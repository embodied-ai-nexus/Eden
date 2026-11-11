"""Listen to an audio sensor while the simulation runs.

:class:`LiveAudioPlayer` streams the blocks an audio sensor's ``read()`` returns to the sound card, one block per
simulation step, so a scene can be heard as it is watched. It needs no Python audio package: the samples are piped
as raw PCM into whichever command-line player the machine already has (``aplay`` or ``ffplay``).

Hearing the simulation in sync means running it in real time, and the player is what enforces that:

- A simulation faster than real time is slowed down. :meth:`LiveAudioPlayer.write` sleeps until the sound card is
  about to need the block, keeping only ``lead`` seconds queued. Without that the pipe would swallow seconds of
  audio ahead of the picture.
- A simulation slower than real time cannot be helped — the sound card runs dry. Each time it does, the player
  counts a dropout and rebuilds its cushion of silence, so one slow step costs one short gap instead of leaving
  the stream on the edge of crackling from then on. :meth:`LiveAudioPlayer.close` reports the count, so "did it
  keep up" is a number rather than an impression.

The cushion is the whole of the delay this module adds, because the players are told not to add their own. Left
at their defaults they would: a stock ``aplay`` holds back 0.5 s of audio before it starts and stays that far
behind (measured: 680 ms), ``ffplay`` 2 s. So ``aplay`` is given a buffer no deeper than ``lead`` and ``ffplay``
its low-delay flags, which leaves ``lead`` plus the sound server's own few tens of milliseconds.
:meth:`LiveAudioPlayer.close` measures that figure — how long the player keeps sounding after the last block — and
reports it. SoX's ``play`` is not tried: a ``play`` on ``PATH`` is not reliably SoX, and its latency was never
measured here.

Eden's interactive viewer already paces ``env.step()`` to ``sim_dt``, so a scene light enough to hold that rate
plays cleanly with the viewer open. It also needs one ``read()`` per simulation step (``decimation=1``); see the
engine addons guide on gapped reads.

This module is not imported by :mod:`eden.engine.sensors` and installs no sensor.
"""

import shutil
import subprocess
import time
from typing import Self

import numpy as np
import torch

import eden as en

# Raw mono signed-16-bit little-endian PCM on stdin, tried in this order. `{rate}` is the sample rate in Hz and
# `{buffer_us}` the cushion in microseconds. Every flag past the format is there to stop the player buffering ahead.
_PLAYERS = (
    ("aplay", "-q", "-t", "raw", "-f", "S16_LE", "-c", "1", "-r", "{rate}", "-B", "{buffer_us}", "-"),
    (
        *("ffplay", "-loglevel", "error", "-nodisp", "-autoexit"),
        *("-fflags", "nobuffer", "-flags", "low_delay", "-probesize", "32", "-analyzeduration", "0"),
        *("-f", "s16le", "-ar", "{rate}", "-i", "pipe:0"),
    ),
)


class LiveAudioPlayer:
    """Stream mono float blocks to the sound card in real time, through a command-line player.

    Use as a context manager around the simulation loop::

        with LiveAudioPlayer(sample_rate=16000, gain=0.2) as speaker:
            for _ in range(steps):
                env.step()
                speaker.write(env.sensors["mic"].read()[0])

    A recording can be peak-normalized after the fact; a live stream cannot, since the peak is not known yet. The
    samples are scaled by a fixed ``gain`` instead and soft-limited with ``tanh``, so a louder moment than expected
    saturates smoothly rather than wrapping around.

    Parameters
    ----------
    sample_rate : int
        Samples per second: ``audio_substeps / sim_dt`` of the sensor being played.
    gain : float
        Scale from the sensor's arbitrary units to full scale ``[-1, 1]``, applied before the soft limiter.
    lead : float
        Seconds of audio kept queued ahead of the sound card, which is also how far the sound trails the
        simulation. Larger rides out slower steps without a dropout; smaller keeps the sound closer to the picture.
    command : tuple[str, ...] | None
        Player command reading raw mono S16_LE PCM from stdin, with ``{rate}`` standing for the sample rate and
        ``{buffer_us}`` for ``lead`` in microseconds. It must not buffer ahead by itself, or the sound will trail
        by that much more. ``None`` picks the first of ``aplay`` and ``ffplay`` found on ``PATH``.

    Attributes
    ----------
    dropouts : int
        Times a block arrived after the queue had run dry, i.e. audible gaps so far.
    trailing : float
        Seconds the player kept sounding after the last block was written, measured by :meth:`close`: how far the
        sound trailed the simulation. About ``lead`` plus the sound server's latency when the player is well behaved.
    """

    def __init__(self, sample_rate: int, gain: float = 1.0, lead: float = 0.06, command: tuple[str, ...] | None = None):
        if command is None:
            command = next((player for player in _PLAYERS if shutil.which(player[0])), None)
            if command is None:
                raise RuntimeError(
                    f"No command-line audio player found; looked for {[player[0] for player in _PLAYERS]} on PATH. "
                    "Install one (ALSA's aplay or ffmpeg's ffplay), or pass `command`."
                )
        self._command = [part.format(rate=sample_rate, buffer_us=round(lead * 1e6)) for part in command]
        self._sample_rate = sample_rate
        self._gain = gain
        self._lead = lead
        self._process = subprocess.Popen(self._command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self._started_at: float | None = None
        self._seconds_written = 0.0  # of the caller's audio; the silences this player inserts are not counted
        self.dropouts = 0
        self._last_write_at: float | None = None
        self.trailing = 0.0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def write(self, block: "torch.Tensor | np.ndarray") -> None:
        """Queue one block, first sleeping until the sound card is ``lead`` seconds from needing it.

        Parameters
        ----------
        block : torch.Tensor | np.ndarray
            Mono float samples of one environment, any shape; flattened in order.
        """
        if isinstance(block, torch.Tensor):
            block = block.detach().cpu().numpy()
        samples = np.tanh(self._gain * np.asarray(block, dtype=np.float64).reshape(-1))

        # The queue holds `lead + due_in` seconds: the sound card has consumed `now - started_at` of the
        # `lead + seconds_written` handed over so far.
        block_seconds = samples.size / self._sample_rate
        now = time.perf_counter()
        ran_dry = self._started_at is not None and self._started_at + self._seconds_written - now < -self._lead
        if self._started_at is None or ran_dry:
            # Open with `lead` seconds of silence. That is the cushion: while the simulation keeps up the queue holds
            # `lead` seconds, and a slow step eats into it instead of into the sound. Once it is used up, start over.
            self.dropouts += ran_dry
            self._started_at = now - self._seconds_written
            samples = np.concatenate((np.zeros(round(self._lead * self._sample_rate)), samples))
        due_in = self._started_at + self._seconds_written - now
        if due_in > 0.0:
            time.sleep(due_in)

        try:
            self._process.stdin.write((samples * 32767.0).astype("<i2").tobytes())
            self._process.stdin.flush()
        except BrokenPipeError:
            stderr = self._process.stderr.read().decode(errors="replace").strip()
            raise RuntimeError(
                f"Audio player `{' '.join(self._command)}` exited early: {stderr or 'no output'}"
            ) from None
        self._seconds_written += block_seconds
        self._last_write_at = time.perf_counter()

    def close(self) -> None:
        """Let the queued audio finish, stop the player, and report how the simulation kept up."""
        if self._process.stdin.closed:
            return
        try:
            self._process.stdin.close()
        except BrokenPipeError:
            pass
        self._process.wait()
        self._process.stderr.close()
        # The player exits once the last sample has sounded, so how long it outlives the last write is how far the
        # sound was trailing: the cushion plus whatever the player and the sound server buffer.
        self.trailing = time.perf_counter() - self._last_write_at if self._last_write_at is not None else 0.0

        en.logger.info(
            f"🔈 played {self._seconds_written:.2f} s of audio with {self.dropouts} dropouts, trailing the simulation "
            f"by ~{1e3 * self.trailing:.0f} ms"
        )
        if self.dropouts:
            en.logger.warning(
                f"The simulation fell behind real time {self.dropouts} times, each one a gap in the sound. Lighten "
                "the scene or raise `lead`, or write a WAV and listen afterwards."
            )
