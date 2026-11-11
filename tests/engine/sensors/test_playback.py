"""``eden.engine.sensors.playback.LiveAudioPlayer``: what reaches the player, and when.

No sound card: the player command is a shell pipeline writing its stdin to a file.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from eden.engine.sensors.playback import LiveAudioPlayer

RATE = 8000


def _to_file(path):
    return ("sh", "-c", f"cat > {path}")


def test_blocks_arrive_as_gained_soft_limited_pcm_after_the_opening_silence(tmp_path):
    """The player receives ``lead`` seconds of silence, then every block as S16_LE of ``tanh(gain * x)``."""
    out = tmp_path / "pcm.raw"
    blocks = [torch.linspace(-4.0, 4.0, 80), np.full(80, 0.5)]  # a tensor and an array; one far beyond full scale
    with LiveAudioPlayer(RATE, gain=0.5, lead=0.01, command=_to_file(out)) as speaker:
        for block in blocks:
            speaker.write(block)

    pcm = np.frombuffer(out.read_bytes(), dtype="<i2")
    silence, played = pcm[:80], pcm[80:]  # 0.01 s at 8 kHz
    assert not silence.any()
    expected = np.tanh(0.5 * np.concatenate([np.asarray(b, dtype=np.float64) for b in blocks])) * 32767.0
    np.testing.assert_allclose(played, expected.astype("<i2"), atol=1)
    assert np.abs(played).max() < 32767, "the limiter saturates; it never wraps"


def test_a_fast_producer_is_held_to_real_time(tmp_path):
    """Writing 0.4 s of audio takes about 0.4 s, however fast the blocks are produced."""
    block = np.zeros(RATE // 20)  # 0.05 s
    started = time.perf_counter()
    with LiveAudioPlayer(RATE, lead=0.1, command=_to_file(tmp_path / "pcm.raw")) as speaker:
        for _ in range(8):
            speaker.write(block)
        in_loop = time.perf_counter() - started
    # The last block is written when the one before it is due, i.e. 0.35 s in; scheduling only ever adds to that.
    assert 0.33 < in_loop < 0.6


def test_a_late_block_counts_one_dropout_and_rebuilds_the_cushion(tmp_path):
    """A producer that stalls past the cushion costs one gap: the player re-primes instead of limping on empty."""
    out = tmp_path / "pcm.raw"
    block = np.full(RATE // 20, 0.5)  # 0.05 s
    lead = 0.05
    with LiveAudioPlayer(RATE, lead=lead, command=_to_file(out)) as speaker:
        speaker.write(block)
        speaker.write(block)
        assert speaker.dropouts == 0
        time.sleep(0.4)  # far longer than the 0.05 s queued plus the 0.05 s cushion
        speaker.write(block)
        speaker.write(block)  # back on schedule: the fresh cushion absorbs it
        assert speaker.dropouts == 1

    pcm = np.frombuffer(out.read_bytes(), dtype="<i2")
    n_lead, n_block = round(lead * RATE), block.size
    layout = [(n_lead, False), (2 * n_block, True), (n_lead, False), (2 * n_block, True)]
    assert pcm.size == sum(n for n, _ in layout)
    at = 0
    for n, sounding in layout:
        assert bool(pcm[at : at + n].all()) == sounding and bool(pcm[at : at + n].any()) == sounding
        at += n


def test_the_player_is_told_the_cushion_and_its_trailing_is_measured(tmp_path):
    """The command receives the rate and ``lead`` as its buffer depth; ``close`` times how long it outlives the feed.

    A stock ``aplay`` buffers 0.5 s and trails by that much for good, so the cushion has to reach the player. The
    stand-in sleeps 0.2 s after its input ends, as a player with 0.2 s still queued would.
    """
    args = tmp_path / "args.txt"
    command = ("sh", "-c", f"cat > /dev/null; echo rate={{rate}} buffer={{buffer_us}} > {args}; sleep 0.2")
    speaker = LiveAudioPlayer(RATE, lead=0.04, command=command)
    speaker.write(np.zeros(RATE // 20))
    speaker.close()
    assert args.read_text().strip() == f"rate={RATE} buffer=40000"
    assert 0.2 <= speaker.trailing < 0.5


def test_a_player_that_dies_is_reported_with_its_own_words(tmp_path):
    """A player that exits early surfaces as an error carrying its stderr, not as a bare ``BrokenPipeError``."""
    marker = tmp_path / "exited"
    speaker = LiveAudioPlayer(RATE, lead=0.0, command=("sh", "-c", f"echo 'no such device' >&2; touch {marker}"))
    while not marker.exists():
        time.sleep(0.01)
    with pytest.raises(RuntimeError, match="exited early: no such device"):
        for _ in range(64):  # enough to overrun any pipe buffering of a dead reader
            speaker.write(np.zeros(4096))


def test_no_player_on_the_machine_names_the_ones_looked_for(monkeypatch):
    """Without any known player on ``PATH`` construction fails, listing what would have worked."""
    monkeypatch.setenv("PATH", "")
    with pytest.raises(RuntimeError, match="aplay.*ffplay"):
        LiveAudioPlayer(RATE)
