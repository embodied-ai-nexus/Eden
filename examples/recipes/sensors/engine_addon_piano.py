"""A piano that plays itself: 88 simulated keys, voiced by an out-of-tree audio sensor.

---
tier: recipe
capabilities: [sensor.piano_audio, sensor.spatial_audio]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/sensors/engine_addon_piano.py
---

The keyboard (:class:`~eden.options.objects.piano.Piano`) is physics only: a fixed base and 88 spring-loaded
hinges, with the dimensions of RoboPianist's piano. ``PianoAudio``, an :mod:`eden.engine` addon, is what makes it
an instrument. It watches the key joints and, when one travels past its strike point, throws a hammer at that
key's string: a bank of decaying partials at the right pitch, as loud as the key was fast, ringing while the key
is held and damped when it comes back up.

Nothing here touches the sound directly. An invisible pianist pushes the keys with joint torques, harder for a
louder note, and everything heard follows from how the keys then move. A robot hand pressing the same keys would
sound the same.

The tune is the opening of Beethoven's *Ode to Joy* over a bass line, played with a crescendo, then one chord left
to ring on the sustain pedal after the keys are let go. A ``SpatialAudio`` microphone half a metre in front of the
keyboard hears it. Pass ``--play`` to listen live (with ``--vis``, in sync with the picture) or ``--wav`` to write
a file.

Usage::

    python examples/recipes/sensors/engine_addon_piano.py --wav /tmp/piano.wav
    python examples/recipes/sensors/engine_addon_piano.py --vis --play
"""

from __future__ import annotations

import wave
from argparse import ArgumentParser
from typing import NamedTuple

import genesis as gs
import numpy as np

import eden as en
from eden.engine.sensors import PianoAudio, SpatialAudio
from eden.engine.sensors.playback import LiveAudioPlayer
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import PlaneOptions
from eden.options.objects.piano import NUM_KEYS, Piano, key_of_midi_note
from eden.options.sensors import SensorOptions, SensorsOptions
from eden.utils.configs import EdenConfig

SIM_DT = 0.005
# 16 kHz, PianoAudio's default: enough for every fundamental on the keyboard.
AUDIO_SUBSTEPS = 80
AUDIO_RATE = int(AUDIO_SUBSTEPS / SIM_DT)

BEAT = 0.4  # seconds
# The key spring takes 0.12 N·m to reach the strike point at all. From there, more torque is a faster key and a
# louder note, up to about 0.6 N·m where the note is at full scale.
SOFTEST_TORQUE = 0.14
LOUDEST_TORQUE = 0.6
# Live playback cannot peak-normalize, so it scales by a fixed gain: the final chord reaches the microphone at ~1.5.
PLAY_GAIN = 0.4


class Note(NamedTuple):
    """One press of one key."""

    midi: int
    start: float  # beats
    length: float  # beats the key is held
    loudness: float  # 0 to 1


def ode_to_joy() -> tuple[list[Note], float, float]:
    """The score: notes, the beat the sustain pedal goes down on, and the length of the piece in beats.

    Returns
    -------
    tuple[list[Note], float, float]
        The notes, the pedal beat, and the total length.
    """
    E4, F4, G4, D4, C4 = 64, 65, 67, 62, 60
    melody = [E4, E4, F4, G4, G4, F4, E4, D4, C4, C4, D4, E4]
    notes = [
        # A crescendo from quiet to loud over the phrase. Each key is let go a little before the next goes down.
        Note(midi, start=beat, length=0.8, loudness=0.15 + 0.6 * beat / len(melody))
        for beat, midi in enumerate(melody)
    ]
    bass = [(48, 0), (43, 4), (48, 8)]  # C3, G2, C3: one to a bar, held through it
    notes += [Note(midi, start=beat, length=3.8, loudness=0.3) for midi, beat in bass]
    # The last bar: E D D, then a C major chord across four octaves, struck hard and let go after half a beat.
    end = len(melody)
    notes += [Note(E4, end, 1.3, 0.8), Note(D4, end + 1.5, 0.4, 0.6), Note(D4, end + 2, 1.8, 0.7)]
    notes.append(Note(43, end, 3.8, 0.4))
    chord_beat = end + 4
    notes += [Note(midi, chord_beat, 0.5, 1.0) for midi in (36, 48, 55, 60, 64, 67, 72)]
    return notes, chord_beat - 0.25, chord_beat + 7.0


def key_torques(notes: list[Note], time: float) -> np.ndarray:
    """The pianist: the torque on every key joint at ``time``.

    Parameters
    ----------
    notes : list[Note]
        The score.
    time : float
        Seconds since the start of the piece.

    Returns
    -------
    np.ndarray
        ``(NUM_KEYS,)`` torques in N·m, zero for a key nobody is pressing.
    """
    torques = np.zeros(NUM_KEYS)
    for note in notes:
        if note.start * BEAT <= time < (note.start + note.length) * BEAT:
            torques[key_of_midi_note(note.midi)] = SOFTEST_TORQUE + note.loudness * (LOUDEST_TORQUE - SOFTEST_TORQUE)
    return torques


def build_env(show_viewer: bool) -> EnvBase:
    """Build the keyboard, its voice and a microphone.

    Parameters
    ----------
    show_viewer : bool
        Whether to open the interactive viewer.

    Returns
    -------
    EnvBase
        The built environment.
    """
    config = EdenConfig(
        # decimation=1 so every simulation step is read: the recording is then a continuous waveform.
        env_options=EnvOptions(num_envs=1, sim_dt=SIM_DT, decimation=1, show_FPS=False),
        scene_options=SceneOptions(floor=PlaneOptions(), piano=Piano(default_root_pos=(0.0, 0.0, 0.7))),
        sensors_options=SensorsOptions(
            strings=SensorOptions(sensor=PianoAudio(audio_substeps=AUDIO_SUBSTEPS), attach_entity_name="piano"),
            # Where a pianist's head would be.
            mic=SensorOptions(sensor=SpatialAudio(audio_substeps=AUDIO_SUBSTEPS, pos_offset=(0.3, 0.0, 1.1))),
        ),
    )
    env = EnvBase.from_config(config, show_viewer=show_viewer)
    env.build()
    if show_viewer:
        env.scene.viewer.set_camera_pose(pos=(1.1, -0.9, 1.5), lookat=(0.0, 0.0, 0.7))
    return env


def write_wav(path: str, samples: np.ndarray) -> None:
    """Write a mono 16-bit WAV, normalized to just under full scale.

    Parameters
    ----------
    path : str
        Output file.
    samples : np.ndarray
        1-D float waveform in arbitrary units.
    """
    peak = float(np.abs(samples).max())
    scaled = samples / peak * 0.95 if peak > 0.0 else samples
    with wave.open(path, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(AUDIO_RATE)
        fh.writeframes((scaled * 32767.0).astype(np.int16).tobytes())


def main():
    parser = ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="Run on the GPU instead of the CPU")
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    parser.add_argument("-s", "--steps", type=int, default=None, help="Simulation steps to run (default: the piece)")
    parser.add_argument("--wav", type=str, default=None, help="Write what the microphone hears to this WAV file")
    parser.add_argument("--play", action="store_true", help="Play what the microphone hears, live, in real time")
    args = parser.parse_args()

    en.init(backend=gs.gpu if args.gpu else gs.cpu)

    notes, pedal_beat, length = ode_to_joy()
    env = build_env(show_viewer=args.vis)
    piano = env.entities["piano"]
    strings = env.sensors["strings"]
    en.logger.info(f"🎹 {len(notes)} notes over {length * BEAT:.1f} s, audio rate {AUDIO_RATE} Hz")

    steps = args.steps if args.steps is not None else round(length * BEAT / SIM_DT)
    if en.is_smoke():
        steps = 5
    # The player paces the loop to real time, so the sound stays with the viewer's picture.
    speaker = LiveAudioPlayer(AUDIO_RATE, gain=PLAY_GAIN) if args.play and not en.is_smoke() else None
    recorded, struck = [], 0
    was_down = np.zeros(NUM_KEYS, dtype=bool)
    for step in range(steps):
        time = step * SIM_DT
        # The keyboard has no actuators of its own; these torques are the pianist's fingers.
        piano._entity.control_dofs_force(key_torques(notes, time))
        strings.sensor.set_sustain(time >= pedal_beat * BEAT)
        env.step()

        down = strings.sensor.keys_down[0].cpu().numpy()
        struck += int((down & ~was_down).sum())
        was_down = down
        recorded.append(env.sensors["mic"].read()[0].flatten().cpu().numpy())
        if speaker is not None:
            speaker.write(recorded[-1])
    if speaker is not None:
        speaker.close()
    heard = np.concatenate(recorded)

    en.logger.info(
        f"🔊 {struck} hammers thrown for {len(notes)} notes; peak level at the microphone {np.abs(heard).max():.2f}"
    )
    if args.wav:
        write_wav(args.wav, heard)
        en.logger.info(f"💾 wrote {heard.size / AUDIO_RATE:.2f}s of audio to {args.wav}")


if __name__ == "__main__":
    main()
