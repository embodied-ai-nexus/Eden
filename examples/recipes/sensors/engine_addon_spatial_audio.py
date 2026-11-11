"""Hear a Franka's motors: an out-of-tree Genesis sensor installed as an eden.engine addon.

---
tier: recipe
capabilities: [sensor.spatial_audio, sensor.actuation_audio, robot.franka]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/sensors/engine_addon_spatial_audio.py
---

Two sensors that stock Genesis does not ship, ported from the
``neuroagents-lab/tactile-genesis`` fork and installed as :mod:`eden.engine` addons:

* ``ActuationAudio`` synthesizes each actuated DOF's motor noise from its torque and speed, and
  registers it with the scene's audio registry.
* ``SpatialAudio`` is a microphone that renders that registry with distance attenuation, propagation
  delay, and Doppler.

Neither one needs a patched or forked Genesis. Importing ``eden.engine.sensors`` is the whole
installation — see that package's docstring for why.

Three microphones listen to the same arm: one close, one four times farther away (to hear the
attenuation), and one riding the arm's own hand (to hear the motors from onboard). Pass ``--play`` to
hear the onboard signal live while the simulation runs (with ``--vis``, in sync with the picture), or
``--wav`` to write it to a file.

Usage::

    python examples/recipes/sensors/engine_addon_spatial_audio.py --steps 400
    python examples/recipes/sensors/engine_addon_spatial_audio.py --wav /tmp/franka.wav
    python examples/recipes/sensors/engine_addon_spatial_audio.py --vis --play
"""

from __future__ import annotations

import wave
from argparse import ArgumentParser

import genesis as gs
import numpy as np
import torch

import eden as en
from eden.engine.sensors import ActuationAudio, ActuationAudioProperties, SpatialAudio
from eden.engine.sensors.playback import LiveAudioPlayer
from eden.envs.base import EnvBase
from eden.options import EnvOptions, PlaneOptions, SceneOptions
from eden.options.sensors import SensorOptions, SensorsOptions

# 20 samples per 5 ms step is a 4 kHz audio rate: coarse for music, plenty for a motor whine whose
# fundamental stays under a few hundred Hz.
SIM_DT = 0.005
AUDIO_SUBSTEPS = 20
AUDIO_RATE = int(AUDIO_SUBSTEPS / SIM_DT)
# Live playback cannot peak-normalize, so it scales by a fixed gain: the onboard signal peaks at ~15.
PLAY_GAIN = 0.06

# A servo voiced to be clearly audible: a loud load-driven whine, a motion-driven hum, bearing hiss,
# and a click each time the joint reverses.
SERVO = ActuationAudioProperties(
    pitch_slope=180.0,
    harmonic_gains=(1.0, 0.6, 0.3),
    idle_freq=120.0,
    idle_velocity_gain=0.05,
    idle_harmonic_gains=(1.0, 0.5, 0.25),
    friction_gain=0.02,
    load_gain=0.02,
    power_gain=0.05,
    reversal_click_gain=0.2,
)


def build_env(show_viewer: bool) -> EnvBase:
    """Build a one-arm scene with a motor source and three microphones.

    ``decimation=1`` (rather than Eden's default 4) so that one ``env.step()`` synthesizes exactly one
    audio block and a single ``read()`` per step yields a continuous waveform. At the default
    decimation a control-rate read samples one block in four: fine when the signal is an observation,
    a gapped stream when it is a recording. Propagation stays correct either way — the delay line
    advances every simulation step regardless of how often anything renders it.

    Parameters
    ----------
    show_viewer : bool
        Open the interactive viewer.

    Returns
    -------
    EnvBase
        The built environment.
    """
    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=SIM_DT,
            decimation=1,
            show_FPS=False,
        ),
        scene_options=SceneOptions(
            floor=PlaneOptions(),
            robot=en.robots.FrankaEmikaPanda(),
        ),
        sensors_options=SensorsOptions(
            motors=SensorOptions(
                sensor=ActuationAudio(audio_substeps=AUDIO_SUBSTEPS, default_properties=SERVO),
                attach_entity_name="robot",
            ),
            mic_near=SensorOptions(sensor=SpatialAudio(audio_substeps=AUDIO_SUBSTEPS, pos_offset=(0.5, 0.0, 0.5))),
            mic_far=SensorOptions(sensor=SpatialAudio(audio_substeps=AUDIO_SUBSTEPS, pos_offset=(2.0, 0.0, 0.5))),
            mic_onboard=SensorOptions(
                sensor=SpatialAudio(audio_substeps=AUDIO_SUBSTEPS),
                attach_entity_name="robot",
                attach_link_name="hand",
            ),
        ),
        show_viewer=show_viewer,
    )
    env.build()
    return env


def write_wav(path: str, samples: np.ndarray) -> None:
    """Write a mono 16-bit WAV, normalized to just under full scale.

    Parameters
    ----------
    path : str
        Destination file.
    samples : np.ndarray
        1-D float waveform in arbitrary units.
    """
    peak = float(np.abs(samples).max())
    scaled = samples / peak * 0.95 if peak > 0.0 else samples
    with wave.open(path, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(AUDIO_RATE)
        fh.writeframes((scaled * 32767.0).astype("<i2").tobytes())


def main():
    parser = ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="Run on the GPU instead of the CPU")
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    parser.add_argument("-s", "--steps", type=int, default=200, help="Number of control steps to run")
    parser.add_argument("--wav", type=str, default=None, help="Write the onboard mic signal to this WAV file")
    parser.add_argument("--play", action="store_true", help="Play the onboard mic signal, live, in real time")
    args = parser.parse_args()

    en.init(backend=gs.gpu if args.gpu else gs.cpu, debug=True)

    env = build_env(show_viewer=args.vis)
    robot = env.entities["robot"]
    n_dofs = robot._entity.n_dofs
    mics = {name: env.sensors[name] for name in ("mic_near", "mic_far", "mic_onboard")}

    en.logger.info(f"🎧 audio rate {AUDIO_RATE} Hz, voicing {env.sensors['motors'].read().shape[1]} DOFs")

    # Sweep the arm back and forth so the motors have torque and speed to voice, and so the reversals
    # produce audible clicks.
    energy = dict.fromkeys(mics, 0.0)
    captured = []
    steps = 5 if en.is_smoke() else args.steps
    # The player paces the loop to real time, so the sound stays with the viewer's picture.
    speaker = LiveAudioPlayer(AUDIO_RATE, gain=PLAY_GAIN) if args.play and not en.is_smoke() else None
    for step in range(steps):
        phase = np.sin(2.0 * np.pi * step / 60.0)
        target = torch.full((env.num_envs, n_dofs), 0.7 * float(phase), device=env.device)
        robot._entity.control_dofs_position(target)
        env.step()

        for name, mic in mics.items():
            block = mic.read()
            energy[name] += float(block.pow(2).sum())
        captured.append(mics["mic_onboard"].read()[0].flatten().cpu().numpy())
        if speaker is not None:
            speaker.write(captured[-1])
    if speaker is not None:
        speaker.close()

    en.logger.info("🔊 accumulated signal energy per microphone:")
    for name, value in energy.items():
        en.logger.info(f"   {name:<12} {value:12.3f}")
    ratio = energy["mic_near"] / max(energy["mic_far"], 1e-12)
    en.logger.info(f"   near/far = {ratio:.1f}x  (mic_far is 4x the distance)")

    if args.wav:
        write_wav(args.wav, np.concatenate(captured))
        en.logger.info(f"💾 wrote {len(np.concatenate(captured)) / AUDIO_RATE:.2f}s of audio to {args.wav}")


if __name__ == "__main__":
    main()
