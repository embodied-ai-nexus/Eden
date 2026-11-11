"""Hear what things are made of: taps and a scrape, voiced by an out-of-tree contact microphone.

---
tier: recipe
capabilities: [sensor.contact_audio, sensor.spatial_audio]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/sensors/engine_addon_contact_audio.py
---

``ContactAudio`` is a Genesis sensor that stock Genesis does not ship, ported from the
``neuroagents-lab/tactile-genesis`` fork and installed as an :mod:`eden.engine` addon. It turns one link's
contacts into sound with a source-filter model: the *struck* object is a bank of damped resonators (its
material), a tap is an impulse into them, and sliding is force- and speed-scaled noise shaped by the surface
roughness.

Three things happen, one after another, so they can be told apart by ear:

* 0.2 s: a ball lands on a **steel** plate — high modes that ring on for most of a second;
* 0.4 s: a ball, dropped from higher, lands on a **wood** block — low modes, gone in a few milliseconds;
* 0.7 s: a box is let go on a rough ramp — no pitch of its own, a scrape that brightens as the box speeds up.

Each body carries its own ``ContactAudio`` sensor, and the material belongs to what it *touches*: the balls are
identical, and sound different only because of what they land on. A ``SpatialAudio`` microphone in the middle
hears all three through the scene's audio registry. Pass ``--play`` to hear it live while the simulation runs
(with ``--vis``, in sync with the picture), or ``--wav`` to write it to a file.

Real materials ring at kilohertz, so this runs the audio at 16 kHz (``audio_substeps=80`` at ``sim_dt=0.005``)
rather than the 4 kHz default, which could carry nothing above 900 Hz.

Usage::

    python examples/recipes/sensors/engine_addon_contact_audio.py
    python examples/recipes/sensors/engine_addon_contact_audio.py --wav /tmp/contacts.wav
    python examples/recipes/sensors/engine_addon_contact_audio.py --vis --play
"""

from __future__ import annotations

import wave
from argparse import ArgumentParser

import genesis as gs
import numpy as np

import eden as en
from eden.engine.sensors import ContactAudio, ContactAudioProperties, SpatialAudio
from eden.engine.sensors.playback import LiveAudioPlayer
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import BoxOptions, PlaneOptions, SphereOptions
from eden.options.sensors import SensorOptions, SensorsOptions
from eden.utils.configs import EdenConfig

SIM_DT = 0.005
AUDIO_SUBSTEPS = 80
AUDIO_RATE = int(AUDIO_SUBSTEPS / SIM_DT)

# Steel: high, inharmonic modes that decay slowly, and a hard click on impact. The ball comes to rest on the plate,
# so the ring is heard *in contact*; a small ball barely damps a plate, hence the low contact_damping.
STEEL = ContactAudioProperties(
    modal_freqs=(1180.0, 2050.0, 3140.0),
    modal_decays=(5.0, 8.0, 14.0),
    modal_gains=(1.0, 0.7, 0.4),
    contact_damping=2.0,
    accel_noise_gain=0.3,
    accel_noise_freq=3400.0,
)
# Wood: low modes that are gone almost at once.
WOOD = ContactAudioProperties(
    modal_freqs=(310.0, 640.0, 1150.0),
    modal_decays=(90.0, 140.0, 220.0),
    modal_gains=(1.0, 0.5, 0.25),
    impact_gain=0.3,
)
# The ramp: nothing to ring, all roughness. 400 bumps per meter puts a 1 m/s slide at 400 Hz.
ROUGH = ContactAudioProperties(
    modal_gains=(0.0,),
    roughness_gain=0.25,
    roughness_spatial_freq=400.0,
    roughness_bandwidth=700.0,
)

# The ramp is pitched 30 degrees about +y (wxyz), so it descends toward +x.
RAMP_QUAT = (0.9659258, 0.0, 0.2588190, 0.0)
# Friction combines as max(a, b). The ramp is slippery from the start; the box holds on with its own friction
# (1.0 > tan 30 deg) until the script makes it slippery too, which is what lets it go.
SLIPPERY = 0.1
RELEASE_STEP = 140
# Live playback cannot peak-normalize, so it scales by a fixed gain: the loudest tap reaches the microphone at ~7.
PLAY_GAIN = 0.12


def build_env(show_viewer: bool) -> EnvBase:
    """Build the three-event scene with one contact microphone per moving body and one airborne microphone.

    Parameters
    ----------
    show_viewer : bool
        Whether to open the interactive viewer.

    Returns
    -------
    EnvBase
        The built environment.
    """

    def contact_mic(body: str, touches: str, material: ContactAudioProperties) -> SensorOptions:
        return SensorOptions(
            sensor=ContactAudio(audio_substeps=AUDIO_SUBSTEPS, n_modes=3, properties={touches: material}),
            attach_entity_name=body,
        )

    config = EdenConfig(
        # decimation=1 so every simulation step is rendered: read() is then a continuous waveform. At the default
        # decimation the taps would still sound (synthesis runs every simulation step) but the recording would
        # have gaps; see the guide's note on `update_sensors_every_substep`.
        env_options=EnvOptions(num_envs=1, sim_dt=SIM_DT, decimation=1, show_FPS=False),
        scene_options=SceneOptions(
            floor=PlaneOptions(),
            steel_plate=BoxOptions(size=(0.3, 0.3, 0.04), default_root_pos=(-0.5, 0.0, 0.02), fixed=True),
            wood_block=BoxOptions(size=(0.3, 0.3, 0.04), default_root_pos=(0.5, 0.0, 0.02), fixed=True),
            steel_ball=SphereOptions(radius=0.025, default_root_pos=(-0.5, 0.0, 0.25)),
            wood_ball=SphereOptions(radius=0.025, default_root_pos=(0.5, 0.0, 0.85)),
            ramp=BoxOptions(
                size=(0.8, 0.3, 0.04),
                default_root_pos=(0.0, 0.6, 0.35),
                default_root_quat=RAMP_QUAT,
                fixed=True,
                default_geoms_friction={".*": SLIPPERY},
            ),
            slider=BoxOptions(
                size=(0.08, 0.08, 0.05),
                # On the ramp's top face, near its high end.
                default_root_pos=(-0.2373, 0.6, 0.5440),
                default_root_quat=RAMP_QUAT,
                default_geoms_friction={".*": 1.0},
            ),
        ),
        sensors_options=SensorsOptions(
            steel_tap=contact_mic("steel_ball", "steel_plate", STEEL),
            wood_tap=contact_mic("wood_ball", "wood_block", WOOD),
            scrape=contact_mic("slider", "ramp", ROUGH),
            mic=SensorOptions(sensor=SpatialAudio(audio_substeps=AUDIO_SUBSTEPS, pos_offset=(0.0, 0.2, 0.4))),
        ),
    )
    env = EnvBase.from_config(config, show_viewer=show_viewer)
    env.build()
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


def describe(name: str, signal: np.ndarray) -> None:
    """Log when a contact microphone first sounded, and where its spectrum peaks.

    Parameters
    ----------
    name : str
        Sensor name.
    signal : np.ndarray
        The sensor's whole 1-D recording.
    """
    peak = float(np.abs(signal).max())
    if peak == 0.0:
        en.logger.info(f"   {name:<10} silent")
        return
    # "Sounded" means within 40 dB of the recording's peak, which skips the box faintly settling onto the ramp.
    loud = np.flatnonzero(np.abs(signal) > 0.01 * peak)
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.size)))
    peak_hz = np.fft.rfftfreq(signal.size, d=1.0 / AUDIO_RATE)[spectrum.argmax()]
    en.logger.info(f"   {name:<10} first sound at {loud[0] / AUDIO_RATE:5.2f} s, loudest at {peak_hz:6.0f} Hz")


def main():
    parser = ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="Run on the GPU instead of the CPU")
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    parser.add_argument("-s", "--steps", type=int, default=320, help="Number of simulation steps to run")
    parser.add_argument("--wav", type=str, default=None, help="Write what the airborne mic hears to this WAV file")
    parser.add_argument("--play", action="store_true", help="Play what the airborne mic hears, live, in real time")
    args = parser.parse_args()

    en.init(backend=gs.gpu if args.gpu else gs.cpu)

    env = build_env(show_viewer=args.vis)
    names = ("steel_tap", "wood_tap", "scrape", "mic")
    en.logger.info(f"🎧 audio rate {AUDIO_RATE} Hz")

    recorded = {name: [] for name in names}
    steps = 5 if en.is_smoke() else args.steps
    # The player paces the loop to real time, so the sound stays with the viewer's picture.
    speaker = LiveAudioPlayer(AUDIO_RATE, gain=PLAY_GAIN) if args.play and not en.is_smoke() else None
    for step in range(steps):
        if step == RELEASE_STEP:
            env.entities["slider"]._entity.set_friction(SLIPPERY)
        env.step()
        for name in names:
            recorded[name].append(env.sensors[name].read()[0].flatten().cpu().numpy())
        if speaker is not None:
            speaker.write(recorded["mic"][-1])
    if speaker is not None:
        speaker.close()
    signals = {name: np.concatenate(blocks) for name, blocks in recorded.items()}

    en.logger.info("🔊 what each microphone picked up:")
    for name in names:
        describe(name, signals[name])

    if args.wav:
        write_wav(args.wav, signals["mic"])
        en.logger.info(f"💾 wrote {signals['mic'].size / AUDIO_RATE:.2f}s of audio to {args.wav}")


if __name__ == "__main__":
    main()
