"""``eden.engine.sensors.actuation_audio``: motor and gearbox noise per actuated DOF."""

from __future__ import annotations

import genesis as gs
import pytest
import torch

from eden.engine.sensors import ActuationAudio, ActuationAudioProperties, AudioContext, SpatialAudio
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.robots import FrankaEmikaPanda
from eden.options.sensors import SensorOptions, SensorsOptions
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env

# 0.01 s steps split into 20 samples: a 2 kHz audio rate, whose partial ceiling (0.45 x Nyquist) is 450 Hz.
SIM_DT = 0.01
K = 20
DT_SUB = SIM_DT / K
LOUD = ActuationAudioProperties(power_gain=1.0, load_gain=0.5, friction_gain=0.3, reversal_click_gain=0.5)
# A pitch slope this steep saturates the whine as soon as the joint moves at all.
SATURATING = dict(pitch_slope=1e6, load_gain=1.0, power_gain=0.0, idle_freq=900.0, idle_gain=1e-4)
WHINE_CASES = {
    "three_partials": ((1.0, 0.5, 0.25), 150.0),
    "fundamental_only": ((1.0,), 450.0),
    "trailing_zero_gain": ((1.0, 0.5, 0.0), 225.0),
}


def _motors(joints=None, **options):
    return SensorOptions(sensor=ActuationAudio(audio_substeps=K, joints=joints, **options), attach_entity_name="robot")


@pytest.fixture(scope="module")
def arm(eden_messages):
    """Two Franka envs voiced by several actuation sensors, swung +-0.3 rad for 64 steps and then held for 120.

    Yields ``(env, observed)``: the build-time log, env 0's whine recordings from the swing, the loudest motor and
    microphone samples of the swing, and the holding torque, residual speed and straining sensor's loudest sample
    once the arm has come to rest.
    """
    sensors = {
        "motors": _motors(default_properties=LOUD),
        "subset": _motors(joints=("joint1", "joint2")),
        "strain": _motors(default_properties=ActuationAudioProperties(load_gain=1.0, power_gain=0.0)),
        "mic": SensorOptions(sensor=SpatialAudio(audio_substeps=K, pos_offset=(0.5, 0.0, 0.5))),
    }
    for name, (gains, _) in WHINE_CASES.items():
        props = ActuationAudioProperties(harmonic_gains=gains, **SATURATING)
        sensors[name] = _motors(joints=("joint2",), n_partials=3, default_properties=props)
    with eden_messages() as messages:
        env = build_env(
            EdenConfig(
                env_options=EnvOptions(num_envs=2, sim_dt=SIM_DT, decimation=1),
                scene_options=SceneOptions(robot=FrankaEmikaPanda()),
                sensors_options=SensorsOptions(**sensors),
            )
        )
    robot = env.entities["robot"]._entity
    hold = robot.get_dofs_position().clone()
    whine = {name: [] for name in WHINE_CASES}
    loudest_motor = loudest_mic = 0.0
    for step in range(64):
        robot.control_dofs_position(hold + (0.3 if (step // 10) % 2 == 0 else -0.3))
        env.step()
        for name in WHINE_CASES:
            whine[name].append(env.sensors[name].read()[0].reshape(-1).clone())
        loudest_motor = max(loudest_motor, float(env.sensors["motors"].read().abs().max()))
        loudest_mic = max(loudest_mic, float(env.sensors["mic"].read().abs().max()))
    for _ in range(120):
        robot.control_dofs_position(hold)
        env.step()
    observed = {
        "messages": list(messages),
        "whine": {name: torch.cat(blocks) for name, blocks in whine.items()},
        "loudest_motor": loudest_motor,
        "loudest_mic": loudest_mic,
        "holding_torque": float(robot.get_dofs_control_force().abs().max()),
        "residual_speed": float(robot.get_dofs_velocity().abs().max()),
        "strain": float(env.sensors["strain"].read().abs().max()),
    }
    yield env, observed
    env.destroy()


def test_every_actuated_dof_is_voiced_and_heard_through_the_registry(arm, audio_context):
    """``read()`` is one block per DOF, and a microphone hears the motors: every sensor publishes to one registry."""
    env, observed = arm
    n_dofs = env.entities["robot"]._entity.n_dofs
    assert tuple(env.sensors["motors"].read().shape) == (env.num_envs, n_dofs, K)
    assert tuple(env.sensors["subset"].read().shape) == (env.num_envs, 2, K)
    assert list(env.scene.sim._sensor_manager._shared_contexts) == [AudioContext]
    assert len(audio_context(env).sources) == 6
    assert observed["loudest_motor"] > 0.0
    assert observed["loudest_mic"] > 0.0


@pytest.mark.parametrize("case", WHINE_CASES)
def test_whine_saturates_where_its_highest_sounding_partial_meets_the_ceiling(arm, case):
    """A saturated whine stops where its top *sounding* partial still fits under the 450 Hz ceiling.

    Three sounding partials stop the fundamental at 150 Hz: bounding only the fundamental would put the third
    partial at 1350 Hz, folding down to 650 Hz. What counts is the highest non-zero gain, not ``n_partials``.
    """
    _, observed = arm
    signal = observed["whine"][case]
    spectrum = torch.fft.rfft(signal * torch.hann_window(signal.numel(), device=signal.device)).abs()
    freqs = torch.fft.rfftfreq(signal.numel(), d=DT_SUB)
    assert float(freqs[spectrum.argmax()]) == pytest.approx(WHINE_CASES[case][1], abs=10.0)


def test_idle_hum_above_the_ceiling_is_clamped_and_says_so(arm):
    """The hum is a lone fundamental, so it is clamped to the full ceiling, with a warning."""
    _, observed = arm
    assert any("idle_freq" in m and "clamped to 450 Hz" in m for m in observed["messages"])


def test_a_joint_holding_still_under_load_adds_no_dc_offset(arm):
    """Holding torque alone makes no sound: a still joint's oscillators freeze, and a frozen sine is a DC step.

    The whine fades out below 20 Hz instead, so an arm holding its pose against gravity is silent.
    """
    _, observed = arm
    assert observed["holding_torque"] > 1.0, "test premise: the arm is holding a load"
    assert observed["residual_speed"] < 1e-2, "test premise: the arm has come to rest"
    assert observed["strain"] < 1e-2 * observed["holding_torque"]


BANDS = {"friction_freq": "friction_gain", "click_freq": "reversal_click_gain", "idle_freq": "idle_velocity_gain"}


@pytest.fixture(scope="module")
def clamp_warnings(eden_messages):
    """Build-time clamp warnings of a scene at the stock 4 kHz rate (900 Hz ceiling).

    It holds a stock sensor with every band audible, and per band two sensors setting it to 5000 Hz: one with
    the band's gain zero, one audible.
    """
    silent = {"idle_gain": 0.0, "idle_velocity_gain": 0.0, "friction_gain": 0.0, "reversal_click_gain": 0.0}
    sensors = {
        "stock": SensorOptions(
            sensor=ActuationAudio(
                default_properties=ActuationAudioProperties(idle_gain=0.1, friction_gain=0.1, reversal_click_gain=0.1)
            ),
            attach_entity_name="robot",
        )
    }
    for attr, gain_attr in BANDS.items():
        for gain in (0.0, 0.1):
            props = ActuationAudioProperties(**{**silent, attr: 5000.0, gain_attr: gain})
            sensors[f"{attr}_{gain}"] = SensorOptions(
                sensor=ActuationAudio(joints=("joint1",), default_properties=props), attach_entity_name="robot"
            )
    with eden_messages() as messages:
        env = build_env(
            EdenConfig(
                env_options=EnvOptions(num_envs=1),
                scene_options=SceneOptions(robot=FrankaEmikaPanda()),
                sensors_options=SensorsOptions(**sensors),
            )
        )
    env.destroy()
    return [m for m in messages if "clamped" in m]


def test_stock_timbre_needs_no_clamping(clamp_warnings):
    """Defaults throughout fit the stock audio rate, so only the deliberately out-of-range bands warn."""
    assert all("5000 Hz" in w for w in clamp_warnings), clamp_warnings


@pytest.mark.parametrize("attr", BANDS)
def test_out_of_range_band_warns_only_when_audible(clamp_warnings, attr):
    """A band the rate cannot carry is clamped either way; only the one that sounds is worth a warning."""
    assert len([w for w in clamp_warnings if attr in w and "5000 Hz" in w]) == 1


@pytest.mark.parametrize(
    ("sensor", "match"),
    [
        (_motors(joints=("joint_typo",)), "joint_typo"),
        (SensorOptions(sensor=ActuationAudio(audio_substeps=K)), "entity_idx is still -1"),
        (
            _motors(n_partials=3, default_properties=ActuationAudioProperties(harmonic_gains=(1.0, 0.5, 0.25, 0.1))),
            "harmonic_gains.*n_partials=3",
        ),
        (
            SensorOptions(sensor=ActuationAudio(), attach_entity_name="robot", attach_link_name="hand"),
            "link_idx_local.*attach_link_name",
        ),
    ],
    ids=["unknown_joint", "unattached", "more_gains_than_partials", "link_attachment"],
)
def test_misconfigured_actuation_audio_fails_at_build(sensor, match):
    """A misspelled joint, a missing entity, a gain that could never sound, or an ignored link attachment raise."""
    config = EdenConfig(
        env_options=EnvOptions(num_envs=1, sim_dt=SIM_DT),
        scene_options=SceneOptions(robot=FrankaEmikaPanda()),
        sensors_options=SensorsOptions(motors=sensor),
    )
    env = EnvBase.from_config(config, show_viewer=False)
    try:
        with pytest.raises(gs.GenesisException, match=match):
            env.build()
    finally:
        env.destroy()


def test_audio_sensor_config_round_trips_through_a_file(tmp_path):
    """A saved config reloads the Eden-side options classes, per-joint overrides as ``ActuationAudioProperties``.

    ``ActuationVoice`` reads an override by attribute, so it must come back as the NamedTuple, not a bare list.
    """
    config = EdenConfig(
        env_options=EnvOptions(num_envs=1, sim_dt=SIM_DT),
        scene_options=SceneOptions(robot=FrankaEmikaPanda()),
        sensors_options=SensorsOptions(
            motors=_motors(
                joints=("joint1", "joint2"), properties={"joint2": ActuationAudioProperties(pitch_slope=999.0)}
            ),
            mic=SensorOptions(sensor=SpatialAudio(audio_substeps=K, attenuation="inverse_square")),
        ),
    )
    path = str(tmp_path / "audio_config.json")
    config.save_as_json(path, lock_config=False)
    loaded = EdenConfig.load_from_file(path).sensors_options

    assert isinstance(loaded.mic.sensor, SpatialAudio) and loaded.mic.sensor.attenuation == "inverse_square"
    motors = loaded.motors.sensor
    assert isinstance(motors, ActuationAudio) and tuple(motors.joints) == ("joint1", "joint2")
    override = motors.properties["joint2"]
    assert isinstance(override, ActuationAudioProperties) and override.pitch_slope == 999.0
