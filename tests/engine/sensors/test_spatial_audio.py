"""``eden.engine.sensors.spatial_audio``: the microphone's attenuation, propagation delay and link attachment."""

from __future__ import annotations

import math

import genesis as gs
import pytest
import torch

from eden.engine.sensors import SpatialAudio
from eden.options import BoxOptions, EnvOptions, SceneOptions
from eden.options.sensors import SensorOptions, SensorsOptions
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env

SIM_DT = 0.01
K = 20
SPEED_OF_SOUND = 343.0
BLOCK_DISTANCE = K * (SIM_DT / K) * SPEED_OF_SOUND  # 3.43 m of propagation per block
IMPULSE_DISTANCE = 5.0


def _mic(pos, **kwargs):
    return SensorOptions(sensor=SpatialAudio(audio_substeps=K, speed_of_sound=SPEED_OF_SOUND, pos_offset=pos, **kwargs))


@pytest.fixture(scope="module")
def field(probe_source, audio_context):
    """Four microphones at 0.4 / 0.8 m from a constant source, and one at the origin for a distant impulse.

    The constant source at the origin emits 1 for three steps, then stops; the impulse (1000, 5 m away) fires on
    step 4. Yields ``(env, heard)`` with each step's microphone blocks, ``heard[step][name]``.
    """
    env = build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=1, sim_dt=SIM_DT, decimation=1),
            scene_options=SceneOptions(
                box=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.0, 3.0, 0.05), fixed=True)
            ),
            sensors_options=SensorsOptions(
                inverse_near=_mic((0.4, 0.0, 0.0), attenuation="inverse"),
                inverse_far=_mic((0.8, 0.0, 0.0), attenuation="inverse"),
                square_near=_mic((-0.4, 0.0, 0.0), attenuation="inverse_square"),
                square_far=_mic((-0.8, 0.0, 0.0), attenuation="inverse_square"),
                origin=_mic((0.0, 0.0, 0.0), enable_doppler=False),
            ),
        )
    )
    context = audio_context(env)
    context.register(probe_source(env, [1.0] * 3), "constant")
    context.register(probe_source(env, [0.0] * 3 + [1000.0], pos=(IMPULSE_DISTANCE, 0.0, 0.0)), "impulse")
    heard = []
    for _ in range(5):
        env.step()
        heard.append({name: sensor.read().reshape(-1).clone() for name, sensor in env.sensors.items()})
    yield env, heard
    env.destroy()


@pytest.mark.parametrize(("attenuation", "power"), [("inverse", 1), ("square", 2)])
def test_distance_attenuation_follows_the_falloff_law(field, attenuation, power):
    """Past ``ref_distance`` (0.1 m) the steady-state gain is ``(ref / r) ** power``.

    Three steps fill the delay line with the constant, so these are gains rather than zero-padded history.
    """
    _, heard = field
    near = float(heard[2][f"{attenuation}_near"].abs().max())
    far = float(heard[2][f"{attenuation}_far"].abs().max())
    assert near == pytest.approx((0.1 / 0.4) ** power, rel=1e-4)
    assert far == pytest.approx((0.1 / 0.8) ** power, rel=1e-4)


def test_propagation_delay_arrives_in_a_later_block(field):
    """Sound farther than one block's flight distance is heard on a later step, not the one that emitted it."""
    assert IMPULSE_DISTANCE > BLOCK_DISTANCE, "test premise: the impulse must be more than one block away"
    _, heard = field
    assert float(heard[3]["origin"].abs().max()) == pytest.approx(0.0, abs=1e-6), "arrived too early"
    assert float(heard[4]["origin"].abs().max()) > 0.0, "never arrived"


def test_scene_read_sensors_cannot_name_an_out_of_tree_sensor(field):
    """Pin a Genesis limitation the engine-addons guide documents: ``scene.read_sensors()`` raises.

    It labels results through ``genesis.options.sensors.types``, which only resolves Genesis-defined names. When a
    Genesis bump makes this pass, drop the caveat from ``docs/guides/engine-addons.md`` and this test.
    """
    env, _ = field
    with pytest.raises(AttributeError, match="SpatialAudio"):
        env.scene.read_sensors()


def test_attached_microphone_rides_its_link_with_a_link_local_offset(probe_source, audio_context):
    """An attached microphone hears from where Eden reports it, and ``pos_offset`` is in the link's frame.

    A source placed exactly at the reported position is heard at unit gain; were the offset added in the world
    frame, the two would sit apart on the tilted box and the gain would drop. Moving the offset 0.1 m away at
    ``ref_distance=0.01`` then gives a gain of 0.1.
    """
    tilt = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)  # 90 deg about x
    offset = (0.0, 0.0, 0.3)
    env = build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=1, sim_dt=SIM_DT, decimation=1),
            scene_options=SceneOptions(
                box=BoxOptions(
                    size=(0.1, 0.1, 0.1), default_root_pos=(0.0, 0.0, 0.5), default_root_quat=tilt, fixed=True
                )
            ),
            sensors_options=SensorsOptions(
                mic=SensorOptions(
                    sensor=SpatialAudio(audio_substeps=K, pos_offset=offset, ref_distance=0.01),
                    attach_entity_name="box",
                )
            ),
        )
    )
    try:
        mic = env.sensors["mic"]
        mic_pos = mic.get_pos()[0]
        link_pos = env.entities["box"].links[0].get_pos(relative=False).reshape(3)
        assert (mic_pos - (link_pos + torch.tensor(offset, device=env.device))).norm() > 0.1

        audio_context(env).register(probe_source(env, [1.0] * 8, pos=tuple(mic_pos.tolist())), "probe")
        env.step()
        assert float(mic.read().reshape(-1)[-1]) == pytest.approx(1.0, rel=1e-3)

        mic.set_pos_offset((0.0, 0.0, 0.4))
        env.step()
        assert float((mic.get_pos()[0] - mic_pos).norm()) == pytest.approx(0.1, abs=1e-4)
        assert float(mic.read().reshape(-1)[-1]) == pytest.approx(0.1, rel=1e-3)
    finally:
        env.destroy()


@pytest.mark.parametrize(
    "kwargs", [{"attenuation": "inverse_cube"}, {"speed_of_sound": -1.0}, {"audio_substeps": 0}, {"mispelled_field": 1}]
)
def test_spatial_audio_options_reject_nonsense(kwargs):
    """Options validation catches a bad value where it is written."""
    with pytest.raises(gs.GenesisException):
        SpatialAudio(**kwargs)
