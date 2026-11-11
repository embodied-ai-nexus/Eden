"""The sensor observation term in ``eden/managers/terms/observations/sensors.py``: ``SensorRead``."""

from __future__ import annotations

import genesis as gs
import pytest
import torch

from eden.managers.terms.observations.sensors import SensorRead
from eden.options import BoxOptions, EnvOptions, SceneOptions
from eden.options.sensors import SensorOptions, SensorsOptions
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env

pytestmark = pytest.mark.requires_genesis


def _probe(z):
    return SensorOptions(
        sensor=gs.sensors.SurfaceDistanceProbe(probe_local_pos=[(0.0, 0.0, z)], track_link_idx=(0,), probe_radius=2.0),
        attach_entity_name="probe",
        track_link_names=["surface"],
    )


def _term(env, **params):
    term = SensorRead(env, options=SensorRead.configure(**params))
    term.build()
    return term


@pytest.fixture(scope="module")
def scene():
    """A probe box falling onto a fixed surface, and apart, a box coming to rest on a fixed base.

    Yields ``(env, early, late)``: the probe's reading after 5 and after 45 steps. The scene is then stepped to
    200, so the top box rests on its base.
    """
    env = build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=2),
            scene_options=SceneOptions(
                probe=BoxOptions(size=(0.05, 0.05, 0.05), default_root_pos=(0.0, 0.0, 1.0)),
                surface=BoxOptions(size=(0.5, 0.5, 0.1), default_root_pos=(0.0, 0.0, 0.05), fixed=True),
                top=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(2.0, 0.0, 0.6)),
                bottom=BoxOptions(size=(0.5, 0.5, 0.1), default_root_pos=(2.0, 0.0, 0.05), fixed=True),
            ),
            sensors_options=SensorsOptions(
                prox_a=_probe(0.0),
                prox_b=_probe(0.01),
                force=SensorOptions(sensor=gs.sensors.ContactForce(), attach_entity_name="top"),
                imu=SensorOptions(
                    sensor=gs.sensors.IMU(acc_noise=(0.1, 0.1, 0.1), gyro_noise=(0.1, 0.1, 0.1)),
                    attach_entity_name="top",
                ),
            ),
        )
    )
    term = _term(env, sensor_names=["prox_a"])
    for _ in range(5):
        env.scene.step()
    early = term.compute().clone()
    for _ in range(40):
        env.scene.step()
    late = term.compute().clone()
    for _ in range(155):
        env.scene.step()
    yield env, early, late
    env.destroy()


def _imu_cat(reading):
    return torch.cat([reading.lin_acc, reading.ang_vel, reading.mag], dim=-1)


def test_reads_live_data_as_the_probe_approaches(scene):
    _, early, late = scene
    assert (late < early).all(), (early, late)
    assert (late >= 0).all()


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (["prox_a", "prox_b"], lambda env: torch.cat([env.sensors["prox_a"].read(), env.sensors["prox_b"].read()], -1)),
        (["imu"], lambda env: _imu_cat(env.sensors["imu"].read())),  # a NamedTuple read: its fields concatenated
        (
            ["imu", "force"],
            lambda env: torch.cat([_imu_cat(env.sensors["imu"].read()), env.sensors["force"].read()], -1),
        ),
        (
            None,  # every sensor, in registration order
            lambda env: torch.cat(
                [env.sensors[n].read() for n in ("prox_a", "prox_b", "force")] + [_imu_cat(env.sensors["imu"].read())],
                -1,
            ),
        ),
    ],
    ids=["two_probes", "namedtuple", "heterogeneous", "all"],
)
def test_concatenates_the_selected_sensors_readings(scene, names, expected):
    env, _, _ = scene
    torch.testing.assert_close(_term(env, sensor_names=names).compute(), expected(env))


def test_read_ground_truth_selects_the_noiseless_channel(scene):
    env, _, _ = scene
    out = _term(env, sensor_names=["imu"], read_ground_truth=True).compute()
    torch.testing.assert_close(out, _imu_cat(env.sensors["imu"].read_ground_truth()))


def test_post_process_reduces_each_reading(scene):
    """A contact-force magnitude: positive once the top box rests on its base (gravity's normal force)."""
    env, _, _ = scene
    out = _term(env, sensor_names=["force"], post_process_func=lambda t: torch.linalg.norm(t, dim=-1, keepdim=True))
    out = out.compute()
    assert out.shape == (2, 1)
    assert (out > 0).all(), out


def test_unknown_sensor_name_fails_at_build(scene):
    env, _, _ = scene
    with pytest.raises(ValueError, match="Not all query patterns are matched"):
        _term(env, sensor_names=["nope"])
