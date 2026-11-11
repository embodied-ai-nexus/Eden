"""``eden.entities.sensor``: sensor pose, offsets, named-link resolution and reads, on one built scene."""

from __future__ import annotations

import math

import genesis as gs
import pytest
import torch
from genesis.utils.geom import transform_by_quat

from eden.entities.sensor import Sensor
from eden.options import BoxOptions, EnvOptions, SceneOptions, SphereOptions
from eden.options.sensors import SensorOptions, SensorsOptions
from eden.utils.configs import EdenConfig
from eden.utils.geom import quat_mul
from tests.support.envs import build_env

YAW90 = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
IDENTITY = (1.0, 0.0, 0.0, 0.0)


@pytest.fixture(scope="module")
def env():
    """Two envs: a falling box carrying most sensors, two spheres a probe tracks, and an obstacle a contact ignores.

    ``late`` is added to the config after construction, as a caller extending a task's sensors would.
    """
    config = EdenConfig(
        env_options=EnvOptions(num_envs=2, sim_dt=0.005),
        scene_options=SceneOptions(
            robot=BoxOptions(size=(0.2, 0.2, 0.2), default_root_pos=(0.0, 0.0, 1.0)),
            sphere1=SphereOptions(radius=0.1, default_root_pos=(1.0, 0.0, 0.5)),
            sphere2=SphereOptions(radius=0.2, default_root_pos=(-1.0, 0.0, 0.5)),
            obstacle=BoxOptions(size=(0.2, 0.2, 0.2), fixed=True, default_root_pos=(0.0, 2.0, 0.1)),
        ),
        sensors_options=SensorsOptions(
            imu=SensorOptions(sensor=gs.sensors.IMU(pos_offset=(0.1, 0.2, 0.3)), attach_entity_name="robot"),
            imu_yawed=SensorOptions(sensor=gs.sensors.IMU(euler_offset=(0.0, 0.0, 90.0)), attach_entity_name="robot"),
            noisy_imu=SensorOptions(
                sensor=gs.sensors.IMU(acc_noise=(0.1, 0.1, 0.1), gyro_noise=(0.1, 0.1, 0.1)),
                attach_entity_name="robot",
            ),
            static=SensorOptions(sensor=gs.sensors.Contact(pos_offset=(1.0, 2.0, 3.0))),
            prox=SensorOptions(
                sensor=gs.sensors.SurfaceDistanceProbe(
                    probe_local_pos=[(0.0, 0.0, 0.0)], probe_radius=1.0, track_link_idx=(0,)
                ),
                attach_entity_name="robot",
                attach_link_name="box_baselink",
                track_link_names=["sphere1", "sphere2"],  # a bare entity name tracks every link of it
            ),
            force=SensorOptions(
                sensor=gs.sensors.ContactForce(), attach_entity_name="robot", filter_link_names=["obstacle"]
            ),
            depth=SensorOptions(
                sensor=gs.sensors.DepthCamera(pattern=gs.sensors.DepthCameraPattern(), return_points=False),
                attach_entity_name="robot",
            ),
        ),
    )
    config.sensors_options.late = SensorOptions(sensor=gs.sensors.Contact(draw_debug=True))
    env = build_env(config)
    env.step()
    yield env
    env.destroy()


def _link_pose(env, sensor_name):
    link = env.sensors[sensor_name]._sensor._link  # the Genesis link the sensor rides; Eden exposes no handle
    return link.get_pos(), link.get_quat()


def test_is_attached_reports_the_mount(env):
    assert env.sensors["imu"].is_attached is True
    assert env.sensors["static"].is_attached is False


def test_attached_sensor_sits_at_its_link_plus_the_rotated_offset(env):
    sensor = env.sensors["imu"]
    link_pos, link_quat = _link_pose(env, "imu")
    offset = torch.tensor([0.1, 0.2, 0.3], device=env.device).expand_as(link_pos)
    torch.testing.assert_close(sensor.get_pos(), link_pos + transform_by_quat(offset, link_quat), atol=1e-4, rtol=0)
    idx = torch.tensor([1], device=env.device)
    torch.testing.assert_close(sensor.get_pos(envs_idx=idx), sensor.get_pos()[idx])


@pytest.mark.parametrize(("name", "offset"), [("imu", IDENTITY), ("imu_yawed", YAW90)])
def test_attached_sensor_orientation_is_link_then_offset(env, name, offset):
    """``link ∘ offset``: the reversed composition, or a dropped offset, would fail the yawed case."""
    _, link_quat = _link_pose(env, name)
    expected = quat_mul(link_quat, torch.tensor(offset, device=env.device).expand_as(link_quat))
    torch.testing.assert_close(env.sensors[name].get_quat(), expected, atol=1e-4, rtol=0)


def test_static_sensor_sits_at_its_offset_in_every_env(env):
    sensor = env.sensors["static"]
    torch.testing.assert_close(sensor.get_pos().cpu(), torch.tensor([[1.0, 2.0, 3.0]] * 2))
    torch.testing.assert_close(sensor.get_quat().cpu(), torch.tensor([IDENTITY] * 2))


def test_set_offsets_move_an_attached_sensor(env):
    """A written offset reaches ``get_pos``/``get_quat`` after the next step; restored afterwards."""
    sensor = env.sensors["imu"]
    pos_offset = torch.tensor([0.5, 0.5, 0.5], device=env.device)
    yaw90 = torch.tensor(YAW90, device=env.device)
    try:
        sensor.set_pos_offset(pos_offset)
        sensor.set_quat_offset(yaw90)
        env.step()
        link_pos, link_quat = _link_pose(env, "imu")
        expected_pos = link_pos + transform_by_quat(pos_offset.expand_as(link_pos), link_quat)
        torch.testing.assert_close(sensor.get_pos(), expected_pos, atol=1e-4, rtol=0)
        torch.testing.assert_close(
            sensor.get_quat(), quat_mul(link_quat, yaw90.expand_as(link_quat)), atol=1e-4, rtol=0
        )
    finally:
        sensor.set_pos_offset(torch.tensor([0.1, 0.2, 0.3], device=env.device))
        sensor.set_quat_offset(torch.tensor(IDENTITY, device=env.device))


@pytest.mark.parametrize(
    "setter",
    [lambda s: s.set_pos_offset(torch.zeros(3)), lambda s: s.set_quat_offset(torch.tensor(IDENTITY))],
    ids=["set_pos_offset", "set_quat_offset"],
)
def test_a_static_sensor_refuses_offsets(env, setter):
    with pytest.raises(RuntimeError, match="static"):
        setter(env.sensors["static"])


def test_named_links_resolve_to_the_genesis_link_indices(env):
    """``track_link_names`` / ``filter_link_names`` land on the Genesis sensor as solver-global link indices."""
    spheres = [env.entities[name].get_link("sphere_baselink").idx for name in ("sphere1", "sphere2")]
    assert env.sensors["prox"]._options.sensor.track_link_idx == tuple(spheres)
    obstacle = env.entities["obstacle"].get_link("box_baselink").idx
    assert env.sensors["force"]._options.sensor.filter_link_idx == (obstacle,)


def test_a_sensor_added_after_config_construction_is_built(env):
    assert env.sensors["late"]._options.sensor.draw_debug is True
    assert env.sensors["static"]._options.sensor.draw_debug is False


def test_read_ground_truth_is_the_noiseless_channel(env):
    sensor = env.sensors["noisy_imu"]
    noisy, truth = sensor.read(), sensor.read_ground_truth()
    assert not torch.allclose(noisy.lin_acc, truth.lin_acc)


def test_distance_only_depth_camera_reads_no_points(env):
    data = env.sensors["depth"].read()
    assert data.points is None
    assert data.distances is not None


def test_a_sensor_copies_its_mutable_options():
    options = SensorOptions(sensor=gs.sensors.Contact(), filter_link_names=["robot/foot"])
    sensor = Sensor(None, options)
    assert sensor.filter_link_names == ["robot/foot"] and sensor.filter_link_names is not options.filter_link_names


def test_sensor_class_defaults_match_the_options_defaults_that_shadow_them():
    """The options value always wins, so a drifted class default would never be observed."""
    drifted = []
    for name in sorted(Sensor.get_parameter_names()):
        field = SensorOptions.model_fields.get(name)
        if field is None or field.is_required():
            continue
        if getattr(Sensor, name, "<missing>") != field.get_default(call_default_factory=True):
            drifted.append(name)
    assert not drifted, f"Sensor defaults drifted from SensorOptions: {drifted}"
