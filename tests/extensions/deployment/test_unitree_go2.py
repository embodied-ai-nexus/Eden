"""``UnitreeGo2Deployment`` constructed on a built Go2 env, over a fake ``unitree_sdk2py`` DDS transport.

:class:`FakeGo2` is the robot on the other end of ``rt/lowstate`` / ``rt/lowcmd``: it publishes the
low state it is given and records every command packet. It is installed where the backend, the
shared DDS transport and ``eden.extensions.deployment.utils.unitree_sdk2`` bind the SDK; nothing on the Eden
side is faked.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from eden.extensions.deployment import resolve_deployment_for
from eden.extensions.deployment import unitree_dds
from eden.extensions.deployment import unitree_go2 as go2mod
from eden.extensions.deployment.unitree_go2 import GO2_JOINT_IDX, UnitreeGo2Deployment
from eden.extensions.deployment.utils import unitree_sdk2
from eden.extensions.deployment.utils.state import RobotCommand
from eden.options.robots.unitree_go2 import Go2, Go2W
from tests.support.deploy import assert_dofs_looked_up_by_name, deployments, robot_env

DOFS_NAME = list(Go2().dofs_name)
#: Where each Eden-order DOF lives on the wire, from the documented layout (leg-major FR, FL, RR, RL).
SDK_LAYOUT = {
    f"{leg}_{joint}_joint": 3 * i + j
    for i, leg in enumerate(("FR", "FL", "RR", "RL"))
    for j, joint in enumerate(("hip", "thigh", "calf"))
}
SDK_SLOT = [SDK_LAYOUT[name] for name in DOFS_NAME]


class TransportDied(BaseException):
    """What the fake raises to kill the republisher: past the loop's per-send ``except Exception``."""


# ---------------------------------------------------------------------------
# the fake robot and its DDS SDK
# ---------------------------------------------------------------------------


class FakeGo2:
    """A Go2 on the DDS bus: 20 motor slots it reports and every ``LowCmd_`` it received."""

    def __init__(self):
        self.q = np.zeros(20)
        self.streams = True
        self.interface: str | None = None
        self.topics: list[str] = []
        self.written: list[dict] = []
        self.crc_calls = 0
        self.die_off_main_thread = False
        self._on_state = None

    def publish_state(self) -> None:
        motors = [SimpleNamespace(q=float(q), dq=0.5 * float(q), tau_est=-float(q)) for q in self.q]
        imu = SimpleNamespace(
            quaternion=(1.0, 0.0, 0.0, 0.0), gyroscope=(0.1, 0.2, 0.3), accelerometer=(0.0, 0.0, 9.81)
        )
        self._on_state(SimpleNamespace(motor_state=motors, imu_state=imu))

    def install(self, monkeypatch) -> None:
        robot = self

        class Subscriber:
            def __init__(self, topic, msg_type):
                robot.topics.append(topic)

            def Init(self, callback, depth):  # noqa: N802 — the SDK's spelling
                robot._on_state = callback
                if robot.streams:
                    robot.publish_state()

        class Publisher:
            def __init__(self, topic, msg_type):
                robot.topics.append(topic)

            def Init(self):  # noqa: N802
                pass

            def Write(self, cmd):  # noqa: N802
                if robot.die_off_main_thread and threading.current_thread() is not threading.main_thread():
                    raise TransportDied("DDS participant gone")
                robot.written.append(
                    {
                        "q": [m.q for m in cmd.motor_cmd],
                        "kp": [m.kp for m in cmd.motor_cmd],
                        "kd": [m.kd for m in cmd.motor_cmd],
                        "crc": cmd.crc,
                        "thread": threading.current_thread() is threading.main_thread(),
                    }
                )

        class Crc:
            def Crc(self, cmd):  # noqa: N802
                robot.crc_calls += 1
                return robot.crc_calls

        def low_cmd():
            motor = dict(mode=0, q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0)
            return SimpleNamespace(
                head=[0, 0], level_flag=0, gpio=0, crc=0, motor_cmd=[SimpleNamespace(**motor) for _ in range(20)]
            )

        def channel_factory_initialize(domain, interface):
            robot.interface = interface

        monkeypatch.setattr(unitree_sdk2, "IS_UNITREE_SDK2PY_AVAILABLE", True)
        monkeypatch.setattr(unitree_sdk2, "ChannelFactoryInitialize", channel_factory_initialize)
        for name, value in (("ChannelSubscriber", Subscriber), ("ChannelPublisher", Publisher), ("CRC", Crc)):
            monkeypatch.setattr(unitree_dds, name, value)
        for name, value in (("unitree_go_msg_dds__LowCmd_", low_cmd), ("LowState_go", object), ("LowCmd_go", object)):
            monkeypatch.setattr(go2mod, name, value)

    def sent_eden(self, field: str, packet: int = -1) -> np.ndarray:
        """``field`` of one received packet, gathered into Eden DOF order."""
        return np.array([self.written[packet][field][slot] for slot in SDK_SLOT])


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def env():
    return robot_env(Go2())


@pytest.fixture
def robot(monkeypatch):
    fake = FakeGo2()
    fake.install(monkeypatch)
    return fake


@pytest.fixture
def backend(env, robot):
    """Return ``backend(connect=True, **options)``: a real ``UnitreeGo2Deployment`` on ``env``, undone after the test."""
    with deployments(env) as make:
        yield lambda connect=True, **options: make(
            UnitreeGo2Deployment,
            connect=connect,
            **{"network_interface": "lo", "publish_rate": None, "init_steps": 4, **options},
        )


def _payload(pos, kp=40.0, kd=1.0) -> RobotCommand:
    return RobotCommand(
        dofs_pos=np.broadcast_to(np.asarray(pos, float), (12,)).copy(),
        dofs_vel=np.zeros(12),
        dofs_torque=np.zeros(12),
        dofs_kp=np.full(12, kp),
        dofs_kd=np.full(12, kd),
    )


# ---------------------------------------------------------------------------
# binding and motor map
# ---------------------------------------------------------------------------


def test_joint_table_follows_the_documented_sdk_layout():
    """Kept separate from ``GO1_JOINT_IDX`` on purpose: the layout belongs to each robot's firmware."""
    assert GO2_JOINT_IDX == SDK_LAYOUT
    assert set(GO2_JOINT_IDX) == set(DOFS_NAME)
    assert SDK_SLOT != list(range(12)), "Eden order is joint-major, the SDK leg-major: the remap is load-bearing"


def test_go2w_reaches_this_backend_through_the_mro_and_is_refused_at_construction():
    """The Go2W's wheel motors have no confirmed SDK mapping, so guessing them is refused."""
    assert resolve_deployment_for(Go2W()) is UnitreeGo2Deployment
    go2w_env = robot_env(Go2W())
    with pytest.raises(NotImplementedError, match="no confirmed SDK motor index"):
        UnitreeGo2Deployment(go2w_env, UnitreeGo2Deployment.configure())


# ---------------------------------------------------------------------------
# connect / read / send
# ---------------------------------------------------------------------------


def test_connect_joins_the_bus_and_waits_for_the_first_low_state(backend, robot):
    backend()
    assert robot.interface == "lo"
    assert robot.topics == ["rt/lowstate", "rt/lowcmd"]
    assert robot.written == [], "nothing may go on the wire before the first command"


def test_connect_times_out_when_the_robot_never_publishes(backend, robot):
    robot.streams = False
    with pytest.raises(RuntimeError, match="Timed out waiting for Unitree Go2 low state"):
        backend(connect_timeout_s=0.1)


def test_read_state_gathers_sdk_slots_into_eden_order(backend, robot):
    dep = backend()
    robot.q[:] = np.arange(20)
    robot.publish_state()
    state = dep.read_state()
    np.testing.assert_array_equal(state.dofs_pos, SDK_SLOT)
    np.testing.assert_array_equal(state.dofs_vel, 0.5 * np.array(SDK_SLOT))
    np.testing.assert_array_equal(state.full_dofs_pos, np.arange(12))


def test_send_payload_low_passes_the_targets_onto_their_sdk_slots(backend, robot):
    """``--target_lpf`` smooths every command path: ``sent = a*new + (1-a)*prev``, warm-started."""
    dep = backend(target_lpf=0.5)
    dep.send_payload(_payload(1.0))
    dep.send_payload(_payload(3.0))
    np.testing.assert_allclose(robot.sent_eden("q", 0), 1.0)  # no start-up transient
    np.testing.assert_allclose(robot.sent_eden("q", 1), 2.0)
    assert [p["crc"] for p in robot.written] == [1, 2], "each packet carries its own CRC"


def test_it_refuses_to_send_before_connect(backend):
    with pytest.raises(RuntimeError, match="not connected"):
        backend(connect=False).send_payload(_payload(0.0))


# ---------------------------------------------------------------------------
# stand-up ramp, reset, shutdown
# ---------------------------------------------------------------------------


def test_init_ramps_from_the_measured_pose_instead_of_snapping(backend, robot):
    """The first INIT tick used to command the default pose outright, at full gains, from anywhere."""
    robot.q[:] = -1.0
    dep = backend(init_dofs_kp_scale=0.5, init_dofs_kd_scale=2.0)
    robot.publish_state()
    dep.reset()
    dep.request_init()
    first_packet = len(robot.written)
    target = dep.default_dof_pos
    for tick in range(6):
        dep.step(None)
        if tick == 1:
            robot.q[:] = 0.7  # the start pose is captured once per ramp, not re-read each tick
            robot.publish_state()

    for tick in range(6):
        alpha = min(tick / 4, 1.0)
        np.testing.assert_allclose(robot.sent_eden("q", first_packet + tick), -1.0 + (target + 1.0) * alpha)
        np.testing.assert_allclose(robot.sent_eden("kp", first_packet + tick), dep.default_dof_kp * 0.5)
        np.testing.assert_allclose(robot.sent_eden("kd", first_packet + tick), dep.default_dof_kd * 2.0)


def test_every_init_ramps_from_where_the_robot_is_now(backend, robot):
    dep = backend()
    robot.publish_state()
    dep.reset()
    dep.request_init()
    for _ in range(dep.init_steps + 2):
        dep.step(None)
    robot.q[:] = 0.3  # knocked into another pose between episodes
    robot.publish_state()
    dep.reset()
    dep.request_init()
    first_packet = len(robot.written)
    dep.step(None)
    np.testing.assert_allclose(robot.sent_eden("q", first_packet), 0.3)


def test_close_ramps_to_damping_and_then_stops_feeding_the_robot(backend, robot):
    dep = backend(publish_rate=500.0, shutdown_damping_steps=3, shutdown_damping_kd_scale=0.5)
    dep.send_payload(_payload(0.25))
    time.sleep(0.05)
    dep.close()
    np.testing.assert_array_equal(robot.sent_eden("kp"), np.zeros(12))
    np.testing.assert_allclose(robot.sent_eden("kd"), dep.default_dof_kd * 0.5)

    after_close = len(robot.written)
    time.sleep(0.05)
    assert len(robot.written) == after_close, "the republisher outlived close()"
    with pytest.raises(RuntimeError, match="not connected"):
        dep.send_payload(_payload(0.0))


# ---------------------------------------------------------------------------
# the 500 Hz republisher
# ---------------------------------------------------------------------------


def test_publish_rate_defaults_to_unitrees_own_low_level_rate():
    """``go2_stand_example.py``: ``dt = 0.002`` and a 2 ms writer thread."""
    assert UnitreeGo2Deployment.configure().publish_rate == 500.0


def test_the_republisher_resends_the_held_command_verbatim_between_ticks(backend, robot):
    """The Go2 ``LowCmd_`` carries no timestamp, so the packet ``send_payload`` built stays valid as is."""
    dep = backend(publish_rate=500.0)
    time.sleep(0.05)
    assert robot.written == [], "the limp initial packet started flowing before any command"

    dep.send_payload(_payload(0.25))
    time.sleep(0.2)
    resent = [p for p in robot.written if not p["thread"]]
    assert len(resent) >= 40, len(resent)  # 100 nominal; a floor for a loaded box
    assert {p["crc"] for p in resent} == {1} and robot.crc_calls == 1, "a re-send recomputed the CRC"
    assert all(q == 0.25 for p in resent for q in (p["q"][slot] for slot in SDK_SLOT))


def test_a_dead_republisher_says_so_without_predicting_a_parallel_joint_ring(backend, robot, eden_caplog):
    """The mixin's default text is the Oli/G1 fault; the Go2 has no parallel joints to ring."""
    robot.die_off_main_thread = True
    dep = backend(publish_rate=500.0)
    dep.send_payload(_payload(0.25))
    time.sleep(0.1)
    errors = [r.message for r in eden_caplog.records if r.levelname == "ERROR"]
    assert errors and "Republisher thread died" in errors[0] and "DDS participant gone" in errors[0]
    assert "parallel" not in errors[0] and "500 Hz low-level rate" in errors[0]


def test_dofs_are_looked_up_by_name_beside_the_sdk_motor_map(backend):
    assert_dofs_looked_up_by_name(backend(), Go2())
