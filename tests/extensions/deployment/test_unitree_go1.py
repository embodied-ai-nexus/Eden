"""``UnitreeGo1Deployment`` constructed on a built Go1 env, talking to a fake legacy ``robot_interface`` SDK.

The SDK is the boundary: :class:`FakeGo1` is a small stateful robot (motor states it reports, every
packet it received), installed where the backend binds the module. Everything on the Eden side
-- env, entity swap, gains, ramps, recording -- is the real thing.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from eden.extensions.deployment import resolve_deployment_for
from eden.extensions.deployment import unitree_go1 as go1mod
from eden.extensions.deployment.unitree_go1 import GO1_JOINT_IDX, UnitreeGo1Deployment
from eden.extensions.deployment.unitree_go2 import UnitreeGo2Deployment
from eden.extensions.deployment.utils import unitree_go1_sdk
from eden.extensions.deployment.utils.record import load_deploy_record
from eden.extensions.deployment.utils.state import RobotCommand
from eden.options.robots.unitree_go1 import Go1
from tests.support.deploy import assert_dofs_looked_up_by_name, deployments, robot_env

DOFS_NAME = list(Go1().dofs_name)
#: Where each Eden-order DOF lives on the wire, from the documented layout (leg-major FR, FL, RR, RL).
SDK_LAYOUT = {
    f"{leg}_{joint}_joint": 3 * i + j
    for i, leg in enumerate(("FR", "FL", "RR", "RL"))
    for j, joint in enumerate(("hip", "thigh", "calf"))
}
SDK_SLOT = [SDK_LAYOUT[name] for name in DOFS_NAME]


# ---------------------------------------------------------------------------
# the fake robot and its SDK
# ---------------------------------------------------------------------------


class FakeGo1:
    """A Go1 control board: 20 motor slots it reports, and every command packet it was sent.

    It streams state only to a client it has heard from, as the real board does.
    """

    def __init__(self):
        self.q = np.zeros(20)
        self.dq = np.zeros(20)
        self.tau = np.zeros(20)
        self.temperature = np.full(20, 25)
        self.mode = np.full(20, 0x0A)
        self.streams = True
        self.heard = False
        self.sent: list = []
        self.protected: list[tuple[int, float]] = []  # (factor, the q[0] the state it was given held)

    def sdk(self):
        robot = self

        class MotorCmd:
            def __init__(self):
                self.mode, self.q, self.dq, self.Kp, self.Kd, self.tau = 0, 0.0, 0.0, 0.0, 0.0, 0.0

        class MotorState:
            def __init__(self):
                self.q = self.dq = self.tauEst = 0.0
                self.temperature, self.mode = 0, 0

        class LowCmd:
            def __init__(self):
                self.motorCmd = [MotorCmd() for _ in range(20)]

        class LowState:
            def __init__(self):
                self.tick = 0
                self.motorState = [MotorState() for _ in range(20)]
                self.imu = type(
                    "IMU",
                    (),
                    {
                        "quaternion": [1.0, 0.0, 0.0, 0.0],
                        "gyroscope": [0.1, 0.2, 0.3],
                        "accelerometer": [0.0, 0.0, 9.81],
                    },
                )()
                self.bms = type("BMS", (), {"SOC": 95, "cell_vol": [4100] * 10})()  # 10 cells x 4.1 V

        class UDP:
            def __init__(self, level, local_port, ip, port):
                self._cmd = None

            def InitCmdData(self, cmd):  # noqa: N802 — the SDK's spelling
                pass

            def Recv(self):  # noqa: N802
                pass

            def GetRecv(self, state):  # noqa: N802
                state.tick = 12345 if (robot.heard and robot.streams) else 0
                for i, m in enumerate(state.motorState):
                    m.q, m.dq, m.tauEst = float(robot.q[i]), float(robot.dq[i]), float(robot.tau[i])
                    m.temperature, m.mode = int(robot.temperature[i]), int(robot.mode[i])

            def SetSend(self, cmd):  # noqa: N802
                self._cmd = cmd

            def Send(self):  # noqa: N802
                robot.sent.append(copy.deepcopy(self._cmd.motorCmd))
                robot.heard = True

        class Safety:
            def __init__(self, legged_type):
                pass

            def PowerProtect(self, cmd, state, factor):  # noqa: N802
                robot.protected.append((factor, state.motorState[0].q))

        return type(
            "robot_interface",
            (),
            {
                "UDP": UDP,
                "Safety": Safety,
                "LowCmd": LowCmd,
                "LowState": LowState,
                "LeggedType": type("LeggedType", (), {"Go1": "Go1"}),
            },
        )

    def sent_eden(self, field: str, packet: int = -1) -> np.ndarray:
        """``field`` of one received packet, gathered into Eden DOF order."""
        return np.array([getattr(self.sent[packet][slot], field) for slot in SDK_SLOT])


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def env():
    return robot_env(Go1())


@pytest.fixture
def robot(monkeypatch):
    fake = FakeGo1()
    monkeypatch.setattr(go1mod, "sdk", fake.sdk())
    monkeypatch.setattr(unitree_go1_sdk, "IS_GO1_SDK_AVAILABLE", True)
    return fake


@pytest.fixture
def backend(env, robot):
    """Return ``backend(connect=True, **options)``: a real ``UnitreeGo1Deployment`` on ``env``, undone after the test."""
    with deployments(env) as make:
        yield lambda connect=True, **options: make(
            UnitreeGo1Deployment, connect=connect, **{"init_steps": 10, "shutdown_damping_steps": 3, **options}
        )


def _payload(pos, kp=None, kd=None) -> RobotCommand:
    return RobotCommand(
        dofs_pos=np.asarray(pos, float), dofs_vel=np.zeros(12), dofs_torque=np.zeros(12), dofs_kp=kp, dofs_kd=kd
    )


# ---------------------------------------------------------------------------
# binding and motor map
# ---------------------------------------------------------------------------


def test_go1_resolves_to_this_backend():
    assert resolve_deployment_for(Go1()) is UnitreeGo1Deployment


def test_joint_table_follows_the_documented_sdk_layout():
    """A wrong value drives the wrong motor with no error; the keys must be exactly the robot's joints."""
    assert GO1_JOINT_IDX == SDK_LAYOUT
    assert set(GO1_JOINT_IDX) == set(DOFS_NAME)
    assert SDK_SLOT != list(range(12)), "Eden order is joint-major, the SDK leg-major: the remap is load-bearing"


# ---------------------------------------------------------------------------
# connect / read / send
# ---------------------------------------------------------------------------


def test_connect_primes_the_link_with_a_limp_command(backend, robot):
    """The board streams only to a client it has heard from, so connect must SEND while it waits."""
    backend()
    assert robot.sent, "connect never sent, so the robot would never start streaming"
    assert all(m.Kp == m.Kd == m.tau == 0.0 for packet in robot.sent for m in packet)
    assert all(packet[slot].mode == 0x0A for packet in robot.sent for slot in range(12))  # servo mode


def test_connect_times_out_when_the_robot_never_streams(backend, robot):
    robot.streams = False
    with pytest.raises(RuntimeError, match="Timed out"):
        backend(connect_timeout_s=0.05)


def test_read_state_gathers_sdk_slots_into_eden_order(backend, robot):
    robot.q[:] = np.arange(20)
    robot.dq[:] = 100 + np.arange(20)
    robot.tau[:] = 200 + np.arange(20)
    robot.temperature[:] = 30 + np.arange(20)
    state = backend().read_state()

    np.testing.assert_array_equal(state.dofs_pos, SDK_SLOT)
    np.testing.assert_array_equal(state.dofs_vel, 100 + np.array(SDK_SLOT))
    np.testing.assert_array_equal(state.dofs_torque, 200 + np.array(SDK_SLOT))
    np.testing.assert_array_equal(state.base_quat, [1.0, 0.0, 0.0, 0.0])  # wxyz, untouched
    np.testing.assert_array_equal(state.base_ang_vel, [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(state.base_lin_acc, [0.0, 0.0, 9.81])
    np.testing.assert_array_equal(state.extra["motor_temp"], 30 + np.array(SDK_SLOT))
    assert state.extra["battery_soc"][0] == 95 and state.extra["battery_v"][0] == pytest.approx(41.0)


def test_an_overheating_motor_warns_once_until_it_recovers(backend, robot, eden_caplog):
    """A derating motor makes its leg go weak -- a common cause of a Go1 freezing mid-deploy."""
    dep = backend()
    robot.mode[GO1_JOINT_IDX["RR_calf_joint"]] = 0x08  # the SDK's Overheat mode
    robot.temperature[GO1_JOINT_IDX["FL_hip_joint"]] = 95
    dep.read_state()
    dep.read_state()

    def warned(joint):
        return sum(f"Go1 motor '{joint}'" in r.message for r in eden_caplog.records if r.levelname == "WARNING")

    assert (warned("RR_calf_joint"), warned("FL_hip_joint"), warned("RL_calf_joint")) == (1, 1, 0)
    robot.mode[:] = 0x0A
    robot.temperature[:] = 40
    dep.read_state()  # recovered ...
    robot.temperature[GO1_JOINT_IDX["FL_hip_joint"]] = 95
    dep.read_state()  # ... so a second overheat warns again
    assert warned("FL_hip_joint") == 2


def test_send_payload_writes_each_dof_to_its_sdk_slot_and_protects_against_fresh_state(backend, robot):
    dep = backend(power_protect_factor=7)
    robot.q[0] = 0.5  # moved since connect: PowerProtect must judge the command against this
    pos, kp, kd = np.arange(12.0), 100 + np.arange(12.0), 200 + np.arange(12.0)
    dep.send_payload(_payload(pos, kp, kd))

    for field, expected in (("q", pos), ("Kp", kp), ("Kd", kd)):  # capital K: the legacy SDK's fields
        np.testing.assert_array_equal(robot.sent_eden(field), expected)
    assert all(robot.sent[-1][slot].Kp == 0.0 for slot in range(12, 20)), "an unused slot was driven"
    assert robot.protected[-1] == (7, 0.5)


def test_power_protect_factor_is_a_go1_option_defaulting_to_full_power():
    """Lower factors clamp the calf torque a gait needs (7 froze a Go1 mid-stance); other backends reject it."""
    assert UnitreeGo1Deployment.configure().power_protect_factor == 10
    with pytest.raises(ValueError, match="power_protect_factor"):
        UnitreeGo2Deployment.configure(power_protect_factor=5)


def test_it_refuses_to_read_or_send_before_connect(backend):
    dep = backend(connect=False)
    with pytest.raises(RuntimeError, match="not connected"):
        dep.read_state()
    with pytest.raises(RuntimeError, match="not connected"):
        dep.send_payload(_payload(np.zeros(12)))


# ---------------------------------------------------------------------------
# stand-up ramp, reset, shutdown
# ---------------------------------------------------------------------------


def test_it_boots_holding_the_pose_it_finds_until_init(backend, robot):
    """Stood up only on the operator's INIT: a folded robot stays folded through the boot hold."""
    robot.q[:] = -2.0  # lying folded
    dep = backend()
    dep.reset()
    sent_before = len(robot.sent)
    for _ in range(3):
        dep.step(None)
    for i in range(sent_before, len(robot.sent)):
        np.testing.assert_allclose(robot.sent_eden("q", i), -2.0)


def test_init_ramps_from_the_measured_pose_to_the_default_and_holds(backend, robot):
    robot.q[:] = -2.0  # lying folded
    dep = backend()
    dep.reset()
    dep.request_init()
    sent_before = len(robot.sent)
    for _ in range(dep.init_steps + 3):
        dep.step(None)
    ramp = np.array([robot.sent_eden("q", i) for i in range(sent_before, len(robot.sent))])

    np.testing.assert_allclose(ramp[0], -2.0)  # tick 0 commands the pose it found
    direction = np.sign(dep.default_dof_pos + 2.0)
    assert np.all(np.diff(ramp, axis=0) * direction >= -1e-9), "a joint moved away from its target"
    np.testing.assert_allclose(ramp[-3:], np.broadcast_to(dep.default_dof_pos, (3, 12)))  # reached at tick init_steps


def test_every_init_ramps_from_where_the_robot_is_now(backend, robot):
    """A second episode's INIT must not replay the first ramp."""
    dep = backend()
    dep.reset()
    dep.request_init()
    for _ in range(dep.init_steps + 2):
        dep.step(None)
    robot.q[:] = 0.3  # knocked into another pose between episodes
    dep.reset()
    dep.request_init()
    first_packet = len(robot.sent)
    dep.step(None)
    np.testing.assert_allclose(robot.sent_eden("q", first_packet), 0.3)


def test_close_ramps_to_a_damping_state_before_letting_go(backend, robot):
    """Dropping the link outright leaves the robot holding its last stiff target until the watchdog trips."""
    dep = backend(shutdown_damping_kd_scale=0.5)
    sent_before = len(robot.sent)
    dep.close()

    assert len(robot.sent) - sent_before == 3
    np.testing.assert_array_equal(robot.sent_eden("Kp"), np.zeros(12))
    np.testing.assert_allclose(robot.sent_eden("Kd"), dep.default_dof_kd * 0.5)
    with pytest.raises(RuntimeError, match="not connected"):
        dep.read_state()
    dep.close()  # a second close, or one before connect, sends nothing and does not raise
    assert len(robot.sent) - sent_before == 3


def test_debug_record_captures_the_motor_health_extras(backend, robot, tmp_path):
    path = tmp_path / "deploy_record.npz"
    dep = backend(debug_record=True, debug_record_path=str(path))
    for _ in range(2):
        dep.step(torch.zeros(1, 12))
    dep.close()

    record = load_deploy_record(str(path))
    assert record["x_motor_temp"].shape == (2, 12) and record["x_motor_mode"].shape == (2, 12)
    np.testing.assert_allclose(record["x_battery_v"], 41.0)
    np.testing.assert_array_equal(record["x_battery_soc"], 95)


def test_dofs_are_looked_up_by_name_beside_the_sdk_motor_map(backend):
    assert_dofs_looked_up_by_name(backend(), Go1())
