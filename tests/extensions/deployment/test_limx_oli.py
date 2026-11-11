"""``LimxOliDeployment`` constructed on a built Oli env, talking to a fake ``limxsdk`` robot.

:class:`FakeOli` is the robot end of the SDK: it reports its motors in *its* order (the reverse of
Eden's, so the name-based map is observable), streams the state / IMU / remote it is given through
the subscription callbacks, and records every published ``RobotCmd``.
"""

from __future__ import annotations

import copy
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eden.extensions.deployment import limx_oli as olimod
from eden.extensions.deployment import resolve_deployment_for
from eden.extensions.deployment.base import resolve_dofs_values
from eden.extensions.deployment.limx_oli import (
    LIMX_OLI_CLAMP_TORQUE_LIMITS,
    LimxOliDeployment,
    resolve_clamp_torque_limits,
)
from eden.extensions.deployment.utils.state import RobotCommand
from eden.options import (
    SceneOptions,
)
from eden.options.robots import Oli
from eden.options.robots.limx_oli import OLI_DOFS_NAME
from eden.options.robots.unitree_go1 import Go1
from tests.support.deploy import assert_dofs_looked_up_by_name, deployments, robot_env_config
from tests.support.envs import build_env

DOFS_NAME = list(Oli().dofs_name)
N = len(DOFS_NAME)
#: The PS4-style remote's button indices, as the SDK's ``SensorJoy`` reports them.
BUTTON = {"cross": 0, "circle": 1, "square": 2, "triangle": 3, "L1": 4, "R1": 7}


# ---------------------------------------------------------------------------
# the fake robot and its SDK
# ---------------------------------------------------------------------------


class FakeOli:
    """An Oli on the other end of ``limxsdk``: motor names it reports, streams it sends, commands it got."""

    def __init__(self):
        self.motor_names = list(reversed(DOFS_NAME))
        self.q = np.zeros(N)
        self.dq = np.zeros(N)
        self.tau = np.zeros(N)
        self.init_ok = True
        self.streams = True
        self.init_ip: str | None = None
        self.published: list = []
        self._callbacks: dict[str, object] = {}

    def install(self, monkeypatch) -> None:
        robot = self

        class Robot:
            def __init__(self, robot_type):
                pass

            def init(self, ip):
                robot.init_ip = ip
                return robot.init_ok

            def getMotorNames(self):  # noqa: N802 — the SDK's spelling
                return list(robot.motor_names)

            def getMotorNumber(self):  # noqa: N802
                return len(robot.motor_names)

            def subscribeRobotState(self, cb):  # noqa: N802
                robot._callbacks["state"] = cb
                if robot.streams:
                    robot.send_state()

            def subscribeImuData(self, cb):  # noqa: N802
                robot._callbacks["imu"] = cb
                if robot.streams:
                    robot.send_imu()

            def subscribeSensorJoy(self, cb):  # noqa: N802
                robot._callbacks["joy"] = cb

            def subscribeDiagnosticValue(self, cb):  # noqa: N802
                robot._callbacks["diagnostic"] = cb

            def publishRobotCmd(self, cmd):  # noqa: N802
                robot.published.append(copy.deepcopy(vars(cmd)))

        monkeypatch.setattr(olimod, "IS_LIMX_SDK_AVAILABLE", True)
        monkeypatch.setattr(olimod, "LimxRobot", Robot)
        monkeypatch.setattr(olimod, "LimxRobotType", SimpleNamespace(Humanoid="Humanoid"))
        monkeypatch.setattr(olimod, "LimxRobotCmd", lambda: SimpleNamespace(stamp=0))

    def slot(self, dof: str) -> int:
        return self.motor_names.index(dof)

    def send_state(self):
        msg = SimpleNamespace(
            stamp=0, q=list(self.q), dq=list(self.dq), tau=list(self.tau), motor_names=self.motor_names
        )
        self._callbacks["state"](msg)
        return msg

    def send_imu(self, quat=(1.0, 0.0, 0.0, 0.0)):
        msg = SimpleNamespace(stamp=0, quat=list(quat), gyro=[0.1, 0.2, 0.3], acc=[0.0, 0.0, 9.81])
        self._callbacks["imu"](msg)
        return msg

    def press(self, *buttons, n_buttons=8):
        pressed = [0] * n_buttons
        for name in buttons:
            if BUTTON[name] < n_buttons:
                pressed[BUTTON[name]] = 1
        msg = SimpleNamespace(stamp=0, axes=[0.0] * 4, buttons=pressed)
        self._callbacks["joy"](msg)
        return msg

    def diagnose(self, name="motor", level=2, code=7, message="overheat"):
        self._callbacks["diagnostic"](SimpleNamespace(stamp=0, name=name, level=level, code=code, message=message))

    def sent_eden(self, field: str, packet: int = -1) -> np.ndarray:
        """``field`` of one published command, gathered into Eden DOF order."""
        return np.array([self.published[packet][field][self.slot(dof)] for dof in DOFS_NAME])


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def env():
    """An Oli, plus a Go1 to point the backend at a robot whose joints are not HU_D04 motors."""
    config = robot_env_config(Oli(is_fixed_base=True), scale=1.0)
    scene = SceneOptions(robot=Oli(is_fixed_base=True), go1=Go1(default_root_pos=(3.0, 0.0, 0.5)))
    return build_env(config.model_copy(update={"scene_options": scene}), reset=True)


@pytest.fixture
def robot(monkeypatch):
    fake = FakeOli()
    fake.install(monkeypatch)
    monkeypatch.delenv("ROBOT_IP", raising=False)
    return fake


@pytest.fixture
def backend(env, robot):
    """Return ``backend(connect=True, **options)``: a real ``LimxOliDeployment`` on ``env``, undone after the test."""
    with deployments(env) as make:
        yield lambda connect=True, **options: make(
            LimxOliDeployment,
            connect=connect,
            **{"publish_rate": None, "soft_torque_limit": 0.0, "init_steps": 10, **options},
        )


def _payload(pos, kp=50.0, kd=2.0) -> RobotCommand:
    return RobotCommand(
        dofs_pos=np.broadcast_to(np.asarray(pos, float), (N,)).copy(),
        dofs_vel=np.zeros(N),
        dofs_torque=np.zeros(N),
        dofs_kp=None if kp is None else np.full(N, kp),
        dofs_kd=None if kd is None else np.full(N, kd),
    )


# ---------------------------------------------------------------------------
# binding and motor map
# ---------------------------------------------------------------------------


def test_oli_resolves_to_this_backend_and_its_dofs_are_the_hu_d04_motor_set():
    """The map is by name, so the two must agree as sets (the order is firmware-owned)."""
    assert resolve_deployment_for(Oli()) is LimxOliDeployment
    assert sorted(DOFS_NAME) == sorted(OLI_DOFS_NAME) and N == 31


def test_connect_maps_motors_by_name_whatever_order_the_robot_reports(backend, robot):
    """LimX's own docs and shipped YAMLs disagree on the head-joint order: a positional map is wrong."""
    dep = backend()
    assert robot.init_ip == "10.192.1.2"  # the real robot's default
    dep.send_payload(_payload(np.arange(N, dtype=float)))
    np.testing.assert_array_equal(robot.sent_eden("q"), np.arange(N))
    assert robot.published[-1]["motor_names"] == robot.motor_names


def test_motor_names_may_drop_the_joint_suffix(backend, robot):
    robot.motor_names = [name.removesuffix("_joint") for name in robot.motor_names]
    backend().send_payload(_payload(np.arange(N, dtype=float)))
    sent = robot.published[-1]["q"]
    assert [sent[robot.motor_names.index(dof.removesuffix("_joint"))] for dof in DOFS_NAME] == list(range(N))


@pytest.mark.parametrize(
    "motor_names, error",
    [
        (lambda names: [n for n in names if n != "left_knee_joint"], "left_knee_joint"),
        (lambda names: [*names, "left_knee"], "collides"),
    ],
    ids=["missing", "collision"],
)
def test_an_unmappable_motor_list_fails_connect(backend, robot, motor_names, error):
    robot.motor_names = motor_names(robot.motor_names)
    with pytest.raises(RuntimeError, match=error):
        backend()


def test_motor_name_map_bridges_a_custom_sdk_vocabulary(backend, robot):
    names = {"left_knee_joint": "J4", "head_yaw_joint": "HY"}
    robot.motor_names = [names.get(n, n) for n in robot.motor_names]
    dep = backend(motor_name_map=names)
    dep.send_payload(_payload(np.arange(N, dtype=float)))
    sent = robot.published[-1]["q"]
    assert sent[robot.motor_names.index("J4")] == DOFS_NAME.index("left_knee_joint")
    assert sent[robot.motor_names.index("HY")] == DOFS_NAME.index("head_yaw_joint")


def test_a_motor_the_config_does_not_drive_is_held_limp_and_named(backend, robot, eden_caplog):
    robot.motor_names.append("gripper_motor")
    robot.q, robot.dq, robot.tau = (np.zeros(N + 1) for _ in range(3))
    backend().send_payload(_payload(1.0))
    slot = robot.motor_names.index("gripper_motor")
    assert (robot.published[-1]["q"][slot], robot.published[-1]["Kp"][slot], robot.published[-1]["Kd"][slot]) == (
        0.0,
        0.0,
        0.0,
    )
    assert "gripper_motor" in eden_caplog.text and "held limp" in eden_caplog.text


@pytest.mark.parametrize(
    "options, error",
    [
        ({"motor_name_map": {"not_a_dof": "J1"}}, "not_a_dof"),
        ({"entity_name": "go1"}, "FR_hip_joint"),  # a robot whose joints are not HU_D04 motors
        ({"soft_torque_limit": 1.5}, "soft_torque_limit"),
        ({"soft_torque_limit": 0.95, "clamp_torque_limits": {".*": 0.0}}, "left_hip_pitch_joint"),
        ({"clamp_torque_limits": {"waist_rol_joint": 80.0}}, "waist_rol_joint"),  # a typo'd pattern
        ({"publish_rate": 20.0}, "publish_rate"),  # below the 50 Hz control rate
    ],
)
def test_a_bad_config_is_refused_before_the_entity_swap(env, backend, options, error):
    """Refused after the swap, the env would keep the wrapper and a retry would wrap the wrapper."""
    entity_name = options.get("entity_name", "robot")
    sim_entity = env.entities[entity_name]
    for _ in range(2):  # a retry fails the same way, not somewhere else
        with pytest.raises(ValueError, match=error):
            backend(connect=False, **options)
        assert env.entities[entity_name] is sim_entity


def test_a_mapped_dof_is_exempt_from_the_hu_d04_check(backend):
    """``motor_name_map`` is an explicit declaration of the DOF's SDK motor."""
    go1_dofs = Go1().dofs_name
    motor_name_map = {name: f"M{i}" for i, name in enumerate(go1_dofs)}
    backend(connect=False, entity_name="go1", motor_name_map=motor_name_map, clamp_torque_limits={})


# ---------------------------------------------------------------------------
# connect / read / send
# ---------------------------------------------------------------------------


def test_the_robot_ip_comes_from_the_option_then_the_environment(backend, robot, monkeypatch):
    monkeypatch.setenv("ROBOT_IP", "127.0.0.1")
    backend()
    assert robot.init_ip == "127.0.0.1"
    backend(robot_ip="10.0.0.5")
    assert robot.init_ip == "10.0.0.5"


@pytest.mark.parametrize(
    "robot_state, error",
    [
        ({"init_ok": False}, "Failed to connect"),
        ({"streams": False}, "Timed out waiting for LimX Oli robot state and IMU"),
    ],
)
def test_connect_fails_fast(backend, robot, robot_state, error):
    vars(robot).update(robot_state)
    with pytest.raises(RuntimeError, match=error):
        backend(connect_timeout_s=0.05)


def test_connect_requires_the_sdk(backend, monkeypatch):
    monkeypatch.setattr(olimod, "IS_LIMX_SDK_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="limxsdk"):
        backend()


def test_read_state_gathers_motor_order_into_eden_order(backend, robot):
    dep = backend()
    robot.q, robot.dq, robot.tau = (
        np.arange(N, dtype=float),
        100 + np.arange(N, dtype=float),
        200 + np.arange(N, dtype=float),
    )
    robot.send_state()
    robot.send_imu(quat=(0.5, 0.5, -0.5, 0.5))
    state = dep.read_state()
    slots = np.array([robot.slot(dof) for dof in DOFS_NAME], dtype=float)
    np.testing.assert_array_equal(state.dofs_pos, slots)
    np.testing.assert_array_equal(state.dofs_vel, 100 + slots)
    np.testing.assert_array_equal(state.dofs_torque, 200 + slots)
    np.testing.assert_array_equal(state.full_dofs_pos, np.arange(N))
    np.testing.assert_array_equal(state.base_quat, [0.5, 0.5, -0.5, 0.5])  # wxyz, untouched


def test_read_state_rejects_a_robot_reporting_another_motor_count(backend, robot):
    dep = backend()
    robot.q = robot.dq = robot.tau = np.zeros(12)
    robot.send_state()
    with pytest.raises(RuntimeError, match="12 motors"):
        dep.read_state()


def test_messages_are_snapshotted_against_the_bridge_reusing_them(backend, robot):
    """The native bridge may mutate a message in place after the callback returns."""
    dep = backend()
    robot.q = np.arange(N, dtype=float)
    state_msg = robot.send_state()
    imu_msg = robot.send_imu()
    state_msg.q[0] = 999.0
    imu_msg.quat[0] = -1.0
    state = dep.read_state()
    assert state.full_dofs_pos[0] == 0.0 and state.base_quat[0] == 1.0


def test_send_payload_writes_pr_mode_commands(backend, robot):
    """The SDK solves the parallel ankle/waist linkages, so q/Kp/Kd stay in Eden's serial-joint space."""
    backend().send_payload(_payload(np.arange(N, dtype=float), kp=50.0, kd=2.0))
    cmd = robot.published[-1]
    np.testing.assert_array_equal(robot.sent_eden("Kp"), np.full(N, 50.0))  # capital K: LimX's field names
    np.testing.assert_array_equal(robot.sent_eden("Kd"), np.full(N, 2.0))
    assert all(cmd["parallel_solve_required"]) and set(cmd["mode"]) == {0} and cmd["stamp"] > 0


def test_it_refuses_to_send_before_connect_and_after_close(backend):
    with pytest.raises(RuntimeError, match="not connected"):
        backend(connect=False).send_payload(_payload(0.0))
    dep = backend()
    dep.close()
    with pytest.raises(RuntimeError, match="not connected"):
        dep.send_payload(_payload(0.0))


# ---------------------------------------------------------------------------
# torque-feasibility clamp
# ---------------------------------------------------------------------------


def test_clamp_torque_limits_default_to_limxs_user_torque_limit():
    """80 N·m on the six parallel PR joints (two 42 N·m motors each); Eden's force limits elsewhere."""
    by_name = dict(zip(DOFS_NAME, resolve_clamp_torque_limits(Oli(), DOFS_NAME, LimxOliDeployment.clamp_torque_limits)))
    parallel = [n for n in DOFS_NAME if "ankle" in n or n in ("waist_roll_joint", "waist_pitch_joint")]
    assert len(parallel) == 6 and {by_name[n] for n in parallel} == {80.0}
    assert (by_name["waist_yaw_joint"], by_name["left_hip_pitch_joint"]) == (42.0, 140.0)
    assert (by_name["left_shoulder_pitch_joint"], by_name["left_wrist_yaw_joint"]) == (42.0, 19.0)
    # The class default is a copy, so mutating one instance's dict can never leak into the constant.
    assert LimxOliDeployment.clamp_torque_limits == LIMX_OLI_CLAMP_TORQUE_LIMITS
    assert LimxOliDeployment.clamp_torque_limits is not LIMX_OLI_CLAMP_TORQUE_LIMITS
    np.testing.assert_array_equal(
        resolve_clamp_torque_limits(Oli(), DOFS_NAME, {}),
        resolve_dofs_values(Oli().default_dofs_force_limits, DOFS_NAME),
    )


@pytest.mark.parametrize(
    "measured_dq, target, clamped",
    [
        (0.0, 10.0, True),
        (0.0, -10.0, True),
        # The window is centred on the measured position only. LimX's formula adds kd*dq/kp; on
        # hardware the PR-space velocity of the parallel joints shifted it past the half-window.
        (1.0, 10.0, True),
        (-50.0, -0.1, False),  # in-window despite a large velocity: reaches the wire untouched
    ],
)
def test_the_clamp_bounds_the_position_implied_torque(backend, robot, measured_dq, target, clamped):
    dep = backend(soft_torque_limit=0.95)
    robot.dq[:] = measured_dq
    robot.send_state()
    dep.state = dep.read_state()
    dep.send_payload(_payload(target, kp=50.0))
    half_window = 0.95 * dep.dof_torque_limits / 50.0
    expected = np.sign(target) * half_window if clamped else np.full(N, target)
    np.testing.assert_allclose(robot.sent_eden("q"), expected)


def test_the_clamp_skips_zero_kp_joints_and_the_disabled_clamp_passes_everything(backend, robot):
    """A zero kp is no position torque, so the soft-shutdown path is never distorted."""
    dep = backend(soft_torque_limit=0.95)
    dep.state = dep.read_state()
    dep.send_payload(_payload(5.0, kp=0.0))
    np.testing.assert_array_equal(robot.sent_eden("q"), np.full(N, 5.0))
    backend(soft_torque_limit=0.0).send_payload(_payload(10.0, kp=50.0))
    np.testing.assert_array_equal(robot.sent_eden("q"), np.full(N, 10.0))


def test_the_clamp_uses_the_wire_gains_and_live_state_when_the_payload_carries_none(backend, robot):
    """A payload without kp inherits the gains already on the wire, and no cached state reads live state."""
    dep = backend(soft_torque_limit=0.95)
    assert dep.state is None
    dep.send_payload(_payload(0.0, kp=50.0))  # puts kp=50 on the wire
    dep.send_payload(_payload(10.0, kp=None, kd=None))
    np.testing.assert_allclose(robot.sent_eden("q"), 0.95 * dep.dof_torque_limits / 50.0)


# ---------------------------------------------------------------------------
# the operator state machine, driven from the LimX remote
# ---------------------------------------------------------------------------


def test_the_remote_drives_stop_init_start_and_back(backend, robot):
    """Boots in STOP/hold; L1+triangle ramps to the default pose; R1+square hands over; L1+cross latches a hold."""
    robot.q[:] = -0.5
    dep = backend(init_dofs_kp_scale=0.5)
    dep.reset()
    action = torch.full((1, N), 0.1)

    dep.step(action)  # nothing pressed: hold the measured pose at the default gains
    np.testing.assert_allclose(robot.sent_eden("q"), -0.5)
    np.testing.assert_allclose(robot.sent_eden("Kp"), dep.default_dof_kp)

    robot.press("L1", "triangle")
    ramp = []
    for _ in range(dep.init_steps + 2):  # held for the whole ramp: fires once, on the rising edge
        dep.step(action)
        ramp.append(robot.sent_eden("q"))
    np.testing.assert_allclose(ramp[0], -0.5)
    np.testing.assert_allclose(ramp[-1], dep.default_dof_pos, atol=1e-6)
    np.testing.assert_allclose(robot.sent_eden("Kp"), dep.default_dof_kp * 0.5)

    robot.press("R1", "square")
    dep.step(action)  # the START tick repeats the INIT target: this action predates START
    assert dep.consume_start_events() == 1
    np.testing.assert_allclose(robot.sent_eden("q"), dep.default_dof_pos, atol=1e-6)
    np.testing.assert_allclose(robot.sent_eden("Kp"), dep.default_dof_kp * 0.5)
    dep.step(action)
    np.testing.assert_allclose(robot.sent_eden("q"), dep.default_dof_pos + 0.1, atol=1e-5)  # the policy's action

    robot.q[:] = 0.2
    robot.send_state()
    dep.step(action)  # the policy's tick that reads the robot at 0.2
    robot.press("L1", "cross")
    dep.step(action)
    dep.step(action)
    np.testing.assert_allclose(robot.sent_eden("q"), 0.2)  # STOP latched where the robot was
    robot.q[:] = -0.4  # and keeps that pose as the robot sags away from it
    robot.send_state()
    dep.step(action)
    dep.step(action)
    np.testing.assert_allclose(robot.sent_eden("q"), 0.2)


def test_policy_switches_are_queued_and_kill_wins_a_button_mash(backend, robot):
    """L1+R1+square completes both KILL and START: KILL must come first, before the policy is enabled."""
    dep = backend()
    dep.reset()
    robot.press("L1", "circle")
    dep.step(None)
    assert dep.consume_policy_switches() == ["next"] and dep.consume_policy_switches() == []

    robot.press("L1", "R1", "square")
    with pytest.raises(KeyboardInterrupt):
        dep.step(None)
    assert dep.consume_start_events() == 0


def test_a_truncated_remote_packet_is_ignored(backend, robot):
    dep = backend()
    dep.reset()
    robot.press("L1", "triangle", n_buttons=2)  # the triangle index is past the end of the packet
    dep.step(None)
    np.testing.assert_allclose(robot.sent_eden("q"), 0.0)  # still holding, not ramping


# ---------------------------------------------------------------------------
# republisher, diagnostics, shutdown
# ---------------------------------------------------------------------------


def test_the_republisher_resends_the_wire_command_between_ticks(backend, robot):
    """In PR mode the robot runs the parallel joints' PD once per received packet: 50 Hz rings the waist."""
    dep = backend(publish_rate=500.0)
    dep.send_payload(_payload(0.25))
    time.sleep(0.2)
    dep.close()
    assert len(robot.published) >= 40, len(robot.published)  # 100 nominal; a floor for a loaded box
    stamps = [cmd["stamp"] for cmd in robot.published[:40]]
    assert len(set(stamps)) == len(stamps), "a re-send reused the packet's own timestamp"
    np.testing.assert_array_equal(robot.sent_eden("q", 20), np.full(N, 0.25))


def test_diagnostics_log_faults_by_default_and_everything_when_verbose(backend, robot, eden_caplog):
    """Only a change is logged; OK/warning chatter only with ``verbose_diagnostics``."""
    backend()
    for level, code, message in ((2, 7, "overheat"), (2, 7, "overheat"), (1, 3, "warm"), (0, 0, "ok"), (2, 8, "fault")):
        robot.diagnose(level=level, code=code, message=message)
    records = [(r.levelname, r.message) for r in eden_caplog.records if "robot diagnostic" in r.message]
    assert [(lvl, "code=7" in msg or "code=8" in msg) for lvl, msg in records] == [("ERROR", True), ("ERROR", True)]

    eden_caplog.clear()
    backend(verbose_diagnostics=True)
    for name, level, code, message in (
        ("motor", 1, 3, "warm"),
        ("motor", 1, 3, "warm"),
        ("motor", 0, 0, "ok"),
        ("battery", 1, 9, "low"),
    ):
        robot.diagnose(name=name, level=level, code=code, message=message)
    assert [r.levelname for r in eden_caplog.records if "robot diagnostic" in r.message] == [
        "WARNING",
        "INFO",
        "WARNING",
    ]


def test_close_ramps_to_a_damping_state_before_letting_go(backend, robot):
    dep = backend(shutdown_damping_steps=3, shutdown_damping_kd_scale=0.5)
    dep.state = dep.read_state()
    before = len(robot.published)
    dep.close()
    assert len(robot.published) - before == 3
    np.testing.assert_array_equal(robot.sent_eden("Kp"), np.zeros(N))
    np.testing.assert_allclose(robot.sent_eden("Kd"), dep.default_dof_kd * 0.5)
    dep.close()  # nothing left to release
    assert len(robot.published) - before == 3


def test_dofs_are_looked_up_by_name_once_connect_has_mapped_the_motors(backend):
    assert_dofs_looked_up_by_name(backend(), Oli(is_fixed_base=True))
