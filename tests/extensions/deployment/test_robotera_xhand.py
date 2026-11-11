"""``RoboTeraXHandDeployment`` constructed on a built XHand1 env, over a fake ``xhand_controller`` SDK.

Two properties of the hand shape this backend: ``FingerState_t`` carries a position and no velocity,
so ``read_state`` differences consecutive reads; and its PD gains are ``int16`` firmware values
with no mapping to N·m/rad, so the command's gains are never forwarded.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from eden.extensions.deployment import robotera_xhand as xhand
from eden.extensions.deployment.robotera_xhand import XHAND_FINGER_IDX, RoboTeraXHandDeployment
from eden.extensions.deployment.utils.state import RobotCommand
from eden.options.robots.robotera_xhand import XHand1_L
from tests.support.deploy import assert_dofs_looked_up_by_name, deployments, robot_env

# Reversed: the URDF order *is* the SDK finger order, so only a permuted deployment order makes the
# Eden->SDK scatter observable.
DOFS_NAME = list(reversed(XHand1_L().dofs_name))
FINGER = [XHAND_FINGER_IDX[name] for name in DOFS_NAME]
N = len(DOFS_NAME)


def _ok(code=0, message=""):
    return SimpleNamespace(error_code=code, error_message=message)


class FakeXHand:
    """The hand behind ``XHandControl``: SDK-order finger positions it reports, commands it received."""

    def __init__(self):
        self.positions: list[np.ndarray] = []  # one SDK-order vector per read; the last one repeats
        self.sent: list[list[SimpleNamespace]] = []
        self.send_error = _ok()

    def install(self, monkeypatch) -> None:
        hand = self

        class XHandControl:
            def open_serial(self, port, baud):
                return _ok()

            def list_hands_id(self):
                return [0]

            def read_state(self, hand_id, force_update):
                pos = hand.positions.pop(0) if len(hand.positions) > 1 else hand.positions[0]
                fingers = [
                    SimpleNamespace(
                        id=i,
                        position=float(pos[i]),
                        torque=0,
                        raw_position=0,
                        sensor_id=0,
                        temperature=30,
                        commboard_err=0,
                        jonitboard_err=0,
                        tipboard_err=0,
                    )  # fmt: skip
                    for i in range(12)
                ]
                force = SimpleNamespace(fx=0.0, fy=0.0, fz=0.0)
                sensors = [
                    SimpleNamespace(calc_force=force, raw_force=[force], calc_temperature=30.0) for _ in range(5)
                ]
                return _ok(), SimpleNamespace(finger_state=fingers, sensor_data=sensors)

            def send_command(self, hand_id, cmd):
                hand.sent.append([SimpleNamespace(**vars(fc)) for fc in cmd.finger_command])
                return hand.send_error

        def hand_command():
            return SimpleNamespace(finger_command=[SimpleNamespace() for _ in range(12)])

        monkeypatch.setattr(
            xhand, "xhand_control", SimpleNamespace(XHandControl=XHandControl, HandCommand_t=hand_command)
        )

    def queue_eden(self, *vectors) -> None:
        """Queue reads given in deployment DOF order, scattered into the SDK's finger order."""
        for vector in vectors:
            sdk = np.zeros(12)
            sdk[FINGER] = vector
            self.positions.append(sdk)


@pytest.fixture(scope="module")
def env():
    return robot_env(XHand1_L(dofs_name=DOFS_NAME, is_fixed_base=True))


@pytest.fixture
def hand(monkeypatch):
    fake = FakeXHand()
    fake.install(monkeypatch)
    fake.queue_eden(np.zeros(N))
    return fake


@pytest.fixture
def backend(env, hand):
    """Return ``backend(connect=True, **options)``: a real ``RoboTeraXHandDeployment`` on ``env``, undone after the test."""
    with deployments(env) as make:
        yield lambda connect=True, **options: make(
            RoboTeraXHandDeployment, connect=connect, **{"serial_port": "/dev/null", **options}
        )


@pytest.fixture
def clock(monkeypatch):
    """A settable ``time.perf_counter`` (the clock the velocity estimate divides by)."""
    now = SimpleNamespace(t=10.0)
    monkeypatch.setattr(time, "perf_counter", lambda: now.t)
    return now


def test_read_state_differences_positions_into_a_joint_velocity(backend, hand, clock):
    """The SDK has no velocity field; policies observing it used to be fed a constant zero."""
    p0 = np.linspace(0.0, 1.0, N)
    p1 = p0 + np.arange(N) * 0.01
    dep = backend()
    hand.positions.clear()
    hand.queue_eden(p0, p1)

    first = dep.read_state()
    clock.t += 0.02
    second = dep.read_state()

    np.testing.assert_array_equal(first.dofs_vel, np.zeros(N))  # no previous sample: zero, not garbage
    np.testing.assert_allclose(second.dofs_pos, p1)  # deployment DOF order, not SDK order
    np.testing.assert_allclose(second.dofs_vel, np.arange(N) * 0.01 / 0.02)


def test_a_stalled_clock_reports_zero_velocity_not_a_division_by_zero(backend, hand, clock):
    dep = backend()
    hand.positions.clear()
    hand.queue_eden(np.zeros(N), np.ones(N))
    dep.read_state()
    np.testing.assert_array_equal(dep.read_state().dofs_vel, np.zeros(N))


def test_a_reconnect_does_not_difference_against_the_previous_session(backend, hand, clock, eden_caplog):
    dep = backend()
    hand.positions.clear()
    hand.queue_eden(np.zeros(N), np.ones(N), np.full(N, 5.0))
    dep.read_state()
    clock.t += 1.0
    dep.connect()
    clock.t += 1.0
    np.testing.assert_array_equal(dep.read_state().dofs_vel, np.zeros(N))

    # connect names the gains the hand is actually running.
    lines = [r.message for r in eden_caplog.records if "firmware PD" in r.message]
    assert lines and "kp=50" in lines[-1] and "kd=25" in lines[-1] and "dofs_kp / dofs_ki / dofs_kd" in lines[-1]


def test_send_payload_forwards_positions_and_leaves_the_firmware_gains_alone(backend, hand):
    """Payload gains are N·m/rad; the hand's are firmware ints. A ``kp = 0`` payload changes nothing."""
    targets = np.linspace(0.1, 0.9, N)
    backend().send_payload(
        RobotCommand(
            dofs_pos=targets, dofs_vel=np.zeros(N), dofs_torque=np.zeros(N), dofs_kp=np.zeros(N), dofs_kd=np.zeros(N)
        )
    )
    (sent,) = hand.sent
    np.testing.assert_allclose([sent[f].position for f in FINGER], targets)
    assert all((fc.kp, fc.kd, fc.mode) == (50, 25, 3) for fc in sent)


def test_target_lpf_smooths_what_reaches_the_fingers_and_the_payload_says_so(backend, hand):
    """``--target_lpf`` used to be accepted and ignored here; a composite rig forwards it to every part."""
    dep = backend(target_lpf=0.5)
    for value in (0.4, 0.0):
        dep.send_payload(RobotCommand(dofs_pos=np.full(N, value), dofs_vel=np.zeros(N), dofs_torque=np.zeros(N)))
    np.testing.assert_allclose([hand.sent[0][f].position for f in FINGER], 0.4, err_msg="warm-started")
    np.testing.assert_allclose([hand.sent[1][f].position for f in FINGER], 0.2)


def test_close_releases_the_fingers_and_reports_a_failed_release(backend, hand, eden_caplog):
    """A failed release leaves the fingers holding, so it must say so."""
    dep = backend()
    hand.send_error = _ok(3, "bus timeout")
    dep.close()
    assert [fc.mode for fc in hand.sent[-1]] == [0] * 12  # powerless
    assert "mode change to 0 failed: bus timeout" in eden_caplog.text
    with pytest.raises(RuntimeError, match="not connected"):
        dep.read_state()


# --- base options that act through command gains are refused ---------------------------


def test_the_default_options_construct(backend):
    backend(connect=False)


def test_deploy_kp_kd_scale_is_refused_naming_the_firmware_gain_fix(backend):
    """A first-run ``--deploy_kp_kd_scale 0.5`` would leave the hand at full firmware gain."""
    with pytest.raises(ValueError, match=r"deploy_kp_kd_scale=0\.5 cannot reach the hand") as exc:
        backend(connect=False, deploy_kp_kd_scale=0.5)
    assert "configure(dofs_kp=25, dofs_kd=12)" in str(exc.value)


def test_disable_joints_is_refused_because_the_finger_would_keep_servoing(backend):
    with pytest.raises(ValueError, match="cannot leave a single finger limp") as exc:
        backend(connect=False, disable_joints=(DOFS_NAME[0],))
    assert DOFS_NAME[0] in str(exc.value)


def test_dofs_are_looked_up_by_name_beside_the_finger_map(backend):
    assert_dofs_looked_up_by_name(backend(), XHand1_L())
