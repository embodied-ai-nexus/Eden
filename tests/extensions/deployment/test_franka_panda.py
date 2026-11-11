"""``FrankaPandaDeployment`` constructed on a built Panda env, over a fake ``panda_py`` / ``libfranka``.

The arm runs on panda-py's JointPosition controller and the Franka Hand on libfranka's blocking
gripper calls, which a background worker dispatches as the commanded width changes -- each new
width stopping the command still in flight.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from eden.extensions.deployment import franka_panda as fp
from eden.extensions.deployment.franka_panda import FrankaPandaDeployment
from eden.extensions.deployment.utils.state import RobotCommand
from eden.options.robots.franka_emika_panda import FrankaEmikaPanda
from tests.support.deploy import deployments, robot_env

OPEN, CLOSED = 0.08, 0.0


class FakeFranka:
    """The arm and the hand as panda-py and libfranka present them, recording every command."""

    def __init__(self, arm_q: np.ndarray) -> None:
        self.arm_q = np.asarray(arm_q, dtype=np.float64)
        self.setpoints: list[np.ndarray] = []
        self.gripper_calls: list[tuple[str, float]] = []
        self._lock = threading.Lock()

    def install(self, monkeypatch) -> None:
        fake = self

        class Panda:
            def __init__(self, ip):
                pass

            def recover(self):
                pass

            def get_state(self):
                return SimpleNamespace(
                    q=fake.arm_q.tolist(),
                    dq=[0.0] * 7,
                    tau_J=[0.0] * 7,
                    O_T_EE=np.eye(4).flatten().tolist(),
                    robot_mode="idle",
                )

            def start_controller(self, controller):
                pass

            def stop_controller(self):
                pass

        class JointPosition:
            def __init__(self, **kwargs):
                pass

            def set_control(self, q):
                fake.setpoints.append(np.array(q, dtype=np.float64))

        class Gripper:
            def __init__(self, ip):
                pass

            def read_once(self):
                return SimpleNamespace(width=OPEN, max_width=OPEN, is_grasped=False)

            def homing(self):
                pass

            def move(self, width, speed):
                fake.record("move", width)
                return True

            def grasp(self, width, speed, force, epsilon_inner, epsilon_outer):
                fake.record("grasp", width)
                return True

            def stop(self):
                fake.record("stop", float("nan"))

        monkeypatch.setattr(fp, "panda_py", SimpleNamespace(Panda=Panda))
        monkeypatch.setattr(fp, "_panda_controllers", SimpleNamespace(JointPosition=JointPosition))
        monkeypatch.setattr(fp, "_libfranka", SimpleNamespace(Gripper=Gripper))

    def record(self, call: str, width: float) -> None:
        with self._lock:
            self.gripper_calls.append((call, float(width)))

    def calls(self, kind: str) -> list[float]:
        with self._lock:
            return [width for call, width in self.gripper_calls if call == kind]

    def wait_for(self, kind: str, count: int, timeout: float = 2.0) -> None:
        deadline = time.monotonic() + timeout
        while len(self.calls(kind)) < count:
            assert time.monotonic() < deadline, f"the gripper worker never sent {count} {kind}(s): {self.gripper_calls}"
            time.sleep(0.002)


@pytest.fixture(scope="module")
def env():
    return robot_env(FrankaEmikaPanda())


@pytest.fixture
def franka(env, monkeypatch):
    default = FrankaEmikaPanda().default_dofs_pos
    fake = FakeFranka([default[f"joint{i}"] for i in range(1, 8)])
    fake.install(monkeypatch)
    return fake


@pytest.fixture
def backend(env, franka):
    """Return ``backend(**options)``: a connected ``FrankaPandaDeployment`` on ``env``, closed after the test."""
    with deployments(env) as make:
        yield lambda **options: make(FrankaPandaDeployment, connect=True, **options)


def _command(dep, arm: np.ndarray, width: float) -> RobotCommand:
    dofs_pos = np.zeros(dep.num_dofs)
    dofs_pos[dep._arm_dof_indices] = arm
    dofs_pos[dep._finger_dof_indices] = width / 2.0  # each finger carries half the jaw opening
    zeros = np.zeros(dep.num_dofs)
    return RobotCommand(dofs_pos=dofs_pos, dofs_vel=zeros, dofs_torque=zeros, dofs_kp=zeros, dofs_kd=zeros)


def test_target_lpf_smooths_the_arm_and_leaves_the_gripper_width_as_commanded(backend, franka):
    """A filtered width ramps over many ticks, and each step of it was a new gripper command stopping the last.

    Closing on an object, that stopped the force-controlled grasp and re-issued it at the control rate.
    """
    dep = backend(target_lpf=0.5)
    home = franka.arm_q.copy()
    reach = home + 0.01

    opening = _command(dep, home, OPEN)
    dep.send_payload(opening)
    franka.wait_for("move", 1)
    closing = [_command(dep, reach, CLOSED) for _ in range(6)]
    for payload in closing:
        dep.send_payload(payload)
        time.sleep(0.02)  # long enough for the worker to dispatch whatever width it was handed
    franka.wait_for("grasp", 1)

    assert franka.calls("grasp") == [CLOSED], "one grasp, at the commanded width"
    assert franka.calls("move") == [OPEN]
    assert franka.calls("stop") == [], "nothing in flight was stopped"
    np.testing.assert_allclose(closing[0].dofs_pos[dep._finger_dof_indices], CLOSED / 2, err_msg="what was sent")
    # The arm is still low-passed: half way to the new target on the first tick, and the payload says so.
    np.testing.assert_allclose(franka.setpoints[1], 0.5 * reach + 0.5 * home)
    np.testing.assert_allclose(closing[0].dofs_pos[dep._arm_dof_indices], franka.setpoints[1])
