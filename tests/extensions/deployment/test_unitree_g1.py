"""``UnitreeG1Deployment``: a command goes out under the command lock, and the republisher re-sends that packet verbatim.

A really-constructed backend on the directory's G1 env. Only the Unitree SDK's wire objects (DDS
publisher, CRC, ``LowCmd_``) are fakes, installed where ``connect()`` would put them.
"""

from __future__ import annotations

import numpy as np
import pytest

from eden.extensions.deployment.unitree_g1 import UnitreeG1Deployment
from eden.extensions.deployment.utils.state import RobotCommand
from tests.support.deploy import assert_dofs_looked_up_by_name, robot_state


class _Publisher:
    """``ChannelPublisher``: records each written packet's first motor target, and whether the lock was held."""

    def __init__(self, lock) -> None:
        self._lock = lock
        self.writes: list[tuple[float, int, bool]] = []

    def Write(self, msg) -> None:  # noqa: N802 -- the SDK's spelling
        self.writes.append((msg.motor_cmd[0].q, msg.crc, self._lock.locked()))


class _Crc:
    def __init__(self) -> None:
        self.calls = 0

    def Crc(self, msg) -> int:  # noqa: N802 -- the SDK's spelling
        self.calls += 1
        return 0xDEADBEEF


class _MotorCmd:
    q = dq = tau = kp = kd = 0.0
    mode = 1


class _LowCmd:
    def __init__(self) -> None:
        self.mode_pr = self.mode_machine = self.crc = 0
        self.motor_cmd = [_MotorCmd() for _ in range(35)]


def _wired(dep: UnitreeG1Deployment) -> UnitreeG1Deployment:
    """``dep`` with the fake wire installed where ``connect()`` would put the SDK's."""
    dep._publisher, dep._crc, dep._low_cmd = _Publisher(dep._cmd_lock), _Crc(), _LowCmd()
    return dep


@pytest.fixture
def g1(deploy):
    return _wired(deploy(UnitreeG1Deployment, publish_rate=None))


def _payload(dep, pos: float) -> RobotCommand:
    n = dep.num_dofs
    return RobotCommand(
        dofs_pos=np.full(n, pos),
        dofs_vel=np.zeros(n),
        dofs_torque=np.zeros(n),
        dofs_kp=np.full(n, 50.0),
        dofs_kd=np.full(n, 2.0),
    )


def test_a_command_is_published_under_the_lock_and_resent_verbatim(g1):
    """``LowCmd_`` carries no timestamp or sequence number, so the packet ``send_payload`` built stays valid."""
    g1.send_payload(_payload(g1, 0.25))
    with g1._cmd_lock:
        g1._publish_held_command()

    assert g1._publisher.writes == [(0.25, 0xDEADBEEF, True), (0.25, 0xDEADBEEF, True)]
    assert g1._crc.calls == 1, "the re-send is not re-CRC'd"


def test_nothing_is_resent_before_the_first_command(g1):
    with g1._cmd_lock:
        g1._publish_held_command()
    assert g1._publisher.writes == []


def test_dofs_are_looked_up_by_name_beside_the_sdk_motor_map(g1):
    assert_dofs_looked_up_by_name(g1, g1.robot_options)


def test_close_damps_from_the_last_read_pose_and_sends_nothing_before_a_read(deploy):
    """The G1 settles holding the pose the loop last read (kp=0, scaled kd); with no read yet there is none to hold."""
    unread = _wired(deploy(UnitreeG1Deployment, publish_rate=None, shutdown_damping_steps=3))
    wire = unread._publisher
    unread.close()
    assert wire.writes == []

    dep = _wired(
        deploy(UnitreeG1Deployment, publish_rate=None, shutdown_damping_steps=3, shutdown_damping_kd_scale=0.5)
    )
    wire, packet = dep._publisher, dep._low_cmd
    dep.state = robot_state(dep.num_dofs, dofs_pos=[0.3] * dep.num_dofs)
    dep.close()

    assert wire.writes == [(0.3, 0xDEADBEEF, True)] * 3
    for i, slot in enumerate(dep._dof_index):
        assert packet.motor_cmd[slot].kp == 0.0
        assert packet.motor_cmd[slot].kd == pytest.approx(0.5 * dep.default_dof_kd[i])
    assert dep._publisher is None, "the channel is released after the ramp"
