"""``eden.extensions.deployment.utils.unitree_sdk2``: the shared low-state unpacker and the SDK CRC fallback.

``read_low_state`` gets a synthetic ``LowState_`` (the Go2 and G1 IDL messages share its shape) in
which motor *i* reports ``q = i``, so an off-by-one or a missing Eden->SDK scatter fails loudly.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from eden.extensions.deployment.utils import unitree_sdk2
from eden.extensions.deployment.utils.unitree_sdk2 import read_low_state


def _low_state(num_motors: int):
    motors = [SimpleNamespace(q=float(i), dq=float(i) + 0.5, tau_est=-float(i)) for i in range(num_motors)]
    imu = SimpleNamespace(quaternion=(1.0, 0.0, 0.0, 0.0), gyroscope=(0.1, 0.2, 0.3), accelerometer=(0.0, 0.0, 9.81))
    return SimpleNamespace(motor_state=motors, imu_state=imu)


def test_driven_dofs_come_back_in_eden_order_and_the_full_vector_in_sdk_order():
    """``dof_index`` is the Eden->SDK map; ``full_dofs_pos`` spans every motor, driven or not."""
    dof_index = [5, 3, 1]  # a policy driving three motors, in a scrambled order
    state = read_low_state(_low_state(35), robot_name="Unitree G1", dof_index=dof_index, num_full_dof=29)

    np.testing.assert_array_equal(state.dofs_pos, [5.0, 3.0, 1.0])
    np.testing.assert_array_equal(state.dofs_vel, [5.5, 3.5, 1.5])
    np.testing.assert_array_equal(state.dofs_torque, [-5.0, -3.0, -1.0])
    np.testing.assert_array_equal(state.full_dofs_pos, np.arange(29))


def test_imu_maps_to_the_base_state_in_float64():
    """Deployment math is float64 end to end; a float32 leak here would be silent."""
    state = read_low_state(_low_state(12), robot_name="Unitree Go2", dof_index=list(range(12)), num_full_dof=12)

    np.testing.assert_array_equal(state.base_quat, [1.0, 0.0, 0.0, 0.0])  # wxyz
    np.testing.assert_array_equal(state.base_ang_vel, [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(state.base_lin_acc, [0.0, 0.0, 9.81])
    assert state.stamp > 0.0
    for field in ("base_quat", "base_ang_vel", "base_lin_acc", "dofs_pos", "dofs_vel", "dofs_torque", "full_dofs_pos"):
        assert getattr(state, field).dtype == np.float64, field


@pytest.mark.parametrize("robot_name", ["Unitree G1", "Unitree Go2"])
def test_no_message_yet_raises_naming_the_robot(robot_name):
    with pytest.raises(RuntimeError, match=f"No {robot_name} low state received yet."):
        read_low_state(None, robot_name=robot_name, dof_index=[0], num_full_dof=1)


def test_crc_works_without_the_sdks_native_library():
    """``pip install git+...`` drops ``unitree_sdk2py/utils/lib/*.so``; the patched ``CRC`` falls back to Python."""
    if unitree_sdk2.CRC is None:
        pytest.skip("unitree_sdk2py not installed; nothing to fall back from")
    crc = unitree_sdk2.CRC()
    assert crc.platform in ("Linux", "PyFallback")  # native lib present, or the fallback sentinel
    # A regression value from the SDK's own ``_crc_py``, not a spec-conformance check.
    assert crc._crc_py([0x01020304, 0x05060708]) == 0x140B8DD8
