"""Shared helpers for the entity tests: pose comparisons, a bundled offset mesh, and build-time log capture."""

from __future__ import annotations

import logging
from contextlib import contextmanager

import genesis.utils.geom as gu
import numpy as np

from eden.utils.misc import get_assets_dir

#: A small rigid mesh, fetched into the asset directory on first use. Its centre of mass is off its origin,
#: so Genesis' solver frame and the authored frame differ for it, and it exercises the file loader's
#: up/front alignment.
HAND_STL = get_assets_dir("grippers", "franka_hand", "assets", "hand.stl")


def to_np(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def euler_quat(rpy_deg) -> np.ndarray:
    """World-frame wxyz quaternion for an intrinsic XYZ euler angle in degrees."""
    return np.asarray(gu.xyz_to_quat(np.array(rpy_deg, dtype=np.float64), rpy=True, degrees=True), dtype=np.float64)


def assert_quat_close(actual, expected, *, atol=1e-4, err_msg=""):
    """Compare wxyz quaternions up to sign, through the rotations they induce."""
    np.testing.assert_allclose(
        gu.quat_to_R(np.asarray(actual, dtype=np.float64)),
        gu.quat_to_R(np.asarray(expected, dtype=np.float64)),
        atol=atol,
        err_msg=err_msg,
    )


@contextmanager
def captured_eden_logs():
    """Collect every record the ``eden`` logger emits inside the block, e.g. during a module-scoped build.

    The function-scoped ``eden_caplog`` fixture cannot observe a build that runs in a module fixture.
    """
    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Collect(level=logging.DEBUG)
    logger = logging.getLogger("eden")
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
