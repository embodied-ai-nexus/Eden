"""Tests for ``eden.extensions.retargeting.dex_retargeter.calibrate.urdf_to_mano_rot``.

For hands, ``up`` reads as "fingers extend" and ``front`` as "palm normal"; with MANO's own declaration,
``align_up_and_front`` then yields the URDF-to-MANO rotation with no per-robot metadata.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("nlopt")  # the optional ``dex-retargeter`` extra, imported by the package

from eden.extensions.retargeting.dex_retargeter.calibrate import urdf_to_mano_rot  # noqa: E402


def _quat_close(q1, q2, tol=1e-4):
    """Equal as rotations: ``q`` and ``-q`` are the same one."""
    q1, q2 = np.asarray(q1, dtype=np.float32), np.asarray(q2, dtype=np.float32)
    return min(np.max(np.abs(q1 - q2)), np.max(np.abs(-q1 - q2))) < tol


def test_the_xhand1_urdf_to_mano_rotation(env):
    R_mano_to_world, q_urdf_to_mano = urdf_to_mano_rot(env.entities["robot"])

    assert _quat_close(q_urdf_to_mano, (0.5, 0.5, -0.5, -0.5)), tuple(float(x) for x in q_urdf_to_mano)
    # MANO's canonical frame: up = fingers = MANO -X, front = palm normal = MANO -Y.
    expected_R = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=np.float32)
    assert np.max(np.abs(R_mano_to_world - expected_R)) < 1e-4, R_mano_to_world


def test_the_mano_hands_own_urdf_is_the_identity(env):
    _, q_urdf_to_mano = urdf_to_mano_rot(env.entities["mano"])
    assert _quat_close(q_urdf_to_mano, (1.0, 0.0, 0.0, 0.0)), tuple(float(x) for x in q_urdf_to_mano)
