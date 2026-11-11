"""URDF↔MANO frame alignment, derived from Eden's ``up`` / ``front`` fields.

A hand URDF's ``up`` and ``front`` declarations encode the same kind of
"asset-frame → world-reference" rotation that :func:`eden.utils.geom.align_up_and_front`
already produces for any Eden entity. For hand robots the convention is:

- ``up`` = the URDF axis along which the **fingers extend**
- ``front`` = the URDF axis along which the **palm normal points**
  (= the URDF axis that aligns with world +X after load)

Eden's defaults (``up=(0,0,1), front=(1,0,0)``) already match XHand1_R's
URDF (fingers along URDF +Z, palm normal along URDF +X), so XHand1_R
loads identity. MANO's right-hand canonical frame is the matching
declaration in MANO-local axes:

- fingers (= ``up``) along ``-X`` in MANO local
- palm normal (= ``front``) along ``-Y`` in MANO local

Applying ``align_up_and_front`` twice (once for the robot, once for MANO)
gives a URDF → canonical-world → MANO chain that the retargeter can compose
into a single rotation — no per-robot ``urdf_to_mano_quat`` metadata
needed.

This module exposes :func:`urdf_to_mano_rot`, which returns:

- ``R_mano_to_world`` — used to rotate MANO-local keypoints into the
  retargeting (= canonical world) frame so the optimizer's targets line
  up with the robot's actually-loaded fingertip workspace.
- ``q_urdf_to_mano`` — composed onto the per-frame MANO wrist quat to get
  the floating-base ``base_quat_w`` that places the robot at the right
  world orientation for the saved ``dofs_pos`` to reproduce MANO
  fingertips at viz time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from eden.utils.geom import align_up_and_front

if TYPE_CHECKING:
    from eden.entities.rigid import RigidEntity


# MANO canonical wrist-local axes, encoded as ``(up, front)`` — same
# convention as Eden's ``up`` / ``front`` on hand robots:
#   up    = fingers extend       (MANO -X)
#   front = palm normal points   (MANO -Y for right; +Y mirror for left)
_MANO_UP_FRONT = {
    "right": ((-1, 0, 0), (0, -1, 0)),
    "left": ((-1, 0, 0), (0, 1, 0)),
}


def urdf_to_mano_rot(robot: "RigidEntity") -> tuple[np.ndarray, np.ndarray]:
    """Derive the URDF↔MANO conversion from the robot's ``up`` / ``front``.

    Returns ``(R_mano_to_world, q_urdf_to_mano)`` where:

    - ``R_mano_to_world`` (3x3) — maps MANO-local vectors into the
      canonical world frame the robot is loaded into via
      ``align_up_and_front(robot.up, robot.front)``. Used at retargeting
      time to bring MANO keypoint targets into the robot's world frame.
    - ``q_urdf_to_mano`` (4,) wxyz — composed onto the per-frame MANO
      wrist quat to produce the floating-base ``base_quat_w`` for viz.

    The robot's MANO side is read from ``metadata.side`` (defaults to right).
    """
    from genesis.utils.geom import R_to_quat

    md = getattr(robot.options, "metadata", None)
    side = getattr(md, "side", "right") if md is not None else "right"
    mano_up, mano_front = _MANO_UP_FRONT[side]

    R_robot_to_world = align_up_and_front(robot.options.up, robot.options.front)
    R_mano_to_world = align_up_and_front(mano_up, mano_front)
    # R_urdf_to_mano = R_world_to_mano @ R_urdf_to_world = R_mano_to_world.T @ R_robot_to_world
    R_urdf_to_mano = (R_mano_to_world.T @ R_robot_to_world).astype(np.float32)
    q_urdf_to_mano = R_to_quat(R_urdf_to_mano).astype(np.float32)
    return R_mano_to_world.astype(np.float32), q_urdf_to_mano
