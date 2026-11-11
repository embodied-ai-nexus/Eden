"""LBM-eval scene options."""

from __future__ import annotations

import math

from eden.options.entities import EntityOptions, PlaneOptions, SceneOptions
from eden.options.objects.lbm_eval import (
    LBM_LEFT_ROBOT_POS,
    LBM_LEFT_ROBOT_QUAT,
    LBM_RIGHT_ROBOT_POS,
    LBM_RIGHT_ROBOT_QUAT,
    LBMRiverwayStation,
)


# 45° outward splay on the franka base joint so the two arms don't collide
# at init. The signs differ because the two bases have different world yaws
# (left=180°, right=90°), and we want both arms to reach into the (-x, +y)
# world quadrant — i.e. away from each other:
#   - Left base +x = world -x. joint1=-pi/4 rotates the arm's reach
#     direction CW in base frame -> CCW in world -> from (-x,0) toward (-x,+y).
#   - Right base +x = world +y. joint1=+pi/4 rotates CCW in base frame
#     -> CCW in world too -> from (0,+y) toward (-x,+y).
# Equal-signed offsets would steer one arm INTO the other.
_LBM_LEFT_JOINT1_OFFSET: float = -math.pi / 4
_LBM_RIGHT_JOINT1_OFFSET: float = math.pi / 4


def _with_joint1_offset(robot: EntityOptions, offset: float) -> dict[str, float]:
    """Return the robot's ``default_dofs_pos`` with ``joint1`` shifted by ``offset``.

    Skips robots that don't expose a ``joint1`` DOF (non-Franka arms) so the
    scene can still be instantiated with a custom robot without raising.
    """
    base = dict(robot.default_dofs_pos)
    if "joint1" in base or "joint1" in robot.dofs_name:
        base["joint1"] = base.get("joint1", 0.0) + offset
    return base


def LBMEvalSceneOptions(
    left_robot: EntityOptions | None = None,
    right_robot: EntityOptions | None = None,
    plane: PlaneOptions | None = None,
    **kwargs: EntityOptions,
) -> SceneOptions:
    """Bimanual scene built around the Riverway dual-Franka station.

    The two arms default to :class:`FrankaEmikaPanda`. The scene re-poses them
    onto the station's left (+x site) and right (-y site) gantry base plates,
    orients each one so its forward (+x) points into the workspace
    (left=180° about z, right=90° about z), and applies a ±π/4 offset to
    ``joint1`` (the base DOF) so the two arms splay away from each other at
    init instead of meeting in the middle of the table.

    Returns a plain :class:`SceneOptions` (station + re-posed robots as named
    children) so the config round-trips through validated load.
    """
    if left_robot is None or right_robot is None:
        # Lazy import: a robot module loads the whole ``eden.options.robots`` package,
        # which this scene needs only for the default arms.
        from eden.options.robots.franka_emika_panda import FrankaEmikaPanda

        if left_robot is None:
            left_robot = FrankaEmikaPanda()
        if right_robot is None:
            right_robot = FrankaEmikaPanda()

    scene_kwargs: dict[str, EntityOptions] = {
        "station": LBMRiverwayStation(),
        "left_robot": left_robot.model_copy(
            update={
                "default_root_pos": LBM_LEFT_ROBOT_POS,
                "default_root_quat": LBM_LEFT_ROBOT_QUAT,
                "default_dofs_pos": _with_joint1_offset(left_robot, _LBM_LEFT_JOINT1_OFFSET),
            }
        ),
        "right_robot": right_robot.model_copy(
            update={
                "default_root_pos": LBM_RIGHT_ROBOT_POS,
                "default_root_quat": LBM_RIGHT_ROBOT_QUAT,
                "default_dofs_pos": _with_joint1_offset(right_robot, _LBM_RIGHT_JOINT1_OFFSET),
            }
        ),
    }
    if plane is not None:
        scene_kwargs["plane"] = plane
    return SceneOptions(**scene_kwargs, **kwargs)


__all__ = ["LBMEvalSceneOptions"]
