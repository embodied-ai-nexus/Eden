"""Observation terms (robot base frame) for the manipulation tasks.

Mirror mjlab's ``*_in_robot_base_frame`` observations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.managers import OBSERVATION_TERM_REGISTRY
from eden.tasks.manipulation.custom_terms.common import _grasp_point_w, _object_pos_w, _to_base_frame
from genesis.utils.geom import inv_quat

from eden.utils.geom import quat_mul, quat_to_rot6d

if TYPE_CHECKING:
    from eden.entities.base import Entity
    from eden.envs.base import EnvBase


@OBSERVATION_TERM_REGISTRY.register()
def object_position_b(env: "EnvBase", *, robot_name: str, object_name: str) -> torch.Tensor:
    """Object position in the robot base frame. (N, 3)."""
    return _to_base_frame(env, robot_name, _object_pos_w(env, object_name))


@OBSERVATION_TERM_REGISTRY.register()
def object_orientation_b(env: "EnvBase", *, robot_name: str, object_name: str) -> torch.Tensor:
    """Object orientation (wxyz) relative to the robot base frame. (N, 4)."""
    robot: Entity = env.entities[robot_name]
    obj: Entity = env.entities[object_name]
    return quat_mul(inv_quat(robot.get_quat()), obj.get_quat())


@OBSERVATION_TERM_REGISTRY.register()
def object_rot6d_b(env: "EnvBase", *, robot_name: str, object_name: str) -> torch.Tensor:
    """Object orientation relative to the robot base frame, as :func:`~eden.utils.geom.quat_to_rot6d`. (N, 6).

    The rotation of :func:`object_orientation_b` without its quaternion's sign: a pose ``q`` and its
    double cover ``-q`` read the same, so a policy fed a camera's pose estimate (whose sign is
    arbitrary) sees what it saw in training.
    """
    return quat_to_rot6d(object_orientation_b(env, robot_name=robot_name, object_name=object_name))


@OBSERVATION_TERM_REGISTRY.register()
def grasp_point_b(
    env: "EnvBase",
    *,
    robot_name: str,
    gripper_name: str,
    grasp_links: list[str],
    approach_offset: float = 0.0,
) -> torch.Tensor:
    """Grasp center in the robot base frame. (N, 3)."""
    return _to_base_frame(env, robot_name, _grasp_point_w(env, gripper_name, grasp_links, approach_offset))
