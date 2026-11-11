"""Reward terms for the manipulation tasks.

Reproduce mjlab's ``lift_env_cfg`` reward structure (``object_ee_distance``,
``object_is_lifted``, ``object_goal_distance``) plus the shaping that made it
learnable in Genesis: a sharp unclamped fine-reach, a dense ``lift_height``, and
a ``grasp_bonus`` for closing while centered. ``reaching_target`` is the
reacher-style reward used by ``manipulation_reach_droid``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.managers import REWARD_TERM_REGISTRY
from eden.tasks.manipulation.custom_terms.common import (
    _base_to_world,
    _grasp_point_w,
    _gripper_closure,
    _object_pos_w,
)

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


def _nan_safe(t: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


def _resolve_rest_z(env: "EnvBase", rest_z: float | None, rest_z_metric: str | None) -> "float | torch.Tensor":
    """Resting-z reference for the lift gates: a scalar, or a per-env metric.

    Exactly one source must be given. ``rest_z`` is the single object's resting z;
    ``rest_z_metric`` names a metric term supplying a *per-env* resting height
    (``metric_manager.get_value``), for heterogeneous-object tasks (see ``manipulation_lift_ycb_rgb``) that
    hold a differently-shaped object per env, where one scalar cannot gate the lift.

    Both guards below are fail-loud on purpose: this value is the *reference* the lift gates
    subtract from, so a wrong one is never a crash — it silently opens the gate (a too-low
    reference makes ``object_is_lifted`` fire on a resting object) or closes it forever.
    """
    if (rest_z is None) == (rest_z_metric is None):
        raise ValueError(
            "the lift gate needs exactly one resting-height source: pass rest_z=<float> for a single "
            "object, or rest_z_metric=<metric name> for a per-env resting height (heterogeneous "
            f"objects); got rest_z={rest_z!r}, rest_z_metric={rest_z_metric!r}."
        )
    if rest_z_metric is None:
        return rest_z
    if env.metric_manager is None:
        raise ValueError(
            f"rest_z_metric='{rest_z_metric}' was requested but no metric manager is configured; "
            f"add the metric term (e.g. 'object_rest_height') to the task's metric_options."
        )
    # The lift gate needs a per-step resting height, not a running episode sum. A cumulative
    # metric's get_value grows every step, silently pushing the gate out of reach — fail loud
    # instead (mirrors the guard in metric_reward). is_cumulative also raises if the name is
    # unregistered, so a typo'd metric surfaces here with the active-term list.
    if env.metric_manager.is_cumulative(rest_z_metric):
        raise ValueError(
            f"rest_z_metric='{rest_z_metric}' is a cumulative metric; the lift gate needs a per-step "
            f"resting height, not an episode sum. Use a non-cumulative metric (e.g. 'object_rest_height')."
        )
    return env.metric_manager.get_value(rest_z_metric)


def _lifted_and_close(
    env: "EnvBase",
    gripper_name: str,
    object_name: str,
    grasp_links: list[str],
    rest_z: "float | torch.Tensor",
    lift_margin: float,
    grasp_threshold: float,
    approach_offset: float,
) -> torch.Tensor:
    grasp = _grasp_point_w(env, gripper_name, grasp_links, approach_offset)
    obj = _object_pos_w(env, object_name)
    ee_close = torch.norm(grasp - obj, dim=-1) < grasp_threshold
    lifted = (obj[:, 2] - rest_z) > lift_margin
    return lifted & ee_close


def _object_goal_dist(env: "EnvBase", robot_name: str, object_name: str, command_name: str) -> torch.Tensor:
    """World-frame distance (N,) from the object to the goal command.

    The command is in the robot base frame; lift it to world via the robot pose.
    Shared by :func:`object_goal_distance` and :func:`settle_at_goal` so the two
    use the same goal/object geometry.
    """
    command = env.command_manager.get_term(command_name).command  # base frame (N, 3)
    goal_w = _base_to_world(env, robot_name, command[..., :3])
    obj = _object_pos_w(env, object_name)
    return torch.norm(goal_w - obj, dim=-1)


@REWARD_TERM_REGISTRY.register()
def object_ee_distance(
    env: "EnvBase",
    *,
    gripper_name: str,
    object_name: str,
    grasp_links: list[str],
    std: float,
    approach_offset: float = 0.0,
    clamp_max: float = 0.7,
) -> torch.Tensor:
    """Shaped reward for the grasp center approaching the object.

    mjlab ``object_ee_distance``: ``clamp(1 - tanh(d/std), max=0.7)``. The
    ``clamp_max`` is exposed so a sharp (small-``std``) *unclamped* copy can
    provide gradient across the final approach into the grasp zone, which the
    clamped coarse term does not.
    """
    grasp = _grasp_point_w(env, gripper_name, grasp_links, approach_offset)
    obj = _object_pos_w(env, object_name)
    dist = torch.norm(grasp - obj, dim=-1)
    return _nan_safe(torch.clamp(1.0 - torch.tanh(dist / std), max=clamp_max))


@REWARD_TERM_REGISTRY.register()
def lift_height(
    env: "EnvBase",
    *,
    object_name: str,
    rest_z: float | None = None,
    rest_z_metric: str | None = None,
    cap: float = 0.3,
) -> torch.Tensor:
    """Dense reward for raising the object above its resting height.

    Unlike :func:`object_is_lifted` (a sparse grasped-and-lifted gate), this
    gives gradient for *any* upward object motion, so an incidental grasp is
    immediately reinforced.

    ``rest_z`` is the object's resting z; pass ``rest_z_metric`` instead to source
    a per-env resting height (heterogeneous objects — see :func:`_resolve_rest_z`).
    """
    rest_z = _resolve_rest_z(env, rest_z, rest_z_metric)
    obj_z = _object_pos_w(env, object_name)[:, 2]
    return _nan_safe(torch.clamp(obj_z - rest_z, min=0.0, max=cap))


@REWARD_TERM_REGISTRY.register()
def grasp_bonus(
    env: "EnvBase",
    *,
    gripper_name: str,
    object_name: str,
    grasp_links: list[str],
    gripper_dofs: list[str],
    close_target: float,
    grasp_threshold: float,
    approach_offset: float = 0.0,
) -> torch.Tensor:
    """Reward closing the gripper while the grasp center is on the object.

    Teaches the (otherwise unrewarded) close action: closure counts only when
    the grasp center is within ``grasp_threshold`` of the object.
    """
    grasp = _grasp_point_w(env, gripper_name, grasp_links, approach_offset)
    obj = _object_pos_w(env, object_name)
    ee_close = (torch.norm(grasp - obj, dim=-1) < grasp_threshold).float()
    closure = _gripper_closure(env, gripper_name, gripper_dofs, close_target)
    return _nan_safe(ee_close * closure)


@REWARD_TERM_REGISTRY.register()
def object_is_lifted(
    env: "EnvBase",
    *,
    gripper_name: str,
    object_name: str,
    grasp_links: list[str],
    rest_z: float | None = None,
    rest_z_metric: str | None = None,
    lift_margin: float = 0.02,
    grasp_threshold: float = 0.06,
    approach_offset: float = 0.0,
) -> torch.Tensor:
    """1.0 when the object is grasped (EE close) and lifted off the surface.

    ``rest_z_metric`` sources a per-env resting height in place of the scalar
    ``rest_z`` (heterogeneous objects — see :func:`_resolve_rest_z`).
    """
    rest_z = _resolve_rest_z(env, rest_z, rest_z_metric)
    return _nan_safe(
        _lifted_and_close(
            env, gripper_name, object_name, grasp_links, rest_z, lift_margin, grasp_threshold, approach_offset
        ).float()
    )


@REWARD_TERM_REGISTRY.register()
def object_goal_distance(
    env: "EnvBase",
    *,
    robot_name: str,
    gripper_name: str,
    object_name: str,
    command_name: str,
    grasp_links: list[str],
    std: float,
    rest_z: float | None = None,
    rest_z_metric: str | None = None,
    lift_margin: float = 0.02,
    grasp_threshold: float = 0.06,
    approach_offset: float = 0.0,
    deadband: float = 0.0,
) -> torch.Tensor:
    """Goal tracking, gated on the object being grasped and lifted.

    mjlab ``object_goal_distance``: ``(grasped & lifted) * (1 - tanh(d/std))``.

    ``deadband`` (m) flattens the reward within that radius of the goal: the
    distance is measured as ``max(d - deadband, 0)`` so there is no gradient (no
    incentive to micro-correct) once the object is "close enough". Pair a small
    deadband on the fine term with ``settle_at_goal`` so the arm can hold still at
    the goal instead of perpetually chasing the exact sub-cm point.

    ``rest_z_metric`` sources a per-env resting height in place of the scalar
    ``rest_z`` for the lift gate (heterogeneous objects — see :func:`_resolve_rest_z`).
    """
    rest_z = _resolve_rest_z(env, rest_z, rest_z_metric)
    dist = _object_goal_dist(env, robot_name, object_name, command_name)
    if deadband > 0.0:
        dist = (dist - deadband).clamp(min=0.0)
    gated = _lifted_and_close(
        env, gripper_name, object_name, grasp_links, rest_z, lift_margin, grasp_threshold, approach_offset
    ).float()
    return _nan_safe(gated * (1.0 - torch.tanh(dist / std)))


@REWARD_TERM_REGISTRY.register()
def reaching_target(
    env: "EnvBase",
    *,
    robot_name: str,
    gripper_name: str,
    command_name: str,
    grasp_links: list[str],
    reaching_std: float,
    approach_offset: float = 0.0,
) -> torch.Tensor:
    """Reacher-style reward: ``exp(-||grasp - target||^2 / std^2)``."""
    command = env.command_manager.get_term(command_name).command  # base frame (N, 3)
    target_w = _base_to_world(env, robot_name, command[..., :3])
    grasp = _grasp_point_w(env, gripper_name, grasp_links, approach_offset)
    dist_sq = torch.sum(torch.square(grasp - target_w), dim=-1)
    return _nan_safe(torch.exp(-dist_sq / reaching_std**2))


@REWARD_TERM_REGISTRY.register()
def settle_at_goal(
    env: "EnvBase",
    *,
    robot_name: str,
    object_name: str,
    command_name: str,
    std: float = 0.06,
) -> torch.Tensor:
    """Penalize arm joint velocity, weighted by object→goal proximity.

    Returns ``(1 - tanh(d/std)) * sum(arm_dof_vel^2)`` — apply a NEGATIVE weight.
    Self-gating: ~0 while the object is far (approach unhindered), growing toward
    1 as it settles at the goal so the arm holds still once the task is achieved.
    Needed because Eden's long lift episode has a hold phase where goal-tracking
    is flat; mjlab's 1 s episode ends at the lift and needs no equivalent.
    """
    dist = _object_goal_dist(env, robot_name, object_name, command_name)
    near = 1.0 - torch.tanh(dist / std)
    arm_vel = env.entities[robot_name].get_dofs_vel()
    vel_sq = torch.sum(torch.square(arm_vel), dim=-1)
    return _nan_safe(near * vel_sq)
