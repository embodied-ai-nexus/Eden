"""Custom reward and event terms for the Go1 getup task.

Ported from: mujoco_playground/_src/locomotion/go1/getup.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
from eden.managers.terms.utils import gravity_vec, projected_gravity

from eden.managers.reward_manager import REWARD_TERM_REGISTRY, RewardTerm
from eden.managers.termination_manager import TERMINATION_TERM_REGISTRY, TerminationTerm
from eden.managers.event_manager import EVENT_TERM_REGISTRY, EventTerm
from eden.options.managers.events import EventTermOptions
from eden.utils.sample import sample_uniform
from eden.utils.misc import env_indices
from eden.utils.string import term_subject
from eden.utils.torch import compile_model, device_constant

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.entities.base import Entity


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


@compile_model
def _reward_orientation(gravity: torch.Tensor, up_vec: torch.Tensor) -> torch.Tensor:
    # up_vec = [0, 0, -1] in body frame when upright
    error = torch.sum(torch.square(up_vec - gravity), dim=1)
    return torch.exp(-2.0 * error)


@REWARD_TERM_REGISTRY.register()
class GetupOrientation(RewardTerm):
    """Reward for keeping the torso upright. exp(-2 * ||up - gravity||^2).

    Caches the robot ref and the constant world-frame ``[0, 0, -1]`` vector in :meth:`build` (reused as
    both the gravity input and the upright reference), instead of rebuilding them every step.
    """

    entity_name: str = "robot"

    def build(self) -> None:
        self._robot = self._env.entities[self.entity_name]
        self._neg_z = gravity_vec(self.num_envs, self.device)

    def compute(self) -> torch.Tensor:
        gravity = projected_gravity(self._robot.get_quat(), self._neg_z)
        return _reward_orientation(gravity, self._neg_z)


@compile_model
def _reward_height(torso_height: torch.Tensor, z_des: float) -> torch.Tensor:
    height = torch.clamp(torso_height, max=z_des)
    return torch.exp(height) - 1.0


@REWARD_TERM_REGISTRY.register()
class GetupTorsoHeight(RewardTerm):
    """Reward for raising the torso to the desired height. exp(min(h, z_des)) - 1."""

    entity_name: str = "robot"
    z_des: float = 0.275

    def build(self) -> None:
        self._robot = self._env.entities[self.entity_name]

    def compute(self) -> torch.Tensor:
        torso_height = self._robot.get_pos()[:, 2]
        return _reward_height(torso_height, self.z_des)


@compile_model
def _reward_posture_fused(
    gravity: torch.Tensor,
    up_vec: torch.Tensor,
    joint_angles: torch.Tensor,
    default_pose: torch.Tensor,
    ori_tol: float,
) -> torch.Tensor:
    # Gate: upright only (reference gates posture by is_upright, not height)
    ori_error = torch.sum(torch.square(up_vec - gravity), dim=1)
    is_up = (ori_error < ori_tol).float()
    # Posture reward
    cost = torch.sum(torch.square(joint_angles - default_pose), dim=1)
    rew = torch.exp(-0.5 * cost)
    return is_up * rew


@REWARD_TERM_REGISTRY.register()
class GetupPosture(RewardTerm):
    """Reward the robot for being in the default pose, gated by upright only."""

    entity_name: str = "robot"
    z_des: float = 0.275
    ori_tol: float = 0.01

    def build(self) -> None:
        self._robot = self._env.entities[self.entity_name]
        self._neg_z = gravity_vec(self.num_envs, self.device)

    def compute(self) -> torch.Tensor:
        gravity = projected_gravity(self._robot.get_quat(), self._neg_z)
        return _reward_posture_fused(
            gravity,
            self._neg_z,
            self._robot.get_dofs_pos(),
            self._robot.default_dofs_pos,
            self.ori_tol,
        )


@compile_model
def _dofs_pose_penalty_fused(
    gravity: torch.Tensor,
    up_vec: torch.Tensor,
    torso_height: torch.Tensor,
    dofs_pos: torch.Tensor,
    default_pos: torch.Tensor,
    ori_tol: float,
    z_des: float,
    pos_tol: float,
) -> torch.Tensor:
    # Gate: upright AND near desired height
    ori_error = torch.sum(torch.square(up_vec - gravity), dim=1)
    is_up = (ori_error < ori_tol).float()
    height = torch.clamp(torso_height, max=z_des)
    height_error = z_des - height
    is_at_h = (height_error < pos_tol).float()
    gate = is_up * is_at_h
    # Quadratic penalty (provides linear gradient even for large deviations)
    cost = torch.sum(torch.square(dofs_pos - default_pos), dim=1)
    return gate * cost


@REWARD_TERM_REGISTRY.register()
class GetupDofsPosePenalty(RewardTerm):
    """Quadratic penalty for DOF deviation from default, gated by upright + at height.

    Unlike getup_posture (exp-based, saturates for large deviations), this
    provides strong gradient to pull splayed legs back to the default pose.
    Use ``range_s`` in the RewardTerm config to control activation timing.
    """

    entity_name: str = "robot"
    z_des: float = 0.275
    ori_tol: float = 0.5
    pos_tol: float = 0.05

    def build(self) -> None:
        self._robot = self._env.entities[self.entity_name]
        self._neg_z = gravity_vec(self.num_envs, self.device)

    def compute(self) -> torch.Tensor:
        gravity = projected_gravity(self._robot.get_quat(), self._neg_z)
        torso_height = self._robot.get_pos()[:, 2]
        return _dofs_pose_penalty_fused(
            gravity,
            self._neg_z,
            torso_height,
            self._robot.get_dofs_pos(),
            self._robot.default_dofs_pos,
            self.ori_tol,
            self.z_des,
            self.pos_tol,
        )


@compile_model
def _reward_stand_still_fused(
    gravity: torch.Tensor,
    up_vec: torch.Tensor,
    torso_height: torch.Tensor,
    action: torch.Tensor,
    ori_tol: float,
    z_des: float,
    pos_tol: float,
) -> torch.Tensor:
    # Gate: upright AND at desired height
    ori_error = torch.sum(torch.square(up_vec - gravity), dim=1)
    is_up = (ori_error < ori_tol).float()
    height = torch.clamp(torso_height, max=z_des)
    height_error = z_des - height
    is_at_h = (height_error < pos_tol).float()
    gate = is_up * is_at_h
    # Stand still reward
    cost = torch.sum(torch.square(action), dim=1)
    rew = torch.exp(-0.5 * cost)
    return gate * rew


@REWARD_TERM_REGISTRY.register()
class GetupStandStill(RewardTerm):
    """Reward for zero action once upright and at desired height."""

    entity_name: str = "robot"
    z_des: float = 0.275
    ori_tol: float = 0.01

    def build(self) -> None:
        self._robot = self._env.entities[self.entity_name]
        self._neg_z = gravity_vec(self.num_envs, self.device)

    def compute(self) -> torch.Tensor:
        gravity = projected_gravity(self._robot.get_quat(), self._neg_z)
        torso_height = self._robot.get_pos()[:, 2]
        # The action manager holds the raw policy delta (ReferenceSource.DELTA controller).
        action = self._env.action_manager.action
        return _reward_stand_still_fused(
            gravity,
            self._neg_z,
            torso_height,
            action,
            self.ori_tol,
            self.z_des,
            0.005,
        )


@compile_model
def _hip_abduction_penalty(dofs_pos: torch.Tensor, default_pos: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.square(dofs_pos - default_pos), dim=1)


@REWARD_TERM_REGISTRY.register()
class GetupHipAbductionPenalty(RewardTerm):
    """Penalize hip abduction DOFs deviating from default (always active).

    Discourages wide leg opening during recovery. Default indices (0,1,2,3)
    correspond to FL_hip, RL_hip, FR_hip, RR_hip in Eden's Go1 DOF ordering.

    The DOF index list and the (sliced) default pose are resolved once in :meth:`build`.
    """

    entity_name: str = "robot"
    dofs_indices: Tuple[int, ...] = (0, 1, 2, 3)

    def build(self) -> None:
        self._robot = self._env.entities[self.entity_name]
        # A device tensor, not a list: indexing with a Python list uploads it on every call.
        self._idx = torch.tensor(list(self.dofs_indices), dtype=torch.long, device=self.device)
        default_pos = self._robot.default_dofs_pos
        if default_pos.dim() > 1:
            self._default_pos = default_pos[:, self._idx]
        else:
            self._default_pos = default_pos[self._idx].unsqueeze(0)

    def compute(self) -> torch.Tensor:
        dofs_pos = self._robot.get_dofs_pos()[:, self._idx]
        return _hip_abduction_penalty(dofs_pos, self._default_pos)


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


@TERMINATION_TERM_REGISTRY.register()
class GetupBodyContact(TerminationTerm):
    """Terminate if trunk, thigh, or hip touches the floor.

    Uses z-position of links as proxy since only feet have collision geometry. Only activates after
    ``delay_s`` seconds to allow the robot time to recover from its initial fallen state. The link indices
    and the delay (in steps) are resolved once in :meth:`build`.
    """

    entity_name: str = "robot"
    links_name: Tuple[str, ...] = (
        "base",
        "FL_thigh",
        "FR_thigh",
        "RL_thigh",
        "RR_thigh",
        "FL_hip",
        "FR_hip",
        "RL_hip",
        "RR_hip",
    )
    contact_threshold: float = 0.05
    delay_s: float = 1.0

    def build(self) -> None:
        self._robot = self._env.entities[self.entity_name]
        _, self._idx = self._robot.find_named_links_idx_local(list(self.links_name))
        self._delay_steps = int(self.delay_s / self._env.dt)

    def compute(self) -> torch.Tensor:
        # Don't terminate during the recovery window.
        # No `.any()` early-out: the `& past_delay` below already clears every env inside the window, and
        # reading the gate back would be a GPU->CPU sync every step.
        past_delay = self._env.episode_length_buf >= self._delay_steps
        link_pos = self._robot.get_links_pos(ls_idx_local=self._idx)  # (N, num_links, 3)
        contact = (link_pos[:, :, 2] < self.contact_threshold).any(dim=1)
        return contact & past_delay


# ---------------------------------------------------------------------------
# Events: Drop from random height with random orientation and joint angles
# ---------------------------------------------------------------------------


@EVENT_TERM_REGISTRY.register()
class DropFromHeight(EventTerm):
    """Reset the robot with a random orientation at a fixed height and randomized joint angles.

    With probability (1 - drop_prob), reset to the default standing pose instead.

    Ported from mujoco_playground getup._get_random_qpos + reset logic.
    """

    entity_name: str = "robot"
    drop_height: float = 0.5
    drop_prob: float = 0.6
    root_vel_range: Tuple[float, float] = (-0.5, 0.5)

    def __init__(self, env: EnvBase, options: EventTermOptions):
        super().__init__(env=env, options=options)
        self.entity: Entity | None = None

    def build(self) -> None:
        self.entity = self._env.entities[self.entity_name]
        self._check_every_dof_is_bounded()

    def _check_every_dof_is_bounded(self) -> None:
        """Reject a robot whose joints do not all have a finite range.

        ``compute`` draws each joint angle uniformly between ``soft_dofs_pos_limits``. An
        unbounded DOF — a wheel, a ball joint, or a hinge the model gives no ``range`` — carries
        -/+inf there, and ``sample_uniform`` turns that into ``rand * inf - inf`` = **NaN** (or
        -/+inf where only one side is unbounded), which the ``set_dofs_pos`` below writes straight
        into the sim as a joint position. A getup robot is a legged one whose every joint is
        range-limited, so an infinite limit here is a robot/task mismatch to name at build time,
        not a range to sample around.

        Refused rather than degraded on purpose. ``compute`` already holds ``default_dofs_pos``,
        so leaving just the unbounded rows at their default angle would cost nothing to write —
        but it would quietly pin a joint the caller asked to randomize, and keep doing so for
        the whole run. That is the fall-back-to-a-default-that-hides-the-error this codebase
        rules out: a wheeled-legged robot arriving here is a robot/task mismatch worth naming,
        not smoothing over. Widen this to a per-row skip only alongside a config flag that says
        so out loud.
        """
        soft_limits = self.entity.soft_dofs_pos_limits
        # ``is not a tensor``, not ``is None``: the two ways an entity carries no resolved limits
        # do not agree. ``_clear_dof_properties`` sets ``None`` on one with no DOFs at all, while
        # a kinematic or unbuilt entity never reaches ``_init_soft_pos_limits`` and simply keeps
        # the mirrored options *dict* (``{}`` by default) — on which ``isfinite`` below would be
        # an ``AttributeError`` rather than the named error this guard exists to give.
        if not torch.is_tensor(soft_limits):
            raise ValueError(
                f"{term_subject(self)} needs a simulated articulated entity, which resolves "
                f"`soft_dofs_pos_limits` to a tensor; this entity carries {type(soft_limits).__name__} "
                f"(it is non-articulated, kinematic, or unbuilt)."
            )
        # ``(1 | num_envs, num_dofs, 2)`` -> one flag per DOF, set if any env is unbounded.
        unbounded = (~soft_limits.isfinite().all(dim=-1)).any(dim=0)
        if not bool(unbounded.any()):
            return
        dofs_name = self.entity.dofs_name
        offenders = [
            dofs_name[i] if i < len(dofs_name) else f"dof[{i}]" for i in unbounded.nonzero().flatten().tolist()
        ]
        raise ValueError(
            f"{term_subject(self)} resets each joint to a uniform "
            f"sample of its soft position limits, but DOFs {offenders} have an infinite limit and "
            f"so no range to sample — the draw would be NaN. Give those joints a finite `range` in "
            f"the model (or a `soft_dofs_pos_limits` entry), or drop this term: continuous joints "
            f"such as wheels cannot be reset to a random angle this way."
        )

    def compute(self, envs_idx: Optional[slice | torch.Tensor] = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        # Int ids, once: every write below takes the selected rows, a sync per gather and per Genesis
        # setter through the auto-reset's bool mask and none through ids.
        envs_idx = env_indices(envs_idx)

        n = self._env.num_envs

        # Decide which envs get the random drop vs default standing
        drop_mask = torch.rand(n, device=self._env.device) < self.drop_prob
        # --- Random orientation ---
        rand_quat = torch.randn(n, 4, device=self._env.device)
        rand_quat = rand_quat / (rand_quat.norm(dim=1, keepdim=True) + 1e-6)
        # Default orientation
        default_quat = self.entity.default_root_quat.expand(n, -1)
        quat = torch.where(drop_mask.unsqueeze(1), rand_quat, default_quat)
        self.entity.set_quat(quat[envs_idx], envs_idx=envs_idx)

        # --- Height ---
        default_pos = device_constant(self.entity._options.default_root_pos, self._env.device)
        rand_pos = torch.zeros(n, 3, device=self._env.device)
        rand_pos[:, 2] = self.drop_height
        pos = torch.where(drop_mask.unsqueeze(1), rand_pos, default_pos.unsqueeze(0).expand(n, -1))
        self.entity.set_pos(pos[envs_idx], envs_idx=envs_idx)

        # --- Random joint angles ---
        soft_limits = self.entity.soft_dofs_pos_limits  # (1 or n, ndof, 2)
        lower = soft_limits[:, :, 0].expand(n, -1)
        upper = soft_limits[:, :, 1].expand(n, -1)
        rand_dofs = sample_uniform(lower, upper, (n, self.entity.num_dofs), device=self._env.device)
        default_dofs = self.entity.default_dofs_pos.expand(n, -1)
        dofs_pos = torch.where(drop_mask.unsqueeze(1), rand_dofs, default_dofs)
        self.entity.set_dofs_pos(dofs_pos[envs_idx], envs_idx=envs_idx)

        # --- Random root velocity ---
        rand_vel = sample_uniform(
            self.root_vel_range[0],
            self.root_vel_range[1],
            (n, 6),
            device=self._env.device,
        )
        # Zero velocity for non-drop envs
        zero_vel = torch.zeros(n, 6, device=self._env.device)
        vel = torch.where(drop_mask.unsqueeze(1), rand_vel, zero_vel)

        # Set linear and angular velocities via dofs of the floating base
        # The list, not a tensor built per reset: the entity resolves it to a cached `range`.
        base_dofs_index = self.entity.find_free_base_dofs_idx_local()
        self.entity.set_dofs_vel(
            vel[envs_idx],
            dofs_idx_local=base_dofs_index,
            envs_idx=envs_idx,
        )
