"""Observation terms for in-hand cube reorientation.

Port of ``wuji-mjlab``'s ``observations.py``. The policy group is deliberately narrow and
*hardware-reproducible*: normalized joint angles, the tracking error against the commanded
joint target, and the cube's pose relative to the wrist tag -- everything a real hand with
joint encoders and one wrist-mounted camera can measure. Everything that needs privileged
simulator state (true velocities, fingertip positions, the domain-randomization draw) goes
to the critic only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from genesis.utils.geom import inv_quat, inv_transform_by_quat

from eden.managers import OBSERVATION_TERM_REGISTRY, ObservationTerm
from eden.tasks.reorient.constants import TIP_LINKS
from eden.utils.geom import quat_mul, quat_to_rot6d_cols, random_quat_uniform
from eden.utils.string import term_subject

if TYPE_CHECKING:
    from eden.entities.dofs_default_mixin import DofSelection
    from eden.envs.base import EnvBase


def _normalized_dofs_pos(entity, dofs: DofSelection) -> torch.Tensor:
    """Joint positions mapped onto [-1, 1] across the soft DOF range.

    Two index spaces, deliberately both: the getter takes Genesis' local indices, while
    ``soft_dofs_pos_limits`` is columned by ``dofs_name``. They coincide for this hand and differ
    for any robot configured out of tree order.
    """
    soft = entity.soft_dofs_pos_limits[:, dofs.columns, :]
    center = 0.5 * (soft[..., 0] + soft[..., 1])
    half_range = 0.5 * (soft[..., 1] - soft[..., 0])
    return ((entity.get_dofs_pos(dofs_idx_local=dofs.dofs_idx_local) - center) / (half_range + 1e-6)).clamp(-1.0, 1.0)


class _HandDofsTerm(ObservationTerm):
    """Shared plumbing for the two joint-space observations.

    Parameters
    ----------
    entity_name:
        Scene entity name of the hand.
    """

    entity_name: str = "robot"

    def build(self) -> None:
        # Every controlled DOF: its columns index the soft limits, its local indices the state reads.
        self._dofs = self._env.entities[self.entity_name].find_named_dofs()


@OBSERVATION_TERM_REGISTRY.register()
class DofsPosNormalized(_HandDofsTerm):
    """Joint positions normalized to [-1, 1] by the soft DOF limits. ``(num_envs, n_dofs)``.

    Range-normalized rather than offset-from-default: this hand's joints have very
    different spans (0.74 rad of abduction against 2.0 rad of flexion), so a raw offset
    would hand the policy inputs of wildly different scale.
    """

    def compute(self) -> torch.Tensor:
        return _normalized_dofs_pos(self.resolve_entity(self.entity_name), self._dofs)


@OBSERVATION_TERM_REGISTRY.register()
class DofsPosTargetError(_HandDofsTerm):
    """Normalized joint tracking error, ``current - commanded``. ``(num_envs, n_dofs)``.

    The commanded value is the action term's *processed* target (post-clip, post-EMA,
    post-clamp), so this is the error the PD loop is actually working against. It tells the
    policy how much authority it has left: a finger pressed against the cube shows a large
    persistent error even though its command is not moving.

    Parameters
    ----------
    entity_name:
        Scene entity name of the hand.
    action_term_name:
        Key of the joint-position action term in ``action_options``.
    """

    action_term_name: str = "joint_pos"

    def build(self) -> None:
        super().build()
        # Touch the action term for fail-fast; it is resolved again per step because the
        # action manager owns its lifetime.
        self._env.action_manager.get_term(self.action_term_name)

    def compute(self) -> torch.Tensor:
        entity = self.resolve_entity(self.entity_name)
        soft = entity.soft_dofs_pos_limits[:, self._dofs.columns, :]
        center = 0.5 * (soft[..., 0] + soft[..., 1])
        half_range = 0.5 * (soft[..., 1] - soft[..., 0])
        target = self._env.action_manager.get_term(self.action_term_name).processed_action
        normalized_target = ((target - center) / (half_range + 1e-6)).clamp(-1.0, 1.0)
        return _normalized_dofs_pos(entity, self._dofs) - normalized_target


class _TagFrameTerm(ObservationTerm):
    """Shared plumbing for the tag-frame observations.

    Parameters
    ----------
    command_name:
        Key of the :class:`InHandReorientCommand` term, which owns the tag transform.
    object_entity_name:
        Scene entity name of the object.
    injection_prob:
        Per-env, per-step probability of replacing this observation with a random value.
        Upstream uses this on the two cube-tracking channels so the policy cannot become
        wholly dependent on a tag detection that will sometimes drop out on hardware.
    """

    command_name: str = "reorient"
    object_entity_name: str = "cube"
    injection_prob: float = 0.0

    def build(self) -> None:
        self._env.entities[self.object_entity_name]
        if not 0.0 <= self.injection_prob <= 1.0:
            raise ValueError(f"{type(self).__name__}: injection_prob must be in [0, 1], got {self.injection_prob}.")
        # Managers are constructed in ``_manager_order``, so the command manager already exists
        # when an observation term builds: a bad ``command_name`` raises at env construction
        # rather than on the first step. The handle is cached, not the state --
        # deployment hot-swaps entities, never terms.
        self._command = self._env.command_manager.get_term(self.command_name)

    def _inject(self, value: torch.Tensor, replacement: torch.Tensor) -> torch.Tensor:
        if self.injection_prob <= 0.0:
            return value
        mask = torch.rand(value.shape[0], 1, device=value.device) < self.injection_prob
        return torch.where(mask, replacement, value)


@OBSERVATION_TERM_REGISTRY.register()
class CubePosInTag(_TagFrameTerm):
    """Object position in the wrist-tag frame. ``(num_envs, 3)``.

    Absolute, not a delta from a reference pose: the policy sees where the cube actually is
    relative to the tag, which is what a camera reports and what generalizes across
    randomized spawn positions.
    """

    def compute(self) -> torch.Tensor:
        tag_pos_w, tag_quat_w = self._command.tag_pose_w()
        cube_pos_w = self.resolve_entity(self.object_entity_name).get_pos()
        cube_pos_tag = inv_transform_by_quat(cube_pos_w - tag_pos_w, tag_quat_w)
        return self._inject(cube_pos_tag, torch.empty_like(cube_pos_tag).uniform_(-0.5, 0.5))


@OBSERVATION_TERM_REGISTRY.register()
class GoalRotErrorInTag(_TagFrameTerm):
    """Residual rotation from goal to cube, in tag frame, as 6D. ``(num_envs, 6)``.

    ``mat(cube_in_tag * goal_in_tag^-1)`` reduced to its first two columns
    (:func:`~eden.utils.geom.quat_to_rot6d_cols`). Both quaternions
    are moved into tag frame first, which is what the deployed policy receives -- the
    hardware side never applies a wrist transform of its own.
    """

    def compute(self) -> torch.Tensor:
        command = self._command
        _, tag_quat_w = command.tag_pose_w()
        tag_quat_inv = inv_quat(tag_quat_w)
        cube_in_tag = quat_mul(tag_quat_inv, self.resolve_entity(self.object_entity_name).get_quat())
        goal_in_tag = quat_mul(tag_quat_inv, command.command)
        error = quat_to_rot6d_cols(quat_mul(cube_in_tag, inv_quat(goal_in_tag)))
        replacement = quat_to_rot6d_cols(random_quat_uniform(error.shape[0], device=error.device))
        return self._inject(error, replacement)


@OBSERVATION_TERM_REGISTRY.register()
class PalmRot6d(ObservationTerm):
    """Palm orientation in world frame as 6D. ``(num_envs, 6)``.

    Critic-only. Gives the value function the full 3-DOF wrist orientation under the wrist
    randomization; projected gravity would only pin two of the three degrees of freedom,
    leaving rotation about the gravity axis ambiguous.

    Parameters
    ----------
    entity_name:
        Scene entity name of the hand.
    palm_link:
        Link pattern selecting the palm.
    """

    entity_name: str = "robot"
    palm_link: str = ".*_palm_link"

    def build(self) -> None:
        entity = self._env.entities[self.entity_name]
        _, palm_idx = entity.find_named_links_idx_local(self.palm_link, preserve_order=True)
        self._palm_idx = torch.as_tensor(palm_idx, dtype=torch.long, device=self.device)

    def compute(self) -> torch.Tensor:
        quat = self.resolve_entity(self.entity_name).get_links_quat(ls_idx_local=self._palm_idx).squeeze(1)
        return quat_to_rot6d_cols(quat)


@OBSERVATION_TERM_REGISTRY.register()
class FingertipPositions(ObservationTerm):
    """Fingertip positions relative to the palm. ``(num_envs, 3 * n_tips)``.

    Critic-only. Palm-relative rather than upstream's world-minus-constant-offset, so the
    value is invariant to the wrist pose randomization instead of confounded with it.

    Parameters
    ----------
    entity_name:
        Scene entity name of the hand.
    tip_links:
        Link patterns selecting the fingertip frames.
    palm_link:
        Link pattern selecting the palm.
    """

    entity_name: str = "robot"
    tip_links: list[str] = [TIP_LINKS]
    palm_link: str = ".*_palm_link"

    def build(self) -> None:
        entity = self._env.entities[self.entity_name]
        _, tip_idx = entity.find_named_links_idx_local(self.tip_links, preserve_order=True)
        _, palm_idx = entity.find_named_links_idx_local(self.palm_link, preserve_order=True)
        if len(tip_idx) == 0:
            raise ValueError(
                f"{term_subject(self)}: tip_links={self.tip_links!r} matched no links. "
                f"Available links: {[link.name for link in entity.links]}"
            )
        self._tip_idx = torch.as_tensor(tip_idx, dtype=torch.long, device=self.device)
        self._palm_idx = torch.as_tensor(palm_idx, dtype=torch.long, device=self.device)

    def compute(self) -> torch.Tensor:
        entity = self.resolve_entity(self.entity_name)
        palm_pos = entity.get_links_pos(ls_idx_local=self._palm_idx).squeeze(1)
        palm_quat = entity.get_links_quat(ls_idx_local=self._palm_idx).squeeze(1)
        tip_rel = entity.get_links_pos(ls_idx_local=self._tip_idx) - palm_pos.unsqueeze(1)
        palm_quat_b = palm_quat.unsqueeze(1).expand(-1, tip_rel.shape[1], -1)
        tip_in_palm = inv_transform_by_quat(tip_rel, palm_quat_b)
        return tip_in_palm.flatten(start_dim=1)


@OBSERVATION_TERM_REGISTRY.register()
def object_lin_vel(env: EnvBase, *, entity_name: str = "cube") -> torch.Tensor:
    """World-frame object linear velocity. ``(num_envs, 3)``."""
    return env.entities[entity_name].get_vel()


@OBSERVATION_TERM_REGISTRY.register()
def object_ang_vel(env: EnvBase, *, entity_name: str = "cube") -> torch.Tensor:
    """World-frame object angular velocity. ``(num_envs, 3)``."""
    return env.entities[entity_name].get_ang()


@OBSERVATION_TERM_REGISTRY.register()
def command_state_progress(env: EnvBase, *, command_name: str = "reorient") -> torch.Tensor:
    """Goal state-machine progress and episode context. ``(num_envs, 4)``.

    ``[hold_progress, window_progress, episode_progress, goal_reach_count]``. Critic-only,
    and left unnormalized -- the critic's observation normalization handles the scales.
    Without it the value function cannot tell "about to bank a success" from "just started
    a fresh goal", two states with identical cube poses and very different returns.

    Reads the same live counters the reward does. That only works because the command measures
    them at phase 5 and leaves them alone until the switch at 17: an observation at phase 19 and
    a reward at phase 9 sit on opposite sides of that switch, so a quantity written *there*
    means two different things to its two readers. Reading a pre-switch snapshot instead would
    report a full success window to the critic for a goal replaced on that very step -- the
    read-side half of the hook rule in ``docs/guides/terms.md``.
    """
    command = env.command_manager.get_term(command_name)
    return torch.stack(
        [
            command.hold_counter.float() / max(command.success_hold_steps, 1),
            command.window_timer.float() / max(command.goal_switch_delay, 1),
            env.episode_length_buf.float() / env.max_episode_length,
            command.goal_reach_count.float(),
        ],
        dim=-1,
    )


@OBSERVATION_TERM_REGISTRY.register()
def cage_counter_progress(env: EnvBase, *, metric_name: str = "cage_escape") -> torch.Tensor:
    """Normalized cage-escape counter. ``(num_envs, 1)``.

    Critic-only. Because the counter decays gradually rather than resetting, a non-zero
    value can mean "recently outside" even while the cube is back inside -- which is exactly
    the near-drop state the value function needs to see.
    """
    term = env.metric_manager.get_term(metric_name)
    return (term.penalty_counter / float(term.max_outside_steps)).clamp(max=1.0).unsqueeze(-1)


class _EventStateTerm(ObservationTerm):
    """Read a buffer owned by an event term.

    The event term is looked up once at build and cached here -- which is why these are class
    terms rather than plain functions.

    Parameters
    ----------
    event_name:
        Key of the event term in ``event_options``.
    buffer_name:
        Attribute on that term to read.
    """

    event_name: str = "object_disturbance"
    buffer_name: str = ""

    def build(self) -> None:
        # The event manager is constructed before the observation manager (``_manager_order``),
        # so both the term name and the buffer name are checkable here.
        for _, name, term in self._env.event_manager.iter_terms():
            if name == self.event_name:
                self._event = term
                break
        else:
            available = [name for _, name, _ in self._env.event_manager.iter_terms()]
            raise ValueError(f"{type(self).__name__}: no event term named {self.event_name!r}. Available: {available}")
        if not hasattr(self._event, self.buffer_name):
            raise ValueError(
                f"{type(self).__name__}: event term {self.event_name!r} "
                f"({type(self._event).__name__}) has no attribute {self.buffer_name!r}."
            )

    def compute(self) -> torch.Tensor:
        return getattr(self._event, self.buffer_name)


@OBSERVATION_TERM_REGISTRY.register()
class PerturbationDirection(_EventStateTerm):
    """Unit direction of the most recent velocity kick. ``(num_envs, 3)``.

    Critic-only. Without it the value function sees an unexplained jolt in the cube's
    motion and can only average over it; with it, it can attribute the jolt to the
    disturbance rather than to the policy's action.
    """

    buffer_name: str = "force_dir"


@OBSERVATION_TERM_REGISTRY.register()
class PerturbationVelocity(_EventStateTerm):
    """Most recent velocity kick as ``concat(linear, zeros)``. ``(num_envs, 6)``.

    Critic-only. The trailing zeros are upstream's layout (it never applies an angular
    impulse) and are kept so a policy exported from either stack sees the same width.
    """

    buffer_name: str = "velocity_cache"


@OBSERVATION_TERM_REGISTRY.register()
class RandomizedParams(ObservationTerm):
    """The domain-randomization draw this env received, as ratios to the configured values.

    Critic-only, and the reason the critic can be accurate under heavy DR: the same
    observed cube motion means different things on a hand with half the rotor inertia, so
    without this the value function has to average over the whole randomization
    distribution.

    ``[armature_ratio, damping (absolute), kp_ratio, kd_ratio, object_mass_ratio]``, the first
    four each a mean over the hand's DOFs.
    Narrower than upstream's six columns, which read MuJoCo model fields directly: Genesis
    has no per-env geom size or geom friction to report, so those two are dropped. Damping
    is absolute rather than a ratio because this hand's configured damping is zero.

    Parameters
    ----------
    entity_name:
        Scene entity name of the hand.
    object_entity_name:
        Scene entity name of the object.
    """

    entity_name: str = "robot"
    object_entity_name: str = "cube"

    def build(self) -> None:
        entity = self._env.entities[self.entity_name]
        self._env.entities[self.object_entity_name]
        dofs = entity.find_named_dofs()
        self._dofs_idx = dofs.dofs_idx_local
        # Configured baselines: ``default_dofs_*`` stay at the configured value, while the DR
        # events write the sim-side values the getters in ``compute`` read. Sliced by column rather
        # than read through ``get_default_dofs_*``: ``compute`` divides by them every step, and the
        # stored ``(1 or num_envs, n)`` keeps a shared row broadcast.
        self._armature0 = entity.default_dofs_armature[:, dofs.columns].clamp_min(1e-9)
        self._kp0 = entity.default_dofs_kp[:, dofs.columns].clamp_min(1e-9)
        self._kd0 = entity.default_dofs_kd[:, dofs.columns].clamp_min(1e-9)
        # ``.mean()``, not ``.sum()``: with ``batch_links_info=True`` (which this task needs)
        # ``get_mass`` returns one row per env, so summing would make the baseline num_envs
        # times too large and the reported ratio ~0 -- silently, since it is a critic-only
        # input. Every env is still identical here (``build`` runs before startup DR), so
        # the mean is the configured mass; the expression also handles the 0-dim shape.
        self._mass0 = float(self._env.entities[self.object_entity_name].get_mass().mean().item())

    def compute(self) -> torch.Tensor:
        entity = self.resolve_entity(self.entity_name)
        obj = self.resolve_entity(self.object_entity_name)
        armature = entity.get_dofs_armature(dofs_idx_local=self._dofs_idx) / self._armature0
        damping = entity.get_dofs_damping(dofs_idx_local=self._dofs_idx)
        kp = entity.get_dofs_kp(dofs_idx_local=self._dofs_idx) / self._kp0
        kd = entity.get_dofs_kd(dofs_idx_local=self._dofs_idx) / self._kd0
        mass = obj.get_mass()
        mass = mass.reshape(-1) if mass.ndim > 0 else mass.reshape(1)
        mass_ratio = (mass / max(self._mass0, 1e-9)).expand(self.num_envs).unsqueeze(-1)
        return torch.cat(
            [
                armature.mean(dim=-1, keepdim=True).expand(self.num_envs, 1),
                damping.mean(dim=-1, keepdim=True).expand(self.num_envs, 1),
                kp.mean(dim=-1, keepdim=True).expand(self.num_envs, 1),
                kd.mean(dim=-1, keepdim=True).expand(self.num_envs, 1),
                mass_ratio,
            ],
            dim=-1,
        )


__all__ = [
    "CubePosInTag",
    "DofsPosNormalized",
    "DofsPosTargetError",
    "FingertipPositions",
    "GoalRotErrorInTag",
    "PalmRot6d",
    "PerturbationDirection",
    "PerturbationVelocity",
    "RandomizedParams",
    "cage_counter_progress",
    "command_state_progress",
    "object_ang_vel",
    "object_lin_vel",
]
