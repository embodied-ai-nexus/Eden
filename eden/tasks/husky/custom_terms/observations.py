"""Observation terms for the HUSKY skateboarding task.

Ports of ``mjlab_husky.tasks.skater.mdp.observations``. Terms that need the
gait-phase / contact state read it from the :class:`~eden.tasks.husky.custom_terms.commands.SkateCommand`
term (refreshed in ``post_physics_refresh``, before observations are computed).

The pelvis frame (``relative=True`` pose reads) is used wherever the source
reads ``root_link_*`` quantities — Genesis re-origins the base link onto its
inertial frame, and the pelvis inertial offset is ~9 cm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from genesis.utils.geom import inv_transform_by_quat

import eden as en
from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY, ObservationTerm
from eden.managers.terms.utils import gravity_vec, projected_gravity
from eden.utils.geom import heading_from_quat, quat_to_tan_norm, wrap_to_pi
from eden.utils.torch import device_constant

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.tasks.husky.custom_terms.commands import SkateCommand


def _skate(env: EnvBase, command_name: str) -> SkateCommand:
    return env.command_manager.get_term(command_name)


def _pelvis_quat(env: EnvBase, entity_name: str) -> torch.Tensor:
    return env.entities[entity_name].get_quat(relative=True)


@OBSERVATION_TERM_REGISTRY.register()
def robot_heading(env: EnvBase, *, entity_name: str = "robot") -> torch.Tensor:
    """World heading of the robot pelvis, shape (num_envs, 1)."""
    return heading_from_quat(_pelvis_quat(env, entity_name)).unsqueeze(-1)


@OBSERVATION_TERM_REGISTRY.register()
def pelvis_projected_gravity(env: EnvBase, *, entity_name: str = "robot") -> torch.Tensor:
    """Gravity direction in the pelvis (MJCF root) frame, shape (num_envs, 3)."""
    return projected_gravity(_pelvis_quat(env, entity_name), gravity_vec(env.num_envs, env.device))


@OBSERVATION_TERM_REGISTRY.register()
def skate_phase(env: EnvBase, *, command_name: str = "skate") -> torch.Tensor:
    """Gait-phase clock in [0, 1), shape (num_envs, 1)."""
    return _skate(env, command_name).phase.unsqueeze(-1)


@OBSERVATION_TERM_REGISTRY.register()
def skate_contact_phase(env: EnvBase, *, command_name: str = "skate") -> torch.Tensor:
    """One-hot ``[push, steer, push2steer, steer2push]`` phase, shape (num_envs, 4)."""
    return _skate(env, command_name).contact_phase.clone()


def _amp_trainer_selected(env: EnvBase) -> bool:
    """Whether this run hands rsl_rl an algorithm that could consume the ``amp`` group.

    ``rsl_rl.utils.resolve_callable`` reads a *bare* ``class_name`` ("PPO") as "a stock
    rsl_rl algorithm" and anything carrying a module path
    ("eden.learning.wrappers.rsl_rl.amp_ppo.ppo:AmpPPO") as a custom class. rsl_rl
    ships no AMP variant, so a bare name means the style reward is off; a qualified
    one is assumed to be ``AmpPPO`` or a peer of it. When there is no such field to read at all — an
    rl_games runner block, say — this answers ``True``: a diagnostic that guesses is
    worse than one that stays quiet.
    """
    algorithm = getattr(getattr(env.config, "runner_options", None), "algorithm", None)
    class_name = getattr(algorithm, "class_name", None)
    if not isinstance(class_name, str):
        return True
    return "." in class_name or ":" in class_name


@OBSERVATION_TERM_REGISTRY.register()
class SkatePushFlag(ObservationTerm):
    """Push-phase indicator, shape (num_envs, 1) — consumed by the AMP discriminator.

    A class term only so that :meth:`build` can say out loud when nothing is going to
    read it. The source augments PPO with an adversarial motion prior over the push
    phase, and the registered config selects that trainer (``AmpPPO``); a run that
    swaps in a stock rsl_rl algorithm computes the whole ``amp`` observation group each
    step for nobody. Training the plain MDP is a legitimate thing to do — it is a
    *harder, sparser* problem, worth about a quarter of the reward at 3000 iterations —
    but it is not what a HUSKY number is comparable to, and nothing else in a run says
    which one you got.
    """

    command_name: str = "skate"

    def build(self) -> None:
        """Warn a *training* run whose algorithm cannot consume this term."""
        # Not on the eval paths (`eden inference`, the example's play): there the
        # algorithm is never constructed, so plain PPO in the config means nothing.
        if self._env.training and not _amp_trainer_selected(self._env):
            en.logger.warning(
                "[husky] The AMP style reward is OFF: runner_options.algorithm.class_name is a "
                "stock rsl_rl algorithm, so the `amp` observation group is computed every step "
                "and read by nobody. This trains the plain MDP (~1/4 the reward at 3k iterations, "
                "and not comparable to a published HUSKY result). The task's default algorithm, "
                "`RslRlAmpPpoAlgorithmOptions` (eden.learning.wrappers.rsl_rl.amp_ppo), is the "
                "AMP-augmented trainer — restore it (algorithm=RslRlAmpPpoAlgorithmOptions(amp_motion_file=...), "
                "not class_name alone: AmpPPO needs the amp_* fields). "
                "Ignore this if the plain MDP is what you meant to train."
            )

    def compute(self) -> torch.Tensor:
        return _skate(self._env, self.command_name).contact_phase[:, 0:1].clone()


@OBSERVATION_TERM_REGISTRY.register()
def skate_heading_error(env: EnvBase, *, command_name: str = "skate") -> torch.Tensor:
    """Wrapped target-minus-board heading error, shape (num_envs, 1)."""
    skate = _skate(env, command_name)
    return wrap_to_pi(skate.target_heading_w - skate.board_heading_w()).unsqueeze(-1)


@OBSERVATION_TERM_REGISTRY.register()
def skate_pose_local(env: EnvBase, *, command_name: str = "skate") -> torch.Tensor:
    """Skateboard pose in the pelvis frame: position + tangent/normal 6D, shape (num_envs, 9)."""
    pos_b, quat_b = _skate(env, command_name).board_pose_in_robot_frame()
    return torch.cat([pos_b, quat_to_tan_norm(quat_b)], dim=-1)


@OBSERVATION_TERM_REGISTRY.register()
def skate_vel_local(env: EnvBase, *, entity_name: str = "robot", skateboard_name: str = "skateboard") -> torch.Tensor:
    """Skateboard linear velocity in the pelvis frame, shape (num_envs, 3)."""
    vel_w = env.entities[skateboard_name].get_vel()
    return inv_transform_by_quat(vel_w, _pelvis_quat(env, entity_name))


@OBSERVATION_TERM_REGISTRY.register()
def skate_ang_vel_local(
    env: EnvBase, *, entity_name: str = "robot", skateboard_name: str = "skateboard"
) -> torch.Tensor:
    """Skateboard angular velocity in the pelvis frame, shape (num_envs, 3)."""
    ang_w = env.entities[skateboard_name].get_ang()
    return inv_transform_by_quat(ang_w, _pelvis_quat(env, entity_name))


@OBSERVATION_TERM_REGISTRY.register()
def skateboard_roll(env: EnvBase, *, skateboard_name: str = "skateboard") -> torch.Tensor:
    """Deck tilt + both truck-roll joint angles, shape (num_envs, 3)."""
    # Columns of the `dofs_name`-order read, through a device index built once (`device_constant`):
    # indexing with the Python list `[0, 1, 4]` would upload it on every call, a GPU->CPU sync.
    columns = device_constant((0, 1, 4), env.device, dtype=torch.long)
    return env.entities[skateboard_name].get_dofs_pos()[:, columns]


@OBSERVATION_TERM_REGISTRY.register()
def transition_target_pos(env: EnvBase, *, command_name: str = "skate") -> torch.Tensor:
    """Blend-target positions of the tracked links (board frame), shape (num_envs, 42)."""
    target = _skate(env, command_name).transition_target_pos_b
    return target.reshape(target.shape[0], -1)


@OBSERVATION_TERM_REGISTRY.register()
def transition_target_quat(env: EnvBase, *, command_name: str = "skate") -> torch.Tensor:
    """Blend-target orientations of the tracked links (board frame), shape (num_envs, 56)."""
    target = _skate(env, command_name).transition_target_quat_b
    return target.reshape(target.shape[0], -1)


@OBSERVATION_TERM_REGISTRY.register()
def foot_contact_force(
    env: EnvBase, *, command_name: str = "skate", surface: str = "ground", foot: int = 0
) -> torch.Tensor:
    """Signed-log contact force on one foot from ``ground`` or ``board``, shape (num_envs, 3)."""
    skate = _skate(env, command_name)
    if surface == "ground":
        force = skate.foot_ground_force[:, foot]
    elif surface == "board":
        force = skate.foot_board_force[:, foot]
    else:
        raise ValueError(f"surface must be 'ground' or 'board', got {surface!r}")
    return torch.sign(force) * torch.log1p(force.abs())
