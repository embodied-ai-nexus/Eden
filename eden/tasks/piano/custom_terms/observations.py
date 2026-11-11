"""Observation terms for the piano task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from genesis.utils.geom import inv_quat

from eden.managers import OBSERVATION_TERM_REGISTRY
from eden.utils.geom import axis_angle_from_quat, quat_mul

if TYPE_CHECKING:
    from eden.entities.rigid import RigidEntity
    from eden.envs.base import EnvBase


def base_offset_from_default(entity: RigidEntity) -> tuple[torch.Tensor, torch.Tensor]:
    """How far an entity's root has moved and turned from its default root pose.

    Both in the authored frame and about world axes, which is the frame and the convention
    :class:`~eden.managers.terms.actions.FreeBasePoseController` takes its action in: a zero action holds a zero
    offset here.

    Parameters
    ----------
    entity : RigidEntity
        A floating-base entity.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(num_envs, 3)`` displacement in metres and ``(num_envs, 3)`` rotation vector in radians.
    """
    default_pos, default_quat = entity.get_default_root_pose()
    turn = quat_mul(entity.get_quat(relative=True), inv_quat(default_quat))
    return entity.get_pos(relative=True) - default_pos, axis_angle_from_quat(turn)


@OBSERVATION_TERM_REGISTRY.register()
def base_pose_from_default(env: EnvBase, *, entity_name: str) -> torch.Tensor:
    """``(num_envs, 6)`` where a floating base is, relative to the pose it rests in.

    Three metres of displacement and a three-component rotation vector, both about world axes — the same six
    numbers, in the same order, that a :class:`~eden.managers.terms.actions.FreeBasePoseController` action asks
    for, so the policy sees its command and its outcome in one frame.

    Parameters
    ----------
    env : EnvBase
        The environment.
    entity_name : str
        The floating-base entity, here a hand.

    Returns
    -------
    torch.Tensor
        The displacement and the rotation vector, concatenated.
    """
    return torch.cat(base_offset_from_default(env.entities[entity_name]), dim=-1)


@OBSERVATION_TERM_REGISTRY.register()
def piano_key_positions(env: EnvBase, *, command_name: str = "song") -> torch.Tensor:
    """``(num_envs, n_keys)`` travel of every key, 0 at rest and 1 at the bottom of its stroke.

    Read from the keyboard at the moment of the call, not from the command term's per-step measurement, so an
    environment that was just reset is observed with its keys up.

    Parameters
    ----------
    env : EnvBase
        The environment.
    command_name : str
        The :class:`~eden.tasks.piano.custom_terms.commands.SongCommand`, which knows each key's stroke.

    Returns
    -------
    torch.Tensor
        The key positions.
    """
    return env.command_manager.get_term(command_name).key_positions()


__all__ = ["base_offset_from_default", "base_pose_from_default", "piano_key_positions"]
