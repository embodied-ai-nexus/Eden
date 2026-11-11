"""Reward terms for the piano task, after RoboPianist's ``PianoWithShadowHands``.

Three of RoboPianist's five terms are here. The sustain-pedal reward is not, because this task has no pedal action;
the forearm-collision reward is not, because these hands fly and have no forearm to collide with anything. Its
fingering reward needs a fingering annotation per note, which only the (non-redistributable) PIG dataset carries, so
the shaping term here is the annotation-free one RoboPianist falls back to, from RP1M: pull *some* fingertip toward
each key that is due.

Every term reads the keyboard through :class:`~eden.tasks.piano.custom_terms.commands.SongCommand`, which measures
it once per step.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from genesis.utils.geom import transform_by_quat

from eden.managers import REWARD_TERM_REGISTRY, RewardTerm
from eden.managers.terms.utils import resolve_link_idx
from eden.tasks.piano.custom_terms.observations import base_offset_from_default

if TYPE_CHECKING:
    from eden.entities.rigid import RigidEntity
    from eden.envs.base import EnvBase

# The Gaussian tolerance falls to this value at `margin` past the bounds.
_VALUE_AT_MARGIN = 0.1
_GAUSSIAN_SCALE = math.sqrt(-2.0 * math.log(_VALUE_AT_MARGIN))


def gaussian_tolerance(error: torch.Tensor, bound: float, margin: float) -> torch.Tensor:
    """Gaussian tolerance: 1 for ``error <= bound``, then a Gaussian falling to 0.1 at ``margin`` past it.

    Parameters
    ----------
    error : torch.Tensor
        Non-negative error.
    bound : float
        Errors up to here earn full credit.
    margin : float
        Distance past ``bound`` at which the credit has fallen to 0.1.

    Returns
    -------
    torch.Tensor
        Credit in (0, 1], same shape as ``error``.
    """
    excess = (error - bound).clamp(min=0.0) / margin
    return torch.exp(-0.5 * (excess * _GAUSSIAN_SCALE) ** 2)


@REWARD_TERM_REGISTRY.register()
def key_press(env: EnvBase, *, command_name: str = "song", bound: float = 0.05, margin: float = 0.5) -> torch.Tensor:
    """RoboPianist's key-press reward: press what is due, and nothing else.

    Half of it is the mean, over the keys that should be down, of how far down each one is. The other half is all
    or nothing: it is lost if *any* key that should be up is down. A step with nothing due pays only that half.

    Parameters
    ----------
    env : EnvBase
        The environment.
    command_name : str
        The :class:`SongCommand`.
    bound : float
        A due key within this fraction of the bottom of its stroke earns full credit.
    margin : float
        Gaussian falloff of that credit, as a fraction of the stroke.

    Returns
    -------
    torch.Tensor
        ``(num_envs,)`` reward in [0, 1].
    """
    song = env.command_manager.get_term(command_name)
    due = song.goal.float()
    pressed = (gaussian_tolerance(1.0 - song.key_pos, bound, margin) * due).sum(dim=-1) / due.sum(dim=-1).clamp(min=1.0)
    wrong_key_down = (song.key_down & ~song.goal).any(dim=-1)
    return 0.5 * pressed + 0.5 * (~wrong_key_down).float()


def _driven_dofs_idx(entity: RigidEntity) -> torch.Tensor:
    """Local indices of every DOF a controller drives: the joints, plus a floating base's six if it has one.

    A free base is not in ``dofs_name`` — Eden keeps that for the joints — so reading an entity's DOFs the
    ordinary way silently leaves out everything a
    :class:`~eden.managers.terms.actions.FreeBasePoseController` does with the hand.

    Parameters
    ----------
    entity : RigidEntity
        The entity to read.

    Returns
    -------
    torch.Tensor
        Local DOF indices, the base's first.
    """
    joints_idx = entity.dofs_idx_local
    if not entity.free_joints:
        return joints_idx
    base_idx = torch.as_tensor(entity.find_free_base_dofs_idx_local(), dtype=joints_idx.dtype, device=joints_idx.device)
    return torch.cat((base_idx, joints_idx))


@REWARD_TERM_REGISTRY.register()
def actuation_energy(env: EnvBase, *, entity_names: list[str]) -> torch.Tensor:
    """Mechanical power drawn by the actuators, ``sum |tau * omega|``, over ``entity_names``.

    A floating base's six DOFs count too: flying a hand along the keyboard is what its controller spends power on,
    and leaving it out would price moving the whole hand at nothing while charging for a finger.

    Parameters
    ----------
    env : EnvBase
        The environment.
    entity_names : list[str]
        Actuated entities to sum over.

    Returns
    -------
    torch.Tensor
        ``(num_envs,)`` power in watts. Weight it negatively.
    """
    power = torch.zeros(env.num_envs, device=env.device)
    for name in entity_names:
        entity = env.entities[name]
        dofs_idx = _driven_dofs_idx(entity)
        force = entity.get_dofs_control_force(dofs_idx_local=dofs_idx)
        power += (force * entity.get_dofs_vel(dofs_idx_local=dofs_idx)).abs().sum(dim=-1)
    return power


@REWARD_TERM_REGISTRY.register()
def rest_pose_deviation(env: EnvBase, *, entity_names: list[str]) -> torch.Tensor:
    """Squared distance of every joint from its rest position, summed over ``entity_names``.

    A floating base counts as a joint here: how far its root has moved from the pose it rests in, and how far it
    has turned. Playing a note takes a finger about half a radian from rest at two joints, which this barely
    charges for. What it charges for is everything else: a hand with nothing to play that drifts into a splayed
    pose, or one that wanders off down the keyboard and twists to reach a key a finger could have reached. The
    key-press reward is indifferent to all of it, so without this a policy that plays every note correctly can
    still look nothing like a pianist.

    Parameters
    ----------
    env : EnvBase
        The environment.
    entity_names : list[str]
        Actuated entities to sum over.

    Returns
    -------
    torch.Tensor
        ``(num_envs,)`` squared deviation (rad^2, and m^2 for a displacement). Weight it negatively.
    """
    deviation = torch.zeros(env.num_envs, device=env.device)
    for name in entity_names:
        entity = env.entities[name]
        deviation += torch.square(entity.get_dofs_pos() - entity.default_dofs_pos).sum(dim=-1)
        if entity.free_joints:
            moved, turned = base_offset_from_default(entity)
            deviation += moved.square().sum(dim=-1) + turned.square().sum(dim=-1)
    return deviation


@REWARD_TERM_REGISTRY.register()
class FingertipToKey(RewardTerm):
    """Shaping: for each key due soon, how close the nearest fingertip is to it.

    The mean, over the keys due within the command's ``shaping_horizon_steps``, of a Gaussian credit on the
    distance from the key's press point to whichever fingertip is nearest. Zero when nothing is due. Without it
    the key-press reward is all the policy has, and that is silent until a finger happens to find the right key.

    RP1M assigns fingertips to keys one-to-one with the Hungarian algorithm. Nearest-fingertip is the batched
    approximation: it differs only when one finger is the nearest to two due keys at once, and there the
    key-press reward is what separates them.

    Parameters
    ----------
    command_name : str
        The :class:`SongCommand`.
    piano_entity_name : str
        The keyboard entity, whose links ``key_<i>`` are its keys.
    fingertip_links : dict[str, list[str]]
        Fingertip link names, by hand entity.
    press_point : tuple[float, float, float]
        Where on a key a finger should land, in the key link's frame (origin at the key's centre, +x toward
        the player, +z up). The default is a white key's top surface, 2.6 cm in front of centre, which is
        RoboPianist's target; it sits 2 mm below a black key's top surface.
    bound : float
        A fingertip within this distance of the press point earns full credit, in metres.
    margin : float
        Gaussian falloff of that credit, in metres.
    """

    command_name: str = "song"
    piano_entity_name: str = "piano"
    fingertip_links: dict[str, list[str]] = {}
    press_point: tuple[float, float, float] = (0.026, 0.0, 0.011)
    bound: float = 0.01
    margin: float = 0.1

    def build(self) -> None:
        if not self.fingertip_links:
            raise ValueError("FingertipToKey needs `fingertip_links`: fingertip link names by hand entity.")
        self._song = self._env.command_manager.get_term(self.command_name)
        piano = self.resolve_entity(self.piano_entity_name)
        self._keys_idx = resolve_link_idx(piano, [f"key_{key}" for key in range(self._song.n_keys)], self.device)
        self._tips_idx = {
            name: resolve_link_idx(self.resolve_entity(name), links, self.device)
            for name, links in self.fingertip_links.items()
        }
        self._press_point = torch.tensor(self.press_point, device=self.device).expand(
            self.num_envs, self._song.n_keys, 3
        )

    def compute(self) -> torch.Tensor:
        piano = self.resolve_entity(self.piano_entity_name)
        targets = piano.get_links_pos(ls_idx_local=self._keys_idx) + transform_by_quat(
            self._press_point, piano.get_links_quat(ls_idx_local=self._keys_idx)
        )
        tips = torch.cat(
            [self.resolve_entity(name).get_links_pos(ls_idx_local=idx) for name, idx in self._tips_idx.items()], dim=1
        )
        nearest = torch.cdist(targets, tips).amin(dim=-1)  # (num_envs, n_keys)
        due = self._song.goal_soon.float()
        credit = gaussian_tolerance(nearest, self.bound, self.margin)
        return (credit * due).sum(dim=-1) / due.sum(dim=-1).clamp(min=1.0)


__all__ = ["FingertipToKey", "actuation_energy", "gaussian_tolerance", "key_press", "rest_pose_deviation"]
