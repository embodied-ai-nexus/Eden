"""SuperMimic observation and termination terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY, ObservationTerm
from eden.managers.termination_manager import TERMINATION_TERM_REGISTRY
from eden.utils.string import term_subject

if TYPE_CHECKING:
    from eden.envs.base import MimicRLEnvBase


@OBSERVATION_TERM_REGISTRY.register()
class RootJointState(ObservationTerm):
    """The robot's generalized state ``[root_pos, root_quat, q, root_lin_vel, root_ang_vel, qd]``.

    The root is the robot's base link: its world position less the env origin, its wxyz orientation and the world
    frame twist of its origin. ``q`` and ``qd`` follow the entity's ``dofs_name``.
    """

    entity_name: str = "robot"

    def build(self) -> None:
        entity = self.resolve_entity(self.entity_name)
        self._root = entity.find_named_links_idx_local([entity.base_link_name])[1]

    def compute(self) -> torch.Tensor:
        entity = self.resolve_entity(self.entity_name)
        root_pos = entity.get_links_pos(ls_idx_local=self._root)[:, 0]
        origins = getattr(self._env, "env_origins", None)
        if origins is not None:
            root_pos = root_pos - origins
        return torch.cat(
            [
                root_pos,
                entity.get_links_quat(ls_idx_local=self._root)[:, 0],
                entity.get_dofs_pos(),
                entity.get_links_vel(ls_idx_local=self._root)[:, 0],
                entity.get_links_ang(ls_idx_local=self._root)[:, 0],
                entity.get_dofs_vel(),
            ],
            dim=-1,
        )


@OBSERVATION_TERM_REGISTRY.register()
class DofsPosBiased(ObservationTerm):
    """Joint positions relative to the default pose as a biased encoder reads them.

    An encoder bias ``b`` shifts the reading to ``q + b`` and the joint position controller's target to
    ``a * scale + q0 - b``. Randomized through the controller's per-env offset (``RandomizeDofsPosOffset`` draws
    ``-b``), it is read back here, so the reading is ``q - effective_offset`` with ``effective_offset = q0 - b``.

    Parameters
    ----------
    entity_name : str
        The robot.
    action_term_name : str
        Its joint position controller, under ``ReferenceSource.DEFAULT`` and controlling every DOF in ``dofs_name``
        order.
    """

    entity_name: str = "robot"
    action_term_name: str = ""

    def build(self) -> None:
        entity = self.resolve_entity(self.entity_name)
        term = self._env.action_manager.get_term(self.action_term_name)
        if list(term.dofs_name) != list(entity.dofs_name):
            raise ValueError(
                f"{term_subject(self)}: action term {self.action_term_name!r} controls {list(term.dofs_name)}, "
                f"but the encoder reading covers every DOF of {self.entity_name!r} in order: {list(entity.dofs_name)}."
            )
        self._controller = term

    def compute(self) -> torch.Tensor:
        return self.resolve_entity(self.entity_name).get_dofs_pos() - self._controller.effective_offset


@TERMINATION_TERM_REGISTRY.register()
def reference_clip_end(env: MimicRLEnvBase, *, entity_name: str) -> torch.Tensor:
    """True on the last frame of each env's reference clip.

    Reads the validity of the reference's next frame, so the motion manager must hold at least two frames
    (``num_frames >= 2``).
    """
    validity = env.motion_manager.data[entity_name].validity
    if validity.shape[1] < 2:
        raise ValueError(
            f"reference_clip_end reads the next reference frame, but the motion manager holds "
            f"num_frames={validity.shape[1]}. Set motion_options.num_frames >= 2."
        )
    return ~validity[:, 1]
