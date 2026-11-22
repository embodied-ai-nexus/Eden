"""The observation and termination terms a task trained by the SuperMimic runner configures.

The runner decodes the state group as :class:`RootJointState`, reads :class:`DofsPosBiased` as the joint term of the
actor's proprioception, and takes the clip-end truncation from :class:`ReferenceClipEnd`.
"""

from __future__ import annotations

import torch

from eden.constants import ReferenceSource
from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY, ObservationTerm
from eden.managers.termination_manager import TERMINATION_TERM_REGISTRY, TerminationTerm
from eden.utils.string import term_subject


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
        if getattr(self._env, "motion_manager", None) is None:
            raise ValueError(
                f"{term_subject(self)}: the state is expressed in the motion library's frame, which needs the "
                f"motion manager of a motion-tracking env, and {type(self._env).__name__} has none. Use the term in "
                "an EdenMimicConfig task."
            )

    def compute(self) -> torch.Tensor:
        entity = self.resolve_entity(self.entity_name)
        root_pos = entity.get_links_pos(ls_idx_local=self._root)[:, 0]
        # The origins the motion manager adds to the reference (None for an env without them), so the state and
        # the motion library share one frame.
        origins = self._env.motion_manager.env_origins
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
        source = getattr(term, "reference_source", None)
        if source != ReferenceSource.DEFAULT:
            raise ValueError(
                f"{term_subject(self)}: action term {self.action_term_name!r} has reference_source={source!r}, but "
                "the encoder reading is q minus the controller's default-pose offset, which leaves out the per-step "
                "reference of every other mode. Set its reference_source to ReferenceSource.DEFAULT."
            )
        self._controller = term

    def compute(self) -> torch.Tensor:
        return self.resolve_entity(self.entity_name).get_dofs_pos() - self._controller.effective_offset


@TERMINATION_TERM_REGISTRY.register()
class ReferenceClipEnd(TerminationTerm):
    """True on the last frame of each env's reference clip.

    Reads the validity of the reference's next frame, so the motion manager must hold at least two frames
    (``num_frames >= 2``).

    Parameters
    ----------
    entity_name : str
        The entity whose reference clip ends the episode.
    """

    entity_name: str = "robot"

    def build(self) -> None:
        motion = self._env.motion_manager
        if self.entity_name not in motion.data:
            raise ValueError(
                f"{term_subject(self)}: no motion term tracks entity {self.entity_name!r}. Tracked entities: "
                f"{sorted(motion.data)}."
            )
        if motion.num_frames < 2:
            raise ValueError(
                f"{term_subject(self)} reads the next reference frame, but the motion manager holds "
                f"num_frames={motion.num_frames}. Set motion_options.num_frames >= 2."
            )

    def compute(self, envs_idx: torch.Tensor | None = None) -> torch.Tensor:
        return ~self._env.motion_manager.data[self.entity_name].validity[:, 1]
