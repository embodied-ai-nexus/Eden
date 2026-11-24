"""A motion term over a root-and-joints dataset that computes link states by forward kinematics.

A large library stored with every tracked link's pose and twist per frame may not fit on one GPU (all of BONES-SEED
is ~47.6M frames: ~13.5 GB as root and joint state alone). :class:`KinematicLinksMotion` reads a dataset that keeps
root and joints only (:class:`~eden.datasets.base.RootJointsOnlyMixin`) and fills the reference buffer's link fields
for the frames the envs read, so the reward, termination and observation terms work unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.managers.motion_manager import MOTION_TERM_REGISTRY, MotionTerm
from eden.utils.motion_reference.kinematics import ForwardKinematics
from eden.utils.motion_reference.motion_data import LINK_FIELDS
from eden.utils.motion_reference.utils import derive_body_frame_link_fields

if TYPE_CHECKING:
    from collections.abc import Sequence

    from eden.envs.base import EnvBase
    from eden.options.managers.motions import MotionTermOptions
    from eden.utils.motion_reference.motion_data import MotionReferenceData


def link_fields(
    fk: ForwardKinematics,
    base_pos: torch.Tensor,
    base_quat: torch.Tensor,
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    dofs_pos: torch.Tensor,
    dofs_vel: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """The eight link fields, in ``LINK_FIELDS`` order, of ``(rows, ...)`` world-frame root and joint states."""
    links_w = fk(base_pos, base_quat, dofs_pos, base_lin_vel, base_ang_vel, dofs_vel)
    links_b = derive_body_frame_link_fields(base_pos, base_quat, base_lin_vel, base_ang_vel, *links_w)
    return (*links_w, *links_b)


#: The root and joint fields the kinematics read, in the order :func:`link_fields` takes them.
_STATE_FIELDS = ("base_pos_w", "base_quat_w", "base_lin_vel_w", "base_ang_vel_w", "dofs_pos", "dofs_vel")


@MOTION_TERM_REGISTRY.register()
class KinematicLinksMotion(MotionTerm):
    """Reference links by forward kinematics.

    The term tracks one entity, the key of ``entities_map``. ``target_links_name`` of its sequence lists the links
    the reference buffer holds, by name and in buffer order: the world-frame link fields, and the body-frame ones
    derived from them, are computed for those links at the frames the envs read. The entity's asset file gives the
    kinematic tree, a free root and hinge joints. Clip selection, start times and every other option are the base
    term's.

    Parameters
    ----------
    compile_kinematics : bool
        Run the kinematics through ``torch.compile`` on a CUDA device, which fuses its few thousand small
        elementwise operations into a few kernels. The first fill of every env and the first fill of a subset each
        wait for a compile, and the link fields differ from the eager ones in the last bits. ``False`` runs the
        pass eagerly, as a CPU device always does.
    """

    compile_kinematics: bool = True

    def __init__(self, env: EnvBase, options: MotionTermOptions) -> None:
        super().__init__(env=env, options=options)
        name = type(self).__name__
        if len(self.entities_map) != 1:
            raise ValueError(f"{name} tracks one entity, got entities_map={self.entities_map}.")
        ((entity_name, key),) = self.entities_map.items()
        self.sequence_key: str = key
        entity_options = getattr(env.scene_options, entity_name, None)
        if entity_options is None:
            raise ValueError(
                f"{name}: entities_map names entity {entity_name!r}, which the scene options do not have "
                f"({list(env.scene_options.keys())})."
            )
        if not getattr(entity_options, "file", None):
            raise ValueError(
                f"{name} reads the kinematic tree from the entity's asset file, and entity {entity_name!r} "
                f"({type(entity_options).__name__}) has none. Track an entity loaded from an MJCF or a URDF."
            )
        #: ``(root pose, root twist, joints) -> link poses and twists``, in ``target_links_name`` order.
        self.fk = ForwardKinematics(
            entity_options.fetch_asset(), self.target_dofs_name[key], self.target_links_name[key]
        ).to(self.device)
        self._compiled = self.compile_kinematics and torch.device(self.device).type == "cuda"
        self._link_fields = torch.compile(link_fields) if self._compiled else link_fields

    def derives_links(self, motion_sequence_key: str) -> bool:
        return True

    def derive_reference_fields(
        self,
        motion_sequence_key: str,
        envs_idx: slice | Sequence[int] | torch.Tensor,
        data_buffer: MotionReferenceData,
    ) -> None:
        """The link fields of ``data_buffer[envs_idx]`` by forward kinematics of its root and joint fields."""
        # torch.compile compiles once per grad mode and input layout. Two layouts are needed: every env through a
        # slice, and a subset through its ids, whose row count is declared dynamic from its first fill. Inference
        # mode is forced, so a training loop's steps and the env's first reset share the compiled passes.
        with torch.inference_mode():
            state = [getattr(data_buffer, field)[envs_idx].flatten(0, 1) for field in _STATE_FIELDS]
            if self._compiled and not isinstance(envs_idx, slice):
                for rows in state:
                    torch._dynamo.maybe_mark_dynamic(rows, 0)
            links = self._link_fields(self.fk, *state)
            for field, value in zip(LINK_FIELDS, links, strict=True):
                dst = getattr(data_buffer, field, None)
                if dst is not None:
                    dst[envs_idx] = value.reshape(-1, *dst.shape[1:])
