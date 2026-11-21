"""A motion term over a root-and-joints dataset that computes link states by forward kinematics.

A large library stored with every tracked link's pose and twist per frame may not fit on one GPU (all of BONES-SEED
is ~47.6M frames: ~13.5 GB as root and joint state alone). :class:`KinematicLinksMotion` reads a dataset that keeps
root and joints only (:class:`~eden.datasets.base.RootJointsOnlyMixin`) and fills the reference buffer's link fields
for the frames the envs read, so the reward, termination and observation terms work unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from eden.managers.motion_manager import MOTION_TERM_REGISTRY, MotionTerm
from eden.utils.motion_reference.kinematics import ForwardKinematics
from eden.utils.motion_reference.motion_data import LINK_FIELDS
from eden.utils.motion_reference.utils import derive_body_frame_link_fields

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch

    from eden.envs.base import EnvBase
    from eden.options.managers.motions import MotionTermOptions
    from eden.utils.motion_reference.motion_data import MotionReferenceData


@MOTION_TERM_REGISTRY.register()
class KinematicLinksMotion(MotionTerm):
    """Reference links by forward kinematics.

    The term tracks one entity, the key of ``entities_map``. ``target_links_name`` of its sequence lists the links
    the reference buffer holds, by name and in buffer order: the world-frame link fields, and the body-frame ones
    derived from them, are computed for those links at the frames the envs read. The entity's asset file gives the
    kinematic tree, a free root and hinge joints. Clip selection, start times and every other option are the base
    term's.
    """

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

    def derives_links(self, motion_sequence_key: str) -> bool:
        return True

    def derive_reference_fields(
        self,
        motion_sequence_key: str,
        envs_idx: slice | Sequence[int] | torch.Tensor,
        data_buffer: MotionReferenceData,
    ) -> None:
        """The link fields of ``data_buffer[envs_idx]`` by forward kinematics of its root and joint fields."""
        base_pos, base_quat, base_lin, base_ang, dofs_pos, dofs_vel = (
            getattr(data_buffer, field)[envs_idx].flatten(0, 1)
            for field in ("base_pos_w", "base_quat_w", "base_lin_vel_w", "base_ang_vel_w", "dofs_pos", "dofs_vel")
        )
        links_w = self.fk(base_pos, base_quat, dofs_pos, base_lin, base_ang, dofs_vel)
        links_b = derive_body_frame_link_fields(base_pos, base_quat, base_lin, base_ang, *links_w)
        for field, value in zip(LINK_FIELDS, (*links_w, *links_b), strict=True):
            dst = getattr(data_buffer, field, None)
            if dst is not None:
                dst[envs_idx] = value.reshape(-1, *dst.shape[1:])
