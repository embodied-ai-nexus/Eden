"""UWLab's OmniReset observations: joint angles in Isaac Lab's order and relative poses as position + axis-angle."""

from __future__ import annotations

import torch

from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY, ObservationTerm
from eden.managers.terms.utils import resolve_link_idx
from eden.utils.geom import axis_angle_from_quat, subtract_frame_transforms
from eden.utils.string import term_subject


@OBSERVATION_TERM_REGISTRY.register()
class JointPositionsByName(ObservationTerm):
    """Joint angles of ``entity_name`` for ``joints_name``, in that order, passive joints included.

    ``dofs_pos`` reads only an entity's controlled DOFs; UWLab's ``joint_pos`` reads every articulation joint,
    the 2F-85's five passive linkage joints among them, in PhysX's breadth-first order.
    """

    entity_name: str = "robot"
    joints_name: list[str] = []

    def build(self) -> None:
        entity = self.resolve_entity(self.entity_name)
        names, idx = entity.find_named_dofs_idx_local(list(self.joints_name), preserve_order=True)
        if names != list(self.joints_name):
            missing = sorted(set(self.joints_name) - set(names))
            raise ValueError(
                f"{term_subject(self)}: the entity has no single-DOF joint(s) {missing}. "
                f"Joints it has: {[joint.name for joint in entity.joints]}."
            )
        self._dofs_idx = idx

    def compute(self, *args, **kwargs) -> torch.Tensor:
        return self.resolve_entity(self.entity_name).get_dofs_pos(dofs_idx_local=self._dofs_idx)


@OBSERVATION_TERM_REGISTRY.register()
class PoseInFrame(ObservationTerm):
    """Pose of a target frame in a root frame: position (3) then axis-angle (3), angle in ``[0, pi]``.

    UWLab's ``target_asset_pose_in_root_asset_frame(rotation_repr="axis_angle")``. A frame is an entity's link
    (``*_link_name``) or, with no link named, the entity's authored root (``get_pos(relative=True)``), which is
    Isaac Lab's ``body_link`` pose of a rigid object.
    """

    target_entity_name: str = ""
    target_link_name: str | None = None
    root_entity_name: str = "robot"
    root_link_name: str | None = None

    def build(self) -> None:
        self._target_link = self._resolve(self.target_entity_name, self.target_link_name)
        self._root_link = self._resolve(self.root_entity_name, self.root_link_name)

    def _resolve(self, entity_name: str, link_name: str | None) -> int | None:
        entity = self.resolve_entity(entity_name)
        return None if link_name is None else int(resolve_link_idx(entity, [link_name], self.device)[0])

    def _pose(self, entity_name: str, link_idx: int | None) -> tuple[torch.Tensor, torch.Tensor]:
        entity = self.resolve_entity(entity_name)
        if link_idx is None:
            return entity.get_pos(relative=True), entity.get_quat(relative=True)
        return entity.get_links_pos(ls_idx_local=[link_idx])[:, 0], entity.get_links_quat(ls_idx_local=[link_idx])[:, 0]

    def compute(self, *args, **kwargs) -> torch.Tensor:
        target_pos, target_quat = self._pose(self.target_entity_name, self._target_link)
        root_pos, root_quat = self._pose(self.root_entity_name, self._root_link)
        pos, quat = subtract_frame_transforms(root_pos, root_quat, target_pos, target_quat)
        return torch.cat([pos, axis_angle_from_quat(quat)], dim=-1)
