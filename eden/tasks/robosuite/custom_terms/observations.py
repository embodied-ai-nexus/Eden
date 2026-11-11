"""Observation terms for the robosuite ports: robot proprioception and object-relative vectors.

robosuite's default proprio (``robot0_joint_pos_cos/sin``, ``joint_vel``, ``eef_pos``, ``eef_quat``,
``gripper_qpos/qvel``) and its per-task ``gripper_to_<obj>`` vectors. All are class terms so the DOF
and link lookups happen once in ``build()``; entities are re-resolved each step (deployment
hot-swaps them). Quaternions are wxyz (robosuite emits xyzw).
"""

from __future__ import annotations

import torch

from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY, ObservationTerm
from eden.managers.terms.utils import resolve_link_idx
from eden.tasks.robosuite.custom_terms.common import GraspFrameMixin, ObjectPointMixin, RigMixin


def _resolve_dofs(term: ObservationTerm, entity_name: str, dofs_name: list[str]) -> torch.Tensor:
    entity = term.resolve_entity(entity_name)
    _, idx = entity.find_named_dofs_idx_local(list(dofs_name), preserve_order=True)
    return torch.as_tensor(idx, dtype=torch.long, device=term.device)


@OBSERVATION_TERM_REGISTRY.register()
class ArmDofsPosSinCos(RigMixin, ObservationTerm):
    """``[cos(q), sin(q)]`` of the arm joints — robosuite's ``joint_pos_cos`` + ``joint_pos_sin``. ``(N, 2k)``."""

    rig: str = "robot"

    def build(self) -> None:
        rig = self._rig_options()
        self._arm_name = rig.arm_name
        self._idx = _resolve_dofs(self, rig.arm_name, rig.arm_dofs)

    def compute(self) -> torch.Tensor:
        q = self.resolve_entity(self._arm_name).get_dofs_pos(dofs_idx_local=self._idx)
        return torch.cat((torch.cos(q), torch.sin(q)), dim=-1)


@OBSERVATION_TERM_REGISTRY.register()
class ArmDofsVel(RigMixin, ObservationTerm):
    """Arm joint velocities (robosuite ``joint_vel``). ``(N, k)``."""

    rig: str = "robot"

    def build(self) -> None:
        rig = self._rig_options()
        self._arm_name = rig.arm_name
        self._idx = _resolve_dofs(self, rig.arm_name, rig.arm_dofs)

    def compute(self) -> torch.Tensor:
        return self.resolve_entity(self._arm_name).get_dofs_vel(dofs_idx_local=self._idx)


@OBSERVATION_TERM_REGISTRY.register()
class GripperDofsPos(RigMixin, ObservationTerm):
    """Finger joint positions (robosuite ``gripper_qpos``). ``(N, k)``."""

    rig: str = "robot"

    def build(self) -> None:
        rig = self._rig_options()
        self._gripper_name = rig.gripper_name
        self._idx = _resolve_dofs(self, rig.gripper_name, rig.gripper_dofs)

    def compute(self) -> torch.Tensor:
        return self.resolve_entity(self._gripper_name).get_dofs_pos(dofs_idx_local=self._idx)


@OBSERVATION_TERM_REGISTRY.register()
class GripperDofsVel(RigMixin, ObservationTerm):
    """Finger joint velocities (robosuite ``gripper_qvel``). ``(N, k)``."""

    rig: str = "robot"

    def build(self) -> None:
        rig = self._rig_options()
        self._gripper_name = rig.gripper_name
        self._idx = _resolve_dofs(self, rig.gripper_name, rig.gripper_dofs)

    def compute(self) -> torch.Tensor:
        return self.resolve_entity(self._gripper_name).get_dofs_vel(dofs_idx_local=self._idx)


@OBSERVATION_TERM_REGISTRY.register()
class EefPos(GraspFrameMixin, ObservationTerm):
    """World position of the grasp point — robosuite's ``eef_pos`` (the ``grip_site``). ``(N, 3)``."""

    rig: str = "robot"

    def build(self) -> None:
        self._build_grasp_frame()

    def compute(self) -> torch.Tensor:
        return self._grasp_pos_w()


@OBSERVATION_TERM_REGISTRY.register()
class EefQuat(RigMixin, ObservationTerm):
    """World orientation (wxyz) of the gripper mount link — robosuite's ``eef_quat``. ``(N, 4)``."""

    rig: str = "robot"

    def build(self) -> None:
        rig = self._rig_options()
        self._arm_name = rig.arm_name
        self._idx = resolve_link_idx(self.resolve_entity(rig.arm_name), [rig.ee_link], self.device)

    def compute(self) -> torch.Tensor:
        return self.resolve_entity(self._arm_name).get_links_quat(ls_idx_local=self._idx)[:, 0]


@OBSERVATION_TERM_REGISTRY.register()
class GripperToObject(ObjectPointMixin, GraspFrameMixin, ObservationTerm):
    """``object_point - grasp_pos`` in the world frame — robosuite's ``gripper_to_<obj>_pos``. ``(N, 3)``.

    The object point is the root by default; ``object_link_name`` targets one link (the door's latch)
    and ``object_offset`` a fixed point in that frame (robosuite's ``handle`` / ``pot_handle0`` sites).
    """

    rig: str = "robot"
    object_name: str = "object"
    object_link_name: str | None = None
    object_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def build(self) -> None:
        self._build_grasp_frame()
        self._build_object_point()

    def compute(self) -> torch.Tensor:
        return self._object_point_w() - self._grasp_pos_w()


@OBSERVATION_TERM_REGISTRY.register()
class ObjectPointPos(ObjectPointMixin, ObservationTerm):
    """World position of a point of an object: its root, a link, or a fixed offset in either frame. ``(N, 3)``.

    robosuite reads sites (``handle``, ``pot_handle0``); Genesis has none, so the site is its offset in
    the parent body's frame.
    """

    object_name: str = "object"
    object_link_name: str | None = None
    object_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def build(self) -> None:
        self._build_object_point()

    def compute(self) -> torch.Tensor:
        return self._object_point_w()


@OBSERVATION_TERM_REGISTRY.register()
class ObjectToObject(ObservationTerm):
    """``pos(object_b) - pos(object_a)`` in the world frame (robosuite's ``cubeA_to_cubeB``). ``(N, 3)``.

    ``relative=True`` on both: robosuite subtracts two body origins, and a difference of two solver
    frames carries each body's centre-of-mass offset rotated by its own orientation, so the vector
    would not vanish when the two origins coincide.
    """

    object_a_name: str = "object_a"
    object_b_name: str = "object_b"

    def build(self) -> None:
        self.resolve_entity(self.object_a_name)
        self.resolve_entity(self.object_b_name)

    def compute(self) -> torch.Tensor:
        return self.resolve_entity(self.object_b_name).get_pos(relative=True) - self.resolve_entity(
            self.object_a_name
        ).get_pos(relative=True)


@OBSERVATION_TERM_REGISTRY.register()
class ObjectLinkPos(ObservationTerm):
    """World position of one link of an object (e.g. the door handle), in the asset's own frame. ``(N, 3)``.

    ``relative=True``, like every other geometric read in the port: Genesis re-origins a link onto
    its inertial frame, and robosuite's link / site numbers are authored-frame numbers. On the root
    link of a free body that is the same 6.8 cm (hammer) the object-pose observables would be off by.
    """

    object_name: str = "object"
    link_name: str = ""

    def build(self) -> None:
        self._idx = resolve_link_idx(self.resolve_entity(self.object_name), [self.link_name], self.device)

    def compute(self) -> torch.Tensor:
        return self.resolve_entity(self.object_name).get_links_pos(ls_idx_local=self._idx, relative=True)[:, 0]
