"""Manipulation observation terms: end-effector pose, gripper state, object distances."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from genesis.utils.geom import R_to_rot6d, quat_to_R

from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY, ObservationTerm
from eden.managers.terms.utils import resolve_link_idx
from genesis.utils.geom import inv_transform_by_quat, transform_by_trans_quat
from eden.utils.geom import (
    axis_angle_from_quat,
    combine_frame_transforms,
    euler_xyz_from_quat,
    quat_mul,
    subtract_frame_transforms,
)
from eden.utils.string import term_subject

if TYPE_CHECKING:
    from eden.entities.base import Entity
    from genesis.engine.entities.rigid_entity.rigid_link import RigidLink


def _resolve_target(term: ObservationTerm) -> Entity | RigidLink:
    """Resolve ``term.entity_name`` (optionally its ``entity_link_name`` link).

    Called in ``build()`` to fail fast on a missing entity/link, and in ``compute()`` for the
    hot-swap-safe lazy read (see ``ManagerTermBase.resolve_entity``).
    """
    entity: Entity = term.resolve_entity(term.entity_name)
    if term.entity_link_name is not None:
        return entity.get_link(term.entity_link_name)
    return entity


@OBSERVATION_TERM_REGISTRY.register()
class EndEffectorToObjectDistance(ObservationTerm):
    """End-effector-to-object vector ``(N, 3)``.

    From the mean of the two end-effector links to the object (or its link). Expressed in the robot's base frame
    (``frame="robot"``, the default) or the world frame.
    """

    robot_name: str = "robot"
    left_ee_link_name: str = "left_gripper"
    right_ee_link_name: str = "right_gripper"
    entity_name: str = "obj"
    entity_link_name: str | None = None
    frame: Literal["world", "robot"] = "robot"

    def build(self) -> None:
        # Cache only derived idx; resolve entities lazily in compute() (see ManagerTermBase.resolve_entity).
        robot = self.resolve_entity(self.robot_name)
        self.ee_idx_local = resolve_link_idx(robot, [self.left_ee_link_name, self.right_ee_link_name], self.device)
        _resolve_target(self)  # fail fast if the object entity / link is missing

    def compute(self, *args, **kwargs):
        robot = self.resolve_entity(self.robot_name)
        ee_pos_w = robot.get_links_pos(ls_idx_local=self.ee_idx_local).mean(dim=1)
        obj_pos_w = _resolve_target(self).get_pos()
        distance_vec_w = obj_pos_w - ee_pos_w

        if self.frame == "world":
            return distance_vec_w
        elif self.frame == "robot":
            base_quat_w = robot.get_quat()
            distance_vec_b = inv_transform_by_quat(distance_vec_w, base_quat_w)
            return distance_vec_b
        else:
            raise ValueError(f"Invalid frame '{self.frame}'. Expected 'world' or 'robot'.")


@OBSERVATION_TERM_REGISTRY.register()
class ObjectToOracleDistance(ObservationTerm):
    """Object-to-oracle vector ``(N, 3)``.

    From the object (or its link) to the oracle target. The target is the first three components of command
    ``command_name``, a position in the robot's base frame. The result is expressed in the robot's base frame
    (``frame="robot"``, the default) or the world frame.
    """

    entity_name: str = "obj"
    entity_link_name: str | None = None
    command_name: str = "command"
    robot_name: str = "robot"
    frame: Literal["world", "robot"] = "robot"

    def build(self) -> None:
        # Command terms aren't swapped, so cache the ref; resolve entities lazily (see ManagerTermBase.resolve_entity).
        self._command = self._env.command_manager.get_term(self.command_name)
        _ = self.resolve_entity(self.robot_name)  # fail fast if the robot entity is missing
        _resolve_target(self)  # fail fast if the object entity / link is missing

    def compute(self, *args, **kwargs):
        robot = self.resolve_entity(self.robot_name)
        oracle_pos_b = self._command.command[..., :3]
        base_pos_w = robot.get_pos()
        base_quat_w = robot.get_quat()
        oracle_pos_w = transform_by_trans_quat(oracle_pos_b, base_pos_w, base_quat_w)

        obj_pos_w = _resolve_target(self).get_pos()
        distance_vec_w = oracle_pos_w - obj_pos_w

        if self.frame == "world":
            return distance_vec_w
        elif self.frame == "robot":
            distance_vec_b = inv_transform_by_quat(distance_vec_w, base_quat_w)
            return distance_vec_b
        else:
            raise ValueError(f"Invalid frame '{self.frame}'. Expected 'world' or 'robot'.")


# ---------------------------------------------------------------------------
# End-effector pose / gripper state
#
# The proprioception every manipulation policy server consumes is a composition of these two
# terms and ``DofsPos(dofs_name=[...])`` for the arm joints (proprio.py); declared in one
# observation group, they concatenate in declaration order.
# Indices are resolved once in ``build()`` so a wrong link or DOF name fails at env
# construction; entities are read through ``resolve_entity`` so deployment hot-swap works.
# ---------------------------------------------------------------------------

RotationFormat = Literal["quat", "euler_xyz", "axis_angle", "rot6d"]
GripperNormalize = Literal["raw", "openness", "closedness"]

_IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)


def rotation_as(quat_wxyz: torch.Tensor, rotation: RotationFormat) -> torch.Tensor:
    """Express a batch of wxyz quaternions ``(N, 4)`` in ``rotation`` format.

    ``euler_xyz`` is extrinsic roll/pitch/yaw (scipy ``"xyz"``), ``axis_angle`` a rotation
    vector, ``rot6d`` the first two rows of the rotation matrix (Zhou et al.).
    """
    if rotation == "quat":
        return quat_wxyz
    if rotation == "euler_xyz":
        roll, pitch, yaw = euler_xyz_from_quat(quat_wxyz)
        return torch.stack([roll, pitch, yaw], dim=-1)
    if rotation == "axis_angle":
        return axis_angle_from_quat(quat_wxyz)
    if rotation == "rot6d":
        return R_to_rot6d(quat_to_R(quat_wxyz))
    raise ValueError(f"Unknown rotation format {rotation!r}; expected quat, euler_xyz, axis_angle or rot6d.")


def end_effector_pose(
    entity,
    link_idx: torch.Tensor,
    *,
    frame: Literal["base", "world"] = "base",
    offset_pos: torch.Tensor | None = None,
    offset_quat: torch.Tensor | None = None,
    correction_quat: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(pos (N, 3), quat_wxyz (N, 4))`` of one link, optionally re-expressed and corrected.

    ``offset_pos`` / ``offset_quat`` ``(1, 3)`` / ``(1, 4)`` place a virtual tool frame on
    the link (RobotArena's ``ee_gripper_link`` sits 9.36 cm ahead of WidowX's
    ``gripper_link``). ``frame="base"`` expresses the result in the entity's base frame.
    ``correction_quat`` ``(1, 4)`` is a fixed rotation right-multiplied onto the
    orientation (``R_out = R @ R_corr``), as the DROID rot6d convention or a hardware
    tool-frame yaw offset require.
    """
    pos_w = entity.get_links_pos(ls_idx_local=link_idx)[:, 0, :]
    quat_w = entity.get_links_quat(ls_idx_local=link_idx)[:, 0, :]
    if offset_pos is not None or offset_quat is not None:
        pos_w, quat_w = combine_frame_transforms(pos_w, quat_w, offset_pos, offset_quat)
    if frame == "base":
        pos, quat = subtract_frame_transforms(entity.get_pos(), entity.get_quat(), pos_w, quat_w)
    elif frame == "world":
        pos, quat = pos_w, quat_w
    else:
        raise ValueError(f"Invalid frame {frame!r}. Expected 'base' or 'world'.")
    if correction_quat is not None:
        quat = quat_mul(quat, correction_quat.expand_as(quat))
    return pos, quat


def gripper_state(
    dofs_pos: torch.Tensor,
    *,
    normalize: GripperNormalize = "raw",
    open_pos: float | None = None,
    close_pos: float | None = None,
    scale: float = 1.0,
) -> torch.Tensor:
    """Map gripper DOF positions ``(N, k)`` to raw (scaled) values, openness or closedness.

    ``closedness = (pos - open_pos) / (close_pos - open_pos)`` and
    ``openness = 1 - closedness``, so a WidowX finger (open at 0.037, closed at 0) and a
    Robotiq driver (open at 0, closed at 0.8) both read 0 = open, 1 = closed for
    ``closedness`` without per-robot special cases.
    """
    if normalize == "raw":
        return dofs_pos * scale
    if open_pos is None or close_pos is None or open_pos == close_pos:
        raise ValueError(
            f"normalize={normalize!r} needs distinct open_pos and close_pos, got open_pos={open_pos}, close_pos={close_pos}."
        )
    closedness = (dofs_pos - open_pos) / (close_pos - open_pos)
    if normalize == "closedness":
        return closedness
    if normalize == "openness":
        return 1.0 - closedness
    raise ValueError(f"Unknown gripper normalize {normalize!r}; expected raw, openness or closedness.")


def _quat_tensor(value, device, *, name: str) -> torch.Tensor:
    quat = torch.as_tensor(value, dtype=torch.float32, device=device).reshape(1, 4)
    norm = quat.norm()
    if not torch.isfinite(norm) or norm < 1e-6:
        raise ValueError(f"{name} must be a non-zero wxyz quaternion, got {value}.")
    return quat / norm


@OBSERVATION_TERM_REGISTRY.register()
class EndEffectorPose(ObservationTerm):
    """End-effector pose ``[pos(3), rotation(k)]`` of a link, in the base or world frame.

    Parameters
    ----------
    entity_name: str
        Robot entity owning the link.
    ee_link_name: str
        The end-effector link.
    frame: {"base", "world"}
        Frame the pose is expressed in.
    offset_pos, offset_quat
        Virtual tool frame relative to the link (position in m, wxyz quaternion).
    frame_correction_quat
        Fixed wxyz rotation right-multiplied onto the orientation (``R @ R_corr``), e.g. the
        DROID ``rot6d`` frame correction or a hardware tool-frame yaw.
    rotation: {"quat", "euler_xyz", "axis_angle", "rot6d"}
        Output rotation format; the term is 7-, 6-, 6- or 9-D accordingly.
    """

    entity_name: str = "robot"
    ee_link_name: str = ""
    frame: Literal["base", "world"] = "base"
    offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    offset_quat: tuple[float, float, float, float] = _IDENTITY_QUAT
    frame_correction_quat: tuple[float, float, float, float] = _IDENTITY_QUAT
    rotation: RotationFormat = "quat"

    def build(self) -> None:
        if self.rotation not in ("quat", "euler_xyz", "axis_angle", "rot6d"):
            raise ValueError(
                f"{term_subject(self)}: rotation={self.rotation!r}; expected quat, euler_xyz, axis_angle or rot6d."
            )
        if self.frame not in ("base", "world"):
            raise ValueError(f"{term_subject(self)}: frame={self.frame!r}; expected 'base' or 'world'.")
        entity = self._env.entities[self.entity_name]
        self._link_idx = resolve_link_idx(entity, [self.ee_link_name], self.device)
        has_offset = tuple(self.offset_pos) != (0.0, 0.0, 0.0) or tuple(self.offset_quat) != _IDENTITY_QUAT
        self._offset_pos = (
            torch.as_tensor(self.offset_pos, dtype=torch.float32, device=self.device).reshape(1, 3)
            if has_offset
            else None
        )
        self._offset_quat = _quat_tensor(self.offset_quat, self.device, name="offset_quat") if has_offset else None
        self._correction_quat = (
            _quat_tensor(self.frame_correction_quat, self.device, name="frame_correction_quat")
            if tuple(self.frame_correction_quat) != _IDENTITY_QUAT
            else None
        )

    def compute(self, *args, **kwargs) -> torch.Tensor:
        entity = self.resolve_entity(self.entity_name)
        pos, quat = end_effector_pose(
            entity,
            self._link_idx,
            frame=self.frame,
            offset_pos=self._offset_pos,
            offset_quat=self._offset_quat,
            correction_quat=self._correction_quat,
        )
        return torch.cat([pos, rotation_as(quat, self.rotation)], dim=-1)


@OBSERVATION_TERM_REGISTRY.register()
class GripperState(ObservationTerm):
    """Gripper DOF positions as raw values, openness or closedness.

    Parameters
    ----------
    entity_name: str
        Entity owning the gripper DOFs (the arm, or a separately mounted gripper).
    dofs_name: list[str]
        Gripper DOFs to report (one driver, or both fingers for a raw two-finger state).
    normalize: {"raw", "openness", "closedness"}
        ``raw`` reports ``pos * scale``; the others map ``[open_pos, close_pos]`` to
        ``[0, 1]`` (see :func:`gripper_state`).
    open_pos, close_pos: float | None
        DOF positions of the fully open / fully closed gripper (required unless ``raw``).
    scale: float
        Multiplier for ``raw`` output (e.g. to match a checkpoint's normaliser units).
    """

    entity_name: str = "robot"
    dofs_name: list[str] = []
    normalize: GripperNormalize = "raw"
    open_pos: float | None = None
    close_pos: float | None = None
    scale: float = 1.0

    def build(self) -> None:
        if not self.dofs_name:
            raise ValueError(f"{term_subject(self)} needs at least one gripper DOF name.")
        if self.normalize not in ("raw", "openness", "closedness"):
            raise ValueError(
                f"{term_subject(self)}: normalize={self.normalize!r}; expected raw, openness or closedness."
            )
        if self.normalize != "raw" and (
            self.open_pos is None or self.close_pos is None or self.open_pos == self.close_pos
        ):
            raise ValueError(
                f"{term_subject(self)}: normalize={self.normalize!r} needs distinct open_pos/close_pos "
                f"(got {self.open_pos}, {self.close_pos})."
            )
        entity = self._env.entities[self.entity_name]
        _, idx = entity.find_named_dofs_idx_local(list(self.dofs_name), preserve_order=True)
        self._dofs_idx = torch.as_tensor(idx, dtype=torch.long, device=self.device)

    def compute(self, *args, **kwargs) -> torch.Tensor:
        pos = self.resolve_entity(self.entity_name).get_dofs_pos(dofs_idx_local=self._dofs_idx)
        return gripper_state(
            pos, normalize=self.normalize, open_pos=self.open_pos, close_pos=self.close_pos, scale=self.scale
        )
