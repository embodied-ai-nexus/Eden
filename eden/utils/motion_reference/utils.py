"""Motion-reference helpers: interpolation and robot-vs-reference differences."""

from __future__ import annotations
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import torch
from genesis.utils.geom import slerp

from genesis.utils.geom import inv_quat, inv_transform_by_quat, transform_by_quat

from eden.utils.geom import (
    quat_box_minus,
    quat_error_magnitude,
    quat_mul,
    subtract_frame_transforms,
    transform_points,
)
from eden.utils.motion_reference.lookup import assert_motion_field_present

if TYPE_CHECKING:
    from eden.envs.base import MimicRLEnvBase
    from eden.entities.base import Entity
    from eden.managers.motion_manager import MotionManager
    from eden.utils.motion_reference.motion_data import MotionReferenceData


def _get_ref_data(
    env: MimicRLEnvBase,
    entity_name: str,
) -> tuple[Entity, MotionManager, MotionReferenceData]:
    """Extract the entity, motion manager, and reference data for the given entity."""
    entity: Entity = env.entities[entity_name]
    motion_manager: MotionManager = env.motion_manager
    ref_data: MotionReferenceData = motion_manager.data[entity_name]
    return entity, motion_manager, ref_data


def get_base_position_distance(
    env: MimicRLEnvBase,
    entity_name: str = "robot",
    frame_idx: int = 0,
    masking: bool = True,
    squared: bool = False,
    return_diff: bool = False,
) -> torch.Tensor:
    """
    Compute the distance between the base position and the reference position.

    Parameters
    ----------
    env : MimicRLEnvBase
        The environment.
    entity_name : str
        The name of the robot entity.
    frame_idx : int
        Which reference frame to compare against. Defaults to 0.
    masking : bool, optional
        Whether to use the reference data's position mask in computing distance. Defaults to True.
    squared : bool, optional
        Whether to return the squared distance. Defaults to False.
    return_diff : bool, optional
        If True, return the difference (3d) rather than the distance (scalar). Defaults to False.

    Returns
    -------
    torch.Tensor
        The difference or distance. (num_envs, 3) if return_diff, (num_envs,) otherwise.
    """
    entity, _, ref_data = _get_ref_data(env, entity_name)

    base_pos = entity.get_pos()
    ref_pos = ref_data.base_pos_w[:, frame_idx]

    diff = base_pos - ref_pos  # (num_envs, 3)
    if masking:
        diff[:, :2] *= ref_data.base_pos_plane_mask[:, frame_idx].unsqueeze(1)
        diff[:, 2] *= ref_data.base_pos_height_mask[:, frame_idx]
    if return_diff:
        return diff

    if squared:
        return torch.sum(torch.square(diff), dim=-1)
    return torch.norm(diff, dim=-1)


def get_base_rotation_distance(
    env: MimicRLEnvBase,
    entity_name: str = "robot",
    frame_idx: int = 0,
    difference_mode: Literal["box_minus", "axis_angle"] = "axis_angle",
    masking: bool = True,
) -> torch.Tensor:
    """
    Compute the distance between the base rotation and the reference rotation.

    Parameters
    ----------
    env : MimicRLEnvBase
        The environment.
    entity_name : str
        The name of the robot entity.
    frame_idx : int
        Which reference frame to compare against. Defaults to 0.
    difference_mode : Literal["box_minus", "axis_angle"], optional
        The mode to compute the difference. Defaults to "axis_angle".
    masking : bool, optional
        Whether to use the reference data's validity mask. Defaults to True.

    Returns
    -------
    torch.Tensor
        The rotation difference. (num_envs,)
    """
    entity, _, ref_data = _get_ref_data(env, entity_name)

    base_rot = entity.get_quat()
    ref_rot = ref_data.base_quat_w[:, frame_idx]

    if difference_mode == "box_minus":
        diff = torch.norm(
            quat_box_minus(base_rot.reshape(-1, 4), ref_rot.reshape(-1, 4)).reshape(-1, 3),
            dim=-1,
        )
    elif difference_mode == "axis_angle":
        diff = quat_error_magnitude(base_rot.reshape(-1, 4), ref_rot.reshape(-1, 4))
    else:
        raise ValueError(f"Unknown difference_mode: {difference_mode}. Must be 'box_minus' or 'axis_angle'.")

    if masking:
        diff *= ref_data.validity[:, frame_idx]

    return diff


def get_base_velocity_difference(
    env: MimicRLEnvBase,
    entity_name: str = "robot",
    frame_idx: int = 0,
    masking: bool = True,
    anchor_frame: Literal["world", "robot", "reference"] = "world",
    return_diff: bool = False,
) -> torch.Tensor:
    """
    Compute the difference between the base velocity and the reference velocity.

    Parameters
    ----------
    env : MimicRLEnvBase
        The environment.
    entity_name : str
        The name of the robot entity.
    frame_idx : int
        Which reference frame to compare against. Defaults to 0.
    masking : bool, optional
        Whether to use the reference data's validity mask. Defaults to True.
    anchor_frame : Literal["world", "robot", "reference"], optional
        The frame to calculate the velocity difference. Defaults to "world".
    return_diff : bool, optional
        If True, return the difference (3d) rather than the distance (scalar). Defaults to False.

    Returns
    -------
    torch.Tensor
        The velocity difference. (num_envs, 3) if return_diff, (num_envs,) otherwise.
    """
    entity, _, ref_data = _get_ref_data(env, entity_name)

    if anchor_frame == "world":
        base_vel = entity.get_vel(frame="world")
        ref_base_vel = ref_data.base_lin_vel_w[:, frame_idx]
    elif anchor_frame == "robot":
        base_vel = entity.get_vel(frame="body")
        anchor_quat = entity.get_quat()
        ref_base_vel = inv_transform_by_quat(ref_data.base_lin_vel_w[:, frame_idx], anchor_quat)
    elif anchor_frame == "reference":
        anchor_quat = ref_data.base_quat_w[:, frame_idx]
        base_vel = inv_transform_by_quat(entity.get_vel(frame="world"), anchor_quat)
        ref_base_vel = inv_transform_by_quat(ref_data.base_lin_vel_w[:, frame_idx], anchor_quat)
    else:
        raise ValueError(f"Unknown anchor frame: {anchor_frame}. Must be 'world', 'robot', or 'reference'.")

    vel_diff = base_vel - ref_base_vel  # (num_envs, 3)
    if masking:
        vel_diff *= ref_data.validity[:, frame_idx].unsqueeze(1)
    if return_diff:
        return vel_diff

    return torch.norm(vel_diff, dim=-1)


def get_dofs_position_difference(
    env: MimicRLEnvBase,
    entity_name: str = "robot",
    frame_idx: int = 0,
    masking: bool = True,
) -> torch.Tensor:
    """
    Compute the difference between the joint positions and the reference joint positions.

    Parameters
    ----------
    env : MimicRLEnvBase
        The environment.
    entity_name : str
        The name of the robot entity.
    frame_idx : int
        Which reference frame to compare against. Defaults to 0.
    masking : bool, optional
        Whether to use the reference data's joint position mask. Defaults to True.

    Returns
    -------
    torch.Tensor
        The joint position difference. (num_envs, num_dofs)
    """
    entity, _, ref_data = _get_ref_data(env, entity_name)

    dofs_pos = entity.get_dofs_pos()
    ref_dofs_pos = ref_data.dofs_pos[:, frame_idx]

    joint_diff = dofs_pos - ref_dofs_pos  # (num_envs, num_dofs)
    if masking:
        joint_diff *= ref_data.dofs_pos_mask[:, frame_idx]
    return joint_diff


def get_dofs_velocity_difference(
    env: MimicRLEnvBase,
    entity_name: str = "robot",
    frame_idx: int = 0,
    masking: bool = True,
) -> torch.Tensor:
    """
    Compute the difference between the joint velocities and the reference joint velocities.

    Parameters
    ----------
    env : MimicRLEnvBase
        The environment.
    entity_name : str
        The name of the robot entity.
    frame_idx : int
        Which reference frame to compare against. Defaults to 0.
    masking : bool, optional
        Whether to use the reference data's joint velocity mask. Defaults to True.

    Returns
    -------
    torch.Tensor
        The joint velocity difference. (num_envs, num_dofs)
    """
    entity, _, ref_data = _get_ref_data(env, entity_name)

    dofs_vel = entity.get_dofs_vel()
    ref_dofs_vel = ref_data.dofs_vel[:, frame_idx]

    joint_diff = dofs_vel - ref_dofs_vel  # (num_envs, num_dofs)
    if masking:
        joint_diff *= ref_data.dofs_vel_mask[:, frame_idx]
        joint_diff *= ref_data.validity[:, frame_idx].unsqueeze(1)
    return joint_diff


def get_link_position_distance(
    env: MimicRLEnvBase,
    entity_name: str = "robot",
    links_name: list[str] | None = None,
    frame_idx: int = 0,
    in_base_frame: bool = False,
    masking: bool = True,
    squared: bool = False,
    return_diff: bool = False,
) -> torch.Tensor:
    """
    Compute the distance between the link positions and the reference link positions.

    Parameters
    ----------
    env : MimicRLEnvBase
        The environment.
    entity_name : str
        The name of the robot entity.
    links_name : list[str] | None
        The entity links to compare: the reference's links, in its link order (the reference side is not
        re-indexed). If None, every link of the entity, which must then match the reference's link axis.
    frame_idx : int
        Which reference frame to compare against. Defaults to 0.
    in_base_frame : bool, optional
        Whether to compute the distance in the base frame. Defaults to False.
    masking : bool, optional
        Whether to use the reference data's link position mask. Defaults to True.
    squared : bool, optional
        Whether to return the squared distance. Defaults to False.
    return_diff : bool, optional
        If True, return the difference (3d) rather than the distance (scalar). Defaults to False.

    Returns
    -------
    torch.Tensor
        The link position distance. (num_envs, num_links) or (num_envs, num_links, 3) if return_diff.
    """
    entity, _, ref_data = _get_ref_data(env, entity_name)

    if links_name is not None:
        _, link_indices = entity.find_named_links_idx_local(links_name, preserve_order=True)
    else:
        link_indices = None

    if in_base_frame:
        # A dataset may leave the body-frame fields out (``derive_body_frame=False``): refuse, don't read the
        # buffer default.
        assert_motion_field_present(env, entity_name, "links_pos_b")
        links_pos = entity.get_links_pos(ls_idx_local=link_indices)
        base_pos = entity.get_pos()
        base_quat = entity.get_quat()
        base_pos_inv, base_quat_inv = subtract_frame_transforms(base_pos, base_quat)
        links_pos = transform_points(links_pos, base_pos_inv, base_quat_inv)
        ref_links_pos = ref_data.links_pos_b[:, frame_idx]
    else:
        links_pos = entity.get_links_pos(ls_idx_local=link_indices)
        ref_links_pos = ref_data.links_pos_w[:, frame_idx]

    link_pos_diff = links_pos - ref_links_pos  # (num_envs, num_links, 3)
    if masking:
        link_pos_diff *= ref_data.links_pos_mask[:, frame_idx].unsqueeze(-1)
    if return_diff:
        return link_pos_diff
    if squared:
        return torch.sum(torch.square(link_pos_diff), dim=-1)
    return torch.norm(link_pos_diff, dim=-1)


def get_link_rotation_distance(
    env: MimicRLEnvBase,
    entity_name: str = "robot",
    links_name: list[str] | None = None,
    frame_idx: int = 0,
    in_base_frame: bool = False,
    difference_mode: Literal["box_minus", "axis_angle"] = "axis_angle",
    masking: bool = True,
    squared: bool = False,
) -> torch.Tensor:
    """
    Compute the distance between the link rotations and the reference link rotations.

    Parameters
    ----------
    env : MimicRLEnvBase
        The environment.
    entity_name : str
        The name of the robot entity.
    links_name : list[str] | None
        The entity links to compare: the reference's links, in its link order (the reference side is not
        re-indexed). If None, every link of the entity, which must then match the reference's link axis.
    frame_idx : int
        Which reference frame to compare against. Defaults to 0.
    in_base_frame : bool, optional
        Whether to compute the distance in the base frame. Defaults to False.
    difference_mode : Literal["box_minus", "axis_angle"], optional
        The mode to compute the difference. Defaults to "axis_angle".
    masking : bool, optional
        Whether to use the reference data's link rotation mask. Defaults to True.
    squared : bool, optional
        Whether to return the squared distance. Defaults to False.

    Returns
    -------
    torch.Tensor
        The link rotation distance. (num_envs, num_links)
    """
    entity, _, ref_data = _get_ref_data(env, entity_name)

    if links_name is not None:
        _, link_indices = entity.find_named_links_idx_local(links_name, preserve_order=True)
    else:
        link_indices = None

    if in_base_frame:
        assert_motion_field_present(env, entity_name, "links_quat_b")
        links_quat = entity.get_links_quat(ls_idx_local=link_indices)
        base_quat_inv = inv_quat(entity.get_quat())
        links_quat = quat_mul(base_quat_inv.unsqueeze(1).expand_as(links_quat), links_quat)
        ref_links_quat = ref_data.links_quat_b[:, frame_idx]
    else:
        links_quat = entity.get_links_quat(ls_idx_local=link_indices)
        ref_links_quat = ref_data.links_quat_w[:, frame_idx]

    num_links = links_quat.shape[1]
    if difference_mode == "box_minus":
        link_rot_diff = torch.norm(
            quat_box_minus(links_quat.reshape(-1, 4), ref_links_quat.reshape(-1, 4)).reshape(-1, num_links, 3),
            dim=-1,
        )
    elif difference_mode == "axis_angle":
        link_rot_diff = quat_error_magnitude(links_quat.reshape(-1, 4), ref_links_quat.reshape(-1, 4)).reshape(
            -1, num_links
        )
    else:
        raise ValueError(f"Unknown difference_mode: {difference_mode}. Must be 'box_minus' or 'axis_angle'.")

    if masking:
        link_rot_diff *= ref_data.links_rot_mask[:, frame_idx]
    if squared:
        return torch.square(link_rot_diff)
    return link_rot_diff


# ---------------------------------------------------------------------------
# Motion interpolation and velocity estimation utilities
# ---------------------------------------------------------------------------


def lerp_rows(seq: torch.Tensor, index_0: torch.Tensor, index_1: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    """Rows ``index_0`` and ``index_1`` of ``seq`` blended linearly, ``(1 - blend) * a + blend * b``, row by row.

    ``seq`` is ``(T, *data_shape)``; ``index_0`` / ``index_1`` / ``blend`` are ``(N,)`` and the result is
    ``(N, *data_shape)``. ``blend`` is used in the dtype it comes in — a caller wanting it in ``seq``'s casts it.
    """
    blend = blend.view((-1,) + (1,) * (seq.ndim - 1))
    return seq[index_0] * (1 - blend) + seq[index_1] * blend


def slerp_rows(seq: torch.Tensor, index_0: torch.Tensor, index_1: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    """:func:`lerp_rows` for a ``(T, *, 4)`` unit-quaternion ``seq``: every quaternion of a row slerped by its blend."""
    q0 = seq[index_0]
    q1 = seq[index_1]
    if q0.numel() == 0:
        # Zero-link case: slerp() on an empty pair is undefined; short-circuit.
        return q0
    if q0.ndim > 2:
        # (N, L_1, ..., L_k, 4): each row's blend repeats once per element of its quaternion grid.
        blend = blend.repeat_interleave(math.prod(q0.shape[1:-1]))
    return slerp(q0.reshape(-1, 4), q1.reshape(-1, 4), blend).reshape(q0.shape)


def nearest_rows(seq: torch.Tensor, index_0: torch.Tensor, index_1: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    """:func:`lerp_rows` for discrete data (contact flags, labels, points): the nearer row, never a blend of two."""
    return seq[torch.where(blend < 0.5, index_0, index_1)]


# How each per-frame motion field resamples between two frames, by ``BatchMotionSequence`` field name: unit
# quaternions slerp, the discrete contact annotations take the nearer frame (a blended flag is not a flag, and a
# blended contact point is one no demo touched), every other field lerps. Shared by ``interpolate_motion`` and
# ``interpolate_body_model`` (a reader's retiming) and ``SpeedScale`` (a dataset modifier), each on its own time grid.
MOTION_FIELD_ROWS: dict[str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "base_pos_w": lerp_rows,
    "base_quat_w": slerp_rows,
    "dofs_pos": lerp_rows,
    "links_pos_w": lerp_rows,
    "links_quat_w": slerp_rows,
    "base_lin_vel_w": lerp_rows,
    "base_ang_vel_w": lerp_rows,
    "dofs_vel": lerp_rows,
    "links_lin_vel_w": lerp_rows,
    "links_ang_vel_w": lerp_rows,
    "links_pos_b": lerp_rows,
    "links_quat_b": slerp_rows,
    "links_lin_vel_b": lerp_rows,
    "links_ang_vel_b": lerp_rows,
    "contact_points_w": nearest_rows,
    "contact_points_valid": nearest_rows,
    "contact_label": nearest_rows,
    "body_pose_quat": slerp_rows,
    "body_shape": lerp_rows,
}


def _retime_grid(
    input_frames: int, source_framerate: float | torch.Tensor, target_framerate: float | torch.Tensor, device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(index_0, index_1, blend)`` of each output frame when ``input_frames`` at ``source_framerate`` are retimed
    to ``target_framerate``: the output frame blends source frames ``index_0`` and ``index_1`` by ``blend``.
    """
    if input_frames < 2:
        raise ValueError(f"interpolate_motion requires at least 2 frames, got {input_frames}")
    if isinstance(source_framerate, torch.Tensor):
        source_framerate = source_framerate.item()
    if isinstance(target_framerate, torch.Tensor):
        target_framerate = target_framerate.item()
    input_dt = 1.0 / source_framerate
    output_dt = 1.0 / target_framerate
    duration = (input_frames - 1) * input_dt

    times = torch.arange(0, duration, output_dt, device=device, dtype=torch.float32)
    phase = times / duration
    index_0 = (phase * (input_frames - 1)).floor().long()
    index_1 = torch.minimum(
        index_0 + 1,
        torch.tensor(input_frames - 1, device=index_0.device),
    )
    blend = phase * (input_frames - 1) - index_0.float()
    return index_0, index_1, blend


def interpolate_motion(
    root_trans: torch.Tensor,
    root_quat: torch.Tensor,
    dofs_pos: torch.Tensor | None,
    links_pos_w: torch.Tensor | None,
    links_quat_w: torch.Tensor | None,
    source_framerate: float,
    target_framerate: float,
    base_lin_vel_w: torch.Tensor | None = None,
    base_ang_vel_w: torch.Tensor | None = None,
    dofs_vel: torch.Tensor | None = None,
    links_lin_vel_w: torch.Tensor | None = None,
    links_ang_vel_w: torch.Tensor | None = None,
    contact_points_w: torch.Tensor | None = None,
    contact_points_valid: torch.Tensor | None = None,
    contact_label: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Resample motion data from ``source_framerate`` to ``target_framerate`` over the clip's duration.

    Each field resamples by its :data:`MOTION_FIELD_ROWS` rule (slerp for quaternions, nearest frame for the contact
    fields, lerp otherwise).

    Parameters
    ----------
    root_trans : torch.Tensor
        The root translation data. (num_frames, 3)
    root_quat : torch.Tensor
        The root quaternion data. (num_frames, 4) in (w, x, y, z)
    dofs_pos : torch.Tensor | None
        The joint position data. (num_frames, num_dofs)
    links_pos_w : torch.Tensor | None
        The link position data. (num_frames, num_links, 3)
    links_quat_w : torch.Tensor | None
        The link quaternion data. (num_frames, num_links, 4) in (w, x, y, z)
    source_framerate : float
        The source framerate.
    target_framerate : float
        The target framerate.
    base_lin_vel_w : torch.Tensor | None
        The base linear velocity. (num_frames, 3)
    base_ang_vel_w : torch.Tensor | None
        The base angular velocity. (num_frames, 3)
    dofs_vel : torch.Tensor | None
        The joint velocity data. (num_frames, num_dofs)
    links_lin_vel_w : torch.Tensor | None
        The link linear velocity data. (num_frames, num_links, 3)
    links_ang_vel_w : torch.Tensor | None
        The link angular velocity data. (num_frames, num_links, 3)
    contact_points_w : torch.Tensor | None
        Demo contact points. (num_frames, num_links, num_partners, 3). Resampled
        with nearest-neighbor gather (contact is a discrete event; blending two
        frames' points across a make/break transition fabricates positions).
    contact_points_valid : torch.Tensor | None
        Contact validity flags, bool. (num_frames, num_links, num_partners).
        Nearest-neighbor gather — linear blending would corrupt booleans.
    contact_label : torch.Tensor | None
        Tri-state per-link contact labels, int8 in {-1, 0, +1}. (num_frames, num_links). Nearest-neighbor
        gather, like the other contact fields.

    Returns
    -------
    tuple
        The interpolated (root_trans, root_quat, dofs_pos, links_pos_w, links_quat_w,
        base_lin_vel_w, base_ang_vel_w, dofs_vel, links_lin_vel_w, links_ang_vel_w,
        contact_points_w, contact_points_valid, contact_label).
    """
    assert root_trans.ndim == 2, "The root_trans must be in shape (num_frames, 3)."
    assert root_quat.ndim == 2, "The root_quat must be in shape (num_frames, 4)."
    if dofs_pos is not None:
        assert dofs_pos.ndim == 2, "The dofs_pos must be in shape (num_frames, num_dofs)."
    if links_pos_w is not None:
        assert links_pos_w.ndim == 3, "The links_pos_w must be in shape (num_frames, num_links, 3)."
    if links_quat_w is not None:
        assert links_quat_w.ndim == 3, "The links_quat_w must be in shape (num_frames, num_links, 4)."

    index_0, index_1, blend = _retime_grid(root_trans.shape[0], source_framerate, target_framerate, root_trans.device)

    fields = {
        "base_pos_w": root_trans,
        "base_quat_w": root_quat,
        "dofs_pos": dofs_pos,
        "links_pos_w": links_pos_w,
        "links_quat_w": links_quat_w,
        "base_lin_vel_w": base_lin_vel_w,
        "base_ang_vel_w": base_ang_vel_w,
        "dofs_vel": dofs_vel,
        "links_lin_vel_w": links_lin_vel_w,
        "links_ang_vel_w": links_ang_vel_w,
        "contact_points_w": contact_points_w,
        "contact_points_valid": contact_points_valid,
        "contact_label": contact_label,
    }
    return tuple(
        None if field is None else MOTION_FIELD_ROWS[name](field, index_0, index_1, blend)
        for name, field in fields.items()
    )


def interpolate_body_model(
    body_pose_quat: torch.Tensor,
    body_shape: torch.Tensor,
    source_framerate: float | torch.Tensor,
    target_framerate: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Retime a sequence's body-model fields on :func:`interpolate_motion`'s time grid.

    Parameters
    ----------
    body_pose_quat : torch.Tensor
        Body-model joint rotations, wxyz. (num_frames, num_body_joints, 4). Slerped per joint.
    body_shape : torch.Tensor
        Body-model shape coefficients. (num_frames, num_betas). Linear.
    source_framerate, target_framerate : float | torch.Tensor
        As for :func:`interpolate_motion`.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        The retimed ``(body_pose_quat, body_shape)``, frame for frame with :func:`interpolate_motion`'s output.
    """
    index_0, index_1, blend = _retime_grid(
        body_pose_quat.shape[0], source_framerate, target_framerate, body_pose_quat.device
    )
    return (
        MOTION_FIELD_ROWS["body_pose_quat"](body_pose_quat, index_0, index_1, blend),
        MOTION_FIELD_ROWS["body_shape"](body_shape, index_0, index_1, blend),
    )


def estimate_motion_velocities(
    root_trans: torch.Tensor,
    root_quat: torch.Tensor,
    dofs_pos: torch.Tensor | None,
    links_pos_w: torch.Tensor | None,
    links_quat_w: torch.Tensor | None,
    dt: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """
    Compute linear, joint, and angular velocities from motion sequences.

    Parameters
    ----------
    root_trans : torch.Tensor
        Root translation data with shape (T, 3).
    root_quat : torch.Tensor
        Root rotation quaternions with shape (T, 4), in (w, x, y, z).
    dofs_pos : torch.Tensor | None
        Joint position data with shape (T, D).
    links_pos_w : torch.Tensor | None
        The link position data with shape (T, L, 3).
    links_quat_w : torch.Tensor | None
        The link quaternion data with shape (T, L, 4).
    dt : float
        The time interval between two frames.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]
        The base linear velocities, base angular velocities, joint velocities, link linear velocities, and link
        angular velocities; each optional one is None when its input is.

    Raises
    ------
    ValueError
        If the sequence has fewer than 2 frames: a single frame spans no time, so it has no velocity.
    """
    if root_trans.shape[0] < 2:
        raise ValueError(
            f"estimate_motion_velocities needs at least 2 frames to finite-difference, got a sequence of "
            f"{root_trans.shape[0]} frame(s) (root_trans shape {tuple(root_trans.shape)}): a single frame spans no "
            f"time, so it has no velocity. Reject the clip before this call rather than padding it with zeros."
        )
    base_lin_vel = torch.gradient(root_trans, spacing=dt, dim=0)[0]
    base_ang_vel = _so3_central_difference(root_quat, dt)
    dofs_vel = None
    if dofs_pos is not None:
        dofs_vel = torch.gradient(dofs_pos, spacing=dt, dim=0)[0]
    links_lin_vel = None
    if links_pos_w is not None:
        links_lin_vel = torch.gradient(links_pos_w, spacing=dt, dim=0)[0]
    links_ang_vel = None
    if links_quat_w is not None:
        links_ang_vel = _so3_central_difference(links_quat_w, dt)
    return base_lin_vel, base_ang_vel, dofs_vel, links_lin_vel, links_ang_vel


def _so3_central_difference(rotations: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Central-difference SO(3) derivative.

    The two end frames copy their neighbour's central difference. A 2-frame sequence has a single interval, so both
    frames get its one-sided difference, which is what ``torch.gradient`` returns for the linear channels there.

    Parameters
    ----------
    rotations : torch.Tensor
        Quaternion rotations (w, x, y, z) with shape (T, 4) or (T, N, 4).
    dt : float
        The time interval between two frames.

    Returns
    -------
    torch.Tensor
        Angular velocity with shape (T, 3) or (T, N, 3).
    """
    if rotations.shape[0] == 2:
        omega = quat_box_minus(rotations[1:], rotations[:1]) / dt
        return torch.cat([omega, omega], dim=0)
    q_prev = rotations[:-2]
    q_next = rotations[2:]
    omega = quat_box_minus(q_next, q_prev) / (2.0 * dt)
    return torch.cat([omega[:1], omega, omega[-1:]], dim=0)


def derive_body_frame_link_fields(
    base_pos_w: torch.Tensor,
    base_quat_w: torch.Tensor,
    base_lin_vel_w: torch.Tensor | None,
    base_ang_vel_w: torch.Tensor | None,
    links_pos_w: torch.Tensor | None,
    links_quat_w: torch.Tensor | None,
    links_lin_vel_w: torch.Tensor | None,
    links_ang_vel_w: torch.Tensor | None,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Derive body-frame ``links_*_b`` fields from world-frame fields.

    Returns ``(links_pos_b, links_quat_b, links_lin_vel_b, links_ang_vel_b)``,
    each ``None`` when the corresponding world-frame input is ``None`` (the
    velocities need every position, quaternion and velocity input). Shared by
    :meth:`DatasetBase._pack_motion_sequence` and the modifiers that rewrite
    world-frame fields (e.g. ``SmoothStartTransition``), so both produce
    bit-identical body-frame fields from the same world-frame inputs.

    All tensors are torch with shape ``(T, …)`` (no batch dim); quaternions are
    wxyz. The velocities are relative to the base, rotated into the base frame:
    the linear one subtracts the base's rigid-body velocity at the link,
    ``v_base + ω_base × r`` with ``r`` the link position relative to the base
    in the world frame; the angular one subtracts ``ω_base``.
    """
    links_pos_b: torch.Tensor | None = None
    links_quat_b: torch.Tensor | None = None
    links_lin_vel_b: torch.Tensor | None = None
    links_ang_vel_b: torch.Tensor | None = None

    if links_pos_w is None and links_quat_w is None:
        return links_pos_b, links_quat_b, links_lin_vel_b, links_ang_vel_b

    base_pos_w_inv, base_quat_w_inv = subtract_frame_transforms(base_pos_w, base_quat_w)
    if links_pos_w is not None:
        links_pos_b = transform_points(links_pos_w, base_pos_w_inv, base_quat_w_inv)
    if links_quat_w is not None:
        links_quat_b = quat_mul(
            base_quat_w_inv.unsqueeze(1).expand_as(links_quat_w),
            links_quat_w,
        )

    if (
        links_pos_w is not None
        and links_quat_w is not None
        and links_lin_vel_w is not None
        and links_ang_vel_w is not None
        and base_lin_vel_w is not None
        and base_ang_vel_w is not None
    ):
        rel_pos_w = links_pos_w - base_pos_w.unsqueeze(1)
        rel_lin_vel_w = (
            links_lin_vel_w
            - base_lin_vel_w.unsqueeze(1)
            - torch.cross(base_ang_vel_w.unsqueeze(1).expand_as(rel_pos_w), rel_pos_w, dim=-1)
        )
        rel_ang_vel_w = links_ang_vel_w - base_ang_vel_w.unsqueeze(1)
        base_quat_w_inv_batched = base_quat_w_inv.unsqueeze(1).expand(-1, links_lin_vel_w.shape[1], -1)
        links_lin_vel_b = transform_by_quat(rel_lin_vel_w, base_quat_w_inv_batched)
        links_ang_vel_b = transform_by_quat(rel_ang_vel_w, base_quat_w_inv_batched)

    return links_pos_b, links_quat_b, links_lin_vel_b, links_ang_vel_b
