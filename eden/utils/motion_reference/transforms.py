"""Pure-tensor motion-reference transforms shared across the motion systems.

Holds the anchor-relative ("yaw-warp") body transform used by both the
``MotionCommand`` command term (``eden.managers.terms.commands.mimic``) and the
motion-manager tracking helpers (``eden.utils.motion_reference.lookup``), so the
two cannot drift apart, plus the full anchor-frame transforms the tracking
observations use.

Functions here take and return plain tensors only — no env / manager coupling —
so they compose with either motion backend.
"""

from __future__ import annotations

import torch

from genesis.utils.geom import inv_quat, inv_transform_by_quat, transform_by_quat

from eden.utils.geom import quat_mul, yaw_quat


def anchor_relative_links(
    robot_anchor_pos_w: torch.Tensor,
    robot_anchor_quat_w: torch.Tensor,
    motion_anchor_pos_w: torch.Tensor,
    motion_anchor_quat_w: torch.Tensor,
    motion_links_pos_w: torch.Tensor,
    motion_links_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Warp motion-reference body poses into the robot's anchor frame.

    The motion bodies are rigidly re-anchored so the motion's anchor link sits at
    the robot's anchor link: a yaw-only delta rotation (heading) plus a planar
    delta position (the robot's xy, the motion's z) is applied to every tracked
    body. Yaw-only keeps roll/pitch from the reference; the z-from-motion keeps
    height tracking referenced to the clip. A NaN robot-anchor quaternion falls
    back to the motion reference (identity warp) so NaN never propagates downstream.

    Shapes are rank-agnostic in the leading dims: anchor tensors are
    ``(*B, 3)`` / ``(*B, 4)`` and body tensors are ``(*B, L, 3)`` / ``(*B, L, 4)``
    for ``L`` tracked links, where ``*B`` is any batch shape — ``(N,)`` for the
    single-frame command path, ``(N, K)`` for a K-frame future window. Quaternions
    are wxyz.

    Returns ``(links_pos_relative_w, links_quat_relative_w)``, each ``(*B, L, 3|4)``.
    """
    # NaN guard: the physics engine may sporadically return NaN quaternions for
    # the robot anchor. Fall back to the motion reference so the relative
    # transform is identity (perfect tracking) rather than cascading NaN into all
    # downstream rewards / observations / metrics.
    nan_mask = torch.isnan(robot_anchor_quat_w).any(dim=-1, keepdim=True)
    robot_anchor_quat_w = torch.where(nan_mask, motion_anchor_quat_w, robot_anchor_quat_w)
    robot_anchor_pos_w = torch.where(nan_mask, motion_anchor_pos_w, robot_anchor_pos_w)

    # Yaw-only heading delta that maps the motion anchor onto the robot anchor.
    anchor_delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w, inv_quat(motion_anchor_quat_w)))
    # Planar delta position: robot's xy, motion's z (height stays clip-referenced).
    anchor_delta_pos_w = robot_anchor_pos_w.clone()
    anchor_delta_pos_w[..., 2] = motion_anchor_pos_w[..., 2]

    # Broadcast the per-anchor delta over the tracked-link axis (inserted at -2).
    delta_ori_exp = anchor_delta_ori_w.unsqueeze(-2).expand(motion_links_quat_w.shape).contiguous()
    delta_pos_exp = anchor_delta_pos_w.unsqueeze(-2).expand(motion_links_pos_w.shape).contiguous()
    anchor_pos_exp = motion_anchor_pos_w.unsqueeze(-2).expand(motion_links_pos_w.shape).contiguous()

    links_quat_relative_w = quat_mul(delta_ori_exp, motion_links_quat_w)
    links_pos_relative_w = delta_pos_exp + transform_by_quat(motion_links_pos_w - anchor_pos_exp, delta_ori_exp)
    return links_pos_relative_w, links_quat_relative_w


def links_pos_in_anchor_frame(
    anchor_pos_w: torch.Tensor,
    anchor_quat_w: torch.Tensor,
    links_pos_w: torch.Tensor,
) -> torch.Tensor:
    """Express world-frame link positions in an anchor link's frame.

    The **full** inverse-anchor transform ``R_anchor⁻¹ · (p − p_anchor)`` — uses
    the anchor's complete orientation (roll, pitch, AND yaw), unlike the
    yaw-only :func:`anchor_relative_links` warp. This is the obs-side
    "body-relative" convention the policy reads (ProtoMotions reduced coords,
    ``eden.managers.terms.observations.tracking``); the yaw-warp is the
    reward/metric convention. The two are intentionally different.

    Shapes are rank-agnostic in the leading dims: ``anchor_pos_w`` is
    ``(*B, 3)`` / ``anchor_quat_w`` is ``(*B, 4)`` and ``links_pos_w`` is
    ``(*B, L, 3)`` for ``L`` links. ``*B`` is any batch shape — ``(N,)`` for a
    single frame, ``(N, K)`` for a K-frame future window (expand the per-env
    anchor over the K axis before calling, since the robot anchor has no future
    axis). Quaternions are wxyz. Returns ``(*B, L, 3)``.
    """
    delta = links_pos_w - anchor_pos_w.unsqueeze(-2)
    anchor_quat_exp = anchor_quat_w.unsqueeze(-2).expand(links_pos_w.shape[:-1] + (4,))
    return inv_transform_by_quat(delta, anchor_quat_exp)


def links_quat_in_anchor_frame(
    anchor_quat_w: torch.Tensor,
    links_quat_w: torch.Tensor,
) -> torch.Tensor:
    """Express world-frame link orientations in an anchor link's frame.

    The **full** inverse-anchor rotation ``R_anchor⁻¹ · R_link`` (wxyz), the
    orientation counterpart of :func:`links_pos_in_anchor_frame`. Rank-agnostic:
    ``anchor_quat_w`` is ``(*B, 4)`` and ``links_quat_w`` is ``(*B, L, 4)``;
    returns ``(*B, L, 4)``. See :func:`links_pos_in_anchor_frame` for the
    batch-shape / future-window convention.
    """
    inv_anchor = inv_quat(anchor_quat_w).unsqueeze(-2).expand(links_quat_w.shape)
    return quat_mul(inv_anchor, links_quat_w)
