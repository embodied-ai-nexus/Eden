"""Quality-filter mixin for motion datasets."""

from __future__ import annotations
from typing import TYPE_CHECKING

import eden as en
import torch

if TYPE_CHECKING:
    from eden.utils.motion_reference.motion_data import MotionSequence, BatchMotionSequence


class FilterMixin:
    """Collection of quality filters for reference motion data."""

    @staticmethod
    def filter_wbm(
        motion: MotionSequence,
        min_height_threshold: float = -0.05,
        max_velocity_threshold: float = 15.0,
        max_dof_vel_threshold: float = 40.0,
        duration_height_filter: float | None = 0.2,
        duration_height_seconds: float = 1.0,
    ) -> bool:
        """
        Filter to exclude whole body motions based on quality criteria.

        Parameters
        ----------
        motion: MotionSequence
            The motion sequence to check.
        min_height_threshold: float
            The minimum world z (m) any link may reach.
        max_velocity_threshold: float
            The maximum finite-difference link velocity (m/s, per axis).
        max_dof_vel_threshold: float
            The maximum absolute DOF velocity.
        duration_height_filter: float | None
            Height (m) of each frame's lowest link above the clip's lowest point beyond which the frame counts as
            airborne for the duration-based filter; if None, this filter is disabled.
        duration_height_seconds: float
            How long (s) the motion may stay airborne before it is rejected.

        Returns
        -------
        bool
            True if motion passes all filters, False otherwise.
        """
        z_positions = motion.links_pos_w[..., 2]
        min_height = z_positions.min()
        framerate = float(motion.framerate.item())

        # Any link below min_height_threshold
        if min_height < min_height_threshold:
            en.logger.info(f"Skipping because it has {min_height} smaller than {min_height_threshold}")
            return False

        # Any link velocity too large, by finite difference rather than the stored link
        # velocities, which can be smoothed
        if motion.links_pos_w.shape[0] > 1:
            global_velocity_fin_diff = (motion.links_pos_w[1:] - motion.links_pos_w[:-1]) * framerate
            max_global_velocity = global_velocity_fin_diff.abs().max()
            if max_global_velocity > max_velocity_threshold:
                en.logger.info(f"Skipping because it has {max_global_velocity} larger than {max_velocity_threshold}")
                return False

        # Check if any dof_vels is too large
        max_dof_vel = motion.dofs_vel.abs().max()
        if max_dof_vel > max_dof_vel_threshold:
            en.logger.info(f"Skipping because it has {max_dof_vel} larger than {max_dof_vel_threshold}")
            return False

        if duration_height_filter is not None and duration_height_seconds > 0:
            floor_estimate = min_height
            lowest_point_per_frame = z_positions.min(dim=-1).values
            lowest_distance_from_floor = lowest_point_per_frame - floor_estimate
            too_high_per_frame = lowest_distance_from_floor > duration_height_filter
            # check if too high for n consecutive frames
            consecutive_frames = max(1, int(duration_height_seconds * framerate))
            num_consecutive = 0
            for is_too_high in too_high_per_frame:
                if is_too_high:
                    num_consecutive += 1
                else:
                    num_consecutive = 0

                if num_consecutive >= consecutive_frames:
                    en.logger.info(
                        f"Skipping because it has {num_consecutive} consecutive frames with height larger than {duration_height_filter}"
                    )
                    return False
        return True

    @staticmethod
    def filter_wbm_batched(
        motion: BatchMotionSequence,
        min_height_threshold: float = -0.05,
        max_velocity_threshold: float = 15.0,
        max_dof_vel_threshold: float = 40.0,
        duration_height_filter: float | None = 0.2,
        duration_height_seconds: float = 1.0,
    ) -> torch.Tensor:
        """
        Batched version of `filter_wbm` for variable-length motion sequences; the parameters are the same.

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape [num_motions]. True means the motion passes all filters.
        """
        links_pos_w, frame_mask = motion.links_pos_w.with_mask[:]
        z_positions = links_pos_w[..., 2]
        valid_pos_mask = frame_mask.unsqueeze(-1)
        num_motions = z_positions.shape[0]
        device = z_positions.device

        passed = torch.ones(num_motions, dtype=torch.bool, device=device)
        inf = torch.tensor(torch.inf, dtype=z_positions.dtype, device=device)
        ninf = torch.tensor(-torch.inf, dtype=z_positions.dtype, device=device)

        # 1) Check minimum link height over valid frames.
        min_height = z_positions.masked_fill(~valid_pos_mask, inf).amin(dim=(1, 2))
        has_valid_frame = frame_mask.any(dim=1)
        passed &= min_height >= min_height_threshold
        passed &= has_valid_frame

        # 2) Check finite-difference link velocity over valid adjacent frames.
        framerate = motion.framerate.to(z_positions.dtype)
        global_velocity_fin_diff = (
            z_positions.new_zeros((num_motions, 0, z_positions.shape[-1], 3))
            if z_positions.shape[1] <= 1
            else (links_pos_w[:, 1:] - links_pos_w[:, :-1]) * framerate[:, None, None, None]
        )
        if global_velocity_fin_diff.numel() > 0:
            valid_velocity_mask = frame_mask[:, 1:] & frame_mask[:, :-1]
            velocity_abs_max = (
                global_velocity_fin_diff.abs()
                .masked_fill(~valid_velocity_mask.unsqueeze(-1).unsqueeze(-1), ninf)
                .amax(dim=(1, 2, 3))
            )
            velocity_abs_max = torch.where(
                valid_velocity_mask.any(dim=1),
                velocity_abs_max,
                torch.zeros_like(velocity_abs_max),
            )
            passed &= velocity_abs_max <= max_velocity_threshold

        # 3) Check DOF velocity over valid frames.
        dofs_vel, dofs_mask = motion.dofs_vel.with_mask[:]
        dof_abs_max = dofs_vel.abs().masked_fill(~dofs_mask.unsqueeze(-1), ninf).amax(dim=(1, 2))
        dof_abs_max = torch.where(dofs_mask.any(dim=1), dof_abs_max, torch.zeros_like(dof_abs_max))
        passed &= dof_abs_max <= max_dof_vel_threshold

        # 4) Check whether the lowest point stays too high above floor for too long.
        if duration_height_filter is not None and duration_height_seconds > 0:
            floor_estimate = min_height
            lowest_point_per_frame = z_positions.masked_fill(~valid_pos_mask, inf).amin(dim=-1)
            too_high_per_frame = frame_mask & (
                (lowest_point_per_frame - floor_estimate[:, None]) > duration_height_filter
            )

            consecutive_frames = torch.clamp((duration_height_seconds * motion.framerate).to(torch.int64), min=1)
            num_consecutive = torch.zeros(num_motions, dtype=torch.int64, device=device)
            rejected_by_duration = torch.zeros(num_motions, dtype=torch.bool, device=device)
            for t in range(too_high_per_frame.shape[1]):
                num_consecutive = torch.where(
                    too_high_per_frame[:, t],
                    num_consecutive + 1,
                    torch.zeros_like(num_consecutive),
                )
                rejected_by_duration |= num_consecutive >= consecutive_frames

            passed &= ~rejected_by_duration

        return passed
