"""Goal interfaces of the multi-encoder actor: motion tracking and a body-frame velocity command.

Every interface is trained on the motion-tracking reward; a goal spec only turns the reference motion into the
goal its encoder reads.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from eden.learning.wrappers.rsl_rl.supermimic.representation import motion_goal
from eden.utils.geom import inv_transform_by_quat

if TYPE_CHECKING:
    from eden.learning.wrappers.rsl_rl.supermimic.options import VelocityGoalOptions
    from eden.learning.wrappers.rsl_rl.supermimic.representation import GoalWindow
    from eden.utils.motion_reference.library import MotionLibrary

#: ``(library, frames, aux_state) -> goal``.
GoalFn = Callable[["MotionLibrary", torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class GoalSpec:
    """One goal interface: its name, goal width and the goal it reads at a reference frame."""

    name: str
    goal_dim: int
    goal: GoalFn


def motion_goal_spec(num_joints: int, window: GoalWindow) -> GoalSpec:
    """Motion tracking: the actor half of the reference window (:func:`representation.motion_goal`)."""
    return GoalSpec(
        name="motion",
        goal_dim=window.dims(num_joints)[1],
        goal=lambda library, frames, aux_state: motion_goal(library, frames, aux_state, window)[1],
    )


def velocity_goal_spec(options: VelocityGoalOptions, device: str | torch.device) -> GoalSpec:
    """A body-frame velocity command ``[vx, vy, wz]``.

    The command is the reference root velocity, averaged over ``window_frames`` frames centred on the current one,
    in the robot's root frame, plus fresh uniform noise of half-width ``command_noise`` per step, clamped to the
    command ranges. The window clamps to the concatenated library, not to the clip, so a frame within half a window
    of a clip's end averages in the neighbouring clip's velocity.
    """
    # On the training device from the start: a CPU tensor moved inside the goal would be a host-to-device copy in
    # the captured CUDA graph of the policy update, which PyTorch refuses.
    lo = torch.tensor(
        [options.lin_vel_x_range[0], options.lin_vel_y_range[0], options.ang_vel_z_range[0]], device=device
    )
    hi = torch.tensor(
        [options.lin_vel_x_range[1], options.lin_vel_y_range[1], options.ang_vel_z_range[1]], device=device
    )
    noise = torch.tensor(options.command_noise, device=device)
    window = options.window_frames

    def centred_mean(track: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        if window <= 1:
            return track[frames]
        offsets = torch.arange(window, device=frames.device) - window // 2
        idx = (frames[:, None] + offsets).clamp(0, track.shape[0] - 1)
        return track[idx].mean(dim=1)

    def goal(library: MotionLibrary, frames: torch.Tensor, aux_state: torch.Tensor) -> torch.Tensor:
        quat = aux_state[..., 3:7]
        v_b = inv_transform_by_quat(centred_mean(library.root_lin_vel_w, frames), quat)
        w_b = inv_transform_by_quat(centred_mean(library.root_ang_vel_w, frames), quat)
        command = torch.stack([v_b[:, 0], v_b[:, 1], w_b[:, 2]], dim=-1)
        command = command + (torch.rand_like(command) * 2 - 1) * noise
        return torch.clamp(command, lo, hi)

    return GoalSpec(name="velocity", goal_dim=3, goal=goal)
