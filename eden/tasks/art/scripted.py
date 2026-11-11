"""The art tasks' scripted demonstrators' shared part: a plan's tip path timed, and followed through the task's action.

:func:`plan_timeline` times a plan's :func:`~eden.tasks.art.plans.tip_path` at ``draw_speed`` on and near the
paper (a stroke, its way in and out included, and a stop such as a dip) and ``travel_speed`` between, from and back
to the arm's home. :class:`TipFollower` drives the task's own action — the differential-IK delta at the tool's
undeformed tip — along it: each step it commands ``gain`` times the offset from the tip to where the path is (at most
90 % of the action's step), and the path's clock waits while the tip trails it by more than ``max_lag`` (``careful_lag``
in a stroke or a stop, where the tip's path is the mark's), so the tip follows the plan's turns rather than cutting
across them. The hand's orientation is held at its tool's :attr:`~eden.tasks.art.workcell.ToolSpec.hand_quat`.
Whatever a demonstrator built on it does, a policy could do through the same interface: it reads the tip and the
plan, and never touches the tool or the marks.
"""

from __future__ import annotations

import numpy as np
import torch

from eden.managers.terms.actions.task_space_actions import DifferentialIKController
from eden.tasks.art import workcell
from eden.tasks.art.plans import TRAVEL, path_times
from eden.tasks.art.terms import CopyMetric
from eden.utils.geom import axis_angle_from_quat, inv_quat, quat_mul


def plan_timeline(
    path: np.ndarray,
    owner: np.ndarray,
    *,
    draw_speed: float,
    travel_speed: float,
    dt: float,
    home: tuple[float, float, float],
    turn_radius: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The tip's goal ``(T, 3)`` ``(u, v, depth)`` per control step, ``(T,)`` flags of the steps in a stroke or a stop
    (*careful* steps), and ``(T,)`` the owner (:func:`~eden.tasks.art.plans.tip_path`'s) of each step's goal.

    ``path (m, 3)`` and ``owner (m,)`` are a plan's :func:`~eden.tasks.art.plans.tip_path`; the timeline starts
    and ends at ``home``, and goes at the speeds of :func:`~eden.tasks.art.plans.path_times`.
    """
    home_np = np.asarray(home, dtype=np.float64)
    points = np.concatenate([home_np[None], path, home_np[None]])
    owners = np.r_[TRAVEL, owner, TRAVEL]
    careful = owners != TRAVEL
    times = path_times(points, owners, draw_speed=draw_speed, travel_speed=travel_speed, turn_radius=turn_radius)
    steps = np.arange(0.0, times[-1] + dt, dt)
    goal = np.stack([np.interp(steps, times, points[:, k]) for k in range(3)], axis=-1)
    goal[-1] = points[-1]
    index = np.clip(np.searchsorted(times, steps, side="right") - 1, 0, len(points) - 2)
    in_careful = careful[index] & careful[index + 1]
    step_owner = np.where(in_careful, owners[index + 1], TRAVEL)
    return goal, in_careful, step_owner


class TipFollower:
    """Batched follower of a timeline (:func:`plan_timeline`) through a built art task's IK action.

    ``act()`` returns the ``(num_envs, 6)`` action for the env's ``action_name`` term; ``reset(envs_idx)`` restarts
    the timeline for the envs the env reset (all of them by default). :attr:`clock` is each env's step of the
    timeline; :attr:`done` is True where it has run out.
    """

    def __init__(
        self,
        env,
        copy: CopyMetric,
        goal: np.ndarray,
        careful: np.ndarray,
        *,
        action_name: str = "arm",
        max_lag: float = 0.008,
        careful_lag: float = 0.0008,
        gain: float = 2.5,
        demonstrator: str = "TipFollower",
    ) -> None:
        self.env, self.copy = env, copy
        self.term = env.action_manager.get_term(action_name)
        if not isinstance(self.term, DifferentialIKController):
            # act() inverts this term's action contract: a delta of effective_scale * tanh(action) at ee_local_pos.
            raise TypeError(
                f"{demonstrator} drives action '{action_name}' by inverting DifferentialIKController's "
                f"effective_scale * tanh(action) delta, but that term is a {type(self.term).__name__}. Point "
                "action_name at the task's DifferentialIKController term (the art tasks' 'arm')."
            )
        device = env.device
        self.path = torch.as_tensor(goal, dtype=torch.float32, device=device)
        self._allowed_lag = torch.where(torch.as_tensor(careful, device=device), careful_lag, max_lag).float()
        self._scale = self.term.effective_scale
        self.gain = gain
        self._target_quat = torch.tensor(workcell.TOOLS[copy.tool_kind].hand_quat, dtype=torch.float32, device=device)
        self.clock = torch.zeros(env.num_envs, dtype=torch.long, device=device)

    @property
    def done(self) -> torch.Tensor:
        return self.clock >= self.path.shape[0] - 1

    def reset(self, envs_idx=None) -> None:
        envs_idx = slice(None) if envs_idx is None else envs_idx
        self.clock[envs_idx] = 0

    def act(self) -> torch.Tensor:
        copy = self.copy
        tip = copy.tip_on_sheet()
        hand = self.env.entities["robot_gripper"].get_link("hand").get_quat()
        last = self.path.shape[0] - 1
        lag = (self.path[self.clock] - tip).norm(dim=-1)
        self.clock = torch.where(lag <= self._allowed_lag[self.clock], (self.clock + 1).clamp(max=last), self.clock)
        error = self.path[self.clock] - tip  # (u, v, depth)
        delta = self.gain * (error[:, :1] * copy.u_axis + error[:, 1:2] * copy.v_axis + error[:, 2:] * copy.inward)
        limit = 0.9 * self._scale[:3].min()
        delta = delta * (limit / delta.norm(dim=-1, keepdim=True).clamp_min(1e-9)).clamp(max=1.0)
        rotation = self.gain * axis_angle_from_quat(quat_mul(self._target_quat.expand_as(hand), inv_quat(hand)))
        rot_limit = 0.9 * self._scale[3:].min()
        rotation = rotation * (rot_limit / rotation.norm(dim=-1, keepdim=True).clamp_min(1e-9)).clamp(max=1.0)
        command = torch.cat([delta, rotation], dim=-1) / self._scale.clamp_min(1e-9)
        return torch.atanh(command.clamp(-0.99, 0.99))
