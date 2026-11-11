"""Adaptive curricula for the lift task.

``GoalRangeCurriculum`` expands the goal sampling ranges from an easy center
point as goal-reach success rises. It is a success-gated EMA and self-gates on
grasping — the cube cannot reach any goal before lifting works, so it
naturally stays easy for as long as grasp acquisition takes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.managers.curriculum_manager import (
    CURRICULUM_TERM_REGISTRY,
    CurriculumTerm,
    CurriculumTermOptions,
)
from eden.tasks.manipulation.custom_terms.common import _base_to_world

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase


def _missing_easy_center_message(command_name: str, robot_name: str) -> str:
    return (
        f"GoalRangeCurriculum: easy_center is required, the first goal in the {command_name!r} command's frame "
        f"(the {robot_name!r} base, not the world). Set it to the point above the object's rest pose at the lowest "
        f"target height, e.g. easy_center=(x, 0.0, rest_z - base_z + 0.10)."
    )


@CURRICULUM_TERM_REGISTRY.register()
class GoalRangeCurriculum(CurriculumTerm):
    """Adaptive goal-range curriculum for the lift target command.

    Starts the goal as a single point (``easy_center``, straight above the cube at
    the lowest target height) and expands the command's x/y/z sampling ranges
    toward their configured (full) values as goal-reach success rises. Tracks an
    EMA of per-episode reach success (did the cube come within ``reach_thresh`` of
    the goal) and grows a ``level`` in [0, 1] when the EMA clears ``advance_thresh``
    (shrinks below ``retreat_thresh``). Self-gates on grasping: the cube cannot
    reach any goal before lifting works, so this naturally advances only after
    lifting is learned.

    The configured ranges on the command term are read once at build as the *full*
    targets; ``level`` then interpolates each range endpoint from ``easy_center``.
    """

    command_name: str = "lift_target"
    object_name: str = "cube"
    robot_name: str = "robot"
    reach_thresh: float = 0.08  # cube within this of the goal counts as reached
    advance_thresh: float = 0.7
    retreat_thresh: float = 0.4
    step_up: float = 0.05
    step_down: float = 0.05
    ema_window: int = 4096
    level_init: float = 0.0
    # Easy goal, in the command's frame (the robot base): straight above the cube at the lowest target.
    # Required: it depends on where the task mounts the arm, so no default fits every task.
    easy_center: tuple[float, float, float] | None = None

    _BASE_NUM_ENVS = 4096

    @classmethod
    def configure(cls, **kwargs) -> CurriculumTermOptions:
        """Refuse a missing ``easy_center`` where the config is written, not after the scene has built."""
        if kwargs.get("easy_center") is None:
            command_name = kwargs.get("command_name", cls.command_name)
            raise ValueError(_missing_easy_center_message(command_name, kwargs.get("robot_name", cls.robot_name)))
        return super().configure(**kwargs)

    def __init__(self, env: "RLEnvBase", options: CurriculumTermOptions):
        super().__init__(env=env, options=options)
        # Backstop for options that skipped configure(): a loaded config.json, or one edited in place.
        if self.easy_center is None:
            raise ValueError(_missing_easy_center_message(self.command_name, self.robot_name))
        n = self._env.num_envs
        dev = self.device
        self._cmd = self._env.command_manager.get_term(self.command_name)
        self._obj = self._env.entities[self.object_name]
        # Configured ranges are the full targets to expand toward.
        self._full = (tuple(self._cmd.x_range), tuple(self._cmd.y_range), tuple(self._cmd.z_range))
        self._reached = torch.zeros(n, dtype=torch.bool, device=dev)
        self._goal_ema = torch.tensor(0.0, device=dev)
        self._level = torch.tensor(float(self.level_init), device=dev)
        self._primed = torch.tensor(False, device=dev)
        self._win = max(1.0, self.ema_window * n / self._BASE_NUM_ENVS)
        self._apply_level(float(self._level))

    def _apply_level(self, level: float) -> None:
        (fx, fy, fz) = self._full
        (cx, cy, cz) = self.easy_center
        self._cmd.x_range = (cx + (fx[0] - cx) * level, cx + (fx[1] - cx) * level)
        self._cmd.y_range = (cy + (fy[0] - cy) * level, cy + (fy[1] - cy) * level)
        self._cmd.z_range = (cz + (fz[0] - cz) * level, cz + (fz[1] - cz) * level)

    def compute(self, *args, **kwargs) -> dict:
        env = self._env
        goal_w = _base_to_world(env, self.robot_name, self._cmd.command[..., :3])
        d = (self._obj.get_pos() - goal_w).norm(dim=-1)
        done = env.episode_length_buf == 0
        # On the device throughout, every step, as `torch.where` over "did any episode end": gating on
        # `done.any()`, gathering the ended envs through the mask or reading the EMA back would each be a
        # GPU->CPU sync. The float64 casts reproduce the rounding of the same update computed with
        # Python floats (float32 where it touches the tensors).
        n_done = done.sum()
        has_done = n_done > 0
        succ_mean = (self._reached & done).sum().to(torch.float32) / n_done.clamp(min=1).to(torch.float32)
        w = (n_done.double() / self._win).clamp(max=1.0).float()
        self._goal_ema = torch.where(has_done, torch.lerp(self._goal_ema, succ_mean, w), self._goal_ema)
        self._primed = self._primed | has_done
        frac = n_done.double() / float(env.num_envs)
        ema = self._goal_ema.double()
        up = (self._level + (self.step_up * frac).float()).clamp(max=1.0)
        down = (self._level - (self.step_down * frac).float()).clamp(min=0.0)
        self._level = torch.where(
            has_done & (ema > self.advance_thresh),
            up,
            torch.where(has_done & (ema < self.retreat_thresh), down, self._level),
        )
        # The one sync left: the command samples from Python-float ranges.
        self._apply_level(float(self._level))
        self._reached &= ~done  # restart the ended envs' tracker
        self._reached |= d < self.reach_thresh
        return {
            "level": self._level,
            "goal_ema": self._goal_ema,
            "primed": self._primed,
        }

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        return

    def state_dict(self) -> dict:
        return {
            "level": self._level.detach().cpu().clone(),
            "goal_ema": self._goal_ema.detach().cpu().clone(),
            "primed": bool(self._primed),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("level") is not None:
            self._level = torch.as_tensor(float(state["level"]), device=self.device)
            self._apply_level(float(self._level))
        if state.get("goal_ema") is not None:
            self._goal_ema = torch.as_tensor(float(state["goal_ema"]), device=self.device)
        if "primed" in state:
            self._primed = torch.tensor(bool(state["primed"]), device=self.device)
