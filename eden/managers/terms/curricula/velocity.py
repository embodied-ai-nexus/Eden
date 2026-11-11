"""Velocity curriculum for progressive velocity range expansion."""

from __future__ import annotations

from typing import TYPE_CHECKING

from eden.managers.curriculum_manager import (
    CURRICULUM_TERM_REGISTRY,
    CurriculumTerm,
    CurriculumTermOptions,
)

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase


@CURRICULUM_TERM_REGISTRY.register()
class VelocityCommandCurriculum(CurriculumTerm):
    """Curriculum that progressively expands velocity command ranges.

    Parameters
    ----------
    command_name : str
        The name of the velocity command term in the command manager.
    velocity_stages : list[dict]
        Stages, in any order (sorted by ``step`` on construction). Each dict has ``step`` (int) and optional
        ``lin_vel_x``, ``lin_vel_y``, ``ang_vel_z`` tuples for the new ranges.
        Example::

            [
                {"step": 0, "lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.5, 0.5)},
                {"step": 120000, "lin_vel_x": (-1.0, 2.0), "lin_vel_y": (-1.0, 1.0)},
            ]
    """

    command_name: str = "velocity_command"
    velocity_stages: list[dict] = []

    def __init__(self, env: RLEnvBase, options: CurriculumTermOptions):
        super().__init__(env=env, options=options)
        self.velocity_stages = sorted(self.velocity_stages, key=lambda x: x["step"], reverse=True)
        self.command_term = self._env.command_manager.get_term(self.command_name)
        self._current_stage_step = -1

    def compute(self, *args, **kwargs):
        for stage in self.velocity_stages:
            if self._env.common_step_counter >= stage["step"]:
                if stage["step"] != self._current_stage_step:
                    self._current_stage_step = stage["step"]
                    if "lin_vel_x" in stage:
                        self.command_term.lin_vel_x_range = tuple(stage["lin_vel_x"])
                    if "lin_vel_y" in stage:
                        self.command_term.lin_vel_y_range = tuple(stage["lin_vel_y"])
                    if "ang_vel_z" in stage:
                        self.command_term.ang_vel_z_range = tuple(stage["ang_vel_z"])
                break
        return {"stage_step": self._current_stage_step}
