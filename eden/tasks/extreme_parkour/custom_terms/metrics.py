"""How far each Extreme Parkour episode got along its course, whichever term chose its terrain cell.

The level an episode ran on needs no metric: ``ParkourCommand`` already logs it per finished episode as
``Commands/parkour_command/terrain_level``. Both read the command, which holds the cell an episode was placed on
whatever sampler chose it (``--terrain_sampler level|uniform|sgs``), so the three modes log the same curves.
Metrics compute before the reset of a finished episode, so the value logged for it is its own last step's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.managers import METRIC_TERM_REGISTRY

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.tasks.extreme_parkour.custom_terms.commands import ParkourCommand


@METRIC_TERM_REGISTRY.register()
def parkour_goal_progress(env: EnvBase, *, command_name: str) -> torch.Tensor:
    """The fraction of its course's goals the env has reached (``ParkourCommand.goal_progress``).

    Its success threshold of 1 is the ``all_goals_reached`` termination, which reads the same property.
    """
    command: ParkourCommand = env.command_manager.get_term(command_name)
    return command.goal_progress
