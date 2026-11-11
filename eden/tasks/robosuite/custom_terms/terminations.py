"""Termination terms for the robosuite ports.

robosuite itself only terminates on the horizon (Wipe excepted); Eden's episodes are shorter, so a
task also ends when its object leaves the workspace — a fallen object cannot be recovered and the
remaining steps would only teach the policy about an unreachable state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.managers import TERMINATION_TERM_REGISTRY

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase


@TERMINATION_TERM_REGISTRY.register()
def entity_below_height(env: "RLEnvBase", *, entity_name: str, height: float) -> torch.Tensor:
    """Terminate when the entity's root drops below ``height`` (it fell off the table).

    ``relative=True`` because ``height`` is reasoned from the table top and the asset's own
    geometry: the default solver frame is re-origined onto the body's centre of mass, which for an
    object like the hammer sits 6.8 cm from its origin and swings with the body as it tilts — enough
    to terminate an episode with the object still in hand.
    """
    return env.entities[entity_name].get_pos(relative=True)[:, 2] < height
