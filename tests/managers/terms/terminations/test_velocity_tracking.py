"""The velocity-tracking termination terms in ``eden/managers/terms/terminations/velocity_tracking.py``."""

from __future__ import annotations

import pytest
import math

import torch

from eden.managers.terms.terminations.velocity_tracking import fell_over

pytestmark = pytest.mark.requires_genesis


def test_fell_over_fires_past_the_limit_angle(box_env, restore_state):
    """A 30 deg lean survives a 45 deg limit; a 60 deg lean does not, whichever way it leans."""
    box = restore_state(box_env.entities["robot"])
    halves = torch.tensor([30.0, 60.0, 30.0, 60.0]).deg2rad() / 2.0
    zeros = torch.zeros(4)
    rolls = torch.stack([halves.cos(), halves.sin(), zeros, zeros], dim=-1)[:2]
    pitches = torch.stack([halves.cos(), zeros, halves.sin(), zeros], dim=-1)[2:]
    box.set_quat(torch.cat([rolls, pitches]))
    assert fell_over(box_env, entity_name="robot", limit_angle=math.radians(45.0)).tolist() == [
        False,
        True,
        False,
        True,
    ]
