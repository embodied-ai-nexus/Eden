"""Actuator math: reflected inertia and the Raibert PD heuristic, pinned to hand-computed values."""

from __future__ import annotations

import pytest

from eden.options.actuators import DamiaoDM4310, DamiaoDM4340
from eden.utils.actuator import (
    raibert_heuristic_action_scale,
    raibert_heuristic_kd,
    raibert_heuristic_kp,
    reflected_inertia_from_two_stage_planetary,
)


def test_two_stage_planetary_reflects_each_stage_through_the_gears_after_it():
    # Unitree N5020-16: 0.139e-4 * (46/18 + 1)^2 (56/16 + 1)^2 + 0.017e-4 * 4.5^2 + 0.169e-4.
    inertia = reflected_inertia_from_two_stage_planetary(
        (0.139e-4, 0.017e-4, 0.169e-4), (1.0, 1.0 + 46 / 18, 1.0 + 56 / 16)
    )
    assert inertia == pytest.approx(3.609725e-3, rel=1e-9)


@pytest.mark.parametrize(
    "actuator, kp, kd, action_scale",
    [
        # armature 1.8e-3 kg m^2, effort 7 N m, at the default 10 Hz: kp = a w^2, kd = 4 a w.
        (DamiaoDM4310, 7.1061152, 0.45238934, 0.24626677),
        # armature 3.2e-2 kg m^2, effort 27 N m.
        (DamiaoDM4340, 126.33094, 8.0424772, 0.053431093),
    ],
)
def test_raibert_heuristic_at_the_default_10_hz_bandwidth(actuator, kp, kd, action_scale):
    assert raibert_heuristic_kp(actuator) == pytest.approx(kp, rel=1e-7)
    assert raibert_heuristic_kd(actuator) == pytest.approx(kd, rel=1e-7)
    assert raibert_heuristic_action_scale(actuator) == pytest.approx(action_scale, rel=1e-7)
    # The classmethod spellings on the spec go through the same helpers.
    assert actuator.RAIBERT_HEURISTIC_KP() == pytest.approx(kp, rel=1e-7)
