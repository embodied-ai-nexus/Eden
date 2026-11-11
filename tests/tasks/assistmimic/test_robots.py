"""AssistMimic's weakened recipients against the caregiver they are derived from."""

from __future__ import annotations

import pytest

from eden.options.robots.smpl_humanoid import PHC_EFFORT_LIMIT, PhcSmplHumanoid
from eden.tasks.assistmimic.robots import (
    RECIPIENT_WEAKNESS_SCALE,
    AssistMimicBedRecipient,
    AssistMimicRecipient,
    _weakened,
)

CAREGIVER = PhcSmplHumanoid()


def test_recipients_load_the_70pct_mass_asset():
    assert CAREGIVER.file == "humanoids/smpl_humanoid/smpl_humanoid_phc.xml"
    assert (
        AssistMimicRecipient().file
        == AssistMimicBedRecipient().file
        == "humanoids/smpl_humanoid/smpl_humanoid_phc_70pct_mass.xml"
    )


@pytest.mark.parametrize(
    ("robot", "dof", "weak", "effort"),
    [
        (AssistMimicRecipient(), "L_Hip_x", True, 80.0),
        (AssistMimicRecipient(), "R_Toe_z", True, 80.0),
        (AssistMimicRecipient(), "Torso_y", False, PHC_EFFORT_LIMIT),
        (AssistMimicRecipient(), "L_Shoulder_x", False, PHC_EFFORT_LIMIT),
        (AssistMimicBedRecipient(), "L_Hip_x", True, 20.0),
        (AssistMimicBedRecipient(), "Chest_z", True, 40.0),
        (AssistMimicBedRecipient(), "R_Knee_y", True, 80.0),
        (AssistMimicBedRecipient(), "L_Shoulder_x", False, PHC_EFFORT_LIMIT),
    ],
)
def test_weak_bodies_keep_half_their_gains_and_take_their_effort_limit(robot, dof, weak, effort):
    scale = RECIPIENT_WEAKNESS_SCALE if weak else 1.0
    assert robot.default_dofs_kp[dof] == pytest.approx(CAREGIVER.default_dofs_kp[dof] * scale)
    assert robot.default_dofs_kd[dof] == pytest.approx(CAREGIVER.default_dofs_kd[dof] * scale)
    assert robot.default_dofs_force_limits[dof] == effort


def test_a_weakness_profile_naming_an_unknown_body_raises():
    with pytest.raises(ValueError, match=r"\['L_Thigh'\].*Pick from"):
        _weakened({50.0: ("L_Thigh",)})
