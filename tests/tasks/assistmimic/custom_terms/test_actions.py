"""``PhcSphericalPDAction``: a PHC rotation-vector target becomes the hinge angles of the same rotation."""

from __future__ import annotations

import math

import pytest
import torch

from eden.tasks.assistmimic.robots import SMPL_DOF_NAMES
from tests.support.envs import build_env
from tests.tasks.assistmimic.conftest import assistmimic_config


@pytest.mark.requires_genesis
def test_a_target_past_the_hinge_limit_is_sent_as_is(assistmimic_env):
    """-0.9 pi about x from a hip at 2.5 rad: the nearest whole-turn shift, 1.1 pi, past the +-180 deg range.

    It is sent unclamped, as PHC does, so the PD keeps driving the hip into its limit: clamping it to pi weakens
    the drive, and neither (-0.9 pi, 0, 0) nor (0.1 pi, pi, pi) — the exact decompositions inside the range —
    is taken, since both swing the hip half a turn.
    """
    env = assistmimic_env
    env.reset()
    term = env.action_manager.get_term("caregiver")
    entity = env.entities["caregiver"]
    hip = [SMPL_DOF_NAMES.index(f"L_Hip_{axis}") for axis in "xyz"]
    current = entity.get_dofs_pos(dofs_idx_local=term.dofs_idx_local).clone()
    current[:, hip[0]] = 2.5
    entity.set_dofs_pos(current, dofs_idx_local=term.dofs_idx_local)

    actions = torch.zeros(env.num_envs, term.action_dim, device=env.device)
    actions[:, hip[0]] = -0.9  # rotation vector (-0.9 pi, 0, 0) on L_Hip
    term.compute(actions)

    target = term.processed_action
    _, upper = entity.get_dofs_limit(dofs_idx_local=term.dofs_idx_local)
    assert (target[:, hip[0]] > upper[:, hip[0]]).all()
    expected = torch.tensor([1.1 * math.pi, 0.0, 0.0], device=env.device).expand(env.num_envs, 3)
    torch.testing.assert_close(target[:, hip], expected, atol=1e-5, rtol=0)
    # Every other body was asked for the identity, i.e. zero hinge angles.
    others = [i for i in range(len(SMPL_DOF_NAMES)) if i not in hip]
    torch.testing.assert_close(target[:, others], torch.zeros_like(target[:, others]), atol=1e-6, rtol=0)
    env.reset()


@pytest.mark.requires_genesis
def test_a_knee_bending_past_90_deg_keeps_its_x_and_z_targets(assistmimic_env):
    """Knee y at 1.5 rad asked for 2 rad (action 0.4 at PHC's knee scale 5): only the y hinge moves.

    The plain decomposition of that rotation is (pi, pi - 2, pi), which would swing the knee's x and z by pi.
    """
    env = assistmimic_env
    env.reset()
    term = env.action_manager.get_term("caregiver")
    entity = env.entities["caregiver"]
    knee = [SMPL_DOF_NAMES.index(f"L_Knee_{axis}") for axis in "xyz"]
    current = entity.get_dofs_pos(dofs_idx_local=term.dofs_idx_local).clone()
    current[:, knee[1]] = 1.5
    entity.set_dofs_pos(current, dofs_idx_local=term.dofs_idx_local)

    actions = torch.zeros(env.num_envs, term.action_dim, device=env.device)
    actions[:, knee[1]] = 0.4
    term.compute(actions)

    expected = torch.tensor([0.0, 2.0, 0.0], device=env.device).expand(env.num_envs, 3)
    torch.testing.assert_close(term.processed_action[:, knee], expected, atol=1e-5, rtol=0)
    env.reset()


@pytest.mark.requires_genesis
def test_a_knee_held_at_90_deg_keeps_its_x_and_z_targets_under_a_small_off_axis_action(assistmimic_env):
    """At the hinges' gimbal lock a 0.02 rad twist about z decomposes exactly as x and z ``+-1.5`` rad apart.

    Within ``lock_band`` the target keeps the current ``a - c`` instead, so x and z stay near zero.
    """
    env = assistmimic_env
    env.reset()
    term = env.action_manager.get_term("caregiver")
    entity = env.entities["caregiver"]
    knee = [SMPL_DOF_NAMES.index(f"L_Knee_{axis}") for axis in "xyz"]
    current = entity.get_dofs_pos(dofs_idx_local=term.dofs_idx_local).clone()
    current[:, knee[1]] = math.pi / 2
    entity.set_dofs_pos(current, dofs_idx_local=term.dofs_idx_local)

    actions = torch.zeros(env.num_envs, term.action_dim, device=env.device)
    actions[:, knee[1]] = math.pi / 2 / term.knee_y_scale
    actions[:, knee[2]] = 0.02 / term.scale
    term.compute(actions)

    target = term.processed_action[:, knee]
    assert target[:, [0, 2]].abs().max() < 0.05
    torch.testing.assert_close(target[:, 1], torch.full_like(target[:, 1], math.pi / 2), atol=0.02, rtol=0)
    env.reset()


@pytest.mark.requires_genesis
def test_a_lock_band_of_1_is_refused_at_build(clips_file):
    cfg = assistmimic_config("assistmimic", clips_file)
    cfg.action_options.caregiver.lock_band = 1.0
    with pytest.raises(ValueError, match=r"PhcSphericalPDAction on entity 'caregiver': lock_band=1.0 bounds \|cos b\|"):
        build_env(cfg)
