"""LimX Oli: per-joint defaults that reproduce the deployed HU_D04 gains, and the hand-stripped variant built."""

from __future__ import annotations

import numpy as np
import pytest
from genesis.utils import geom as gu

from eden.options.robots.limx_oli import OLI_ACTION_SCALE, Oli, OliNoHands
from tests.options.robots.conftest import build_scene, dofs_in_declared_order, link_pose, links_pos

# (dofs_spec pattern, armature, effort limit, deployed kp, kd, action scale), identified from the
# deployed HU_D04 controllers and the RL MJCF -- an oracle independent of the Raibert helpers.
_DEPLOYED = [
    ("*_hip_*_joint", 0.14125, 140.0, 139.41, 17.75, 0.2511),
    ("*_knee_joint", 0.14125, 140.0, 139.41, 17.75, 0.2511),
    ("*_ankle_*_joint", 0.09488, 42.0, 93.65, 11.92, 0.1121),
    ("waist_*_joint", 0.09488, 42.0, 93.65, 11.92, 0.1121),
    ("*_shoulder_*_joint", 0.0886706, 42.0, 87.51, 11.14, 0.12),
    ("*_elbow_joint", 0.0886706, 42.0, 87.51, 11.14, 0.12),
    ("*_wrist_*_joint", 0.0153218, 19.0, 15.12, 1.93, 0.3141),
    ("head_*_joint", 0.0153218, 19.0, 15.12, 1.93, 0.3141),
]


def test_every_spec_pattern_carries_a_deployed_gain():
    assert set(Oli.dofs_spec) == {pattern for pattern, *_ in _DEPLOYED}


@pytest.mark.parametrize("pattern, armature, effort, kp, kd, action_scale", _DEPLOYED)
def test_defaults_match_the_deployed_gains(pattern, armature, effort, kp, kd, action_scale):
    """Oli's heuristic runs at 5 Hz, not the global 10 Hz; at 10 Hz every kp here would be 4x off."""
    oli = Oli()
    assert oli.default_dofs_kp[pattern] == pytest.approx(kp, abs=0.02)
    assert oli.default_dofs_kd[pattern] == pytest.approx(kd, abs=0.02)
    assert oli.default_dofs_armature[pattern] == pytest.approx(armature, rel=1e-9)
    assert oli.default_dofs_force_limits[pattern] == effort
    assert OLI_ACTION_SCALE[pattern] == pytest.approx(action_scale, abs=2e-3)


# ── OliNoHands, built beside the stock Oli ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def olis():
    """Both variants at one pose, fixed-base (nothing steps here, so they may overlap)."""
    env = build_scene(stock=Oli(is_fixed_base=True), no_hands=OliNoHands(is_fixed_base=True))
    return env.entities["stock"], env.entities["no_hands"]


def test_no_hands_keeps_the_31_dof_tree_and_every_link_where_it_was(olis):
    """The only change to the link set is the two ``*_hand_mount`` flange frames it adds."""
    stock, no_hands = olis
    declared = Oli.model_fields["dofs_name"].default
    assert no_hands.num_dofs == stock.num_dofs == 31
    assert dofs_in_declared_order(no_hands, declared) == declared
    stock_links = [link.name for link in stock.links]
    no_hands_links = [link.name for link in no_hands.links]
    assert [n for n in stock_links if n not in no_hands_links] == []
    assert [n for n in no_hands_links if n not in stock_links] == ["left_hand_mount", "right_hand_mount"]
    np.testing.assert_allclose(links_pos(no_hands, stock_links), links_pos(stock, stock_links), atol=1e-6)


@pytest.mark.parametrize("side", ["left", "right"])
def test_the_flange_frame_is_a_massless_frame_58_mm_down_the_wrist(olis, side):
    """An end effector welds to it with identity; upstream's ``*_hand_contact`` sits at the same origin.

    It must survive the fixed-link merge (only ``links_to_keep`` guarantees that) and add nothing a solver
    can feel: Genesis floors a massless link at ~1.2e-7 kg, four orders below LimX's own 1 g marker frames.
    """
    _, no_hands = olis
    wrist_pos, wrist_quat = link_pose(no_hands, f"{side}_wrist_roll_link")
    mount_pos, mount_quat = link_pose(no_hands, f"{side}_hand_mount")
    np.testing.assert_allclose(gu.quat_to_R(wrist_quat).T @ (mount_pos - wrist_pos), (0.0, 0.0, -0.058), atol=1e-6)
    np.testing.assert_allclose(mount_quat, wrist_quat, atol=1e-6)
    mount = no_hands.get_link(f"{side}_hand_mount")
    assert len(mount.geoms) == 0
    assert float(no_hands.get_links_inertial_mass(ls_idx_local=[mount.idx_local])[0, 0]) < 1e-6


def test_the_bare_wrists_drop_the_moulded_hands_mass(olis):
    """Two wrists recomputed for bare metal (0.581445 -> 0.205092 kg) and four 1 g marker boxes dropped."""
    stock, no_hands = olis
    drop = float(stock.get_links_inertial_mass().sum() - no_hands.get_links_inertial_mass().sum())
    assert drop == pytest.approx(2 * (0.581445 - 0.205092) + 4 * 0.001, abs=1e-4)


@pytest.mark.parametrize("link_name", ["left_wrist_roll_link", "right_wrist_roll_link"])
def test_the_hand_geometry_is_gone_from_the_wrist_hull(olis, link_name):
    """The hand lives in the stock wrist *mesh*, ~15 cm past the wrist; the ``_nohand`` mesh stops at the flange."""

    def z_extent(entity):
        verts = np.concatenate([np.asarray(g.init_verts) for g in entity.get_link(link_name).geoms])
        return float(np.ptp(verts[:, 2]))

    stock, no_hands = olis
    assert z_extent(no_hands) < 0.15 and z_extent(stock) - z_extent(no_hands) > 0.10
