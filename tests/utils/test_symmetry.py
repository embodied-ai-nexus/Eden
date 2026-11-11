"""Tests for ``eden.utils.symmetry``: the XZ-plane mirror primitives and ``SymmetryAugmentation``."""

from __future__ import annotations

import pytest
import torch

from eden.utils.symmetry import (
    SymmetryAugmentation,
    mirror_xz_ang_vel_,
    mirror_xz_joint,
    mirror_xz_quat_,
    mirror_xz_y_negate_,
)


def test_mirror_xz_joint_reorders_then_signs_over_any_leading_dims():
    joints = torch.arange(24.0).view(2, 3, 4)  # (B, T, n_joints)
    out = mirror_xz_joint(joints, torch.tensor([1, 0, 3, 2]), torch.tensor([-1.0, 1.0, -1.0, 1.0]))
    # [20, 21, 22, 23] -> reorder [21, 20, 23, 22] -> sign [-21, 20, -23, 22]
    assert out[1, 2].tolist() == [-21.0, 20.0, -23.0, 22.0]
    assert out[0, 0].tolist() == [-1.0, 0.0, -3.0, 2.0]


@pytest.mark.parametrize(
    "mirror,value,expected",
    [
        (mirror_xz_ang_vel_, [1.0, 2.0, 3.0], [-1.0, 2.0, -3.0]),  # roll and yaw rates flip
        (mirror_xz_quat_, [1.0, 2.0, 3.0, 4.0], [1.0, -2.0, 3.0, -4.0]),  # wxyz: x and z flip
        (mirror_xz_y_negate_, [5.0, 6.0, 7.0], [5.0, -6.0, 7.0]),
    ],
    ids=["ang_vel", "quat", "y_negate"],
)
def test_in_place_mirrors_negate_their_components(mirror, value, expected):
    tensor = torch.tensor([value])
    assert mirror(tensor) is tensor
    assert tensor.tolist() == [expected]


def test_mirror_xz_y_negate_along_a_non_last_axis():
    v = torch.ones(2, 3, 4)
    mirror_xz_y_negate_(v, axis=1)
    assert (v[:, 1] == -1.0).all() and (v[:, [0, 2]] == 1.0).all()


@pytest.fixture
def augmentation():
    """A 4-DoF bilateral robot whose roll joints flip sign under the mirror."""
    return SymmetryAugmentation(
        dofs_name=["L_hip_roll", "L_hip_pitch", "R_hip_roll", "R_hip_pitch"],
        symmetry_joint_names={
            "L_hip_roll": "R_hip_roll",
            "R_hip_roll": "L_hip_roll",
            "L_hip_pitch": "R_hip_pitch",
            "R_hip_pitch": "L_hip_pitch",
        },
        flip_sign_joint_names=["L_hip_roll", "R_hip_roll"],
        obs_components={
            "base_lin_vel": 3,
            "base_ang_vel": 3,
            "dofs_pos": 4,
            "command_lin_vel": 2,
            "command_ang_vel": 1,
        },
    )


def test_mirror_actions_swaps_sides_and_flips_roll(augmentation):
    # [L_roll, L_pitch, R_roll, R_pitch] -> [-R_roll, R_pitch, -L_roll, L_pitch]
    assert augmentation.mirror_actions(torch.tensor([[1.0, 2.0, 3.0, 4.0]])).tolist() == [[-3.0, 4.0, -1.0, 2.0]]


_OBS = [[1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 10.0, 20.0, 30.0, 40.0, 0.7, 0.9, 0.3]]
#: lin_vel y, ang_vel x/z, the dofs (swapped, roll flipped), command y and yaw all change sign.
_MIRRORED_OBS = [[1.0, -2.0, 3.0, -0.5, 1.5, -2.5, -30.0, 40.0, -10.0, 20.0, 0.7, -0.9, -0.3]]


def test_mirror_observations_dispatches_each_component(augmentation):
    torch.testing.assert_close(augmentation.mirror_observations(torch.tensor(_OBS)), torch.tensor(_MIRRORED_OBS))


def test_augment_appends_the_mirrored_batch(augmentation):
    torch.testing.assert_close(
        augmentation.augment_observations(torch.tensor(_OBS)), torch.tensor(_OBS + _MIRRORED_OBS)
    )
    augmented = augmentation.augment_actions(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    assert augmented.tolist() == [[1.0, 2.0, 3.0, 4.0], [-3.0, 4.0, -1.0, 2.0]]


def test_an_obs_component_without_a_mirror_handler_warns():
    with pytest.warns(UserWarning, match="no _mirror_unknown_thing handler"):
        SymmetryAugmentation(
            dofs_name=["a", "b"],
            symmetry_joint_names={"a": "b", "b": "a"},
            flip_sign_joint_names=[],
            obs_components={"unknown_thing": 4},
        )
