"""Tests for ``eden.datasets.modifiers.smooth_start_end.SmoothStartTransition``, which runs its IK in its own scene.

The IK helper itself is tested in ``tests/utils/motion_reference/test_bipedal_ik.py``.
"""

from __future__ import annotations

import pytest
import torch

from eden.datasets.modifiers import DATASET_MODIFIER_REGISTRY, SmoothStartTransition
from eden.options.robots.unitree_g1 import G1_IK_PROFILE

DEFAULT_Z = float(G1_IK_PROFILE.robot_options.default_root_pos[2])


def _modifier():
    options = SmoothStartTransition.configure(
        ik_profile=G1_IK_PROFILE, default_pose_sec=0.4, step_sec=0.2, transition_sec=0.2
    )
    return DATASET_MODIFIER_REGISTRY.get(options.name)(options=options)


def test_it_prepends_a_deterministic_transition_with_finite_differenced_velocities(g1_clips):
    """Clips start with the hips rolled open, so the transition from the default pose has to move the legs."""
    mod = _modifier()

    first = mod.apply({"g1": g1_clips([DEFAULT_Z] * 2, T=50, hip_roll=0.2)})["g1"]
    second = mod.apply({"g1": g1_clips([DEFAULT_Z] * 2, T=50, hip_roll=0.2)})["g1"]

    assert all(length > 50 for length in first.buffer_length.tolist())
    assert second.buffer_length.tolist() == first.buffer_length.tolist()
    dofs_vel = first.dofs_vel.concatenated_tensor
    assert torch.isfinite(dofs_vel).all()
    assert 1e-3 < dofs_vel.abs().max().item() < 50.0  # the legs move, and the finite difference did not blow up


def test_it_owns_one_robot_and_refuses_several_keys_before_building_a_scene():
    with pytest.raises(ValueError, match="exactly one motion-sequences key"):
        _modifier().apply({"g1": object(), "object": object()})


def test_it_refuses_a_sequence_with_body_model_parameters_before_building_a_scene(make_sequence):
    from tests.datasets.modifiers.conftest import with_body_model

    with pytest.raises(ValueError, match=r"SmoothStartTransition cannot apply to 'g1': it carries body-model"):
        _modifier().apply({"g1": with_body_model(make_sequence())})


def test_a_configured_key_the_dataset_lacks_is_refused_rather_than_skipped():
    options = SmoothStartTransition.configure(ik_profile=G1_IK_PROFILE, motion_sequence_keys=("robot",))
    with pytest.raises(KeyError, match=r"\['robot'\] not in motion_sequences \['g1'\]"):
        SmoothStartTransition(options=options).apply({"g1": object()})
