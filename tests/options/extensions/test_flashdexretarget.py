"""Tests for ``eden.options.extensions.flashdexretarget``: the pipeline options' ``name`` groups and ranges."""

from __future__ import annotations

import pydantic
import pytest

from eden.options.extensions.flashdexretarget import (
    FDRCoacdConvexOptions,
    FDRCuacdConvexOptions,
    FDRImportOptions,
    FDRKinematicRetargetingOptions,
)


def test_a_convex_override_without_a_name_sets_a_field_of_the_default_backend():
    options = FDRKinematicRetargetingOptions.model_validate({"dataset": "d", "convex": {"force": True}})
    assert isinstance(options.convex, FDRCoacdConvexOptions) and options.convex.force


def test_a_convex_name_picks_its_backend():
    options = FDRKinematicRetargetingOptions.model_validate({"dataset": "d", "convex": {"name": "cuacd", "depth": 2}})
    assert isinstance(options.convex, FDRCuacdConvexOptions) and options.convex.depth == 2


def test_an_unknown_convex_name_lists_the_backends():
    with pytest.raises(pydantic.ValidationError, match=r"convex\.name must be one of 'coacd', 'cuacd'"):
        FDRKinematicRetargetingOptions.model_validate({"dataset": "d", "convex": {"name": "vhacd"}})


def test_a_source_without_a_name_lists_the_releases():
    with pytest.raises(
        pydantic.ValidationError, match=r"source\.name must be one of 'taco', 'oakink2', 'hot3d', 'dexycb'"
    ):
        FDRImportOptions.model_validate({"source": {"table_z": 0.0}, "raw": "r", "dataset": "d", "mano_dir": "m"})


@pytest.mark.parametrize(
    "override",
    [
        {"retarget": {"attempts": 0}},  # no attempt: the clip would vanish from the report
        {"retarget": {"smooth": 0}},
        {"retarget": {"n_guess": 0}},
        {"filters": {"edge_frames": 0}},  # x[-0:] is the whole clip
        {"sdf": {"workers_per_gpu": 0}},
        {"sdf": {"gpus": [-1]}},
        {"workers": 0},
        {"convex": {"coacd": {"apx_mode": "hull"}}},  # CoACD takes "ch" or "box"
    ],
)
def test_a_value_the_pipeline_cannot_run_with_is_refused_up_front(override):
    with pytest.raises(pydantic.ValidationError):
        FDRKinematicRetargetingOptions.model_validate({"dataset": "d", **override})


def test_dexycb_with_both_hands_is_refused_naming_the_side_to_pick():
    """One hand per clip: sides='both' would skip every clip (and an import drops the clips it skips)."""
    with pytest.raises(pydantic.ValidationError, match=r"source\.name='dexycb' has one hand per clip.*--sides right"):
        FDRImportOptions.model_validate({"source": {"name": "dexycb"}, "raw": "r", "dataset": "d", "mano_dir": "m"})
    FDRImportOptions.model_validate(
        {"source": {"name": "dexycb"}, "raw": "r", "dataset": "d", "mano_dir": "m", "sides": "right"}
    )


def test_the_contact_pairs_take_the_overridden_contact_settings_not_their_own():
    """mjENBL_OVERRIDE replaces every contact's solref and friction, so per-pair settings would do nothing."""
    with pytest.raises(pydantic.ValidationError, match="pair_friction"):
        FDRKinematicRetargetingOptions.model_validate(
            {"dataset": "d", "retarget": {"scene": {"pair_friction": [1] * 5}}}
        )
