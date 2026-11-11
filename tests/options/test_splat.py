"""``eden.options.splat``: contradictory or unsupported settings are rejected at construction."""

from __future__ import annotations

import pytest

from eden.options import SplatOptions
from eden.options.splat import NyxSplatOptions, PCDOptions


@pytest.mark.parametrize(
    "kwargs",
    [
        {"camera_mode": "single"},
        {"camera_mode": "batched", "per_env_gaussians": True},
        {"camera_mode": "auto", "per_env_gaussians": True},
    ],
)
def test_consistent_modes_are_accepted(kwargs):
    SplatOptions(**kwargs)


def test_single_view_cannot_render_per_env_gaussians():
    """One shared view cannot show per-env geometry; a config-level contradiction fails at validation, not build."""
    with pytest.raises(ValueError, match="per_env_gaussians"):
        SplatOptions(camera_mode="single", per_env_gaussians=True)


def test_nyx_options_reject_a_path_depth_no_renderer_can_apply():
    """gs-nyx-plugin has no path-depth setting, so a depth would be silently dropped; it fails at validation instead."""
    NyxSplatOptions()  # the default leaves the depth to the renderer
    with pytest.raises(ValueError, match="max_pt_depth=2"):
        NyxSplatOptions(max_pt_depth=2)


def test_a_config_saved_with_the_old_path_depth_default_says_how_to_load_it():
    """Every NyxSplatOptions saved before the plugin bump carries `max_pt_depth: 2`; the load names the key to delete."""
    from eden.utils.configs import to_options

    with pytest.raises(ValueError, match="delete that key"):
        to_options({"max_pt_depth": 2}, NyxSplatOptions)


def test_a_pcd_caches_under_its_assets_subdir_and_a_saved_config_keeps_local_dir_unset(user_assets_dir):
    """Pre-seeded: a cached .ply is returned without a Hub request."""
    cached = user_assets_dir / "polaris" / "scenes" / "bg.ply"
    cached.parent.mkdir(parents=True)
    cached.write_text("ply")

    pcd = PCDOptions(path="scenes/bg.ply", dataset="eden_polaris", assets_subdir="polaris")

    assert pcd.path == str(cached)
    assert pcd.model_dump()["local_dir"] is None


def test_an_explicit_pcd_local_dir_wins_over_its_assets_subdir(user_assets_dir, tmp_path):
    """Pre-seeded: a cached .ply is returned without a Hub request."""
    cached = tmp_path / "mine" / "scenes" / "bg.ply"
    cached.parent.mkdir(parents=True)
    cached.write_text("ply")

    pcd = PCDOptions(
        path="scenes/bg.ply", dataset="eden_polaris", assets_subdir="polaris", local_dir=str(tmp_path / "mine")
    )

    assert pcd.path == str(cached)
