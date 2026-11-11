"""Tests for ``eden.tasks.polaris.base_config``: the ``*_batched`` variants (``polaris_batch_splat``).

Skips without gsplat, which ``env_class_for_config`` imports to resolve a splat env.
"""

from __future__ import annotations

import pytest

pytest.importorskip("gsplat")

from eden.envs.base import env_class_for_config  # noqa: E402
from eden.envs.splat import SplatRLEnvBase  # noqa: E402
from eden.tasks import TASK_REGISTRY  # noqa: E402
from eden.utils.configs import EdenConfig  # noqa: E402


@pytest.mark.parametrize(
    "key",
    [
        "polaris_block_stack_kitchen",
        "polaris_food_bussing",
        "polaris_move_latte_cup",
        "polaris_organize_tools",
        "polaris_pan_clean",
        "polaris_tape_into_container",
    ],
)
def test_batched_task_is_a_splat_rl_env_reusing_the_single_env_splat(key):
    batched, single = TASK_REGISTRY.build(key + "_batched"), TASK_REGISTRY.build(key)
    assert env_class_for_config(batched) is SplatRLEnvBase
    assert batched.splat_options.pcds
    assert set(batched.splat_options.pcds) == set(single.splat_options.pcds)


@pytest.mark.parametrize("key", ["polaris_block_stack_kitchen", "polaris_block_stack_kitchen_batched"])
def test_the_splat_composite_survives_a_saved_config(key, tmp_path):
    """Both cameras replace the hidden plane with the splat, and ``config.json`` keeps that."""
    cfg = TASK_REGISTRY.build(key)
    cfg.save_as_json(str(tmp_path / "config.json"), lock_config=False)
    loaded = EdenConfig.load_from_file(str(tmp_path / "config.json"))
    for camera in ("external_camera", "wrist_camera"):
        (composite,) = getattr(cfg.cameras_options, camera).overlays
        (reloaded,) = getattr(loaded.cameras_options, camera).overlays
        assert (composite.name, composite.entities) == ("splat_composite", ["plane"])
        assert reloaded == composite
