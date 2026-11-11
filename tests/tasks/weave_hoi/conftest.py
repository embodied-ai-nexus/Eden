"""The ``appolyn/Weave`` snapshot the task tests build against, taken from the local HuggingFace cache only."""

from __future__ import annotations

import os

import pytest

#: The objects the task tests build with: two, so the heterogeneous path and the per-object groups are exercised.
TEST_OBJECTS = ("smallbox", "largebox")


@pytest.fixture(scope="session")
def weave_root() -> str:
    """Root of a cached ``appolyn/Weave`` snapshot holding the test split and geometry of :data:`TEST_OBJECTS`.

    Skips rather than downloads: the task tests need ~0.4 GB of reference data, and a test run must not reach the
    network. ``eden smoke --task weave_hoi --objects smallbox,largebox`` fills the cache.
    """
    try:
        from huggingface_hub import snapshot_download

        root = snapshot_download(
            repo_type="dataset",
            repo_id="appolyn/Weave",
            allow_patterns=[f"reference/test/{o}_*" for o in TEST_OBJECTS]
            + [f"objects/{o}/*" for o in TEST_OBJECTS]
            + ["objects/geometry/*"],
            local_files_only=True,
        )
    except Exception as exc:  # no cache entry at all
        pytest.skip(f"appolyn/Weave is not in the local HuggingFace cache ({type(exc).__name__})")
    needed = [f"objects/{o}/sdf_128.npz" for o in TEST_OBJECTS] + ["objects/geometry/bps_128.npy"]
    if not all(os.path.isfile(os.path.join(root, p)) for p in needed):
        pytest.skip(f"the cached appolyn/Weave snapshot lacks {needed}")
    return root


def build_weave_env(root: str, objects: str, num_envs: int):
    """Build and reset the registered ``weave_hoi`` task on ``objects`` from the cached snapshot."""
    from eden.tasks import TASK_REGISTRY
    from tests.support.envs import build_env

    cfg = TASK_REGISTRY.build("weave_hoi", objects=objects, raw_data_path=root)
    cfg.env_options.num_envs = num_envs
    return build_env(cfg, reset=True)
