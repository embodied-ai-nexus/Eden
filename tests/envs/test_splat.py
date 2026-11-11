"""``eden.envs.splat``: which splat camera each camera gets, and a renderer ready before the managers probe it.

Skips without gsplat or plyfile.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("gsplat")
plyfile = pytest.importorskip("plyfile")

from eden.entities.camera import BatchSplatCamera, SplatCamera  # noqa: E402
from eden.envs.base import env_class_for_config  # noqa: E402
from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY  # noqa: E402
from eden.options import BoxOptions, EnvOptions, SceneOptions, SplatOptions  # noqa: E402
from eden.options.camera import CameraOptions, CamerasOptions  # noqa: E402
from eden.options.managers.observations import ObservationGroupOptions, ObservationManagerOptions  # noqa: E402
from eden.options.splat import PCDOptions  # noqa: E402
from eden.terms import ObsTerm  # noqa: E402
from eden.utils.configs import EdenSplatConfig, EdenSplatRLConfig  # noqa: E402

FIXED = CameraOptions(cam_pos=(1.5, 0.0, 1.0), cam_lookat=(0.0, 0.0, 0.3), cam_resolution=(32, 24))
ATTACHED = CameraOptions(
    attach_entity_name="box",
    attach_link_name="box_baselink",
    attach_offset=(0.5, 0.0, 0.5),
    attach_quat=(1.0, 0.0, 0.0, 0.0),
    cam_resolution=(32, 24),
)


def _write_ply(path, n=64, seed=0):
    rng = np.random.default_rng(seed)
    fields = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2"]
    fields += [f"rot_{i}" for i in range(4)]
    arr = np.empty(n, dtype=[(f, "f4") for f in fields])
    for f in fields:
        arr[f] = rng.standard_normal(n).astype(np.float32)
    plyfile.PlyData([plyfile.PlyElement.describe(arr, "vertex")]).write(str(path))


@pytest.mark.parametrize("config_cls", [EdenSplatConfig, EdenSplatRLConfig], ids=["SplatEnvBase", "SplatRLEnvBase"])
@pytest.mark.parametrize(
    ("mode", "per_env", "fixed_cls", "attached_cls"),
    [
        ("auto", False, SplatCamera, BatchSplatCamera),  # a fixed camera over a shared splat is broadcast
        ("auto", True, BatchSplatCamera, BatchSplatCamera),  # per-env Gaussians force a view per env
        ("batched", False, BatchSplatCamera, BatchSplatCamera),
        ("single", False, SplatCamera, SplatCamera),  # forced, even for an attached camera
    ],
)
def test_each_camera_gets_the_splat_camera_its_mode_calls_for(config_cls, mode, per_env, fixed_cls, attached_cls):
    """Both env classes build splat cameras at all.

    The mixin's ``_setup_cameras`` must win the MRO over ``EnvBase``'s, or the env silently builds plain cameras
    that render no splat.
    """
    config = config_cls(
        env_options=EnvOptions(num_envs=2),
        scene_options=SceneOptions(box=BoxOptions(size=(0.2, 0.2, 0.2))),
        splat_options=SplatOptions(camera_mode=mode, per_env_gaussians=per_env),
        cameras_options=CamerasOptions(fixed=FIXED, attached=ATTACHED),
    )
    env = env_class_for_config(config).from_config(config, show_viewer=False)  # construction only; no build
    try:
        assert type(env.cameras["fixed"]) is fixed_cls
        assert type(env.cameras["attached"]) is attached_cls
    finally:
        env.destroy()


@OBSERVATION_TERM_REGISTRY.register()
def splat_renderer_probe(env) -> torch.Tensor:
    """Fail as a camera term would if the splat renderer does not exist yet when the term is computed."""
    if env._splat_renderer is None:
        raise RuntimeError("the splat renderer was not built before the observation manager probed its terms")
    return torch.zeros((env.num_envs, 1), device=env.device)


def test_the_renderer_exists_before_the_managers_probe_their_terms(tmp_path):
    """``ObservationManager`` sizes every term by computing it once at build, and a camera term renders the splat.

    So the renderer is built before the managers load. Built after them, or from a post-build hook (drained below
    ``_load_managers``), the probe meets a ``None`` renderer. A real camera term cannot stand in here: gsplat
    rasterizes on CUDA only and the suite runs Genesis on ``gs.cpu``.
    """
    _write_ply(tmp_path / "bg.ply")
    config = EdenSplatConfig(
        env_options=EnvOptions(num_envs=1),
        scene_options=SceneOptions(box=BoxOptions(size=(0.2, 0.2, 0.2), default_root_pos=(0.0, 0.0, 0.3))),
        splat_options=SplatOptions(pcds={"bg": PCDOptions(path=str(tmp_path / "bg.ply"))}),
        observation_options=ObservationManagerOptions(
            policy=ObservationGroupOptions(probe=ObsTerm.configure(func=splat_renderer_probe))
        ),
    )
    env = env_class_for_config(config).from_config(config, show_viewer=False)
    try:
        env.build()
    finally:
        env.destroy()
