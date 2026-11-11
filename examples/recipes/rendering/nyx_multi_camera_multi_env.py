"""Two Nyx cameras over four parallel envs, one batched read, tiled into a grid PNG.

---
tier: recipe
capabilities: [render.nyx, sensor.camera]
hardware: gpu
optional_deps: [nyx]
run: uv run python examples/recipes/rendering/nyx_multi_camera_multi_env.py
---

Eden-native port of genesis-nyx ``07_multi_camera_multi_env`` — no external HDRI/GLB
assets. Builds an ``EnvBase`` with ``num_envs=4`` and two declarative Nyx cameras, then
teleports one primitive sphere to a different spot on the plane in each env.

* the **wide** camera is a static overhead view, the same world pose in every env;
* the **follow** camera is declared with ``attach_entity_name="rig"`` so it rides an
  invisible rig entity (``visualization=False``) whose position **and** orientation are
  set per env — each follow cell approaches the sphere from a different radial angle.

Because the scene is batched, a single ``env.cameras[name].read()`` per camera returns
a ``(N_ENVS, H, W, 3)`` tensor; the script tiles both batches into one PNG with rows =
cameras (overview on top, follow below) and columns = envs.

Usage:
    uv run python examples/recipes/rendering/nyx_multi_camera_multi_env.py
"""

import math

import numpy as np
import torch

import genesis as gs

import eden as en
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.camera import CameraOptions, CamerasOptions
from eden.options.entities import PlaneOptions, SphereOptions
from eden.options.surfaces import BSDFSurfaceOptions

N_ENVS = 4
CAM_RES = (480, 360)

# One sphere position per env, arranged on a circle of radius R so the overview shows
# the sphere at four clearly different spots. Z lifts it onto the plane.
R = 1.2
RADIUS = 0.3
SPHERE_POSITIONS = [
    [0.0, R, RADIUS],
    [R, 0.0, RADIUS],
    [0.0, -R, RADIUS],
    [-R, 0.0, RADIUS],
]

# Per-env rig orientation (Z-axis rotation, WXYZ quaternion). Rotates the rig so the
# follow camera's "behind" direction points outward from the plane centre — every env
# sees the sphere from a different radial angle.
S = math.sqrt(0.5)
RIG_OUTWARD_QUATS = [
    [0.0, 0.0, 0.0, 1.0],  # 180 deg about Z
    [S, 0.0, 0.0, -S],  # -90 deg about Z
    [1.0, 0.0, 0.0, 0.0],  #   0 deg about Z
    [S, 0.0, 0.0, S],  # +90 deg about Z
]


def main() -> None:
    en.init(backend=gs.gpu)

    from genesis.utils.geom import R_to_quat, pos_lookat_up_to_T
    from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions

    # Follow-camera pose in the rig's local frame: 1.1 m back along -Y and 0.45 m up,
    # aimed at the sphere. Eden's declarative attach takes an offset + quat, so decompose
    # the look-at transform into its translation and rotation.
    offset_T = pos_lookat_up_to_T(
        torch.tensor((0.0, -1.1, 0.45)),
        torch.tensor((0.0, 0.0, RADIUS)),
        torch.tensor((0.0, 0.0, 1.0)),
    )
    offset_T = np.asarray(offset_T.cpu() if torch.is_tensor(offset_T) else offset_T, dtype=np.float64)
    attach_offset = offset_T[:3, 3].tolist()
    attach_quat = np.asarray(R_to_quat(torch.tensor(offset_T[:3, :3])).cpu()).tolist()

    lights = [
        {"type": "point", "pos": (2.5, -2.5, 4.0), "color": (1.0, 0.97, 0.92), "intensity": 80.0},
        {"type": "point", "pos": (-2.5, 2.5, 3.0), "color": (0.75, 0.85, 1.0), "intensity": 45.0},
        {"type": "directional", "dir": (0.2, 0.3, -1.0), "color": (1.0, 1.0, 1.0), "intensity": 2.0},
    ]

    env = EnvBase(
        env_options=EnvOptions(num_envs=N_ENVS),
        scene_options=SceneOptions(
            plane=PlaneOptions(),
            # Polished metallic sphere so each follow camera's reflection reads as a
            # distinct slice. Free-base (not fixed) so per-env set_pos is allowed.
            sphere=SphereOptions(
                radius=RADIUS,
                fixed=False,
                default_root_pos=(0.0, 0.0, RADIUS),
                surface=BSDFSurfaceOptions(color=(0.15, 0.35, 0.85), metallic=1.0, roughness=0.15),
            ),
            # Invisible rig the follow camera attaches to; its per-env pose is what puts
            # the follow camera at a different world pose in every env.
            rig=SphereOptions(radius=0.01, fixed=False, visualization=False, default_root_pos=(0.0, 0.0, RADIUS)),
        ),
        cameras_options=CamerasOptions(
            # Wide overhead — same pose in every env; lights live here and are merged
            # scene-wide (shared by both cameras and every env).
            wide=CameraOptions(
                cam_pos=(0.5, -2.5, 4.0),
                cam_lookat=(0.0, 0.0, 0.3),
                cam_resolution=CAM_RES,
                cam_fov=42.0,
                sensor_options=NyxCameraOptions(spp=32, denoise=True, lights=lights),
            ),
            # Follow — attached to the rig; rotating the rig per env rotates this pose.
            follow=CameraOptions(
                cam_resolution=CAM_RES,
                cam_fov=45.0,
                attach_entity_name="rig",
                attach_link_name="sphere_baselink",
                attach_offset=attach_offset,
                attach_quat=attach_quat,
                sensor_options=NyxCameraOptions(spp=32, denoise=True),
            ),
        ),
    )
    env.build()

    # Per-env state: sphere position, rig position (follows the sphere), rig orientation
    # (points outward). All share the outer dimension N_ENVS.
    env.entities["sphere"].set_pos(torch.tensor(SPHERE_POSITIONS))
    env.entities["rig"].set_pos(torch.tensor(SPHERE_POSITIONS))
    env.entities["rig"].set_quat(torch.tensor(RIG_OUTWARD_QUATS))

    # One step drives every sensor; one read() per camera returns the full batch.
    env.scene.step()
    wide_batch = env.cameras["wide"].read().rgb.cpu().numpy()
    follow_batch = env.cameras["follow"].read().rgb.cpu().numpy()
    print(f"wide batch {wide_batch.shape}, follow batch {follow_batch.shape}")

    _save_grid(wide_batch, follow_batch)


def _save_grid(wide_batch, follow_batch) -> None:
    """Tile the two batched renders into one PNG (rows = cameras, cols = envs)."""
    try:
        from PIL import Image
    except ImportError:
        print("Pillow not installed; skipping tiled PNG")
        return

    W, H = CAM_RES
    grid = Image.new("RGB", (W * N_ENVS, H * 2))
    for env_idx in range(N_ENVS):
        grid.paste(Image.fromarray(wide_batch[env_idx]), (env_idx * W, 0))
        grid.paste(Image.fromarray(follow_batch[env_idx]), (env_idx * W, H))
    out = "nyx_multi_camera_multi_env.png"
    grid.save(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
