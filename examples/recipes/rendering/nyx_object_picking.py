"""Ray-pick primitive spheres through a Nyx camera and recover per-object world hits.

---
tier: recipe
capabilities: [render.nyx, sensor.camera]
hardware: gpu
optional_deps: [nyx]
run: uv run python examples/recipes/rendering/nyx_object_picking.py
---

Eden-native port of genesis-nyx ``06_object_picking`` — no external HDRI/GLB assets.
Builds an ``EnvBase`` with four ``SphereOptions`` balls (Eden surface options) and a
declarative Nyx ``CameraOptions``, then demonstrates the camera's ray API:
``env.cameras["cam"].pick_pixel(0, x, y)`` casts a ray through a pixel and returns the
first entity hit (or ``None`` for background) with its world-space hit ``position``.

The scene renders once, then a coarse grid of rays is cast via ``pick_pixel``. Each
hit is tied back to its ball by entity identity (``env.entities[name].entity``), and
the script prints per-ball hit counts plus the mean recovered world hit position —
the API's return value is the only data source. With Pillow available an overlay PNG
is written that dots every hit so the image reads like a low-resolution ID buffer.

Usage:
    uv run python examples/recipes/rendering/nyx_object_picking.py
"""

import genesis as gs

import eden as en
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.camera import CameraOptions, CamerasOptions
from eden.options.entities import PlaneOptions, SphereOptions
from eden.options.surfaces import MetalSurfaceOptions, PlasticSurfaceOptions

CAM_RES = (640, 480)
RADIUS = 0.4

# Four balls named by colour: ``name -> (SurfaceOptions, world_pos)``. The plane is
# rendered for shadows but kept out of this dict, so any ray that hits the ground
# falls into the "not in SPHERES" branch and is silently ignored.
SPHERES = {
    "gold": (MetalSurfaceOptions(color=(1.00, 0.84, 0.40)), (-0.9, 0.5, RADIUS)),
    "copper": (MetalSurfaceOptions(color=(0.95, 0.60, 0.42)), (0.0, 0.0, RADIUS)),
    "red": (PlasticSurfaceOptions(color=(0.85, 0.10, 0.10)), (0.9, 0.5, RADIUS)),
    "cyan": (PlasticSurfaceOptions(color=(0.10, 0.60, 0.65)), (0.4, -0.8, RADIUS)),
}

# Dot colour (RGBA) used for the optional overlay, one per ball.
DOT_RGBA = {
    "gold": (255, 200, 50, 200),
    "copper": (255, 120, 60, 200),
    "red": (255, 60, 60, 200),
    "cyan": (60, 220, 220, 200),
}


def main() -> None:
    en.init(backend=gs.gpu)

    from genesis.utils.misc import tensor_to_array
    from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions

    # Coarser grid under smoke so the example stays harness-runnable.
    grid_w, grid_h = (12, 9) if en.is_smoke() else (48, 36)

    spheres = {
        name: SphereOptions(radius=RADIUS, fixed=True, default_root_pos=pos, surface=surface)
        for name, (surface, pos) in SPHERES.items()
    }
    env = EnvBase(
        env_options=EnvOptions(num_envs=1),
        scene_options=SceneOptions(plane=PlaneOptions(), **spheres),
        cameras_options=CamerasOptions(
            cam=CameraOptions(
                cam_pos=(2.5, -3.5, 2.6),
                cam_lookat=(0.0, 0.0, 0.3),
                cam_resolution=CAM_RES,
                cam_fov=30.0,
                sensor_options=NyxCameraOptions(
                    spp=32,
                    denoise=True,
                    lights=[
                        {"type": "point", "pos": (3.0, -3.0, 4.0), "color": (1.0, 0.98, 0.95), "intensity": 60.0},
                        {"type": "point", "pos": (-3.0, 2.0, 3.0), "color": (0.7, 0.8, 1.0), "intensity": 30.0},
                    ],
                ),
            ),
        ),
    )
    env.build()
    env.scene.step()

    cam = env.cameras["cam"]
    rgb = tensor_to_array(cam.read().rgb[0])
    W, H = CAM_RES

    # Map each ball's underlying Genesis entity identity back to its colour name.
    # The plane is absent, so ground hits fall through and are ignored.
    name_by_eid = {id(env.entities[name].entity): name for name in SPHERES}

    # Per-ball accumulator: [hit count, sum world x, sum world y, sum world z].
    hits: dict[str, list[float]] = {n: [0.0, 0.0, 0.0, 0.0] for n in SPHERES}
    dots: list[tuple[int, int, str]] = []

    for gy in range(grid_h):
        py = int((gy + 0.5) * H / grid_h)
        for gx in range(grid_w):
            px = int((gx + 0.5) * W / grid_w)
            res = cam.pick_pixel(0, px, py)  # camera_index 0 for a single sensor
            if res is None:
                continue  # background (sky)
            name = name_by_eid.get(id(res.entity))
            if name is None:
                continue  # entity outside the picking set (plane)
            wx, wy, wz = res.position
            acc = hits[name]
            acc[0] += 1
            acc[1] += wx
            acc[2] += wy
            acc[3] += wz
            dots.append((px, py, name))

    print(f"Cast {grid_w}x{grid_h} rays through {CAM_RES[0]}x{CAM_RES[1]} camera")
    for name, (n, sx, sy, sz) in hits.items():
        if n:
            print(f"  {name:<6} {int(n):>4} hits   mean pos=({sx / n:+.2f}, {sy / n:+.2f}, {sz / n:+.2f})")
        else:
            print(f"  {name:<6}    0 hits")

    _save_overlay(rgb, dots, W, H, grid_w, grid_h)


def _save_overlay(rgb, dots, W, H, grid_w, grid_h) -> None:
    """Composite hit dots over the render and save a PNG. No-op if Pillow is missing."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("Pillow not installed; skipping overlay PNG")
        return

    base = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    dot_r = max(1, min(W // grid_w, H // grid_h) // 2 - 1)
    for px, py, name in dots:
        draw.ellipse((px - dot_r, py - dot_r, px + dot_r, py + dot_r), fill=DOT_RGBA[name])

    out = "nyx_object_picking.png"
    Image.alpha_composite(base, overlay).convert("RGB").save(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
