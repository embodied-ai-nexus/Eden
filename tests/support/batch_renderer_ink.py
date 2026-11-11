"""Paint the overlay tests' floor square on the batch renderer (Madrona), on CUDA, and measure it against the model.

Run as ``python -m tests.support.batch_renderer_ink`` from the repo root; prints one line, ``RESULT {...}``: the
renderer the camera used, how many floor pixels inside the square the camera sees, the darkest and brightest of them
(green channel, so the cube's shadow shows), and the largest difference between the painted pixels and the floor's
pixels decoded from sRGB, tinted by ``ink / floor`` and re-encoded, for a green ink and for a grey one (every channel
tinted). A process of its own because the test suite
initialises Genesis on ``gs.cpu``, where the batch renderer does not run, and Genesis cannot be re-initialised
in-process.

The scene is the overlay tests' own, so the two cannot drift apart: it reads ``_config``, ``_shadowing_cube``,
``LOOK``, ``SQUARE``, ``INK``, ``INK_TINT``, ``MID_INK`` and ``MID_TINT`` from ``tests/entities/test_camera_overlays.py``
(and registers that module's test metric by importing it). Renaming one of those breaks this script, which shows
only on a CUDA machine.
"""

from __future__ import annotations

import json


def _srgb_tinted(pixels, tint):
    """``pixels`` (uint8) decoded from sRGB, multiplied by ``tint`` in linear light, and re-encoded, in [0, 255]."""
    import numpy as np

    decoded = pixels / 255.0
    linear = np.where(decoded <= 0.04045, decoded / 12.92, ((decoded + 0.055) / 1.055) ** 2.4) * tint
    encoded = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.maximum(linear, 0.0031308) ** (1 / 2.4) - 0.055)
    return np.round(encoded * 255.0)


def _first_image(frame):
    """Env 0's image of a frame, as numpy: the batch renderer returns a torch batch, one image per env."""
    import numpy as np

    frame = frame.cpu().numpy() if hasattr(frame, "cpu") else np.asarray(frame)
    return frame[0] if frame.ndim in (3, 4) and frame.shape[0] == 1 else frame


def main() -> None:
    import eden as en  # before genesis and torch (DDP import order)
    import genesis as gs

    en.init(backend=gs.gpu, logging_level="WARNING")
    import numpy as np

    from eden.options.camera import CameraOptions
    from eden.options.renderer import BatchRendererOptions
    from tests.entities import test_camera_overlays as overlays
    from tests.support.envs import build_env

    light = {"pos": (0.0, 0.0, 3.0), "dir": (-1.0, -1.0, -1.0), "directional": True, "castshadow": True}
    config = overlays._config(
        cube_pos=overlays._shadowing_cube(light["dir"]),
        plain=CameraOptions(**overlays.LOOK),
        # Not _paint_square(): that camera is a debug one, and a debug camera rasterizes even in a batch-rendered scene.
        square=CameraOptions(
            **overlays.LOOK, overlays=[overlays.PlanePaintOverlay.configure(metric_name="square", color=overlays.INK)]
        ),
        mid=CameraOptions(
            **overlays.LOOK,
            overlays=[overlays.PlanePaintOverlay.configure(metric_name="square", color=overlays.MID_INK)],
        ),
    )
    config.renderer_options = BatchRendererOptions(lights=[light])
    env = build_env(config)
    plain, painted, mid = (_first_image(env.cameras[name].render_rgb())[..., :3] for name in ("plain", "square", "mid"))
    points, valid = (_first_image(x) for x in env.cameras["square"].render_pointcloud(world_frame=True))
    inside = valid & (np.abs(points[..., 2]) < 2e-3) & (np.abs(points[..., :2]).max(axis=-1) < overlays.SQUARE - 0.03)
    floor = plain[inside].astype(float)
    result = {
        "renderer": env.cameras["square"].renderer_kind,
        "inside": int(inside.sum()),
        "floor_green": [float(plain[inside][:, 1].min()), float(plain[inside][:, 1].max())],
        "max_error": float(np.abs(painted[inside] - _srgb_tinted(floor, overlays.INK_TINT)).max()),
        "max_error_mid": float(np.abs(mid[inside] - _srgb_tinted(floor, overlays.MID_TINT)).max()),
    }
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
