"""Point / directional / spot lights on primitive balls, shaded by Nyx.

---
tier: recipe
capabilities: [render.nyx]
hardware: gpu
optional_deps: [nyx]
run: uv run python examples/recipes/rendering/nyx_light_types.py
---

Eden-native port of the genesis-nyx ``04_light_types.py`` demo — the declarative
counterpart to ``camera_as_sensor_with_nyx.py``. Instead of raw ``gs.Scene`` +
``scene.add_sensor``, it builds an ``EnvBase`` from options: three ``SphereOptions``
balls (a shared ``MetalSurfaceOptions`` gold, whose sharp speculars read each light
clearly) and a declarative ``CameraOptions`` whose ``sensor_options`` is a
``NyxCameraOptions`` carrying the three lights.

It keeps the trio of light types the plugin exposes through ``NyxCameraOptions.lights``:

* a saturated **point** light hovering in front of the left ball,
* a soft green **directional** "sun" tilted from the front-top, and
* a blue **spot** cone aimed down-and-in at the right ball.

Self-contained: primitive spheres + the three explicit lights, no HDRI/GLB assets.
Renders a single frame via ``env.cameras["cam"].read()`` and writes
``nyx_light_types.png`` next to this script.

Usage:
    uv run python examples/recipes/rendering/nyx_light_types.py
"""

import matplotlib.pyplot as plt

import genesis as gs

import eden as en
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.camera import CameraOptions, CamerasOptions
from eden.options.entities import PlaneOptions, SphereOptions
from eden.options.surfaces import MetalSurfaceOptions

BALL_RADIUS = 0.15
BALL_LEFT = (-0.93, 0.0, BALL_RADIUS)
BALL_MID = (0.00, 0.0, BALL_RADIUS)
BALL_RIGHT = (0.93, 0.0, BALL_RADIUS)

# Point light: hovers just above and in front of the left ball (toward the camera so
# the red lands on the visible face), saturated red, short range so the falloff stays
# local and doesn't wash the middle ball.
POINT_LIGHT = {
    "type": "point",
    "pos": (-0.93, 0.0, 0.5),
    "color": (1.0, 0.15, 0.05),
    "intensity": 4.0,
    "range": 0.8,
}

# Directional light: a soft green "sun" tilted from the front-top. Reaches every ball
# equally; aimed at the middle ball so it dominates only there.
DIRECTIONAL_LIGHT = {
    "type": "directional",
    "dir": (0.0, 0.3, -0.95),
    "color": (0.25, 1.0, 0.35),
    "intensity": 2.0,
}

# Spot light: high and to the right, aimed down-and-in at the right ball. A narrow
# inner / outer cone produces a clear blue hotspot with a soft edge.
SPOT_LIGHT = {
    "type": "spot",
    "pos": (0.93, -0.6, 1.0),
    "dir": (0.0, 0.5, -0.85),
    "color": (0.15, 0.4, 1.0),
    "intensity": 15.0,
    "inner_angle": 10.0,
    "outer_angle": 20.0,
    "range": 3.0,
}


def main():
    en.init(backend=gs.gpu)

    from genesis.utils.misc import tensor_to_array
    from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions

    gold = MetalSurfaceOptions(color=(1.0, 0.84, 0.40))
    balls = {
        f"ball_{i}": SphereOptions(radius=BALL_RADIUS, fixed=True, default_root_pos=pos, surface=gold)
        for i, pos in enumerate((BALL_LEFT, BALL_MID, BALL_RIGHT))
    }

    env = EnvBase(
        env_options=EnvOptions(num_envs=1),
        scene_options=SceneOptions(plane=PlaneOptions(), **balls),
        cameras_options=CamerasOptions(
            cam=CameraOptions(
                cam_pos=(0.0, 4.5, 1.8),
                cam_lookat=(0.0, 0.0, 0.15),
                cam_up=(0.0, 0.0, 1.0),
                cam_resolution=(1920, 1080),
                cam_fov=21.0,
                # Nyx sensor backend: pose/fov/res are auto-filled from the cam_* fields
                # above; the three lights (no HDRI env map) do all the shading.
                sensor_options=NyxCameraOptions(
                    spp=64,
                    denoise=True,
                    lights=[POINT_LIGHT, DIRECTIONAL_LIGHT, SPOT_LIGHT],
                ),
            ),
        ),
    )
    env.build()

    env.scene.step()
    rgb = env.cameras["cam"].read().rgb
    print(f"RGB shape: {rgb.shape}")

    plt.imsave("nyx_light_types.png", tensor_to_array(rgb[0]))
    print("Saved to nyx_light_types.png")


if __name__ == "__main__":
    main()
