"""Row of spheres, each a different Eden surface option, shaded by Nyx.

---
tier: recipe
capabilities: [render.nyx]
hardware: gpu
optional_deps: [nyx]
run: uv run python examples/recipes/rendering/nyx_materials.py
---

Eden-native port of the genesis-nyx ``03_materials.py`` demo — the declarative
counterpart to ``camera_as_sensor_with_nyx.py``. Instead of raw ``gs.Scene`` +
``scene.add_sensor`` + ``gs.surfaces.*``, it builds an ``EnvBase`` from options:

* materials are ``eden.options.surfaces`` wrappers (``PlasticSurfaceOptions``,
  ``MetalSurfaceOptions``, ``GlassSurfaceOptions``, ``EmissionSurfaceOptions``)
  attached to ``SphereOptions.surface``;
* the Nyx camera is a declarative ``CameraOptions`` whose ``sensor_options`` is a
  ``NyxCameraOptions`` — the camera's pose / fov / resolution are read from the
  ``cam_*`` fields, so only Nyx-specific settings (lights, spp) live on the sensor.

Self-contained: primitive spheres + explicit lights, no HDRI/GLB assets. Renders a
single frame via ``env.cameras["cam"].read()`` and writes ``nyx_materials.png``.

Usage:
    uv run python examples/recipes/rendering/nyx_materials.py
"""

import matplotlib.pyplot as plt

import genesis as gs

import eden as en
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.camera import CameraOptions, CamerasOptions
from eden.options.entities import PlaneOptions, SphereOptions
from eden.options.surfaces import (
    BSDFSurfaceOptions,
    EmissionSurfaceOptions,
    GlassSurfaceOptions,
    MetalSurfaceOptions,
    PlasticSurfaceOptions,
)

RADIUS = 0.4
SPACING = 1.1

# A representative spread of Eden surface wrappers. Nyx has no transparency, so the
# glass ball reads as a tinted opaque dielectric — kept anyway to show the option.
MATERIALS = [
    PlasticSurfaceOptions(color=(0.85, 0.20, 0.20)),
    BSDFSurfaceOptions(color=(0.20, 0.40, 0.85), metallic=0.0, roughness=0.25),
    MetalSurfaceOptions(color=(1.00, 0.84, 0.40)),  # gold
    MetalSurfaceOptions(color=(0.95, 0.60, 0.42)),  # copper
    GlassSurfaceOptions(color=(0.70, 0.95, 1.00)),
    EmissionSurfaceOptions(emissive=(4.0, 1.5, 0.4)),
]


def main():
    en.init(backend=gs.gpu)

    from genesis.utils.misc import tensor_to_array
    from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions

    # One fixed sphere per material, laid out in a centered row.
    x0 = -SPACING * (len(MATERIALS) - 1) * 0.5
    spheres = {
        f"sphere_{i}": SphereOptions(
            radius=RADIUS,
            fixed=True,
            default_root_pos=(x0 + i * SPACING, 0.0, RADIUS),
            surface=surface,
        )
        for i, surface in enumerate(MATERIALS)
    }

    env = EnvBase(
        env_options=EnvOptions(num_envs=1),
        scene_options=SceneOptions(plane=PlaneOptions(), **spheres),
        cameras_options=CamerasOptions(
            cam=CameraOptions(
                cam_pos=(0.0, -4.5, 1.5),
                cam_lookat=(0.0, 0.0, RADIUS),
                cam_resolution=(1600, 600),
                cam_fov=35.0,
                # Nyx sensor backend: pose/fov/res are auto-filled from the cam_* fields
                # above, so only the Nyx-specific lighting/quality settings go here.
                sensor_options=NyxCameraOptions(
                    spp=64,
                    denoise=True,
                    lights=[
                        {"type": "directional", "dir": (-0.3, 0.6, -1.0), "color": (1.0, 1.0, 1.0), "intensity": 3.0},
                        {"type": "point", "pos": (-3.0, -3.0, 3.0), "color": (1.0, 1.0, 1.0), "intensity": 40.0},
                        {"type": "point", "pos": (3.0, -3.0, 3.0), "color": (1.0, 1.0, 1.0), "intensity": 40.0},
                    ],
                ),
            ),
        ),
    )
    env.build()

    env.scene.step()
    rgb = env.cameras["cam"].read().rgb
    print(f"RGB shape: {rgb.shape}")

    plt.imsave("nyx_materials.png", tensor_to_array(rgb[0]))
    print("Saved to nyx_materials.png")


if __name__ == "__main__":
    main()
