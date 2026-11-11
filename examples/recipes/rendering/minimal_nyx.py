"""Minimal Nyx camera sensor example (using primitives, no Eden RL stack).

---
tier: recipe
capabilities: [render.nyx, sensor.camera]
hardware: gpu
optional_deps: [nyx]
run: uv run python examples/recipes/rendering/minimal_nyx.py
---

Counterpart to ``minimal_apollo.py`` from the legacy apollo examples.
Renders a single sphere with the Nyx path tracer and writes ``nyx_output.png``.

Usage:
    uv run python examples/recipes/rendering/minimal_nyx.py
"""

import matplotlib.pyplot as plt

import genesis as gs


def main():
    gs.init(seed=0, precision="32", backend=gs.gpu, logging_level="warning")

    # Nyx auto-registers ``startup``/``shutdown`` with Genesis on import via
    # ``gs.register_external_module(...)`` inside ``nyx_camera_options``, so we
    # only need to import the options class -- no manual startup/shutdown calls.
    from genesis.utils.misc import tensor_to_array
    from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions

    scene = gs.Scene(
        renderer=gs.renderers.Rasterizer(),
        show_viewer=False,
    )

    scene.add_entity(
        gs.morphs.Sphere(
            pos=(0.0, 0.0, 0.0),
            radius=1.0,
        ),
        surface=gs.surfaces.Rough(
            color=(0.9, 0.3, 0.6),
        ),
    )

    cam = scene.add_sensor(
        NyxCameraOptions(
            res=(640, 480),
            pos=(5.0, 5.0, 3.0),
            lookat=(0.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
            fov=60.0,
            spp=16,
            denoise=True,
            # Nyx requires at least one light or env map to illuminate the scene.
            lights=[
                {
                    "type": "point",
                    "pos": (5.0, 5.0, 5.0),
                    "color": (1.0, 1.0, 1.0),
                    "intensity": 50.0,
                },
            ],
        )
    )

    scene.build(n_envs=1)

    scene.step()
    data = cam.read()
    print(f"RGB shape: {data.rgb.shape}")

    plt.imsave("nyx_output.png", tensor_to_array(data.rgb[0]))
    print("Saved to nyx_output.png")


if __name__ == "__main__":
    main()
