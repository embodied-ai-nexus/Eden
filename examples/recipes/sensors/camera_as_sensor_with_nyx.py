"""Camera-as-sensor example with Nyx + Rasterizer side-by-side.

---
tier: recipe
capabilities: [sensor.camera, render.nyx]
hardware: gpu
optional_deps: [nyx]
run: uv run python examples/recipes/sensors/camera_as_sensor_with_nyx.py
---

Counterpart to ``camera_as_sensor_with_apollo.py`` from the legacy apollo
examples. Demonstrates ``scene.add_sensor()`` with two backends (Rasterizer
and Nyx), free-floating cameras, and cameras attached to a moving entity via
``offset_T``.

Usage:
    uv run python examples/recipes/sensors/camera_as_sensor_with_nyx.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import genesis as gs


def main():
    gs.init(seed=0, precision="32", backend=gs.gpu, logging_level="info")

    from genesis.utils.misc import tensor_to_array
    from genesis.options.sensors import RasterizerCameraOptions

    # Nyx auto-registers startup/shutdown via ``gs.register_external_module``
    # the moment we import the options module.
    from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions
    from gs_nyx import nyx_py_renderer as npr

    if gs.backend != gs.cuda:
        print("Nyx requires a CUDA backend; aborting.")
        return

    # ------------------------------ Scene ------------------------------
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            enable_collision=True,
            gravity=(0.0, 0.0, -9.8),
        ),
        renderer=gs.renderers.Rasterizer(),
        show_viewer=False,
    )

    scene.add_entity(
        morph=gs.morphs.Plane(),
        surface=gs.surfaces.Rough(color=(0.4, 0.4, 0.4)),
    )

    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(pos=(0.0, 0.0, 2.0), radius=0.5),
        surface=gs.surfaces.Smooth(color=(1.0, 0.5, 0.5)),
    )

    scene.add_entity(
        morph=gs.morphs.Box(pos=(1.0, 1.0, 1.0), size=(0.3, 0.3, 0.3)),
        surface=gs.surfaces.Rough(color=(0.5, 1.0, 0.5)),
    )

    # ------------------------------ Cameras ----------------------------
    CAMERA_COMMON = dict(up=(0.0, 0.0, 1.0), near=0.1, far=100.0)

    # Pose 2 units along +X from the sphere, looking back at it (Ry(90°)).
    OFFSET_X = np.array(
        [
            [0, 0, 1, 2],
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    # Pose 2 units along +Y from the sphere, looking back at it (Rx(-90°)).
    OFFSET_Y = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 2],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    camera_specs = [
        {
            "name": "cam0",
            "pos": (3.0, 0.0, 2.0),
            "lookat": (0.0, 0.0, 1.0),
            "fov": 60.0,
            "attachment": None,
            "lights": [
                {
                    "type": "point",
                    "pos": (2.0, 2.0, 5.0),
                    "color": (1.0, 1.0, 1.0),
                    "intensity": 1.0,
                }
            ],
        },
        {
            "name": "cam1",
            "pos": (0.0, 3.0, 2.0),
            "lookat": (0.0, 0.0, 1.0),
            "fov": 60.0,
            "attachment": None,
            "lights": [],
        },
        {
            "name": "cam_attached",
            "pos": (0.0, 0.0, 0.0),
            "lookat": (0.0, 0.0, 0.0),
            "fov": 70.0,
            "attachment": {"link_idx_local": 0, "offset_T": OFFSET_X},
            "lights": [],
        },
        {
            "name": "cam_offset_T",
            "pos": (0.0, 0.0, 0.0),
            "lookat": (0.0, 0.0, 0.0),
            "fov": 70.0,
            "attachment": {"link_idx_local": 0, "offset_T": OFFSET_Y},
            "lights": [],
        },
    ]

    backends = [
        ("raster", RasterizerCameraOptions),
        ("nyx", NyxCameraOptions),
    ]

    cameras = {}
    for backend_name, options_cls in backends:
        print(f"\n=== {backend_name} cameras ===")
        for spec in camera_specs:
            kwargs = dict(
                res=(500, 600),
                pos=spec["pos"],
                lookat=spec["lookat"],
                up=CAMERA_COMMON["up"],
                fov=spec["fov"],
                lights=spec["lights"],
            )

            attachment = spec["attachment"]
            if attachment is not None:
                kwargs["entity_idx"] = sphere.idx
                kwargs["link_idx_local"] = attachment["link_idx_local"]
                kwargs["offset_T"] = attachment["offset_T"]

            if backend_name == "raster":
                kwargs["near"] = CAMERA_COMMON["near"]
                kwargs["far"] = CAMERA_COMMON["far"]
            elif backend_name == "nyx":
                kwargs["near"] = CAMERA_COMMON["near"]
                kwargs["far"] = CAMERA_COMMON["far"]
                kwargs["spp"] = 32
                kwargs["denoise"] = True
                kwargs["render_mode"] = npr.ERenderMode.FastPathTracer
                # Nyx uses physically-based lighting in lumens — boost
                # rasterizer-tuned intensities so the scene isn't pitch-black.
                if spec["lights"]:
                    kwargs["lights"] = [
                        {**light, "intensity": light.get("intensity", 1.0) * 50} for light in spec["lights"]
                    ]

            name = f"{backend_name}_{spec['name']}"
            cameras[name] = scene.add_sensor(options_cls(**kwargs))
            print(f"  added {name}")

    # ------------------------------ Build ------------------------------
    n_envs = 1
    scene.build(n_envs=n_envs)

    # ----------------------------- Simulate ----------------------------
    out_dir = "camera_sensor_output"
    os.makedirs(out_dir, exist_ok=True)

    for step in range(100):
        scene.step()
        if step % 10 != 0:
            continue
        print(f"\n--- step {step} ---")
        for cam_name, camera in cameras.items():
            data = camera.read()
            print(f"  {cam_name} rgb shape: {data.rgb.shape}")
            rgb = data.rgb[0] if data.rgb.ndim > 3 else data.rgb
            suffix = "_env0" if n_envs > 1 else ""
            plt.imsave(
                os.path.join(out_dir, f"{cam_name}{suffix}_step{step:03d}.png"),
                tensor_to_array(rgb),
            )

    print(f"\nSaved images to {out_dir}/")


if __name__ == "__main__":
    main()
