"""The Newton's cradle of ``newton_cradle.py``, path traced by Nyx: polished steel balls in a studio light.

---
tier: recipe
capabilities: [solver.mochi, render.nyx]
hardware: gpu-fp64
optional_deps: [nyx]
run: uv run python examples/recipes/mochi/newton_cradle_nyx.py
---

The pendulums, the contact and the integrator are ``newton_cradle.py``'s, built by its own ``build()``. Read
that file for what the Mochi solve is doing. What differs is how it looks. The camera is a Nyx path-tracer
sensor instead of the rasterizer, and the balls are steel instead of flat colour. Around the rail stands the
rest of a cradle: an A-frame of four uprights on a plinth, all visual only and out of the swing's way. The
uprights stand clear of the balls' plane, so a ball released at 45 degrees passes between them.

Steel is mostly what it reflects, and a constant environment map reflects one flat grey. So the environment
here is a studio panorama instead: a dark floor and backdrop with two softboxes overhead. Their highlights on
each ball, and the balls' images in one another, are what read as polished metal.

``--shot`` picks the framing. ``wide`` holds the whole cradle, both 45-degree swings included. ``impact``
sits low and close on the pack at the height of the balls, where the end balls swing out of frame and back.

Nyx reaches Genesis one solver at a time and does not know about the Mochi solver, so a Mochi scene would
otherwise export at its rest pose and never move. ``_nyx.py`` bridges the two and has to be installed before
the scene is built.

    uv run python examples/recipes/mochi/newton_cradle_nyx.py
    uv run python examples/recipes/mochi/newton_cradle_nyx.py --shot impact --spp 256
"""

import argparse
import math

import genesis as gs
import numpy as np
from _mochi_common import VideoRecorder, init
from newton_cradle import BALL_X, INITIAL_ANGLE, build

import eden as en
from eden.options import CameraOptions, CamerasOptions
from eden.options.entities import BoxOptions, PlaneOptions
from eden.options.materials import MochiRigidMaterialOptions

# The A-frame. The rail of ``newton_cradle.py`` hangs at z = 0, spans x = +-0.324 and carries every string at
# y = 0, so the uprights stand at y = +-FRAME_HALF_DEPTH: a ball 5 cm in radius clears the 8.5 cm between
# them. A crossbar at each end of the rail carries it across to the uprights.
FRAME_X = 0.339
FRAME_HALF_DEPTH = 0.10
BAR = 0.03
PLINTH_TOP = -1.13  # the balls hang to -1.05 at rest
PLINTH_SIZE = (0.92, 0.38, 0.04)

# Polished steel. A metalness of 1 renders a near-black ball against anything but a bright environment, so
# this keeps some diffuse lobe for the lights to land on. The reflections of the softboxes still read as
# chrome.
STEEL = {"metal_type": "aluminium", "color": (0.88, 0.89, 0.91), "roughness": 0.10, "metallic": 0.75}
WIRE = gs.surfaces.Plastic(color=(0.82, 0.80, 0.76), roughness=0.5)
FRAME = gs.surfaces.Metal(metal_type="aluminium", roughness=0.28, metallic=0.6, color=(0.62, 0.63, 0.65))
PLINTH = gs.surfaces.Plastic(color=(0.02, 0.02, 0.022), roughness=0.12)
FLOOR = gs.surfaces.Plastic(color=(0.16, 0.16, 0.17), roughness=0.7)

# A key spot high on the camera side, a cool fill opposite it and a warm rim behind, at the two-metre standoff
# of a scene this size. The intensities are lumens: a two-figure value here clips the frame to white.
LIGHTS = [
    {
        "type": "spot",
        "pos": (1.4, -2.0, 1.4),
        "dir": (-0.55, 0.8, -0.95),
        "color": (1.0, 0.96, 0.90),
        "intensity": 22.0,
        "inner_angle": 30.0,
        "outer_angle": 60.0,
    },
    {"type": "point", "pos": (-1.8, -1.6, 0.2), "color": (0.62, 0.72, 1.0), "intensity": 4.0},
    {"type": "point", "pos": (0.4, 1.6, 0.6), "color": (1.0, 0.86, 0.72), "intensity": 8.0},
]

SHOTS = {
    # pos, lookat, vertical fov (deg)
    "wide": ((0.95, -3.0, 0.55), (-0.05, 0.0, -0.58), 27.0),
    "impact": ((0.55, -1.25, -0.55), (0.0, 0.0, -0.98), 34.0),
}


def shade_pendulums(ball_surface, string_surface) -> None:
    """Shade the pendulums' sphere and string geoms with real surfaces, not the MJCF's flat ``rgba``.

    The plugin gives every MJCF visual geom that carries a vertex colour a bare albedo material, and an MJCF
    primitive always carries one, so the entity's own surface never reaches it. This sets the override
    per geom instead, by the geom's shape. Install it before the scene is built.

    Parameters
    ----------
    ball_surface : gs.surfaces.Surface
        Surface of the sphere geoms.
    string_surface : gs.surfaces.Surface
        Surface of the capsule (string) geoms.
    """
    from gs_nyx_plugin import nyx_scene_exporter

    exporter = nyx_scene_exporter.NyxSceneExporter
    stock = exporter._build_vgeom_instances

    def surface_for(vgeom):
        # A visual geom carries no primitive type, only its mesh: the string is the long thin one.
        extent = np.ptp(vgeom.vmesh.verts, axis=0)
        return string_surface if extent.max() > 3.0 * extent.min() else ball_surface

    def wrapper(self, entity, instance_idx):
        end = stock(self, entity, instance_idx)
        if isinstance(entity.morph, gs.morphs.MJCF) and entity.morph.file.endswith(
            tuple(f"ball_{i}.xml" for i in range(len(BALL_X)))
        ):
            for idx, vgeom in zip(range(instance_idx, end), entity.vgeoms, strict=True):
                instance = self._scene_asset.get_instance(idx)
                instance.matOverride = self._build_material_override(surface_for(vgeom), entity, vgeom.vmesh, True)
                instance.smooth = True
                self._scene_asset.set_instance(idx, instance)
        return end

    exporter._build_vgeom_instances = wrapper


def decor() -> dict:
    """The A-frame, plinth and floor: fixed, collision-free, and on a Mochi material like every body here."""

    def bar(size, pos, surface=FRAME):
        return BoxOptions(
            size=size,
            default_root_pos=pos,
            is_fixed_base=True,
            collision=False,
            material=MochiRigidMaterialOptions(),
            surface=surface,
        )

    height = 0.5 * BAR - PLINTH_TOP  # from the plinth top to the top of the rail
    parts = {}
    for side, x in (("l", -FRAME_X), ("r", FRAME_X)):
        parts[f"crossbar_{side}"] = bar((BAR, 2 * FRAME_HALF_DEPTH + BAR, BAR), (x, 0.0, 0.0))
        for end, y in (("f", -FRAME_HALF_DEPTH), ("b", FRAME_HALF_DEPTH)):
            parts[f"upright_{side}{end}"] = bar((BAR, BAR, height), (x, y, 0.5 * (PLINTH_TOP + 0.5 * BAR)))
    parts["plinth"] = bar(PLINTH_SIZE, (0.0, 0.0, PLINTH_TOP - 0.5 * PLINTH_SIZE[2]), PLINTH)
    parts["floor"] = PlaneOptions(
        default_root_pos=(0.0, 0.0, PLINTH_TOP - PLINTH_SIZE[2]),
        collision=False,
        material=MochiRigidMaterialOptions(),
        surface=FLOOR,
    )
    return parts


def nyx_camera(shot: str, resolution: tuple[int, int], spp: int, env_map) -> CamerasOptions:
    """The ``rec`` camera, backed by a Nyx path-tracer sensor.

    Parameters
    ----------
    shot : str
        ``wide`` or ``impact``.
    resolution : tuple[int, int]
        Render resolution.
    spp : int
        Samples per pixel.
    env_map : gs_nyx.nyx_py_sdk.EnvironmentMapAsset
        The studio environment.

    Returns
    -------
    eden.options.CamerasOptions
        The single recording camera.
    """
    from gs_nyx import nyx_py_renderer as npr
    from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions

    from eden.envs.nyx_splat import write_nyx_camera_profile

    pos, lookat, fov = SHOTS[shot]
    return CamerasOptions(
        rec=CameraOptions(
            cam_pos=pos,
            cam_lookat=lookat,
            cam_resolution=resolution,
            cam_fov=fov,
            sensor_options=NyxCameraOptions(
                spp=spp,
                denoise=True,
                # The reference path tracer: the balls reflect one another, and that takes the extra bounces.
                render_mode=npr.ERenderMode.RefPathTracer,
                # gs-nyx-plugin >= 0.1.7 takes the tone mapper (and the optics) from a camera profile.
                camera_profile=write_nyx_camera_profile(fov, "ACES_1_3"),
                lights=LIGHTS,
                env_maps=(env_map,),
            ),
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--steps", type=int, default=600, help="Number of simulation steps")
    parser.add_argument(
        "-o", "--output-dir", type=str, default="renders/mochi", help="Directory the recorded video is written to"
    )
    parser.add_argument("--shot", choices=tuple(SHOTS), default="wide", help="Camera framing")
    parser.add_argument("--spp", type=int, default=128, help="Nyx samples per pixel (noise vs. render time)")
    parser.add_argument("--res", type=int, nargs=2, default=(1920, 1080), help="Render resolution, width height")
    parser.add_argument("--substeps", type=int, default=50, help="Implicit substeps per 60 Hz frame")
    args = parser.parse_args()
    # No --cpu: Nyx renders on CUDA, so a CPU run has no outcome but a crash inside the first render.
    init(args)
    steps = 5 if en.is_smoke() else args.steps

    import _nyx

    _nyx.install()  # teach the plugin about the Mochi solver, before the scene is built
    shade_pendulums(gs.surfaces.Metal(**STEEL), WIRE)

    env_map = _nyx.studio_env_map()
    env = build(
        substeps=args.substeps,
        cameras_options=nyx_camera(args.shot, tuple(args.res), args.spp, env_map),
        frame_surface=FRAME,
        **decor(),
    )
    balls = [env.entities[f"ball_{i}"] for i in range(len(BALL_X))]
    balls[0].set_dofs_pos(np.array([[INITIAL_ANGLE]]))

    recorder = VideoRecorder(env, args, f"newton_cradle_nyx_{args.shot}", fps=60)
    peak = np.zeros(len(balls))
    for i_step in range(steps):
        env.scene.step()
        peak = np.maximum(peak, np.abs([float(ball.get_dofs_pos()[0, 0]) for ball in balls]))
        recorder.frame()
        if i_step % 60 == 0:
            print(f"step {i_step:4d}: peak swing angles [deg] {np.array2string(np.degrees(peak), precision=1)}")
    recorder.close()
    print(f"Last-ball transfer ratio: {np.degrees(peak[-1]) / math.degrees(INITIAL_ANGLE):.3f}")


if __name__ == "__main__":
    main()
