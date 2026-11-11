"""The RJ45 insertion of ``contacts_rj45_plug.py``, path traced by Nyx from a close or a wide shot.

---
tier: recipe
capabilities: [solver.mochi, render.nyx]
hardware: gpu-fp64
optional_deps: [nyx]
run: uv run python examples/recipes/mochi/contacts_rj45_plug_nyx.py
---

The scene, the solve and the insertion ramp are ``contacts_rj45_plug.py``'s -- read that one, and
``_rj45.py``, for what the Mochi solve is doing. What differs is the camera: a Nyx path-tracer sensor
instead of the rasterizer, over a three-point rig and a studio panorama, with materials stated rather than
left at flat vertex colour -- the floor included, which would otherwise carry Genesis' checker tiles. A connector is a millimetre-scale part in glossy plastic beside a
matte rubber cable, and that contrast is most of what a render of it is for.

``--shot`` picks the framing. ``wide`` holds the whole coil and the connector in one static frame -- the
cable is a third of a metre of rod and the reason to render this at all. ``latch`` rides in beside the plug
at the scale of the latch itself, where the tab is pressed into the socket's window and springs back behind
its lip; at 6 mm across that is invisible in any wider shot.

Nyx reaches Genesis one solver at a time and does not know about the Mochi solver, so a Mochi scene would
otherwise export at its rest pose and never move -- see ``_nyx.py``, which bridges the two and has to be
installed before the scene is built.

Rendering cost is per frame, not per step: ``--spp`` trades noise for time, and the run renders one frame
per simulation step.

    uv run python examples/recipes/mochi/contacts_rj45_plug_nyx.py
    uv run python examples/recipes/mochi/contacts_rj45_plug_nyx.py --shot latch --spp 256
"""

import argparse

import _rj45
import genesis as gs
from _mochi_common import VideoRecorder, init, newton_iterations

import eden as en
from eden.options import CameraOptions, CamerasOptions

# A key light high on the camera side, a cool fill opposite it and a warm rim behind. The whole scene is
# 0.15 m across, so the rig sits an order of magnitude closer than a robot-scale one and its intensities
# (lumens) come down with the square of that distance.
LIGHTS = [
    {
        # Wide enough that the edge of the cone falls outside the frame: at this distance a tighter
        # one draws its own terminator across the ground, which reads as a seam in the floor.
        "type": "spot",
        "pos": (0.16, -0.16, 0.20),
        "dir": (-0.6, 0.6, -1.0),
        "color": (1.0, 0.96, 0.90),
        "intensity": 0.40,
        "inner_angle": 55.0,
        "outer_angle": 88.0,
    },
    {"type": "point", "pos": (-0.12, -0.20, 0.10), "color": (0.62, 0.72, 1.0), "intensity": 0.07},
    {"type": "point", "pos": (0.05, 0.18, 0.12), "color": (1.0, 0.86, 0.72), "intensity": 0.09},
]

STUDIO = 0.7  # the panorama's softboxes are the fill here; at full strength they flatten the key light

# Moulded connector plastic against the cable's matte jacket: the same distinction the render is for.
SURFACES = {
    "connector": gs.surfaces.Plastic(color=(0.90, 0.90, 0.92), roughness=0.25),
    "socket": gs.surfaces.Plastic(color=(0.13, 0.32, 0.52), roughness=0.35),
    "cable": gs.surfaces.Plastic(color=(0.06, 0.06, 0.07), roughness=0.75),
    # A mid grey, light enough that the black cable reads against it and dark enough for the white plug.
    "plane": gs.surfaces.Plastic(color=(0.24, 0.24, 0.25), roughness=0.6),
}

WIDE_POS, WIDE_LOOKAT, WIDE_FOV = (0.118, -0.118, 0.076), (-0.015, -0.040, 0.007), 35.0
# The latch shot rides with the plug -- both of these are its pose at full insertion, and the whole rig
# slides back by however far the plug still has to travel. So the connector holds still in frame and the
# socket comes to meet it, which is the only way to watch a 6 mm tab at the scale it moves.
LATCH_POS, LATCH_LOOKAT, LATCH_FOV = (0.062, -0.046, 0.046), (-0.003, -0.016, 0.0155), 24.0
# The connector is 25 mm long and the camera sits 80 mm off it, well inside Genesis' default near plane.
CAM_NEAR, CAM_FAR = 2e-3, 2.0


def shot_pose(shot: str, insertion: float):
    """Camera position and lookat for the requested shot at this insertion depth.

    Parameters
    ----------
    shot : str
        ``wide`` (static) or ``latch`` (a dolly that rides with the plug).
    insertion : float
        How far the plug has travelled along the insertion axis (m).

    Returns
    -------
    tuple
        ``(pos, lookat)``, each a 3-tuple in metres.
    """
    if shot == "wide":
        return WIDE_POS, WIDE_LOOKAT
    lag = insertion + _rj45.PLUG_Y_OFFSET  # zero once seated, -25 mm at the start
    return (
        (LATCH_POS[0], LATCH_POS[1] + lag, LATCH_POS[2]),
        (LATCH_LOOKAT[0], LATCH_LOOKAT[1] + lag, LATCH_LOOKAT[2]),
    )


def nyx_camera(shot: str, resolution: tuple[int, int], spp: int, env_map) -> CamerasOptions:
    """The ``rec`` camera, backed by a Nyx path-tracer sensor framed on the start of the shot.

    Parameters
    ----------
    shot : str
        ``wide`` or ``latch``.
    resolution : tuple[int, int]
        Render resolution.
    spp : int
        Samples per pixel.
    env_map : gs_nyx.nyx_py_sdk.EnvironmentMapAsset
        The ambient light, and what shows in the plastic's reflections.

    Returns
    -------
    eden.options.CamerasOptions
        The single recording camera.
    """
    from gs_nyx import nyx_py_renderer as npr
    from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions

    from eden.envs.nyx_splat import write_nyx_camera_profile

    pos, lookat = shot_pose(shot, 0.0)
    fov = LATCH_FOV if shot == "latch" else WIDE_FOV
    return CamerasOptions(
        rec=CameraOptions(
            cam_pos=pos,
            cam_lookat=lookat,
            cam_resolution=resolution,
            cam_fov=fov,
            cam_near=CAM_NEAR,
            cam_far=CAM_FAR,
            sensor_options=NyxCameraOptions(
                spp=spp,
                denoise=True,
                # The reference path tracer over the fast one: a handful of objects under three lights, so
                # the extra bounces cost little and are what light the inside of the socket at all.
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
    parser.add_argument("-s", "--steps", type=int, default=260, help="Number of simulation steps")
    parser.add_argument(
        "-o", "--output-dir", type=str, default="renders/mochi", help="Directory the recorded video is written to"
    )
    parser.add_argument("--shot", choices=("wide", "latch"), default="wide", help="Camera framing")
    parser.add_argument("--spp", type=int, default=128, help="Nyx samples per pixel (noise vs. render time)")
    parser.add_argument("--res", type=int, nargs=2, default=(1920, 1080), help="Render resolution, width height")
    parser.add_argument("--no-cable", action="store_true", help="Insert the bare connector, without the cable")
    args = parser.parse_args()
    init(args)
    steps = 5 if en.is_smoke() else args.steps

    import _nyx

    _nyx.install()  # teach the plugin about the Mochi solver, before the scene is built

    env_map = _nyx.studio_env_map()
    env_map.multiplier = STUDIO

    env = _rj45.build(
        with_cable=not args.no_cable,
        surfaces=SURFACES,
        cameras_options=nyx_camera(args.shot, tuple(args.res), args.spp, env_map),
    )
    connector, camera = env.entities["connector"], env.cameras["rec"]

    recorder = VideoRecorder(env, args, f"contacts_rj45_plug_nyx_{args.shot}")
    for i_step in range(steps):
        qpos = connector.get_qpos()[0]
        pos, lookat = shot_pose(args.shot, float(qpos[1]))
        camera.sensor.update_camera_pose(pos=pos, lookat=lookat)
        _rj45.drive(env, i_step)
        env.scene.step()
        if i_step % 20 == 0 or i_step == steps - 1:
            qpos = connector.get_qpos()[0]
            cable_state = "" if args.no_cable else f" {_rj45.cable_metrics(env.entities['cable'])}"
            print(
                f"step {i_step:4d}: insertion={float(qpos[1]) * 1e3:6.3f} mm latch={float(qpos[3]):+.4f} rad"
                f"{cable_state} newton iterations={newton_iterations(env)}"
            )
        recorder.frame()
    recorder.close()


if __name__ == "__main__":
    main()
