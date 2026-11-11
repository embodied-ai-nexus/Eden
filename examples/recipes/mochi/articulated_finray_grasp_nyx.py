"""A Franka with soft FinRay fingers lifting a YCB object, path traced by Nyx from two camera shots.

---
tier: recipe
capabilities: [solver.mochi, render.nyx, robot.franka]
hardware: gpu-fp64
optional_deps: [nyx]
run: uv run python examples/recipes/mochi/articulated_finray_grasp_nyx.py
---

The rig, the solve and the control schedule are ``articulated_finray_grasp.py``'s — read that one
for what the Mochi solve is doing. Two things differ. The object is a YCB one rather than a 3 cm
cube — ``--object`` picks it, and each carries the grasp height and closing stroke tuned for its own
shape. That is what a FinRay pad is actually for: on the default apple, the pads cannot meet a round
surface flat, so every rib along them takes a different deflection and the two close on a patch a
rigid jaw could only touch at a point. And the recording camera is a Nyx path-tracer sensor rather
than the rasterizer, lit by a three-point rig over a constant environment map.

``--shot`` picks what that camera does. ``orbit`` arcs around the whole arm and pushes in on the
grasp. ``closeup`` instead hangs the camera off the gripper's own mount frame, so the gripper is
still in frame and the only thing moving is the pads folding around the object — the part of the run
that is a finite-element solve rather than a pair of jaws, and hard to read from any wider shot.

Nyx reaches Genesis one solver at a time and does not know about the Mochi solver, so a Mochi scene
would otherwise export at its rest pose and never move — see ``_nyx.py``, which bridges the two and
has to be installed before the scene is built.

Rendering cost is per frame, not per step: ``--spp`` trades noise for time, and the run renders one
frame per simulation step. Requires the LBM eval models
(https://huggingface.co/datasets/toyota-research-institute/lbm_eval_models); point ``--asset-dir``
at its ``robots/schunk_grippers`` directory or set ``$LBM_EVAL_MODELS``.

    uv run python examples/recipes/mochi/articulated_finray_grasp_nyx.py
    uv run python examples/recipes/mochi/articulated_finray_grasp_nyx.py --shot closeup
    uv run python examples/recipes/mochi/articulated_finray_grasp_nyx.py --object can --spp 256
"""

import argparse
from typing import NamedTuple

import genesis as gs
import genesis.utils.geom as gu
import numpy as np
import torch
from _finray import (
    DRIVE_DOF,
    GRIPPER_DOFS,
    add_asset_dir_arg,
    finger_color_texture,
    finger_mesh,
    finger_visual_mesh,
    mount_finger,
    resolve_urdf,
    trim_urdf,
)
from _mochi_common import MOCHI_GRAVITY, VideoRecorder, init, newton_iterations

import eden as en
from eden.engine.solvers.mochi.options import MochiOptions
from eden.envs.base import EnvBase
from eden.options import CameraOptions, CamerasOptions, EnvOptions, SceneOptions
from eden.options.entities import EntityOptions, PlaneOptions
from eden.options.materials import MochiElasticMaterialOptions, MochiRigidMaterialOptions
from eden.options.objects.ycb import YCB005TomatoSoupCan, YCB013Apple, YCB017Orange, YCBOptions

OBJECT_XY = (0.45, 0.0)


class Grasp(NamedTuple):
    """One object and the grasp tuned for it.

    A FinRay pad has no single right stroke: it meets a wide object sooner, so the same closing
    target that wraps one would crush -- or miss -- another. Heights follow from the mesh; the
    stroke is how far past first contact the pads are driven, which is what the ribs absorb.

    ``grasp_height`` also decides *where along the pad* the object lands. The pads hang ~12.5 cm
    below the mount frame and are most compliant at their tips, so raising it slides a short object
    towards those tips -- which folds them more, at the cost of less pad behind the contact.
    """

    options: type[YCBOptions]
    half_height: float  # mesh half-extent in z, so it rests on the ground with its centre here
    density: float  # kg/m3, near the real object where this gripper's preload carries it
    grasp_height: float  # mount frame height at the grasp
    lift_height: float  # mount frame height at the top of the lift
    close_stroke: float  # per-pad travel in from the open pose; the free gap closes by twice it


# Young's modulus of the pads, an order of magnitude below the 30 MPa the source example prints them
# in. That 30 MPa is about right for the real TPU, and at this scale it deflects too little to read
# on screen; at 3 MPa the ribs visibly fold, which is the whole reason to solve them as finite
# elements rather than as jaws. Deliberately soft, then, not measured -- and softer pads push back
# less for the same closing target, so the strokes below carry the extra travel that costs
# (~25 N settled on the apple, against ~69 N at the printed stiffness).
FINGER_E = 3e6

# A sphere is the shape that makes the point: the pads cannot meet it flat, so every rib along them
# takes a different deflection and the contact patch is one a rigid jaw could only touch at a point.
# The can is kept for the contrasting case, a straight side the pads flatten against.
OBJECTS = {
    "apple": Grasp(YCB013Apple, 0.0359, 800.0, 0.182, 0.422, 0.0300),
    "orange": Grasp(YCB017Orange, 0.0357, 700.0, 0.182, 0.422, 0.0305),
    # Water would make the can 370 g, more than this preload carries, so it keeps the source
    # example's light plastic.
    "can": Grasp(YCB005TomatoSoupCan, 0.0509, 300.0, 0.200, 0.440, 0.0280),
}
# Standoff of the gripper mount frame from the flange along the tool axis. The URDF carries the mount
# frame at the center of the Schunk body, so bolting the body onto the flange face takes half its
# depth of clearance.
MOUNT_OFFSET = 0.03625
# Roll of the gripper about the tool axis, turning the finger separation axis a quarter turn off the
# flange x.
MOUNT_EULER = (0.0, 0.0, 90.0)
# Elbow-up seed for the IK solves: the parsed model's neutral configuration is a singular straight-up
# posture outside the elbow joint limit.
SEED_QPOS = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)
# Tool orientation of both waypoints: the tool axis points down, the fingers separate along world y.
TOOL_QUAT = (0.0, 1.0, 0.0, 0.0)

# Camera orbit: a slow arc around the grasp that rises with the arm, so the pads are read from near
# their own height while they wrap and from above once the object is up. Azimuth is measured from
# world +x about it; the lookat climbs with the tool so the gripper stays on the same third of frame.
ORBIT_AXIS = OBJECT_XY
ORBIT_AZIMUTH = (-72.0, -22.0)  # degrees, start -> end
ORBIT_RADIUS = (1.15, 0.72)  # horizontal distance from the axis: a slow push in on the grasp
# Both ends keep the camera more than a half-FOV above the lookat, so the horizon stays out of
# frame and the dark floor, not a strip of sky, is what the arm is read against.
# Heights are offsets from the mount frame at the grasp and at the top of the lift, so the shot
# follows the object rather than having to be re-framed for each one.
ORBIT_CAMERA_RISE = 0.48  # camera height above those two
ORBIT_LOOKAT_RISE = (0.02, -0.06)  # lookat height relative to those two, start -> end
CAM_FOV = 32.0

# The close-up shot. The camera holds a fixed offset from the gripper's own mount frame instead of a
# fixed point in the world, so the gripper is motionless in frame and the only thing that moves is
# the pads collapsing around the object -- which is the whole point of simulating them as finite
# elements rather than as jaws. Heights are relative to that mount frame; the target sits on the
# contact band, a hand's width below it, and the azimuth drifts just enough to keep the shot alive.
# The pads separate along world y, so the shot looks down world x: both flank the object, and the bend
# each one takes is read in silhouette rather than hidden behind its own flat back.
CLOSEUP_AZIMUTH = (-34.0, -6.0)  # degrees, start -> end
CLOSEUP_RADIUS = 0.38  # horizontal distance from the mount frame
CLOSEUP_RISE = -0.01  # camera height relative to the mount frame
# Aimed a little above the held object's own centre, which is where the pads meet it; taking it from
# the grasp rather than fixing it keeps the framing right for a short object and a tall one alike.
CLOSEUP_TARGET_RISE = 0.035
CLOSEUP_FOV = 34.0

# A key / fill / rim rig over a constant ambient. Nyx is physically based and reads these in lumens,
# so the numbers are small: at these standoffs a two-figure intensity blows the frame to white, and
# the albedo goes with it — every surface renders as its highlight. The key is the warm spot that
# rakes across the pads and draws their contact shadow, the cool point fills the shadow side, and
# the third lifts the far edge of the arm off the background.
LIGHTS = [
    {
        "type": "spot",
        "pos": (1.1, -0.9, 1.6),
        "dir": (-0.5, 0.45, -1.0),
        "color": (1.0, 0.96, 0.90),
        "intensity": 12.0,
        "inner_angle": 25.0,
        "outer_angle": 55.0,
    },
    {"type": "point", "pos": (-0.6, -1.2, 0.9), "color": (0.62, 0.72, 1.0), "intensity": 3.0},
    {"type": "point", "pos": (0.2, 1.3, 1.1), "color": (1.0, 0.86, 0.72), "intensity": 4.0},
]
AMBIENT = 0.25  # constant environment map: the soft fill, and the background the scene sits on

# Both gltfs address their base-color map through `KHR_texture_basisu`, and neither Genesis nor Nyx
# reads that form: Genesis drops the texture and falls back to grey, Nyx renders the mesh black. The
# pads take the plain `.png` the gltf names as that texture's own fallback, which `_finray` hands
# back — a deformable renders as a dynamic mesh, which has no source file to take a material from at
# all, so its surface has to state one either way. The Schunk body's own map is one near-uniform light
# grey, so rather than restate that flatly it is given the anodised aluminium it is made of — a
# metal surface the exporter knows by name, at a roughness that reads as brushed rather than mirror.
GRIPPER_METAL = {"metal_type": "aluminium", "roughness": 0.22, "metallic": 0.45}


def orbit_pose(alpha: float, grasp: Grasp) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Camera position and lookat at ``alpha`` in [0, 1] along the orbit."""
    # Smoothstep, so the move eases out of the opening frame and settles into the closing one
    # instead of starting and stopping on a hard velocity step.
    eased = alpha * alpha * (3.0 - 2.0 * alpha)

    def lerp(pair):
        return pair[0] + eased * (pair[1] - pair[0])

    mounts = (grasp.grasp_height, grasp.lift_height)
    azimuth = np.radians(lerp(ORBIT_AZIMUTH))
    radius = lerp(ORBIT_RADIUS)
    pos = (
        ORBIT_AXIS[0] + radius * np.cos(azimuth),
        ORBIT_AXIS[1] + radius * np.sin(azimuth),
        lerp(tuple(m + ORBIT_CAMERA_RISE for m in mounts)),
    )
    lookat_z = lerp(tuple(m + rise for m, rise in zip(mounts, ORBIT_LOOKAT_RISE, strict=True)))
    return tuple(float(v) for v in pos), (*ORBIT_AXIS, float(lookat_z))


def closeup_pose(
    alpha: float, grasp: Grasp, mount_pos
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Camera position and lookat at ``alpha`` in [0, 1], held off the gripper's mount frame."""
    azimuth = np.radians(CLOSEUP_AZIMUTH[0] + alpha * (CLOSEUP_AZIMUTH[1] - CLOSEUP_AZIMUTH[0]))
    pos = (
        mount_pos[0] + CLOSEUP_RADIUS * np.cos(azimuth),
        mount_pos[1] + CLOSEUP_RADIUS * np.sin(azimuth),
        mount_pos[2] + CLOSEUP_RISE,
    )
    # The object hangs this far below the mount frame for the whole run, lift included.
    target_z = mount_pos[2] + grasp.half_height - grasp.grasp_height + CLOSEUP_TARGET_RISE
    return tuple(float(v) for v in pos), (float(mount_pos[0]), float(mount_pos[1]), float(target_z))


def shot_pose(shot: str, alpha: float, grasp: Grasp, mount_pos):
    """The camera pose for the requested shot at ``alpha`` in [0, 1] along the run."""
    return closeup_pose(alpha, grasp, mount_pos) if shot == "closeup" else orbit_pose(alpha, grasp)


def nyx_camera(shot: str, grasp: Grasp, resolution: tuple[int, int], spp: int) -> CamerasOptions:
    """The ``rec`` camera, backed by a Nyx path-tracer sensor framed on the start of the orbit."""
    import _nyx
    from gs_nyx import nyx_py_renderer as npr
    from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions

    from eden.envs.nyx_splat import write_nyx_camera_profile

    # The IK below puts the mount frame exactly here at the first frame, which is what the close-up
    # camera is hung off; the orbit ignores it.
    pos, lookat = shot_pose(shot, 0.0, grasp, (*OBJECT_XY, grasp.grasp_height))
    fov = CLOSEUP_FOV if shot == "closeup" else CAM_FOV
    return CamerasOptions(
        rec=CameraOptions(
            cam_pos=pos,
            cam_lookat=lookat,
            cam_resolution=resolution,
            cam_fov=fov,
            sensor_options=NyxCameraOptions(
                spp=spp,
                denoise=True,
                # The reference path tracer over the fast one: the scene is a handful of objects
                # under three lights, so the extra bounces cost little and are what carry the light
                # into the pads.
                render_mode=npr.ERenderMode.RefPathTracer,
                # gs-nyx-plugin >= 0.1.7 takes the tone mapper (and the optics) from a camera profile.
                camera_profile=write_nyx_camera_profile(fov, "ACES_1_3"),
                lights=LIGHTS,
                env_maps=(_nyx.constant_env_map(AMBIENT),),
            ),
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--steps", type=int, default=400, help="Number of simulation steps")
    parser.add_argument(
        "-o", "--output-dir", type=str, default="renders/mochi", help="Directory the recorded video is written to"
    )
    parser.add_argument(
        "--object",
        type=str,
        choices=tuple(OBJECTS),
        default="apple",
        help="Which YCB object the pads close on (each carries its own grasp height and stroke)",
    )
    parser.add_argument(
        "--shot",
        type=str,
        choices=("orbit", "closeup"),
        default="orbit",
        help="Camera: 'orbit' arcs around the whole arm, 'closeup' rides the gripper and watches the pads deform",
    )
    parser.add_argument("--spp", type=int, default=96, help="Nyx samples per pixel (noise vs. render time)")
    parser.add_argument("--res", type=int, nargs=2, default=(1280, 720), help="Render resolution, width height")
    add_asset_dir_arg(parser)
    args = parser.parse_args()
    if (urdf_path := resolve_urdf(args.asset_dir)) is None:
        return
    # No --cpu: Nyx renders on CUDA, so a CPU run has no outcome but a crash inside the first render.
    init(args)
    steps = 5 if en.is_smoke() else args.steps
    grasp = OBJECTS[args.object]

    import _nyx

    _nyx.install()

    rubber = MochiElasticMaterialOptions(E=FINGER_E, nu=0.4, rho=1200.0, friction=1.5, viscous_friction=1.0)
    fingers = {
        f"finger_{side}": EntityOptions(
            file=finger_mesh(args.asset_dir, side),
            material=rubber,
            surface=gs.surfaces.Default(
                diffuse_texture=gs.textures.ImageTexture(image_path=finger_color_texture(args.asset_dir))
            ),
        )
        for side in ("L", "R")
    }
    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=1.0 / 60.0,
            mochi_options=MochiOptions(gravity=MOCHI_GRAVITY, n_newton_iterations=8),
        ),
        scene_options=SceneOptions(
            plane=PlaneOptions(material=MochiRigidMaterialOptions()),
            franka=en.robots.FrankaEmikaPandaNoHand(
                material=MochiRigidMaterialOptions(friction=1.0, viscous_friction=1.0),
            ),
            # The mount rotation that turns the Schunk body along the tool axis is baked into the
            # URDF; the weld below stands it clear of the wrist and rolls it about that axis.
            gripper=EntityOptions(
                file=trim_urdf(urdf_path),
                is_articulated=True,
                is_actuated=True,
                is_fixed_base=True,
                dofs_name=GRIPPER_DOFS,
                batch_fixed_verts=True,  # a fixed-base child welded onto a moving parent link
                material=MochiRigidMaterialOptions(friction=1.5, viscous_friction=1.0),
                surface=gs.surfaces.Metal(**GRIPPER_METAL),
            ),
            **fingers,
            # The YCB mesh ships its own .obj/.mtl/.png, which Nyx loads for itself — no surface here.
            object=grasp.options(
                default_root_pos=(*OBJECT_XY, grasp.half_height),
                material=MochiRigidMaterialOptions(rho=grasp.density, friction=1.5, viscous_friction=1.0),
            ),
            attachments_dict={
                "gripper": (
                    "franka",
                    "attachment",
                    (0.0, 0.0, MOUNT_OFFSET),
                    gu.xyz_to_quat(np.array(MOUNT_EULER), rpy=True, degrees=True),
                )
            },
        ),
        cameras_options=nyx_camera(args.shot, grasp, tuple(args.res), args.spp),
        show_viewer=False,
    )

    verts_local: dict[str, np.ndarray] = {}

    def mount_fingers():
        gripper = env.entities["gripper"]
        for side in ("L", "R"):
            verts_local[side] = mount_finger(
                env,
                gripper,
                env.entities[f"finger_{side}"],
                f"finray_finger_{side}",
                finger_mesh(args.asset_dir, side),
                finger_visual_mesh(args.asset_dir, side),
            )

    env.register_pre_build_hook(mount_fingers)
    env.build()

    franka, gripper = env.entities["franka"], env.entities["gripper"]
    held = env.entities["object"]
    finger_l, finger_r = env.entities["finger_L"], env.entities["finger_R"]
    # The gripper body abuts the flange it is welded to, and the fingers overlap the gripper body
    # near their mounts and each other when closed: contact acts only against the object and the ground.
    gripper.set_contact_enabled(franka, False)
    finger_l.set_contact_enabled(finger_r, False)
    for finger in (finger_l, finger_r):
        finger.set_contact_enabled(gripper, False)

    # Both waypoints place the gripper mount frame, which the standoff carries ahead of the flange
    # the solve steers, so the heights keep meaning the height of the gripper itself.
    mount_point = (0.0, 0.0, MOUNT_OFFSET)
    qpos_grasp = franka.inverse_kinematics(
        "attachment",
        poss=(*OBJECT_XY, grasp.grasp_height),
        quats=TOOL_QUAT,
        local_points=mount_point,
        init_qpos=SEED_QPOS,
    )
    qpos_lift = franka.inverse_kinematics(
        "attachment",
        poss=(*OBJECT_XY, grasp.lift_height),
        quats=TOOL_QUAT,
        local_points=mount_point,
        init_qpos=qpos_grasp,
    )
    franka.set_qpos(qpos_grasp)
    # Carry the finger vertices along with the link they are tied to, so every attachment starts at
    # zero violation despite the arm having moved off the pose the fingers were mounted at.
    for side, finger in (("L", finger_l), ("R", finger_r)):
        link = gripper.get_link(f"finray_finger_{side}")
        link_pos = np.asarray(link.get_pos().cpu()).reshape(3)
        link_quat = np.asarray(link.get_quat().cpu()).reshape(4)
        finger.set_vertex_positions(gu.transform_by_trans_quat(verts_local[side], link_pos, link_quat))
    franka.control_dofs_pos(qpos_grasp)

    # Drive only the first prismatic joint; the URDF mimic coupling carries the second finger.
    _, drive = gripper.find_named_dofs_idx_local([DRIVE_DOF])
    gripper.set_dofs_kp(torch.tensor([2e4], device=env.device), dofs_idx_local=drive)
    gripper.set_dofs_kd(torch.tensor([5.0], device=env.device), dofs_idx_local=drive)
    gripper.control_dofs_pos(torch.zeros(1, 1, device=env.device), dofs_idx_local=drive)

    camera = env.cameras["rec"]
    mount = gripper.get_link("mount_base")
    name = f"articulated_finray_grasp_nyx_{args.object}_{args.shot}"
    recorder = VideoRecorder(env, args, name, fps=30.0)
    for i_step in range(steps):
        if 60 <= i_step < 200:
            # Close onto it over slightly more than two seconds, the pads wrapping the surface.
            stroke = grasp.close_stroke * (i_step - 59) / 140.0
            gripper.control_dofs_pos(torch.full((1, 1), stroke, device=env.device), dofs_idx_local=drive)
        elif 200 <= i_step < 320:
            # Interpolate the joint targets: a step target to the raised pose would jerk the object out
            # of the pads at more than a meter per second.
            alpha = (i_step - 199) / 120.0
            franka.control_dofs_pos(qpos_grasp + (qpos_lift - qpos_grasp) * alpha)
        env.scene.step()
        # The sensor renders the state of the step just taken, so the pose is set before the frame —
        # and for the close-up that means reading the mount frame the gripper has just moved to.
        pos, lookat = shot_pose(args.shot, i_step / max(steps - 1, 1), grasp, mount.get_pos()[0].tolist())
        camera.sensor.update_camera_pose(pos=pos, lookat=lookat)
        if i_step % 60 == 0:
            grip_force = float(np.linalg.norm(np.asarray(finger_l.get_vertex_contact_forces()[0].cpu()).sum(axis=0)))
            print(
                f"step {i_step:4d}: {args.object} z={float(held.get_pos()[0, 2]):.4f} "
                f"tool z={float(gripper.get_link('mount_base').get_pos()[0, 2]):.4f} "
                f"grip force={grip_force:6.3f} N newton iterations={newton_iterations(env)}"
            )
        recorder.frame()
    recorder.close()


if __name__ == "__main__":
    main()
