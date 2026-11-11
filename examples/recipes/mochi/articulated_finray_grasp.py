"""Mochi arm + soft fingers: a Franka wearing the FinRay gripper picks a box off the ground and lifts it.

---
tier: recipe
capabilities: [solver.mochi, robot.franka]
hardware: gpu-fp64
optional_deps: []
run: uv run python examples/recipes/mochi/articulated_finray_grasp.py -v
---

Eden port of the Genesis fork's ``examples/mochi/articulated_finray_grasp.py``. The arm is loaded without
its hand (``FrankaEmikaPandaNoHand``) and the gripper is welded onto the flange through
``SceneOptions.attachments_dict`` -- in its offset form, which stands the Schunk body clear of the wrist
instead of burying it -- so the seven arm joints, the gripper prismatic drive and its mimic coupling form
one articulated body. Each FinRay finger is a tetrahedral deformable body whose base vertices
are tied to the prismatic finger link. The joint drives, the finite elements, the attachments and the contact
all enter one implicit Newton solve, so the compliant fingers wrap around the box and carry it as the arm lifts.

An attachment anchors each vertex in the frame of the link it coincides with when the scene is built, so the
fingers are mounted from a pre-build hook at the pose the parsed model gives their link. The arm then moves
to the grasp configuration and the finger vertices follow rigidly, which leaves every attachment at zero
violation.

Requires the LBM eval models (https://huggingface.co/datasets/toyota-research-institute/lbm_eval_models);
point ``--asset-dir`` at its ``robots/schunk_grippers`` directory or set ``$LBM_EVAL_MODELS``.

    uv run python examples/recipes/mochi/articulated_finray_grasp.py -v
"""

import argparse

import genesis as gs
import genesis.utils.geom as gu
import numpy as np
import torch
from _finray import (
    DRIVE_DOF,
    GRIPPER_DOFS,
    add_asset_dir_arg,
    finger_mesh,
    finger_visual_mesh,
    mount_finger,
    resolve_urdf,
    trim_urdf,
)
from _mochi_common import MOCHI_GRAVITY, VideoRecorder, init, newton_iterations, recording_cameras

import eden as en
from eden.engine.solvers.mochi.options import MochiOptions
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import BoxOptions, EntityOptions, PlaneOptions
from eden.options.materials import MochiElasticMaterialOptions, MochiRigidMaterialOptions

CAM_POS, CAM_LOOKAT = (1.1, -1.1, 0.8), (0.35, 0.0, 0.3)
BOX_SIZE, BOX_XY = 0.03, (0.45, 0.0)
# Standoff of the gripper mount frame from the flange along the tool axis. The URDF carries the mount frame at
# the center of the Schunk body, so bolting the body onto the flange face takes half its depth of clearance.
MOUNT_OFFSET = 0.03625
# Roll of the gripper about the tool axis, turning the finger separation axis a quarter turn off the flange x.
MOUNT_EULER = (0.0, 0.0, 90.0)
# Gripper mount frame height at the grasp: it puts the box between the finger tips, where the angled FinRay
# pads close tightest, and leaves the tips a few millimeters clear of the ground.
GRASP_HEIGHT, LIFT_HEIGHT = 0.16, 0.40
# The angled pads meet closer than the box is wide, so the box holds them open and they press back on it.
CLOSE_STROKE = 0.03
# Elbow-up seed for the IK solves: the parsed model's neutral configuration is a singular straight-up posture
# outside the elbow joint limit.
SEED_QPOS = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)
# Tool orientation of both waypoints: the tool axis points down, the fingers separate along world y.
TOOL_QUAT = (0.0, 1.0, 0.0, 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on the CPU backend instead of the GPU")
    parser.add_argument("-s", "--steps", type=int, default=400, help="Number of simulation steps")
    parser.add_argument("-r", "--record", action="store_true", help="Record a video of the run")
    parser.add_argument(
        "-o", "--output-dir", type=str, default="renders/mochi", help="Directory the recorded video is written to"
    )
    add_asset_dir_arg(parser)
    args = parser.parse_args()
    if (urdf_path := resolve_urdf(args.asset_dir)) is None:
        return
    init(args)
    steps = 5 if en.is_smoke() else args.steps

    rubber = MochiElasticMaterialOptions(E=3e7, nu=0.4, rho=1200.0, friction=1.5, viscous_friction=1.0)
    fingers = {
        f"finger_{side}": EntityOptions(
            file=finger_mesh(args.asset_dir, side),
            material=rubber,
            # No color: the skinned visual mesh keeps the base-color texture of the model's gltf.
            surface=gs.surfaces.Default(),
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
            # The mount rotation that turns the Schunk body along the tool axis is baked into the URDF; the
            # weld below stands it clear of the wrist and rolls it about that axis.
            gripper=EntityOptions(
                file=trim_urdf(urdf_path),
                is_articulated=True,
                is_actuated=True,
                is_fixed_base=True,
                dofs_name=GRIPPER_DOFS,
                batch_fixed_verts=True,  # a fixed-base child welded onto a moving parent link
                material=MochiRigidMaterialOptions(friction=1.5, viscous_friction=1.0),
            ),
            **fingers,
            box=BoxOptions(
                size=(BOX_SIZE,) * 3,
                default_root_pos=(*BOX_XY, 0.5 * BOX_SIZE),
                material=MochiRigidMaterialOptions(rho=300.0, friction=1.5, viscous_friction=1.0),
                surface=gs.surfaces.Default(color=(0.3, 0.7, 0.9)),
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
        cameras_options=recording_cameras(args, pos=CAM_POS, lookat=CAM_LOOKAT),
        show_viewer=args.vis,
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

    franka, gripper, box = env.entities["franka"], env.entities["gripper"], env.entities["box"]
    finger_l, finger_r = env.entities["finger_L"], env.entities["finger_R"]
    # The gripper body abuts the flange it is welded to, and the fingers overlap the gripper body near their
    # mounts and each other when closed: contact acts only against the box and the ground.
    gripper.set_contact_enabled(franka, False)
    finger_l.set_contact_enabled(finger_r, False)
    for finger in (finger_l, finger_r):
        finger.set_contact_enabled(gripper, False)

    # Both waypoints place the gripper mount frame, which the standoff carries ahead of the flange the solve
    # steers, so the heights keep meaning the height of the gripper itself.
    mount_point = (0.0, 0.0, MOUNT_OFFSET)
    qpos_grasp = franka.inverse_kinematics(
        "attachment", poss=(*BOX_XY, GRASP_HEIGHT), quats=TOOL_QUAT, local_points=mount_point, init_qpos=SEED_QPOS
    )
    qpos_lift = franka.inverse_kinematics(
        "attachment", poss=(*BOX_XY, LIFT_HEIGHT), quats=TOOL_QUAT, local_points=mount_point, init_qpos=qpos_grasp
    )
    franka.set_qpos(qpos_grasp)
    # Carry the finger vertices along with the link they are tied to, so every attachment starts at zero
    # violation despite the arm having moved off the pose the fingers were mounted at.
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

    recorder = VideoRecorder(env, args, "articulated_finray_grasp")
    for i_step in range(steps):
        if 60 <= i_step < 200:
            # Close onto the box over slightly more than two seconds, the fingers deforming around it.
            stroke = CLOSE_STROKE * (i_step - 59) / 140.0
            gripper.control_dofs_pos(torch.full((1, 1), stroke, device=env.device), dofs_idx_local=drive)
        elif 200 <= i_step < 320:
            # Interpolate the joint targets: a step target to the raised pose would jerk the box out of the
            # pads at more than a meter per second.
            alpha = (i_step - 199) / 120.0
            franka.control_dofs_pos(qpos_grasp + (qpos_lift - qpos_grasp) * alpha)
        env.scene.step()
        if i_step % 60 == 0:
            grip_force = float(np.linalg.norm(np.asarray(finger_l.get_vertex_contact_forces()[0].cpu()).sum(axis=0)))
            print(
                f"step {i_step:4d}: box z={float(box.get_pos()[0, 2]):.4f} "
                f"tool z={float(gripper.get_link('mount_base').get_pos()[0, 2]):.4f} "
                f"grip force={grip_force:6.3f} N newton iterations={newton_iterations(env)}"
            )
        recorder.frame()
    recorder.close()


if __name__ == "__main__":
    main()
