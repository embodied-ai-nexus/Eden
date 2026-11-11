"""Mochi soft fingers on a rigid gripper: a Schunk WSG-50 with deformable FinRay fingers closing on a box.

---
tier: recipe
capabilities: [solver.mochi]
hardware: gpu-fp64
optional_deps: []
run: uv run python examples/recipes/mochi/finray_gripper.py -v
---

Eden port of the Genesis fork's ``examples/mochi/finray_gripper.py``. The gripper URDF is loaded as an
articulated rigid body with the finger geometry stripped; each finger is a tetrahedral mesh (the ``.vtk``
deformable meshes shipped with the model) whose base vertices are rigidly attached to the corresponding
prismatic finger link, and is rendered through the model's detailed ``.gltf`` visual mesh skinned by the
simulation tetrahedra. The prismatic drive, the mimic coupling, the finite elements, the attachments and the
contact all enter one implicit Newton solve, so closing the compliant fingers on the box makes them wrap
around it.

Mounting a soft body on a link is pre-build work (an attachment anchors each vertex where it stands when the
scene is built), so it runs from an ``env.register_pre_build_hook`` callback.

Requires the LBM eval models (https://huggingface.co/datasets/toyota-research-institute/lbm_eval_models);
point ``--asset-dir`` at its ``robots/schunk_grippers`` directory or set ``$LBM_EVAL_MODELS``.

    uv run python examples/recipes/mochi/finray_gripper.py -v
"""

import argparse

import genesis as gs
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

CAM_POS, CAM_LOOKAT = (0.45, -0.45, 0.32), (0.0, 0.0, 0.1)
BASE_POS = (0.0, 0.0, 0.16)
BOX_SIZE = 0.03
FLIP_X = (0.0, 1.0, 0.0, 0.0)  # 180 degrees about x, wxyz: the tool axis points down


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on the CPU backend instead of the GPU")
    parser.add_argument("-s", "--steps", type=int, default=800, help="Number of simulation steps")
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
            gripper=EntityOptions(
                file=trim_urdf(urdf_path),
                default_root_pos=BASE_POS,
                default_root_quat=FLIP_X,
                is_fixed_base=True,
                is_articulated=True,
                is_actuated=True,
                dofs_name=GRIPPER_DOFS,
                material=MochiRigidMaterialOptions(friction=1.5, viscous_friction=1.0),
            ),
            **fingers,
            box=BoxOptions(
                size=(BOX_SIZE,) * 3,
                default_root_pos=(0.0, 0.0, 0.5 * BOX_SIZE),
                material=MochiRigidMaterialOptions(rho=300.0, friction=1.5, viscous_friction=1.0),
                surface=gs.surfaces.Default(color=(0.3, 0.7, 0.9)),
            ),
        ),
        cameras_options=recording_cameras(args, pos=CAM_POS, lookat=CAM_LOOKAT),
        show_viewer=args.vis,
    )

    def mount_fingers():
        gripper = env.entities["gripper"]
        for side in ("L", "R"):
            mount_finger(
                env,
                gripper,
                env.entities[f"finger_{side}"],
                f"finray_finger_{side}",
                finger_mesh(args.asset_dir, side),
                finger_visual_mesh(args.asset_dir, side),
            )

    env.register_pre_build_hook(mount_fingers)
    env.build()

    gripper, box = env.entities["gripper"], env.entities["box"]
    finger_l, finger_r = env.entities["finger_L"], env.entities["finger_R"]
    # The fingers overlap the gripper body near their mounts, and each other when fully closed: contact acts
    # only against the box and the ground.
    finger_l.set_contact_enabled(finger_r, False)
    for finger in (finger_l, finger_r):
        finger.set_contact_enabled(gripper, False)

    # Drive only the first prismatic joint; the URDF mimic coupling carries the second finger.
    _, drive = gripper.find_named_dofs_idx_local([DRIVE_DOF])
    gripper.set_dofs_kp(torch.tensor([2e4], device=env.device), dofs_idx_local=drive)
    gripper.set_dofs_kd(torch.tensor([5.0], device=env.device), dofs_idx_local=drive)
    gripper.control_dofs_pos(torch.zeros(1, 1, device=env.device), dofs_idx_local=drive)

    recorder = VideoRecorder(env, args, "finray_gripper")
    for i_step in range(steps):
        if 60 <= i_step < 200:
            # Close onto the box: the commanded opening (2.4 cm) is narrower than the box, so the compliant
            # fingers deform around it.
            stroke = 0.025 * (i_step - 59) / 60.0
            gripper.control_dofs_pos(torch.full((1, 1), stroke, device=env.device), dofs_idx_local=drive)
        env.scene.step()
        if i_step % 60 == 0:
            grip_force = -float(np.asarray(finger_l.get_vertex_contact_forces()[0].cpu())[:, 0].sum())
            tip_gap = float(
                finger_r.get_vertex_positions()[0, :, 0].min() - finger_l.get_vertex_positions()[0, :, 0].max()
            )
            print(
                f"step {i_step:4d}: box z={float(box.get_pos()[0, 2]):.4f} finger gap={tip_gap:+.4f} "
                f"grip force={grip_force:+.3f} N newton iterations={newton_iterations(env)}"
            )
        recorder.frame()
    recorder.close()


if __name__ == "__main__":
    main()
