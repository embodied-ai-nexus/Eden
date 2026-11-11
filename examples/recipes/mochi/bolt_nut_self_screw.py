"""Mochi thread contact: a nut screws itself down a fixed bolt at plain 60 Hz steps.

---
tier: recipe
capabilities: [solver.mochi]
hardware: gpu-fp64
optional_deps: []
run: uv run python examples/recipes/mochi/bolt_nut_self_screw.py -v
---

Eden port of the Genesis fork's ``examples/mochi/bolt_nut_self_screw.py`` (the Mochi counterpart of
Genesis' ``examples/rigid/bolt_nut_self_screw.py``): the procedurally generated M24 x 3 bolt and nut shipped
with Genesis, the bolt fixed shaft up, a steady torque turning the free nut whose threads convert the rotation
into axial travel. The smooth penalty contact between the nut's boundary samples and the bolt's signed
distance field enters the Newton solve of every step, so the nut follows the 3 mm pitch with no substeps.
Thread contact wants parameters at the scale of the thread (0.3 mm radial clearance): the penalty threshold
and smoothing are shrunk to 0.1 mm / 0.2 mm and the bolt's SDF is sampled at 0.2 mm cells. The torque ramps
up only after the nut has settled into the threads and is released at the end of the travel, where the
self-locking thread holds the nut in place. ``--torque 1.0`` unscrews it back up.

    uv run python examples/recipes/mochi/bolt_nut_self_screw.py -v
    uv run python examples/recipes/mochi/bolt_nut_self_screw.py -v --torque 1.0
"""

import argparse
import os

import numpy as np
import torch
from _mochi_common import MOCHI_GRAVITY, VideoRecorder, init, recording_cameras

import eden as en
from eden.engine.solvers.mochi.options import MochiOptions
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import EntityOptions, PlaneOptions
from eden.options.materials import MochiRigidMaterialOptions
from eden.utils.misc import get_gs_assets_dir

CAM_POS, CAM_LOOKAT, CAM_FOV = (0.2, 0.1, 0.1), (0.0, 0.0, 0.03), 35.0
SETTLE_STEPS = 12
RAMP_STEPS = 30


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on the CPU backend instead of the GPU")
    parser.add_argument("-s", "--steps", type=int, default=400, help="Number of simulation steps")
    parser.add_argument("-r", "--record", action="store_true", help="Record a video of the run")
    parser.add_argument(
        "-o", "--output-dir", type=str, default="renders/mochi", help="Directory the recorded video is written to"
    )
    parser.add_argument("--torque", type=float, default=-1.0, help="Driving torque about z [N*m]; positive unscrews")
    args = parser.parse_args()
    # Verified stable over the full travel for magnitudes up to 4 N*m in either direction.
    if not (0.0 < abs(args.torque) <= 4.0):
        raise ValueError(f"--torque magnitude must be in (0, 4] N*m, got {args.torque}")
    init(args)
    steps = 5 if en.is_smoke() else args.steps

    # Contact at the scale of the thread: the penalty ramp must live inside the 0.3 mm clearance, and the bolt's
    # distance field must resolve the 1.6 mm flanks.
    thread_contact = {
        "friction": 0.2,
        "penalty_coefficient": 1e9,
        "penalty_threshold": 1e-4,
        "penalty_smoothing_half_distance": 2e-4,
    }
    meshes = os.path.join(get_gs_assets_dir(), "meshes", "bolt_nut")
    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=1.0 / 60.0,
            mochi_options=MochiOptions(gravity=MOCHI_GRAVITY, n_newton_iterations=8),
        ),
        scene_options=SceneOptions(
            plane=PlaneOptions(material=MochiRigidMaterialOptions()),
            # Fixed, hex head down on the ground plane, threaded shaft pointing up (head top z = 11 mm, shaft tip
            # z = 43 mm). The helix must survive loading untouched: no decimation, no convex decomposition.
            bolt=EntityOptions(
                file=os.path.join(meshes, "bolt.stl"),
                default_root_pos=(0.0, 0.0, 0.011),
                is_fixed_base=True,
                decimate=False,
                convexify=False,
                material=MochiRigidMaterialOptions(rho=7850.0, sdf_cell_size=2e-4, sdf_max_res=256, **thread_contact),
            ),
            # Pre-engaged near the top of the shaft. The bolt is fixed and so provides no contact samples: nothing
            # ever queries the nut's own collider (the nut meets the plane through its samples too).
            nut=EntityOptions(
                file=os.path.join(meshes, "nut.stl"),
                default_root_pos=(0.0, 0.0, 0.024),
                decimate=False,
                convexify=False,
                material=MochiRigidMaterialOptions(rho=7850.0, collider_type="none", **thread_contact),
            ),
        ),
        cameras_options=recording_cameras(args, pos=CAM_POS, lookat=CAM_LOOKAT, fov=CAM_FOV),
        show_viewer=args.vis,
    )
    env.build()
    nut = env.entities["nut"]
    # The free base joint's DOFs are (x, y, z, rx, ry, rz); the torque acts about z.
    spin_dof = nut.find_free_base_dofs_idx_local()[5:6]

    recorder = VideoRecorder(env, args, "bolt_nut_self_screw")
    # The travel limits below are heights of the nut's base, its authored origin: ``relative=True``. The default
    # solver frame sits at the centre of mass, 9 mm higher, so the screw-down limit would never be reached.
    z0 = float(nut.get_pos(relative=True)[0, 2])
    drive_on = True
    for i_step in range(steps):
        nut_z = float(nut.get_pos(relative=True)[0, 2])
        # Release the torque at the end of the travel and latch it off: screwing down, when the nut base reaches
        # the bolt head; unscrewing, when only the last turn of thread is still gripping.
        if (args.torque < 0.0 and nut_z < 0.0114) or (args.torque > 0.0 and nut_z > 0.038):
            drive_on = False
        ramp = 0.0 if i_step < SETTLE_STEPS else min((i_step - SETTLE_STEPS) / RAMP_STEPS, 1.0)
        torque = torch.tensor([[args.torque * ramp * drive_on]], device=env.device)
        nut.control_dofs_force(torque, dofs_idx_local=spin_dof)
        env.scene.step()
        if i_step % 40 == 0:
            wz = float(np.asarray(nut.entity.get_dofs_velocity(dofs_idx_local=spin_dof).cpu()).reshape(-1)[0])
            print(
                f"step {i_step:4d}: nut z = {nut_z * 1e3:6.2f} mm  travelled = {abs(nut_z - z0) * 1e3:5.2f} mm  "
                f"wz = {wz:6.2f} rad/s  drive {'on' if drive_on else 'off'}"
            )
        recorder.frame()
    recorder.close()


if __name__ == "__main__":
    main()
