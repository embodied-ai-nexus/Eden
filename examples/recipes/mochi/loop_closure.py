"""Mochi loop closure: a four-bar linkage closed by a connect equality, next to two welded boxes.

---
tier: recipe
capabilities: [solver.mochi]
hardware: gpu-fp64
optional_deps: []
run: uv run python examples/recipes/mochi/loop_closure.py -v
---

Eden port of the Genesis fork's ``examples/mochi/loop_closure.py``. Equality constraints are enforced as
stiff penalties inside the implicit Newton solve (``MochiOptions.equality_stiffness``), so the closing joint
of the linkage holds within a fraction of a millimeter while a constant crank torque turns the mechanism.
The two MJCF models are authored inline and written to a temporary directory that lives until the scene is
built. The closing-joint gap is read through the Genesis entity's ``equalities`` (a documented reach-through:
Eden does not wrap equality descriptors).

    uv run python examples/recipes/mochi/loop_closure.py -v
"""

import argparse
import tempfile

import genesis as gs
import numpy as np
import torch
from _mochi_common import (
    MOCHI_GRAVITY,
    VideoRecorder,
    init,
    recording_cameras,
    write_text,
)

import eden as en
from eden.engine.solvers.mochi.options import MochiOptions
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import EntityOptions, PlaneOptions
from eden.options.materials import MochiRigidMaterialOptions

CAM_POS, CAM_LOOKAT = (1.5, -2.5, 1.5), (0.2, 0.3, 0.7)

FOUR_BAR = """<mujoco>
  <worldbody>
    <body name="crank" pos="0 0 1">
      <joint name="crank_hinge" type="hinge" axis="0 1 0" damping="0.05"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.012" mass="0.2"/>
      <body name="coupler" pos="0 0 0.1">
        <joint name="coupler_hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.4 0 0.1" size="0.01" mass="0.4"/>
      </body>
    </body>
    <body name="rocker" pos="0.4 0 1">
      <joint name="rocker_hinge" type="hinge" axis="0 1 0" damping="0.05"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.012" mass="0.3"/>
    </body>
  </worldbody>
  <equality>
    <connect body1="coupler" body2="rocker" anchor="0.4 0 0.1"/>
  </equality>
</mujoco>
"""

# Authored relative to box ``a``, which ``WELDED_POS`` places: Eden writes ``default_root_pos`` over the root body's
# pose after build, while the morph position shifts every body, so an absolute pose here would part the two boxes.
WELDED_POS = (0.0, 0.6, 0.5)
WELDED_BOXES = """<mujoco>
  <worldbody>
    <body name="a" pos="0 0 0"><freejoint/><geom type="box" size="0.1 0.05 0.05" mass="1"/></body>
    <body name="b" pos="0.3 0 0" euler="0 30 0"><freejoint/><geom type="box" size="0.1 0.05 0.05" mass="1"/></body>
  </worldbody>
  <equality>
    <weld body1="a" body2="b" anchor="0.15 0 0"/>
  </equality>
</mujoco>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on the CPU backend instead of the GPU")
    parser.add_argument("-s", "--steps", type=int, default=1000, help="Number of simulation steps")
    parser.add_argument("-r", "--record", action="store_true", help="Record a video of the run")
    parser.add_argument(
        "-o", "--output-dir", type=str, default="renders/mochi", help="Directory the recorded video is written to"
    )
    args = parser.parse_args()
    init(args)
    steps = 5 if en.is_smoke() else args.steps

    with tempfile.TemporaryDirectory(prefix="eden_mochi_loop_") as directory:
        env = EnvBase(
            env_options=EnvOptions(
                num_envs=1,
                sim_dt=1.0 / 120.0,
                mochi_options=MochiOptions(gravity=MOCHI_GRAVITY, n_newton_iterations=8),
            ),
            scene_options=SceneOptions(
                plane=PlaneOptions(material=MochiRigidMaterialOptions()),
                # The models declare no armature; keep Genesis' zero rather than Eden's 0.1 kg m^2 default.
                linkage=EntityOptions(
                    file=write_text(directory, "four_bar.xml", FOUR_BAR),
                    is_articulated=True,
                    is_fixed_base=True,
                    dofs_name=["crank_hinge", "coupler_hinge", "rocker_hinge"],
                    # The fork's armature, Genesis' default for an MJCF hinge. It dominates these light links'
                    # inertia and carries the crank over the linkage's gravitational peak: at zero armature the
                    # 0.3 N*m torque stalls the crank at 4.35 rad for good.
                    default_armature=0.1,
                    material=MochiRigidMaterialOptions(),
                    surface=gs.surfaces.Default(color=(0.3, 0.5, 0.9)),
                ),
                welded=EntityOptions(
                    file=write_text(directory, "welded.xml", WELDED_BOXES),
                    default_root_pos=WELDED_POS,
                    default_armature=0.0,
                    material=MochiRigidMaterialOptions(),
                    surface=gs.surfaces.Default(color=(0.9, 0.5, 0.3)),
                ),
            ),
            cameras_options=recording_cameras(args, pos=CAM_POS, lookat=CAM_LOOKAT),
            show_viewer=args.vis,
        )
        env.build()  # the XML must still exist here: Genesis parses it at build

    linkage, welded = env.entities["linkage"], env.entities["welded"]
    equality = linkage.entity.equalities[0]
    coupler = linkage.entity.links[equality.eq_obj1id - linkage.entity.link_start]
    rocker = linkage.entity.links[equality.eq_obj2id - linkage.entity.link_start]
    anchors = np.asarray(equality.eq_data, dtype=np.float64)
    _, crank_dof = linkage.find_named_dofs_idx_local(["crank_hinge"])
    crank_torque = torch.tensor([[0.3]], device=env.device)

    def anchor_world(link, anchor):
        pos = link.get_pos()[0].cpu().numpy()
        quat = link.get_quat()[0].cpu().numpy()
        return pos + gs.utils.geom.transform_by_quat(anchor, quat)

    recorder = VideoRecorder(env, args, "loop_closure")
    for i_step in range(steps):
        linkage.control_dofs_force(crank_torque, dofs_idx_local=crank_dof)  # constant torque turns the crank
        env.scene.step()
        if i_step % 60 == 0:
            gap = np.linalg.norm(anchor_world(coupler, anchors[0:3]) - anchor_world(rocker, anchors[3:6]))
            print(
                f"step {i_step:4d}: crank angle={float(linkage.get_qpos()[0, 0]):7.3f} rad, "
                f"closing joint gap={gap:.2e} m, welded pair z={float(welded.entity.links[0].get_pos()[0, 2]):.3f} m"
            )
        recorder.frame()
    recorder.close()


if __name__ == "__main__":
    main()
