"""A Franka grasps and lifts a rigid cube using the SAP convex frictional-contact coupler.

---
tier: recipe
capabilities: [solver.sap, solver.rigid, robot.franka]
hardware: gpu-fp64
optional_deps: []
run: uv run python examples/recipes/physics/sap_franka_grasp_cube.py -v
---

Eden port of Genesis' ``examples/sap_coupling/franka_grasp_rigid_cube.py``.

A Franka grasps and lifts a rigid cube using the **SAP** (Semi-Analytic Primal)
convex contact coupler. SAP resolves the frictional finger/cube contact that
holds the cube up under a constant closing force.

Three things make SAP work in Eden, all surfaced by this example:

- **Double precision.** SAP needs fp64; ``en.init(precision="64")`` (added for this)
  threads it through to ``gs.init``.
- **Fine epsilon.** Eden defaults ``gs.init(eps=1e-8)`` for rigid-solver stability, but
  that is coarser than SAP's ``1e-10`` convergence tolerances and makes contacts
  explode. ``en.init(eps=1e-15)`` (also added for this) restores stability — this
  matches the Genesis default eps that the reference example relies on.
- **SAP coupler options** flow through ``EnvOptions.coupler_options`` exactly like the
  IPC coupler (``gs.options.SAPCouplerOptions``).

The cube is held up purely by friction from a constant ``control_dofs_force`` on the
fingers. The SAP coupler takes its floor from the scene's rigid ground plane, which
must be a single fixed plane facing up: without one, nothing holds the cube up.

    uv run python examples/recipes/physics/sap_franka_grasp_cube.py -v
"""

import argparse

import genesis as gs
import torch

import eden as en
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import BoxOptions, PlaneOptions
from eden.options.materials import RigidMaterialOptions
from eden.options.robots import FrankaEmikaPanda


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    args = parser.parse_args()

    en.init(backend=gs.gpu, precision="64", eps=1e-15)

    grippy = RigidMaterialOptions(coup_friction=1.0, friction=1.0)

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=1.0 / 60,
            sim_substeps=2,
            enable_self_collision=False,
            coupler_options=gs.options.SAPCouplerOptions(
                pcg_threshold=1e-10,
                sap_convergence_atol=1e-10,
                sap_convergence_rtol=1e-10,
                linesearch_ftol=1e-10,
            ),
        ),
        scene_options=SceneOptions(
            # The SAP coupler's floor: a rigid plane, not a kinematic one (which no rigid contact sees)
            plane=PlaneOptions(),
            franka=FrankaEmikaPanda(material=grippy),
            cube=BoxOptions(size=(0.04, 0.04, 0.04), default_root_pos=(0.65, 0.0, 0.02), material=grippy),
        ),
        show_viewer=args.vis,
    )

    env.build()

    franka = env.entities["franka"].entity
    motors_dof, fingers_dof = slice(0, 7), slice(7, 9)
    end_effector = franka.get_link("hand")
    B, device = env.num_envs, env.device

    def run(steps: int):
        for _ in range(steps if not en.is_smoke() else 1):
            env.scene.step()

    # Start near the cube (matches the original's hand-tuned init qpos).
    init_qpos = torch.tensor([[-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04]], device=device)
    franka.set_qpos(init_qpos)

    target_quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device).repeat(B, 1)

    def reach(z: float):
        pos = torch.tensor([[0.65, 0.0, z]], device=device).repeat(B, 1)
        qpos = franka.inverse_kinematics(link=end_effector, pos=pos, quat=target_quat)
        franka.control_dofs_position(qpos[..., motors_dof], dofs_idx_local=motors_dof)

    closing_force = torch.full((B, 2), -1.0, device=device)

    reach(0.13)  # hold above the cube
    run(15)

    # close on the cube with a constant force (SAP resolves the frictional hold)
    for _ in range(10 if not en.is_smoke() else 1):
        franka.control_dofs_force(closing_force, dofs_idx_local=fingers_dof)
        env.scene.step()

    reach(0.3)  # lift, holding the grasp force below
    cube = env.entities["cube"]
    z0 = cube.get_pos()[0, 2].item()
    for _ in range(40 if not en.is_smoke() else 1):
        franka.control_dofs_force(closing_force, dofs_idx_local=fingers_dof)
        env.scene.step()

    print(f"\ncube z: start={z0:.3f} m  ->  final={cube.get_pos()[0, 2].item():.3f} m")


if __name__ == "__main__":
    main()
