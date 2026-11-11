"""FEM soft bodies pinned by vertex constraints, contrasting soft-spring and hard-kinematic modes.

---
tier: recipe
capabilities: [solver.fem]
hardware: gpu
optional_deps: []
run: uv run python examples/recipes/physics/fem_hard_soft_constraint.py -v
---

Eden port of Genesis' ``examples/fem_hard_and_soft_constraint.py``.

FEM (finite-element) soft bodies with vertex constraints. Two deformable spheres/
cubes are pinned by a single vertex each, contrasting the two constraint modes:

- **soft** (spring): the compliant blob (low ``E``) hangs from a soft constraint on
  vertex 0 and sags under gravity around the pinned point.
- **hard** (kinematic): the stiff cube's vertex 0 is driven exactly along a circular
  path, dragging the elastic body with it.

Vertex constraints require ``FEMOptions(enable_vertex_constraints=True)``, passed via
``EnvOptions.fem_options``. The initial constraint is set through Eden's wrapped
``FEMEntity.set_vertex_constraints``; per-step target updates use the Genesis entity's
``update_constraint_targets`` (not yet mirrored on the Eden wrapper — see
examples/recipes/physics/README.md "Known gaps").

    uv run python examples/recipes/physics/fem_hard_soft_constraint.py -v
    uv run python examples/recipes/physics/fem_hard_soft_constraint.py --solver explicit
"""

import math
import argparse

import genesis as gs
import torch

import eden as en
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import BoxOptions, PlaneOptions, SphereOptions
from eden.options.materials import FEMElasticMaterialOptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=["explicit", "implicit"], default="implicit", help="Solver variant to run")
    parser.add_argument("-t", "--seconds", type=float, default=5.0, help="Seconds of simulated time to run")
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    args = parser.parse_args()

    # Implicit is stable at a larger timestep; explicit needs a small dt + substeps.
    dt, substeps = (1e-3, 1) if args.solver == "implicit" else (1e-4, 5)
    seconds = 0.01 if en.is_smoke() else args.seconds

    en.init(backend=gs.gpu)

    # Both bodies sit around this point (matches the original's SCENE_POS).
    sx, sy, sz = 0.5, 0.5, 1.0

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=dt,
            sim_substeps=substeps,
            fem_options=gs.options.FEMOptions(
                use_implicit_solver=args.solver == "implicit",
                enable_vertex_constraints=True,
            ),
        ),
        scene_options=SceneOptions(
            plane=PlaneOptions(),
            blob=SphereOptions(
                radius=0.1,
                default_root_pos=(sx - 0.3, sy - 0.3, sz),
                material=FEMElasticMaterialOptions(E=1.0e4, nu=0.45, rho=1000.0, model="linear_corotated"),
            ),
            cube=BoxOptions(
                size=(0.2, 0.2, 0.2),
                default_root_pos=(sx + 0.3, sy + 0.3, sz),
                material=FEMElasticMaterialOptions(E=1.0e6, nu=0.45, rho=1000.0, model="linear_corotated"),
            ),
        ),
        show_viewer=args.vis,
    )

    env.build()

    blob, cube = env.entities["blob"], env.entities["cube"]
    pinned = [0]  # constrain vertex 0 of each body
    device = env.device

    # Soft constraint: pin the blob's vertex 0 at its rest position (a spring).
    blob_v0 = blob.get_vertex_positions()[0, pinned]  # (1, 3)
    blob.set_vertex_constraints(pinned, target_poss=blob_v0, is_soft_constraint=True, stiffness=1e4)

    # Hard constraint: drive the cube's vertex 0 around a circle of radius 0.3.
    radius, period = 0.3, 10.0
    angle_step = 2 * math.pi * dt / period

    def circle_offset(angle: float) -> torch.Tensor:
        return torch.tensor([-radius * math.cos(angle), -radius * math.sin(angle), 0.0], device=device)

    cube_v0 = cube.get_vertex_positions()[0, pinned[0]]  # (3,)
    center = cube_v0 - circle_offset(0.0)
    cube.set_vertex_constraints(pinned, target_poss=(center + circle_offset(0.0)).unsqueeze(0))

    angle = 0.0
    for _ in range(int(seconds / dt)):
        angle += angle_step
        target = (center + circle_offset(angle)).unsqueeze(0)  # (1, 3)
        # update_constraint_targets is a Genesis FEMEntity call not yet wrapped by Eden.
        cube.entity.update_constraint_targets(pinned, target)
        env.scene.step()

    print(f"blob centroid: {blob.get_pos()[0].tolist()}")
    print(f"cube centroid: {cube.get_pos()[0].tolist()}")


if __name__ == "__main__":
    main()
