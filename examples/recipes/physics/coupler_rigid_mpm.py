"""An MPM-elastic cube welded to a kinematically-driven rigid box via particle constraints.

---
tier: recipe
capabilities: [solver.mpm, solver.rigid]
hardware: gpu
optional_deps: []
run: uv run python examples/recipes/physics/coupler_rigid_mpm.py -v
---

Eden port of Genesis' ``examples/coupling/rigid_mpm_attachment.py``.

An MPM-elastic cube is welded to a rigid box by a set of *particle constraints*:
the particles in a thin slab near the cube's top are pinned to the rigid box's
link with a soft spring. The rigid box is then driven up and down kinematically
(``set_qpos`` each step), stretching the deformable cube like a piston.

Particle selection and constraints use Eden's ``ParticleEntity`` wrappers
(``get_particles_in_bbox`` / ``set_particle_constraints``), which forward to the
Genesis MPM entity — no reaching through ``entity.entity`` required.

    uv run python examples/recipes/physics/coupler_rigid_mpm.py -v
"""

import argparse

import genesis as gs
import torch

import eden as en
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import BoxOptions, PlaneOptions
from eden.options.materials import MPMElasticMaterialOptions, RigidMaterialOptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    args = parser.parse_args()

    en.init(backend=gs.gpu)

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=2e-3,
            sim_substeps=20,
            mpm_options=gs.options.MPMOptions(
                lower_bound=(-1.0, -1.0, 0.0),
                upper_bound=(1.0, 1.0, 1.5),
                grid_density=64,
            ),
        ),
        scene_options=SceneOptions(
            plane=PlaneOptions(),
            rigid_box=BoxOptions(
                size=(0.12, 0.12, 0.05),
                default_root_pos=(0.0, 0.0, 0.55),
                fixed=False,
                # this box is driven kinematically and only holds the particle
                # constraint, so it does not need solver-level coupling.
                material=RigidMaterialOptions(needs_coup=False),
            ),
            mpm_cube=BoxOptions(
                size=(0.15, 0.15, 0.15),
                default_root_pos=(0.0, 0.0, 0.35),
                material=MPMElasticMaterialOptions(E=5e4, nu=0.3, rho=1000.0),
                surface=gs.surfaces.Default(vis_mode="particle"),
                vis_mode="particle",
            ),
        ),
        show_viewer=args.vis,
    )

    env.build()

    rigid_box = env.entities["rigid_box"]
    mpm_cube = env.entities["mpm_cube"]

    # Pin the top slab of MPM particles to the rigid box's link.
    mask = mpm_cube.get_particles_in_bbox((-0.08, -0.08, 0.41), (0.08, 0.08, 0.44))
    mpm_cube.set_particle_constraints(mask, rigid_box.links[0].idx, stiffness=1e5)

    initial_z = 0.55
    horizon = 500 if not en.is_smoke() else 2
    for i in range(horizon):
        # triangle-wave vertical motion, period 200 steps, amplitude 0.15 m
        z_offset = 0.15 * (1 - abs((i % 200) - 100) / 100.0)
        qpos = torch.tensor(
            [[0.0, 0.0, initial_z + z_offset, 1.0, 0.0, 0.0, 0.0]], device=env.device
        )  # (B=1, 7): pos + wxyz quat
        rigid_box.set_qpos(qpos)
        env.scene.step()


if __name__ == "__main__":
    main()
