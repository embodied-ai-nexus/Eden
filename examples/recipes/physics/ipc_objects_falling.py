"""
IPC cloth + soft ball falling onto a rigid cube.

---
tier: recipe
capabilities: [solver.ipc, solver.fem, solver.rigid]
hardware: gpu
optional_deps: [libuipc]
run: uv run python examples/recipes/physics/ipc_objects_falling.py -v
---

Demonstrates Eden's FEM + IPC interface:
- FEMClothMaterialOptions for cloth
- FEMElasticMaterialOptions for volumetric soft body
- RigidMaterialOptions with coup_type="ipc_only" for IPC-driven rigid objects

Equivalent to references/Genesis/examples/IPC_Solver/ipc_objects_falling.py
but expressed through Eden's config-driven scene construction.
"""

import argparse

import genesis as gs
from huggingface_hub import snapshot_download

import eden as en
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import BoxOptions, EntityOptions, PlaneOptions, SphereOptions
from eden.options.materials import (
    FEMClothMaterialOptions,
    FEMElasticMaterialOptions,
    RigidMaterialOptions,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    args = parser.parse_args()

    en.init(backend=gs.gpu)

    # Download cloth mesh from HuggingFace
    asset_path = snapshot_download(
        repo_type="dataset",
        repo_id="Genesis-Intelligence/assets",
        revision="8aa8fcd60500b9f3a36c356080224bdb1be9ee59",
        allow_patterns="IPC/grid20x20.obj",
        max_workers=1,
    )

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=0.02,
            sim_substeps=1,
            decimation=1,  # this example advances via env.step(), so control decimation is live
            # IPC coupler requires non-batched rigid info
            batch_dofs_info=False,
            coupler_options=gs.options.IPCCouplerOptions(
                contact_d_hat=0.01,
                two_way_coupling=True,
            ),
        ),
        # Initial poses are chosen with a clear gap between every entity so the IPC
        # GPU sanity check does not flag any penetration at build time.
        # (Note: Eden currently bakes `default_root_pos` into file-mesh morphs
        # but not `default_root_quat` — rotated initial poses would need the
        # rotation baked directly into the asset.)
        scene_options=SceneOptions(
            plane=PlaneOptions(),
            cloth=EntityOptions(
                file=f"{asset_path}/IPC/grid20x20.obj",
                scale=1.5,
                default_root_pos=(0.0, 0.0, 1.2),
                # The grid20x20 mesh is flat in the XZ plane (mesh "up" is Y),
                # so realign its Y axis to world Z to get a horizontal cloth.
                up=(0, 1, 0),
                front=(1, 0, 0),
                material=FEMClothMaterialOptions(
                    E=1e5,
                    nu=0.499,
                    rho=200.0,
                    thickness=0.001,
                    bending_stiffness=50.0,
                ),
                surface=gs.surfaces.Plastic(color=(0.3, 0.5, 0.8, 1.0)),
            ),
            cube=BoxOptions(
                size=(0.2, 0.2, 0.2),
                default_root_pos=(-0.25, 0.0, 0.4),
                material=RigidMaterialOptions(
                    rho=500,
                    coup_type="ipc_only",
                ),
                surface=gs.surfaces.Plastic(color=(0.8, 0.3, 0.2, 0.8)),
            ),
            ball=SphereOptions(
                radius=0.08,
                default_root_pos=(0.25, 0.0, 0.2),
                material=FEMElasticMaterialOptions(
                    E=1.0e3,
                    nu=0.3,
                    rho=1000.0,
                    model="stable_neohookean",
                ),
                surface=gs.surfaces.Plastic(color=(0.2, 0.8, 0.3, 0.8)),
            ),
        ),
        show_viewer=args.vis,
    )

    env.build()

    horizon = 100 if not en.is_smoke() else 5
    for _ in range(horizon):
        env.step()


if __name__ == "__main__":
    main()
