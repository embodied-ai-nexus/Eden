"""Mochi contact quadrature A/B: how far a box edge shows through a coarse drape at three vs six samples per triangle.

---
tier: recipe
capabilities: [solver.mochi]
hardware: gpu-fp64
optional_deps: []
run: uv run python examples/recipes/mochi/cloth_p1q6.py -v
---

Eden port of the Genesis fork's ``examples/mochi/cloth_p1q6.py``: the drape half of ``cloth.py`` with six
boundary samples per triangle (``MochiOptions.boundary_element_type="P1Q6"``) instead of the default three.

Contact is sampled, so the box is only felt at the quadrature points of the cloth triangles and its edges
poke through the flat cloth between samples when the mesh is coarse relative to the box (the original mochi
shows the same artifact at the same magnitude). More samples hold the cloth closer to the surface: on this
scene the deepest point of the cloth *surface* below the box edge improves from about -7.6 mm (P1Q3) to
about -6.1 mm, and the deepest cloth *vertex* from about -1.6 mm to about -0.3 mm. Run it twice to compare.
A finer sheet, or a larger ``penalty_threshold`` on the box material, reduces the artifact further.

    uv run python examples/recipes/mochi/cloth_p1q6.py -v
    uv run python examples/recipes/mochi/cloth_p1q6.py -v --quadrature P1Q3
"""

import argparse

import genesis as gs
import numpy as np
from _mochi_common import (
    MOCHI_GRAVITY,
    VideoRecorder,
    init,
    newton_iterations,
    recording_cameras,
    sheet_mesh,
)

import eden as en
from eden.engine.solvers.mochi.options import MochiOptions
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import BoxOptions, EntityOptions, PlaneOptions
from eden.options.materials import MochiRigidMaterialOptions, MochiShellMaterialOptions

# Framed tight on the box edge, where the artifact shows.
CAM_POS, CAM_LOOKAT = (-0.35, -0.45, 0.55), (-0.72, -0.05, 0.3)
BOX_POS = np.array([-0.8, 0.0, 0.15])
BOX_HALF = np.array([0.15, 0.15, 0.15])


def box_sdf(points: np.ndarray) -> np.ndarray:
    """Signed distance of points to the fixed box (negative inside)."""
    q = np.abs(points - BOX_POS) - BOX_HALF
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(axis=-1), 0.0)


def surface_points(verts: np.ndarray, faces: np.ndarray, n_sub: int = 8) -> np.ndarray:
    """Dense barycentric samples of the cloth surface (the mid-triangle points are where an edge pokes through)."""
    ij = [(i, j) for i in range(n_sub + 1) for j in range(n_sub + 1 - i)]
    bary = np.array([[1.0 - (i + j) / n_sub, i / n_sub, j / n_sub] for i, j in ij])
    return np.einsum("qk,fkd->fqd", bary, verts[faces]).reshape(-1, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on the CPU backend instead of the GPU")
    parser.add_argument("-s", "--steps", type=int, default=300, help="Number of simulation steps")
    parser.add_argument("-r", "--record", action="store_true", help="Record a video of the run")
    parser.add_argument(
        "-o", "--output-dir", type=str, default="renders/mochi", help="Directory the recorded video is written to"
    )
    parser.add_argument(
        "--quadrature", choices=("P1Q3", "P1Q6"), default="P1Q6", help="Boundary samples per triangle (3 or 6)"
    )
    args = parser.parse_args()
    init(args)
    steps = 5 if en.is_smoke() else args.steps

    sheet_path, faces = sheet_mesh(12, 0.8)
    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=1.0 / 60.0,
            mochi_options=MochiOptions(
                gravity=MOCHI_GRAVITY, n_newton_iterations=8, boundary_element_type=args.quadrature
            ),
        ),
        scene_options=SceneOptions(
            plane=PlaneOptions(material=MochiRigidMaterialOptions()),
            box=BoxOptions(
                size=(0.3, 0.3, 0.3),
                default_root_pos=tuple(BOX_POS.tolist()),
                is_fixed_base=True,
                material=MochiRigidMaterialOptions(),
            ),
            drape=EntityOptions(
                file=sheet_path,
                default_root_pos=(-0.8, 0.0, 0.6),
                material=MochiShellMaterialOptions(
                    E=2e4,
                    nu=0.3,
                    rho=200.0,
                    thickness=2e-3,
                    friction=0.6,
                    collider_radius=0.02,
                    penalty_coefficient=1e7,
                ),
                surface=gs.surfaces.Default(color=(0.9, 0.4, 0.3)),
            ),
        ),
        cameras_options=recording_cameras(args, pos=CAM_POS, lookat=CAM_LOOKAT),
        show_viewer=args.vis,
    )
    env.build()
    drape = env.entities["drape"]

    recorder = VideoRecorder(env, args, f"cloth_{args.quadrature.lower()}")
    for i_step in range(steps):
        env.scene.step()
        if i_step % 30 == 0:
            nodes = np.asarray(drape.get_vertex_positions()[0].cpu(), dtype=np.float64)
            print(
                f"step {i_step:4d} [{args.quadrature}]: box edge sdf "
                f"surface={float(box_sdf(surface_points(nodes, faces)).min()) * 1e3:7.2f} mm "
                f"node={float(box_sdf(nodes).min()) * 1e3:7.2f} mm newton iterations={newton_iterations(env)}"
            )
        recorder.frame()
    recorder.close()


if __name__ == "__main__":
    main()
