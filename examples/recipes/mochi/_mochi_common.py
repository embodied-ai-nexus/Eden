"""Helpers shared by the Mochi recipes: fp64 init, the smoke cap, video recording, and mesh generators.

The CLI itself is *not* shared: every example declares its own flags, so the convention checker
(``tests/utils/test_example_flags.py``) can see them and a reader sees the whole interface in one file.

Not a runnable example (listed in ``eden.utils.examples_catalog.EXCLUDE``). Every recipe in this directory
imports it by name, which works because Python puts a script's own directory first on ``sys.path``.
"""

import argparse
import os

import genesis as gs
import numpy as np
import trimesh

import eden as en
from eden.options import CameraOptions, CamerasOptions
from eden.utils.misc import get_cache_dir
from eden.utils.procedural_meshes import write_atomically

# The gravity the fork's mochi examples and the mochi benchmark scenes run at. Eden's ``SimOptions``
# defaults to -9.81, and the 0.1% difference is enough to settle a draped cloth into a different fold
# (the box-edge metric of ``cloth_p1q6.py`` moves from -7.5 mm to -5.1 mm), so the recipes state it.
# ``MochiOptions`` carries a per-solver gravity; ``EnvOptions`` has no gravity field.
MOCHI_GRAVITY = (0.0, 0.0, -9.8)

# The joint-space press used by ``cloth_arm.py`` / ``rope_arm.py``: the hand starts ~0.30 m above the work
# point ``(0.65, 0.0)`` and is driven down to ~0.13 m while the fingertips press with 1 N each.
FRANKA_PRESS_START_QPOS = (-1.0418, 1.3805, 1.5885, -1.7021, -1.3798, 1.5783, 1.4467, 0.04, 0.04)
FRANKA_PRESS_TARGET_QPOS = (-1.0095, 1.5617, 1.3595, -1.6830, -1.5855, 1.7817, 1.4595, 0.04, 0.04)
FRANKA_PRESS_WORK_XY = (0.65, 0.0)
FRANKA_ARM_DOFS = [f"joint{i}" for i in range(1, 8)]
FRANKA_FINGER_DOFS = ["finger_joint1", "finger_joint2"]


def init(args: argparse.Namespace) -> None:
    """Initialize Eden for a Mochi scene: the default 1e9 Pa/m contact stiffness needs double precision.

    ``--cpu`` is optional: a recipe whose renderer or sensor is CUDA-only exposes no such flag (a
    CPU run would have no outcome but a crash), and is read here as the GPU default.
    """
    en.init(backend=gs.cpu if getattr(args, "cpu", False) else gs.gpu, precision="64")


def recording_cameras(
    args: argparse.Namespace, *, pos, lookat, res=(640, 360), fov: float = 30.0
) -> CamerasOptions | None:
    """A single ``rec`` camera framed like the viewer, or None when not recording.

    ``fov`` defaults to Genesis' camera default (30 degrees) rather than Eden's ``CameraOptions`` default, so
    the recordings frame the scene the way the original examples do.
    """
    if not args.record:
        return None
    return CamerasOptions(rec=CameraOptions(cam_pos=pos, cam_lookat=lookat, cam_resolution=res, cam_fov=fov))


class VideoRecorder:
    """Record one frame per :meth:`frame` call into ``<output_dir>/<name>.mp4``; a no-op when not recording."""

    def __init__(self, env, args: argparse.Namespace, name: str, fps: float = 30.0):
        # As with ``--cpu``: a recipe whose whole output is the video offers no ``--record`` flag,
        # and always records.
        self._camera = env.cameras["rec"] if getattr(args, "record", True) else None
        if self._camera is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            self.path = os.path.join(args.output_dir, f"{name}.mp4")
            self._camera.start_recording(self.path, fps=fps)

    def frame(self) -> None:
        if self._camera is not None:
            # ``render_rgb`` rather than ``render``: it is the one pass both camera backends share,
            # and the multi-output ``render`` is not implemented for a sensor-backed camera (Nyx).
            self._camera.render_rgb()

    def close(self) -> None:
        if self._camera is not None:
            self._camera.stop_recording()
            print(f"Saved video to {self.path}")


def newton_iterations(env) -> int:
    """Newton iterations the Mochi solver spent on the last substep of environment 0."""
    return int(env.mochi_solver.get_convergence_info()["n_iter"][0])


def sheet_mesh(n_cells: int, size: float) -> tuple[str, np.ndarray]:
    """Write a flat ``n_cells x n_cells`` triangle sheet of side ``size`` (centered, z = 0) to Eden's cache, once.

    Returns the ``.obj`` path and the ``(2 n_cells^2, 3)`` face array. Consistently wound, as Mochi shells require,
    with the fork's recipes' winding (normals down): :func:`eden.utils.procedural_meshes.grid_sheet` winds the other
    way round, so the recipes keep this sheet to reproduce the fork's runs.
    """
    axis = np.linspace(-0.5 * size, 0.5 * size, n_cells + 1)
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
    verts = np.stack([grid_x.reshape(-1), grid_y.reshape(-1), np.zeros(grid_x.size)], axis=-1)
    faces = []
    for i in range(n_cells):
        for j in range(n_cells):
            a = i * (n_cells + 1) + j
            faces.append([a, a + 1, a + n_cells + 2])
            faces.append([a, a + n_cells + 2, a + n_cells + 1])
    faces = np.array(faces)
    path = os.path.join(get_cache_dir("procedural_meshes"), f"recipe_sheet_{n_cells}_{size:g}.obj")
    if not os.path.exists(path):
        write_atomically(
            path, trimesh.exchange.obj.export_obj(trimesh.Trimesh(vertices=verts, faces=faces, process=False))
        )
    return path, faces


def rod_section(radius: float, *, E: float, G: float, rho: float) -> dict[str, float]:
    """Explicit stiffness / inertia fields of a circular rod section for ``MochiRodMaterialOptions``.

    The material derives the same quantities from ``E``, ``nu``, ``rho`` and the radius on its own; the explicit
    form is for benchmark scenes that set ``E`` and ``G`` independently.
    """
    area = np.pi * radius**2
    second_moment = 0.25 * np.pi * radius**4
    polar_moment = 0.5 * np.pi * radius**4
    return {
        "axial_stiffness": E * area,
        "flexural_stiffness": E * second_moment,
        "torsional_stiffness": G * polar_moment,
        "linear_density": rho * area,
        "linear_rotational_inertia": rho * polar_moment,
    }


def write_text(directory: str, name: str, text: str) -> str:
    """Write ``text`` to ``directory/name`` and return the path (for MJCF models authored inline)."""
    path = os.path.join(directory, name)
    with open(path, "w") as f:
        f.write(text)
    return path
