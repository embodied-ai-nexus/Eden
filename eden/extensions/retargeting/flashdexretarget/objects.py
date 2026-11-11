"""Object URDFs for simulation (``<dataset>/objects/<id>/object.urdf``): the convex parts collide, ``visual.obj`` is
drawn, and the inertial follows FlashDexRetarget's rule — density 800 kg/m^3, the mass clamped to [0.05, 0.5] kg with
the inertia scaled to match. This replaces FlashDexRetarget's Isaac USD conversion, which authored the same URDF.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

import eden as en
from eden.extensions.retargeting.flashdexretarget.convex import convex_parts

#: FlashDexRetarget's object density (kg/m^3) and mass clamp (kg).
OBJECT_DENSITY = 800.0
OBJECT_MASS_CLAMP = (0.05, 0.5)


def object_urdf_path(obj_dir: str | Path, scale: float = 1.0) -> Path:
    """The URDF of object ``obj_dir`` at mesh ``scale`` (``object.urdf``, or ``object_s<scale>.urdf``)."""
    name = "object.urdf" if float(scale) == 1.0 else f"object_s{float(scale):g}.urdf"
    return Path(obj_dir) / name


def object_mass_properties(obj_dir: str | Path, scale: float = 1.0) -> tuple[float, float, np.ndarray, np.ndarray]:
    """``(mass, raw mass, com, inertia about the com)`` of the scaled object at :data:`OBJECT_DENSITY`.

    The visual mesh's volume when it is watertight, else the union of the convex parts'. The mass is clamped to
    :data:`OBJECT_MASS_CLAMP` and the inertia scaled by the same factor.
    """
    obj_dir = Path(obj_dir)
    solid = trimesh.load(obj_dir / "visual.obj", force="mesh")
    if not solid.is_watertight or solid.volume <= 0.0:
        solid = trimesh.util.concatenate([trimesh.load(p, force="mesh") for p in convex_parts(obj_dir)])
    if scale != 1.0:
        solid.apply_scale(scale)
    solid.density = OBJECT_DENSITY
    mass_raw = float(solid.mass)
    mass = min(max(mass_raw, OBJECT_MASS_CLAMP[0]), OBJECT_MASS_CLAMP[1])
    return mass, mass_raw, np.asarray(solid.center_mass), np.asarray(solid.moment_inertia) * (mass / mass_raw)


def write_object_urdf(obj_dir: str | Path, scale: float = 1.0) -> Path:
    """Write the single-link URDF of one object at one mesh scale and return its path.

    Parameters
    ----------
    obj_dir : str | Path
        ``<dataset>/objects/<id>``, holding ``visual.obj`` and ``convex/<i>.obj``.
    scale : float
        Uniform mesh scale.
    """
    obj_dir = Path(obj_dir)
    parts = convex_parts(obj_dir)
    if not parts:
        raise FileNotFoundError(f"No convex parts in {obj_dir / 'convex'}; run the convex step (steps.convex=true).")
    mass, mass_raw, com, inertia = object_mass_properties(obj_dir, scale)
    s = f'scale="{scale:g} {scale:g} {scale:g}"'
    lines = [
        '<?xml version="1.0"?>',
        f"<!-- Written by eden.extensions.retargeting.flashdexretarget.objects. Inertial: density {OBJECT_DENSITY:g} kg/m^3,",
        f"     raw mass {mass_raw:.6g} kg clamped to {mass:.6g} kg (inertia scaled to match). -->",
        f'<robot name="object_{obj_dir.name}">',
        '  <link name="baseLink">',
        "    <inertial>",
        f'      <origin rpy="0 0 0" xyz="{com[0]:.9g} {com[1]:.9g} {com[2]:.9g}"/>',
        f'      <mass value="{mass:.9g}"/>',
        f'      <inertia ixx="{inertia[0, 0]:.9g}" ixy="{inertia[0, 1]:.9g}" ixz="{inertia[0, 2]:.9g}"'
        f' iyy="{inertia[1, 1]:.9g}" iyz="{inertia[1, 2]:.9g}" izz="{inertia[2, 2]:.9g}"/>',
        "    </inertial>",
        "    <visual>",
        f'      <geometry><mesh filename="visual.obj" {s}/></geometry>',
        "    </visual>",
    ]
    for i, part in enumerate(parts):
        lines += [
            f'    <collision name="collision_{i}">',
            f'      <geometry><mesh filename="convex/{part.name}" {s}/></geometry>',
            "    </collision>",
        ]
    lines += ["  </link>", "</robot>", ""]
    path = object_urdf_path(obj_dir, scale)
    path.write_text("\n".join(lines))
    return path


def write_object_urdfs(pool: str | Path, objects: list[tuple[str, float]]) -> list[Path]:
    """URDFs of every ``(object id, scale)`` in ``objects`` under ``<pool>/objects/``; returns their paths."""
    out = [write_object_urdf(Path(pool) / "objects" / oid, scale) for oid, scale in objects]
    en.logger.info(f"[urdf] {len(out)} object URDFs")
    return out
