"""Procedural simulation meshes for deformable bodies, written once to Eden's cache and loaded by path.

An entity takes its mesh as a ``file=`` path, so a structured mesh a task generates (a towel sheet, a
sponge lattice) is written to ``~/.cache/eden/procedural_meshes/`` under a name that spells out every
parameter, and the path goes into the options. A second call with the same parameters reuses the file.

Tetgen meshes a box primitive into a handful of tetrahedra (nine vertices for a sponge), far too coarse to
deform; :func:`box_tet_lattice_file` builds a regular lattice instead.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

from eden.utils.misc import get_cache_dir

# Five tetrahedra per hexahedral cell, as corner indices of the cell (corner ``a + 2 b + 4 c`` is offset
# ``(a, b, c)``). The two variants alternate with the cell parity so neighbouring cells share face diagonals.
_EVEN_CELL_TETS = ((0, 1, 2, 4), (1, 3, 2, 7), (1, 4, 5, 7), (2, 4, 7, 6), (1, 2, 4, 7))
_ODD_CELL_TETS = ((0, 1, 3, 5), (0, 3, 2, 6), (0, 4, 5, 6), (3, 5, 7, 6), (0, 3, 5, 6))


def write_atomically(path: str, text: str) -> None:
    """Write ``text`` to ``path`` (creating its directory) so no reader ever sees a partial file.

    Several processes (DDP ranks, parallel tests) may generate the same cached file at once: each writes its own
    temporary file and the rename makes whichever lands last the (identical) result.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def derived_file_path(subdir: str, source: str, name: str) -> str:
    """Where Eden caches a file derived from the file ``source``: one path per source file, named ``name``.

    A derived file (an asset with parts stripped, a mesh rescaled) often names the source's own files by absolute
    path, so it is keyed on the source's real path: two checkouts never share one. Put any parameter the derivation
    takes in ``name``. Callers rewrite the file (with :func:`write_atomically`) on every call, which costs
    milliseconds and keeps it following both the source and the code that derives it.
    """
    key = hashlib.sha1(os.path.realpath(source).encode()).hexdigest()[:16]
    stem, ext = os.path.splitext(name)
    return os.path.join(get_cache_dir(subdir), f"{stem}_{key}{ext}")


def box_tet_lattice(size: tuple[float, float, float], cells: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Vertices ``(n, 3)`` and positively oriented tetrahedra ``(m, 4)`` of a box centred at the origin.

    Parameters
    ----------
    size : tuple[float, float, float]
        Box extents along x, y, z (m).
    cells : tuple[int, int, int]
        Hexahedral cells along x, y, z; each holds five tetrahedra.
    """
    if any(c < 1 for c in cells):
        raise ValueError(f"box_tet_lattice needs at least one cell per axis, got cells={cells}.")
    axes = [np.linspace(-0.5 * s, 0.5 * s, c + 1) for s, c in zip(size, cells)]
    verts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    _, ny, nz = (c + 1 for c in cells)
    tets = []
    for i in range(cells[0]):
        for j in range(cells[1]):
            for k in range(cells[2]):
                corners = [((i + a) * ny + (j + b)) * nz + (k + c) for c in (0, 1) for b in (0, 1) for a in (0, 1)]
                for tet in _EVEN_CELL_TETS if (i + j + k) % 2 == 0 else _ODD_CELL_TETS:
                    tets.append([corners[t] for t in tet])
    tets = np.asarray(tets)
    p = verts[tets]
    volume = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])
    flip = volume < 0.0
    tets[flip, 1], tets[flip, 2] = tets[flip, 2], tets[flip, 1].copy()
    return verts, tets


def box_tet_lattice_file(size: tuple[float, float, float], cells: tuple[int, int, int]) -> str:
    """Path of a legacy ASCII VTK file holding :func:`box_tet_lattice` (a ``MochiElasticMaterialOptions`` mesh)."""
    name = "box_{:g}x{:g}x{:g}_{}x{}x{}.vtk".format(*size, *cells)
    path = os.path.join(get_cache_dir("procedural_meshes"), name)
    if not os.path.exists(path):
        verts, tets = box_tet_lattice(size, cells)
        lines = ["# vtk DataFile Version 2.0", name, "ASCII", "DATASET UNSTRUCTURED_GRID"]
        lines.append(f"POINTS {len(verts)} double")
        lines += [f"{x:.9g} {y:.9g} {z:.9g}" for x, y, z in verts]
        lines.append(f"CELLS {len(tets)} {5 * len(tets)}")
        lines += [f"4 {a} {b} {c} {d}" for a, b, c, d in tets]
        lines.append(f"CELL_TYPES {len(tets)}")
        lines += ["10"] * len(tets)
        write_atomically(path, "\n".join(lines) + "\n")
    return path


def grid_sheet(size: tuple[float, float], cells: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Vertices ``(n, 3)`` and consistently wound triangles ``(m, 3)`` of a flat sheet in z = 0, centred."""
    if any(c < 1 for c in cells):
        raise ValueError(f"grid_sheet needs at least one cell per axis, got cells={cells}.")
    axes = [np.linspace(-0.5 * s, 0.5 * s, c + 1) for s, c in zip(size, cells)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
    verts = np.concatenate([grid, np.zeros((len(grid), 1))], axis=1)
    ny = cells[1] + 1
    faces = []
    for i in range(cells[0]):
        for j in range(cells[1]):
            a = i * ny + j
            faces.append([a, a + ny, a + ny + 1])
            faces.append([a, a + ny + 1, a + 1])
    return verts, np.asarray(faces)


def grid_sheet_file(size: tuple[float, float], cells: tuple[int, int]) -> str:
    """Path of a Wavefront OBJ holding :func:`grid_sheet` (a ``MochiShellMaterialOptions`` mesh)."""
    name = "sheet_{:g}x{:g}_{}x{}.obj".format(*size, *cells)
    path = os.path.join(get_cache_dir("procedural_meshes"), name)
    if not os.path.exists(path):
        verts, faces = grid_sheet(size, cells)
        lines = [f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in verts]
        lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
        write_atomically(path, "\n".join(lines) + "\n")
    return path


def swept_tube(path: np.ndarray, radius: float, sides: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and outward-wound triangles of a closed tube of circular section swept along a planar polyline.

    The ends are capped, so the mesh is watertight (a signed-distance collider needs an inside). The path must
    lie in a plane; the section turns with the path, one ring of ``sides`` vertices per path point, and a bend
    made of short segments reads as a smooth elbow.

    Parameters
    ----------
    path : np.ndarray
        Points ``(n, 3)`` of the tube's centreline, ``n >= 2``, no two consecutive ones equal.
    radius : float
        Section radius (m).
    sides : int
        Vertices per ring.
    """
    path = np.asarray(path, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2:
        raise ValueError(f"swept_tube needs a path of shape (n >= 2, 3), got {path.shape}.")
    segments = np.diff(path, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError(f"swept_tube's path repeats a point at index {int(np.argmin(lengths))}.")
    segments /= lengths[:, None]
    # Tangent at each point: the mean of the segments meeting there, so a ring bisects the bend.
    tangents = np.concatenate([segments[:1], segments[:-1] + segments[1:], segments[-1:]])
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    # The plane's normal is one fixed axis of every ring.
    normal = np.zeros(3)
    for i in range(len(segments) - 1):
        normal = np.cross(segments[i], segments[i + 1])
        if np.linalg.norm(normal) > 1e-9:
            break
    if np.linalg.norm(normal) <= 1e-9:  # a straight path: any axis across it
        normal = np.cross(segments[0], [1.0, 0.0, 0.0] if abs(segments[0][0]) < 0.9 else [0.0, 1.0, 0.0])
    normal /= np.linalg.norm(normal)
    angles = 2.0 * np.pi * np.arange(sides) / sides
    rings = []
    for point, tangent in zip(path, tangents):
        # A ring bisecting a bend of angle a is radius / cos(a / 2) across it; at the few degrees a segment of
        # an elbow turns, the tube stays round to well under a percent.
        side = np.cross(tangent, normal)
        rings.append(point + radius * (np.cos(angles)[:, None] * normal + np.sin(angles)[:, None] * side))
    verts = np.concatenate(rings + [path[:1], path[-1:]])
    faces = []
    for i in range(len(path) - 1):
        for k in range(sides):
            a, b = i * sides + k, i * sides + (k + 1) % sides
            c, d = a + sides, b + sides
            faces += [[a, c, b], [b, c, d]]
    start, end = len(path) * sides, len(path) * sides + 1
    last = (len(path) - 1) * sides
    for k in range(sides):
        faces.append([start, k, (k + 1) % sides])
        faces.append([end, last + (k + 1) % sides, last + k])
    faces = np.asarray(faces)
    # Wind outward: a closed mesh wound inward has negative signed volume.
    p = verts[faces]
    if np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() < 0.0:
        faces = faces[:, ::-1]
    return verts, faces


def swept_tube_file(name: str, path: np.ndarray, radius: float, sides: int = 24) -> str:
    """Path of a Wavefront OBJ holding :func:`swept_tube`, cached under ``name`` (which must spell out the path)."""
    file = os.path.join(get_cache_dir("procedural_meshes"), f"tube_{name}_r{radius:g}_{sides}.obj")
    if not os.path.exists(file):
        verts, faces = swept_tube(path, radius, sides)
        lines = [f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in verts]
        lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
        write_atomically(file, "\n".join(lines) + "\n")
    return file
