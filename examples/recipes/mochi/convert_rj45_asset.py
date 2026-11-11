"""Convert Newton's ``rj45_plug.usd`` into the meshes and cable centerline the RJ45 recipes read.

Newton ships the RJ45 scene as one USD stage (``newton/examples/assets/rj45_plug.usd``,
newton-physics/newton, Apache-2.0). Eden's Mochi recipes read plain files, so the three connector
meshes become Z-up ``.obj`` solids in their own prim frames and the cable's ``BasisCurves`` becomes an
``(n, 3)`` world-space polyline.

The socket needs one repair on the way. Two of its polygons (a triangle inside the quad next to it, on
each side of the latch window) are redundant: they leave six edges with a face count other than two, so
the mesh is not watertight exactly where the latch works. Newton never notices -- it builds a
narrow-band SDF with a margin and its VBD contact is compliant -- but Genesis signs its distance field
with a winding number over the whole mesh, and there the duplicated sheet flips the sign of a pocket
inside the socket cavity. The latch then jams on ~0.5 mm of phantom solid at z ~ 15.4 mm instead of
springing back. Dropping the two redundant faces makes the mesh watertight and leaves its volume unchanged.

Needs USD to open the stage. Upload ``<out_dir>`` as ``objects/rj45`` of the ``Kashu7100/eden_assets`` Hub
dataset, which ``_rj45.py`` reads::

    uv run --with usd-core --with trimesh python examples/recipes/mochi/convert_rj45_asset.py <newton_repo> <out_dir>
"""

import argparse
import collections
import os

import numpy as np
import trimesh
from pxr import Usd, UsdGeom

MESHES = (("socket", "/World/Socket"), ("plug", "/World/Plug"), ("latch", "/World/Latch"))
# Only the socket is a collider in the recipe, so only its distance field has to be signed correctly.
# The plug and the latch carry contact samples, which are plain triangle quadrature points.
COLLIDERS = frozenset({"socket"})


def read_polygons(prim):
    """Points and per-face vertex-index lists of a USD mesh prim."""
    mesh = UsdGeom.Mesh(prim)
    points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
    indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())
    polygons, offset = [], 0
    for count in counts:
        polygons.append(indices[offset : offset + count])
        offset += count
    return points, polygons


def non_manifold_edges(polygons):
    """Edges of a polygon soup that are not shared by exactly two faces."""
    counts = collections.Counter()
    for poly in polygons:
        for k in range(len(poly)):
            counts[tuple(sorted((int(poly[k]), int(poly[(k + 1) % len(poly)]))))] += 1
    return {edge: count for edge, count in counts.items() if count != 2}


def drop_redundant_polygons(points, polygons):
    """Drop the fewest faces that make the soup edge-manifold, smallest first.

    A face is removed only when its removal strictly reduces the number of bad edges, so a sound mesh
    is returned untouched and a repair can never eat into real surface.
    """
    kept = list(range(len(polygons)))
    areas = {}
    for i, poly in enumerate(polygons):
        verts = points[poly]
        fan = np.cross(verts[1:-1] - verts[0], verts[2:] - verts[0])
        areas[i] = 0.5 * float(np.linalg.norm(fan.sum(axis=0)))
    dropped = []
    bad = len(non_manifold_edges(polygons))
    while bad:
        broken = {v for edge in non_manifold_edges([polygons[i] for i in kept]) for v in edge}
        candidates = [i for i in kept if broken & {int(v) for v in polygons[i]}]
        for i in sorted(candidates, key=lambda i: areas[i]):
            trial = [j for j in kept if j != i]
            if len(non_manifold_edges([polygons[j] for j in trial])) < bad:
                kept, dropped = trial, dropped + [i]
                bad = len(non_manifold_edges([polygons[j] for j in kept]))
                break
        else:
            break
    return [polygons[i] for i in kept], dropped


def triangulate(polygons):
    """Fan-triangulate a convex polygon soup."""
    tris = []
    for poly in polygons:
        for k in range(1, len(poly) - 1):
            tris.append([poly[0], poly[k], poly[k + 1]])
    return np.array(tris, dtype=np.int64)


def world_translate(prim):
    matrix = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    return matrix[:3, 3]


def main(newton_repo, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    stage = Usd.Stage.Open(f"{newton_repo}/newton/examples/assets/rj45_plug.usd")
    for name, path in MESHES:
        prim = stage.GetPrimAtPath(path)
        points, polygons = read_polygons(prim)
        polygons, dropped = drop_redundant_polygons(points, polygons)
        mesh = trimesh.Trimesh(vertices=points, faces=triangulate(polygons), process=True)
        trimesh.repair.fill_holes(mesh)
        mesh.remove_unreferenced_vertices()
        if name in COLLIDERS and not mesh.is_watertight:
            raise SystemExit(f"{name}: not watertight after repair, so its signed distance field would be unreliable.")
        print(
            f"{name}: V={len(mesh.vertices)} F={len(mesh.faces)} watertight={mesh.is_watertight} "
            f"volume={mesh.volume:.4e} m^3 origin={world_translate(prim)} dropped_polygons={dropped}"
        )
        mesh.export(f"{out_dir}/{name}.obj")

    prim = stage.GetPrimAtPath("/World/CableCurve")
    points = np.array(UsdGeom.BasisCurves(prim).GetPointsAttr().Get(), dtype=np.float64) + world_translate(prim)
    np.save(f"{out_dir}/cable_centerline.npy", points)
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    print(f"cable: {len(points)} nodes, {segments.sum() * 1e3:.1f} mm long, {segments.mean() * 1e3:.2f} mm segments")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("newton_repo", help="newton-physics/newton checkout holding rj45_plug.usd")
    parser.add_argument("out_dir", help="directory to write the meshes and centerline to (upload it as objects/rj45)")
    args = parser.parse_args()
    main(args.newton_repo, args.out_dir)
