"""Probe clouds: dense surface samples carried by the human, the robot links and the objects (Sec. III-B).

Every entity carries a probe cloud, sampled uniformly by area. Robot and object probes are rigid in
their carrier's frame (a link, an object). Human probes ride a deforming mesh, so they are stored as
barycentric coordinates on its triangles and re-posed every frame, each labelled with the body part
(the heaviest skinning joint) of the triangle it lies on.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import trimesh

from eden.utils.mesh import get_link_mesh

if TYPE_CHECKING:
    from eden.extensions.retargeting.otretarget.human import HumanSurfaceMotion
    from eden.extensions.retargeting.otretarget.kinematics import RigKinematics


def resolve_density(name: str, densities: Mapping[str, float]) -> float:
    """The density of the first pattern in ``densities`` that fully matches ``name``.

    Parameters
    ----------
    name : str
        A link or body-part name.
    densities : Mapping[str, float]
        Ordered ``{regex: points per m^2}``; put specific patterns before a catch-all ``".*"``.
    """
    for pattern, density in densities.items():
        if re.fullmatch(pattern, name):
            return float(density)
    raise ValueError(
        f"No probe density pattern matches {name!r}. Patterns: {list(densities)}. Add a catch-all '.*' entry."
    )


def sample_surface(
    mesh: trimesh.Trimesh, density: float, min_points: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Area-uniform samples of a mesh surface at ``density`` points per m^2 (at least ``min_points``).

    Returns
    -------
    points : np.ndarray
        ``(N, 3)``.
    normals : np.ndarray
        ``(N, 3)`` face normals at the samples.
    faces : np.ndarray
        ``(N,)`` face index of each sample.
    bary : np.ndarray
        ``(N, 3)`` barycentric coordinates in that face.
    """
    n = max(int(np.ceil(mesh.area * density)), int(min_points))
    points, faces = trimesh.sample.sample_surface_even(mesh, n, seed=seed)
    if len(points) < max(min_points, n // 2):
        # Even sampling rejects aggressively on tiny or sliver-heavy meshes; fall back to plain area sampling.
        points, faces = trimesh.sample.sample_surface(mesh, n, seed=seed)
    bary = trimesh.triangles.points_to_barycentric(mesh.triangles[faces], points)
    return np.asarray(points), mesh.face_normals[faces], np.asarray(faces), bary


@dataclass
class RigidProbes:
    """Probes rigidly attached to carriers (robot links or objects).

    Attributes
    ----------
    local : np.ndarray
        ``(N, 3)`` positions in the carrier frame.
    normals : np.ndarray
        ``(N, 3)`` outward surface normals in the carrier frame.
    carrier : np.ndarray
        ``(N,)`` carrier index (rig link index, or object index).
    area : np.ndarray
        ``(N,)`` surface area each probe stands for (its carrier's area over its probe count), m^2.
    """

    local: np.ndarray
    normals: np.ndarray
    carrier: np.ndarray
    area: np.ndarray

    def __len__(self) -> int:
        return len(self.local)

    def world(self, carrier_pos: np.ndarray, carrier_rot: np.ndarray) -> np.ndarray:
        """``(N, 3)`` world positions given carrier poses ``(C, 3)``, ``(C, 3, 3)``."""
        return carrier_pos[self.carrier] + np.einsum("nab,nb->na", carrier_rot[self.carrier], self.local)


def sample_rig_probes(
    kin: RigKinematics,
    densities: Mapping[str, float],
    min_points: int,
    exclude: list[str] | None = None,
    use_visual_mesh: bool = True,
    seed: int = 0,
) -> RigidProbes:
    """Sample every rig link that has geometry, in its link frame.

    Parameters
    ----------
    kin : RigKinematics
        The rig.
    densities : Mapping[str, float]
        ``{link-name regex: points per m^2}``.
    min_points : int
        Floor on samples per link, so small links (finger phalanges) still carry probes.
    exclude : list[str], optional
        Link-name regexes that carry no probes.
    use_visual_mesh : bool
        Sample the visual meshes (the paper's choice) rather than the collision geometry.
    seed : int
        Sampling seed.
    """
    exclude = exclude or []
    links = {link.name: link for ent in kin.entities for link in ent.links}
    local, normals, carrier, area = [], [], [], []
    for li, name in enumerate(kin.link_names):
        if any(re.fullmatch(p, name) for p in exclude):
            continue
        geoms = links[name].vgeoms if use_visual_mesh else links[name].geoms
        if len(geoms) == 0:
            continue
        mesh = get_link_mesh(links[name], use_visual_mesh=use_visual_mesh)
        if mesh.area <= 1e-8:
            continue
        pts, nrm, _, _ = sample_surface(mesh, resolve_density(name, densities), min_points, seed=seed + li)
        local.append(pts)
        normals.append(nrm)
        carrier.append(np.full(len(pts), li, dtype=np.int64))
        area.append(np.full(len(pts), mesh.area / len(pts)))
    if not local:
        raise ValueError(f"sample_rig_probes: no link of rig {kin.entity_names} has geometry to sample.")
    return RigidProbes(np.concatenate(local), np.concatenate(normals), np.concatenate(carrier), np.concatenate(area))


def sample_object_probes(meshes: list[trimesh.Trimesh], density: float, min_points: int, seed: int = 0) -> RigidProbes:
    """Sample each object mesh in its own frame; the carrier index is the object's position in ``meshes``."""
    local, normals, carrier, area = [], [], [], []
    for oi, mesh in enumerate(meshes):
        pts, nrm, _, _ = sample_surface(mesh, density, min_points, seed=seed + oi)
        local.append(pts)
        normals.append(nrm)
        carrier.append(np.full(len(pts), oi, dtype=np.int64))
        area.append(np.full(len(pts), mesh.area / len(pts)))
    if not local:
        return RigidProbes(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, dtype=np.int64), np.zeros(0))
    return RigidProbes(np.concatenate(local), np.concatenate(normals), np.concatenate(carrier), np.concatenate(area))


@dataclass
class HumanProbes:
    """Probes on the human mesh, as barycentric coordinates on its triangles.

    Attributes
    ----------
    corners : np.ndarray
        ``(N, 3)`` vertex indices of each probe's triangle.
    bary : np.ndarray
        ``(N, 3)`` barycentric weights.
    part : np.ndarray
        ``(N,)`` index of the skeleton joint whose part the probe lies on.
    area : np.ndarray
        ``(N,)`` rest-surface area each probe stands for (its part's area over its probe count), m^2.
    """

    corners: np.ndarray
    bary: np.ndarray
    part: np.ndarray
    area: np.ndarray

    def __len__(self) -> int:
        return len(self.corners)

    def positions(self, vertices: np.ndarray) -> np.ndarray:
        """Probe positions on a posed mesh: ``(V, 3) -> (N, 3)`` or ``(T, V, 3) -> (T, N, 3)``."""
        return np.einsum("...nkc,nk->...nc", vertices[..., self.corners, :], self.bary)

    def normals(self, vertices: np.ndarray) -> np.ndarray:
        """Unit outward normals of the probes' triangles on a posed ``(V, 3)`` mesh: ``(N, 3)``."""
        tri = np.asarray(vertices, dtype=np.float64)[self.corners]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def sample_human_probes(
    human: HumanSurfaceMotion, densities: Mapping[str, float], min_points: int, seed: int = 0
) -> HumanProbes:
    """Sample the human rest mesh part by part, each part at the density its joint name resolves to.

    A triangle belongs to the part of the joint with the largest summed skinning weight over its
    three vertices.

    Parameters
    ----------
    human : HumanSurfaceMotion
        The motion (its rest mesh is sampled).
    densities : Mapping[str, float]
        ``{joint-name regex: points per m^2}``.
    min_points : int
        Floor on samples per part.
    seed : int
        Sampling seed.
    """
    faces = np.asarray(human.faces)
    face_part = human.skin_weights[faces].sum(axis=1).argmax(axis=1)
    corners, bary, part, area = [], [], [], []
    for j, name in enumerate(human.joint_names):
        mask = face_part == j
        if not mask.any():
            continue
        sub_faces = faces[mask]
        mesh = trimesh.Trimesh(human.rest_vertices, sub_faces, process=False)
        _, _, fidx, b = sample_surface(mesh, resolve_density(name, densities), min_points, seed=seed + j)
        corners.append(sub_faces[fidx])
        bary.append(b)
        part.append(np.full(len(fidx), j, dtype=np.int64))
        area.append(np.full(len(fidx), mesh.area / len(fidx)))
    return HumanProbes(np.concatenate(corners), np.concatenate(bary), np.concatenate(part), np.concatenate(area))
