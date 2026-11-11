"""Interaction channels: the surfaces probes read proximity triples ``(d, w, n)`` from (Sec. III-B).

A channel is a rigid surface — the ground or a manipulated object — with, in its own frame,

* a signed distance ``Phi(x)`` and a witness ``W(x)`` (the closest surface point), and
* a geodesic distance ``rho(w; w_hat)`` between two surface points (the placement residual ``r^x``).

For a mesh, ``Phi`` and ``W`` are baked once on a voxel grid around the object (1 cm isotropic by
default, reaching ``max_distance`` past the surface) and read back by trilinear interpolation, so a
query costs no mesh search and the grid never needs recomputing as the object moves. For the ground,
both are affine and exact. Geodesics are all-pairs shortest paths over the channel's surface point
cloud, interpolated to arbitrary witness points through their nearest samples.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import igl
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial import cKDTree

from eden.utils.distribution import install_hint, installed_version


@dataclass
class ChannelQuery:
    """Proximity of query points to a channel, all in the channel frame.

    Attributes
    ----------
    distance : np.ndarray
        ``(N,)`` signed distance ``Phi(x)`` (negative inside).
    witness : np.ndarray
        ``(N, 3)`` witness ``W(x)``, the closest surface point.
    distance_grad : np.ndarray
        ``(N, 3)`` gradient of ``Phi``.
    witness_jac : np.ndarray
        ``(N, 3, 3)`` Jacobian ``dW/dx``.
    valid : np.ndarray
        ``(N,)`` bool, False where the point lies outside the baked region (then ``Phi >= max_distance``
        is all that is known, and the other fields are meaningless).
    """

    distance: np.ndarray
    witness: np.ndarray
    distance_grad: np.ndarray
    witness_jac: np.ndarray
    valid: np.ndarray


class Channel(ABC):
    """A rigid surface probes measure proximity to.

    Parameters
    ----------
    name : str
        The channel's name (the scene entity name for an object, ``"ground"`` for the floor).
    max_distance : float
        ``d_bar``: the distance below which a probe-channel pair is active.
    """

    def __init__(self, name: str, max_distance: float) -> None:
        self.name = name
        self.max_distance = float(max_distance)

    @abstractmethod
    def query(self, points: np.ndarray) -> ChannelQuery:
        """Proximity of ``(N, 3)`` channel-frame points to the surface."""

    @abstractmethod
    def geodesic(
        self, w: np.ndarray, w_hat: np.ndarray, n_w: np.ndarray | None = None, n_hat: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Geodesic distance ``rho(w; w_hat)`` between surface points, and its gradient in ``w``.

        Parameters
        ----------
        w, w_hat : np.ndarray
            ``(N, 3)`` channel-frame surface points.
        n_w, n_hat : np.ndarray, optional
            ``(N, 3)`` outward surface normals at ``w`` and ``w_hat``: which face of a thin part each point is on.

        Returns
        -------
        rho : np.ndarray
            ``(N,)`` distances.
        grad : np.ndarray
            ``(N, 3)`` gradient of ``rho`` with respect to ``w``.
        """

    @abstractmethod
    def exact_distance(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Exact (not grid-interpolated) signed distance and closest point, for evaluation."""

    @abstractmethod
    def exact_query(self, points: np.ndarray) -> ChannelQuery:
        """Like :meth:`query`, from the exact surface (for points beyond the baked region)."""


class PlaneChannel(Channel):
    """The ground: the plane ``z = 0`` of its frame, with ``+z`` outside.

    ``Phi(x) = x_z`` and ``W(x) = (x_x, x_y, 0)`` exactly, at any distance; the geodesic between two
    plane points is their Euclidean distance.
    """

    def query(self, points: np.ndarray) -> ChannelQuery:
        points = np.asarray(points, dtype=np.float64)
        n = len(points)
        witness = points.copy()
        witness[:, 2] = 0.0
        grad = np.zeros((n, 3))
        grad[:, 2] = 1.0
        jac = np.zeros((n, 3, 3))
        jac[:, 0, 0] = 1.0
        jac[:, 1, 1] = 1.0
        return ChannelQuery(points[:, 2].copy(), witness, grad, jac, np.ones(n, dtype=bool))

    def geodesic(
        self, w: np.ndarray, w_hat: np.ndarray, n_w: np.ndarray | None = None, n_hat: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        delta = np.asarray(w, dtype=np.float64) - np.asarray(w_hat, dtype=np.float64)
        delta[:, 2] = 0.0
        rho = np.linalg.norm(delta, axis=1)
        grad = delta / np.maximum(rho, 1e-12)[:, None]
        return rho, grad

    def exact_distance(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float64)
        witness = points.copy()
        witness[:, 2] = 0.0
        return points[:, 2].copy(), witness

    def exact_query(self, points: np.ndarray) -> ChannelQuery:
        return self.query(points)


class MeshChannel(Channel):
    """A rigid mesh, with ``Phi``/``W`` baked on a voxel grid in the mesh frame.

    Parameters
    ----------
    name : str
        The channel's name.
    mesh : trimesh.Trimesh
        The surface, in the channel (object) frame. The sign comes from the generalised winding number, which is
        only right on a closed, consistently oriented surface: on an open scan (a bin) it reads the empty cavity as
        solid and flips sign across the opening, so the distance jumps from +5 to -5 cm between two neighbouring
        points, and walls scanned as zero-thickness sheets have no inside at all. Pass ``solid_voxel`` for such a mesh
        (see :func:`winding_sign_is_reliable` and :func:`watertight_solid`).
    max_distance : float
        ``d_bar``; the grid reaches this far past the mesh's bounding box.
    resolution : float
        Grid spacing (m).
    geodesic_density : float
        Surface samples per m^2 for the geodesic table.
    solid_voxel : float
        If > 0 and the winding-number sign of ``mesh`` is not reliable (:func:`winding_sign_is_reliable`), the channel
        uses :func:`watertight_solid` of it at this voxel size (m) instead; ``source_mesh`` keeps the original. The
        solid lies one voxel outside the scan, so a scan that is already closed keeps its own surface.
    """

    def __init__(
        self,
        name: str,
        mesh: trimesh.Trimesh,
        max_distance: float,
        resolution: float = 0.01,
        geodesic_density: float = 1000.0,
        solid_voxel: float = 0.0,
    ) -> None:
        super().__init__(name, max_distance)
        if not hasattr(igl, "FastWindingNumberBVH"):  # libigl < 2.6.3, as genesis-world installs it without the extra
            raise ImportError(
                f"MeshChannel '{name}' needs libigl >= 2.6.3 (igl.FastWindingNumberBVH, its winding-number tree); the "
                f"installed version is {installed_version('libigl') or 'not installed'}. Install the extra with "
                f"{install_hint('ot-retargeter')}; or upgrade alone with `uv pip install 'libigl>=2.6.3'`."
            )
        if resolution <= 0.0:
            raise ValueError(f"MeshChannel '{name}': resolution={resolution!r} must be > 0 (the paper uses 0.01 m).")
        if solid_voxel < 0.0:
            raise ValueError(
                f"MeshChannel '{name}': solid_voxel={solid_voxel!r} must be >= 0 (0 keeps the mesh as is)."
            )
        self.source_mesh = mesh
        if solid_voxel > 0.0 and not winding_sign_is_reliable(mesh):
            mesh = watertight_solid(mesh, solid_voxel)
        self.mesh = mesh
        self.resolution = float(resolution)
        self._vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
        self._faces = np.ascontiguousarray(mesh.faces, dtype=np.int64)
        # Search trees built once: igl.signed_distance rebuilds both on every call, 6x the cost of the query.
        self._aabb = igl.AABB()
        self._aabb.init(self._vertices, self._faces)
        self._winding = igl.FastWindingNumberBVH()
        self._winding.init(self._vertices, self._faces)

        margin = self.max_distance + 2.0 * self.resolution
        lo = self._vertices.min(axis=0) - margin
        hi = self._vertices.max(axis=0) + margin
        self.origin = lo
        self.shape = np.ceil((hi - lo) / self.resolution).astype(int) + 1
        axes = [self.origin[k] + self.resolution * np.arange(self.shape[k]) for k in range(3)]
        nodes = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        distance, witness = self.exact_distance(nodes)
        self._phi = distance.reshape(*self.shape).astype(np.float32)
        self._witness = witness.reshape(*self.shape, 3).astype(np.float32)

        self.geodesics = SurfaceGeodesics.from_mesh(mesh, geodesic_density)

    def exact_distance(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        distance, _, witness = self._closest(points)
        return distance, witness

    def _closest(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Signed distance (sign from the generalised winding number), closest face and closest point."""
        points = np.ascontiguousarray(points, dtype=np.float64)
        if len(points) == 0:
            return np.zeros(0), np.zeros(0, dtype=np.int64), np.zeros((0, 3))
        squared, face, witness = self._aabb.squared_distance(self._vertices, self._faces, points)
        inside = np.asarray(self._winding.winding_number(points)) > 0.5
        return np.where(inside, -1.0, 1.0) * np.sqrt(squared), np.asarray(face), np.asarray(witness)

    def exact_query(self, points: np.ndarray) -> ChannelQuery:
        """Exact distance, witness and their derivatives.

        The closest point of a triangle mesh slides with the query point only along the feature it lies
        on: ``dW/dx`` is the tangent projector ``I - n n^T`` inside a face, ``t t^T`` on an edge (``t`` its
        direction) and zero at a vertex. Using the face projector everywhere overstates or misdirects the
        witness motion near edges, which a Gauss-Newton step then follows in the wrong direction.
        """
        points = np.ascontiguousarray(points, dtype=np.float64)
        distance, face, witness = self._closest(points)
        n = len(points)
        delta = points - witness
        length = np.linalg.norm(delta, axis=1)
        tri = self._vertices[self._faces[face]]  # (N, 3, 3)
        face_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        face_n /= np.maximum(np.linalg.norm(face_n, axis=1, keepdims=True), 1e-12)
        sign = np.where(distance < 0, -1.0, 1.0)
        grad = np.where((length > 1e-9)[:, None], sign[:, None] * delta / np.maximum(length, 1e-12)[:, None], face_n)
        bary = trimesh.triangles.points_to_barycentric(tri, witness)
        on_zero = bary < 1e-6
        n_zero = on_zero.sum(axis=1)
        jac = np.eye(3)[None] - face_n[:, :, None] * face_n[:, None, :]
        edge = n_zero == 1
        if edge.any():
            # The edge joins the two vertices whose barycentric weight is non-zero.
            k = np.argmax(on_zero[edge], axis=1)
            a = tri[edge][np.arange(edge.sum()), (k + 1) % 3]
            b = tri[edge][np.arange(edge.sum()), (k + 2) % 3]
            t = (b - a) / np.maximum(np.linalg.norm(b - a, axis=1, keepdims=True), 1e-12)
            jac[edge] = t[:, :, None] * t[:, None, :]
        jac[n_zero >= 2] = 0.0
        return ChannelQuery(distance, witness, grad, jac, np.ones(n, dtype=bool))

    def query(self, points: np.ndarray) -> ChannelQuery:
        points = np.asarray(points, dtype=np.float64)
        u = (points - self.origin) / self.resolution
        upper = self.shape - 1
        valid = np.all((u >= 0.0) & (u <= upper), axis=1)
        i0 = np.clip(np.floor(u).astype(int), 0, upper - 1)
        f = np.clip(u - i0, 0.0, 1.0)

        n = len(points)
        distance = np.zeros(n)
        grad = np.zeros((n, 3))
        witness = np.zeros((n, 3))
        jac = np.zeros((n, 3, 3))
        for corner in range(8):
            bits = np.array([(corner >> 2) & 1, (corner >> 1) & 1, corner & 1])
            idx = i0 + bits
            # Trilinear weight and its derivative along each axis.
            factors = np.where(bits[None, :] == 1, f, 1.0 - f)
            dfactors = np.where(bits == 1, 1.0, -1.0)
            weight = factors.prod(axis=1)
            dweight = (
                np.stack(
                    [
                        dfactors[0] * factors[:, 1] * factors[:, 2],
                        dfactors[1] * factors[:, 0] * factors[:, 2],
                        dfactors[2] * factors[:, 0] * factors[:, 1],
                    ],
                    axis=1,
                )
                / self.resolution
            )
            phi_c = self._phi[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float64)
            w_c = self._witness[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float64)
            distance += weight * phi_c
            grad += dweight * phi_c[:, None]
            witness += weight[:, None] * w_c
            jac += w_c[:, :, None] * dweight[:, None, :]
        return ChannelQuery(distance, witness, grad, jac, valid)

    def geodesic(
        self,
        w: np.ndarray,
        w_hat: np.ndarray,
        n_w: np.ndarray | None = None,
        n_hat: np.ndarray | None = None,
        hat: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """See :meth:`Channel.geodesic`; ``hat`` is :meth:`SurfaceGeodesics.endpoint_weights` at ``w_hat``, if known."""
        return self.geodesics.distance(w, w_hat, n_w, n_hat, hat=hat)


def winding_sign_is_reliable(mesh: trimesh.Trimesh, offset: float = 5e-4, samples: int = 4000) -> bool:
    """Whether the generalised winding number of ``mesh`` gives the right inside/outside sign at its surface.

    True when, with duplicate vertices merged, the surface has no boundary edge and a consistent orientation, and at
    least 98% of surface samples read inside ``offset`` behind their face and outside ``offset`` in front of it. A
    scan stored as loose triangles (every triangle with its own vertices, as the OMOMO objects are) is closed once
    merged and passes; ``Trimesh.is_watertight`` on the unmerged mesh reads it as open. An open bin, or walls scanned
    as zero-thickness sheets (no inside behind the face), fail.
    """
    merged = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.faces), process=True)
    _, uses = np.unique(np.sort(merged.edges, axis=1), axis=0, return_counts=True)
    if (uses == 1).any() or not merged.is_winding_consistent:
        return False
    points, face = trimesh.sample.sample_surface(mesh, samples, seed=0)
    normal = mesh.face_normals[face]
    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int64)
    inside = np.asarray(igl.fast_winding_number(vertices, faces, points - offset * normal)) > 0.5
    outside = np.asarray(igl.fast_winding_number(vertices, faces, points + offset * normal)) < 0.5
    return bool(inside.mean() >= 0.98 and outside.mean() >= 0.98)


def watertight_solid(mesh: trimesh.Trimesh, voxel: float, cache: bool = True) -> trimesh.Trimesh:
    """A closed surface around the solid an open or fragmented scan stands for.

    Space within ``voxel`` of the scan is a shell (a single-sheet wall gets that thickness); free space reachable
    from outside the bounding box without crossing the shell is outside, so the cavity of an open bin stays free
    space, while space the shell encloses (the inside of a closed box) is solid. The surface is the zero level of
    ``distance - voxel`` over outside and shell space and ``-distance`` inside, extracted by marching cubes on a grid
    of spacing ``voxel``: closed, and ``voxel`` outside the scanned surface. The result is cached by mesh content.
    """
    import hashlib

    from scipy import ndimage

    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int64)
    key = hashlib.sha1(vertices.tobytes() + faces.tobytes() + np.float64(voxel).tobytes()).hexdigest()[:20]
    path = None
    if cache:
        from eden.utils.misc import get_cache_dir

        folder = get_cache_dir("otretarget", "watertight")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{key}.npz")
        if os.path.exists(path):
            data = np.load(path)
            return trimesh.Trimesh(data["vertices"], data["faces"], process=False)
    try:
        from skimage.measure import marching_cubes
    except ImportError as exc:  # pragma: no cover - exercised only without the optional dependency
        raise ImportError(
            "watertight_solid extracts the closed surface with scikit-image: install it with `pip install "
            "scikit-image` (the `ot-retargeter` extra of eden)."
        ) from exc
    pad = 4.0 * voxel
    lo = vertices.min(axis=0) - pad
    shape = np.ceil((vertices.max(axis=0) + pad - lo) / voxel).astype(int) + 1
    axes = [lo[k] + voxel * np.arange(shape[k]) for k in range(3)]
    tree = igl.AABB()
    tree.init(vertices, faces)
    # Slab by slab along x: the full node array of a 1.7 m lamp at 4 mm is 12 M points (300 MB in float64).
    yz = np.stack(np.meshgrid(axes[1], axes[2], indexing="ij"), axis=-1).reshape(-1, 2)
    distance = np.empty(shape, dtype=np.float32)
    for i, x in enumerate(axes[0]):
        slab = np.concatenate([np.full((len(yz), 1), x), yz], axis=1)
        distance[i] = np.sqrt(tree.squared_distance(vertices, faces, slab)[0]).reshape(shape[1], shape[2])
    free = distance >= voxel
    labels, _ = ndimage.label(free)
    rim = np.concatenate([labels[[0, -1]].ravel(), labels[:, [0, -1]].ravel(), labels[:, :, [0, -1]].ravel()])
    outside = np.isin(labels, np.unique(rim[rim > 0]))
    field = np.where(outside | ~free, distance - voxel, -distance)
    verts, tris, _, _ = marching_cubes(field, level=0.0, spacing=(voxel, voxel, voxel))
    solid = trimesh.Trimesh(verts + lo, tris, process=True)
    solid.fix_normals()
    # Closed means no boundary edge: every edge on an even number of faces. Marching cubes puts an edge on four
    # faces where two cells touch only along it, which ``is_watertight`` (exactly two) refuses but which encloses.
    _, uses = np.unique(np.sort(solid.edges, axis=1), axis=0, return_counts=True)
    if (uses % 2).any():
        raise RuntimeError(
            f"watertight_solid: the extracted surface of a {len(faces)}-face mesh has {int((uses % 2).sum())} boundary "
            f"edges at voxel={voxel}; a surface strictly inside the grid has none -- report the mesh."
        )
    if path is not None:
        tmp = f"{path}.{os.getpid()}.tmp.npz"
        np.savez_compressed(tmp, vertices=solid.vertices, faces=solid.faces)
        os.replace(tmp, path)
    return solid


class SurfaceGeodesics:
    """All-pairs geodesic distances over a surface point cloud, interpolated to arbitrary surface points.

    The cloud is connected into a graph by k-nearest-neighbour edges between samples whose normals
    agree (so a thin plate's two faces are not short-circuited through its thickness); components left
    disconnected — separate shells of one object — are bridged at their closest pair. Distances between
    samples are Dijkstra shortest paths.

    For two arbitrary surface points the distance is interpolated *continuously* — the solver's
    Gauss-Newton model must describe the cost it is accepted against, and a nearest-sample minimum
    jumps whenever the nearest set changes. Each endpoint spreads over the samples within
    ``support * spacing`` with compactly supported weights ``(1 - r / R)^2``; the distance is the sample
    table interpolated at both ends, ``sum_ab alpha_a(w) alpha_b(w_hat) G[a, b]``; and, for endpoints a few
    spacings apart, it blends by a smoothstep into the straight segment, which is what a geodesic reduces
    to locally.

    Parameters
    ----------
    points : np.ndarray
        ``(N, 3)`` surface samples.
    normals : np.ndarray
        ``(N, 3)`` unit outward normals at the samples.
    spacing : float
        The mean sample spacing (m).
    k : int
        Graph neighbours per sample.
    """

    def __init__(self, points: np.ndarray, normals: np.ndarray, spacing: float, k: int = 10) -> None:
        self.points = np.asarray(points, dtype=np.float64)
        self.normals = np.asarray(normals, dtype=np.float64)
        self.spacing = float(spacing)
        n = len(self.points)
        self._tree = cKDTree(self.points)
        # The k nearest samples *of the same facing*: on a part thinner than the spacing about half the plain
        # nearest neighbours lie on the opposite face, and dropping them left the face so sparsely connected that
        # paths across it ran up to 10x the straight line.
        k_eff = min(4 * k + 1, n)
        dist, nbr = self._tree.query(self.points, k=k_eff)
        dist, nbr = dist[:, 1:], nbr[:, 1:]
        ok = (np.einsum("na,nka->nk", self.normals, self.normals[nbr]) > -0.5) & (dist <= 4.0 * self.spacing)
        ok &= np.cumsum(ok, axis=1) <= k
        rows = np.repeat(np.arange(n), nbr.shape[1])[ok.reshape(-1)]
        cols, lengths = nbr[ok], dist[ok]
        rows, cols, lengths = list(rows), list(cols), list(lengths)

        # Bridge disconnected shells at their closest pair until the graph is connected.
        while True:
            graph = coo_matrix((lengths, (rows, cols)), shape=(n, n)).tocsr()
            n_comp, labels = connected_components(graph, directed=False)
            if n_comp == 1:
                break
            sizes = np.bincount(labels)
            comp = int(np.argmin(sizes))
            inside = np.flatnonzero(labels == comp)
            outside = np.flatnonzero(labels != comp)
            d_out, j_out = cKDTree(self.points[outside]).query(self.points[inside])
            best = int(np.argmin(d_out))
            rows.append(int(inside[best]))
            cols.append(int(outside[j_out[best]]))
            lengths.append(float(max(d_out[best], 1e-9)))
        self.table = shortest_path(graph, method="D", directed=False).astype(np.float32)

    @property
    def tree(self) -> cKDTree:
        """k-d tree of the samples."""
        return self._tree

    @classmethod
    def from_mesh(cls, mesh: trimesh.Trimesh, density: float, seed: int = 0) -> SurfaceGeodesics:
        """Sample ``mesh`` at ``density`` points per m^2 and build the geodesic table on the samples."""
        n = max(int(np.ceil(mesh.area * density)), 16)
        points, faces = trimesh.sample.sample_surface_even(mesh, n, seed=seed)
        if len(points) < 16:
            points, faces = trimesh.sample.sample_surface(mesh, n, seed=seed)
        return cls(points, mesh.face_normals[faces], spacing=1.0 / np.sqrt(density))

    def _weights(self, x: np.ndarray, k: int, support: float, normal: np.ndarray | None = None):
        """Neighbour indices and distances ``(M, k)``, normalised weights ``(M, k)``, their gradients ``(M, k, 3)``
        in ``x``, and the offsets ``x - sample`` ``(M, k, 3)``.

        With ``normal`` (the outward normal at each ``x``), samples on the far face of a part thinner than the
        support are left out: full weight up to 90 deg from ``normal``, none beyond 120 deg, a smoothstep between.
        Otherwise a point on a 1 cm chair back interpolates the table rows of both faces, whose distances to a
        target differ by the way round the edge (tens of cm over 1 cm: a gradient in the thousands), and the
        interpolated normal of opposite faces nearly cancels. The factor is held fixed in the gradient: it is 1 or
        0 except on the rim of a part.
        """
        radius = support * self.spacing
        k = min(k, len(self.points))
        dist, idx = self._tree.query(x, k=k)
        dist, idx = dist.reshape(len(x), k), idx.reshape(len(x), k)
        # A sample outside the support has zero weight and zero gradient, so membership changes are continuous.
        base = np.clip(1.0 - dist / radius, 0.0, None)
        face = np.ones_like(dist)
        if normal is not None:
            cos = np.einsum("mka,ma->mk", self.normals[idx], np.asarray(normal, dtype=np.float64))
            tf = np.clip((cos + 0.5) / 0.5, 0.0, 1.0)
            face = tf * tf * (3.0 - 2.0 * tf)
        u = face * base**2
        lone = u.sum(axis=1) <= 0.0
        if lone.any():
            # No sample within the support (a point off the sampled surface, or only far-face samples there):
            # fall back to the nearest sample on the point's own face, else the nearest one.
            first = np.argmax(face[lone] > 0.0, axis=1)
            first[~(face[lone] > 0.0).any(axis=1)] = 0
            u[lone] = 0.0
            u[np.flatnonzero(lone), first] = 1.0
        diff = x[:, None, :] - self.points[idx]  # (M, k, 3)
        du = (-2.0 / radius) * (face * base)[:, :, None] * diff / np.maximum(dist, 1e-12)[:, :, None]
        du[lone] = 0.0
        total = u.sum(axis=1, keepdims=True)
        alpha = u / total
        dalpha = (du - alpha[:, :, None] * du.sum(axis=1, keepdims=True)) / total[:, :, None]
        return idx, dist, alpha, dalpha, diff

    def endpoint_weights(
        self, x: np.ndarray, normal: np.ndarray | None = None, k: int = 16, support: float = 1.5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Neighbour indices ``(M, k)`` and normalised weights ``(M, k)`` of the table at the surface points ``x``.

        This is the ``w_hat`` end of :meth:`distance`. Each row depends on its own point only, so a caller whose
        targets stay fixed over several :meth:`distance` calls computes it once and passes it as ``hat``.
        """
        idx, _, alpha, _, _ = self._weights(np.asarray(x, dtype=np.float64), k, support, normal)
        return idx, alpha

    def distance(
        self,
        w: np.ndarray,
        w_hat: np.ndarray,
        n_w: np.ndarray | None = None,
        n_hat: np.ndarray | None = None,
        k: int = 16,
        support: float = 1.5,
        blend: tuple[float, float] = (2.0, 4.0),
        soften: float = 0.05,
        hat: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Interpolated geodesic distance and its gradient in ``w`` (see the class docstring).

        Parameters
        ----------
        w, w_hat : np.ndarray
            ``(M, 3)`` surface points.
        n_w, n_hat : np.ndarray, optional
            ``(M, 3)`` outward normals at ``w`` and ``w_hat``; each endpoint then interpolates over the samples of
            its own face only (see :meth:`_weights`). Without them a point near a thin part's rim blends both faces.
        k : int
            Neighbours examined per endpoint (must cover the support).
        support : float
            Weight support radius, in sample spacings.
        soften : float
            ``eps`` of the softened end normals (see the comment in the body): bounds the agreement's gradient by
            ``1 / eps`` where the interpolated normal cancels, and shifts it little elsewhere.
        blend : tuple[float, float]
            Chord lengths (in spacings) over which the distance blends from the straight segment (below)
            to the graph distance (above).
        hat : tuple[np.ndarray, np.ndarray], optional
            :meth:`endpoint_weights` at ``w_hat`` (with ``n_hat``, ``k`` and ``support``), computed here if None.
        """
        w = np.asarray(w, dtype=np.float64)
        w_hat = np.asarray(w_hat, dtype=np.float64)
        ia, _, aa, daa, _ = self._weights(w, k, support, n_w)
        ib, ab = self.endpoint_weights(w_hat, n_hat, k, support) if hat is None else hat
        # Interpolated outward normals at both ends; the straight segment is a valid local geodesic only on a
        # consistently oriented patch (not across a thin plate, whose faces are close but opposite). Each is divided by
        # sqrt(|sum alpha n|^2 + eps^2), not its length: at the narrow end of a thin slat the interpolation averages the
        # slat's two opposite sides to nearly nothing (|sum alpha n| = 4e-6), and divided by that the agreement
        # gradient reached 8e5. The softened normal is short there, so the agreement is small and the graph distance
        # is used. (Holding the caller's normals fixed instead left the gradient inexact wherever the agreement blends,
        # and a cold first frame wandered off.)
        eps = soften
        raw_w = (aa[:, :, None] * self.normals[ia]).sum(axis=1)
        raw_h = (ab[:, :, None] * self.normals[ib]).sum(axis=1)
        len_w = np.sqrt((raw_w * raw_w).sum(axis=1) + eps * eps)
        unit_w = raw_w / len_w[:, None]
        unit_h = raw_h / np.sqrt((raw_h * raw_h).sum(axis=1) + eps * eps)[:, None]
        agree = (unit_w * unit_h).sum(axis=1)
        # d(agree)/dw = ((I - u u^T) / L) (sum_a n_a dalpha_a^T) applied to n_h,
        # with u = raw / L, L = sqrt(|raw|^2 + eps^2)
        draw = np.einsum("mka,mkb->mab", self.normals[ia], daa)  # d raw_w / dw
        proj_h = unit_h - agree[:, None] * unit_w
        d_agree = np.einsum("ma,mab->mb", proj_h, draw) / len_w[:, None]
        ta = np.clip((agree - 0.2) / 0.4, 0.0, 1.0)
        h = ta * ta * (3.0 - 2.0 * ta)
        dh = np.where((agree > 0.2) & (agree < 0.6), 6.0 * ta * (1.0 - ta) / 0.4, 0.0)[:, None] * d_agree
        # The sample table interpolated at both endpoints: exact at the samples, smooth in between.
        to_b = (self.table[ia[:, :, None], ib[:, None, :]] * ab[:, None, :]).sum(axis=2)  # (M, k)
        graph = (aa * to_b).sum(axis=1)
        g_graph = (daa * to_b[:, :, None]).sum(axis=1)

        chord_vec = w - w_hat
        chord = np.linalg.norm(chord_vec, axis=1)
        g_chord = chord_vec / np.maximum(chord, 1e-12)[:, None]
        lo, hi = blend[0] * self.spacing, blend[1] * self.spacing
        tt = np.clip((chord - lo) / (hi - lo), 0.0, 1.0)
        near = 1.0 - tt * tt * (3.0 - 2.0 * tt)  # 1 for a short chord, 0 beyond ``blend[1]``
        dnear = -np.where((chord > lo) & (chord < hi), 6.0 * tt * (1.0 - tt) / (hi - lo), 0.0)[:, None] * g_chord
        # Weight of the straight segment: a short chord between agreeing normals.
        c = near * h
        dc = dnear * h[:, None] + near[:, None] * dh
        rho = c * chord + (1.0 - c) * graph
        grad = c[:, None] * g_chord + (1.0 - c)[:, None] * g_graph + (chord - graph)[:, None] * dc
        return rho, grad
