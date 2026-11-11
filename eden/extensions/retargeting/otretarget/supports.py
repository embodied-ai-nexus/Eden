"""Untracked supports: the tables and stools a demonstrated object rests on but the capture does not track.

A demonstration such as "put the monitor on the table" tracks the monitor, not the table. Retargeted without the
table, nothing holds the resting object up: within the interaction range the robot's hands pull it down toward
them (up to 22 cm on HiPHI). :func:`find_untracked_supports` reads each support off the object track and gives
it back as a thin plate, which the retargeter takes as a fixed object.

A support is found where the object rests still above the floor with no hand on it (so it is not being held still
in the air), and only if it could be resting on a level surface: one horizontal level of the object (its lowest
one, or the seat of an upside-down chair whose backrest hangs over the table edge) must hold the object's centre
of mass in plan view, with nothing of the object directly below it. A tilted object touching down on one corner,
or one floating a few centimetres over the floor (a capture offset, not furniture), gives no support.

The plate is the support's top only: a block or a slab on legs fills the space where the demonstrator's (and so
the robot's) legs stand against the table, and the non-penetration rows then reject every robot step.
"""

from __future__ import annotations

from dataclasses import dataclass

import igl
import numpy as np
import trimesh
from scipy.spatial import Delaunay

from eden.extensions.retargeting.otretarget.retargeter import InteractionObject
from eden.utils.geom import R_to_quat, axis_angle_from_quat, quat_from_axis_angle, quat_to_R


@dataclass
class Support:
    """A plate standing in for an untracked support.

    Attributes
    ----------
    name : str
        The fixed object's name (``support<i>``).
    height : float
        Height of the plate's top surface (m): the level the object rests on.
    center : np.ndarray
        ``(2,)`` horizontal position of the plate's centre.
    yaw : float
        Plate heading (rad) about the vertical.
    extents : np.ndarray
        ``(3,)`` plate size along its own axes (m); the last is its thickness.
    frames : tuple[int, int]
        The resting segment ``[start, stop)`` the support was read from.
    """

    name: str
    height: float
    center: np.ndarray
    yaw: float
    extents: np.ndarray
    frames: tuple[int, int]

    def pose(self) -> np.ndarray:
        """``(7,)`` ``[pos, quat_wxyz]`` of the plate's centre."""
        z = self.height - 0.5 * self.extents[2]
        return np.r_[self.center[0], self.center[1], z, quat_from_axis_angle(np.array([0.0, 0.0, self.yaw]))]

    def to_dict(self) -> dict:
        """JSON-ready record: name, top height, plate pose ``[pos, quat_wxyz]``, extents and source frames."""
        return {
            "name": self.name,
            "height": float(self.height),
            "pose_wxyz": self.pose().tolist(),
            "extents": [float(e) for e in self.extents],
            "frames": [int(f) for f in self.frames],
        }

    def as_object(self, n_frames: int) -> InteractionObject:
        """The plate as a fixed :class:`InteractionObject` over ``n_frames`` frames."""
        mesh = trimesh.creation.box(extents=self.extents)
        return InteractionObject(self.name, mesh, np.tile(self.pose(), (n_frames, 1)), fixed=True)


def find_untracked_supports(
    obj: InteractionObject,
    hand_points: np.ndarray,
    *,
    min_height: float = 0.15,
    floor_gap: float = 0.05,
    static_speed: float = 0.002,
    static_turn_deg: float = 0.5,
    hand_clearance: float = 0.06,
    min_frames: int = 5,
    level_band: float = 0.015,
    horizontal_deg: float = 18.0,
    com_tolerance: float = 0.01,
    margin: float = 0.02,
    clearance: float = 0.01,
    min_side: float = 0.1,
    thickness: float = 0.03,
) -> list[Support]:
    """The untracked supports ``obj`` rests on in a demonstration (see the module docstring).

    Parameters
    ----------
    obj : InteractionObject
        The tracked object, z-up, floor at ``z = 0``.
    hand_points : np.ndarray
        ``(T, H, 3)`` world points on the demonstrator's hands, same frames as ``obj.poses``.
    min_height : float
        A resting level lower than this is the floor seen through a capture offset, not furniture.
    floor_gap : float
        The object's lowest point must be this far above the floor for a frame to count as resting elevated.
    static_speed, static_turn_deg : float
        Per-frame translation (m) and rotation (deg) below which the object is still.
    hand_clearance : float
        Every hand point must be this far from the object (m): a held object is not resting.
    min_frames : int
        Shortest resting segment.
    level_band : float
        Thickness of one horizontal level of the object (m).
    horizontal_deg : float
        Faces within this angle of horizontal (either facing) make up the levels.
    com_tolerance : float
        How far outside the level's footprint the centre of mass may lie (m).
    margin : float
        The plate reaches this far beyond the level's footprint (m), before clearing hanging parts.
    clearance : float
        Gap kept between the plate and any part of the object below the level (m).
    min_side : float
        A plate narrower than this (after clearing hanging parts) is no support.
    thickness : float
        Plate thickness (m).

    Returns
    -------
    list[Support]
        One per distinct support (resting segments at the same place and height share one).
    """
    poses = np.asarray(obj.poses, dtype=np.float64)
    hand_points = np.asarray(hand_points, dtype=np.float64)
    if hand_points.ndim != 3 or hand_points.shape[0] != len(poses) or hand_points.shape[2] != 3:
        raise ValueError(
            f"find_untracked_supports: hand_points has shape {hand_points.shape}; expected (T, H, 3) with "
            f"T = {len(poses)}, the object's frame count."
        )
    mesh = obj.mesh
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    com = _center_of_mass(mesh)
    rot = quat_to_R(poses[:, 3:])
    pos = poses[:, :3]
    zmin = np.einsum("tab,vb->tva", rot, verts)[:, :, 2].min(axis=1) + pos[:, 2]
    aabb = igl.AABB()
    aabb.init(verts, faces)
    # Dense surface points for the parts hanging below a level: a coarse mesh's vertices miss a face crossing the plate.
    n_samples = int(min(max(mesh.area / 0.005**2, 1000), 50000))
    surface = np.vstack([verts, trimesh.sample.sample_surface_even(mesh, n_samples, seed=0)[0]])
    speed = np.r_[0.0, np.linalg.norm(np.diff(pos, axis=0), axis=1)]
    step = axis_angle_from_quat(R_to_quat(rot[1:] @ rot[:-1].transpose(0, 2, 1)))
    turn = np.r_[0.0, np.degrees(np.linalg.norm(step, axis=1))]
    resting = (zmin > floor_gap) & (speed < static_speed) & (turn < static_turn_deg)
    for t in np.flatnonzero(resting):  # the hand test only where the rest already holds: it is the costly one
        local = np.ascontiguousarray((hand_points[t] - pos[t]) @ rot[t])
        resting[t] = aabb.squared_distance(verts, faces, local)[0].min() > hand_clearance**2

    supports: list[Support] = []
    for start, stop in _runs(resting, min_frames):
        k = (start + stop) // 2
        found = _resting_level(
            verts @ rot[k].T + pos[k],
            faces,
            surface @ rot[k].T + pos[k],
            com @ rot[k].T + pos[k],
            min_height,
            level_band,
            np.cos(np.radians(horizontal_deg)),
            com_tolerance,
            margin,
            clearance,
            min_side,
        )
        if found is None:
            continue
        height, center, yaw, size = found
        if any(abs(s.height - height) < 0.02 and np.linalg.norm(s.center - center) < 0.1 for s in supports):
            continue
        extents = np.array([size[0], size[1], thickness])
        supports.append(Support(f"support{len(supports)}", height, center, yaw, extents, (start, stop)))
    return supports


def _center_of_mass(mesh: trimesh.Trimesh) -> np.ndarray:
    """Volume centroid of a closed mesh; area-weighted surface centroid of an open scan."""
    merged = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.faces), process=True)
    if merged.is_watertight and merged.volume > 0:
        return np.asarray(merged.center_mass, dtype=np.float64)
    area = mesh.area_faces
    return (mesh.triangles_center * area[:, None]).sum(axis=0) / area.sum()


def _runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return [(int(s), int(e)) for s, e in zip(edges[::2], edges[1::2], strict=True) if e - s >= min_len]


def _resting_level(
    world, faces, surface, com, min_height, band, cos_horizontal, com_tolerance, margin, clearance, min_side
):
    """The lowest horizontal level the object can rest on, and a plate for it: ``(height, center, yaw, (sx, sy))``.

    Levels are tried from the bottom up. The first that holds the centre of mass and leaves room for a plate is the
    one the object rests on; if it is lower than ``min_height`` the object rests on the floor (None).
    """
    tri = world[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(normal, axis=1)
    horizontal = np.abs(normal[:, 2]) > cos_horizontal * np.maximum(length, 1e-12)
    horizontal &= length > 1e-12
    face_z = tri[:, :, 2].mean(axis=1)
    for h in np.unique(np.floor(face_z[horizontal] / band)) * band:
        contact = np.unique(world[faces[horizontal & (face_z >= h) & (face_z < h + band)]].reshape(-1, 3), axis=0)
        height = float(contact[:, 2].min())
        if not _inside(contact[:, :2], com[:2], com_tolerance):
            continue
        plate = _plate(contact[:, :2], surface[surface[:, 2] < height - band, :2], com[:2], margin, clearance)
        if plate is None or min(plate[2]) < min_side:
            continue
        # The plate's top: no higher than the lowest object point over it (a slightly tilted object rests on a corner
        # of its level; a plate at the level's height would start inside it).
        center, yaw, size = plate
        axes = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        over = np.all(np.abs((surface[:, :2] - center) @ axes.T) <= 0.5 * size, axis=1)
        height = min(height, float(surface[over, 2].min())) if over.any() else height
        return None if height < min_height else (height, *plate)
    return None


def _inside(points: np.ndarray, p: np.ndarray, tolerance: float) -> bool:
    """Whether ``p`` lies in the convex hull of ``points`` (2-D), or within ``tolerance`` of one of them."""
    spread = np.linalg.svd(points - points.mean(axis=0), compute_uv=False) if len(points) >= 3 else np.zeros(2)
    # Collinear points span no area (Delaunay refuses them): only the distance test applies.
    if spread[1] > 1e-6 * max(spread[0], 1e-12) and Delaunay(points).find_simplex(p[None])[0] >= 0:
        return True
    return bool(np.linalg.norm(points - p, axis=1).min() < tolerance)


def _plate(contact, below, com, margin, clearance):
    """A rectangle over the contact footprint (+``margin``), shrunk off every point of ``below``; keeps ``com``.

    The rectangle starts as the footprint's minimum-area bounding rectangle. Returns ``(center (2,), yaw, (sx, sy))``,
    or None if no rectangle clears ``below`` and still holds ``com``.
    """
    to_local, size = trimesh.bounds.oriented_bounds_2D(contact)

    def local(p):
        return p @ to_local[:2, :2].T + to_local[:2, 2]

    lo, hi = -0.5 * size - margin, 0.5 * size + margin
    c = local(com)
    b = np.unique(np.round(local(below) / 0.005), axis=0) * 0.005  # 5 mm voxels: the shrink is per point
    while len(b):
        q = b[np.all((b > lo - clearance) & (b < hi + clearance), axis=1)]
        if not len(q):
            break
        # Each side's position if it alone cleared every offending point: the valid one losing least area is moved.
        # Parts hanging on several sides need several sides: each then clears the points nearest it.
        alone = [(0, 0, q[:, 0].max() + clearance), (0, 1, q[:, 0].min() - clearance)]
        alone += [(1, 0, q[:, 1].max() + clearance), (1, 1, q[:, 1].min() - clearance)]
        best, best_area = None, -1.0
        for axis, upper, value in alone:
            new_lo, new_hi = lo.copy(), hi.copy()
            (new_hi if upper else new_lo)[axis] = value
            span = new_hi - new_lo
            if np.all(span > 0) and np.all(c >= new_lo) and np.all(c <= new_hi) and np.prod(span) > best_area:
                best, best_area = (new_lo, new_hi), float(np.prod(span))
        if best is not None:
            lo, hi = best
            continue
        moves = np.stack([q[:, 0] - lo[0], hi[0] - q[:, 0], q[:, 1] - lo[1], hi[1] - q[:, 1]], axis=1)
        nearest = np.argmin(moves, axis=1)
        for side in np.unique(nearest):
            axis, upper = divmod(int(side), 2)
            group = q[nearest == side, axis]
            if upper:
                hi[axis] = min(hi[axis], group.min() - clearance)
            else:
                lo[axis] = max(lo[axis], group.max() + clearance)
        if np.any(hi <= lo) or np.any(c < lo) or np.any(c > hi):
            return None
    to_world = np.linalg.inv(to_local)
    center = to_world[:2, :2] @ (0.5 * (lo + hi)) + to_world[:2, 2]
    return center, float(np.arctan2(to_world[1, 0], to_world[0, 0])), hi - lo
