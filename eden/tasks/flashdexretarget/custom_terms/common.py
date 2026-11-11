"""The hand / object state every FlashDexRetarget term reads, and the parameters they share.

:class:`HandObjectState` resolves the palms, fingertips, finger bodies, contact links, objects and their SDF
grids and surface points once per env; :class:`HandTermMixin` declares the parameters (``hand_model``,
``sides``, ``objects``, ``dataset``) and builds it. Imported by the per-manager term modules in this package.
"""

from __future__ import annotations

import weakref
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch
import torch.nn.functional as F

from eden.tasks.flashdexretarget.assets import motion_summary, robot_spec
from eden.utils.common import ConfigurableMixin
from eden.utils.geom import inv_transform_by_quat, quat_mul, transform_by_quat
from eden.utils.string import term_subject

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


#: SDF grid of every object (``eden retarget flashdex kinematic``'s ``sdf`` defaults; must match them).
SDF_EXTENT, SDF_GRID_N = 0.30, 128


def _object_geometry(urdfs: tuple[str, ...], n_points: int, n_pool: int, device) -> dict[str, torch.Tensor]:
    """Per object slot: the SDF grid ``(S, 4, N, N, N)`` and surface keypoints ``(S, P, 3)``, in the mesh-origin
    frame of the slot's object (scale applied).
    """
    import trimesh

    from eden.extensions.retargeting.flashdexretarget.sdf import bake_object_sdf_grid

    grids, points = [], []
    for urdf in urdfs:
        scale = float(_urdf_link(urdf).find("visual/geometry/mesh").get("scale").split()[0])
        visual = Path(urdf).parent / "visual.obj"
        grids.append(bake_object_sdf_grid(str(visual), scale, SDF_EXTENT, SDF_GRID_N, device=str(device)))
        mesh = trimesh.load(visual, force="mesh", process=False, skip_materials=True)
        pool = _surface_pool(mesh, n_pool) * scale
        points.append(torch.as_tensor(pool[_farthest_point_sample(pool, n_points)], dtype=torch.float32))
    return {"sdf": torch.stack(grids).to(device), "points": torch.stack(points).to(device)}


#: Per env, the object geometry its terms share, keyed by (slot URDFs, keypoint count). Every term reads the same
#: grids (one 4 x 128^3 float32 grid, 33.5 MB, per object slot), so they are loaded once per env rather than once
#: per term; held weakly, they go with the env.
_SHARED_GEOMETRY: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _shared_object_geometry(env: EnvBase, urdfs: list[str], n_points: int) -> dict[str, torch.Tensor]:
    """:func:`_object_geometry` of ``urdfs``, loaded once per ``env``."""
    per_env = _SHARED_GEOMETRY.setdefault(env, {})
    key = (tuple(urdfs), n_points)
    if key not in per_env:
        per_env[key] = _object_geometry(key[0], n_points, 2048, env.device)
    return per_env[key]


def _urdf_link(urdf: str):
    import xml.etree.ElementTree as ET

    return ET.parse(urdf).getroot().find("link")


def _object_inertia(urdfs: list[str], device) -> torch.Tensor:
    """Per object slot, the URDF's inertia about the com ``(S, 3, 3)``, in the object's axes."""
    inertia = []
    for urdf in urdfs:
        i = _urdf_link(urdf).find("inertial/inertia").attrib
        inertia.append(
            [
                [float(i["ixx"]), float(i["ixy"]), float(i["ixz"])],
                [float(i["ixy"]), float(i["iyy"]), float(i["iyz"])],
                [float(i["ixz"]), float(i["iyz"]), float(i["izz"])],
            ]
        )
    return torch.tensor(inertia, dtype=torch.float32, device=device)


def _surface_pool(mesh, n: int) -> np.ndarray:
    """Deterministic area-uniform points on the faces: even spacing, topped up area-weighted (FlashDexRetarget)."""
    import trimesh

    pts, _ = trimesh.sample.sample_surface_even(mesh, n, seed=0)
    if len(pts) < n:
        extra, _ = trimesh.sample.sample_surface(mesh, n - len(pts), seed=1)
        pts = np.vstack([pts, extra])
    return np.asarray(pts[:n], dtype=np.float32)


def _farthest_point_sample(pts: np.ndarray, n: int) -> np.ndarray:
    """Greedy farthest-point sampling, starting from the point farthest from the centroid; ``(n,)`` indices."""
    if len(pts) <= n:
        return np.concatenate([np.arange(len(pts)), np.full(n - len(pts), len(pts) - 1)])
    idx = np.empty(n, dtype=np.int64)
    idx[0] = int(np.argmax(np.linalg.norm(pts - pts.mean(0), axis=1)))
    dist = np.linalg.norm(pts - pts[idx[0]], axis=1)
    for k in range(1, n):
        idx[k] = int(np.argmax(dist))
        dist = np.minimum(dist, np.linalg.norm(pts - pts[idx[k]], axis=1))
    return idx


class HandObjectState:
    """Per-hand reads of the robot, its objects and their references, resolved once in a term's ``build()``.

    Every method returns tensors stacked over ``sides`` on dim 1: ``(N, S, ...)``.

    Parameters
    ----------
    env : EnvBase
        The built env.
    robot : str
        The FlashDexRetarget robot (``xhand`` / ``sharpa``).
    sides : list[str]
        Hand sides, in the motion file's order.
    objects : dict[str, str]
        Side -> the object entity that hand holds (one shared entity, or one per side).
    dataset : str
        The ``motion.pt``; its object slots give the object geometry.
    entity_name : str
        The robot entity.
    geometry : bool
        Load the object SDF grids and surface points (only the terms that query them need to; shared by every term
        of the env). The inertia is always read.
    n_points : int
        Surface keypoints per object, when ``geometry``.
    """

    def __init__(
        self,
        env: EnvBase,
        robot: str,
        sides: list[str],
        objects: dict[str, str],
        dataset: str,
        entity_name: str = "robot",
        geometry: bool = False,
        n_points: int = 128,
    ) -> None:
        self.env, self.sides, self.device = env, list(sides), env.device
        spec = robot_spec(robot, tuple(sides))
        self.spec = spec
        missing = [s for s in self.sides if s not in objects]
        if missing:
            raise ValueError(f"objects names no object entity for hand(s) {missing}; got {objects}.")
        self.objects = {s: objects[s] for s in self.sides}
        self.entity_name = entity_name
        robot_entity = env.entities[entity_name]
        for name in set(self.objects.values()):
            env.entities[name]  # fail fast on a missing object entity

        def links(names) -> torch.Tensor:
            _, idx = robot_entity.find_named_links_idx_local(list(names), preserve_order=True)
            if len(idx) != len(names):
                raise ValueError(
                    f"Robot entity {entity_name!r} lacks links {sorted(set(names) - set(robot_entity.links_name))}; "
                    "keep them with RobotOptions.links_to_keep."
                )
            return torch.tensor(idx, dtype=torch.long, device=self.device)

        self._palm = links([spec.palm_link[s] for s in self.sides])
        self._palm_quat = torch.tensor([spec.palm_quat[s] for s in self.sides], device=self.device)
        self._tips = links([n for s in self.sides for n in spec.tip_links[s]])
        self._tip_offsets = torch.tensor(
            [o for s in self.sides for o in spec.tip_offsets[s]], dtype=torch.float32, device=self.device
        )
        self._bodies = links([r for s in self.sides for r, _ in spec.body_map[s]])
        self._contacts = links([n for s in self.sides for n in spec.contact_links[s]])
        self.n_tips, self.n_bodies = len(spec.tip_links[self.sides[0]]), len(spec.body_map[self.sides[0]])
        self.n_contacts = len(spec.contact_links[self.sides[0]])
        self.contact_margins = torch.tensor(
            [spec.contact_margins[s] for s in self.sides], dtype=torch.float32, device=self.device
        )
        from eden.extensions.retargeting.flashdexretarget.clips import MANO_JOINT_NAMES, FINGERS

        self._mano_tips = torch.tensor(
            [MANO_JOINT_NAMES.index(f"{f}_tip") for f in FINGERS], dtype=torch.long, device=self.device
        )
        self._mano_bodies = torch.tensor(
            [MANO_JOINT_NAMES.index(m) for _, m in spec.body_map[self.sides[0]]], dtype=torch.long, device=self.device
        )
        for s in self.sides:
            env.motion_manager.data[f"mano_{s}"]  # fail fast on an untracked reference

        # Object slots: the variant each env holds (one per GroupedEntityOptions entry, in motion.pt slot order).
        summary = motion_summary(dataset)
        self.slot = {}
        for s in self.sides:
            obj = env.entities[self.objects[s]]
            if obj.is_heterogeneous:
                self.slot[s] = obj.env_variant_idx.to(self.device)
            else:
                self.slot[s] = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        self._inertia = {s: _object_inertia(summary.slot_urdfs[self.objects[s]], self.device) for s in self.sides}
        self.geometry: dict[str, dict[str, torch.Tensor]] = {}
        if geometry:
            for s in self.sides:
                self.geometry[s] = _shared_object_geometry(env, summary.slot_urdfs[self.objects[s]], n_points)
            self._sdf_plan = {
                s: self._grid_plan(self.slot[s], len(summary.slot_urdfs[self.objects[s]])) for s in self.sides
            }

    # ---------------------------------------------------------------- robot

    def _robot(self):
        return self.env.entities[self.entity_name]

    def palm_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Palm position ``(N, S, 3)`` and orientation ``(N, S, 4)``."""
        r = self._robot()
        pos = r.get_links_pos(ls_idx_local=self._palm, relative=True)
        quat = r.get_links_quat(ls_idx_local=self._palm, relative=True)
        return pos, quat_mul(quat, self._palm_quat.expand_as(quat))

    def palm_vel(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Palm linear and angular velocity, world frame, ``(N, S, 3)`` each."""
        r = self._robot()
        return r.get_links_vel(ls_idx_local=self._palm, relative=True), r.get_links_ang(ls_idx_local=self._palm)

    def _per_side(self, x: torch.Tensor, k: int) -> torch.Tensor:
        return x.reshape(x.shape[0], len(self.sides), k, *x.shape[2:])

    def _tip_lever(self) -> torch.Tensor:
        """Each fingertip site's offset from its link origin, world frame ``(N, S*5, 3)``."""
        quat = self._robot().get_links_quat(ls_idx_local=self._tips, relative=True)
        return transform_by_quat(self._tip_offsets.expand(quat.shape[0], -1, -1), quat)

    def tips_pos(self) -> torch.Tensor:
        """Fingertip (tracking site) positions ``(N, S, 5, 3)``."""
        pos = self._robot().get_links_pos(ls_idx_local=self._tips, relative=True) + self._tip_lever()
        return self._per_side(pos, self.n_tips)

    def tips_vel(self) -> torch.Tensor:
        """Fingertip (tracking site) linear velocities ``(N, S, 5, 3)``."""
        r = self._robot()
        vel = r.get_links_vel(ls_idx_local=self._tips, relative=True)
        vel = vel + torch.cross(r.get_links_ang(ls_idx_local=self._tips), self._tip_lever(), dim=-1)
        return self._per_side(vel, self.n_tips)

    def bodies_pos(self) -> torch.Tensor:
        """Mapped finger-body positions ``(N, S, M, 3)``."""
        return self._per_side(self._robot().get_links_pos(ls_idx_local=self._bodies, relative=True), self.n_bodies)

    def bodies_vel(self) -> torch.Tensor:
        """Mapped finger-body linear velocities ``(N, S, M, 3)``."""
        return self._per_side(self._robot().get_links_vel(ls_idx_local=self._bodies, relative=True), self.n_bodies)

    def contact_links_pos(self) -> torch.Tensor:
        """Contact-link origins ``(N, S, C, 3)``."""
        pos = self._robot().get_links_pos(ls_idx_local=self._contacts, relative=True)
        return self._per_side(pos, self.n_contacts)

    def contact_links_force(self) -> torch.Tensor:
        """Net contact force on every contact link ``(N, S, C, 3)``, world frame (all contacts, as PhysX reports)."""
        force = self._robot().get_links_net_contact_force()[:, self._contacts]
        return self._per_side(force, self.n_contacts)

    # ---------------------------------------------------------------- objects

    def object_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Object positions ``(N, S, 3)`` and orientations ``(N, S, 4)`` (mesh-origin frame)."""
        poses = {
            n: (self.env.entities[n].get_pos(relative=True), self.env.entities[n].get_quat(relative=True))
            for n in set(self.objects.values())
        }
        return (
            torch.stack([poses[self.objects[s]][0] for s in self.sides], 1),
            torch.stack([poses[self.objects[s]][1] for s in self.sides], 1),
        )

    def object_vel(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Object linear velocity at its mesh origin and angular velocity, world frame, ``(N, S, 3)`` each."""
        vels = {}
        for n in set(self.objects.values()):
            e = self.env.entities[n]
            vels[n] = (e.get_links_vel(ls_idx_local=[0], relative=True)[:, 0], e.get_ang())
        return (
            torch.stack([vels[self.objects[s]][0] for s in self.sides], 1),
            torch.stack([vels[self.objects[s]][1] for s in self.sides], 1),
        )

    # ---------------------------------------------------------------- references

    def _ref(self, name: str):
        return self.env.motion_manager.data[name]

    def mano_wrist(self, frame: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """MANO wrist position ``(N, S, 3)`` and orientation ``(N, S, 4)`` at reference ``frame``."""
        refs = [self._ref(f"mano_{s}") for s in self.sides]
        return torch.stack([r.base_pos_w[:, frame] for r in refs], 1), torch.stack(
            [r.base_quat_w[:, frame] for r in refs], 1
        )

    def mano_joints(self, frame: int = 0, field: str = "links_pos_w") -> torch.Tensor:
        """The 20 MANO joints ``(N, S, 20, 3)`` (``links_pos_w`` or ``links_lin_vel_w``) at reference ``frame``."""
        return torch.stack([getattr(self._ref(f"mano_{s}"), field)[:, frame] for s in self.sides], 1)

    def mano_tips(self, frame: int = 0, field: str = "links_pos_w") -> torch.Tensor:
        return self.mano_joints(frame, field)[:, :, self._mano_tips]

    def mano_bodies(self, frame: int = 0, field: str = "links_pos_w") -> torch.Tensor:
        return self.mano_joints(frame, field)[:, :, self._mano_bodies]

    def ref_object(self, frame: int = 0) -> dict[str, torch.Tensor]:
        """Reference object ``pos`` / ``quat`` / ``lin_vel`` / ``ang_vel``, ``(N, S, ·)``, at ``frame``."""
        refs = [self._ref(self.objects[s]) for s in self.sides]
        return {
            "pos": torch.stack([r.base_pos_w[:, frame] for r in refs], 1),
            "quat": torch.stack([r.base_quat_w[:, frame] for r in refs], 1),
            "lin_vel": torch.stack([r.base_lin_vel_w[:, frame] for r in refs], 1),
            "ang_vel": torch.stack([r.base_ang_vel_w[:, frame] for r in refs], 1),
        }

    # ---------------------------------------------------------------- geometry

    def surface_points(self, side_idx: int) -> torch.Tensor:
        """Each env's object surface keypoints ``(N, P, 3)``, object frame."""
        s = self.sides[side_idx]
        return self.geometry[s]["points"][self.slot[s]]

    def inertia(self, side_idx: int) -> torch.Tensor:
        """Each env's object inertia about its com ``(N, 3, 3)``, object axes."""
        s = self.sides[side_idx]
        return self._inertia[s][self.slot[s]]

    def _grid_plan(self, slot: torch.Tensor, n_slots: int) -> dict:
        """Envs grouped by slot for one batched ``grid_sample`` (the assignment is fixed at build, so this is too)."""
        order = torch.argsort(slot, stable=True)
        counts = torch.bincount(slot, minlength=n_slots)
        starts = torch.cumsum(torch.cat([counts.new_zeros(1), counts[:-1]]), 0)
        within = torch.arange(slot.shape[0], device=slot.device) - starts[slot[order]]
        return {"order": order, "within": within, "n_max": int(counts.max()), "slot": slot[order]}

    def sdf(self, side_idx: int, points_w: torch.Tensor, obj_pos: torch.Tensor, obj_quat: torch.Tensor):
        """Signed distance ``(N, K)`` and unit outward gradient ``(N, K, 3)`` (world) of each env's object at world
        points ``(N, K, 3)``, the object at ``obj_pos`` ``(N, 3)`` / ``obj_quat`` ``(N, 4)``. Points outside the
        grid read its border (FlashDexRetarget's ``padding_mode="border"``).
        """
        s = self.sides[side_idx]
        grid, plan = self.geometry[s]["sdf"], self._sdf_plan[s]
        N, K = points_w.shape[:2]
        quat = obj_quat[:, None].expand(N, K, 4)
        local = inv_transform_by_quat(points_w - obj_pos[:, None], quat)
        uvw = (local / SDF_EXTENT).flip(-1)  # grid axes are (x, y, z) = (D, H, W): grid_sample wants (W, H, D)
        S = grid.shape[0]
        packed = uvw.new_zeros(S, plan["n_max"], K, 3)
        packed[plan["slot"], plan["within"]] = uvw[plan["order"]]
        out = F.grid_sample(
            grid,
            packed.reshape(S, plan["n_max"] * K, 1, 1, 3),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        out = out.view(S, 4, plan["n_max"], K).permute(0, 2, 3, 1)
        res = uvw.new_empty(N, K, 4)
        res[plan["order"]] = out[plan["slot"], plan["within"]]
        grad = res[..., 1:] / res[..., 1:].norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return res[..., 0], transform_by_quat(grad, quat)


class HandTermMixin(ConfigurableMixin):
    """The parameters every FlashDexRetarget term shares, and its :class:`HandObjectState`.

    A :class:`ConfigurableMixin` so its fields count as the term's parameters.
    """

    hand_model: str = "xhand"
    sides: list[str] = ["right"]
    objects: dict[str, str] = {}
    dataset: str = ""
    entity_name: str = "robot"
    #: Whether the term queries the object SDF or surface points (loads them at build).
    _geometry_: ClassVar[bool] = False

    def _build_state(self, n_points: int = 128) -> None:
        self.state = HandObjectState(
            self._env,
            self.hand_model,
            self.sides,
            self.objects,
            self.dataset,
            entity_name=self.entity_name,
            geometry=self._geometry_,
            n_points=n_points,
        )

    def _side_indices(self, side: str | None) -> list[int]:
        if side is None:
            return list(range(len(self.sides)))
        if side not in self.sides:
            raise ValueError(f"{term_subject(self)}: side {side!r} is not a hand of this task ({self.sides}).")
        return [self.sides.index(side)]
