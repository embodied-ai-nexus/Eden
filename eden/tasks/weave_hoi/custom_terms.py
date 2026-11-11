"""Weave G1-Inspire-HOI terms that have no generic Eden counterpart.

Each is a port of the like-named Weave term (``g1_hoi_learning/tasks/hoi/mdp``), rewritten against Eden's
multi-entity scene: Weave's robot is one articulation whose 54 bodies include both hands, while here the G1 body and
the two Inspire hands are three entities (``robot``, ``left_hand``, ``right_hand``). Terms that span them therefore
take their links per entity — ``links={"left_hand": [...], "right_hand": [...]}`` — and concatenate in that order.

* :class:`ObjectNearestSurfaceVector` / :class:`ObjectBpsSdf` — object-geometry observations read off Weave's
  per-object surface samples and signed-distance grid (the ``objects/`` folder of ``appolyn/Weave``).
* :class:`HandOppositionReward` / :class:`ObjectContactLabelReward` — the grasp-shape and contact-matching rewards.
* :class:`LostObjectContact` — terminate once the hands have lost the object for too long.
* :class:`WeaveFeetAirTime` / :class:`WeaveFeetSlide` — Weave's command-free feet terms (Eden's ``FeetAirTime`` /
  ``FeetSlip`` gate on a velocity command, which a tracking task has none of).

The object's per-env mesh comes from its variant (``Entity.env_variant_idx``) when the object is heterogeneous, and is
the single listed object otherwise. Object poses are read with ``get_pos()`` / ``get_quat()``: build the object with
``align=False`` so they are the mesh-origin frame the geometry files are expressed in.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch

from eden.constants import MetricDirection, MetricMode
from eden.datasets.weave_dataset import weave_object_dir
from eden.managers.metric_manager import METRIC_TERM_REGISTRY
from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY, ObservationTerm
from eden.managers.reward_manager import REWARD_TERM_REGISTRY, RewardTerm
from eden.managers.termination_manager import TERMINATION_TERM_REGISTRY, TerminationTerm
from eden.managers.terms.utils import resolve_link_idx
from eden.utils.geom import inv_quat, inv_transform_by_quat, quat_error_magnitude, quat_mul, transform_by_quat
from eden.utils.motion_reference.lookup import assert_motion_field_present, get_motion_link_idx

if TYPE_CHECKING:
    from eden.envs.base import EnvBase

#: Weave's signed-distance grids span ``[-SDF_GRID_BOUND, SDF_GRID_BOUND]^3`` of the normalised mesh.
SDF_GRID_BOUND = 1.25


# ---------------------------------------------------------------------------------------------------- helpers


def env_object_index(env: EnvBase, object_entity_name: str, objects: list[str]) -> torch.Tensor:
    """Which of ``objects`` each env holds, ``(num_envs,)`` long — the variant, or all zeros for one object."""
    obj = env.entities[object_entity_name]
    if obj.is_heterogeneous:
        if obj.num_variants != len(objects):
            raise ValueError(
                f"objects lists {len(objects)} names ({objects}) but entity '{object_entity_name}' has "
                f"{obj.num_variants} variants; list one object per GroupedEntityOptions.grouped_entities entry, in order."
            )
        return obj.env_variant_idx.to(env.device)
    if len(objects) != 1:
        raise ValueError(
            f"Entity '{object_entity_name}' is a single object, so objects must name exactly one; got {objects}."
        )
    return torch.zeros(env.num_envs, dtype=torch.long, device=env.device)


def load_surface_points(objects: list[str], geometry_root: str | None, device) -> torch.Tensor:
    """Weave's per-object surface samples in the mesh frame, stacked ``(num_objects, 512, 3)``."""
    return torch.stack(
        [
            torch.as_tensor(
                np.load(os.path.join(weave_object_dir(o, geometry_root), "surface.npy")), dtype=torch.float32
            )
            for o in objects
        ]
    ).to(device)


def nearest_surface_points(surface: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """The nearest of each env's surface samples ``(N, S, 3)`` to each query point ``(N, Q, 3)``; returns ``(N, Q, 3)``."""
    idx = torch.cdist(query, surface).argmin(-1)  # (N, Q)
    return torch.gather(surface, 1, idx.unsqueeze(-1).expand(-1, -1, 3))


def _entity_links(env: EnvBase, links: dict[str, list[str]], term: str, field: str = "links"):
    if not links:
        raise ValueError(
            f"{term}: {field} is empty; name the links per entity, e.g. {{'left_hand': ['thumb_distal']}}."
        )
    return {
        name: resolve_link_idx(env.entities[name], names, env.device, term=term, field=f"{field}[{name!r}]")
        for name, names in links.items()
    }


def object_contact_force(env: EnvBase, object_entity_name: str, links_idx: dict[str, torch.Tensor]) -> torch.Tensor:
    """Per-link contact force magnitude against the object, concatenated over entities: ``(N, sum L_e)``.

    ``force_norm_sum`` of :meth:`RigidEntity.get_grouped_contacts`, summed over the object's links — contacts with
    anything else (the ground, the other hand) do not count, as with Weave's object-filtered contact sensor.
    """
    obj = env.entities[object_entity_name]
    forces = [
        env.entities[name].get_grouped_contacts(obj, links_a_idx_local=idx)["force_norm_sum"].sum(-1)
        for name, idx in links_idx.items()
    ]
    return torch.cat(forces, dim=-1)


def reference_contact_label(env: EnvBase, links: dict[str, list[str]]) -> torch.Tensor:
    """The current-frame reference contact labels of ``links``, concatenated over entities: ``(N, sum L_e)``."""
    labels = []
    for name, names in links.items():
        assert_motion_field_present(env, name, "contact_label")
        idx = get_motion_link_idx(env, name, names)
        labels.append(env.motion_manager.data[name].contact_label[:, 0].index_select(1, idx))
    return torch.cat(labels, dim=-1)


def _links_pos(env: EnvBase, links_idx: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([env.entities[n].get_links_pos(ls_idx_local=idx) for n, idx in links_idx.items()], dim=1)


def _anchor_quat(env: EnvBase, entity_name: str, idx: torch.Tensor) -> torch.Tensor:
    return env.entities[entity_name].get_links_quat(ls_idx_local=idx)[:, 0]


# ---------------------------------------------------------------------------------------------------- observations


@OBSERVATION_TERM_REGISTRY.register()
class ObjectNearestSurfaceVector(ObservationTerm):
    """Vector from each named link to its nearest object-surface sample, in the robot anchor frame: ``(N, 3 * L)``.

    Weave's ``object_nearest_point_b``: the links are carried into the object frame, matched against the object's
    512 surface samples (``surface.npy``), and the offset rotated back out through the world into the live anchor
    link's frame. Brute force over the samples, as in Weave.

    Parameters
    ----------
    object_entity_name : str
        The manipulated object.
    anchor_entity_name : str
        The entity carrying the anchor link (the humanoid).
    anchor_link_name : str
        The frame the vectors are expressed in (Weave: ``pelvis``).
    links : dict[str, list[str]]
        Query links per entity; the output concatenates entities in this order.
    objects : list[str]
        The Weave object of each variant of ``object_entity_name``, in ``grouped_entities`` order (one name for a
        single object).
    geometry_root : str | None
        Root of an ``appolyn/Weave`` snapshot, or ``None`` to fetch the geometry into the HuggingFace cache.
    """

    object_entity_name: str = "object"
    anchor_entity_name: str = "robot"
    anchor_link_name: str = "pelvis"
    links: dict[str, list[str]] = {}
    objects: list[str] = []
    geometry_root: str | None = None

    def build(self) -> None:
        self._links_idx = _entity_links(self._env, self.links, type(self).__name__)
        self._anchor_idx = resolve_link_idx(
            self._env.entities[self.anchor_entity_name], [self.anchor_link_name], self.device, term=type(self).__name__
        )
        env_obj = env_object_index(self._env, self.object_entity_name, self.objects)
        self._surface = load_surface_points(self.objects, self.geometry_root, self.device)[env_obj]  # (N, S, 3)

    def compute(self) -> torch.Tensor:
        obj = self._env.entities[self.object_entity_name]
        obj_pos, obj_quat = obj.get_pos(), obj.get_quat()
        links_w = _links_pos(self._env, self._links_idx)  # (N, L, 3)
        n = links_w.shape[1]
        quat_l = obj_quat[:, None].expand(-1, n, -1)
        links_o = inv_transform_by_quat(links_w - obj_pos[:, None], quat_l)
        diff_w = transform_by_quat(nearest_surface_points(self._surface, links_o) - links_o, quat_l)
        anchor_quat = _anchor_quat(self._env, self.anchor_entity_name, self._anchor_idx)
        diff_b = inv_transform_by_quat(diff_w, anchor_quat[:, None].expand(-1, n, -1))
        return diff_b.reshape(diff_b.shape[0], -1)


@OBSERVATION_TERM_REGISTRY.register()
class ObjectBpsSdf(ObservationTerm):
    """Weave's 128-D basis-point shape descriptor of the object, as the anchor link sees it: ``(N, 128)``.

    A fixed basis (``objects/geometry/bps_128.npy``, in the normalised mesh space of ``sdf_128.npz``) is rotated by
    the anchor-to-object rotation — rotation only, so the descriptor encodes the object's shape and relative
    orientation but not where it is — and the object's signed-distance grid is sampled trilinearly at each point.
    Values are normalised distances (outside positive), clamped at the grid bound.

    Parameters
    ----------
    object_entity_name : str
        The manipulated object.
    anchor_entity_name : str
        The entity carrying the anchor link.
    anchor_link_name : str
        The link whose orientation the basis rides on (Weave: ``pelvis``).
    objects : list[str]
        The Weave object of each variant, as in :class:`ObjectNearestSurfaceVector`.
    geometry_root : str | None
        Root of an ``appolyn/Weave`` snapshot, or ``None`` to fetch the geometry.
    """

    object_entity_name: str = "object"
    anchor_entity_name: str = "robot"
    anchor_link_name: str = "pelvis"
    objects: list[str] = []
    geometry_root: str | None = None

    def build(self) -> None:
        self._anchor_idx = resolve_link_idx(
            self._env.entities[self.anchor_entity_name], [self.anchor_link_name], self.device, term=type(self).__name__
        )
        self._env_obj = env_object_index(self._env, self.object_entity_name, self.objects)
        root = weave_object_dir(self.objects[0], self.geometry_root)
        basis = np.load(os.path.join(os.path.dirname(root), "geometry", "bps_128.npy"))
        self._basis = torch.as_tensor(basis, dtype=torch.float32, device=self.device)  # (B, 3)
        grids = []
        for o in self.objects:
            with np.load(os.path.join(weave_object_dir(o, self.geometry_root), "sdf_128.npz")) as data:
                grids.append(np.asarray(data["sdf"], dtype=np.float32))
        self._sdf = torch.as_tensor(np.stack(grids), device=self.device)  # (O, R, R, R), indexed (x, y, z)

    def compute(self) -> torch.Tensor:
        obj_quat = self._env.entities[self.object_entity_name].get_quat()
        anchor_quat = _anchor_quat(self._env, self.anchor_entity_name, self._anchor_idx)
        object_from_anchor = quat_mul(inv_quat(obj_quat), anchor_quat)
        b = self._basis.shape[0]
        query = transform_by_quat(
            self._basis[None].expand(obj_quat.shape[0], -1, -1), object_from_anchor[:, None].expand(-1, b, -1)
        )
        return sample_sdf_trilinear(self._sdf, self._env_obj, query, SDF_GRID_BOUND)


def sample_sdf_trilinear(
    sdf: torch.Tensor, object_idx: torch.Tensor, points: torch.Tensor, grid_bound: float
) -> torch.Tensor:
    """Trilinearly sample per-env object grids ``sdf[object_idx]`` ``(O, R, R, R)`` at ``points`` ``(N, Q, 3)``.

    Weave's ``sample_sdf_trilinear``: grid axes are ``(x, y, z)`` spanning ``[-grid_bound, grid_bound]``, queries
    outside clamp to the boundary, and the eight corners are gathered from one flat table — no per-env grid copy.
    """
    res = sdf.shape[1]
    voxel = ((points + grid_bound) * ((res - 1) / (2.0 * grid_bound))).clamp(0.0, float(res - 1))
    lower = voxel.floor().long()
    frac = voxel - lower
    upper = (lower + 1).clamp(max=res - 1)
    x0, y0, z0 = lower.unbind(-1)
    x1, y1, z1 = upper.unbind(-1)
    wx, wy, wz = frac.unbind(-1)
    base = object_idx.long().unsqueeze(1) * res**3
    flat = sdf.reshape(-1)

    def at(x, y, z):
        return flat[base + x * res * res + y * res + z]

    c00 = torch.lerp(at(x0, y0, z0), at(x0, y0, z1), wz)
    c01 = torch.lerp(at(x0, y1, z0), at(x0, y1, z1), wz)
    c10 = torch.lerp(at(x1, y0, z0), at(x1, y0, z1), wz)
    c11 = torch.lerp(at(x1, y1, z0), at(x1, y1, z1), wz)
    return torch.lerp(torch.lerp(c00, c01, wy), torch.lerp(c10, c11, wy), wx)


# ---------------------------------------------------------------------------------------------------- rewards


@REWARD_TERM_REGISTRY.register()
class ObjectContactLabelReward(RewardTerm):
    """Match each link's object contact to its reference label, in ``[0, 1]``.

    Weave's ``contact_reward``: per link, the contact strength ``clamp(|F| / saturate_force, 0, 1)`` against the
    object scores ``1 - |target - strength|`` with ``target = 1`` for a ``+1`` label and ``0`` for ``-1``; ``0``
    labels are skipped, and the reward is the mean over the labelled links (``0`` when none is labelled).

    Parameters
    ----------
    object_entity_name : str
        The manipulated object.
    links : dict[str, list[str]]
        Scored links per entity (Weave: every hand link past the proximals).
    saturate_force : float
        Contact force (N) that counts as full contact. Defaults to 5.0.
    """

    object_entity_name: str = "object"
    links: dict[str, list[str]] = {}
    saturate_force: float = 5.0

    def build(self) -> None:
        self._links_idx = _entity_links(self._env, self.links, type(self).__name__)

    def compute(self) -> torch.Tensor:
        label = reference_contact_label(self._env, self.links)
        strength = (
            object_contact_force(self._env, self.object_entity_name, self._links_idx) / self.saturate_force
        ).clamp(0, 1)
        mask = (label != 0).float()
        score = 1.0 - ((label > 0).float() - strength).abs()
        return (score * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)


@REWARD_TERM_REGISTRY.register()
class HandOppositionReward(RewardTerm):
    """Reward the thumb for opposing the fingers across the object, in ``[0, 1]`` (Weave's ``hand_opposition_reward``).

    Per hand, each digit's outward direction ``u = unit(tip - nearest surface sample)`` (object frame); the thumb
    opposes finger ``k`` by ``(1 - u_thumb . u_k) / 2``, averaged over the fingers. A hand is scored only while the
    reference labels any of its digits ``+1``; the reward averages the scored hands (``0`` when neither is).

    Parameters
    ----------
    object_entity_name : str
        The manipulated object.
    hands : dict[str, list[str]]
        Per hand entity, its digit links thumb first (Weave: ``thumb_distal``, then the four ``*_intermediate``).
    objects : list[str]
        The Weave object of each variant, as in :class:`ObjectNearestSurfaceVector`.
    geometry_root : str | None
        Root of an ``appolyn/Weave`` snapshot, or ``None`` to fetch the geometry.
    """

    object_entity_name: str = "object"
    hands: dict[str, list[str]] = {}
    objects: list[str] = []
    geometry_root: str | None = None

    def build(self) -> None:
        self._links_idx = _entity_links(self._env, self.hands, type(self).__name__, field="hands")
        counts = {len(v) for v in self.hands.values()}
        if len(counts) != 1 or counts.pop() < 2:
            raise ValueError(
                f"HandOppositionReward: every hand needs the same digit list, thumb first then >= 1 finger; got "
                f"{ {k: len(v) for k, v in self.hands.items()} } links."
            )
        env_obj = env_object_index(self._env, self.object_entity_name, self.objects)
        self._surface = load_surface_points(self.objects, self.geometry_root, self.device)[env_obj]

    def compute(self) -> torch.Tensor:
        obj = self._env.entities[self.object_entity_name]
        obj_pos, obj_quat = obj.get_pos(), obj.get_quat()
        tips_w = _links_pos(self._env, self._links_idx)  # (N, H*D, 3)
        n = tips_w.shape[1]
        tips_o = inv_transform_by_quat(tips_w - obj_pos[:, None], obj_quat[:, None].expand(-1, n, -1))
        bearing = tips_o - nearest_surface_points(self._surface, tips_o)
        u = (bearing / bearing.norm(dim=-1, keepdim=True).clamp_min(1e-6)).reshape(
            tips_o.shape[0], len(self.hands), -1, 3
        )
        opposition = (1.0 - (u[:, :, :1] * u[:, :, 1:]).sum(-1)) / 2.0  # (N, H, F)
        gate = (reference_contact_label(self._env, self.hands) > 0).reshape(tips_o.shape[0], len(self.hands), -1)
        gate = gate.any(-1).float()  # (N, H)
        return (opposition.mean(-1) * gate).sum(-1) / gate.sum(-1).clamp(min=1.0)


@REWARD_TERM_REGISTRY.register()
class WeaveFeetAirTime(RewardTerm):
    """Count feet that land after at least ``threshold`` seconds in the air (Weave's ``feet_air_time``).

    On the step a foot makes contact (net contact force above ``contact_force_threshold``), it scores ``1`` if its
    swing that just ended lasted ``>= threshold``. No command gate — a tracking task steps when its reference does.

    Parameters
    ----------
    entity_name : str
        The legged entity.
    feet_link_names : list[str]
        The foot links.
    threshold : float
        Minimum rewarded swing, seconds. Defaults to 0.3.
    contact_force_threshold : float
        Net contact force (N) that counts as ground contact. Defaults to 1.0 (Isaac Lab's contact-sensor default).
    """

    entity_name: str = "robot"
    feet_link_names: list[str] = []
    threshold: float = 0.3
    contact_force_threshold: float = 1.0

    def build(self) -> None:
        robot = self._env.entities[self.entity_name]
        self._feet_idx = resolve_link_idx(
            robot, self.feet_link_names, self.device, term=type(self).__name__, field="feet_link_names"
        )
        self._air_time = torch.zeros(self.num_envs, self._feet_idx.numel(), device=self.device)
        self._in_contact = torch.zeros(self.num_envs, self._feet_idx.numel(), dtype=torch.bool, device=self.device)
        self._dt = float(self._env.dt)

    def compute(self) -> torch.Tensor:
        forces = self.resolve_entity(self.entity_name).get_links_net_contact_force()[:, self._feet_idx]
        contact = forces.norm(dim=-1) > self.contact_force_threshold
        first_contact = contact & ~self._in_contact
        reward = (first_contact & (self._air_time >= self.threshold)).float().sum(-1)
        self._air_time.copy_(torch.where(contact, torch.zeros_like(self._air_time), self._air_time + self._dt))
        self._in_contact.copy_(contact)
        return reward

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        if envs_idx is None:
            self._air_time.zero_()
            self._in_contact.zero_()
        else:
            self._air_time[envs_idx] = 0.0
            self._in_contact[envs_idx] = False


@REWARD_TERM_REGISTRY.register()
class WeaveFeetSlide(RewardTerm):
    """Planar foot speed while the foot bears load, summed over feet (Weave's ``feet_slide``; use a negative weight).

    A foot is loaded when its net contact force exceeded ``force_threshold`` on any of the last ``history_length``
    control steps (Isaac Lab's contact-sensor history).

    Parameters
    ----------
    entity_name : str
        The legged entity.
    feet_link_names : list[str]
        The foot links.
    force_threshold : float
        Load threshold, N. Defaults to 0.5.
    history_length : int
        Control steps of force history. Defaults to 3.
    """

    entity_name: str = "robot"
    feet_link_names: list[str] = []
    force_threshold: float = 0.5
    history_length: int = 3

    def build(self) -> None:
        robot = self._env.entities[self.entity_name]
        self._feet_idx = resolve_link_idx(
            robot, self.feet_link_names, self.device, term=type(self).__name__, field="feet_link_names"
        )
        self._history = torch.zeros(self.num_envs, self.history_length, self._feet_idx.numel(), device=self.device)

    def compute(self) -> torch.Tensor:
        robot = self.resolve_entity(self.entity_name)
        force = robot.get_links_net_contact_force()[:, self._feet_idx].norm(dim=-1)
        self._history.copy_(torch.cat([force[:, None], self._history[:, :-1]], dim=1))
        loaded = self._history.amax(dim=1) > self.force_threshold
        speed_xy = robot.get_links_vel(ls_idx_local=self._feet_idx)[..., :2].norm(dim=-1)
        return (speed_xy * loaded).sum(-1)

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        if envs_idx is None:
            self._history.zero_()
        else:
            self._history[envs_idx] = 0.0


# ---------------------------------------------------------------------------------------------------- terminations


@TERMINATION_TERM_REGISTRY.register()
class LostObjectContact(TerminationTerm):
    """Terminate once the reference expects hand contact but none of the links has touched the object for too long.

    Weave's ``bad_contact``: a per-env counter counts consecutive control steps on which some link is labelled
    ``+1`` yet no link carries more than ``force_threshold`` against the object, is cleared on any other step, and
    fires at ``max_lost_steps``.

    Parameters
    ----------
    object_entity_name : str
        The manipulated object.
    links : dict[str, list[str]]
        The links per entity that can hold the object.
    max_lost_steps : int
        Consecutive lost steps that terminate (Weave: 10, i.e. 0.2 s at 50 Hz). Defaults to 10.
    force_threshold : float
        Contact force (N) that counts as touching. Defaults to 1.0.
    """

    object_entity_name: str = "object"
    links: dict[str, list[str]] = {}
    max_lost_steps: int = 10
    force_threshold: float = 1.0

    def build(self) -> None:
        self._links_idx = _entity_links(self._env, self.links, type(self).__name__)
        self._lost = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def compute(self) -> torch.Tensor:
        expected = (reference_contact_label(self._env, self.links) > 0).any(-1)
        touching = (
            object_contact_force(self._env, self.object_entity_name, self._links_idx) > self.force_threshold
        ).any(-1)
        lost = expected & ~touching
        self._lost.copy_(torch.where(lost, self._lost + 1, torch.zeros_like(self._lost)))
        return self._lost >= self.max_lost_steps

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        if envs_idx is None:
            self._lost.zero_()
        else:
            self._lost[envs_idx] = 0


# ---------------------------------------------------------------------------------------------------- metrics


@METRIC_TERM_REGISTRY.register(
    success_threshold=0.1, is_cumulative=True, direction=MetricDirection.LIB, metric_mode=MetricMode.RESET
)
def weave_error_object_pos(env: EnvBase, *, entity_name: str = "object") -> torch.Tensor:
    """Object position error vs the reference, metres (Weave's ``error_object_pos``)."""
    ref = env.motion_manager.data[entity_name]
    return torch.norm(env.entities[entity_name].get_pos() - ref.base_pos_w[:, 0], dim=-1)


@METRIC_TERM_REGISTRY.register(
    success_threshold=0.3, is_cumulative=True, direction=MetricDirection.LIB, metric_mode=MetricMode.RESET
)
def weave_error_object_rot(env: EnvBase, *, entity_name: str = "object") -> torch.Tensor:
    """Object orientation error vs the reference, radians (Weave's ``error_object_rot``)."""
    ref = env.motion_manager.data[entity_name]
    return quat_error_magnitude(ref.base_quat_w[:, 0], env.entities[entity_name].get_quat())
