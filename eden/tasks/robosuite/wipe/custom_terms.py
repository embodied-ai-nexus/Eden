"""Terms only the robosuite ``Wipe`` port needs: the dirt path, the wiped-marker bookkeeping, the arm's safety and the wrist force.

robosuite scatters 100 visual-only cylinders along a random walk on the table and, whenever the
wiping pad touches something, marks every marker that lies within 2 cm under the pad's rectangle
as wiped. Eden's markers are kinematic cylinders (no contacts, settable pose), so:

* :class:`SampleWipePath` (RESET event) re-runs robosuite's random walk with a longer stride and
  teleports the markers onto the table top.
* :class:`MarkersWiped` (metric) owns the per-episode latched ``(num_envs, num_markers)`` wiped mask
  and a geometric stand-in for the corner-plane test — a marker is wiped once the grasp point (pad
  midpoint) passes within ``xy_tolerance`` horizontally and ``z_tolerance`` vertically of it. It
  reports the wiped fraction and exposes the derived quantities (newly wiped count, unwiped
  centroid / radius) the other terms pull by name through ``metric_manager.get_term``, resolved in
  ``build()`` (the metric manager is constructed ahead of every consumer — see
  ``RLEnvBase._manager_order``), so a bad ``metric_name`` fails at env construction.
* :class:`ArmLinksInContact` and :class:`ArmNearJointLimits` (metrics) are robosuite's two unsafe
  states: a link of the arm proper touching the table (``check_contact(robot_model)``) and a joint
  within 0.1 rad of a stop (``check_q_limits``). The task ORs them into one ``unsafe`` metric; it
  ends the episode, pays the collision penalty and — as in robosuite, where an unsafe step returns
  the penalty *instead of* everything else — zeroes every other Wipe reward term
  (:class:`_PaidWhileSafe`).
* :class:`GripperContactForce` (metric) stands in for robosuite's wrist force-torque sensor: the
  magnitude of the net contact force on the gripper from everything that is not the robot, which is
  what the bias-subtracted sensor reads when the hand is still. The contact reward, the pressure
  reward and the ``robot0_contact`` observation read it.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from eden.managers.event_manager import EVENT_TERM_REGISTRY, EventTerm
from eden.managers.metric_manager import METRIC_TERM_REGISTRY, MetricTerm
from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY, ObservationTerm
from eden.managers.reward_manager import REWARD_TERM_REGISTRY, RewardTerm
from eden.managers.terms.utils import resolve_metric_manager
from eden.tasks.robosuite.custom_terms.common import GraspFrameMixin, RigMixin
from eden.utils.misc import sanitize_envs_idx
from eden.utils.sample import sample_uniform

if TYPE_CHECKING:
    from eden.entities.rigid import RigidEntity
    from eden.managers.base import ManagerTermBase


# --- the dirt path --------------------------------------------------------------------------------


def _first_true_index(mask: torch.Tensor, *, fallback: int) -> torch.Tensor:
    """Index of the first ``True`` along the last dim, or ``fallback`` where a row is all ``False``.

    ``argmax`` cannot express this: it also returns 0 for an all-``False`` row, which here would mean
    "the first redraw fit" when none did.
    """
    order = torch.arange(mask.shape[-1], device=mask.device).expand_as(mask)
    return torch.where(mask, order, fallback).amin(dim=-1)


@EVENT_TERM_REGISTRY.register()
class SampleWipePath(EventTerm):
    """Teleport the markers onto a fresh random-walk path on the table top (robosuite ``WipeArena.sample_path_pos``).

    The walk starts uniformly inside the ``+-x_half`` x ``+-y_half`` box around ``origin`` with a
    uniform heading; each following marker advances ``step`` along the heading, which turns by
    ``N(0, turn_std)`` with probability ``turn_prob``. When the next point would leave the box the
    heading is re-drawn — uniformly, where robosuite keeps nudging it by ``N(0, 0.5)`` until it
    fits: a fixed number of uniform re-draws (then a clamp) is what keeps the sampler sync-free
    while still turning a wall-bound walk around. Marker centres sit ``lift`` above ``origin[2]``.

    The ``redraws`` are drawn in one batch and the first that fits is taken, rather than looped over
    one at a time: the walk itself is sequential in the markers, so a nested loop would put
    ``markers x redraws`` rounds of kernel launches on every reset — and this event fires whenever
    any env's episode ends, which for Wipe (the one early-terminating task) is most steps.
    """

    marker_names: list[str] = []
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    x_half: float = 0.13
    y_half: float = 0.22
    step: float = 0.03
    turn_prob: float = 0.3
    turn_std: float = 0.5
    lift: float = 0.001
    redraws: int = 12

    def build(self) -> None:
        if not self.marker_names:
            raise ValueError("SampleWipePath needs at least one marker in marker_names.")
        for name in self.marker_names:
            self.resolve_entity(name)
        origin = torch.tensor(self.origin, dtype=torch.float32, device=self.device)
        self._origin = origin + torch.tensor([0.0, 0.0, self.lift], device=self.device)
        self._half = torch.tensor([self.x_half, self.y_half], dtype=torch.float32, device=self.device)

    def _advance(self, prev: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
        return prev + self.step * torch.stack((torch.sin(heading), torch.cos(heading)), dim=-1)

    def _sample_path(self, n: int) -> torch.Tensor:
        """Table-local xy of every marker on ``n`` fresh paths. ``(n, num_markers, 2)``."""
        m = len(self.marker_names)
        heading = sample_uniform(-math.pi, math.pi, (n,), device=self.device)
        path = torch.empty(n, m, 2, device=self.device)
        path[:, 0] = sample_uniform(-self._half, self._half, (n, 2), device=self.device)
        for k in range(1, m):
            turn = torch.rand(n, device=self.device) < self.turn_prob
            heading = heading + turn * torch.randn(n, device=self.device) * self.turn_std
            prev = path[:, k - 1]
            cand = self._advance(prev, heading)
            out = (cand.abs() >= self._half).any(dim=-1)
            retry = sample_uniform(-math.pi, math.pi, (n, self.redraws), device=self.device)
            steps = self._advance(prev[:, None], retry)  # (n, redraws, 2)
            fits = (steps.abs() < self._half).all(dim=-1)  # (n, redraws)
            # First heading that fits, or the last draw when none does (the clamp below rescues it).
            first = _first_true_index(fits, fallback=self.redraws - 1)
            picked_step = steps.gather(1, first[:, None, None].expand(n, 1, 2))[:, 0]
            picked_heading = retry.gather(1, first[:, None])[:, 0]
            cand = torch.where(out[:, None], picked_step, cand)
            heading = torch.where(out, picked_heading, heading)
            path[:, k] = torch.clamp(cand, -self._half + 1e-3, self._half - 1e-3)
        return path

    def compute(self, envs_idx: slice | torch.Tensor | None) -> None:
        envs_idx, n_envs = sanitize_envs_idx(envs_idx, self.num_envs, return_n_envs=True)
        path = self._sample_path(n_envs)  # only the resetting envs' paths: this fires on most Wipe steps
        pos = torch.cat((path, torch.zeros(*path.shape[:2], 1, device=self.device)), dim=-1) + self._origin
        for k, name in enumerate(self.marker_names):
            self.resolve_entity(name).set_pos(pos[:, k], envs_idx=envs_idx)


# --- wiped-marker bookkeeping ---------------------------------------------------------------------


@METRIC_TERM_REGISTRY.register()
class MarkersWiped(GraspFrameMixin, MetricTerm):
    """Fraction of markers wiped so far this episode (success at 1.0); owns the latched wiped mask.

    A marker counts as wiped once the grasp point is within ``xy_tolerance`` of it horizontally and
    ``z_tolerance`` vertically. The mask is latched per episode and cleared in ``reset()``; the
    unwiped centroid / radius are refreshed every ``compute()`` and again in ``post_reset()`` so the
    first post-reset observation sees the freshly sampled path. Other terms read
    :attr:`newly_wiped_count`, :attr:`wipe_centroid`, :attr:`wipe_radius` and :attr:`all_wiped`.

    Marker positions are read once per reset, not once per step: the markers are kinematic and
    :class:`SampleWipePath` is the only thing that moves them, so a per-step read would be one
    kernel launch per marker to fetch a buffer that cannot have changed.
    """

    rig: str = "robot"
    marker_names: list[str] = []
    xy_tolerance: float = 0.02
    z_tolerance: float = 0.03

    def build(self) -> None:
        self._build_grasp_frame()
        if not self.marker_names:
            raise ValueError("MarkersWiped needs at least one marker in marker_names.")
        for name in self.marker_names:
            self.resolve_entity(name)
        n, m = self.num_envs, len(self.marker_names)
        self._wiped = torch.zeros(n, m, dtype=torch.bool, device=self.device)
        self._newly_wiped = torch.zeros(n, device=self.device)
        self._centroid = torch.zeros(n, 3, device=self.device)
        self._radius = torch.zeros(n, device=self.device)
        self._all_wiped = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._markers_w = self._read_marker_pos_w()

    @property
    def newly_wiped_count(self) -> torch.Tensor:
        """Markers wiped for the first time on the last step. ``(num_envs,)`` float."""
        return self._newly_wiped

    @property
    def wipe_centroid(self) -> torch.Tensor:
        """Mean world position of the unwiped markers; zeros once all are wiped. ``(num_envs, 3)``."""
        return self._centroid

    @property
    def wipe_radius(self) -> torch.Tensor:
        """Largest distance of an unwiped marker from the centroid. ``(num_envs,)``."""
        return self._radius

    @property
    def all_wiped(self) -> torch.Tensor:
        """``(num_envs,)`` bool: every marker wiped."""
        return self._all_wiped

    def _read_marker_pos_w(self) -> torch.Tensor:
        """Stack every marker's root position — one read per marker, so only on a reset."""
        return torch.stack([self.resolve_entity(name).get_pos() for name in self.marker_names], dim=1)

    def _refresh_summary(self, markers: torch.Tensor) -> None:
        unwiped = ~self._wiped
        count = unwiped.sum(dim=-1)
        centroid = (markers * unwiped[..., None]).sum(dim=1) / torch.clamp_min(count, 1)[:, None]
        self._all_wiped.copy_(count == 0)
        self._centroid.copy_(torch.where(self._all_wiped[:, None], torch.zeros_like(centroid), centroid))
        self._radius.copy_((torch.norm(markers - centroid[:, None], dim=-1) * unwiped).amax(dim=-1))

    def compute(self) -> torch.Tensor:
        markers = self._markers_w
        delta = markers - self._grasp_pos_w()[:, None]
        hit = (torch.norm(delta[..., :2], dim=-1) < self.xy_tolerance) & (delta[..., 2].abs() < self.z_tolerance)
        self._newly_wiped.copy_((hit & ~self._wiped).sum(dim=-1).float())
        self._wiped.logical_or_(hit)
        self._refresh_summary(markers)
        return self._wiped.float().mean(dim=-1)

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        self._wiped[envs_idx] = False
        self._newly_wiped[envs_idx] = 0.0

    def post_reset(self, envs_idx: slice | torch.Tensor) -> None:
        # The one place the markers can have moved: ``SampleWipePath`` ran in the reset that precedes
        # this hook. Idempotent for the envs that did not reset, so re-read and refresh all of them.
        self._markers_w = self._read_marker_pos_w()
        self._refresh_summary(self._markers_w)


def _markers_wiped_term(term: ManagerTermBase, metric_name: str) -> MarkersWiped:
    """The :class:`MarkersWiped` metric ``metric_name`` names; raises if it is another kind of term."""
    metric = term._env.metric_manager.get_term(metric_name)
    if not isinstance(metric, MarkersWiped):
        raise TypeError(
            f"{type(term).__name__} needs metric '{metric_name}' to be a MarkersWiped term, got "
            f"{type(metric).__name__}; it reads the wiped-marker buffers only that term owns."
        )
    return metric


class _PullsMarkersWiped:
    """Resolve the :class:`MarkersWiped` metric named by ``metric_name`` in ``build()``.

    Cooperative: subclasses with build work of their own call ``super().build()``. The metric
    manager is constructed before the reward and observation managers that host these terms
    (``RLEnvBase._manager_order``), so the handle resolves here and a bad name raises at env
    construction rather than on the first step.
    """

    metric_name: str

    def build(self) -> None:
        self._metric = _markers_wiped_term(self, self.metric_name)
        super().build()

    def _markers(self) -> MarkersWiped:
        return self._metric


# --- observations ---------------------------------------------------------------------------------


@OBSERVATION_TERM_REGISTRY.register()
class WipeCentroid(_PullsMarkersWiped, ObservationTerm):
    """World centroid of the unwiped markers (robosuite ``wipe_centroid``). ``(N, 3)``."""

    metric_name: str = "markers_wiped"

    def compute(self) -> torch.Tensor:
        return self._markers().wipe_centroid


@OBSERVATION_TERM_REGISTRY.register()
class WipeRadius(_PullsMarkersWiped, ObservationTerm):
    """Spread of the unwiped markers about their centroid (robosuite ``wipe_radius``). ``(N, 1)``."""

    metric_name: str = "markers_wiped"

    def compute(self) -> torch.Tensor:
        return self._markers().wipe_radius[:, None]


@OBSERVATION_TERM_REGISTRY.register()
class GripperToWipeCentroid(_PullsMarkersWiped, GraspFrameMixin, ObservationTerm):
    """``centroid - grasp_pos`` (robosuite ``gripper_to_wipe_centroid``). ``(N, 3)``."""

    rig: str = "robot"
    metric_name: str = "markers_wiped"

    def build(self) -> None:
        super().build()  # _PullsMarkersWiped: resolve the metric handle
        self._build_grasp_frame()

    def compute(self) -> torch.Tensor:
        return self._markers().wipe_centroid - self._grasp_pos_w()


# --- rewards --------------------------------------------------------------------------------------


class _PaidWhileSafe:
    """Pay this reward only while the ``unsafe`` metric is 0 — robosuite Wipe's reward structure.

    robosuite returns ``arm_limit_collision_penalty`` *instead of* every other component on a step
    where the arm touches the table or a joint is at its limit, so wiping markers while colliding is
    not worth it. Summing the terms (the convention of the other ports) would pay 50 per marker on
    that step and make a collision profitable. The host term declares ``unsafe_metric`` (a field must
    sit on the term class to be configurable); cooperative, the host's own ``build()`` work runs via
    ``super().build()``.
    """

    unsafe_metric: str

    def build(self) -> None:
        resolve_metric_manager(self._env, term=type(self).__name__, metric_name=self.unsafe_metric)
        super().build()

    def _safe(self) -> torch.Tensor:
        """``(N,)`` 1.0 while the arm is safe, 0.0 on an unsafe step."""
        return 1.0 - self._env.metric_manager.get_value(self.unsafe_metric)


@REWARD_TERM_REGISTRY.register()
class NewlyWipedMarkers(_PaidWhileSafe, _PullsMarkersWiped, RewardTerm):
    """Number of markers wiped for the first time this step (robosuite ``unit_wiped_reward`` x count). ``(N,)``."""

    metric_name: str = "markers_wiped"
    unsafe_metric: str = "unsafe"

    def compute(self) -> torch.Tensor:
        return self._markers().newly_wiped_count * self._safe()


@REWARD_TERM_REGISTRY.register()
class WipeCentroidDistanceTanh(_PaidWhileSafe, _PullsMarkersWiped, GraspFrameMixin, RewardTerm):
    """``1 - tanh(coef * ||centroid - grasp_pos||)`` while markers remain (robosuite's distance shaping). ``(N,)``."""

    rig: str = "robot"
    metric_name: str = "markers_wiped"
    unsafe_metric: str = "unsafe"
    coef: float = 5.0

    def build(self) -> None:
        super().build()  # _PaidWhileSafe / _PullsMarkersWiped: resolve the metric handles
        self._build_grasp_frame()

    def compute(self) -> torch.Tensor:
        markers = self._markers()
        dist = torch.norm(markers.wipe_centroid - self._grasp_pos_w(), dim=-1)
        return (1.0 - torch.tanh(self.coef * dist)) * (~markers.all_wiped).float() * self._safe()


@REWARD_TERM_REGISTRY.register()
class WipeTaskComplete(_PaidWhileSafe, _PullsMarkersWiped, RewardTerm):
    """1.0 once every marker is wiped (robosuite ``task_complete_reward``). ``(N,)``."""

    metric_name: str = "markers_wiped"
    unsafe_metric: str = "unsafe"

    def compute(self) -> torch.Tensor:
        return self._markers().all_wiped.float() * self._safe()


class _ReadsGripperForce:
    """Resolve the :class:`GripperContactForce` metric named by ``force_metric`` in ``build()``; read it with :meth:`_force`.

    The host term declares ``force_metric``.
    """

    force_metric: str

    def build(self) -> None:
        resolve_metric_manager(self._env, term=type(self).__name__, metric_name=self.force_metric)
        super().build()

    def _force(self) -> torch.Tensor:
        """``(N,)`` wrist force [N] this step."""
        return self._env.metric_manager.get_value(self.force_metric)


@REWARD_TERM_REGISTRY.register()
class GripperContact(_PaidWhileSafe, _ReadsGripperForce, RewardTerm):
    """1.0 while the wrist force exceeds ``threshold`` (robosuite ``wipe_contact_reward``, ``contact_threshold`` 1 N). ``(N,)``."""

    force_metric: str = "gripper_force"
    unsafe_metric: str = "unsafe"
    threshold: float = 1.0

    def compute(self) -> torch.Tensor:
        return (self._force() > self.threshold).float() * self._safe()


@REWARD_TERM_REGISTRY.register()
class WipePressure(_PaidWhileSafe, _ReadsGripperForce, RewardTerm):
    """robosuite Wipe's force terms on the wrist force ``F``, configured with ``weight=1``. ``(N,)``.

    ``-excess_penalty * F`` above ``max_force`` (60 N); otherwise ``contact_reward + force_gain * F``
    above ``min_force`` (0.5 N); else 0 — robosuite's ``excess_force_penalty_mul`` and its "pressing
    into the table" bonus, with its multipliers.
    """

    force_metric: str = "gripper_force"
    unsafe_metric: str = "unsafe"
    min_force: float = 0.5
    max_force: float = 60.0
    excess_penalty: float = 0.05
    contact_reward: float = 0.01
    force_gain: float = 0.01

    def compute(self) -> torch.Tensor:
        force = self._force()
        pressing = torch.where(force > self.min_force, self.contact_reward + self.force_gain * force, 0.0)
        return torch.where(force > self.max_force, -self.excess_penalty * force, pressing) * self._safe()


@OBSERVATION_TERM_REGISTRY.register()
class GripperInContact(_ReadsGripperForce, ObservationTerm):
    """1.0 while the wrist force exceeds ``threshold`` (robosuite ``robot0_contact``, ``contact_threshold`` 1 N). ``(N, 1)``."""

    force_metric: str = "gripper_force"
    threshold: float = 1.0

    def compute(self) -> torch.Tensor:
        return (self._force() > self.threshold).float()[:, None]


# --- the arm's safety and the wrist force ----------------------------------------------------------------------------


@METRIC_TERM_REGISTRY.register()
class ArmLinksInContact(RigMixin, MetricTerm):
    """1.0 when an arm link touches ``other_name`` (robosuite ``check_contact(robot arm geoms, ...)``).

    The arm is every link of the rig's arm entity except its ``ee_link`` and that link's descendants —
    the gripper mount and fingers are allowed to touch (the Panda carries them on the same entity; a
    composite rig's gripper is a separate entity and never in the set). Configure with
    ``contributes_to_success=False``: it is a penalty / termination input.
    """

    rig: str = "robot"
    other_name: str = "table"

    def build(self) -> None:
        rig = self._rig_options()
        self._arm_name = rig.arm_name
        arm: RigidEntity = self.resolve_entity(rig.arm_name)
        self.resolve_entity(self.other_name)
        ee_idx = arm.get_link(rig.ee_link).idx
        arm_links = [link.idx for link in arm.links if not self._is_or_descends_from(arm, link, ee_idx)]
        if not arm_links:
            raise ValueError(
                f"Every link of '{rig.arm_name}' is '{rig.ee_link}' or hangs below it; the rig's ee_link must be "
                f"the gripper mount so the arm proper is non-empty. Links: {[link.name for link in arm.links]}."
            )
        self._arm_links = torch.tensor(arm_links, dtype=torch.long, device=self.device)

    @staticmethod
    def _is_or_descends_from(arm: RigidEntity, link, ancestor_idx: int) -> bool:
        while link is not None and link.idx != -1:
            if link.idx == ancestor_idx:
                return True
            parent = link.parent_idx
            if parent < arm.link_start or parent >= arm.link_end:
                return False
            link = arm.links[parent - arm.link_start]
        return False

    def compute(self) -> torch.Tensor:
        arm = self.resolve_entity(self._arm_name)
        contacts = arm.get_contacts(with_entity=self.resolve_entity(self.other_name))
        on_arm = torch.isin(contacts["link_a"], self._arm_links) | torch.isin(contacts["link_b"], self._arm_links)
        return (contacts["valid_mask"] & on_arm).any(dim=-1).float()


@METRIC_TERM_REGISTRY.register()
class ArmNearJointLimits(RigMixin, MetricTerm):
    """1.0 when an arm joint is within ``tolerance`` [rad] of a stop (robosuite ``Robot.check_q_limits``). ``(N,)``.

    The arm DOFs are the rig's ``arm_dofs``, as robosuite checks only the arm's reference joints. A
    joint without a range (equal or infinite limits) never counts. Configure with
    ``contributes_to_success=False``: it is a penalty / termination input.
    """

    rig: str = "robot"
    tolerance: float = 0.1

    def build(self) -> None:
        rig = self._rig_options()
        self._arm_name = rig.arm_name
        arm: RigidEntity = self.resolve_entity(rig.arm_name)
        self._dofs = arm.find_named_dofs(list(rig.arm_dofs))
        low, high = (limit[:, self._dofs.columns] for limit in arm.get_dofs_limit())
        bounded = torch.isfinite(low) & torch.isfinite(high) & (high > low)
        # An unbounded joint gets a band no position can leave.
        self._low = torch.where(bounded, low + self.tolerance, -math.inf)
        self._high = torch.where(bounded, high - self.tolerance, math.inf)

    def compute(self) -> torch.Tensor:
        q = self.resolve_entity(self._arm_name).get_dofs_pos(dofs_idx_local=self._dofs.dofs_idx_local)
        return ((q <= self._low) | (q >= self._high)).any(dim=-1).float()


@METRIC_TERM_REGISTRY.register()
class GripperContactForce(RigMixin, MetricTerm):
    """Magnitude [N] of the net contact force on the gripper from everything but the robot. ``(N,)``.

    Stands in for robosuite's wrist force-torque reading (``ee_force`` less its first-step bias, which
    removes the hand's own weight): with the hand still, that sensor reads the external contact
    force on the gripper. The gripper is the rig's gripper entity; contacts with the rig's own links
    (the arm it is welded to, its own fingers) are left out. Configure with
    ``contributes_to_success=False``.
    """

    rig: str = "robot"

    def build(self) -> None:
        rig = self._rig_options()
        self._gripper_name = rig.gripper_name
        gripper: RigidEntity = self.resolve_entity(rig.gripper_name)
        arm: RigidEntity = self.resolve_entity(rig.arm_name)
        own = {link.idx for entity in (arm, gripper) for link in entity.links}
        self._gripper_links = torch.tensor([link.idx for link in gripper.links], dtype=torch.long, device=self.device)
        self._robot_links = torch.tensor(sorted(own), dtype=torch.long, device=self.device)

    def compute(self) -> torch.Tensor:
        contacts = self.resolve_entity(self._gripper_name).get_contacts()
        link_a, link_b, valid = contacts["link_a"], contacts["link_b"], contacts["valid_mask"]
        on_a = valid & torch.isin(link_a, self._gripper_links) & ~torch.isin(link_b, self._robot_links)
        on_b = valid & torch.isin(link_b, self._gripper_links) & ~torch.isin(link_a, self._robot_links)
        force = (contacts["force_a"] * on_a[..., None]).sum(dim=1) + (contacts["force_b"] * on_b[..., None]).sum(dim=1)
        return torch.norm(force, dim=-1)
