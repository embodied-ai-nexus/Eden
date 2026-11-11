"""Terms the Mochi deformable tasks share: tool centre, soft-body keypoints, reaching gated by the grip, letting go."""

from __future__ import annotations

import torch

from eden.constants import MetricDirection
from eden.managers.metric_manager import METRIC_TERM_REGISTRY, MetricTerm
from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY, ObservationTerm
from eden.managers.reward_manager import REWARD_TERM_REGISTRY, RewardTerm
from eden.managers.terms.utils import resolve_metric_manager
from eden.tasks.mochi.common import link_point


def _soft_vertices(term, entity_name: str, vertices: list[int]) -> torch.Tensor:
    body = term.resolve_entity(entity_name)
    bad = [v for v in vertices if not -body.n_vertices <= v < body.n_vertices]
    if not vertices or bad:
        raise ValueError(
            f"{type(term).__name__} needs vertices of '{entity_name}' ({body.n_vertices} of them, indices "
            f"0..{body.n_vertices - 1} or negative from the end), got {vertices}."
        )
    return torch.tensor([v % body.n_vertices for v in vertices], dtype=torch.long, device=term.device)


@OBSERVATION_TERM_REGISTRY.register()
class ToolCentre(ObservationTerm):
    """World position of a robot's tool centre (a point fixed in one of its links), ``(num_envs, 3)``."""

    robot_name: str = "robot"
    link_name: str = "link_6"
    link_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def build(self) -> None:
        self._link = self.resolve_entity(self.robot_name).get_link(self.link_name)
        self._offset = torch.tensor(self.link_offset, device=self.device)

    def compute(self) -> torch.Tensor:
        return link_point(self._link, self._offset)


@OBSERVATION_TERM_REGISTRY.register()
class SoftBodyPoints(ObservationTerm):
    """Chosen vertices of a Mochi soft body relative to a robot's tool centre, ``(num_envs, 3 * n_vertices)``."""

    entity_name: str = "sponge"
    vertices: list[int] = []
    robot_name: str = "robot"
    link_name: str = "link_6"
    link_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def build(self) -> None:
        self._vertices = _soft_vertices(self, self.entity_name, self.vertices)
        self._link = self.resolve_entity(self.robot_name).get_link(self.link_name)
        self._offset = torch.tensor(self.link_offset, device=self.device)

    def compute(self) -> torch.Tensor:
        points = self.resolve_entity(self.entity_name).get_vertex_positions()[:, self._vertices]
        return (points - link_point(self._link, self._offset)[:, None, :]).reshape(self.num_envs, -1)


@REWARD_TERM_REGISTRY.register()
class ReachSoftBody(RewardTerm):
    """``exp(-d^2 / std^2)``, ``d`` from a robot's tool centre to the mean of chosen soft-body vertices.

    While the grip metric ``grip_metric_name`` reads ``grip_threshold`` or more the term pays its full 1:
    dropping to 0 there made taking hold cost more than it earned, and policies learned to hover by the part
    and never close (measured on the cloth and cable tasks). Leave ``grip_metric_name`` empty for a
    hand that only presses, and ``vertices`` empty for the whole body's centroid.
    """

    entity_name: str = "sponge"
    vertices: list[int] = []
    robot_name: str = "robot"
    link_name: str = "link_6"
    link_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    grip_metric_name: str = "grip"
    grip_threshold: float = 2.0
    std: float = 0.1

    def build(self) -> None:
        self._vertices = _soft_vertices(self, self.entity_name, self.vertices) if self.vertices else None
        self._link = self.resolve_entity(self.robot_name).get_link(self.link_name)
        self._offset = torch.tensor(self.link_offset, device=self.device)
        if self.grip_metric_name:
            resolve_metric_manager(self._env, term="reach_soft_body", metric_name=self.grip_metric_name)

    def compute(self) -> torch.Tensor:
        verts = self.resolve_entity(self.entity_name).get_vertex_positions()
        point = (verts if self._vertices is None else verts[:, self._vertices]).mean(dim=1)
        d2 = (point - link_point(self._link, self._offset)).square().sum(dim=-1)
        reach = torch.exp(-d2 / self.std**2)
        if not self.grip_metric_name:
            return reach
        held = self._env.metric_manager.get_value(self.grip_metric_name) >= self.grip_threshold
        return torch.where(held, 1.0, reach)


@REWARD_TERM_REGISTRY.register()
class GripHeld(RewardTerm):
    """1 while the grip metric ``grip_metric_name`` reads ``grip_threshold`` or more and the part is off the bench.

    "Off the bench" is the soft body's lowest vertex at least ``min_height`` up, so squeezing a part where it
    lies pays nothing.
    """

    entity_name: str = "sponge"
    grip_metric_name: str = "grip"
    grip_threshold: float = 2.0
    min_height: float = 0.01

    def build(self) -> None:
        self.resolve_entity(self.entity_name)
        resolve_metric_manager(self._env, term="grip_held", metric_name=self.grip_metric_name)

    def compute(self) -> torch.Tensor:
        lowest = self.resolve_entity(self.entity_name).get_vertex_positions()[..., 2].amin(dim=-1)
        held = self._env.metric_manager.get_value(self.grip_metric_name) >= self.grip_threshold
        return (held & (lowest >= self.min_height)).float()


@METRIC_TERM_REGISTRY.register()
class HandsClear(MetricTerm):
    """The distance (m) from the nearest of several tool centres to a soft body: the hands have let go once it is clear.

    RoboSimStudio's goals end on the part released: no towel particle within 0.02 m of a tool centre (its
    ``held_particles``), both hands more than 0.05 m from the cable. Succeeds at or above ``success_threshold``.

    Parameters
    ----------
    entity_name : str
        The soft body.
    robots_name : list[str]
        The robots whose tool centres count.
    link_name : str
        Each robot's tool link.
    link_offset : tuple[float, float, float]
        The tool centre in that link's frame (m).
    success_threshold : float
        The clearance (m) at or above which the hands count as let go.
    """

    entity_name: str = "towel"
    robots_name: list[str] = ["left", "right"]
    link_name: str = "link_6"
    link_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    success_threshold: float = 0.02

    def build(self) -> None:
        self.resolve_entity(self.entity_name)
        self._links = [self.resolve_entity(name).get_link(self.link_name) for name in self.robots_name]
        self._offset = torch.tensor(self.link_offset, device=self.device)

    def compute(self) -> torch.Tensor:
        verts = self.resolve_entity(self.entity_name).get_vertex_positions()
        tools = torch.stack([link_point(link, self._offset) for link in self._links], dim=1)  # (B, n_tools, 3)
        return torch.cdist(tools, verts).amin(dim=(-1, -2))


@METRIC_TERM_REGISTRY.register()
class SoftBodySettled(MetricTerm):
    """The 95th-percentile vertex speed of a soft body (m/s): it has settled at or below ``success_threshold``.

    RoboSimStudio calls a fold done once the towel lies still: 95th-percentile speed under 0.02 m/s.
    """

    entity_name: str = "towel"
    quantile: float = 0.95
    success_threshold: float = 0.02
    direction: MetricDirection = MetricDirection.LIB

    def build(self) -> None:
        self.resolve_entity(self.entity_name)

    def compute(self) -> torch.Tensor:
        speed = self.resolve_entity(self.entity_name).get_vertex_velocities().norm(dim=-1)
        return torch.quantile(speed, self.quantile, dim=-1)


@REWARD_TERM_REGISTRY.register()
class MetricsHold(RewardTerm):
    """1 on a step where every named metric meets its own success threshold (and every ``zero_metrics_name`` reads 0).

    A staged bonus on the way to the task's success: ``["fold_score", "fold_extent"]`` pays for a towel lying folded
    whether or not it has been let go; adding ``zero_metrics_name=["holds"]`` pays for a folded towel the latch has
    let go of. ``metrics_name`` must be success metrics (their thresholds and directions are their own);
    ``zero_metrics_name`` may be shared quantities (``contributes_to_success=False``). With ``scale_metric_name``
    the pay is graded by that metric's value over ``scale_target`` (capped at 1): routed, and the hands half way
    clear of the cable, pays half.
    """

    metrics_name: list[str] = []
    zero_metrics_name: list[str] = []
    scale_metric_name: str = ""
    scale_target: float = 1.0

    def build(self) -> None:
        manager = self._env.metric_manager
        names = [
            *self.metrics_name,
            *self.zero_metrics_name,
            *([self.scale_metric_name] if self.scale_metric_name else []),
        ]
        missing = [name for name in names if manager is None or name not in manager.active_terms]
        if not self.metrics_name or missing:
            raise ValueError(
                f"MetricsHold needs metrics_name naming the task's metrics, got {self.metrics_name} and "
                f"zero_metrics_name {self.zero_metrics_name} (unknown: {missing}); the task's metrics are "
                f"{[] if manager is None else manager.active_terms}."
            )

    def compute(self) -> torch.Tensor:
        manager = self._env.metric_manager
        hold = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for name in self.metrics_name:
            hold &= manager.is_current_step_success(name)
        for name in self.zero_metrics_name:
            hold &= manager.get_value(name) == 0.0
        if not self.scale_metric_name:
            return hold.float()
        grade = (manager.get_value(self.scale_metric_name) / self.scale_target).clamp(0.0, 1.0)
        return torch.where(hold, grade, 0.0)
