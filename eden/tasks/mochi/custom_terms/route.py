"""Terms of ``mochi_route_the_cable``: RoboSimStudio's ``routed`` goal for a Mochi rod (``route_the_cable/board.py``).

RoboSimStudio's cable is a chain of 50 capsules, and its goal counts capsule centres in boxes: at least two of
the fifty in each spool's seat (3 to 45 mm off its axis on the side the cable must pass, 35 mm either way along
the line), three in the port's 90 mm box, and the free end in the port. A Mochi rod's segments are the capsules:
:class:`RodShareInBox` counts their midpoints, :class:`RodEndInBox` places the free end.
"""

from __future__ import annotations

import torch

from eden.managers.metric_manager import METRIC_TERM_REGISTRY, MetricTerm
from eden.managers.reward_manager import REWARD_TERM_REGISTRY, RewardTerm
from eden.utils.string import term_subject


def _resolve_rod(term, entity_name: str):
    rod = term.resolve_entity(entity_name)
    if not getattr(rod, "is_rod", False):
        raise ValueError(
            f"{type(term).__name__} measures a Mochi rod, but '{entity_name}' is a {type(rod).__name__}"
            f"{' that is not a rod' if hasattr(rod, 'is_rod') else ''}. Give the cable RodOptions with a "
            f"MochiRodMaterialOptions material."
        )
    return rod


def _box(term, lower: tuple[float, float, float], upper: tuple[float, float, float]) -> tuple[torch.Tensor, ...]:
    if any(lo > hi for lo, hi in zip(lower, upper)):
        raise ValueError(f"{type(term).__name__} box has lower {lower} above upper {upper} on some axis.")
    return (
        torch.tensor(lower, dtype=torch.float, device=term.device),
        torch.tensor(upper, dtype=torch.float, device=term.device),
    )


@METRIC_TERM_REGISTRY.register()
class RodShareInBox(MetricTerm):
    """Share of a rod's segments whose midpoints lie in an axis-aligned world box (success at ``success_threshold``).

    Parameters
    ----------
    entity_name : str
        The rod.
    lower, upper : tuple[float, float, float]
        The box's corners, world frame (m).
    """

    entity_name: str = "cable"
    lower: tuple[float, float, float] = (0.0, 0.0, 0.0)
    upper: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def build(self) -> None:
        _resolve_rod(self, self.entity_name)
        self._lower, self._upper = _box(self, self.lower, self.upper)

    def compute(self) -> torch.Tensor:
        nodes = self.resolve_entity(self.entity_name).get_vertex_positions()
        centres = 0.5 * (nodes[:, 1:] + nodes[:, :-1])
        inside = ((centres >= self._lower) & (centres <= self._upper)).all(dim=-1)
        return inside.float().mean(dim=-1)


@METRIC_TERM_REGISTRY.register()
class RodEndInBox(MetricTerm):
    """1 when a rod's free end (its last node) lies in an axis-aligned world box, else 0.

    Parameters
    ----------
    entity_name : str
        The rod.
    lower, upper : tuple[float, float, float]
        The box's corners, world frame (m).
    """

    entity_name: str = "cable"
    lower: tuple[float, float, float] = (0.0, 0.0, 0.0)
    upper: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def build(self) -> None:
        _resolve_rod(self, self.entity_name)
        self._lower, self._upper = _box(self, self.lower, self.upper)

    def compute(self) -> torch.Tensor:
        end = self.resolve_entity(self.entity_name).get_vertex_positions()[:, -1]
        return ((end >= self._lower) & (end <= self._upper)).all(dim=-1).float()


@REWARD_TERM_REGISTRY.register()
class RodNodeToPoint(RewardTerm):
    """``exp(-d^2 / std^2)``, ``d`` from one node of a rod to a fixed world point: shaping toward one placement.

    RoboSimStudio's routine lays the cable down at fixed marks, a point along it at a time (beside the first spool,
    across the line, round the second, into the port); one term per mark pays for that point of
    the cable reaching it, in whatever order.

    Parameters
    ----------
    entity_name : str
        The rod.
    node : int
        The node (negative counts from the free end).
    point : tuple[float, float, float]
        The mark, world frame (m).
    std : float
        Kernel width (m).
    """

    entity_name: str = "cable"
    node: int = -1
    point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: float = 0.05

    def build(self) -> None:
        rod = _resolve_rod(self, self.entity_name)
        if not -rod.n_vertices <= self.node < rod.n_vertices:
            raise ValueError(
                f"{term_subject(self)}: node {self.node} is not a node of the rod, which has {rod.n_vertices} "
                f"(0..{rod.n_vertices - 1}, or negative from the end)."
            )
        self._node = self.node % rod.n_vertices
        self._point = torch.tensor(self.point, dtype=torch.float, device=self.device)

    def compute(self) -> torch.Tensor:
        node = self.resolve_entity(self.entity_name).get_vertex_positions()[:, self._node]
        return torch.exp(-(node - self._point).square().sum(dim=-1) / self.std**2)
