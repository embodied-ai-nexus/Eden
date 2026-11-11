"""Terms only the robosuite ``TwoArmPegInHole`` port needs.

robosuite's ``_compute_orientation()`` reduces the peg / hole pose pair to three numbers that the
reward, the success test and three observables all read:

* ``t`` — signed distance of the hole centre from the peg centre **along the peg axis**,
* ``d`` — perpendicular distance of the hole centre from the peg's axis line,
* ``cos`` — ``|hole_normal . peg_axis|``, 1 when the peg is square to the plate.

The peg axis is the peg body's local z; the hole centre is the plate body origin shifted by
``(0.1, 0, 0)`` in the plate frame (robosuite uses 0.1 here although the hole is cut at 0.11).
:class:`_PegHoleGeometryMixin` computes the three from the two entity poses; the four metric terms
below each expose one of them (or the success predicate over all three). They recompute the same
handful of ``(N, 3)`` ops rather than reading one another — metric terms have no ordering
guarantee among themselves, and the work is a few vector products per env.

The two reward terms turn a metric or an entity-pair distance into robosuite's ``1 - tanh(.)``
shaping without recomputing the geometry.
"""

from __future__ import annotations

import torch
from genesis.utils.geom import transform_by_quat

from eden.managers.metric_manager import METRIC_TERM_REGISTRY, MetricTerm
from eden.managers.reward_manager import REWARD_TERM_REGISTRY, RewardTerm


class _PegHoleGeometryMixin:
    """Resolve the peg / hole entities in ``build()``; read robosuite's ``(t, d, cos)`` each step."""

    peg_name: str
    hole_name: str
    hole_center_offset: tuple[float, float, float]

    def _build_peg_hole(self) -> None:
        self.resolve_entity(self.peg_name)
        self.resolve_entity(self.hole_name)
        z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device)
        self._z_axis = z_axis.expand(self.num_envs, 3).contiguous()
        offset = torch.tensor(self.hole_center_offset, dtype=torch.float32, device=self.device)
        self._center_offset = offset.expand(self.num_envs, 3).contiguous()

    def _peg_hole_geometry(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(t, d, cos)`` of robosuite's ``_compute_orientation``, each ``(num_envs,)``."""
        peg = self.resolve_entity(self.peg_name)
        hole = self.resolve_entity(self.hole_name)
        # relative=True: robosuite reads ``body_xpos`` / ``body_xmat``, and ``hole_center_offset`` is
        # a plate-frame number. The default solver frame is re-origined onto the inertial frame.
        peg_pos, peg_quat = peg.get_pos(relative=True), peg.get_quat(relative=True)
        hole_pos, hole_quat = hole.get_pos(relative=True), hole.get_quat(relative=True)
        peg_axis = transform_by_quat(self._z_axis, peg_quat)
        hole_normal = transform_by_quat(self._z_axis, hole_quat)
        center = hole_pos + transform_by_quat(self._center_offset, hole_quat)
        t = ((center - peg_pos) * peg_axis).sum(dim=-1)
        d = torch.linalg.cross(peg_axis, peg_pos - center, dim=-1).norm(dim=-1)
        cos = (hole_normal * peg_axis).sum(dim=-1).abs()
        return t, d, cos


@METRIC_TERM_REGISTRY.register()
class PegHoleAxialOffset(_PegHoleGeometryMixin, MetricTerm):
    """robosuite's ``t``: signed offset of the hole centre from the peg centre along the peg axis (m)."""

    peg_name: str = "peg"
    hole_name: str = "hole"
    hole_center_offset: tuple[float, float, float] = (0.1, 0.0, 0.0)

    def build(self) -> None:
        self._build_peg_hole()

    def compute(self) -> torch.Tensor:
        t, _, _ = self._peg_hole_geometry()
        return t


@METRIC_TERM_REGISTRY.register()
class PegHolePerpDistance(_PegHoleGeometryMixin, MetricTerm):
    """robosuite's ``d``: perpendicular distance of the hole centre from the peg's axis line (m)."""

    peg_name: str = "peg"
    hole_name: str = "hole"
    hole_center_offset: tuple[float, float, float] = (0.1, 0.0, 0.0)

    def build(self) -> None:
        self._build_peg_hole()

    def compute(self) -> torch.Tensor:
        _, d, _ = self._peg_hole_geometry()
        return d


@METRIC_TERM_REGISTRY.register()
class PegHoleAxisCos(_PegHoleGeometryMixin, MetricTerm):
    """robosuite's ``cos``: ``|hole_normal . peg_axis|``, 1 when the peg is square to the plate."""

    peg_name: str = "peg"
    hole_name: str = "hole"
    hole_center_offset: tuple[float, float, float] = (0.1, 0.0, 0.0)

    def build(self) -> None:
        self._build_peg_hole()

    def compute(self) -> torch.Tensor:
        _, _, cos = self._peg_hole_geometry()
        return cos


@METRIC_TERM_REGISTRY.register()
class PegInHole(_PegHoleGeometryMixin, MetricTerm):
    """robosuite's ``_check_success``: ``d < d_max and t_min <= t <= t_max and cos > cos_min``.

    The defaults are robosuite's (6 cm off-axis, the hole centre between 12 cm behind and 14 cm
    ahead of the peg centre along a 26 cm peg, within ~18 degrees of square).
    """

    peg_name: str = "peg"
    hole_name: str = "hole"
    hole_center_offset: tuple[float, float, float] = (0.1, 0.0, 0.0)
    d_max: float = 0.06
    t_min: float = -0.12
    t_max: float = 0.14
    cos_min: float = 0.95

    def build(self) -> None:
        self._build_peg_hole()

    def compute(self) -> torch.Tensor:
        t, d, cos = self._peg_hole_geometry()
        return ((d < self.d_max) & (t >= self.t_min) & (t <= self.t_max) & (cos > self.cos_min)).float()


@REWARD_TERM_REGISTRY.register()
class EntitiesDistanceTanh(RewardTerm):
    """``1 - tanh(coef * ||pos(object_b) - pos(object_a)||)`` between two entity roots. ``(N,)`` in ``[0, 1]``."""

    object_a_name: str = "object_a"
    object_b_name: str = "object_b"
    coef: float = 1.0

    def build(self) -> None:
        self.resolve_entity(self.object_a_name)
        self.resolve_entity(self.object_b_name)

    def compute(self) -> torch.Tensor:
        # relative=True on both, as in ``ObjectToObject``: robosuite measures between body origins.
        dist = torch.norm(
            self.resolve_entity(self.object_b_name).get_pos(relative=True)
            - self.resolve_entity(self.object_a_name).get_pos(relative=True),
            dim=-1,
        )
        return 1.0 - torch.tanh(self.coef * dist)


@REWARD_TERM_REGISTRY.register()
class OneMinusTanhMetric(RewardTerm):
    """``1 - tanh(coef * m)`` of a metric's cached value ``m`` (``|m|`` with ``absolute=True``). ``(N,)``.

    The tanh-shaped counterpart of ``rewards.MetricReward``: the metric manager runs before the
    reward manager, so the value read here is this step's.
    """

    metric_name: str = ""
    coef: float = 1.0
    absolute: bool = False

    def build(self) -> None:
        # The metric manager is constructed ahead of the reward manager (RLEnvBase._manager_order), so
        # the name resolves here and an unknown one raises at env construction, listing the active
        # metrics, rather than on the first step.
        self._env.metric_manager.get_value(self.metric_name)

    def compute(self) -> torch.Tensor:
        value = self._env.metric_manager.get_value(self.metric_name)
        if self.absolute:
            value = value.abs()
        return 1.0 - torch.tanh(self.coef * value)
