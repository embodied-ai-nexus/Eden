"""Terms only the robosuite ``TwoArmLift`` port needs: the ``direction_coef``-gated elevation reward and success.

The pot's handle *sites* (``(0, +-0.155, 0.06)`` in the pot frame) are reproduced with the shared
terms' ``object_offset`` — read off the root pose with ``relative=True``, because the handle child
links' ``get_links_pos`` is shifted by the root's centre-of-mass re-origin while the geoms are not.
"""

from __future__ import annotations

import math

import torch
from genesis.utils.geom import transform_by_quat

from eden.managers.metric_manager import METRIC_TERM_REGISTRY, MetricTerm
from eden.managers.reward_manager import REWARD_TERM_REGISTRY, RewardTerm


class _UprightMixin:
    """robosuite ``direction_coef``: the object's body ``+z`` is within ``max_tilt_deg`` of the world ``+z``."""

    object_name: str
    max_tilt_deg: float

    def _build_upright(self) -> None:
        self.resolve_entity(self.object_name)
        self._z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3).contiguous()
        self._cos_max_tilt = math.cos(math.radians(self.max_tilt_deg))

    def _upright(self) -> torch.Tensor:
        quat = self.resolve_entity(self.object_name).get_quat(relative=True)
        return transform_by_quat(self._z_axis, quat)[:, 2] > self._cos_max_tilt

    def _bottom_z(self, bottom_offset: float) -> torch.Tensor:
        return self.resolve_entity(self.object_name).get_pos(relative=True)[:, 2] - bottom_offset


@METRIC_TERM_REGISTRY.register()
class ObjectLiftedUpright(_UprightMixin, MetricTerm):
    """1.0 when the object's bottom (``root z - bottom_offset``) is above ``height`` and the object is upright.

    robosuite TwoArmLift success: ``pot_center_z - 0.07 > table_top + 0.10``, and its sparse reward
    is zeroed when the pot tilts more than 30 degrees, so the tilt test is part of the predicate here.
    """

    object_name: str = "object"
    bottom_offset: float = 0.07
    height: float = 0.0
    max_tilt_deg: float = 30.0

    def build(self) -> None:
        self._build_upright()

    def compute(self) -> torch.Tensor:
        return ((self._bottom_z(self.bottom_offset) > self.height) & self._upright()).float()


@REWARD_TERM_REGISTRY.register()
class ObjectElevationUpright(_UprightMixin, RewardTerm):
    """``[upright] * clip(bottom_z - table_top - min_elevation, 0, max_elevation)`` — TwoArmLift's lift shaping.

    Zero until the object's bottom is ``min_elevation`` above the table, then linear in height up to
    ``max_elevation`` (robosuite: 5 cm and 15 cm, times 10 — put the 10 in ``weight``). ``(N,)``.
    """

    object_name: str = "object"
    bottom_offset: float = 0.07
    table_top_z: float = 0.0
    min_elevation: float = 0.05
    max_elevation: float = 0.15
    max_tilt_deg: float = 30.0

    def build(self) -> None:
        self._build_upright()

    def compute(self) -> torch.Tensor:
        elevation = self._bottom_z(self.bottom_offset) - self.table_top_z - self.min_elevation
        return self._upright().float() * elevation.clamp(0.0, self.max_elevation)
