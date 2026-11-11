"""Terms only the robosuite ``NutAssembly`` port needs: :class:`ObjectNearPointXY`, the nut-on-peg test."""

from __future__ import annotations

import torch

from eden.managers.metric_manager import METRIC_TERM_REGISTRY, MetricTerm
from eden.managers.terms.utils import resolve_link_idx


@METRIC_TERM_REGISTRY.register()
class ObjectNearPointXY(MetricTerm):
    """1.0 when the object's xy is within ``radius`` (per axis, a box) of ``point_xy`` and it is below ``z_max``.

    robosuite's nut-on-peg test: ``|dx| < 0.03 and |dy| < 0.03 and z < table_top + 0.05``.
    ``object_link_name`` measures a link instead of the root.
    """

    object_name: str = "object"
    object_link_name: str | None = None
    point_xy: tuple[float, float] = (0.0, 0.0)
    radius: float = 0.03
    z_max: float = float("inf")

    def build(self) -> None:
        obj = self.resolve_entity(self.object_name)
        self._link = resolve_link_idx(obj, [self.object_link_name], self.device) if self.object_link_name else None
        self._point = torch.tensor(self.point_xy, device=self.device, dtype=torch.float32)

    def compute(self) -> torch.Tensor:
        obj = self.resolve_entity(self.object_name)
        # relative=True: the box and z_max are robosuite numbers in the asset's own frame. The
        # default solver frame is re-origined onto the centre of mass (1.1 cm off for the square
        # nut), and since the reset randomises yaw the offset would displace the test in a
        # different direction every episode.
        pos = (
            obj.get_pos(relative=True)
            if self._link is None
            else obj.get_links_pos(ls_idx_local=self._link, relative=True)[:, 0]
        )
        near = ((pos[:, :2] - self._point).abs() < self.radius).all(dim=-1)
        return (near & (pos[:, 2] < self.z_max)).float()
