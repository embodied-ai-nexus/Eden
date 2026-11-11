"""Terms of the towel-folding task: RoboSimStudio's fold measures on a Mochi shell.

A fold carries the *moving* half of the cloth over a crease onto the *base* half. Each moving vertex is paired
once, on the rest geometry, with the base vertex its mirror image across the crease lands nearest to:

* :class:`ClothFoldScore` is RoboSimStudio's ``fold_score`` (SoftGym's ClothFold): ``1 - (max(0, g - t) +
  1.2 f) / (g0 - t)``, ``g`` the mean distance between paired vertices now, ``g0`` the same at rest, ``f`` how
  far the base half has drifted since the episode began and ``t`` the thickness of one layer, which a perfect
  fold cannot close. A sheet carried halfway scores 0.75; RoboSimStudio's gate is 0.85.
* :class:`ClothFoldExtent` is its ``fold_extent``: the cloth's span across the crease over the span at the
  start of the episode, 0.5 for a perfect fold; the gate is 0.6.

Both are measured in the cloth's own frame (vertex pairs, and the crease-normal direction recorded at reset),
so a towel placed at a yaw is scored the same way.
"""

from __future__ import annotations

import torch

from eden.constants import MetricDirection
from eden.managers.metric_manager import METRIC_TERM_REGISTRY, MetricTerm

# Weight of the base half's drift in the fold score (RoboSimStudio's FOLD_DRAG_WEIGHT, after SoftGym).
FOLD_DRAG_WEIGHT = 1.2
# How far past the crease (m) a rest vertex must lie to be in the moving half.
_CREASE_TOLERANCE = 1e-6


class _FoldGeometry:
    """The rest-geometry split of a cloth into a moving and a base half across a crease, and their pairing."""

    def __init__(self, rest: torch.Tensor, fold_axis: int, crease_offset: float):
        along = rest[:, fold_axis]
        crease = 0.5 * (along.max() + along.min()) + crease_offset
        # A row of vertices on the crease belongs to the base half; the tolerance keeps float rounding of the rest
        # positions from deciding that.
        self.moving = torch.nonzero(along > crease + _CREASE_TOLERANCE).squeeze(-1)
        self.base = torch.nonzero(along <= crease + _CREASE_TOLERANCE).squeeze(-1)
        if len(self.moving) == 0 or len(self.base) == 0:
            raise ValueError(
                f"The crease at {crease_offset:+.3f} m from the cloth's middle along axis {fold_axis} leaves one half "
                f"empty; keep |crease_offset| below half the cloth's span ({float(along.max() - along.min()) / 2:.3f} m)."
            )
        mirrored = rest[self.moving].clone()
        mirrored[:, fold_axis] = 2.0 * crease - mirrored[:, fold_axis]
        self.pairs = self.base[torch.cdist(mirrored, rest[self.base]).argmin(dim=-1)]
        self.rest_gap = float((rest[self.moving] - rest[self.pairs]).norm(dim=-1).mean())
        # The far edge of the moving half: what a gripper takes hold of to fold.
        self.far_edge = torch.nonzero(along >= along.max() - 1e-6).squeeze(-1)


def _cloth_geometry(term, entity_name: str, fold_axis: int, crease_offset: float) -> _FoldGeometry:
    cloth = term.resolve_entity(entity_name)
    if not getattr(cloth, "is_shell", False):
        raise ValueError(
            f"{type(term).__name__} measures a fold of a Mochi shell, but '{entity_name}' is a "
            f"{type(cloth).__name__}{'' if not hasattr(cloth, 'material') else f' with {type(cloth.material).__name__}'}. "
            f"Give the cloth MochiShellMaterialOptions."
        )
    rest = torch.as_tensor(cloth.init_positions, dtype=torch.float, device=term.device)
    return _FoldGeometry(rest, fold_axis, crease_offset)


@METRIC_TERM_REGISTRY.register()
class ClothFoldScore(MetricTerm):
    """RoboSimStudio's ``fold_score`` of a Mochi shell folded across a crease (success at 0.85 by default).

    Parameters
    ----------
    entity_name : str
        The cloth (a Mochi shell).
    fold_axis : int
        Axis of the cloth's rest geometry the crease is perpendicular to (0: x, 1: y).
    crease_offset : float
        Crease position from the middle of the cloth along ``fold_axis`` (m); the side beyond it moves.
    layer_thickness : float
        Thickness of one cloth layer to anything else (m): twice the shell's ``collider_radius``.
    success_threshold : float
        Score at or above which the fold succeeds (RoboSimStudio's gate).
    """

    entity_name: str = "towel"
    fold_axis: int = 0
    crease_offset: float = 0.0
    layer_thickness: float = 0.006
    success_threshold: float = 0.85

    def build(self) -> None:
        self._fold = _cloth_geometry(self, self.entity_name, self.fold_axis, self.crease_offset)
        cloth = self.resolve_entity(self.entity_name)
        self._base_start = cloth.get_vertex_positions()[:, self._fold.base].clone()

    def compute(self) -> torch.Tensor:
        verts = self.resolve_entity(self.entity_name).get_vertex_positions()
        gap = (verts[:, self._fold.moving] - verts[:, self._fold.pairs]).norm(dim=-1).mean(dim=-1)
        drift = (verts[:, self._fold.base] - self._base_start).norm(dim=-1).mean(dim=-1)
        forgiven = (gap - self.layer_thickness).clamp_min(0.0)
        return 1.0 - (forgiven + FOLD_DRAG_WEIGHT * drift) / (self._fold.rest_gap - self.layer_thickness)

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        # The reset events have placed this episode's cloth by now: its base half is where drift is measured from.
        envs_idx = slice(None) if envs_idx is None else envs_idx
        verts = self.resolve_entity(self.entity_name).get_vertex_positions()
        self._base_start[envs_idx] = verts[envs_idx][:, self._fold.base]


@METRIC_TERM_REGISTRY.register()
class ClothFoldExtent(MetricTerm):
    """RoboSimStudio's ``fold_extent``: the cloth's span across the crease over its span at the episode's start.

    0.5 for a perfect half fold; succeeds at or below ``success_threshold`` (0.6, RoboSimStudio's gate). The
    span is measured along the crease normal the cloth had when the episode began (the direction from the
    base half's centroid to the moving half's), so it follows the cloth's yaw.
    """

    entity_name: str = "towel"
    fold_axis: int = 0
    crease_offset: float = 0.0
    success_threshold: float = 0.6
    direction: MetricDirection = MetricDirection.LIB

    def build(self) -> None:
        self._fold = _cloth_geometry(self, self.entity_name, self.fold_axis, self.crease_offset)
        self._normal = torch.zeros(self.num_envs, 3, device=self.device)
        self._start_span = torch.ones(self.num_envs, device=self.device)
        self.reset()

    def _span(self, verts: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
        along = (verts * normal[:, None, :]).sum(dim=-1)
        return along.amax(dim=-1) - along.amin(dim=-1)

    def compute(self) -> torch.Tensor:
        verts = self.resolve_entity(self.entity_name).get_vertex_positions()
        return self._span(verts, self._normal) / self._start_span

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        envs_idx = slice(None) if envs_idx is None else envs_idx
        verts = self.resolve_entity(self.entity_name).get_vertex_positions()[envs_idx]
        normal = verts[:, self._fold.moving].mean(dim=1) - verts[:, self._fold.base].mean(dim=1)
        normal[:, 2] = 0.0
        self._normal[envs_idx] = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        self._start_span[envs_idx] = self._span(verts, self._normal[envs_idx])
