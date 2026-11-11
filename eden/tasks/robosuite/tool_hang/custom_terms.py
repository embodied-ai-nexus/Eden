"""Terms only the robosuite ``ToolHang`` port needs.

robosuite's ``_check_frame_assembled`` / ``_check_tool_on_frame`` read MuJoCo *sites* on the frame,
stand and wrench (tip, mount, intersection, hang point, hole centre). Genesis exposes no sites, so
every site is reproduced as a constant offset in the object's root frame and placed by rigid-body
FK from the root pose. The root pose is read with ``relative=True``: Genesis re-origins a free body
onto its inertial frame (the frame's centre of mass is 5 cm from its MJCF origin and its principal
axes are rotated), and the default ``get_pos()`` / ``get_quat()`` report that solver frame. The
poses of the fixed child links (``frame_horizontal_frame``, ``tool_hole1``) are *not* used for
geometry — ``get_links_pos`` reports them shifted by the root's centre-of-mass offset — they only
serve to restrict the contact check to the bar and the ring.
"""

from __future__ import annotations

import math
import torch
from genesis.utils.geom import transform_by_quat

from eden.managers.metric_manager import METRIC_TERM_REGISTRY, MetricTerm
from eden.tasks.robosuite.custom_terms.common import RigMixin
from eden.tasks.robosuite.custom_terms.metrics import shared_per_metric_pass


def _const(term: MetricTerm, values: tuple[float, ...]) -> torch.Tensor:
    """A ``(num_envs, len(values))`` constant buffer for a body-frame offset (built once in ``build()``)."""
    return torch.tensor(values, dtype=torch.float32, device=term.device).expand(term.num_envs, -1).contiguous()


def _point_w(entity, offset: torch.Tensor) -> torch.Tensor:
    """World position of ``offset`` (a robosuite site) given in the entity's MJCF root frame. ``(num_envs, 3)``."""
    return entity.get_pos(relative=True) + transform_by_quat(offset, entity.get_quat(relative=True))


def _opposite_sides(a: torch.Tensor, b: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """``(N,)`` bool: points ``a`` and ``b`` lie on opposite sides of the line along ``axis``.

    Both points are relative to a point on the line.

    robosuite's test, verbatim: ``dot(cross(a, axis), cross(b, axis)) < 0``.
    """
    return (torch.linalg.cross(a, axis) * torch.linalg.cross(b, axis)).sum(dim=-1) < 0


@METRIC_TERM_REGISTRY.register()
class FrameAssembled(MetricTerm):
    """1.0 when the frame stands in the stand's socket — robosuite ``_check_frame_assembled``, verbatim.

    Its three tests, on the sites and geoms of the two assets (constants in their root frames):

    * the stand is upright: ``stand_mount_site - stand_base`` within 10 degrees of vertical;
    * the frame's tip is close to the base: ``||frame_tip_site - stand_base|| < 0.05``;
    * the rod passes between the socket walls: with the rod line from ``frame_mount_site`` to
      ``frame_intersection_site``, walls 0 and 2 lie on opposite sides of it, and so do walls 1 and 3.

    The stand is a fixed fixture here, so the first test always passes; it is kept so the predicate is
    robosuite's.
    """

    frame_name: str = "frame"
    stand_name: str = "stand"
    #: robosuite ``frame_tip_site``, ``frame_mount_site`` and ``frame_intersection_site`` in the frame root frame.
    tip_offset: tuple[float, float, float] = (0.04375, 0.0, -0.13445)
    mount_offset: tuple[float, float, float] = (0.04375, 0.0, -0.09)
    intersection_offset: tuple[float, float, float] = (0.04375, 0.0, 0.08625)
    #: robosuite ``stand_base`` geom centre and ``stand_mount_site`` in the stand root frame.
    base_offset: tuple[float, float, float] = (0.0, 0.0, -0.075)
    stand_mount_offset: tuple[float, float, float] = (0.0, 0.0, 0.08)
    #: Centres of robosuite's ``stand_wall_0`` .. ``stand_wall_3`` in the stand root frame.
    wall_offsets: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0395, 0.005),
        (-0.0055, 0.045, 0.005),
        (0.0, 0.0505, 0.005),
        (0.0055, 0.045, 0.005),
    )
    insertion_tolerance: float = 0.05
    max_tilt_deg: float = 10.0

    def build(self) -> None:
        self.resolve_entity(self.frame_name)
        self.resolve_entity(self.stand_name)
        if len(self.wall_offsets) != 4:
            raise ValueError(
                f"FrameAssembled.wall_offsets lists {len(self.wall_offsets)} walls; robosuite's socket test pairs "
                f"walls 0/2 and 1/3, so it needs exactly 4."
            )
        self._tip = _const(self, self.tip_offset)
        self._mount = _const(self, self.mount_offset)
        self._intersection = _const(self, self.intersection_offset)
        self._base = _const(self, self.base_offset)
        self._stand_mount = _const(self, self.stand_mount_offset)
        self._walls = [_const(self, wall) for wall in self.wall_offsets]
        self._cos_max_tilt = math.cos(math.radians(self.max_tilt_deg))

    @shared_per_metric_pass
    def compute(self) -> torch.Tensor:
        frame = self.resolve_entity(self.frame_name)
        stand = self.resolve_entity(self.stand_name)
        base = _point_w(stand, self._base)
        shaft = _point_w(stand, self._stand_mount) - base
        upright = shaft[:, 2] / torch.norm(shaft, dim=-1) > self._cos_max_tilt
        close = torch.norm(_point_w(frame, self._tip) - base, dim=-1) < self.insertion_tolerance
        hook_end = _point_w(frame, self._mount)
        rod = _point_w(frame, self._intersection) - hook_end
        rod = rod / torch.norm(rod, dim=-1, keepdim=True)
        walls = [_point_w(stand, wall) - hook_end for wall in self._walls]
        between = _opposite_sides(walls[0], walls[2], rod) & _opposite_sides(walls[1], walls[3], rod)
        return (upright & close & between).float()


@METRIC_TERM_REGISTRY.register()
class ToolHoleOnBar(MetricTerm):
    """1.0 when the wrench's hole is threaded on the frame's hook bar.

    The geometric half of robosuite ``_check_tool_on_frame``.

    With ``h`` the bar's free end (``frame_hang_site``), ``u`` the unit bar direction (to
    ``frame_intersection_site``) and ``v`` the hole centre relative to ``h``, robosuite's three
    geometric tests, verbatim:

    * the hole centre is within ``radial_tol`` of the bar line, ``||v - (v.u) u||`` (robosuite:
      ``inner_radius_1 - frame_thickness / 2 = 0.00675``);
    * two opposite ring segments (``tool_hole1_hc_0`` and ``tool_hole1_hc_4``) lie on opposite sides
      of the bar line — the bar passes through the ring;
    * the hole is threaded on, not past the end: ``(v.u) / bar_length`` in ``(0.05, 1.0)``.

    Combine with an :class:`~eden.tasks.robosuite.custom_terms.EntitiesInContact` (ring touches bar)
    and a negated :class:`GripperTouching` (no gripper contact with the wrench) for the full test.
    """

    tool_name: str = "tool"
    frame_name: str = "frame"
    #: robosuite ``tool_hole1_center`` in the wrench root frame.
    hole_offset: tuple[float, float, float] = (-0.0935328, 0.0, 0.0)
    #: Centres of ``tool_hole1_hc_0`` and ``tool_hole1_hc_4`` (robosuite's ``ngeoms // 2``) in the wrench root frame.
    ring_segment_offsets: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (-0.0935328 - 0.012934, 0.0, 0.0),
        (-0.0935328 + 0.012934, 0.0, 0.0),
    )
    #: robosuite ``frame_hang_site`` (free end of the bar) and ``frame_intersection_site`` in the frame root frame.
    hang_offset: tuple[float, float, float] = (-0.0475, 0.0, 0.08625)
    intersection_offset: tuple[float, float, float] = (0.04375, 0.0, 0.08625)
    radial_tol: float = 0.00675
    along_range: tuple[float, float] = (0.05, 1.0)

    def build(self) -> None:
        self.resolve_entity(self.tool_name)
        self.resolve_entity(self.frame_name)
        self._hole = _const(self, self.hole_offset)
        self._segments = [_const(self, offset) for offset in self.ring_segment_offsets]
        self._hang = _const(self, self.hang_offset)
        self._intersection = _const(self, self.intersection_offset)

    @shared_per_metric_pass
    def compute(self) -> torch.Tensor:
        tool = self.resolve_entity(self.tool_name)
        frame = self.resolve_entity(self.frame_name)
        hang = _point_w(frame, self._hang)
        bar = _point_w(frame, self._intersection) - hang
        bar_length = torch.norm(bar, dim=-1)
        u = bar / bar_length.unsqueeze(-1)
        v = _point_w(tool, self._hole) - hang
        t = (v * u).sum(dim=-1)
        radial = torch.norm(v - t.unsqueeze(-1) * u, dim=-1)
        along = t / bar_length
        straddles = _opposite_sides(*(_point_w(tool, segment) - hang for segment in self._segments), u)
        threaded = (along > self.along_range[0]) & (along < self.along_range[1])
        return ((radial < self.radial_tol) & straddles & threaded).float()


@METRIC_TERM_REGISTRY.register()
class GripperTouching(RigMixin, MetricTerm):
    """1.0 when any link of the rig's gripper touches ``object_name`` — robosuite's "robot is not touching the tool".

    Any contact, not a grasp: robosuite's ToolHang counts the wrench as hanging on its own only while
    none of the gripper's geoms touches it (negate this with ``MetricTermNOT``). Configure with
    ``contributes_to_success=False`` when it is only a gate input.
    """

    rig: str = "robot"
    object_name: str = "object"

    def build(self) -> None:
        self._gripper_name = self._rig_options().gripper_name
        self.resolve_entity(self._gripper_name)
        self.resolve_entity(self.object_name)

    @shared_per_metric_pass
    def compute(self) -> torch.Tensor:
        contacts = self.resolve_entity(self._gripper_name).get_contacts(
            with_entity=self.resolve_entity(self.object_name)
        )
        return contacts["valid_mask"].any(dim=-1).float()
