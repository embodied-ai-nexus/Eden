"""The writing tools: how a Chinese brush and a pointed pen mark the sheet, batched in torch.

Both are kinematic models, as in RoboPaint's: what a tool lays down follows from where its tip is, not from a
contact force. A sample is the tip's point on the sheet ``(u, v)`` and its ``depth``, how far below the paper's
surface the *undeformed* tip has gone (0 or less: in the air). The hairs and the nib do not collide; the rest of the
tool does, so pressing deeper than they are long drives the ferrule into the paper.

* :class:`DynamicBrush` — the dynamic virtual brush of Wang, Chen, Deng, Hutchinson and Dellaert, *Robot
  Calligraphy using Pseudospectral Optimal Control in Conjunction with a Novel Dynamic Brush Model* (IROS 2020),
  with that paper's fitted width, drag and offset (eqs. 1, 6-8, Fig. 3) and RoboPaint's ``wang2020`` settings.
* :class:`PointedPen` — a pointed (flexible) pen whose tines spread under pressure: a hairline at a light touch,
  swelling to a shade across the nib, RoboPaint's ``pointed_pen`` settings.

A tool's :meth:`~WritingTool.step` advances its state by one sample and returns that sample's :class:`Footprint`:
a mark in its own frame on the sheet, whose shape :meth:`~WritingTool.inside` tests. The sheet's ink
(:class:`~eden.tasks.art.calligraphy.grading.SheetInk`) is the union of the footprints.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class Footprint:
    """One mark per env, ``(B,)`` each: its frame on the sheet and its size.

    ``root`` ``(B, 2)`` is the frame's origin and ``along`` ``(B, 2)`` its unit first axis, both in the sheet's
    ``(u, v)``; the second axis is ``along`` turned 90 degrees counter-clockwise. ``length`` and ``width`` size the
    shape along and across it (m), and ``live`` is False where the tool lays nothing.
    """

    root: torch.Tensor
    along: torch.Tensor
    length: torch.Tensor
    width: torch.Tensor
    live: torch.Tensor


def stack_footprints(footprints: list[Footprint]) -> Footprint:
    """``footprints`` stacked along a new sample axis: every field ``(B, S, ...)``."""
    return Footprint(
        *(torch.stack([getattr(f, name) for f in footprints], dim=1) for name in Footprint.__dataclass_fields__)
    )


class WritingTool:
    """A writing tool's state per env, advanced one sample at a time.

    ``reach`` (m) bounds how far any point a footprint inks lies from its :meth:`centre`.
    """

    reach: float

    def step(self, tip: torch.Tensor, depth: torch.Tensor, lateral: torch.Tensor) -> Footprint:
        """Advance by one sample: the tip at ``tip (B, 2)`` on the sheet, ``depth (B,)`` below the paper, the tool's
        lateral axis projected on the sheet ``lateral (B, 2)`` (not normalized). Returns the sample's footprint.
        """
        raise NotImplementedError

    def centre(self, footprint: Footprint) -> torch.Tensor:
        """The point ``(..., 2)`` a footprint's ink lies within :attr:`reach` of."""
        raise NotImplementedError

    def inside(self, s: torch.Tensor, t: torch.Tensor, length: torch.Tensor, width: torch.Tensor) -> torch.Tensor:
        """Whether the points ``(s, t)`` of a footprint's frame lie in its mark (broadcasting)."""
        raise NotImplementedError

    def width_at(self, depth: torch.Tensor) -> torch.Tensor:
        """The mark's settled width (m) at ``depth``: what a tool held there long enough draws."""
        raise NotImplementedError

    def reset(self, envs_idx) -> None:
        """Forget ``envs_idx``'s state: its next sample starts the tool afresh, undeformed, where its tip is."""
        raise NotImplementedError


def _polynomial(coefficients: tuple[float, ...], x: torch.Tensor) -> torch.Tensor:
    """``coefficients`` highest power first, evaluated at ``x`` (Horner)."""
    out = torch.zeros_like(x)
    for c in coefficients:
        out = out * x + c
    return out


class DynamicBrush(WritingTool):
    """Wang et al.'s dynamic virtual brush (IROS 2020), per env.

    The brush's state is its width ``w``, its drag ``d``, and the root of its mark: the middle of the mark's flat
    end, ``o`` away from the tip along the unit ``along`` (the way the hairs bend). The mark runs from the root
    ``d`` further along ``along``, its flat end ``w`` wide, narrowing on a parabola to a rounded point (Fig. 3)::

        0 <= s <= d,   |t| <= (w / 2) sqrt(1 - s / d)

    Each sample at depth ``z`` (clamped to ``[0, z_max]``) updates it as the paper's eqs. 6-8, with the fitted
    ``Width``, ``Drag`` and ``Offset`` of ``z`` (each kept >= 0)::

        w <- k_w w + (1 - k_w) Width(z)          d <- k_d d + (1 - k_d) Drag(z)
        o <- min(Offset(z), |root - tip|)       along <- (root - tip) / |root - tip|,   root <- tip + o along

    So the hairs' root stays put while the tip moves, until the tip has gone ``Offset(z)`` from it; then it is
    dragged after the tip. Both updates wait until the tip is more than ``theta_eps`` from the root
    (``theta_eps_contact`` while the brush touches the paper), where the direction is defined: until then the root
    and ``along`` stay as they are. That is where this differs from a literal reading of eq. 8, which in the air
    (``Offset`` 0) would snap the root onto the tip every sample and so turn ``along`` only for samples more than
    ``theta_eps`` apart: here the root trails the tip by up to ``theta_eps`` in the air, so ``along`` follows the
    way the brush travels at any speed, and a brush lowered onto the paper starts with its hairs trailing the way
    it came. A brush with no state yet (built, or reset) starts undeformed at the tip, its hairs trailing toward the
    sheet's ``-v``.

    Parameters
    ----------
    num_envs : int
    width, drag : tuple[float, float]
        ``Width(z) = a z + b`` and ``Drag(z)``, SI: the paper's ``(1.187, -0.03)`` and ``(1.075, 0.138)`` cm per cm
        are ``(1.187, -0.0003)`` and ``(1.075, 0.00138)``.
    offset : tuple[float, float, float]
        ``Offset(z) = a z^2 + b z + c``, SI: the paper's ``(-0.967, 2.11, -0.036)`` in cm is
        ``(-96.7, 2.11, -0.00036)``.
    z_max : float
        Depth (m) beyond which the brush deforms no further: RoboPaint's 1.5 cm.
    k_w, k_d : float
        The width's and the drag's inertia per sample: the paper's 0.02.
    theta_eps, theta_eps_contact : float
        The root-to-tip distance (m) below which ``along`` keeps its direction, in the air and on the paper.
    device : torch.device | str
    """

    def __init__(
        self,
        num_envs: int,
        *,
        width: tuple[float, float] = (1.187, -0.0003),
        drag: tuple[float, float] = (1.075, 0.00138),
        offset: tuple[float, float, float] = (-96.7, 2.11, -0.00036),
        z_max: float = 0.015,
        k_w: float = 0.02,
        k_d: float = 0.02,
        theta_eps: float = 1.0e-4,
        theta_eps_contact: float = 1.0e-3,
        device="cpu",
    ) -> None:
        self.width_poly, self.drag_poly, self.offset_poly = width, drag, offset
        self.z_max, self.k_w, self.k_d = z_max, k_w, k_d
        self.theta_eps, self.theta_eps_contact = theta_eps, theta_eps_contact
        depths = torch.linspace(0.0, z_max, 301, dtype=torch.float64)
        # How far the mark reaches from its centre (halfway along it): its flat end's corners, half its drag along
        # and half its width across, at the depth where that is largest. The filtered width and drag never exceed
        # their largest settled values, so no mark reaches further.
        self.reach = float(torch.hypot(self._drag(depths), self._width(depths)).max()) / 2
        self.w = torch.zeros(num_envs, device=device)
        self.d = torch.zeros(num_envs, device=device)
        self.root = torch.zeros(num_envs, 2, device=device)
        self.along = torch.zeros(num_envs, 2, device=device)
        self.started = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # Built once: a constant made from a Python value on every step is a blocking copy to the GPU each time.
        self._trailing = torch.tensor([0.0, -1.0], device=device)

    def _width(self, z: torch.Tensor) -> torch.Tensor:
        return _polynomial(self.width_poly, z).clamp_min(0.0)

    def _drag(self, z: torch.Tensor) -> torch.Tensor:
        return _polynomial(self.drag_poly, z).clamp_min(0.0)

    def _offset(self, z: torch.Tensor) -> torch.Tensor:
        return _polynomial(self.offset_poly, z).clamp_min(0.0)

    def width_at(self, depth: torch.Tensor) -> torch.Tensor:
        return self._width(depth.clamp(0.0, self.z_max))

    def step(self, tip: torch.Tensor, depth: torch.Tensor, lateral: torch.Tensor) -> Footprint:
        z = depth.clamp(0.0, self.z_max)
        width, drag = self._width(z), self._drag(z)
        fresh = ~self.started
        trailing = self._trailing.expand_as(tip)
        root = torch.where(fresh[:, None], tip, self.root)
        along = torch.where(fresh[:, None], trailing, self.along)
        # The state is written in place: the metric steps the brush under inference mode, and a reset outside it
        # writes these same tensors.
        self.w.copy_(torch.where(fresh, width, self.k_w * self.w + (1.0 - self.k_w) * width))
        self.d.copy_(torch.where(fresh, drag, self.k_d * self.d + (1.0 - self.k_d) * drag))
        away = root - tip
        distance = away.norm(dim=-1)
        pulled = distance > torch.where(depth > 0.0, self.theta_eps_contact, self.theta_eps)
        along = torch.where(pulled[:, None], away / distance.clamp_min(1e-12)[:, None], along)
        offset = torch.minimum(self._offset(z), distance)
        self.root.copy_(torch.where(pulled[:, None], tip + offset[:, None] * along, root))
        self.along.copy_(along)
        self.started.fill_(True)
        live = (depth > 0.0) & (self.w > 0.0) & (self.d > 0.0)
        return Footprint(self.root.clone(), self.along.clone(), self.d.clone(), self.w.clone(), live)

    def centre(self, footprint: Footprint) -> torch.Tensor:
        return footprint.root + footprint.along * (footprint.length / 2)[..., None]

    def inside(self, s, t, length, width):
        fraction = (s / length.clamp_min(1e-12)).clamp(max=1.0)
        return (s >= 0.0) & (s <= length) & (2.0 * t.abs() <= width * (1.0 - fraction).clamp_min(0.0).sqrt())

    def reset(self, envs_idx) -> None:
        self.started[envs_idx] = False


class PointedPen(WritingTool):
    """A pointed pen, per env: the tines spread across the nib as it is pressed.

    The mark is an ellipse at the tip, ``hairline`` long along the slit and ``w(z)`` wide across the nib (along the
    pen's lateral axis on the sheet): a hairline up to ``flex_start`` deep, swelling linearly to ``max_width`` at
    ``full_flex`` and no further. A stroke pulled along the slit is as wide as the tines are spread (a shade); one
    across it stays a hairline, as with a real pointed pen. Where the lateral axis stands nearly upright on the
    sheet (its projection under 0.1), the mark is a hairline circle. The pen has no state.

    Parameters
    ----------
    hairline, max_width : float
        The mark's width (m) unflexed and fully flexed: RoboPaint's 0.4 and 2 mm.
    flex_start, full_flex : float
        Depths (m) where the tines start to spread and where they are fully spread: RoboPaint's 1 and 3.5 mm.
    """

    def __init__(
        self,
        *,
        hairline: float = 0.0004,
        max_width: float = 0.002,
        flex_start: float = 0.001,
        full_flex: float = 0.0035,
    ) -> None:
        if not 0.0 < hairline <= max_width or not 0.0 <= flex_start < full_flex:
            raise ValueError(
                "PointedPen: needs 0 < hairline <= max_width and 0 <= flex_start < full_flex, got "
                f"hairline={hairline}, "
                f"max_width={max_width}, flex_start={flex_start}, full_flex={full_flex}."
            )
        self.hairline, self.max_width = hairline, max_width
        self.flex_start, self.full_flex = flex_start, full_flex
        self.reach = max_width / 2

    def width_at(self, depth: torch.Tensor) -> torch.Tensor:
        flex = ((depth - self.flex_start) / (self.full_flex - self.flex_start)).clamp(0.0, 1.0)
        return self.hairline + (self.max_width - self.hairline) * flex

    def step(self, tip: torch.Tensor, depth: torch.Tensor, lateral: torch.Tensor) -> Footprint:
        norm = lateral.norm(dim=-1)
        upright = norm < 0.1
        across = lateral / norm.clamp_min(1e-12)[:, None]
        along = torch.stack([across[:, 1], -across[:, 0]], dim=-1)  # turned 90 degrees clockwise: `across` is second
        # +u for an upright pen, made on the device: a tensor of Python values would be a blocking copy each step.
        plus_u = torch.stack([torch.ones_like(norm), torch.zeros_like(norm)], dim=-1)
        along = torch.where(upright[:, None], plus_u, along)
        width = torch.where(upright, self.hairline, self.width_at(depth))
        return Footprint(tip, along, torch.full_like(depth, self.hairline), width, depth > 0.0)

    def centre(self, footprint: Footprint) -> torch.Tensor:
        return footprint.root

    def inside(self, s, t, length, width):
        return (2.0 * s / length) ** 2 + (2.0 * t / width) ** 2 <= 1.0

    def reset(self, envs_idx) -> None:
        pass


#: The writing tools (the art suite's tools these tasks hold, ``eden.tasks.art.workcell.TOOLS``).
WritingToolKind = Literal["brush", "pen"]

#: Each writing tool's model for ``num_envs`` envs on ``device``, RoboPaint's settings.
TOOL_MODELS: dict[WritingToolKind, Callable[[int, torch.device | str], WritingTool]] = {
    "brush": lambda num_envs, device: DynamicBrush(num_envs, device=device),
    "pen": lambda num_envs, device: PointedPen(),
}


def lerp_samples(start: torch.Tensor, end: torch.Tensor, count: int) -> list[torch.Tensor]:
    """``count`` points from just past ``start`` to ``end`` (``end`` last), evenly spaced: a sample's sub-steps."""
    return [start + (end - start) * ((k + 1) / count) for k in range(count)]


def tool_reach_cells(tool: WritingTool, cell: float) -> int:
    """The half-side, in cells, of a square window round a footprint's centre cell that holds all its ink."""
    return math.ceil(tool.reach / cell) + 1
