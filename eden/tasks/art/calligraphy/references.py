"""What a work to copy is: a plan of the tool's tip, stroke by stroke, and the target its tool draws from it.

A RoboPaint writing work is a target image and a stroke budget; its exemplars (calligraphy rubbings, a 1741
copybook) and the brush runs it rendered them with are not released. The works here are drawn for Eden instead:
each is a :class:`Plan`, the path of the tool's tip through every stroke in the art suite's plan format
(:mod:`~eden.tasks.art.plans`), and its target is what the work's own tool (:mod:`~eden.tasks.art.calligraphy.tools`)
draws along that path (:func:`render_plan`). A copy can therefore match its target exactly, and a plan replayed
through the robot (the scripted writer) shows the work can be written.

The works themselves (their key points and the tasks' :data:`~eden.tasks.art.calligraphy.works.WORKS`) are in
:mod:`~eden.tasks.art.calligraphy.works`.
"""

from __future__ import annotations

import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import torch

from eden.tasks.art.calligraphy.grading import SheetInk
from eden.tasks.art.calligraphy.tools import TOOL_MODELS, Footprint, WritingTool, WritingToolKind, stack_footprints
from eden.tasks.art.plans import SAMPLE_SPACING, Stroke, tip_path
from eden.tasks.art.workcell import sheet_grid
from eden.utils.plane_grid import PlaneGrid


@dataclass(frozen=True)
class Plan:
    """A work's strokes (:class:`~eden.tasks.art.plans.Stroke`) in writing order, and the tool that writes them
    (``"brush"`` or ``"pen"``).
    """

    tool: WritingToolKind
    strokes: tuple[Stroke, ...]

    def __post_init__(self) -> None:
        if self.tool not in typing.get_args(WritingToolKind):
            raise ValueError(
                f"Plan: unknown tool {self.tool!r}; the writing tools are {list(typing.get_args(WritingToolKind))}."
            )

    def path(self, spacing: float = SAMPLE_SPACING) -> tuple[np.ndarray, np.ndarray]:
        """The tip's whole path ``(m, 3)``, travel between strokes included, and the stroke each sample belongs to
        ``(m,)`` (``-1`` on the way between strokes): :func:`~eden.tasks.art.plans.tip_path`.
        """
        return tip_path(self.strokes, spacing=spacing)


@dataclass(frozen=True)
class Work:
    """A work to copy: its plan, the sheet it is written on and the grid its ink is kept on."""

    name: str
    title: str
    plan: Plan
    #: The sheet's width and height (m), centred on the sheet frame's origin.
    sheet: tuple[float, float]
    #: The ink grid's cell (m).
    cell: float
    #: The tool's lateral axis on the sheet as the plan holds it (``(u, v)``, unit): what spreads a pen's tines.
    lateral: tuple[float, float] = (1.0, 0.0)
    #: RoboPaint's thresholds where this work's differ from its writing defaults: CalligraphyWriting parameters
    #: (``iou``, ``boundary_f``, ``min_stroke_coverage``, …), which a task's metric is configured with. Read-only,
    #: and out of the hash (a mapping has none), so a work stays hashable.
    rule: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}), hash=False)
    about: str = field(default="", compare=False)

    @property
    def grid(self) -> PlaneGrid:
        return sheet_grid(self.sheet, self.cell)


@dataclass(frozen=True)
class Rendering:
    """A plan drawn by its tool: the target and what the hand rules are scaled from."""

    #: ``(rows, cols)``: the cells the plan inks.
    target: np.ndarray
    #: ``(rows, cols)``: the stroke each target cell belongs to (the first that inks it), ``-1`` off the target.
    owner: np.ndarray
    #: The tip's path on the paper (m), and the mark's mean width along it (m).
    path_length: float
    mark_width: float


def run_tool(tool: WritingTool, path: np.ndarray, lateral: tuple[float, float], device) -> list[Footprint]:
    """``tool`` (one env) along ``path (m, 3)`` with its lateral axis held at ``lateral``: every sample's footprint."""
    path_t = torch.as_tensor(path, dtype=torch.float32, device=device)
    lateral_t = torch.tensor([lateral], dtype=torch.float32, device=device)
    return [tool.step(p[None, :2], p[None, 2], lateral_t) for p in path_t]


def render_plan(work: Work, *, device="cpu") -> Rendering:
    """Draw ``work``'s plan with a fresh tool of its kind, its lateral axis held at the work's :attr:`Work.lateral`."""
    path, stroke_of = work.plan.path()
    tool = TOOL_MODELS[work.plan.tool](1, device)
    footprints = stack_footprints(run_tool(tool, path, work.lateral, device))  # (1, m)
    strokes = len(work.plan.strokes)
    # One "env" per stroke, each laying only its own stroke's footprints.
    owner_t = torch.as_tensor(stroke_of, device=device)
    mine = owner_t[None, :] == torch.arange(strokes, device=device)[:, None]  # (K, m)
    per_stroke = Footprint(
        *(
            getattr(footprints, f).expand(strokes, *getattr(footprints, f).shape[1:])
            for f in ("root", "along", "length", "width")
        ),
        footprints.live & mine,
    )
    ink = SheetInk(work.grid, strokes, tool, device=device)
    ink.deposit(per_stroke)
    inked = ink.inked.cpu().numpy()  # (K, rows, cols)
    target = inked.any(axis=0)
    owner = np.where(target, inked.argmax(axis=0), -1)
    touching = path[:, 2] > 0.0
    step = np.r_[0.0, np.hypot(*np.diff(path[:, :2], axis=0).T)] * (touching & np.r_[False, touching[:-1]])
    widths = footprints.width[0].cpu().numpy()
    length = float(step.sum())
    return Rendering(target, owner, length, float((widths * step).sum() / max(length, 1e-12)))
