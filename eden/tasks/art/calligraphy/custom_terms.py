"""Terms of the calligraphy tasks.

:class:`CalligraphyWriting` is the task's success metric and owns everything the ink does. It is a
:class:`~eden.tasks.art.terms.CopyMetric`, which samples the tool's tip after every sim step and counts the hand;
once per control step it runs the work's tool model (:mod:`~eden.tasks.art.calligraphy.tools`) along the tip's samples,
lays its footprints (:class:`~eden.tasks.art.calligraphy.grading.SheetInk`), and grades the copy with RoboPaint's
writing rules.

The observation and reward terms read it through the metric manager (resolved in ``build()``), so the ink is laid
once per step: :class:`ToolMark`, :class:`TargetCoverageGain` and :class:`StraySheetInkGain` here, and the art suite's
plan-following terms (``eden.tasks.art.terms``). The tool is welded in the hand by the scene
(``eden.tasks.art.workcell``).

The ink exists only here, not in the scene: a camera shows it through a
:class:`~eden.entities.camera_overlays.PlanePaintOverlay`, which paints :meth:`CalligraphyWriting.painted_plane`
onto its frames (the camera the tasks give ``eden inference --rec`` has one).
"""

from __future__ import annotations

from typing import ClassVar, Literal

import torch

from eden.entities.camera_overlays import PaintedPlane
from eden.managers.metric_manager import METRIC_TERM_REGISTRY
from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY, ObservationTerm
from eden.managers.reward_manager import REWARD_TERM_REGISTRY, RewardTerm
from eden.managers.terms.utils import resolve_metric_term
from eden.tasks.art.calligraphy.grading import SheetInk, WritingJudge, WritingRule
from eden.tasks.art.calligraphy.references import render_plan
from eden.tasks.art.calligraphy.tools import TOOL_MODELS, Footprint, stack_footprints
from eden.tasks.art.calligraphy.works import WORKS
from eden.tasks.art.terms import CopyMetric, WorkBrief

WorkName = Literal["dai", "san", "ei", "minimum"]


@METRIC_TERM_REGISTRY.register()
class CalligraphyWriting(CopyMetric):
    """The copy's grade (the success metric), and the ink on the sheet it is graded on.

    Its value is the IoU of the ink with the work's target while every other rule passes, and otherwise the IoU
    held just under ``iou``: so with ``success_threshold`` equal to ``iou`` (``configure()`` sets it so; any other
    is refused), success is RoboPaint's *all rules met*. The rules: the picture's
    (:class:`~eden.tasks.art.calligraphy.grading.WritingRule`) and the hand's — no more strokes than the work has, a
    path on the paper at most ``path_allowance`` times the plan's, and a mean mark width at least
    ``mark_width_floor`` of the plan's (so a work is written, not coloured in with a fine tip). The tip, the strokes
    and the plan guide are the :class:`~eden.tasks.art.terms.CopyMetric`'s (its parameters: ``tool_name``,
    ``sheet_name``, ``substeps``, ``lift_height``, ``guide_spacing``, ``guide_radius``).

    Parameters
    ----------
    work : {"dai", "san", "ei", "minimum"}
        The work to copy (:data:`~eden.tasks.art.calligraphy.works.WORKS`); its tool must be the held tool's.
    iou, boundary_f, min_stroke_coverage, max_stray_frac, boundary_tolerance, stray_tolerance : float
        The picture rule (RoboPaint's writing thresholds by default).
    path_allowance, mark_width_floor : float
        The hand rule's path budget and mark-width floor, relative to the plan's.
    """

    work: WorkName = "ei"
    iou: float = 0.8
    boundary_f: float = 0.75
    min_stroke_coverage: float = 0.8
    max_stray_frac: float = 0.04
    boundary_tolerance: float = 0.001
    stray_tolerance: float = 0.001
    path_allowance: float = 2.5
    mark_width_floor: float = 1.0 / 3.0
    success_threshold: float = 0.8

    _grade_rule: ClassVar[str] = "iou"

    def build_work(self) -> WorkBrief:
        work = WORKS[self.work]
        self.work_spec = work
        reference = render_plan(work, device=self.device)
        self.reference = reference
        self.grid = work.grid
        self.tool = TOOL_MODELS[work.plan.tool](self.num_envs, self.device)
        self.ink = SheetInk(self.grid, self.num_envs, self.tool, device=self.device)
        rule = WritingRule(
            iou=self.iou,
            boundary_f=self.boundary_f,
            min_stroke_coverage=self.min_stroke_coverage,
            max_stray_frac=self.max_stray_frac,
            boundary_tolerance=self.boundary_tolerance,
            stray_tolerance=self.stray_tolerance,
        )
        self.judge = WritingJudge(self.grid, reference.target, reference.owner, rule, device=self.device)
        #: The hand's budget beyond the strokes: path on the paper (m), and the floor of the mean mark width (m).
        self.max_path = self.path_allowance * reference.path_length
        self.min_mark_width = self.mark_width_floor * reference.mark_width
        n = self.num_envs
        #: The picture measures of the last grading (see WritingJudge.judge) and whether every rule passed.
        self.measures: dict[str, torch.Tensor] = {}
        self.passed = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.coverage = torch.zeros(n, device=self.device)  # share of the target inked
        self.new_coverage = torch.zeros(n, device=self.device)
        self.stray = torch.zeros(n, device=self.device)
        self.new_stray = torch.zeros(n, device=self.device)
        self.score = torch.zeros(n, device=self.device)
        # The largest grade under `iou` in the score's own precision: a float64 nextafter rounds back up to `iou`.
        self._below_iou = torch.nextafter(
            torch.tensor(self.iou, device=self.device), torch.tensor(0.0, device=self.device)
        )
        #: The last tool-model sample's footprint, for the observations.
        self.last_footprint = Footprint(
            torch.zeros(n, 2, device=self.device),
            torch.zeros(n, 2, device=self.device),
            torch.zeros(n, device=self.device),
            torch.zeros(n, device=self.device),
            torch.zeros(n, dtype=torch.bool, device=self.device),
        )
        return WorkBrief(
            label=f"work={self.work!r}",
            tool=work.plan.tool,
            plan_path=work.plan.path(),
            max_strokes=len(work.plan.strokes),
            sheet=work.sheet,
        )

    def painted_plane(self, envs_idx: torch.Tensor) -> PaintedPlane:
        """The ink of the envs ``envs_idx`` on the sheet, for a camera's
        :class:`~eden.entities.camera_overlays.PlanePaintOverlay` (which paints it in its own ``color``).
        """
        return PaintedPlane(
            origin=self.origin,
            u_axis=self.u_axis,
            v_axis=self.v_axis,
            grid=self.grid,
            weight=self.ink.inked[envs_idx],
            entity_name=self.sheet_name,
        )

    def advance(self, uv: torch.Tensor, depth: torch.Tensor, lateral: torch.Tensor) -> torch.Tensor:
        footprints = stack_footprints(
            [self.tool.step(uv[:, j], depth[:, j], lateral[:, j]) for j in range(depth.shape[1])]
        )
        # Copied into the buffers built in build(), not replaced: compute runs under inference mode, and reset()
        # (outside it) writes these same tensors.
        for f in Footprint.__dataclass_fields__:
            getattr(self.last_footprint, f).copy_(getattr(footprints, f)[:, -1])
        self.ink.deposit(footprints)
        self.count_hand(uv, depth, footprints.width)
        self._grade()
        return self.score

    def _grade(self) -> None:
        measures = self.judge.judge(self.ink.inked)
        self.measures = measures
        hand = (
            (self.strokes <= self.max_strokes)
            & (self.path_length <= self.max_path)
            & (self.mark_width >= self.min_mark_width)
        )
        self.passed.copy_(measures["passed"] & hand)
        iou = measures["iou"]
        coverage = measures["coverage"]
        self.new_coverage.copy_(coverage - self.coverage)
        self.coverage.copy_(coverage)
        self.new_stray.copy_(measures["stray_frac"] - self.stray)
        self.stray.copy_(measures["stray_frac"])
        self.score.copy_(torch.where(self.passed, iou, torch.minimum(iou, self._below_iou)))

    def reset_work(self, envs_idx) -> None:
        self.ink.reset(envs_idx)
        self.tool.reset(envs_idx)
        self.passed[envs_idx] = False
        for buffer in (self.coverage, self.new_coverage, self.stray, self.new_stray, self.score):
            buffer[envs_idx] = 0.0
        # The auto-reset runs before the observations: the new episode's first ToolMark must not show the last
        # episode's final mark. Back to the state at build: no mark.
        mark = self.last_footprint
        for field in (mark.root, mark.along, mark.length, mark.width):
            field[envs_idx] = 0.0
        mark.live[envs_idx] = False


@OBSERVATION_TERM_REGISTRY.register()
class ToolMark(ObservationTerm):
    """The tool's last footprint: its root relative to the tip, its direction, length and width, and whether it
    inked, ``(num_envs, 7)``. For the brush, where its bent hairs lie on the paper.
    """

    metric_name: str = "writing"

    def build(self) -> None:
        self._writing = resolve_metric_term(
            self._env, term="ToolMark", metric_name=self.metric_name, expected=CalligraphyWriting
        )

    def compute(self) -> torch.Tensor:
        writing = self._writing
        mark = writing.last_footprint
        tip = writing.to_sheet(writing.tip_pose()[0])[:, :2]
        return torch.cat(
            [mark.root - tip, mark.along, mark.length[:, None], mark.width[:, None], mark.live[:, None].float()], dim=-1
        )


@REWARD_TERM_REGISTRY.register()
class TargetCoverageGain(RewardTerm):
    """Share of the target first inked this step (dense; sums to the share inked by the end)."""

    metric_name: str = "writing"

    def build(self) -> None:
        self._writing = resolve_metric_term(
            self._env, term="TargetCoverageGain", metric_name=self.metric_name, expected=CalligraphyWriting
        )

    def compute(self) -> torch.Tensor:
        return self._writing.new_coverage


@REWARD_TERM_REGISTRY.register()
class StraySheetInkGain(RewardTerm):
    """Ink laid off the target this step, as a share of the target's area. Weight it negative."""

    metric_name: str = "writing"

    def build(self) -> None:
        self._writing = resolve_metric_term(
            self._env, term="StraySheetInkGain", metric_name=self.metric_name, expected=CalligraphyWriting
        )

    def compute(self) -> torch.Tensor:
        return self._writing.new_stray
