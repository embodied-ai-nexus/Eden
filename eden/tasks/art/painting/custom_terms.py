"""Terms of the painting tasks.

:class:`PaintingCopy` is the task's success metric and owns everything the paint does. It is a
:class:`~eden.tasks.art.terms.CopyMetric`, which samples the brush's tip after every sim step and counts the hand;
once per control step it lays the tip's samples on the paper as the brush's mark in wet paint
(:class:`~eden.tasks.art.painting.paint.Canvas`, which uses up the brush's load for the paper it sweeps), counts the
strokes (a touch-down on the sheet starts one, ending the last) and the dips: a sample in a paint well
(:mod:`~eden.tasks.art.painting.palette`) dips the brush, which comes out of the well full of that paint at the end
of the step. The canvas's clock runs with the episode, so the paint dries. Then it grades the copy with RoboPaint's
acrylic rules. A step is laid as one run, with no host sync (``STEP_SYNC_BUDGETS`` in ``tests/envs/test_base.py``
pins it): it is the same work every step.

The observation and reward terms read it through the metric manager (resolved in ``build()``), so the paint is laid
once per step: :class:`BrushPaint`, :class:`NextPaint` and :class:`PaintingErrorDrop` here, and the art suite's
plan-following terms (``eden.tasks.art.terms``). The brush is welded in the hand by the scene
(``eden.tasks.art.workcell``).

The paint exists only here, not in the scene: a camera shows it through a
:class:`~eden.entities.camera_overlays.PlanePaintOverlay`, which paints :meth:`PaintingCopy.painted_plane` in the
paint's own colours (the camera the tasks give ``eden inference --rec`` has one).
"""

from __future__ import annotations

import math
from typing import ClassVar

import torch

from eden.entities.camera_overlays import PaintedPlane
from eden.managers.metric_manager import METRIC_TERM_REGISTRY
from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY, ObservationTerm
from eden.managers.reward_manager import REWARD_TERM_REGISTRY, RewardTerm
from eden.managers.terms.utils import resolve_metric_term
from eden.tasks.art.painting.grading import PaintingJudge, PaintingRule
from eden.tasks.art.painting.paint import Canvas
from eden.tasks.art.painting.palette import WELL_RADIUS, PaletteOptions
from eden.tasks.art.painting.references import render_painting
from eden.tasks.art.painting.works import WorkName, painting
from eden.tasks.art.terms import CopyMetric, WorkBrief
from eden.utils.misc import assign_over_envs
from eden.utils.string import term_subject


@METRIC_TERM_REGISTRY.register()
class PaintingCopy(CopyMetric):
    """The copy's grade (the success metric), and the paint on the sheet it is graded on.

    Its value is the lightness SSIM of the copy with the work's target while every other rule passes, and otherwise
    the SSIM held just under ``ssim_l``: so with ``success_threshold`` equal to ``ssim_l`` (``configure()`` sets it
    so; any other is refused), success is RoboPaint's *all rules met*. The rules: the picture's
    (:class:`~eden.tasks.art.painting.grading.PaintingRule`) and the hand's — at most ``stroke_allowance`` times the
    plan's strokes, a path on the paper at most ``path_allowance`` times the plan's, and sweeps rather than
    scribbles: no more than ``max_scribble_share`` of the path in strokes at least ``scribble_min`` long whose path
    is over ``max_scribble`` times the distance from where they start to where they end (a zigzag, a loop or a
    there-and-back). The tip, the strokes and the plan guide are the :class:`~eden.tasks.art.terms.CopyMetric`'s (its
    parameters: ``tool_name``, ``sheet_name``, ``substeps``, ``lift_height``, ``guide_spacing``, ``guide_radius``); a
    stroke is on the sheet: a dip is not one.

    A dip fills the brush as it comes out of the well: at the end of the control step the dip began in. A dip counts
    within ``WELL_RADIUS`` (11 mm) of a well's centre, so the tip can be dipping 19 mm from the sheet's edge, on a
    tray level with the paper. Samples it lays after a dip in the same step (a tip sliding from a well onto the paper
    at about 40 cm/s or more) are laid with the brush as it was before the dip.

    Parameters
    ----------
    work : {"sunset", "fuji_study", "fuji"}
        The painting to copy (:data:`~eden.tasks.art.painting.works.WORKS`), made with the paintbrush.
    de_mean, de_p90, ssim_l, edge_corr, paint_iou, blur : float
        The picture rule (RoboPaint's acrylic thresholds by default) and the blur (m) it is read after.
    stroke_allowance, path_allowance : float
        The hand's stroke and path budgets, relative to the plan's: RoboPaint's acrylic budgets are 1.3 times the
        reference's strokes and about 3 times its path.
    max_scribble, max_scribble_share, scribble_min : float
        The scribble rule (RoboPaint's): the path-to-span ratio over which a stroke is a scribble, the share of the
        path scribbles may take, and the length (m) under which a stroke is never one.
    dip_depth : float
        How far (m) into a well the tip must go to dip the brush.
    palette_name : str
        The palette entity: a :class:`~eden.tasks.art.painting.palette.PaletteOptions` of this ``work``, since a dip is
        read off the work's wells.
    """

    work: WorkName = "sunset"
    de_mean: float = 2.0
    de_p90: float = 6.0
    ssim_l: float = 0.92
    edge_corr: float = 0.8
    paint_iou: float = 0.95
    blur: float = 0.001
    stroke_allowance: float = 1.3
    path_allowance: float = 3.0
    max_scribble: float = 2.5
    max_scribble_share: float = 0.1
    scribble_min: float = 0.01
    dip_depth: float = 0.001
    palette_name: str = "palette"
    success_threshold: float = 0.92

    _grade_rule: ClassVar[str] = "ssim_l"

    def build_work(self) -> WorkBrief:
        subject = term_subject(self, self.tool_name)
        if self.dip_depth <= 0.0:
            raise ValueError(f"{subject}: dip_depth={self.dip_depth}; a dip needs the tip in the paint (> 0 m).")
        for name in ("stroke_allowance", "path_allowance", "max_scribble"):
            if not getattr(self, name) > 0.0:
                raise ValueError(
                    f"{subject}: {name}={getattr(self, name)}; it is a budget relative to the plan's and must be "
                    "positive (RoboPaint's acrylic budgets: stroke_allowance=1.3, path_allowance=3.0, "
                    "max_scribble=2.5)."
                )
        if not 0.0 <= self.max_scribble_share <= 1.0:
            raise ValueError(
                f"{subject}: max_scribble_share={self.max_scribble_share}; it is a share of the path, in [0, 1]."
            )
        rule = PaintingRule(
            de_mean=self.de_mean,
            de_p90=self.de_p90,
            ssim_l=self.ssim_l,
            edge_corr=self.edge_corr,
            paint_iou=self.paint_iou,
            blur=self.blur,
        )
        self._check_palette(subject)
        work = painting(self.work)
        self.work_spec = work
        reference = render_painting(work, device=self.device)
        self.reference = reference
        self.grid = work.grid
        self.canvas = Canvas(self.grid, self.num_envs, device=self.device)
        self.judge = PaintingJudge(self.grid, reference.rgb, reference.painted, rule, device=self.device)
        self.max_path = self.path_allowance * reference.path_length
        self.palette = torch.tensor(work.palette, dtype=torch.float32, device=self.device)
        self.wells = torch.as_tensor(work.wells, dtype=torch.float32, device=self.device)
        #: Each plan stroke's paint (a palette index).
        self.stroke_paint = torch.tensor([s.paint for s in work.strokes], dtype=torch.long, device=self.device)
        n = self.num_envs
        #: The paint on the brush (a palette index, -1 before the first dip; its pigment and load are the canvas's),
        #: and the dips made.
        self.paint = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.dips = torch.zeros(n, dtype=torch.long, device=self.device)
        self._dipping = torch.zeros(n, dtype=torch.bool, device=self.device)
        # The stroke being made (its path on the paper, where it started and where it is), and the path in scribbles.
        self._stroke_path = torch.zeros(n, device=self.device)
        self._stroke_start = torch.zeros(n, 2, device=self.device)
        self._stroke_end = torch.zeros(n, 2, device=self.device)
        self._in_stroke = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.scribble_path = torch.zeros(n, device=self.device)
        # The bare sheet's picture measures: each env's until its first grading, and again after a reset.
        self._bare_measures = self.judge.judge(self.canvas.rgb()[:1], self.canvas.painted_cells()[:1])
        #: The picture measures of the last grading (see PaintingJudge.judge) and whether every rule passed. Copied
        #: into these buffers, not replaced: compute runs under inference mode, and reset() (outside it) writes them.
        self.measures = {key: value.expand(n).clone() for key, value in self._bare_measures.items()}
        self.passed = torch.zeros(n, dtype=torch.bool, device=self.device)
        #: The bare sheet's mean ΔE from the target: what :class:`PaintingErrorDrop` normalizes by.
        self.bare_de_mean = float(self._bare_measures["de_mean"][0])
        self.de = torch.full((n,), self.bare_de_mean, device=self.device)
        self.de_drop = torch.zeros(n, device=self.device)
        # The largest grade under `ssim_l` in the score's own precision: a float64 nextafter rounds back up to it.
        self._below = torch.nextafter(
            torch.tensor(self.ssim_l, device=self.device), torch.tensor(0.0, device=self.device)
        )
        # The bare sheet's grade, graded as _grade does (the hand, with nothing done, meets every hand rule): each
        # env's until its first grading and again after a reset, so the grade does not jump on an episode's first step.
        bare_ssim = self._bare_measures["ssim_l"][0]
        self._bare_score = torch.where(
            self._bare_measures["passed"][0], bare_ssim, torch.minimum(bare_ssim, self._below)
        )
        self.score = self._bare_score.expand(n).clone()
        return WorkBrief(
            label=f"work={self.work!r}",
            tool="paintbrush",
            plan_path=work.path(),
            max_strokes=math.ceil(self.stroke_allowance * len(work.strokes)),
            sheet=work.sheet,
        )

    def _check_palette(self, subject: str) -> None:
        """Refuse a scene whose palette is not this work's: its wells, where a dip is read, would not be where the
        scene draws the paints, or would hold other paints.
        """
        scene = self._env.config.scene_options
        options = getattr(scene, self.palette_name, None)
        if isinstance(options, PaletteOptions) and options.work == self.work:
            self.resolve_entity(self.palette_name)
            return
        if options is None:
            found = "missing"
        elif isinstance(options, PaletteOptions):
            found = f"the palette of work={options.work!r}"
        else:
            found = f"a {type(options).__name__}"
        palettes = {
            name: entity.work for name, entity in scene.named_entities().items() if isinstance(entity, PaletteOptions)
        }
        own = [name for name, work in palettes.items() if work == self.work]
        if own:
            fix = f"Set palette_name={own[0]!r}, the scene's palette of this work"
        elif options is None or isinstance(options, PaletteOptions):
            fix = (
                f"Put PaletteOptions(work={self.work!r}) in the scene as '{self.palette_name}' "
                f"(painting.config._scene({self.work!r}) does), or pick the work the scene's palette is for"
            )
        else:
            fix = (
                f"Put PaletteOptions(work={self.work!r}) in the scene under a name of its own and set palette_name to "
                "it, or pick the work the scene's palette is for"
            )
        raise ValueError(
            f"{subject}: work={self.work!r} dips in its own palette, but the scene's '{self.palette_name}' is {found} "
            f"(the scene's palettes, by work: {palettes}). {fix}; don't loosen this check, a copy would dip in wells "
            "the scene does not show."
        )

    def painted_plane(self, envs_idx: torch.Tensor) -> PaintedPlane:
        """The paint of the envs ``envs_idx`` on the sheet, in its own colours, for a camera's
        :class:`~eden.entities.camera_overlays.PlanePaintOverlay` (configured with no ``color``).
        """
        return PaintedPlane(
            origin=self.origin,
            u_axis=self.u_axis,
            v_axis=self.v_axis,
            grid=self.grid,
            weight=self.canvas.painted_cells(envs_idx),
            entity_name=self.sheet_name,
            rgb=self.canvas.rgb(envs_idx),
        )

    def markable_points(self) -> torch.Tensor:
        """The plan points the brush can lay now, ``(B, n)``: those of the paint on it, while it lays that paint at
        full opacity (a load of at least ``PaintLoad.dry_load``, as the plan, dipping with a reserve, always paints).
        A brush never dipped, or running dry, lays none. Read before the step's samples are laid, so the step is
        credited by the brush it began with, as :meth:`~eden.tasks.art.painting.paint.Canvas.lay` lays it.
        """
        point_paint = self.stroke_paint[self.guide.stroke]  # (n,)
        full = self.canvas.load >= self.canvas.paint_load.dry_load
        return (self.paint[:, None] == point_paint) & full[:, None]

    def scribble(self, path: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        """Whether a stroke of ``path`` (m) on the paper from ``start`` to ``end`` ``(..., 2)`` is a scribble."""
        span = (end - start).norm(dim=-1)
        return (path >= self.scribble_min) & (path > self.max_scribble * span)

    def advance(self, uv: torch.Tensor, depth: torch.Tensor, lateral: torch.Tensor) -> torch.Tensor:
        self.canvas.tick(self._env.dt)
        starts, moved, touching = self.count_hand(uv, depth, self.canvas.round_brush.diameter_at(depth))
        # A sample whose mark can reach the paper is laid, as the target's are: one whose tip is past the sheet's edge
        # paints the part of its mark on the paper (the canvas drops the rest, laid on the desk the sheet lies on, a
        # paper's thickness below: it uses the paint all the same). One out of reach (in a well, 19 mm and more off
        # the sheet) is not: it neither paints nor uses paint.
        near = (uv.abs() <= self.half_sheet + self.canvas.round_brush.reach).all(dim=-1)
        self.canvas.lay(uv, torch.where(near, depth, -1.0))
        self._track_strokes(uv, touching, starts, moved)
        self._dip(uv, depth)
        self._grade()
        return self.score

    def _track_strokes(self, uv: torch.Tensor, touching: torch.Tensor, starts: torch.Tensor, moved: torch.Tensor):
        """Follow the strokes through the samples ``uv (B, T, 2)``, ``touching``, ``starts`` and ``moved (B, T)``
        (:meth:`count_hand`'s): each stroke's path on the paper, where it started and where it last touched. A stroke
        ends where the next starts; one that ends is counted into the scribbles if it is one.

        The samples are split into segments, one per stroke: segment 0 is the stroke being made when the step began
        (if one was), segment ``k`` the ``k``-th started in it.
        """
        n, t = starts.shape
        segment = starts.long().cumsum(dim=1)  # (B, T)
        steps = torch.arange(t, device=self.device).expand(n, t)
        none = torch.full((n, t + 1), -1, dtype=torch.long, device=self.device)
        path = torch.zeros(n, t + 1, device=self.device).scatter_add_(1, segment, moved)
        path[:, 0] += self._stroke_path
        first = none.scatter_reduce(1, segment, torch.where(starts, steps, -1), "amax")
        last = none.scatter_reduce(1, segment, torch.where(touching, steps, -1), "amax")
        rows = torch.arange(n, device=self.device)[:, None]
        start = torch.where((first >= 0)[..., None], uv[rows, first.clamp_min(0)], self._stroke_start[:, None])
        end = torch.where((last >= 0)[..., None], uv[rows, last.clamp_min(0)], self._stroke_end[:, None])
        current = segment[:, -1]  # the stroke being made as the step ends
        was_stroke = torch.cat([self._in_stroke[:, None], starts.new_ones(n, t)], dim=1)
        ended = (torch.arange(t + 1, device=self.device) < current[:, None]) & was_stroke
        self.scribble_path += torch.where(ended & self.scribble(path, start, end), path, 0.0).sum(dim=1)
        rows = rows[:, 0]
        self._stroke_path.copy_(path[rows, current])
        self._stroke_start.copy_(start[rows, current])
        self._stroke_end.copy_(end[rows, current])
        self._in_stroke |= current > 0

    def _dip(self, uv: torch.Tensor, depth: torch.Tensor) -> None:
        """Count the dips the samples ``uv (B, T, 2)``, ``depth (B, T)`` begin, and fill each env's brush with the
        paint of the last well it was dipped in this step, as it comes out of it.
        """
        distance = torch.cdist(uv, self.wells[None].expand(uv.shape[0], -1, -1))  # (B, T, K)
        nearest = distance.argmin(dim=-1)
        in_well = (distance.gather(-1, nearest[..., None])[..., 0] < WELL_RADIUS) & (depth > self.dip_depth)
        # A dip begins where the tip was not in a well the sample before.
        dips = in_well & ~torch.cat([self._dipping[:, None], in_well[:, :-1]], dim=1)
        self.dips += dips.sum(dim=1)
        self._dipping.copy_(in_well[:, -1])
        last = torch.where(dips, torch.arange(uv.shape[1], device=self.device), -1).amax(dim=1)
        dipped = last >= 0
        well = nearest.gather(1, last.clamp_min(0)[:, None])[:, 0]
        self.canvas.dip(dipped, self.palette[well])
        self.paint.copy_(torch.where(dipped, well, self.paint))

    def scribble_share(self) -> torch.Tensor:
        """The share of the path on the paper in scribbles, the stroke being made counted as it stands, ``(B,)``."""
        current = self._in_stroke & self.scribble(self._stroke_path, self._stroke_start, self._stroke_end)
        scribbled = self.scribble_path + torch.where(current, self._stroke_path, 0.0)
        return scribbled / self.path_length.clamp_min(1e-12)

    def _grade(self) -> None:
        measures = self.judge.judge(self.canvas.rgb(), self.canvas.painted_cells())
        for key, value in measures.items():
            self.measures[key].copy_(value)
        hand = (
            (self.strokes <= self.max_strokes)
            & (self.path_length <= self.max_path)
            & (self.scribble_share() <= self.max_scribble_share)
        )
        self.passed.copy_(measures["passed"] & hand)
        self.de_drop.copy_((self.de - measures["de_mean"]) / self.bare_de_mean)
        self.de.copy_(measures["de_mean"])
        ssim = measures["ssim_l"]
        self.score.copy_(torch.where(self.passed, ssim, torch.minimum(ssim, self._below)))

    def reset_work(self, envs_idx) -> None:
        self.canvas.reset(envs_idx)
        for key, value in self.measures.items():
            assign_over_envs(value, envs_idx, self._bare_measures[key].expand_as(value))
        self.paint[envs_idx] = -1
        self.dips[envs_idx] = 0
        self._dipping[envs_idx] = False
        self._stroke_path[envs_idx] = 0.0
        self._in_stroke[envs_idx] = False
        self.scribble_path[envs_idx] = 0.0
        self.passed[envs_idx] = False
        assign_over_envs(self.score, envs_idx, self._bare_score.expand_as(self.score))
        self.de[envs_idx] = self.bare_de_mean
        self.de_drop[envs_idx] = 0.0


def _resolve(term: ObservationTerm | RewardTerm, name: str) -> PaintingCopy:
    return resolve_metric_term(term._env, term=name, metric_name=term.metric_name, expected=PaintingCopy)


@OBSERVATION_TERM_REGISTRY.register()
class BrushPaint(ObservationTerm):
    """The paint on the brush: its load and its colour (sRGB, as it has picked up wet paint since its dip; black
    before the first dip), ``(num_envs, 4)``.
    """

    metric_name: str = "painting"

    def build(self) -> None:
        self._copy = _resolve(self, "BrushPaint")

    def compute(self) -> torch.Tensor:
        copy = self._copy
        colour = torch.where(copy.paint[:, None] >= 0, copy.canvas.brush_rgb(), 0.0)
        return torch.cat([copy.canvas.load[:, None], colour], dim=-1)


@OBSERVATION_TERM_REGISTRY.register()
class NextPaint(ObservationTerm):
    """The paint the plan's next point is painted in: its colour (sRGB), its well's offset ``(du, dv)`` from the
    tip, and 1 where the brush holds another paint, ``(num_envs, 6)``.
    """

    metric_name: str = "painting"

    def build(self) -> None:
        self._copy = _resolve(self, "NextPaint")

    def compute(self) -> torch.Tensor:
        copy = self._copy
        paint = copy.stroke_paint[copy.guide.stroke[copy.guide.next_index()]]
        well = copy.wells[paint] - copy.tip_on_sheet()[:, :2]
        other = (copy.paint != paint).float()[:, None]
        return torch.cat([copy.palette[paint], well, other], dim=-1)


@REWARD_TERM_REGISTRY.register()
class PaintingErrorDrop(RewardTerm):
    """How much this step brought the copy's mean ΔE toward the target, as a share of the bare sheet's (dense; sums
    to the share of the bare sheet's error removed by the end). Negative where the step made it worse.
    """

    metric_name: str = "painting"

    def build(self) -> None:
        self._copy = _resolve(self, "PaintingErrorDrop")

    def compute(self) -> torch.Tensor:
        return self._copy.de_drop
