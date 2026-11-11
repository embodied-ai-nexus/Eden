"""What every art task's metric shares, and the terms that read only that.

:class:`CopyMetric` is the base of an art task's success metric (``CalligraphyWriting``, ``PaintingCopy``): it
samples the tool's tip after every sim step (its point on the sheet, how far below the paper it is, and the tool's
lateral axis, in the workcell's sheet frame: ``workcell.SHEET_CENTER``, ``SHEET_U``, ``SHEET_V``), hands the tool
model's samples between them to its subclass once per control step (:meth:`CopyMetric.advance`), counts the hand's
strokes and path, and latches the plan's points the touching tip reaches while it can lay them (:class:`PlanGuide`,
:meth:`CopyMetric.markable_points`). The tip is read off the tool's pose; marks need no contact (the hairs, the tuft
and the nib do not collide, see ``workcell.py``). A sample with a non-finite pose (an env the solver failed on, reset
this step) is taken as the tool in the air where it last was.

The observation and reward terms here read a :class:`CopyMetric` through the metric manager (resolved in
``build()``; ``metric_name`` is required, since the two task families key their metric differently): the tip's pose
(:class:`TipPose`), the plan's next points (:class:`NextPlanPoints`), the strokes left (:class:`StrokesLeft`), the
grade (:class:`CopyGrade`), and the plan followed (:class:`PlanProgressGain`, :class:`ReachNextPlanPoint`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import genesis as gs
import numpy as np
import torch

from eden.managers.metric_manager import MetricTerm
from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY, ObservationTerm
from eden.managers.reward_manager import REWARD_TERM_REGISTRY, RewardTerm
from eden.managers.terms.utils import resolve_metric_term
from eden.tasks.art import workcell
from eden.utils.geom import transform_by_quat
from eden.utils.string import term_subject


def vector(values, device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, device=device)


class PlanGuide:
    """The plan's points on the paper every ``spacing`` metres, in order, each latched *reached* once the tip touches
    the paper within ``radius`` of it while it can lay that point's mark (a brush holding that point's paint, say).
    ``newly_reached`` is the share the last update reached.

    ``path (m, 3)`` and ``stroke_of (m,)`` are a plan's :func:`~eden.tasks.art.plans.tip_path`: only its strokes'
    samples on the paper are kept.
    """

    def __init__(
        self, path: np.ndarray, stroke_of: np.ndarray, num_envs: int, *, spacing: float, radius: float, device
    ):
        on_paper = (path[:, 2] > 0.0) & (stroke_of >= 0)
        path, stroke_of = path[on_paper], stroke_of[on_paper]
        arc = np.r_[0.0, np.cumsum(np.hypot(*np.diff(path[:, :2], axis=0).T))]
        keep = np.r_[True, np.diff(np.floor(arc / spacing)) > 0] | np.r_[stroke_of[1:] != stroke_of[:-1], True]
        self.points = torch.as_tensor(path[keep], dtype=torch.float32, device=device)  # (n, 3): u, v, depth
        self.stroke = torch.as_tensor(stroke_of[keep], dtype=torch.long, device=device)
        self.radius = radius
        self.reached = torch.zeros(num_envs, len(self.points), dtype=torch.bool, device=device)
        self.newly_reached = torch.zeros(num_envs, device=device)

    @property
    def num_points(self) -> int:
        return self.points.shape[0]

    def update(self, tips: torch.Tensor, touching: torch.Tensor, markable: torch.Tensor | None = None) -> None:
        """Latch the points within reach of the tip samples ``tips (B, S, 2)`` taken while ``touching (B, S)``: of
        them, only those ``markable (B, n)`` marks, if given (a point the tool passes but cannot lay stays unreached,
        so the guide still leads back to it).
        """
        before = self.reached.sum(dim=-1)
        near = torch.cdist(tips, self.points[:, :2]) <= self.radius  # (B, S, n)
        reached = (near & touching[..., None]).any(dim=1)
        self.reached |= reached if markable is None else reached & markable
        self.newly_reached.copy_((self.reached.sum(dim=-1) - before).float() / self.num_points)

    def next_index(self) -> torch.Tensor:
        """Index of each env's first point not reached, in plan order; the last point once all are reached."""
        order = torch.arange(self.num_points, device=self.reached.device).expand_as(self.reached)
        return torch.where(~self.reached, order, self.num_points - 1).amin(dim=-1)

    def reset(self, envs_idx) -> None:
        self.reached[envs_idx] = False
        self.newly_reached[envs_idx] = 0.0


@dataclass(frozen=True)
class WorkBrief:
    """What a :class:`CopyMetric` subclass's work tells the base (:meth:`CopyMetric.build_work` returns it): how errors
    name it (e.g. ``"work='ei'"``), the tool it is made with, its plan's
    :func:`~eden.tasks.art.plans.tip_path` ``(path (m, 3), stroke_of (m,))``, the hand's stroke budget, and the
    sheet's ``(width, height)`` (m), centred on the sheet frame's origin.
    """

    label: str
    tool: workcell.ToolKind
    plan_path: tuple[np.ndarray, np.ndarray]
    max_strokes: int
    sheet: tuple[float, float]


class CopyMetric(MetricTerm):
    """Base of an art task's success metric: the tip sampled, the hand counted, the plan guided (see the module
    docstring). Not a term of its own: a subclass grades the copy.

    A stroke is a touch-down after the tool was lifted at least ``lift_height`` clear of the paper: a tip grazing
    on and off the paper within that is still making the same stroke.

    A subclass builds its work in :meth:`build_work` (called first by :meth:`build`) and returns the
    :class:`WorkBrief` the base reads (kept as :attr:`work_label`, :attr:`tool_kind`, :attr:`plan_path` and
    :attr:`max_strokes`). Each control step :meth:`compute` hands it the tool model's samples in :meth:`advance`,
    which returns the metric's value; :meth:`reset_work` resets its own state.

    The grade is the subclass's main picture measure while every rule passes, held just under that rule's threshold
    otherwise (the field :attr:`_grade_rule` names), so success is *all rules met* exactly when
    ``success_threshold`` equals that threshold: :meth:`configure` sets it so unless it is given, and :meth:`build`
    refuses any other.

    Parameters
    ----------
    tool_name : str
        The tool entity, a :class:`~eden.tasks.art.workcell.HeldToolOptions`.
    sheet_name : str
        The sheet entity the marks lie on: a recording camera shades them as it shades the sheet. Its frame is the
        workcell's (``workcell.SHEET_CENTER``, ``SHEET_U``, ``SHEET_V``), where the scene puts the sheet, the
        palette and the arm's home.
    substeps : int
        Tool-model samples per sim step, spaced evenly between the sim steps' tips. One, 5 ms apart, is RoboPaint's
        0.25 mm sub-step at 5 cm/s, and finer at slower speeds; a faster tool leaves gaps, as a skipping pen does.
    lift_height : float
        How far (m) above the paper the tool must have been for its next touch-down to count as a new stroke.
    guide_spacing, guide_radius : float
        The plan guide's spacing (m) and how near (m) the touching tip must come to reach a point.

    A subclass whose tool does not always mark (a brush that has run dry, or holds another paint than a point's)
    tells the guide which points it can lay now in :meth:`markable_points`.
    """

    tool_name: str = "tool"
    sheet_name: str = "sheet"
    substeps: int = 1
    lift_height: float = 0.001
    guide_spacing: float = 0.002
    guide_radius: float = 0.003

    #: The subclass's field whose threshold the grade is held under when another rule fails (e.g. ``"iou"``); a
    #: subclass that does not name one of its fields is refused when it is defined.
    _grade_rule: ClassVar[str]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        rule = getattr(cls, "_grade_rule", None)
        threshold = getattr(cls, rule, None) if isinstance(rule, str) else None
        if not isinstance(threshold, int | float) or isinstance(threshold, bool):
            raise TypeError(
                f"{cls.__name__} is a CopyMetric, so it must set _grade_rule to the name of its numeric field whose "
                f"threshold the grade is held under when another rule fails (CalligraphyWriting's is 'iou', "
                f"PaintingCopy's 'ssim_l'); got _grade_rule={rule!r}."
            )

    #: The grade, ``(B,)``: the metric's value, which the subclass keeps (made in :meth:`build_work`, graded in
    #: :meth:`advance`) and sets back to the bare sheet's grade in :meth:`reset_work`.
    score: torch.Tensor
    #: From :meth:`build_work`'s :class:`WorkBrief`: the work's name in errors, the tool it is made with, the plan's
    #: tip path and the stroke budget.
    work_label: str
    tool_kind: workcell.ToolKind
    plan_path: tuple[np.ndarray, np.ndarray]
    max_strokes: int

    @classmethod
    def configure(cls, **kwargs):
        """The options, with ``success_threshold`` the grade rule's threshold (:attr:`_grade_rule`) unless given."""
        kwargs.setdefault("success_threshold", kwargs.get(cls._grade_rule, getattr(cls, cls._grade_rule)))
        return super().configure(**kwargs)

    def build_work(self) -> WorkBrief:
        """Build the subclass's work and return what the base reads of it."""
        raise NotImplementedError

    def advance(self, uv: torch.Tensor, depth: torch.Tensor, lateral: torch.Tensor) -> torch.Tensor:
        """Run the tool model along the samples ``uv (B, T, 2)``, ``depth (B, T)``, ``lateral (B, T, 2)``, count the
        hand (:meth:`count_hand`), and return the metric's value ``(B,)``.
        """
        raise NotImplementedError

    def reset_work(self, envs_idx) -> None:
        """Reset the subclass's state of ``envs_idx``."""
        raise NotImplementedError

    def markable_points(self) -> torch.Tensor | None:
        """Which of the plan guide's points the tool can lay now, ``(B, n)`` (bool), read before the step's samples
        are laid; ``None`` (the default): all of them, for a tool that marks whenever it touches (the calligraphy
        tools never run dry). Only the points it can lay are reached, so progress is paid for marks the plan asks for.
        """
        return None

    def build(self) -> None:
        subject = term_subject(self, self.tool_name)
        if gs.tc_float != torch.float32:
            raise ValueError(
                f"{subject}: Eden was initialized at precision='64', but the art tasks' tool models, plans and grading "
                "run in float32 (the published plans are pinned as planned in float32). Initialize Eden with "
                "eden.init(precision='32'), the default; casting the samples down would not do, the plan guide and "
                "the scripted demos read the float64 poses too."
            )
        if self.substeps < 1:
            raise ValueError(f"{subject}: substeps={self.substeps}; it must be at least 1.")
        for name in ("guide_spacing", "guide_radius"):
            if not getattr(self, name) > 0.0:
                raise ValueError(
                    f"{subject}: {name}={getattr(self, name)}; it is a distance on the paper (m) and must be positive "
                    "(the defaults: guide_spacing=0.002, guide_radius=0.003)."
                )
        rule = self._grade_rule
        if self.success_threshold != getattr(self, rule):
            raise ValueError(
                f"{subject}: the grade is the {rule} when every rule passes and held under {rule}="
                f"{getattr(self, rule)} otherwise, so success_threshold={self.success_threshold} would not mean all "
                f"rules met. Configure success_threshold={getattr(self, rule)}, equal to {rule}: configure() sets it "
                f"so when it is not given, but a {rule} changed on the options afterwards needs success_threshold "
                "changed with it."
            )
        work = self.build_work()
        self.work_label, self.tool_kind = work.label, work.tool
        self.plan_path, self.max_strokes = work.plan_path, work.max_strokes
        tool_options = getattr(self._env.config.scene_options, self.tool_name, None)
        if not isinstance(tool_options, workcell.HeldToolOptions) or tool_options.tool != self.tool_kind:
            held = (
                tool_options.tool if isinstance(tool_options, workcell.HeldToolOptions) else type(tool_options).__name__
            )
            raise ValueError(
                f"{subject}: {self.work_label} is made with the {self.tool_kind!r}, but the scene's "
                f"'{self.tool_name}' is {held!r}. Put HeldToolOptions(tool={self.tool_kind!r}) in the scene, or pick a "
                "work for the tool held."
            )
        self.resolve_entity(self.tool_name)
        self.resolve_entity(self.sheet_name)
        self._tip = vector((0.0, 0.0, workcell.TOOLS[self.tool_kind].tip), self.device)
        # Built once: a constant made from a Python value on every step is a blocking copy to the GPU each time.
        self._lateral_axis = vector(workcell.TOOL_LATERAL, self.device)
        self.origin = vector(workcell.SHEET_CENTER, self.device)
        self.u_axis, self.v_axis = vector(workcell.SHEET_U, self.device), vector(workcell.SHEET_V, self.device)
        #: Half the sheet's width and height (m): where the paper ends.
        self.half_sheet = vector(work.sheet, self.device) / 2
        self.inward = torch.linalg.cross(self.v_axis, self.u_axis)
        path, stroke_of = self.plan_path
        self.guide = PlanGuide(
            path, stroke_of, self.num_envs, spacing=self.guide_spacing, radius=self.guide_radius, device=self.device
        )
        n = self.num_envs
        #: The hand: strokes made, the tip's path on the paper (m), and that path weighted by the mark's width.
        self.strokes = torch.zeros(n, dtype=torch.long, device=self.device)
        self.path_length = torch.zeros(n, device=self.device)
        self.width_path = torch.zeros(n, device=self.device)
        self.touching = torch.zeros(n, dtype=torch.bool, device=self.device)
        # Whether the tool has been lifted clear (lift_height) since it last touched: its next touch is a new stroke.
        self.lifted = torch.ones(n, dtype=torch.bool, device=self.device)
        self._hand_uv = torch.zeros(n, 2, device=self.device)
        # The tip's samples, a ring of one control step's sim steps: the ring is read oldest first from the slot
        # after the last written (as latest_ring_slots orders them), so a control step's samples are its own, in order.
        steps = self._env.env_options.decimation
        self._sampled = 0
        self._uv = torch.zeros(n, steps, 2, device=self.device)
        self._depth = torch.zeros(n, steps, device=self.device)
        self._lateral = torch.zeros(n, steps, 2, device=self.device)
        self._valid = torch.zeros(n, steps, dtype=torch.bool, device=self.device)
        # Where the tool model last was (it interpolates from there), and whether it has been anywhere since a reset.
        self._last_uv = torch.zeros(n, 2, device=self.device)
        self._last_depth = torch.zeros(n, device=self.device)
        self._last_lateral = torch.zeros(n, 2, device=self.device)
        self._has_last = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._env.register_substep_callback(self._sample_tip)

    @property
    def strokes_left(self) -> torch.Tensor:
        """The strokes (touch-downs) the hand may still make before it is over budget, ``(B,)``."""
        return self.max_strokes - self.strokes

    def to_sheet(self, points: torch.Tensor) -> torch.Tensor:
        """``(u, v, depth)`` of world ``points (..., 3)``: in the sheet frame, ``depth`` below the paper's face."""
        d = points - self.origin
        return torch.stack([(d * self.u_axis).sum(-1), (d * self.v_axis).sum(-1), (d * self.inward).sum(-1)], dim=-1)

    def tip_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The undeformed tip's world position ``(B, 3)`` and the tool's rotation ``(B, 4)`` (wxyz), read now.

        The tool is one body, so its authored frame is the entity's ``relative=True`` pose; Genesis' own link frame
        of it sits at the centre of mass, several centimetres up the handle.
        """
        tool = self.resolve_entity(self.tool_name)
        pos, quat = tool.get_pos(relative=True), tool.get_quat(relative=True)
        return pos + transform_by_quat(self._tip.expand_as(pos), quat), quat

    def tip_on_sheet(self) -> torch.Tensor:
        """The undeformed tip's ``(u, v, depth)`` in the sheet frame, ``(B, 3)``, read now."""
        return self.to_sheet(self.tip_pose()[0])

    def lateral_on_sheet(self, quat: torch.Tensor) -> torch.Tensor:
        """The tool's lateral axis (:data:`~eden.tasks.art.workcell.TOOL_LATERAL`) projected on the sheet,
        ``(B, 2)`` (not normalized).
        """
        lateral = transform_by_quat(self._lateral_axis.expand(quat.shape[0], 3), quat)
        return torch.stack([(lateral * self.u_axis).sum(-1), (lateral * self.v_axis).sum(-1)], dim=-1)

    def _sample_tip(self) -> None:
        """After each sim step: record where the tip is, how deep, and how the tool's lateral axis lies."""
        k = self._sampled % self._uv.shape[1]
        tip, quat = self.tip_pose()
        sheet = self.to_sheet(tip)
        lateral = self.lateral_on_sheet(quat)
        valid = torch.isfinite(tip).all(-1) & torch.isfinite(quat).all(-1)
        self._uv[:, k] = torch.where(valid[:, None], sheet[:, :2], 0.0)
        self._depth[:, k] = torch.where(valid, sheet[:, 2], -1.0)
        self._lateral[:, k] = torch.where(valid[:, None], lateral, 0.0)
        self._valid[:, k] = valid
        self._sampled += 1

    def compute(self) -> torch.Tensor:
        sampled, self._sampled = self._sampled, 0
        if sampled == 0:
            raise RuntimeError(
                f"{term_subject(self, self.tool_name)}: the metric was computed with no sim step sampled since the "
                "last compute, so there is no tip path to mark. It is computed by env.step(), after the physics."
            )
        # The latest control step's samples, oldest first, and the tool model's sub-steps between them: each sample
        # is reached in `substeps` even steps from the one before (from itself, right after a reset).
        size = self._uv.shape[1]
        n = min(sampled, size)
        order = (torch.arange(n, device=self.device) + (sampled - n)) % size  # latest_ring_slots, on the device
        valid = self._valid[:, order]
        # An invalid sample is the tool in the air where it last was.
        uv = torch.where(valid[..., None], self._uv[:, order], self._last_uv[:, None])
        depth = torch.where(valid, self._depth[:, order], -1.0)
        lateral = torch.where(valid[..., None], self._lateral[:, order], self._last_lateral[:, None])
        start = self._has_last[:, None]
        prev_uv = torch.cat([torch.where(start[..., None], self._last_uv[:, None], uv[:, :1]), uv[:, :-1]], dim=1)
        prev_depth = torch.cat([torch.where(start, self._last_depth[:, None], depth[:, :1]), depth[:, :-1]], dim=1)
        prev_lateral = torch.cat(
            [torch.where(start[..., None], self._last_lateral[:, None], lateral[:, :1]), lateral[:, :-1]], dim=1
        )
        fraction = torch.arange(1, self.substeps + 1, device=self.device) / self.substeps  # (S,), S = substeps
        p_uv = (prev_uv[:, :, None] + (uv - prev_uv)[:, :, None] * fraction[:, None]).flatten(1, 2)  # (B, n*S, 2)
        p_depth = (prev_depth[..., None] + (depth - prev_depth)[..., None] * fraction).flatten(1)
        p_lateral = (prev_lateral[:, :, None] + (lateral - prev_lateral)[:, :, None] * fraction[:, None]).flatten(1, 2)
        self._last_uv.copy_(uv[:, -1])
        self._last_depth.copy_(depth[:, -1])
        self._last_lateral.copy_(lateral[:, -1])
        self._has_last.fill_(True)
        return self.advance(p_uv, p_depth, p_lateral)

    def on_sheet(self, uv: torch.Tensor) -> torch.Tensor:
        """Whether each point ``uv (..., 2)`` of the sheet frame is on the sheet, ``(...)``."""
        return (uv.abs() <= self.half_sheet).all(dim=-1)

    def count_hand(self, uv: torch.Tensor, depth: torch.Tensor, width: torch.Tensor):
        """Count the tool-model samples ``uv (B, T, 2)``, ``depth (B, T)``, the mark's ``width (B, T)``, in order,
        into the hand's strokes, path and width; latch the plan guide's points they reach. Returns ``(B, T)`` each:
        where a sample starts a stroke, how far (m) it moved on the paper from the sample before (0 unless both
        touch), and where the tool *touches* the paper (below its face, on the sheet).

        The tool is *lifted* once it has been ``lift_height`` above the paper or off the sheet (past the paper's
        edge, which stands a paper's thickness above the desk), until it touches the paper again; a touch while
        lifted starts a stroke. Whether it is lifted before sample ``t`` is decided by which came last before it, a
        sample so clear or a touch (neither yet: as it was before this step).
        """
        on_sheet = self.on_sheet(uv)
        touching = (depth > 0.0) & on_sheet
        self.guide.update(uv, touching, self.markable_points())
        high = (depth < -self.lift_height) | ~on_sheet
        steps = torch.arange(depth.shape[1], device=self.device).expand_as(depth)
        last_high = torch.where(high, steps, -1).cummax(dim=1).values
        last_touch = torch.where(touching, steps, -1).cummax(dim=1).values
        undecided = (last_high < 0) & (last_touch < 0)
        lifted_after = torch.where(undecided, self.lifted[:, None], last_high > last_touch)  # (B, T)
        lifted_before = torch.cat([self.lifted[:, None], lifted_after[:, :-1]], dim=1)
        starts = touching & lifted_before
        self.strokes += starts.sum(dim=1)
        self.lifted.copy_(lifted_after[:, -1])
        was_touching = torch.cat([self.touching[:, None], touching[:, :-1]], dim=1)
        previous = torch.cat([self._hand_uv[:, None], uv[:, :-1]], dim=1)
        moved = torch.where(touching & was_touching, (uv - previous).norm(dim=-1), 0.0)
        self.path_length += moved.sum(dim=1)
        self.width_path += (moved * width).sum(dim=1)
        self.touching.copy_(touching[:, -1])
        self._hand_uv.copy_(uv[:, -1])
        return starts, moved, touching

    @property
    def mark_width(self) -> torch.Tensor:
        """The mean width (m) of the tool's mark along its path on the paper, ``(B,)``."""
        return self.width_path / self.path_length.clamp_min(1e-12)

    def reset(self, envs_idx=None) -> None:
        envs_idx = slice(None) if envs_idx is None else envs_idx
        self.guide.reset(envs_idx)
        self.strokes[envs_idx] = 0
        self.path_length[envs_idx] = 0.0
        self.width_path[envs_idx] = 0.0
        self.touching[envs_idx] = False
        self.lifted[envs_idx] = True
        # Nothing of the last episode's tip carries over: its first sample moves from itself, and one with no pose
        # (an env the solver failed on) is in the air at the origin.
        self._has_last[envs_idx] = False
        for buffer in (self._last_uv, self._last_depth, self._last_lateral):
            buffer[envs_idx] = 0.0
        self.reset_work(envs_idx)


def _resolve(term: ObservationTerm | RewardTerm, name: str) -> CopyMetric:
    if not term.metric_name:
        metrics = getattr(term._env, "metric_manager", None)
        keys = metrics.active_terms if metrics is not None else []
        copies = [key for key in keys if isinstance(metrics.get_term(key), CopyMetric)]
        raise ValueError(
            f"{name} reads an art task's metric (a CopyMetric), but metric_name is not set. Configure "
            f"{name}.configure(metric_name=...) with one of the CopyMetric metrics: {copies}."
        )
    return resolve_metric_term(term._env, term=name, metric_name=term.metric_name, expected=CopyMetric)


@OBSERVATION_TERM_REGISTRY.register()
class TipPose(ObservationTerm):
    """The tool's tip in the sheet frame ``(u, v, depth)`` and the tool's axis in it, ``(num_envs, 6)``."""

    metric_name: str = ""

    def build(self) -> None:
        self._copy = _resolve(self, "TipPose")
        self._z_axis = vector((0.0, 0.0, 1.0), self.device)

    def compute(self) -> torch.Tensor:
        copy = self._copy
        tip, quat = copy.tip_pose()
        axis = transform_by_quat(self._z_axis.expand_as(tip), quat)
        axis_sheet = torch.stack(
            [(axis * copy.u_axis).sum(-1), (axis * copy.v_axis).sum(-1), (axis * copy.inward).sum(-1)], dim=-1
        )
        return torch.cat([copy.to_sheet(tip), axis_sheet], dim=-1)


@OBSERVATION_TERM_REGISTRY.register()
class NextPlanPoints(ObservationTerm):
    """The next ``count`` plan points to reach, ``stride`` guide points apart from the first not reached.

    Per point ``(du, dv, ddepth, lift)``: its offset from the tip in the sheet frame (depth included), and 1 where a
    new stroke starts between it and the point before it (at the plan's first point, always), ``(num_envs,
    4 * count)``. Past the last point the list repeats it.
    """

    metric_name: str = ""
    count: int = 4
    stride: int = 3

    def build(self) -> None:
        self._copy = _resolve(self, "NextPlanPoints")
        self._steps = torch.arange(self.count, device=self.device) * self.stride

    def compute(self) -> torch.Tensor:
        copy, guide = self._copy, self._copy.guide
        index = (guide.next_index()[:, None] + self._steps).clamp(max=guide.num_points - 1)  # (B, count)
        points = guide.points[index]  # (B, count, 3)
        stroke = guide.stroke[index]
        # The stroke of the point before each; before the plan's first point, none (-1): a stroke starts there.
        before_first = torch.where(index[:, :1] > 0, guide.stroke[(index[:, :1] - 1).clamp(min=0)], -1)
        previous = torch.cat([before_first, stroke[:, :-1]], dim=1)
        tip = copy.tip_on_sheet()[:, None, :]
        return torch.cat([points - tip, (stroke != previous).float()[..., None]], dim=-1).reshape(self.num_envs, -1)


@OBSERVATION_TERM_REGISTRY.register()
class StrokesLeft(ObservationTerm):
    """The strokes (touch-downs) the hand may still make before it is over budget, ``(num_envs, 1)``."""

    metric_name: str = ""

    def build(self) -> None:
        self._copy = _resolve(self, "StrokesLeft")

    def compute(self) -> torch.Tensor:
        return self._copy.strokes_left.float()[:, None]


@OBSERVATION_TERM_REGISTRY.register()
class CopyGrade(ObservationTerm):
    """The copy's grade, the metric's value (:attr:`CopyMetric.score`), ``(num_envs, 1)``.

    Read off the metric itself rather than the metric manager's buffer (``MetricObservation``), which reads 0 after
    a reset until the next step: a reset env's grade is its bare sheet's, so the observation does not jump on an
    episode's first step.
    """

    metric_name: str = ""

    def build(self) -> None:
        self._copy = _resolve(self, "CopyGrade")

    def compute(self) -> torch.Tensor:
        return self._copy.score[:, None]


@REWARD_TERM_REGISTRY.register()
class PlanProgressGain(RewardTerm):
    """Share of the plan's points the touching tip first reached this step, while it could lay them
    (:meth:`CopyMetric.markable_points`: for a brush, holding their paint).
    """

    metric_name: str = ""

    def build(self) -> None:
        self._copy = _resolve(self, "PlanProgressGain")

    def compute(self) -> torch.Tensor:
        return self._copy.guide.newly_reached


@REWARD_TERM_REGISTRY.register()
class ReachNextPlanPoint(RewardTerm):
    """``exp(-d^2 / std^2)``, ``d`` from the tip to the next plan point not reached, its depth included.

    The point is on (in) the paper, so the full reward needs the tip there at the plan's depth; between strokes the
    task's own terms (stray marks, the stroke budget) are what teach the tool to lift. It pays whether or not the
    tool can lay the point (:meth:`CopyMetric.markable_points`), deliberately: it is the dense pull toward the plan,
    small beside :class:`PlanProgressGain`, which pays only for points laid.
    """

    metric_name: str = ""
    std: float = 0.03

    def build(self) -> None:
        self._copy = _resolve(self, "ReachNextPlanPoint")

    def compute(self) -> torch.Tensor:
        guide = self._copy.guide
        target = guide.points[guide.next_index()]
        return torch.exp(-(self._copy.tip_on_sheet() - target).square().sum(-1) / self.std**2)
