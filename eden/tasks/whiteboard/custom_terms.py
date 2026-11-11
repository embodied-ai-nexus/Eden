"""Terms of the whiteboard drawing tasks.

:class:`WhiteboardDrawing` is the task's success metric and owns everything the ink does. After every sim step
it samples the nib's pose and its contact force against the board off the solver; once per control step it lays
ink where the nib pressed in those samples (:class:`~eden.tasks.whiteboard.grading.InkLayer`) and grades the
drawing with the original's verifier for that drawing (:class:`~eden.tasks.whiteboard.grading.TextJudge` /
:class:`~eden.tasks.whiteboard.grading.TableJudge`). Ink is laid only where the solver reports the ``nib`` geom
pressing on the board — never on proximity — and only while the marker faces the board (its axis within 78
degrees of the board's normal, the original's gate). A sample with a non-finite pose or press (an env the solver
failed on, reset this step) draws nothing.

The observation and reward terms read it through the metric manager (resolved in ``build()``), so the ink is
integrated once per step. :class:`PlaceMarkerInGrasp` puts the marker between the Franka Hand's pads at every
reset and commands the fingers shut; nothing holds the marker after that but the pads' friction.

The ink exists only here, not in the scene: a camera shows it through a
:class:`~eden.entities.camera_overlays.PlanePaintOverlay`, which paints :meth:`WhiteboardDrawing.painted_plane`
onto its frames (the camera the tasks give ``eden inference --rec`` has one).
"""

from __future__ import annotations

from typing import Literal

import torch

from eden.entities.camera_overlays import PaintedPlane
from eden.managers.event_manager import EVENT_TERM_REGISTRY, EventTerm
from eden.managers.metric_manager import METRIC_TERM_REGISTRY, MetricTerm
from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY, ObservationTerm
from eden.managers.reward_manager import REWARD_TERM_REGISTRY, RewardTerm
from eden.managers.termination_manager import TERMINATION_TERM_REGISTRY, TerminationTerm
from eden.managers.terms.utils import resolve_link_idx, resolve_metric_term
from eden.tasks.whiteboard import common
from eden.tasks.whiteboard.grading import BoardGrid, InkLayer, StrokeGuide, TableJudge, TextJudge
from eden.tasks.whiteboard.strokes import DRAWINGS
from eden.utils.buffers.circular_buffer import latest_ring_slots
from eden.utils.geom import transform_by_quat
from eden.utils.misc import sanitize_envs_idx
from eden.utils.string import term_subject

DrawingName = Literal["hello", "welcome", "table"]


def _vector(values, device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, device=device)


@METRIC_TERM_REGISTRY.register()
class WhiteboardDrawing(MetricTerm):
    """The drawing's grade (the success metric), and the ink on the board it is graded on.

    Its value is the original verifier's: the lettering's F1 for a text drawing (``success_threshold`` 0.85),
    the table's progress for ``"table"`` (1.0 exactly when every line passes and the board is clean;
    ``success_threshold`` 1.0).

    Parameters
    ----------
    drawing : {"hello", "welcome", "table"}
        The reference drawing (:data:`~eden.tasks.whiteboard.strokes.DRAWINGS`).
    marker_name, board_name : str
        The marker and board entities.
    nib_link, nib_geom : str
        The marker's nib link (the slide's child) and its collision geom, by MJCF name.
    nib_tip : tuple[float, float, float]
        The nib's tip in the nib link's frame (m).
    nib_semi_axes : tuple[float, float]
        The ink footprint's semi-axes along the nib's x and y axes (m).
    board_origin, board_u, board_v : tuple[float, float, float]
        The drawing frame on the board's face: the drawing centre (world, m), and the unit ``u`` (right) and ``v``
        (up) axes. ``v x u`` points into the board.
    margin, cell : float
        The pigment grid covers the drawing and ``margin`` round it, in ``cell``-metre cells.
    rate, force_cap : float
        The original's transfer law: ``rate`` (1/s) at a press of ``force_cap`` newtons or more.
    min_facing : float
        The marker draws only while its axis' cosine with the board's inward normal is at least this.
    guide_spacing : float
        Spacing (m) of the reference points :class:`~eden.tasks.whiteboard.grading.StrokeGuide` latches.
    """

    drawing: DrawingName = "hello"
    marker_name: str = "marker"
    board_name: str = "board"
    nib_link: str = "nib"
    nib_geom: str = "nib"
    nib_tip: tuple[float, float, float] = (0.0, 0.0, common.NIB_SIZE[2] / 2)
    nib_semi_axes: tuple[float, float] = common.NIB_SEMI_AXES
    board_origin: tuple[float, float, float] = common.DRAWING_CENTER
    board_u: tuple[float, float, float] = common.BOARD_U
    board_v: tuple[float, float, float] = common.BOARD_V
    margin: float = 0.04
    cell: float = 0.001
    rate: float = 350.0
    force_cap: float = 1.0
    min_facing: float = 0.2
    guide_spacing: float = 0.002
    success_threshold: float = 0.85

    def build(self) -> None:
        if self.drawing == "table" and self.success_threshold != 1.0:
            raise ValueError(
                f"{term_subject(self, self.marker_name)}: drawing='table' grades 1.0 exactly when the table passes and less "
                f"otherwise, so success_threshold={self.success_threshold} would count an unfinished table as "
                "drawn. Configure success_threshold=1.0 (0.85 is the text drawings' F1 threshold)."
            )
        marker = self.resolve_entity(self.marker_name)
        self.resolve_entity(self.board_name)
        geoms = [geom for geom in marker.geoms if geom.metadata.get("name") == self.nib_geom]
        if len(geoms) != 1:
            names = sorted({str(geom.metadata.get("name")) for geom in marker.geoms})
            raise ValueError(
                f"{term_subject(self, self.marker_name)}: the entity has {len(geoms)} geoms named "
                f"nib_geom={self.nib_geom!r} (its geoms: {names}); ink is laid where that one geom presses on "
                f"'{self.board_name}'. Set nib_geom to the nib's MJCF geom name."
            )
        self._nib_geom_idx = geoms[0].idx
        self._nib_link = resolve_link_idx(
            marker, [self.nib_link], self.device, term=term_subject(self, self.marker_name), field="nib_link"
        )
        self._nib_tip = _vector(self.nib_tip, self.device)
        # The drawing frame: its origin (the drawing centre), its axes, and the board's inward normal.
        self.origin = _vector(self.board_origin, self.device)
        self.u_axis, self.v_axis = _vector(self.board_u, self.device), _vector(self.board_v, self.device)
        self.inward = torch.linalg.cross(self.v_axis, self.u_axis)
        strokes = DRAWINGS[self.drawing]
        self.grid = BoardGrid.around(strokes, margin=self.margin, cell=self.cell)
        if self.drawing == "table":
            self.judge = TableJudge(self.grid, strokes, self.device)
        else:
            self.judge = TextJudge(self.grid, strokes, self.num_envs, self.device)
        self.ink = InkLayer(
            self.grid,
            self.num_envs,
            semi_axes=self.nib_semi_axes,
            rate=self.rate,
            force_cap=self.force_cap,
            stray_map=self.judge.stray_map(),
            device=self.device,
        )
        self.guide = StrokeGuide(self.grid, strokes, self.num_envs, spacing=self.guide_spacing, device=self.device)
        #: The board's push on the nib (N), averaged over the last control step's sim steps.
        self.nib_force = torch.zeros(self.num_envs, device=self.device)
        #: Visible stray ink (m^2) the last control step added.
        self.new_stray_m2 = torch.zeros(self.num_envs, device=self.device)
        self.score = torch.zeros(self.num_envs, device=self.device)
        # The nib's samples, a ring of one control step's sim steps: `_sample_nib` writes sample k at slot
        # k % decimation, `compute` draws the latest control step's in order. The ring keeps a term the metric
        # manager does not compute (built on its own, or replaced by a rebuild) harmless: its callback, which the
        # env keeps, only overwrites its own ring.
        steps = self._env.env_options.decimation
        self._sim_dt = self._env.env_options.sim_dt
        self._sampled = 0
        self._uv = torch.zeros(self.num_envs, steps, 2, device=self.device)
        self._press_samples = torch.zeros(self.num_envs, steps, device=self.device)
        self._draw_force = torch.zeros(self.num_envs, steps, device=self.device)
        self._axes = torch.eye(2, device=self.device).repeat(self.num_envs, steps, 1, 1)
        self._env.register_substep_callback(self._sample_nib)

    def painted_plane(self, envs_idx: torch.Tensor) -> PaintedPlane:
        """The visible ink of the envs ``envs_idx`` on the board, for a camera's
        :class:`~eden.entities.camera_overlays.PlanePaintOverlay` (which paints it in its own ``color``). Ink off the
        pigment grid is not kept, so not shown.
        """
        return PaintedPlane(
            origin=self.origin,
            u_axis=self.u_axis,
            v_axis=self.v_axis,
            grid=self.grid,
            weight=self.ink.weight_of(envs_idx),
            entity_name=self.board_name,
        )

    def to_board(self, points: torch.Tensor) -> torch.Tensor:
        """``(u, v, gap)`` of world ``points (..., 3)``: in the drawing frame, ``gap`` the height off the face."""
        d = points - self.origin
        return torch.stack([(d * self.u_axis).sum(-1), (d * self.v_axis).sum(-1), -(d * self.inward).sum(-1)], dim=-1)

    def nib_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The nib tip's world position ``(B, 3)`` and the nib frame's rotation ``(B, 4)`` (wxyz), read now."""
        marker = self.resolve_entity(self.marker_name)
        pos = marker.get_links_pos(ls_idx_local=self._nib_link)[:, 0]
        quat = marker.get_links_quat(ls_idx_local=self._nib_link)[:, 0]
        return pos + transform_by_quat(self._nib_tip.expand_as(pos), quat), quat

    def _press(self) -> torch.Tensor:
        """The board's push on the nib along the board's outward normal (N), from the last physics step."""
        contacts = self.resolve_entity(self.marker_name).get_contacts(with_entity=self.resolve_entity(self.board_name))
        geom_a, geom_b = contacts["geom_a"], contacts["geom_b"]
        force_a, force_b = contacts["force_a"], contacts["force_b"]
        on_nib = torch.where(
            (geom_a == self._nib_geom_idx)[..., None],
            force_a,
            torch.where((geom_b == self._nib_geom_idx)[..., None], force_b, 0.0),
        )
        on_nib = torch.where(contacts["valid_mask"][..., None], on_nib, 0.0)  # (B, contacts, 3)
        return (-(on_nib * self.inward).sum(dim=-1)).sum(dim=-1).clamp_min(0.0)

    def _sample_nib(self) -> None:
        """After each sim step: record where the nib is, how it lies on the board, and how hard it presses."""
        k = self._sampled % self._uv.shape[1]
        tip, quat = self.nib_pose()
        uv = self.to_board(tip)[:, :2]
        press = self._press()
        axis = transform_by_quat(_vector((0.0, 0.0, 1.0), self.device).expand_as(tip), quat)
        facing = (axis * self.inward).sum(-1) >= self.min_facing
        nib_x = transform_by_quat(_vector((1.0, 0.0, 0.0), self.device).expand_as(tip), quat)
        nib_y = transform_by_quat(_vector((0.0, 1.0, 0.0), self.device).expand_as(tip), quat)
        axes = self.ink.footprint(
            torch.stack([(nib_x * self.u_axis).sum(-1), (nib_x * self.v_axis).sum(-1)], -1),
            torch.stack([(nib_y * self.u_axis).sum(-1), (nib_y * self.v_axis).sum(-1)], -1),
        )
        # An env the solver failed on reads garbage this step (and is reset in it): it draws nothing.
        valid = torch.isfinite(tip).all(-1) & torch.isfinite(quat).all(-1) & torch.isfinite(press)
        self._uv[:, k] = torch.where(valid[:, None], uv, 0.0)
        self._press_samples[:, k] = torch.where(valid, press, 0.0)
        self._draw_force[:, k] = torch.where(valid & facing, press, 0.0)
        self._axes[:, k] = torch.where(valid[:, None, None], axes, torch.eye(2, device=self.device))
        self._sampled += 1

    def compute(self) -> torch.Tensor:
        sampled, self._sampled = self._sampled, 0
        if sampled == 0:
            raise RuntimeError(
                f"{term_subject(self, self.marker_name)}: the metric was computed with no sim step sampled since the "
                "last compute, so there is no nib path to ink. It is computed by env.step(), after the physics."
            )
        # The latest control step's samples, oldest first. More than a control step's worth were taken only when a
        # step was cut short (an exception mid-physics) before this one: that step's samples are overwritten.
        slots = latest_ring_slots(sampled, self._uv.shape[1])
        order = slice(None, len(slots)) if slots == list(range(len(slots))) else torch.tensor(slots, device=self.device)
        uv, force, axes, press = (
            buf[:, order] for buf in (self._uv, self._draw_force, self._axes, self._press_samples)
        )
        stray_before = self.ink.stray_m2
        change = self.ink.deposit(uv, force, axes, self._sim_dt)
        self.new_stray_m2.copy_(self.ink.stray_m2 - stray_before)
        self.nib_force.copy_(press.mean(dim=1))
        self.guide.update(change)
        if isinstance(self.judge, TableJudge):
            self.score.copy_(self.judge.judge(self.ink)["progress"])
        else:
            self.judge.update(change)
            self.score.copy_(self.judge.f1(self.ink))
        return self.score

    def reset(self, envs_idx=None) -> None:
        envs_idx = slice(None) if envs_idx is None else envs_idx
        self.ink.reset(envs_idx)
        self.guide.reset(envs_idx)
        if isinstance(self.judge, TextJudge):
            self.judge.reset(envs_idx)
        self.nib_force[envs_idx] = 0.0
        self.new_stray_m2[envs_idx] = 0.0
        self.score[envs_idx] = 0.0


@EVENT_TERM_REGISTRY.register()
class PlaceMarkerInGrasp(EventTerm):
    """Put the marker between the Franka Hand's pads, along the hand's approach axis, and close the fingers on it.

    Run after the arm's reset (a higher ``priority``): it reads the pads where the arm's reset left them. The
    fingers are written open to the barrel's radius and commanded shut (``close_pos``), so they squeeze it at
    their force limit; the marker's frame is the hand's, shifted so the pads hold it at ``grip_z`` on its axis,
    and its nib spring is relaxed. The fingers' command stays shut — this task has no gripper action.

    It reads the pads' and the hand's poses right after writing the fingers' (and, in the arm's reset event, the
    arm's) joint positions: Genesis' DOF setters update the links' poses in the same call. The grasp test
    (``test_the_marker_starts_in_the_hand_and_stays_there_by_friction``) holds it to 1 mm across the arm's reset noise.
    """

    marker_name: str = "marker"
    gripper_name: str = "robot_gripper"
    hand_link: str = "hand"
    pad_links: list[str] = ["left_gripper_pad", "right_gripper_pad"]
    finger_dofs: list[str] = ["finger_joint1", "finger_joint2"]
    nib_dof: str = "nib_slide"
    barrel_radius: float = common.MARKER_RADIUS
    close_pos: float = 0.0
    grip_z: float = common.GRIP_Z

    def build(self) -> None:
        gripper = self.resolve_entity(self.gripper_name)
        marker = self.resolve_entity(self.marker_name)
        self._hand = resolve_link_idx(
            gripper, [self.hand_link], self.device, term=term_subject(self, self.gripper_name), field="hand_link"
        )
        self._pads = resolve_link_idx(
            gripper, self.pad_links, self.device, term=term_subject(self, self.gripper_name), field="pad_links"
        )
        self._fingers = gripper.find_named_dofs(self.finger_dofs)
        # Not find_named_dofs: it selects only an entity's controlled joints (its dofs_name), and the marker, a passive
        # prop, has none. Its nib slide is resolved straight to Genesis' local index, all set_dofs_pos needs.
        _, nib = marker.find_named_dofs_idx_local([self.nib_dof])
        self._nib = torch.tensor(nib, dtype=torch.int32, device=self.device)
        self._grip_point = _vector((0.0, 0.0, self.grip_z), self.device)

    def compute(self, envs_idx) -> None:
        envs_idx, n = sanitize_envs_idx(envs_idx, self.num_envs, prefer_slice=False, return_n_envs=True)
        gripper = self.resolve_entity(self.gripper_name)
        marker = self.resolve_entity(self.marker_name)
        open_pos = torch.full((n, len(self._fingers)), self.barrel_radius, device=self.device)
        gripper.set_dofs_pos(open_pos, dofs_idx_local=self._fingers.dofs_idx_local, envs_idx=envs_idx)
        gripper.control_dofs_pos(
            torch.full_like(open_pos, self.close_pos), dofs_idx_local=self._fingers.dofs_idx_local, envs_idx=envs_idx
        )
        pads = gripper.get_links_pos(ls_idx_local=self._pads, envs_idx=envs_idx)  # (n, 2, 3)
        quat = gripper.get_links_quat(ls_idx_local=self._hand, envs_idx=envs_idx)[:, 0]
        pos = pads.mean(dim=1) - transform_by_quat(self._grip_point.expand(n, 3), quat)
        marker.set_pos(pos, envs_idx=envs_idx, relative=True)
        marker.set_quat(quat, envs_idx=envs_idx, relative=True)
        marker.set_dofs_pos(torch.zeros(n, 1, device=self.device), dofs_idx_local=self._nib, envs_idx=envs_idx)


@TERMINATION_TERM_REGISTRY.register()
class MarkerDropped(TerminationTerm):
    """The marker has left the hand: its grip point is more than ``max_offset`` from the pads' midpoint."""

    marker_name: str = "marker"
    gripper_name: str = "robot_gripper"
    pad_links: list[str] = ["left_gripper_pad", "right_gripper_pad"]
    grip_z: float = common.GRIP_Z
    max_offset: float = 0.02

    def build(self) -> None:
        gripper = self.resolve_entity(self.gripper_name)
        self.resolve_entity(self.marker_name)
        self._pads = resolve_link_idx(
            gripper, self.pad_links, self.device, term=term_subject(self, self.gripper_name), field="pad_links"
        )
        self._grip_point = _vector((0.0, 0.0, self.grip_z), self.device)

    def compute(self) -> torch.Tensor:
        marker = self.resolve_entity(self.marker_name)
        pads = self.resolve_entity(self.gripper_name).get_links_pos(ls_idx_local=self._pads).mean(dim=1)
        pos, quat = marker.get_pos(relative=True), marker.get_quat(relative=True)
        grip = pos + transform_by_quat(self._grip_point.expand_as(pos), quat)
        return (grip - pads).norm(dim=-1) > self.max_offset


@OBSERVATION_TERM_REGISTRY.register()
class NibPose(ObservationTerm):
    """The nib tip in the drawing frame ``(u, v, gap)`` and the marker's axis in it, ``(num_envs, 6)``."""

    metric_name: str = "drawing"

    def build(self) -> None:
        self._drawing = resolve_metric_term(
            self._env, term="NibPose", metric_name=self.metric_name, expected=WhiteboardDrawing
        )

    def compute(self) -> torch.Tensor:
        drawing = self._drawing
        tip, quat = drawing.nib_pose()
        axis = transform_by_quat(_vector((0.0, 0.0, 1.0), self.device).expand_as(tip), quat)
        axis_board = torch.stack(
            [(axis * drawing.u_axis).sum(-1), (axis * drawing.v_axis).sum(-1), (axis * drawing.inward).sum(-1)], dim=-1
        )
        return torch.cat([drawing.to_board(tip), axis_board], dim=-1)


@OBSERVATION_TERM_REGISTRY.register()
class NibForce(ObservationTerm):
    """The board's push on the nib (N), averaged over the last control step, ``(num_envs, 1)``: 0 after a reset."""

    metric_name: str = "drawing"

    def build(self) -> None:
        self._drawing = resolve_metric_term(
            self._env, term="NibForce", metric_name=self.metric_name, expected=WhiteboardDrawing
        )

    def compute(self) -> torch.Tensor:
        return self._drawing.nib_force[:, None]


@OBSERVATION_TERM_REGISTRY.register()
class NextStrokePoints(ObservationTerm):
    """The next ``count`` reference points to ink, ``stride`` guide points apart from the first uncovered one.

    Per point ``(du, dv, lift)``: its offset from the nib tip in the drawing frame, and 1 where a new stroke starts
    between it and the point before it (the pen has to lift on the way), ``(num_envs, 3 count)``. For the first
    point that is the reference point just before it, so the flag says whether the next point to ink begins a new
    stroke. Past the last point the list repeats it.
    """

    metric_name: str = "drawing"
    count: int = 4
    stride: int = 5

    def build(self) -> None:
        self._drawing = resolve_metric_term(
            self._env, term="NextStrokePoints", metric_name=self.metric_name, expected=WhiteboardDrawing
        )
        self._steps = torch.arange(self.count, device=self.device) * self.stride

    def compute(self) -> torch.Tensor:
        drawing, guide = self._drawing, self._drawing.guide
        index = (guide.next_index()[:, None] + self._steps).clamp(max=guide.num_points - 1)  # (B, count)
        points = guide.points[index]  # (B, count, 2)
        stroke = guide.stroke[index]
        previous = torch.cat([guide.stroke[(index[:, :1] - 1).clamp(min=0)], stroke[:, :-1]], dim=1)
        tip = drawing.to_board(drawing.nib_pose()[0])[:, None, :2]
        return torch.cat([points - tip, (stroke != previous).float()[..., None]], dim=-1).reshape(self.num_envs, -1)


@REWARD_TERM_REGISTRY.register()
class StrokeCoverageGain(RewardTerm):
    """Share of the reference's points first inked this step (dense; sums to the share covered by the end)."""

    metric_name: str = "drawing"

    def build(self) -> None:
        self._drawing = resolve_metric_term(
            self._env, term="StrokeCoverageGain", metric_name=self.metric_name, expected=WhiteboardDrawing
        )

    def compute(self) -> torch.Tensor:
        return self._drawing.guide.newly_covered


@REWARD_TERM_REGISTRY.register()
class StrayInkGain(RewardTerm):
    """Visible ink laid off the drawing this step (cm^2), as the judge counts it stray. Weight it negative."""

    metric_name: str = "drawing"

    def build(self) -> None:
        self._drawing = resolve_metric_term(
            self._env, term="StrayInkGain", metric_name=self.metric_name, expected=WhiteboardDrawing
        )

    def compute(self) -> torch.Tensor:
        return self._drawing.new_stray_m2 * 1.0e4


@REWARD_TERM_REGISTRY.register()
class ReachNextStrokePoint(RewardTerm):
    """``exp(-d^2 / std^2)``, ``d`` from the nib tip to the next uncovered reference point on the board's face.

    The point is on the face, so the full reward needs the nib touching the board there; between strokes the
    stray-ink penalty is what teaches the pen to lift.
    """

    metric_name: str = "drawing"
    std: float = 0.05

    def build(self) -> None:
        self._drawing = resolve_metric_term(
            self._env, term="ReachNextStrokePoint", metric_name=self.metric_name, expected=WhiteboardDrawing
        )

    def compute(self) -> torch.Tensor:
        drawing, guide = self._drawing, self._drawing.guide
        target = guide.points[guide.next_index()]
        tip = drawing.to_board(drawing.nib_pose()[0])
        d2 = (tip[:, :2] - target).square().sum(-1) + tip[:, 2].square()
        return torch.exp(-d2 / self.std**2)
