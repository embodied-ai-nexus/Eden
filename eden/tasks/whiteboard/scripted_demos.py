"""Scripted demonstrators for the whiteboard tasks (``eden datagen --task whiteboard_write_text``, …).

One privileged :class:`ScriptedDrawer` per task, registered under the task's name in
:data:`~eden.extensions.scripted_demos.registry.SCRIPTED_DEMO_REGISTRY`; it is also the proof that each drawing
can be made (``tests/tasks/whiteboard/test_scripted_demos.py``). It has no randomness and makes no deliberate
mistakes, so ``seed`` changes nothing it does and ``p_mistake`` must be 0. An ``eden datagen`` episode ends when the
task does: on success, which for the text tasks comes once the lettering's F1 passes 0.85, before the word is
finished.

The drawer drives the task's own action — the differential-IK delta at the nib's tip — so whatever it draws, a policy
could draw through the same interface. It reads the arm's own kinematics (the IK control point's pose) and the
reference strokes; it never touches the marker or the ink.

The plan, per stroke: travel to ``hover`` off the board above its first point, press in to ``press_depth`` past
the face (the nib's spring and the arm's compliance turn that into ~1 N), trace the stroke at ``draw_speed``, and
lift back to ``hover``. While the path presses in, the press depth is trimmed each step toward ``press_force``
on the nib (an integral force loop: the arm's tracking error, 1-3 mm, is larger than the nib's 2 mm of spring,
so a fixed depth skips). The plan is a time-parameterized path of the control point in the drawing frame; each
step commands ``gain`` times the delta from where the control point is to where the path is (at most 90 % of
the action's step: the IK term turns each command into a joint target from where the arm *is*, and the arm's
servos cover only part of it in one step), and the path's clock waits while the arm trails it by more than
``max_lag`` — or, at a corner (a stroke vertex, a touch-down, a lift-off), by more than ``corner_lag``, so the
arm turns where the stroke turns rather than cutting across. The hand's orientation is held at
:func:`~eden.tasks.whiteboard.common.hand_quat`.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from eden.extensions.scripted_demos.registry import SCRIPTED_DEMO_REGISTRY
from eden.managers.terms.actions.task_space_actions import DifferentialIKController
from eden.tasks.whiteboard import common
from eden.tasks.whiteboard.strokes import DRAWINGS
from eden.utils.geom import axis_angle_from_quat, inv_quat, quat_mul, transform_by_quat


def plan_path(
    strokes,
    *,
    hover: float,
    press_depth: float,
    draw_speed: float,
    travel_speed: float,
    dt: float,
    start: tuple[float, float, float] = (0.0, 0.0, 0.03),
) -> tuple[np.ndarray, np.ndarray]:
    """The control point's path ``(T, 3)`` as ``(u, v, gap)`` per control step (``gap`` < 0 pressing in), and
    ``(T,)`` flags marking its corners: every stroke vertex, touch-down and lift-off.

    ``start`` is where the path begins (the arm's home: 30 mm off the drawing centre).
    """
    points = [np.asarray(start, dtype=np.float64)]
    speeds = []

    def go(target, speed):
        points.append(np.asarray(target, dtype=np.float64))
        speeds.append(speed)

    for _, _, path in strokes:
        path = np.asarray(path, dtype=np.float64)
        go((*path[0], hover), travel_speed)
        go((*path[0], -press_depth), travel_speed / 4)
        for p in path[1:]:
            go((*p, -press_depth), draw_speed)
        go((*path[-1], hover), travel_speed / 4)
    go((0.0, 0.0, start[2]), travel_speed)
    samples, corners = [points[0]], [True]
    for a, b, speed in zip(points[:-1], points[1:], speeds):
        n = max(1, math.ceil(np.linalg.norm(b - a) / (speed * dt)))
        samples.extend(a + (b - a) * (k / n) for k in range(1, n + 1))
        corners.extend([False] * (n - 1) + [True])
    return np.stack(samples), np.asarray(corners)


class ScriptedDrawer:
    """Batched privileged drawer for a built whiteboard env (``WhiteboardDrawingConfig`` and its tasks).

    ``act()`` returns the ``(num_envs, 6)`` action for the env's ``arm`` term; call ``reset(envs_idx)`` for the envs
    the env reset (all of them by default). ``done`` is True for an env whose plan has run out. For ``eden datagen``
    it is a demonstrator: :meth:`update_subgoals` returns ``done``, and :attr:`subgoal_counts` counts the strokes
    finished.
    """

    def __init__(
        self,
        env,
        *,
        metric_name: str = "drawing",
        action_name: str = "arm",
        hover: float = 0.012,
        press_depth: float = 0.001,
        draw_speed: float = 0.05,
        travel_speed: float = 0.10,
        max_lag: float = 0.010,
        corner_lag: float = 0.002,
        gain: float = 2.5,
        press_force: float = 0.8,
        force_gain: float = 0.0005,
        depth_trim: tuple[float, float] = (-0.006, 0.002),
    ) -> None:
        self.env = env
        self.drawing = env.metric_manager.get_term(metric_name)
        self.term = env.action_manager.get_term(action_name)
        if not isinstance(self.term, DifferentialIKController):
            # act() inverts this term's action contract: a delta of effective_scale * tanh(action) at ee_local_pos.
            raise TypeError(
                f"ScriptedDrawer drives action '{action_name}' by inverting DifferentialIKController's "
                f"effective_scale * tanh(action) delta, but that term is a {type(self.term).__name__}. Point "
                "action_name at the task's DifferentialIKController term (the whiteboard tasks' 'arm')."
            )
        device = env.device
        path, corners = plan_path(
            DRAWINGS[self.drawing.drawing],
            hover=hover,
            press_depth=press_depth,
            draw_speed=draw_speed,
            travel_speed=travel_speed,
            dt=env.dt,
        )
        self.path = torch.as_tensor(path, dtype=torch.float32, device=device)
        # Where each stroke ends: the first path point off the board after a run pressing into it.
        pressing = path[:, 2] < 0.0
        self._stroke_ends = torch.as_tensor(
            np.flatnonzero(pressing[:-1] & ~pressing[1:]) + 1, dtype=torch.long, device=device
        )
        self._strokes_finished = torch.zeros((), dtype=torch.long, device=device)
        self._allowed_lag = torch.where(torch.as_tensor(corners, device=device), corner_lag, max_lag).float()
        self._scale = self.term.effective_scale
        self._control_point = torch.tensor(self.term.ee_local_pos, dtype=torch.float32, device=device)
        self.gain = gain
        self.press_force, self.force_gain, self.depth_trim = press_force, force_gain, depth_trim
        self.trim = torch.zeros(env.num_envs, device=device)  # added to the path's depth while pressing (m)
        self._target_quat = torch.tensor(common.hand_quat(), dtype=torch.float32, device=device)
        self.clock = torch.zeros(env.num_envs, dtype=torch.long, device=device)

    @property
    def done(self) -> torch.Tensor:
        return self.clock >= self.path.shape[0] - 1

    @property
    def subgoal_counts(self) -> dict[str, int]:
        """The strokes finished so far, over every env and episode (``eden datagen``'s report)."""
        return {"stroke": int(self._strokes_finished)}

    def update_subgoals(self) -> torch.Tensor:
        """``eden datagen``'s per-step check: :attr:`done` (the plan advances in :meth:`act`)."""
        return self.done

    def reset(self, envs_idx=None) -> None:
        envs_idx = slice(None) if envs_idx is None else envs_idx
        self.clock[envs_idx] = 0
        self.trim[envs_idx] = 0.0

    def control_point(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The IK control point's world position and the hand's orientation, ``(B, 3)`` and ``(B, 4)``."""
        arm = self.env.entities[self.term.entity_name]
        link = arm.get_link(self.term.ee_link_name)
        pos, quat = link.get_pos(), link.get_quat()
        hand = self.env.entities["robot_gripper"].get_link("hand").get_quat()
        return pos + transform_by_quat(self._control_point.expand_as(pos), quat), hand

    def act(self) -> torch.Tensor:
        drawing = self.drawing
        pos, hand = self.control_point()
        nib = drawing.to_board(drawing.nib_pose()[0])
        last = self.path.shape[0] - 1
        pressing = self.path[self.clock, 2] < 0.0
        # Too light: press deeper (the trim goes negative, into the board); too hard: ease off.
        step = self.force_gain * (drawing.nib_force - self.press_force)
        self.trim = torch.where(pressing, (self.trim + step).clamp(*self.depth_trim), self.trim)

        def error() -> torch.Tensor:
            # Along the board, where the nib tip is (the marker can turn a little in the pinch); across it, where the
            # control point is (a pressed nib sits on the face whatever the arm does), against the trimmed depth.
            goal = self.path[self.clock]
            depth = goal[:, 2] + torch.where(pressing, self.trim, 0.0)
            board = torch.cat([goal[:, :2] - nib[:, :2], (depth - drawing.to_board(pos)[:, 2])[:, None]], dim=-1)
            return board[:, :1] * drawing.u_axis + board[:, 1:2] * drawing.v_axis - board[:, 2:] * drawing.inward

        lag = error().norm(dim=-1)
        before = self.clock
        self.clock = torch.where(lag <= self._allowed_lag[self.clock], (self.clock + 1).clamp(max=last), self.clock)
        passed = (before[:, None] < self._stroke_ends) & (self._stroke_ends <= self.clock[:, None])
        self._strokes_finished += passed.sum()
        pressing = self.path[self.clock, 2] < 0.0
        delta = self.gain * error()
        limit = 0.9 * self._scale[:3]
        delta = delta * (limit.min() / delta.norm(dim=-1, keepdim=True).clamp_min(1e-9)).clamp(max=1.0)
        rotation = self.gain * axis_angle_from_quat(quat_mul(self._target_quat.expand_as(hand), inv_quat(hand)))
        rot_limit = 0.9 * self._scale[3:].min()
        rotation = rotation * (rot_limit / rotation.norm(dim=-1, keepdim=True).clamp_min(1e-9)).clamp(max=1.0)
        command = torch.cat([delta, rotation], dim=-1) / self._scale.clamp_min(1e-9)
        return torch.atanh(command.clamp(-0.99, 0.99))


def _drawer(env, task: str, *, p_mistake: float) -> ScriptedDrawer:
    if p_mistake != 0.0:
        raise ValueError(
            f"The scripted demonstrator of '{task}' makes no deliberate mistakes, so p_mistake={p_mistake} cannot be "
            "honoured. Generate with --p_mistake 0 (or leave it out)."
        )
    return ScriptedDrawer(env)


@SCRIPTED_DEMO_REGISTRY.register(name="whiteboard_write_text")
def write_text(env, *, seed: int = 0, p_mistake: float = 0.0) -> ScriptedDrawer:
    """HELLO in black, stroke by stroke in the reference's order. ``seed`` changes nothing: the drawer is exact."""
    return _drawer(env, "whiteboard_write_text", p_mistake=p_mistake)


@SCRIPTED_DEMO_REGISTRY.register(name="whiteboard_write_in_colour")
def write_in_colour(env, *, seed: int = 0, p_mistake: float = 0.0) -> ScriptedDrawer:
    """WELCOME in green, stroke by stroke in the reference's order. ``seed`` changes nothing: the drawer is exact."""
    return _drawer(env, "whiteboard_write_in_colour", p_mistake=p_mistake)


@SCRIPTED_DEMO_REGISTRY.register(name="whiteboard_draw_table")
def draw_table(env, *, seed: int = 0, p_mistake: float = 0.0) -> ScriptedDrawer:
    """The blank table in blue: frame, header line, column lines. ``seed`` changes nothing: the drawer is exact."""
    return _drawer(env, "whiteboard_draw_table", p_mistake=p_mistake)


# Each task's own episode length (``config.py``), which the drawer's plan fits in: the generator's default is 90 s.
write_text.episode_length_s = 90.0
write_in_colour.episode_length_s = 120.0
draw_table.episode_length_s = 240.0
