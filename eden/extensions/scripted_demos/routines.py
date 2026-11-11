"""Routine demonstrators: a scripted policy written for one robot, run batched over every env.

A *routine* is a generator function that drives one environment's robot through its
:class:`~eden.extensions.scripted_demos.tools.Tools` handle. It yields motions and gets each
one's :class:`~eden.extensions.scripted_demos.tools.Outcome` back, so checks and retries are plain
Python::

    def pick(tools):
        cube = tools.object_footprint("cube")
        above = cube.center + (0.0, 0.0, 0.10)
        yield tools.move_to(above, yaw=cube.yaw - tools.finger_axis_yaw + math.pi / 2)
        yield tools.move_to(cube.center)
        yield tools.close_gripper()
        yield tools.move_to(above)
        if not tools.holding():
            tools.note("missed the cube")

:class:`RoutineDemonstrator` gives each env its own instance of the routine and runs their motions
together: the routines are resumed only when an env's motion has ended (once per motion, not per
step), and every motion in flight is evaluated as batched torch on the env device and turned into
the task's own actions by :class:`~eden.extensions.scripted_demos.rig.RigDriver`. It has the
demonstrator interface ``eden datagen`` drives (``reset`` / ``update_subgoals`` / ``act``), so a
routine registered for a task in :data:`~eden.extensions.scripted_demos.registry.SCRIPTED_DEMO_REGISTRY`
generates datasets like the keyframe demonstrators do.

Every motion and every :meth:`~eden.extensions.scripted_demos.tools.Tools.note` lands in the env's
:meth:`~RoutineDemonstrator.trace` — the record to read when a routine fails, one line per motion
with how it ended.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from genesis.utils.misc import tensor_to_array

from eden.extensions.scripted_demos.primitives import GRIPPER_PERIOD, effector_rot
from eden.extensions.scripted_demos.rig import RigDriver
from eden.extensions.scripted_demos.tools import (
    BLOCKED,
    CONTACT,
    DONE,
    NO_CONTACT,
    REACHED,
    STATUSES,
    TIMEOUT,
    Footprint,
    Motion,
    MotionKind,
    Outcome,
    Tools,
    footprint_from_points,
)
from eden.extensions.scripted_demos.trajectory import smoothstep5
from eden.utils.geom import quat_slerp, yaw_from_quat

if TYPE_CHECKING:
    from eden.envs.base import EnvBase

#: A routine: given one env's tools, a generator of motions that receives each one's outcome.
Routine = Callable[[Tools], Generator[Motion, Outcome, Any]]

#: A move has reached its target only with its heading within this [rad] of the asked one, too: a
#: grasp closed along the wrong axis misses.
HEADING_TOLERANCE = 0.15
#: A move to a tilted rotation (``move_to(rot=...)``) has reached it only within this angle [rad] of it.
ROTATION_TOLERANCE = 0.15
#: A target rotation this far [rad] from the straight-down pose at its own heading is *tilted*: its
#: move is judged on the full rotation, as the heading of a hand pointing sideways says little.
TILTED = 0.05
#: Shortest a move lasts [s], however short the distance.
MIN_MOVE_S = 0.3
#: Peak heading rate [rad/s] of a move at the nominal speed; scaled with the episode's speed.
PEAK_TURN_RATE = 1.2
#: A guarded move is ``BLOCKED`` once its tool point trails the command by more than this [m] while
#: moving slower than :data:`STALL_SPEED` [m/s] for :data:`STALL_TIME` [s].
BLOCK_LAG = 0.015
STALL_SPEED = 0.01
STALL_TIME = 0.3
#: After travelling its full distance, a guarded move waits this long [s] for the hand to catch up
#: (and touch) before ending ``NO_CONTACT``.
GUARD_SETTLE_S = 0.5
#: A gripper motion or a wait after a move keeps commanding that move's target when the hand is within
#: this [m] of it, so the IK closes the residual a move ends with (it ends once within its tolerance)
#: during the pause and the close. Farther, the move was held back, and the hand holds where it is.
HOLD_TARGET_WITHIN = 0.02
#: Fewer pixels than this of an object and :meth:`Tools.locate` reports it unseen.
MIN_LOCATE_POINTS = 20

_KIND_NONE = -1
_STATUS_CODE = {status: code for code, status in enumerate(STATUSES)}


class RoutineDemonstrator:
    """Runs one routine per environment, batched, through an arm + two-finger-gripper rig.

    Parameters
    ----------
    env : EnvBase
        A built environment holding the rig (:class:`~eden.extensions.scripted_demos.rig.RigDriver`).
    routine : Routine
        Called with each env's :class:`~eden.extensions.scripted_demos.tools.Tools` at the start of
        every episode; the env's task is complete when the generator returns. A routine may list the
        scene entities it looks at as an ``entity_names`` attribute (the put-on routine does), and
        each must then be in the scene.
    rig : str
        Scene entry holding the rig.
    seed : int
        Seed of the per-episode random streams (:attr:`Tools.rng`, the speed draw).
    workspace : tuple[tuple[float, float, float], tuple[float, float, float]]
        Low / high corners every tool target is clipped to; the low ``z`` bounds the fingertips.
    peak_speed : float
        Nominal peak tool speed [m/s] of a move. Size it to what the task's action realizes per
        step (``docs/guides/scripted-demos.md``, *Speed and saturation*).
    speed_range : tuple[float, float]
        Each episode's moves run at ``peak_speed`` times a factor drawn uniformly from this range.
    max_step_translation, max_step_rotation : float
        Largest end-effector move per control step ([m] per axis, [rad]).
    privileged : bool
        Whether routines may read the simulator's object state. ``False`` refuses
        :meth:`Tools.object_footprint` / :meth:`Tools.grasped`, leaving the robot's own state and
        :meth:`Tools.locate`.
    cameras : tuple[str, ...]
        Cameras :meth:`Tools.locate` may look through, the first being its default. Each must render
        every env (``env_options.env_separate_rigid`` or the batch renderer) when there are several.
    contact_force : float
        Smallest force [N] on a finger link that stops a guarded move.
    holding_closure : float
        Fingers commanded closed that stop below this closure (0 open, 1 closed) are holding something.
    exclusive : bool
        Whether this rig's IK and gripper terms must be every action term of the env. A multi-arm
        demonstrator runs one routine demonstrator per arm with ``False``
        (:class:`~eden.extensions.scripted_demos.multi_arm.MultiArmDemonstrator`).
    """

    def __init__(
        self,
        env: EnvBase,
        routine: Routine,
        *,
        rig: str = "robot",
        seed: int = 0,
        workspace: tuple[tuple[float, float, float], tuple[float, float, float]],
        peak_speed: float = 0.12,
        speed_range: tuple[float, float] = (0.8, 1.25),
        max_step_translation: float = 0.05,
        max_step_rotation: float = 0.3,
        privileged: bool = True,
        cameras: tuple[str, ...] = (),
        contact_force: float = 1.0,
        holding_closure: float = 0.9,
        exclusive: bool = True,
    ):
        if not callable(routine):
            raise TypeError(f"routine must be a generator function taking a Tools handle, got {routine!r}.")
        if peak_speed <= 0 or not 0 < speed_range[0] <= speed_range[1]:
            raise ValueError(
                f"peak_speed must be positive and speed_range 0 < low <= high, got {peak_speed} and {speed_range}."
            )
        self.env = env
        self.routine = routine
        self.privileged = privileged
        self.peak_speed = peak_speed
        self.speed_range = speed_range
        self.contact_force = contact_force
        self.holding_closure = holding_closure
        self.seed = seed
        self.num_envs = n = env.num_envs
        self.device = device = env.device
        self.dt = env.dt
        self.driver = RigDriver(
            env,
            rig,
            workspace=workspace,
            max_step_translation=max_step_translation,
            max_step_rotation=max_step_rotation,
            exclusive=exclusive,
        )
        self.cameras = tuple(cameras)
        #: Host copies of batched readings and camera frames, valid for the current step.
        self._cache: dict[Any, Any] = {}
        self._check_cameras()
        self._check_entities()
        self._stall_steps_needed = math.ceil(STALL_TIME / self.dt)

        # Batched motion state, one row per env.
        self._kind = torch.full((n,), _KIND_NONE, dtype=torch.long, device=device)
        self._elapsed = torch.zeros(n, device=device)
        self._duration = torch.zeros(n, device=device)
        self._timeout = torch.zeros(n, device=device)
        self._tolerance = torch.zeros(n, device=device)
        self._start_pos = torch.zeros(n, 3, device=device)
        self._start_rot = torch.zeros(n, 4, device=device)
        self._target_pos = torch.zeros(n, 3, device=device)
        self._target_rot = torch.zeros(n, 4, device=device)
        self._direction = torch.zeros(n, 3, device=device)
        self._distance = torch.zeros(n, device=device)
        self._speed = torch.zeros(n, device=device)
        self._grasp = torch.zeros(n, device=device)
        self._stall = torch.zeros(n, dtype=torch.long, device=device)
        self._prev_tool_pos = torch.zeros(n, 3, device=device)
        self._fresh = torch.zeros(n, dtype=torch.bool, device=device)
        self._tilted = torch.zeros(n, dtype=torch.bool, device=device)
        # The effector rotation last asked for: a move without a heading, a gripper motion or a wait
        # keeps it, so tilt and heading error do not build up from one motion to the next.
        self._held_rot = torch.zeros(n, 4, device=device)
        self._has_held_rot = torch.zeros(n, dtype=torch.bool, device=device)
        # Likewise the last move's target position, held by the gripper motions and waits after it.
        self._held_pos = torch.zeros(n, 3, device=device)
        self._has_held_pos = torch.zeros(n, dtype=torch.bool, device=device)
        # What the fingers touched (any force) when a guarded move started, by scene link: only a
        # touch of something else ends it — a held object's grip force comes and goes.
        self._touched_at_start: torch.Tensor | None = None
        #: ``(N,)`` bool: the routine has returned, this env's task is complete.
        self.done = torch.zeros(n, dtype=torch.bool, device=device)

        # Per-env Python state, touched only when an env's motion ends.
        self._routines: list[Generator[Motion, Outcome, Any] | None] = [None] * n
        self._motions: list[Motion | None] = [None] * n
        self._motion_start_s = np.zeros(n)
        self._traces: list[list[str]] = [[] for _ in range(n)]
        self._episode_steps = np.zeros(n, dtype=np.int64)
        self._episodes = np.zeros(n, dtype=np.int64)
        self._speed_scale = np.ones(n)
        self._grasp_np = np.zeros(n)
        self._blackboards: list[dict] = [{} for _ in range(n)]
        #: Motions ended so far, by ``<tool>:<status>`` — the generation report's tally.
        self.subgoal_counts: dict[str, int] = {}

    # --- episode lifecycle ------------------------------------------------------------------------

    def reset(self, envs_idx: torch.Tensor | None = None, *, blackboards: dict[int, dict] | None = None) -> None:
        """Start a fresh routine for ``envs_idx`` (bool mask or long indices; all when None).

        Call it for every env after the env itself has been reset — including once for all of them
        before the first :meth:`act`, and after every ``env.step`` for the envs it auto-reset
        (``env.reset_buf``: a task's time limit or termination). A routine left running across an
        env reset keeps acting on the scene it saw before. ``eden datagen`` does both.

        ``blackboards`` maps an env id to the dict its routine sees as :attr:`Tools.shared` this
        episode; a multi-arm demonstrator hands both arms of an env the same one. A fresh, empty
        dict otherwise.
        """
        env_ids = self._env_ids(envs_idx)
        if not env_ids:
            return
        low, high = self.speed_range
        for env_id in env_ids:
            if self._routines[env_id] is not None:
                self._routines[env_id].close()  # run its finally blocks now, not at garbage collection
            self._episodes[env_id] += 1
            rng = np.random.default_rng((self.seed, env_id, int(self._episodes[env_id])))
            self._speed_scale[env_id] = rng.uniform(low, high)
            self._routines[env_id] = self.routine(Tools(self, env_id, rng))
            self._motions[env_id] = None
            self._traces[env_id] = []
            self._episode_steps[env_id] = 0
            self._grasp_np[env_id] = 0.0
            self._blackboards[env_id] = {} if blackboards is None else blackboards[env_id]
        idx = torch.tensor(env_ids, device=self.device)
        self._kind[idx] = _KIND_NONE
        self._grasp[idx] = 0.0
        self.done[idx] = False
        self._fresh[idx] = False
        self._has_held_rot[idx] = False
        self._has_held_pos[idx] = False
        self._cache.clear()

    # --- stepping ---------------------------------------------------------------------------------

    def update_subgoals(self) -> torch.Tensor:
        """End the motions that are over, resume their routines, and start their next motions; returns :attr:`done`.

        :meth:`act` calls it first; calling it before is optional and idempotent within a step.
        """
        status = self._motion_status()
        idle = (self._kind == _KIND_NONE) & ~self.done
        resume = (idle | (status >= 0)) & ~self.done
        if bool(resume.any()):
            env_ids = resume.nonzero(as_tuple=False).flatten().tolist()
            self._resume(env_ids, tensor_to_array(status).tolist())
        return self.done

    def act(self) -> dict[str, torch.Tensor]:
        """The action dict for this control step, keyed by the task's IK and gripper term names."""
        self.update_subgoals()
        moving = self._kind >= 0
        t = self._elapsed
        # MOVE: minimum-jerk along the straight line; GUARDED: constant speed along its direction;
        # GRIPPER / WAIT: hold where the motion started (the previous move's target when within
        # HOLD_TARGET_WITHIN of it), at the rotation last asked for.
        s = smoothstep5(t / self._duration.clamp(min=1e-6)).unsqueeze(-1)
        move_pos = self._start_pos + (self._target_pos - self._start_pos) * s
        move_rot = quat_slerp(self._start_rot, self._target_rot, s)
        move_rot = move_rot / move_rot.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        travel = torch.minimum(self._speed * t, self._distance).unsqueeze(-1)
        guarded_pos = self._start_pos + self._direction * travel

        is_move = (self._kind == MotionKind.MOVE).unsqueeze(-1)
        is_guarded = (self._kind == MotionKind.GUARDED).unsqueeze(-1)
        pos = torch.where(is_move, move_pos, torch.where(is_guarded, guarded_pos, self._target_pos))
        rot = torch.where(is_move, move_rot, self._target_rot)

        # Stall tracking for guarded moves: the hand trails the command and has (nearly) stopped.
        tool_pos, _ = self.driver.tool_pose()
        # Against the unclipped command: a hand held at the workspace floor is blocked as much as one on the table.
        lag = (guarded_pos - tool_pos).norm(dim=-1)
        still = (tool_pos - self._prev_tool_pos).norm(dim=-1) < STALL_SPEED * self.dt
        stalling = is_guarded.squeeze(-1) & ~self._fresh & (lag > BLOCK_LAG) & still
        self._stall = torch.where(stalling, self._stall + 1, 0)
        self._prev_tool_pos = tool_pos

        command = self.driver.command(pos, rot, self._grasp, hold=self.done | ~moving)
        self._elapsed += self.dt * moving
        self._fresh[:] = False
        self._episode_steps += 1
        self._cache.clear()
        return command

    def _motion_status(self) -> torch.Tensor:
        """``(N,)`` status code of each env's motion if it has ended now (:data:`STATUSES` index), else -1."""
        kind, t = self._kind, self._elapsed
        active = (kind >= 0) & ~self._fresh & ~self.done
        none = torch.full_like(kind, -1)
        if not bool(active.any()):
            return none
        tool_pos, tool_rot = self.driver.tool_pose()
        error = (tool_pos - self._target_pos).norm(dim=-1)
        heading_error = _wrap(yaw_from_quat(tool_rot) - yaw_from_quat(self._target_rot)).abs()
        aligned = torch.where(
            self._tilted,
            _angle_between(tool_rot, self._target_rot) <= ROTATION_TOLERANCE,
            heading_error <= HEADING_TOLERANCE,
        )

        is_move = active & (kind == MotionKind.MOVE)
        on_target = (error <= self._tolerance) & aligned
        reached = is_move & (t >= self._duration) & on_target
        timed_out = is_move & ~reached & (t >= self._duration + self._timeout)

        is_guarded = active & (kind == MotionKind.GUARDED)
        if bool(is_guarded.any()):
            new_touch = (self.driver.finger_touches(self.contact_force) & ~self._touched_at_start).any(dim=-1)
            touched = is_guarded & new_touch
        else:
            touched = torch.zeros_like(is_guarded)
        blocked = is_guarded & ~touched & (self._stall >= self._stall_steps_needed)
        travel_time = self._distance / self._speed.clamp(min=1e-6)
        untouched = is_guarded & ~touched & ~blocked & (t >= travel_time + GUARD_SETTLE_S)

        timed = active & ((kind == MotionKind.GRIPPER) | (kind == MotionKind.WAIT)) & (t >= self._duration)

        status = none
        for mask, name in (
            (reached, REACHED),
            (timed_out, TIMEOUT),
            (touched, CONTACT),
            (blocked, BLOCKED),
            (untouched, NO_CONTACT),
            (timed, DONE),
        ):
            status = torch.where(mask, _STATUS_CODE[name], status)
        return status

    def _resume(self, env_ids: list[int], status: list[int]) -> None:
        """Hand each env's routine the outcome of its ended motion (if any) and install what it yields next."""
        elapsed = tensor_to_array(self._elapsed)
        error = tensor_to_array((self.driver.tool_pose()[0] - self._target_pos).norm(dim=-1))
        started: list[tuple[int, Motion]] = []
        finished: list[int] = []
        for env_id in env_ids:
            motion = self._motions[env_id]
            outcome = None
            if motion is not None:
                code = status[env_id]
                kind_error = 0.0 if motion.kind in (MotionKind.GRIPPER, MotionKind.WAIT) else float(error[env_id])
                outcome = Outcome(motion.label, STATUSES[code], float(elapsed[env_id]), kind_error)
                self._log_outcome(env_id, motion, outcome)
            next_motion = self._advance_routine(env_id, outcome)
            if next_motion is None:
                finished.append(env_id)
            else:
                started.append((env_id, next_motion))
        if finished:
            idx = torch.tensor(finished, device=self.device)
            self.done[idx] = True
            self._kind[idx] = _KIND_NONE
            for env_id in finished:
                self._motions[env_id] = None
        if started:
            self._install(started)

    def _advance_routine(self, env_id: int, outcome: Outcome | None) -> Motion | None:
        """The routine's next motion, or None once it has returned. Its errors are raised with its trace."""
        routine = self._routines[env_id]
        if routine is None:
            raise RuntimeError(f"Env {env_id} has no routine: call RoutineDemonstrator.reset() before act().")
        try:
            motion = next(routine) if outcome is None else routine.send(outcome)
        except StopIteration:
            return None
        except Exception as error:
            trace = " | ".join(self._traces[env_id][-8:]) or "(no motion yet)"
            raise RuntimeError(
                f"Routine {getattr(self.routine, '__name__', self.routine)!r} raised in env {env_id} at "
                f"t={self.episode_time(env_id):.2f} s: {type(error).__name__}: {error}. Last trace lines: {trace}"
            ) from error
        if not isinstance(motion, Motion):
            raise TypeError(
                f"Routine {getattr(self.routine, '__name__', self.routine)!r} yielded {motion!r} in env {env_id}; a "
                f"routine yields motions built by its Tools handle (tools.move_to(...), tools.close_gripper(), ...)."
            )
        return motion

    def _install(self, started: list[tuple[int, Motion]]) -> None:
        """Start the motions ``started`` (env id, motion) from where each hand is now."""
        device = self.device
        env_ids = [env_id for env_id, _ in started]
        idx = torch.tensor(env_ids, device=device)
        tool_pos, tool_rot = self.driver.tool_pose()
        start_pos, start_rot = tool_pos[idx], tool_rot[idx]
        held = self._has_held_rot[idx].unsqueeze(-1)
        target_pos, target_rot = start_pos.clone(), torch.where(held, self._held_rot[idx], start_rot)
        duration = torch.zeros(len(env_ids), device=device)

        moves = [i for i, (_, m) in enumerate(started) if m.kind == MotionKind.MOVE]
        if moves:
            rows = torch.tensor(moves, device=device)
            requested = torch.tensor([started[i][1].target for i in moves], device=device, dtype=torch.float32)
            target_pos[rows] = self.driver.clip(requested)
            with_yaw = [i for i in moves if started[i][1].yaw is not None]
            if with_yaw:
                yaw_rows = torch.tensor(with_yaw, device=device)
                yaw = torch.tensor([started[i][1].yaw for i in with_yaw], device=device, dtype=torch.float32)
                window = self.driver.yaw_window()[idx[yaw_rows]]
                current = yaw_from_quat(start_rot[yaw_rows])
                target_rot[yaw_rows] = effector_rot(_reachable_heading(yaw, current, window))
            with_rot = [i for i in moves if started[i][1].rot is not None]
            if with_rot:
                rot_rows = torch.tensor(with_rot, device=device)
                target_rot[rot_rows] = torch.tensor(
                    [started[i][1].rot for i in with_rot], device=device, dtype=torch.float32
                )
            speeds = torch.tensor([started[i][1].speed for i in moves], device=device, dtype=torch.float32)
            scale = torch.tensor(self._speed_scale[[env_ids[i] for i in moves]], device=device, dtype=torch.float32)
            distance = (target_pos[rows] - start_pos[rows]).norm(dim=-1)
            turn = _angle_between(start_rot[rows], target_rot[rows])
            # Minimum-jerk peak speed is 1.875 d / T: stretch T so neither the tool speed nor the turn rate exceeds
            # its peak.
            duration[rows] = torch.maximum(
                torch.maximum(1.875 * distance / speeds, 1.875 * turn / (PEAK_TURN_RATE * scale)),
                torch.full_like(distance, MIN_MOVE_S),
            )
            for i in moves:
                clipped = target_pos[i].tolist()
                if any(abs(a - b) > 1e-4 for a, b in zip(clipped, started[i][1].target)):
                    self.add_note(env_ids[i], f"{started[i][1].label} clipped to the workspace: ({_fmt(clipped)})")

        guarded = [i for i, (_, m) in enumerate(started) if m.kind == MotionKind.GUARDED]
        direction = torch.zeros(len(env_ids), 3, device=device)
        if guarded:
            rows = torch.tensor(guarded, device=device)
            direction[rows] = torch.tensor([started[i][1].direction for i in guarded], device=device)
            reach = torch.tensor([started[i][1].distance for i in guarded], device=device).unsqueeze(-1)
            target_pos[rows] = start_pos[rows] + direction[rows] * reach

        timed = [i for i, (_, m) in enumerate(started) if m.kind in (MotionKind.GRIPPER, MotionKind.WAIT)]
        if timed:
            rows = torch.tensor(timed, device=device)
            duration[rows] = torch.tensor([started[i][1].duration_s for i in timed], device=device)
            held_pos = self._held_pos[idx[rows]]
            near = self._has_held_pos[idx[rows]] & ((held_pos - start_pos[rows]).norm(dim=-1) <= HOLD_TARGET_WITHIN)
            target_pos[rows] = torch.where(near.unsqueeze(-1), held_pos, start_pos[rows])

        def column(attribute: str) -> torch.Tensor:
            return torch.tensor([float(getattr(m, attribute)) for _, m in started], device=device)

        self._kind[idx] = torch.tensor([int(m.kind) for _, m in started], device=device)
        self._elapsed[idx] = 0.0
        self._duration[idx] = duration
        self._timeout[idx] = column("timeout_s")
        self._tolerance[idx] = column("tolerance")
        self._start_pos[idx], self._start_rot[idx] = start_pos, start_rot
        self._target_pos[idx], self._target_rot[idx] = target_pos, target_rot
        self._held_rot[idx], self._has_held_rot[idx] = target_rot, True
        self._tilted[idx] = _angle_between(target_rot, effector_rot(yaw_from_quat(target_rot))) > TILTED
        # A move's target is held after it; a guarded move's is not (it lies past where the hand stopped).
        kinds = self._kind[idx]
        self._held_pos[idx] = torch.where((kinds == MotionKind.GUARDED).unsqueeze(-1), start_pos, target_pos)
        self._has_held_pos[idx] = kinds != MotionKind.GUARDED
        if guarded:
            touches = self.driver.finger_touches(0.0)
            if self._touched_at_start is None:
                self._touched_at_start = torch.zeros_like(touches)
            self._touched_at_start[idx] = touches[idx]
        self._direction[idx] = direction
        self._distance[idx] = column("distance")
        self._speed[idx] = column("speed")
        self._stall[idx] = 0
        self._fresh[idx] = True
        grasping = [(env_id, m.grasp) for env_id, m in started if m.grasp is not None]
        if grasping:
            self._grasp[torch.tensor([e for e, _ in grasping], device=device)] = torch.tensor(
                [g for _, g in grasping], device=device
            )
            for env_id, grasp in grasping:
                self._grasp_np[env_id] = grasp
        for env_id, motion in started:
            self._motions[env_id] = motion
            self._motion_start_s[env_id] = self.episode_time(env_id)

    # --- trace and tally --------------------------------------------------------------------------

    def trace(self, env_id: int) -> list[str]:
        """This episode's record for ``env_id``: one line per ended motion and per note, in order."""
        return list(self._traces[env_id])

    def add_note(self, env_id: int, text: str) -> None:
        self._traces[env_id].append(f"{self.episode_time(env_id):6.2f} s  # {text}")

    def _log_outcome(self, env_id: int, motion: Motion, outcome: Outcome) -> None:
        line = f"{self._motion_start_s[env_id]:6.2f} s  {motion.label} -> {outcome.status} in {outcome.elapsed_s:.2f} s"
        if motion.kind in (MotionKind.MOVE, MotionKind.GUARDED):
            line += f" (off target {1000 * outcome.error_m:.0f} mm)"
        self._traces[env_id].append(line)
        key = f"{motion.label.split('(')[0]}:{outcome.status}"
        self.subgoal_counts[key] = self.subgoal_counts.get(key, 0) + 1

    # --- what Tools reads -------------------------------------------------------------------------

    def episode_time(self, env_id: int) -> float:
        return float(self._episode_steps[env_id]) * self.dt

    def episode_speed(self, env_id: int) -> float:
        return self.peak_speed * float(self._speed_scale[env_id])

    def grasp_command(self, env_id: int) -> float:
        return float(self._grasp_np[env_id])

    def blackboard(self, env_id: int) -> dict:
        """The dict ``env_id``'s routine shares this episode (:attr:`Tools.shared`)."""
        return self._blackboards[env_id]

    def read(self, key: str) -> np.ndarray:
        """A batched robot reading as a host array, read once per step: ``tool_pos``, ``tool_rot``, ``tool_yaw``,
        ``gripper_closure``, ``yaw_window``.
        """
        if key not in self._cache:
            if key == "tool_pos":
                value = self.driver.tool_pose()[0]
            elif key == "tool_rot":
                value = self.driver.tool_pose()[1]
            elif key == "tool_yaw":
                value = yaw_from_quat(self.driver.tool_pose()[1])
            elif key == "gripper_closure":
                value = self.driver.gripper_closure()
            elif key == "yaw_window":
                value = self.driver.yaw_window()
            else:
                raise KeyError(
                    f"No robot reading '{key}'; readings: tool_pos, tool_rot, tool_yaw, gripper_closure, yaw_window."
                )
            self._cache[key] = tensor_to_array(value)
        return self._cache[key]

    def object_footprint(self, name: str, env_id: int) -> Footprint:
        """Footprint of entity ``name`` in ``env_id`` from its collision vertices (the vertices read once per step)."""
        key = ("verts", name)
        if key not in self._cache:
            entity = self._entity(name)
            # A kinematic entity's links carry no collision geoms (and no ``n_geoms`` at all).
            verts = [link.get_verts() for link in entity.links if getattr(link, "n_geoms", 0)]
            if not verts:
                raise ValueError(
                    f"Entity '{name}' has no collision geometry to measure a footprint from; read its pose with "
                    f"link_pose('{name}') instead."
                )
            batched = [v if v.ndim == 3 else v.unsqueeze(0).expand(self.num_envs, -1, -1) for v in verts]
            self._cache[key] = tensor_to_array(torch.cat(batched, dim=1))
        return footprint_from_points(self._cache[key][env_id])

    def grasped(self, name: str) -> np.ndarray:
        """``(N,)`` bool: fingers on both sides of the closing line touch entity ``name``."""
        key = ("grasped", name)
        if key not in self._cache:
            self._cache[key] = tensor_to_array(self.driver.pinching(self._entity(name)))
        return self._cache[key]

    def link_pose(self, name: str, link: str | None, env_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Authored-frame pose of entity ``name``'s ``link`` (its root when None) in ``env_id``, read once per step."""
        key = ("link_pose", name, link)
        if key not in self._cache:
            entity = self._entity(name)
            if link is None:
                pos, quat = entity.get_pos(relative=True), entity.get_quat(relative=True)
            else:
                links = [candidate.name for candidate in entity.links]
                if link not in links:
                    raise KeyError(f"Entity '{name}' has no link '{link}'; its links: {links}.")
                idx = torch.tensor([entity.get_link(link).idx_local], device=self.device)
                pos = entity.get_links_pos(ls_idx_local=idx, relative=True)[:, 0]
                quat = entity.get_links_quat(ls_idx_local=idx, relative=True)[:, 0]
            pos, quat = (v if v.ndim == 2 else v.unsqueeze(0).expand(self.num_envs, -1) for v in (pos, quat))
            self._cache[key] = (tensor_to_array(pos), tensor_to_array(quat))
        pos, quat = self._cache[key]
        return pos[env_id], quat[env_id]

    def locate(self, name: str, camera: str | None, env_id: int) -> Footprint | None:
        """Footprint of the pixels of ``camera`` showing entity ``name`` in ``env_id``; None below
        :data:`MIN_LOCATE_POINTS`.
        """
        if not self.cameras:
            raise ValueError(
                "locate() looks through a camera, and this demonstrator was given none: pass "
                "cameras=('<camera name>', ...) to RoutineDemonstrator; the scene's cameras: "
                f"{sorted(self.env.cameras)}."
            )
        camera = camera or self.cameras[0]
        if camera not in self.cameras:
            raise KeyError(
                f"locate() camera '{camera}' is not one of this demonstrator's cameras {list(self.cameras)}."
            )
        segmentation, points = self._camera_frame(camera)
        entity = self._entity(name)
        ids = self.env.entity_segmentation_ids.get(entity.idx)
        if not ids:
            raise KeyError(
                f"locate('{name}'): no segmentation id belongs to entity '{name}' (idx {entity.idx}), so no pixel could "
                f"ever show it; the ids map to entities {sorted(self.env.entity_segmentation_ids)}. Check the scene's "
                f"segmentation level — reading it as 'not seen' would hide the mistake."
            )
        mask = np.isin(segmentation[env_id], ids)
        seen = points[env_id][mask]
        if len(seen) < MIN_LOCATE_POINTS:
            return None
        return footprint_from_points(seen)

    def _camera_frame(self, camera: str) -> tuple[np.ndarray, np.ndarray]:
        """``(N, H, W)`` segmentation and ``(N, H, W, 3)`` world points of ``camera``, rendered once per step."""
        key = ("camera", camera)
        if key not in self._cache:
            cam = self.env.cameras[camera]
            segmentation = tensor_to_array(cam.render_segm())
            points, _ = cam.render_pointcloud(world_frame=True)
            points = tensor_to_array(points)
            if segmentation.ndim == 2:
                segmentation, points = segmentation[None], points[None]
            self._cache[key] = (segmentation, points)
        return self._cache[key]

    def _check_entities(self) -> None:
        """Every entity the routine lists in ``entity_names`` is in the scene: a misspelt one would otherwise raise
        mid-episode.
        """
        missing = [name for name in getattr(self.routine, "entity_names", ()) if name not in self.env.entities]
        if missing:
            raise KeyError(
                f"The routine looks at entities {missing}, which are not in the scene; its entities: "
                f"{sorted(self.env.entities)}."
            )

    def _check_cameras(self) -> None:
        """Every camera exists and, with several envs, renders each of them — or locate() would see env 0
        everywhere.
        """
        if not self.cameras:
            return
        available = sorted(getattr(self.env, "cameras", {}) or {})
        missing = [name for name in self.cameras if name not in available]
        if missing:
            raise KeyError(f"Cameras {missing} are not in the scene; its cameras: {available}.")
        for name in self.cameras:
            segmentation, _ = self._camera_frame(name)
            if self.num_envs > 1 and segmentation.shape[0] != self.num_envs:
                raise ValueError(
                    f"Camera '{name}' renders {segmentation.shape[0]} view(s) for {self.num_envs} envs, so locate() would "
                    f"give every env what env 0 sees. Render each env: --override 'env_options: "
                    f"{{env_separate_rigid: true}}' (or use the batch renderer)."
                )
        self._cache.clear()

    def _entity(self, name: str):
        if name not in self.env.entities:
            raise KeyError(f"No entity '{name}' in this scene; its entities: {sorted(self.env.entities)}.")
        return self.env.entities[name]

    def _env_ids(self, envs_idx: torch.Tensor | None) -> list[int]:
        if envs_idx is None:
            return list(range(self.num_envs))
        if envs_idx.dtype == torch.bool:
            envs_idx = envs_idx.nonzero(as_tuple=False).flatten()
        env_ids = envs_idx.tolist()
        bad = [env_id for env_id in env_ids if not 0 <= env_id < self.num_envs]
        if bad:
            raise IndexError(f"Env ids {bad} are out of range for {self.num_envs} envs.")
        return env_ids


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    """``angle`` wrapped into ``[-pi, pi)``."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _reachable_heading(yaw: torch.Tensor, current: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """The heading equivalent to ``yaw`` under a two-finger gripper's half-turn symmetry that the wrist reaches,
    nearest ``current``.

    Inside ``window`` first (the last joint's reachable range), then nearest the current heading — so
    the hand never turns the long way round, which a slerp toward a far equivalent would make it do.
    """
    shifts = torch.arange(-4, 5, device=yaw.device, dtype=yaw.dtype) * GRIPPER_PERIOD
    candidates = yaw.unsqueeze(-1) + shifts  # (B, 9)
    low, high = window[:, :1], window[:, 1:]
    outside = (low - candidates).clamp(min=0.0) + (candidates - high).clamp(min=0.0)
    score = outside + 1e-3 * (candidates - current.unsqueeze(-1)).abs()
    return candidates.gather(1, score.argmin(dim=1, keepdim=True)).squeeze(1)


def _angle_between(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    """``(B,)`` rotation angle [rad] between wxyz quaternions ``q0`` and ``q1``."""
    dot = (q0 * q1).sum(-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def _fmt(values) -> str:
    return ", ".join(f"{v:.3f}" for v in values)
