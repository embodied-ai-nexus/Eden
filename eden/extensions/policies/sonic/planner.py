"""SONIC locomotion planner: joystick commands -> full-body reference motion.

Port of the deploy stack's kinematic motion planner. A large generative ONNX
model (``planner_sonic.onnx``) turns a locomotion command (mode, movement and
facing directions, speed, height) plus a 4-frame context window into a 30 Hz
world-frame ``qpos`` trajectory (root pose + 29 joints, MuJoCo order). The
runtime resamples it to the 50 Hz control rate, cross-fades it into the live
planner motion over 8 frames, and lets the g1 encoder track the result.

Pieces, matching the reference:

* :class:`LocomotionMode` + predicates and motion sets — the mode vocabulary.
* :class:`MovementState` — the per-tick command from the operator.
* :class:`PlannerOnnx` — the 11-input / 2-output graph wrapper.
* :class:`SonicPlanner` — a 10 Hz background thread that decides when to
  replan and runs inference, plus the control-thread ``advance()`` that
  blends fresh output into the live motion, advances playback, and runs the
  idle re-adaptation loop.

Threading: the planner thread runs inference and hands the result to the
control thread through a generation buffer; the control thread (via
``advance``) blends it into the live ``planner_motion`` and runs the idle
re-adaptation. A single lock (``_motion_lock``) guards every access to
``planner_motion`` and that hand-off buffer on both threads — the planner
thread's context read and (re)initialization, and the control thread's blend,
in-place idle re-adaptation, and reassignment — mirroring the reference's one
``planner_motion_mutex_``. The scheduler keeps its own lock and is never held
while ``_motion_lock`` is held, so the two never nest. Robot state crosses
through an injected snapshot callable so the planner thread never touches the
simulator.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
from genesis.utils import geom as gu

from eden.extensions.policies.onnx import _OnnxWrapperBase, _run_with_cpu_fallback
from eden.extensions.policies.sonic.joint_maps import IL_TO_MJ, LOWER_BODY_IL_INDICES, MJ_TO_IL
from eden.extensions.policies.sonic.motion import MotionFrames, MotionScheduler

_IL_TO_MJ = np.asarray(IL_TO_MJ)
_MJ_TO_IL = np.asarray(MJ_TO_IL)
_LOWER_BODY_IL = np.asarray(LOWER_BODY_IL_INDICES)


class LocomotionMode(IntEnum):
    """Planner locomotion modes (values match the reference enum exactly)."""

    IDLE = 0
    SLOW_WALK = 1  # 0.1 .. 0.8 m/s
    WALK = 2  # 0.8 .. 2.5 m/s
    RUN = 3  # 2.5 .. 7.5 m/s
    IDLE_SQUAT = 4
    IDLE_KNEEL_TWO_LEGS = 5
    IDLE_KNEEL = 6
    IDLE_LYING_FACE_DOWN = 7
    CRAWLING = 8
    IDLE_BOXING = 9
    WALK_BOXING = 10
    LEFT_PUNCH = 11
    RIGHT_PUNCH = 12
    RANDOM_PUNCH = 13
    ELBOW_CRAWLING = 14
    LEFT_HOOK = 15
    RIGHT_HOOK = 16
    FORWARD_JUMP = 17
    STEALTH_WALK = 18
    INJURED_WALK = 19
    LEDGE_WALKING = 20
    OBJECT_CARRYING = 21
    STEALTH_WALK_2 = 22
    HAPPY_DANCE_WALK = 23
    ZOMBIE_WALK = 24
    GUN_WALK = 25
    SCARE_WALK = 26


_STATIC_MODES = frozenset(
    {
        LocomotionMode.IDLE, LocomotionMode.IDLE_SQUAT, LocomotionMode.IDLE_KNEEL_TWO_LEGS,
        LocomotionMode.IDLE_KNEEL, LocomotionMode.IDLE_LYING_FACE_DOWN, LocomotionMode.IDLE_BOXING,
    }
)  # fmt: skip

_SQUAT_MODES = frozenset(
    {
        LocomotionMode.IDLE_SQUAT, LocomotionMode.IDLE_KNEEL_TWO_LEGS, LocomotionMode.IDLE_KNEEL,
        LocomotionMode.IDLE_LYING_FACE_DOWN, LocomotionMode.CRAWLING, LocomotionMode.ELBOW_CRAWLING,
    }
)  # fmt: skip


# Operator-cyclable locomotion modes, by set: 0 standing, 1 squat, 2 boxing,
# 3 styled. Built once (the reference uses a fixed table), not per call.
_MOTION_SETS: dict[int, tuple[LocomotionMode, ...]] = {
    0: (
        LocomotionMode.SLOW_WALK, LocomotionMode.WALK, LocomotionMode.RUN,
        LocomotionMode.FORWARD_JUMP, LocomotionMode.STEALTH_WALK, LocomotionMode.INJURED_WALK,
    ),
    1: (
        LocomotionMode.IDLE_SQUAT, LocomotionMode.IDLE_KNEEL_TWO_LEGS, LocomotionMode.IDLE_KNEEL,
        LocomotionMode.CRAWLING, LocomotionMode.ELBOW_CRAWLING,
    ),
    2: (
        LocomotionMode.IDLE_BOXING, LocomotionMode.WALK_BOXING, LocomotionMode.LEFT_PUNCH,
        LocomotionMode.RIGHT_PUNCH, LocomotionMode.RANDOM_PUNCH, LocomotionMode.LEFT_HOOK,
        LocomotionMode.RIGHT_HOOK,
    ),
    3: (
        LocomotionMode.LEDGE_WALKING, LocomotionMode.OBJECT_CARRYING, LocomotionMode.STEALTH_WALK_2,
        LocomotionMode.HAPPY_DANCE_WALK, LocomotionMode.ZOMBIE_WALK, LocomotionMode.GUN_WALK,
        LocomotionMode.SCARE_WALK,
    ),
}  # fmt: skip


def is_static_mode(mode: int) -> bool:
    """A mode that stands in place (stick magnitude does not drive replanning).

    Unknown mode ids are simply "not static", like the reference's enum cast.
    Membership uses the precomputed frozenset directly (the reference's
    ``constexpr is_static_motion_mode`` compile-time predicate); IntEnum members
    compare/hash equal to their int value, so an int or numpy id matches.
    """
    return mode in _STATIC_MODES


def is_squat_mode(mode: int) -> bool:
    """A mode whose height command is clamped to stay above the crouch floor."""
    return mode in _SQUAT_MODES


def get_motion_set(index: int) -> list[LocomotionMode]:
    """Locomotion modes cyclable from the operator, by set: 0 standing, 1 squat, 2 boxing, 3 styled."""
    return list(_MOTION_SETS.get(index, _MOTION_SETS[0]))


def valid_mode_count(version: int) -> int:
    """Number of mode ids the planner graph accepts for a model version."""
    counts = {0: 4, 1: 20, 2: 27}
    if version not in counts:
        raise ValueError(
            f"unsupported planner model version {version!r}; supported: {sorted(counts)} "
            "(PlannerConfig.version must match the planner ONNX — the V2 release is 2)"
        )
    return counts[version]


@dataclass
class MovementState:
    """Per-tick locomotion command from the operator (planner-topic payload)."""

    locomotion_mode: int = 0
    movement_direction: np.ndarray = field(default_factory=lambda: np.zeros(3))
    facing_direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    movement_speed: float = 0.0  # -1 = mode default
    height: float = 0.0  # -1 = mode default


@dataclass
class PlannerConfig:
    """Planner runtime configuration (reference defaults)."""

    version: int = 2  # the V2 release model
    look_ahead_steps: int = 2  # 50 Hz frames of context look-ahead (0.04 s)
    default_height: float = 0.788740
    initial_random_seed: int = 1234
    planner_dt: float = 0.1  # 10 Hz planner thread
    # Replan timers per mode family, seconds.
    replan_interval_running: float = 0.1
    replan_interval_crawling: float = 0.2
    replan_interval_boxing: float = 1.0
    replan_interval: float = 1.0


@dataclass
class PlannerState:
    """Lifecycle flags (set by the ZMQ manager, read by the planner thread)."""

    enabled: bool = False
    initialized: bool = False


QPOS_WIDTH = 36  # root xyz (3) + root wxyz quat (4) + 29 joints, MuJoCo order
_BLEND_NUM_FRAMES = 8  # cross-fade width, ~ the planner's context length
# Idle re-adaptation thresholds (rad, mean over lower-body joints).
_ADAPT_TRIGGER = 0.10
_ADAPT_STOP = 0.05
_RECOVER_TRIGGER = 0.045


class PlannerOnnx(_OnnxWrapperBase):
    """The ``planner_sonic.onnx`` graph: locomotion command + context -> 30 Hz qpos.

    Inputs (11) and outputs (2) are fed/read by name; ``generate`` returns the
    valid prediction as numpy, so the wrapper stays torch-free for callers.
    """

    ALLOWED_PRED_NUM_TOKENS = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=np.int64)  # 9-11 tokens

    _INPUTS = (
        "context_mujoco_qpos", "mode", "target_vel", "height", "movement_direction", "facing_direction",
        "random_seed", "has_specific_target", "specific_target_positions", "specific_target_headings",
        "allowed_pred_num_tokens",
    )  # fmt: skip

    def __init__(self, onnx_path: str, device: str | None = None) -> None:
        super().__init__(onnx_path, device=device, label="SONIC Planner")
        missing = [n for n in self._INPUTS if n not in self._input_names]
        if missing:
            raise ValueError(f"planner graph at {onnx_path} lacks inputs {missing}; got {sorted(self._input_names)}")
        for name in ("mujoco_qpos", "num_pred_frames"):
            if name not in self._output_names:
                raise ValueError(f"planner graph at {onnx_path} lacks output '{name}'; got {self._output_names}")

    def generate(
        self,
        context_qpos: np.ndarray,
        mode: int,
        target_vel: float,
        target_height: float,
        movement_direction: np.ndarray,
        facing_direction: np.ndarray,
        random_seed: int,
    ) -> np.ndarray:
        """Run one planning inference; returns the ``(num_pred_frames, 36)`` 30 Hz qpos."""
        feed = {
            "context_mujoco_qpos": np.asarray(context_qpos, dtype=np.float32).reshape(1, 4, QPOS_WIDTH),
            "mode": np.array([mode], dtype=np.int64),
            "target_vel": np.array([target_vel], dtype=np.float32),
            "height": np.array([target_height], dtype=np.float32),
            "movement_direction": np.asarray(movement_direction, dtype=np.float32).reshape(1, 3),
            "facing_direction": np.asarray(facing_direction, dtype=np.float32).reshape(1, 3),
            "random_seed": np.array([random_seed], dtype=np.int64),
            "has_specific_target": np.zeros((1, 1), dtype=np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
            "specific_target_headings": np.zeros((1, 4), dtype=np.float32),
            "allowed_pred_num_tokens": self.ALLOWED_PRED_NUM_TOKENS,
        }
        qpos, num_frames = _run_with_cpu_fallback(
            self, feed, output_names=["mujoco_qpos", "num_pred_frames"], label=self._label
        )
        n = int(np.asarray(num_frames).reshape(-1)[0])
        return np.asarray(qpos, dtype=np.float64).reshape(-1, QPOS_WIDTH)[:n]


def resample_30hz_to_50hz(qpos_30hz: np.ndarray) -> MotionFrames:
    """Resample a 30 Hz planner qpos trajectory to a 50 Hz motion.

    Positions and joints interpolate linearly, the root quaternion slerps, and
    joint velocities are forward finite differences (last frame repeats the
    previous one). Joint columns are reordered MuJoCo -> IsaacLab. Rejects a
    trajectory that is longer than the graph's 64-frame output buffer or
    carries NaNs, like the reference.
    """
    qpos_30hz = np.asarray(qpos_30hz, dtype=np.float64)
    timesteps_30hz = qpos_30hz.shape[0]
    if timesteps_30hz > 64:
        raise ValueError(f"planner generated {timesteps_30hz} frames (> 64)")
    if np.isnan(qpos_30hz).any():
        raise ValueError("planner qpos output contains NaNs")

    timesteps_50hz = int(np.floor(timesteps_30hz / 30.0 * 50.0))
    joint_pos = np.zeros((timesteps_50hz, 29))
    body_pos = np.zeros((timesteps_50hz, 1, 3))
    body_quat = np.zeros((timesteps_50hz, 1, 4))
    joints_il = qpos_30hz[:, 7:][:, _MJ_TO_IL]  # MuJoCo columns -> IsaacLab order

    for f_50hz in range(timesteps_50hz):
        f_30hz = f_50hz / 50.0 * 30.0
        f0 = int(np.floor(f_30hz))
        f1 = min(f0 + 1, timesteps_30hz - 1)
        w1 = f_30hz - f0
        body_pos[f_50hz, 0] = (1.0 - w1) * qpos_30hz[f0, :3] + w1 * qpos_30hz[f1, :3]
        body_quat[f_50hz, 0] = gu.slerp(qpos_30hz[f0, 3:7], qpos_30hz[f1, 3:7], np.asarray(w1))
        joint_pos[f_50hz] = (1.0 - w1) * joints_il[f0] + w1 * joints_il[f1]

    joint_vel = np.zeros_like(joint_pos)
    if timesteps_50hz >= 2:
        joint_vel[:-1] = (joint_pos[1:] - joint_pos[:-1]) * 50.0
        joint_vel[-1] = joint_vel[-2]
    return MotionFrames(
        name="planner_generated", joint_pos=joint_pos, joint_vel=joint_vel, body_quat=body_quat, body_pos=body_pos
    )


def blend_planner_motion(
    live: MotionFrames, generated: MotionFrames, current_frame: int, gen_frame: int
) -> MotionFrames:
    """Cross-fade a fresh planner generation into the live motion, rebased to the cursor.

    The result's frame 0 is the current playback frame; the generation starts
    at ``gen_frame - current_frame`` on that timeline and fades in linearly
    over 8 frames. Returns a new motion (the reference blends in place, but
    every read there sees pre-blend data, so building fresh is equivalent).
    """
    new_length = gen_frame - current_frame + generated.timesteps
    if new_length <= 0:
        raise ValueError("planner generation arrived too late to blend")
    blend_start = max(0, gen_frame - current_frame)

    joint_pos = np.zeros((new_length, live.joint_pos.shape[1]))
    joint_vel = np.zeros_like(joint_pos)
    body_pos = np.zeros((new_length, 1, 3))
    body_quat = np.zeros((new_length, 1, 4))
    for f in range(new_length):
        f_old = int(np.clip(f + current_frame, 0, live.timesteps - 1))
        f_new = int(np.clip(f + current_frame - gen_frame, 0, generated.timesteps - 1))
        w_new = float(np.clip((f - blend_start) / _BLEND_NUM_FRAMES, 0.0, 1.0))
        w_old = 1.0 - w_new
        joint_pos[f] = w_old * live.joint_pos[f_old] + w_new * generated.joint_pos[f_new]
        joint_vel[f] = w_old * live.joint_vel[f_old] + w_new * generated.joint_vel[f_new]
        body_pos[f, 0] = w_old * live.body_pos[f_old, 0] + w_new * generated.body_pos[f_new, 0]
        body_quat[f, 0] = gu.slerp(live.body_quat[f_old, 0], generated.body_quat[f_new, 0], np.asarray(w_new))

    blended = MotionFrames(
        name="planner_motion", joint_pos=joint_pos, joint_vel=joint_vel, body_quat=body_quat, body_pos=body_pos
    )
    blended.encode_mode = live.encode_mode  # runtime mode state carries over
    return blended


def sample_context_from_motion(motion: MotionFrames, gen_frame: int) -> np.ndarray:
    """Sample the 4-frame 30 Hz context window from a 50 Hz motion at ``gen_frame``.

    Returns ``(4, 36)`` world-frame qpos rows (joints back in MuJoCo order).
    """
    if motion.timesteps == 0:
        raise ValueError("cannot sample planner context: motion has no frames")
    context = np.zeros((4, QPOS_WIDTH))
    gen_time = gen_frame / 50.0
    for n in range(4):
        f_50hz = (gen_time + n / 30.0) * 50.0
        f0 = min(int(np.floor(f_50hz)), motion.timesteps - 1)
        f1 = min(f0 + 1, motion.timesteps - 1)
        w1 = f_50hz - f0
        context[n, :3] = (1.0 - w1) * motion.body_pos[f0, 0] + w1 * motion.body_pos[f1, 0]
        context[n, 3:7] = gu.slerp(motion.body_quat[f0, 0], motion.body_quat[f1, 0], np.asarray(w1))
        joints_il = (1.0 - w1) * motion.joint_pos[f0] + w1 * motion.joint_pos[f1]
        context[n, 7:] = joints_il[_IL_TO_MJ]  # IsaacLab -> MuJoCo columns
    return context


def initial_context(joint_positions_mj: np.ndarray, default_height: float) -> np.ndarray:
    """The 4-frame context used before any motion exists: standing at the origin.

    All frames are identical: identity orientation at the default height with
    the robot's measured joints (the planner works in a yaw-normalized frame,
    so the robot's actual heading is deliberately not encoded).
    """
    context = np.zeros((4, QPOS_WIDTH))
    context[:, 2] = default_height
    context[:, 3] = 1.0  # identity quaternion
    context[:, 7:] = np.asarray(joint_positions_mj, dtype=np.float64)
    return context


def should_replan(previous: MovementState, current: MovementState, timer_expired: bool) -> bool:
    """Replan decision: mode/facing/height changes always replan; a non-static
    mode also replans on speed/direction changes or on timer expiry while moving.
    """
    mode_changed = current.locomotion_mode != previous.locomotion_mode
    facing_changed = not np.array_equal(current.facing_direction, previous.facing_direction)
    height_changed = current.height != previous.height
    if mode_changed or facing_changed or height_changed:
        return True
    if is_static_mode(current.locomotion_mode):
        return False
    speed_changed = current.movement_speed != previous.movement_speed
    direction_changed = not np.array_equal(current.movement_direction, previous.movement_direction)
    return speed_changed or direction_changed or (timer_expired and current.movement_speed != 0)


def replan_interval_for(mode: int, config: PlannerConfig) -> float:
    """Timer period for periodic replans of a mode (running/crawling/punching families)."""
    if mode == LocomotionMode.RUN:
        return config.replan_interval_running
    if mode == LocomotionMode.CRAWLING:
        return config.replan_interval_crawling
    if mode in (
        LocomotionMode.LEFT_PUNCH, LocomotionMode.RIGHT_PUNCH, LocomotionMode.RANDOM_PUNCH,
        LocomotionMode.LEFT_HOOK, LocomotionMode.RIGHT_HOOK,
    ):  # fmt: skip
        return config.replan_interval_boxing
    return config.replan_interval


class SonicPlanner:
    """10 Hz replanning thread + control-side blending for the live planner motion.

    Parameters
    ----------
    onnx : PlannerOnnx
        The loaded planner graph.
    scheduler : MotionScheduler
        The shared playback state; ``advance()`` swaps the live planner motion in through it.
    robot_state_fn : Callable[[], tuple[np.ndarray, np.ndarray] | None]
        Returns the latest ``(base_quat wxyz (4,), q_mj (29,))`` robot snapshot, or None before
        the first one. Captured on the control thread; the planner thread never touches the simulator.
    movement_state_fn : Callable[[], MovementState | None]
        Returns the latest operator :class:`MovementState`, or None when nothing has arrived yet.
    config : PlannerConfig | None
        Planner constants; ``None`` uses the defaults.
    initial_encode_mode : int
        Encoder mode stamped on the live planner motion when (re)initializing (0 with an encoder present).
    """

    def __init__(
        self,
        onnx: PlannerOnnx,
        scheduler: MotionScheduler,
        robot_state_fn: Callable[[], tuple[np.ndarray, np.ndarray] | None],
        movement_state_fn: Callable[[], MovementState | None],
        config: PlannerConfig | None = None,
        initial_encode_mode: int = 0,
    ) -> None:
        self._onnx = onnx
        self._scheduler = scheduler
        self._robot_state_fn = robot_state_fn
        self._movement_state_fn = movement_state_fn
        self.config = config or PlannerConfig()
        self._valid_mode_count = valid_mode_count(self.config.version)  # a bad version fails here, not per replan
        self.state = PlannerState()
        #: The exception that stopped the planner thread, for the control thread to surface.
        self.error: BaseException | None = None
        self._initial_encode_mode = initial_encode_mode
        self._random_seed = self.config.initial_random_seed

        # Live planner motion (what the policy tracks). Empty until the first
        # generation is adopted by advance().
        self.planner_motion = self._empty_planner_motion()

        # One lock guards every access to `planner_motion` (control-thread
        # blend / idle-readapt / reassign; planner-thread context read / (re)init)
        # AND the generation hand-off buffer below — the reference's single
        # `planner_motion_mutex_` covering both planner buffers on both threads.
        # Never held while a scheduler method (its own lock) is called.
        self._motion_lock = threading.Lock()
        self._generated: MotionFrames | None = None
        self._generated_at_frame = 0  # gen_frame the context was sampled for
        self._motion_available = False

        self._last_movement_state = MovementState()
        self._replan_timer = 0.0

        # Idle re-adaptation state (control thread only).
        self._readapt_state = "idle"
        self._readapt_stored = False
        self._readapt_original_targets = np.zeros(29)

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def _empty_planner_motion(self) -> MotionFrames:
        motion = MotionFrames(
            name="planner_motion",
            joint_pos=np.zeros((0, 29)),
            joint_vel=np.zeros((0, 29)),
            body_quat=np.zeros((0, 1, 4)),
            body_pos=np.zeros((0, 1, 3)),
        )
        motion.encode_mode = self._initial_encode_mode
        return motion

    # ------------------------------------------------------------------
    # planner thread (10 Hz)
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="sonic-planner", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop_event.wait(self.config.planner_dt):
            try:
                self.step()
            except Exception as e:  # noqa: BLE001 — handed to the control thread, which ends the session
                # Stop planning and leave the shared state alone: the control
                # thread owns the scheduler (a set_motion(None) from here could
                # also wipe a streamed motion mid mode switch) and raises this
                # through ``raise_if_failed`` with the traceback attached.
                self.state.enabled = False
                self.state.initialized = False
                self.error = e
                return

    def raise_if_failed(self) -> None:
        """Re-raise a planner-thread failure on the calling (control) thread.

        The planner thread stops on failure; without this re-raise the robot would hold
        its stance indefinitely with no planner and no visible error.
        """
        if self.error is not None:
            raise RuntimeError(
                f"the SONIC planner thread failed ({type(self.error).__name__}: {self.error}); the planner "
                "is stopped, so the session ends here rather than holding the stance with no planner"
            ) from self.error

    def step(self) -> None:
        """One planner-thread iteration: initialize, or decide and replan."""
        if not self.state.enabled:
            self.state.initialized = False
            return
        robot_state = self._robot_state_fn()
        if robot_state is None:
            return

        if not self.state.initialized:
            self._initialize(robot_state)
            return

        if self.planner_motion.timesteps == 0:
            return  # first generation not yet adopted by the control thread

        movement = self._movement_state_fn() or MovementState()
        self._replan_timer += self.config.planner_dt
        timer_expired = self._replan_timer >= replan_interval_for(movement.locomotion_mode, self.config)
        if timer_expired:
            self._replan_timer = 0.0
        if not should_replan(self._last_movement_state, movement, timer_expired):
            return
        self._last_movement_state = movement

        gen_frame = self._scheduler.frame + self.config.look_ahead_steps
        with self._motion_lock:
            if self.planner_motion.timesteps == 0:
                return  # control thread cleared/replaced the motion since the check above
            context = sample_context_from_motion(self.planner_motion, gen_frame)
        self._generate(context, movement, gen_frame)

    def _initialize(self, robot_state: tuple[np.ndarray, np.ndarray]) -> None:
        """First inference: idle in place from the robot's measured joints."""
        _base_quat, joint_positions_mj = robot_state
        with self._motion_lock:
            self.planner_motion = self._empty_planner_motion()
        context = initial_context(joint_positions_mj, self.config.default_height)
        idle = MovementState(locomotion_mode=int(LocomotionMode.IDLE), movement_speed=-1.0, height=-1.0)
        self._generate(context, idle, gen_frame=0)
        self._last_movement_state = idle
        self.state.initialized = True

    def _generate(self, context: np.ndarray, movement: MovementState, gen_frame: int) -> None:
        mode = int(movement.locomotion_mode)
        if not 0 <= mode < self._valid_mode_count:
            print(f"[SonicPlanner] invalid mode {mode} for version {self.config.version}; using IDLE")
            mode = int(LocomotionMode.IDLE)
        qpos = self._onnx.generate(
            context_qpos=context,
            mode=mode,
            target_vel=float(movement.movement_speed),
            target_height=float(movement.height),
            movement_direction=movement.movement_direction,
            facing_direction=movement.facing_direction,
            random_seed=self._random_seed,
        )
        generated = resample_30hz_to_50hz(qpos)
        with self._motion_lock:
            self._generated = generated
            self._generated_at_frame = gen_frame
            self._motion_available = True

    # ------------------------------------------------------------------
    # control thread (50 Hz)
    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        """True while the planner owns the reference motion."""
        return self.state.enabled and self.state.initialized

    def advance(self, measured_q_mj: np.ndarray | None) -> None:
        """End-of-tick planner bookkeeping (reference ``CurrentFrameAdvancement``).

        Adopts/blends a fresh generation into the live motion (resetting the
        cursor to 0), advances playback clamped to the last frame, and runs
        the idle re-adaptation blend against the measured joints.
        """
        if not self.active:
            return

        with self._motion_lock:
            generated = self._generated if self._motion_available else None
            gen_frame = self._generated_at_frame
            self._motion_available = False

        if generated is not None and generated.timesteps > 0:
            current_frame = self._scheduler.frame  # read outside _motion_lock (the two locks never nest)
            with self._motion_lock:
                first_time = self.planner_motion.timesteps == 0
                if first_time:
                    adopted = MotionFrames(
                        name="planner_motion",
                        joint_pos=generated.joint_pos,
                        joint_vel=generated.joint_vel,
                        body_quat=generated.body_quat,
                        body_pos=generated.body_pos,
                    )
                    adopted.encode_mode = self.planner_motion.encode_mode
                    self.planner_motion = adopted
                else:
                    try:
                        self.planner_motion = blend_planner_motion(
                            self.planner_motion, generated, current_frame, gen_frame
                        )
                    except ValueError:
                        generated = None  # arrived too late; keep the motion, still advance below
            if generated is not None:
                self._scheduler.set_motion(self.planner_motion, frame=0, play=None, reinitialize_heading=first_time)
                self._readapt_stored = False

        motion = self._scheduler.motion
        if not self._scheduler.play or motion is None or motion.timesteps == 0:
            return
        new_frame = self._scheduler.frame + 1
        if new_frame >= motion.timesteps:
            new_frame = motion.timesteps - 1
            movement = self._movement_state_fn()
            if (
                motion is self.planner_motion
                and measured_q_mj is not None
                and movement is not None
                and movement.locomotion_mode == LocomotionMode.IDLE
            ):
                with self._motion_lock:
                    self._idle_readapt(new_frame, np.asarray(measured_q_mj, dtype=np.float64))
        self._scheduler.set_frame(new_frame)

    def _idle_readapt(self, frame: int, measured_q_mj: np.ndarray) -> None:
        """Blend held lower-body targets toward the measured pose (and back).

        Double-threshold state machine that stops the policy from fighting a
        drifted standing pose: ADAPTING pulls the clamped frame's lower-body
        targets 2 % per tick toward the robot, RECOVERING pulls them back
        toward the original planner targets. Mutates the live motion in place
        (between ticks, like the reference). Runs under ``_motion_lock``.
        """
        if frame >= self.planner_motion.timesteps:
            return  # planner_motion was replaced (e.g. planner error) — nothing to adapt
        lower = _LOWER_BODY_IL
        targets = self.planner_motion.joint_pos[frame]
        if not self._readapt_stored:
            self._readapt_original_targets = targets.copy()
            self._readapt_stored = True
            self._readapt_state = "idle"

        measured_il = measured_q_mj[_MJ_TO_IL]  # hardware order -> IsaacLab
        avg_error = float(np.mean(np.abs(targets[lower] - measured_il[lower])))

        if self._readapt_state == "idle":
            if avg_error > _ADAPT_TRIGGER:
                self._readapt_state = "adapting"
            elif avg_error < _RECOVER_TRIGGER:
                self._readapt_state = "recovering"
        elif self._readapt_state == "adapting" and avg_error < _ADAPT_STOP:
            self._readapt_state = "idle"
        elif self._readapt_state == "recovering" and avg_error > _ADAPT_TRIGGER:
            self._readapt_state = "adapting"

        if self._readapt_state == "adapting":
            targets[lower] = 0.98 * targets[lower] + 0.02 * measured_il[lower]
        elif self._readapt_state == "recovering":
            targets[lower] = 0.98 * targets[lower] + 0.02 * self._readapt_original_targets[lower]
