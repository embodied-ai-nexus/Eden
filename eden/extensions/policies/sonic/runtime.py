"""SONIC teleoperation robot session: robot body + policy + planner + ZMQ, in one process.

This is the Eden counterpart of the reference stack's T1 (MuJoCo sim) and T2
(deploy binary) combined: a plant (:mod:`eden.extensions.policies.sonic.plant`)
is the robot body, the SONIC policy is the brain, and the operator station
talks to this process over the reference-compatible ZMQ channels. Run it with
``eden teleop sonic robot``. The body is the Genesis env by default; ``--mujoco``
(sim-to-sim) and ``--real`` (the Unitree G1) drive the body-only G1 through
the same deployment backends ``eden deploy --policy sonic`` uses.

Per 50 Hz control tick (wall-clock paced):

1. ``SonicZmqServer.apply`` — consume operator messages, drive mode state.
2. One body read, then the policy step (once the operator has engaged, a
   motion exists and — on a deployment — START handed control over) or the
   hold at the SONIC standing pose; Dex3 hand targets on a body with hands.
3. ``plant.step`` (elastic-band wrench + ``env.step``, or the deployment's step).
4. Planner adopt/advance (or plain frame advance), robot-state snapshot for
   the planner thread, ``g1_debug`` publish.

Keys (global pynput listener): ``9`` drop/suspend band (the gantry under
``--mujoco``), ``7``/``8`` band length, ``o`` e-stop, ``i`` re-capture
heading, ``q``/``e`` delta heading +/- 0.1 rad. A deployment adds its own
operator sequence (INIT, START, STOP, KILL) on the terminal or the G1 remote.

With ``--auto-reset``, a sustained fall while the band is off (engaged or
not) returns the whole session to the launch state — robot re-hung on the
band at the origin, policy/planner/motion state cleared, operator disengaged
— and the operator (re-)engages with A+B+X+Y.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from typing import TYPE_CHECKING

import numpy as np
import torch
from genesis.utils.geom import quat_to_R

from eden.extensions.policies.sonic.joint_maps import DEFAULT_ANGLES_MJ

if TYPE_CHECKING:
    from eden.extensions.policies.sonic.plant import RobotPlant

FALL_MAX_UP_DEVIATION_COS = 0.5  # 60° between the measured and reference up-axes
FALL_TRIGGER_TICKS = 20  # 0.4 s sustained at 50 Hz

_UPRIGHT_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


def fall_reference_quat(engaged: bool, motion, frame: int) -> np.ndarray:
    """Reference root quat for fall detection.

    The commanded motion's root while engaged (so crawling/lying modes don't
    read as falls); upright for the disengaged standing hold, which is how the
    reference sim resets a robot that topples before or between engagements.
    """
    if engaged and motion is not None and motion.timesteps > 0 and motion.body_quat.shape[1] > 0:
        return motion.body_quat[min(frame, motion.timesteps - 1), 0]
    return _UPRIGHT_QUAT


class FallDetector:
    """Detects a fallen base by comparing its up-axis against a commanded reference.

    Absolute tilt/height thresholds would false-positive on commanded crawling
    and lying locomotion modes (and deep squats), so the signal is the angle
    between the measured base up-axis and the reference root's up-axis (see
    :func:`fall_reference_quat`) — yaw-invariant (heading capture and
    delta-heading never enter) and tracking whatever posture is currently
    commanded.
    """

    def __init__(
        self, max_deviation_cos: float = FALL_MAX_UP_DEVIATION_COS, trigger_ticks: int = FALL_TRIGGER_TICKS
    ) -> None:
        self._max_deviation_cos = max_deviation_cos
        self._trigger_ticks = trigger_ticks
        self._counter = 0

    def update(self, armed: bool, base_quat_wxyz: np.ndarray, ref_root_quat_wxyz: np.ndarray | None) -> bool:
        """Advance one control tick; True when a sustained fall is detected."""
        if not armed or ref_root_quat_wxyz is None:
            self._counter = 0
            return False
        # Third row of R = world z expressed in the body frame; the dot product of the
        # two is the cosine of the tilt deviation, unaffected by either frame's yaw.
        cos_dev = float(quat_to_R(base_quat_wxyz)[2] @ quat_to_R(ref_root_quat_wxyz)[2])
        self._counter = self._counter + 1 if cos_dev < self._max_deviation_cos else 0
        if self._counter >= self._trigger_ticks:
            self._counter = 0
            return True
        return False


HOLD_RAMP_RATE_RAD_S = 0.5  # joint speed of the pre-engage approach to the standing pose
JOINT_VEL_LIMIT_RAD_S = 35.0  # the reference deploy binary's hardware runaway trip


class HoldRamp:
    """The pre-engage hold target: a rate-limited approach to the SONIC standing pose.

    Seeded from the measured pose (session start, every reset, every deployment START), it
    moves each joint toward the stand by at most one step per tick and snaps onto the
    stand within a step. At the stand it returns the exact float32 ``DEFAULT_ANGLES_MJ``
    row, so in a simulator that spawns at the stand the hold is the reference's fixed
    hold; on hardware it removes the full-gain jump a START from a different pose (no
    INIT, or a STOP then START) would otherwise command.
    """

    def __init__(self, control_dt: float, device: torch.device | str, rate_rad_s: float = HOLD_RAMP_RATE_RAD_S) -> None:
        self._target = np.asarray(DEFAULT_ANGLES_MJ, dtype=np.float64)
        self._target_row = torch.tensor(DEFAULT_ANGLES_MJ, dtype=torch.float32, device=device).reshape(1, -1)
        self._step = rate_rad_s * control_dt
        self._device = device
        self._current = self._target

    def seed(self, q_body: np.ndarray) -> None:
        """Restart the approach from a measured body pose."""
        q_body = np.asarray(q_body, dtype=np.float64).reshape(-1)
        if q_body.shape != self._target.shape:
            raise ValueError(f"HoldRamp.seed expects {self._target.shape[0]} body joints, got {q_body.shape[0]}")
        self._current = q_body

    def step(self) -> torch.Tensor:
        """Advance one tick and return this tick's ``(1, 29)`` hold target."""
        delta = self._target - self._current
        self._current = np.where(np.abs(delta) <= self._step, self._target, self._current + np.sign(delta) * self._step)
        if np.array_equal(self._current, self._target):
            return self._target_row
        return torch.as_tensor(self._current, dtype=torch.float32, device=self._device).reshape(1, -1)


def joint_over_velocity_limit(qd_body: np.ndarray, limit_rad_s: float) -> int | None:
    """Index of the first body joint faster than ``limit_rad_s``, or ``None`` (``limit <= 0`` disables).

    The reference deploy binary stops at 35 rad/s on the real robot and skips the check in simulation.
    """
    if limit_rad_s <= 0.0:
        return None
    over = np.flatnonzero(np.abs(qd_body) > limit_rad_s)
    return int(over[0]) if over.size else None


LOOP_REPORT_TICKS = 50  # one health line per second at 50 Hz (reference cadence)
DEADLINE_RESYNC_S = 0.5  # a stall longer than this abandons the lost ticks


class _LoopStats:
    """Control-loop health accumulator; the reference deploy prints its own every 50 ticks."""

    def __init__(self) -> None:
        self.deadline_resyncs = 0
        self._work_sum = 0.0
        self._work_max = 0.0
        self._ticks = 0
        self._window_start = time.monotonic()

    def note(self, work_s: float) -> None:
        self._work_sum += work_s
        self._work_max = max(self._work_max, work_s)
        self._ticks += 1

    def report(self, stream_lag: int | None, midstream_catchups: int) -> str:
        """One health line; resets the reporting window."""
        now = time.monotonic()
        rate = self._ticks / max(now - self._window_start, 1e-9)
        line = (
            f"loop {rate:5.1f} Hz | work {1e3 * self._work_sum / max(self._ticks, 1):5.2f} ms avg"
            f" / {1e3 * self._work_max:5.2f} ms max"
            f" | stream lag {'-' if stream_lag is None else stream_lag} frames"
            f" | stream catchups {midstream_catchups} | resyncs {self.deadline_resyncs}"
        )
        self._work_sum = self._work_max = 0.0
        self._ticks = 0
        self._window_start = now
        return line


def _pace(deadline: float, control_dt: float, stats: _LoopStats) -> float:
    """Absolute-deadline pacing: sleep to the deadline, resync only after a long stall.

    A single overrun is amortized by running the following ticks back-to-back
    (the reference recurrent control thread behaves the same); only a stall
    longer than DEADLINE_RESYNC_S abandons the lost ticks, so a streamed
    motion's cursor cannot fall arbitrarily far behind the operator's clock.
    """
    now = time.monotonic()
    if now < deadline:
        time.sleep(deadline - now)
    elif now - deadline > DEADLINE_RESYNC_S:
        stats.deadline_resyncs += 1
        deadline = now
    return deadline + control_dt


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eden teleop sonic robot",
        description=(
            "SONIC whole-body teleoperation: robot body + policy process. The body is the Genesis simulator "
            "by default; --mujoco (sim-to-sim) and --real (the Unitree G1) drive the body-only G1 through the "
            "`eden deploy` backends, and the deployment flags keep their `eden deploy` spelling."
        ),
    )
    body = parser.add_mutually_exclusive_group()
    body.add_argument("--mujoco", action="store_true", help="Sim-to-sim: drive the MuJoCo deployment backend")
    body.add_argument("--real", action="store_true", help="Drive the real Unitree G1 over DDS")
    parser.add_argument("-v", "--vis", action="store_true", help="Show the viewer (MuJoCo's own under --mujoco)")
    parser.add_argument("-c", "--cpu", action="store_true", help="Use the CPU backend")
    parser.add_argument("-s", "--steps", type=int, default=None, help="Stop after N control ticks (default: run)")
    parser.add_argument("--operator-host", default="localhost", help="Host running the operator station (PUB :5556)")
    parser.add_argument("--cmd-port", type=int, default=5556, help="Operator command/planner/pose port")
    parser.add_argument("--debug-port", type=int, default=5557, help="g1_debug publish port (bound here)")
    parser.add_argument("--encoder-path", default=None, help="Local override for the SONIC encoder ONNX")
    parser.add_argument("--decoder-path", default=None, help="Local override for the SONIC decoder ONNX")
    parser.add_argument("--planner-path", default=None, help="Local override for the planner ONNX")
    parser.add_argument(
        "--max-close-ratio", type=float, default=1.0, help="Dex3 close limit in [0.2, 1.0] (Genesis body only)"
    )
    parser.add_argument(
        "--no-band", action="store_true", help="Start with the elastic band (--mujoco: the gantry) released"
    )
    parser.add_argument(
        "--drop-band-after",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Release the elastic band this many seconds in (headless hosts have no '9' key)",
    )
    parser.add_argument(
        "--auto-reset",
        action="store_true",
        help="Reset to the launch state when the base falls with the band off (band re-armed unless --no-band)",
    )
    deploy = parser.add_argument_group("deployment body (--mujoco / --real)")
    deploy.add_argument(
        "--deploy_kp_kd_scale", type=float, default=1.0, help="Scale on the default kp/kd (ramp up from 0.2)"
    )
    deploy.add_argument("--target_lpf", type=float, default=1.0, help="Target low-pass in (0, 1]; 1.0 disables")
    deploy.add_argument(
        "--disable_joints", nargs="+", default=(), metavar="JOINT", help="Joint names / patterns sent with kp = kd = 0"
    )
    deploy.add_argument("--debug_record", action="store_true", help="Record commanded vs measured state to an .npz")
    deploy.add_argument("--debug_record_path", default=None, help="Destination for --debug_record")
    deploy.add_argument(
        "--joint-vel-limit",
        type=float,
        default=JOINT_VEL_LIMIT_RAD_S,
        help="--real: stop when any body joint exceeds this speed in rad/s (0 disables)",
    )
    return parser


def _check_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject bad flags before the Genesis build and the ONNX loads, not minutes into them."""
    for flag, path in (
        ("--encoder-path", args.encoder_path),
        ("--decoder-path", args.decoder_path),
        ("--planner-path", args.planner_path),
    ):
        if path is not None and not os.path.isfile(path):
            parser.error(f"{flag} {path!r} is not a file; pass a local ONNX path, or drop {flag} for the released one")
    if not 0.2 <= args.max_close_ratio <= 1.0:
        parser.error(f"--max-close-ratio must be in [0.2, 1.0], got {args.max_close_ratio}")
    if not 0.0 < args.target_lpf <= 1.0:
        parser.error(f"--target_lpf must be in (0, 1] (1.0 disables), got {args.target_lpf}")

    def passed(flags: dict[str, bool]) -> str:
        return ", ".join(flag for flag, on in flags.items() if on)

    deployed = args.mujoco or args.real
    deploy_only = passed(
        {
            "--deploy_kp_kd_scale": args.deploy_kp_kd_scale != 1.0,
            "--target_lpf": args.target_lpf != 1.0,
            "--disable_joints": bool(args.disable_joints),
            "--debug_record": args.debug_record,
            "--debug_record_path": args.debug_record_path is not None,
        }
    )
    if deploy_only and not deployed:
        parser.error(f"{deploy_only}: these configure a deployment body; pass --mujoco or --real")
    if args.joint_vel_limit != JOINT_VEL_LIMIT_RAD_S and not args.real:
        parser.error("--joint-vel-limit is the real robot's runaway trip; pass --real")
    if deployed and args.max_close_ratio != 1.0:
        parser.error("--max-close-ratio: the deployment body is the body-only G1, which drives no hands")
    if args.real:
        sim_only = passed(
            {
                "--auto-reset": args.auto_reset,
                "--no-band": args.no_band,
                "--drop-band-after": args.drop_band_after is not None,
            }
        )
        if sim_only:
            parser.error(
                f"{sim_only}: the real robot has no elastic band and cannot be reset to the launch pose — "
                "hang it from a gantry and use the remote's STOP"
            )
        from eden.extensions.deployment.utils.unitree_sdk2 import ensure_unitree_sdk

        try:
            ensure_unitree_sdk()
        except RuntimeError as e:
            parser.error(f"--real: {e}")


def _discard_terminal_input() -> None:
    """Drop whatever is typed but unread on the terminal's input line (no-op without a TTY)."""
    if not sys.stdin.isatty():
        return
    import termios

    termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)


def _build_plant(args: argparse.Namespace, env, robot_options) -> RobotPlant:
    """The body this session drives: the Genesis env itself, a MuJoCo deployment, or the real G1.

    Called after the policy is constructed: a deployment swaps the env's robot entity for a
    state wrapper, and the policy resolves its DOF and link indices against the real one.
    """
    from eden.extensions.policies.sonic.plant import DeploymentPlant, GantryBand, GenesisPlant

    if not (args.mujoco or args.real):
        return GenesisPlant(env)

    from eden.extensions.deployment import DEPLOYMENT_REGISTRY, resolve_deployment_for

    # sync=False: this loop paces itself (absolute deadlines with catch-up, like the reference
    # control thread). auto_yaw_align=False: the policy captures its own heading on engage.
    options = {
        "entity_name": "robot",
        "sync": False,
        "auto_yaw_align": False,
        "connect_timeout_s": 10.0,
        "deploy_kp_kd_scale": args.deploy_kp_kd_scale,
        "target_lpf": args.target_lpf,
        "disable_joints": tuple(args.disable_joints),
        "debug_record": args.debug_record,
        "debug_record_path": args.debug_record_path,
    }
    if args.mujoco:
        deployment_cls = DEPLOYMENT_REGISTRY.get("mujoco_deployment")
        # The gantry is this body's band: the robot hangs from it until the operator releases it.
        dep = deployment_cls(env, deployment_cls.configure(**options, show_viewer=args.vis, gantry=not args.no_band))
        return DeploymentPlant(dep, is_hardware=False, band=GantryBand(dep))
    sim_robot = env.entities["robot"]  # before the deployment swaps it for its state wrapper
    deployment_cls = resolve_deployment_for(robot_options)
    dep = deployment_cls(env, deployment_cls.configure(**options))
    return DeploymentPlant(dep, is_hardware=True, mirror=(env, sim_robot) if args.vis else None)


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _check_args(parser, args)

    import genesis as gs

    import eden as en
    from eden.extensions.policies.sonic.motion import MotionScheduler
    from eden.extensions.policies.sonic.planner import PlannerOnnx, SonicPlanner
    from eden.extensions.policies.sonic.plant import SonicTeleopEnv
    from eden.extensions.policies.sonic.policy import SonicPolicy
    from eden.extensions.policies.sonic.teleop_policy import ReferenceUnavailableError, SonicTeleopPolicy
    from eden.extensions.policies.sonic.wire import G1DebugPublisher
    from eden.extensions.policies.sonic.zmq_runtime import Dex3HandController, SonicZmqServer
    from eden.options import ActionManagerOptions, CommandManagerOptions, EnvOptions, ObservationManagerOptions
    from eden.options.robots.unitree_g1 import G1_DOFS29_NAME

    en.init(backend=gs.cpu if args.cpu else gs.gpu)

    # Bound before the slow env/model construction so a port collision fails
    # fast instead of after the Genesis build and ONNX loads.
    try:
        debug_pub = G1DebugPublisher(port=args.debug_port)
    except Exception as e:  # surface the actionable cause, whatever zmq raises
        raise SystemExit(
            f"[sonic robot] cannot bind the g1_debug port {args.debug_port} ({e}) — a previous robot "
            "session (or a default-port pico bridge) is still running or shutting down; stop it, or "
            "pass --debug-port"
        ) from e

    if args.encoder_path is not None:
        SonicTeleopPolicy.use_local_checkpoint(args.encoder_path)
    if args.decoder_path is not None:
        SonicTeleopPolicy.use_local_checkpoint(args.decoder_path, checkpoint_attr="decoder_checkpoint_path")
    if args.planner_path is not None:
        SonicTeleopPolicy.use_local_checkpoint(args.planner_path, checkpoint_attr="planner_checkpoint_path")

    # A deployment body is the robot `eden deploy --policy sonic` runs a clip on — the
    # body-only G1 at the checkpoint's gains — so its env comes from that policy's spec.
    spec = SonicPolicy if args.mujoco or args.real else SonicTeleopPolicy
    robot_options = spec.robot_options()
    env = SonicTeleopEnv(
        env_options=EnvOptions(num_envs=1, episode_length_s=1e9, **spec.env_options()),
        scene_options=en.scenes.Empty(robot=robot_options),
        command_options=CommandManagerOptions(**SonicTeleopPolicy.command_options()),
        action_options=ActionManagerOptions(**spec.action_options()),
        observation_options=ObservationManagerOptions(**spec.observation_options()),
        # MuJoCo opens its own viewer; the real robot is mirrored in the Genesis one.
        show_viewer=args.vis and not args.mujoco,
        eval_mode=True,
    )
    env.build()

    policy = SonicTeleopPolicy(env=env, options=SonicTeleopPolicy.configure())
    plant = _build_plant(args, env, robot_options)
    band = plant.band
    if band is None and (args.auto_reset or args.drop_band_after is not None):
        raise SystemExit("[sonic robot] --auto-reset / --drop-band-after need a body with a band; this one has none")
    scheduler = MotionScheduler()
    policy.attach_runtime(scheduler)
    hands = Dex3HandController(max_close_ratio=args.max_close_ratio)

    # Robot-state snapshot for the planner thread (it never touches the body).
    state_lock = threading.Lock()
    state_snapshot: dict[str, np.ndarray] = {}

    def robot_state_fn() -> tuple[np.ndarray, np.ndarray] | None:
        with state_lock:
            if not state_snapshot:
                return None
            return state_snapshot["base_quat"], state_snapshot["q_mj"]

    planner_graph = PlannerOnnx(policy.resolve_planner_path(), device=str(env.device))
    planner = SonicPlanner(
        onnx=planner_graph,
        scheduler=scheduler,
        robot_state_fn=robot_state_fn,
        movement_state_fn=lambda: server.movement_state,
    )
    server = SonicZmqServer(scheduler, planner, hands, host=args.operator_host, port=args.cmd_port)
    if band is not None:
        band.enable = not args.no_band

    estop_requested = threading.Event()
    owns_terminal = args.mujoco or args.real

    def on_key(key: str):
        def _cb(pressed: bool) -> None:
            if not pressed:
                return
            if owns_terminal:
                # The key was also typed into the terminal, where the deployment reads its
                # INIT/START/STOP/KILL lines: left there it would prefix the next one
                # ("9x" is not STOP). Dropped a moment later, once the tty has it.
                threading.Timer(0.1, _discard_terminal_input).start()
            if band is not None and band.handle_key(key):
                return
            if key == "o":
                # Consumed on the control thread: the keyboard thread must not
                # write scheduler state (snapshot() would race the clear).
                print("[sonic robot] EMERGENCY STOP (o)")
                estop_requested.set()
            elif key == "i":
                scheduler.request_heading_reinitialize()
                print("[sonic robot] heading re-capture requested")
            elif key == "q":
                scheduler.add_heading_increment(+0.1)
            elif key == "e":
                scheduler.add_heading_increment(-0.1)

        return _cb

    from eden.utils.devices.keyboard import Keyboard

    try:
        kb = Keyboard(key_bindings={})
    except ImportError as e:  # pynput missing, or a headless session (no display to read keys from)
        kb = None
        print(f"[sonic robot] keyboard unavailable ({e}); band/e-stop keys disabled")
    else:
        for key in ("9", "7", "8", "o", "i", "q", "e"):
            kb.add_callback(key, on_key(key))
        band_keys = "" if band is None else f"{band.key_help}, "
        print(f"[sonic robot] keys: {band_keys}o e-stop, i reinit heading, q/e delta yaw")

    control_dt = plant.control_dt
    hold = HoldRamp(control_dt, env.device)
    fall_detector = FallDetector() if args.auto_reset else None
    reset_count = 0

    stats = _LoopStats()
    band_drop_pending = args.drop_band_after is not None
    tick = 0
    plant.connect()  # outside the try: a body that never opened has nothing to close
    try:
        plant.reset()
        policy.reset()
        hold.seed(plant.read().q_body)
        server.start()
        planner.start()

        deadline = time.monotonic() + control_dt
        if plant.is_hardware:
            print(
                "[sonic robot] REAL ROBOT: ending this session (o, KILL, Ctrl-C, the joint-velocity trip, a "
                "reference streamer's stop) damps the motors for about a second and then releases them — keep "
                "the robot on the gantry whenever you end it. The PICO's A+B+X+Y only disengages: the robot "
                "keeps balancing on the planner."
            )
        print(f"[sonic robot] listening on tcp://{args.operator_host}:{args.cmd_port}, control dt {control_dt:.3f}s")
        print("[sonic robot] waiting for the operator (A+B+X+Y on the PICO engages)")

        while args.steps is None or tick < args.steps:
            tick_start = time.monotonic()

            if band_drop_pending and tick * control_dt >= args.drop_band_after:
                band_drop_pending = False  # one-shot: an auto-reset re-arms the band and keeps it
                band.enable = False
                print("[sonic robot] band released (--drop-band-after)")

            if estop_requested.is_set():
                server.emergency_stop()
            planner.raise_if_failed()
            server.apply()
            if server.stop_requested:
                print("[sonic robot] stop requested — ending session")
                break

            # One body read per tick: the reference logs a single LowState at the
            # top of the control tick and serves every consumer — policy obs,
            # planner context, fall check, and the g1_debug publish — from it.
            reading = plant.read()
            hands_q = reading.q_hands
            if plant.is_hardware:
                over = joint_over_velocity_limit(reading.qd_body, args.joint_vel_limit)
                if over is not None:
                    print(
                        f"[sonic robot] joint velocity trip: {G1_DOFS29_NAME[over]} at "
                        f"{reading.qd_body[over]:.1f} rad/s (limit {args.joint_vel_limit:g}) — ending session"
                    )
                    break
            if not plant.policy_active:
                # A deployment in STOP / INIT ignores this loop's targets. Keep the hold
                # at the measured pose, so the tick START lands on — the deployment
                # forwards that tick's target — approaches the stand from where the robot
                # is instead of stepping to it at full gains.
                hold.seed(reading.q_body)
            if plant.consume_start_events():
                # Make the operator engage afresh: a session that kept running through
                # the STOP holds a motion the robot never followed.
                if server.operator_start:
                    reset_count += 1
                    server.reset_session()
                    policy.reset()
                print("[sonic robot] START — holding the standing pose; A+B+X+Y (re-)engages")
            robot_state = reading.snapshot(env.device)
            with state_lock:
                state_snapshot["base_quat"] = reading.base_quat
                state_snapshot["q_mj"] = reading.q_body

            # Read the motion once: a second scheduler.motion access could see a
            # planner/ZMQ swap in between (and `.timesteps` on a None it just
            # returned would raise).
            current_motion = scheduler.motion
            engaged = (
                server.operator_start
                and current_motion is not None
                and current_motion.timesteps > 0
                and plant.policy_active
            )
            if engaged:
                try:
                    action = policy.step(None, robot_state=robot_state)
                except ReferenceUnavailableError as e:
                    print(f"[sonic robot] {e} — stopping control")
                    break
            else:
                # WAIT_FOR_CONTROL: hold the SONIC standing pose (the reference
                # INIT ramp target) while hanging from the band. The state
                # history still rolls — the reference serves it from the
                # LowState it logs every tick — so the first engaged tick sees
                # the robot's real recent past, not nine zero frames (zero
                # gravity included, which the policy never saw in training).
                policy.record_state(robot_state)
                action = {"dofs_pos_controller": hold.step()}
            if plant.has_hands:
                action["hands_pos_controller"] = torch.as_tensor(
                    hands.compute(hands_q[:7], hands_q[7:]), dtype=torch.float32, device=env.device
                ).reshape(1, -1)

            plant.step(action)

            if planner.active:
                planner.advance(measured_q_mj=reading.q_body)
            else:
                scheduler.advance_frame(is_planner_motion=False)

            if fall_detector is not None:
                # Armed whenever the band is off (engaged or not — the reference
                # sim resets any toppled robot); a band-hung robot never triggers.
                # motion+frame read atomically so the frame indexes its own motion.
                ref_motion, ref_frame = scheduler.motion_and_frame()
                ref_quat = fall_reference_quat(engaged, ref_motion, ref_frame)
                if fall_detector.update(
                    armed=not band.enable, base_quat_wxyz=reading.base_quat, ref_root_quat_wxyz=ref_quat
                ):
                    reset_count += 1
                    print(f"[sonic robot] FALL detected — auto-reset #{reset_count}; A+B+X+Y (re-)engages")
                    band.enable = not args.no_band
                    server.reset_session()
                    plant.reset()
                    policy.reset()
                    hold.seed(plant.read().q_body)
                    tick += 1
                    stats.note(time.monotonic() - tick_start)
                    deadline = _pace(deadline, control_dt, stats)
                    continue  # next tick publishes the incremented reset_count with fresh state

            plant.render()

            # Subset of the reference g1_debug keys: everything the operator
            # station reads, plus the base pose; policy internals (body_q,
            # last_action, token_state) are not republished. Every value comes
            # from this tick's single top-of-tick read, so the published
            # state is one consistent instant (like the reference's logged
            # LowState), not a mix of pre- and post-step reads.
            debug_motion = scheduler.motion
            debug_pub.publish(
                {
                    "control_loop_type": "eden",
                    "index": tick,
                    "ros_timestamp": time.time(),
                    "base_quat": reading.base_quat,  # wxyz
                    "base_quat_measured": reading.base_quat,
                    "base_ang_vel": reading.base_ang,
                    "base_trans_measured": reading.base_pos,  # reference key name; zeros when unmeasured
                    "body_q_measured": reading.q_body,  # MuJoCo/MJCF order, absolute radians
                    "left_hand_q_measured": hands_q[:7],  # zeros on a body without hands
                    "right_hand_q_measured": hands_q[7:],
                    "delta_heading": scheduler.delta_heading,
                    "encode_mode": debug_motion.encode_mode if debug_motion is not None else -2,
                    "operator_start": server.operator_start,
                    # Eden extra (not a reference key): the operator's reset
                    # watcher disengages its state machine on an increment.
                    "reset_count": reset_count,
                }
            )

            tick += 1
            stats.note(time.monotonic() - tick_start)
            if tick % LOOP_REPORT_TICKS == 0:
                print(f"[sonic robot] {stats.report(scheduler.stream_lag_frames(), server.midstream_catchups)}")
            deadline = _pace(deadline, control_dt, stats)
    except KeyboardInterrupt:
        print("[sonic robot] interrupted")
    finally:
        # The body first, whatever else fails: on the real robot this is the soft-damping
        # shutdown, and until it runs the last target is still being held at full gains.
        try:
            if plant.is_hardware:
                print("[sonic robot] ending: damping the motors, then releasing them — the robot must be supported")
            plant.close()
        finally:
            # Port next: the thread joins below can take seconds, and a relaunch
            # must not find :5557 still bound to this dying process.
            debug_pub.close()
            if kb is not None:
                kb.stop()
            planner.stop()
            server.stop()


if __name__ == "__main__":
    main()
