"""SONIC operator station: PICO body tracking -> ZMQ commands for the robot process.

The Eden counterpart of the reference stack's T3. One manager loop polls the
controllers, advances the operator state machine, runs the active mode's
streamer (POSE at ~50 Hz, planner modes at ~20 Hz), and publishes the
``command`` / ``manager_state`` messages. Run it with ``eden teleop sonic operator``.

PICO input arrives through ``eden pico bridge``, which this process spawns
(pose-only, port ``--bridge-port``) unless ``--bridge-host`` points at an
external one or ``--no-bridge-start`` is given; ``--sdk`` reads the
XRoboToolkit SDK in-process instead (the reference-parity path).

Controls (PICO):
    A+B+X+Y   engage + calibrate (stand in the neutral pose) / stop + exit
    A+X       toggle POSE <-> PLANNER
    B+Y       toggle POSE <-> PLANNER_FROZEN_UPPER_BODY
    L-stick   click: toggle VR_3PT from either planner mode
    A+B/X+Y   next / previous locomotion mode (planner modes)
    triggers  close/open that hand
    menu hold pause the pose stream (re-align before releasing)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from eden.extensions.policies.sonic.wire import build_command_message, build_manager_state_message
from eden.extensions.teleop.base import TeleopOperator
from eden.extensions.teleop.sonic.inputs import SONIC_BRIDGE_POSE_PORT
from eden.extensions.teleop.sonic.state_machine import (
    PLANNER_MODES,
    ButtonSnapshot,
    Effect,
    OperatorStateMachine,
    StreamMode,
    Transition,
)
from eden.options.extensions.teleop import TeleopOptions

_SERVICE_SCRIPT = "/opt/apps/roboticsservice/runService.sh"

# Modes that stream body poses, so a stale body stream means their input is gone.
_BODY_STREAM_MODES = (StreamMode.POSE, StreamMode.PLANNER_VR_3PT)


def input_lost(mode: StreamMode, *, bridge_died: bool, body_stale: bool) -> bool:
    """Whether an engaged operator has lost the input its current mode streams from.

    A dead bridge freezes everything. Body staleness only matters to the modes that
    stream body poses: PLANNER / FROZEN_UPPER run on the controllers (whose stalls the
    robot's planner timeout already handles), so a tracker occlusion must not end
    joystick walking.
    """
    if mode == StreamMode.OFF:
        return False
    return bridge_died or (body_stale and mode in _BODY_STREAM_MODES)


# How long the command socket may keep flushing a final command (stop, or the planner
# hand-off) at teardown. Ample for a loopback/LAN PUB (a ~1.3 kB message already queued);
# bounded so a robot that has gone away cannot hang the exit.
_FINAL_COMMAND_LINGER_MS = 500


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eden teleop sonic operator", description="SONIC operator station (PICO body tracking -> robot commands)."
    )
    parser.add_argument(
        "--robot-host", default="localhost", help="Host running 'eden teleop sonic robot' (g1_debug SUB)"
    )
    parser.add_argument("--port", type=int, default=5556, help="Command/planner/pose publish port (bound here)")
    parser.add_argument("--feedback-port", type=int, default=5557, help="Robot g1_debug port")
    parser.add_argument("--target-fps", type=int, default=50, help="Pose stream rate")
    parser.add_argument("--num-frames", type=int, default=5, help="Pose window size (frames per message)")
    parser.add_argument("--record-dir", default="", help="Record each sent pose message as .npz here")
    parser.add_argument("--no-service-start", action="store_true", help="Do not spawn the XRoboToolkit PC service")
    parser.add_argument(
        "--bridge-host",
        default="",
        help="Connect to an 'eden pico bridge' on this host (skips spawning the bridge and the PC service)",
    )
    parser.add_argument(
        "--bridge-port",
        type=int,
        default=SONIC_BRIDGE_POSE_PORT,
        help="Bridge pose PUB port (the default avoids the SONIC wire's 5556/5557)",
    )
    parser.add_argument(
        "--bridge-fps",
        type=float,
        default=120.0,
        help="Publish rate for the spawned bridge; keep it above the device body rate (~90 Hz)",
    )
    parser.add_argument(
        "--no-bridge-start", action="store_true", help="Do not spawn the bridge; connect to one already running locally"
    )
    parser.add_argument(
        "--sdk", action="store_true", help="Read the XRoboToolkit SDK in-process instead of via the bridge"
    )
    parser.add_argument("--vis-vr3pt", action="store_true", help="Visualize the calibrated VR 3-point pose")
    parser.add_argument("--vis-smpl", action="store_true", help="Also render the SMPL body (needs --vis-vr3pt)")
    parser.add_argument("--waist-tracking", action="store_true", help="Viz only: robot waist follows the neck")
    parser.add_argument("--no-g1", action="store_true", help="Viz only: hide the G1 mesh")
    return parser


def _bridge_argv(cfg: SonicTeleopOperator | argparse.Namespace) -> list[str]:
    """Command line for the spawned pose-only bridge (same interpreter/venv).

    ``cfg`` is anything carrying ``bridge_port`` / ``bridge_fps`` — the operator
    itself at runtime, or an ``argparse.Namespace`` in tests.
    """
    # --no-video: sonic never streams frames back, and the bridge's frame port
    # default (5556) would collide with the operator's own command PUB bind.
    # --no-derived-body: sonic reads only body_raw + controllers; skipping the
    # per-joint body dict and trackers keeps the bridge's per-frame cost low so
    # the publish rate (and with it the 50 Hz pose resampler) never starves.
    return [
        sys.executable,
        "-m",
        "eden.utils.devices.pico.bridge",
        "--no-video",
        "--no-derived-body",
        "--pose-port",
        str(cfg.bridge_port),
        "--fps",
        str(cfg.bridge_fps),
    ]


def _die_with_parent() -> None:
    """Popen preexec: SIGKILL the bridge when the operator dies, however it dies.

    The teardown terminate only runs on a clean exit or Ctrl-C; a hard-killed
    operator would otherwise orphan the bridge, which then holds the pose port
    (and can sit uninterruptibly inside the SDK). Linux-only (PR_SET_PDEATHSIG).
    """
    import ctypes
    import signal as _signal

    PR_SET_PDEATHSIG = 1
    ctypes.CDLL(None, use_errno=True).prctl(PR_SET_PDEATHSIG, _signal.SIGKILL)


class SonicTeleopOperator(TeleopOperator):
    """The SONIC operator station as a :class:`TeleopOperator`.

    ``setup`` builds the reader, publish socket, calibration FK, streamers and
    state machine (blocking until body data arrives); ``step`` runs one control
    tick (poll -> state machine -> effects -> stream -> publish); ``teardown``
    releases everything, publish socket first. The loop cadence is set by the
    active streamer's own pacing sleep inside its ``run_once``. Config is mirrored
    from a :class:`~eden.options.extensions.teleop.TeleopOptions` (build one with
    ``SonicTeleopOperator.configure(**vars(args))``); the PICO/SONIC-specific
    parameters below ride along as options extras.
    """

    #: Pose stream rate (Hz).
    target_fps: int = 50
    #: Pose window size (frames per message).
    num_frames: int = 5
    #: If set, record each sent pose message as an ``.npz`` here.
    record_dir: str = ""
    #: Do not spawn the XRoboToolkit PC service.
    no_service_start: bool = False
    #: Connect to an ``eden pico bridge`` on this host (skips spawning it and the service).
    bridge_host: str = ""
    #: Bridge pose PUB port (off the SONIC wire's 5556/5557).
    bridge_port: int = SONIC_BRIDGE_POSE_PORT
    #: Publish rate for the spawned bridge (keep above the ~90 Hz device body rate).
    bridge_fps: float = 120.0
    #: Do not spawn the bridge; connect to one already running locally.
    no_bridge_start: bool = False
    #: Read the XRoboToolkit SDK in-process instead of via the bridge.
    sdk: bool = False
    #: Visualize the calibrated VR 3-point pose.
    vis_vr3pt: bool = False
    #: Also render the SMPL body (needs ``vis_vr3pt``).
    vis_smpl: bool = False
    #: Viz only: robot waist follows the neck.
    waist_tracking: bool = False
    #: Viz only: hide the G1 mesh.
    no_g1: bool = False

    def __init__(self, options: TeleopOptions) -> None:
        super().__init__(options)
        self._stop = False
        self._final_command_queued = False  # a planner hand-off is queued: teardown must flush it
        # The robot was handed to the planner and kept its session: it is still engaged, with the
        # heading it captured at the first engage, so the next engage must not reset the facing.
        self._robot_kept_heading = False
        # Built by setup(); None until then, so teardown can release a partial stack.
        self._context = self._socket = self._reader = self._bridge_proc = None
        self._feedback = self._reset_watcher = self._fk = self._viz = self._sm = None

    def setup(self) -> None:
        import genesis as gs
        import zmq

        import eden as en
        from eden.extensions.teleop.sonic.feedback import FeedbackReader, RobotResetWatcher
        from eden.extensions.teleop.sonic.g1_fk import G1KeyFrameFk
        from eden.extensions.teleop.sonic.planner_streamer import PlannerStreamer
        from eden.extensions.teleop.sonic.pose_streamer import PoseStreamer
        from eden.extensions.teleop.sonic.three_point import ThreePointPose

        # CPU backend: the calibration FK is a single-env kinematics scene, and
        # the operator pipelines are numpy — claiming a GPU (and torch's default
        # device) here would only fight the robot process.
        en.init(backend=gs.cpu)

        # A --bridge-host bridge runs on another machine, next to its own service.
        self._bridge_proc = None
        if not self.no_service_start and not self.bridge_host and os.path.exists(_SERVICE_SCRIPT):
            subprocess.Popen(["bash", _SERVICE_SCRIPT])
            print(f"[sonic operator] started the XRoboToolkit PC service ({_SERVICE_SCRIPT})")

        if self.sdk:
            from eden.extensions.teleop.sonic.inputs import PicoReader

            self._reader = PicoReader()
        else:
            from eden.extensions.teleop.sonic.inputs import BridgeReader

            if not self.bridge_host and not self.no_bridge_start:
                # PR_SET_PDEATHSIG is Linux-only; elsewhere the bridge is reaped by teardown alone.
                preexec_fn = _die_with_parent if sys.platform.startswith("linux") else None
                self._bridge_proc = subprocess.Popen(_bridge_argv(self), preexec_fn=preexec_fn)  # noqa: PLW1509
                print(
                    f"[sonic operator] started 'eden pico bridge' (pose port {self.bridge_port}, {self.bridge_fps:g} fps)"
                )
            self._reader = BridgeReader(host=self.bridge_host or "127.0.0.1", port=self.bridge_port)
        self._reader.start()

        # Bound before the slow startup (FK compile, headset wait) so a port
        # collision fails fast instead of minutes in. The command socket gets its
        # own context so teardown can term() it — the only call that waits for a
        # queued stop command to be written — without waiting on the feedback /
        # reader sockets that live on the shared instance.
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        try:
            self._socket.bind(f"tcp://*:{self.port}")
        except zmq.ZMQError as e:
            self._socket.close(linger=0)
            self._context.term()
            self._context = self._socket = None
            raise SystemExit(
                f"[sonic operator] cannot bind tcp://*:{self.port} ({e}) — a previous operator (or the "
                "reference streamer) is still running or still shutting down; stop it, or pass --port"
            ) from e
        print(f"[sonic operator] publishing on tcp://*:{self.port}")

        # Built eagerly, while the headset connects: the first calibration (the
        # engage press) must not stall on the kernel compile.
        print("[sonic operator] building the G1 kinematics scene (first run compiles, up to ~30 s)...")
        self._fk = G1KeyFrameFk()

        print("[sonic operator] waiting for body tracking data (headset in Full-body mode, service WORKING)...")
        while self._reader.get_latest() is None:
            if self._bridge_proc is not None and self._bridge_proc.poll() is not None:
                raise SystemExit(
                    f"[sonic operator] the pico bridge exited with code {self._bridge_proc.returncode} — "
                    f"check the xrobotoolkit_sdk build and that port {self.bridge_port} is free"
                )
            print("waiting for body data...")
            time.sleep(1)

        on_frame = None
        self._viz = None
        if self.vis_vr3pt:
            from eden.extensions.teleop.sonic.viz import Vr3PtPoseVisualizer

            self._viz = Vr3PtPoseVisualizer(
                with_g1_robot=not self.no_g1,
                enable_waist_tracking=self.waist_tracking,
                enable_smpl_vis=self.vis_smpl,
            )
            on_frame = self._viz.on_frame

        self._three_point = ThreePointPose(self._fk)
        self._feedback = FeedbackReader(host=self.robot_host, port=self.feedback_port)
        self._reset_watcher = RobotResetWatcher(host=self.robot_host, port=self.feedback_port)
        self._pose_streamer = PoseStreamer(
            send=self._socket.send,
            three_point=self._three_point,
            num_frames_to_send=self.num_frames,
            target_fps=self.target_fps,
            record_dir=self.record_dir,
            on_frame=on_frame,
        )
        self._planner_streamer = PlannerStreamer(
            send=self._socket.send, three_point=self._three_point, feedback=self._feedback, on_frame=on_frame
        )
        self._sm = OperatorStateMachine()

        print("[sonic operator] A+B+X+Y engages (stand in the neutral pose: forearms bent 90° forward)")
        print(
            "[sonic operator] A+B+X+Y again disengages: the robot balances in place on the planner; "
            "Ctrl-C exits the operator (the robot keeps idling); end the robot session from its own terminal"
        )
        self._prev_toggle_dc = False
        self._prev_toggle_da = False
        self._bridge_died = False

    def step(self) -> None:
        reader = self._reader
        socket = self._socket
        sm = self._sm

        if self._bridge_proc is not None and not self._bridge_died and self._bridge_proc.poll() is not None:
            self._bridge_died = True
            print(
                f"[sonic operator] WARNING: the pico bridge exited with code {self._bridge_proc.returncode}; "
                "the input stream is frozen — Ctrl-C to exit"
            )
        if input_lost(sm.mode, bridge_died=self._bridge_died, body_stale=reader.disconnected):
            # Frozen input would otherwise leave the robot on the last streamed pose
            # (POSE never times out robot-side) or the last VR 3-point target.
            forced = self._force_off("input lost", "A+B+X+Y re-engages once tracking is back")
            self._release_robot_to_planner(forced.previous)
            return
        controllers = reader.controllers()

        if self._reset_watcher.poll() and sm.mode != StreamMode.OFF:
            self._force_off("robot fell and auto-reset", "stand in the neutral pose and press A+B+X+Y to re-engage")
            self._robot_kept_heading = False  # the reset disengaged the robot: the next engage re-captures it
            return

        transition = sm.step(
            ButtonSnapshot(
                a=controllers.a,
                b=controllers.b,
                x=controllers.x,
                y=controllers.y,
                left_axis_click=controllers.left_axis_click,
                left_menu=controllers.left_menu,
            )
        )

        for effect in transition.effects:
            if effect == Effect.POSE_EXIT:
                self._pose_streamer.on_mode_exit()
            elif effect == Effect.CALIBRATE_FULL:
                sample = reader.get_latest()
                if sample is not None:
                    self._three_point.calibrate_now(sample.body_poses)
                else:
                    print("[sonic operator] WARNING: no body data available for calibration")
            elif effect == Effect.RESET_POSE_YAW:
                self._pose_streamer.reset_yaw()
            elif effect == Effect.RESET_PLANNER_YAW:
                if transition.previous == StreamMode.OFF and self._robot_kept_heading:
                    # The robot stayed engaged on the planner, so it re-captures no heading on
                    # this engage: the facing it is idling with must carry over, not snap to forward.
                    self._robot_kept_heading = False
                else:
                    self._planner_streamer.reset_yaw()
            elif effect == Effect.RELEASE_TO_PLANNER:
                self._release_robot_to_planner(transition.previous)
                print(
                    "[sonic operator] disengaged — A+B+X+Y re-engages (full calibration: neutral pose, "
                    "looking ahead); Ctrl-C exits the operator"
                )
            elif effect == Effect.SAVE_FROZEN_TARGETS:
                self._planner_streamer.save_upper_body_position_target()
            elif effect == Effect.RECALIBRATE_WRISTS:
                self._planner_streamer.recalibrate_for_vr3pt()

        # Run one tick of the new mode (its pacing sleep sets the loop rate).
        if transition.mode == StreamMode.POSE:
            self._pose_streamer.run_once(reader.get_latest(), controllers)
        elif transition.mode in PLANNER_MODES:
            self._planner_streamer.run_once(
                transition.mode, reader.get_latest(), controllers, reader.get_timestamp_ns()
            )
        else:
            time.sleep(0.005)  # OFF / POSE_PAUSE: idle poll (no hot spin)

        # Command messages go out after the first tick of the new mode, so its
        # data is already in flight when the robot switches.
        if transition.changed:
            print(f"[sonic operator] {transition.previous.name} -> {transition.mode.name}")
            if transition.command is not None:
                kind, planner_flag = transition.command
                socket.send(build_command_message(start=kind == "start", stop=False, planner=planner_flag))

        # manager_state every tick (consumed by data recorders, not the robot).
        toggle_dc_now = controllers.a and controllers.left_grip > 0.5
        toggle_da_now = controllers.b and controllers.left_grip > 0.5
        socket.send(
            build_manager_state_message(
                stream_mode=transition.mode.value,
                toggle_data_collection=toggle_dc_now and not self._prev_toggle_dc,
                toggle_data_abort=toggle_da_now and not self._prev_toggle_da,
            )
        )
        self._prev_toggle_dc = toggle_dc_now
        self._prev_toggle_da = toggle_da_now

    def should_stop(self) -> bool:
        return self._stop

    def _force_off(self, reason: str, next_step: str) -> Transition:
        """Disengage the operator state machine outside a button transition (re-engage runs a full calibration)."""
        forced = self._sm.force_off()
        if Effect.POSE_EXIT in forced.effects:
            self._pose_streamer.on_mode_exit()
        print(f"[sonic operator] {reason} ({forced.previous.name} -> OFF) — {next_step}")
        time.sleep(0.005)
        return forced

    def _release_robot_to_planner(self, engaged_mode: StreamMode) -> None:
        """Leave the robot standing on the planner, with no operator-driven upper body.

        The mode command alone is not enough: from a planner mode it changes nothing, and
        the robot's planner timeout keeps VR 3-point control (reference behaviour), so a
        VR_3PT engagement would keep tracking the last wrist targets. One IDLE planner message
        without the optional fields clears the VR control (g1 encoder again) and the
        frozen upper body; the planner timeout then holds IDLE.
        """
        self._socket.send(build_command_message(start=False, stop=False, planner=True))
        if engaged_mode not in PLANNER_MODES:
            # Coming from the pose stream: the robot re-captures its heading on the switch,
            # so the planner's facing restarts at forward, as on a normal POSE -> PLANNER.
            self._planner_streamer.reset_yaw()
        self._planner_streamer.send_idle()
        self._final_command_queued = True
        self._robot_kept_heading = True
        print("[sonic operator] robot handed to the planner (it idles in place)")

    def on_interrupt(self) -> None:
        print("\n[sonic operator] interrupted")

    def teardown(self) -> None:
        if self._socket is not None:
            # Leaving while engaged (Ctrl-C, an exception in step) without the stop
            # combo (which leaves the machine OFF): don't leave the robot holding the
            # last streamed pose.
            if self._sm is not None and self._sm.mode != StreamMode.OFF:
                self._release_robot_to_planner(self._sm.mode)
            # Release the command port first: the viz window and the FK scene can
            # take a while to tear down, and an immediate operator relaunch must not
            # find :5556 still bound to this dying process. linger=0 discards
            # whatever ZMQ has not yet written, though, and the last command is sent
            # microseconds before this — so after one, linger and term() (which
            # blocks until the queue is written or the linger expires) so the robot
            # actually receives it.
            self._socket.close(linger=_FINAL_COMMAND_LINGER_MS if self._final_command_queued else 0)
            self._context.term()
        if self._reader is not None:
            self._reader.stop()
        if self._feedback is not None:
            self._feedback.close()
        if self._reset_watcher is not None:
            self._reset_watcher.close()
        if self._bridge_proc is not None and self._bridge_proc.poll() is None:
            self._bridge_proc.terminate()
            try:
                self._bridge_proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._bridge_proc.kill()
        if self._viz is not None:
            self._viz.close()
        if self._fk is not None:
            self._fk.close()
        print("[sonic operator] shutdown complete")


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.vis_smpl and not args.vis_vr3pt:
        raise SystemExit("--vis-smpl requires --vis-vr3pt (they share one window)")
    # Every argparse dest is a configurable parameter of the operator, so the
    # namespace maps straight onto TeleopOptions via configure().
    options = SonicTeleopOperator.configure(**vars(args))
    SonicTeleopOperator(options).run()


if __name__ == "__main__":
    main()
