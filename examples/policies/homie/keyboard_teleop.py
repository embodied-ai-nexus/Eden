"""Keyboard teleoperation for the HOMIE walk/stand tracker (sim/viser + real G1).

---
tier: policy
capabilities: [task.locomotion, robot.g1, teleop.keyboard, deploy.real_robot]
hardware: cpu
optional_deps: []
run: uv run python examples/policies/homie/keyboard_teleop.py --dummy --steps 50
---

HOMIE consumes a 4-D command ``[vel_x, vel_y, ang_vel_z, height_delta]``; this
example sets it live from the keyboard instead of a constant CLI value. Two
modes share one control scheme:

* **Sim (default)** — rolls the policy out in Genesis and streams the scene to
  a ViserViewer web client (``http://localhost:8080``) with a read-only command
  readout. Practice here before touching hardware.
* **Deploy (``--deploy``)** — drives a real Unitree G1 through
  ``UnitreeG1Deployment``. The deployment keeps its INIT/START/STOP/KILL state
  machine (G1 remote or ``s``/``x``/``i``/``k`` + Enter); this script only
  feeds the locomotion command, and policy actions are ignored until the
  operator presses START.

Key layout (chosen to avoid the deployment's stdin command letters
``s/x/i/k/n/1-9`` — typing a state-machine command never injects a velocity)::

          DRIVE                 TURN          HEIGHT
        ^  forward            q  yaw left     c  rise
   <  strafe L   >  strafe R  e  yaw right    z  crouch
        v  backward

        SPACE = instant zero command (stand)

Velocity keys are dead-man style: release and the command slews back to zero.
Height is latched: it integrates while ``z``/``c`` is held and stays where you
leave it (delta about the 0.74 m default, clamped to [-0.5, 0]). Velocity
targets are slew-limited (``--accel``) and snapped to exactly zero near zero so
the tracker's walk/stand switch does not chatter.

The keyboard listener (pynput) needs a display. Without one (e.g. a headless
cluster node over SSH), sim mode falls back to **browser controls**: a "Drive
(browser)" panel with vel/height sliders and a STOP button appears in the
viser GUI, fed through the same slew filter — forward the port
(``ssh -L 8080:localhost:8080``) and drive from the web client. Deploy mode
has no such fallback (run it from a machine with a display, near the robot);
there pynput captures keys globally regardless of window focus — stray typing
anywhere moves the robot; SPACE zero-stops.

Usage::

    python examples/policies/homie/keyboard_teleop.py --dummy --steps 50   # plumbing smoke test
    python examples/policies/homie/keyboard_teleop.py --walk-model W --stand-model S
    python examples/policies/homie/keyboard_teleop.py --deploy --deploy-kp-kd-scale 0.2 --vx-max 0.3
"""

from __future__ import annotations

import argparse
import os
import tempfile

import torch

import eden as en
import genesis as gs
from eden.envs.base import RLEnvBase
from eden.extensions.policies.homie.modules import HOMIE_INIT_DICT, HIMActorCritic
from eden.extensions.policies.homie.policy import HomiePolicy
from eden.options import (
    ActionManagerOptions,
    EnvOptions,
    ObservationManagerOptions,
)
from eden.utils.devices.keyboard import Keyboard, KeyBinding


class HomieKeyboard(Keyboard):
    """Four-axis keyboard for the HOMIE command: arrows + q/e (yaw) + z/c (height).

    Axes are derived from ``active_keys`` membership rather than the base
    class's accumulated readings — OS key auto-repeat fires repeated press
    events at a machine-dependent rate, and a release force-zeros an axis even
    if the opposite key is still held. Membership is idempotent under repeats
    and opposite keys simply cancel.

    SPACE requests an instant zero command. The callback runs on the pynput
    thread *while the device lock is held*, so it must not call ``reset()``
    (same non-reentrant lock — deadlock); it only sets a flag that the control
    loop drains via :meth:`consume_zero_stop`.
    """

    def __init__(self) -> None:
        from pynput.keyboard import Key

        super().__init__(
            key_bindings={
                Key.up: KeyBinding(0, +1),
                Key.down: KeyBinding(0, -1),
                Key.left: KeyBinding(1, +1),  # G1 +y is left
                Key.right: KeyBinding(1, -1),
                "q": KeyBinding(2, +1),  # +wz is CCW = turn left
                "e": KeyBinding(2, -1),
                "z": KeyBinding(3, -1),  # crouch
                "c": KeyBinding(3, +1),  # rise
            }
        )
        self._zero_stop = False
        self.add_callback(Key.space, self._on_space)

    def _on_space(self, on_press: bool) -> None:
        if on_press:
            self._zero_stop = True

    def consume_zero_stop(self) -> bool:
        """Return and clear the pending SPACE zero-stop request."""
        pending = self._zero_stop
        self._zero_stop = False
        return pending

    def get_axes(self) -> torch.Tensor:
        """Current digital axes ``[vx, vy, wz, height]`` in {-1, 0, +1}."""
        axes = torch.zeros(4, dtype=torch.float32)
        with self.lock:
            for key, binding in self.key_bindings.items():
                if key in self.active_keys:
                    axes[binding.index] += binding.value
        return axes.clamp_(-1.0, 1.0)


class CommandFilter:
    """Turn digital axes into a smooth, bounded 4-D HOMIE command.

    Velocities slew toward ``axes * limits`` at ``accel`` per second and snap
    to exactly zero once the target is zero and the value is small — hovering
    near the tracker's 0.1 command-norm walk/stand threshold would chatter
    between the two models. Height integrates at ``height_rate`` while a key is
    held (latched) and is clamped to HOMIE's reachable delta range. A zero-stop
    bypasses the slew entirely.
    """

    _SNAP_EPS = 0.05
    _HEIGHT_DELTA_RANGE = (-0.5, 0.0)  # about HOMIE_DEFAULT_HEIGHT (0.74 m)

    def __init__(self, limits: tuple[float, float, float], accel: float, height_rate: float, dt: float) -> None:
        self.limits = torch.tensor(limits, dtype=torch.float32)
        self.accel = accel
        self.height_rate = height_rate
        self.dt = dt
        self.cmd = torch.zeros(4, dtype=torch.float32)

    def zero(self) -> None:
        self.cmd.zero_()

    def _slew_vel(self, target: torch.Tensor) -> None:
        step = self.accel * self.dt
        self.cmd[:3] += (target - self.cmd[:3]).clamp_(-step, step)
        snap = (target == 0.0) & (self.cmd[:3].abs() < self._SNAP_EPS)
        self.cmd[:3] = torch.where(snap, torch.zeros(3), self.cmd[:3])

    def update(self, axes: torch.Tensor, zero_stop: bool) -> torch.Tensor:
        """Keyboard mode: velocities from digital axes, height rate-integrated while held."""
        if zero_stop:
            self.cmd.zero_()
            return self.cmd.clone()
        self._slew_vel(axes[:3] * self.limits)
        self.cmd[3] = (self.cmd[3] + axes[3] * self.height_rate * self.dt).clamp(*self._HEIGHT_DELTA_RANGE)
        return self.cmd.clone()

    def update_targets(self, vel_target: torch.Tensor, height_target: float, zero_stop: bool) -> torch.Tensor:
        """Slider mode: absolute targets, slewed at the same rates as keyboard mode."""
        if zero_stop:
            self.cmd.zero_()
            return self.cmd.clone()
        self._slew_vel(vel_target.clamp(-self.limits, self.limits))
        height_target = min(max(height_target, self._HEIGHT_DELTA_RANGE[0]), self._HEIGHT_DELTA_RANGE[1])
        height_step = self.height_rate * self.dt
        self.cmd[3] += min(max(height_target - float(self.cmd[3]), -height_step), height_step)
        return self.cmd.clone()


_KEY_GUIDE = """\
Teleop keys (global — work regardless of window focus):
  Up/Down      forward / backward        q / e   yaw left / right
  Left/Right   strafe left / right       z / c   crouch / rise (latched)
  SPACE        instant zero command (stand)"""


def _make_keyboard() -> HomieKeyboard | None:
    """Build the teleop keyboard, or return None if unavailable (callers pick the fallback)."""
    if not os.environ.get("DISPLAY") and os.name == "posix":
        print("[teleop] pynput keyboard unavailable (no display)")
        return None
    try:
        kb = HomieKeyboard()
    except Exception as exc:  # pynput missing or listener failed to start
        print(f"[teleop] pynput keyboard unavailable ({exc})")
        return None
    print(_KEY_GUIDE)
    return kb


def _make_dummy_checkpoint(path: str, seed: int) -> None:
    """Write a random-weight HIMActorCritic state dict to ``path``.

    Used by ``--dummy`` to exercise the load + inference path without the
    released checkpoints. The output does not walk — it is for plumbing only.
    """
    torch.manual_seed(seed)
    model = HIMActorCritic(**HOMIE_INIT_DICT)
    torch.save(model.state_dict(), path)


def _resolve_checkpoints(args: argparse.Namespace) -> tempfile.TemporaryDirectory | None:
    """Apply --dummy / --walk-model / --stand-model overrides (mirrors homie/main.py)."""
    if args.dummy:
        tmpdir = tempfile.TemporaryDirectory()
        walk_path = os.path.join(tmpdir.name, "model_walk.pt")
        stand_path = os.path.join(tmpdir.name, "model_stand.pt")
        _make_dummy_checkpoint(walk_path, seed=0)
        _make_dummy_checkpoint(stand_path, seed=1)
        HomiePolicy.walk_model_path = walk_path
        HomiePolicy.stand_model_path = stand_path
        HomiePolicy.walk_md5_hash = HomiePolicy.stand_md5_hash = None  # custom weights
        print(f"[dummy] random walk/stand checkpoints in {tmpdir.name} (plumbing only — will not walk)")
        return tmpdir
    if args.walk_model is not None:
        HomiePolicy.walk_model_path = args.walk_model
        HomiePolicy.walk_md5_hash = None  # custom checkpoint, drop the pinned hash
    if args.stand_model is not None:
        HomiePolicy.stand_model_path = args.stand_model
        HomiePolicy.stand_md5_hash = None
    return None


def _build_env(args: argparse.Namespace, show_viewer: bool) -> RLEnvBase:
    env = RLEnvBase(
        env_options=EnvOptions(
            num_envs=1,
            episode_length_s=1e9,
            **HomiePolicy.env_options(),
        ),
        scene_options=en.scenes.Empty(robot=HomiePolicy.robot_options()),
        action_options=ActionManagerOptions(**HomiePolicy.action_options()),
        observation_options=ObservationManagerOptions(**HomiePolicy.observation_options()),
        show_viewer=show_viewer,
        eval_mode=True,
    )
    env.build()
    return env


def _make_filter(args: argparse.Namespace) -> CommandFilter:
    control_dt = HomiePolicy.env_options()["sim_dt"] * HomiePolicy.env_options()["decimation"]
    return CommandFilter(
        limits=(args.vx_max, args.vy_max, args.wz_max),
        accel=args.accel,
        height_rate=args.height_rate,
        dt=control_dt,
    )


def main(args: argparse.Namespace) -> None:
    """Sim mode: HOMIE in Genesis, ViserViewer streaming, keyboard command."""
    from eden.extensions.deployment.utils.rate import RateLimiter
    from eden.extensions.visualization.viser import ViserViewer

    env = _build_env(args, show_viewer=args.vis)
    keyboard = _make_keyboard()

    vis = ViserViewer(env, port=args.port)
    vis.build()
    # Read-only live readout of the active (filtered) command in the web GUI.
    with vis.server.gui.add_folder("HOMIE command"):
        readout = {
            name: vis.server.gui.add_number(name, initial_value=0.0, step=0.01, disabled=True)
            for name in ("vel_x", "vel_y", "ang_vel_z", "height_delta")
        }

    # Browser fallback: with no keyboard (headless node over SSH), drive from
    # viser sliders instead. Slider values are targets fed through the same
    # CommandFilter slew, so the safety behavior matches keyboard mode.
    sliders = None
    stop_requested = {"flag": False}
    if keyboard is None:
        with vis.server.gui.add_folder("Drive (browser)"):
            sliders = {
                "vel_x": vis.server.gui.add_slider(
                    "vel_x [m/s]", min=-args.vx_max, max=args.vx_max, step=0.05, initial_value=0.0
                ),
                "vel_y": vis.server.gui.add_slider(
                    "vel_y [m/s]", min=-args.vy_max, max=args.vy_max, step=0.05, initial_value=0.0
                ),
                "ang_vel_z": vis.server.gui.add_slider(
                    "ang_vel_z [rad/s]", min=-args.wz_max, max=args.wz_max, step=0.05, initial_value=0.0
                ),
                "height_delta": vis.server.gui.add_slider(
                    "height_delta [m]", min=-0.5, max=0.0, step=0.01, initial_value=0.0
                ),
            }
            stop_button = vis.server.gui.add_button("STOP — zero command")

            @stop_button.on_click
            def _(_) -> None:
                stop_requested["flag"] = True

        print("[teleop] no keyboard — drive with the sliders in the viser GUI ('Drive (browser)' panel)")

    def _zero_sliders() -> None:
        if sliders is not None:
            for handle in sliders.values():
                handle.value = 0.0

    policy = HomiePolicy(env=env, options=HomiePolicy.configure())
    obs, _ = env.reset()
    policy.reset()

    filt = _make_filter(args)
    rate = RateLimiter(1.0 / filt.dt)

    print(f"[teleop] viser at http://localhost:{args.port} — Ctrl-C to exit")

    prev_axes = torch.zeros(4)
    step = 0
    try:
        while args.steps is None or step < args.steps:
            if keyboard is not None:
                axes = keyboard.get_axes()
                zero_stop = keyboard.consume_zero_stop()
                if zero_stop:
                    keyboard.reset()
                cmd = filt.update(axes, zero_stop)
            elif sliders is not None:
                zero_stop = stop_requested["flag"]
                stop_requested["flag"] = False
                if zero_stop:
                    _zero_sliders()
                vel_target = torch.tensor(
                    [sliders["vel_x"].value, sliders["vel_y"].value, sliders["ang_vel_z"].value],
                    dtype=torch.float32,
                )
                cmd = filt.update_targets(vel_target, sliders["height_delta"].value, zero_stop)
            else:
                cmd = filt.update(torch.zeros(4), False)
            policy.set_command(cmd)

            action = policy.step(obs)
            obs, _, _, _, _ = env.step(action)

            gui_info = vis.update()
            if gui_info["reset"]:
                # The web Reset button only raises this flag; do the reset here.
                env.reset()
                policy.reset()
                filt.zero()
                if keyboard:
                    keyboard.reset()
                _zero_sliders()
                print("[teleop] env reset — command zeroed")

            if keyboard is not None and not torch.equal(axes, prev_axes):
                print(f"[teleop] axes {axes.tolist()} -> cmd {[f'{v:+.2f}' for v in filt.cmd.tolist()]}")
                prev_axes = axes
            if step % 5 == 0:
                for i, handle in enumerate(readout.values()):
                    handle.value = round(float(filt.cmd[i]), 3)

            rate.sleep()
            step += 1
    except KeyboardInterrupt:
        print(f"\n[teleop] stopped after {step} steps")

    base = env.entities["robot"].get_pos()[0]
    print(f"[teleop] final base pos: x={base[0]:+.3f} y={base[1]:+.3f} z={base[2]:.3f}m")


def deploy(args: argparse.Namespace) -> None:
    """Deploy mode: HOMIE on a real Unitree G1, keyboard command.

    The deployment owns the safety state machine — the robot boots in
    STOP/hold and ignores policy actions until the operator presses START
    (remote A, or ``s`` + Enter). On every START the policy history and the
    keyboard command are re-armed because the deployment re-captures its yaw
    offset. Ctrl-C (or KILL) triggers the soft damping shutdown.
    """
    from eden.extensions.deployment.unitree_g1 import UnitreeG1Deployment
    from eden.extensions.deployment.utils.republisher import publish_rate_configure_kwargs

    env = _build_env(args, show_viewer=False)

    # Construct the policy before the deployment swaps the sim entity for the
    # RobotStateEntity: __init__ resolves DOF indices against the sim robot,
    # while step() re-looks-up env.entities["robot"] and reads the real state.
    policy = HomiePolicy(env=env, options=HomiePolicy.configure())

    dep = UnitreeG1Deployment(
        env,
        UnitreeG1Deployment.configure(
            entity_name="robot",
            sync=True,
            connect_timeout_s=10.0,
            deploy_kp_kd_scale=args.deploy_kp_kd_scale,
            auto_yaw_align=True,
            **publish_rate_configure_kwargs(args.publish_rate),
        ),
    )
    dep.connect()
    obs = dep.reset()

    # Start the pynput listener only after dep.reset() printed its own guide.
    keyboard = _make_keyboard()
    if keyboard is None:
        print("[deploy] no keyboard — locomotion command stays zero (robot stands); run from a machine with a display")
    filt = _make_filter(args)

    print("=" * 60)
    print("State machine: remote A=START B=STOP Y=INIT L1+R1=KILL,")
    print("or type s / x / i / k + Enter.")
    print("NOTE: arrow keys leak escape codes into the terminal line buffer.")
    print("Before typing a state command: SPACE (zero-stop), press Enter once")
    print("to flush the dirty line, then type the command on a clean line.")
    print("The G1 remote is the primary safety channel.")
    print("=" * 60)

    step = 0
    try:
        while args.steps is None or step < args.steps:
            if dep.consume_start_events():
                # START re-captures the yaw offset; stale tracker history and
                # any held command reference the pre-START frame.
                policy.reset()
                filt.zero()
                if keyboard:
                    keyboard.reset()
                print(f"[deploy] START at step {step} — HOMIE re-armed, command zeroed")

            axes = keyboard.get_axes() if keyboard else torch.zeros(4)
            zero_stop = keyboard.consume_zero_stop() if keyboard else False
            if zero_stop and keyboard:
                keyboard.reset()
            cmd = filt.update(axes, zero_stop)
            policy.set_command(cmd)

            with torch.no_grad():
                action = policy.step(obs)
            # Paced by the deployment's RateLimiter; the action is ignored
            # until the operator has pressed START.
            obs = dep.step(action)
            step += 1
    except KeyboardInterrupt:
        print(f"\n[deploy] stopped after {step} steps")
    finally:
        dep.close()  # soft shutdown: kp -> 0 damping ramp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deploy", action="store_true", help="Run on a real Unitree G1 instead of sim.")
    parser.add_argument(
        "--deploy-kp-kd-scale",
        type=float,
        default=1.0,
        help="Scale applied to kp/kd before sending to the G1 SDK. Try 0.2 / 0.5 / 0.8 first.",
    )
    parser.add_argument(
        "--publish-rate",
        type=float,
        default=None,
        help="Hz at which the held wire command is re-sent between control ticks (default: the "
        "backend's 500 Hz, Unitree's own low-level rate). The G1's parallel ankle/waist joints "
        "close their PD once per received packet. Pass 0 for one packet per control tick -- the "
        "A/B against that default.",
    )
    parser.add_argument("--walk-model", default=None, help="Path to model_walk.pt")
    parser.add_argument("--stand-model", default=None, help="Path to model_stand.pt")
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Generate random-weight walk/stand checkpoints (plumbing smoke test only, sim only).",
    )
    parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU backend")
    parser.add_argument("-v", "--vis", action="store_true", help="Show the Genesis viewer (sim only)")
    parser.add_argument("--port", type=int, default=8080, help="Viser web server port (sim only)")
    parser.add_argument(
        "-s", "--steps", type=int, default=None, help="Stop after N control steps (default: run forever)"
    )
    parser.add_argument(
        "--vx-max", type=float, default=0.6, help="Max forward/backward velocity [m/s] (HOMIE clip 0.8)"
    )
    parser.add_argument("--vy-max", type=float, default=0.4, help="Max lateral velocity [m/s] (HOMIE clip 0.5)")
    parser.add_argument("--wz-max", type=float, default=0.6, help="Max yaw rate [rad/s] (HOMIE clip 0.8)")
    parser.add_argument("--accel", type=float, default=1.5, help="Velocity slew rate [unit/s] (0 -> vx_max in ~0.4 s)")
    parser.add_argument("--height-rate", type=float, default=0.15, help="Height change rate while z/c held [m/s]")
    args = parser.parse_args()
    if args.dummy and args.deploy:
        parser.error("--dummy generates random weights and must not be deployed to hardware.")
    return args


if __name__ == "__main__":
    args = parse_args()
    en.init(backend=gs.cpu if args.cpu else gs.gpu)
    tmpdir = _resolve_checkpoints(args)
    try:
        if args.deploy:
            deploy(args)
        else:
            main(args)
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()
