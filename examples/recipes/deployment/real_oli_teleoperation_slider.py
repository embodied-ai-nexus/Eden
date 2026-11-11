"""Teleoperate a real LimX Oli with GUI joint sliders in the interactive viewer.

---
tier: recipe
capabilities: [teleop.gui, robot.oli, deploy.real_robot]
hardware: real-robot
optional_deps: []
run: uv run python examples/recipes/deployment/real_oli_teleoperation_slider.py
---

Opens the Genesis viewer with an ImGui slider per joint and sends the same
joint-position command to the real robot (through ``LimxOliDeployment``, so
the target low-pass and the torque-feasibility clamp apply) and to the sim
robot (so the viewer previews the command; the sim is also resynced from
real state every cycle by the deployment's state wrapper).

The deployment's operator state machine gates when the sliders take effect
(the LimX remote is PS4-style: □ square, △ triangle, ✕ cross, ○ circle —
LimX's docs call the same buttons X/Y/A/B respectively):

    boot             STOP/hold — the robot holds its measured pose
    L1+△ / i+Enter   INIT — ramp to the default stance
    R1+□ / s+Enter   START — hand the slider-covered joints to the sliders
    L1+✕ / x+Enter   STOP — back to holding the measured pose
    L1+□ / k+Enter   KILL — soft-damp and exit

Until START, the sliders (and the targets of joints without a slider) are
continuously glued to the measured pose, so the handover is jump-free: the
first commanded target equals the pose the robot is actually in. Joints you
exclude from ``--joints`` stay held at their handover pose. The panel's
Reset button snaps the sliders back to the default stance.

Sliders are step inputs, so the wire target does not follow them directly:
it is interpolated toward the slider value at ``--max-joint-vel`` (the same
trick LimX's own stand controller uses, and what keeps the official motions
quiet). Sending the raw slider jumps instead excites the PD at the command
rate — audible ticking — and any target further than ``tau_limit/kp``
(~0.45 rad on waist/shoulders) from the measured pose pins the deployment's
torque-feasibility clamp at 95% of rated torque: loud, shaky, barely-moving.
The gains themselves are LimX's shipped walk-controller gains. Commands go
out at 200 Hz (LimX's own stack publishes at 1 kHz) while the sim + viewer
preview runs at 50 Hz. Error-level robot diagnostics are logged as they
change; pass ``--verbose-diagnostics`` to also see warning/OK transitions.

!!! danger
    This commands a real humanoid. Keep the robot supported (gantry) for
    first runs, start with a low ``--kp-kd-scale``, and remember the leg
    sliders move load-bearing joints. Closing the viewer (or KILL) soft-damps
    every motor — the robot will sag; support it or STOP first.

Hardware prep (LimX's procedure, both steps required): after power-on, click
the RIGHT JOYSTICK button on the remote to activate the motors, then R1+DOWN
for developer mode. Without developer mode the onboard controller fights the
SDK commands (buzzing, shaking); without motor activation the SDK commands
are ignored (robot stays limp). The robot should feel stiff — holding its
pose — from the moment this script boots into STOP/hold.

Usage:
    uv run examples/recipes/deployment/real_oli_teleoperation_slider.py
    ROBOT_IP=127.0.0.1 uv run examples/recipes/deployment/real_oli_teleoperation_slider.py  # LimX MuJoCo sim
    # arms + head only, soft gains, snappier slider tracking:
    uv run examples/recipes/deployment/real_oli_teleoperation_slider.py \
        --joints '*_shoulder_*' '*_elbow_*' '*_wrist_*' 'head_*' --kp-kd-scale 0.5 --max-joint-vel 1.0
"""

import fnmatch
import time
from argparse import ArgumentParser

import genesis as gs
import numpy as np
import torch

import eden as en
from eden.constants import ReferenceSource
from eden.envs.base import RLEnvBase
from eden.extensions.deployment.limx_oli import LimxOliDeployment
from eden.managers.terms.actions.joint_actions import ImplicitPDController
from eden.options import ActionManagerOptions, EnvOptions
from eden.options.robots.limx_oli import Oli
from eden.options.scenes.empty import EmptySceneOptions
from eden.utils.devices.gui_slider import SliderGUI

# LimX's official API example and all four shipped controllers publish RobotCmd
# at 1000 Hz; 200 Hz is what a Python loop with Genesis state syncs in it
# sustains, and it keeps the per-tick chase step (max_joint_vel / CONTROL_FREQ)
# small enough to be inaudible. The sim + viewer only need ~50 Hz.
CONTROL_FREQ = 200  # Hz - wire command rate (deployment rate limiter)
RENDER_EVERY = 4  # sim + viewer step every N control ticks (=> 50 Hz preview)


def _resolve_joints(patterns: list[str], dofs_name: list[str]) -> list[str]:
    """Expand fnmatch patterns into DOF names, preserving deployment DOF order."""
    matched: set[str] = set()
    for pattern in patterns:
        hits = [name for name in dofs_name if fnmatch.fnmatchcase(name, pattern)]
        if not hits:
            raise SystemExit(f"--joints pattern {pattern!r} matched no DOF. Robot DOFs: {dofs_name}")
        matched.update(hits)
    return [name for name in dofs_name if name in matched]


def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    parser.add_argument(
        "--joints",
        nargs="+",
        default=["*"],
        metavar="PATTERN",
        help="Joint name(s) or fnmatch patterns to expose as sliders (default: all 31). "
        "Joints not matched are held at their pose from the moment of START.",
    )
    parser.add_argument(
        "--kp-kd-scale",
        type=float,
        default=1.0,
        help="Scale on the default Kp/Kd sent to the robot. Ramp up from ~0.5 when first commissioning.",
    )
    parser.add_argument(
        "--max-joint-vel",
        type=float,
        default=0.5,
        help="Speed [rad/s] at which the commanded target chases the slider value. Keeps the "
        "position error (and thus the PD torque) small however fast the sliders are dragged; "
        "raise for snappier tracking once commissioned.",
    )
    parser.add_argument(
        "--target-lpf",
        type=float,
        default=0.5,
        help="First-order low-pass on commanded targets in (0, 1] (1.0 = off). Smooths the "
        "residual per-cycle target steps left after the --max-joint-vel interpolation.",
    )
    parser.add_argument(
        "--ip",
        type=str,
        default=None,
        help="Robot IP. Default resolves ROBOT_IP env var, then 10.192.1.2 (LimX MuJoCo sim: 127.0.0.1).",
    )
    parser.add_argument(
        "--verbose-diagnostics",
        action="store_true",
        help="Also log warning/OK robot diagnostics as they change (error-level faults are always logged).",
    )
    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu, debug=True)

    joints = _resolve_joints(args.joints, list(Oli().dofs_name))

    env = RLEnvBase(
        env_options=EnvOptions(
            num_envs=1,
            # env.step runs once per RENDER_EVERY control ticks, so this keeps
            # the sim preview's clock tracking the wall clock.
            sim_dt=RENDER_EVERY / CONTROL_FREQ,
            decimation=1,
            enable_default_keybinds=False,
            show_world_frame=True,
        ),
        scene_options=EmptySceneOptions(
            robot=Oli(),
        ),
        action_options=ActionManagerOptions(
            joint_sliders=ImplicitPDController.configure(
                entity_name="robot",
                dofs_name=joints,
                reference_source=ReferenceSource.ZERO,
                scale=1.0,
            ),
        ),
        # The viewer IS the teleop interface (ImGui sliders), so it is not optional.
        show_viewer=True,
    )
    env.build()

    # Slider limits come from the sim entity, grabbed before the deployment
    # swaps it for the RobotStateEntity wrapper.
    sim_robot = env.entities["robot"]
    _, dof_idx = sim_robot.find_named_dofs_idx_local(joints, preserve_order=True)
    lo, hi = sim_robot.get_dofs_limit(dof_idx)
    limits = torch.stack([lo, hi], dim=-1).squeeze(0).cpu()

    dep = LimxOliDeployment(
        env,
        LimxOliDeployment.configure(
            entity_name="robot",
            control_freq=CONTROL_FREQ,
            sync=True,
            connect_timeout_s=10.0,
            robot_ip=args.ip,
            deploy_kp_kd_scale=args.kp_kd_scale,
            target_lpf=args.target_lpf,
            verbose_diagnostics=args.verbose_diagnostics,
        ),
    )
    dep.connect()
    env.reset()
    # Boots the STOP/hold state machine and prints the control guide; unlike
    # the Franka backend this does not block on Enter — INIT/START are given
    # live via the LimX remote or the terminal keyboard.
    dep.reset()

    en.logger.info(env.action_manager.summary())
    en.logger.info(f"[teleop] Sliders for {len(joints)}/{dep.num_dofs} joints: {joints}")

    slider = SliderGUI(
        viewer=env.scene.viewer,
        dofs_name=joints,
        dofs_pos_limits=limits,
        # Panel Reset snaps the sliders to the default stance (the INIT target).
        initial_position=torch.from_numpy(dep.default_dof_pos[[dep.dofs_name.index(j) for j in joints]]).float(),
    )

    slider_idx = np.array([dep.dofs_name.index(j) for j in joints])
    # Wire target for the slider-covered joints. It chases the slider values at
    # --max-joint-vel instead of jumping to them: sliders are step inputs, and
    # commanding a raw jump at the official gains means a multi-Nm torque step
    # per cycle (and a pegged torque clamp for far targets) — the buzzy, shaky
    # motion this interpolation exists to avoid.
    target = None
    last_t = time.monotonic()
    tick = 0

    try:
        while env.scene.viewer.is_alive():
            now = time.monotonic()
            # Wall-clock dt keeps the chase speed constant under render-rate
            # jitter; the cap stops a stall from turning into a target jump.
            dt = min(now - last_t, 0.1)
            last_t = now

            # Until START hands control to the sliders (``_use_policy``), keep
            # the panel, the chase target, and the full direct-target buffer
            # glued to the measured pose: at the handover the first commanded
            # target equals the pose the robot is in (no jump), and non-slider
            # joints hold it after.
            if not dep._use_policy and dep.state is not None:
                measured = dep.state.dofs_pos
                slider.set_command(measured[slider_idx])
                dep.write_dofs_targets(torch.from_numpy(measured).float(), dep.dofs_name)
                target = measured[slider_idx].copy()

            command = slider.get_command().numpy().reshape(-1).astype(np.float64)
            if target is None:
                target = command.copy()
            step_limit = args.max_joint_vel * dt
            target += np.clip(command - target, -step_limit, step_limit)

            action = {"joint_sliders": torch.from_numpy(target).float().unsqueeze(0).to(env.device)}
            # Real robot, every tick at CONTROL_FREQ: the state machine sends
            # hold/init payloads until START, then the chased targets
            # (low-passed + torque-clamped).
            dep.step(action)
            # Sim, every RENDER_EVERY ticks: same command, so the viewer
            # previews it at ~50 Hz. The sim pose is resynced from real state
            # by the wrapper on every dep.step, so sim-side drift never
            # outlives a render cycle.
            if tick % RENDER_EVERY == 0:
                env.step(action)
            tick += 1
    except KeyboardInterrupt:
        en.logger.info("[teleop] KILL / Ctrl-C received, shutting down deployment.")
    finally:
        slider.close()
        dep.close()


if __name__ == "__main__":
    main()
