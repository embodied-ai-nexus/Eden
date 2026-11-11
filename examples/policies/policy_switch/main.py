"""Hot-swap between the Holosoma loco policy and the SONIC tracking policy on one G1.

---
tier: policy
capabilities: [task.locomotion, task.mimic, robot.g1, deploy.real_robot]
hardware: cpu
optional_deps: [onnx]
run: uv run python examples/policies/policy_switch/main.py --vis
---

Demonstrates runtime policy switching (the sim-side counterpart of
``eden deploy``'s multi-checkpoint switching): both ONNX policies are constructed
up front against a single shared environment, and a scripted schedule swaps the
active one mid-run — Holosoma walks the robot around, SONIC takes over in place
to track the reference motion, then Holosoma resumes.

Why one env can serve both policies:

* Both emit absolute joint-position targets through an identical
  ``dofs_pos_controller`` action term (``ImplicitPDController``, 29 DOFs,
  ``reference_source=ZERO``, ``scale=1.0``), so a single action manager works.
* Both run at 50 Hz control and read raw simulator state directly in
  ``step()`` (the env observation group is SONIC's build-time placeholder).
* The env uses SONIC's ``env_options`` (sim_dt=0.01, substeps=2, decimation=2);
  Holosoma trained at sim_dt=0.005/substeps=1/decimation=4 — the same 200 Hz
  substep rate and 50 Hz control cadence, so it tolerates the swap.

Switching mechanics (the part worth copying):

* While Holosoma drives, the ``MotionCommand`` clock is pinned to frame 0 every
  step. Left free, ``time_steps`` would eventually wrap past the motion end and
  ``_resample_command`` would *teleport* the robot mid-walk.
* Pinning also keeps the command's anchor-relative reference tensors tracking
  the robot's current pose, so when SONIC takes over the reference is "motion
  start, relative to wherever the robot is now".
* On switch, the incoming policy's ``reset()`` clears its internal state
  (gait-phase clock / history ring buffers); SONIC then re-captures its yaw
  heading offset on the first step, aligning the motion to the robot's current
  heading. No env reset, no teleport.

Like ``eden deploy``'s switching, this is a hot swap: the incoming policy sees a
one-control-step transient at the handover.

Deploy mode (``--deploy``) runs the same pair on a real Unitree G1 and switches
on **controller input**, using the deployment's bindings: remote **Select**
(ignored while L1 is held) or keyboard ``n`` / ``1``-``2`` + Enter. When the
deployment has ``consume_policy_switches()`` it is used directly; otherwise
the example parses the remote's Select button from the
low-state ``wireless_remote`` buffer itself.

Usage::

    python examples/policies/policy_switch/main.py --vis
    python examples/policies/policy_switch/main.py --record --cycles 2
    python examples/policies/policy_switch/main.py --holosoma-steps 30 --sonic-steps 30  # smoke test
    python examples/policies/policy_switch/main.py --deploy --vx 0.3                     # real G1
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import threading

import torch
from tqdm import tqdm

import eden as en
import genesis as gs
from eden.envs.base import RLEnvBase
from eden.extensions.policies.holosoma.policy import HolosomaLocomotionPolicy
from eden.extensions.policies.sonic.policy import SonicPolicy
from eden.options import (
    ActionManagerOptions,
    CommandManagerOptions,
    EnvOptions,
    ObservationManagerOptions,
)
from eden.options.camera import CameraOptions, CamerasOptions

_DEFAULT_MOTION = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "projects",
    "mimic",
    "beyondmimic",
    "sample",
    "G1_gangnam_style_V01_bvh_60hz.npz",
)

# Policy slot order: slot 1 = holosoma, slot 2 = sonic (keyboard "1"/"2").
_POLICY_ORDER = ("holosoma", "sonic")


class _RemoteSelectMonitor:
    """Rising-edge detector for the G1 remote's Select button.

    Self-contained stand-in for ``DeploymentBase.consume_policy_switches()``, for
    a deployment that lacks it. The 40-byte ``wireless_remote`` buffer mirrored into every
    low-state message packs the button ``keys`` as a little-endian uint16 at
    offset 2 (Unitree layout, same as unitree_rl_gym). Select with L1 held is
    ignored, matching the deployment's chord suppression.
    """

    _SELECT = 1 << 3
    _L1 = 1 << 1

    def __init__(self, dep) -> None:
        self._dep = dep
        self._prev_keys = 0

    def pressed(self) -> bool:
        msg = getattr(self._dep, "_latest_msg", None)
        remote = getattr(msg, "wireless_remote", None) if msg is not None else None
        if remote is None:
            return False
        keys = struct.unpack_from("<H", bytes(remote), 2)[0]
        rising = keys & ~self._prev_keys
        self._prev_keys = keys
        return bool(rising & self._SELECT) and not (keys & self._L1)


def _start_stdin_reader(queue: list[str]) -> None:
    """Drain stdin lines into ``queue`` from a daemon thread (the deployment's keyboard path).

    Must be started only *after* ``dep.reset()`` returns — a deployment without the
    remote start/stop state machine used ``input()`` in its reset for its own "Press Enter" gate and a
    concurrent reader would steal it.
    """

    def _reader() -> None:
        for line in sys.stdin:
            queue.append(line.strip().lower())

    threading.Thread(target=_reader, daemon=True).start()


def _activate(name: str, holosoma, sonic, cmd, walk_command: torch.Tensor) -> None:
    """Per-policy handover: reset the incoming policy's state, rewind/pin the motion clock."""
    if name == "holosoma":
        holosoma.reset()
        holosoma.set_command(walk_command)
    else:
        cmd.time_steps.zero_()
        sonic.reset()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion-file",
        default=_DEFAULT_MOTION,
        help="Reference motion npz for the SONIC phases (MJCF DOF order).",
    )
    parser.add_argument("--holosoma-model", default=None, help="Override the Holosoma loco ONNX with a local path.")
    parser.add_argument("--encoder-path", default=None, help="Override the SONIC encoder ONNX with a local path.")
    parser.add_argument("--decoder-path", default=None, help="Override the SONIC decoder ONNX with a local path.")
    parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU backend")
    parser.add_argument("-v", "--vis", action="store_true", help="Show viewer")
    parser.add_argument("-r", "--record", action="store_true", help="Record video")
    parser.add_argument("--vx", type=float, default=0.0, help="Walk-phase forward velocity command [m/s]")
    parser.add_argument("--vy", type=float, default=0.0, help="Walk-phase lateral velocity command [m/s]")
    parser.add_argument("--wz", type=float, default=0.0, help="Walk-phase yaw-rate command [rad/s]")
    parser.add_argument("--holosoma-steps", type=int, default=300, help="Control steps per Holosoma walk phase")
    parser.add_argument("--sonic-steps", type=int, default=400, help="Control steps per SONIC tracking phase")
    parser.add_argument("--cycles", type=int, default=2, help="Number of walk→track cycles")
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Run on a real Unitree G1; switch policies with remote Select or keyboard n / 1-2 + Enter.",
    )
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
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=None,
        help="Deploy only: stop after N control steps (default: run until Ctrl-C).",
    )
    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu)

    if args.deploy:
        deploy(args)
        return

    if args.holosoma_model is not None:
        HolosomaLocomotionPolicy.use_local_checkpoint(args.holosoma_model)
    if args.encoder_path is not None:
        SonicPolicy.use_local_checkpoint(args.encoder_path)
    if args.decoder_path is not None:
        SonicPolicy.use_local_checkpoint(args.decoder_path, checkpoint_attr="decoder_checkpoint_path")

    # SONIC's robot: the crouch default pose Holosoma also pins (the two constants are
    # identical) plus SONIC's 7520-22 hip-pitch gains. One robot means one PD gain set for
    # both policies here; this example does not switch gains with the policy. SONIC is the
    # gain-sensitive one, so it wins; Holosoma's hips run 2.47x stiffer than trained.
    # `eden deploy` does switch them: each slot runs at its own policy's gains, ramped on a
    # switch. Run either policy alone, or under `eden deploy`, to get its own gains.
    scene_options = en.scenes.Empty(robot=SonicPolicy.robot_options())

    env = RLEnvBase(
        env_options=EnvOptions(
            num_envs=1,
            episode_length_s=1e9,
            **SonicPolicy.env_options(),
        ),
        scene_options=scene_options,
        command_options=CommandManagerOptions(
            **SonicPolicy.command_options(
                motion_file=args.motion_file,
                sampling_mode="start",
            )
        ),
        # Identical term in both policies' action_options — see module docstring.
        action_options=ActionManagerOptions(**SonicPolicy.action_options()),
        observation_options=ObservationManagerOptions(**SonicPolicy.observation_options()),
        cameras_options=(
            CamerasOptions(
                rec=CameraOptions(
                    cam_pos=(3.0, 0.0, 1.6),
                    cam_lookat=(0.0, 0.0, 0.9),
                    cam_fov=50.0,
                    follow_entity_name="robot",
                )
            )
            if args.record
            else None
        ),
        show_viewer=args.vis,
        eval_mode=True,
    )
    env.build()

    # Both policies are loaded up front and stay resident; switching only
    # changes which one consumes the step (``eden deploy``'s multi-checkpoint pattern).
    holosoma = HolosomaLocomotionPolicy(env=env, options=HolosomaLocomotionPolicy.configure())
    sonic = SonicPolicy(env=env, options=SonicPolicy.configure())
    policies = {"holosoma": holosoma, "sonic": sonic}

    cmd = env.command_manager.get_term("motion")
    robot = env.entities["robot"]

    walk_command = torch.tensor([args.vx, args.vy, args.wz])

    def activate(name: str) -> None:
        _activate(name, holosoma, sonic, cmd, walk_command)

    sonic_steps = min(args.sonic_steps, int(cmd.motion.time_step_total) - 1)
    schedule: list[tuple[str, int]] = []
    for _ in range(args.cycles):
        schedule.append(("holosoma", args.holosoma_steps))
        schedule.append(("sonic", sonic_steps))
    schedule.append(("holosoma", args.holosoma_steps))

    obs, _ = env.reset()

    if args.record:
        out_path = os.path.join(os.path.dirname(__file__), "policy_switch_run.mp4")
        env.cameras["rec"].start_recording(out_path)

    for phase_idx, (name, n_steps) in enumerate(schedule):
        activate(name)
        base = robot.get_pos()[0]
        print(f"[phase {phase_idx}] -> {name:8s} ({n_steps} steps)  base x={base[0]:+.2f} y={base[1]:+.2f}")
        policy = policies[name]
        for _ in tqdm(range(n_steps), desc=name):
            action = policy.step(obs)
            obs, _, _, _, _ = env.step(action)
            if name == "holosoma":
                # Pin the reference at motion start: prevents the wrap-around
                # teleport and keeps the handover reference current (see
                # module docstring).
                cmd.time_steps.zero_()
            if args.record:
                env.cameras["rec"].render_rgb()

    base = robot.get_pos()[0]
    print(f"\nFinal base pos: x={base[0]:+.3f} y={base[1]:+.3f} z={base[2]:.3f}m")

    if args.record:
        env.cameras["rec"].stop_recording()
        print(f"Recording written to {out_path}")


def deploy(args: argparse.Namespace) -> None:
    """Run the SONIC / Holosoma pair on a real Unitree G1, switching on controller input.

    Controls (the deployment's own bindings):

    ========================  =======================================
    remote Select             switch to the next policy (L1+Select is
                              ignored, as the deployment does)
    keyboard ``n`` + Enter    switch to the next policy
    keyboard ``1``/``2`` + Enter  switch to holosoma / sonic
    Ctrl-C                    exit
    ========================  =======================================

    If the deployment exposes ``consume_policy_switches()`` it is used
    as the switch source; otherwise the remote is parsed directly from the
    low-state ``wireless_remote`` buffer (works on today's main).

    Notes
    -----
    - Starts in the Holosoma walk policy with the ``--vx/--vy/--wz`` command
      held constant; pass ``--vx 0`` to start standing.
    - Under the G1 deployment the robot boots in STOP/hold and ignores policy
      actions until the operator presses INIT then START (see the guide the
      deployment prints); the switch bindings select which policy takes over.
    - Same handover mechanics as sim: the motion clock is pinned to frame 0
      while Holosoma drives (also keeps the SONIC handover reference current),
      and the incoming policy is ``reset()`` on switch. During SONIC phases
      the motion loops; the wrap re-captures SONIC's heading offset.
    - Both policies are constructed *before* the deployment swap so their
      ``__init__`` can resolve dof / link indices against the sim entity.
    """
    from eden.extensions.deployment.unitree_g1 import UnitreeG1Deployment
    from eden.extensions.deployment.utils.republisher import publish_rate_configure_kwargs

    if args.holosoma_model is not None:
        HolosomaLocomotionPolicy.use_local_checkpoint(args.holosoma_model)
    if args.encoder_path is not None:
        SonicPolicy.use_local_checkpoint(args.encoder_path)
    if args.decoder_path is not None:
        SonicPolicy.use_local_checkpoint(args.decoder_path, checkpoint_attr="decoder_checkpoint_path")

    env = RLEnvBase(
        env_options=EnvOptions(
            num_envs=1,
            episode_length_s=1e9,
            **SonicPolicy.env_options(),
        ),
        # SONIC's robot, for the same shared-gain reason as the sim demo above.
        scene_options=en.scenes.Empty(robot=SonicPolicy.robot_options()),
        command_options=CommandManagerOptions(
            **SonicPolicy.command_options(
                motion_file=args.motion_file,
                sampling_mode="start",
            )
        ),
        action_options=ActionManagerOptions(**SonicPolicy.action_options()),
        observation_options=ObservationManagerOptions(**SonicPolicy.observation_options()),
        show_viewer=False,
        eval_mode=True,
    )
    env.build()

    holosoma = HolosomaLocomotionPolicy(env=env, options=HolosomaLocomotionPolicy.configure())
    sonic = SonicPolicy(env=env, options=SonicPolicy.configure())
    policies = {"holosoma": holosoma, "sonic": sonic}
    cmd = env.command_manager.get_term("motion")
    walk_command = torch.tensor([args.vx, args.vy, args.wz])

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

    # Switch sources. A deployment with consume_policy_switches owns all input (its own
    # remote state machine + keyboard reader feeding consume_policy_switches);
    # starting our own stdin reader there would steal its keyboard lines. Without
    # it, parse the remote ourselves and read stdin directly —
    # started only after dep.reset() (its "Press Enter" gate also reads stdin).
    consume = getattr(dep, "consume_policy_switches", None)
    consume_starts = getattr(dep, "consume_start_events", None)
    remote = None
    stdin_queue: list[str] = []
    if consume is None:
        remote = _RemoteSelectMonitor(dep)
        _start_stdin_reader(stdin_queue)

    active_idx = 0
    _activate(_POLICY_ORDER[active_idx], holosoma, sonic, cmd, walk_command)

    print("=" * 60)
    print("Policy switching:")
    print("  remote Select / n+Enter  switch to next policy")
    print(f"  1/2 + Enter              switch to {' / '.join(_POLICY_ORDER)}")
    print("  Ctrl-C                   exit")
    print(f"Active: {_POLICY_ORDER[active_idx]}  (walk command {walk_command.tolist()})")
    print("=" * 60)

    def _drain_switch_targets() -> list[int]:
        """Collect this step's requested slot indices from all input sources."""
        requests: list[int | str] = []
        if consume is not None:
            requests += consume()
        elif remote.pressed():
            requests.append("next")
        while stdin_queue:
            line = stdin_queue.pop(0)
            if line == "n":
                requests.append("next")
            elif line in ("1", "2"):
                requests.append(int(line) - 1)
        targets = []
        for req in requests:
            target = (active_idx + 1) % len(_POLICY_ORDER) if req == "next" else req
            if isinstance(target, int) and 0 <= target < len(_POLICY_ORDER):
                targets.append(target)
        return targets

    step = 0
    try:
        while args.steps is None or step < args.steps:
            # Re-arm the active policy at the START handover (START re-captures
            # the deployment's yaw offset; SONIC's heading state must follow).
            if consume_starts is not None and consume_starts():
                _activate(_POLICY_ORDER[active_idx], holosoma, sonic, cmd, walk_command)
                print(f"[step {step}] START — re-armed {_POLICY_ORDER[active_idx]}")
            for target in _drain_switch_targets():
                if target != active_idx:
                    active_idx = target
                    name = _POLICY_ORDER[active_idx]
                    _activate(name, holosoma, sonic, cmd, walk_command)
                    print(f"[step {step}] switched to {name}")
            with torch.no_grad():
                action = policies[_POLICY_ORDER[active_idx]].step(obs)
            obs = dep.step(action)
            if _POLICY_ORDER[active_idx] == "holosoma":
                cmd.time_steps.zero_()
            step += 1
    except KeyboardInterrupt:
        print(f"\nStopped after {step} steps.")
    finally:
        # Soft shutdown (damping ramp) before dropping the channel.
        dep.close()


if __name__ == "__main__":
    main()
