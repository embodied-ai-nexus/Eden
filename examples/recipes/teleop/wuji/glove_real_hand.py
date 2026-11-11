"""Teleoperate a real Wuji Hand 2 from a Wuji Glove.

---
tier: recipe
capabilities: [teleop.wuji_glove, retarget.hand, robot.wuji_hand2beta2, deploy.real_robot]
hardware: real-robot
optional_deps: [wuji]
run: WUJI_TELEMETRY_DISABLE=1 uv run python examples/recipes/teleop/wuji/glove_real_hand.py
---

``glove_teleop.py``'s retargeting loop with the simulated hand's ``control_dofs_pos`` replaced by
the ``WujiHandDeployment`` backend that ``examples/recipes/deployment/real_wuji_hand2.py`` drives.
Once per 50 Hz control tick::

    glove.keypoints()          (21, 3) landmarks
      -> retarget              the hand's 20 joint targets (see below)
      -> write_dofs_targets
      -> step(None)            check the hand, stream 200 Hz sub-targets, read it back

**Retargeters.** ``--retargeter wuji`` (default) is ``wuji_sdk.RetargetSession``, the vendor's
solver with its built-in tuning for the Wuji Hand 2: it follows every finger joint, PIP and DIP
included, at ~0.3 ms a frame. ``--retargeter dexpilot`` is Eden's ``DexRetargeter`` preset, which
matches fingertip vectors only; a fingertip a curled finger reaches can also be reached by bending
the MCP alone, so on intermediate poses it tends to leave PIP / DIP straight. It needs the CPU backend (the
default) to keep the 50 Hz tick: its FK costs ~8 ms a frame there and ~106 ms under ``--gpu``.

**Order matters (dexpilot).** ``DexRetargeter`` caches ``env.entities["robot"]`` when it is built,
and the deployment swaps that slot for a ``RobotStateEntity``, so the retargeter is built first: its
FK keeps running on the sim hand, whose qpos ``step`` then resyncs to the measured pose.

**Startup.** Nothing moves until the glove is tracking and the operator confirms. The hand then
eases from its measured pose to the glove's pose at confirmation over ``--ease-seconds``, and the
script presses START (``request_start``) so the deployment sends its targets rather than holding.

**Gains** are the hand's firmware MIT gains (kp 3.0, kd 0.05), as in ``real_wuji_hand2.py``.

**Stopping.** Ctrl-C, a hand fault or lost hand feedback disables the motors (the fingers go limp
where they are). A glove whose pose stops updating is *not* a stop: the hand holds the last target
and the script warns until the pose moves again.

``--retargeter dexpilot`` also needs the ``dex-retargeter`` extra (``uv sync --inexact --extra dex-retargeter``).

Usage:
    WUJI_TELEMETRY_DISABLE=1 uv run python examples/recipes/teleop/wuji/glove_real_hand.py
    WUJI_TELEMETRY_DISABLE=1 uv run python examples/recipes/teleop/wuji/glove_real_hand.py \
        --glove-sn WG1KA06260801526 --hand-sn WH2KA01260918006 --seconds 60
    # retarget the glove and print the targets, without connecting the hand
    WUJI_TELEMETRY_DISABLE=1 uv run python examples/recipes/teleop/wuji/glove_real_hand.py --dry-run
"""

import time
from argparse import ArgumentParser
from typing import Callable

import genesis as gs
import numpy as np
import torch

import eden as en
from eden import actions, observations
from eden.envs.base import RLEnvBase
from eden.extensions.deployment.utils.rate import RateLimiter
from eden.extensions.deployment.wuji_hand import WujiHandDeployment, sdk_index_map
from eden.extensions.retargeting.dex_retargeter.presets import HAND_PRESETS
from eden.options import ObservationGroupOptions
from eden.utils.configs import (
    ActionManagerOptions,
    EdenRLConfig,
    EnvOptions,
    ObservationManagerOptions,
    SceneOptions,
)
from eden.utils.devices.wuji import WujiGlove
from eden.utils.devices.wuji import protocol as P
from eden.utils.devices.wuji.glove import load_sdk

CONTROL_DT = 0.02  # 50 Hz control tick; the backend streams sub-targets inside it
PRESET_FOR_SIDE = {"right": "wuji_hand2beta2_right_dexpilot", "left": "wuji_hand2beta2_left_dexpilot"}

# The hand's own firmware MIT gains (see real_wuji_hand2.py).
MIT_KP, MIT_KD = 3.0, 0.05

GLOVE_WAIT_S = 10.0  # how long to wait for the first tracked hand pose
GLOVE_FROZEN_S = 0.5  # a pose unchanged for this long is reported as frozen


def make_config(robot) -> EdenRLConfig:
    """One hand, one joint-position controller and a ``policy`` group: what the deploy loop computes."""
    return EdenRLConfig(
        env_options=EnvOptions(num_envs=1, sim_dt=CONTROL_DT, decimation=1),
        scene_options=SceneOptions(robot=robot),
        action_options=ActionManagerOptions(
            joint_pos=actions.ImplicitPDController.configure(entity_name="robot", dofs_name=list(robot.dofs_name))
        ),
        observation_options=ObservationManagerOptions(
            policy=ObservationGroupOptions(dofs_pos=observations.DofsPos.configure(entity_name="robot"))
        ),
    )


def make_retargeter(kind: str, side: str, env: RLEnvBase, preset) -> tuple[str, Callable[[np.ndarray], np.ndarray]]:
    """``(keypoint frame, keypoints -> 20 joint targets in the sim hand's dofs_name order)``.

    The targets are clipped to the hand's joint limits here rather than only inside the deployment
    (``clip_targets_to_joint_limits``), so the printed target and tracking error are what was sent.
    Build it before the deployment: the dexpilot retargeter caches the sim entity.
    """
    dofs_name = list(env.entities["robot"].dofs_name)
    lower, upper = (limit[0].cpu().numpy() for limit in env.entities["robot"].get_dofs_limit())
    if kind == "wuji":
        sdk = load_sdk()
        session = sdk.RetargetSession.for_hand(sdk.HandModel.WujiHand2, getattr(sdk.Handedness, side.capitalize()))
        index = sdk_index_map(dofs_name, owner="glove_real_hand")  # dofs_name[i] is SDK joint index[i]
        return "wuji", lambda kp: np.clip(np.asarray(session.step(kp), dtype=np.float64)[index], lower, upper)

    # Only dexpilot needs the SLSQP solver (``nlopt``, the ``dex-retargeter`` extra).
    from eden.extensions.retargeting.dex_retargeter import DexRetargeter

    retargeter = DexRetargeter(env=env, options=preset.retargeter_options())
    human_indices = retargeter.optimizer.target_link_human_indices

    def dexpilot(kp: np.ndarray) -> np.ndarray:
        ref = DexRetargeter.extract_ref_value(P.keypoints_to_mano(kp), human_indices, retargeter.retargeting_type)
        return np.clip(retargeter.retarget(ref), lower, upper)

    return "hand", dexpilot


def wait_for_pose(glove: WujiGlove, frame: str) -> torch.Tensor:
    """The first tracked hand pose, or a loud error after ``GLOVE_WAIT_S``."""
    deadline = time.monotonic() + GLOVE_WAIT_S
    while time.monotonic() < deadline:
        keypoints = glove.keypoints(frame)
        if keypoints is not None:
            return keypoints
        time.sleep(0.02)
    raise RuntimeError(
        f"Wuji Glove {glove.serial_number}: no hand pose in {GLOVE_WAIT_S:g} s. The glove is connected but "
        "not solving a hand; check it is worn and that `uv run eden wuji monitor` shows a moving skeleton."
    )


def main() -> None:
    parser = ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--side", default="right", choices=["left", "right"], help="hand to teleoperate")
    parser.add_argument("--glove-sn", default=None, help="glove serial number (default: the only glove found)")
    parser.add_argument("--hand-sn", default="", help="hand serial number (default: the only hand found)")
    parser.add_argument(
        "--retargeter", default="wuji", choices=["wuji", "dexpilot"], help="glove -> hand joint mapping (see above)"
    )
    parser.add_argument("--seconds", type=float, default=0.0, help="run for this long (0 = until Ctrl-C)")
    parser.add_argument("--ease-seconds", type=float, default=2.0, help="ease to the first glove pose over this")
    parser.add_argument("--effort-limit", type=float, default=1.0, help="joint current limit programmed at connect, A")
    parser.add_argument("--servo-hz", type=float, default=200.0, help="rate of the sub-targets inside each tick")
    parser.add_argument("--dry-run", action="store_true", help="retarget and print targets; never connect the hand")
    parser.add_argument("--yes", action="store_true", help="start moving without asking for confirmation")
    parser.add_argument("--gpu", action="store_true", help="run on the GPU instead of the CPU (slower here)")
    args = parser.parse_args()

    # The backend is WujiHandDeployment, so the preset is always the Wuji Hand 2 of --side.
    preset = HAND_PRESETS[PRESET_FOR_SIDE[args.side]]

    # CPU by default: this env does no physics, only the dexpilot solver's FK, which syncs the
    # device ~13 times per evaluation -- 8 ms a frame on CPU vs 106 ms on GPU (the 20 ms tick).
    en.init(backend=gs.gpu if args.gpu else gs.cpu)

    env = RLEnvBase.from_config(make_config(preset.robot_options_factory()))
    env.build()
    env.reset()

    # Before the deployment: the dexpilot retargeter must keep the sim entity, not the RobotStateEntity.
    frame, retarget = make_retargeter(args.retargeter, args.side, env, preset)
    dofs_name = list(env.entities["robot"].dofs_name)

    dep = None
    if not args.dry_run:
        dep = WujiHandDeployment(
            env,
            WujiHandDeployment.configure(
                entity_name="robot",
                serial_number=args.hand_sn,
                mit_kp=MIT_KP,
                mit_kd=MIT_KD,
                effort_limit_a=args.effort_limit,
                servo_hz=args.servo_hz,
                clip_targets_to_joint_limits=True,
            ),
        )
        # Deployment DOF order, as a permutation of the retargeter's output order.
        to_dep = [dofs_name.index(name) for name in dep.dofs_name]

    glove = WujiGlove(sn=args.glove_sn, side=args.side, streams=("hand_skeleton",))
    try:
        keypoints = wait_for_pose(glove, frame)
        if dep is not None:
            print(
                f"About to enable the {args.side} Wuji Hand 2 (MIT kp {MIT_KP:g}, kd {MIT_KD:g}, "
                f"{args.effort_limit:g} A), ease it to your hand's pose over {args.ease_seconds:g} s and follow "
                "the glove. Keep the fingers' path clear."
            )
            if not args.yes:
                input("Press Enter to start, Ctrl-C to abort... ")
                # The operator's hand moved while the prompt waited: ease to where it is now, or the
                # first live tick jumps from the pre-prompt pose to this one with no ramp.
                keypoints = wait_for_pose(glove, frame)
        target = retarget(keypoints.numpy())

        if dep is not None:
            dep.connect()
            dep.observe()
            dep.ease_to(target[to_dep], args.ease_seconds, rate_hz=args.servo_hz)
            # ease_to moves the hand outside the control tick, so read it back: START's handover tick
            # holds the pose it reads here, not the one from before the ease.
            dep.observe()
            # The confirmation above was the operator's START: without it the deployment stays in
            # STOP and every step(None) holds that pose, dropping the glove's targets.
            dep.request_start()
        en.logger.info(f"following the {args.side} glove — Ctrl-C to stop")

        rate = RateLimiter(1.0 / CONTROL_DT) if dep is None else None  # the deployment paces itself
        deadline = time.monotonic() + args.seconds if args.seconds > 0 else None
        last_kp, last_change, frozen, ticks, t0 = keypoints.numpy(), time.monotonic(), False, 0, time.monotonic()
        while deadline is None or time.monotonic() < deadline:
            keypoints = glove.keypoints(frame)
            now = time.monotonic()
            if keypoints is not None and not np.array_equal(kp := keypoints.numpy(), last_kp):
                last_kp, last_change = kp, now
                if frozen:
                    en.logger.info("glove pose moving again; following")
                    frozen = False
                target = retarget(kp)
                if dep is not None:
                    dep.write_dofs_targets(torch.as_tensor(target[to_dep], dtype=torch.float32), dep.dofs_name)
            elif not frozen and now - last_change > GLOVE_FROZEN_S:
                en.logger.warning(f"glove pose unchanged for {GLOVE_FROZEN_S:g} s; holding the last target")
                frozen = True

            if dep is not None:
                dep.step(None)  # check the hand, stream the target over the period, read it back
            else:
                rate.sleep()
            ticks += 1
            if ticks % 50 == 0:
                line = f"[{now - t0:6.1f} s] target (deg) {np.degrees(target).round(0).astype(int).tolist()}"
                if dep is not None:
                    err = np.abs(dep.state.dofs_pos - target[to_dep])
                    line += f"; tracking max {np.degrees(err.max()):4.1f} deg, mean {np.degrees(err.mean()):3.1f} deg"
                print(line)
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        if dep is not None:
            dep.close()  # disables the motors and releases this hand's connection only
            print("Hand disabled and disconnected.")
        glove.close()


if __name__ == "__main__":
    main()
