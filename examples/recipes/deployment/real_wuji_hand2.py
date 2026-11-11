"""Open and close a real Wuji Hand 2 through the ``WujiHandDeployment`` backend.

---
tier: recipe
capabilities: [robot.wuji_hand2beta2, deploy.real_robot]
hardware: real-robot
optional_deps: [wuji]
run: WUJI_TELEMETRY_DISABLE=1 uv run python examples/recipes/deployment/real_wuji_hand2.py
---

The deploy loop a policy runs, with a scripted trajectory in the policy's place. Each control
tick writes absolute joint targets with ``write_dofs_targets`` and calls ``step(None)``, which
checks the hand (stale feedback, latched faults), streams the target to it across the control
period, and reads it back. The hand eases from its measured pose to open, then closes and opens
``--cycles`` times on a cosine profile, and the script prints how closely it tracked each pose.

**Gains.** The hand is programmed with the MIT gains its firmware ships with (kp 3.0, kd 0.05 on
every joint), not with ``WujiHand2Beta2``'s ``default_dofs_kp`` / ``default_dofs_kd`` (50 / 5,
simulation gains), which the backend programs when ``mit_kp`` / ``mit_kd`` are left unset.

**Stopping.** Ctrl-C, a fault or lost feedback ends the run and disables the motors, so the
fingers go limp wherever they are. ``--closure`` scales the fist (0 = open, 1 = full); start low.

``WUJI_TELEMETRY_DISABLE=1`` keeps ``wuji_sdk`` from sending usage events to its analytics host.

Usage:
    WUJI_TELEMETRY_DISABLE=1 uv run python examples/recipes/deployment/real_wuji_hand2.py
    WUJI_TELEMETRY_DISABLE=1 uv run python examples/recipes/deployment/real_wuji_hand2.py \
        --side left --serial-number WH2KA01260918006 --closure 0.8 --cycles 5
"""

import argparse
import time

import genesis as gs
import numpy as np
import torch

import eden as en
from eden import actions, observations
from eden.envs.base import RLEnvBase
from eden.extensions.deployment.base import resolve_dofs_values
from eden.extensions.deployment.wuji_hand import WujiHandDeployment
from eden.options import ObservationGroupOptions
from eden.options.robots.wuji_hand2beta2 import WujiHand2Beta2_L, WujiHand2Beta2_R
from eden.utils.configs import (
    ActionManagerOptions,
    EdenRLConfig,
    EnvOptions,
    ObservationManagerOptions,
    SceneOptions,
)

CONTROL_DT = 0.02  # 50 Hz policy rate; the backend streams 200 Hz sub-targets inside each tick

# The hand's own firmware MIT gains, read back from a Hand 2 with ``hand.mit_params().get()``.
MIT_KP, MIT_KD = 3.0, 0.05

# Open is the straight hand. Closed is a fist whose fingers curl well inside their joint limits,
# with the thumb only lightly flexed so it stays clear of the index finger. Keys are regexes over
# either side's joint names, one key per joint.
OPEN = {".*": 0.0}
CLOSED = {
    ".*_thumb_cmc_flex": 0.3,
    ".*_thumb_cmc_abd": 0.0,
    ".*_thumb_mcp": 0.4,
    ".*_thumb_ip": 0.4,
    ".*_mcp_flex": 1.1,
    ".*_mcp_abd": 0.0,
    ".*_pip": 1.2,
    ".*_dip": 0.9,
}


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


def move(dep: WujiHandDeployment, start: np.ndarray, goal: np.ndarray, seconds: float) -> tuple[np.ndarray, float]:
    """Stream a cosine-eased path from ``start`` to ``goal`` one control tick at a time.

    Returns each joint's worst lag behind the commanded path (rad) and the move's wall-clock duration.
    """
    n = max(1, round(seconds / dep.control_dt))
    lag = np.zeros(len(goal))
    t0 = time.perf_counter()
    for i in range(1, n + 1):
        q = start + 0.5 * (1.0 - np.cos(np.pi * i / n)) * (goal - start)
        dep.write_dofs_targets(torch.as_tensor(q, dtype=torch.float32), dep.dofs_name)
        dep.step(None)  # check the hand, stream the target over the period, read it back
        lag = np.maximum(lag, np.abs(dep.state.dofs_pos - q))
    return lag, time.perf_counter() - t0


def hold(dep: WujiHandDeployment, seconds: float) -> None:
    """Keep sending the last target for ``seconds``."""
    for _ in range(max(1, round(seconds / dep.control_dt))):
        dep.step(None)


def report(dep: WujiHandDeployment, label: str, goal: np.ndarray, lag: np.ndarray, elapsed: float) -> None:
    """Print the worst lag during the move and the error left after the hold."""
    err = np.abs(dep.state.dofs_pos - goal)
    lag_j, err_j = int(np.argmax(lag)), int(np.argmax(err))
    print(
        f"[{label:>6s}] move {elapsed:4.2f} s, worst lag {np.degrees(lag[lag_j]):5.1f} deg ({dep.dofs_name[lag_j]}); "
        f"after hold max {np.degrees(err[err_j]):4.1f} deg ({dep.dofs_name[err_j]}), mean {np.degrees(err.mean()):3.1f} "
        f"deg; MCU {dep.transport.temp_summary() or 'n/a'}"
    )


def main():
    parser = argparse.ArgumentParser(description="Open and close a real Wuji Hand 2")
    parser.add_argument("--side", type=str, default="right", choices=["right", "left"], help="Which hand is connected")
    parser.add_argument(
        "--serial-number", type=str, default="", help="Hand to connect; empty connects the only hand on the bus"
    )
    parser.add_argument("--cycles", type=int, default=3, help="Number of close/open cycles")
    parser.add_argument("--closure", type=float, default=0.6, help="Fraction of the full fist to close to, in [0, 1]")
    parser.add_argument("--move-seconds", type=float, default=1.5, help="Duration of each open or close motion")
    parser.add_argument("--hold-seconds", type=float, default=0.5, help="Pause at each pose")
    parser.add_argument("--effort-limit", type=float, default=1.0, help="Joint current limit programmed at connect, A")
    parser.add_argument(
        "--servo-hz", type=float, default=200.0, help="Rate of the sub-targets streamed inside each control tick"
    )
    parser.add_argument("--yes", action="store_true", help="Start moving without asking for confirmation")
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    args = parser.parse_args()
    if not 0.0 <= args.closure <= 1.0:
        parser.error(f"--closure must be in [0, 1]; got {args.closure}")

    en.init(backend=gs.cpu if args.cpu else gs.gpu)

    robot = WujiHand2Beta2_R() if args.side == "right" else WujiHand2Beta2_L()
    env = RLEnvBase.from_config(make_config(robot))
    env.build()
    env.reset()

    dep = WujiHandDeployment(
        env,
        WujiHandDeployment.configure(
            entity_name="robot",
            serial_number=args.serial_number,
            mit_kp=MIT_KP,
            mit_kd=MIT_KD,
            effort_limit_a=args.effort_limit,
            servo_hz=args.servo_hz,
            clip_targets_to_joint_limits=True,
        ),
    )
    open_pose = resolve_dofs_values(OPEN, dep.dofs_name)
    closed_pose = open_pose + args.closure * (resolve_dofs_values(CLOSED, dep.dofs_name) - open_pose)

    print(
        f"About to enable the {args.side} Wuji Hand 2 (MIT kp {MIT_KP:g}, kd {MIT_KD:g}, {args.effort_limit:g} A) "
        f"and close it to {args.closure:.0%} of a fist {args.cycles} times, {args.move_seconds:g} s per motion, "
        f"streaming at {args.servo_hz:g} Hz. Keep the fingers' path clear."
    )
    if not args.yes:
        input("Press Enter to start, Ctrl-C to abort... ")

    dep.connect()
    try:
        dep.observe()  # read the hand once: the first motion starts where it actually is
        dep.request_start()  # the confirmation above was the operator's START: the script's targets now go out
        lag, elapsed = move(dep, dep.state.dofs_pos.copy(), open_pose, seconds=2.0)  # slow, from wherever it is
        hold(dep, args.hold_seconds)
        report(dep, "open", open_pose, lag, elapsed)
        for cycle in range(1, args.cycles + 1):
            print(f"cycle {cycle}/{args.cycles}")
            for label, start, goal in (("closed", open_pose, closed_pose), ("open", closed_pose, open_pose)):
                lag, elapsed = move(dep, start, goal, args.move_seconds)
                hold(dep, args.hold_seconds)
                report(dep, label, goal, lag, elapsed)
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        dep.close()  # disables the motors and releases this hand's connection only
        print("Hand disabled and disconnected.")


if __name__ == "__main__":
    main()
