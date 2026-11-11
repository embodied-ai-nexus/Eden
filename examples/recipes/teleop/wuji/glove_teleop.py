"""Teleoperate a Wuji Hand 2 from a Wuji Glove, with the operator's hand drawn beside it.

---
tier: recipe
capabilities: [teleop.wuji_glove, retarget.hand, robot.wuji_hand2beta2]
hardware: cpu
optional_deps: [wuji]
run: uv run python examples/recipes/teleop/wuji/glove_teleop.py
---

Input and output in one viewer: the glove's live 21-landmark hand skeleton as debug
spheres on the left, the Wuji Hand 2 it drives on the right. Both fingertip sets swell
and redden with the glove's tactile pressure, so a grasp shows up as contact on the
operator's hand and on the robot finger it maps to.

The loop, once per frame::

    glove.keypoints()          (21, 3) MediaPipe landmarks, Eden hand frame
      -> draw_hand()           debug spheres + bones, rotated to face like the robot
      -> retarget              the hand's 20 joint targets (``--retargeter``)
      -> control_dofs_pos

``--retargeter wuji`` (default) is ``wuji_sdk.RetargetSession``, the vendor's solver with its
built-in Wuji Hand 2 tuning: it follows every finger joint, PIP and DIP included.
``--retargeter dexpilot`` is Eden's ``DexRetargeter`` (nlopt SLSQP over Genesis FK on
``keypoints_to_mano()`` output). It matches fingertip vectors only, and a fingertip a curled
finger reaches can also be reached by bending the MCP alone, so on intermediate poses it
tends to leave PIP / DIP straight. Its FK teleports the hand it solves on, so the loop puts
the simulated hand back where physics left it after each solve, and the PD alone moves it.

For dexpilot, ``keypoints_to_mano`` replaces the per-frame wrist-frame SVD a MediaPipe pipeline needs:
a camera detection arrives in an arbitrary frame, but the glove's output is already
anchored to the hand, so the map is a constant (pinned in ``tests/utils/devices/wuji/test_protocol.py``).

The tactile display is exactly that — a *display*. It never enters the control path, and
the sim's own contact forces are independent of it.

Needs the ``wuji`` extra; ``--retargeter dexpilot`` also needs ``dex-retargeter``::

    uv sync --inexact --extra wuji --extra dex-retargeter

Usage:
    uv run python examples/recipes/teleop/wuji/glove_teleop.py
    uv run python examples/recipes/teleop/wuji/glove_teleop.py --side left
    uv run python examples/recipes/teleop/wuji/glove_teleop.py --retargeter dexpilot
"""

import time
from argparse import ArgumentParser

import genesis as gs
import numpy as np
import torch

import eden as en
from eden.envs.base import EnvBase
from eden.extensions.deployment.wuji_hand import sdk_index_map
from eden.extensions.deployment.utils.rate import RateLimiter
from eden.extensions.retargeting.dex_retargeter.presets import HAND_PRESETS, POSITION_PRESET_NAMES, HandPreset
from eden.options import EnvOptions, PlaneOptions, SceneOptions
from eden.utils.devices.wuji import WujiGlove
from eden.utils.devices.wuji import protocol as P
from eden.utils.devices.wuji.glove import load_sdk

FPS = 60.0
PRESET_FOR_SIDE = {"right": "wuji_hand2beta2_right_dexpilot", "left": "wuji_hand2beta2_left_dexpilot"}
# The operator's wrist, parked 25 cm to the left of the robot at the height the preset
# spawns its hand root (0.2 m), so the two hands sit side by side.
HAND_ORIGIN = np.array([0.0, 0.25, 0.2])

# The glove's hand frame is x distal / y left / z dorsal, but the Wuji Hand 2 spawns
# fingers-up and palm-forward (its `up`/`front` axes map to world +Z / +X at load). Turn
# the drawn hand to match, or the operator's hand and the robot's face different ways and
# the mimicry is impossible to read. Columns are the images of x / y / z_hand:
# distal -> world +z (up), dorsal -> world -x (palm faces forward, like the robot's).
R_VIEW = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

BONE_COLOR = (0.35, 0.45, 0.55, 0.9)
JOINT_COLOR = (0.20, 0.75, 0.45, 0.9)
TIP_RADIUS = (0.005, 0.016)  # (no contact, full pressure)


def contact_style(pressure: float) -> tuple[float, tuple[float, float, float, float]]:
    """``(radius, rgba)`` for a fingertip marker: mint and small when free, red and fat when pressing."""
    force = min(max(pressure, 0.0), 1.0)
    radius = TIP_RADIUS[0] + (TIP_RADIUS[1] - TIP_RADIUS[0]) * force
    return radius, (0.25 + 0.75 * force, 0.70 - 0.50 * force, 0.45 - 0.35 * force, 0.85)


def zone_peak(values: np.ndarray) -> float:
    """Peak of a zone's live taxels, or 0.0 if it has none (see ``glove_demo.zone_peak``).

    ``tactile_zones`` masks padding taxels to ``nan``; a zone with none live (a dead finger sheet)
    would reduce to ``nan``, which survives the clamp in :func:`contact_style` as a nan radius.
    """
    live = values[~np.isnan(values)]
    return float(live.max()) if live.size else 0.0


def finger_pressure(glove: WujiGlove) -> np.ndarray:
    """Peak pressure per finger in FINGERS order, zeros before the first tactile frame."""
    zones = glove.tactile_zones()
    if zones is None:
        return np.zeros(len(P.FINGERS))
    return np.array([zone_peak(zones[finger].numpy()) for finger in P.FINGERS])


def resolve_preset(name: str | None, side: str, retargeter: str) -> HandPreset:
    """The ``--preset`` to drive, checked against ``--side`` and ``--retargeter`` before anything is built."""
    default = PRESET_FOR_SIDE[side]
    if retargeter == "wuji" and name not in (None, default):
        raise SystemExit(
            f"--preset {name!r} needs --retargeter dexpilot: the vendor solver (--retargeter wuji) only drives the "
            f"Wuji Hand 2, so with --side {side} it uses {default!r}. Drop --preset, or add --retargeter dexpilot."
        )
    if name in POSITION_PRESET_NAMES:
        # Absolute MANO positions would be solved against a hand spawned 0.2 m up, in another frame.
        valid = sorted(
            key for key, other in HAND_PRESETS.items() if other.side == side and key not in POSITION_PRESET_NAMES
        )
        raise SystemExit(
            f"--preset {name!r} is a POSITION preset; the glove drives relative-vector presets only. "
            f"Pass one of {valid}."
        )
    preset = HAND_PRESETS[name or default]
    if preset.side != side:
        matching = sorted(key for key, other in HAND_PRESETS.items() if other.side == side)
        raise SystemExit(
            f"--preset {name!r} is a {preset.side} hand but --side is {side!r}: the {side} glove would drive a "
            f"mirrored hand. Pass --side {preset.side}, or a {side} preset: {matching}."
        )
    return preset


def draw_hand(env: EnvBase, keypoints: np.ndarray, pressure: np.ndarray) -> None:
    """Draw the operator's hand: bones, joint spheres, and pressure-swollen fingertips."""
    points = keypoints @ R_VIEW.T + HAND_ORIGIN
    for a, b in P.HAND_CONNECTIONS:
        env.draw_debug_line(points[a], points[b], radius=0.0015, color=BONE_COLOR)
    env.draw_debug_spheres(points, radius=0.005, color=JOINT_COLOR)
    for landmark, force in zip(P.FINGERTIP_LANDMARKS, pressure):
        radius, color = contact_style(float(force))
        env.draw_debug_sphere(points[landmark], radius=radius, color=color)


def draw_contact(env: EnvBase, tip_positions: np.ndarray, pressure: np.ndarray) -> None:
    """Echo the operator's per-finger contact onto the robot fingertip that finger drives."""
    for position, force in zip(tip_positions, pressure):
        radius, color = contact_style(float(force))
        env.draw_debug_sphere(position, radius=radius, color=color)


def main() -> None:
    parser = ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sn", default=None, help="serial number of the glove (default: the only one found)")
    parser.add_argument("--side", default="right", choices=["left", "right"], help="which hand to teleoperate")
    parser.add_argument(
        "--retargeter", default="wuji", choices=["wuji", "dexpilot"], help="glove -> hand joint mapping (see above)"
    )
    parser.add_argument(
        "--preset",
        default=None,
        choices=sorted(HAND_PRESETS),
        help="dexpilot preset (default: the Wuji Hand 2 matching --side)",
    )
    parser.add_argument("--seconds", type=float, default=0.0, help="run for this long (0 = until Ctrl-C)")
    parser.add_argument(
        "--rate-hz",
        type=int,
        default=None,
        help="lower the device push rate from its 120 Hz default (firmware 0.11.1 cannot; it warns and ignores)",
    )
    parser.add_argument("--no-skeleton", action="store_false", dest="skeleton", help="hide the operator's hand")
    parser.add_argument("--no-vis", action="store_false", dest="vis", help="run headless (no Genesis viewer)")
    parser.add_argument("--gpu", action="store_true", help="run on the GPU instead of the CPU (slower here)")
    args = parser.parse_args()

    preset = resolve_preset(args.preset, args.side, args.retargeter)

    # CPU by default: one hand steps faster there (0.8 vs 1.7 ms), and the dexpilot solver's FK,
    # which syncs the device ~13 times per evaluation, takes 8 ms a frame vs 106 ms on GPU.
    en.init(backend=gs.gpu if args.gpu else gs.cpu)

    env = EnvBase(
        env_options=EnvOptions(num_envs=1, show_world_frame=False),
        scene_options=SceneOptions(robot=preset.robot_options_factory(), floor=PlaneOptions()),
        show_viewer=args.vis and not en.is_smoke(),
    )
    env.build()
    env.reset()

    robot = env.entities["robot"]
    if args.retargeter == "wuji":
        sdk = load_sdk()
        session = sdk.RetargetSession.for_hand(sdk.HandModel.WujiHand2, getattr(sdk.Handedness, args.side.capitalize()))
        sdk_index = sdk_index_map(list(robot.dofs_name), owner="glove_teleop")  # dofs_name[i] is SDK joint sdk_index[i]
    else:
        # Only dexpilot needs the SLSQP solver (``nlopt``, the ``dex-retargeter`` extra).
        from eden.extensions.retargeting.dex_retargeter import DexRetargeter

        retargeter = DexRetargeter(env=env, options=preset.retargeter_options())
        human_indices = retargeter.optimizer.target_link_human_indices
    # The hand's fingertip links, resolved once so the contact echoes can follow them.
    _, tip_links = robot.find_named_links_idx_local(
        list(type(preset.robot_options_factory()).metadata.target_hand_links)
    )

    glove = WujiGlove(
        sn=args.sn,
        side=args.side,
        streams=("hand_skeleton", "tactile_zones"),
        rate_hz=args.rate_hz,
    )
    en.logger.info(f"teleoperating the {args.side} Wuji Hand 2 — open and close your hand, Ctrl-C to stop")

    deadline = time.monotonic() + args.seconds if args.seconds > 0 else None
    remaining = 5 if en.is_smoke() else -1  # -1 = run until Ctrl-C
    rate = RateLimiter(FPS)
    try:
        while remaining != 0 and (deadline is None or time.monotonic() < deadline):
            keypoints = glove.keypoints()
            if keypoints is None:
                en.logger.warning("no hand pose yet — is the glove worn and solving?")
            else:
                if args.retargeter == "wuji":
                    qpos = np.asarray(session.step(glove.keypoints("wuji").numpy()))[sdk_index]
                else:
                    mano = P.keypoints_to_mano(keypoints.numpy())
                    ref = DexRetargeter.extract_ref_value(mano, human_indices, retargeter.retargeting_type)
                    # The solver's FK teleports `robot` (set_qpos, zero velocity, per evaluation): put the
                    # simulated hand back where physics left it, so the PD below is what moves it.
                    sim_qpos, sim_vel = robot.get_qpos(), robot.get_dofs_vel()
                    qpos = retargeter.retarget(ref)
                    robot.set_qpos(sim_qpos, zero_velocity=False)
                    robot.set_dofs_vel(sim_vel)
                robot.control_dofs_pos(torch.as_tensor(qpos, dtype=torch.float32, device=robot.device).unsqueeze(0))

            env.step()

            # Debug geometry is redrawn wholesale each tick, so clear before drawing.
            env.clear_debug_objects()
            pressure = finger_pressure(glove)
            if keypoints is not None and args.skeleton:
                draw_hand(env, keypoints.numpy(), pressure)
            draw_contact(env, robot.get_links_pos(ls_idx_local=tip_links)[0].cpu().numpy(), pressure)

            if remaining > 0:
                remaining -= 1
            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        glove.close()


if __name__ == "__main__":
    main()
