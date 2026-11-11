"""Live Wuji Glove hand skeleton and tactile contact in the Genesis viewer.

---
tier: recipe
capabilities: [teleop.wuji_glove]
hardware: cpu
optional_deps: [wuji]
run: uv run python examples/recipes/teleop/wuji/glove_demo.py
---

The "is my glove working" recipe, one step past ``eden wuji monitor``: it draws the
glove's 21 MediaPipe landmarks as a hand skeleton in the scene and swells each fingertip
sphere with that finger's tactile pressure, so both streams are visible at once. No robot
and no retargeting — see ``glove_teleop.py`` for that.

Poses come from :meth:`WujiGlove.keypoints`, already in Eden's hand frame (x out along
the fingers, y left, z out of the back of the hand). The hand is wrist-relative, so it is
drawn at a fixed ``ORIGIN`` above the floor; a glove reports no wrist pose in the world
(pair it with an arm, a headset, or a tracker for that).

Usage:
    uv run python examples/recipes/teleop/wuji/glove_demo.py
    uv run python examples/recipes/teleop/wuji/glove_demo.py --side right --seconds 60
    uv run python examples/recipes/teleop/wuji/glove_demo.py --no-vis   # headless
"""

import time
from argparse import ArgumentParser

import genesis as gs
import numpy as np

import eden as en
from eden.envs.base import EnvBase
from eden.extensions.deployment.utils.rate import RateLimiter
from eden.options import EnvOptions, PlaneOptions, SceneOptions
from eden.utils.devices.wuji import WujiGlove
from eden.utils.devices.wuji import protocol as P

ORIGIN = np.array([0.0, 0.0, 0.35])  # where the wrist landmark is parked in the world
FPS = 60.0

BONE_COLOR = (0.35, 0.45, 0.55, 0.9)
# A fingertip marker interpolates between these two ends as its tactile zone loads up.
JOINT_COLOR = (0.20, 0.75, 0.45, 0.9)  # mint: touching nothing
CONTACT_COLOR = (1.00, 0.20, 0.10, 0.9)  # red: full pressure
TIP_RADIUS = (0.006, 0.020)  # (no contact, full pressure)


def contact_style(pressure: float) -> tuple[float, tuple[float, ...]]:
    """``(radius, rgba)`` for a fingertip marker: mint and small when free, red and fat when pressing."""
    force = min(max(pressure, 0.0), 1.0)
    radius = TIP_RADIUS[0] + (TIP_RADIUS[1] - TIP_RADIUS[0]) * force
    return radius, tuple(free + (pressed - free) * force for free, pressed in zip(JOINT_COLOR, CONTACT_COLOR))


def zone_peak(values: np.ndarray) -> float:
    """Peak of a zone's live taxels, or 0.0 if it has none.

    ``tactile_zones`` masks its padding taxels to ``nan``, and a zone with no live taxels
    at all (a dead finger sheet) would reduce to ``nan`` — which then survives the clamp in
    :func:`contact_style`, because ``min(max(nan, 0.0), 1.0)`` is ``nan``, and reaches the
    viewer as a nan-radius sphere. So drop the masked taxels rather than reduce over them.
    """
    live = values[~np.isnan(values)]
    return float(live.max()) if live.size else 0.0


def finger_pressure(glove: WujiGlove) -> np.ndarray:
    """Peak pressure per finger in FINGERS order, zeros before the first tactile frame."""
    zones = glove.tactile_zones()
    if zones is None:
        return np.zeros(len(P.FINGERS))
    return np.array([zone_peak(zones[finger].numpy()) for finger in P.FINGERS])


def draw_hand(env: EnvBase, keypoints: np.ndarray, pressure: np.ndarray) -> None:
    """Draw one frame of the hand: bones, joints, and pressure-swollen fingertips."""
    points = keypoints + ORIGIN
    for a, b in P.HAND_CONNECTIONS:
        env.draw_debug_line(points[a], points[b], radius=0.0015, color=BONE_COLOR)
    env.draw_debug_spheres(points, radius=0.005, color=JOINT_COLOR)
    for landmark, force in zip(P.FINGERTIP_LANDMARKS, pressure):
        radius, color = contact_style(float(force))
        env.draw_debug_sphere(points[landmark], radius=radius, color=color)


def main() -> None:
    parser = ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sn", default=None, help="serial number of the glove (default: the only one found)")
    parser.add_argument("--side", default=None, choices=["left", "right"], help="pick between two connected gloves")
    parser.add_argument("--seconds", type=float, default=0.0, help="run for this long (0 = until Ctrl-C)")
    parser.add_argument(
        "--rate-hz",
        type=int,
        default=None,
        help="lower the device push rate from its 120 Hz default (firmware 0.11.1 cannot; it warns and ignores)",
    )
    parser.add_argument("--no-vis", action="store_false", dest="vis", help="run headless (no Genesis viewer)")
    parser.add_argument("-c", "--cpu", action="store_true", help="run on CPU instead of the GPU")
    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu)

    env = EnvBase(
        env_options=EnvOptions(num_envs=1, show_world_frame=False),
        scene_options=SceneOptions(floor=PlaneOptions()),
        show_viewer=args.vis and not en.is_smoke(),
    )
    env.build()
    env.reset()

    glove = WujiGlove(
        sn=args.sn,
        side=args.side,
        streams=("hand_skeleton", "tactile_zones"),
        rate_hz=args.rate_hz,
    )
    en.logger.info("wiggle your fingers, then press something — Ctrl-C to stop")

    deadline = time.monotonic() + args.seconds if args.seconds > 0 else None
    remaining = 5 if en.is_smoke() else None  # None = run until Ctrl-C, as for `deadline`
    rate = RateLimiter(FPS)
    try:
        while (remaining is None or remaining > 0) and (deadline is None or time.monotonic() < deadline):
            keypoints = glove.keypoints()
            env.clear_debug_objects()
            if keypoints is None:
                en.logger.warning("no hand pose yet — is the glove worn and solving?")
            else:
                draw_hand(env, keypoints.numpy(), finger_pressure(glove))
            env.scene.visualizer.update()
            if remaining is not None:
                remaining -= 1
            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        glove.close()


if __name__ == "__main__":
    main()
