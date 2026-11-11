"""XR bridge process for Meta Quest teleoperation.

Owns the televuer XR session: it spins up a Vuer WebXR server that the Meta
Quest connects to over Wi-Fi, captures head / wrist poses plus either the
hand-joint poses *or* the motion-controller state, and displays the stereo
frames that Eden renders.

WebXR exposes hand tracking and motion controllers as *mutually exclusive*
input sources, so the bridge runs in one of two modes (``--tracking``):
    * ``hand``       : head/wrist poses + 25 hand-joint positions + pinch/squeeze
    * ``controller`` : head/wrist poses + per-hand trigger / grip / A,B buttons /
                       thumbstick (button + 2D axis)

Data flow (see protocol.py):
    * publishes head/wrist + hand-or-controller state  ->  Eden  (PUB on POSE_PORT)
    * receives stereo RGB frames                        <-  Eden  (PULL on FRAME_PORT) -> headset

Each published dict carries a ``"mode"`` key (``"hand"`` or ``"controller"``) so
the Eden side knows which fields to expect.

All published poses are converted from the OpenXR basis (x right, y up, z back)
into the Genesis/robot basis (x forward, y left, z up) so Eden can consume them
directly in its world frame.

Requires the ``metaquest`` extra and televuer, which is not on PyPI
(``uv sync --inexact --extra metaquest --group metaquest-git`` in a checkout).

Run:
    eden metaquest bridge                       # hand tracking (default)
    eden metaquest bridge --tracking controller # motion controllers
    # or, equivalently:
    uv run python -m eden.utils.devices.metaquest.xr_bridge
"""

import argparse
import signal
import time

import numpy as np
import zmq

import eden as en
from eden.utils.devices.metaquest import protocol as P
from eden.utils.distribution import install_hint
from eden.utils.misc import list_host_ips

try:
    # televuer ships the OpenXR<->robot basis matrices; reuse them here.
    from televuer.televuer import TeleVuer
    from televuer.tv_wrapper import R_ROBOT_OPENXR, T_OPENXR_ROBOT, T_ROBOT_OPENXR
except ModuleNotFoundError as e:
    if e.name != "televuer":
        raise
    raise ModuleNotFoundError(
        "The Meta Quest bridge needs televuer, which is not installed. It is not on PyPI, so the `metaquest` extra "
        f"alone does not bring it: install it with {install_hint('metaquest')}.",
        name="televuer",
    ) from e


def to_robot_se3(m_xr):
    """OpenXR-basis 4x4 pose -> robot/Genesis-basis 4x4 (z up, x forward)."""
    return T_ROBOT_OPENXR @ np.asarray(m_xr) @ T_OPENXR_ROBOT


def to_robot_points(p_xr):
    """OpenXR-basis (N, 3) points -> robot/Genesis-basis (N, 3)."""
    return (R_ROBOT_OPENXR @ np.asarray(p_xr).T).T


def build_pose_msg(tv: TeleVuer):
    """Snapshot the current hand-tracking XR state into a JSON dict (robot frame)."""
    msg = {
        "mode": "hand",
        "ready": bool(tv.motion_data_ready),
        "head": to_robot_se3(tv.head_pose).tolist(),
        "left_wrist": to_robot_se3(tv.left_arm_pose).tolist(),
        "right_wrist": to_robot_se3(tv.right_arm_pose).tolist(),
        "left_hand_pos": to_robot_points(tv.left_hand_positions).tolist(),
        "right_hand_pos": to_robot_points(tv.right_hand_positions).tolist(),
        # pinchValue is a distance (~0 when pinching); expose both the bool and value.
        "left_pinch": bool(tv.left_hand_pinch),
        "right_pinch": bool(tv.right_hand_pinch),
        "left_pinch_value": float(tv.left_hand_pinchValue),
        "right_pinch_value": float(tv.right_hand_pinchValue),
        "left_squeeze": float(tv.left_hand_squeezeValue),
        "right_squeeze": float(tv.right_hand_squeezeValue),
    }
    return msg


def build_controller_msg(tv: TeleVuer):
    """Snapshot the current motion-controller XR state into a JSON dict (robot frame).

    ``left_wrist`` / ``right_wrist`` are the controller grip poses (already in the
    robot frame). Button / trigger / grip booleans and analog values plus the
    thumbstick click + 2D axis come straight from the WebXR gamepad mapping:

      * ``*_trigger_value`` / ``*_squeeze_value``: 0.0 (released) -> 1.0 (fully pressed)
      * ``*_thumbstick_value``: (x, y) in [-1, 1], +x right, +y back (see televuer)
    """
    msg = {
        "mode": "controller",
        "ready": bool(tv.motion_data_ready),
        "head": to_robot_se3(tv.head_pose).tolist(),
        "left_wrist": to_robot_se3(tv.left_arm_pose).tolist(),
        "right_wrist": to_robot_se3(tv.right_arm_pose).tolist(),
    }
    for side in ("left", "right"):
        msg.update(
            {
                f"{side}_trigger": bool(getattr(tv, f"{side}_ctrl_trigger")),
                f"{side}_trigger_value": float(getattr(tv, f"{side}_ctrl_triggerValue")),
                f"{side}_squeeze": bool(getattr(tv, f"{side}_ctrl_squeeze")),
                f"{side}_squeeze_value": float(getattr(tv, f"{side}_ctrl_squeezeValue")),
                f"{side}_a_button": bool(getattr(tv, f"{side}_ctrl_aButton")),
                f"{side}_b_button": bool(getattr(tv, f"{side}_ctrl_bButton")),
                f"{side}_thumbstick": bool(getattr(tv, f"{side}_ctrl_thumbstick")),
                f"{side}_thumbstick_value": [float(x) for x in getattr(tv, f"{side}_ctrl_thumbstickValue")],
            }
        )
    return msg


def main(argv=None):
    """Run the XR bridge loop (parses ``argv`` or ``sys.argv`` when ``None``)."""
    ap = argparse.ArgumentParser(description="Eden <-> Quest XR bridge (televuer side)")
    ap.add_argument("--width", type=int, default=1280, help="full stereo frame width (both eyes)")
    ap.add_argument("--height", type=int, default=480, help="frame height")
    ap.add_argument("--fps", type=float, default=30.0, help="headset display fps")
    ap.add_argument("--display-mode", default="immersive", choices=["immersive", "ego", "pass-through"])
    ap.add_argument(
        "--tracking",
        default="hand",
        choices=["hand", "controller"],
        help="XR input source: hand joints (default) or motion controllers (mutually exclusive in WebXR)",
    )
    args = ap.parse_args(argv)

    use_hand_tracking = args.tracking == "hand"
    build_msg = build_pose_msg if use_hand_tracking else build_controller_msg

    # The bridge runs televuer, not Genesis, so en.init() (and its en.logger) never
    # runs here -- stand up Eden's logger directly.
    logger = getattr(en, "logger", None) or en.Logger("INFO")

    tv = TeleVuer(
        use_hand_tracking=use_hand_tracking,
        binocular=True,
        img_shape=(args.height, args.width),
        display_fps=args.fps,
        display_mode=args.display_mode,
        zmq=True,  # in-process shared-memory frame display (no external image server)
        webrtc=False,
    )

    ctx = zmq.Context.instance()
    frame_sock = P.frame_pull(ctx)
    pose_sock = P.pose_pub(ctx)

    # Eden-styled banner: ~<...>~ mint, ~~<...>~~ mint-underline, ~~~~<...>~~~~ bold.
    rule = "━" * 64
    logger.info(f"~<{rule}>~")
    logger.info(f" 🍏 ~~~~<Meta Quest XR bridge>~~~~  ({args.tracking} tracking)  —  waiting for the headset")
    logger.info(" On the Quest: join this host's Wi-Fi, then open in the browser ~<→>~")
    for ip in list_host_ips() or ["<host-ip>"]:
        logger.info(f"     ~~<https://{ip}:8012/?ws=wss://{ip}:8012>~~")
    logger.info(" Tap ~<Virtual Reality>~ and allow all prompts")
    logger.info(" (first run: accept the self-signed cert — Advanced ~<→>~ Proceed)")
    logger.info(f" ZMQ   poses PUB ~<:{P.POSE_PORT}>~   frames PULL ~<:{P.FRAME_PORT}>~")
    logger.info(f"~<{rule}>~")

    running = {"v": True}

    def stop(*_):
        running["v"] = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    period = 1.0 / max(args.fps, 1.0)
    frames_shown = 0
    last_log = time.time()
    try:
        while running["v"]:
            loop_start = time.time()

            # Newest frame from Eden -> headset. Eden sends RGB; render_to_xr treats
            # its input as BGR (it does BGR2RGB internally), so flip channels here.
            frame = P.recv_frame(frame_sock, flags=zmq.NOBLOCK)
            if frame is not None:
                tv.render_to_xr(np.ascontiguousarray(frame[:, :, ::-1]))
                frames_shown += 1

            # Publish current XR pose state to Eden.
            P.send_pose(pose_sock, build_msg(tv))

            now = time.time()
            if now - last_log > 5.0:
                state = "tracking" if tv.motion_data_ready else "waiting for headset"
                logger.info(f"{state} | frames shown: {frames_shown}")
                last_log = now

            dt = time.time() - loop_start
            if dt < period:
                time.sleep(period - dt)
    finally:
        logger.info("shutting down...")
        tv.close()


if __name__ == "__main__":
    main()
