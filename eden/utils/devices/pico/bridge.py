"""PICO teleoperation bridge: XRoboToolkit PC Service <-> Eden.

Owns everything device-side of a PICO teleop session so the Eden process only
speaks ZMQ: it reads the headset's tracking state from the PC Service (via
``xrobotoolkit_sdk``), publishes it in the Genesis/robot world frame, and streams
Eden's rendered stereo frames back to the headset as H.264 over the app's
*Remote Vision* link.

Data flow (see :mod:`eden.utils.devices.pico.protocol` for the headset link and
:mod:`eden.utils.devices.metaquest.protocol` for the ZMQ channels)::

    PICO ──Wi-Fi──▶ XRoboToolkit PC Service ──▶ eden pico bridge
                                                 │  poses  (PUB :5557) ──▶ Eden
      headset decoder ◀──H.264 / TCP─────────────┤  frames (PULL :5556) ◀──┘

The published state dict uses the **same schema** as
``eden metaquest bridge --tracking controller``, so an Eden-side teleop loop runs
against either headset unchanged; PICO's extras (menu buttons, motion trackers,
full-body joints) are additive keys. See
:meth:`eden.utils.devices.pico.xr_client.PicoXrClient.state`.

Requires the ``pico`` extra (``uv sync --inexact --extra pico``) plus a locally built
``xrobotoolkit_sdk`` — see :mod:`eden.utils.devices.pico.xr_client`.

Run:
    eden pico bridge
    # or, equivalently:
    uv run python -m eden.utils.devices.pico.bridge
"""

import argparse
import signal
import time

import zmq

import eden as en

# The pose/frame ZMQ channels are shared with the Meta Quest bridge on purpose:
# same ports, same framing, same dict schema, so the Eden-side consumers are
# device-agnostic and the recipes work against either headset.
from eden.utils.devices.metaquest import protocol as P
from eden.utils.devices.pico.protocol import CONTROL_PORT
from eden.utils.devices.pico.video import PicoVideoLink
from eden.utils.devices.pico.xr_client import PicoXrClient
from eden.utils.misc import list_host_ips


def main(argv=None):
    """Run the PICO bridge loop (parses ``argv`` or ``sys.argv`` when ``None``)."""
    ap = argparse.ArgumentParser(description="Eden <-> PICO teleop bridge (XRoboToolkit side)")
    ap.add_argument("--fps", type=float, default=30.0, help="pose publish rate (frames are forwarded as they arrive)")
    ap.add_argument(
        "--control-address",
        default=f"0.0.0.0:{CONTROL_PORT}",
        help="host:port to listen on for the app's OPEN_CAMERA / CLOSE_CAMERA commands",
    )
    ap.add_argument("--pose-port", type=int, default=P.POSE_PORT, help="ZMQ PUB port for the tracking state")
    ap.add_argument("--frame-port", type=int, default=P.FRAME_PORT, help="ZMQ PULL port for Eden's rendered frames")
    ap.add_argument("--no-video", action="store_true", help="publish tracking only; do not stream frames back")
    ap.add_argument(
        "--no-derived-body",
        action="store_true",
        help="skip the derived per-joint body dict and trackers; publish the lossless body_raw only",
    )
    ap.add_argument(
        "--tracker-role",
        action="append",
        default=[],
        metavar="SERIAL=ROLE",
        help="label a motion tracker by serial, e.g. --tracker-role PA1234=waist "
        "(repeatable; unmapped trackers are logged so you can add them)",
    )
    args = ap.parse_args(argv)

    tracker_roles = {}
    for spec in args.tracker_role:
        serial, sep, role = spec.partition("=")
        if not (serial and sep and role):
            ap.error(f"--tracker-role expects SERIAL=ROLE, got {spec!r}")
        tracker_roles[serial] = role

    # The bridge does not run Genesis, so en.init() (and its en.logger) never
    # runs here -- stand up Eden's logger directly.
    logger = getattr(en, "logger", None) or en.Logger("INFO")

    client = PicoXrClient(tracker_roles=tracker_roles)  # raises with a build hint if the SDK is missing
    video = None if args.no_video else PicoVideoLink(args.control_address, logger=logger)

    ctx = zmq.Context.instance()
    # No video, no frame channel: nothing would consume the frames, and not
    # binding keeps the port free for whoever shares this host (the SONIC
    # operator's command PUB uses the same default number, 5556).
    frame_sock = None if video is None else P.frame_pull(ctx, port=args.frame_port)
    pose_sock = P.pose_pub(ctx, port=args.pose_port)

    # Eden-styled banner: ~<...>~ mint, ~~<...>~~ mint-underline, ~~~~<...>~~~~ bold.
    rule = "━" * 64
    logger.info(f"~<{rule}>~")
    logger.info(" 🥽 ~~~~<PICO XR bridge>~~~~  (XRoboToolkit)  —  waiting for the headset")
    logger.info(" On the PICO: open ~<XRoboToolkit>~, point it at this host and tap ~<Connect>~")
    for ip in list_host_ips() or ["<host-ip>"]:
        logger.info(f"     ~~<{ip}>~~")
    logger.info(" Status must read ~<Working>~; then enable ~<Send>~ under Data & Control")
    if video is not None:
        logger.info(f" Stereo view: ~<Remote Vision>~ ~<→>~ ~<Listen>~ (Eden listens on {args.control_address})")
    frames_note = "frames off (--no-video)" if frame_sock is None else f"frames PULL ~<:{args.frame_port}>~"
    logger.info(f" ZMQ   poses PUB ~<:{args.pose_port}>~   {frames_note}")
    logger.info(f"~<{rule}>~")

    if video is not None:
        video.start()

    running = {"v": True}

    def stop(*_):
        running["v"] = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    period = 1.0 / max(args.fps, 1.0)
    frames_forwarded = 0
    last_log = time.time()
    warned_at = 0.0
    seen_serials: set[str] = set()
    # Probe once up front: a fundamentally wrong SDK contract fails loud here, at
    # startup, rather than deep in the loop below.
    derived_body = not args.no_derived_body
    last_state = client.state(derived_body=derived_body)
    try:
        while running["v"]:
            loop_start = time.time()

            # Newest frame from Eden -> headset. Eden renders RGB and the encoder
            # takes RGB, so no channel flip is needed here.
            if video is not None:
                frame = P.recv_frame(frame_sock, flags=zmq.NOBLOCK)
                if frame is not None:
                    video.push(frame)
                    frames_forwarded += 1

            # Transient device noise (a momentarily malformed read) must not tear
            # down a live teleop session: log it at most every few seconds and
            # republish the last good state (the consumers CONFLATE to latest).
            try:
                last_state = client.state(derived_body=derived_body)
            except (ValueError, RuntimeError) as exc:
                now = time.time()
                if now - warned_at > 5.0:
                    logger.warning(f"skipping a bad PICO read: {exc}")
                    warned_at = now
            P.send_pose(pose_sock, last_state)

            # Surface each new unlabeled tracker once so the operator can map it.
            for tr in last_state.get("trackers", {}).values():
                if tr.get("role") is None and tr["serial"] not in seen_serials:
                    seen_serials.add(tr["serial"])
                    logger.info(
                        f"motion tracker {tr['serial']} is unlabeled — map it with --tracker-role {tr['serial']}=<role>"
                    )

            now = time.time()
            if now - last_log > 5.0:
                logger.info(_status(last_state, video, frames_forwarded))
                last_log = now

            dt = time.time() - loop_start
            if dt < period:
                time.sleep(period - dt)
    finally:
        logger.info("shutting down...")
        if video is not None:
            video.stop()
        client.close()


def _status(state: dict, video: PicoVideoLink | None, frames_forwarded: int) -> str:
    """One-line status from the last published state (no extra SDK reads)."""
    if not state.get("ready"):
        return "waiting for the headset (is XRoboToolkit connected, with Send enabled?)"

    body = "on" if state.get("body_ready") else "off"
    trackers = state.get("trackers", {})
    labels = ",".join(sorted(trackers)) or "none"
    parts = [
        f"tracking | trackers: {len(trackers)} [{labels}] | full-body: {body}",
        f"L grip={state['left_squeeze_value']:.2f} trig={state['left_trigger_value']:.2f}",
        f"R grip={state['right_squeeze_value']:.2f} trig={state['right_trigger_value']:.2f}",
    ]
    if video is not None:
        vstate = f"{video.frames_sent} sent" if video.streaming else "idle (tap Remote Vision → Listen)"
        parts.append(f"video: {vstate}, {frames_forwarded} received")
    return " | ".join(parts)


if __name__ == "__main__":
    main()
