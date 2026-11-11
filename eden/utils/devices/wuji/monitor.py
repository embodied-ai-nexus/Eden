"""``eden wuji`` — find a Wuji Glove and watch its hand-pose / tactile streams live.

The first thing to run after plugging a glove in: it proves the SDK sees the device,
that the glove is solving a hand pose, and that the tactile sheet responds to a touch,
without building a Genesis scene.

Run::

    eden wuji scan                                  # list the gloves on the bus
    eden wuji monitor                               # live status, Ctrl-C to stop
    eden wuji monitor --side right --seconds 30
    eden wuji monitor --streams hand_skeleton tactile_zones --rate-hz 30
    # or, equivalently:
    uv run python -m eden.utils.devices.wuji.monitor monitor
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import eden as en
from eden.extensions.deployment.utils.rate import RateLimiter
from eden.utils.devices.wuji import protocol as P
from eden.utils.devices.wuji.glove import DEFAULT_STREAMS, STREAMS, WujiGlove, scan_gloves
from eden.utils.format import ColorHelpFormatter

REFRESH_HZ = 5.0
LOADED_THRESHOLD = 0.05  # raw pressure above which a taxel is worth counting in the status line


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eden wuji",
        description="Wuji Glove discovery and a live stream monitor.",
        formatter_class=ColorHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("scan", help="list every Wuji Glove on the bus", formatter_class=ColorHelpFormatter)

    monitor = sub.add_parser(
        "monitor",
        help="connect to a glove and print its live hand pose and tactile state",
        formatter_class=ColorHelpFormatter,
    )
    monitor.add_argument("--sn", default=None, help="serial number of the glove (default: the only one found)")
    monitor.add_argument("--side", default=None, choices=["left", "right"], help="pick between two connected gloves")
    monitor.add_argument(
        "--streams",
        nargs="+",
        default=list(DEFAULT_STREAMS),
        choices=list(STREAMS),
        help=f"streams to subscribe to (default: {' '.join(DEFAULT_STREAMS)})",
    )
    monitor.add_argument("--rate-hz", type=int, default=None, help="lower the device push rate (default: 120 Hz)")
    monitor.add_argument("--seconds", type=float, default=0.0, help="run for this long (0 = until Ctrl-C)")
    monitor.add_argument("--exclusive", action="store_true", help="take the glove for this process alone")
    return parser


def _pose_line(glove: WujiGlove) -> str:
    """Fingertip reach, in cm from the wrist landmark — a one-glance liveness check."""
    keypoints = glove.keypoints()
    if keypoints is None:
        return "hand pose: waiting for a frame"
    tips = keypoints[list(P.FINGERTIP_LANDMARKS)] - keypoints[0]
    reach = torch.linalg.norm(tips, dim=-1) * 100.0
    spread = "  ".join(f"{name} {d:5.1f}" for name, d in zip(P.FINGERS, reach.tolist()))
    return f"tip reach (cm): {spread}"


def _sheet_line(name: str, grid: torch.Tensor | None) -> str:
    """Loaded-taxel count and peak of one nan-masked 24x31 sheet."""
    if grid is None:
        return f"{name}: waiting for a frame"
    values = grid.numpy()
    valid = ~np.isnan(values)
    if not valid.any():
        return f"{name}: every taxel masked"
    peak = float(np.nanmax(values))
    row, col = np.unravel_index(int(np.nanargmax(values)), values.shape)
    # Raw pressure, not detected contact: an unworn glove still reads well above zero over
    # much of the sheet from its own resting deformation. Contact detection is what
    # `tactile_binary` is for, and that needs a calibration.
    loaded = int((values[valid] > LOADED_THRESHOLD).sum())
    return (
        f"{name}: {loaded:3d}/{int(valid.sum())} taxels >{LOADED_THRESHOLD}   "
        f"peak {peak:.3f} @ (r{row:02d}, c{col:02d})"
    )


def _zone_peak(values: torch.Tensor) -> str:
    """A zone's peak pressure, or ``--`` when every one of its taxels is masked."""
    live = values[~torch.isnan(values)]
    return f"{float(live.max()):.2f}" if live.numel() else "--"


def _zones_line(zones: dict[str, torch.Tensor] | None) -> str:
    """Peak pressure per hand region — the fallback when no full sheet is subscribed."""
    if zones is None:
        return "tactile_zones: waiting for a frame"
    return "tactile_zones: " + "  ".join(f"{z} {_zone_peak(v)}" for z, v in zones.items())


def _tactile_line(glove: WujiGlove) -> str:
    """Summarise whichever tactile stream is subscribed, a full sheet in preference to zones."""
    # `tactile_residual` is a sheet too, and is deliberately not offered: it is signed and
    # baseline-relative, so counting cells over LOADED_THRESHOLD would mean nothing.
    for name, read in (("tactile", glove.tactile), ("tactile_binary", glove.tactile_binary)):
        if name in glove.streams:
            return _sheet_line(name, read())
    if "tactile_zones" in glove.streams:
        return _zones_line(glove.tactile_zones())
    if "tactile_residual" in glove.streams:
        return "tactile_residual: subscribed (signed and baseline-relative — no contact summary)"
    return "tactile: not subscribed"


def _run_monitor(args: argparse.Namespace) -> int:
    glove = WujiGlove(
        sn=args.sn,
        side=args.side,
        streams=args.streams,
        rate_hz=args.rate_hz,
        exclusive=args.exclusive,
    )
    rule = "━" * 72
    en.logger.info(f"~<{rule}>~")
    en.logger.info(f" 🧤 ~~~~<Wuji Glove {glove.serial_number}>~~~~  ({glove.side} hand)")
    en.logger.info(f" streams: ~<{' '.join(glove.streams)}>~")
    en.logger.info(" wiggle your fingers, then press something — Ctrl-C to stop")
    en.logger.info(f"~<{rule}>~")

    deadline = time.monotonic() + args.seconds if args.seconds > 0 else None
    rate = RateLimiter(REFRESH_HZ)
    try:
        while deadline is None or time.monotonic() < deadline:
            state = "tracking" if glove.tracking else "waiting for the glove to solve a pose"
            en.logger.info(f"[{state}] {_pose_line(glove)} | {_tactile_line(glove)}")
            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        glove.close()
    return 0


def _run_scan() -> int:
    gloves = scan_gloves()
    if not gloves:
        en.logger.warning("no Wuji Glove found on the bus")
        return 1
    en.logger.info(f"found {len(gloves)} Wuji Glove(s):")
    for entry in gloves:
        en.logger.info(f"  ~<{entry['sn']}>~  {entry['address']}  ({entry['transport']})")
    en.logger.info("the glove's side is only readable over a connection — see `eden wuji monitor`")
    return 0


def main(argv=None) -> int:
    """Run ``eden wuji scan`` or ``eden wuji monitor`` (parses ``argv``, or ``sys.argv`` when ``None``)."""
    args = _build_parser().parse_args(argv)
    return _run_scan() if args.command == "scan" else _run_monitor(args)


if __name__ == "__main__":
    raise SystemExit(main())
