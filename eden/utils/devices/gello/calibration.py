"""GELLO leader-arm joint-offset calibration."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from eden.utils.format import (
    ANSI_BOLD,
    ANSI_CYAN,
    ANSI_DIM,
    ANSI_GREEN,
    ANSI_MAGENTA,
    ANSI_YELLOW,
    ColorHelpFormatter,
    colorize,
)


def get_offset(
    current_dofs: Sequence[float] | np.ndarray,
    start_dofs: Sequence[float] | np.ndarray,
    signs: Sequence[float] | np.ndarray,
    max_pi_multiple: int = 8,
) -> np.ndarray:
    """Search per-joint offsets (multiples of pi/2) that align current DOFs with known start DOFs.

    For each joint i, picks the offset that minimizes
    ``abs(signs[i] * (current_dofs[i] - offset) - start_dofs[i])``
    over ``offset`` in ``[-max_pi_multiple * pi, +max_pi_multiple * pi]`` stepped by pi/2.
    """
    if max_pi_multiple <= 0:
        raise ValueError("'max_pi_multiple' must be greater than 0.")

    current_arr = np.asarray(current_dofs, dtype=np.float64).reshape(-1)
    start_arr = np.asarray(start_dofs, dtype=np.float64).reshape(-1)
    sign_arr = np.asarray(signs, dtype=np.float64).reshape(-1)

    if start_arr.size == 0:
        raise ValueError("'start_dofs' must not be empty.")
    if start_arr.size != sign_arr.size or start_arr.size != current_arr.size:
        raise ValueError("'current_dofs', 'start_dofs', and 'signs' must have the same length.")
    if not np.all(np.isin(sign_arr, (-1.0, 1.0))):
        raise ValueError("'signs' must only contain -1 or 1.")

    candidates = np.linspace(
        -max_pi_multiple * np.pi,
        max_pi_multiple * np.pi,
        max_pi_multiple * 4 + 1,
    )

    errors = np.abs(sign_arr[:, None] * (current_arr[:, None] - candidates[None, :]) - start_arr[:, None])
    return candidates[np.argmin(errors, axis=1)]


def format_offsets_as_pi_half_terms(offsets: Sequence[float] | np.ndarray) -> list[str]:
    """Return each offset formatted as ``k*np.pi/2``."""
    arr = np.asarray(offsets, dtype=np.float64).reshape(-1)
    return [f"{int(np.round(v / (np.pi / 2.0)))}*np.pi/2" for v in arr]


def estimate_gripper_range_deg(
    current_gripper_rad: float,
    open_delta_deg: float = -0.2,
    close_delta_deg: float = -42.0,
) -> tuple[float, float]:
    """Estimate gripper open/close positions (degrees) from the current resting position.

    The gripper is assumed to be held near its open position during calibration.
    """
    current_deg = float(np.rad2deg(float(current_gripper_rad)))
    return current_deg + float(open_delta_deg), current_deg + float(close_delta_deg)


def calibrate_gello_leader(argv: list[str] | None = None):
    from argparse import ArgumentParser
    from pathlib import Path

    from eden.utils.devices.gello.dynamixel_driver import DynamixelDriver
    from eden.utils.devices.gello.leader import GelloLeaderConfig

    description = (
        f"{colorize('Gello leader arm calibration tool.', ANSI_BOLD, ANSI_CYAN)}\n"
        "\n"
        "Hold the leader arm at a known joint configuration (passed via "
        f"{colorize('--start-dofs', ANSI_GREEN)},\n"
        "in radians), then run this tool to compute per-joint offsets and a gripper\n"
        "open/close range. The result is printed as a ready-to-paste\n"
        f"{colorize('GelloLeaderConfig(...)', ANSI_MAGENTA)}."
    )
    epilog = (
        f"{colorize('Examples:', ANSI_BOLD, ANSI_YELLOW)}\n"
        f"  {colorize('# 6-DoF arm (IDs 1..6) at zero pose, with a gripper on ID 7 (default N+1):', ANSI_DIM)}\n"
        "  eden gello calibrate \\\n"
        f"      {colorize('--port', ANSI_GREEN)} /dev/ttyUSB0 \\\n"
        f"      {colorize('--start-dofs', ANSI_GREEN)} 0 0 0 0 0 0 \\\n"
        f"      {colorize('--signs', ANSI_GREEN)} 1 1 -1 1 1 1 \\\n"
        f"      {colorize('--gripper', ANSI_GREEN)}\\\n"
        f"      {colorize('--output', ANSI_GREEN)} configs/gello_ur5e.yaml\n"
        "\n"
        f"  {colorize('# Custom Dynamixel IDs and high-speed baudrate, no gripper:', ANSI_DIM)}\n"
        "  eden gello calibrate \\\n"
        f"      {colorize('--port', ANSI_GREEN)} /dev/ttyUSB0 \\\n"
        f"      {colorize('--start-dofs', ANSI_GREEN)} 0 0 0 0 0 0 \\\n"
        f"      {colorize('--signs', ANSI_GREEN)} 1 1 1 1 1 1 \\\n"
        f"      {colorize('--joint-ids', ANSI_GREEN)} 1 2 4 6 7 8 \\\n"
        f"      {colorize('--baudrate', ANSI_GREEN)} 1000000"
    )
    parser = ArgumentParser(
        prog="eden gello calibrate",
        description=description,
        epilog=epilog,
        formatter_class=ColorHelpFormatter,
    )
    parser.add_argument("--port", type=str, required=True, help="Dynamixel serial port (e.g. /dev/ttyUSB0).")
    parser.add_argument(
        "--start-dofs", type=float, nargs="+", required=True, help="Known arm DOFs (rad) at calibration pose."
    )
    parser.add_argument("--signs", type=float, nargs="+", required=True, help="Sign (-1 or 1) per arm DOF.")
    parser.add_argument(
        "--joint-ids", type=int, nargs="+", default=None, help="Dynamixel IDs for arm DOFs. Default: 1..N."
    )
    parser.add_argument("--gripper", action="store_true", help="Whether the gripper is attached.")
    parser.add_argument("--gripper-id", type=int, default=None, help="Dynamixel ID for the gripper. Default: N+1.")
    parser.add_argument("--baudrate", type=int, default=57600)
    parser.add_argument("--warmup-reads", type=int, default=10, help="Read iterations before sampling offsets.")
    parser.add_argument("--max-pi-multiple", type=int, default=8)
    parser.add_argument("--gripper-open-delta-deg", type=float, default=-0.2)
    parser.add_argument("--gripper-close-delta-deg", type=float, default=-42.0)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Save the resulting GelloLeaderConfig to this path (format inferred from .json, .yaml, or .yml).",
    )
    args = parser.parse_args(argv)

    if args.output is not None and args.output.suffix.lower() not in (".json", ".yaml", ".yml"):
        parser.error(f"--output must end in .json, .yaml, or .yml (got {args.output.suffix!r}).")

    start_dofs = np.asarray(args.start_dofs, dtype=np.float64)
    signs = np.asarray(args.signs, dtype=np.float64)
    if start_dofs.size != signs.size:
        raise ValueError("'--start-dofs' and '--signs' must have the same length.")

    num_arm = start_dofs.size
    joint_ids = args.joint_ids if args.joint_ids is not None else list(range(1, num_arm + 1))
    if len(joint_ids) != num_arm:
        raise ValueError(f"'--joint-ids' must have length {num_arm}, got {len(joint_ids)}.")

    ids = list(joint_ids)
    if args.gripper:
        gripper_id = int(args.gripper_id) if args.gripper_id is not None else num_arm + 1
        if gripper_id in ids:
            raise ValueError("'--gripper-id' must not overlap with arm joint IDs.")
        ids.append(gripper_id)

    driver = DynamixelDriver(ids=ids, port=args.port, baudrate=args.baudrate)
    try:
        for _ in range(max(0, args.warmup_reads)):
            driver.get_positions()

        positions = driver.get_positions()
        arm_positions = positions[:num_arm]
        offsets = get_offset(
            current_dofs=arm_positions,
            start_dofs=start_dofs,
            signs=signs,
            max_pi_multiple=args.max_pi_multiple,
        )

        def _label(text: str) -> str:
            return colorize(text, ANSI_CYAN)

        def _value(text: str) -> str:
            return colorize(text, ANSI_YELLOW)

        def _key(text: str) -> str:
            return colorize(text, ANSI_GREEN)

        print(f"{_label('current arm dofs (rad)    :')} {_value(str([f'{v:.6f}' for v in arm_positions.tolist()]))}")
        print(f"{_label('best offsets (rad)        :')} {_value(str([f'{v:.6f}' for v in offsets.tolist()]))}")
        print(f"{_label('best offsets (pi/2 terms) :')} {_value(str(format_offsets_as_pi_half_terms(offsets)))}")

        gripper_block = ""
        if args.gripper:
            open_deg, close_deg = estimate_gripper_range_deg(
                current_gripper_rad=float(positions[-1]),
                open_delta_deg=args.gripper_open_delta_deg,
                close_delta_deg=args.gripper_close_delta_deg,
            )
            print(f"{_label('gripper current (deg)     :')} {_value(f'{np.rad2deg(positions[-1]):.6f}')}")
            print(f"{_label('gripper open (deg)        :')} {_value(f'{open_deg:.6f}')}")
            print(f"{_label('gripper close (deg)       :')} {_value(f'{close_deg:.6f}')}")
            gripper_block = (
                f"    {_key('gripper_id')}={_value(str(gripper_id))},\n"
                f"    {_key('gripper_open_deg')}={_value(f'{open_deg:.6f}')},\n"
                f"    {_key('gripper_close_deg')}={_value(f'{close_deg:.6f}')},\n"
            )

        config = GelloLeaderConfig(
            joint_ids=[int(v) for v in joint_ids],
            joint_offsets_rad=[float(v) for v in offsets],
            joint_signs=[int(s) for s in signs],
            gripper_id=gripper_id if args.gripper else None,
            gripper_open_deg=(open_deg if args.gripper else None),
            gripper_close_deg=(close_deg if args.gripper else None),
            baudrate=args.baudrate,
        )

        print(f"\n{colorize('GelloLeaderConfig(', ANSI_BOLD, ANSI_MAGENTA)}")
        print(f"    {_key('joint_ids')}={_value(str(tuple(config.joint_ids)))},")
        print(f"    {_key('joint_offsets_rad')}={_value(str(tuple(config.joint_offsets_rad)))},")
        print(f"    {_key('joint_signs')}={_value(str(tuple(config.joint_signs)))},")
        if gripper_block:
            print(gripper_block, end="")
        print(f"    {_key('baudrate')}={_value(str(config.baudrate))},")
        print(colorize(")", ANSI_BOLD, ANSI_MAGENTA))

        if args.output is not None:
            saved = config.save(args.output)
            print(f"\n{colorize('Saved config to:', ANSI_BOLD, ANSI_GREEN)} {saved}")
    finally:
        driver.close()
