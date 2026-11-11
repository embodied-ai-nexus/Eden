"""GELLO leader-arm runtime and configuration."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from eden.utils.format import (
    ANSI_BOLD,
    ANSI_CYAN,
    ANSI_GREEN,
    ANSI_YELLOW,
    ColorHelpFormatter,
    colorize,
)

from eden.utils.devices.gello.dynamixel_driver import DynamixelDriver


@dataclass
class GelloLeaderConfig:
    """Calibration config for a single-arm Gello leader.

    Attributes
    ----------
    joint_ids:
        Dynamixel IDs for the arm DOFs, in controller order.
    joint_offsets_rad:
        Offset (rad) subtracted from each raw joint reading. Typically multiples of pi/2.
    joint_signs:
        +1 or -1 per arm DOF to match the follower's positive rotation direction.
    gripper_id:
        (Optional) Dynamixel ID for the gripper DOF.
    gripper_open_deg:
        (Optional) Gripper position in degrees when fully open.
    gripper_close_deg:
        (Optional) Gripper position in degrees when fully closed.
    baudrate:
        Serial baudrate for Dynamixel communication.
    """

    joint_ids: Sequence[int]
    joint_offsets_rad: Sequence[float]
    joint_signs: Sequence[int]
    gripper_id: int | None = None
    gripper_open_deg: float | None = None
    gripper_close_deg: float | None = None
    baudrate: int = 57600

    def __post_init__(self):
        if len(self.joint_ids) != len(self.joint_offsets_rad):
            raise ValueError("'joint_ids' and 'joint_offsets_rad' must have the same length.")
        if len(self.joint_ids) != len(self.joint_signs):
            raise ValueError("'joint_ids' and 'joint_signs' must have the same length.")
        if not all(s in (-1, 1) for s in self.joint_signs):
            raise ValueError("'joint_signs' entries must be -1 or 1.")
        has_gripper = self.gripper_id is not None
        has_range = self.gripper_open_deg is not None and self.gripper_close_deg is not None
        if has_gripper != has_range:
            raise ValueError("'gripper_id', 'gripper_open_deg', and 'gripper_close_deg' must be set together.")

    @property
    def has_gripper(self) -> bool:
        return self.gripper_id is not None

    @property
    def num_arm_dofs(self) -> int:
        return len(self.joint_ids)

    @property
    def num_dofs(self) -> int:
        return self.num_arm_dofs + (1 if self.has_gripper else 0)

    @property
    def all_ids(self) -> list[int]:
        return [*self.joint_ids, self.gripper_id] if self.has_gripper else list(self.joint_ids)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["joint_ids"] = [int(v) for v in self.joint_ids]
        data["joint_offsets_rad"] = [float(v) for v in self.joint_offsets_rad]
        data["joint_signs"] = [int(v) for v in self.joint_signs]
        return data

    def save(self, path: str | Path) -> Path:
        """Save config to ``path``. Format is inferred from the extension (.json, .yaml, .yml)."""
        path = Path(path)
        text = _dump_config(self.to_dict(), path.suffix.lower())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GelloLeaderConfig:
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path) -> GelloLeaderConfig:
        """Load config from ``path``. Format is inferred from the extension (.json, .yaml, .yml)."""
        path = Path(path)
        return cls.from_dict(_load_config(path.read_text(encoding="utf-8"), path.suffix.lower()))


def _dump_config(data: dict[str, Any], suffix: str) -> str:
    if suffix == ".json":
        return json.dumps(data, indent=2) + "\n"
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to save YAML configs. Install with `pip install pyyaml`.") from exc
        return yaml.safe_dump(data, sort_keys=False)
    raise ValueError(f"Unsupported config extension: {suffix!r}. Expected .json, .yaml, or .yml.")


def _load_config(text: str, suffix: str) -> dict[str, Any]:
    if suffix == ".json":
        return json.loads(text)
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to load YAML configs. Install with `pip install pyyaml`.") from exc
        return yaml.safe_load(text)
    raise ValueError(f"Unsupported config extension: {suffix!r}. Expected .json, .yaml, or .yml.")


class GelloLeader:
    """Gello leader agent: reads calibrated joint positions from Dynamixel hardware.

    The arm DOFs are returned in radians after applying
    ``(raw - offset) * sign``. When a gripper is configured, the gripper DOF is
    normalized to ``[0, 1]`` with ``0 == open`` and ``1 == closed``.
    """

    def __init__(
        self,
        port: str,
        config: GelloLeaderConfig,
        start_dofs: np.ndarray | Sequence[float] | None = None,
    ):
        self._config = config
        self._driver = DynamixelDriver(ids=config.all_ids, port=port, baudrate=config.baudrate)
        self._driver.set_torque_mode(False)

        arm_offsets = np.asarray(config.joint_offsets_rad, dtype=np.float64)
        arm_signs = np.asarray(config.joint_signs, dtype=np.float64)
        if config.has_gripper:
            self._offsets = np.concatenate([arm_offsets, np.zeros(1)])
            self._signs = np.concatenate([arm_signs, np.ones(1)])
            self._gripper_range_rad = (
                float(np.deg2rad(config.gripper_open_deg)),
                float(np.deg2rad(config.gripper_close_deg)),
            )
        else:
            self._offsets = arm_offsets
            self._signs = arm_signs
            self._gripper_range_rad = None

        if start_dofs is not None:
            self._align_offsets_to_start(np.asarray(start_dofs, dtype=np.float64).reshape(-1))

    @property
    def num_dofs(self) -> int:
        return self._config.num_dofs

    @property
    def config(self) -> GelloLeaderConfig:
        return self._config

    def get_command(self) -> np.ndarray:
        """Return calibrated arm DOFs (rad) concatenated with a normalized gripper value (if any)."""
        raw = self._driver.get_positions()
        dofs = (raw - self._offsets) * self._signs
        if self._gripper_range_rad is not None:
            dofs[-1] = self._normalize_gripper(dofs[-1])
        return dofs

    def close(self):
        self._driver.close()

    def _align_offsets_to_start(self, start_dofs: np.ndarray):
        num_arm = self._config.num_arm_dofs
        if start_dofs.size == num_arm + 1:
            start_dofs = start_dofs[:-1]
        if start_dofs.size != num_arm:
            raise ValueError(f"'start_dofs' must have length {num_arm} or {num_arm + 1}, got {start_dofs.size}.")

        raw_arm = self._driver.get_positions()[:num_arm]
        current_arm = (raw_arm - self._offsets[:num_arm]) * self._signs[:num_arm]
        two_pi = 2.0 * np.pi
        delta = two_pi * np.round((current_arm - start_dofs) / two_pi) * self._signs[:num_arm]
        self._offsets[:num_arm] = self._offsets[:num_arm] + delta

    def _normalize_gripper(self, gripper_rad: float) -> float:
        g_open, g_close = self._gripper_range_rad
        if abs(g_close - g_open) < 1e-8:
            raise RuntimeError("Invalid gripper calibration: open and close positions are identical.")
        return float(np.clip((gripper_rad - g_open) / (g_close - g_open), 0.0, 1.0))


def run_gello_leader(argv: list[str] | None = None):
    from argparse import ArgumentParser
    from pathlib import Path

    from eden.utils.devices.gello.zmq_client import GelloClient

    description = (
        f"{colorize('Gello leader arm teleoperation tool.', ANSI_BOLD, ANSI_CYAN)}\n"
        "\n"
        "Run this tool to continuously read calibrated joint positions from Dynamixel hardware\n"
        "and send them to a GelloZMQServer over ZMQ. The arm DOFs are returned in radians after\n"
        "applying (raw - offset) * sign. When a gripper is configured, the gripper DOF is\n"
        "normalized to [0, 1] with 0 == open and 1 == closed."
    )
    epilog = (
        f"{colorize('Examples:', ANSI_BOLD, ANSI_YELLOW)}\n"
        "  eden gello leader \\\n"
        f"      {colorize('-c', ANSI_GREEN)} configs/gello_ur5e.yaml \\\n"
        f"      {colorize('--port', ANSI_GREEN)} /dev/ttyUSB0 \\\n"
        f"      {colorize('--zmq-host', ANSI_GREEN)} 127.0.0.1 \\\n"
        f"      {colorize('--zmq-port', ANSI_GREEN)} 6001"
    )

    parser = ArgumentParser(
        prog="eden gello leader",
        description=description,
        epilog=epilog,
        formatter_class=ColorHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="Path to the GelloLeaderConfig file. First run 'eden gello calibrate' to create a config file.",
    )
    parser.add_argument("--port", type=str, required=True, help="Dynamixel serial port (e.g. /dev/ttyUSB0).")
    parser.add_argument("--zmq-host", type=str, help="ZMQ host (e.g. 127.0.0.1).", default="127.0.0.1")
    parser.add_argument("--zmq-port", type=int, help="ZMQ port (e.g. 6001).", default=6001)

    args = parser.parse_args(argv)

    config = GelloLeaderConfig.load(args.config)
    leader = GelloLeader(port=args.port, config=config, start_dofs=np.zeros(config.num_dofs))
    client = GelloClient(host=args.zmq_host, port=args.zmq_port)

    assert client.num_dofs() == leader.num_dofs

    print(f"{colorize('Gello leader arm teleoperation tool started.', ANSI_BOLD, ANSI_CYAN)}")

    try:
        while True:
            client.command_joint_state(leader.get_command())
            time.sleep(1 / 100)  # 100 Hz
    finally:
        leader.close()
        client.close()
        print(f"{colorize('Gello leader arm teleoperation tool stopped.', ANSI_BOLD, ANSI_CYAN)}")
