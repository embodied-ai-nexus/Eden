"""Dynamixel servo driver for the GELLO leader arm."""

from __future__ import annotations

from typing import Sequence

import numpy as np


ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
LEN_GOAL_POSITION = 4
ADDR_PRESENT_POSITION = 132
LEN_PRESENT_POSITION = 4
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0


class DynamixelDriver:
    """Minimal Dynamixel position driver for Gello leader arms.

    Reads joint positions (rad) and, when torque is enabled, commands goal positions.
    """

    def __init__(self, ids: Sequence[int], port: str, baudrate: int = 57600):
        if len(ids) == 0:
            raise ValueError("'ids' must not be empty.")

        try:
            from dynamixel_sdk.group_sync_read import GroupSyncRead
            from dynamixel_sdk.group_sync_write import GroupSyncWrite
            from dynamixel_sdk.packet_handler import PacketHandler
            from dynamixel_sdk.port_handler import PortHandler
            from dynamixel_sdk.robotis_def import COMM_SUCCESS
            from dynamixel_sdk.robotis_def import DXL_HIBYTE, DXL_HIWORD, DXL_LOBYTE, DXL_LOWORD
        except ImportError as exc:
            raise ImportError("dynamixel_sdk is required. Install with `pip install dynamixel-sdk`.") from exc

        self._ids = tuple(int(i) for i in ids)
        self._port_handler = PortHandler(port)
        self._packet_handler = PacketHandler(2.0)
        self._group_read = GroupSyncRead(
            self._port_handler, self._packet_handler, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
        )
        self._group_write = GroupSyncWrite(
            self._port_handler, self._packet_handler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION
        )
        self._comm_success = COMM_SUCCESS
        self._byte_pack = (DXL_LOBYTE, DXL_HIBYTE, DXL_LOWORD, DXL_HIWORD)
        self._torque_enabled = False

        if not self._port_handler.openPort():
            raise RuntimeError(f"Failed to open Dynamixel port: {port}")
        if not self._port_handler.setBaudRate(int(baudrate)):
            raise RuntimeError(f"Failed to set Dynamixel baudrate: {baudrate}")

        for dxl_id in self._ids:
            if not self._group_read.addParam(dxl_id):
                raise RuntimeError(f"Failed to add group read param for ID {dxl_id}")

    @property
    def ids(self) -> tuple[int, ...]:
        return self._ids

    def set_torque_mode(self, enable: bool):
        target = TORQUE_ENABLE if enable else TORQUE_DISABLE
        for dxl_id in self._ids:
            comm_result, error = self._packet_handler.write1ByteTxRx(
                self._port_handler, dxl_id, ADDR_TORQUE_ENABLE, target
            )
            self._raise_if_comm_failed(comm_result, error, f"set torque mode for ID {dxl_id}")
        self._torque_enabled = bool(enable)

    def get_positions(self) -> np.ndarray:
        comm_result = self._group_read.txRxPacket()
        if comm_result != self._comm_success:
            raise RuntimeError(f"Failed to read joint positions: {self._packet_handler.getTxRxResult(comm_result)}")

        positions = np.zeros(len(self._ids), dtype=np.float64)
        for i, dxl_id in enumerate(self._ids):
            if not self._group_read.isAvailable(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
                raise RuntimeError(f"Position data is not available for ID {dxl_id}")
            raw = self._group_read.getData(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
            signed = raw - (1 << 32) if raw & (1 << 31) else raw
            positions[i] = signed * np.pi / 2048.0
        return positions

    def set_positions(self, joint_angles: Sequence[float]):
        if not self._torque_enabled:
            raise RuntimeError("Torque must be enabled before commanding joint positions.")

        joint_angles_arr = np.asarray(joint_angles, dtype=np.float64).reshape(-1)
        if joint_angles_arr.size != len(self._ids):
            raise ValueError(f"Expected {len(self._ids)} joint angles, got {joint_angles_arr.size}.")

        lo_byte, hi_byte, lo_word, hi_word = self._byte_pack
        try:
            for dxl_id, angle in zip(self._ids, joint_angles_arr):
                value = int(np.round(angle * 2048.0 / np.pi)) & 0xFFFFFFFF
                data = [
                    lo_byte(lo_word(value)),
                    hi_byte(lo_word(value)),
                    lo_byte(hi_word(value)),
                    hi_byte(hi_word(value)),
                ]
                if not self._group_write.addParam(dxl_id, data):
                    raise RuntimeError(f"Failed to add write param for ID {dxl_id}")
            comm_result = self._group_write.txPacket()
        finally:
            self._group_write.clearParam()

        if comm_result != self._comm_success:
            raise RuntimeError(f"Failed to write joint positions: {self._packet_handler.getTxRxResult(comm_result)}")

    def close(self):
        handler = getattr(self, "_port_handler", None)
        if handler is not None:
            handler.closePort()

    def _raise_if_comm_failed(self, comm_result: int, error: int, context: str):
        if comm_result != self._comm_success:
            raise RuntimeError(f"Failed to {context}: {self._packet_handler.getTxRxResult(comm_result)}")
        if error != 0:
            raise RuntimeError(
                f"Dynamixel error while trying to {context}: {self._packet_handler.getRxPacketError(error)}"
            )
