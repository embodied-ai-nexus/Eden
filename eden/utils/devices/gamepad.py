"""Gamepad teleoperation device (SE2/SE3)."""

from dataclasses import dataclass
from threading import Thread, Lock
import time

import torch

# Top-level `import eden` is safe: devices/ is not on eden's package-init
# import chain, and `en.logger` is seeded at package import.
import eden as en
from eden.utils.devices.base import DeviceBase

IS_INPUTS_INSTALLED = True
try:
    from inputs import devices as input_devices, get_gamepad
except ImportError:
    IS_INPUTS_INSTALLED = False


@dataclass
class GamepadBinding:
    index: int
    scale: float = 1.0
    offset: float = 0.0
    deadzone: float = 0.0
    is_toggle: bool = False


SCALE_PS4 = 1.0 / 128  # PS4 controller provides readings from 0 ~ 255
SCALE_XBOX = 1.0 / 32768  # Xbox controller provides readings from -32768 ~ 32767
OFFSET_PS4 = 1.0
OFFSET_XBOX = 0.0
DEADZONE_PS4 = 0.005
DEADZONE_XBOX = 0.005


class Gamepad(DeviceBase):
    """
    Gamepad device.

    Parameters
    ----------
    bindings: dict[str, GamepadBinding]
        The bindings for the gamepad.
    """

    def __init__(self, bindings: dict[str, GamepadBinding] = None):
        if not IS_INPUTS_INSTALLED:
            raise ImportError("inputs is not installed. Please install it with `pip install inputs`.")
        if not input_devices.gamepads:
            # ``get_gamepad`` raises per call when nothing is plugged in, and the reader
            # thread swallows that; without this check the device looks alive and reports a
            # steady zero command forever. Say so here instead, while the caller can act.
            raise RuntimeError(
                "No gamepad found. Connect one (and check it is readable, e.g. it appears in "
                "/dev/input) before constructing a Gamepad device."
            )

        super().__init__()
        self.bindings = bindings or {}
        self.connected = True

        # Determine size of readings vector based on max index in bindings
        if self.bindings:
            max_index = max(b.index for b in self.bindings.values())
            self._readings = torch.zeros(max_index + 1, dtype=torch.float32)
            self._is_toggle_mask = torch.zeros_like(self._readings, dtype=torch.bool)
            for binding in self.bindings.values():
                if binding.is_toggle:
                    self._is_toggle_mask[binding.index] = True
        else:
            self._readings = torch.zeros(0, dtype=torch.float32)
            self._is_toggle_mask = torch.zeros(0, dtype=torch.bool)

        self.lock = Lock()
        self.running = True
        self.thread = Thread(target=self._read_inputs, daemon=True)
        self.thread.start()

    def _read_inputs(self):
        while self.running:
            try:
                events = get_gamepad()
            except Exception:
                # The pad was unplugged (or the process is exiting). Axis readings must not
                # persist: a stick held forward and a pad yanked out of the socket look
                # identical to a consumer that only sees the last value, and on a robot the
                # second one has to mean stop.
                #
                # Latched toggles are deliberately left alone. A toggle is state the operator
                # *set*, not a deflection they are holding, and on SE3Gamepad it is the
                # gripper: zeroing it would open the hand on disconnect and drop whatever it
                # was carrying. Losing the pad means "stop moving", not "let go".
                if self.connected:
                    self.connected = False
                    en.logger.warning("Lost the gamepad; zeroing the axes until it comes back (toggles are kept).")
                with self.lock:
                    self._readings[~self._is_toggle_mask] = 0.0
                time.sleep(0.1)
                continue
            self.connected = True

            with self.lock:
                for event in events:
                    if event.code in self.bindings:
                        binding = self.bindings[event.code]
                        val = float(event.state)

                        # The deadzone applies to the normalized (scaled + offset) value.
                        normalized_val = val * binding.scale + binding.offset

                        if abs(normalized_val) < binding.deadzone:
                            normalized_val = 0.0

                        if binding.is_toggle:
                            if event.state == 1:  # Only toggle on press
                                self._readings[binding.index] = not bool(self._readings[binding.index])
                        else:
                            self._readings[binding.index] = normalized_val

                    # Handle callbacks
                    if event.code in self._callbacks:
                        self._callbacks[event.code](event.state)

    def stop(self):
        self.running = False
        # Thread might be blocked on get_gamepad(), so we can't easily join it.
        # Since it's a daemon thread, it will exit when the program exits.

    def reset(self):
        with self.lock:
            self._readings.fill_(0)

    def get_raw_data(self) -> torch.Tensor:
        with self.lock:
            return self._readings.clone()

    def get_command(self) -> torch.Tensor:
        return self.get_raw_data()


class SE2Gamepad(Gamepad):
    """
    A gamepad controller for sending SE(2) commands as velocity commands.

    Default bindings (``inputs`` event codes; axis scaling defaults to a PS4 pad's 0-255 range, pass
    ``scale=SCALE_XBOX, offset=OFFSET_XBOX`` for an Xbox pad):
    - Left Stick Y: Linear Velocity X
    - Left Stick X: Linear Velocity Y
    - Right Stick X: Angular Velocity Z
    """

    def __init__(
        self,
        bindings: dict[str, GamepadBinding] = None,
        vel_max: float = 1.0,
        ang_max: float = 1.0,
        scale: float = SCALE_PS4,
        offset: float = OFFSET_PS4,
        deadzone: float = DEADZONE_PS4,
    ):
        bindings = bindings or self.default_bindings(scale, offset, deadzone)
        super().__init__(bindings)

        # Multipliers on the normalized axes, not clamps.
        self.vel_max = vel_max
        self.ang_max = ang_max

    @classmethod
    def default_bindings(cls, scale: float, offset: float = 0.0, deadzone: float = 0.0):
        return {
            "ABS_Y": GamepadBinding(0, scale=-scale, deadzone=deadzone, offset=offset),  # Y axis (inverted often)
            "ABS_X": GamepadBinding(1, scale=-scale, deadzone=deadzone, offset=offset),  # X axis
            "ABS_RX": GamepadBinding(2, scale=-scale, deadzone=deadzone, offset=offset),  # Angular Z
        }

    def get_command(self) -> torch.Tensor:
        readings = self.get_raw_data()
        cmd = torch.zeros(3, dtype=torch.float32)
        # Assuming index 0 is VX, 1 is VY, 2 is WZ based on default bindings
        if readings.shape[0] >= 3:
            cmd[0] = readings[0] * self.vel_max
            cmd[1] = readings[1] * self.vel_max
            cmd[2] = readings[2] * self.ang_max
        return cmd


class SE3Gamepad(Gamepad):
    """
    A gamepad controller for sending SE(3) commands as delta poses and binary command (open/close).

    Default bindings (``inputs`` event codes; axis scaling defaults to a PS4 pad's 0-255 range, pass
    ``scale=SCALE_XBOX, offset=OFFSET_XBOX`` for an Xbox pad):
    - Left Stick Y: X axis
    - Left Stick X: Y axis
    - Right Stick Y: Z axis
    - Right Stick X: Yaw (Rot Z)
    - D-Pad Up/Down: Pitch (Rot Y)
    - D-Pad Left/Right: Roll (Rot X)
    - A Button: Toggle Gripper

    ``pos_max`` and ``rot_max`` default to ``1.0``, i.e. this device emits a **normalized** delta
    pose and the task-space controller downstream owns the conversion to metres and radians via
    its ``position_scale`` / ``orientation_scale``. Lower them only to clamp the sticks below full
    deflection, and expect the physical step to be the product of the two.
    """

    def __init__(
        self,
        bindings: dict[str, GamepadBinding] = None,
        pos_max: float = 1.0,
        rot_max: float = 1.0,
        scale: float = SCALE_PS4,
        offset: float = OFFSET_PS4,
        deadzone: float = DEADZONE_PS4,
    ):
        bindings = bindings or self.default_bindings(scale, offset, deadzone)
        super().__init__(bindings)
        self.pos_max = pos_max
        self.rot_max = rot_max

    @classmethod
    def default_bindings(cls, scale: float, offset: float = 0.0, deadzone: float = 0.0):
        return {
            "ABS_Y": GamepadBinding(0, scale=-scale, deadzone=deadzone, offset=offset),  # Tx
            "ABS_X": GamepadBinding(1, scale=-scale, deadzone=deadzone, offset=offset),  # Ty
            "ABS_RY": GamepadBinding(2, scale=-scale, deadzone=deadzone, offset=offset),  # Tz
            # Rotation
            "ABS_HAT0X": GamepadBinding(3, scale=1.0, deadzone=0.0),  # Rx (D-pad is -1/0/1 usually)
            "ABS_HAT0Y": GamepadBinding(4, scale=-1.0, deadzone=0.0),  # Ry
            "ABS_RX": GamepadBinding(5, scale=-scale, deadzone=deadzone, offset=offset),  # Rz
            # Gripper
            "BTN_SOUTH": GamepadBinding(6, is_toggle=True),  # A button
        }

    def get_command(self) -> torch.Tensor:
        readings = self.get_raw_data()
        cmd = torch.zeros(7, dtype=torch.float32)

        if readings.shape[0] >= 7:
            # Translation
            cmd[:3] = readings[:3] * self.pos_max
            # Rotation
            cmd[3:6] = readings[3:6] * self.rot_max
            # Gripper
            cmd[6] = readings[6]

        return cmd


if __name__ == "__main__":
    import time

    # Example usage
    try:
        gamepad = SE3Gamepad()
        print("Gamepad initialized. Press Ctrl+C to exit.")
        while True:
            cmd = gamepad.get_command()
            print(f"Command: {cmd}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")
        if "gamepad" in locals():
            gamepad.stop()
