"""Keyboard teleoperation device (SE2/SE3)."""

import threading
from dataclasses import dataclass

import torch

from eden.utils.devices.base import DeviceBase

IS_PYNPUT_INSTALLED = True
PYNPUT_IMPORT_ERROR: Exception | None = None
try:
    from pynput import keyboard
    from pynput.keyboard import Key
except ImportError as exc:
    # Not only a missing package: on a headless Linux box pynput raises ImportError
    # from its backend selection ("this platform is not supported"), which reads as a
    # packaging problem unless the original message survives. Keep it for the raise.
    IS_PYNPUT_INSTALLED = False
    PYNPUT_IMPORT_ERROR = exc


@dataclass
class KeyBinding:
    index: int
    value: int
    reset_on_release: bool = True
    is_toggle: bool = False


class Keyboard(DeviceBase):
    """
    Keyboard device.

    Parameters
    ----------
    key_bindings: dict[str, KeyBinding]
        The key bindings for the keyboard.
    """

    def __init__(self, key_bindings: dict[str, KeyBinding] = None):
        super().__init__()
        self._require_pynput()

        self.key_bindings = key_bindings or {
            "w": KeyBinding(0, +1),
            "s": KeyBinding(0, -1),
            "a": KeyBinding(1, +1),
            "d": KeyBinding(1, -1),
            "q": KeyBinding(2, +1),
            "e": KeyBinding(2, -1),
            Key.space: KeyBinding(3, 1, is_toggle=True),
        }

        if self.key_bindings:
            max_index = max(v.index for v in self.key_bindings.values())
            self._readings = torch.zeros(max_index + 1, dtype=torch.float32)
        else:
            self._readings = torch.zeros(0, dtype=torch.float32)
        self._is_toggle_mask = torch.zeros_like(self._readings, dtype=torch.bool)
        for key_binding in self.key_bindings.values():
            if key_binding.is_toggle:
                self._is_toggle_mask[key_binding.index] = True
        self.lock = threading.Lock()
        self.active_keys = set()

        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.daemon = True
        self.listener.start()

    @classmethod
    def _require_pynput(cls):
        if not IS_PYNPUT_INSTALLED:
            raise ImportError(
                "pynput could not be imported, so keyboard input is unavailable "
                f"({PYNPUT_IMPORT_ERROR}). Install it with `pip install pynput`; if it is already "
                "installed, this is a headless session — pynput reads the keyboard through the "
                "windowing system, so it needs a local display, not an SSH shell."
            )

    def _on_press(self, key):
        k = key.char.lower() if hasattr(key, "char") and key.char else key
        if k in self.key_bindings:
            with self.lock:
                self.active_keys.add(k)
                if self.key_bindings[k].is_toggle:
                    self._readings[self.key_bindings[k].index] = not bool(self._readings[self.key_bindings[k].index])
                else:
                    self._readings[self.key_bindings[k].index] += self.key_bindings[k].value

        if k in self._callbacks:
            with self.lock:
                self.active_keys.add(k)
                self._callbacks[k](True)

    def _on_release(self, key):
        k = key.char.lower() if hasattr(key, "char") and key.char else key
        if k in self.key_bindings:
            with self.lock:
                self.active_keys.discard(k)
                key_binding = self.key_bindings[k]
                if key_binding.reset_on_release and not key_binding.is_toggle:
                    self._readings[key_binding.index] = 0

        if k in self._callbacks:
            with self.lock:
                self.active_keys.discard(k)
                self._callbacks[k](False)

    def reset(self) -> None:
        with self.lock:
            self.active_keys.clear()
            self._readings.fill_(0)

    def stop(self) -> None:
        """Stop listening and zero the readings. Idempotent.

        Mirrors :meth:`~eden.utils.devices.gamepad.Gamepad.stop` so a caller can
        release either device without knowing which it holds.
        """
        self.listener.stop()
        self.reset()

    def get_raw_data(self) -> torch.Tensor:
        return self._readings

    def get_command(self) -> torch.Tensor:
        return self._readings


class SE2Keyboard(Keyboard):
    """
    A keyboard controller for sending SE(2) commands as velocity commands.

    Parameters
    ----------
    key_bindings: dict[str, KeyBinding]
        The key bindings for the keyboard.
    vel_max: float
        The maximum value of the velocity commands.
    ang_max: float
        The maximum value of the angular velocity commands.

    ====================== ================= =================
    Command                Key (+ve axis)    Key (-ve axis)
    ====================== ================= =================
    Move along x-axis      8                 2
    Move along y-axis      4                 6
    Rotate along z-axis    7                 9
    ====================== ================= =================
    """

    def __init__(
        self,
        key_bindings: dict[str, KeyBinding] = None,
        vel_max: float = 1.0,
        ang_max: float = 1.0,
    ):
        key_bindings = key_bindings or self.default_key_bindings()
        super().__init__(key_bindings)
        self.limit = torch.tensor(
            [
                vel_max,
            ]
            * 2
            + [
                ang_max,
            ]
            * 1,
            dtype=torch.float32,
        )

    @classmethod
    def default_key_bindings(cls):
        return {
            "8": KeyBinding(0, +1),
            "2": KeyBinding(0, -1),
            "4": KeyBinding(1, +1),
            "6": KeyBinding(1, -1),
            "7": KeyBinding(2, +1),
            "9": KeyBinding(2, -1),
        }

    @classmethod
    def wasd_key_bindings(cls):
        return {
            "w": KeyBinding(0, +1),
            "s": KeyBinding(0, -1),
            "a": KeyBinding(1, +1),
            "d": KeyBinding(1, -1),
            "q": KeyBinding(2, +1),
            "e": KeyBinding(2, -1),
        }

    def get_command(self) -> torch.Tensor:
        return torch.where(
            self._is_toggle_mask,
            self._readings,
            torch.tanh(self._readings) * self.limit,
        )


class TeleopSE2Keyboard(SE2Keyboard):
    """:class:`SE2Keyboard` whose deflection saturates instead of ramping.

    ``SE2Keyboard`` smooths its readings through ``tanh``. Because :meth:`Keyboard._on_press`
    *accumulates* (``+=``) and the OS delivers key-repeat while a key is held, that makes the
    reading a ramp: one press is ``tanh(1) = 0.76``, and full deflection is only approached as
    repeats pile up — so the commanded velocity depends on the machine's key-repeat rate and
    on how long the operator held the key. Fine for nudging a camera, wrong for a velocity
    command a robot acts on, where the same gesture has to mean the same speed everywhere.

    Here a held key is full deflection from the first press, exactly like a gamepad stick
    pushed to its stop, and releasing still returns to zero (``reset_on_release``). The
    ceiling that deflection maps to is the trained range, which
    :func:`~eden.utils.devices.se2.scale_se2_deflection` owns.
    """

    def get_command(self) -> torch.Tensor:
        return torch.where(
            self._is_toggle_mask,
            self._readings,
            self._readings.clamp(-1.0, 1.0) * self.limit,
        )


class SE3Keyboard(Keyboard):
    """
    A keyboard controller for sending SE(3) commands as delta poses and binary command (open/close).

    Parameters
    ----------
    key_bindings: dict[str, KeyBinding]
        The key bindings for the keyboard.
    pos_max: float
        Gain on the three position axes. Defaults to ``1.0``, i.e. a **normalized** delta pose:
        the task-space controller downstream owns the conversion to metres via its
        ``position_scale``. Lower it only to clamp the device below full deflection, and expect
        the physical step to be the product of the two.
    rot_max: float
        Gain on the three rotation axes, on the same normalized contract as ``pos_max`` — the
        controller's ``orientation_scale`` owns radians.

    Key bindings
    ------------
    ============================== ================= =================
    Description                    Key (+ve axis)    Key (-ve axis)
    ============================== ================= =================
    Toggle gripper (open/close)    Space
    Move along x-axis              8                 2
    Move along y-axis              4                 6
    Move along z-axis              7                 9
    Rotate along x-axis            Up                Down
    Rotate along y-axis            Right             Left
    Rotate along z-axis            1                 3
    ============================== ================= =================

    """

    def __init__(
        self,
        key_bindings: dict[str, KeyBinding] = None,
        pos_max: float = 1.0,
        rot_max: float = 1.0,
    ):
        key_bindings = key_bindings or self.default_key_bindings()
        super().__init__(key_bindings)
        self.limit = torch.tensor(
            [
                pos_max,
            ]
            * 3
            + [
                rot_max,
            ]
            * 3
            + [1.0],
            dtype=torch.float32,
        )

    @classmethod
    def default_key_bindings(cls):
        cls._require_pynput()
        return {
            "8": KeyBinding(0, +1),
            "2": KeyBinding(0, -1),
            "4": KeyBinding(1, +1),
            "6": KeyBinding(1, -1),
            "7": KeyBinding(2, +1),
            "9": KeyBinding(2, -1),
            Key.up: KeyBinding(3, +1),
            Key.down: KeyBinding(3, -1),
            Key.right: KeyBinding(4, +1),
            Key.left: KeyBinding(4, -1),
            "1": KeyBinding(5, +1),
            "3": KeyBinding(5, -1),
            Key.space: KeyBinding(6, 0, is_toggle=True),
        }

    @classmethod
    def wasd_key_bindings(cls):
        cls._require_pynput()
        return {
            "w": KeyBinding(0, +1),
            "s": KeyBinding(0, -1),
            "a": KeyBinding(1, +1),
            "d": KeyBinding(1, -1),
            "q": KeyBinding(2, +1),
            "e": KeyBinding(2, -1),
            Key.up: KeyBinding(3, +1),
            Key.down: KeyBinding(3, -1),
            Key.right: KeyBinding(4, +1),
            Key.left: KeyBinding(4, -1),
            "z": KeyBinding(5, +1),
            "c": KeyBinding(5, -1),
            Key.space: KeyBinding(6, 0, is_toggle=True),
        }

    def get_command(self) -> torch.Tensor:
        # NOTE: apply a hyperbolic tangent to the readings to smooth the keyboard reading rate
        return torch.where(
            self._is_toggle_mask,
            self._readings,
            torch.tanh(self._readings) * self.limit,
        )


if __name__ == "__main__":
    import time

    keyboard = SE3Keyboard()

    while True:
        print(keyboard.get_command())
        time.sleep(0.01)
