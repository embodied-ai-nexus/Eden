"""SpaceMouse teleoperation device (SE2/SE3)."""

import threading
import time
from collections import defaultdict

import numpy as np
import torch

# Top-level `import eden` is safe: devices/ is not on eden's package-init
# import chain, and `en.logger` is seeded at package import.
import eden as en
from eden.utils.devices.base import DeviceBase

IS_SPNAV_INSTALLED = True
try:
    from spnav import (
        spnav_open,
        spnav_poll_event,
        spnav_close,
        SpnavMotionEvent,
        SpnavButtonEvent,
    )
except ImportError:
    IS_SPNAV_INSTALLED = False


class Spacemouse(DeviceBase):
    """
    Spacemouse device.

    Parameters
    ----------
    pos_max: float
        Gain on the three position axes. Defaults to ``1.0``, i.e. a **normalized** delta pose:
        the task-space controller downstream owns the conversion to metres via its
        ``position_scale``. Lower it only to clamp the device below full deflection, and expect
        the physical step to be the product of the two.
    rot_max: float
        Gain on the three rotation axes, on the same normalized contract as ``pos_max`` — the
        controller's ``orientation_scale`` owns radians.
    deadzone: float | tuple
        The deadzone of the spacemouse reading. Default is 0.3.
    polling_rate: int
        The polling rate of the spacemouse. Default is 200.
    is_toggle_btn0: bool
        Whether the first button is a toggle button. Default is False.
    is_toggle_btn1: bool
        Whether the second button is a toggle button. Default is False.
    """

    def __init__(
        self,
        pos_max: float = 1.0,
        rot_max: float = 1.0,
        deadzone: float | tuple = 0.3,
        polling_rate: int = 200,
        is_toggle_btn0: bool = False,
        is_toggle_btn1: bool = False,
    ):
        if not IS_SPNAV_INSTALLED:
            error_message = """
            spnav is not installed.
            Please install it with the following commands:
            ```bash
            sudo apt install libspnav-dev spacenavd
            sudo systemctl start spacenavd
            pip install https://github.com/cheng-chi/spnav/archive/c1c938ebe3cc542db4685e0d13850ff1abfdb943.tar.gz
            ```
            Then reboot the system before using the spacemouse.
            """
            raise ImportError(error_message)

        super().__init__()

        self.pos_max = pos_max
        self.rot_max = rot_max
        self.polling_rate = polling_rate
        self.is_toggle_btn0 = is_toggle_btn0
        self.is_toggle_btn1 = is_toggle_btn1

        if isinstance(deadzone, (int, float)):
            self.deadzone = np.full(6, fill_value=deadzone, dtype=np.float32)
        else:
            self.deadzone = np.array(deadzone, dtype=np.float32)
        assert (self.deadzone >= 0).all()

        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        self.motion_event = SpnavMotionEvent([0, 0, 0], [0, 0, 0], 0)
        self.button_state = defaultdict(lambda: False)

        # Coordinate transform from Spacemouse to Eden (Z-up)
        self.tx_zup_spnav = torch.tensor([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device="cpu")

        self.thread = threading.Thread(target=self._run_worker)
        self.thread.daemon = True
        self.thread.start()

    def _run_worker(self):
        try:
            spnav_open()
        except Exception as e:
            en.logger.warning(f"Failed to open spacemouse: {e}")
            return

        try:
            while not self.stop_event.is_set():
                event = spnav_poll_event()
                if isinstance(event, SpnavMotionEvent):
                    with self.lock:
                        self.motion_event = event
                elif isinstance(event, SpnavButtonEvent):
                    with self.lock:
                        if event.bnum == 0:
                            if self.is_toggle_btn0:
                                if event.press:
                                    self.button_state[event.bnum] = not self.button_state[event.bnum]
                            else:
                                self.button_state[event.bnum] = event.press
                        elif event.bnum == 1:
                            if self.is_toggle_btn1:
                                if event.press:
                                    self.button_state[event.bnum] = not self.button_state[event.bnum]
                            else:
                                self.button_state[event.bnum] = event.press

                    # Handle callbacks
                    if event.bnum in self._callbacks:
                        self._callbacks[event.bnum](event.press)
                else:
                    time.sleep(1.0 / self.polling_rate)
        except Exception as e:
            en.logger.error(f"Spacemouse thread error: {e}")
        finally:
            try:
                spnav_close()
            except Exception:
                pass

    def get_raw_data(self) -> torch.Tensor:
        """Return the raw normalized state without coordinate transformation."""
        with self.lock:
            me = self.motion_event
            state = np.array(me.translation + me.rotation, dtype=np.float32) / 500.0

        is_dead = (-self.deadzone < state) & (state < self.deadzone)
        state[is_dead] = 0.0
        return torch.from_numpy(state)

    def get_command(self) -> torch.Tensor:
        """Return the transformed command in Eden coordinate system (Z-up)."""
        state = self.get_raw_data()

        tf_state = torch.zeros_like(state)
        tf_state[:3] = (self.tx_zup_spnav @ state[:3]) * self.pos_max
        tf_state[3:] = (self.tx_zup_spnav @ state[3:]) * self.rot_max

        return tf_state

    def is_button_pressed(self, button_id: int) -> bool:
        with self.lock:
            return self.button_state[button_id]

    def close(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def __del__(self):
        self.close()


class SE2Spacemouse(Spacemouse):
    """
    A spacemouse controller for sending SE(2) commands as velocity commands.

    Unlike the delta-pose devices, this one feeds a velocity command directly rather than a
    task-space controller, so it keeps physical units, capped by default at 0.1 m/s and 0.1 rad/s.

    Parameters
    ----------
    pos_max: float
        The maximum linear velocity, in m/s.
    rot_max: float
        The maximum angular velocity, in rad/s.
    **kwargs
        Forwarded to :class:`Spacemouse`.
    """

    def __init__(self, pos_max: float = 0.1, rot_max: float = 0.1, **kwargs):
        super().__init__(pos_max=pos_max, rot_max=rot_max, **kwargs)

    def get_command(self) -> torch.Tensor:
        state = self.get_raw_data()

        tf_state = torch.zeros(3)
        tf_state[:2] = (self.tx_zup_spnav @ state[3:5]) * self.rot_max

        return tf_state


class SE3Spacemouse(Spacemouse):
    """A spacemouse controller for sending SE(3) commands as delta poses and binary command (open/close)."""

    def get_command(self) -> torch.Tensor:
        """Return the transformed command in Eden coordinate system (Z-up)."""
        state = self.get_raw_data()

        tf_state = torch.zeros(7)
        tf_state[:3] = (self.tx_zup_spnav @ state[:3]) * self.pos_max
        tf_state[3:6] = (self.tx_zup_spnav @ state[3:6]) * self.rot_max
        tf_state[6] = 1.0 if self.is_button_pressed(0) else 0.0

        return tf_state


if __name__ == "__main__":
    sm = Spacemouse(deadzone=0.3, is_toggle_btn0=True, is_toggle_btn1=False)
    print("Spacemouse initialized. Press Ctrl+C to exit.")
    try:
        while True:
            cmd = sm.get_command()
            btns = [sm.is_button_pressed(i) for i in range(2)]
            print(f"Cmd: {cmd.numpy().round(3)}, Buttons: {btns}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")
        sm.close()
