"""Building an SE(2) input device by name, and mapping its deflection to a velocity.

Both :class:`~eden.utils.devices.keyboard.TeleopSE2Keyboard` and
:class:`~eden.utils.devices.gamepad.SE2Gamepad` emit a *normalized* ``(vx, vy, wz)``
deflection in ``[-1, 1]``; the consumer owns the conversion to m/s and rad/s, the
contract :class:`~eden.utils.devices.keyboard.SE3Keyboard` already documents. This
module holds the two halves of that contract shared by every consumer: naming a
device, and scaling its deflection into a velocity range.

A held key is **full** deflection, from the first press -- the same thing a stick
pushed to its stop means. That is why the keyboard here is ``TeleopSE2Keyboard`` and
not its ``SE2Keyboard`` base, whose ``tanh`` smoothing turns one press into ``0.76``
and only approaches full deflection as OS key-repeat accumulates, making the
commanded speed depend on the operator's key-repeat rate.

The keyboard bindings are arrows and ``,`` / ``.`` -- deliberately not letters or
digits. ``pynput`` listens to the keyboard *globally* rather than to the focused
window, so it also sees the keys the operator types at the terminal to drive the
deployment state machine (:mod:`eden.extensions.deployment.utils.operator_input`
reads ``s`` START, ``x`` STOP, ``i`` INIT, ``k`` KILL, ``n`` and ``1``-``9`` from
stdin). On WASD bindings, typing START would also command the robot backwards.

======================= ================= =================
Command                 Key (+ve axis)    Key (-ve axis)
======================= ================= =================
Forward / backward vx   Up                Down
Strafe vy               Left              Right
Turn wz                 ,                 .
======================= ================= =================
"""

from __future__ import annotations

import torch

#: Device names :func:`make_se2_device` accepts.
DEVICES: tuple[str, ...] = ("keyboard", "gamepad")

#: Controller families :func:`make_se2_device` knows the axis scaling for.
GAMEPAD_LAYOUTS: tuple[str, ...] = ("ps4", "xbox")


def teleop_key_bindings() -> dict:
    """SE(2) key bindings that do not collide with the deploy operator keys.

    See the module docstring for the table and for why letters and digits are
    unusable. Requires ``pynput`` (the ``all`` extra), since the arrow keys are
    ``pynput`` key objects rather than characters.
    """
    from eden.utils.devices.keyboard import KeyBinding, Keyboard

    Keyboard._require_pynput()
    from pynput.keyboard import Key

    return {
        Key.up: KeyBinding(0, +1),
        Key.down: KeyBinding(0, -1),
        Key.left: KeyBinding(1, +1),
        Key.right: KeyBinding(1, -1),
        ",": KeyBinding(2, +1),
        ".": KeyBinding(2, -1),
    }


def make_se2_device(device: str, options: dict | None = None):
    """Build a normalized SE(2) input device by name.

    Parameters
    ----------
    device: str
        One of :data:`DEVICES`.
    options: dict | None
        Device options. ``layout`` ("ps4" | "xbox") picks the gamepad axis scaling;
        every other key reaches the device constructor.

    Returns
    -------
    DeviceBase
        The device, already listening. ``vel_max`` / ``ang_max`` are pinned to ``1.0``
        so it emits a deflection, not a velocity -- :func:`scale_se2_deflection` owns
        the units.
    """
    options = dict(options or {})
    if device not in DEVICES:
        raise ValueError(f"Unknown SE(2) device {device!r}; expected one of {list(DEVICES)}.")

    if device == "keyboard":
        # TeleopSE2Keyboard, not SE2Keyboard: see the module docstring for why a ramped
        # deflection is the wrong contract for a velocity command.
        from eden.utils.devices.keyboard import TeleopSE2Keyboard

        options.pop("layout", None)
        return TeleopSE2Keyboard(key_bindings=teleop_key_bindings(), vel_max=1.0, ang_max=1.0, **options)

    from eden.utils.devices.gamepad import (
        DEADZONE_PS4,
        DEADZONE_XBOX,
        OFFSET_PS4,
        OFFSET_XBOX,
        SCALE_PS4,
        SCALE_XBOX,
        SE2Gamepad,
    )

    layout = options.pop("layout", "ps4")
    if layout not in GAMEPAD_LAYOUTS:
        raise ValueError(f"Unknown gamepad layout {layout!r}; expected one of {list(GAMEPAD_LAYOUTS)}.")
    scale, offset, deadzone = {
        "ps4": (SCALE_PS4, OFFSET_PS4, DEADZONE_PS4),
        "xbox": (SCALE_XBOX, OFFSET_XBOX, DEADZONE_XBOX),
    }[layout]
    return SE2Gamepad(vel_max=1.0, ang_max=1.0, scale=scale, offset=offset, deadzone=deadzone, **options)


def scale_se2_deflection(deflection: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Map a normalized ``[-1, 1]`` deflection into ``[low, high]`` per axis.

    Full deflection maps to the *larger* side of each axis and the result is clamped
    into the range, so an asymmetric range (e.g. ``(-2.0, 3.0)``) reaches both ends and
    saturates early on the short side rather than overshooting it. Clamping is what
    bounds an operator to the velocities a policy was trained on, so it is applied to
    the device reading too -- a miscalibrated stick cannot deflect past 1.0 and out of
    the range that way.
    """
    return torch.clamp(deflection.clamp(-1.0, 1.0) * torch.maximum(high.abs(), low.abs()), min=low, max=high)
