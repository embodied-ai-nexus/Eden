"""Constants for the dex-retargeting port.

Mirrors ``dex_retargeting.constants`` but trimmed to what the Eden port
actually consumes.  Robot-name enums upstream are tied to YAML config files
shipped with the package; in Eden we plug in any ``RigidEntity`` directly, so
those enums are not needed.
"""

from __future__ import annotations

import enum

import numpy as np


# Frame conversion matrices from the operator (MediaPipe) frame to the MANO
# wrist frame. Same values as upstream ``dex_retargeting.constants``.
OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ]
)


class HandType(enum.Enum):
    right = enum.auto()
    left = enum.auto()


OPERATOR2MANO = {
    HandType.right: OPERATOR2MANO_RIGHT,
    HandType.left: OPERATOR2MANO_LEFT,
}
