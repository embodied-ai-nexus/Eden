"""Intel RealSense camera (colour, and depth aligned to it).

:mod:`camera` imports ``pyrealsense2`` (Eden's ``realsense`` extra) only when a camera opens, so this
package imports without it.
"""

from eden.utils.devices.realsense.camera import (
    DEFAULT_DEPTH_SIZE,
    MODES,
    RealSenseCamera,
    RealSenseFrame,
    list_devices,
    parse_mode,
)

__all__ = ["DEFAULT_DEPTH_SIZE", "MODES", "RealSenseCamera", "RealSenseFrame", "list_devices", "parse_mode"]
