"""Wuji Glove teleoperation device (hand pose + tactile).

:mod:`protocol` is pure numpy — stream layouts, the MediaPipe landmark table, and the
basis change from the vendor's anatomical wrist frame into Eden's hand frame — so it
imports without ``wuji_sdk``. :mod:`glove` owns the connection; it raises an install
hint if the SDK (Eden's ``wuji`` extra) is missing.
"""

from eden.utils.devices.wuji.glove import (
    CALIBRATED_STREAMS,
    DEFAULT_STREAMS,
    RATE_SOURCES,
    STREAMS,
    TACTILE_SHEETS,
    WujiGlove,
    load_sdk,
    scan_gloves,
)
from eden.utils.devices.wuji.protocol import (
    FINGERS,
    HAND_LANDMARKS,
    N_ACTIVE_TAXELS,
    N_HAND_DOFS,
    N_LANDMARKS,
    N_UNMASKED_TAXELS,
    R_HAND_WUJI,
    TACTILE_COLS,
    TACTILE_INVALID,
    TACTILE_ROWS,
    TACTILE_ZONE_SIZES,
    tactile_grid,
    tactile_taxels,
    tactile_valid,
    wuji_points_to_hand,
    wuji_pose_to_hand,
)

__all__ = [
    "CALIBRATED_STREAMS",
    "DEFAULT_STREAMS",
    "FINGERS",
    "HAND_LANDMARKS",
    "N_ACTIVE_TAXELS",
    "N_HAND_DOFS",
    "N_LANDMARKS",
    "N_UNMASKED_TAXELS",
    "RATE_SOURCES",
    "R_HAND_WUJI",
    "STREAMS",
    "TACTILE_COLS",
    "TACTILE_INVALID",
    "TACTILE_ROWS",
    "TACTILE_SHEETS",
    "TACTILE_ZONE_SIZES",
    "WujiGlove",
    "load_sdk",
    "scan_gloves",
    "tactile_grid",
    "tactile_taxels",
    "tactile_valid",
    "wuji_points_to_hand",
    "wuji_pose_to_hand",
]
