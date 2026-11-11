"""Thin re-export of the shared dex-retargeting hand presets.

The presets now live in the eden package so the ``eden retarget dex`` CLI and
these example scripts consume one source of truth; see
:mod:`eden.extensions.retargeting.dex_retargeter.presets`.
"""

from __future__ import annotations

from eden.extensions.retargeting.dex_retargeter.presets import (
    HAND_PRESETS,
    HandPreset,
    POSITION_PRESET_NAMES,
)

__all__ = ["HAND_PRESETS", "HandPreset", "POSITION_PRESET_NAMES"]
