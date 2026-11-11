"""Bundled robot + retargeter presets for dex-retargeting.

Single source of truth shared by the ``eden retarget dex`` CLI and the
``examples/recipes/retargeting/dex_retargeter`` scripts. Each :class:`HandPreset` pairs a
``RobotOptions`` factory with a ``RetargeterOptions`` factory. The
``RetargeterOptions`` factories import ``DexRetargeter`` lazily (it pulls
``nlopt``), so importing this module for :data:`POSITION_PRESET_NAMES` — used to
build the CLI's ``--preset`` choices — stays free of the ``dex-retargeter``
extra.

Fingertip link names and MANO keypoint indices are read off the robot's
``MetadataOptions`` (``target_hand_links`` / ``mano_hand_idx``), so adding a
hand only needs a new entry in :data:`HAND_PRESETS`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np

import eden as en

if TYPE_CHECKING:
    from eden.options.entities import RobotOptions
    from eden.options.extensions.retargeting import RetargeterOptions

# MediaPipe MANO-keypoint indices for wrist + thumb/index/middle/ring/pinky tips
# (universal to MediaPipe's hand model, independent of the target robot).
_MEDIAPIPE_WRIST = 0
_MEDIAPIPE_FINGERTIPS = [4, 8, 12, 16, 20]

_XHAND1_R_WRIST = "right_hand_link"
_XHAND1_R_TIPS = list(en.robots.XHand1_R.metadata.target_hand_links)
_XHAND1_R_MANO_IDX = list(en.robots.XHand1_R.metadata.mano_hand_idx)

# The Wuji Hand 2 (beta2) is rooted at `{r,l}_wrist` in the unmounted variants, which is
# also the frame the Wuji Glove reports its skeleton in -- so a glove drives its
# same-brand hand through the standard MANO-keypoint path (see the wuji teleop recipe).
_WUJI_HAND2 = {
    "right": (en.robots.WujiHand2Beta2_R, "r_wrist"),
    "left": (en.robots.WujiHand2Beta2_L, "l_wrist"),
}


@dataclass
class HandPreset:
    """A robot-options factory paired with its retargeter-options factory, for one ``side`` of hand."""

    robot_options_factory: Callable[[], "RobotOptions"]
    retargeter_options: Callable[[], "RetargeterOptions"]
    side: Literal["left", "right"]


def _xhand1_r_robot() -> "RobotOptions":
    return en.robots.XHand1_R(is_fixed_base=True, default_root_pos=(0.0, 0.0, 0.2))


def _xhand1_right_dexpilot() -> "RetargeterOptions":
    from eden.extensions.retargeting.dex_retargeter import DexRetargeter

    return DexRetargeter.configure(
        entity_name="robot",
        type="dexpilot",
        finger_tip_link_names=_XHAND1_R_TIPS,
        wrist_link_name=_XHAND1_R_WRIST,
        scaling_factor=1.5,
        low_pass_alpha=0.2,
    )


def _xhand1_right_vector() -> "RetargeterOptions":
    from eden.extensions.retargeting.dex_retargeter import DexRetargeter

    n = len(_XHAND1_R_TIPS)
    return DexRetargeter.configure(
        entity_name="robot",
        type="vector",
        target_origin_link_names=[_XHAND1_R_WRIST] * n,
        target_task_link_names=_XHAND1_R_TIPS,
        target_link_human_indices=np.array([[_MEDIAPIPE_WRIST] * n, _MEDIAPIPE_FINGERTIPS[:n]]),
        scaling_factor=1.5,
        low_pass_alpha=0.2,
    )


def _xhand1_right_position() -> "RetargeterOptions":
    # Position retargeting consumes MANO keypoints (e.g. from DexYCB / ARCTIC);
    # use MANO indices, not MediaPipe ones.
    from eden.extensions.retargeting.dex_retargeter import DexRetargeter

    return DexRetargeter.configure(
        entity_name="robot",
        type="position",
        target_link_names=_XHAND1_R_TIPS,
        target_link_human_indices=np.array(_XHAND1_R_MANO_IDX),
        low_pass_alpha=1.0,
    )


def _wuji_hand2_robot(side: str) -> "RobotOptions":
    robot, _ = _WUJI_HAND2[side]
    return robot(is_fixed_base=True, default_root_pos=(0.0, 0.0, 0.2))


def _wuji_hand2_dexpilot(side: str) -> "RetargeterOptions":
    from eden.extensions.retargeting.dex_retargeter import DexRetargeter

    robot, wrist = _WUJI_HAND2[side]
    return DexRetargeter.configure(
        entity_name="robot",
        type="dexpilot",
        finger_tip_link_names=list(robot.metadata.target_hand_links),
        wrist_link_name=wrist,
        # The Wuji Hand 2 is human-sized: its straight wrist->tip reach is 1.0-1.1x the glove's
        # default hand (index 0.179 vs 0.167 m). At 1.5 an open hand's scaled vectors outreach
        # every robot finger and a curled one is matched by the MCP alone, so PIP / DIP never bend.
        scaling_factor=1.05,
        low_pass_alpha=0.2,
    )


HAND_PRESETS: dict[str, HandPreset] = {
    "xhand1_right_dexpilot": HandPreset(_xhand1_r_robot, _xhand1_right_dexpilot, "right"),
    "xhand1_right_vector": HandPreset(_xhand1_r_robot, _xhand1_right_vector, "right"),
    "xhand1_right_position": HandPreset(_xhand1_r_robot, _xhand1_right_position, "right"),
    "wuji_hand2beta2_right_dexpilot": HandPreset(
        partial(_wuji_hand2_robot, "right"), partial(_wuji_hand2_dexpilot, "right"), "right"
    ),
    "wuji_hand2beta2_left_dexpilot": HandPreset(
        partial(_wuji_hand2_robot, "left"), partial(_wuji_hand2_dexpilot, "left"), "left"
    ),
}

# Presets usable by the offline position path (`eden retarget dex`). Kept as
# bare names so argparse choices need no backend import.
POSITION_PRESET_NAMES: tuple[str, ...] = ("xhand1_right_position",)
