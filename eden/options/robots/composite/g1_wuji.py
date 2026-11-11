"""G1Wuji = Unitree G1 29-DOF (hands stripped) + a Wuji Hand2 beta2 bolted to each wrist.

The G1 counterpart of :class:`~eden.options.robots.composite.oli_wuji.OliWuji`. The flange is the
``*_ee_link`` frame the no-hand G1 MJCF declares 4 cm past each ``*_wrist_yaw_link``, along the
forearm (+x of the wrist frame).

**The orientation.** Unlike Oli, the G1 wrist frame runs its forearm along +x, while a Wuji hand's
fingers run along its base link's -z. The mount is therefore -90 deg about the flange's y axis, which
takes the hand's -z onto the flange's +x: fingers continue the forearm, palm toward the body midline
and thumb on the wrist frame's +z (forward in the arms-down pose). One quaternion serves both sides,
because the per-side chirality lives in the ``_L`` / ``_R`` hand assets.
"""

from __future__ import annotations

import math

from genesis.typing import UnitVec4FType
from pydantic import Field

from eden.options.entities import RobotOptions
from eden.options.robots.composite.base import COMPOSITE_ROBOT_REGISTRY, HumanoidOptions
from eden.options.robots.unitree_g1 import G1Dofs29NoHands
from eden.options.robots.wuji_hand2beta2 import WujiHand2Beta2WithMount_L, WujiHand2Beta2WithMount_R

_HALF = math.sqrt(0.5)


@COMPOSITE_ROBOT_REGISTRY.register()
class G1Wuji(HumanoidOptions):
    """Unitree G1 + two Wuji Hand2 beta2 hands, a 69-DOF bimanual humanoid (29 body + 2 x 20).

    Expands into three scene entities — the humanoid under
    :attr:`~eden.options.robots.composite.base.HumanoidOptions.robot_name` plus a hand per wrist::

        scene = en.scenes.Empty(robot=G1Wuji(base_pos=(0.0, 0.0, 0.793)))
        # -> entities "robot", "left_hand", "right_hand"
    """

    #: The hand-stripped G1, whose ``*_ee_link`` frames are the flanges.
    robot: RobotOptions = Field(default_factory=G1Dofs29NoHands)
    #: Free-base ``WithMount`` hands (a welded child is posed by its parent link).
    left_hand: RobotOptions = Field(default_factory=lambda: WujiHand2Beta2WithMount_L(is_fixed_base=False))
    right_hand: RobotOptions = Field(default_factory=lambda: WujiHand2Beta2WithMount_R(is_fixed_base=False))

    left_ee_link: str = "left_ee_link"
    right_ee_link: str = "right_ee_link"

    #: -90 deg about the flange y axis: the hand's finger axis (-z) onto the forearm (+x).
    mount_quat: UnitVec4FType = (_HALF, 0.0, -_HALF, 0.0)

    left_grasp_links: list[str] = list(WujiHand2Beta2WithMount_L.metadata.target_hand_links)
    right_grasp_links: list[str] = list(WujiHand2Beta2WithMount_R.metadata.target_hand_links)
