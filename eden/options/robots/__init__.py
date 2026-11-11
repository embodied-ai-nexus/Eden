"""Robot configuration option presets."""

from eden.options.robots.aloha_vx300s import Aloha
from eden.options.robots.franka_emika_panda import FrankaEmikaPanda, FrankaEmikaPandaNoHand, FrankaResearch3
from eden.options.robots.franka_hand import FrankaHand
from eden.options.robots.i2rt_yam import Yam, YAM_ACTION_SCALE
from eden.options.robots.inspire_rh56 import InspireRH56_L, InspireRH56_R, InspireRH56DFQ_L, InspireRH56DFQ_R
from eden.options.robots.kuka_iiwa14 import Iiwa14
from eden.options.robots.leap_hand import LeapHand_L, LeapHand_R
from eden.options.robots.limx_oli import (
    Oli,
    OLI_ACTION_SCALE,
    OliNoHands,
)
from eden.options.robots.mano_hand import ManoHand_L, ManoHand_R
from eden.options.robots.omnireset_ur5e_2f85 import OmniResetUR5e2f85
from eden.options.robots.robotera_xhand import XHand1_L, XHand1_R
from eden.options.robots.robotiq_2f85 import Robotiq2f85
from eden.options.robots.sharpa_wave import SharpaWave_L, SharpaWave_R
from eden.options.robots.smpl_humanoid import PhcSmplHumanoid, SmplHumanoid
from eden.options.robots.tesollo_dg5f import TesolloDG5F_L, TesolloDG5F_R
from eden.options.robots.trossen_widowx250 import WidowX250
from eden.options.robots.ufactory_xarm7 import Xarm7
from eden.options.robots.unitree_g1 import (
    G1Dofs23,
    G1Dofs29NoWristDof,
    G1Dofs29,
    G1Dofs29Dex3,
    G1Dofs29NoHands,
    G1_ACTION_SCALE,
)
from eden.options.robots.unitree_go1 import Go1, GO1_ACTION_SCALE
from eden.options.robots.unitree_go2 import (
    Go2,
    GO2_ACTION_SCALE,
    Go2W,
    GO2W_ACTION_SCALE,
)
from eden.options.robots.universal_ur5e import UR5e
from eden.options.robots.universal_ur10e import UR10e
from eden.options.robots.wuji_hand1 import (
    WujiHand1_L,
    WujiHand1_R,
    WujiHand1Simplified_L,
    WujiHand1Simplified_R,
)
from eden.options.robots.wuji_hand2beta2 import (
    WujiHand2Beta2_L,
    WujiHand2Beta2_R,
    WujiHand2Beta2WithMount_L,
    WujiHand2Beta2WithMount_R,
)

from eden.options.catalog import ROBOT_REGISTRY
from eden.options.robots.composite import (
    COMPOSITE_ROBOT_REGISTRY,
    DexManipulatorOptions,
    HumanoidOptions,
    ManipulatorOptions,
    Droid,
    FrankaXHand,
    G1Inspire,
    G1Wuji,
    KukaSharpa,
    OliWuji,
    PandaFrankaHand,
    UR5eRobotiq,
    UR10eRobotiq,
    XArm7XHand1_L,
    XArm7XHand1_R,
    has_frame_alignment,
    weld_quat,
)


__all__ = [
    "ROBOT_REGISTRY",
    "COMPOSITE_ROBOT_REGISTRY",
    "ManipulatorOptions",
    "DexManipulatorOptions",
    "HumanoidOptions",
    "Droid",
    "FrankaXHand",
    "KukaSharpa",
    "OliWuji",
    "G1Inspire",
    "G1Wuji",
    "PandaFrankaHand",
    "UR5eRobotiq",
    "UR10eRobotiq",
    "Aloha",
    "FrankaEmikaPanda",
    "FrankaEmikaPandaNoHand",
    "FrankaResearch3",
    "FrankaHand",
    "SmplHumanoid",
    "PhcSmplHumanoid",
    "G1Dofs23",
    "G1Dofs29NoWristDof",
    "G1Dofs29",
    "G1Dofs29Dex3",
    "G1Dofs29NoHands",
    "G1_ACTION_SCALE",
    "Go1",
    "GO1_ACTION_SCALE",
    "Go2",
    "GO2_ACTION_SCALE",
    "Go2W",
    "GO2W_ACTION_SCALE",
    "Iiwa14",
    "InspireRH56DFQ_L",
    "InspireRH56DFQ_R",
    "InspireRH56_L",
    "InspireRH56_R",
    "LeapHand_L",
    "LeapHand_R",
    "Oli",
    "OliNoHands",
    "OLI_ACTION_SCALE",
    "ManoHand_L",
    "ManoHand_R",
    "Robotiq2f85",
    "SharpaWave_L",
    "SharpaWave_R",
    "TesolloDG5F_L",
    "TesolloDG5F_R",
    "UR5e",
    "OmniResetUR5e2f85",
    "UR10e",
    "WidowX250",
    "XArm7XHand1_L",
    "XArm7XHand1_R",
    "Xarm7",
    "XHand1_L",
    "XHand1_R",
    "WujiHand1_L",
    "WujiHand1_R",
    "WujiHand1Simplified_L",
    "WujiHand1Simplified_R",
    "WujiHand2Beta2_L",
    "WujiHand2Beta2_R",
    "WujiHand2Beta2WithMount_L",
    "WujiHand2Beta2WithMount_R",
    "Yam",
    "YAM_ACTION_SCALE",
    "has_frame_alignment",
    "weld_quat",
]

# Robots register themselves via @ROBOT_REGISTRY.register in their own modules (imported
# above); ROBOT_REGISTRY is re-exported here for convenience.
