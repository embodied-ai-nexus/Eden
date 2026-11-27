"""Wuji Hand 2 (hand2_2) dexterous hand configurations, left and right.

The current revision of the Wuji Hand 2, and the model Wuji's ``wuji-sim`` simulates: twenty DOFs (5 fingers x
4), with the same joint names, axes and limits as :mod:`eden.options.robots.wuji_hand2beta2`. The palm's MCP
origins, the fingertip frames, the meshes and the inertials differ, so check a policy trained on one model in
simulation on the other before reusing it. Assets (URDF + STL meshes, MIT-licensed) live under
``hands/wuji_hand2_2`` in the ``Kashu7100/eden_assets`` dataset, converted from
``wuji-technology/wuji-description`` v2026.11.18 ``hand2/hand2_2/body``, the release ``wuji-sim`` v2026.11.23
pins.

Upstream's bare URDF is rooted at the wrist adapter plate (``{r,l}_mount``), so these presets correspond to
``WujiHand2Beta2WithMount_*``: the root pose getters and ``default_root_pos`` refer to the plate, and
``{r,l}_wrist`` is kept as a link so lookups written against the wrist still resolve.

The joint dynamics are wuji-sim's (``wuji_sim/model_overrides.toml`` and ``profiles/mujoco.toml``): per-joint
armature, the torque limit of a 1.5 A current through each motor's torque constant, and zero joint friction and
damping, which wuji-sim leaves unidentified. ``WujiHand2Beta2`` instead carries the friction loss and damping
wuji-mjlab identified on one right hand; pass them as ``default_dofs_frictionloss`` / ``default_dofs_damping``
to train against them. The gains are the pen-spin operating point the beta2 presets use, which wuji-sim's
pen-spin example also runs this model at.
"""

from typing import ClassVar

from eden.options.actuators import ActuatorSpecOptions
from eden.options.catalog import ROBOT_REGISTRY
from eden.options.entities import MetadataOptions, RobotOptions
from eden.options.robots.wuji_hand2beta2 import _CURRENT_LIMIT_A, _MIT_KP, _PEN_SPIN_MOTORS


# Finger-major, thumb first: the SDK's joint order, and upstream's ``WUJI_HAND2_TASK_NAMES``.
_JOINTS = tuple(
    f"{finger}_{joint}"
    for finger, joints in (
        ("thumb", ("cmc_flex", "cmc_abd", "mcp", "ip")),
        ("index_finger", ("mcp_flex", "mcp_abd", "pip", "dip")),
        ("middle_finger", ("mcp_flex", "mcp_abd", "pip", "dip")),
        ("ring_finger", ("mcp_flex", "mcp_abd", "pip", "dip")),
        ("pinky", ("mcp_flex", "mcp_abd", "pip", "dip")),
    )
    for joint in joints
)
_TIPS = ("thumb_tip", "index_finger_tip", "middle_finger_tip", "ring_finger_tip", "pinky_tip")

# Joint armature [kg*m^2] from wuji-sim v2026.11.23 ``wuji_sim/model_overrides.toml`` (source "estimate"; the left
# and right hands use the same values). They are the per-joint values the ``WujiHand2*`` motor specs average.
_ARMATURE = {
    "thumb_cmc_flex": 0.0001457162066834878,
    "thumb_cmc_abd": 0.00013663892981144644,
    "thumb_mcp": 0.00012735528418833324,
    "thumb_ip": 8.768902139971928e-05,
    "index_finger_mcp_flex": 0.00011304740105532539,
    "index_finger_mcp_abd": 5.1271605241470905e-05,
    "index_finger_pip": 4.843782959990254e-05,
    "index_finger_dip": 7.906192627320932e-05,
    "middle_finger_mcp_flex": 0.00020110478286346846,
    "middle_finger_mcp_abd": 7.106655434321573e-05,
    "middle_finger_pip": 6.05474605842666e-05,
    "middle_finger_dip": 5.659271669130923e-05,
    "ring_finger_mcp_flex": 0.00013938973043403448,
    "ring_finger_mcp_abd": 3.544107855351169e-05,
    "ring_finger_pip": 5.233973575843903e-05,
    "ring_finger_dip": 6.0479482928204684e-05,
    "pinky_mcp_flex": 9.70625901007757e-05,
    "pinky_mcp_abd": 3.8260542056944675e-05,
    "pinky_pip": 4.664456024117704e-05,
    "pinky_dip": 4.74377771614785e-05,
}


class _WujiHand2_2(RobotOptions):
    dofs_spec: ClassVar[dict[str, type[ActuatorSpecOptions]]] = {p: m for p, (m, _) in _PEN_SPIN_MOTORS.items()}

    # The current the force limits below are set at. ``WujiHandDeployment`` programs it when ``effort_limit_a`` is
    # unset.
    current_limit_a: ClassVar[float] = _CURRENT_LIMIT_A

    default_dofs_pos: dict[str, float] = {"*": 0.0}

    default_dofs_kp: dict[str, float] = {p: m.JOINT_GAIN(_MIT_KP) for p, (m, _) in _PEN_SPIN_MOTORS.items()}
    default_dofs_kd: dict[str, float] = {p: m.JOINT_GAIN(kd) for p, (m, kd) in _PEN_SPIN_MOTORS.items()}
    # wuji-sim's ``joint_torque_limit``, "1.5 A current limit x torque constant", from the torque constants
    # themselves: wuji-sim's table agrees to its 1e-3 N*m except at thumb IP (0.200 there, 1.5 x 0.1329 = 0.19935).
    default_dofs_force_limits: dict[str, float] = {
        p: m.TORQUE_CONSTANT * _CURRENT_LIMIT_A for p, (m, _) in _PEN_SPIN_MOTORS.items()
    }

    # Per joint, so they replace the motor specs' means (armature) and beta2-identified values (friction loss).
    default_dofs_armature: dict[str, float] = {f".*_{joint}": value for joint, value in _ARMATURE.items()}
    default_dofs_frictionloss: dict[str, float] = {"*": 0.0}
    default_dofs_damping: dict[str, float] = {"*": 0.0}


@ROBOT_REGISTRY.register(name="wuji_hand2_2_r")
class WujiHand2_2_R(_WujiHand2_2):
    file: str = "hands/wuji_hand2_2/wuji_hand2_2_right.urdf"

    # As on ``WujiHand2Beta2_R``: the fingers extend along URDF -Z and the palm normal is +Y, so the hand spawns
    # fingers-up, palm-forward.
    up: tuple[int, int, int] = (0, 0, -1)
    front: tuple[int, int, int] = (0, 1, 0)

    dofs_name: list[str] = [f"r_{joint}" for joint in _JOINTS]

    # ``r_wrist`` is a fixed child of the ``r_mount`` root, which Genesis would merge into the base.
    links_to_keep: list[str] = ["r_wrist", "r_wrist_imu_frame"] + [f"r_{tip}" for tip in _TIPS]

    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        side="right",
        # NOTE: the order of ``target_hand_links`` matches the order of ``mano_hand_idx``.
        target_hand_links=[f"r_{tip}" for tip in _TIPS],
        mano_hand_idx=[16, 17, 18, 19, 20],
    )


@ROBOT_REGISTRY.register(name="wuji_hand2_2_l")
class WujiHand2_2_L(_WujiHand2_2):
    file: str = "hands/wuji_hand2_2/wuji_hand2_2_left.urdf"

    # The left URDF mirrors the right about the palm plane: the palm normal flips to -Y.
    up: tuple[int, int, int] = (0, 0, -1)
    front: tuple[int, int, int] = (0, -1, 0)

    dofs_name: list[str] = [f"l_{joint}" for joint in _JOINTS]

    links_to_keep: list[str] = ["l_wrist", "l_wrist_imu_frame"] + [f"l_{tip}" for tip in _TIPS]

    metadata: ClassVar[MetadataOptions] = MetadataOptions(
        side="left",
        # NOTE: the order of ``target_hand_links`` matches the order of ``mano_hand_idx``.
        target_hand_links=[f"l_{tip}" for tip in _TIPS],
        mano_hand_idx=[16, 17, 18, 19, 20],
    )
