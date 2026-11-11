"""Skateboard asset for the HUSKY humanoid-skateboarding policy port.

Asset: ``objects/skateboard/skateboard.xml`` in the ``Kashu7100/eden_assets``
dataset, adapted from `TeleHuman/humanoid_skateboarding
<https://github.com/TeleHuman/humanoid_skateboarding>`_ (CC BY-NC 4.0). A
free-floating deck with a tilt joint, two steering trucks, and four free
wheels. MJCF ``equality/joint`` couplings make the trucks steer when the deck
tilts (``tilt = -0.577 * truck_roll``, a ~60 deg rake) — Genesis parses these
natively, so leaning the board steers it exactly as in the source MuJoCo model.

The source model centers the tilt and truck joints with position actuators
(kp/kd from the compiled training scene, targets 0). Genesis anchors passive
joint springs at ``qpos0`` — which is 0 for every skateboard joint — so plain
``default_dofs_stiffness``/``default_dofs_damping`` reproduce those servos
without an action term. The wheels are torque-free; their MJCF friction
(sliding 1.0) carries over, and ``default_armature=0.0`` keeps Genesis's 0.1
armature default off the wheel joints (0.1 is ~1e4x the wheel's real rotary
inertia and would stop the board from rolling).
"""

from eden.options.entities import EntityOptions


class HuskySkateboard(EntityOptions):
    """HUSKY skateboard: deck + coupled steering trucks + free wheels."""

    file: str = "objects/skateboard/skateboard.xml"

    is_articulated: bool = True
    # MJCF declaration order (= Genesis qpos order after the free base).
    dofs_name: list[str] = [
        "board_tilt_joint",
        "front_truck_roll_joint",
        "front_left_wheel_joint",
        "front_right_wheel_joint",
        "rear_truck_roll_joint",
        "rear_left_wheel_joint",
        "rear_right_wheel_joint",
    ]

    # Deck center 0.10 m above ground: wheel axles sit at -0.07 relative to the
    # deck and the wheel radius is 0.03, so the wheels touch a z=0 plane.
    default_root_pos: tuple[float, float, float] = (0.0, 0.0, 0.10)

    # Joints carry no MJCF armature; without this Genesis defaults them to 0.1.
    default_armature: float = 0.0

    # Centering springs = the source model's position servos (targets are 0 and
    # Genesis anchors springs at qpos0=0). Gains are the compiled training
    # scene's actuator gainprm/biasprm values.
    default_dofs_stiffness: dict[str, float] = {
        "board_tilt_joint": 14.67,
        "*_truck_roll_joint": 10.0,
        "*_wheel_joint": 0.0,
    }
    default_dofs_damping: dict[str, float] = {
        "board_tilt_joint": 0.402,
        "*_truck_roll_joint": 0.5,
        "*_wheel_joint": 0.0,
    }
