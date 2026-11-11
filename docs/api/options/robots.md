# Robot Presets

Robot presets are `RobotOptions` subclasses with the asset, DOF order, PD gains, armature and limits of one robot
already filled in; construct one and put it in a scene. Composite rigs bundle several presets (an arm and a gripper,
a humanoid and two hands) into one validated options object. The [robots guide](../../guides/robots.md) covers the
catalog, composite rigs and adding a robot.

```python
import eden as en
from eden.options.robots import COMPOSITE_ROBOT_REGISTRY, ROBOT_REGISTRY, Droid

robot = en.robots.Go1()
rig = Droid(base_pos=(0.0, 0.0, 0.4))           # hand-less Panda + Robotiq 2F-85
cls = COMPOSITE_ROBOT_REGISTRY.get("ur5e_2f85")  # -> UR5eRobotiq
```

`ROBOT_REGISTRY` looks unit robots up by key and `COMPOSITE_ROBOT_REGISTRY` composite rigs. Each section below
renders one module of `eden.options.robots`.

## Humanoids

### Unitree G1

::: eden.options.robots.unitree_g1
    options:
      show_root_heading: false
      show_if_no_docstring: true

### LimX Oli

::: eden.options.robots.limx_oli
    options:
      show_root_heading: false
      show_if_no_docstring: true

### SMPL humanoids

::: eden.options.robots.smpl_humanoid
    options:
      show_root_heading: false
      show_if_no_docstring: true

## Quadrupeds

### Unitree Go1

::: eden.options.robots.unitree_go1
    options:
      show_root_heading: false
      show_if_no_docstring: true

### Unitree Go2 and Go2-W

::: eden.options.robots.unitree_go2
    options:
      show_root_heading: false
      show_if_no_docstring: true

## Arms

### Franka Emika Panda and Franka Research 3

::: eden.options.robots.franka_emika_panda
    options:
      show_root_heading: false
      show_if_no_docstring: true

### Universal Robots UR5e

::: eden.options.robots.universal_ur5e
    options:
      show_root_heading: false
      show_if_no_docstring: true

### Universal Robots UR10e

::: eden.options.robots.universal_ur10e
    options:
      show_root_heading: false
      show_if_no_docstring: true

### UFACTORY xArm7

::: eden.options.robots.ufactory_xarm7
    options:
      show_root_heading: false
      show_if_no_docstring: true

### KUKA LBR iiwa 14

::: eden.options.robots.kuka_iiwa14
    options:
      show_root_heading: false
      show_if_no_docstring: true

### i2RT YAM

::: eden.options.robots.i2rt_yam
    options:
      show_root_heading: false
      show_if_no_docstring: true

### Trossen WidowX-250

::: eden.options.robots.trossen_widowx250
    options:
      show_root_heading: false
      show_if_no_docstring: true

### ALOHA (ViperX 300 S)

::: eden.options.robots.aloha_vx300s
    options:
      show_root_heading: false
      show_if_no_docstring: true

### OmniReset UR5e rig

::: eden.options.robots.omnireset_ur5e_2f85
    options:
      show_root_heading: false
      show_if_no_docstring: true

### LIBERO Panda

::: eden.options.robots.libero
    options:
      show_root_heading: false
      show_if_no_docstring: true

## Grippers

### Franka Hand

::: eden.options.robots.franka_hand
    options:
      show_root_heading: false
      show_if_no_docstring: true

### Robotiq 2F-85

::: eden.options.robots.robotiq_2f85
    options:
      show_root_heading: false
      show_if_no_docstring: true

## Dexterous hands

### RobotEra XHand1

::: eden.options.robots.robotera_xhand
    options:
      show_root_heading: false
      show_if_no_docstring: true

### Inspire RH56

::: eden.options.robots.inspire_rh56
    options:
      show_root_heading: false
      show_if_no_docstring: true

### LEAP Hand

::: eden.options.robots.leap_hand
    options:
      show_root_heading: false
      show_if_no_docstring: true

### SharPa Wave

::: eden.options.robots.sharpa_wave
    options:
      show_root_heading: false
      show_if_no_docstring: true

### Tesollo DG-5F

::: eden.options.robots.tesollo_dg5f
    options:
      show_root_heading: false
      show_if_no_docstring: true

### Wuji Hand 1

::: eden.options.robots.wuji_hand1
    options:
      show_root_heading: false
      show_if_no_docstring: true

### Wuji Hand2 (beta2)

::: eden.options.robots.wuji_hand2beta2
    options:
      show_root_heading: false
      show_if_no_docstring: true

### MANO hand model

::: eden.options.robots.mano_hand
    options:
      show_root_heading: false
      show_if_no_docstring: true

## Composite robots

A composite is a `CompositeEntityOptions`: it round-trips through the config system and is added to a scene by name
like any other entity, and the scene expands it into its unit robots plus the welds that join them. It exposes the
controller metadata as properties (`arm_dofs` on every arm rig, plus `gripper_dofs`, `gripper_open`,
`gripper_close`, ... on a `ManipulatorOptions` and `hand_dofs` on a `DexManipulatorOptions`; `robot_dofs`,
`left_hand_dofs`, `right_hand_dofs` on a humanoid rig). `ManipulatorOptions` carries a two-finger
gripper, `DexManipulatorOptions` a dexterous hand, and `HumanoidOptions` one end-effector per wrist.

### Composite base classes

::: eden.options.robots.composite.base
    options:
      show_root_heading: false
      show_if_no_docstring: true

### Droid

::: eden.options.robots.composite.droid
    options:
      show_root_heading: false
      show_if_no_docstring: true

### Panda + Franka Hand

::: eden.options.robots.composite.panda_hand
    options:
      show_root_heading: false
      show_if_no_docstring: true

### UR5e / UR10e + Robotiq 2F-85

::: eden.options.robots.composite.ur_robotiq
    options:
      show_root_heading: false
      show_if_no_docstring: true

### Franka + XHand (LeFranX)

::: eden.options.robots.composite.franka_xhand
    options:
      show_root_heading: false
      show_if_no_docstring: true

### xArm7 + XHand

::: eden.options.robots.composite.xarm_xhand
    options:
      show_root_heading: false
      show_if_no_docstring: true

### KUKA iiwa 14 + SharPa Wave

::: eden.options.robots.composite.kuka_sharpa
    options:
      show_root_heading: false
      show_if_no_docstring: true

### G1 + Inspire hands

::: eden.options.robots.composite.g1_inspire
    options:
      show_root_heading: false
      show_if_no_docstring: true

### G1 + Wuji hands

::: eden.options.robots.composite.g1_wuji
    options:
      show_root_heading: false
      show_if_no_docstring: true

### Oli + Wuji hands

::: eden.options.robots.composite.oli_wuji
    options:
      show_root_heading: false
      show_if_no_docstring: true

## See also

- [Robots guide](../../guides/robots.md)
- [Actuator options](actuators.md)
- [Entity options](entities.md)
