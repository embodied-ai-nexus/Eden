# Deployment

A deployment backend runs a trained task policy on a real robot, or against a MuJoCo model of it for sim-to-sim
validation. It builds the task's own env for its observation and action terms, reads the robot's state, and sends
the action terms' commands over the robot's SDK at the control rate. The [deployment guide](../../guides/deployment.md)
covers `eden deploy`, every supported robot, safety and timing.

```python
from eden.extensions.deployment.base import DEPLOYMENT_REGISTRY, DeploymentBase, resolve_deployment_for
from eden.options.extensions.deployment import DeploymentOptions
```

A backend registers in `DEPLOYMENT_REGISTRY` and is bound to a robot options class with `bind_deployment`;
`resolve_deployment_for(robot_options)` picks the backend for a task's robot. Importing `eden.extensions.deployment`
imports every backend module so those bindings exist; a backend whose SDK is missing is skipped. The MuJoCo backend
is robot-agnostic and is looked up by name (`eden deploy ... --mujoco`), and a composite rig resolves to
`CompositeDeployment`, which drives each part through that part's own binding.

## Base class and registry

::: eden.extensions.deployment.base
    options:
      show_root_heading: false
      members:
        - DeploymentBase
        - SoftShutdownMixin
        - DEPLOYMENT_REGISTRY
        - bind_deployment
        - resolve_deployment_for
        - resolve_pinned_gains
        - resolve_dofs_values

## Options

::: eden.options.extensions.deployment
    options:
      show_root_heading: false

## Hardware backends

### Unitree G1

::: eden.extensions.deployment.unitree_g1
    options:
      show_root_heading: false

### Unitree Go2

::: eden.extensions.deployment.unitree_go2
    options:
      show_root_heading: false

### Unitree Go1

::: eden.extensions.deployment.unitree_go1
    options:
      show_root_heading: false

### LimX Oli

::: eden.extensions.deployment.limx_oli
    options:
      show_root_heading: false

### Franka Panda

::: eden.extensions.deployment.franka_panda
    options:
      show_root_heading: false

### RobotEra XHand

::: eden.extensions.deployment.robotera_xhand
    options:
      show_root_heading: false

### Wuji Hand

::: eden.extensions.deployment.wuji_hand
    options:
      show_root_heading: false

### Composite rigs

::: eden.extensions.deployment.composite
    options:
      show_root_heading: false

## MuJoCo sim-to-sim

::: eden.extensions.deployment.mujoco
    options:
      show_root_heading: false

::: eden.extensions.deployment.mujoco_composite
    options:
      show_root_heading: false

The MuJoCo world itself (`MujocoSim`, `MujocoRig`, the `MujocoFloor` both add), what both MuJoCo backends read off
the built env (`mujoco_scene.py`), the offscreen recording (`MujocoVideo`, `mujoco_video.py`), the
Eden-robot-to-Menagerie map (`menagerie_path_for`), the
state and command dataclasses (`RobotState`, `RobotCommand`) and the loop helpers (`RateLimiter`,
`StreamingRecorder`, `LinVelKalmanEstimator`) live in `eden/extensions/deployment/utils/`, which has no
`__init__.py`, so they are not rendered here; read them in the source.

## See also

- [Deployment guide](../../guides/deployment.md)
- [Policy export](../../guides/policy-export.md)
- [System identification](sysid.md)
