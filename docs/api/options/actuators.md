# Actuator Options

Actuator options describe a physical motor (rotor inertia, gear ratio, effort and velocity limits, ...) so a robot
preset can derive its armature and PD gains from the hardware instead of hand-tuned numbers. `ActuatorSpecOptions`
is the base; the presets below model the motors of the built-in Unitree, DAMIAO and LimX robots. The
[robots guide](../../guides/robots.md) shows how a robot preset uses them.

```python
from eden.options.actuators import Unitree7520_14
```

::: eden.options.actuators
    options:
      show_root_heading: false

The helpers that turn a motor spec into armature and gains (`reflected_inertia`, `raibert_heuristic_kp`, ...) are on
the [robotics utilities page](../utils/robotics.md#actuators).

## See also

- [Robots guide](../../guides/robots.md)
- [Robot presets](robots.md)
