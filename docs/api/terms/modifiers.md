# Modifiers

Modifiers post-process a term's input or output. **Action modifiers** sit between an action term and the actuator
and model its imperfections (delay, friction, effort clipping, motor strength, backlash, deadband); chain several
with `Compose`. On the observation side, **noise models** corrupt a term's output during training only, and the
**observation modifiers** in a term's `modifiers=[...]` list (a moving average, a delay) run in every mode. The
[actions guide](../../guides/actions.md) and [observations guide](../../guides/observations.md) show them in
context.

```python
from eden.managers.modifiers.actions.actuators import ActionDelay, Compose
from eden.managers.modifiers.observations.noise import UniformNoise
```

## Base classes and registries

::: eden.managers.modifiers.base
    options:
      show_root_heading: false

## Action modifiers

::: eden.managers.modifiers.actions.actuators
    options:
      show_root_heading: false

## Observation noise

::: eden.managers.modifiers.observations.noise
    options:
      show_root_heading: false

## Temporal observation modifiers

::: eden.managers.modifiers.observations.temporal
    options:
      show_root_heading: false

## See also

- [Modifier options](../options/managers.md#modifiers)
- [Domain-randomization events](events.md#domain-randomization), several of which drive an action modifier
