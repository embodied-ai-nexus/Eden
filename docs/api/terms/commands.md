# Command Terms

Command terms produce the goal a policy is conditioned on (a base velocity, a target pose, a gait phase, a motion
clip) and resample it on their own schedule. Other terms read a command by its config key through
`env.command_manager`. The [commands guide](../../guides/commands.md) covers resampling and debug visualization;
the [term catalog](catalog.md#command-terms) lists every parameter.

```python
from eden import commands
```

The base class `CommandTerm` is on the [managers page](../managers.md#commandmanager).

## Velocity

SE(2) base-velocity commands: sampled, held constant, or driven by an operator's input device.

::: eden.managers.terms.commands.velocity
    options:
      show_root_heading: false

## Target poses

::: eden.managers.terms.commands.oracle
    options:
      show_root_heading: false

## Gait phase

::: eden.managers.terms.commands.gait
    options:
      show_root_heading: false

## Motion clips

`MotionCommand` plays `.npz` motion clips on a plain `RLEnvBase`, for frozen-policy inference and single-clip
tracking. Training-grade motion tracking goes through `MimicRLEnvBase` and the
[motion manager](../motion_reference.md) instead.

::: eden.managers.terms.commands.mimic
    options:
      show_root_heading: false

## See also

- [Commands guide](../../guides/commands.md)
- [Command manager options](../options/managers.md#commands)
