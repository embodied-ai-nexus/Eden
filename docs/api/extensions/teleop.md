# Teleoperation

Operator-side teleoperation stacks: the process an operator runs to stream commands to a robot process. A stack is a
`TeleopOperator` built from a `TeleopInputSource` (what the operator wears or holds) and a `TeleopStreamer` (what it
sends), configured by `TeleopOptions`. The stack that ships is SONIC whole-body teleoperation: PICO body tracking on
the operator station, sent over ZMQ to the robot process that runs the SONIC policy. The input devices themselves
(keyboard, gamepad, SpaceMouse, GELLO, Meta Quest, PICO, Wuji Glove) are on the
[devices page](../utils/devices.md). The [teleoperation guide](../../guides/teleoperation.md) covers every pipeline
and its `eden` command.

```python
from eden.extensions.teleop.base import TeleopInputSource, TeleopOperator, TeleopStreamer
from eden.options.extensions.teleop import TeleopOptions
```

## Base classes and options

::: eden.extensions.teleop.base
    options:
      show_root_heading: false

::: eden.options.extensions.teleop
    options:
      show_root_heading: false

## SONIC operator station

::: eden.extensions.teleop.sonic.manager
    options:
      show_root_heading: false

::: eden.extensions.teleop.sonic.state_machine
    options:
      show_root_heading: false

::: eden.extensions.teleop.sonic.inputs
    options:
      show_root_heading: false

::: eden.extensions.teleop.sonic.pose_streamer
    options:
      show_root_heading: false

::: eden.extensions.teleop.sonic.planner_streamer
    options:
      show_root_heading: false

::: eden.extensions.teleop.sonic.three_point
    options:
      show_root_heading: false

::: eden.extensions.teleop.sonic.smpl
    options:
      show_root_heading: false

::: eden.extensions.teleop.sonic.wrist
    options:
      show_root_heading: false

::: eden.extensions.teleop.sonic.hands
    options:
      show_root_heading: false

::: eden.extensions.teleop.sonic.feedback
    options:
      show_root_heading: false

::: eden.extensions.teleop.sonic.g1_fk
    options:
      show_root_heading: false

::: eden.extensions.teleop.sonic.viz
    options:
      show_root_heading: false

The robot-side half (the SONIC policy runtime and its ZMQ server) is in `eden.extensions.policies.sonic`; its policy
class is on the [policies page](policies.md).

## See also

- [Teleoperation guide](../../guides/teleoperation.md)
- [Teleoperation devices](../utils/devices.md)
