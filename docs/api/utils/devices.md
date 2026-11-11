# Teleoperation Devices

Input-device backends for teleoperation and for driving a policy by hand: keyboard, gamepad, SpaceMouse and GUI
sliders for SE(2)/SE(3) commands, the GELLO leader arm, the Meta Quest and PICO headsets, and the Wuji Glove. Each
device imports its driver lazily, so its optional extra is needed only when you use it. The
[teleoperation guide](../../guides/teleoperation.md) covers the `eden gello`, `eden metaquest`, `eden pico`,
`eden wuji` and `eden teleop` pipelines.

```python
from eden.utils.devices.se2 import make_se2_device
```

## Base class

::: eden.utils.devices.base
    options:
      show_root_heading: false

## SE(2) and SE(3) devices

`make_se2_device(name)` builds a keyboard or gamepad SE(2) device by name; `OperatorVelocityCommand`
(a [velocity command term](../terms/commands.md#velocity)) reads one.

::: eden.utils.devices.se2
    options:
      show_root_heading: false

::: eden.utils.devices.keyboard
    options:
      show_root_heading: false

::: eden.utils.devices.gamepad
    options:
      show_root_heading: false

::: eden.utils.devices.spacemouse
    options:
      show_root_heading: false

::: eden.utils.devices.gui_slider
    options:
      show_root_heading: false

## GELLO leader arm

::: eden.utils.devices.gello.leader
    options:
      show_root_heading: false

::: eden.utils.devices.gello.calibration
    options:
      show_root_heading: false

::: eden.utils.devices.gello.dynamixel_driver
    options:
      show_root_heading: false

::: eden.utils.devices.gello.zmq_server
    options:
      show_root_heading: false

::: eden.utils.devices.gello.zmq_client
    options:
      show_root_heading: false

## Meta Quest

A WebXR bridge process (televuer) streams controller poses to Eden over ZMQ.

::: eden.utils.devices.metaquest.xr_bridge
    options:
      show_root_heading: false

::: eden.utils.devices.metaquest.protocol
    options:
      show_root_heading: false

::: eden.utils.devices.metaquest.pose_mapping
    options:
      show_root_heading: false

## PICO

The XRoboToolkit tracking client, the bridge process, and the Remote Vision video link that streams Eden's stereo
frames to the headset.

::: eden.utils.devices.pico.xr_client
    options:
      show_root_heading: false

::: eden.utils.devices.pico.bridge
    options:
      show_root_heading: false

::: eden.utils.devices.pico.video
    options:
      show_root_heading: false

::: eden.utils.devices.pico.protocol
    options:
      show_root_heading: false

## Wuji Glove

::: eden.utils.devices.wuji.glove
    options:
      show_root_heading: false

::: eden.utils.devices.wuji.protocol
    options:
      show_root_heading: false

::: eden.utils.devices.wuji.monitor
    options:
      show_root_heading: false

## See also

- [Teleoperation guide](../../guides/teleoperation.md)
- [Teleoperation extensions](../extensions/teleop.md)
