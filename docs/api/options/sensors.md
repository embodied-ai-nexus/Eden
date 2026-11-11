# Sensor and Camera Options

`SensorsOptions` and `CamerasOptions` are the `sensors_options` and `cameras_options` fields of a config. Each holds
named children: one `SensorOptions` or `CameraOptions` per sensor or camera, keyed by the name the env exposes it
under (`env.sensors[...]`, `env.cameras[...]`). A `SensorOptions` wraps any Genesis sensor options (`gs.sensors.*`,
or one of Eden's [engine sensors](../engine.md#audio-sensors)) and says which entity and link to attach it to. The
[sensors and rendering guide](../../guides/sensors-and-rendering.md) explains each sensor and camera mode.

```python
from eden.options.camera import CameraOptions, CamerasOptions
from eden.options.sensors import SensorOptions, SensorsOptions
```

## Sensors

::: eden.options.sensors.SensorOptions

::: eden.options.sensors.SensorsOptions

## Cameras

::: eden.options.camera.CameraOptions

::: eden.options.camera.CamerasOptions

::: eden.options.camera.CameraOverlayOptions

The overlay classes themselves (`PlanePaintOverlay`, `SplatComposite`, ...) are on the
[entities page](../entities.md#camera-overlays).

## See also

- [Sensors and rendering guide](../../guides/sensors-and-rendering.md)
- [Entities: cameras and sensors](../entities.md#cameras)
- [Sensor observations](../terms/observations.md#sensors), [camera observations](../terms/observations.md#cameras)
