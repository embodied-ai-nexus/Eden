# Splat and Renderer Options

Options for photorealistic backgrounds and the non-default renderers. `SplatOptions` configures a Gaussian-splat
background rendered with gsplat and `NyxSplatOptions` one rendered with the Nyx path tracer; `PCDOptions` loads a
raw point cloud. `RayTracerOptions` and `BatchRendererOptions` are Eden's versions of Genesis' ray tracer and
Madrona batch renderer options. The [sensors and rendering guide](../../guides/sensors-and-rendering.md) explains
when to use each renderer, and the [scenes guide](../../guides/scenes-and-terrains.md) covers real-to-sim splat
scenes.

```python
from eden.options.splat import SplatOptions
from eden.options.renderer import BatchRendererOptions
```

A camera shows the splat only when its `CameraOptions.overlays` declares a `SplatComposite`
([camera overlays](../entities.md#camera-overlays)).

## Gaussian splats and point clouds

::: eden.options.splat
    options:
      show_root_heading: false

## Renderers

::: eden.options.renderer
    options:
      show_root_heading: false

## See also

- [Sensors and rendering guide](../../guides/sensors-and-rendering.md)
- [Gaussian-splat environments](../envs.md#gaussian-splat-environments) and [configs](../configs.md#gaussian-splat-configs)
- [Splat rendering utilities](../utils/rendering.md)
