# Rendering Utilities

The batched Gaussian-splat renderer behind `SplatCamera` and `BatchSplatCamera` (gsplat backend), its `.ply`
loading and conversion, and the shadow transfer that darkens a composited background where the simulation casts a
shadow. The [sensors and rendering guide](../../guides/sensors-and-rendering.md) explains when each is used.

## Batched splat renderer

::: eden.utils.batch_splat_renderer.renderer
    options:
      show_root_heading: false

::: eden.utils.batch_splat_renderer.gaussiandata
    options:
      show_root_heading: false

::: eden.utils.batch_splat_renderer.batch_rasterization
    options:
      show_root_heading: false

## PLY files

::: eden.utils.batch_splat_renderer.ply_io
    options:
      show_root_heading: false

::: eden.utils.splat_ply_converter
    options:
      show_root_heading: false

## Background shadows

::: eden.utils.background_shadows
    options:
      show_root_heading: false

## See also

- [Splat and renderer options](../options/splat.md)
- [Cameras and overlays](../entities.md#cameras)
