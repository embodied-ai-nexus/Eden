# Motion Optimization

Offline optimization of motion clips before training. The one optimizer that ships is TCRS, which turns
flat-ground biped clips into terrain-adapted ones: gait-phase and contact detection, MPPI foot-swing planning against
a terrain height field, support-aware pelvis-height filtering, and whole-body Jacobian IK on a MuJoCo model of the
robot. It needs no Eden env (numpy, scipy and mujoco only). Datasets reach it through the `TCRSTerrainAdapt`
[dataset modifier](../datasets.md#modifiers). The [TCRS terrain adaptation guide](../../guides/tcrs-terrain-adaptation.md)
explains the method and its settings.

```python
from eden.extensions.motion_optimization.tcrs import HeightFieldTerrain, TCRSPlannerOptions, TCRSRobotOptions
```

## Options

::: eden.options.extensions.motion_optimization
    options:
      show_root_heading: false

## Pipeline

::: eden.extensions.motion_optimization.tcrs.pipeline
    options:
      show_root_heading: false

## Motion clips and terrain references

::: eden.extensions.motion_optimization.tcrs.motion
    options:
      show_root_heading: false

## Gait phase

::: eden.extensions.motion_optimization.tcrs.gait_phase
    options:
      show_root_heading: false

## Foot-swing optimization (MPPI)

::: eden.extensions.motion_optimization.tcrs.mppi_foot
    options:
      show_root_heading: false

## Pelvis-height filtering

::: eden.extensions.motion_optimization.tcrs.terrain_warp
    options:
      show_root_heading: false

## Whole-body IK

::: eden.extensions.motion_optimization.tcrs.ik
    options:
      show_root_heading: false

## Terrain sources

::: eden.extensions.motion_optimization.tcrs.terrain_source
    options:
      show_root_heading: false

## Synthetic demo clip

::: eden.extensions.motion_optimization.tcrs.demo
    options:
      show_root_heading: false

## See also

- [TCRS terrain adaptation guide](../../guides/tcrs-terrain-adaptation.md)
- [Dataset modifiers](../datasets.md#modifiers)
