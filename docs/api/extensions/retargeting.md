# Retargeting

Retargeting maps motion captured on one morphology (human keypoints, SMPL-family bodies, MANO hands, interaction
meshes) onto an Eden robot. The four solvers are `RetargetingBase` subclasses configured by a `RetargeterOptions`, and the
FlashDexRetarget pipeline is a set of functions with its own options; each imports its backend lazily. The [retargeting guide](../../guides/retargeting.md) covers `eden retarget` and the
body backends; the [dex retargeting guide](../../guides/dex-retargeting.md) covers hands.

```python
from eden.extensions.retargeting.base import RetargetingBase
from eden.options.extensions.retargeting import RetargeterOptions
```

| Backend | Maps | CLI |
|---|---|---|
| `DexRetargeter` | MANO hand keypoints → a dexterous robot hand | `eden retarget dex` |
| `OmniRetargeter` | human motion with an interaction mesh → a humanoid (Laplacian deformation) | `eden retarget omniretarget` |
| `OTRetargeter` | a human surface and objects → a robot and objects (optimal transport) | `eden retarget otretarget` |
| `SomaRetargeter` | SMPL / SMPL-H / SMPL-X / SOMA → a floating-base humanoid (Unitree G1, LimX Oli), through the `eden-soma-retargeter` library | `eden retarget soma` |
| FlashDexRetarget | MANO hand-object demos → a bimanual robot-hand `motion.pt` | `eden retarget flashdex` |

## Base class and options

::: eden.extensions.retargeting.base.RetargetingBase
    options:
      show_bases: true

::: eden.options.extensions.retargeting.RetargeterOptions
    options:
      show_bases: true

::: eden.extensions.retargeting.constants
    options:
      show_root_heading: false

## DexRetargeter

::: eden.extensions.retargeting.dex_retargeter.retargeter
    options:
      show_root_heading: false

::: eden.extensions.retargeting.dex_retargeter.optimizer
    options:
      show_root_heading: false

::: eden.extensions.retargeting.dex_retargeter.presets
    options:
      show_root_heading: false

::: eden.extensions.retargeting.dex_retargeter.calibrate
    options:
      show_root_heading: false

::: eden.extensions.retargeting.dex_retargeter.constants
    options:
      show_root_heading: false

::: eden.extensions.retargeting.dex_retargeter.utils
    options:
      show_root_heading: false

## OmniRetarget

::: eden.extensions.retargeting.omniretarget.retargeter
    options:
      show_root_heading: false

::: eden.extensions.retargeting.omniretarget.g1_presets
    options:
      show_root_heading: false

::: eden.extensions.retargeting.omniretarget.utils
    options:
      show_root_heading: false

## OTRetarget

Joint robot and object retargeting from surface interactions via entropic optimal transport. `OTRetargeter` is the
solver; `FloorStance` and `HandGrasp` are its add-ons; the presets bundle G1 and LimX Oli rigs, with or without Wuji
hands.

::: eden.extensions.retargeting.otretarget.retargeter
    options:
      show_root_heading: false

::: eden.extensions.retargeting.otretarget.presets
    options:
      show_root_heading: false

::: eden.extensions.retargeting.otretarget.stance
    options:
      show_root_heading: false

::: eden.extensions.retargeting.otretarget.grasp
    options:
      show_root_heading: false

::: eden.extensions.retargeting.otretarget.human
    options:
      show_root_heading: false

::: eden.extensions.retargeting.otretarget.sources
    options:
      show_root_heading: false

::: eden.extensions.retargeting.otretarget.supports
    options:
      show_root_heading: false

::: eden.extensions.retargeting.otretarget.kinematics
    options:
      show_root_heading: false

::: eden.extensions.retargeting.otretarget.channels
    options:
      show_root_heading: false

::: eden.extensions.retargeting.otretarget.probes
    options:
      show_root_heading: false

::: eden.extensions.retargeting.otretarget.transport
    options:
      show_root_heading: false

::: eden.extensions.retargeting.otretarget.capsules
    options:
      show_root_heading: false

::: eden.extensions.retargeting.otretarget.solver
    options:
      show_root_heading: false

::: eden.extensions.retargeting.otretarget.metrics
    options:
      show_root_heading: false

## SomaRetargeter

`SomaRetargeter` runs the `eden-soma-retargeter` library in a worker process. That extra pins a `trimesh` that only
a checkout's uv override reconciles with Genesis, so install it with `uv sync --inexact --extra soma-retargeter`
(it cannot be pip-installed).

::: eden.extensions.retargeting.soma_retargeter.retargeter
    options:
      show_root_heading: false

## FlashDexRetarget

The kinematic half of FlashDexRetarget: import a released hand-object dataset, preprocess and filter its clips,
retarget each with physics IK in MuJoCo, and pack the kept clips into `motion.pt`.

::: eden.options.extensions.flashdexretarget
    options:
      show_root_heading: false

::: eden.extensions.retargeting.flashdexretarget.importer
    options:
      show_root_heading: false

::: eden.extensions.retargeting.flashdexretarget.pipeline
    options:
      show_root_heading: false

::: eden.extensions.retargeting.flashdexretarget.preprocess
    options:
      show_root_heading: false

::: eden.extensions.retargeting.flashdexretarget.retarget
    options:
      show_root_heading: false

::: eden.extensions.retargeting.flashdexretarget.pack
    options:
      show_root_heading: false

::: eden.extensions.retargeting.flashdexretarget.clips
    options:
      show_root_heading: false

::: eden.extensions.retargeting.flashdexretarget.filters
    options:
      show_root_heading: false

::: eden.extensions.retargeting.flashdexretarget.quality
    options:
      show_root_heading: false

## See also

- [Retargeting guide](../../guides/retargeting.md)
- [Dex retargeting guide](../../guides/dex-retargeting.md)
- [Datasets](../datasets.md)
