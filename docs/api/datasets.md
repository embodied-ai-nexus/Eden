# Datasets

Motion datasets load human or robot motion (motion capture, retargeted clips, hand-object recordings) into the
`BatchMotionSequence` layout the [motion manager](motion_reference.md) plays back. Each loader is a `DatasetBase`
subclass that fetches its files, parses them, caches the result and applies `DatasetModifier`s. The
[motion datasets guide](../guides/motion-datasets.md) covers downloading, licences and the runnable examples.

```python
from eden.datasets import DATASET_REGISTRY, LAFAN1RetargetedG1Dataset
from eden.options.datasets import DatasetOptions
```

Datasets and modifiers register under a snake-case key derived from the class name.

## Base class and registry

::: eden.datasets.base
    options:
      show_root_heading: false
      members:
        - DatasetBase
        - DATASET_REGISTRY
        - cache_identity

## Options

::: eden.options.datasets
    options:
      show_root_heading: false

## Humanoid motion

::: eden.datasets.amass_dataset.AMASSRetargetedG1Dataset
    options:
      show_bases: true
      inherited_members: false

::: eden.datasets.lafan1_dataset.LAFAN1RetargetedG1Dataset
    options:
      show_bases: true
      inherited_members: false

::: eden.datasets.seed_dataset.SEEDRetargetedG1Dataset
    options:
      show_bases: true
      inherited_members: false

::: eden.datasets.phuma_dataset.PHUMADataset
    options:
      show_bases: true
      inherited_members: false

## Human-object and human-human interaction

::: eden.datasets.omomo_dataset.OMOMOHolosomaDataset
    options:
      show_bases: true
      inherited_members: false

::: eden.datasets.weave_dataset.WeaveG1InspireHOIDataset
    options:
      show_bases: true
      inherited_members: false

::: eden.datasets.hhi_assist_dataset.HHIAssistDataset
    options:
      show_bases: true
      inherited_members: false

## Hand-object manipulation

::: eden.datasets.arctic_dataset.ArcticDataset
    options:
      show_bases: true
      inherited_members: false

::: eden.datasets.dex_ycb_dataset.DexYCBDataset
    options:
      show_bases: true
      inherited_members: false

### MANO hand-object releases

TACO, OakInk2 and HOT3D ship MANO parameters rather than keypoints, so their loaders need the licence-gated MANO
model files (`body_model_dir`). They share `HandObjectDatasetBase`.

::: eden.datasets.hand_object_base
    options:
      show_root_heading: false
      members:
        - HandObjectDatasetBase
        - HandObjectClip
        - MANOHand

::: eden.datasets.taco_dataset.TacoDataset
    options:
      show_bases: true
      inherited_members: false

::: eden.datasets.oakink2_dataset.OakInk2Dataset
    options:
      show_bases: true
      inherited_members: false

::: eden.datasets.hot3d_dataset.Hot3dDataset
    options:
      show_bases: true
      inherited_members: false

## Modifiers

Offline transforms applied to loaded sequences, in order. `DatasetModifierOptions.motion_sequence_keys` selects
the sequences each one targets.

::: eden.datasets.modifiers.base
    options:
      show_root_heading: false

::: eden.datasets.modifiers.transforms
    options:
      show_root_heading: false

::: eden.datasets.modifiers.temporal
    options:
      show_root_heading: false

::: eden.datasets.modifiers.noise
    options:
      show_root_heading: false

::: eden.datasets.modifiers.ground
    options:
      show_root_heading: false

::: eden.datasets.modifiers.smooth_start_end
    options:
      show_root_heading: false

::: eden.datasets.modifiers.terrain_adapt
    options:
      show_root_heading: false

## Body models

Parametric body models (MANO, SMPL, SMPL-H, SMPL-X) behind the opt-in `body_model_params=True` fields, run on a
numpy linear-blend-skinning rig with no `smplx` dependency.

::: eden.datasets.utils.body_model
    options:
      show_root_heading: false

## BVH parsing

::: eden.datasets.utils.bvh
    options:
      show_root_heading: false

## Keypoint names

::: eden.datasets.constants
    options:
      show_root_heading: false

## See also

- [Motion datasets guide](../guides/motion-datasets.md)
- [Motion reference](motion_reference.md)
- [Retargeting extensions](extensions/retargeting.md)
