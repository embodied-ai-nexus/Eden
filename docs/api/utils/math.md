# Math and Geometry Utilities

Batched math helpers: quaternion, rotation and pose-transform operations, bilateral symmetry for augmentation,
sampling and seeding, and torch configuration. Quaternions are **wxyz** throughout.

```python
from eden.utils.geom import quat_mul, subtract_frame_transforms, transform_by_quat
```

## Geometry

`eden.utils.geom` re-exports, as the same objects, every `genesis.utils.geom` quaternion and pose-transform helper
that has a batched torch path (`quat_to_R`, `transform_by_quat`, `inv_transform_by_trans_quat`, `slerp`, ...), so
either import works, and adds the helpers Genesis does not have (`quat_mul`, `subtract_frame_transforms`,
`rot6d_to_quat`, `align_up_and_front`, ...). Genesis' Euler converters are deliberately not re-exported: their
convention differs from this module's `quat_from_euler_xyz`. Use these helpers rather than re-implementing quaternion
math at a call site; if an operation is missing, add it here with a test in `tests/utils/test_geom.py`.

::: eden.utils.geom
    options:
      show_root_heading: false

## Symmetry

Mirroring across the robot's XZ plane, and `SymmetryAugmentation` for flat observation and action tensors.

::: eden.utils.symmetry
    options:
      show_root_heading: false

## Sampling and seeding

::: eden.utils.sample
    options:
      show_root_heading: false

::: eden.utils.random
    options:
      show_root_heading: false

## Torch helpers

Backend configuration (TF32, determinism), model compilation, checkpoint inspection, device resolution and
numpy/torch conversion.

::: eden.utils.torch
    options:
      show_root_heading: false

## See also

- [Entities guide](../../guides/entities.md) — pose frames
- [Robotics utilities](robotics.md)
