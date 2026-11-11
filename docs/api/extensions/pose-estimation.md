# Pose Estimation

Pose estimators recover human motion from monocular video as an AMASS-style SMPL-X clip that the retargeting
extensions consume. An estimator is a `PoseEstimatorBase` subclass configured by `PoseEstimatorOptions`; the backend
that ships is GVHMR, installed with the `pose` extra (`pip install 'eden-world[pose]'`, or
`uv sync --inexact --extra pose` in a checkout) and imported lazily. The
[pose estimation guide](../../guides/pose-estimation.md) covers `eden pose gvhmr`.

```python
from eden.extensions.pose_estimation.base import PoseEstimatorBase
from eden.options.extensions.pose_estimation import PoseEstimatorOptions
```

## Base class and options

::: eden.extensions.pose_estimation.base
    options:
      show_root_heading: false

::: eden.options.extensions.pose_estimation
    options:
      show_root_heading: false

## GVHMR

::: eden.extensions.pose_estimation.gvhmr.estimator
    options:
      show_root_heading: false

### Conversion to an AMASS-style clip

GVHMR's world-frame output is Y-up; these helpers rotate it to Eden's Z-up and write an AMASS-style SMPL-X `.npz`.

::: eden.extensions.pose_estimation.gvhmr.convert
    options:
      show_root_heading: false

## See also

- [Pose estimation guide](../../guides/pose-estimation.md)
- [Retargeting](retargeting.md)
