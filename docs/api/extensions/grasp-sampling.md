# Grasp Sampling

Grasp samplers propose 6-DoF grasps of an object in a built env, from a camera's point cloud or the object's own
cloud, for a given end effector. A sampler is a `GraspSamplerBase`
subclass registered in `GRASP_SAMPLER_REGISTRY` and returns a `GraspResult`. Two backends ship: GraspGen (one model
per gripper) and GraspGenX (one cross-embodiment model, the gripper selected by name). The
[grasp sampling guide](../../guides/grasp-sampling.md) covers installation and use.

```python
from eden.extensions.grasp_sampling.base import GRASP_SAMPLER_REGISTRY, GraspResult, GraspSamplerBase
from eden.options.extensions.grasp_sampling import GraspSamplerOptions
```

## Base class, result and registry

::: eden.extensions.grasp_sampling.base
    options:
      show_root_heading: false

## Options

::: eden.options.extensions.grasp_sampling
    options:
      show_root_heading: false

## GraspGenX

::: eden.extensions.grasp_sampling.graspgenx.sampler
    options:
      show_root_heading: false

## GraspGen

::: eden.extensions.grasp_sampling.graspgen.sampler
    options:
      show_root_heading: false

## See also

- [Grasp sampling guide](../../guides/grasp-sampling.md)
- [Motion planning](motion-planning.md)
