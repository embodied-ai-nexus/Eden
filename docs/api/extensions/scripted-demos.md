# Scripted Demonstrations

Scripted demonstrators produce demo-like trajectories, the kind a person teleoperating would record: smooth
minimum-jerk-timed motion through randomized keyframes, a per-episode speed, pauses, off-centre grasps, retries and
deliberate mistakes the demonstrator recovers from. They are a port of OCBench's scripted policies onto Eden's
batched envs. `SCRIPTED_DEMO_REGISTRY` says which demonstrator drives which task, and `generate_demos` (what
`eden datagen` runs) rolls one out in parallel envs and records every episode. The
[scripted demonstrations guide](../../guides/scripted-demos.md) covers `eden datagen` and writing a demonstrator.

```python
from eden.extensions.scripted_demos.generate import generate_demos
from eden.extensions.scripted_demos.registry import SCRIPTED_DEMO_REGISTRY
```

## Generation and registry

::: eden.extensions.scripted_demos.generate
    options:
      show_root_heading: false

::: eden.extensions.scripted_demos.registry
    options:
      show_root_heading: false

## Keyframe trajectories and pick-and-place primitives

::: eden.extensions.scripted_demos.trajectory
    options:
      show_root_heading: false

::: eden.extensions.scripted_demos.primitives
    options:
      show_root_heading: false

## Pick-and-place controller

::: eden.extensions.scripted_demos.controller
    options:
      show_root_heading: false

## Block-world demonstrators

::: eden.extensions.scripted_demos.block_world
    options:
      show_root_heading: false

## The robot a demonstrator drives

::: eden.extensions.scripted_demos.rig
    options:
      show_root_heading: false

## Tool routines

A routine is a scripted policy written for one robot as a generator of motions (`move_to`, `move_until_contact`,
`close_gripper`, ...), run batched over every env.

::: eden.extensions.scripted_demos.tools
    options:
      show_root_heading: false

::: eden.extensions.scripted_demos.routines
    options:
      show_root_heading: false

::: eden.extensions.scripted_demos.put_on_routine
    options:
      show_root_heading: false

::: eden.extensions.scripted_demos.multi_arm
    options:
      show_root_heading: false

## See also

- [Scripted demonstrations guide](../../guides/scripted-demos.md)
- [Datasets and recording guide](../../guides/datasets-and-recording.md)
