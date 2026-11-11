# Motion Planning

Motion planning for one entity of a built env: a planner takes a goal in its backend's form (joint space or task
space) and returns joint-space waypoints. A planner is a `MotionPlannerBase` subclass registered in
`MOTION_PLANNER_REGISTRY` and configured by `PlannerOptions`; two backends ship, cuRobo and OMPL. Each imports
its library lazily and reports whether it is installed (`IS_CUROBO_AVAILABLE`, `IS_OMPL_AVAILABLE`). The
[planning and sysid guide](../../guides/planning-and-sysid.md) shows them in use.

```python
from eden.extensions.motion_planning.base import MOTION_PLANNER_REGISTRY, MotionPlannerBase
from eden.options.extensions.motion_planning import PlannerOptions
```

## Base class and registry

::: eden.extensions.motion_planning.base
    options:
      show_root_heading: false

## Options

::: eden.options.extensions.motion_planning
    options:
      show_root_heading: false

## cuRobo

::: eden.extensions.motion_planning.curobo.planner
    options:
      show_root_heading: false

## OMPL

::: eden.extensions.motion_planning.ompl.planner
    options:
      show_root_heading: false

## See also

- [Planning and sysid guide](../../guides/planning-and-sysid.md)
- [Grasp sampling](grasp-sampling.md)
