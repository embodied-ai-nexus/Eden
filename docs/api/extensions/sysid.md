# System Identification

System identification fits per-DOF dynamics parameters of an entity (`damping`, `armature`, `stiffness`,
`frictionloss`, and the PD gains `kp` / `kd`; masses are not fittable) so the
simulation reproduces recorded trajectories. A `SystemIdentificationBase` rolls candidate `ParameterSet`s out in
parallel environments, scores them with a residual against `Trajectory` recordings, and runs an optimizer (CMA-ES or
SciPy least squares). Excitation signals drive the robot while you record, and `DeploymentRecorder` captures
real-robot trajectories during a deployment. The [planning and sysid guide](../../guides/planning-and-sysid.md)
walks through a fit.

```python
from eden.extensions.sysid import CMAES, Parameter, ParameterSet, SYSID_REGISTRY, Trajectory
from eden.options.extensions.sysid import SystemIdentificationOptions
```

## Base class, registry and result

::: eden.extensions.sysid.base
    options:
      show_root_heading: false

## Options

::: eden.options.extensions.sysid
    options:
      show_root_heading: false

## Parameters and trajectories

::: eden.extensions.sysid.parameter
    options:
      show_root_heading: false

::: eden.extensions.sysid.trajectory
    options:
      show_root_heading: false

## Optimizers

::: eden.extensions.sysid.optimizers.cmaes
    options:
      show_root_heading: false

::: eden.extensions.sysid.optimizers.scipy_ls
    options:
      show_root_heading: false

## Excitation signals

::: eden.extensions.sysid.excitation
    options:
      show_root_heading: false

## Rollouts, residuals and applying parameters

::: eden.extensions.sysid.rollout
    options:
      show_root_heading: false

::: eden.extensions.sysid.residual
    options:
      show_root_heading: false

::: eden.extensions.sysid.modifier
    options:
      show_root_heading: false

## Recording on the robot

::: eden.extensions.sysid.deployment_recorder
    options:
      show_root_heading: false

## Reporting

::: eden.extensions.sysid.report
    options:
      show_root_heading: false

## See also

- [Planning and sysid guide](../../guides/planning-and-sysid.md)
- [System-identification recorder term](../terms/recorders.md#system-identification)
- [Deployment](deployment.md)
