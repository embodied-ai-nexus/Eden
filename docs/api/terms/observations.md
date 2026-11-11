# Observation Terms

Observation terms read the simulation, commands and motion reference into the tensors a policy sees. Observation
`compute()` is a read, not a tick: it runs at build, at reset, in several step phases and from
`get_observations()`, so per-step state advances in `post_physics_refresh()`. The
[observations guide](../../guides/observations.md) covers groups, noise, history and modifiers; the
[term catalog](catalog.md#observation-terms) lists every parameter.

```python
from eden import observations
```

The base classes `ObservationTerm` and `ObservationTermFuncWrapper` are on the
[managers page](../managers.md#observationmanager).

## Proprioception

::: eden.managers.terms.observations.proprio
    options:
      show_root_heading: false

## Locomotion

::: eden.managers.terms.observations.locomotion
    options:
      show_root_heading: false

## Commands, actions and episode state

::: eden.managers.terms.observations.common
    options:
      show_root_heading: false

## Privileged dynamics

Per-env domain-randomized quantities (gain and motor-strength scales, added mass, foot friction, CoM offset) for a
critic or a teacher policy.

::: eden.managers.terms.observations.privileged
    options:
      show_root_heading: false

## Manipulation

::: eden.managers.terms.observations.manipulation
    options:
      show_root_heading: false

## Contact

::: eden.managers.terms.observations.contact
    options:
      show_root_heading: false

## Sensors

::: eden.managers.terms.observations.sensors
    options:
      show_root_heading: false

## Cameras

::: eden.managers.terms.observations.camera
    options:
      show_root_heading: false

## Motion tracking

Terms that expose `env.motion_manager` reference state on a `MimicRLEnvBase`.

::: eden.managers.terms.observations.tracking
    options:
      show_root_heading: false

## See also

- [Observations guide](../../guides/observations.md)
- [Observation modifiers and noise](modifiers.md)
- [Observation manager options](../options/managers.md#observations)
