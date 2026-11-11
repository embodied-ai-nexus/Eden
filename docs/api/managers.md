# Managers

A manager owns one aspect of the step loop (actions, observations, rewards, ...) and holds the **terms** that
implement it. Each section below documents one manager module: the manager class, its term base class, and, for
the kinds that accept plain functions, the function wrapper. For how to write a term, read the
[terms guide](../guides/terms.md); for the order managers run in, read the
[environments guide](../guides/environments.md).

```python
from eden.managers import REWARD_TERM_REGISTRY, RewardTerm
from eden.terms import DoneTerm, ObsTerm, RewardTerm  # short aliases of the function wrappers
```

Some term kinds declare their shared fields (a reward's `weight`, an event's `mode`, ...) on a private mixin. Those
blocks list the fields with `inherited_members` so they render on the term classes.

## Base classes

`ManagerTermBase` is the base of every term, `ConfigurableTermBase` of every class term, and
`ConfigurableTermFuncWrapper` of every function wrapper. `ManagerBase` builds terms from options through
`build_term`, which adds a note naming the config entry to any exception a term's `build()` raises.

::: eden.managers.base
    options:
      show_root_heading: false

## ActionManager

::: eden.managers.action_manager
    options:
      show_root_heading: false

## ObservationManager

::: eden.managers.observation_manager
    options:
      show_root_heading: false
      inherited_members: [effective_scale]

## RewardManager

::: eden.managers.reward_manager
    options:
      show_root_heading: false
      inherited_members: [weight, range_s, tags, is_temporal_dependent]

## TerminationManager

::: eden.managers.termination_manager
    options:
      show_root_heading: false
      inherited_members: [time_out]

## CommandManager

::: eden.managers.command_manager
    options:
      show_root_heading: false

## EventManager

::: eden.managers.event_manager
    options:
      show_root_heading: false
      inherited_members: [mode, interval_range_s, is_global_time, min_step_count_between_reset, priority]

## CurriculumManager

`CurriculumManager.get_state(name)` returns what another curriculum term published this step and raises, naming
the fix, whenever there is no such value. See the [curricula guide](../guides/curricula.md).

::: eden.managers.curriculum_manager
    options:
      show_root_heading: false

## MetricManager

::: eden.managers.metric_manager
    options:
      show_root_heading: false
      inherited_members: [success_threshold, direction, metric_mode, is_cumulative, contributes_to_success]

## RecorderManager

::: eden.managers.recorder_manager
    options:
      show_root_heading: false

## MotionManager

The motion manager of `MimicRLEnvBase`, with `MotionTerm` and `MOTION_TERM_REGISTRY`, is documented on the
[motion reference page](motion_reference.md#motion-manager-and-motionterm) together with the collaborators
`MotionTerm` delegates to.

## Reset-configuration sampling

Which reset configuration each env starts its next episode from, for tasks that reset from a bank of states.

::: eden.managers.resets.sampling
    options:
      show_root_heading: false

## Term catalog

The machinery behind `eden terms` and the generated [term catalog](terms/catalog.md).

::: eden.managers.catalog
    options:
      show_root_heading: false
      members:
        - TermKind
        - TermEntry
        - KINDS
        - builtin_catalog
        - render_markdown
        - render_index
        - main

## Registries

Each term kind has a registry. A term registers under `to_snake_case(<ClassOrFunctionName>)`; `register()` takes no
`name=`.

| Registry | Import |
|----------|--------|
| `ACTION_TERM_REGISTRY` | `from eden.managers import ACTION_TERM_REGISTRY` |
| `OBSERVATION_TERM_REGISTRY` | `from eden.managers import OBSERVATION_TERM_REGISTRY` |
| `REWARD_TERM_REGISTRY` | `from eden.managers import REWARD_TERM_REGISTRY` |
| `TERMINATION_TERM_REGISTRY` | `from eden.managers import TERMINATION_TERM_REGISTRY` |
| `COMMAND_TERM_REGISTRY` | `from eden.managers import COMMAND_TERM_REGISTRY` |
| `EVENT_TERM_REGISTRY` | `from eden.managers import EVENT_TERM_REGISTRY` |
| `CURRICULUM_TERM_REGISTRY` | `from eden.managers import CURRICULUM_TERM_REGISTRY` |
| `METRIC_TERM_REGISTRY` | `from eden.managers import METRIC_TERM_REGISTRY` |
| `RECORDER_TERM_REGISTRY` | `from eden.managers import RECORDER_TERM_REGISTRY` |
| `MOTION_TERM_REGISTRY` | `from eden.managers.motion_manager import MOTION_TERM_REGISTRY` |

The registry classes themselves are documented under [core utilities](utils/core.md#registries).

## See also

- [Terms guide](../guides/terms.md)
- [Built-in terms](terms/index.md) and the [term catalog](terms/catalog.md)
- [Manager options](options/managers.md)
