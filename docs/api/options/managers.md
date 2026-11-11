# Manager Options

Each manager has a `*ManagerOptions` container and a `*TermOptions` class for one of its terms. You rarely build a
term options object by hand: `<Term>.configure(...)` returns one. The container holds terms as named fields, and the
name is the key other code uses (`env.reward_manager`, `get_term("<name>")`, a reward's `command_name`, ...).
Observation and reward terms are grouped one level further (`ObservationGroupOptions`, `RewardGroupOptions`). The
[terms guide](../../guides/terms.md) explains configuring terms.

```python
from eden.options.managers.observations import ObservationGroupOptions, ObservationManagerOptions
from eden.options.managers.rewards import RewardManagerOptions
```

## Base

`ManagerTermOptions` is the base of every term's options; `FunctionTermOptions` adds the `params` dict that
`configure(func=...)` fills, for the kinds that accept function terms. `ManagerOptions` is the base of the containers.

::: eden.options.managers.base
    options:
      show_root_heading: false

## Actions

::: eden.options.managers.actions
    options:
      show_root_heading: false

## Observations

::: eden.options.managers.observations
    options:
      show_root_heading: false

## Rewards

::: eden.options.managers.rewards
    options:
      show_root_heading: false

## Terminations

::: eden.options.managers.terminations
    options:
      show_root_heading: false

## Commands

::: eden.options.managers.commands
    options:
      show_root_heading: false

## Events

::: eden.options.managers.events
    options:
      show_root_heading: false

## Curricula

::: eden.options.managers.curricula
    options:
      show_root_heading: false

## Metrics

::: eden.options.managers.metrics
    options:
      show_root_heading: false

## Recorders

::: eden.options.managers.recorders
    options:
      show_root_heading: false

## Motions

::: eden.options.managers.motions
    options:
      show_root_heading: false

## Modifiers

Options for action modifiers, observation modifiers and noise models.

::: eden.options.managers.modifiers
    options:
      show_root_heading: false

## See also

- [Terms guide](../../guides/terms.md)
- [Managers](../managers.md)
- [Built-in terms](../terms/index.md)
