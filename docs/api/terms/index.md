# Built-in Terms

Eden ships ready-made terms for every manager: action controllers, observations, rewards, terminations,
commands, events, curricula, metrics and recorders, plus the modifiers that post-process term inputs and outputs.
This section renders their API docs, one page per kind and one section per module.

**To find what a term accepts**, use the [term catalog](catalog.md). It is generated from the registries, so it
lists every registered term with its parameters, their types and defaults, and which are required. In a terminal:

```bash
eden terms --grep height           # terms whose name, parameters or summary match a regex
eden terms --kind reward           # one kind only
eden terms --json                  # the whole catalog, for tooling
```

## Using a built-in term

Configure a term in a config with `.configure(...)`, which returns its serializable options. A class term takes its
parameters as keyword arguments. A function term goes through the kind's function wrapper (short aliases in
`eden.terms`), with its parameters in `params={...}`. Those aliases are the function wrappers only:
`eden.terms.RewardTerm` is `RewardTermFuncWrapper`, not the class-term base `eden.managers.reward_manager.RewardTerm`
that shares its name (the same holds for `EventTerm` and `MetricTerm`). Subclass the base from its manager module. Excerpted from `eden/tasks/benchmark/reacher/config.py` and
`eden/tasks/benchmark/move_forward/config.py`:

```python
from eden import observations, rewards
from eden.terms import RewardTerm

dofs_pos = observations.DofsPos.configure(entity_name="robot")
joint_vel_hinge = RewardTerm.configure(
    func=rewards.dofs_vel_limits,
    weight=-0.01,
    params={"entity_name": "robot", "max_vel": 0.5},
)
tracking_lin_vel = rewards.TrackingLinVel.configure(
    weight=1.0,
    command_name="velocity_command",
    entity_name="robot",
    sigma=0.25,
)
```

`from eden import observations, rewards, terminations, ...` imports the built-in term modules lazily, on first
use. A term is looked up by its registry key, never by object identity, so a class term you write yourself must be
registered with `@<KIND>_TERM_REGISTRY.register()`. See the [terms guide](../../guides/terms.md).

## Function wrappers

::: eden.terms
    options:
      show_root_heading: false

## Pages

| Kind | Page | Lazy namespace |
|---|---|---|
| Actions | [Action terms](actions.md) | `eden.actions` |
| Observations | [Observation terms](observations.md) | `eden.observations` |
| Rewards | [Reward terms](rewards.md) | `eden.rewards` |
| Terminations | [Termination terms](terminations.md) | `eden.terminations` |
| Commands | [Command terms](commands.md) | `eden.commands` |
| Events | [Event terms](events.md) | `eden.events` |
| Curricula | [Curriculum terms](curricula.md) | `eden.curricula` |
| Metrics | [Metric terms](metrics.md) | `eden.metrics` |
| Recorders | [Recorder terms](recorders.md) | `eden.recorders` |
| Modifiers | [Modifiers](modifiers.md) | — |

## Shared term helpers

Building blocks the built-in terms reuse: resolve-once helpers for `build()` (`resolve_link_idx`,
`resolve_terrain`, `resolve_metric_manager`, ...) and cheap math helpers for `compute()` (`projected_gravity`,
`base_height_above_ground`, ...).

::: eden.managers.terms.utils
    options:
      show_root_heading: false

## See also

- [Terms guide](../../guides/terms.md)
- [Managers](../managers.md)
- [Manager options](../options/managers.md)
