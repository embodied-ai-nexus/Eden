# Curricula

A curriculum term changes the task while it trains: it widens a command range on a schedule, ramps a penalty
weight up as the policy gets better, or tightens a termination tolerance as the success rate rises. This page
covers when curriculum terms run, the built-in terms, what happens to them at evaluation, and how to write one
that other terms can read.

Curricula exist only on RL environments, configured through `EdenRLConfig.curriculum_options`.

## A first example

The velocity-tracking tasks widen the commanded velocity in three stages
(`eden/tasks/velocity_tracking/base_config.py`):

```python
from eden.managers.terms import VelocityCommandCurriculum
from eden.options.managers.curricula import CurriculumManagerOptions

curriculum_options = CurriculumManagerOptions(
    command_vel=VelocityCommandCurriculum.configure(
        command_name="velocity_command",
        velocity_stages=[
            {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
            {"step": 5000 * 24, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
            {"step": 10000 * 24, "lin_vel_x": (-2.0, 3.0)},
        ],
    ),
)
```

`step` is compared with `env.common_step_counter`, the number of control steps (`env.step()` calls) the env has
taken. All envs step together, so it is not multiplied by `num_envs`. With RSL-RL collecting
`num_steps_per_env=24` steps per iteration (that task's runner setting), `5000 * 24` is reached after 5000
iterations.

The reacher benchmark raises a penalty weight the same way (`eden/tasks/benchmark/reacher/config.py`):

```python
from eden.managers.terms.curricula.rewards import StageRewardWeightCurriculum

curriculum_options = CurriculumManagerOptions(
    reaching_weight=StageRewardWeightCurriculum.configure(
        reward_term_name="joint_vel_hinge",
        weight_stages=[(0, -0.01), (100 * 24, -0.1), (200 * 24, -1.0)],
    )
)
```

## When curriculum terms run

| Phase | Call | Runs at eval? |
|---|---|---|
| 13 | `CurriculumManager.reset(envs_idx)`: log every term's last published state, then call each class term's `reset(envs_idx)` for the envs that finished an episode | logging only (there is nothing to log); term `reset()` does not run |
| 16 | `CurriculumManager.compute()`: call every term's `compute()` in declaration order and cache what it returns | no |

Phase 16 is after the auto-reset and before the commands are published (phase 17), so a range a curriculum
writes takes effect on the next resample. The full step order is in [Environments](environments.md).

`compute()` runs on **every control step**, not once per iteration. A term that moves a level by a fixed amount
per call (`PenaltyCurriculum` moves its scale by `degree`) moves it once per step.

A term's `reset(envs_idx)` is where it accumulates the statistics `compute()` adapts from: an episode-length
average, a success rate. It runs during the reset at phase 13, before `episode_length_buf` is zeroed, so
`env.episode_length_buf[envs_idx]` still holds the finished episodes' lengths.

`compute()` returns the term's published state: a dict of scalars, logged as `Curriculum/<term_name>/<key>`
(a non-dict value is logged as `Curriculum/<term_name>`).

### Curricula are frozen at eval

When the env is built with `eval_mode=True` (`env.training` is `False`), the curriculum manager's `compute()` and
the terms' `reset()` do nothing. An eval rollout's difficulty is therefore fixed, and no `Curriculum/*` key is
logged for the whole rollout.

What a term holds at eval is what it installed at build. That is the level the **config** pins, not the level the
training run ended at:

- `VelocityCommandCurriculum` and `StageRewardWeightCurriculum` write nothing until their first `compute()`, so
  at eval the command ranges and the reward weight are the ones configured on the command and reward terms.
- `PenaltyCurriculum` installs `initial_scale` on its reward terms in `build()`, so at eval those terms run at
  `initial_scale` times their configured weight.

To evaluate at a later level, set that level in the config the eval builds from.

!!! note "A resumed run restarts every curriculum"
    `env.common_step_counter` starts at 0 when an env is constructed, and Eden does not restore it, or any
    curriculum term's internal state, from a checkpoint. A run resumed from a checkpoint replays its
    step-scheduled stages from the first one, and adaptive terms restart their averages.

## Built-in curriculum terms

All built-ins live in `eden.managers.terms.curricula` (also `eden.curricula`). `eden terms --kind curriculum`
prints them with every parameter, and the generated [term catalog](../api/terms/catalog.md) has the same list.

| Term | Drives | Adapts from |
|---|---|---|
| `VelocityCommandCurriculum` | the `lin_vel_x_range`, `lin_vel_y_range`, `ang_vel_z_range` of command term `command_name` | `velocity_stages`: dicts with a `step` and any of `lin_vel_x`, `lin_vel_y`, `ang_vel_z` |
| `StageRewardWeightCurriculum` | the weight of reward term `reward_term_name` | `weight_stages`: `(step, weight)` pairs; the latest stage reached wins |
| `PenaltyCurriculum` | a scale on the weights of several reward terms (`configured_weight * scale`) | the average episode length: up by `degree` above `level_up_threshold`, down below `level_down_threshold`, within `[min_scale, max_scale]` |
| `AverageEpisodeLengthTracker` | nothing; publishes `average_episode_length` and `is_primed` | an EMA of episode length |
| `MotionBasePoseThresholdCurriculum` | the `pos_threshold` / `rot_threshold` of a `MotionBasePoseTrackingBreak` termination | an EMA of the per-episode success rate (an episode succeeds when its clip runs out rather than the pose gate firing) |
| `SuccessGuidedSampling` | which configuration each resetting env draws, for the RESET event that binds it as its `sampler` | each configuration's success rate over its last `history` episodes (graded by termination `success_term`); draws peak at `target` |

Each term resolves what it drives in `build()`, so a wrong `command_name`, `reward_term_name` or
`termination_term_name` raises at env construction: the command, termination and reward managers are built
before the curriculum manager.

`PenaltyCurriculum` selects its reward terms either by `reward_term_names` or by `reward_term_tags` (matched
against each reward term's `tags`), never both; setting both or neither raises. A tag selection that matches no
term logs a warning and does nothing. It normally keeps its own buffer of the last `num_episode_average` episode
lengths; with `tracker_name` it reads an `AverageEpisodeLengthTracker` instead, which several penalty curricula
can share:

```python
from eden.managers.terms.curricula import AverageEpisodeLengthTracker, PenaltyCurriculum

curriculum_options = CurriculumManagerOptions(
    # Declared first: the consumers below read it in the same step.
    ep_tracker=AverageEpisodeLengthTracker.configure(num_compute_average_epl=1000),
    penalty_torques=PenaltyCurriculum.configure(reward_term_tags=["penalty"], tracker_name="ep_tracker"),
    penalty_smoothness=PenaltyCurriculum.configure(reward_term_tags=["smoothness"], tracker_name="ep_tracker"),
)
```

The tags are the ones you set on the targeted reward terms (`tags=["penalty"]` in their `configure()`), not the
curriculum's own key. `AverageEpisodeLengthTracker`'s effective window is
`max(1, int(num_compute_average_epl * num_envs / 4096))` resets, so the same setting gives the same time constant
at any env count; it reports `is_primed=False` until its first finished episode, and `PenaltyCurriculum` does not
move its scale before then.

Task packages add their own, for example `TerrainLevelCurriculum` (`eden/tasks/extreme_parkour/custom_terms/curricula.py`)
and `GoalRangeCurriculum` (`eden/tasks/manipulation/custom_terms/curricula.py`).

## Reading another term's state

`env.curriculum_manager.get_state(name)` returns what the named term published **in this step**. It never returns
`None` and never serves an older value: every way there can be no current state raises, and each message names
the fix.

| Situation | Raises | Fix |
|---|---|---|
| No curriculum term has that name | `KeyError`, listing the declared terms | Use one of the listed keys, or add the term. |
| The env is in eval mode (every step of the rollout) | `RuntimeError` | Read the term itself (below). |
| The named term is declared after the term reading it | `RuntimeError`, naming both | Move the producer above the consumer in `curriculum_options`. |
| The read happens before the run's first `compute()` (in `build()`, the first reset, task setup) | `RuntimeError` | Read the term itself. |
| The named term's `compute()` returned `None` | `RuntimeError` | Return a dict from that term's `compute()`. |
| The read runs earlier in the step than phase 16 (a curriculum `reset()`, a metric, a termination, a reward) | `RuntimeError`, naming both steps | Read the term itself, or move the read after phase 16. |

Terms compute in declaration order, so a producer must be declared above every curriculum term that reads it. A
term must publish on every step: one that returns a dict on some steps and `None` on others makes `get_state`
raise on exactly the steps it skipped.

!!! tip "Need the level at eval too? Read the term, not the manager"
    `get_state` raises for the whole of an eval rollout, by design: a consumer that fell back to a default there
    would grade the eval at a difficulty no training step used. A term that needs the current level in both modes
    resolves the curriculum term once in `build()` with `env.curriculum_manager.get_term(name)` and reads the
    attribute it exposes. `ObjectVelocityDisturbance` does this with `AdaptiveEpisodeCurriculum.value`
    (`eden/tasks/reorient/custom_terms/`). Do not catch the error and substitute a default.

Do not call `CurriculumManager.compute()` yourself to refresh a value: terms adapt as a side effect of computing,
so a second call in one step advances the curriculum twice.

## Reset-configuration samplers

A curriculum term can also decide which stored start state each env resets to. A RESET event that owns a fixed
set of starts, such as `ResetFromStateBuffer`, names a curriculum term with `sampler="<config key>"`; that term must
subclass `ResetConfigSampler` (`eden/managers/resets/sampling.py`) next to `CurriculumTerm`. The event resolves it
in `build()` (the curriculum manager is built before the event manager), binds it to the set's size once, and asks
it for indices on every reset. A name that is missing, or a term that is not a `ResetConfigSampler`, raises at env
construction. Without `sampler`, rows are drawn uniformly. See [Events](events.md).

`SuccessGuidedSampling` is the built-in sampler (Success Guided Sampling, Zhang et al., CoRL 2026). It keeps the
last `history` outcomes of every configuration, grades each finished episode in the RESET event that starts the next, and draws the
next configuration from a softmax over a Beta-shaped kernel peaked at `target`, so most episodes go to
configurations the policy solves sometimes. Its defaults are the paper's locomotion settings, it draws uniformly
at eval, and a term no reset event binds raises at env construction. Extreme parkour wires it in with
`--terrain_sampler sgs`.

## Writing a curriculum term

Subclass `CurriculumTerm`, register it, resolve what it drives in `build()`, and return a dict from `compute()`:

```python
from eden.managers import CURRICULUM_TERM_REGISTRY, CurriculumTerm


@CURRICULUM_TERM_REGISTRY.register()
class LinearSpeedRamp(CurriculumTerm):
    """Widen a velocity command's x range linearly over ramp_steps control steps."""

    command_name: str = "velocity_command"
    start_speed: float = 0.5
    max_speed: float = 2.0
    ramp_steps: int = 100_000

    def build(self) -> None:
        self._command = self._env.command_manager.get_term(self.command_name)

    def compute(self, *args, **kwargs) -> dict[str, float]:
        frac = min(self._env.common_step_counter / self.ramp_steps, 1.0)
        speed = self.start_speed + frac * (self.max_speed - self.start_speed)
        self._command.lin_vel_x_range = (-speed, speed)
        return {"max_speed": speed}
```

- `compute()` runs every control step in training, so keep it cheap and free of GPU→CPU syncs. A per-episode
  statistic belongs in `reset(envs_idx)`; reduce it on the device (`eden.utils.misc.sum_over_envs`), as
  `MotionBasePoseThresholdCurriculum.reset` does, because `envs_idx` is the auto-reset's bool mask on most steps.
- Install the starting level in `build()` if eval should see it: `compute()` never runs at eval.
- Expose the current level as an attribute if other terms need it in eval as well as in training.

A function term works too, through `eden.terms.CurrTerm.configure(func=..., params={...})`. Its function is called
as `func(env=env, **params)` once per step and its return value is published the same way.

## See also

- [Commands](commands.md) — the command terms `VelocityCommandCurriculum` drives.
- [Rewards and terminations](rewards-and-terminations.md) — reward weights, tags and the termination terms curricula tighten.
- [Training runs](training-runs.md) — iterations, `num_steps_per_env`, resuming.
- [Terms](terms.md) — the general term lifecycle.
- [Term catalog](../api/terms/catalog.md) — every registered curriculum term with its parameters.
- API reference: [`CurriculumManager`](../api/managers.md), [curriculum terms](../api/terms/curricula.md), [options](../api/options/managers.md).
