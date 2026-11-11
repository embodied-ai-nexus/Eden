# Rewards and Terminations

The reward manager turns a set of weighted reward terms into the scalar reward of each step, and the termination
manager decides which episodes end — and whether each ending is a true terminal state or a time-out. This page
covers how the reward is computed, weights, time windows, tags and reward groups, what gets logged, how
terminations split into terminated and truncated, and the built-in terms. The general term contract is in
[Writing terms](terms.md).

Both managers belong to the RL env (`RLEnvBase`). In the [step loop](environments.md#the-step-loop) terminations
run at phase 8 and rewards at phase 9, both on the post-physics state of the step that just happened and before
the auto-reset, so a reward term can read this step's termination verdicts.

## The step reward

Each entry of `reward_options` is one reward term with a `weight`. The step reward is

```text
reward = dt * Σ_i  weight_i * mask_i * term_i
```

where `dt` is the control step (`EnvOptions.sim_dt * EnvOptions.decimation`), `term_i` is the term's
`(num_envs,)` output and `mask_i` is 1 except outside the term's `range_s` window. The reward is multiplied by
`dt`, so weights are per second: a term that outputs 1 every step with weight 1 contributes 1 over each second of
episode, whatever the control rate. When you port weights from a framework that does not scale by `dt` (or
does), convert them.

An excerpt of the velocity-tracking rewards (`eden/tasks/velocity_tracking/base_config.py`):

```python
import math

from eden import rewards
from eden.options.managers.rewards import RewardManagerOptions
from eden.terms import RewardTerm

reward_options = RewardManagerOptions(
    track_linear_velocity=RewardTerm.configure(
        func=rewards.track_lin_vel_with_z_penalty,
        weight=2.0,
        params={"command_name": "velocity_command", "entity_name": "robot", "std": math.sqrt(0.25)},
    ),
    joint_pos_limits=RewardTerm.configure(
        func=rewards.dofs_pos_limits,
        weight=-1.0,
        params={"entity_name": "robot"},
    ),
    action_rate_l2=RewardTerm.configure(func=rewards.action_rate_l2, weight=-0.1),
    action_rate_l2_smooth=rewards.ActionRateL2Smooth.configure(weight=-0.05),
    # feet_link_names is filled in per robot by the task's robot-binding TaskMod.
    foot_slip=rewards.FeetSlip.configure(
        weight=-0.1,
        entity_name="robot",
        feet_link_names=[],
        command_name="velocity_command",
        command_threshold=0.05,
    ),
)
```

Penalty terms output a non-negative cost and take a **negative** weight; the sign lives in the weight, not in
the term.

A few things happen around the sum:

- **A weight of `0.0` means the term is not built.** The manager logs a warning naming the entry, the term never
  runs and gets no log key, and reading it (`get_term_value`) raises a `KeyError` that says why. To keep a term
  that a curriculum will switch on later, give it a small non-zero weight.
- **Envs Genesis flagged as corrupt** this step (a NaN contact force, a diverged solve) contribute 0 for every
  term. Those envs are reset in the same step; see [Terminations](#terminations).
- **Term names must be unique** across the whole reward manager, groups included.

Reward terms are importable from `eden.rewards` (`from eden import rewards`); the full list is in the
[Term Catalog](../api/terms/catalog.md#reward-terms) and from `eden terms --kind reward`.

## Reward term fields

Every reward term — function or class — has these fields beside its own parameters:

| Field | Default | Meaning |
|---|---|---|
| `weight` | `1.0` | multiplies the term's output in the sum |
| `range_s` | `None` | a time window `(start_s, end_s)` in seconds since the episode began, or a list of them, during which the term is active; both ends inclusive. Outside the window the term contributes 0 and is not accumulated in its episode sums. `None` is active for the whole episode |
| `tags` | `[]` | labels for grouping terms (`["penalty"]`, `["tracking", "feet"]`), so curricula and other consumers can select terms by tag instead of by name |

`range_s` is converted to integer step bounds once at build, `round(start_s / dt)` to `round(end_s / dt)`, and compared against the env's `episode_length_buf`. The getup task
(`eden/tasks/mujoco_playground/getup/config.py`) only rewards resembling the default pose between 1 and 6 s:

```python
similar_to_default=RewardTerm.configure(
    func=rewards.similar_to_default,
    weight=-1.0,
    params={"entity_name": "robot"},
    range_s=[(1.0, 6.0)],
),
```

## Changing weights during training

A term's `weight` can change between steps; the manager compares the terms' weights with its device copy each
step and uploads them only when one moved. Curricula are the usual way to move them — for example
`StageRewardWeightCurriculum` (stepwise weights for one term, used by the reacher task) and `PenaltyCurriculum`
(scales a set of terms selected by `reward_term_names` or by `reward_term_tags`). Curricula are frozen during
evaluation. See [Curricula](curricula.md).

## Reward groups

Entries of `reward_options` may be flat terms, named **groups** of terms (`RewardGroupOptions`), or a mix. Groups
never change the scalar reward — the manager always sums every term — but they add a per-group readout for
multi-channel consumers: a CMDP-style task/style/penalty split, a multi-critic algorithm, per-group logging. The
ConTrack task (`eden/tasks/contrack/config.py`) splits its reward three ways, abridged:

```python
from eden.options import RewardGroupOptions
from eden.options.managers.rewards import RewardManagerOptions
from eden.terms import RewardTerm

reward_options = RewardManagerOptions(
    rg=RewardGroupOptions(  # object tracking (task)
        tracking_obj_pos=RewardTerm.configure(
            func=motion_base_position_tracking_exp,
            weight=500.0,
            params={"entity_name": "object", "std": _OBJ_POS_STD},
        ),
    ),
    rs=RewardGroupOptions(  # joint and contact imitation (style)
        contact_match=MotionContactMatchReward.configure(weight=100.0, ...),
    ),
    rp=RewardGroupOptions(  # smoothness penalties
        qvel_penalty_arm=RewardTerm.configure(func=contrack_dofs_vel_penalty, weight=-1.0, params={...}),
    ),
)
```

Flat terms belong to an implicit group named `"default"` (`eden.managers.DEFAULT_REWARD_GROUP`), which exists only
when at least one flat term is declared; iterate `reward_manager.group_term_names` rather than assuming it.
Unlike observation groups, reward groups **partition** one flat set of terms: a term name may appear only once
across all groups.

To override a grouped term, address it through its group — `{"rg": {"tracking_obj_pos": {...}}}` in a config
override, `config.reward_options.rg.tracking_obj_pos` in code. A flat-name override of a grouped term raises,
naming the group to go through. `reward_options.disable_term(name)` removes a flat term, a whole group, or a term
nested in exactly one group; an ambiguous name raises and tells you to call `disable_term` on the group.

## Reading rewards

`env.reward_manager` exposes the last step's values and the running episode sums. Every per-term readout raises
`KeyError` listing the active terms for an unknown name.

| Method | Returns |
|---|---|
| `get_term_value(name)` | the term's output at the last step, `(num_envs,)`: masked, zero for corrupt envs, **without** weight and `dt` — what an off-policy learner that weights its own reward channels stores |
| `get_group_reward(group)` | the group's share of the last step's scalar reward: weighted, masked and `dt`-scaled. The groups' shares add up to the scalar reward |
| `get_episode_sum(name, raw=True)` | the running episode sum of one term; `raw=True` is unweighted, `raw=False` weighted |
| `get_group_episode_sum(group, *, raw)` | the same for a group; `raw` is keyword-only and has no default |
| `iter_terms()`, `group_term_names` | the active terms, flat, and the active term names by group |
| `summary()` | a table of the active terms with their weights (and groups) |

The per-step readouts are the manager's own buffers, overwritten in place by the next step: clone a value to
keep it. After an auto-reset they still read the step that just ended, consistent with the reward `env.step()`
returned.

## What gets logged

When envs reset, the reward manager reports, averaged over the envs that reset and divided by
`episode_length_s`:

| Key | Value | Emitted |
|---|---|---|
| `Episode_Reward/<term>` | the weighted episode sum | when per-term logging is on |
| `Episode_Reward_Raw/<term>` | the unweighted episode sum | when per-term logging is on |
| `Episode_Reward_Group/<group>` | the sum of the group's `Episode_Reward` values | whenever the config declares groups |

`RewardManagerOptions.log_per_term_rewards` controls per-term logging. `None` (the default) turns it **on** for a
config with no groups and **off** for one that declares groups, where the group sums are the intended readout —
and logs an info line at build saying so. Set `True` to always emit per-term keys, or `False` to never. It
changes logging only, never the reward.

The termination manager reports `Episode_Termination/<term>`: how many of the envs that reset this step were
ended by that term (an env can be counted by several). `Episode_Termination/solver_error` counts the envs reset
because a solver flagged them.

## Terminations

Each entry of `termination_options` is a termination term returning a `(num_envs,)` bool. An env whose episode
any term ends is reset at phase 13 of the same step, so the observation `env.step()` returns for it is already
its next episode's first frame. Set `EnvOptions.record_final_observations=True` to also get the terminal frame,
in `extras["final_observations"]`.

The velocity-tracking tasks use the two most common terms:

```python
import math

from eden import terminations
from eden.options.managers.terminations import TerminationManagerOptions
from eden.terms import DoneTerm

termination_options = TerminationManagerOptions(
    time_out=DoneTerm.configure(func=terminations.time_out, time_out=True),
    fell_over=DoneTerm.configure(
        func=terminations.fell_over,
        params={"entity_name": "robot", "limit_angle": math.radians(70.0)},
    ),
)
```

### Terminated or truncated: `time_out`

Every termination term has a `time_out` field (default `False`), and it decides what kind of ending the term
reports:

- **`time_out=False`** — the episode reached a true terminal state (the robot fell, the object dropped). The env
  lands in `terminated`, and a learner does **not** bootstrap the value of the next state.
- **`time_out=True`** — the episode was cut off by a deadline, not by anything the agent did. The env lands in
  `truncated` (the manager's `timeouts`), and a learner bootstraps from the value of the final state.

`env.step()` returns `(obs, reward, terminated, timeouts, extras)`. An env can be in both on one step — it failed
on the step its episode timed out — and is then terminal. The rsl_rl wrapper reports
`extras["time_outs"] = truncated & ~terminated` on every step, which is what rsl_rl bootstraps from.

Mark a term `time_out=True` only when the ending is genuinely not the agent's doing. A deadline you want treated
as a real terminal — a task that must be finished within the episode — is declared with `time_out=False`.

`env.termination_manager` exposes `dones`, `terminated`, `timeouts`, `solver_errors`, `get_term_dones(name)` and
`summary()`.

### Episode length and the `time_out` term

The episode horizon is `EnvOptions.episode_length_s` (default `20.0` s); the env converts it to
`max_episode_length = ceil(episode_length_s / dt)` control steps. Nothing ends an episode at that length on its
own: the built-in `terminations.time_out` term does, by returning `episode_length_buf >= max_episode_length`. So
include it, with `time_out=True`, in every task that should time out.

`episode_length_s` must be positive and finite. For an unbounded episode, keep a finite horizon and leave the
`time_out` term out — truncation then never fires. (`max_episode_length` still sizes integer buffers and the
rsl_rl runner, and `episode_phase` observations and the per-episode log averages divide by it.)

### Corrupt envs

When a physics solver flags an env as corrupt (a NaN contact force, a diverged implicit solve), the termination
manager resets it as **truncated**, never terminated, regardless of what any term says about its state: the
simulator failed, not the task, so the learner bootstraps the step instead of reading it as a terminal outcome.
The reward and metric managers drop what their terms read of that env on that step. The name `solver_error` is
reserved for the log key; a termination term named `solver_error` raises at build.

## Built-in termination terms

Termination terms are importable from `eden.terminations` (`from eden import terminations`); the full list is in
the [Term Catalog](../api/terms/catalog.md#termination-terms) and from `eden terms --kind termination`.

| Family | Terms |
|---|---|
| Time and success | `time_out`, `on_success`, `on_failure`, `OnMetricSuccess`, `MetricAbove`, `cumulative_reward_below_threshold` |
| Contact | `illegal_contact` (which *entities* touched), `IllegalLinkContact` (which *links* of one entity carry force above `force_threshold` — the knee-and-torso fall detector) |
| Pose and limits | `fell_over`, `base_orientation_roll_greater_than`, `base_orientation_pitch_greater_than`, `root_height_below_minimum`, `dofs_pos_limit_exceeded`, `dofs_vel_limit_exceeded`, `dofs_torque_limit_exceeded` |
| Manipulation | `object_out_of_reach`, and the success conditions also registered as metrics: `object_lifted`, `object_in_region`, `object_above_height`, `object_a_is_on_b`, `entity_near_target`, `entity_near_entity_xy`, `joint_angle_at_target` |
| Motion tracking | `bad_anchor_ori`, `bad_anchor_pos_z_only`, `bad_base_ori`, `BadMotionLinkPosZOnly`, `motion_ends`, `MotionBasePoseTrackingBreak`, and the `motion_*_tracking_error_above` family |

The success conditions in the manipulation row are registered both as metric terms and as termination terms, so
one function can count a success and end the episode on it. To end an episode on a metric you already compute,
use `OnMetricSuccess` (the metric's own success check) or `MetricAbove` (its value against a threshold); see
[Share a computation across terms](terms.md#share-a-computation-across-terms-with-a-metricterm).

## Built-in reward terms

| Family | Terms |
|---|---|
| Action smoothness and effort | `action_l2`, `action_rate_l2`, `ActionRateL2Smooth`, `DofsAccL2`, `dofs_torques` |
| Joint limits and posture | `dofs_pos_limits`, `dofs_vel_limits`, `similar_to_default`, `VariablePosture`, `WeightedPosePenalty` |
| Contact | `UndesiredContacts`, `self_collision_cost` |
| Velocity tracking and base | `track_lin_vel_with_z_penalty`, `track_ang_vel_with_xy_penalty`, `TrackingLinVel`, `TrackingAngVel`, `lin_vel_z`, `FlatOrientationL2`, `Upright`, `BodyAngVelXYPenalty`, `base_height_l2`, `constant_base_height`, `is_terminated_penalty` |
| Feet | `FeetAirTime`, `FeetClearance`, `FeetSlip`, `FeetPhase`, `FeetContactBalancePenalty`, `FeetOrientationPenalty`, `CloseFeetPenalty`, `CloseFeetYawProjectedPenalty`, `StumblePenalty` |
| Motion tracking | the `motion_*_tracking_exp` family, `MotionRelativeBodyPositionErrorExp`, `MotionGlobalBodyLinearVelocityErrorExp` and siblings, `MotionContactMatchReward`, `MotionContactDistanceReward`, `MotionContactChamferReward` |
| Metrics and success | `MetricReward`, `TaskSuccess` |

`action_rate_l2` and `action_l2` penalize the whole action vector by default; pass `action_term_name` to restrict
them to one action term's slice. `is_terminated_penalty` returns 1 for envs the termination manager marked
`terminated` this step — time-outs excluded.

## Writing a reward or termination term

Both kinds take the function form for a pure function of state and the class form (subclass
`eden.managers.reward_manager.RewardTerm` or `eden.managers.termination_manager.TerminationTerm`) for anything
resolved once or carried between steps. [Writing terms](terms.md) has the rules; the ones that matter most here:

- Return `(num_envs,)` — float for a reward, bool for a termination.
- `RewardManager.compute` runs under `torch.inference_mode()`, so a class term's persistent buffers must be
  allocated in `build()` and written in place
  ([Term state lifecycle](terms.md#term-state-lifecycle-buffers-reset-and-torch-modes)).
- Grade the step that just happened. A command term's measurement of this step (a hold counter, a contact latch)
  must be computed in its `post_physics_refresh()`, or the reward reads it a step late
  ([Which hook](terms.md#which-hook-grade-the-step-or-prepare-the-next-one)).
- When a reward, a termination and an observation need one quantity, compute it once as a metric and pull it
  with `MetricReward`, `MetricAbove` / `OnMetricSuccess` and `MetricObservation`.

## See also

- [Writing terms](terms.md) — the term contract
- [Curricula](curricula.md) — moving reward weights during training
- [Metrics and recorders](metrics-and-recorders.md) — [how success is decided](metrics-and-recorders.md#how-success-is-decided) and pulling a metric into other terms
- [Training runs](training-runs.md) — what the logged keys look like in a run
- [Term Catalog](../api/terms/catalog.md), the [reward term API](../api/terms/rewards.md) and the
  [termination term API](../api/terms/terminations.md)
