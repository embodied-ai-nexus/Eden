# Commands

A *command* is the goal a policy is asked to reach: a base velocity to track, a target position for an
end-effector, a gait phase, a frame of a motion clip. The command manager owns these goals, resamples them on
a timer and on every reset, and publishes them for observation and reward terms to read. This page covers the
built-in command terms, the per-step timing a command term has to respect, and how to write your own.

Commands exist only on RL environments ([`RLEnvBase`](../api/envs.md) and `MimicRLEnvBase`), configured through
`EdenRLConfig.command_options`.

## A first example

The velocity-tracking tasks sample a planar base velocity every 3 to 8 seconds
(`eden/tasks/velocity_tracking/base_config.py`):

```python
import math

from eden.managers.terms import UniformVelocityCommand
from eden.options.managers.commands import CommandManagerOptions

command_options = CommandManagerOptions(
    velocity_command=UniformVelocityCommand.configure(
        entity_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.3,
        heading_command=True,
        heading_control_stiffness=0.5,
        lin_vel_x_range=(-1.0, 1.0),
        lin_vel_y_range=(-1.0, 1.0),
        ang_vel_z_range=(-0.5, 0.5),
        heading_range=(-math.pi, math.pi),
    )
)
```

The key (`velocity_command`) is the name every other term uses to find the command. The policy sees it
through the `generated_commands` observation, and the tracking rewards read it by the same name:

```python
from eden import observations
from eden.terms import ObsTerm

velocity_command = ObsTerm.configure(
    func=observations.generated_commands,
    params={"command_name": "velocity_command"},
)
```

Inside a term, read it with `env.command_manager.get_command(name)` (the command tensor) or
`env.command_manager.get_term(name)` (the term object, for its extra attributes). Both raise a `KeyError`
listing the active command terms when the name is wrong. A task that declares no command terms has
`env.command_manager is None`.

## Fields every command term has

| Field | Type | Default | Meaning |
|---|---|---|---|
| `resampling_time_range` | `tuple[float, float] \| None` | `None` | Seconds until the next resample, drawn uniformly per env after every resample. |
| `debug_vis` | `bool` | `False` | Call the term's `draw_vis()` every step, outside training only. |

!!! warning "A resampled term needs `resampling_time_range`"
    The base default is `None`, and there is no check for it: a term that resamples (such as
    `UniformVelocityCommand` or `UniformT3Command`) and is configured without a range fails with a `TypeError`
    from `uniform_()` on the first reset. Terms that never resample set a very long range themselves:
    `ConstantVelocityCommand`, `OperatorVelocityCommand` and `GaitPhaseCommand` default to `(1.0e9, 1.0e9)`.

## Built-in command terms

All of them are in `eden.managers.terms.commands` (also importable from `eden.managers.terms` or as
`eden.commands`). `eden terms --kind command` prints each one with every parameter and default, and the full
list is in the generated [term catalog](../api/terms/catalog.md).

| Term | Command shape | What it does |
|---|---|---|
| `UniformVelocityCommand` | `(num_envs, 3)` | `(vx, vy, wz)` in the robot's base frame, sampled uniformly from `lin_vel_x_range`, `lin_vel_y_range`, `ang_vel_z_range`. |
| `ConstantVelocityCommand` | `(num_envs, 3)` | The same command held at `command_value` (or whatever `set_command()` was last given), clamped into the configured ranges. |
| `OperatorVelocityCommand` | `(num_envs, 3)` | A `ConstantVelocityCommand` re-read from a keyboard or gamepad every control step (`device_name`, `device_options`). |
| `UniformT3Command` | `(num_envs, 3)` | A target position sampled from `x_range`, `y_range`, `z_range`. |
| `UniformSE3Command` | `(num_envs, 7)` | A target pose: position from `x_range`/`y_range`/`z_range`, then a wxyz quaternion from `roll_range`/`pitch_range`/`yaw_range`. |
| `GaitPhaseCommand` | `(num_envs, num_feet)` | A per-foot phase in `[-pi, pi]` advancing at `gait_frequency` Hz, offset by `phase_offset` between feet. |
| `MotionCommand` | `(num_envs, 2 × clip joints)` | The current frame of one or more motion clips (`.npz`): the clip's joint positions, then its joint velocities. Link and base poses are separate accessors. |

### Velocity commands

`UniformVelocityCommand` has two optional behaviours on top of the uniform draw:

- **Heading control.** With `heading_command=True`, a fraction `rel_heading_envs` of the resampled envs gets a
  target heading drawn from `heading_range` instead of a yaw rate; each step their `wz` is set to
  `heading_control_stiffness * heading_error`, clipped to `ang_vel_z_range`. `heading_command=True` with
  `heading_range=None` raises at construction.
- **Standing envs.** A fraction `rel_standing_envs` of the resampled envs is commanded all zeros.

`ConstantVelocityCommand` and `OperatorVelocityCommand` are what a deployment substitutes for a trained
`UniformVelocityCommand`, so that the robot goes where the operator says instead of where the training
distribution says. Both clamp into `lin_vel_x_range`, `lin_vel_y_range` and `ang_vel_z_range`, which a deploy
fills from the term it replaces. A degenerate range such as `(0.5, 0.5)` pins that axis: the policy only ever
saw one value there. `OperatorVelocityCommand` opens its input device in `build()` and is not meant for
training, since a training run would sit at whatever the sticks read. See [Deployment](deployment.md).

### Pose and position targets

`UniformT3Command` can show its goal: set `debug_vis=True` and either `marker_entity_name` (a scene entity,
for example a non-colliding sphere, that is moved to the goal each step) or nothing (a debug sphere of
`marker_radius` and `marker_color` is drawn). Prefer the marker entity when you record video: debug draws
appear only in the interactive viewer, not in camera recordings. `marker_frame_entity` names the entity whose
frame the command is expressed in (for example the robot base); left empty, the command is world-frame.

The reacher benchmark (`eden/tasks/benchmark/reacher/config.py`) is a complete task built on
`UniformT3Command`.

### Gait phase

`GaitPhaseCommand.command` holds raw phase angles. Feed them to the policy with the `sin_phase` / `cos_phase`
observation terms (`eden/managers/terms/observations/locomotion.py`). With `velocity_command_name` set, the
phase of every foot is pinned to `stand_phase_value` in the envs whose velocity command is below
`stand_velocity_threshold`, so a gait reward stops asking for steps while the robot is told to stand.

### Motion clips

`MotionCommand` plays BeyondMimic-format `.npz` clips (`motion_file`, or `motion_files` for one clip per env)
and exposes the current reference frame through `links_*_w`, `base_*_w` and `anchor_*_w` accessors. It is the
lightweight motion path, used by the frozen-policy ports and single-clip tracking on a plain `RLEnvBase`; the
training-grade path is the motion manager of `MimicRLEnvBase`. `motion_command_options(...)` builds the
`{"motion": MotionCommand}` entry for a policy. Both paths, their sampling modes and the clip format are
described in [Motion tracking](motion-tracking.md).

## When a command term runs

A command term has three per-step hooks, and which one you put state in decides whether the reward sees it in
the same step or a step late. The full step order is the table in [Environments](environments.md); the rows
that concern commands are:

| Phase | Call | What belongs there |
|---|---|---|
| 3 | `CommandTerm.pre_physics_advance()` | Advance a time-based reference *before* physics, so the post-physics robot is graded against the matching frame (`MotionCommand` advances its clip clock here). |
| 5 | `CommandTerm.post_physics_refresh()` | Everything that **grades this step**: caches of the post-physics robot state, hold counters, contact latches, tracking errors. The metric, termination and reward managers (phases 6–9) read it in the same step. |
| 13 | `CommandTerm.reset(envs_idx)` | Runs for the envs being reset: logs `stats`, then resamples them. |
| 17 | `CommandTerm.compute(dt)` | Ticks the resampling clock, calls `_resample_command()` for expired envs, then `_update_command()`. This **publishes** the command the next action is chosen from. |
| 20 | `CommandTerm.draw_vis()` | Only for terms with `debug_vis=True`, and only when the env is not training. |

Commands are published at phase 17, after the reward at phase 9, on purpose: the policy chose this step's
action from the command it observed at the end of the previous step, so the reward must grade against that
command, not a freshly resampled one.

!!! warning "Grading state goes in `post_physics_refresh`, not `_update_command`"
    `_update_command()` runs at phase 17, below the reward. A value derived from the post-physics robot and
    written there is read by the reward one control step late, and a value *cleared* there is never read at
    all: the step that earned it is graded through the cleared buffer and pays zero. Compute such state in
    `post_physics_refresh()`. The test is what the state is, not who reads it — the command itself still
    belongs in `_update_command()`. [Terms](terms.md) explains the rule in full.

On the deployment path (`eden deploy`) only some hooks run: `advance_reference()` (which calls
`pre_physics_advance()` by default), `compute()` and `draw_vis()`. `post_physics_refresh()` never runs there,
because there is no physics to bracket and no reward to grade. State the policy's observations read must
therefore also be produced from `compute()`; an idempotent cache refresh can live in both hooks, as
`MotionCommand`'s robot-state cache does. A term whose `pre_physics_advance()` resets or teleports the robot
(a clip wrap) must override `advance_reference()` with a clock-only advance, and `rewind_reference()` with a
clock-only rewind.

### Resampling

A command is resampled in two situations:

- **On reset.** `CommandManager.reset()` resamples the reset envs (phase 13, after the scene reset and the
  RESET events). The command an env starts its episode with is therefore always fresh.
- **On its timer.** Every step, `compute()` subtracts `dt` from each env's `time_left`; the envs at or below
  zero are resampled and get a new `time_left` drawn from `resampling_time_range`.

`command_counter` counts each env's resamples since its last reset.

Per-env statistics a term puts in its `stats` dict (each a `(num_envs,)` tensor) are averaged over the envs
being reset, logged as `Commands/<term_name>/<stat>`, and zeroed for those envs.

## Writing a command term

Subclass `CommandTerm`, register it, and implement three members:

```python
import torch

from eden.managers import COMMAND_TERM_REGISTRY, CommandTerm
from eden.utils.sample import sample_uniform_ranges


@COMMAND_TERM_REGISTRY.register()
class UniformHeightCommand(CommandTerm):
    """A target base height, resampled on a timer."""

    height_range: tuple[float, float] = (0.6, 0.8)

    def __init__(self, env, options):
        super().__init__(env, options)
        self._target = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._target

    def _resample_command(self, envs_idx: torch.Tensor) -> None:
        sampled = sample_uniform_ranges((self.height_range,), self.num_envs, self.device)
        self._target.copy_(torch.where(envs_idx.unsqueeze(-1), sampled, self._target))

    def _update_command(self) -> None:
        pass  # the target is constant between resamples
```

```python
command_options = CommandManagerOptions(
    base_height=UniformHeightCommand.configure(height_range=(0.5, 0.9), resampling_time_range=(4.0, 6.0)),
)
```

The contract for each member:

- **`command`** — a property returning the `(num_envs, ...)` tensor other terms read. Return the same buffer
  every time and update it in place.
- **`_resample_command(envs_idx)`** — always receives a `(num_envs,)` bool mask with at least one `True`,
  whatever selector reached the manager (the timer mask, the auto-reset's `reset_buf`, a slice, an index
  tensor). An empty selection never reaches it, so do not guard for one. Write through `torch.where` over a
  full-batch draw, as above, rather than indexing with the mask: `self.cmd[envs_idx] = ...` is correct but
  costs a GPU→CPU sync for the gather and another for the write, and this method runs every control step. A
  scalar fill such as `self.flag[envs_idx] = False` is fine. `len(envs_idx)` is `num_envs`, not the number of
  selected envs.
- **`_update_command()`** — publishes the command for the next step (phase 17). Closing the loop of a goal
  state machine (deciding a goal was reached and drawing the next one) belongs here too.

Resolve entities and other terms in `build()`, so a bad name raises at env construction; see
`UniformVelocityCommand.build()` for the pattern, and [Terms](terms.md) for the general rules (state in
`build()`, entity handles re-resolved through `resolve_entity`, no GPU syncs on the hot path).

## Replacing a command term after build

`CommandManager.swap_term(name, options)` builds a new term from `options`, installs it under `name`, and returns
the old term. It is how a deployment puts an operator-driven or held command in place of the trained generator.
The replacement's command must have the same shape as the one it replaces, or `swap_term` raises a
`ValueError`: the observation layout and the policy's input width were fixed by the old term.
`restore_term(name, term)` puts back the exact object `swap_term` returned.

`swap_term` does not repoint terms that cached the replaced term in their own `build()`
(`self._command = env.command_manager.get_term(...)`). Code that needs those to follow the swap must repoint
them itself, as `DeploymentBase` does.

## See also

- [Environments](environments.md) — the full step-order table.
- [Terms](terms.md) — the general term lifecycle and the grade-versus-prepare hook rule.
- [Curricula](curricula.md) — `VelocityCommandCurriculum` widens a velocity command's ranges over training.
- [Motion tracking](motion-tracking.md) — `MotionCommand` and the motion manager.
- [Deployment](deployment.md) — operator velocity commands on hardware.
- [Term catalog](../api/terms/catalog.md) — every registered command term with its parameters.
- API reference: [`CommandManager`](../api/managers.md), [command terms](../api/terms/commands.md), [options](../api/options/managers.md).
