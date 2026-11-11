# Events and domain randomization

Event terms change the simulation at defined moments: once at startup, whenever an env resets, or on a timer
while it runs. Resetting a robot to a random pose, pushing it mid-episode and randomizing its friction, mass
or PD gains are all event terms. This page covers when events fire and in what order, the built-in reset,
placement and domain-randomization terms with the scene settings some of them need, and how to write your own.

Events are configured through `event_options` (an [`EventManagerOptions`](../api/options/managers.md)) on any
config, RL or not.

## A first example

The velocity-tracking tasks reset the robot's base and joints every episode, push it every 1 to 3 seconds, and
randomize its foot friction, centre of mass and encoder offsets once per env
(`eden/tasks/velocity_tracking/base_config.py`, trimmed):

```python
import eden as en
from eden import events
from eden.options.envs import EnvOptions
from eden.options.managers.events import EventManagerOptions
from eden.terms import EventTerm

env_options = EnvOptions(
    batch_links_info=True,  # RandomizeComShift writes per-env link COMs
)

event_options = EventManagerOptions(
    reset_base=EventTerm.configure(
        func=en.events.reset_base_state_uniform,
        mode="reset",
        params={
            "entity_name": "robot",
            "pos_x_range": (-0.5, 0.5),
            "pos_y_range": (-0.5, 0.5),
            "pos_z_range": (0.01, 0.05),
            "yaw_range": (-3.14, 3.14),
        },
    ),
    push_robot=events.PushByVelocity.configure(
        mode="interval",
        interval_range_s=(1.0, 3.0),
        entity_name="robot",
        velocity_range={"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.78, 0.78)},
    ),
    randomize_friction=events.RandomizeFrictionRatio.configure(
        mode="startup",
        entity_name="robot",
        links_name=[".*_foot"],
        friction_range=(0.3, 1.2),
    ),
    randomize_com=events.RandomizeComShift.configure(
        mode="startup",
        entity_name="robot",
        links_name=["base"],
        com_shift_x_range=(-0.025, 0.025),
    ),
    encoder_bias=events.RandomizeDofsPosOffset.configure(
        mode="startup",
        entity_name="robot",
        dofs_pos_range=(-0.015, 0.015),
    ),
)
```

(The task fills `links_name` per robot; the patterns above are placeholders.) Two spellings appear: function
terms go through `EventTerm.configure(func=..., params={...})`, class terms through their own `.configure(...)`
with the parameters as keyword arguments.

!!! note "Two classes called `EventTerm`"
    `eden.terms.EventTerm` is the wrapper for *function* event terms (`EventTerm.configure(func=...)`).
    The base class you subclass for a *class* event term is `eden.managers.EventTerm`
    (`eden.managers.event_manager.EventTerm`). Import the one you mean.

## When events fire

Every event term carries five scheduling fields:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `mode` | `EventMode` or its string | `"reset"` | `"startup"`, `"reset"` or `"interval"`. Several class terms change the default (pushes and wrenches default to `"interval"`, `RandomizeStartupDofsPosBias` to `"startup"`). |
| `priority` | `int` | `0` | Order among the terms of the same mode: lower runs first. |
| `interval_range_s` | `tuple[float, float] \| None` | `None` | Interval mode only, and required there: seconds between fires, redrawn after each fire. |
| `is_global_time` | `bool` | `False` | Interval mode only: one clock shared by all envs instead of one per env. |
| `min_step_count_between_reset` | `int` | `0` | Reset mode only: fire for an env at most once every N control steps. |

| Mode | Fires | Where in the step |
|---|---|---|
| `"startup"` | Once, for all envs, at the end of `env.build()` after every manager is constructed | — |
| `"reset"` | For the envs being reset, right after the scene reset | phase 13 (and every `env.reset()`) |
| `"interval"` | For the envs whose countdown expired | phase 18, after the auto-reset, before observations |

The phase numbers refer to the step-order table in [Environments](environments.md).

### Startup

STARTUP events run once, with `envs_idx=None`, before Eden snapshots the scene state that every later reset
restores. Two consequences:

- A property a STARTUP event writes (friction, mass, COM, gains, a joint-zero bias) holds for the env's
  lifetime. This is the usual choice for physical domain randomization: one draw per env per run.
- A *state* a STARTUP event writes (a joint position, a base pose) becomes part of the reset baseline: every
  scene reset returns to it.

### Reset

RESET events run after `scene.reset()` has put the reset envs back to the post-build state, for exactly those
envs, before any manager's `reset()`. A task with no reset events therefore starts every episode from the same
pose. The events receive `envs_idx` in whatever form the caller used: the auto-reset in `RLEnvBase.step` passes
the full-length bool `reset_buf`, `env.reset()` passes a slice or an index tensor. Index with it; never take its
length (see [Writing event terms](#writing-event-terms)).

`min_step_count_between_reset` throttles an event for an env that keeps resetting: the event fires for an env
only if at least that many control steps passed since it last fired there (or it has never fired there).

### Interval

An interval term keeps a countdown per env (one CPU-side countdown with `is_global_time=True`). Each control
step subtracts `env.dt`; when it reaches zero the term fires for those envs and draws a new interval from
`interval_range_s`. The countdown is in control-step time, so an interval equal to `env.dt` fires every step.
A mode-`"interval"` term without `interval_range_s` raises a `ValueError` when the event manager is built.

- A class term's per-env countdown restarts with a fresh draw when its env resets. A *function* term's
  countdown does not restart on reset.
- A global-time term is called with no `envs_idx` at all, so a function used that way must give `envs_idx` a
  default.

### Ordering events with priority

Within one mode, terms run in ascending `priority`; equal priorities keep their declaration order (the sort is
stable). Ordering is computed separately for each mode. Set `priority` whenever one event has to see another's
result, and leave gaps (0, 10, 20) so a term can be slotted in later. Examples:

- An event that converts a world-frame angular velocity through the base orientation must run after the event
  that randomizes that orientation (`reset_base_state_uniform`), or it converts against the previous episode's
  heading. A free base stores its angular DOF velocity in the base body frame; see [Entities](entities.md).
- Two events that write the same quantity, such as a whole-robot friction pass at `priority=0` and a tighter
  per-foot pass at `priority=10`.
- `ResetFromStateBuffer` must run after any RESET event that writes the same entities, or they overwrite the
  restored state.

`env.event_manager.summary()` prints the resolved order, one table per mode with a `Priority` column. It is the
quickest way to confirm what runs.

### A class term's `reset()` always runs

Separately from firing, the event manager calls `reset(envs_idx)` on every *class* event term whenever an env
resets, whatever the term's mode. This happens after `action_manager.reset()`, which the env enforces at build.
Several domain-randomization terms rely on it: an action modifier's per-env buffer is restored to its baseline
by the action manager's reset, so a randomization written earlier in the same reset would be overwritten. The
actuator randomizers write in `reset()` instead (see [below](#actuator-and-controller-randomization)).

### Events do not stop at eval

The event manager has no eval switch: every event fires in `eval_mode=True` exactly as in training. (Eval mode
turns off observation corruption and freezes curricula; see [Observations](observations.md) and
[Curricula](curricula.md).) To evaluate without a randomization, remove it from the config before building the
env:

```python
config.event_options.disable_term("push_robot")
```

`disable_term` raises a `KeyError` listing the declared terms if the name is wrong. The keyboard-driven
velocity-tracking player disables every reset and randomization event this way before it runs a trained policy
(`examples/projects/locomotion/velocity_tracking/main.py`).

## Built-in event terms

All built-ins live in `eden.managers.terms.events` (also `eden.events`). `eden terms --kind event` prints every
one with its parameters, defaults and `requires:` line, and the generated [term catalog](../api/terms/catalog.md)
has the same list. The groups below say what each is for and what it needs.

### Resetting state

Function terms; use them with `EventTerm.configure(func=..., mode="reset", params={...})`.

| Term | What it does |
|---|---|
| `reset_base_state_uniform` | The entity's default root pose plus a uniform offset (`pos_*_range`) and rotation (`roll_range`, `pitch_range`, `yaw_range`, composed as `R_rpy @ R_default`). Writes the authored root frame (`relative=True`), so the asset's declared origin lands on the target. Rigid entities only. |
| `reset_dofs_by_offset` | Joint positions `default + U(dofs_pos_range)`, clamped to the soft limits; velocities zero, or `U(dofs_vel_range)` when given. |
| `reset_dofs_by_scale` | Joint positions `default * U(dofs_pos_scale_range)`, clamped; velocities zero. |
| `reset_dofs_by_scale_and_offset` | `default * U(dofs_scale_range) + U(dofs_offset_range)`, clamped; velocities zero. |

Class terms for joint positions: `SetRandomDofsPos` (default: a fraction of each joint's limit range,
`apply_as_ratio=True`, `(0.0, 0.2)`) and `RandomizeStartupDofsPos` (default: absolute radians, ±1°; despite the
name its default mode is `"reset"`).

For motion-tracking tasks on `MimicRLEnvBase`, `ReferenceStateInitialization` (robot) and
`ObjectReferenceStateInitialization` (a free object) start each episode from the motion reference, with optional
pose, velocity and joint noise; see [Motion tracking](motion-tracking.md). `ResetFromStateBuffer(path=...)`
restores each resetting env to a stored row of a reset-state buffer file, drawn uniformly, by `category_weights`,
or by a curriculum sampler named in `sampler` (see [Curricula](curricula.md#reset-configuration-samplers)).

### Placing objects on a surface

These sample a position on another entity's support surface and drop the entity onto it, after resetting its
orientation to its default (plus the sampled rotation, where there is one). The support entity must be built with
support sampling enabled: `is_support_enabled=True` on its options, for example
`en.scenes.Empty(is_support_enabled=True)` for the ground plane. See [Scenes and terrains](scenes-and-terrains.md).

| Term | Kind | Samples |
|---|---|---|
| `place_constant_xy` | function | A fixed `pos_xy` (`frame="world"` or `"local"` to the support). |
| `place_range_xy` | function | Uniform `range_x` × `range_y`, plus optional `range_roll` / `range_pitch` / `range_yaw`. |
| `place_uniform` | function | Uniformly over the support surface (`samples_per_area`). |
| `PlaceGaussian` | class | Gaussian-weighted around `(mu_x, mu_y)` with variances `var_x`, `var_y`. |
| `PlaceFromLayouts` | class | One of a fixed list of `layouts`, each mapping entity names to `(x, y, yaw_deg)`. Drawing every entity from one layout keeps them from overlapping, which independent ranges over the same area cannot guarantee. |

### Pushes and external wrenches

| Term | Default mode | What it does |
|---|---|---|
| `PushByVelocity` | interval | **Adds** a random velocity to the free base. `velocity_range` keys: `x`, `y`, `z` (world frame) and `roll`, `pitch`, `yaw` (added in the base body frame). |
| `PushSetPlanarVelocity` | interval | **Sets** the base's `vx`, `vy` to `U(velocity_xy_range)`, keeping `vz` and the angular velocity. |
| `ApplyExternalForce` | interval | A random force (`force_*_range`, N) on `links_name`, at `ref="link_origin"` or `"link_com"`, world or `local` frame. |
| `ApplyExternalTorque` | interval | The same for a torque (`torque_*_range`, N·m). |

The two wrench terms take `prob`, a per-env probability of applying the wrench at each fire. Each term draws its
own mask, so a force term and a torque term configured together gate independently. Genesis applies the wrench
over one simulation step (`RigidSolver.apply_links_external_wrench`).

### Physical properties

| Term | Requires | What it does |
|---|---|---|
| `RandomizeFrictionRatio` | — | Per-link sliding friction from `friction_range`, as an **absolute** coefficient (see the warning below). |
| `RandomizeMassShift` | `EnvOptions(batch_links_info=True)` | Adds `U(mass_shift_range)` kg to each named link's mass. |
| `RandomizeLinkMassScale` | `EnvOptions(batch_links_info=True)` | Multiplies each named link's mass by `U(mass_scale_range)`. |
| `RandomizeComShift` | `EnvOptions(batch_links_info=True)` | Shifts each named link's centre of mass by an xyz offset (m, link frame). |
| `RandomizeKpKdGains` | `EnvOptions(batch_dofs_info=True)` | Scales the joint PD gains by `U(kp_range)` / `U(kd_range)` per env and DOF. |
| `set_contact_sol_params` | STARTUP only | Sets the constraint-solver parameters (7 values: timeconst, dampratio, dmin, dmax, width, mid, power) on the collision geoms of `entity_links`. |

`batch_links_info` defaults to `False` and `batch_dofs_info` to `True` (`EnvOptions`). A term whose flag is off
raises a `ValueError` naming the term and the flag when the env is built, rather than failing on its first write.

The mass and COM terms also refuse, at build:

- **A link fixed to the world** (a fixed-base arm's base, a mounted hand's palm, a `fixed=True` prop), whose mass
  and COM the solver never reads. The error suggests a `links_name` that selects exactly the moving links, so
  `links_name=["*"]` suits only an entity with no world-fixed link.
- **A body Genesis re-anchored at build.** By default a single free rigid body is *aligned*: its frame is put on
  its centre of mass and principal axes, and Genesis then does not support writing its COM (or, for a body of
  several links, its mass). The fix is to load the asset with `align=False`; primitives always align, so load
  the shape from a file instead. `batch_links_info=True` does not help here.
- **A range that can make a mass non-positive** (`RandomizeMassShift` checks the lightest selected link,
  `RandomizeLinkMassScale` requires a positive lower bound).

`RandomizeKpKdGains` has two paths. With `action_term_name=None` it writes the entity's gains
(`set_dofs_kp` / `set_dofs_kd`), which is what Genesis' built-in PD (`ImplicitPDController`) reads. With
`action_term_name` set to an `ExplicitPDController` or `VelocityFeedforwardPDController`, it scales that
controller's own gain caches and leaves the entity alone; the factors are re-applied after every action-manager
reset so they survive it.

`set_contact_sol_params` rejects any `envs_idx` other than all envs, because Genesis stores geom solver
parameters unbatched. Genesis mixes a contact pair's parameters as the mean of its two geoms, so stiffening one
object's geoms also stiffens its contacts with everything else (the table, say). To stiffen an object without an
event, set `default_geoms_sol_params` on it; see [Entities](entities.md).

!!! warning "Friction randomization has absolute semantics, and contact friction combines as `max`"
    Every `RandomizeFrictionRatio` fire first sets the **whole entity's** base friction to 1.0 — discarding any
    authored `default_geoms_friction`, including on links outside `links_name` — and then writes per-link ratios
    on `links_name`. `friction_range` is therefore the absolute coefficient range, whatever the asset authored;
    convert a source config written as "scale × authored value" before porting it. Where torsional and rolling
    friction are enabled, the same ratio scales them too.

    Genesis resolves a contact pair's friction as `max(a, b)`, so the randomized value only matters against
    surfaces whose friction is below it. Author the ground low (for example
    `PlaneOptions(default_geoms_friction={"*": 0.01})`) so the robot side decides. The full porting recipe is in
    [Entities](entities.md).

### Actuator and controller randomization

These randomize an action term or one of its action modifiers rather than the simulator. Each names its action
term with `action_term_name` and raises at build if the term, or the modifier it needs, is missing. Modifiers
are configured in the action term's `modifier` chain; see [Actions](actions.md).

| Term | Needs | What it randomizes |
|---|---|---|
| `RandomizeMotorStrength` | `MotorStrength` | The torque scale, one factor per env from `motor_strength_range`. |
| `RandomizeGearBacklash` | `GearBacklash` | The backlash per env and DOF (`backlash_range`, rad, or multipliers of the configured value with `apply_as_ratio=True`). |
| `RandomizeConstantTorqueKick` | `ConstantTorqueKick` | The kick magnitude (`torque_kick_range`, or multipliers). |
| `RandomizeDeadbandEpsilon` | `Deadband` | The position-error deadband (`deadband_epsilon_range`, or multipliers). |
| `RandomizeTorqueNoise` | `TorqueOffset` | Random force injection: per-DOF torque noise in `[-rfi_scale, rfi_scale]` at each fire. Interval mode; set `interval_range_s` to the control period. |
| `ERFI50` | `TorqueOffset` | Per env, either a constant actuation offset for the episode or per-step random torque noise (`erfi_ratio` splits the envs). Interval mode, `interval_range_s` set to the control period. |
| `RandomizeDofsPosOffset` | a joint position controller | A per-env joint-position offset added to the controller's target (encoder bias). `action_term_name=None` selects every joint position controller on `entity_name`. |
| `RandomizeStartupDofsPosBias` | — | Bakes a per-env constant bias into `entity.default_dofs_pos` at startup (calibration drift). |

The first four randomize in `reset()`, not in their event `compute()`, so their draw lands after the action
manager has restored the modifier's baseline. They therefore support only `mode="reset"` with
`min_step_count_between_reset=0`, and reject anything else at construction. Managers that reset before the event
manager (action, command, metric, termination, reward, curriculum) still see the previous episode's draw in the
modifier buffers.

`RandomizeStartupDofsPosBias` and `RandomizeDofsPosOffset` compose: a fixed drift baked in at startup plus
per-episode jitter on top, in either event order. Because the bias moves `default_dofs_pos`, every PD controller
on the entity with `reference_source=DEFAULT` must be listed in `action_terms_name` so its cached offset moves in
lockstep; `build()` raises for a controller left out, for duplicates, and for a `NullJointAction` on the same
entity (which caches the default with no refresh path).

## Writing event terms

### Function terms

An event function takes the env and the env selector, then keyword-only parameters:

```python
import torch

from eden.managers import EVENT_TERM_REGISTRY
from eden.terms import EventTerm
from eden.utils.misc import sanitize_envs_idx


@EVENT_TERM_REGISTRY.register()
def my_reset(env, envs_idx: slice | torch.Tensor | None, *, entity_name: str, scale: float = 1.0) -> None:
    envs_idx = sanitize_envs_idx(envs_idx, env.num_envs)
    entity = env.get_entity(entity_name)
    ...  # write the selected envs' state through the entity's setters, with envs_idx=envs_idx


reset_thing = EventTerm.configure(func=my_reset, mode="reset", params={"entity_name": "robot", "scale": 0.5})
```

The built-ins in `eden/managers/terms/events/resets.py` are complete examples. Points to get right:

- `envs_idx` is `None` at startup (all envs), a full-length bool mask on the auto-reset and on interval fires,
  and a slice or index tensor from `env.reset()`. `sanitize_envs_idx` turns `None` into a slice. Index with it;
  `len(envs_idx)` on the step path is `num_envs`, not the reset count.
- Sample for every env and keep the selected rows (`sample_uniform(..., (env.num_envs, n), ...)[envs_idx]`, or
  pass every row with the mask where the setter accepts it), rather than gathering through the mask, which
  costs GPU→CPU syncs on every step that resets anything. [Terms](terms.md) covers the sync rules.
- A function used with `is_global_time=True` is called without `envs_idx`; give it a default.
- A function that needs an `EnvOptions` flag declares it at registration:
  `@EVENT_TERM_REGISTRY.register(_requires_env_options=("batch_links_info",))`. The env then refuses to build
  without it, and `eden terms` shows it on the term's `requires:` line.

### Class terms

Subclass `eden.managers.EventTerm` when the term needs state resolved once (link indices, a cached controller) or
a `reset()` hook. Declare parameters as annotated class attributes (they become `configure()` arguments), resolve
names in `build()` so a typo raises at env construction, and implement `compute(self, envs_idx=None)`. Set
`_requires_env_options = ("batch_links_info",)` as a class attribute for a scene-wide flag. `RandomizeComShift`
in `eden/managers/terms/events/domain_rand.py` is a short, complete example.

### Firing several reset terms as one write

Each Genesis state setter resets the selected envs' collider state and runs forward kinematics, so N terms that
each seed one entity pay N passes. A class term can override the classmethod
`EventTerm.compute_batch(terms, envs_idx)` to fire several terms of its kind in one write. The event manager hands
a run of RESET terms to one call when they are adjacent in priority order, each term's class takes `compute` and
`compute_batch` from the same class, and none sets `min_step_count_between_reset`. A term of another kind between
them breaks the run, and a subclass that overrides `compute` alone runs alone. A batch must leave exactly what
firing its terms one after another would leave: the same random draws in the same order, last write winning. The
reference-state-initialization terms do this through `StagedStateWrite` (`eden/entities/rigid.py`).
`EventManager.iter_reset_batches()` lists the batches; their terms' own `compute` never runs on a reset.

## See also

- [Environments](environments.md) — the step order the phases above refer to.
- [Terms](terms.md) — the general term lifecycle, validation and performance rules.
- [Entities](entities.md) — pose frames, free-base velocity frames, friction and contact parameters.
- [Physics](physics.md) — `EnvOptions` batching flags and solver settings.
- [Actions](actions.md) — the action modifiers the actuator randomizers write to.
- [Term catalog](../api/terms/catalog.md) — every registered event term with its parameters.
- API reference: [`EventManager`](../api/managers.md), [event terms](../api/terms/events.md), [options](../api/options/managers.md).
