# Writing Terms

Every observation, reward, termination, command, event, curriculum, metric, action and recorder in an Eden task
is a **term**: a small unit that a manager builds from its options, calls on a fixed schedule, and resets with
the episode. This page is the contract for writing one — which form to pick, what each lifecycle hook is for,
where per-step state belongs, what validation raises, and how terms are registered. Read it before you write or
review a term; the per-kind pages ([actions](actions.md), [observations](observations.md),
[rewards and terminations](rewards-and-terminations.md), [commands](commands.md), [events](events.md),
[curricula](curricula.md), [metrics and recorders](metrics-and-recorders.md)) build on it.

!!! tip "Coding agents"
    In an Eden checkout, the `writing-eden-terms` skill (`.claude/skills/writing-eden-terms/`) turns this page
    into a checklist to verify a term against. It is repo-only: `pip install` ships `eden/`, not `.claude/`.

## Terms at a glance

Each manager owns one kind of term and resolves it from one registry. A task config holds one options object
per manager (`reward_options`, `observation_options`, …), and each named entry in it is one term.

| Kind | Config field | Registry (`eden.managers`) | Function terms? | Class base to subclass | Function wrapper (`eden.terms`) |
|---|---|---|---|---|---|
| action | `action_options` | `ACTION_TERM_REGISTRY` | no | `eden.managers.action_manager.ActionTerm` | — |
| observation | `observation_options` | `OBSERVATION_TERM_REGISTRY` | yes | `eden.managers.observation_manager.ObservationTerm` | `ObsTerm` |
| reward | `reward_options` | `REWARD_TERM_REGISTRY` | yes | `eden.managers.reward_manager.RewardTerm` | `RewardTerm` |
| termination | `termination_options` | `TERMINATION_TERM_REGISTRY` | yes | `eden.managers.termination_manager.TerminationTerm` | `DoneTerm` |
| command | `command_options` | `COMMAND_TERM_REGISTRY` | no | `eden.managers.command_manager.CommandTerm` | — |
| event | `event_options` | `EVENT_TERM_REGISTRY` | yes | `eden.managers.event_manager.EventTerm` | `EventTerm` |
| curriculum | `curriculum_options` | `CURRICULUM_TERM_REGISTRY` | yes | `eden.managers.curriculum_manager.CurriculumTerm` | `CurrTerm` |
| metric | `metric_options` | `METRIC_TERM_REGISTRY` | yes | `eden.managers.metric_manager.MetricTerm` | `MetricTerm` |
| recorder | `recorder_options` | `RECORDER_TERM_REGISTRY` | no | `eden.managers.recorder_manager.RecorderTerm` | — |

What `compute()` returns, whichever form you use:

| Kind | Output |
|---|---|
| observation | `(num_envs, ...)` tensor; usually `(num_envs, dim)` |
| reward | `(num_envs,)` float |
| termination | `(num_envs,)` bool |
| metric | `(num_envs,)` float |

!!! warning "Three names mean two different things"
    `eden.terms.RewardTerm`, `eden.terms.EventTerm` and `eden.terms.MetricTerm` are the **function wrappers**
    — you call `.configure(func=...)` on them. The class bases with the same names live in the manager modules
    (`eden.managers.reward_manager.RewardTerm`, `eden.managers.event_manager.EventTerm`,
    `eden.managers.metric_manager.MetricTerm`, also re-exported from `eden.managers`) — you **subclass** those.
    Import from the module that matches the form you are writing. Subclassing `eden.terms.RewardTerm` gives you
    a function wrapper with no function in it.

## Function term or class term?

Pick by **whether the term has anything to keep between calls**:

| | Function term | Class term |
|---|---|---|
| Use when | the term is a cheap, pure function of the current env state, with nothing to cache and nothing to carry | the term resolves something once (link or DOF indices, a constant tensor, another manager's term) and reuses it, **or** it carries per-step or per-episode state (an integral, a counter, a latch) |
| Written as | a plain function `def f(env, *, param: type) -> torch.Tensor` | a subclass of the kind's class base, with config fields as class attributes |
| Wired in as | `RewardTerm.configure(func=f, params={...})` | `MyTerm.configure(field=...)` |
| Hooks | none: `compute()` only | `build()`, `compute()`, `reset()`, `post_reset()`, plus per-kind hooks |

### Function terms

A function term takes the env and keyword-only parameters, and returns a tensor. This is the built-in
`dofs_vel_limits` reward (`eden/managers/terms/rewards/common.py`):

```python
import torch

from eden.managers.reward_manager import REWARD_TERM_REGISTRY


@REWARD_TERM_REGISTRY.register()
def dofs_vel_limits(env, *, entity_name: str, max_vel: float | torch.Tensor) -> torch.Tensor:
    """Quadratic hinge penalty on joint velocities exceeding a symmetric limit."""
    dofs_vel = env.get_entity(entity_name).get_dofs_vel()
    excess = (dofs_vel.abs() - max_vel).clamp_min(0.0)
    return (excess**2).sum(dim=-1)
```

You wire it into a config with the kind's function wrapper. `params` holds the function's keyword arguments;
the term-level fields (`weight` here) sit beside it:

```python
from eden import rewards
from eden.options.managers.rewards import RewardManagerOptions
from eden.terms import RewardTerm

reward_options = RewardManagerOptions(
    joint_vel_hinge=RewardTerm.configure(
        func=rewards.dofs_vel_limits,
        weight=-0.01,
        params={"entity_name": "robot", "max_vel": 0.5},
    ),
)
```

That is an excerpt of `eden/tasks/benchmark/reacher/config.py`.

A function term is **stateless by construction**. The manager never calls `reset()`, `post_reset()` or any
per-step hook on it, so state kept in a closure or a module global is never zeroed at an episode reset, and it
advances on every one of the several `compute()` calls a step can make (see
[`compute()` is a read](#compute-is-a-read-not-a-tick)). Nothing warns you. A term that has to remember
anything between steps is a class term.

### Class terms

A class term declares its configuration as class attributes with defaults, resolves static data once in
`build()`, and reads state in `compute()`. This is the built-in `UndesiredContacts` reward, lightly trimmed:

```python
import torch

from eden.managers.reward_manager import REWARD_TERM_REGISTRY, RewardTerm
from eden.managers.terms.utils import resolve_link_idx


@REWARD_TERM_REGISTRY.register()
class UndesiredContacts(RewardTerm):
    """Penalty count of links whose net contact-force magnitude exceeds ``threshold``."""

    entity_name: str = "robot"
    links_name: list[str] = []
    threshold: float = 1.0

    def build(self) -> None:
        # Touch the entity so a bad name raises at env construction, and keep only derived data.
        entity = self.resolve_entity(self.entity_name)
        self._link_idx = resolve_link_idx(entity, self.links_name, self.device, term=type(self).__name__)

    def compute(self) -> torch.Tensor:
        forces = self.resolve_entity(self.entity_name).get_links_net_contact_force()  # (N, n_links, 3)
        magnitudes = torch.norm(forces[:, self._link_idx], dim=-1)
        return (magnitudes > self.threshold).float().sum(dim=-1)
```

You configure a class term by calling `configure()` on the class itself, with its fields as keyword arguments —
not `func=` and not `params=`:

```python
from eden import rewards

knee_contacts = rewards.UndesiredContacts.configure(
    weight=-0.1,
    entity_name="robot",
    links_name=[".*knee.*"],
)
```

Mutable defaults are deep-copied for each configured instance, so `links_name: list[str] = []` is safe.
Every public class attribute becomes a parameter; a name with a leading underscore does not, which is how class
constants such as `_function_era_params` and `_requires_env_options` stay out of the config.

!!! warning "A class term must be registered"
    Eden resolves every term — function and class — by its registry key, never by the object you passed.
    A class term that was never decorated with `@<KIND>_TERM_REGISTRY.register()` configures without error and
    then fails to build with a `KeyError` from the registry. See [Registries](#registries).

### `.configure()` returns a specification, not a term

`configure()` builds a Pydantic options object (`RewardTermOptions`, `ObservationTermOptions`, …) that names
the term by its registry key and carries its parameters. Nothing is constructed until the env builds its
managers. That is what lets a config be validated when it is written, saved to JSON or YAML next to a
checkpoint, and loaded back.

## The term lifecycle

A class term goes through these hooks, in this order:

1. **`__init__`** — wiring only. The scene is not built yet; do not touch entities here.
2. **`build()`** — called once, right after construction, while the env builds its managers. Validate names
   (entities, links, DOFs, other terms), resolve indices and constant tensors, allocate persistent buffers.
3. **`compute()`** — the hot path, called every step (several times a step for observations). Read state and
   return the output. No name resolution, no constant tensors built here.
4. **`reset(envs_idx)`** — called when envs restart. Zero the per-episode rows of your buffers.
5. **`post_reset(envs_idx)`** — called after *every* manager has finished resetting those envs. Rebuild
   caches that derive from the final post-reset scene.

Some kinds add hooks of their own:

| Kind | Extra hooks | Where they are documented |
|---|---|---|
| observation | `post_physics_refresh()` — the once-per-step tick (phase 7) | [`compute()` is a read](#compute-is-a-read-not-a-tick) |
| command | `pre_physics_advance()` (phase 3), `post_physics_refresh()` (phase 5), `_resample_command(envs_idx)`, `_update_command()` (phase 17), `advance_reference()` and `rewind_reference()` (deployment) | [Which hook](#which-hook-grade-the-step-or-prepare-the-next-one), [When a command term runs](commands.md#when-a-command-term-runs) |
| action | `compute(actions)` once per control step, `apply_actions()` once per physics step | [Actions](actions.md) |
| event | `mode` and `priority` decide when and in what order it fires; `compute_batch(terms, envs_idx)` for batched reset terms | [When events fire](events.md#when-events-fire), [Ordering events](events.md#ordering-events-with-priority), [Batched reset terms](events.md#firing-several-reset-terms-as-one-write) |

Managers are constructed in their reset order, producers before consumers (`_manager_order()`; see
[Manager order](environments.md#manager-order)). For an RL env that is action → command → metric →
termination → reward → curriculum → event → observation → recorder. So in `build()` a term can already resolve
a term of any manager **earlier** in that list: a reward term can look up an action term, an observation term
can look up a command term, a curriculum can look up the reward term it drives. A wrong name then raises at env
construction rather than mid-rollout. Don't defer such a lookup to the first `compute()`.

Two managers may be absent. `env.metric_manager` is `None` when the task declares no metric terms, and on an
RL env `env.command_manager` and `env.curriculum_manager` are `None` when the task declares none. Test with
`if env.metric_manager is not None:` — these attributes exist, set to `None`, even before the env is built.

The step loop's full twenty-phase order, and the test that keeps it honest, are in
[The step loop](environments.md#the-step-loop). The phase numbers on this page refer to that table.

## Writing good terms: state, caching, and sharing

A few rules keep terms fast and correct. Each of them has been broken in-tree at least once, and none of the
breakages raised an exception.

### Cache derived state in `build()` — never on `env`

Resolve link and DOF indices and constant tensors **once**, in `build()`, and store them on the term instance.
Never stash them on the env object:

```python
# WRONG: leaks across terms, never reset, and re-checked on every step
cache_key = f"_upright_{entity_name}_{link_name}"
if not hasattr(env, cache_key):
    setattr(env, cache_key, {"ls_idx": ..., "grav": ...})

# RIGHT: resolve once in build(), store on the term
def build(self) -> None:
    robot = self.resolve_entity(self.entity_name)
    self._ls_idx = resolve_link_idx(robot, [self.link_name], self.device)
    self._grav = gravity_vec(self.num_envs, self.device)
```

A value written to `env` is visible to every other term, has no reset, and forces a `hasattr` branch onto the
hot path. If you find a function term doing this, convert it to a class term
([Converting a function term to a class term](#converting-a-function-term-to-a-class-term)).

Use the shared helpers in `eden/managers/terms/utils.py` rather than open-coding the resolution; they are what
the built-in terms call:

| Helper | What it does |
|---|---|
| `resolve_link_idx(entity, names, device, *, term=None)` | link-name patterns → a `long` index tensor; raises naming the entity's links |
| `resolve_per_dof_std(std_dict, robot, device, *, default=None)` | a `regex -> value` dict → a per-DOF tensor. A DOF no pattern matches raises; add a `".*"` entry (or pass `default=`) for "everything else" |
| `gravity_vec(num_envs, device)` | the world-frame `[0, 0, -1]`, expanded to `(num_envs, 3)` |
| `projected_gravity(quat, grav)` | `grav` expressed in the frame of a wxyz `quat` |
| `soft_dof_pos_violation(dofs_pos, soft_limits)` | per-DOF soft-limit violation |
| `parallel_jaw_is_grasping(...)` | whether both pads of a parallel-jaw gripper press the same link |
| `resolve_metric_manager(env, *, term, metric_name)` | the metric manager, after checking it exists and has `metric_name` |

A hand-rolled `find_named_links_idx_local(...)` plus `torch.tensor(...)` at the call site, every step, is what
to avoid.

### Resolve entities in `compute()`, cache only what they *derive*

The one thing `build()` must **not** cache is the entity handle itself. Deployment swaps `env.entities[name]`
for a `RobotStateEntity` so observation terms read the real robot; a handle captured in `build()` keeps pointing
at the simulated entity, and the policy is silently fed simulated numbers on hardware. Reach the entity through
`ManagerTermBase.resolve_entity()` each time instead — it is a dictionary lookup on `env.entities`, cheap
enough for the hot path, and an unknown name raises listing the names the scene has:

```python
# WRONG: the handle goes stale when deployment swaps the entity
def build(self) -> None:
    self._robot = self._env.entities[self.entity_name]

def compute(self) -> torch.Tensor:
    return projected_gravity(self._robot.get_quat(), self._grav)

# RIGHT: touch it in build() to fail fast, re-resolve it in compute()
def build(self) -> None:
    self.resolve_entity(self.entity_name)  # a bad name raises at env construction
    self._grav = gravity_vec(self.num_envs, self.device)

def compute(self) -> torch.Tensor:
    return projected_gravity(self.resolve_entity(self.entity_name).get_quat(), self._grav)
```

Lazy resolution does not mean lazy validation: still touch the entity, and any link or DOF names you index, in
`build()`, so a typo fails at env construction rather than on the first step. Derived data — resolved link and
DOF indices, constant tensors — stays valid across the swap and belongs in `build()`. `IllegalLinkContact`
(`eden/managers/terms/terminations/common.py`) and `UndesiredContacts` above are the pattern to copy.

### Term state lifecycle: buffers, reset, and torch modes

Terms that carry per-step state (latched contacts, air-time accumulators, action histories, pose caches) live
inside two framework contracts that are easy to violate silently.

**Allocate persistent buffers in `build()`, mutate them in place, never rebind.** Some manager entry points
run under `@torch.inference_mode()` — `RewardManager.compute`, `MetricManager.compute`, `MetricManager.reset`
and the `post_reset()` fan-out — and others do not (`reset()` on most managers, a plain step loop such as
`eden smoke`). A tensor *created* inside an inference-mode region is an inference tensor for good, and the
first in-place write to it from outside raises
`RuntimeError: Inplace update to inference tensor outside InferenceMode`. Training usually hides this, because
rsl_rl wraps the whole rollout in inference mode, so the crash surfaces later, in `eden smoke` or an eval
script. The discipline is the latch idiom of `FeetAirTime` (`eden/managers/terms/rewards/locomotion.py`):

```python
def build(self) -> None:
    # Allocated outside any inference-mode region: writable from every path.
    self._last_contacts = torch.zeros(self.num_envs, n_feet, dtype=torch.bool, device=self.device)

def compute(self) -> torch.Tensor:
    contact = forces.norm(dim=-1) > self.threshold
    contact_filt = contact | self._last_contacts  # a fresh temporary is fine
    self._last_contacts.copy_(contact)  # copy_, not `self._last_contacts = contact`
    ...
```

`copy_`, `mul_`, `add_` into a `build()`-time buffer are legal under inference mode. Rebinding the attribute to
a tensor produced there is the trap.

`build()` runs outside the inference-mode entry points because no caller in the tree builds an env inside
`torch.inference_mode()` — that is the caller's doing, not the term's. A term that would rather own the
guarantee wraps its allocation in `with torch.inference_mode(False):`, as `ActionRateL2Smooth`
(`eden/managers/terms/rewards/common.py`) does for the buffer it sizes off `action_manager.prev_action`.

**When `build()` genuinely cannot size a buffer, allocate it lazily under `torch.inference_mode(False)`.** This
is the sanctioned exception, for a shape no manager knows yet — not for one that is merely inconvenient to
reach (that is an ordering problem; fix the order instead):

```python
def compute(self) -> torch.Tensor:
    if self._buf is None:
        # A normal tensor even inside RewardManager.compute's inference mode,
        # so reset() can still write it from a non-inference path.
        with torch.inference_mode(False):
            self._buf = torch.zeros_like(value)
    ...
```

`eden/utils/buffers/circular_buffer.py` uses this for storage sized by the first value pushed into it. Every
later write still follows the in-place rule.

**Per-step caches must be coherent for reset envs before observations read them.** The step loop, abridged to
the phases this rule turns on:

- **1–4** — actions, then the physics sub-steps (`command_manager.pre_physics_advance()` is phase 3)
- **5** — `command_manager.post_physics_refresh()`: per-step command caches refresh here
- **6–9** — metrics, the observation-term tick (7), terminations, rewards — all graded on the pre-reset state
- **13** — **auto-reset** of done envs: scene reset, `reset`-mode events, and every manager's `reset()`
- **14** — **`post_reset()`** on every term, for the envs phase 13 reset
- **17–18** — command resample and publish, `interval`-mode events
- **19** — **observations**: the reset envs are observed now, in their new episode

A term that refreshes its caches at phase 5 and does nothing at reset serves phase 5's *pre*-reset caches to
phase 19's *post*-reset observations: the freshly reset envs observe the dead episode's contact forces or poses,
and nothing raises. Keeping the caches coherent is your job, and the split is principled:

- **Zero in `reset()`** what is honestly unknown until the next physics step: contact forces, latched contact
  states, air time.
- **Recompute in `post_reset()`** what is derivable from the post-reset scene: poses, targets,
  phase-derived quantities.

Why two hooks: phase 13 resets managers *in sequence*, so your `reset()` runs while other managers still have
theirs to go, and anything you derive there from another entity or term can still change before phase 19.
`post_reset(envs_idx)` runs after the whole sequence — including the `episode_length_buf` zeroing and any
subclass's extra reset work, such as the mimic env refilling its motion buffer — so it is the first moment the
scene is final:

```python
def reset(self, envs_idx=None):
    idx = slice(None) if envs_idx is None else envs_idx
    self.foot_contact_force[idx] = 0.0  # unknowable until the next physics step

def post_reset(self, envs_idx):
    self._refresh_kinematic_caches(envs_idx)  # derivable now, from the final scene state
```

`envs_idx` reaches `post_reset` sanitized and never `None`, but **index with it, never measure it**: the
auto-reset path hands you the full-length boolean `reset_buf`, while `env.reset()` hands you a slice or an
index tensor. `buf[envs_idx] = ...` is always right; `len(envs_idx)` silently means `num_envs` on the step
path. The fan-out runs under `torch.inference_mode()`, so the in-place rule applies with full force.

`SkateCommand` in `eden/tasks/husky/custom_terms/commands.py` is the worked example, split exactly this way.
Audit any term with a `post_physics_refresh` or a per-step cache against this rule.

### `compute()` is a read, not a tick

An observation term's `compute()` is **not** called exactly once per control step, and nothing promises which
call is "the" step. It runs:

| When | Why | History advances? |
|---|---|---|
| once per term at **build**, before any `reset()` | the manager learns the output shape from it (term dimensions are not declared) | — |
| in **`EnvBase.reset()`** | the fresh observation the reset returns | yes |
| at **phase 19** of `RLEnvBase.step` | the canonical observation | yes (`update_history=True`) |
| at **phase 12**, before the auto-reset | the final-observation snapshot, when `EnvOptions.record_final_observations` is on | no |
| from **`RslRlVecEnvWrapper.get_observations()`** | rsl_rl reads observations outside the step, at the top of `learn()` | no |
| from the **deploy pre-flight probe** | shapes the policy's probe input; the manager is reset afterwards | no |

The cross-group cache goes the other way: two identically configured instances (a `policy` and a `critic` copy
of one term) compute **once** per observation pass, so one instance's `compute()` may never run in a step.

The "history advances?" column is also when a term's stateful [observation
modifiers](observations.md#observation-modifiers) advance: a moving average or a delay moves on each
`update_history=True` pass, together with the history window, and never on the reads.

So a term that advances state inside `compute()` — `self._integral += err * dt`, a step counter, a contact
latch — double-counts under the snapshot, counts the build probe and the reset as steps, and drifts apart across
the cache. **The failure is a wrong number, not an exception.** Eden's own Extreme Parkour port paid for this:
its stateful observation terms guarded `compute()` with a step-counter stamp, which still advanced before the
reset with the snapshot on and after it with the snapshot off, so one config observed a freshly reset env two
ways depending on a recording flag. They now tick in `post_physics_refresh`.

The contract:

- **`compute()` is a read.** It returns a value derived from the current env state and the term's buffers, and
  changes nothing a second call would see. Under that rule every caller above is harmless, and so is the cache.
- **Per-step state advances in `ObservationTerm.post_physics_refresh()`**, dispatched at **phase 7** by
  `observation_manager.post_physics_refresh()`, exactly once per control step: after the command caches (5) and
  the metrics (6), so a term may integrate either and see this step's value; and above the graded layer and the
  auto-reset. That position gives the state the semantics you expect: the phase-12 snapshot carries this step's
  increment, `reset()` then zeroes the envs that restart, and the new episode's first observation reads zero.
  `EnvBase.reset()` and the build probe run no phase 7, so the term reads zero there too. Every instance is
  ticked, including the duplicates the cross-group cache computes once, which keeps their states identical.
- **On hardware**, the deploy loop ticks it once per control tick (`DeploymentBase.state_to_observation(...,
  post_physics_refresh=True)`) after the robot state is synced, and resets the manager (`reset`, then
  `post_reset`) on each episode's first read. Unlike a command term's `post_physics_refresh`, this one runs on
  the robot: the policy's observations *are* the hardware path.
- **`build()` allocates, `reset(envs_idx)` zeroes rows, `post_reset(envs_idx)` rebuilds** what the post-reset
  scene determines, as for any per-step state.

`VelocityTrackingErrorIntegral` in `eden/managers/terms/observations/locomotion.py` is the shipped worked
example — `∫ (commanded − measured) dt` over `[vx, vy, wz]`, clamped — and the shape to copy (validation
trimmed):

```python
import torch

from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY, ObservationTerm


@OBSERVATION_TERM_REGISTRY.register()
class VelocityTrackingErrorIntegral(ObservationTerm):
    entity_name: str = "robot"
    command_name: str = "velocity_command"
    max_abs: float = 1.0

    def build(self) -> None:
        self._command = self._env.command_manager.get_term(self.command_name)  # a bad name raises here
        self.resolve_entity(self.entity_name)  # so does a bad entity
        self._integral = torch.zeros(self.num_envs, 3, device=self.device)  # allocated once, written in place

    def post_physics_refresh(self) -> None:  # phase 7: exactly once per control step
        entity = self.resolve_entity(self.entity_name)
        measured = torch.cat([entity.get_vel(frame="body")[:, :2], entity.get_ang(frame="body")[:, 2:3]], dim=-1)
        self._integral.add_((self._command.command[:, :3] - measured) * self._env.dt).clamp_(
            -self.max_abs, self.max_abs
        )

    def reset(self, envs_idx=None) -> None:  # phase 13: the restarting rows
        self._integral[slice(None) if envs_idx is None else envs_idx] = 0.0

    def compute(self) -> torch.Tensor:  # a read, however often it runs
        return self._integral
```

`tests/envs/test_base.py` runs the real `RLEnvBase.step` against a probe term like this one and pins the spec:
zero after `env.reset()`, exactly `k·dt` after `k` steps, unchanged by `get_observations()` and by the
final-observation snapshot, and zero again for an env the step auto-resets while the others keep counting.

### Which hook: grade the step, or prepare the next one

The step loop has two epochs, split by the auto-reset:

- **Grading** — metrics, the observation-term tick, terminations and rewards at phases 6–9, fed by
  `command_manager.post_physics_refresh()` at 5. Everything here describes *the step that just happened*, read
  off the pre-reset state the action produced.
- **Preparation** — curricula, `command_manager.compute()`, `interval` events and observations at phases
  16–20. Everything here sets up *the next* step, on post-reset state.

Which hook a piece of per-step state belongs in follows from what that state *is*. For a command term it is
three hooks and one rule (an observation term with per-step state has the same first row, ticked at phase 7
rather than 5, with `compute()` as its read):

| The state is… | Write it in | Phase |
|---|---|---|
| a **measurement of what this step produced** — progress counters, hold or contact latches, tracking errors, an observation's integral: anything derived from the post-physics robot | `post_physics_refresh()` | 5 (command), 7 (observation) |
| the **command itself** — the value the policy is given, resampled and published for the step to come | `_resample_command()`, `_update_command()` | 17, and 13 for a resample on reset |
| a **time-advancing reference** the physics must be graded *against* ("advance, then grade" — `MotionCommand`'s clip clock) | `pre_physics_advance()` | 3 |

The split is **what the state is, not who reads it**. Rewards read the command too — `TrackingLinVel` reads
`self._command.command` off `UniformVelocityCommand`, and `GaitPhaseCommand`'s phase *is* its command — and
those stay at 17. Ask whether the value measures the outcome the action produced (grading) or is the command
the policy was given and is about to be given next (publication).

Resampling runs twice: `CommandTerm.reset()` also calls `_resample()`, so `_resample_command` fires at phase 13
on the auto-reset path as well as at 17. Either way it receives a `(num_envs,)` bool mask with at least one
`True`, every control step — which is why its docstring asks you to write through `torch.where` rather than
index with the mask.

Every other kind that derives state other terms read is pinned to one epoch by its manager: a metric computes
at phase 6 and a reward at 9 (grading); an observation reads at 19 (preparation). Command terms and stateful
observation terms are the two kinds with hooks on both sides of the reset, so they are the two kinds that can
put a value in the wrong epoch — for an observation term the wrong place is `compute()` itself. (Recorder terms
straddle the reset too, but they only write data out, so there is nothing downstream to serve stale.)

**Grading state written in `_update_command` is read one step late, and whatever that same call clears is
never read at all.** The reward at phase 9 of step *k*+1 sees what `_update_command` wrote at phase 17 of step
*k*. A counter that call *zeroes* — on a goal switch, a resample, a phase wrap — takes its last value with it,
so the step that earned it is graded through the cleared one. Nothing raises; the reward is simply 0 where it
should have been largest, which is invisible in a training curve.

That is a real bug, and it is where this rule comes from. `InHandReorientCommand`
(`eden/tasks/reorient/custom_terms/commands.py`, in-hand cube reorientation) once ran its whole success state
machine — on-target hold counter, success window, window timer — in `_update_command`, and paid its escalating
hold reward through `reward_*` shadow buffers snapshotted just before the goal switch cleared the live ones.
The snapshot covered the counter but not the gate beside it, so the highest-paying step of every goal cycle
paid zero for a whole training run:

```text
paid:     … 15, 16, 17, 18, 0
expected: … 16, 17, 18, 19, 0
```

Snapshotting the gate too would patch one term. The rule removes the class of bug: grade in
`post_physics_refresh`, four phases above the reward in the same step, and there is no lag to carry a value
back across and nothing to shadow. The goal *switch* stays in `_update_command`, because a switch is
publication. `InHandReorientCommand`, `SkateCommand` (`eden/tasks/husky/custom_terms/commands.py`) and
`ParkourCommand` (`eden/tasks/extreme_parkour/custom_terms/commands.py`) are split this way today; none carries
a shadow buffer. A `reward_*` shadow buffer in a command term is the smell.

!!! warning "A reward and an observation read different epochs of the same attribute"
    Phase 9 and phase 19 read the *same* command attribute at opposite ends of the step, with phase 17 between
    them, so one attribute can mean two things depending on which manager reads it. The reorient task's
    `command_state_progress` observation once reported a full success window to the critic for a goal that had
    been replaced on that very step. When a quantity has both kinds of reader, compute it in
    `post_physics_refresh` and leave it alone for the rest of the step; then both see one value.

!!! danger "Deployment runs two of the three command hooks"
    The rule above is about the *training* loop. `DeploymentBase` (and the backends that route their own step
    through its `_update_commands`) advances the reference with `command_manager.advance_reference()` — which
    defaults to each term's `pre_physics_advance()` — then publishes with `command_manager.compute()` and
    `draw_vis()`. **A command term's `post_physics_refresh` never runs on hardware**: there are no physics
    sub-steps to bracket and no reward to grade. A command term whose deployed observations depend on a side
    effect of that hook reads a value frozen at its last simulated step, silently.

    The advance reaches hardware only because the deploy loop drives it explicitly. A refactor once moved
    `MotionCommand`'s clip clock from `_update_command` (which deployment reached, through `compute()`) into
    `pre_physics_advance` (which it then did not): correct in simulation, and a clip frozen at its first frame on
    every deployed `MotionCommand` policy. `tests/extensions/deployment/test_base.py` now pins both reference
    paths. A term whose simulation advance also teleports or resamples the robot (a clip wrap) overrides
    `advance_reference()` with a clock-only advance.

    A term with a clock needs the reset side too. `CommandTerm.rewind_reference()` runs **only** on the
    deployment path — `DeploymentBase.reset_motion_reference` drives it at deploy reset, at START and on every
    policy switch — and it must rewind the clock *alone*: the simulation's way of restarting a clip places the
    robot and perturbs its pose, which is right for a new episode and meaningless for a robot already standing
    where it is.

    What survives the crossing is an **idempotent** refresh — a cache recomputed from current state, safe to
    run in two hooks. `MotionCommand` refreshes its robot-state cache in `post_physics_refresh` for the reward
    and again in `compute()` for the post-reset state the observation reads, which also covers the hardware
    path. A **non-idempotent** advance — a counter, a latch, a state machine — cannot simply be duplicated,
    because running it twice per control step is a different bug. So: if the term is training-only, follow the
    rule and stop. If it will be deployed, check what the *policy's* observations read off it (the reward is
    not on the hardware path) and make sure `compute()` — or `pre_physics_advance`, for a clock — produces that.

**Why not move the command update above the reward and have one epoch?** Because the policy chose this step's
action from observations computed at phase 19 of the *previous* step, against the command those observations
carried. Grading at phase 9 against that same command is what keeps `observation → action → reward` aligned on
one command. Resampling before the reward would grade the robot against a goal drawn *after* the physics that
produced its state — a goal the policy never saw — giving a reward drop attributed to an action that could not
have caused it. Phase 17 is the only position that keeps the triple aligned, so what moves is the hook you
pick, not the step order.

### Share a computation across terms with a MetricTerm

If more than one term needs the same quantity — a tracking error used by a reward and an observation, a
contact state used by a reward and a termination — compute it **once** as a metric term and pull it by name,
instead of recomputing it in each term or caching it on `env`. The metric manager runs at phase 6, before the
observation tick (7), terminations (8) and rewards (9), so the pulled value is current. Metric terms themselves
are covered in [Metrics and recorders](metrics-and-recorders.md#using-a-metric-from-other-terms).

```python
from eden import observations, rewards, terminations
from eden.options import ObservationGroupOptions
from eden.options.managers.rewards import RewardManagerOptions
from eden.options.managers.terminations import TerminationManagerOptions

reward_options = RewardManagerOptions(
    # the metric's value times the weight (negative for a penalty)
    score=rewards.MetricReward.configure(metric_name="my_score", weight=1.0),
)
policy = ObservationGroupOptions(
    # the same value as a (num_envs, 1) observation
    score=observations.MetricObservation.configure(metric_name="my_score"),
)
termination_options = TerminationManagerOptions(
    too_far=terminations.MetricAbove.configure(metric_name="my_score", threshold=0.5),
)
```

The pull terms read `metric_manager.get_value(name)`, the manager's stable, in-place-updated buffer:

| Pull term | Reads | Notes |
|---|---|---|
| `rewards.MetricReward(metric_name, allow_cumulative=False)` | `get_value` | refuses a cumulative metric (its value is the running episode sum, which a per-step reward would integrate again) unless `allow_cumulative=True` |
| `observations.MetricObservation(metric_name)` | `get_value`, as `(num_envs, 1)` | an env that reset this step reads the metric's post-reset value |
| `terminations.OnMetricSuccess(metric_name)` | the metric's own current-step success check | INTERVAL metrics that contribute to success only |
| `terminations.MetricAbove(metric_name, threshold=0.0)` | `get_value(name) > threshold` | also accepts a `contributes_to_success=False` metric, such as a collision flag; on a cumulative metric it bounds the episode total |
| `rewards.TaskSuccess()` | `metric_manager.success_buf` | 1 on a step where every success metric holds |

All of them resolve their metric in `build()` through `resolve_metric_manager`, so a missing metric manager or
an unknown `metric_name` raises at env construction, listing the active metrics. Write your own pull terms the
same way.

When several terms derive *different* values from one intermediate that is not itself a metric — a
body-position and a body-rotation error from the same anchor-relative transform, say — compute it through
`env.compute_once(key, producer)`. `producer()` runs at most once per pass for each `key`, and the memo is
dropped when the pass ends; outside a pass it simply runs, so a value is never served stale. Each manager's term
loop opens a pass, and `RLEnvBase.step` opens one over its whole graded phase (metrics, the observation tick,
terminations, rewards), so a key a metric produced is served to a reward that asks for it. A key must therefore
name its value completely — the producer and everything it depends on (entity, link names, …) — and every
caller receives the same object, so treat it as read-only. `anchor_state_once` and `anchor_relative_links_once`
in `eden/utils/motion_reference/lookup.py` are the in-tree users. Each `observation_manager.compute()` is a
pass of its own, because it runs after the auto-reset on a different state.

## Validation fails fast

Eden surfaces config mistakes at the earliest well-defined point — construction, `configure()`, or `build()` —
rather than inside a `compute()` step. When you write or wire up terms, expect these to raise:

- **Unknown fields on fixed-schema options raise at construction.** Options classes default to
  `extra="forbid"`, so `EnvOptions(decimatoin=4)` is a `pydantic.ValidationError`. Only classes that hold
  dynamically named children (manager options, `ObservationGroupOptions`, `SceneOptions`, camera and sensor
  holders) allow extras.
- **Misspelled term parameters raise.** `RewardTerm.configure(func=..., wieght=3.0)` raises `ValueError`; so
  does `RewardTermOptions(name="upright", wieght=2.0)` — at construction when `name` is set and the kind's
  registry is importable, on later attribute assignment (rolled back), on `updated(...)` and
  `model_copy(update=...)`, and as a backstop at build. The message names the term and the bad parameter and
  lists every accepted one with its type and default, in the form the [Term Catalog](../api/terms/catalog.md)
  uses.
- **Override one field of an inherited term with `updated()`.** `term_options.updated(weight=0.0)` is a deep
  copy with those fields replaced and checked. `model_copy(update={"weigth": 0.0})` checks the names too (its
  values stay unvalidated, as in pydantic), for term options, manager options, observation groups and every
  fixed-schema class.
- **Observation groups check their entries.** An entry that is neither a group field nor a term
  (`terms_ordr=[...]`) raises `TypeError` at construction, on `model_copy`/`updated`, and on assignment. A term
  written as a plain `dict` must name its term (`{"name": "base_lin_vel", ...}`); one that does not is treated
  as the misspelled group field it usually is. Details are in [Observations](observations.md#observation-groups).
- **Manager options check an assigned entry.** `reward_options.bogus = "x"` raises on that line, not at build.
- **A term cannot take a name its holder's class already has** (`lock`, `clone`, `keys`, `term_keys`, …), in a
  manager's options or an observation group — it would shadow the method. Both construction and assignment
  raise `TypeError`; rename the entry (the piano task's `keys` became `key_positions`).
- **An unregistered `func=` or a missing required parameter raises at `configure()`**, before any env is
  built. The message names the function, the missing parameters and the full set it accepts.
- **`func=` must be the function, not its name.** `RewardTerm.configure(func="dofs_vel_limits")` raises
  `TypeError`.
- **`params=` on a class term raises** — unless the term used to be a function term; see
  [Converting a function term to a class term](#converting-a-function-term-to-a-class-term).
- **Build errors name the entry.** An exception raised while a manager constructs or builds a term carries a
  note naming the entry and its registry key, for example
  `while building RewardManager term 'lift_milk' (name='metric_reward')`, since a task can hold many entries of
  one term. A nameless entry (`name=None`) raises `ValueError` saying so.
- **An entity-bound term opens its own errors with `term_subject(self)`.** From `eden.utils.string`, it renders
  `"<Class> on entity '<name>'"`:
  `raise ValueError(f"{term_subject(self)}: std must be > 0, got {self.std}.")`. Don't write the class or entity
  by hand, and don't add the config key — the build note already does. How to word the rest of the message is
  in [Error messages](error-messages.md).
- **Env-options preconditions raise while the env is built.** A few terms need a scene-wide `EnvOptions` flag
  no parameter of theirs can supply — the per-env domain-randomization terms need batched solver data
  (`RandomizeKpKdGains` → `batch_dofs_info`; `RandomizeComShift`, `RandomizeMassShift`, `RandomizeLinkMassScale`
  → `batch_links_info`). Declare one as a `_requires_env_options` tuple — a class attribute on a class term, or a
  registration keyword on a function term
  (`@EVENT_TERM_REGISTRY.register(_requires_env_options=("batch_links_info",))`) — and the term raises
  `ValueError` at construction naming itself and the flag. A malformed declaration (a bare string, or a name that
  is not an `EnvOptions` field) is refused when the class is defined. `eden terms` prints the flags on a
  `requires:` line and the catalog has a `Requires` column. Use it for env-options flags only; a requirement the
  term can inspect itself (a sibling term, a named link) stays a check in its own `build()`. Keep the leading
  underscore, or the name becomes a configurable parameter.
- **Action terms** refuse a multi-DOF joint in `dofs_name`, two terms commanding one DOF, and a term that
  neither draws an action column nor commands a DOF. See [Rules the action manager
  enforces](actions.md#rules-the-action-manager-enforces).

## Gotchas when writing terms

- **DOF ordering.** Full-width DOF reads (`get_dofs_*()`), `get_dofs_limit`, the `default_dofs_*` tensors and
  `soft_dofs_pos_limits` are in the entity's configured `dofs_name` order — the *column* space. `get_jacobian`
  uses the same order after six base columns on a floating base: index it with `columns + 6` unless
  `entity.has_immobile_base`. `find_named_dofs_idx_local` returns Genesis tree order — the *local* space — which goes only
  into a `dofs_idx_local=` argument. Resolve once in `build()` with `find_named_dofs(names)` and hand its
  `.columns` to tensors and its `.dofs_idx_local` to Genesis calls; read configured properties by name with
  `get_default_dofs_*` in `build()`. See [Entities](entities.md).
- **Quaternions are wxyz** everywhere in Eden, config fields included.
- **Particle entities can't be moved at runtime.** `ParticleEntity.set_pos()` / `set_quat()` (MPM, SPH, PBD)
  raise `NotImplementedError`; bake the pose into the morph at creation time.
- **Auto-reset happens before observation.** In `RLEnvBase.step()`, resets run after rewards and terminations
  but before observations, so the observations returned for done envs are their new episode's first frame. Turn
  on `EnvOptions.record_final_observations` to get the terminal frame in `extras["final_observations"]`. This
  ordering is also why a term with per-step caches needs the [reset
  discipline](#term-state-lifecycle-buffers-reset-and-torch-modes).
- **`eval_mode` turns training off, not envs.** `eval_mode` sets `env.training = False`, which skips
  observation noise and other training-only behaviour; `num_envs` is unchanged. Gate randomization on
  `env.training`.
- **Observation post-compute fields are options-only.** `modifiers`, `noise`, `clip`, `scale`,
  `history_length`, `flatten_history_dim` and `backfill` are accepted by `configure()` but not mirrored as term
  attributes, so a `compute()` cannot read one. Don't declare a field named `scale` on an observation term.
  `noise` and `scale` are resolved once at build; change a scale afterwards with
  `env.observation_manager.set_term_scale(group, name, scale)`. See [Observations](observations.md).
- **Stay sync-free on the hot path.** No `.item()`, `.cpu()`, `.tolist()` or `if tensor.any():` in `compute()`
  unless you must — and then say why in a comment. A boolean mask used as an *index* syncs too: `x[mask]` and
  `x[mask] = y` are sized on the host. Prefer `torch.where(mask, y, x)` or a scalar fill (`x[mask] = 0.0`, which
  stays on the device). This matters most in `reset()`, which the auto-reset calls with the bool `reset_buf`
  every step: reduce a log value with `eden.utils.misc.sum_over_envs` / `mean_over_envs`, and write rows from a
  full-batch source with `eden.utils.misc.assign_over_envs`. Where a reset must gather the selected rows, take
  `ids = eden.utils.misc.env_indices(envs_idx)`: the auto-reset shares one `nonzero()` of `reset_buf` with every
  consumer. Build constant tensors in `build()`, or with `eden.utils.torch.device_constant`; index with a 0-dim
  CUDA tensor through `index_select`, not `x[:, idx[0]]`; use `torch.linalg.solve_ex` / `inv_ex` where the
  system cannot be singular. `torch.cuda.set_sync_debug_mode("warn")` flags syncs as they happen, and
  `count_cuda_syncs` in `tests/conftest.py` pins a budget in tests.
- **Pass `envs_idx=None` for "every env"**, not an all-envs index tensor: the default is generated on the
  device. An index tensor you cached in `build()` is fine to pass to a `RigidEntity` accessor, which resolves it
  to a `range` once per tensor.
- **Build every tensor with `device=`.** There is no torch default device — neither Genesis nor Eden sets one,
  because torch implements a default device as a Python `TorchFunctionMode` that every torch call is dispatched
  through, which `eden/__init__.py` measured at 1.7–2.0 ms per velocity-tracking control step.
  `torch.zeros(n)` is a CPU tensor, as in plain torch.
  Allocate with `device=self.device` (in a term) or `device=env.device`, derive from an existing tensor
  (`torch.zeros_like`, `x.new_zeros`), or pass `device="cpu"` deliberately, with a comment saying why. The
  default *dtype* is set by `eden.init` to Genesis' float, so a dtype-less float tensor matches Genesis' own.
  The test suite runs on CPU, where a missing `device=` cannot show, so `tests/envs/test_base.py` steps a few
  tasks under a `TorchFunctionMode` and fails on any factory call without one.
- **Compile pure tensor math** with `@compile_model` (`eden.utils.torch`) in a free function and call it from
  `compute()`, as the locomotion and tracking terms do.

### Profiling a term you just wrote

A term that is correct but expensive does not announce itself — you get a training run that takes hours longer
than it should. The profiler's own docstring (`eden/utils/profiling/term_profiler.py`) records two runs of the
same task on the same GPU differing 35× in throughput, purely because of per-call allocations inside the task's
own terms. Before tuning hyperparameters or adding envs, measure where the step goes:

```bash
eden smoke --task <name> --profile
```

This builds the task, steps it for a short window (`--profile_warmup`, default 10 steps; `--profile_steps`,
default 50), attributes wall time to each per-step term, prints a ranked table and writes `term_cost.json` to
Eden's log directory (`eden.get_log_dir()`):

```json
{"num_steps": 50, "step_us": 2684.8,
 "terms": [{"manager": "observation", "term": "policy.slow_upright",
            "us_per_step": 2488.2, "pct_step": 92.7, "calls_per_step": 1.0}]}
```

One term at 92% of the step is the bug, and the fix is almost always the advice above: hoist constants and
index resolution into `build()`, drop the host syncs. Profiling is opt-in: without `--profile`, nothing is
wrapped and the framework pays no profiler overhead.

## Registries

Every term kind has its own `TermRegistry`, importable from `eden.managers` (see [the table
above](#terms-at-a-glance)). A registry maps a key to the term, and every manager resolves the terms of its
options through it.

### Naming convention

- **The key is always `to_snake_case(__name__)`** — for classes and functions alike:
  `RelativeLinkPosition` → `"relative_link_position"`, `UniformT3Command` → `"uniform_t3_command"`,
  `distance_penalty` → `"distance_penalty"`.
- **A term registry's `register()` takes no `name=`.** `configure()` stamps `to_snake_case(__name__)` into
  `options.name` and the manager resolves the term back by that name, so a registration cannot choose a
  different key — `@REWARD_TERM_REGISTRY.register(name="custom_key")` raises `ValueError`. Rename the term to
  change its key.
- **`to_snake_case` leaves digit suffixes alone**: `Modern1` → `"modern1"`, not `"modern_1"`.
- **Other keyword arguments to `register()` are the term's defaults**, function terms included:
  `@METRIC_TERM_REGISTRY.register(direction=MetricDirection.LIB, is_cumulative=True)` on a function makes those
  its defaults whether it is configured with `MetricTerm.configure(func=...)` or named with
  `MetricTermOptions(name=...)`. A field you pass explicitly still wins. Naming a *class* term in its options
  fills the fields you leave out from the class's own attributes the same way, so
  `EventTermOptions(name="push_by_velocity")` is an interval event, as `PushByVelocity.configure()` is.
- **One flat namespace per kind.** Two terms of one kind with the same key raise at import, naming both
  modules. An out-of-tree term whose class name collides with a built-in must be renamed (`MyExtUpright`, not
  `Upright`).
- **A key starting with `_` warns** — that is usually an internal class that should not be registered.
- **Every module a registry's terms live in must be listed in its `populated_by`**, because `import eden`
  imports no term module; the registry imports those modules on its first read. A term defined in a module the
  registry does not list reads as "no such term". This is about Eden's built-in term modules; a term defined in
  a task's own module is registered when that module is imported, which happens when the task's config is.

Asset, robot and task registries are the opposite case: they are looked up only by key, so `name=` there is a
normal alias (`@TASK_REGISTRY.register(name="beyondmimic")`).

### Renaming a term

Renaming a term changes its key, so configs saved to disk next to checkpoints stop resolving. If you need the
old key to keep loading, register a subclass under the old name:

```python
@REWARD_TERM_REGISTRY.register()
class NewRestHeight(RewardTerm):
    ...


@REWARD_TERM_REGISTRY.register()
class OldRestHeight(NewRestHeight):
    """Deprecated alias for NewRestHeight, kept so saved configs still load."""
```

Both keys now exist and `old_rest_height` builds the subclass, which inherits every field and behaviour.

### Converting a function term to a class term

A function term that needs to resolve something once, or keep state, becomes a class term. To keep saved
configs loading:

1. **Name the class after the function** (`foo_bar` → `FooBar`), so the registry key does not change.
2. **If the function took `params`, list those keys in `_function_era_params`.** A config saved before the
   conversion still carries `params={"metric_name": ...}`, which a class term never reads. At construction,
   a `params` dict holding only keys from `_function_era_params` is moved onto the class's top-level fields;
   any other key, `params` on a term that was never a function, or a key given both ways still raises.

`DofsPos` (`eden/managers/terms/observations/proprio.py`) is an example:

```python
@OBSERVATION_TERM_REGISTRY.register()
class DofsPos(_DofsObservation):
    offset_from_default: bool = False

    _function_era_params = frozenset({"entity_name", "offset_from_default", "terms_idx"})
```

`MetricReward` (`metric_name`, `allow_cumulative`), `MetricObservation` and `OnMetricSuccess` (`metric_name`)
were converted the same way.

### Inline registration

You can define and register a term right in a task's config module; no separate module is needed. The reacher
task registers its `reaching_reward` this way (`eden/tasks/benchmark/reacher/config.py`). The shape is:

```python
import torch

from eden.managers import REWARD_TERM_REGISTRY
from eden.options.managers.rewards import RewardManagerOptions
from eden.terms import RewardTerm


@REWARD_TERM_REGISTRY.register()
def my_inline_reward(env, *, scale: float) -> torch.Tensor:
    return torch.full((env.num_envs,), scale, device=env.device)


reward_options = RewardManagerOptions(
    bonus=RewardTerm.configure(func=my_inline_reward, weight=1.0, params={"scale": 2.0}),
)
```

Larger tasks keep their terms in a `custom_terms` package next to `config.py`; the layout is in
[Task configs](task-config.md).

## Finding built-in terms

Every registered term, with the parameters it accepts, their types and defaults, and which are required, is in
the generated **[Term Catalog](../api/terms/catalog.md)**. The same listing is available from the shell:

```bash
eden terms                                    # every term, grouped by kind
eden terms --kind reward                      # one kind
eden terms --grep height                      # name, parameters and summary are all searched
eden terms --kind reward --grep "feet|foot"   # --grep is a case-insensitive regex
eden terms --json                             # machine-readable, for tooling
```

`--grep` is the form to reach for first, and the one a failed registry lookup points you at: it searches
parameter names too, so "which term takes `max_vel`?" is `eden terms --grep max_vel`. It exits 1 when nothing
matches, so a script can branch on it.

An index of the catalog — every term's key, class, module and summary, without parameters — ships inside the
installed package at `eden/resources/term_catalog.md`, for an Eden installed without the docs. Both pages are
generated from the registries: after adding, renaming or re-parameterizing a term, regenerate them from a repo
checkout with `uv run python -m eden.managers.catalog` (`--check` reports staleness without writing).
`tests/resources/test_term_catalog.py` fails while either is stale.

Look a term's parameters up before writing it into a config: they are not guessable from its name, and getting
them wrong costs an env build.

## See also

- [Actions](actions.md), [Observations](observations.md), [Rewards and terminations](rewards-and-terminations.md)
- [Commands](commands.md), [Events](events.md), [Curricula](curricula.md),
  [Metrics and recorders](metrics-and-recorders.md)
- [Environments](environments.md) — the step loop and manager order
- [Error messages](error-messages.md) — what a term's error must say
- [Testing](testing.md) — how to test a term
- [Term Catalog](../api/terms/catalog.md) and the [managers API](../api/managers.md)
