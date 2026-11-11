# Observations

The observation manager computes what the policy (and the critic) see each step. This page covers observation
groups, the per-frame pipeline every term's output goes through — modifiers, noise, clip, scale, the non-finite
check and history — the built-in terms, sensors and cameras as observations, and how to write your own term or
modifier. The general term contract is in [Writing terms](terms.md); read its
[`compute()` is a read](terms.md#compute-is-a-read-not-a-tick) section before you write a stateful observation.

## Observation groups

`observation_options` is an `ObservationManagerOptions` whose entries are **groups**
(`ObservationGroupOptions`), and each group's entries are **terms**. The manager computes every group and returns
a `dict` from group name to that group's tensor. The velocity-tracking tasks
(`eden/tasks/velocity_tracking/base_config.py`) define a noisy, history-stacked policy group and a clean critic
group that adds a privileged term:

```python
from eden import observations
from eden.managers.modifiers.observations import UniformNoise
from eden.options import ObservationGroupOptions, ObservationManagerOptions
from eden.terms import ObsTerm

_POLICY_OBSERVATIONS = dict(
    base_ang_vel=ObsTerm.configure(
        func=observations.base_ang_vel,
        noise=UniformNoise.configure(n_min=-0.2, n_max=0.2),
        params={"entity_name": "robot", "frame": "body"},
    ),
    projected_gravity=observations.ProjectedGravity.configure(
        entity_name="robot",
        noise=UniformNoise.configure(n_min=-0.05, n_max=0.05),
    ),
    dofs_pos=observations.DofsPos.configure(
        noise=UniformNoise.configure(n_min=-0.01, n_max=0.01), entity_name="robot", offset_from_default=True
    ),
    dofs_vel=observations.DofsVel.configure(
        entity_name="robot",
        noise=UniformNoise.configure(n_min=-1.5, n_max=1.5),
        scale=0.05,
    ),
    actions=ObsTerm.configure(func=observations.last_action),
    commands=ObsTerm.configure(func=observations.generated_commands, params={"command_name": "velocity_command"}),
)

_CRITIC_OBSERVATIONS = dict(
    base_lin_vel=ObsTerm.configure(
        func=observations.base_lin_vel,
        noise=UniformNoise.configure(n_min=-0.5, n_max=0.5),
        params={"entity_name": "robot", "frame": "body"},
    ),
    **_POLICY_OBSERVATIONS,
)

observation_options = ObservationManagerOptions(
    policy=ObservationGroupOptions(**_POLICY_OBSERVATIONS, enable_corruption=True, history_length=5),
    critic=ObservationGroupOptions(**_CRITIC_OBSERVATIONS, enable_corruption=False),
)
```

Group names are yours to choose. With rsl_rl, the runner's `obs_groups` maps groups onto the actor and the
critic; when it is left empty, rsl_rl falls back to the group named `policy` for the actor and to `critic` (or
`policy`, when there is no `critic`) for the critic, and warns that `obs_groups` is unset. See
[Training runs](training-runs.md).

A group's own fields:

| Field | Default | Meaning |
|---|---|---|
| `concatenate_terms` | `True` | concatenate the terms into one tensor; `False` returns a `dict` from term name to tensor |
| `concatenate_dim` | `-1` | the dimension to concatenate along |
| `terms_order` | `None` | the order of the terms in the output; `None` is declaration order. If set, it must name every term of the group exactly once |
| `enable_corruption` | `True` | `False` drops the `noise` of every term in the group (see [Noise](#noise)) |
| `history_length` | `None` | a history window enforced on the whole group (see [History](#history)) |
| `flatten_history_dim` | `True` | flatten a group history into the feature axis |
| `backfill` | `True` | fill a fresh group history with the first frame instead of zeros |
| `nan_policy` | `"disabled"` | what to do with a NaN or ±Inf a term delivers (see [Non-finite values](#non-finite-values-nan_policy)) |

**Sharing a term dict between groups is safe.** A group takes its own deep copy of every term handed to it —
whether through the constructor, `model_copy(update=...)`, `updated(...)` or attribute assignment — so the
`policy` and `critic` groups above never share a term object, and editing one group's term never edits the
other's. A term may be written as a plain `dict`, but it must then name its registered term
(`{"name": "base_lin_vel", ...}`); a nameless dict is refused as the misspelled group field it usually is.

**Adding or removing a term** on an inherited config:

```python
group = config.observation_options.policy
group.base_height = ObsTerm.configure(func=observations.base_height, params={"entity_name": "robot"})
group.disable_term("actions")  # also removes it from terms_order, if the group declares one
```

An assigned term is checked on that line — an unknown name, a bad parameter, or a name the group class already
has (`keys`, `clone`, …) raises there, not at build. When a group declares `terms_order`, an added term must
also be listed in it.

## The per-frame pipeline

Every frame of a term's output goes through the same stages, in this order:

```text
raw compute() → modifiers (in list order) → noise → clip → scale → group nan_policy → history window
```

These are the fields every observation term accepts, beside its own parameters:

| Field | Default | Stage |
|---|---|---|
| `modifiers` | `[]` | transforms such as a moving average or a delay; [Observation modifiers](#observation-modifiers) |
| `noise` | `None` | a noise model, applied in training only; [Noise](#noise) |
| `clip` | `None` | `(min, max)` clamp |
| `scale` | `1.0` | a float, or a list matched against the term's feature dimension |
| `history_length` | `0` | a per-term history window; [History](#history) |
| `flatten_history_dim` | `True` | flatten the window into the feature axis |
| `backfill` | `True` | fill a fresh window with the first frame instead of zeros |

They are **options-only**: `configure()` accepts them and the manager applies them from outside, but they are
not mirrored as attributes on the term, so a term's `compute()` cannot read one. Don't declare a parameter named
`scale` (or any of these) on an observation term of your own.

The order matters when you port a config. In the velocity-tracking `dofs_vel` term above, the noise
(`±1.5` rad/s) is added before `scale=0.05`, so it stays in rad/s rather than being rescaled.

### Noise

Three noise models are built in, in `eden.managers.modifiers.observations`:

| Model | Parameters |
|---|---|
| `UniformNoise` | `n_min=-1.0`, `n_max=1.0` |
| `GaussianNoise` | `mean=0.0`, `std=1.0` |
| `ConstantNoise` | `bias=0.0` |

Each also takes `operation`: `"add"` (the default) adds the noise, `"scale"` multiplies by it. Values may be
floats or tensors (per-component).

Noise is **training-time corruption**. It is skipped whenever `env.training` is `False` — eval, play
(`eval_mode`) and deployment — and dropped for every term of a group with `enable_corruption=False`. The group
keeps the `noise` in its config, and the manager logs one info line at build naming the group and the terms
whose noise it is not applying. That is what lets one term dict serve a noisy `policy` group and a clean
`critic`.

!!! note "Noise is added after the modifiers"
    Modifiers belong to the signal the policy was trained on, so they run in every mode; noise does not. In
    training, a moving average therefore smooths the *clean* signal and the noise reaches the policy
    unsmoothed, while on hardware the same average smooths the real sensor's noise. To train against filtered
    noise, add the noise inside the term, or in a modifier of your own placed ahead of the average.

### Clip and scale

`clip` is a `clamp`, so it passes NaN through — it is no guard against NaN on its own (see
[`nan_policy`](#non-finite-values-nan_policy)). It is read every step, so changing `term._options.clip` after
build takes effect at once.

`scale` is a float, or a list giving a per-component scale matched against the term's **feature dimension**:
the last dimension of its raw `compute()` output, before history stacking. The length is checked at build, and
a term whose raw output is `(num_envs,)` — no feature dimension — refuses a list scale. An all-ones list or
`1.0` costs nothing on the hot path.

`scale` and `noise` are resolved **once, at build**. Assigning `term._options.scale` afterwards changes what the
options report but not what the manager multiplies by, and nothing raises. Read what is applied off
`term.effective_scale`, and change it at runtime with:

```python
env.observation_manager.set_term_scale("policy", "dofs_vel", 0.1)
```

which re-runs the build-time check and updates both. Call it at setup: because scale is applied before a frame
enters the history window, retuning a term whose window already holds frames would mix two scales in one
window, so `set_term_scale` refuses that with a `RuntimeError`. The window counts as empty before the first
`compute()` and again after a full `env.reset()` — which is where the training wrappers leave you, since their
`from_config` builds and resets.

### History

A term's `history_length` keeps a window of its last `history_length` frames, oldest first. With
`flatten_history_dim=True` (the default) a `(num_envs, dim)` term becomes `(num_envs, history_length * dim)`;
with `False` it is `(num_envs, history_length, dim)`. With `backfill=True` (the default) a fresh window — after a
reset — is filled with the first frame; with `False` it starts at zeros. `history_length=1` logs a warning; use
`0` for no history.

A group's `history_length` instead keeps **one** window over the concatenated group, which requires
`concatenate_terms=True` and replaces every term's own history settings (the manager logs which ones it
overrides). The two layouts differ when flattened: per-term history gives
`[term_a frames…, term_b frames…]`, a group history gives `[frame_oldest (a, b), …, frame_newest (a, b)]`. Match
the layout the policy you are porting expects.

The window advances once per control step and once per `env.reset()`, on the manager's `update_history=True`
pass. Other reads — the final-observation snapshot, `get_observations()` — compute the current frame and show it
in the newest slot without pushing it.

### Observation modifiers

`modifiers` is a list of transforms applied to the term's output in list order, before noise. Unlike noise, they
run in **every** mode — training, eval and deployment — because a policy trained on a delayed or smoothed
observation must be deployed with the same delay or smoothing. Two are built in, in
`eden.managers.modifiers.observations`:

| Modifier | Parameters | What it delivers |
|---|---|---|
| `ExponentialMovingAverage` | `alpha` (required, in `(0, 1]`), `backfill=True` | `y(t) = alpha * x(t) + (1 - alpha) * y(t-1)`; `alpha` weights the newest frame (as in pandas' `ewm`). With `backfill`, the first frame after a reset is delivered unchanged. Needs a floating-point frame |
| `ObservationDelay` | `control_steps` (required, `>= 1`), `backfill=True` | the frame from `control_steps` control steps earlier; until enough frames exist, the first frame (or zeros without `backfill`) |

```python
from eden import observations
from eden.managers.modifiers.observations import ExponentialMovingAverage, ObservationDelay, UniformNoise
from eden.options import ObservationGroupOptions

policy = ObservationGroupOptions(
    dofs_vel_smooth=observations.DofsVel.configure(
        entity_name="robot",
        modifiers=[ExponentialMovingAverage.configure(alpha=0.2)],
        noise=UniformNoise.configure(n_min=-0.01, n_max=0.01),
    ),
    dofs_pos_late=observations.DofsPos.configure(
        entity_name="robot",
        modifiers=[ObservationDelay.configure(control_steps=2)],
    ),
    nan_policy="sanitize",
)
```

Stateful modifiers advance on the same `update_history=True` pass as the history window, so a delay of `k` is
`k` window pushes. Each term instance owns its modifier instances: the same modifier on a `policy` and a
`critic` copy of a term keeps two independent states.

### Non-finite values: `nan_policy`

`nan_policy` is a group field. It checks each term's frame at the end of that term's pipeline — after clip and
scale, before the history window — so every report names the term, and a `clip` that bounds an infinity (a depth
or raycast term's `+inf` for "no hit") is never reported:

| `nan_policy` | A NaN, or an infinity no clip bounded | Cost |
|---|---|---|
| `"disabled"` (default) | reaches the policy unchanged | none |
| `"sanitize"` | replaced with 0.0 | on the device, no sync |
| `"warn"` | replaced with 0.0; the first occurrence per term is logged with the env ids | one GPU sync per group per compute, while any of the group's terms has not warned yet |
| `"error"` | raises `FloatingPointError` naming the term, the kind of value and the envs | one GPU sync per group per compute |

The check covers every computation, the final-observation snapshot included: a simulation that diverged to NaN
just before an env terminated shows up there, and that frame is what a value bootstrap reads. What it cannot do
is clean a stateful modifier upstream of it. A NaN that reaches an `ExponentialMovingAverage` stays in the
average until the env resets, and for that time the policy receives 0.0 for the term (under `"sanitize"` or
`"warn"`).

A modifier runs before the clip, so keeping an infinity an infinity is the modifier's job: a NaN made there is
one no clip can bound. The shipped `ExponentialMovingAverage` is written for that (an infinite input stays
infinite); a filter of your own written with `torch.lerp` meets `inf - inf` and returns NaN.

## Built-in observation terms

Built-in terms are importable from `eden.observations` (`from eden import observations`). The full list, with
every parameter, is in the [Term Catalog](../api/terms/catalog.md#observation-terms) and from
`eden terms --kind observation`. By family:

| Family | Module (`eden/managers/terms/observations/`) | Examples |
|---|---|---|
| Proprioception | `proprio.py` | `DofsPos`, `DofsVel`, `DofsForce`, `DofsControlForce`, `base_lin_vel`, `base_ang_vel`, `base_pos`, `base_quat`, `base_rpy`, `links_pos`, `links_quat` |
| Commands, actions, time | `common.py` | `generated_commands`, `last_action`, `episode_phase`, `MetricObservation` |
| Locomotion | `locomotion.py` | `ProjectedGravity`, `base_height`, `sin_phase`, `cos_phase`, `ScaledCommands`, `GridScanDots`, `VelocityTrackingErrorIntegral` |
| Contact | `contact.py` | `ContactForceNorm`, `GroupedContactFlags`, `motion_contact_flags` |
| Manipulation | `manipulation.py` | `EndEffectorPose`, `EndEffectorToObjectDistance`, `ObjectToOracleDistance`, `GripperState` |
| Privileged (domain-randomization state) | `privileged.py` | `GainScale`, `MotorStrengthScale`, `AddedMass`, `ComOffset`, `FootFriction` |
| Motion tracking | `tracking.py` | `motion_dofs_pos`, `motion_future_dofs_pos`, `motion_links_pos_b`, `robot_body_pos_b`, `motion_phase` |
| Cameras | `camera.py` | `rgb_image`, `depth_image`, `segmentation_image`, `normal_image`, `pointcloud_image` |
| Sensors | `sensors.py` | `SensorRead` |

`DofsPos` and the other DOF terms read every DOF of the entity by default, in the entity's `dofs_name` order;
pass `dofs_name=[...]` to select joints in an output order of your choosing. `DofsPos(offset_from_default=True)`
subtracts `default_dofs_pos`. The base-pose terms read the solver frame by default; `relative=True` gives the
authored frame — see [Entities](entities.md) for why the two differ.

The motion-tracking terms need a motion reference; they are covered in [Motion tracking](motion-tracking.md).

## Sensors and cameras as observations

`SensorRead` concatenates sensor readings into one observation: `sensor_names` (a list of names or patterns;
`None`, the default, reads every sensor in the scene), `read_ground_truth=False` (read the noiseless value
instead of the sensor's noisy reading) and `post_process_func` (applied to each reading). Readings with more than
two dimensions are flattened.

The camera terms read a camera by `camera_name`. `rgb_image` returns the renderer's native `(N, H, W, C)` uint8
by default; for a CNN policy, `normalize=True` gives floats in `[0, 1]` and `channels_first=True` gives
`(N, C, H, W)` (either flag also drops an alpha channel). `pointcloud_image` zeroes invalid pixels so an infinite
depth never reaches the policy. Image terms are usually put in a group of their own with
`concatenate_terms=False`, or in a group no flat-vector policy reads. Cameras, renderers and sensors themselves
are covered in [Sensors and rendering](sensors-and-rendering.md).

## Shared computation across groups

Many tasks reuse one term in several groups — an identical `policy` and `critic` copy that differ only in noise
or history. The manager detects terms whose raw output is provably identical (same registered term, same
parameters, ignoring the post-compute fields) and computes them **once** per observation pass, reusing the
result for the other copies. This is only sound because `compute()` is a pure read; it is one more reason
per-step state must live in [`post_physics_refresh()`](terms.md#compute-is-a-read-not-a-tick), which reaches every
instance, including the duplicates the cache skips.

To share a value between an observation and a reward or termination, compute it once as a metric and read it
with `observations.MetricObservation` — see
[Share a computation across terms](terms.md#share-a-computation-across-terms-with-a-metricterm).

## Writing an observation term

A **function term** is the right form for a pure read of env state:

```python
import torch

from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY


@OBSERVATION_TERM_REGISTRY.register()
def episode_phase(env, *, centered: bool = False) -> torch.Tensor:
    """Normalized episode time as a (num_envs, 1) tensor."""
    phase = env.episode_length_buf.float() / env.max_episode_length
    if centered:
        phase = 2.0 * phase - 1.0
    return phase.unsqueeze(-1)
```

That is the built-in `episode_phase` (`eden/managers/terms/observations/common.py`). Return a
`(num_envs, dim)` tensor — `unsqueeze(-1)` a scalar per env — so the term has a feature dimension to scale and
concatenate.

A **class term** (`ObservationTerm`) is the form for anything resolved once or carried between steps. The rules
specific to observations:

- `compute()` runs several times a step and at build, before any reset; it must change nothing a second call
  would see. Per-step state advances in `post_physics_refresh()` (phase 7, once per control step, also on
  hardware). `VelocityTrackingErrorIntegral` is the worked example, in
  [`compute()` is a read](terms.md#compute-is-a-read-not-a-tick).
- The output shape is learned from the build-time call, so `compute()` must already return the right shape then.
- Resolve the entity with `self.resolve_entity(name)` in `compute()`: deployment swaps the sim entity for the
  real robot's state, and observations are exactly what must read the real robot.

## Writing a modifier

Subclass `ObservationModifier` and register it with `@OBSERVATION_MODIFIER_REGISTRY.register()` (both from
`eden.managers.modifiers`). A modifier follows the same read/commit split as an observation term:

- `compute(frame)` is a **read**. It returns this frame's output given the modifier's state, and it runs as
  often as the observation is computed (the snapshot, `get_observations()`, the build probe). It must not change
  the state.
- `advance(frame)` is the **commit**. The manager calls it right after `compute(frame)` on the
  `update_history=True` pass — once per control step and once per `env.reset()`. Stateless modifiers leave it as
  the no-op it is.
- `compute` returns a tensor it does not write afterwards: the manager uses the output after `advance` has run,
  so a view of state that `advance` updates in place would change under it. Return a fresh tensor, or `.clone()`.
- Neither method writes into `frame`, and `advance` copies what it keeps (`self._last.copy_(frame)`, not
  `self._last = frame`): `frame` may be the cross-group cached output, the term's own buffer, or a view of
  simulator state.
- `build(frame)` allocates state from a sample frame (size it with `torch.zeros_like(frame)`; don't seed it from
  the values, which are read before any reset). `reset(envs_idx)` restarts the rows of envs whose episode
  restarts, with the same indices as the history window.
- Return the shape, dtype and device you receive. The manager checks this at build, because scale, history and
  concatenation are sized from the raw output.

```python
import torch

from eden.managers.modifiers import OBSERVATION_MODIFIER_REGISTRY, ObservationModifier


@OBSERVATION_MODIFIER_REGISTRY.register()
class RunningMax(ObservationModifier):
    """The largest value each component has reached since the episode began."""

    def build(self, frame):
        self._max = torch.full_like(frame, -torch.inf)  # allocated once, written in place

    def reset(self, envs_idx=None):
        self._max[slice(None) if envs_idx is None else envs_idx] = -torch.inf

    def compute(self, frame):  # a read, however often it runs
        return torch.maximum(self._max, frame)

    def advance(self, frame):  # once per control step
        torch.maximum(self._max, frame, out=self._max)
```

A modifier inherits one quirk from the history window: a read made *after* the step's push and before the next
step (`get_observations()` between steps) applies the current frame a second time. With backfill right after a
reset — where rsl_rl makes its one such call — that gives the same value; otherwise it runs one push ahead.
Nothing in the training loop reads there. `ExponentialMovingAverage` and `ObservationDelay` in
`eden/managers/modifiers/observations/temporal.py` are the shipped examples.

## See also

- [Writing terms](terms.md) — the term contract, `compute()` as a read, per-step hooks
- [Sensors and rendering](sensors-and-rendering.md) — cameras, renderers and sensors
- [Motion tracking](motion-tracking.md) — the motion-reference observations
- [Training runs](training-runs.md) — `obs_groups`, observation normalization
- [Term Catalog: observation terms](../api/terms/catalog.md#observation-terms), the
  [observation term API](../api/terms/observations.md) and the [modifier API](../api/terms/modifiers.md)
