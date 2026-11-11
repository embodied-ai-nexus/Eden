# Metrics and recorders

The metric manager measures how well each env is doing: it computes named quantities once per step, decides
whether the task succeeded, and logs per-episode means such as `Metrics/Success`. The recorder manager writes
rollouts to disk: actions, observations, states and RL buffers, one episode per file entry. This page covers
both, for task authors who need a success signal or a quantity shared by several terms, and for anyone
recording demonstrations or system-identification data.

Every environment class supports both managers, configured through `metric_options` and `recorder_options`
(fields of `EdenConfig`); a task that declares no metric terms has `env.metric_manager is None`.

## Metrics

### A first example

A pick-and-place task declares success as "the source object rests on the target"
(`eden/tasks/simpler/bridge_table/config.py`):

```python
from eden.managers.terms.metrics.proximity import object_a_is_on_b
from eden.options.managers.metrics import MetricManagerOptions
from eden.terms import MetricTerm

metric_options = MetricManagerOptions(
    placement=MetricTerm.configure(
        func=object_a_is_on_b,
        params={"entity_a_name": "source", "entity_b_name": "target"},
    ),
)
```

`object_a_is_on_b` returns 1.0 or 0.0 per env. With the defaults below (higher is better, threshold 1.0, judged
every step), an episode counts as a success when the object is on the target on its last step. The env logs
`Metrics/placement`, `Metrics/Success` and `Metrics/Progress` for every finished episode, and the
`on_success` termination can end the episode as soon as it happens.

A task with no metric terms has `env.metric_manager is None`.

### Fields every metric term has

| Field | Type | Default | Meaning |
|---|---|---|---|
| `success_threshold` | `float` | `1.0` | The success threshold, inclusive. |
| `direction` | `MetricDirection` or `"hib"` / `"lib"` | `"hib"` | Higher is better (`value >= threshold` succeeds) or lower is better (`value <= threshold`). |
| `metric_mode` | `MetricMode` or `"interval"` / `"reset"` | `"interval"` | When success is judged: every step, or once on the episode's value at reset. |
| `is_cumulative` | `bool` | `False` | Sum the value over the episode. Success checks and logs use the sum divided by the episode length (the per-step mean). |
| `contributes_to_success` | `bool` | `True` | Whether the term counts toward the task's success at all. |

The metric manager reads these five once, when it is built; a later write to one (from a curriculum, say) has
no effect. Set them through `configure()`.

A function term can carry its own defaults for them at registration, so every use of it starts right:

```python
from eden.constants import MetricDirection, MetricMode
from eden.managers import METRIC_TERM_REGISTRY


@METRIC_TERM_REGISTRY.register(is_cumulative=True, direction=MetricDirection.LIB, metric_mode=MetricMode.RESET)
def error_vel_xy(env, *, command_name: str = "velocity"):
    ...
```

That is how `error_vel_xy`, `error_vel_yaw` (`eden/managers/terms/metrics/error.py`) and the distance metrics
(`entity_distance`, `entity_distance_xy`: lower is better) are registered. A value passed to `configure()`
still wins.

### How success is decided

Each step (phase 6, after physics and before terminations and rewards), the manager computes every term, then:

- **`success_buf`** is `True` for an env when every INTERVAL term that contributes to success succeeds on this
  step's value. RESET-mode terms never enter it, and with no contributing INTERVAL term it is never `True`.
  The `on_success` termination, the `TaskSuccess` reward and the recorder's per-episode success label all read
  it.

When an env's episode ends (phase 13), the manager judges the episode:

- each INTERVAL term counts if it succeeded on the episode's **last step**;
- each RESET-mode term counts if its episode value succeeds (the last step's value, or the per-step mean for a
  cumulative term);
- `Metrics/Progress` is the fraction of contributing terms that counted, and `Metrics/Success` is 1 when all of
  them did;
- `Metrics/<name>` is each term's episode value.

The `Metrics/*` values are means over the envs that ended an episode in that step, returned in the step's
`extras["log"]` (the RSL-RL runner logs them every step). An env whose physics Genesis flagged as corrupt in that
step (a NaN contact, a diverged solve) has its metric values zeroed and is left out of every success check and
every mean.

!!! warning "`metric_mode="reset"` does not exclude a term from `Metrics/Success`"
    A RESET-mode term is still judged, only at the end of the episode instead of every step, and
    `Metrics/Success` requires it. For a metric that exists for monitoring or to feed other terms and has no
    notion of success, set `contributes_to_success=False`: it is then left out of `success_buf`,
    `Metrics/Success`, `Metrics/Progress` and the recorder's success label, and its threshold and direction are
    unused.

!!! warning "A diagnostic error metric left at the defaults pins `Metrics/Success` at 0"
    The defaults are higher-is-better with threshold 1.0. A tracking error of a few centimetres never reaches
    1.0, so it fails every episode and drags `Metrics/Success` to zero with it. Give such a term
    `direction="lib"` and a meaningful threshold, or `contributes_to_success=False`.

`eden inference` prints the mean of every `Metrics/*` key over the finished episodes when the rollout ends,
reading each env's last episode from `metric_manager.last_episode()`. With `--episodes N` it counts each env's
first `ceil(N / num_envs)` episodes and runs until every env has finished them.

### Using a metric from other terms

Compute a quantity once as a metric and pull it by name wherever it is needed, instead of recomputing it in each
term or storing it on `env`. The metric manager runs at phase 6, before the observation tick (7), terminations (8)
and rewards (9), so what they pull is this step's value.

| Term | Kind | Reads |
|---|---|---|
| `rewards.MetricReward(metric_name=...)` | reward | The metric's value, times the reward weight. Rejects a cumulative metric unless `allow_cumulative=True`. |
| `observations.MetricObservation(metric_name=...)` | observation | The value as a `(num_envs, 1)` observation. |
| `terminations.OnMetricSuccess(metric_name=...)` | termination | The metric's own success check on this step (INTERVAL, success-contributing metrics only). |
| `terminations.MetricAbove(metric_name=..., threshold=...)` | termination | `value > threshold`; accepts a `contributes_to_success=False` metric, such as a collision flag. |
| `terminations.on_success` | termination | `success_buf`: every success metric holds this step. |
| `rewards.TaskSuccess` | reward | 1 on a step where `success_buf` holds; raises at build when there is no INTERVAL success metric. |

```python
from eden import observations, rewards
from eden.options import ObservationGroupOptions
from eden.options.managers.rewards import RewardManagerOptions

reward_options = RewardManagerOptions(
    score=rewards.MetricReward.configure(metric_name="my_score", weight=1.0),
)
policy = ObservationGroupOptions(
    score=observations.MetricObservation.configure(metric_name="my_score"),
)
```

All of these resolve the metric in `build()`, so a missing metric manager or an unknown name raises at env
construction with the active metrics listed. In your own terms, the accessors are:

- `env.metric_manager.get_value(name)` — the per-env value buffer, updated in place every step (hold on to it if
  you like; treat it as read-only). For a cumulative metric it is the running episode sum.
- `is_cumulative(name)`, `is_current_step_success(name)`, `success_buf`, `progress_buf`, `num_step_success_terms`.
- `last_episode()` — each env's last finished episode as `Metrics/*` values plus a healthy mask, for evaluations
  that pick which episodes to count.

For an intermediate shared by several terms that is not itself a metric, use `env.compute_once(key, producer)`;
see [Terms](terms.md).

### Built-in metric terms

All built-ins live in `eden.managers.terms.metrics` (also `eden.metrics`). `eden terms --kind metric` prints them
with every parameter, and the generated [term catalog](../api/terms/catalog.md) has the same list.

| Group | Terms |
|---|---|
| Proximity | `entity_distance`, `entity_distance_xy` (lower is better), `entity_near_target`, `entity_near_entity_xy`, `EeNearEntity` (mean end-effector position within `threshold` of an entity), `object_a_is_on_b`, `object_a_within_b_xy` |
| Containment | `object_above_height`, `object_bottom_above_height`, `object_in_region` |
| Grasp | `object_lifted`, `IsGrasping` (parallel-jaw grasp from contact-force alignment) |
| Articulation | `joint_angle_at_target` |
| Velocity tracking | `error_vel_xy`, `error_vel_yaw` (cumulative, lower is better, judged at reset; read a velocity command term) |
| Composition | `MetricTermAND`, `MetricTermOR`, `MetricTermNOT`, `SequentialMetricTerm`, `RubricMetricTerm` |

The boolean combinators work on the child terms' raw values: AND is the element-wise minimum, OR the maximum,
NOT is `1.0 - value`. The children's own thresholds and directions are not consulted, so combine predicates that
already return 0 or 1.

### Ordered sub-goals

`SequentialMetricTerm` tracks phases that must be completed in order, per env; its value is the fraction of phases
completed. Each `PhaseOptions` gives either an inline `term` or a `ref` to a sibling metric by its key, a
`threshold` (default 1.0) the phase's value must reach, and optionally a hold: with `hold=True` the phase's own
predicate (or `hold_term`) must keep holding while later phases run, until the phase named by `hold_until`, or the
env reverts to it. `hold_guard` suppresses a hold while another phase's hold still passes (lowering an object
while still grasping it is not a failure). `allow_skip=True` lets an env advance through several phases in one
step. From `examples/recipes/teleop/keyboard/teleoperation_keyboard_metrics.py`, trimmed:

```python
from eden.managers.terms.metrics import IsGrasping, SequentialMetricTerm, object_a_is_on_b
from eden.options.managers.metrics import MetricManagerOptions, PhaseOptions
from eden.terms import MetricTerm

metric_options = MetricManagerOptions(
    is_grasping=IsGrasping.configure(
        robot_name="robot",
        left_gripper_link_name="left_gripper_pad",
        right_gripper_link_name="right_gripper_pad",
        contributes_to_success=False,
    ),
    task_progress=SequentialMetricTerm.configure(
        phases=[
            PhaseOptions(name="grasp", ref="is_grasping", hold=True, hold_until="place"),
            PhaseOptions(
                name="place",
                term=MetricTerm.configure(
                    func=object_a_is_on_b, params={"entity_a_name": "apple", "entity_b_name": "plate"}
                ),
            ),
        ],
    ),
)
```

(The example script marks its helper metrics `metric_mode=MetricMode.RESET` instead; per the warning above,
`contributes_to_success=False` is what keeps them out of `Metrics/Success`.)

`RubricMetricTerm` scores a list of `CriterionOptions(name=..., term=..., after=[...])`. A criterion is reached the
first step its term succeeds (by the term's own direction and threshold) while every criterion in its `after` list
has been reached, and it stays reached until the env resets. Criteria without a dependency between them are
reached in any order. The value is the fraction reached. `eden/tasks/polaris` uses it.

### Writing a metric term

A function metric takes the env and keyword parameters and returns one value per env:

```python
import torch

from eden.managers import METRIC_TERM_REGISTRY


@METRIC_TERM_REGISTRY.register()
def entity_speed(env, *, entity_name: str) -> torch.Tensor:
    """Linear speed of an entity's root, in m/s."""
    return torch.linalg.vector_norm(env.get_entity(entity_name).get_vel(), dim=-1)
```

```python
from eden.terms import MetricTerm

cup_at_rest = MetricTerm.configure(
    func=entity_speed, params={"entity_name": "cup"}, direction="lib", success_threshold=0.01
)
```

The value is cast to float. For state across steps, a `build()` to resolve names, or a custom `reset()`, subclass
`eden.managers.MetricTerm` and implement `compute(self)`; `IsGrasping` in
`eden/managers/terms/metrics/grasp.py` is a short example. A class term may also override
`is_success(value, envs_idx=None)` for a custom success rule. The general rules (resolve names in `build()`, no
state on `env`, no GPU→CPU syncs) are in [Terms](terms.md).

## Recorders

### Recording a rollout

The recorder manager is idle until you start it. Configure the recorder terms you want, then bracket the steps to
record (`eden/extensions/scripted_demos/generate.py` does exactly this):

```python
from eden.constants import DatasetExportMode
from eden.managers.terms.recorders import (
    ActionRecorder,
    InitialStateRecorder,
    ObservationRecorder,
    RLStepRecorder,
    StateRecorder,
)
from eden.options.managers.recorders import RecorderManagerOptions
from eden.utils.file_handler.hdf5_file_handler import HDF5FileHandler

config.recorder_options = RecorderManagerOptions(
    file_handler_options=HDF5FileHandler.configure(),
    dataset_export_dir_path="datasets",
    dataset_filename="demos",
    dataset_export_mode=DatasetExportMode.EXPORT_ALL,
    override=True,
    observations=ObservationRecorder.configure(),
    actions=ActionRecorder.configure(),
    rl_step=RLStepRecorder.configure(),
    state=StateRecorder.configure(entity_name="robot"),
    initial_state=InitialStateRecorder.configure(),
)

# ... build and reset the env ...
recorder = env.recorder_manager
recorder.start_recording()
for _ in range(num_steps):
    env.step(policy(obs))
recorder.end_recording()
recorder.close()
```

- `start_recording(reset_scene=False)` opens the dataset file and starts an episode per env from the current
  state; with `reset_scene=True` it resets the env first. It does nothing (with a warning) when no recorder term
  is configured.
- Every episode that ends while recording is exported when it ends.
- `end_recording(export_unfinished=True)` stops recording and, by default, exports the episodes still in progress
  as they stand. Pass `export_unfinished=False` to drop them when only complete episodes belong in the dataset.
- `close()` ends a running recording and closes the file.

`eden datagen`, the teleoperation tools and `eden lerobot convert` build on this; see
[Scripted demonstrations](scripted-demos.md), [Teleoperation](teleoperation.md) and
[Datasets and recording](datasets-and-recording.md) for the HDF5 layout and conversion.

### Recorder manager options

| Field | Default | Meaning |
|---|---|---|
| `file_handler_options` | `NPZFileHandler.configure()` | The file format: `NPZFileHandler` (`.npz`) or `HDF5FileHandler` (`.hdf5`), both in `eden.utils.file_handler`. |
| `dataset_export_dir_path` | `<system temp dir>/eden/logs` | Directory the dataset is written to. |
| `dataset_filename` | `"dataset"` | File name without extension. |
| `dataset_export_mode` | `DatasetExportMode.EXPORT_ALL` | Which episodes are written (below). |
| `override` | `False` | Replace an existing file instead of appending to it. |
| `<term name>` | — | Any number of recorder terms. |

`DatasetExportMode` (`eden.constants`): `EXPORT_ALL` writes every episode to one file; `EXPORT_SUCCEEDED_ONLY`
writes successful ones and drops the rest; `EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES` writes failures to
`<dataset_filename>_failed`; `EXPORT_NONE` writes nothing.

!!! warning "An existing dataset is appended to by default"
    With `override=False`, a file already at `dataset_export_dir_path/dataset_filename` is opened in append mode
    and new episodes are added after the ones it holds. Set `override=True` for a fresh file, and change the
    default directory: the system temp directory may not survive a reboot.

An episode's success label is `metric_manager.success_buf` on its last step, set by `RLEnvBase.step`. It therefore
reflects only the INTERVAL metrics that contribute to success; a task whose success metrics are all RESET-mode records every
episode as unsuccessful, an env with no metric manager records no label (treated as a failure), and
`EXPORT_SUCCEEDED_ONLY` then writes nothing.

### Built-in recorder terms

All built-ins live in `eden.managers.terms.recorders` (also `eden.recorders`).

| Term | Keys written | When |
|---|---|---|
| `ActionRecorder` | `action` | every step, at its start |
| `ObservationRecorder(groups=None)` | `obs/<group>` (every group, or the listed ones) | every step, at its start |
| `StateRecorder(entity_name="robot")` | `state/qpos`, `state/dofs_vel`, `state/pos`, `state/quat` (several entities are concatenated along the last axis) | every step, at its start |
| `RLStepRecorder` | `reward`, `done`, `terminated`, `timeout`, `timestamp` (each optional) | every step, after rewards and terminations |
| `SensorRecorder` | `sensor/<sensor name>` (`use_ground_truth=True` reads ground truth) | every step, at its start, and once per new episode |
| `InitialStateRecorder` | `initial_state/entities/<entity>/{qpos,dofs_vel,pos,quat}` for every entity | once per new episode |
| `SysIDRecorder(entity_name="robot")` | `sysid/...`: initial state, action and time stamp, joint positions, velocities and torques, base orientation | per episode and every step |

Row `t` of `action` and of `obs/<group>` form one `(o_t, a_t)` pair: the observation the action was chosen from
and the action itself, which is the shape a behaviour-cloning dataset needs. `SysIDRecorder` records what
`eden.extensions.sysid.Trajectory.from_recorder_episode` reloads; see [Planning and system identification](planning-and-sysid.md).

### When recorder hooks run

A recorder term implements any of four hooks, each returning a dict of `(num_envs, ...)` tensors (nested dicts
become slash-separated keys). The step positions refer to the table in [Environments](environments.md).

| Hook | Runs | Recorded as |
|---|---|---|
| `record_pre_step()` | phase 2, after the action is processed and before physics | one row of the step |
| `record_post_step()` | phase 11, after terminations and rewards, before done envs are reset | one row of the step |
| `record_pre_reset(envs_idx)` | phase 11 for the envs whose episode ends this step, and at the start of a manual `env.reset()` | part of that last step |
| `record_post_reset(envs_idx)` | after a reset (phase 15, the end of `env.reset()`) and at `start_recording()` | episode-level data |

`record_post_step()` runs before the INTERVAL events (phase 18), so an event that perturbs the robot for the next
step never shows up as the outcome of this step's action.

### Writing a recorder term

```python
import torch

from eden.managers import RECORDER_TERM_REGISTRY, RecorderTerm


@RECORDER_TERM_REGISTRY.register()
class BaseHeightRecorder(RecorderTerm):
    entity_name: str = "robot"

    def build(self) -> None:
        self.resolve_entity(self.entity_name)  # fail at build on a bad name

    def record_post_step(self) -> dict[str, torch.Tensor]:
        pos = self.resolve_entity(self.entity_name).get_pos()
        return {"base_height": pos[:, 2:3].clone()}
```

Return clones, or values you will not modify in place, and leave `compute()` alone: recorder terms are driven only
by the hooks. `close(file_path)` is called when the manager closes, while the dataset file is still open.

## See also

- [Rewards and terminations](rewards-and-terminations.md) — the reward and termination managers that read metrics.
- [Datasets and recording](datasets-and-recording.md) — dataset layout and `eden lerobot convert`.
- [Scripted demonstrations](scripted-demos.md) — `eden datagen` on top of the recorder.
- [Environments](environments.md) — the step order.
- [Terms](terms.md) — writing terms, `compute_once`.
- [Term catalog](../api/terms/catalog.md) — every registered metric and recorder term.
- API reference: [`MetricManager` and `RecorderManager`](../api/managers.md), [metric terms](../api/terms/metrics.md), [recorder terms](../api/terms/recorders.md), [options](../api/options/managers.md).
