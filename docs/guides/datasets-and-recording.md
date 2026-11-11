# Datasets and recording

Any Eden env can record itself: you list recorder terms in the config, start recording, and every finished
episode is written to an HDF5 (or NPZ) file. Teleoperation sessions, scripted demonstrations (`eden datagen`) and
policy rollouts all produce the same layout, and `eden lerobot convert` turns it into a LeRobot dataset. This page
is the how-to for demonstration data. The recorder manager itself — the term hooks and how to write a recorder
term — is covered in [Metrics and recorders](metrics-and-recorders.md).

## Record a session

Add `recorder_options` to the env, choose the HDF5 file handler, and bracket the part you want with
`start_recording()` / `end_recording()`:

```python
from eden.constants import DatasetExportMode
from eden.managers.terms.recorders import ActionRecorder, InitialStateRecorder, ObservationRecorder, StateRecorder
from eden.options import RecorderManagerOptions
from eden.utils.file_handler import HDF5FileHandler

recorder_options = RecorderManagerOptions(
    file_handler_options=HDF5FileHandler.configure(),
    dataset_export_dir_path="data/teleop",
    dataset_filename="franka_pick",                 # -> data/teleop/franka_pick.hdf5
    dataset_export_mode=DatasetExportMode.EXPORT_ALL,
    observations=ObservationRecorder.configure(),   # obs/<group>, needs observation groups
    actions=ActionRecorder.configure(),             # action
    state=StateRecorder.configure(entity_name="robot"),
    initial_state=InitialStateRecorder.configure(),
)
```

Pass it as `recorder_options=` to the env constructor (or set `config.recorder_options` on a task config), then:

```python
recorder = env.recorder_manager
recorder.start_recording(reset_scene=True)   # reset first, so the episode starts from a fresh state
...                                          # step the env from your device or policy
recorder.end_recording()                     # exports the episodes in progress
recorder.close()                             # flushes and closes the file
```

Each key under `RecorderManagerOptions` other than the manager's own fields is one recorder term; the key name is
yours. `examples/recipes/teleop/keyboard/teleoperation_keyboard_viser_recorder.py` is a complete teleop session
with recording and replay of the last demo.

!!! warning "Nothing is recorded until `start_recording()`"
    A configured recorder manager stays idle until you call `start_recording()`; `eden train` and
    `eden inference` never call it. Without any recorder term, `start_recording()` only logs a warning.

## Recorder terms

The built-in terms live in [`eden.managers.terms.recorders`](../api/terms/recorders.md). Each writes one key (or a group of keys) per episode.

| Term | Writes | When | Notes |
| --- | --- | --- | --- |
| `ActionRecorder` | `action` | start of each step | The raw action vector the action manager received (every term's slice, concatenated). |
| `ObservationRecorder` | `obs/<group>` | start of each step | The observations the step's action was chosen from, so row `t` of `obs` and `action` is one `(o_t, a_t)` pair. `groups=` limits it; an unknown group raises at build. |
| `StateRecorder` | `state/qpos`, `state/dofs_vel`, `state/pos`, `state/quat` | start of each step | One entity (`entity_name="robot"`) or several, concatenated in order along the last axis. `pos` / `quat` are the root pose (wxyz). `include_qpos`, `include_dofs_vel`, `include_pos_quat` switch parts off. |
| `InitialStateRecorder` | `initial_state/entities/<name>/{qpos,dofs_vel,pos,quat}` | after each reset | Every entity's state at the start of the episode, for replay. |
| `RLStepRecorder` | `reward`, `done`, `terminated`, `timeout`, `timestamp` | end of each step | Reads `RLEnvBase` buffers, so it records nothing on a plain `EnvBase`. `timestamp` is `episode_length_buf * dt`. |
| `SensorRecorder` | `sensor/<name>` | start of each step (and after reset) | Every sensor's `read()`; `use_ground_truth=True` reads `read_ground_truth()`. |
| `SysIDRecorder` | `sysid/...` | reset, start and end of each step | The trace `Trajectory.from_recorder_episode` reloads for system identification; see [Planning and system identification](planning-and-sysid.md#system-identification). |

To record something else, write a `RecorderTerm` subclass — see [Metrics and recorders](metrics-and-recorders.md).

## Where episodes go

`RecorderManagerOptions` fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `file_handler_options` | `NPZFileHandler.configure()` | The file format. Pass `HDF5FileHandler.configure()` for HDF5. |
| `dataset_export_dir_path` | `<tempdir>/eden/logs` (`/tmp/eden/logs` on Linux and macOS) | Output directory. |
| `dataset_filename` | `"dataset"` | File name without extension; the handler adds `.hdf5` or `.npz`. |
| `dataset_export_mode` | `DatasetExportMode.EXPORT_ALL` | Which episodes are written (below). |
| `override` | `False` | `False` appends to an existing file (logged as "Resuming recorder dataset"); `True` replaces it. |

`DatasetExportMode` (`eden.constants`):

- `EXPORT_ALL` — every episode, in one file.
- `EXPORT_SUCCEEDED_ONLY` — only episodes labelled successful.
- `EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES` — successes to `<filename>`, the rest to `<filename>_failed`.
- `EXPORT_NONE` — nothing.

!!! tip "Prefer HDF5 for anything large"
    NPZ has no incremental write: every flush rewrites the whole archive (atomically, through a temporary file),
    and the manager flushes after each exported episode. `eden datagen` and `eden lerobot convert` use HDF5;
    `Trajectory.from_recorder_episode` (system identification) reads NPZ.

Appending to an existing file warns when the stored `env_cfg` differs from the current env's config: the episodes
would mix configurations.

### When an episode ends, and its success label

The recorder closes and exports an episode whenever its env resets:

- On an `RLEnvBase`, an env that terminates or times out is reset inside `step()`, and its episode is exported with
  `success` set from the metric manager's `success_buf` (the task's metric terms with `contributes_to_success`).
- A manual `env.reset()` or `env.reset(envs_idx)` exports the episodes of the envs it resets.
- `end_recording()` exports the episodes still running, cut short (`export_unfinished=False` drops them instead).

A plain `EnvBase` has no terminations, so its episodes end only on a manual reset or `end_recording()`. Label them
yourself with `recorder.set_episode_success(envs_idx, success_values)` before the reset.

!!! warning "Unlabelled episodes count as failed"
    An episode with no success label is exported under `EXPORT_ALL`, but `EXPORT_SUCCEEDED_ONLY` drops it and the
    separate-files mode writes it to `<filename>_failed`. A task without a success metric needs
    `set_episode_success` for those modes.

## File layout

HDF5 (`HDF5FileHandler`):

```text
<filename>.hdf5
└── data/                          attrs: env_cfg (JSON of the env config), total (episode count)
    ├── demo_0/                    attrs: env_id, success (when labelled)
    │   ├── action                 (T, action_dim)
    │   ├── obs/<group>            (T, ...)            ObservationRecorder
    │   ├── state/qpos             (T, n_qs)           StateRecorder
    │   ├── state/dofs_vel, state/pos, state/quat
    │   ├── reward, done, terminated, timeout, timestamp   RLStepRecorder, (T, 1)
    │   └── initial_state/entities/<name>/...           InitialStateRecorder
    ├── demo_1/
    └── ...
```

Nested dicts become groups; every array is gzip-compressed. Demos are numbered in the order they finish, across
envs, so `demo_N` says nothing about which env produced it — read the `env_id` attribute. A resumed file continues
after its highest `demo_N`.

NPZ (`NPZFileHandler`) stores the same tree as flat, slash-separated keys (`demo_0/state/qpos`), with
`__meta__env_cfg` / `__meta__total` for the file and `demo_0/__attrs__` (JSON) for an episode's attributes.

### Reading a dataset back

With `h5py`:

```python
import h5py

with h5py.File("data/teleop/franka_pick.hdf5", "r") as f:
    for name, demo in f["data"].items():
        actions = demo["action"][:]                 # (T, action_dim)
        success = bool(demo.attrs.get("success", False))
```

Inside a running session, `env.recorder_manager.load_episode("demo_0")` returns an `EpisodeData` whose `.data` is
the nested dict of tensors on the env's device. It refuses before a recording session has opened the file, rather
than creating an empty one. The keyboard recorder recipe replays a demo this way: it writes the episode's
`initial_state` back into the entities, then steps the recorded actions.

## Convert to LeRobot

```bash
pip install -U "lerobot[dataset]>=0.4"   # 0.4+ writes LeRobot v3; bare lerobot leaves out its dataset code; no Eden extra
eden lerobot convert \
    --hdf5 data/demos/robosuite_stack.hdf5 \
    --output-dir lerobot/robosuite_stack \
    --repo-id local/robosuite_stack \
    --robot-type panda \
    --task "stack the red cube on the green cube"
```

Flags: `--hdf5` and `--output-dir` (both required), `--repo-id` (default `local/eden_teleoperation`),
`--robot-type` (default `eden`), `--task` (the task string stored on every frame; default
`teleoperate the robot`), `--image-writer-threads` (default `4`) and `--force`. The output is a LeRobot v3 dataset
rooted at `--output-dir`. The same function is importable: `from eden.utils.lerobot import
convert_hdf5_to_lerobot` takes an argv-style list.

What the converter reads and writes:

- Every `demo_<n>` group under `/data`, in numeric order. Each one needs `action`, `timestamp`, `state/qpos`,
  `state/dofs_vel`, `state/pos` and `state/quat` — that is, **`ActionRecorder`, `StateRecorder` and
  `RLStepRecorder`** (for `timestamp`, so the env must be an `RLEnvBase`). `eden datagen` records all three.
- `observation.state` is `qpos ‖ dofs_vel ‖ pos ‖ quat`, with feature names `qpos_i`, `dofs_vel_i`,
  `root_pos_{x,y,z}` and `root_quat_{w,x,y,z}` (`root0_…`, `root1_…` when `StateRecorder` recorded several
  entities). `action` is the recorded action vector.
- The frame rate is inferred from the first demo's timestamps.
- Every demo must have the same feature dimensions; a mismatch raises instead of padding.

!!! note "Not converted"
    Images, `obs/*` groups and the reward/done keys are not carried over: the LeRobot dataset holds
    `observation.state`, `action` and the task string.

`--force` replaces an existing output directory only if it is empty or already a LeRobot dataset (it has
`meta/info.json`), so a mistyped path cannot delete an unrelated directory.

## Scripted and teleoperated data together

`eden datagen` records with exactly the recorder set above (`ObservationRecorder`, `ActionRecorder`,
`RLStepRecorder`, `StateRecorder` on the task's arm or arms, `InitialStateRecorder`) into HDF5 — see
[Scripted demonstrations](scripted-demos.md). A teleoperated recording made with the same terms on the same task
converts the same way, so the two can feed one training set.

## See also

- [Metrics and recorders](metrics-and-recorders.md) — the recorder manager, its hooks, and writing a recorder term.
- [Teleoperation](teleoperation.md) — the input devices.
- [Scripted demonstrations](scripted-demos.md) — `eden datagen`.
- [Planning and system identification](planning-and-sysid.md) — recording traces for sysid.
