# Task Configuration

A task in Eden is one Pydantic config class. It declares the scene, the managers and their terms, the
simulation settings and, for RL tasks, the training runner. This page covers the config classes, every
option group they hold, the task registry that finds them by name, the layout of a task package, and
how configs are validated, derived from one another, saved and loaded. Read it before you write or
change a task.

!!! tip "Reference template"
    `eden/tasks/benchmark/reacher/config.py` is the canonical task config. Copy its shape when you start
    a new task.

## A task at a glance

An abridged excerpt of `ReacherConfig` from `eden/tasks/benchmark/reacher/config.py`. Each field is an
option group, and the decorator registers the class as the task `reacher`:

```python
import eden as en
from eden import actions, rewards, terminations
from eden.constants import ReferenceSource
from eden.managers.terms.commands.oracle import UniformT3Command
from eden.options import (
    ActionManagerOptions,
    CommandManagerOptions,
    EnvOptions,
    RewardManagerOptions,
    SceneOptions,
    TerminationManagerOptions,
)
from eden.tasks import TASK_REGISTRY
from eden.terms import DoneTerm, RewardTerm
from eden.utils.configs import EdenRLConfig


@TASK_REGISTRY.register(modifiers=(ReacherDifficultyMod,))  # key "reacher", one TaskMod
class ReacherConfig(EdenRLConfig):
    env_options: EnvOptions = EnvOptions(num_envs=4096, episode_length_s=9.0, sim_dt=0.01, sim_substeps=2, decimation=1)

    scene_options: SceneOptions = en.scenes.Empty(
        is_support_enabled=True,
        robot=en.robots.Yam(default_root_pos=(0.0, 0.0, 0.0), default_root_quat=(1.0, 0.0, 0.0, 0.0)),  # wxyz
    )

    action_options: ActionManagerOptions = ActionManagerOptions(
        joint_pos_controller=actions.ImplicitPDController.configure(
            entity_name="robot", dofs_name="joint*", scale=0.05, reference_source=ReferenceSource.DELTA
        ),
    )

    command_options: CommandManagerOptions = CommandManagerOptions(
        reaching_target=UniformT3Command.configure(
            x_range=(0.3, 0.5), y_range=(-0.1, 0.1), z_range=(0.25, 0.35), resampling_time_range=(2.0, 3.0)
        )
    )

    reward_options: RewardManagerOptions = RewardManagerOptions(
        action_rate=RewardTerm.configure(func=rewards.action_rate_l2, weight=-0.01),
        joint_pos_limits=RewardTerm.configure(func=rewards.dofs_pos_limits, weight=-10.0, params={"entity_name": "robot"}),
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        timeout=DoneTerm.configure(func=terminations.time_out, time_out=True),
    )
```

The real file also declares the observation groups, a custom reward registered inline, a curriculum and
the RSL-RL runner. `ReacherDifficultyMod` is a [TaskMod](task-mods.md): it adds the `--difficulty` flag.

Every value is a field default. Do not build fields in `__init__` or `model_post_init`;
[Deriving a config from another](#deriving-a-config-from-another) explains why.

## The config hierarchy

Pick the config class whose environment your task needs, and subclass it. Each config class is paired
with exactly one environment class, which
[`env_class_for_config`](environments.md#picking-the-env-class-env_class_for_config) looks up:

| Config (`eden.utils.configs`) | Subclasses | Adds | Environment |
|---|---|---|---|
| `EdenConfig` | — | the base option groups (below) | `EnvBase` |
| `EdenRLConfig` | `EdenConfig` | `reward_options`, `termination_options`, `command_options`, `curriculum_options`, `runner_options` | `RLEnvBase` |
| `EdenMimicConfig` | `EdenRLConfig` | `motion_options` | `MimicRLEnvBase` |
| `EdenSplatConfig` | `EdenConfig` | `splat_options` (`SplatOptions`) | `SplatEnvBase` |
| `EdenSplatRLConfig` | `EdenRLConfig` | `splat_options` (`SplatOptions`) | `SplatRLEnvBase` |
| `EdenNyxSplatConfig` | `EdenConfig` | `splat_options` (`NyxSplatOptions`) | `NyxSplatEnvBase` |

Most tasks subclass `EdenRLConfig`; motion-imitation tasks subclass `EdenMimicConfig`. A plain
`EdenConfig` has no reward or termination, which suits teleoperation, data collection and viewers. The
splat configs render a Gaussian-splat background. Their environment classes need the `batch-splat`
extra (`pip install 'eden-world[batch-splat]'`), or the `nyx` extra for Nyx. `splat_options` is a root
field rather than part of the scene because it selects the environment class.

API reference: [Configs](../api/configs.md).

## Option groups

Every option group is a field whose name ends in `_options`, except `language_instructions`.
`cameras_options` and `sensors_options` are plural because each holds several named items.

| Field | Type | Read by | Guide |
|---|---|---|---|
| `env_options` | `EnvOptions` | the environment and the Genesis scene: env count, timing, solver | [Physics](physics.md) |
| `scene_options` | `SceneOptions` | the entities, by name | [Entities](entities.md), [Scenes and terrains](scenes-and-terrains.md) |
| `observation_options` | `ObservationManagerOptions` | `ObservationManager` (groups of terms) | [Observations](observations.md) |
| `action_options` | `ActionManagerOptions` | `ActionManager` (controllers) | [Actions](actions.md) |
| `event_options` | `EventManagerOptions` | `EventManager` (resets, randomization) | [Events](events.md) |
| `cameras_options` | `CamerasOptions` | cameras, by name | [Sensors and rendering](sensors-and-rendering.md) |
| `sensors_options` | `SensorsOptions` | sensors, by name | [Sensors and rendering](sensors-and-rendering.md) |
| `metric_options` | `MetricManagerOptions` | `MetricManager` (shared computations, success) | [Metrics and recorders](metrics-and-recorders.md) |
| `recorder_options` | `RecorderManagerOptions` | `RecorderManager` (episode recording) | [Metrics and recorders](metrics-and-recorders.md) |
| `renderer_options` | Genesis `RendererOptions` | the Genesis scene's renderer | [Sensors and rendering](sensors-and-rendering.md) |
| `language_instructions` | `list[str]` | not the environment: carried on the config for language-conditioned policies | [Remote policies](remote-policies.md) |
| `reward_options` | `RewardManagerOptions` | `RewardManager` | [Rewards and terminations](rewards-and-terminations.md) |
| `termination_options` | `TerminationManagerOptions` | `TerminationManager` | [Rewards and terminations](rewards-and-terminations.md) |
| `command_options` | `CommandManagerOptions` | `CommandManager` (goals, targets) | [Commands](commands.md) |
| `curriculum_options` | `CurriculumManagerOptions` | `CurriculumManager` | [Curricula](curricula.md) |
| `runner_options` | a runner options class, or `None` | the trainer, not the environment | [Training runs](training-runs.md) |
| `motion_options` | `MotionManagerOptions` | `MotionManager` (motion reference) | [Motion tracking](motion-tracking.md) |
| `splat_options` | `SplatOptions` / `NyxSplatOptions` | the splat renderer | [Scenes and terrains](scenes-and-terrains.md) |

Most groups accept `None`, and the environment substitutes an empty group of the right type. The splat
environments are the exception: a splat config with `splat_options=None` raises `ValueError` when its
environment is constructed.

The manager option groups (`*ManagerOptions`) hold **named terms**: each keyword is a term key, and its
value is a term configured with `.configure(...)`. How to write and configure terms is in
[Terms](terms.md), and `eden terms --grep <text>` lists every registered term with its parameters.

### `env_options`

`EnvOptions` (`eden.options.envs`) holds the simulation settings. These are the fields every task sets:

| Field | Default | Meaning |
|---|---|---|
| `num_envs` | `4096` | Parallel environments. Under multi-GPU training this is per rank. |
| `sim_dt` | `0.005` | Physics time step, in seconds. |
| `sim_substeps` | `1` | Genesis substeps inside one physics step. |
| `decimation` | `4` | Physics steps per control step (the action repeat). |
| `episode_length_s` | `20.0` | Episode length, in seconds. Must be positive and finite. |

The control step is `env.dt = sim_dt * decimation` seconds, so the control rate is
`1 / (sim_dt * decimation)` Hz, and an episode lasts `ceil(episode_length_s / env.dt)` control steps.
The solver, integrator, contact and batching fields, and the options of the other solvers, are in
[Physics](physics.md). API reference: [EnvOptions](../api/options/envs.md).

!!! note "An episode with no time limit"
    `episode_length_s` cannot be infinite: `RLEnvBase` raises on `inf`, `NaN` or a value that is not
    positive. For an episode with no time limit, keep a finite length and leave the `time_out`
    termination term out.

### `scene_options`

A scene is a `SceneOptions` whose extra keywords are entities, keyed by the name that terms use to find
them. The presets in `en.scenes` are factory functions that return a plain `SceneOptions`:

```python
import eden as en
from eden.options import BoxOptions

scene = en.scenes.Empty(
    is_support_enabled=True,
    robot=en.robots.Yam(default_root_pos=(0.0, 0.0, 0.0), default_root_quat=(1.0, 0.0, 0.0, 0.0)),  # wxyz
    cube=BoxOptions(size=(0.05, 0.05, 0.05), default_root_pos=(0.4, 0.0, 0.05)),
)
```

`en.scenes.Empty` adds a ground plane named `plane`. Passing `plane=TerrainOptions(...)` replaces the
flat ground under the same name. `is_support_enabled` and `surface` configure only the default plane, so
passing either one together with `plane=` raises `ValueError`. Entity types, robots and terrains are
covered in [Entities](entities.md), [Robots](robots.md) and [Scenes and terrains](scenes-and-terrains.md).

### `runner_options`

`runner_options` configures the learner; no environment reads it. The base classes are:

- `RslRlOnPolicyRunnerOptions` (PPO) and `RslRlDistillationRunnerOptions`, from
  `eden.options.learning.rsl_rl`. `eden train` uses RSL-RL by default.
- `RlGamesPpoRunnerOptions`, from `eden.options.learning.rl_games`. `eden train` dispatches to rl_games
  when `runner_options` is an instance of it.

Algorithm-specific runners subclass the RSL-RL ones: `RslRlFlashSacRunnerOptions`,
`RslRlRmaDepthDistillationRunnerOptions` and `RslRlContrackRunnerOptions` (listed in
[Where an algorithm lives](training-runs.md#where-an-algorithm-lives)), and `RslRlWeaveRunnerOptions` in
`eden/tasks/weave_hoi/runner.py`.

`runner_options.seed` seeds the run. The training wrappers apply it before they construct the
environment (see [Environments](environments.md#building-an-environment)). Runner fields, custom
algorithms and checkpoints are covered in [Training runs](training-runs.md).

## The task registry

`TASK_REGISTRY` (`eden.tasks`) maps task names to config classes. A task registers itself where it is
defined, so there is no central list to update.

### Registering a task

```python
from eden.tasks import TASK_REGISTRY
from eden.utils.configs import EdenRLConfig


@TASK_REGISTRY.register()  # key "my_reach", derived from the class name
class MyReachConfig(EdenRLConfig):
    ...
```

The key is the class name in snake case with a trailing `_config` removed: `ReacherConfig` registers as
`reacher`, and `PolarisPanCleanConfig` as `polaris_pan_clean`. `register()` takes three keyword
arguments:

- `name=` sets the key explicitly. Use it only when no class name produces the key you want (see
  [Task Package Layout](#task-package-layout)).
- `modifiers=` attaches [TaskMods](task-mods.md), the task's variants and CLI flags.
- `override=True` replaces an existing registration with the same name. It is for a script that
  registers a class at run time, for example
  `TASK_REGISTRY.register(MyReacher, name="reacher", override=True)`. It does not resolve a clash between
  two scanned files (see [Name clashes](#name-clashes-and-broken-entry-points)).

Registering a name that is already taken, without `override=True`, raises `ValueError`. Registering
anything other than a class raises `TypeError`.

### Looking a task up

```python
from eden.tasks import TASK_REGISTRY

TASK_REGISTRY.list_tasks()                         # every task name, sorted
config = TASK_REGISTRY.build("reacher")            # a config instance, with the task's TaskMods applied
config = TASK_REGISTRY.build("reacher", difficulty="hard")
ReacherConfig = TASK_REGISTRY.get("reacher")       # the bare class
```

!!! warning "`get()` returns the class, and instantiating it runs no TaskMod"
    `TASK_REGISTRY.get(name)()` builds the config **without** its TaskMods. For a task whose robot or
    dataset is bound by a mod, that config constructs and saves without complaint and then fails at
    `env.build()`, unless the config overrides `check_bindings()` (see
    [TaskMods](task-mods.md#check_bindings-name-what-is-still-unset)). Use `build()` for a config you
    will run. Use `get()` when you need the class itself: for an `issubclass` check, or to subclass it.

A miss raises `KeyError` with the closest name and every available task. `eden list` prints every task on
the command line, with its source (`eden` for built-in tasks) and its modifiers.

### How discovery works

The first lookup scans `eden/tasks/**/*.py` with Python's `ast` module and records every class
decorated with `@TASK_REGISTRY.register`, **without importing** any task module. Importing `eden.tasks`
imports no task. Only the module that defines the task you ask for is imported, by `get()` or `build()`.
That keeps `eden list` fast and offline, since a task module can pull in heavy dependencies or download
assets.

The scan follows three rules:

- It recognizes `@<name>.register` only where `<name>` is bound in the same file by
  `from eden.tasks import TASK_REGISTRY` or `from eden.tasks.registry import TASK_REGISTRY` (an `as`
  alias works). A file that uses the bare name `TASK_REGISTRY` without that import gets a warning, so
  the miss is not silent.
- It skips `__init__.py` and every file whose name starts with `_`. Put decorated classes in a module
  such as `config.py`.
- A file that cannot be read or parsed is skipped, with a warning that names it.

### Tasks in your own package

A task does not have to live in the Eden tree. Put it in your own installable package, and name the
package that holds the task configs under the `eden.tasks` entry-point group in your `pyproject.toml`:

```toml
[project.entry-points."eden.tasks"]
my_project = "my_project.tasks"   # the key is free-form; the value is the package to scan
```

```python
# my_project/tasks/reach.py: the same decorator and import as a built-in task
from eden.tasks import TASK_REGISTRY
from eden.utils.configs import EdenRLConfig


@TASK_REGISTRY.register()
class MyReachConfig(EdenRLConfig):  # key "my_reach"
    ...
```

After you install the package (`uv pip install -e .`, or any install), `eden list` shows `my_reach` with
your distribution's name in its **Source** column, and both `eden train --task my_reach` and
`TASK_REGISTRY.build("my_reach")` find it. The same scan rules apply: no tasks in `__init__.py` or in
files whose name starts with `_`.

The scan reads your task files without importing them, but locating a dotted package imports the
packages above it. For `my_project.tasks` that is `my_project/__init__.py`, which the first task lookup
of every `eden` process imports, `eden list` included. Keep that file light: no task imports, no heavy
dependencies, no downloads. An entry point that names a top-level package
(`my_project_tasks = "my_project_tasks"`) imports nothing at all.

A script that never goes through the `eden` CLI can add a package itself, before its first lookup. Only
that process sees it, and a package that cannot be found is skipped without an error:

```python
TASK_REGISTRY.add_search_path("my_project.tasks")
```

### Name clashes and broken entry points

Both of these problems raise only from a lookup they could change, so a broken or clashing plugin never
blocks the tasks it has nothing to do with. Each also warns once when the scan runs.

- **Task names are global.** A name registered in two scanned files, in Eden or in a plugin, raises
  `ValueError` naming both files, from every lookup of that name and from `eden list`. Neither class
  registers when its module is imported, so the answer never depends on import order. Adding
  `override=True` does not help, because the scan has no import order to apply it in. Give your task a
  name of its own; to start from a built-in task, subclass its config under a new name.
- **An entry point that does not resolve fails the lookups it could answer.** If the value is not a
  scannable package (a typo, a `module:attr` reference, a plain module, or a parent package that fails
  to import), the scan cannot know which tasks it holds. A lookup that finds nothing, and `eden list`,
  raise `RuntimeError` naming the distribution, the value and the fix. A task found elsewhere still
  resolves. Entry points are re-read on every rescan.

## Task Package Layout

The rules to follow when you add a task to `eden/tasks`. `tests/tasks/test_init.py` checks the naming,
file and `__init__.py` rules across every built-in task.

- **The task name is snake case and starts with its package's name.** A task in `eden/tasks/polaris/`
  is `polaris_pan_clean`, and one in `eden/tasks/robosuite/` is `robosuite_lift`. A package with a single
  task is named for it (`piano`). `benchmark/` is the one exception, because it holds Eden's own
  reference tasks (`reacher`, `move_forward`).
- **The config class is `<Package><Task>Config`, so the key derives from it.** The registry snake-cases
  the class name and strips the suffix (`PolarisPanCleanConfig` → `polaris_pan_clean`), and every
  registered config class ends in `Config`. A saved `config.json` records the class's module and name, so
  renaming or moving a registered config breaks loading the runs saved from it.
- **Pass `name=` only where no class name produces the key.** A `name=` equal to the derived key fails
  the test. The cases in the tree are:
    - a name the package runs together while the class keeps its capitals (`name="beyondmimic"` on
      `BeyondMimicConfig`; likewise `assistmimic`, `flashdexretarget`, `omnireset`);
    - an acronym the snake-casing would split differently (`name="manipulation_lift_ycb_rgb"`);
    - a class name already used elsewhere (`name="homie"` on `HomieLocomotionConfig`).
- **The config lives in a `config.py`.** That is `eden/tasks/<package>/config.py`, or
  `eden/tasks/<suite>/<task>/config.py` once each of a suite's tasks needs a module of its own
  (`robosuite/lift/config.py`, `manipulation/lift_box/config.py`). A suite's shared base config and
  helpers go in other modules at the suite's top level (`polaris/base_config.py`, `robosuite/common.py`,
  `art/common.py`). Tasks that differ from one base by a few fields share one `config.py`
  (`whiteboard/config.py`, `art/calligraphy/config.py`).
- **Terms go where their number suggests.** A few task-specific terms are registered inline in
  `config.py`, as `reaching_reward` is in reacher. More go in a `custom_terms.py`, and many in a
  `custom_terms/` package with one module per manager (`observations.py`, `rewards.py`, ...). A term that
  any task could use belongs in `eden/managers/terms/` instead. See
  [Inline registration](terms.md#inline-registration).
- **A task package's `__init__.py` is its docstring and nothing else.** Importing
  `eden.tasks.<pkg>.custom_terms.rewards` must not build the task's config, and the registry needs no
  import to find a task. A `custom_terms/__init__.py` may re-export its terms.
  `eden/tasks/libero/__init__.py` is the one exception, because it is a documented API surface.
- **Variants are TaskMods.** See [A TaskMod or a separate task?](task-mods.md#a-taskmod-or-a-separate-task)

## Strict fields

`ConfigurableOptions`, the base of every Eden options class, sets Pydantic's `extra="forbid"`. A
misspelled field raises `ValidationError` on the line that contains it, in a task config, `EnvOptions`,
entity, camera, sensor and runner options alike. `EnvOptions(decimatoin=4)` fails at construction, not
at build, and the message names the closest valid field and lists every field the class has.

Classes that hold **dynamically named children** subclass `DynamicConfigurableOptions`, which restores
`extra="allow"`:

| Class | Its extras are |
|---|---|
| `SceneOptions` | named entities (`robot=...`, `cube=...`) |
| `ManagerOptions`: every `*ManagerOptions`, and `RewardGroupOptions` | named terms; named groups in `ObservationManagerOptions`, and optionally in `RewardManagerOptions` |
| `ObservationGroupOptions` | named observation terms |
| `CamerasOptions` / `SensorsOptions` | named cameras / sensors |
| `SplatOptions` / `NyxSplatOptions` | named point clouds (`PCDOptions`) |
| `ManagerTermOptions`, the base of every `*TermOptions` | the parameters `.configure()` passes to the term |
| `CameraOverlayOptions`, `TerrainTermOptions`, `DatasetOptions`, `DatasetModifierOptions`, `FileHandlerOptions`, `MetadataOptions` | `.configure()` parameters |
| extension options: `PolicyOptions`, `ClientOptions`, `RetargeterOptions`, `DeploymentOptions`, `TeleopOptions`, `PlannerOptions`, `PoseEstimatorOptions`, `GraspSamplerOptions`, `SystemIdentificationOptions` | `.configure()` parameters |

`extra="allow"` does not mean unchecked. Term options validate their extras against the registered
term's parameter names when they are constructed, so `RewardTermOptions(name="upright", wieght=2.0)`
raises, and the term checks them again when it is built. Extension options are checked on the
`.configure()` path only.

A term key cannot shadow an attribute of the options object that holds it: a term named `lock` or
`clone` raises `TypeError` and suggests another key.

## Deriving a config from another

### Subclass, and override fields as class defaults

A variant of a task is a subclass that redeclares the fields it changes. To change one part of an
inherited option group, start from the parent's default and call `updated()`:

```python
from eden.options import RewardManagerOptions
from eden.tasks.velocity_tracking.base_config import VelocityTrackingBaseConfig

base = VelocityTrackingBaseConfig.model_fields["reward_options"].default


class MyConfig(VelocityTrackingBaseConfig):
    reward_options: RewardManagerOptions = base.updated(
        track_linear_velocity=base.track_linear_velocity.updated(weight=0.0),
    )
```

`options.updated(field=value)` returns a deep copy with those fields replaced. It raises `ValueError` on
a name the class (or, for a term, the registered term) does not accept, and the copy is always
unlocked, so you can edit it further. `model_copy(update={...})` checks names the same way, which
Pydantic's own `model_copy` does not, but it is a shallow copy unless you pass `deep=True`, and it keeps
the source's lock state. Neither method validates the *values* you pass.

Pydantic deep-copies option defaults for each instance, so a class-level default is never shared
between two config objects.

!!! warning "Never build a config's fields in `__init__` or `model_post_init`"
    Loading a config validates it with `model_validate`, and Pydantic v2 calls a custom `__init__` when
    it validates. A config that sets fields in `__init__` re-applies them on every reconstruction: in
    `load_from_file`, in `with_overrides_from_dict`, and so in `eden train --config` and `--override`.
    Those overrides are then reverted without any message. Declare every value as a field default. For
    a scene, write a factory function that builds on the parent's; the `en.scenes` presets are factories
    for this reason.

### Copying, locking and removing children

- `options.clone()` returns a deep copy that keeps the source's lock state, and `clone(unlock=True)`
  returns an editable one. Prefer `clone()` or `copy.deepcopy` to `model_copy(deep=True)`: only they
  tolerate values that cannot be deep-copied, which they share by reference instead.
- `options.lock()` freezes an options object and everything below it, and writing to a locked object
  raises `RuntimeError`. `unlock()` reverses it. Both apply downward only, so a child can be unlocked
  inside a locked parent. Saving a config locks it by default (below).
- `scene_options.remove_entity(name)` and `disable_term(name)` on a manager's options remove a named
  child. A TaskMod uses them to drop an entity or a term; see
  [TaskMods](task-mods.md#removing-an-entity-or-a-term).

## Saving and loading configs

### Saving

```python
config.save_as_json("logs/my_run/config.json")   # or save_as_yaml(path), or save_as_file(path) by extension
```

`save_as_json` and `save_as_yaml` take an optional path (by default `config.json` / `config.yaml` in the
run's log directory) and raise `ValueError` on the wrong extension. Every save **locks the config**
unless you pass `lock_config=False`, so editing it afterwards raises `RuntimeError`; use
`config.clone(unlock=True)` for an editable copy. A training run saves the config it trains with next to
its checkpoints.

A saved file is an envelope. `_meta` records the config class and the Eden and Genesis that saved it;
`config` holds the fields, and every nested options object is stamped with its own class so that the
exact types come back on load:

```json
{
    "_meta": {
        "eden": {"version": "...", "commit": "..."},
        "genesis": {"version": "...", "commit": null},
        "_option_module_": "eden.tasks.benchmark.reacher.config",
        "_option_class_": "ReacherConfig"
    },
    "config": {
        "env_options": {"num_envs": 4096, "...": "...", "_option_module_": "eden.options.envs", "_option_class_": "EnvOptions"},
        "...": "..."
    }
}
```

### Loading

```python
from eden.utils.configs import EdenConfig

config = EdenConfig.load_from_file("logs/reacher/<run>/config.json")   # returns a ReacherConfig
```

`load_from_file` rebuilds the class the file records, so a generic loader does not need to know the
task. The class you call it on still constrains the result: the saved class must be that class or a
subclass of it, or the load raises `TypeError`. A file without class metadata loads as the class you
called it on. If the saved class cannot be imported (it was renamed, moved, or defined in a script run
as `__main__`), the load raises `ImportError` naming it, rather than falling back to a base class. When
the installed Eden or Genesis differs from the one that saved the file, the load logs a warning.

Loading validates every Eden options object, so an unknown or stale field, or a term parameter that no
longer exists, raises at load and names the class and the field. Embedded Genesis options are not
validated: an unknown key on one is dropped with a warning.

!!! note "The subclass rule is not task identity"
    `load_from_file`'s "this class or a subclass" is right for a library loader, but several tasks
    register a config that subclasses another task's (`ManipulationLiftBoxRGBConfig` is a
    `ManipulationLiftBoxConfig`). When a saved config has to *be* one particular task's, compare the exact
    class. `eden train --config` does; see
    [TaskMods](task-mods.md#overriding-fields-config-and-override).

### Partial overrides

A partial override is a nested mapping of only the fields to change:

```python
config = config.with_overrides_from_dict({"env_options": {"num_envs": 64}})
config = config.with_overrides_from_file("overrides.yaml")   # .json, .yaml or .yml
```

Both return a new config and leave the original untouched. The override is deep-merged into the
config: nested mappings merge key by key, and a key the override does not mention keeps its value. A
**list replaces** the old list whole, so an override of `hidden_dims: [512, 256]` sets exactly that list.
An override that names a different options class (`_option_class_`) for a nested object replaces that
object whole instead of merging into it. The result is validated like a load. On the command line the
same mechanism is `--config <partial file>` and `--override '<inline YAML>'`; see
[TaskMods](task-mods.md#overriding-fields-config-and-override).

## See also

- [TaskMods](task-mods.md): variants, CLI flags, `--config` and `--override`
- [Environments](environments.md): building an environment from a config, and `env.config`
- [Terms](terms.md): writing the terms that the manager option groups hold
- [Physics](physics.md): every `EnvOptions` field
- [Training runs](training-runs.md): runner options and where saved configs land
