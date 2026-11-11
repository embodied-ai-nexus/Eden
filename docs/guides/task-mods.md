# TaskMods

A **TaskMod** is a typed, composable preset that edits a task's config when the task is built. TaskMods
give a task its variants (difficulty levels, robots, controllers, runner backends, datasets) without a
subclass or a parallel config file per variant, and each of their parameters becomes a CLI flag. This
page covers writing a TaskMod, attaching it to a task, how mods compose and are inherited, the edits a
mod typically makes, and how mod flags, `--config` and `--override` behave on the command line.

!!! tip "Reference templates"
    - One axis: `ReacherDifficultyMod` in `eden/tasks/benchmark/reacher/config.py`.
    - Several axes: `eden/tasks/benchmark/move_forward/config.py`, where three mods (`MoveForwardRobotMod`,
      `MoveForwardControllerMod`, `MoveForwardRunnerMod`) combine into every robot × controller × runner
      variant.
    - Binding a robot robustly: `VelocityTrackingRobotMod` in `eden/tasks/velocity_tracking/base_config.py`.

## The contract

- A TaskMod is a plain Python class that subclasses `TaskMod` (`eden.tasks.registry`, also exported from
  `eden.tasks`). It is logic with a few parameters, not a Pydantic model.
- Its CLI-tunable parameters are the annotated **positional-or-keyword** parameters of `__init__`.
- `apply(self, config)` edits the config and returns it.
- A mod is attached to a task with `@TASK_REGISTRY.register(modifiers=(...))`.
- `TASK_REGISTRY.build("task_name", **kwargs)` constructs the config class fresh (`cls()`) and runs each
  attached mod's `apply()` in order, routing each keyword to the mod that owns it.

## A TaskMod or a separate task?

A setting over the same task is a TaskMod: the robot, the controller, the difficulty, the dataset. That
holds even when the upstream project gives the variant a task id of its own; UWLab's other-robot task is
`omnireset --robot ur5e_2f85` here.

A separately registered task is one of two things:

- a different task, with a reward or observation contract of its own that is trained and reported on its
  own. `manipulation_lift_box_rgb` gives the actor a wrist camera in place of the object pose;
- a variant a TaskMod cannot express because it changes the config **class**, and with it the
  environment class. PolaRiS's `*_batched` tasks are `EdenSplatRLConfig`s (an RL environment), where the
  single-env tasks are `EdenSplatConfig`s.

## Writing a TaskMod

`ReacherDifficultyMod`, from `eden/tasks/benchmark/reacher/config.py`:

```python
from typing import Annotated, Literal

from eden.tasks.registry import TASK_REGISTRY, TaskMod
from eden.utils.configs import EdenRLConfig

_DIFFICULTY_PRESETS = {
    # name → (target half-extent xy, target half-extent z, reaching_std, episode_length_s)
    "easy": (0.15, 0.075, 0.30, 12.0),
    "medium": (0.10, 0.050, 0.20, 9.0),  # matches ReacherConfig defaults
    "hard": (0.05, 0.025, 0.10, 6.0),
}


class ReacherDifficultyMod(TaskMod):
    """Coordinated preset for target spread, reward sharpness, and episode length."""

    def __init__(
        self,
        difficulty: Annotated[
            Literal["easy", "medium", "hard"],
            "Reaching difficulty: scales target spread, reward sharpness, episode length.",
        ] = "medium",
    ):
        self.difficulty = difficulty

    def apply(self, config: "ReacherConfig") -> "ReacherConfig":
        half_xy, half_z, reaching_std, episode_len = _DIFFICULTY_PRESETS[self.difficulty]

        target = config.command_options.reaching_target
        target.x_range = (0.4 - half_xy, 0.4 + half_xy)
        target.y_range = (-half_xy, half_xy)
        target.z_range = (0.3 - half_z, 0.3 + half_z)

        config.reward_options.reaching_task.params["reaching_std"] = reaching_std
        config.env_options.episode_length_s = episode_len
        return config


@TASK_REGISTRY.register(modifiers=(ReacherDifficultyMod,))
class ReacherConfig(EdenRLConfig):
    ...
```

That is the whole authoring surface: no central index, and the decorator's `modifiers=` tuple is the
only place a mod is attached. `eden train --task reacher --difficulty hard` and
`TASK_REGISTRY.build("reacher", difficulty="hard")` now both select the hard preset.

Note that the bare class and the mod's default describe the same task: `ReacherConfig`'s own field
values are the `medium` preset spelled out. Keep that property in your own tasks where you can, so that
`ReacherConfig()` and `TASK_REGISTRY.build("reacher")` agree.

### CLI parameters and constructor state

CLI-tunable parameters are **positional-or-keyword** `__init__` parameters. Constructor state that is not
a CLI flag goes in **keyword-only** parameters, after `*`:

```python
def __init__(
    self,
    difficulty: Literal["easy", "hard"] = "easy",  # becomes --difficulty
    *,
    entity_name: str = "robot",                    # set at registration, not a flag
):
    self.difficulty = difficulty
    self.entity_name = entity_name
```

Store every CLI parameter as an attribute of the same name. The CLI reads the flag's default from that
attribute, and `build()` overrides a value by setting it.

If you put a CLI knob after `*` by mistake and then pass it to `build()`, the error says so:

```text
ValueError: Unknown modifier kwargs for task 'my_task': ['entity_name']. Known: ['difficulty']
  Hint:
  'entity_name' is keyword-only on MyMod (as 'entity_name') — keyword-only params are non-CLI ctor state. Set it at registration with modifiers=(MyMod(entity_name=...),); if you own MyMod and want a CLI flag, move 'entity_name' before `*` in __init__.
```

### Supported parameter types and help text

A CLI parameter's annotation must be one of `str`, `int`, `float`, `bool` or `Literal[...]`, optionally
wrapped as `T | None`. A `Literal` becomes an argparse `choices=` list, and a `bool` becomes a
`--flag` / `--no-flag` pair. Any other annotation raises `TypeError` when the parser is built.

Wrap the type in `Annotated[T, "help text"]` to fill in the flag's `--help` line. The inner type still
drives validation. The first string in the metadata is the help text; other metadata is ignored, so
other tools can add their own markers.

A mod flag shares the command line with the command's own flags. `task`, `run_name`, `config`,
`override`, `checkpoint` and `cpu` are reserved, and a mod field that resolves to one of them raises
`ValueError` naming the mod.

### What `apply()` may do

`apply(config)` may edit `config` in place; return it either way. `build()` constructs a fresh config on
every call, and Pydantic deep-copies option defaults for each instance, so an edit never leaks into the
next build.

Treat `self` as read-only inside `apply()`. `build()` copies a mod only when a keyword overrides one of
its fields; otherwise it reuses the registered instance, so a change to `self` would persist into later
builds. Keep per-build state on the config you return.

Calling `super().__init__()` is optional: `TaskMod` keeps no instance state of its own.

## Building a task with mods

```python
from eden.tasks import TASK_REGISTRY

config = TASK_REGISTRY.build("move_forward")                                  # every mod at its default
config = TASK_REGISTRY.build("move_forward", robot="go1")                     # one axis
config = TASK_REGISTRY.build("move_forward", robot="go1", controller="explicit", runner="rl_games")
```

Keywords are routed by flag name to the mod that owns them. An unknown keyword raises `ValueError`
listing the accepted flags. Each value is checked against its annotation: a `Literal` value must be one
of the choices, and a `str`, `int`, `float` or `bool` field must receive that type. The CLI path gets
the same checks from argparse.

!!! warning "`get()` returns the class, and instantiating it yourself skips every mod"
    `TASK_REGISTRY.get("my_task")()`, or `MyTaskConfig()`, constructs the bare config and runs no
    attached mod. For a task whose robot binding lives in a mod (any `--robot`-style variant), that
    leaves the mod-managed fields at their empty placeholders. The config constructs and saves without
    complaint, then fails in `env.build()` on a link or joint lookup that says nothing about modifiers,
    unless the config overrides [`check_bindings()`](#check_bindings-name-what-is-still-unset).

    Use `build()` unless you specifically want the class itself. For a task without mods, `build()` with
    no keywords is the same as `get()()`. For a task with mods it applies each mod's registered default,
    which is the config `eden train --task <name>` trains.

The Python smoke harness takes mod keywords the same way:
`eden.utils.smoke.build_and_step("move_forward", modifier_kwargs={"robot": "go1"})` is the Python side of
`eden smoke --task move_forward --robot go1`.

## Composing mods

Several mods on one task run in registration order. Each mod usually owns a separate slice of the
config; when two touch the same field, the later mod sees the earlier one's edits.

```python
@TASK_REGISTRY.register(modifiers=(MoveForwardRobotMod, MoveForwardControllerMod, MoveForwardRunnerMod))
class MoveForwardConfig(EdenRLConfig):
    ...
```

That task exposes `--robot`, `--controller` and `--runner`, which combine freely:

```bash
eden train --task move_forward                                                      # all defaults
eden train --task move_forward --controller explicit                                # one axis
eden train --task move_forward --robot go1 --controller explicit --runner rl_games  # all three
```

### The same mod twice: `with_prefix`

To register one mod class twice, for example once per arm of a two-arm task, give each instance a
prefix with `with_prefix()`. The flags become `<prefix>_<field>`:

```python
@TASK_REGISTRY.register(modifiers=(
    RobotMod(robot="shadow", entity_name="robot_left").with_prefix("left"),
    RobotMod(entity_name="robot_right").with_prefix("right"),
))
class TwoArmTaskConfig(EdenRLConfig):
    ...

# CLI: --left_robot allegro --right_robot shadow
config = TASK_REGISTRY.build("two_arm_task", left_robot="allegro", right_robot="shadow")
```

`mod.flag_for(field)` returns a field's flag name. Two mods that would claim the same flag raise
`ValueError` from `build()`, naming both owners.

### Per-registration defaults

Pass an instance instead of a class to give a registration its own default:

```python
@TASK_REGISTRY.register(modifiers=(ReacherDifficultyMod(difficulty="hard"),))  # key "reacher_hard"
class ReacherHardConfig(ReacherConfig):
    pass
```

`TASK_REGISTRY.build("reacher_hard")` then uses `hard`, and the CLI shows `hard` as the flag's default; a
keyword or flag still overrides it. (A bare mod class in `modifiers=` is instantiated with no arguments.)

## Subclassing a task: modifiers are inherited

A config registered **without** `modifiers=` inherits the modifiers of its nearest registered base class,
so a subclass of a mod-bound task keeps the mods that bind it:

```python
@TASK_REGISTRY.register()   # inherits VelocityTrackingRobotMod and VelocityTrackingObsMod
class G1StiffArmsConfig(VelocityTrackingFlatConfig):
    ...

TASK_REGISTRY.build("g1_stiff_arms", robot="oli")   # --robot still works
```

| `register(...)` | Modifiers the task gets |
|---|---|
| no `modifiers=` (or a bare `@TASK_REGISTRY.register`) | the nearest registered base class's; a class with no registered base gets none |
| `modifiers=()` | none: an explicit opt-out |
| `modifiers=(...)` | exactly these, replacing the inherited ones |

Opt out when a parent's mod does not fit the subclass. `manipulation_lift_box_rgb` does:
`LiftBoxManipulatorMod` would swap the arm under a wrist camera calibrated to the DROID flange, and
would log the run to the parent task's directory. `eden list` shows inherited modifiers too, without
importing the task module.

The details of "nearest", for when you shadow or re-register tasks:

- The walk follows the class's MRO, passes through unregistered intermediate classes, and counts the
  class itself when it is already registered under another name, so a second name for one class builds
  what the first does.
- **Inheritance is resolved on every read**, not frozen at registration. What a task inherits tracks the
  registry as it stands, whatever order the registrations ran in: re-register the class the mods come
  from, or register it after the subclass, and the subclass follows. Pass `modifiers=` on the subclass to
  pin it.
- Shadowing a built-in task with a subclass keeps the built-in's mods:
  `register(MyFlatConfig, name="velocity_tracking_flat", override=True)` inherits from the registration it
  replaces. That one inherited value is frozen, because nothing live holds those mods any more.
- If the class the mods would come from is registered under several names with **different** modifiers,
  and none of them is the registration being replaced, registration raises `ValueError` and asks you to
  pass `modifiers=` explicitly. A later registration that leaves an existing task in that position makes
  `get_modifiers()` and `build()` raise the same error for it; `eden list` falls back to the names the
  decorators declare rather than failing.
- A subclass whose registered base is replaced by a class it does not derive from inherits nothing. Only
  a config that overrides `check_bindings()` reports the unset fields at `from_config`; anywhere else
  they surface wherever they are first used.

The inherited mod instances are shared, not copied, which is safe because `build()` never mutates a
registered mod.

## Editing a config inside a mod

### Removing an entity or a term

Entities and terms are dynamically named children of their options object. A variant that needs fewer
of them removes them by name, never by editing `__pydantic_extra__` directly, which would skip the lock
check:

```python
def apply(self, config):
    config.scene_options.remove_entity("milk")
    config.reward_options.disable_term("in_bin_milk")         # a flat term, a whole group, or a term inside one group
    config.observation_options.policy.disable_term("milk_pos")  # observation terms live on their group,
    config.observation_options.critic.disable_term("milk_pos")  # and one term is often in several groups
    return config
```

All three raise `KeyError` on an unknown name, with a message that lists the names that can be removed,
and `RuntimeError` on a locked config. `RewardManagerOptions.disable_term` raises `ValueError` when the
name matches more than one entry (a term in two groups, say); call `disable_term` on the owning group
instead. An observation group's `disable_term` also drops the name from the group's `terms_order`, when
the group declares one.

`remove_entity` removes only an entity that was added by name. One declared as a field of a
`SceneOptions` subclass (`DroidSceneOptions.robot`) is part of that scene class, and naming it raises
`KeyError`. A composite (an arm and gripper rig) is removed whole under its placement key; naming one of
its bodies raises `KeyError` naming the composite. Removing an entity also drops its own
`attachments_dict` entry, and raises `ValueError` while another entity is still attached to it.

`remove_entity` touches nothing else that names the entity: manager terms, cameras
(`attach_entity_name`, `follow_entity_name`) and sensors (`attach_entity_name`, `track_link_names`,
`filter_link_names`). Update those too, or they fail to resolve the entity when the environment is
built. `drop_object` in `eden/tasks/robosuite/variants.py` is the worked example: it removes an object and
every manager term whose key names it, and raises if no term did.

### Adding an observation term

The counterpart of `disable_term` is assignment. A term written onto a group under a new name becomes one
of its terms:

```python
from eden import observations
from eden.terms import ObsTerm


def apply(self, config):
    policy = config.observation_options.policy
    if policy.terms_order is not None:
        # Where a term sits in terms_order is where it sits in the observation vector. Set the order
        # BEFORE assigning the terms: each assignment is checked against it.
        policy.terms_order = [*policy.terms_order, "milk_pos", "milk_quat"]
    policy.milk_pos = ObsTerm.configure(func=observations.base_pos, params={"entity_name": "milk"})
    policy.milk_quat = ObsTerm.configure(func=observations.base_quat, params={"entity_name": "milk"})
    return config
```

A group adopts an assigned term exactly as its constructor does. It deep-copies the term, so assigning
one term object to both `policy` and `critic` gives each group its own copy, and it builds a term written
as a plain `dict` into an `ObservationTermOptions`. The group's checks run on the assignment line, not at
build:

- a name that is not a term (a misspelled group field) raises `TypeError` naming the field it resembles;
- in a group that declares a `terms_order`, the assigned term must already be listed in it. The message
  names every term the order is missing and gives the whole `terms_order` to type.

A refused assignment changes nothing, and an assignment onto a locked config raises `RuntimeError`.

`terms_order` itself is a declared field, so writing it is not checked on that line. That is what lets
you set the order before the terms, and it means a bare `policy.terms_order = [...]` can leave the order
and the terms out of step until the environment is built. The build then checks the whole group again:
a misspelled name in the order is reported with the group's actual terms and a corrected order to type,
and a repeated name is reported too, since the order is the layout and a repeat would build that term
twice.

!!! warning "Extend `terms_order`, not `term_keys()`"
    `group.term_keys()` returns the group's terms in **declaration** order. Building a new order from it
    silently re-lays out any group whose `terms_order` differs from its declaration order: the same width,
    different channels, and no error anywhere. Extend `terms_order` when it is set. `VelocityTrackingObsMod`
    in `eden/tasks/velocity_tracking/base_config.py` builds its order from `term_keys()` only because it
    runs on a group whose `terms_order` is `None`, where there is no order to extend.

### Binding a robot by term type

A robot mod rewrites fields that live on terms. Addressing those terms by config key
(`config.action_options.dofs_pos_controller`) couples the mod to names a variant is free to change: a
subclass that splits the controller into `legs` and `upper`, or drops a reward, then fails inside the mod
with an `AttributeError` that names neither the mod nor the fix. `VelocityTrackingRobotMod`
(`eden/tasks/velocity_tracking/base_config.py`) shows the robust shape:

- **Find terms by type and entity, never by key.** Resolve each term's class through its registry
  (`ACTION_TERM_REGISTRY.get(term.name)`), keep the terms of the classes the mod binds whose
  `entity_name` is the robot, and descend into reward groups. For action terms,
  `is_joint_position_controller` (`eden.managers.terms.actions.joint_actions`) is the test.
- **Fill only unset placeholders.** Ship the robot-specific fields empty (`""`, `[]`, and `None` for a
  controller's `scale`) and let the mod fill a field only while it is still empty, so a value a variant
  sets for itself survives. The placeholder must be a value no author would set on purpose: a
  controller's `scale` defaults to `None` rather than `1.0` for exactly that reason, so an authored unit
  scale is kept, and a controller that sets `scale_ratio` instead counts as scaled.
- **Share one whole-body dict across split controllers.** A joint position controller with
  `dict_keys_scope="entity"` skips `scale` / `offset` keys that match only joints other controllers own,
  so one whole-robot action-scale dict (`en.robots.G1_ACTION_SCALE`) serves a legs-only controller as is.
  A key that matches no joint of the robot still raises.
- **A whole-entity default must be the only one.** A field that means "everything" when unset, such as a
  controller's empty `dofs_name` (which the mod fills with every joint of the robot), collides with any
  other term that commands part of the same entity. Count what the manager will count, not only what
  the mod binds: the velocity-tracking mod counts every action term that commands DOFs of the robot, so a
  `NullJointAction` pinning the arms next to an unbound controller is refused by the mod, with the fix,
  instead of reaching `ActionManager`'s overlap check at build.

Event terms that target action terms follow the same idea: `RandomizeDofsPosOffset(action_term_name=None)`
covers every joint position controller on its entity instead of naming one.

### `check_bindings()`: name what is still unset

`EdenConfig.check_bindings()` does nothing by default. `EnvBase.from_config` calls it before it constructs
anything, so a config whose fields a mod fills should override it to raise, listing every field still at
its placeholder and the mod that fills it. That is where a config built by `get(name)()`, a subclass
registered with `modifiers=()`, or a subclass of an unregistered base fails, with a message that names
the problem. `VelocityTrackingBaseConfig.check_bindings` is the worked example.

## From the command line

### Mod flags

`eden train`, `eden inference`, `eden export`, `eden deploy` and `eden smoke` build their parsers with
`get_task_argparser`, which adds the selected task's mod flags. The parser reads `--task`
from the command line first, so `--help` shows only the flags of the task you named:

```bash
eden list                                       # every task, its source and its modifiers
eden train --task reacher --help                # includes --difficulty {easy,medium,hard}
eden train --task reacher --difficulty hard
eden smoke --task move_forward --robot go1      # build the variant and step it twice
```

### Overriding fields: `--config` and `--override`

Two flags change a task's config from the command line. They apply in this order, and a later one wins:

| Flag | Takes | Effect |
|---|---|---|
| `--config` | a `.json` / `.yaml` / `.yml` **path** | A config Eden *saved* (it carries class and version metadata) **replaces** the task's build. Any other file **layers onto** it as a partial override. |
| `--override` | an inline **YAML mapping** | Layers onto whatever `--config` produced. Repeatable; later ones win. |

```bash
eden train --task reacher --override 'env_options: {num_envs: 64}'
eden train --task reacher --override 'env_options: {num_envs: 64}' --override 'runner_options: {max_iterations: 500}'
eden train --task reacher --config overrides.yaml                   # partial file: layered
eden train --task reacher --config logs/reacher/<run>/config.json   # saved config: replaces

# The whole chain: the mod flag builds the task, then the partial file, then the inline override.
eden train --task reacher --difficulty hard --config overrides.yaml --override 'env_options: {num_envs: 64}'
```

Each layer deep-merges into the one before (see
[Partial overrides](task-config.md#partial-overrides)): a key set only by an earlier layer survives, and a
list is replaced whole.

A full file and a partial one are told apart by **content**, not by a flag: `save_as_file` writes a `_meta`
envelope with the config's class, and a hand-written override has none. Two things are hard errors rather
than surprises:

- **A saved config that is not the one `--task` builds.** The comparison is an *exact* class match,
  stricter than `EdenConfig.load_from_file`'s "this class or a subclass". Several tasks register a config
  that subclasses another task's (`manipulation_lift_box_rgb` from `manipulation_lift_box`,
  `extreme_parkour_student` from `extreme_parkour_teacher`), so accepting a subclass would run one task's
  config under another task's name and in its `logs/<task>/` directory, where `--resume` and checkpoint
  replay look later. The error names the task to re-run with.
- **A mod flag set away from its default alongside a saved config**, because the saved config replaces
  the whole build, mods included, and the flag would be silently dropped. Express the change as
  `--override` instead.

Each flag names the other when handed the other's input: `--override` given a file path points at
`--config`, and `--config` given an inline mapping points at `--override`.

`--override` is parsed with `yaml.safe_load`, so JSON works too (it is a subset of YAML). PyYAML follows
YAML 1.1, where `off`, `no` and `yes` read as booleans: quote a value that must stay a string.

### Replaying a checkpoint's config

`eden inference`, `eden export` and `eden deploy` reach a saved config by another route. A checkpoint at
`<run>/model_<n>.pt` replays its sibling `<run>/config.json` automatically, so the run uses exactly the
config it was trained with, not the task's current defaults. A partial `--config` and any `--override`
still apply on top; a saved `--config` replaces the sibling. Mod flags set away from their default are an
error here too, for the same reason as above. The fields in `EnvOptions.training_only_fields`
(`cuda_graph_options`) are reset to their defaults for the replay.

That config is authoritative, so `--task` is only a label there, but the label picks the `logs/<task>/`
directory the run is written to.

!!! warning "A mismatched `--task` moves the run, it does not stop it"
    When the sibling config is not the one `--task` builds, the replay follows the **checkpoint's** task:
    `eden inference --task reacher --checkpoint logs/move_forward/<run>/model_100.pt` logs under
    `logs/move_forward/`, not `logs/reacher/`, and warns, naming both. A script that looks for a replay's
    output under the `--task` it passed finds nothing in that case; fix the `--task` the warning names.

    It is a warning rather than the error a mismatched `--config` gets, because here the config was picked
    up for you. A saved class that no registered task declares also warns, but keeps `logs/<task>/`, since
    there is no better directory to name. If that class was renamed or moved as well, loading it raises
    `ImportError` right after.

    The comparison runs forward: the saved class against the class `--task` declares
    (`TASK_REGISTRY.config_qualname_for_task`). So registering one config class under two names does not
    read as a mismatch.

### The parser in your own script

`eden.tasks.parser` is what the commands use, and a script can use it too:

```python
from eden.tasks.parser import get_task_argparser, get_task_config_from_args

parser = get_task_argparser("Train my task.", default_task_name="my_task")
parser.add_argument("--num_envs", type=int, default=4096)
args = parser.parse_args()

config = get_task_config_from_args(args, run_name=args.run_name)   # build, then --config, then --override
config.env_options.num_envs = args.num_envs
```

`get_task_argparser` adds `--task`, `--config`, `--override`, `--checkpoint`, `--cpu` and `--run_name`,
plus the selected task's mod flags. Pass `argv=` when your script parses its own slice of the command
line, so that the `--task` it reads first and the final parse agree. `get_task_config` does the same
build without argparse, from keyword arguments.

## API reference

`TaskMod` (`prefix`, `with_prefix`, `flag_for`, `apply`) and the registry methods used on this page
(`TaskRegistry.register`, `build`, `add_search_path`) are documented in the
[Configs and Task Registry](../api/configs.md) API page.

## See also

- [Task Configuration](task-config.md): config classes, the task registry, saving and loading
- [Training runs](training-runs.md): `eden train`, `eden inference`, logs and checkpoints
- [Policy zoo](policy-zoo.md): `eden inference --policy` for the shipped frozen policies
- [Reference: CLI](../reference/cli.md): every command and flag
