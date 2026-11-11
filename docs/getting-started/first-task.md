# Your first task

This tutorial builds a task from scratch and trains it with `eden train`: a Yam arm learns to bring its gripper to a
target. On the way it writes a custom reward term, adds a variant flag with a `TaskMod`, and checks the task before
training. It follows the reacher task,
[`eden/tasks/benchmark/reacher/config.py`](https://github.com/embodied-ai-nexus/Eden/blob/main/eden/tasks/benchmark/reacher/config.py),
which is the template every built-in task is written to.

You need Eden installed ([Installation](installation.md)) and the ideas from [Key concepts](concepts.md).

## Where the task lives

A task is a config class in a Python module, registered with `@TASK_REGISTRY.register()`. Eden finds tasks by
scanning packages for that decorator, so the module has to be in a package Eden scans. There are two places:

- **Your own package** (this tutorial). The package declares itself to Eden through an entry point, and works the
  same with Eden from PyPI or from a clone.
- **Inside a clone of Eden**, under `eden/tasks/`, if you are adding a task to Eden itself. See
  [Adding the task to Eden itself](#adding-the-task-to-eden-itself) at the end.

Create this layout:

```text
my_tasks/
├── pyproject.toml
└── my_tasks/
    ├── __init__.py
    └── tasks/
        ├── __init__.py
        └── reach/
            ├── __init__.py
            └── config.py
```

The `__init__.py` files can be empty. The scan skips `__init__.py` files and any module whose name starts with an
underscore, so the task itself goes in `config.py`.

`pyproject.toml` names the package to scan under the `eden.tasks` entry-point group:

```toml
[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"

[project]
name = "my-tasks"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["eden-world"]

[project.entry-points."eden.tasks"]
my_tasks = "my_tasks.tasks"
```

The entry point's value is the dotted name of a package (not a module, and without a `:attribute`). Eden scans
every `.py` file under it.

## Write the config

The rest of the tutorial fills in `my_tasks/tasks/reach/config.py`. The complete file is at the
[end of this section](#the-complete-file).

### The scene, the actions and the command

Start with the imports, the simulation settings, the scene and the action. These are the reacher's, unchanged:

```python
from __future__ import annotations

from typing import Annotated

import torch

import eden as en
from eden import actions, observations, rewards, terminations
from eden.constants import ReferenceSource
from eden.managers import REWARD_TERM_REGISTRY
from eden.managers.reward_manager import RewardTerm as RewardTermBase
from eden.managers.terms.commands.oracle import UniformT3Command
from eden.managers.terms.observations.manipulation import ObjectToOracleDistance
from eden.managers.terms.utils import resolve_link_idx
from eden.options import ObservationGroupOptions
from eden.options.learning.rsl_rl import (
    RslRlActorOptions,
    RslRlCriticOptions,
    RslRlGaussianDistributionOptions,
    RslRlOnPolicyRunnerOptions,
    RslRlPpoAlgorithmOptions,
)
from eden.tasks import TASK_REGISTRY
from eden.tasks.registry import TaskMod
from eden.terms import DoneTerm, ObsTerm, RewardTerm
from eden.utils.configs import (
    ActionManagerOptions,
    CommandManagerOptions,
    EdenRLConfig,
    EnvOptions,
    ObservationManagerOptions,
    RewardManagerOptions,
    SceneOptions,
    TerminationManagerOptions,
)
from eden.utils.string import term_subject
```

The config class declares each option group as an annotated field with a default:

```python
class MyReachConfig(EdenRLConfig):
    env_options: EnvOptions = EnvOptions(
        num_envs=4096,
        episode_length_s=9.0,
        iterations=10,
        ls_iterations=20,
        sim_dt=0.01,
        sim_substeps=2,
        decimation=1,
        max_collision_pairs=30,
        multiplier_collision_broad_phase=12,
        enable_multi_contact=True,
    )

    scene_options: SceneOptions = en.scenes.Empty(
        is_support_enabled=True,
        robot=en.robots.Yam(default_root_pos=(0.0, 0.0, 0.0), default_root_quat=(1.0, 0.0, 0.0, 0.0)),
    )

    action_options: ActionManagerOptions = ActionManagerOptions(
        joint_pos_controller=actions.ImplicitPDController.configure(
            entity_name="robot",
            dofs_name="joint*",
            scale=0.05,
            reference_source=ReferenceSource.DELTA,
        ),
    )

    command_options: CommandManagerOptions = CommandManagerOptions(
        reaching_target=UniformT3Command.configure(
            x_range=(0.3, 0.5),
            y_range=(-0.1, 0.1),
            z_range=(0.25, 0.35),
            resampling_time_range=(2.0, 3.0),
        ),
    )
```

What each part does:

- `env_options`: 4096 parallel environments, 9-second episodes, and a 0.01 s physics step with no decimation, so
  the policy acts at 100 Hz. The solver settings (`iterations`, `max_collision_pairs`, ...) are the reacher's.
- `scene_options`: a ground plane and the Yam arm, placed at the origin under the name `robot`. Quaternions are
  `(w, x, y, z)`, so `(1.0, 0.0, 0.0, 0.0)` is no rotation.
- `action_options`: one action term, an implicit PD controller on every joint whose name matches `joint*`. With
  `ReferenceSource.DELTA` the target is the current joint position plus `0.05 * action`, a small increment each
  step.
- `command_options`: a 3D target point sampled uniformly in a box in front of the arm, resampled every 2 to 3
  seconds. Other terms find it by its name, `reaching_target`. The arm sits at the origin without rotation, so the
  point is in both the world frame and the arm's base frame.

### Observations

The policy sees the joint positions and velocities, the vector from the gripper to the target, and its own last
action. The reacher defines the terms once and uses them in two groups: `policy` for the actor and `critic` for the
value function.

```python
_OBSERVATIONS = dict(
    dofs_pos=observations.DofsPos.configure(entity_name="robot"),
    dofs_vel=observations.DofsVel.configure(entity_name="robot"),
    gripper_to_target=ObjectToOracleDistance.configure(
        entity_name="robot",
        entity_link_name="link_left_finger",
        command_name="reaching_target",
        robot_name="robot",
    ),
    last_action=ObsTerm.configure(func=observations.last_action, params={"action_name": "joint_pos_controller"}),
)
```

Inside the class:

```python
    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        policy=ObservationGroupOptions(**_OBSERVATIONS),
        critic=ObservationGroupOptions(**_OBSERVATIONS),
    )
```

`DofsPos` and `DofsVel` are class terms, configured directly. `last_action` is a function term, configured through
the `ObsTerm` wrapper with `func=` and `params=`. To find other observation terms and their parameters, run
`eden terms --kind observation`.

### A custom reward term

The reward pays for bringing the gripper to the target: `exp(-d² / std²)`, where `d` is the distance from the
midpoint of the two fingers to the target. It is a class term, because it has names to resolve once (two links and
a command) and a parameter worth validating:

```python
@REWARD_TERM_REGISTRY.register()
class GripperToTarget(RewardTermBase):
    """exp(-d^2 / std^2) of the distance d between the midpoint of the gripper's fingers and the target."""

    entity_name: str = "robot"
    links_name: list[str] = ["link_left_finger", "link_right_finger"]
    command_name: str = "reaching_target"
    std: float = 0.2

    def build(self) -> None:
        if self.std <= 0:
            raise ValueError(
                f"{term_subject(self)}: std={self.std} must be positive; it is the distance (m) at which the "
                f"reward has fallen to exp(-1)."
            )
        robot = self.resolve_entity(self.entity_name)
        self._links_idx = resolve_link_idx(robot, self.links_name, self.device, term=type(self).__name__)
        self._command = self._env.command_manager.get_term(self.command_name)

    def compute(self) -> torch.Tensor:
        robot = self.resolve_entity(self.entity_name)
        gripper_pos = robot.get_links_pos(ls_idx_local=self._links_idx).mean(dim=1)  # (num_envs, 3)
        target_pos = self._command.command[..., :3]
        dist_sq = torch.sum(torch.square(gripper_pos - target_pos), dim=-1)
        return torch.exp(-dist_sq / self.std**2)  # (num_envs,)
```

How the term is put together:

- **The class fields are its parameters**, each with a default. The config sets them through
  `GripperToTarget.configure(...)`, and a misspelled one raises when the config is constructed.
- **`build()` runs once, when the environment is built.** Everything that can be resolved ahead of time is resolved
  here: the link indices and the command term. A wrong link or command name raises now, with a message that lists
  what exists, instead of on the first training step.
- **`compute()` runs every step** and returns one value per environment. It looks the robot up again with
  `resolve_entity()` rather than keeping the handle from `build()`: deployment swaps the simulated robot for the
  real one, and a handle kept from `build()` would keep reading the simulation.
- **The key is the class name in snake case**, `gripper_to_target`. `register()` takes no `name=`.

The base class is imported as `RewardTermBase` because `eden.terms` also exports a `RewardTerm`, the wrapper for
function terms that the next section uses for the two built-in penalties. [Writing terms](../guides/terms.md) covers
the other hooks a class term can implement (`reset()`, `post_reset()`, `post_physics_refresh()`) and when each is
the right place for state.

### Rewards, terminations and the runner

Inside the class, weight the custom reward against two built-in penalties, end each episode on a time-out, and set
up PPO:

```python
    reward_options: RewardManagerOptions = RewardManagerOptions(
        reach=GripperToTarget.configure(weight=1.0, std=0.2),
        action_rate=RewardTerm.configure(func=rewards.action_rate_l2, weight=-0.01),
        joint_pos_limits=RewardTerm.configure(
            func=rewards.dofs_pos_limits, weight=-10.0, params={"entity_name": "robot"}
        ),
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        timeout=DoneTerm.configure(func=terminations.time_out, time_out=True),
    )

    runner_options: RslRlOnPolicyRunnerOptions = RslRlOnPolicyRunnerOptions(
        algorithm=RslRlPpoAlgorithmOptions(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        actor=RslRlActorOptions(
            activation="elu",
            hidden_dims=[256, 128],
            distribution_cfg=RslRlGaussianDistributionOptions(init_std=1.0),
        ),
        critic=RslRlCriticOptions(activation="elu", hidden_dims=[256, 128]),
        max_iterations=100,
        save_interval=100,
        experiment_name="my_reach",
    )
```

`time_out=True` marks the termination as a time limit rather than a failure, which the learner treats
differently when it bootstraps the value of the last state.

### A variant flag

A `TaskMod` turns a choice into a command-line flag. Its constructor arguments become flags, and `apply()` edits
the config. This one sets how far from the centre of the box the target may be sampled:

```python
class TargetSpreadMod(TaskMod):
    """How far from the centre of the workspace the reaching target may be sampled."""

    def __init__(
        self,
        target_spread: Annotated[float, "Half-width (m) in x and y of the box the target is sampled in."] = 0.1,
    ):
        self.target_spread = target_spread

    def apply(self, config: MyReachConfig) -> MyReachConfig:
        if self.target_spread <= 0:
            raise ValueError(f"--target_spread={self.target_spread} must be positive; it is a half-width in metres.")
        target = config.command_options.reaching_target
        target.x_range = (0.4 - self.target_spread, 0.4 + self.target_spread)
        target.y_range = (-self.target_spread, self.target_spread)
        return config
```

Three rules make a `TaskMod` work:

- A **positional-or-keyword** argument with a type annotation becomes a flag (`--target_spread`). Arguments after a
  bare `*` are keyword-only and are not flags; use them for settings fixed at registration time.
- `Annotated[T, "help text"]` gives the flag its `--help` text. A `Literal[...]` type becomes a flag with fixed
  choices.
- The default must describe the same task as the bare config. Here `0.1` reproduces the config's own
  `x_range=(0.3, 0.5)` and `y_range=(-0.1, 0.1)`, so `MyReachConfig()` and `TASK_REGISTRY.build("my_reach")` agree.

### Register the task

Decorate the config class and attach the mod. The decorator goes on the class, the mod is passed as an instance (a
bare class is instantiated with no arguments), and the task name is the class name in snake case without the
`Config` suffix:

```python
@TASK_REGISTRY.register(modifiers=(TargetSpreadMod(),))
class MyReachConfig(EdenRLConfig):
    ...
```

`TASK_REGISTRY` must be imported with `from eden.tasks import TASK_REGISTRY` (or from `eden.tasks.registry`): the
scan recognizes the decorator only through those imports, and warns when it sees the name used without one.

### The complete file

??? example "`my_tasks/tasks/reach/config.py`"

    ```python
    """A Yam arm brings its gripper to a sampled target (the Eden first-task tutorial)."""

    from __future__ import annotations

    from typing import Annotated

    import torch

    import eden as en
    from eden import actions, observations, rewards, terminations
    from eden.constants import ReferenceSource
    from eden.managers import REWARD_TERM_REGISTRY
    from eden.managers.reward_manager import RewardTerm as RewardTermBase
    from eden.managers.terms.commands.oracle import UniformT3Command
    from eden.managers.terms.observations.manipulation import ObjectToOracleDistance
    from eden.managers.terms.utils import resolve_link_idx
    from eden.options import ObservationGroupOptions
    from eden.options.learning.rsl_rl import (
        RslRlActorOptions,
        RslRlCriticOptions,
        RslRlGaussianDistributionOptions,
        RslRlOnPolicyRunnerOptions,
        RslRlPpoAlgorithmOptions,
    )
    from eden.tasks import TASK_REGISTRY
    from eden.tasks.registry import TaskMod
    from eden.terms import DoneTerm, ObsTerm, RewardTerm
    from eden.utils.configs import (
        ActionManagerOptions,
        CommandManagerOptions,
        EdenRLConfig,
        EnvOptions,
        ObservationManagerOptions,
        RewardManagerOptions,
        SceneOptions,
        TerminationManagerOptions,
    )
    from eden.utils.string import term_subject


    @REWARD_TERM_REGISTRY.register()
    class GripperToTarget(RewardTermBase):
        """exp(-d^2 / std^2) of the distance d between the midpoint of the gripper's fingers and the target."""

        entity_name: str = "robot"
        links_name: list[str] = ["link_left_finger", "link_right_finger"]
        command_name: str = "reaching_target"
        std: float = 0.2

        def build(self) -> None:
            if self.std <= 0:
                raise ValueError(
                    f"{term_subject(self)}: std={self.std} must be positive; it is the distance (m) at which the "
                    f"reward has fallen to exp(-1)."
                )
            robot = self.resolve_entity(self.entity_name)
            self._links_idx = resolve_link_idx(robot, self.links_name, self.device, term=type(self).__name__)
            self._command = self._env.command_manager.get_term(self.command_name)

        def compute(self) -> torch.Tensor:
            robot = self.resolve_entity(self.entity_name)
            gripper_pos = robot.get_links_pos(ls_idx_local=self._links_idx).mean(dim=1)  # (num_envs, 3)
            target_pos = self._command.command[..., :3]
            dist_sq = torch.sum(torch.square(gripper_pos - target_pos), dim=-1)
            return torch.exp(-dist_sq / self.std**2)  # (num_envs,)


    _OBSERVATIONS = dict(
        dofs_pos=observations.DofsPos.configure(entity_name="robot"),
        dofs_vel=observations.DofsVel.configure(entity_name="robot"),
        gripper_to_target=ObjectToOracleDistance.configure(
            entity_name="robot",
            entity_link_name="link_left_finger",
            command_name="reaching_target",
            robot_name="robot",
        ),
        last_action=ObsTerm.configure(func=observations.last_action, params={"action_name": "joint_pos_controller"}),
    )


    class TargetSpreadMod(TaskMod):
        """How far from the centre of the workspace the reaching target may be sampled."""

        def __init__(
            self,
            target_spread: Annotated[float, "Half-width (m) in x and y of the box the target is sampled in."] = 0.1,
        ):
            self.target_spread = target_spread

        def apply(self, config: MyReachConfig) -> MyReachConfig:
            if self.target_spread <= 0:
                raise ValueError(
                    f"--target_spread={self.target_spread} must be positive; it is a half-width in metres."
                )
            target = config.command_options.reaching_target
            target.x_range = (0.4 - self.target_spread, 0.4 + self.target_spread)
            target.y_range = (-self.target_spread, self.target_spread)
            return config


    @TASK_REGISTRY.register(modifiers=(TargetSpreadMod(),))
    class MyReachConfig(EdenRLConfig):
        env_options: EnvOptions = EnvOptions(
            num_envs=4096,
            episode_length_s=9.0,
            iterations=10,
            ls_iterations=20,
            sim_dt=0.01,
            sim_substeps=2,
            decimation=1,
            max_collision_pairs=30,
            multiplier_collision_broad_phase=12,
            enable_multi_contact=True,
        )

        scene_options: SceneOptions = en.scenes.Empty(
            is_support_enabled=True,
            robot=en.robots.Yam(default_root_pos=(0.0, 0.0, 0.0), default_root_quat=(1.0, 0.0, 0.0, 0.0)),
        )

        observation_options: ObservationManagerOptions = ObservationManagerOptions(
            policy=ObservationGroupOptions(**_OBSERVATIONS),
            critic=ObservationGroupOptions(**_OBSERVATIONS),
        )

        action_options: ActionManagerOptions = ActionManagerOptions(
            joint_pos_controller=actions.ImplicitPDController.configure(
                entity_name="robot",
                dofs_name="joint*",
                scale=0.05,
                reference_source=ReferenceSource.DELTA,
            ),
        )

        command_options: CommandManagerOptions = CommandManagerOptions(
            reaching_target=UniformT3Command.configure(
                x_range=(0.3, 0.5),
                y_range=(-0.1, 0.1),
                z_range=(0.25, 0.35),
                resampling_time_range=(2.0, 3.0),
            ),
        )

        reward_options: RewardManagerOptions = RewardManagerOptions(
            reach=GripperToTarget.configure(weight=1.0, std=0.2),
            action_rate=RewardTerm.configure(func=rewards.action_rate_l2, weight=-0.01),
            joint_pos_limits=RewardTerm.configure(
                func=rewards.dofs_pos_limits, weight=-10.0, params={"entity_name": "robot"}
            ),
        )

        termination_options: TerminationManagerOptions = TerminationManagerOptions(
            timeout=DoneTerm.configure(func=terminations.time_out, time_out=True),
        )

        runner_options: RslRlOnPolicyRunnerOptions = RslRlOnPolicyRunnerOptions(
            algorithm=RslRlPpoAlgorithmOptions(
                value_loss_coef=1.0,
                use_clipped_value_loss=True,
                clip_param=0.2,
                entropy_coef=0.005,
                num_learning_epochs=5,
                num_mini_batches=4,
                learning_rate=1.0e-3,
                schedule="adaptive",
                gamma=0.99,
                lam=0.95,
                desired_kl=0.01,
                max_grad_norm=1.0,
            ),
            actor=RslRlActorOptions(
                activation="elu",
                hidden_dims=[256, 128],
                distribution_cfg=RslRlGaussianDistributionOptions(init_std=1.0),
            ),
            critic=RslRlCriticOptions(activation="elu", hidden_dims=[256, 128]),
            max_iterations=100,
            save_interval=100,
            experiment_name="my_reach",
        )
    ```


## Install the package

Install `my_tasks` in editable mode into the environment Eden is installed in, so that edits to `config.py` take
effect without reinstalling:

=== "pip"

    ```bash
    cd my_tasks
    pip install -e .
    ```

=== "uv (Eden from a clone)"

    ```bash
    source /path/to/Eden/.venv/bin/activate    # the environment Eden is installed in
    cd my_tasks
    uv pip install -e .
    ```

Reinstall only when you change `pyproject.toml`, for example to rename the entry point.

## Check the task

```bash
eden list
```

`my_reach` appears with `my-tasks`, the distribution's name, as its source and `TargetSpreadMod` as its modifier.
If it is missing, the scan did not find the package: check the entry point's value, and that the decorated class is
not in an `__init__.py` or a module whose name starts with `_`. A broken entry point (a package that does not
import) is reported with a warning when the registry scans.

Then build the task and step it, which runs every term's `build()` and a few `compute()` calls:

```bash
eden smoke --task my_reach                 # GPU, 2 environments, 2 steps
eden smoke --task my_reach --cpu
eden smoke --task my_reach --profile       # also time each term, and write term_cost.json
```

A wrong entity, link or command name fails here, with a message naming what exists. Building the task takes a
fraction of the time a training run does, so run it after every change.

## Train it

```bash
eden train --task my_reach
eden train --task my_reach --target_spread 0.15     # the variant flag
eden train --task my_reach --help                   # every flag, --target_spread with its help text
```

The run lands in `logs/my_reach/<timestamp>-<id>/`, with the checkpoint, `config.json` and `run.json` described in
the [Quick start](quickstart.md#2-train-it). On an RTX 3080 the 100 iterations take about 30 s once the kernels
are compiled, and the `reach` term's episode reward climbs from about 0.01 to 0.3 and is still rising at the
last iteration. Raise `max_iterations` to train further.

## Watch it

```bash
eden inference --task my_reach --checkpoint "$(ls -t logs/my_reach/*/model_*.pt | head -n1)" --vis
```

`eden inference` rebuilds the task from the `config.json` next to the checkpoint, so it plays the variant you
trained. Passing a variant flag such as `--target_spread` together with that checkpoint is an error, because the
saved config already decides it; `--override` still applies on top of the saved config. The saved config names
your classes by their import path, so `my_tasks` must stay installed for the run to replay.

## Using the task from a script

The entry point is what makes `eden` commands see the task. A script of your own can also add a package to the
scan for its own process only:

```python
import eden as en
import genesis as gs

from eden.tasks import TASK_REGISTRY

en.init(backend=gs.gpu)
TASK_REGISTRY.add_search_path("my_tasks.tasks")
config = TASK_REGISTRY.build("my_reach", target_spread=0.15)
```

`build()` takes the mod's flags as keyword arguments. Always construct a registered task with `build()`, never by
calling `MyReachConfig()` or `TASK_REGISTRY.get("my_reach")()`: those skip the task's mods, which for many tasks
leaves required settings empty. A script that trains directly instead of through `eden train` also has to set the
logger itself, since `RslRlOnPolicyRunnerOptions.logger` defaults to `"wandb"` (`eden train` passes its own
`--logger`, which defaults to `tensorboard`). [Training runs](../guides/training-runs.md) shows a complete training
script.

## Adding the task to Eden itself

To contribute a task to Eden, put it in a clone under `eden/tasks/`. Built-in tasks follow conventions that the tests
in `tests/tasks/test_init.py` check:

- The config class is defined in `eden/tasks/<package>/config.py`, or `eden/tasks/<suite>/<task>/config.py` for
  a task in a suite.
- The task name starts with the package's name: name the class `<Package><Task>Config`, so that the key derives as
  `<package>_<task>` (`PolarisPanCleanConfig` in `eden/tasks/polaris` is `polaris_pan_clean`). A task named
  exactly like its package is fine too.
- Task names are snake case.

No entry point is needed: `eden/tasks/` is always scanned. [Task configuration](../guides/task-config.md) describes
the package layout in full, and [Contributing](../contributing/index.md) covers tests and pull requests.

## Next steps

- [Writing terms](../guides/terms.md): the term lifecycle, per-step hooks, validation and the gotchas.
- [TaskMods](../guides/task-mods.md): composable variants, prefixes, mods that change the robot.
- [Task configuration](../guides/task-config.md): every option group a config can set.
- [Events and domain randomization](../guides/events.md): randomize the scene on reset.
- [Training runs](../guides/training-runs.md): resuming, loggers, other runners.

## See also

- The reacher task, [`eden/tasks/benchmark/reacher/config.py`](https://github.com/embodied-ai-nexus/Eden/blob/main/eden/tasks/benchmark/reacher/config.py)
- [Key concepts](concepts.md)
- [Examples](../examples/index.md)
