# Training Runs

This page covers training a task with `eden train`, replaying a checkpoint with `eden inference`, and what a run
leaves on disk: the run directory, checkpoints, the progress file, seeding and logging. It ends with how to plug in an
RL algorithm or runner of your own. Read it when you train a task for the first time, when you resume or replay a
run, or when a run behaves in a way you did not expect.

`eden train` drives two RL libraries: [rsl_rl](https://github.com/leggedrobotics/rsl_rl) (`rsl-rl-lib`, the default)
and rl_games. To train with another library (stable-baselines3, skrl, CleanRL) see
[Using Eden with Another RL Library](rl-libraries.md); to train on several GPUs see
[Multi-GPU Training](ddp.md).

![Many G1 humanoids training in parallel](../assets/readme/train_at_scale.webp)

## Train a task

Every registered task with `runner_options` trains from the command line:

```bash
eden list                                       # the registered tasks
eden train --task reacher                       # train with the task's defaults
eden train --task reacher --num_envs 1024       # fewer parallel envs
eden train --task velocity_tracking_flat --run_name lower_lr \
    --override "runner_options: {algorithm: {learning_rate: 3.0e-4}}"
```

A task may add flags of its own (its [TaskMods](task-mods.md)); `eden train --task <name> --help` lists them. For
example, `move_forward` exposes `--runner rl_games` to train with rl_games instead of rsl_rl.

| Flag | What it does |
|------|--------------|
| `--task TASK` | The registered task to train. Required. |
| `--num_envs N` | Replace the task's `env_options.num_envs`. Under DDP this is the count **per GPU**. |
| `--config PATH` | A `.json` / `.yaml` config file. A config Eden saved (one that carries its class) replaces the task's config; any other file is a partial override layered on it. |
| `--override YAML` | An inline YAML mapping merged into the config, for example `'env_options: {num_envs: 64}'`. Repeatable; later ones win; applied after `--config`. |
| `--checkpoint PATH` | Load these weights before training. |
| `--resume REF` | Continue from an earlier run of this task (see [Resume a run](#resume-a-run)). Mutually exclusive with `--checkpoint`. |
| `--run_name NAME` | Folded into the run directory name and used as the wandb run name. |
| `--logger {tensorboard,neptune,wandb}` | The rsl_rl logger. Default `tensorboard`. Ignored for rl_games. |
| `--wandb_group GROUP` | The wandb group for this run (default: the task name). Ignored unless `--logger wandb`. |
| `--cpu` | Run on the CPU backend. |

`eden train` picks the library from the type of the task's `runner_options`: an
`RlGamesPpoRunnerOptions` trains with rl_games, anything else with rsl_rl. A task whose `runner_options` is `None`
has its learner outside Eden, and `eden train` exits with a message that points at the task's standalone example.

!!! tip "Check a task before a long run"
    `eden smoke --task <name>` builds the task with two envs and steps it twice. It catches a broken config in
    seconds, before a training run spends minutes building thousands of envs. `eden smoke --task <name> --profile`
    measures what each term costs per step and writes `term_cost.json`.

### What is checked before the build

A scene with thousands of envs can take minutes to build, so `eden train` checks the learner first and exits before
anything is built or written when:

- `runner_options.class_name` names a runner that cannot be imported, lacks the runner interface, or cannot train
  under the current distributed launch;
- a model or algorithm `class_name` given as an import path does not resolve;
- the algorithm class does not accept the fields its options carry (see
  [Custom algorithms](#custom-algorithms-algorithmclass_name));
- the runner needs a checkpoint to start (a distillation needs its teacher) and you passed neither `--checkpoint` nor
  `--resume`;
- the algorithm options define a `preflight()` and it fails (AMP-PPO loads its expert clips there).

## The run directory

Each run writes into its own directory:

```text
logs/<task>/<timestamp>[-<run_name>]-<run id>/
├── config.json        # the full config the run trained with
├── run.json           # provenance: versions, seed, world size, argv, parent run
├── progress.json      # a small status snapshot, refreshed every iteration
├── model_<iteration>.pt
└── events.out.tfevents.*   # TensorBoard (with --logger tensorboard)
```

The timestamp sorts chronologically (for example `26Nov10T141532`), and the run id is a random 8-character string
that keeps two runs started in the same second apart. `--run_name` is sanitized and placed between them.

`config.json` is the source of truth for the run: `eden inference`, `eden export` and `eden deploy` rebuild the env
from the `config.json` next to a checkpoint, not from the task's current defaults. Keep it, and `run.json`, next to
the `.pt` when you copy a checkpoint elsewhere.

!!! note "The directory is created on first use"
    `en.init()` only *resolves* the run directory; the first read of `en.log_dir` (or `en.get_log_dir()`) creates
    it. A script that never logs anything leaves no empty directory behind. If you look up "my newest run" with
    something like `max(glob.glob("logs/<task>/*"), key=os.path.getmtime)`, do that glob **before** your script
    reads `en.log_dir`, or the glob finds the empty directory of the run you just started.

### `run.json`

`run.json` records what you need to reproduce or audit the run: `run_id`, `run_name`, `timestamp`, `log_dir`, `task`,
the base `seed`, `world_size`, the command line (`argv`), `resumed_from` (the parent run of a `--resume`), the Eden and
Genesis versions and commits (`eden`, `genesis`), and the RL library that wrote the checkpoints (`rl_library`:
`{backend, distribution, version}`). It is written by the main process only, and a failure to write it warns instead
of stopping the run.

### `progress.json`

Every rsl_rl run also writes `progress.json`, an atomically updated snapshot that a script or a coding agent can
poll far more cheaply than it can tail the training log:

```json
{"status": "running", "iteration": 42, "total_iterations": 1500,
 "total_timesteps": 4128768, "mean_reward": 12.7, "mean_episode_length": 480.2,
 "best_mean_reward": 12.9, "elapsed_s": 613.4, "updated_at": 1786781380.82}
```

`status` is `"running"` while training and `"completed"` once `learn()` returns normally. `updated_at` is wall-clock
epoch seconds, so a poller can tell a finished run (`"completed"`) from a stalled or killed one (still `"running"`
with a stale `updated_at`: a hard stop never reaches the final write), and compute progress from `iteration /
total_iterations`. The file is written only by rank 0 and never by an evaluation runner.

A FlashSAC run (`RslRlFlashSacRunnerOptions`) refreshes the file once per log line, every `log_interval` iterations,
and its `"running"` snapshots trail slightly: `total_timesteps` and `elapsed_s` leave out the latest window's
iterations but one. The `"completed"` snapshot is exact.

### Keeping only the newest checkpoints

rsl_rl saves `model_<iteration>.pt` every `save_interval` iterations and once more at the end. A long run at a short
interval would hold hundreds of them, so Eden deletes the older ones after each save and keeps the newest
`runner_options.keep_last_checkpoints` (default `1`, which is all `--resume` needs). To keep every checkpoint, for a
learning-curve sweep say, pass `None`:

```bash
eden train --task <task> --override "runner_options: {keep_last_checkpoints: null}"
```

Details that matter when you rely on it:

- Only checkpoints of an **earlier** iteration than the one just saved are deleted. A later one, left in a reused
  directory or from before a rollback, is kept and reported once, because `--resume` would pick it.
- A save to any other file name (an export, say) prunes nothing. Checkpoints already uploaded to wandb stay there.
- While wandb or neptune uploads a checkpoint in the background, one extra checkpoint is kept. With wandb it goes when
  `learn()` returns and the wandb run has finished; with neptune, or under rsl-rl-lib 3, nothing signals the end of the
  upload and the extra checkpoint stays.
- rl_games (`RlGamesPpoRunnerOptions`) takes the same option. rl_games writes
  `nn/last_<name>_ep_<epoch>_rew_<reward>.pth` every `save_frequency` epochs and never deletes one, so Eden prunes those
  the same way. Its best-reward checkpoint `nn/<name>.pth` is never deleted.

## Loggers

`eden train` logs to TensorBoard unless you pass `--logger wandb` or `--logger neptune`. The runner options default
to `logger="wandb"`, so a training **script** that builds the runner itself logs to wandb unless it sets the field;
`eden inference` and `eden export` always force TensorBoard so a replay never creates a wandb run.

With wandb, runs go to the project `runner_options.wandb_project` (default `"eden"`) and are grouped by task, or by
`--wandb_group` to compare the arms of an ablation across tasks. The group is passed through the `WANDB_RUN_GROUP`
environment variable, and a value you exported yourself wins.

## Resume a run

`--resume REF` continues a run of the same task. `REF` is resolved under `logs/<task>/`:

- `latest`: the most recently modified run directory;
- a run id: the trailing 8-character segment of a run directory's name;
- any other string: a substring of the directory name, such as part of a `--run_name`.

A typo or an ambiguous substring fails before anything is built, and the error lists the runs it found. The resumed
run loads the newest `model_<iteration>.pt` of that directory and writes into a **new** run directory, whose `run.json`
names the parent in `resumed_from`. The old run's directory, TensorBoard events and wandb run are left alone.

```bash
eden train --task velocity_tracking_flat --resume latest
eden train --task velocity_tracking_flat --resume lower_lr       # by run_name
```

`--checkpoint PATH` also loads weights before training, from any file, without the run lookup.

## Replay a checkpoint

```bash
eden inference --task reacher --checkpoint logs/reacher/<run>/model_1500.pt --vis
eden inference --task reacher --checkpoint <ckpt> --num_envs 64 --episodes 200   # evaluate metrics
eden inference --task reacher --checkpoint <ckpt> --rec                          # record an mp4
eden inference --task reacher --checkpoint <ckpt> --async                        # deployment-style timing
```

`eden inference` rebuilds the env from the `config.json` next to the checkpoint, so the replay runs the exact config
the checkpoint trained with. `--config` (a partial one) and `--override` still apply on top. Three consequences:

- **Task-mod flags are refused** when a sibling `config.json` exists: the saved config is authoritative and the flags
  would be silently ignored. Drop them, or pass a saved config to `--config` to use your own.
- **`--task` is only a label** for a replay. If the saved config belongs to another task, the run is logged under that
  task's `logs/<task>/` and Eden warns; re-run with the right `--task` to silence it.
- Fields that only speed up or instrument training (`EnvOptions.training_only_fields`, such as
  `cuda_graph_options`) are reset to their defaults first, so a checkpoint trained on a GPU host replays on a host that
  cannot use them.

A checkpoint with no `config.json` beside it is replayed on the task's current config, rebuilt from `--task` and any
task-mod flags.

| Flag | What it does |
|------|--------------|
| `--checkpoint PATH` | The checkpoint to replay. Required unless you pass `--policy`. |
| `--num_envs N` | Parallel evaluation envs (default `1`). |
| `--steps N` | Stop after N env steps (default: one episode, `max_episode_length`). |
| `--episodes N` | Run until each env has finished its first `ceil(N / num_envs)` episodes, then print the mean of every metric over them. |
| `--vis` | Open the Genesis viewer. |
| `--rec` | Add a `rec` camera and save an mp4 under the replay's own run directory. |
| `--async` (`--sim2sim`) | Run inference on a background thread with the sim paced to wall-clock `dt`, repeating the last action when inference lags, as `eden deploy` does. Reports stale steps, overruns and the real-time factor. |
| `--policy NAME` | Roll out a frozen policy from the [policy zoo](policy-zoo.md) instead of a checkpoint. |

The checkpoint's library is detected from its keys: an rl_games checkpoint for a task configured for rsl_rl (or the
reverse) exits with a message naming the task-mod flag that selects the other library, rather than failing inside
`load`.

`--episodes` counts each env's first episodes rather than stopping at the N-th reset overall, because stopping early
would leave out the longest episodes still running and bias a metric such as success rate. Episodes in which Genesis
flagged the state as corrupt count toward the cap but not toward the means.

## Train from a script

`eden train` is a thin layer over three calls you can make yourself:

```python
import eden as en  # import eden before genesis and torch (see Multi-GPU Training)
import genesis as gs

from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper, make_rsl_rl_runner
from eden.tasks import TASK_REGISTRY

en.init(backend=gs.gpu, log_root_path="logs/reacher", performance_mode=True)

config = TASK_REGISTRY.build("reacher")
config.env_options.num_envs = 4096
config.runner_options.logger = "tensorboard"  # the options default to wandb
config.save_as_json()                         # writes config.json into en.log_dir

env = RslRlVecEnvWrapper.from_config(config)  # seeds, builds and resets the env
runner = make_rsl_rl_runner(env)              # Eden's runner: retention, progress.json, DDP
runner.learn(num_learning_iterations=config.runner_options.max_iterations, init_at_random_ep_len=True)
```

`RslRlVecEnvWrapper.from_config` builds and resets the env, unlike `RLEnvBase.from_config`, which only constructs it.
Prefer `make_rsl_rl_runner(env, checkpoint=...)` to constructing rsl_rl's `OnPolicyRunner` directly: it builds the
runner `runner_options.class_name` names (the DDP variant under `torchrun`), adds checkpoint retention, the progress
file and the agent log theme, and loads a checkpoint across rsl-rl-lib versions. For a replay, pass
`for_inference=True`, build the env with `eval_mode=True`, and call `runner.get_inference_policy(device=env.device)`.
`examples/projects/reaching/reacher.py` is a complete train-and-replay script.

## Seeding

`RslRlVecEnvWrapper.from_config` and `RlGamesVecEnvWrapper.from_config` apply `runner_options.seed` (default `7100`)
**before** constructing the env, so the scene build, the first reset and every domain-randomization draw come from
the seeded RNGs. rsl-rl-lib never applies the field itself.

- **Under DDP, rank `r` uses `seed + r`**, so ranks sample different randomization instead of replaying rank 0.
  `seed` stays the base: the applied value is logged and returned, never written back. `config.json` and `run.json`
  record the base, and `run.json`'s `world_size` pins the rest.
- **Do not call `set_random_seed()` around a wrapper's `from_config()`**: the wrapper overwrites it with the
  configured seed. Set `config.runner_options.seed` instead.
- **Seeding is process-global.** Building a second env mid-run restarts the RNG stream the first one draws from. Pass
  `apply_seed=False` to `from_config` for a caller that must not disturb it (hardware deployment does).
- **If you build the raw env yourself** (`RLEnvBase.from_config(cfg)` and `build()`) and wrap it by hand, call
  `eden.utils.random.apply_run_seed(cfg.runner_options.seed)` first; nothing else on that path will.
- **Seeds are non-negative.** A negative seed is rejected when the config loads.

## Checkpoints need a matching rsl-rl-lib

The layout of a `.pt` file belongs to rsl-rl-lib, not to Eden: rsl-rl-lib 3.x saves one `model_state_dict`, while
4.x and 5.x split it into `actor_state_dict` and `critic_state_dict`. Eden's dependency (`rsl-rl-lib>=3.0`) admits both,
so a checkpoint may not replay under the rsl-rl-lib you have installed.

Eden handles the two directions differently:

- **A 3.x checkpoint under 4.x/5.x** is converted: the file is rewritten into a `<name>.<std_type>.converted.pt`
  sibling and loaded from there (your file is never modified). This also covers a `DistillationRunner`. The conversion
  needs the destination's `std_type`, read from the runner's `actor` (else `teacher`, else `student`) options; a
  runner config that states none raises instead of guessing.
- **A 4.x/5.x checkpoint under 3.x** cannot load. The error names both versions and the keys the file holds, instead
  of rsl_rl's bare `KeyError: 'model_state_dict'`.

Before loading, Eden reads the checkpoint's sibling `run.json` (`rl_library`) and warns when the version that trained
it differs from the installed one. A checkpoint copied out of its run directory loses that record, so move `run.json`
along with the `.pt`.

## Console output: the human and agent log themes

Eden renders its own output, and Genesis', in one of two themes. **`agent` is the default, on a terminal too**,
because Eden's console output is read out of files, CI logs and a model's context window more often than watched:

| | `human` | `agent` |
|---|---|---|
| ANSI color | yes | none |
| Greeting box | yes | skipped |
| Line prefix | `[Eden 🍏] [12:17:11] [INFO] ` | `[Eden 12:17:11] `; the level is named only when it is not INFO |
| Emoji in messages | yes | stripped (status marks such as `✓` / `✕` are kept) |
| Tables (manager summaries, `eden list`) | boxed | a `Title: col \| col` header and one line per row |
| rsl_rl per-iteration block | rsl_rl's own layout | `[it 12/3000] total_steps=... reward=...`, then one line per `Episode_Reward` / `Episode_Termination` / ... group, every number kept |
| `--help` and CLI color | yes | none |

`agent` removes layout, never content. A third setting, `auto`, picks `human` on a terminal and `agent` through a
pipe, honoring `NO_COLOR` and `FORCE_COLOR`.

```bash
EDEN_LOG_THEME=human python train.py           # color for the whole process tree
eden --log-theme human train --task reacher    # one command; the flag goes before the subcommand
```

```python
en.init(logger_theme="human")
```

`en.init(logger_theme=...)` also exports `EDEN_LOG_THEME`, so the help formatters and every subprocess (a DDP worker,
a spawned viewer) follow it. It sets Genesis' theme too; to choose a different Genesis theme, assign
`genesis.styles.style.theme` after `en.init()`.

## Runner options

The task config's `runner_options` describes the learner. For rsl_rl the usual class is [`RslRlOnPolicyRunnerOptions`](../api/options/learning.md)
(`from eden.options.learning.rsl_rl import ...`), with `actor` / `critic` model options and an `algorithm`
(`RslRlPpoAlgorithmOptions`). The fields you most often touch, all on `RslRlBaseRunnerOptions`:

| Field | Default | Meaning |
|-------|---------|---------|
| `seed` | `7100` | Base seed (see [Seeding](#seeding)). |
| `device` | `"auto"` | The learner's device; `"auto"` follows the Genesis backend. |
| `num_steps_per_env` | `24` | Rollout length per env per iteration. |
| `max_iterations` | `30000` | Iterations `eden train` runs. |
| `obs_groups` | `{}` | Which observation groups feed `"actor"` and `"critic"` (and optionally `"rnd_state"`). |
| `clip_actions` | `None` | Clamp actions to `[-clip_actions, clip_actions]`; must be finite and positive, or `None`. |
| `save_interval` | `500` | Iterations between checkpoints. |
| `keep_last_checkpoints` | `1` | Newest checkpoints kept on disk; `None` keeps all. |
| `logger` | `"wandb"` | `"tensorboard"`, `"neptune"` or `"wandb"`; `eden train --logger` overrides it. |
| `wandb_project` / `neptune_project` | `"eden"` | Project names for the two hosted loggers. |

rl_games uses `RlGamesPpoRunnerOptions` (`eden.options.learning.rl_games`), with its own names (`max_epochs`,
`save_frequency`, `horizon_length`, `network`, …) and the same `seed`, `clip_actions` and `keep_last_checkpoints`.
Install it with `pip install 'eden-world[rl-games]'` (or `uv sync --inexact --extra rl-games` in a checkout).

How these options fit into a task config is covered in [Task Configuration](task-config.md).

## Custom algorithms: `algorithm.class_name`

rsl_rl 5.x resolves the algorithm class from a string, so a task can train with its own PPO variant without forking
rsl_rl or touching Eden's runner code:

```python
algorithm=RslRlPpoAlgorithmOptions(
    class_name="my_package.trainers:MyPPO",   # "module.path:ClassName"; default "PPO"
    ...,                                       # the usual PPO fields still apply
)
```

The string is resolved when the runner is built, so the module must be importable in **every** process that builds
the runner: the saved `config.json` names it the same way, so `eden train --resume`, `eden inference` and `--config`
must import it too.

### Where an algorithm lives

An algorithm is not part of a task: a task defines the problem, and the algorithm its paper used is one way to solve
it. An algorithm therefore lives in `eden/learning/wrappers/<library>/<algorithm>/`, written against no particular
task, and a task selects it from its config:

- **The algorithm takes the task's pieces as options.** Whatever ties it to one observation layout (a mirror function,
  which group carries a privileged target) is a field of its options class, not a constant in its code.
- **The task owns those pieces**, next to its config (`eden/tasks/homie/symmetry.py` holds HOMIE's mirror).
- **The task's config names the algorithm its reference used**, so dropping it is a visible config change.
- **`examples/` holds scripts only**, never code a saved config imports: a path into `examples/` resolves only when
  that directory is on `sys.path`, so `eden train --resume` and `eden inference` could not load the config.

Each algorithm package ships an options class whose `class_name` defaults to its implementation, so a task selects it
by using that class:

```python
from eden.learning.wrappers.rsl_rl.him import HIMPPOAlgorithmOptions

algorithm=HIMPPOAlgorithmOptions(
    data_augmentation_func="eden.tasks.homie.symmetry:mirror_observations_and_actions",
    ...,
)
```

The algorithms that ship with Eden (API reference: [eden.learning](../api/learning.md)):

| Options class | Package | Replaces | Used by |
|---------------|---------|----------|---------|
| `HIMPPOAlgorithmOptions` | `eden.learning.wrappers.rsl_rl.him` | `algorithm` (HIMLoco: history estimator, symmetry losses) | `homie` |
| `RslRlAmpPpoAlgorithmOptions` | `eden.learning.wrappers.rsl_rl.amp_ppo` | `algorithm` (PPO plus an adversarial motion prior reward) | `husky` |
| `RslRlRmaActorOptions`, `RslRlRmaPpoAlgorithmOptions` | `eden.learning.wrappers.rsl_rl.rma` | `actor`, `algorithm` (RMA teacher) | `extreme_parkour_teacher` |
| `RslRlRmaDepthDistillationRunnerOptions` | `eden.learning.wrappers.rsl_rl.rma` | `runner_options` (depth-student distillation) | `extreme_parkour_student` |
| `RslRlSimBaActorOptions`, `RslRlSimBaCriticOptions` | `eden.learning.wrappers.rsl_rl.simba` | `actor`, `critic` (SimBaV2) | `weave_hoi` |
| `RslRlMuonPpoAlgorithmOptions` | `eden.learning.wrappers.rsl_rl.muon_ppo` | `algorithm` (Muon + AdamW optimizers) | `weave_hoi` |
| `RslRlFlashSacRunnerOptions` | `eden.learning.wrappers.rsl_rl.flashsac` | `runner_options` (off-policy FlashSAC) | `flashdexretarget` |
| `RslRlContrackRunnerOptions` | `eden.learning.wrappers.rsl_rl_contrack` | `runner_options` (constrained-MDP PPO, external fork) | `contrack` |

FlashSAC's backend is not on PyPI; in a checkout install it with
`uv sync --inexact --extra flashsac --group flashsac-git`. The ConTrack runner needs its rsl_rl fork, which the
`eden.learning.wrappers.rsl_rl_contrack` module docstring names.

### What the runner checks and calls

The class need not subclass rsl_rl's `PPO`: the runner calls only its `construct_algorithm(obs, env, cfg, device)` and
the rollout, update, save and load methods. HIMPPO brings its own actor-critic and rollout storage that way.

Before it builds the env, `eden train` imports every `class_name` given as an import path and checks that the
algorithm class accepts the fields its options carry. Changing `class_name` alone (say, to `"PPO"` on
`RslRlAmpPpoAlgorithmOptions`) is refused, because the options' extra fields would reach a class that does not take
them: give the runner the options class made for the algorithm. The check covers subclasses of rsl_rl's `PPO` and
`Distillation`; a class with its own `construct_algorithm` (HIMPPO) reads and checks its fields there.

Optional hooks Eden reads off these classes:

- `preflight()` on the algorithm options runs at that same pre-build check.
- `for_inference()` on the algorithm options returns the options to **replay** a checkpoint with: `eden inference`,
  `eden deploy` and `make_rsl_rl_runner(..., for_inference=True)` use it. AMP-PPO returns stock PPO, so its
  discriminator, clips and replay buffer are never built for a replay.
- `distributed_unsupported` on the algorithm class, a message: under `torchrun`, `eden train` exits with it before the
  build (`RmaPPO`). An algorithm that does train under DDP extends rsl_rl's parameter broadcast and gradient reduction
  to its own modules (`AmpPPO` syncs its discriminator).
- `adapt_runner_dict(runner_dict, major)` on the algorithm options runs on the runner dict after Eden translates it
  for the installed rsl-rl-lib major version. Eden's translation knows only rsl_rl's own models, so an algorithm whose
  construction differs adapts its own entries here.

!!! note "Across rsl-rl-lib versions"
    rsl-rl-lib 4 and 5 build every algorithm through `construct_algorithm`; 3 builds the model from
    `policy.class_name` and calls `AlgClass(model, storage, device=..., **alg_cfg)`. HIM covers 3 with two factories
    its `adapt_runner_dict` routes to (`HIMActorCritic.from_rsl_rl_v3`, `HIMPPO.from_rsl_rl_v3`). AMP-PPO and RMA need
    rsl-rl-lib 5.5.1 or newer. To check another major version without changing your environment:
    `uv run --with rsl-rl-lib==3.3.0 pytest tests/learning/wrappers/rsl_rl/him/test_ppo.py`.

### Custom runners: `runner_options.class_name`

When the training **loop** itself differs (a distillation that backpropagates through the rollout, a fork of rsl_rl
with its own runner), set the runner options' `class_name` to an import path. `eden train` and `eden inference` then
build that class instead of Eden's `OnPolicyRunner` / `DistillationRunner`:

- It is called as `cls(env, runner_dict, log_dir=..., device=...)` and must provide `learn`, `load` and
  `get_inference_policy`. It composes Eden's runner mixins (checkpoint retention, progress file) itself; they are
  listed in `eden.envs.wrappers.rsl_rl_env.EDEN_RUNNER_MIXINS`.
- A `checkpoint_required` message on the class makes `eden train` exit before anything is written when neither
  `--checkpoint` nor `--resume` is given.
- Under a distributed launch it must compose `DDPRunnerMixin`, or it is refused.
- A bare `class_name` other than `"OnPolicyRunner"` / `"DistillationRunner"` is an error, never a fallback to the
  stock runner. Write an import path with a colon: `"module.path:ClassName"`.
- `eden deploy` loads checkpoints into rsl_rl's stock `OnPolicyRunner`, so it refuses a task whose runner is named by
  import path; replay those checkpoints with `eden inference`.

Examples: `RslRlRmaDepthDistillationRunnerOptions` (`extreme_parkour_student`) and `RslRlContrackRunnerOptions`
(`contrack`).

### Data only the algorithm sees

If the algorithm needs data the actor and critic must not see, publish it as an extra observation group and leave
that group out of `obs_groups`:

```python
runner_options = RslRlOnPolicyRunnerOptions(
    obs_groups={"actor": ["policy"], "critic": ["critic"]},   # the extra group is routed to neither model
    ...
)
```

The full observation TensorDict, extra groups included, reaches the algorithm's `process_env_step`; rsl_rl's own RND
extension reads its `rnd_state` group the same way. Worked example: the `husky` task publishes an `amp` group (a joint
position history window plus a phase flag) that only the AMP discriminator reads (`eden/tasks/husky/config.py` and
`eden/learning/wrappers/rsl_rl/amp_ppo/`).

## Training dynamics that look like bugs

Two behaviors have cost real debugging time in ports because they look like bugs and are not:

- **Early collapse to one-step episodes.** On tasks with per-step action penalties and cheap terminations, PPO can
  briefly learn to terminate immediately, because ending the episode beats paying the penalties while flailing at the
  initial action noise. It recovers on its own as the policy improves. Give a fresh run a few hundred iterations
  before you debug the MDP.
- **Contact-rich rollouts can differ between runs at a fixed seed.** GPU contact-solver atomics can reorder
  floating-point sums between runs, and chaotic contacts (rough terrain especially) amplify the difference. Judge a
  terrain policy on statistics over many episodes and seeds, not on one rollout that differs from the last.

## See also

- [Task Configuration](task-config.md): where `runner_options` sits in a task config
- [Using Eden with Another RL Library](rl-libraries.md): gymnasium, stable-baselines3, skrl, CleanRL
- [Multi-GPU Training](ddp.md): `torchrun` and SLURM
- [Exporting a Policy](policy-export.md): ONNX / TorchScript for your own runtime
- [Deployment](deployment.md): running a checkpoint on a robot, and MuJoCo sim-to-sim
