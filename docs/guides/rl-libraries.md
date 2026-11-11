# Using Eden with Another RL Library

`eden train` drives rsl_rl and rl_games through a task's `runner_options` ([Training Runs](training-runs.md)). Any
other RL library reaches an Eden task through a wrapper around the `RLEnvBase`. This page covers the gymnasium vector
env (for CleanRL or a loop of your own), stable-baselines3 and skrl, and how to add a wrapper for another library.

| Library | Entry point | Install |
|---------|-------------|---------|
| CleanRL, or any loop of your own | `gym.make_vec("Eden/<task>", num_envs=N)` or `GymnasiumVecEnvWrapper` (a gymnasium `VectorEnv` of torch tensors) | core |
| stable-baselines3 | `Sb3VecEnvWrapper` (numpy, one observation group) | `pip install 'eden-world[sb3]'` |
| skrl | `SkrlVecEnvWrapper` (torch, optional critic state) | `pip install 'eden-world[skrl]'` |

In a checkout, install the extras with `uv sync --inexact --extra sb3` or `uv sync --inexact --extra skrl`.

Runnable examples train the `reacher` task with each library:
`examples/recipes/rl_libraries/cleanrl_ppo.py`, `sb3_ppo.py` and `skrl_ppo.py`.

## The gymnasium vector env

```python
import eden as en
import genesis as gs
import gymnasium as gym

from eden.envs.wrappers.gymnasium_env import register_gymnasium_envs

en.init(backend=gs.gpu)
register_gymnasium_envs()                     # every registered task, as "Eden/<task>"
envs = gym.make_vec("Eden/reacher", num_envs=4096, seed=0)

obs, info = envs.reset(seed=0)
obs, reward, terminated, truncated, info = envs.step(actions)   # actions: (4096, action_dim)
```

### Registration

`register_gymnasium_envs()` registers every task in the task registry as `Eden/<task>` and returns the ids it added.
It builds nothing and imports no task module or Genesis, so it is cheap, but `import eden` does not run it: call it
once. Calling it again registers only tasks added since (a task registered at runtime, or one from an installed
package).

### `make_vec`, not `make`

Every Eden env is batched, so `gym.make("Eden/<task>")` raises and names `make_vec`, as does `gym.make_vec` with
`vectorization_mode="sync"` or `"async"`. Do not wrap a `num_envs=1` env to fake a single one: its tensors keep the
leading env dimension. `make_vec` needs `eden.init()` first and takes:

- `num_envs`: replaces the task's `env_options.num_envs`. Gymnasium passes its own default of 1 if you leave it out,
  never the task's.
- `seed`: seeds every RNG before the build ([Seeding](training-runs.md#seeding)), so the build, the first reset and
  domain randomization are reproducible. Without it the process RNG is left alone.
- `clip_actions`: clamps actions to `[-clip_actions, clip_actions]` and bounds the action space by it. Without it
  actions reach the action manager unclamped and the space is unbounded.
- `final_observations` (default on): report `final_obs` in `info` (below). Turning it off saves an observation compute
  per step.
- `random_episode_starts` (default off): stagger the episode clocks on every `reset` (below).
- `show_viewer`, `eval_mode`: as for any env's `from_config`.
- Any other keyword: the task's TaskMod parameters, as `TASK_REGISTRY.build(task, **kwargs)` takes them, for example
  `gym.make_vec("Eden/reacher", num_envs=64, difficulty="hard")`.

From a config you already hold, `GymnasiumVecEnvWrapper.from_config(cfg, clip_actions=..., seed=...)` does the same,
as do the SB3 and skrl wrappers' `from_config`. `cfg` needs no `runner_options` and is left untouched. Each wrapper
also wraps an env you built yourself: `GymnasiumVecEnvWrapper(env)`.

### Tensors stay on the device

Observations, rewards, `terminated` and `truncated` are torch tensors on the sim's device (`envs.device`); actions
may be a tensor or a numpy array. Gymnasium's own vector wrappers (`RecordEpisodeStatistics`, `NormalizeObservation`,
…) are numpy-based and do not apply; the wrapper reports episode statistics itself (below).

Everything `step` and `reset` return is a **copy**, so a learner may keep a step's tensors past the next step (the env
itself reuses its buffers in place). That costs one copy of the observation per step, two with `final_obs`. The SB3
and skrl wrappers copy, and return, only the groups they read.

### Observations are the env's observation groups

`obs` is a dict with one entry per observation group, and `single_observation_space` is a `gymnasium.spaces.Dict` of
them: the actor reads `obs["policy"]`, an asymmetric critic another group. A group that concatenates its terms is a
`Box`; one with `concatenate_terms=False` is a `Dict` of its terms. The spaces are read off a real observation when the
wrapper is constructed (which resets every env), so shapes and dtypes are the manager's.

### Done envs reset in the same step

`metadata["autoreset_mode"]` is gymnasium's `SAME_STEP`: as in `RLEnvBase.step()`, the observation `step` returns for
a done env already starts its next episode. `info` carries what the episode ended with:

| Key | Value |
|-----|-------|
| `final_obs` / `_final_obs` (with `final_observations`) | The observation each done env ended its episode on, and the mask of envs it is valid for: done, and not a state Genesis flagged as corrupt (a solver error, which Eden reports as truncated). Rows outside the mask are zero. |
| `episode` / `_episode` | `{"r": return, "l": length}` of each episode that ended this step (zero elsewhere), and the done mask. |
| `log` | The managers' episode statistics, as the rsl_rl and rl_games wrappers log them. |

`terminated` and `truncated` split the done envs by the termination terms' `time_out` flag. An env can be both on one
step (a failure on the step the episode timed out); treat it as terminal. To bootstrap a time-out correctly, value
`info["final_obs"]`, not the returned observation, for the envs in `truncated & ~terminated & info["_final_obs"]`.
Select with `torch.where` rather than multiplying by the mask, so no other row's value can leak in;
`cleanrl_ppo.py` shows the two lines it takes.

`final_obs` needs `env_options.record_final_observations`, which costs one extra observation compute per step (for a
camera task, a second render). `from_config` and `make_vec` set it to match `final_observations`. Wrapping an env built
without it raises and says so, unless you pass `final_observations=False`. Turn it off for a loop that never reads
`final_obs`: an evaluation rollout, or skrl, whose wrapper leaves it off by default. The SB3 wrapper always needs it.

### Staggered episode starts

Every env resets together and the episode length is fixed, so without help the envs that never terminate early all
time out on the same step, and a learner's per-episode work (SB3 values each ended episode separately) arrives in
bursts. `random_episode_starts=True` starts each env's episode clock at a random step in `[0, max_episode_length)` on
every `reset`, as `eden train` does (rsl_rl's `init_at_random_ep_len`). The first episodes are then shorter, and end
as real time-outs. The clocks are drawn after `reset(seed=...)` seeds the RNGs, so they are reproducible. The three
recipes turn it on.

### Seeding and closing

`reset(seed=s)` seeds the env's RNGs before resetting every env. Eden's envs share one set of RNGs, so a list of
per-env seeds is refused, as are reset `options` (configure the task through its config or TaskMod keywords).

`close()` destroys the scene only when the wrapper built the env (`from_config`, `make_vec`); an env you passed to
`GymnasiumVecEnvWrapper(env)` stays yours to destroy. No wrapper destroys anything on garbage collection, because a
finalizer can run after Genesis has shut down.

## stable-baselines3

```python
from stable_baselines3 import PPO

from eden.envs.wrappers.sb3_env import Sb3VecEnvWrapper
from eden.tasks import TASK_REGISTRY

cfg = TASK_REGISTRY.build("reacher")
cfg.env_options.num_envs = 256
env = Sb3VecEnvWrapper.from_config(cfg, clip_actions=1.0)
model = PPO("MlpPolicy", env, n_steps=24, batch_size=256 * 6)
model.learn(total_timesteps=1_000_000)
```

(Call `eden.init()` first, as above.)

- **The action space must be bounded.** Every SB3 algorithm asserts finite action bounds, so `clip_actions` is
  required; use the task's `runner_options.clip_actions` when it has one. SB3 clips each sampled action to that bound.
- **One observation group**, `obs_group` (default `"policy"`). A `Box` group trains with `"MlpPolicy"`; a group with
  `concatenate_terms=False` is a `Dict` and needs `"MultiInputPolicy"`.
- **Episode ends.** Each done env's info carries `terminal_observation` (SB3 bootstraps time-outs from it, and the
  off-policy algorithms store it as the transition's next observation), `TimeLimit.truncated`
  (`truncated and not terminated`) and `episode`, which SB3 logs as `rollout/ep_rew_mean`; no `VecMonitor` is needed.
  A solver-error env gets no `terminal_observation`, so SB3 does not value its corrupt state. The wrapper therefore
  refuses an env that does not record final observations; `from_config` turns them on.
- **No reset options.** `set_options` with anything in it raises at the next reset.
- **Throughput.** SB3 is numpy-based and single-device: each step copies observations, rewards and dones to the host,
  and SB3 builds one info dict per env. Expect a fraction of `eden train`'s throughput, and use a few hundred envs
  rather than thousands.

`get_attr` returns the batched env's attribute once per env index; `set_attr` and `env_method` raise, since there are
no per-env objects to address.

## skrl

```python
from skrl.utils.runner.torch import Runner

from eden.envs.wrappers.skrl_env import SkrlVecEnvWrapper
from eden.tasks import TASK_REGISTRY

env = SkrlVecEnvWrapper.from_config(TASK_REGISTRY.build("reacher"), state_group="critic")
Runner(env, cfg).run("train")   # cfg: skrl's YAML layout, as a dict
```

- The wrapper keeps everything on the device. The policy observes `obs_group` (default `"policy"`); with `state_group`
  set, that group is skrl's `state`, what a value network with `input: STATES` reads. Both are flattened to
  `(num_envs, D)`.
- Rewards, `terminated` and `truncated` are `(num_envs, 1)`, and `truncated` leaves out an env that also terminated.
- `info` is the shared core's, with every `log` value a 0-d tensor, so a trainer `environment_info: "log"` puts Eden's
  per-term episode statistics (`Episode_Reward/<term>`, …) in skrl's TensorBoard.
- skrl values a time-out from the observation `step` returns, which here (as in Isaac Lab) already starts the next
  episode, whether `time_limit_bootstrap` is on or off. Its agents never read `final_obs`, so the skrl wrapper leaves
  final observations off (`final_observations=True` to have them).
- skrl's `Wrapper` answers any attribute the wrapper lacks from the wrapped env, so `wrapper.episode_length_buf`
  works here but raises `AttributeError` on the gymnasium and SB3 wrappers. Reach the env through `wrapper.env` to
  keep code portable between them.

!!! tip "Normalize the inputs"
    On `reacher`, skrl's PPO needed `RunningStandardScaler` for the observation, state and value preprocessors;
    without them it stayed at a do-nothing policy's reward. `skrl_ppo.py` carries a configuration that learns.

## CleanRL

CleanRL is a set of single-file algorithms, not a library, so there is nothing to adapt: its continuous PPO runs on
`gym.make_vec(...)` as it is, once it indexes `obs["policy"]`. The tensors never leave the device, as in CleanRL's
`ppo_continuous_action_isaacgym.py`, which `cleanrl_ppo.py` follows.

## What these entry points do not do

- `eden train`, `eden inference`, `eden export` and `eden deploy` stay rsl_rl/rl_games-only. The libraries above take
  no `runner_options` and write their checkpoints in their own formats.
- They run on one device: Eden's DDP setup is rsl_rl's.
- The wrappers do not split multi-agent tasks; rl_games' `agents` option does.

## Adding another library

All five wrappers (rsl_rl, rl_games, gymnasium, SB3, skrl) live in `eden/envs/wrappers/` and share
`core.py`'s `EnvWrapperMixin`: the build and env ownership, the action clamp and space, the observation spaces and the
split of `terminated` from timed-out envs. The gymnasium, SB3 and skrl wrappers add `VecEnvMixin`
(`vec_env_mixin.py`): final observations, episode statistics and the copies. To add a library:

1. Mix `VecEnvMixin` in beside the library's base class: `class FooVecEnvWrapper(VecEnvMixin, FooVecEnv)`.
2. In `__init__`, call `_init_vec_env(env, ...)`, after the base class's own `__init__` if that routes attribute
   lookups to the env (as skrl's does). The mixin sets no `num_envs` or `device`: those names belong to the library's
   base class.
3. Build `reset`, `step` and `close` on what `_reset_env`, `_step_env` and `_close_env` return, converted to the
   library's types. A learner that bootstraps time-outs reads `core.time_outs(terminated, truncated)`, which leaves
   out an env that also terminated.
4. Have `from_config` call `_from_config`.
5. Put the wrapper in `eden/envs/wrappers/<library>_env.py`, import the library at module level, declare it as an
   extra in `pyproject.toml`, and `pytest.importorskip` it in the wrapper's test.

## See also

- [Training Runs](training-runs.md): `eden train` and seeding
- [Environments](environments.md): the step loop and auto-reset
- [Observations](observations.md): observation groups and `concatenate_terms`
