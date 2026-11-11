# Environments

The environment classes own the Genesis scene, its entities and the managers, and run the step loop. This page is
the class reference; the [environments guide](../guides/environments.md) explains construction, the step-phase
order and auto-reset.

```python
from eden.envs.base import EnvBase, MimicRLEnvBase, RLEnvBase, env_class_for_config
```

Environment classes **construct only**: `from_config(cfg)` returns an unbuilt env, and you call `build()` and
`reset()` yourself. Use `env_class_for_config(cfg).from_config(cfg)` when the class is not known statically. The
training wrappers (below) build and reset inside their own `from_config`.

## Base classes

::: eden.envs.base.EnvBase
    options:
      show_bases: true

::: eden.envs.base.RLEnvBase
    options:
      show_bases: true
      inherited_members: false

::: eden.envs.base.MimicRLEnvBase
    options:
      show_bases: true
      inherited_members: false

::: eden.envs.base.env_class_for_config

## Gaussian-splat environments

Environments that composite a Gaussian-splat background behind the simulated scene, with the gsplat backend
(`SplatEnvBase`, `SplatRLEnvBase`) or the Nyx path tracer (`NyxSplatEnvBase`). They pair with the splat configs on
the [configs page](configs.md#gaussian-splat-configs).

::: eden.envs.splat.SplatEnvBase
    options:
      show_bases: true
      inherited_members: false

::: eden.envs.splat.SplatRLEnvBase
    options:
      show_bases: true
      inherited_members: false

::: eden.envs.nyx_splat.NyxSplatEnvBase
    options:
      show_bases: true
      inherited_members: false

## Debug drawing

::: eden.envs.draw_debug_mixin.DrawDebugMixin
    options:
      show_bases: true

## CUDA-graph replay of the graded phase

`EnvOptions.cuda_graph_options` replays the metric, observation-refresh, termination and reward phase of
`RLEnvBase.step` from a CUDA graph. `GradedPhaseGraph` is the implementation; `BakedHostReadError` is what it raises
when a term reads a value back to the host during capture.

::: eden.envs.graded_graph
    options:
      show_root_heading: false
      members:
        - GradedPhaseGraph
        - BakedHostReadError

## RL-library wrappers

The wrappers adapt an `RLEnvBase` to an RL library's vectorized-env interface. They live in `eden/envs/wrappers/`,
which has no `__init__.py`, so the API generator cannot render them here; read their docstrings in the source.

| Module | Classes and functions | Library |
|---|---|---|
| `eden/envs/wrappers/rsl_rl_env.py` | `RslRlVecEnvWrapper`, `make_rsl_rl_runner`, `load_runner_checkpoint`, `DDPOnPolicyRunner`, `MimicOnPolicyRunner` | rsl-rl-lib |
| `eden/envs/wrappers/rl_games_env.py` | `RlGamesVecEnvWrapper` | rl_games |
| `eden/envs/wrappers/gymnasium_env.py` | `GymnasiumVecEnvWrapper`, `make_vec_env`, `register_gymnasium_envs` | Gymnasium (`gym.make_vec("Eden/<task>")`) |
| `eden/envs/wrappers/sb3_env.py` | `Sb3VecEnvWrapper` | Stable-Baselines3 |
| `eden/envs/wrappers/skrl_env.py` | `SkrlVecEnvWrapper` | skrl |
| `eden/envs/wrappers/vec_env_mixin.py` | `VecEnvMixin`, shared by the Gymnasium, SB3 and skrl wrappers | — |
| `eden/envs/wrappers/core.py` | `EnvWrapperMixin`, `action_space`, `observation_space`, `time_outs` | — |

See [Training runs](../guides/training-runs.md) and [Other RL libraries](../guides/rl-libraries.md).

## See also

- [Environments guide](../guides/environments.md) — construction, step phases, auto-reset
- [Final observations](../guides/environments.md#final-observations) — `EnvOptions.record_final_observations` and `extras["final_observations"]`
- [Environment options](options/envs.md)
- [Managers](managers.md)
