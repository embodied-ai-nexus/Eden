# Environment Options

`EnvOptions` is the `env_options` field of every config: the number of parallel environments, the control and
simulation time steps, the integrator, solver and contact settings, precision-dependent tolerances and the episode
length. `CudaGraphOptions` (its `cuda_graph_options` field) turns on CUDA-graph replay of the graded phase of the
step. The [physics guide](../../guides/physics.md) explains the simulation settings.

```python
from eden.options.envs import EnvOptions
```

`num_envs` is per process: under multi-GPU training each rank builds its own `num_envs` environments.

::: eden.options.envs.EnvOptions

::: eden.options.envs.CudaGraphOptions

## See also

- [Physics guide](../../guides/physics.md)
- [Environments](../envs.md)
- [Multi-GPU training](../../guides/ddp.md)
