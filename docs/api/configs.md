# Configs and Task Registry

A task is an `EdenConfig` subclass: one validated Pydantic object that holds every option group an environment is
built from. This page documents the config hierarchy, its serialization helpers, and the task registry that finds
configs by name. For how to write a task config, read the [task configuration guide](../guides/task-config.md);
for variants and their CLI flags, read the [TaskMods guide](../guides/task-mods.md).

```python
from eden.tasks import TASK_REGISTRY, TaskMod
from eden.utils.configs import EdenConfig, EdenMimicConfig, EdenRLConfig
```

Each config class pairs with exactly one environment class (`EdenConfig` with `EnvBase`, `EdenRLConfig` with
`RLEnvBase`, `EdenMimicConfig` with `MimicRLEnvBase`); a mismatch raises `TypeError` at construction. See
[Environments](envs.md).

## Config hierarchy

::: eden.utils.configs.EdenConfig
    options:
      show_bases: true

::: eden.utils.configs.EdenRLConfig
    options:
      show_bases: true
      inherited_members: false

::: eden.utils.configs.EdenMimicConfig
    options:
      show_bases: true
      inherited_members: false

### Gaussian-splat configs

Configs for environments that render a Gaussian-splat background (see
[Gaussian-splat environments](envs.md#gaussian-splat-environments)).

::: eden.utils.configs.EdenSplatConfig
    options:
      show_bases: true
      inherited_members: false

::: eden.utils.configs.EdenSplatRLConfig
    options:
      show_bases: true
      inherited_members: false

::: eden.utils.configs.EdenNyxSplatConfig
    options:
      show_bases: true
      inherited_members: false

### Reading and converting config files

::: eden.utils.configs
    options:
      show_root_heading: false
      members:
        - read_config_file
        - is_full_config
        - saved_config_qualname
        - overrides_from_str
        - serialize_obj_with_metadata
        - get_options_class_from_qualname
        - to_options

## Task registry

Tasks register themselves with `@TASK_REGISTRY.register()` on the config class. The registry scans the source of
`eden.tasks` (and of every package an installed distribution names under the `eden.tasks` entry-point group) on
first lookup without importing it, so there is no central index to update. `TASK_REGISTRY.build(name, **kwargs)`
returns a configured instance with the task's TaskMods applied, which is what `eden train` does;
`TASK_REGISTRY.get(name)` returns the bare class, and instantiating that yourself skips every mod.

::: eden.tasks.registry.TaskRegistry

::: eden.tasks.registry.TaskMod

## Command-line parsing

The helpers the task commands (`eden train`, `eden inference`, ...) use to turn `--task`, TaskMod flags and config
overrides into a config instance.

::: eden.tasks.parser
    options:
      show_root_heading: false
      members:
        - get_task_argparser
        - get_task_config
        - get_task_config_from_args
        - overrides_from_args
        - read_config_arg
        - explicit_task_mods

## See also

- [Task configuration guide](../guides/task-config.md)
- [TaskMods guide](../guides/task-mods.md)
- [Options reference](options/index.md)
