# Options

Every configurable piece of Eden (the env, the scene and its entities, materials, sensors, managers and their terms,
runners) is described by an options class: a Pydantic model that validates when you construct it and serializes into
the run's saved config. This page documents the base classes; the pages in this section document each family.

```python
from eden.options.envs import EnvOptions
from eden.options.entities import BoxOptions, SceneOptions
```

## Strict and dynamic options

`ConfigurableOptions` defaults to `extra="forbid"`: a misspelled field raises at construction and names the fields
the class does accept. Containers that hold *named children* (`SceneOptions` holds entities, `*ManagerOptions` hold
terms, `SensorsOptions` and `CamerasOptions` hold sensors and cameras) subclass `DynamicConfigurableOptions`, which
allows extra fields; each extra is one child, keyed by its name. Genesis' own options classes (`gs.morphs.*`,
`gs.sensors.*`) keep Genesis' strict validation.

Copy an options object with `clone()` (or `copy.deepcopy`), not with Pydantic's `model_copy(deep=True)`; and use
`updated(**changes)` to get a validated copy with some fields changed. `lock()` freezes an options tree and
`unlock()` thaws it. The [task configuration guide](../../guides/task-config.md) covers serialization.

::: eden.options.options
    options:
      show_root_heading: false

## Pages

| Page | Module | Contents |
|---|---|---|
| [Environment](envs.md) | `eden.options.envs` | `EnvOptions` (simulation, solver, batching), `CudaGraphOptions` |
| [Entities](entities.md) | `eden.options.entities` | entity, primitive, terrain, grouped and composite options, `SceneOptions` |
| [Materials](materials.md) | `eden.options.materials` | rigid, kinematic, MPM, SPH, PBD, FEM and Mochi materials |
| [Surfaces](surfaces.md) | `eden.options.surfaces` | surface appearance presets |
| [Scenes](scenes.md) | `eden.options.scenes` | scene presets |
| [Robots](robots.md) | `eden.options.robots` | robot presets and composite rigs |
| [Actuators](actuators.md) | `eden.options.actuators` | actuator specs and motor presets |
| [Sensors and cameras](sensors.md) | `eden.options.sensors`, `eden.options.camera` | sensor and camera options |
| [Splats and renderers](splat.md) | `eden.options.splat`, `eden.options.renderer` | Gaussian splats, point clouds, ray tracer, batch renderer |
| [Managers](managers.md) | `eden.options.managers` | manager and term options |
| [Learning](learning.md) | `eden.options.learning` | rsl_rl and rl_games runner options |

Dataset options are on the [datasets page](../datasets.md#options) and extension options on each
[extension page](../extensions/index.md).

## Resolving options classes by name

`eden.options.catalog` answers "which options class does this name mean?" for the `eden view` and `eden snap`
commands and for robots: `ROBOT_REGISTRY` maps a robot key to its class, and the curated `MANIFEST` describes assets
semantically (category, placement, tags).

::: eden.options.catalog
    options:
      show_root_heading: false
      members:
        - ROBOT_REGISTRY
        - RobotRegistry
        - MANIFEST
        - AssetManifest
        - AssetSpec
        - find_entity_options
        - entity_options_index

## See also

- [Task configuration guide](../../guides/task-config.md)
- [Configs and task registry](../configs.md)
