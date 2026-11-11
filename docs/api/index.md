# API Reference

This section is generated from Eden's source and docstrings. Each page opens with what the module family is for and
links the guide that explains it, then renders the classes and functions; follow a symbol's source link to read its
implementation. Start from a guide when you are learning a concept, and come here for exact signatures, fields and
defaults.

**To find a term's parameters**, the [term catalog](terms/catalog.md) (or `eden terms --grep <text>`) is faster
than browsing the term pages.

```python
import eden as en  # import Eden before genesis and torch
import genesis as gs

en.init(backend=gs.gpu)  # once per process, before building a scene; the default backend is gs.cpu
```

## Core

| Page | Modules | What it covers |
|---|---|---|
| [Configs and task registry](configs.md) | `eden.utils.configs`, `eden.tasks` | `EdenConfig` / `EdenRLConfig` / `EdenMimicConfig`, serialization, `TASK_REGISTRY`, `TaskMod` |
| [Environments](envs.md) | `eden.envs` | `EnvBase`, `RLEnvBase`, `MimicRLEnvBase`, splat envs, RL-library wrappers |
| [Entities](entities.md) | `eden.entities` | rigid, terrain, deformable and Mochi entities; cameras, overlays, sensors |
| [Managers](managers.md) | `eden.managers` | every manager, its term base classes, the term registries |
| [Constants and types](constants.md) | `eden.constants`, `eden.types` | `EventMode`, `MetricDirection`, `MetricMode`, `ReferenceSource`, ... |

## Options

| Page | Module | What it covers |
|---|---|---|
| [Options overview](options/index.md) | `eden.options.options` | `ConfigurableOptions`, strict vs. dynamic options |
| [Environment](options/envs.md) | `eden.options.envs` | `EnvOptions`, `CudaGraphOptions` |
| [Entities](options/entities.md) | `eden.options.entities` | entity, primitive, terrain, grouped and composite options; `SceneOptions` |
| [Materials](options/materials.md) | `eden.options.materials` | rigid, MPM, SPH, PBD, FEM and Mochi materials |
| [Surfaces](options/surfaces.md) | `eden.options.surfaces` | appearance presets |
| [Scenes](options/scenes.md) | `eden.options.scenes` | scene presets |
| [Robots](options/robots.md) | `eden.options.robots` | robot presets and composite rigs |
| [Actuators](options/actuators.md) | `eden.options.actuators` | motor specs and presets |
| [Sensors and cameras](options/sensors.md) | `eden.options.sensors`, `eden.options.camera` | sensor and camera options |
| [Splats and renderers](options/splat.md) | `eden.options.splat`, `eden.options.renderer` | Gaussian splats, point clouds, ray tracer, batch renderer |
| [Managers](options/managers.md) | `eden.options.managers` | manager and term options |
| [Learning](options/learning.md) | `eden.options.learning` | rsl_rl and rl_games runner options |

## Terms

| Page | Module | What it covers |
|---|---|---|
| [Built-in terms](terms/index.md) | `eden.terms`, `eden.managers.terms` | using built-in terms, function wrappers |
| [Term catalog](terms/catalog.md) | generated | every registered term and its parameters |
| [Actions](terms/actions.md), [observations](terms/observations.md), [rewards](terms/rewards.md), [terminations](terms/terminations.md) | `eden.managers.terms.*` | per-kind term reference |
| [Commands](terms/commands.md), [events](terms/events.md), [curricula](terms/curricula.md), [metrics](terms/metrics.md), [recorders](terms/recorders.md) | `eden.managers.terms.*` | per-kind term reference |
| [Modifiers](terms/modifiers.md) | `eden.managers.modifiers` | action modifiers, observation noise and modifiers |

## Data, engine and learning

| Page | Module | What it covers |
|---|---|---|
| [Datasets](datasets.md) | `eden.datasets` | motion dataset loaders, dataset modifiers, body models |
| [Motion reference](motion_reference.md) | `eden.managers.motion_manager`, `eden.managers.motion` | the motion manager and its collaborators |
| [Engine](engine.md) | `eden.engine` | audio sensors, the MochiSolver |
| [Learning](learning.md) | `eden.learning` | policy export, algorithms built on rsl-rl-lib |

## Extensions

[Extensions overview](extensions/index.md) · [Deployment](extensions/deployment.md) ·
[Policies](extensions/policies.md) · [Retargeting](extensions/retargeting.md) ·
[System identification](extensions/sysid.md) · [Motion planning](extensions/motion-planning.md) ·
[Motion optimization](extensions/motion-optimization.md) · [Grasp sampling](extensions/grasp-sampling.md) ·
[Pose estimation](extensions/pose-estimation.md) · [Scripted demonstrations](extensions/scripted-demos.md) ·
[Teleoperation](extensions/teleop.md) · [Visualization](extensions/visualization.md)

## Utilities

| Page | Modules | What it covers |
|---|---|---|
| [Core](utils/core.md) | `eden`, `eden.utils.registry`, `eden.utils.common`, ... | `eden.init`, registries, configurables, DDP, profiling |
| [Math and geometry](utils/math.md) | `eden.utils.geom`, `eden.utils.symmetry`, ... | quaternions (wxyz), transforms, symmetry, sampling |
| [Robotics and assets](utils/robotics.md) | `eden.utils.actuator`, `eden.utils.terrains`, `eden.utils.assets`, ... | actuator math, meshes, terrain generators, asset download |
| [Teleoperation devices](utils/devices.md) | `eden.utils.devices` | keyboard, gamepad, SpaceMouse, GELLO, Meta Quest, PICO, Wuji Glove |
| [I/O](utils/io.md) | `eden.utils.file_handler`, `eden.utils.lerobot`, ... | episode files, LeRobot conversion, images |
| [Rendering](utils/rendering.md) | `eden.utils.batch_splat_renderer`, ... | batched splat rendering, PLY files, background shadows |
| [CLI internals](utils/cli.md) | `eden.__main__`, `eden.utils.cli` | the command table and how to add a subcommand |

## What is not rendered

A few directories have no `__init__.py`, so the generator cannot load them as part of the `eden` package:
`eden/envs/wrappers/`, `eden/utils/motion_reference/`, `eden/utils/buffers/`, `eden/options/objects/`,
`eden/extensions/deployment/utils/` and the ReplicaCAD and RoboCasa scene modules. The pages above name their main
classes and point at the source.
