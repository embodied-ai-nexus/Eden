# Scene Presets

Scene presets are functions that return a `SceneOptions` with a ground plane (or a full room) already in it. The
robot is the `robot` argument (`left_robot` / `right_robot` on the two-arm `BimanualEmpty` and `LBMEval`), and
every other entity is a keyword argument named after it. The core presets are also exported as short names: `en.scenes.Empty`,
`BimanualEmpty`, `Tabletop`, `KitchenCountertop` and `LBMEval`. The
[scenes and terrains guide](../../guides/scenes-and-terrains.md) covers scenes, objects and terrain.

```python
import eden as en

# excerpted from eden/tasks/benchmark/reacher/config.py
scene_options = en.scenes.Empty(
    is_support_enabled=True,
    robot=en.robots.Yam(default_root_pos=(0.0, 0.0, 0.0), default_root_quat=(1.0, 0.0, 0.0, 0.0)),
)
```

## Core presets

::: eden.options.scenes.empty
    options:
      show_root_heading: false

::: eden.options.scenes.tabletop
    options:
      show_root_heading: false

::: eden.options.scenes.lbm_eval
    options:
      show_root_heading: false

## Benchmark scenes

Scene constructors derived from benchmark environments. A scene preset alone is not a registered task; the task
suites are documented under [Tasks](../../tasks/index.md).

### LIBERO

::: eden.options.scenes.libero.base
    options:
      show_root_heading: false

### CALVIN

::: eden.options.scenes.calvin.base
    options:
      show_root_heading: false

The ReplicaCAD and RoboCasa scene modules (`eden/options/scenes/replica_cad/`, `robocasa/`, `robocasa365/`) have no
`__init__.py`, so they are not rendered here; read them in the source.

## See also

- [Scenes and terrains guide](../../guides/scenes-and-terrains.md)
- [Entity options](entities.md)
