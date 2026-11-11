# Engine

`eden.engine` holds Genesis engine components that upstream Genesis does not ship: code that subclasses a Genesis
engine class and runs inside `scene.step()`. It has two parts, the audio sensors and the MochiSolver. The
[engine addons guide](../guides/engine-addons.md) explains what belongs here and how each part plugs into stock
Genesis; the [physics guide](../guides/physics.md) covers when to use Mochi.

## Audio sensors

Importing `eden.engine.sensors` installs every sensor below: a Genesis `Sensor` subclass registers itself with
Genesis' sensor manager on definition, so you pass these options as `SensorOptions(sensor=...)` like any
`gs.sensors.*` options. The sensors form one audio pipeline: synthesis (`ActuationAudio`, `ContactAudio`,
`PianoAudio`), propagation (`AudioContext`) and rendering (`SpatialAudio`).

```python
from eden.engine.sensors import ContactAudio, SpatialAudio
from eden.options.sensors import SensorOptions
```

### Motor and gearbox noise

::: eden.engine.sensors.actuation_audio
    options:
      show_root_heading: false

### Contact sound

::: eden.engine.sensors.contact_audio
    options:
      show_root_heading: false

### Piano keyboard

::: eden.engine.sensors.piano_audio
    options:
      show_root_heading: false

### Audio sources and propagation

::: eden.engine.sensors.audio
    options:
      show_root_heading: false

### Microphone

::: eden.engine.sensors.spatial_audio
    options:
      show_root_heading: false

### Live playback

::: eden.engine.sensors.playback
    options:
      show_root_heading: false

## MochiSolver

A fully implicit multi-physics solver hosted by Eden on stock Genesis: each substep solves one nonlinear system over
the rigid and articulated bodies and the vertex positions of deformable ones (tetrahedral solids, thin shells,
discrete elastic rods), with a smooth signed-distance penalty contact model. Eden installs it in a scene on the
scene's first Mochi entity. Every entity in a Mochi scene, the ground plane included, takes a `Mochi*MaterialOptions`
([material options](options/materials.md)), and the default contact stiffness needs `eden.init(precision="64")`.

The Eden-side entity wrappers are `MochiRigidEntity` and `MochiSoftEntity` on the [entities page](entities.md#mochi-entities).

### Options

::: eden.engine.solvers.mochi.options
    options:
      show_root_heading: false

### Materials

::: eden.engine.solvers.mochi.materials.base
    options:
      show_root_heading: false

::: eden.engine.solvers.mochi.materials.rigid
    options:
      show_root_heading: false

::: eden.engine.solvers.mochi.materials.elastic
    options:
      show_root_heading: false

::: eden.engine.solvers.mochi.materials.shell
    options:
      show_root_heading: false

::: eden.engine.solvers.mochi.materials.rod
    options:
      show_root_heading: false

### Rod morph

::: eden.engine.solvers.mochi.morphs
    options:
      show_root_heading: false

### Hosting on stock Genesis

::: eden.engine.solvers.mochi.hosting
    options:
      show_root_heading: false

### Solver, entities and states

::: eden.engine.solvers.mochi.mochi_solver.MochiSolver
    options:
      show_bases: true
      inherited_members: false

::: eden.engine.solvers.mochi.mochi_entity.MochiEntity
    options:
      show_bases: true
      inherited_members: false

::: eden.engine.solvers.mochi.mochi_soft_entity.MochiSoftEntity
    options:
      show_bases: true
      inherited_members: false

::: eden.engine.solvers.mochi.states
    options:
      show_root_heading: false

### Error codes

The flags the solver's kernels set in the per-environment `errno` field.

::: eden.engine.solvers.mochi.error_codes
    options:
      show_root_heading: false

## See also

- [Engine addons guide](../guides/engine-addons.md)
- [Physics guide](../guides/physics.md)
- [Sensors and rendering guide](../guides/sensors-and-rendering.md)
