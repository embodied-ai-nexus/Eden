# Material Options

A material selects the solver that simulates an entity and that solver's physical parameters: rigid (and
kinematic) bodies, MPM, SPH, PBD and FEM deformables, and the Eden-hosted Mochi solver. Set it as an entity's
`material` field. The [physics guide](../../guides/physics.md) explains each solver.

```python
from eden.options.materials import RigidMaterialOptions

RigidMaterialOptions(gravity_compensation=1.0)  # gravity compensation is a material field
```

A Mochi scene is all-Mochi: every entity, the ground plane included, takes a `Mochi*MaterialOptions`.

::: eden.options.materials
    options:
      show_root_heading: false

## See also

- [Physics guide](../../guides/physics.md)
- [Engine: MochiSolver](../engine.md#mochisolver)
