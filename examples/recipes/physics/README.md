# Physics examples

Eden ports of the examples in the Genesis **Catalogue → Physics** section. Genesis
is the underlying physics engine; Eden wraps it with a declarative, Pydantic-based
options API. Each script here is meant to be short and idiomatic — build an
environment from options, then step it:

```python
env = EnvBase(
    env_options=EnvOptions(...),      # sim dt/substeps + per-solver options
    scene_options=SceneOptions(...),  # entities, each with a material
    show_viewer=args.vis,
)
env.build()
for _ in range(horizon):
    env.scene.step()
```

Every script takes `-v`/`--vis` to open the Genesis viewer, and runs headless
otherwise. Under pytest the step horizon is shortened automatically
(`"PYTEST_VERSION" in os.environ`).

## Catalogue

Mirrors the Genesis "Physics" catalogue (the *Stable Fluid: smoke* example is
intentionally omitted). Scripts are landing in a series of PRs ("rounds"); ✅ =
available, 🔜 = planned.

| Genesis example | Eden script | Solver | Status |
| --- | --- | --- | --- |
| Rigid: franka cube | `rigid_franka_cube.py` | Rigid | ✅ |
| Rigid: collision tower | `rigid_collision_tower.py` | Rigid | ✅ |
| Rigid: contype | `rigid_contype.py` | Rigid | ✅ |
| FEM: hard & soft constraint | `fem_hard_soft_constraint.py` | FEM | ✅ |
| MPM: tutorial | `mpm_tutorial.py` | MPM | ✅ |
| MPM: sand wheel | `mpm_sand_wheel.py` | MPM + rigid | ✅ |
| SPH: rigid | `sph_rigid.py` | SPH + rigid | ✅ |
| SPH + MPM | `sph_mpm.py` | SPH + MPM | ✅ |
| PBD: liquid | `pbd_liquid.py` | PBD | ✅ |
| PBD: cloth | `pbd_cloth.py` | PBD | ✅ |
| IPC: robot cloth teleop | `ipc_robot_cloth_teleop.py` | IPC + FEM + rigid | ✅ |
| Coupler: cloth on rigid | `coupler_cloth_on_rigid.py` | PBD + rigid | ✅ |
| Coupler: rigid + MPM | `coupler_rigid_mpm.py` | MPM + rigid | ✅ |
| Coupler: cut dragon | `coupler_cut_dragon.py` | MPM + rigid | ✅ |
| Coupler: water wheel | `coupler_water_wheel.py` | SPH/MPM + rigid | ✅ |
| Coupler: flush cubes | `coupler_flush_cubes.py` | MPM | ✅ |
| SAP: franka grasp rigid cube | `sap_franka_grasp_cube.py` | SAP + rigid | ✅ |

**Additional IPC demos** (not in the Genesis catalogue, kept here as extra
coverage of the FEM + IPC interface): `ipc_objects_falling.py`,
`ipc_robot_grasp_cube.py`, and the Eden-only keyboard variant
`coupler_cut_dragon_teleop.py`.

## How entities pick a solver

There is **no** separate `MpmEntityOptions`/`FemEntityOptions`. You attach a
*material* to a normal entity (`BoxOptions`, `SphereOptions`, or
`EntityOptions(file=...)`), and the material type alone decides the solver and
the backing entity class:

| Material options | Solver | Entity class |
| --- | --- | --- |
| `RigidMaterialOptions` | Rigid | `RigidEntity` |
| `MPM{Elastic,ElastoPlastic,Liquid,Sand}MaterialOptions` | MPM | `ParticleEntity` |
| `SPHLiquidMaterialOptions` | SPH | `ParticleEntity` |
| `PBD{Liquid,Cloth,Elastic,Particle}MaterialOptions` | PBD | `ParticleEntity` |
| `FEM{Elastic,Cloth,Muscle}MaterialOptions` | FEM | `FEMEntity` |

Solver-level tuning is passed through `EnvOptions` as raw Genesis option objects:
`mpm_options`, `sph_options`, `pbd_options`, `fem_options`, and `coupler_options`
(`gs.options.{IPC,SAP,Legacy}CouplerOptions`).

Particle/FEM entities bake their pose into the morph at creation time
(`default_root_pos` / `default_root_quat`); `set_pos`/`set_quat` raise. Two-way
coupling with rigid bodies is enabled per-entity on `RigidMaterialOptions`
(`needs_coup`, `coup_friction`, `coup_softness`, `coup_type`, `coup_links`).

## IPC coupler

The IPC examples exercise Eden's FEM material + IPC coupler interface
(`EnvOptions.coupler_options=gs.options.IPCCouplerOptions(...)` plus
`RigidMaterialOptions.coup_type`). Requirements and current workarounds:

1. **libuipc** Python bindings must be installed (`uv pip install -e references/libuipc`),
   and the GPU backend is required (`en.init(backend=gs.gpu)`).
2. `batch_dofs_info=False` in `EnvOptions` — IPC does not support batched rigid info.
3. Articulated robots under IPC currently need two documented workarounds, both
   carried in the example scripts:
   - Use a **no-tendon** Panda MJCF — Eden's bundled `panda.xml` uses
     tendon-approximated actuators that break `RigidEntity.post_build`'s
     `get_dofs_kp` read.
   - `sanity_check_enable=False` on `IPCCouplerOptions` — Eden applies
     `default_dofs_pos` *after* `scene.build()`, so IPC's build-time sanity check
     otherwise runs on the neutral (self-colliding) qpos.

## Requirements

- **GPU backend.** All non-rigid solvers (MPM/SPH/PBD/FEM/IPC/SAP) require
  `en.init(backend=gs.gpu)`.
- **IPC** additionally requires libuipc (see above).

## Known Eden gaps

Surfaced while porting these examples. Where a clean options-native path exists,
the example uses it; the items below are the exceptions.

- **Particle emitters — no Eden wrapper (planned).** Genesis' fluid/sand emitter
  (`scene.add_emitter(...)` + `emitter.emit(...)`) has no options-level equivalent
  in Eden. The emitter-based examples (sand wheel, water wheel, flush cubes) reach
  through to `env.scene.add_emitter(...)` directly, with an inline note. A proper
  `EmitterOptions` / `env.add_emitter(...)` API is a tracked follow-up.
- **FEM `update_constraint_targets` not wrapped.** Eden's `FEMEntity` wraps
  `set_vertex_constraints`, but per-step *target updates* (moving a vertex
  constraint along a path) still go through the Genesis entity
  (`entity.entity.update_constraint_targets`). Used by `fem_hard_soft_constraint.py`;
  a one-line wrapper is a cheap follow-up.
- **Particle `vis_mode` default.** Eden's entity `vis_mode` defaults to `"visual"`,
  which the SPH/particle solvers reject — particle entities must set
  `vis_mode="particle"` explicitly (as the examples do).
- **SAP coupler (resolved).** `gs.options.SAPCouplerOptions` flows through
  `coupler_options` and is exercised by `sap_franka_grasp_cube.py`. Two `en.init`
  params were added to make it usable: `precision="64"` (SAP needs double precision)
  and `eps=1e-15` — Eden's default `eps=1e-8` (a rigid-solver stability choice) is
  coarser than SAP's `1e-10` convergence tolerances and makes contacts explode.
