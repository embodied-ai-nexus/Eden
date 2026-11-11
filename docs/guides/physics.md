# Physics

This page covers the simulation settings of an Eden environment: time stepping, the backend and floating-point
precision, which physics solver an entity runs in, the rigid solver's contact and integrator settings, the
Eden-hosted Mochi solver, and what happens when a solver fails. Read it when you write the `env_options` of a task,
port a task from another simulator, or add deformable bodies to a scene.

Almost everything here is a field of [`EnvOptions`](../api/options/envs.md), which `EnvBase` turns into Genesis'
`SimOptions`, `RigidOptions` and the per-solver options when it creates the scene. Per-entity physical properties
(mass, armature, friction, contact softness, gains) live on the entity options instead: see
[Entities](entities.md).

## Time stepping

Three fields decide how simulated time advances:

| Field | Default | Meaning |
|---|---|---|
| `sim_dt` | `0.005` | Length of one `scene.step()`, in seconds. |
| `sim_substeps` | `1` | Genesis substeps inside one `scene.step()`. Every solver integrates at `sim_dt / sim_substeps`. |
| `decimation` | `4` | `scene.step()` calls per control step (action repeat). |

The control step is `env.dt = decimation * sim_dt`, so the control frequency is `1 / (sim_dt * decimation)`.
With the defaults a policy acts at 50 Hz on 200 Hz physics. The episode horizon `episode_length_s` (default `20.0`)
becomes `max_episode_length = ceil(episode_length_s / env.dt)` control steps on an `RLEnvBase`; it must be positive
and finite.

The reacher task (`eden/tasks/benchmark/reacher/config.py`) runs physics at 100 Hz with two substeps and acts on
every physics step:

```python
from eden.options.envs import EnvOptions

env_options = EnvOptions(
    num_envs=4096,
    episode_length_s=9.0,
    iterations=10,
    ls_iterations=20,
    sim_dt=0.01,
    sim_substeps=2,
    decimation=1,
    max_collision_pairs=30,
    multiplier_collision_broad_phase=12,
    enable_multi_contact=True,
)
```

LIBERO (`eden/tasks/libero/config.py`) does the opposite: `sim_dt=0.002` and `decimation=25`, a 20 Hz policy on
500 Hz physics.

Two things are easy to get wrong:

- **The substep dt is what stiff constraints see.** Genesis clamps every constraint time constant to
  `2 * sim_dt / sim_substeps`, so a stiff contact only takes full effect when the substep is short enough.
  Closed-loop grippers such as the Robotiq 2F-85 need a substep dt of 0.001 s or less (see
  [Closed-loop grippers](entities.md#closed-loop-grippers-and-constraint-fidelity)).
- **A substep callback runs per `scene.step()`, not per Genesis substep.** `env.register_substep_callback(fn)`
  runs `fn` around each of the `decimation` scene steps; Genesis' own `sim_substeps` happen inside one call and are
  never visible to it.

`num_envs` (default `4096`) is the number of parallel environments. Under multi-GPU training it is the count
**per rank** (see [Multi-GPU training](ddp.md)).

## Backend and precision: `eden.init()`

`eden.init()` initializes Genesis and torch for the process. Call it once, before you build an environment; a second
call warns, does not re-initialize Genesis, and moves the run's log directory.

```python
import eden as en
import genesis as gs

en.init(backend=gs.gpu)                     # the common case
en.init(backend=gs.gpu, precision="64")     # double precision (Mochi at its default stiffness, SAP)
```

| Argument | Default | Meaning |
|---|---|---|
| `backend` | `gs.cpu` | Genesis backend. The `eden` CLI commands pass `gs.gpu` unless you give `--cpu`. |
| `precision` | `"32"` | Floating-point precision of the simulation, `"32"` or `"64"`. |
| `eps` | `1e-8` | Genesis' global epsilon, chosen for rigid-solver stability. |
| `allow_tf32` | `True` | Let torch use TF32 matmuls on Ampere and newer GPUs. |
| `deterministic` | `False` | Use deterministic torch and Genesis algorithms. |
| `performance_mode` | `False` | Passed to `gs.init`. The CLI training and evaluation commands set it. |

After `eden.init()`, torch's default dtype is Genesis' float type, so a dtype-less `torch.zeros(n)` matches Genesis'
tensors under `precision="64"`. The default device is **not** set: a tensor created without `device=` is a CPU tensor.
Use `env.device` (or `gs.device`).

When you need `precision="64"`:

- **Mochi at its default contact stiffness.** The Mochi materials default to `penalty_coefficient=1e9` Pa/m, which
  makes the Newton system ill-conditioned in single precision; the solver warns at build when a stiffness of
  `1e8` or more runs at 32 bits. Either initialize with `precision="64"` or lower the stiffness, as the Mochi tasks
  do (see [Mochi](#mochi-the-eden-hosted-implicit-solver)).
- **The SAP coupler.** `examples/recipes/physics/sap_franka_grasp_cube.py` initializes with
  `en.init(backend=gs.gpu, precision="64", eps=1e-15)`: SAP needs double precision, and Eden's default `eps=1e-8`
  is coarser than SAP's convergence tolerances, which makes contacts explode.

The `eden train` / `eden inference` commands always initialize at `precision="32"`; there is no CLI flag for it.
A task that only works in double precision has to run from its own script. Some tasks refuse double precision
outright: the art tasks raise at build when Eden was initialized at `precision="64"`.

## Which solver an entity runs in

Genesis has one solver per physics family. You never pick a solver directly: the **material** of an entity decides
it, and with it the Eden entity class that wraps the entity.

| Material options (`eden.options.materials`) | Solver | Eden entity class |
|---|---|---|
| `RigidMaterialOptions` (the default), `KinematicMaterialOptions` | rigid | `RigidEntity` |
| `MPMElasticMaterialOptions`, `MPMElastoPlasticMaterialOptions`, `MPMLiquidMaterialOptions`, `MPMSandMaterialOptions` | MPM | `ParticleEntity` |
| `SPHLiquidMaterialOptions` | SPH | `ParticleEntity` |
| `PBDLiquidMaterialOptions`, `PBDClothMaterialOptions`, `PBDElasticMaterialOptions`, `PBDParticleMaterialOptions` | PBD | `ParticleEntity` |
| `FEMElasticMaterialOptions`, `FEMClothMaterialOptions`, `FEMMuscleMaterialOptions` | FEM | `FEMEntity` |
| `MochiRigidMaterialOptions` | Mochi | `MochiRigidEntity` |
| `MochiElasticMaterialOptions`, `MochiShellMaterialOptions`, `MochiRodMaterialOptions` | Mochi | `MochiSoftEntity` |

A terrain (`TerrainOptions`) always builds as a `Terrain`, a rigid entity. There are no separate "MPM entity" or
"FEM entity" options: you put the material on an ordinary entity (`BoxOptions`, `SphereOptions`,
`EntityOptions(file=...)`). Solver-wide settings go on `EnvOptions` as Genesis option objects: `mpm_options`,
`sph_options`, `pbd_options`, `fem_options` (`gs.options.MPMOptions` and so on), `mochi_options`
(`MochiOptions`, Eden's own), and `coupler_options`.

From `examples/recipes/physics/mpm_tutorial.py`, three MPM bodies on a rigid plane:

```python
import genesis as gs

from eden.options import EnvOptions, SceneOptions
from eden.options.entities import BoxOptions, PlaneOptions, SphereOptions
from eden.options.materials import MPMElasticMaterialOptions, MPMElastoPlasticMaterialOptions, MPMLiquidMaterialOptions

env_options = EnvOptions(
    num_envs=1,
    sim_dt=4e-3,
    sim_substeps=10,
    mpm_options=gs.options.MPMOptions(lower_bound=(-0.5, -1.0, 0.0), upper_bound=(0.5, 1.0, 1.0)),
)
scene_options = SceneOptions(
    plane=PlaneOptions(),
    elastic=BoxOptions(
        size=(0.2, 0.2, 0.2),
        default_root_pos=(0.0, -0.5, 0.25),
        material=MPMElasticMaterialOptions(),
        surface=gs.surfaces.Default(color=(1.0, 0.4, 0.4), vis_mode="visual"),
        vis_mode="visual",
    ),
    liquid=BoxOptions(
        size=(0.3, 0.3, 0.3),
        default_root_pos=(0.0, 0.0, 0.25),
        material=MPMLiquidMaterialOptions(),
        surface=gs.surfaces.Default(color=(0.3, 0.3, 1.0), vis_mode="particle"),
        vis_mode="particle",
    ),
    plastic=SphereOptions(
        radius=0.1,
        default_root_pos=(0.0, 0.5, 0.35),
        material=MPMElastoPlasticMaterialOptions(),
        surface=gs.surfaces.Default(color=(0.4, 1.0, 0.4), vis_mode="particle"),
        vis_mode="particle",
    ),
)
```

`examples/recipes/physics/` has one script per solver and coupler (rigid, FEM, MPM, SPH, PBD, IPC, SAP); its
`README.md` lists them. They run on `backend=gs.gpu`, except the rigid collision tower and the keyboard-teleop
variant of the MPM dragon cut, which run on the CPU.

!!! warning "Deformable bodies are posed at creation"
    `ParticleEntity.set_pos()` / `set_quat()` raise (MPM, SPH, PBD), and so does `FEMEntity.set_quat()`. Give the
    pose through `default_root_pos` / `default_root_quat`, which is baked into the morph when the entity is created.
    The DOF and constraint-solver fields of the entity options are refused on a non-rigid material. See
    [Deformable entities](entities.md#deformable-entities) for their API.

### Couplers

`coupler_options` decides how the solvers of one scene exchange forces. `None` (the default) is Genesis' legacy
coupler; `gs.options.SAPCouplerOptions` and `gs.options.IPCCouplerOptions` are the alternatives. Two-way coupling of a
rigid body is configured on its `RigidMaterialOptions` (`needs_coup`, `coup_friction`, `coup_softness`,
`coup_restitution`, and, under IPC only, `coup_type`, `coup_links`, `enable_coup_collision`,
`coup_collision_links`, `contact_resistance`).

`EnvOptions` refuses two IPC settings at construction: `batch_dofs_info=True` (set it `False`) and `num_envs > 1`
(the IPC coupler can only reset the whole scene). The IPC recipes also need the libuipc Python bindings installed.

## The rigid solver

Every robot, object, terrain and ground plane with a rigid material runs in Genesis' rigid solver. These
`EnvOptions` fields configure it.

### Constraint solver

| Field | Default | Meaning |
|---|---|---|
| `solver` | `"newton"` | Constraint solver: `"newton"` or `"cg"`. |
| `iterations` | `50` | Maximum constraint-solver iterations. |
| `tolerance` | `None` | Convergence tolerance, as a ratio against the scene's free-motion cost. `None` lets Genesis pick from the precision (`1e-6` single, `1e-9` double). |
| `ls_iterations` | `50` | Line-search iterations. |
| `ls_tolerance` | `1e-2` | Line-search tolerance. |
| `noslip_iterations` | `0` | Iterations of the noslip post-pass, which suppresses slip and drift. `0` disables it. |

The noslip pass is a cheap fix for objects creeping in a grasp; the LIBERO, SimplerEnv and PolaRiS ports run with
`noslip_iterations=5`.

### Integrators and implicit damping

`integrator` selects how the rigid solver advances velocities:

- `"approximate_implicitfast"` (default): Genesis' own fast scheme.
- `"implicitfast"`: MuJoCo's integrator of that name. It integrates joint damping and the position actuators'
  velocity gain implicitly, as MuJoCo does. Use it for tasks ported from a MuJoCo stack.
- `"euler"`: MuJoCo's explicit Euler. The PD controller's damping is integrated explicitly.

!!! warning "Implicit damping changes armature under `euler` only"
    Under `integrator="euler"`, an actuated entity's `post_build()` adds `kd * substep_dt` to the armature of its
    DOFs, which emulates implicit damping and keeps the PD loop stable. The armature the solver holds then differs
    from the configured `default_dofs_armature`. Under the two implicit integrators nothing is added.

    Only damping Genesis applies itself is ever implicit: the `kv` of a DOF Genesis drives in position or velocity
    mode. A torque-mode action term (one that calls `control_dofs_force`, such as `ExplicitPDController` or
    `OperationalSpaceController`) computes `-kd * qdot` in Python, so its damping is explicit under every
    integrator. Read `entity.get_dofs_armature()` on the built entity to see what the solver actually uses. See
    [Implicit damping](entities.md#implicit-damping) and [Actions](actions.md).

### Collision detection

| Field | Default | Meaning |
|---|---|---|
| `use_gjk_collision` | `True` | Use GJK for convex collision; `False` uses MPR+GJK. |
| `max_collision_pairs` | `50` | Contact capacity per environment. |
| `multiplier_collision_broad_phase` | `8` | Broad-phase capacity multiplier. |
| `enable_multi_contact` | `True` | Generate several contact points per pair. |
| `enable_self_collision` | `True` | Collide the links of one entity with each other. |
| `enable_neutral_collision` | `False` | Keep self-collision pairs that already overlap at the neutral pose. |

A contact capacity that overflows is a *misconfiguration*: the env keeps its error flag across the reset and the
next control step raises with a message naming the option to raise (see [Solver errors](#solver-errors-and-recovery)).

!!! warning "`enable_neutral_collision` is scene-wide"
    With the default `False`, Genesis drops every self-collision pair that overlaps at an entity's neutral
    configuration (qpos0), using a voxel approximation that can also drop a pair that only comes within millimetres.
    `True` keeps the asset's full collision set, but for **every** articulated entity in the scene: a second robot
    whose collision meshes genuinely interpenetrate at qpos0 then fights itself from the first step.

### Friction

| Field | Default | Meaning |
|---|---|---|
| `friction_cone` | `"pyramidal"` | `"pyramidal"` (robust, easy to solve) or `"elliptic"` (the exact isotropic cone). |
| `contact_resolution` | `None` | `"signorini"` or `"convex"`. `None` resolves to `"signorini"` with the elliptic cone and the Newton solver, `"convex"` otherwise. |
| `impratio` | `None` | Ratio of frictional to normal constraint impedance (MuJoCo's `impratio`). `None` lets Genesis pick: 100 with the elliptic cone, 1 otherwise. |
| `enable_torsional_friction` | `False` | Add one constraint axis per contact that resists spin about the normal. |
| `enable_rolling_friction` | `False` | Add two axes that resist rolling. Requires `enable_torsional_friction`. |

`"signorini"` bounds friction by the normal force the contact has developed, so fast sliding never inflates the
normal force: a body launched horizontally decelerates at mu times g instead of lifting off. It needs
`friction_cone="elliptic"` and `solver="newton"`, and costs extra solver iterations. The elliptic cone is not
supported with the noslip pass (`noslip_iterations > 0`) or with `requires_grad`.

Friction coefficients themselves are per geom, on the entity and its material; the entities guide covers them. In
short:

- **A contact pair takes the `max` of the two coefficients**, not the minimum or the mean
  ([Friction combines as `max`](entities.md#friction-combines-as-max-not-min)).
- **The two switches above only add constraint axes.** The strength is per geom (`friction_torsional` /
  `friction_rolling`, in meters), MJCF geoms without `condim >= 4` parse to 0, and a coefficient set while its switch
  is off raises at build
  ([Torsional and rolling friction](entities.md#torsional-and-rolling-friction-are-opt-in-and-their-strength-is-per-geom)).

### Contact softness

`constraint_timeconst` (default `None`) is the rigid solver's global constraint time constant. `None` resolves to
`max(0.01, 2 * sim_dt / sim_substeps)`.

!!! danger "Stiffen contacts per object, never globally"
    Lowering `EnvOptions.constraint_timeconst` to stiffen one grasp ejects mesh objects on contact elsewhere in the
    scene. Stiffen the objects that need it through the entity's `default_geoms_sol_params` instead, on both sides of
    the pair: see [Closed-loop grippers and constraint fidelity](entities.md#closed-loop-grippers-and-constraint-fidelity).

### Per-environment physical properties

Genesis stores link, joint and DOF properties once for the whole batch unless asked otherwise. Randomizing them per
environment needs the matching switch:

| Field | Default | Needed for |
|---|---|---|
| `batch_links_info` | `False` | Per-env link mass, inertia and center of mass. |
| `batch_joints_info` | `False` | Per-env joint properties. |
| `batch_dofs_info` | `True` | Per-env DOF gains, armature, damping, limits. |

An event term that writes one of these declares the flag it needs, `eden terms` lists it on the term, and the env
refuses to build without it, naming the flag. For example `RandomizeLinkMassScale`, `RandomizeMassShift` and
`RandomizeComShift` need `batch_links_info=True`. See [Events](events.md).

### Other rigid-solver switches

- `use_hibernation` (default `False`) is passed to Genesis' `RigidOptions.use_hibernation`.
- **Gravity compensation is a material field**, `RigidMaterialOptions(gravity_compensation=1.0)` cancels gravity on
  the entity. It is not a morph, `RigidOptions` or `add_entity` argument
  ([Gravity compensation](entities.md#per-entity-gravity-compensation)).
- `requires_grad` (default `False`) turns on Genesis' differentiable mode. The Mochi solver and the elliptic cone
  do not support it.

## Mochi: the Eden-hosted implicit solver

The MochiSolver is a fully implicit multi-physics solver that Eden hosts on stock Genesis
(`eden/engine/solvers/mochi`). Every substep it solves one nonlinear system over all rigid and articulated bodies,
tetrahedral solids, thin shells and discrete elastic rods, with a smooth signed-distance penalty contact model.
Contact is re-detected at every Newton iterate and enters the same system as the inertia, which keeps large time
steps stable and couples deformable bodies to rigid ones without a coupler. Genesis does not ship it; how Eden
installs it on one scene at a time is described in [Engine add-ons](engine-addons.md). This section covers the
scene and solver settings; the Mochi entity API (fixing and driving vertices, attaching soft bodies, contact
parameters per link) is in [Mochi entities](entities.md#mochi-entities).

A Mochi scene is configured like any other: give every entity a Mochi material, and optionally tune the solver
through `EnvOptions.mochi_options`. From `examples/recipes/mochi/rigid_bodies.py`:

```python
from eden.engine.solvers.mochi.options import MochiOptions
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import BoxOptions, PlaneOptions, SphereOptions
from eden.options.materials import MochiRigidMaterialOptions

env_options = EnvOptions(
    num_envs=1,
    sim_dt=1.0 / 60.0,
    mochi_options=MochiOptions(n_newton_iterations=8),
)
scene_options = SceneOptions(
    plane=PlaneOptions(material=MochiRigidMaterialOptions()),
    table=BoxOptions(
        size=(1.6, 1.0, 0.1),
        default_root_pos=(0.0, 0.0, 0.75),
        is_fixed_base=True,
        material=MochiRigidMaterialOptions(collider_type="sdf"),
    ),
    sphere=SphereOptions(radius=0.2, default_root_pos=(-0.5, 0.0, 1.5), material=MochiRigidMaterialOptions()),
)
```

The solver is installed when the first entity with a Mochi material is added, whether or not `mochi_options` is
set. `env.mochi_solver` returns it, and raises if no entity of the scene takes a Mochi material.

### Rules of a Mochi scene

- **A Mochi scene is all-Mochi.** Every entity, the ground plane included, needs a Mochi material (kinematic entities
  are allowed). `SceneOptions` rejects the mix and names the offending entities; remember that an entity with no
  material defaults to `RigidMaterialOptions`.
- **Double precision at the default stiffness.** Run with `en.init(precision="64")`, or lower every
  `penalty_coefficient`. The Mochi tasks (`eden/tasks/mochi/`) use `1e7` Pa/m so they train under `eden train`'s
  single precision. Light cloth also uses `1e7`.
- **Leave `MochiOptions.dt` unset.** The solver substep is `sim_dt / sim_substeps`, as for every other solver;
  unset `dt` and `gravity` are filled from the scene's `SimOptions`.
- **No `requires_grad`.** `EnvOptions` refuses `requires_grad=True` with `mochi_options` set, and the env refuses it
  when the first Mochi entity is added.
- **Rasterizer or Nyx only.** The solver refuses a scene whose renderer is the ray tracer or the batch renderer. Use
  the rasterizer, or render through a Nyx camera (`examples/recipes/mochi/_nyx.py` is the bridge). See
  [Sensors and rendering](sensors-and-rendering.md).
- **Options refused at validation.** A `MochiRigidMaterialOptions` entity rejects `default_dofs_frictionloss`,
  `default_dofs_force_limits` and the constraint-solver (sol params) fields, since the solver clamps no drive force,
  models no joint friction loss and has no constraint solver. Support-surface fields, `visualize_contact=True`,
  `GroupedEntityOptions` and `TerrainOptions` are refused with a Mochi material too.
- **Per-env DOF properties** are `MochiOptions.batch_dofs_info` (default `False`), not `EnvOptions.batch_dofs_info`,
  which configures the rigid solver only.

### Mochi materials

| Material | Body | Entity |
|---|---|---|
| `MochiRigidMaterialOptions` | rigid and articulated bodies; keeps the rigid-entity API (DOFs, PD control, IK, `attach_to`) | `MochiRigidEntity` |
| `MochiElasticMaterialOptions` | tetrahedral solids (stable neo-Hookean, StVK or linear); primitives and meshes are tetrahedralized, `.node` / `.vtk` files are read as they are | `MochiSoftEntity` |
| `MochiShellMaterialOptions` | thin shells and cloth on a mesh's surface triangles | `MochiSoftEntity` |
| `MochiRodMaterialOptions` | discrete elastic rods; needs a `RodOptions` morph | `MochiSoftEntity` |

All four share the contact fields of `MochiMaterialOptions`: `friction` (default `0.5`, combined per pair as the
geometric mean), `penalty_coefficient` (`1e9` Pa/m), `penalty_smoothing_half_distance` (`5e-3` m),
`penalty_threshold` (`1e-3` m), `friction_falloff_vel`, `viscous_friction`, `normal_viscous_damping` (the
restitution knob), `max_alignment_normals`, `has_gravity` and `contact_layer`. Contact between two layers is
switched at runtime with `env.mochi_solver.enable_layer_contact(layer_a, layer_b, is_enabled)`, and the contact
parameters of a built entity with its `set_contact_params(...)`.

### Solver options

`MochiOptions` carries the Newton loop (`n_newton_iterations`, default 4; `newton_abs_tol`, `newton_rel_tol`), the
line search, the linear solver (`linear_solver`: `"auto"`, `"ldlt"` or `"pcg"`), the time integrator
(`integrator`: `"backward_euler"` or `"bdf2"`), friction regularization, contact quadrature
(`boundary_element_type`), soft joint limits (`joint_limit_stiffness`, default `1e4`), body sleeping
(`use_sleeping`), divergence handling (`explosion_control`, default `True`) and the step kernel (`step_kernel`). The
[API reference](../api/options/envs.md) documents every field. With `explosion_control`, a substep whose residual
blows up is flagged: the env is put back to its previous pose with zero velocity and the error reaches Eden's
solver-error handling below.

The scripts in `examples/recipes/mochi/` cover each body type, articulated grasps, cloth and rods; the
[Mochi tasks](../tasks/mochi.md) are trainable examples.

## Solver errors and recovery

A physics solver latches an error flag per environment. Eden splits them in two:

- **A corrupt state** (a NaN contact force, acceleration or force in the rigid solver, a diverged Mochi solve) is
  recovered: the termination manager truncates the affected envs, the reward and metric managers drop what their
  terms read of them, and the reset clears the flag before Genesis' error check would raise for the whole scene.
  `env.get_solver_error_envs_mask()` returns these envs, `(num_envs,)` bool.
- **A misconfiguration** (contact capacity overflow, an unsupported collision pair, a NaN gradient under
  `requires_grad`) is reset and dropped the same way, but the env keeps its flag across the reset, and the next
  control step raises it (on CUDA, the one after) with Genesis' message naming the option to change.

So a training run survives an occasional exploding env, but not a scene that is too small for its contacts.

## Throughput switches

These `EnvOptions` fields change speed, not physics:

- `update_sensors_every_substep` (default `False`) skips sensor updates on the intermediate scene steps of a control
  step. See [Sensors and rendering](sensors-and-rendering.md#when-sensors-update).
- `cuda_graph_options` (default `None`) replays the graded phase of `RLEnvBase.step` (metrics, terminations,
  rewards) from a CUDA graph; `CudaGraphOptions()` turns it on. CUDA only. It is a training-only field: `eden
  inference`, `eden export` and `eden deploy` reset it when they replay a run's config. See
  [Environments](environments.md).
- The `batch_*_info` switches above cost memory and some speed; leave them off unless a term needs them.

## See also

- [Entities](entities.md): mass, armature, gains, contact parameters, deformable entities.
- [Sensors and rendering](sensors-and-rendering.md): cameras, renderers, sensors.
- [Engine add-ons](engine-addons.md): how the Mochi solver and Eden's sensors are hosted on stock Genesis.
- [Environments](environments.md): the step loop and where physics runs in it.
- [`EnvOptions` API](../api/options/envs.md) and [material options API](../api/options/materials.md).
