# Mochi examples

Eden ports of the examples in the Genesis fork's `examples/mochi/`, which carried the solver before Eden hosted
it (`eden.engine.solvers.mochi`). The **MochiSolver** is a port of the
implicit multi-physics solver of Meta's [mochi](https://github.com/facebookresearch/project_superdex)
engine: every substep solves one nonlinear system in which the inertia of all bodies and a smooth
signed-distance penalty contact model are assembled together, so large time steps (10-25 ms) stay stable and
contact needs no coupler.

Each script builds an environment from options and steps it, like the other recipes:

```python
env = EnvBase(
    env_options=EnvOptions(sim_dt=1 / 60, mochi_options=MochiOptions(...)),  # from eden.engine.solvers.mochi
    scene_options=SceneOptions(plane=PlaneOptions(material=MochiRigidMaterialOptions()), ...),
    show_viewer=args.vis,
)
env.build()
for _ in range(steps):
    env.scene.step()
```

Every script takes `-v`/`--vis` for the viewer, `-c`/`--cpu` for the CPU backend, `-s`/`--steps`,
and `-r`/`--record` with `-o`/`--output-dir` for an mp4. Under `eden examples smoke` the step horizon
shortens to a handful of steps. The one exception is `articulated_finray_grasp_nyx.py`, whose whole
output is the render: it always records, and offers neither viewer nor CPU backend because Nyx is
CUDA-only.

## Catalogue

| Fork example | Eden script | What it shows |
| --- | --- | --- |
| `rigid_bodies.py` | `rigid_bodies.py` | A sphere and a cube dropped onto a table standing on the ground |
| `soft_bodies.py` | `soft_bodies.py` | Soft spheres and a soft cube on the ground; a rigid box landing on a soft slab |
| `rods.py` | `rods.py` | A clamped sagging rope, a stiff cantilever, a loose rope dropped on the ground |
| `articulated_arm.py` | `articulated_arm.py` | A Franka Panda grasping and lifting a soft box under PD joint control |
| `loop_closure.py` | `loop_closure.py` | A four-bar linkage closed by a connect equality, next to two welded boxes |
| `bolt_nut_self_screw.py` | `bolt_nut_self_screw.py` | A nut screwing itself down a fixed bolt through non-convex thread contact |
| `newton_cradle.py` | `newton_cradle.py` | Newton's five-ball cradle; BDF2 + substeps vs backward Euler |
| — | `newton_cradle_nyx.py` | The same cradle path traced by Nyx: polished steel balls under studio light (needs the `nyx` extra) |
| `cloth.py` | `cloth.py` | A sheet draped over a box, and a ball caught by a sheet whose corners sway |
| `cloth_p1q6.py` | `cloth_p1q6.py` | Contact-quadrature A/B: how far a box edge shows through a coarse drape |
| `soft_duck.py` | `soft_duck.py` | The mochi duck (1899 nodes) dropped on the ground (benchmark scene) |
| `cloth_tshirt.py` | `cloth_tshirt.py` | The mochi t-shirt (3593 nodes, self-contact) falling (benchmark scene); a 1 mm shell where the fork uses 2 mm |
| `rod_helix.py` | `rod_helix.py` | A helical spring hanging from its first node (benchmark scene) |
| `cloth_arm.py` | `cloth_arm.py` | A Franka pressing a 31x31 self-colliding cloth (benchmark scene) |
| `rope_arm.py` | `rope_arm.py` | A Franka pressing the middle of a 64-node rope (benchmark scene) |
| `cloth_teleop.py` | `cloth_teleop.py` | Keyboard (or `--spacemouse`) teleoperation of a Franka grabbing cloths draped over cubes |
| `finray_gripper.py` | `finray_gripper.py` | A Schunk WSG-50 with deformable FinRay fingers closing on a box |
| `articulated_finray_grasp.py` | `articulated_finray_grasp.py` | The same gripper on a Franka, picking a box off the ground |
| — | `articulated_finray_grasp_nyx.py` | The same rig lifting a YCB object (`--object apple`/`orange`/`can`), path traced by Nyx; `--shot closeup` watches the pads fold around it (needs the `nyx` extra) |

The three benchmark-scene meshes (duck, t-shirt, helix) are fetched on first use from `objects/mochi` of the
[`Kashu7100/eden_assets`](https://huggingface.co/datasets/Kashu7100/eden_assets) dataset; the three FinRay
examples need the [LBM eval models](https://huggingface.co/datasets/toyota-research-institute/lbm_eval_models)
and skip with a message when they are absent (`--asset-dir`, or `$LBM_EVAL_MODELS`).

`articulated_finray_grasp_nyx.py` is the same rig and the same solve as the script above it, closing
on a YCB object instead of a cube and path traced instead of rasterized. Each object carries its own
grasp height and closing stroke, since a FinRay pad meets a wide object sooner than a narrow one, and
the pads run an order of magnitude softer than the printed TPU so the ribs visibly fold. Nyx bridges
Genesis one solver at a time and knows nothing about the Mochi solver, so a Mochi scene otherwise
exports at its rest pose and renders a still frame with no error at all; `_nyx.py` is the shim that
hands the plugin the Mochi solver where it reaches for a rigid or FEM one, and belongs upstream in
`nyx-for-genesis` once the plugin learns the solver itself. The shim also has to say that a Mochi
deformable is exportable on its own account: the plugin admits one only when its morph is a file mesh or a
primitive, which covers a solid or a shell but not a rod (`Rod` is a polyline and inherits
`Morph` alone), and an unexported entity is dropped silently, then surfaces as a `TypeError` on a `None`
UUID deep in the renderer's reference table, naming nothing.

Two recipes have no counterpart in the fork:

| Eden script | What it shows |
| --- | --- |
| `contacts_rj45_plug.py` | An RJ45 plug driven into a socket past a sprung latch, trailing a self-colliding deformable cable (port of Newton's `contacts_rj45_plug`) |
| `contacts_rj45_plug_nyx.py` | The same run path traced by Nyx; `--shot latch` rides with the plug and watches the tab deflect and click (needs the `nyx` extra) |
| `teleop/meta_cloth_folding.py` | Two i2rt YAM arms folding the benchmark t-shirt, driven from Meta Quest controllers (clutch-relative IK, head-tracked stereo back to the headset) |

`teleop/meta_cloth_folding.py` is the Mochi build of `examples/recipes/teleop/metaquest/controller_teleop.py`:
same controller stack, a task the rigid solver cannot do. Start the bridge first
(`eden metaquest bridge --tracking controller`).

`contacts_rj45_plug.py` comes from a different engine — NVIDIA's
[Newton](https://github.com/newton-physics/newton), whose `contacts_rj45_plug` example is a connector
insertion with a cable. Newton approximates the cable by a chain of rigid capsules with a bending
stiffness, teleports its first four bodies onto the plug every substep, and swing-corrects each capsule's
orientation afterwards so the collision geometry follows the deformed centerline. Mochi assembles the rod's
nodes and segment twists, the two rigid connector bodies, their joints and the penalty contact into one
implicit Newton system per step, so the cable is a discrete elastic rod tied to the plug by a constraint
*inside* that solve — no teleport kernel, no orientation fix-up, and no collision filter for the nodes
buried in the plug — and it collides with itself, which a chain of capsules with a bending stiffness
cannot. The shared scene is in `_rj45.py`; `convert_rj45_asset.py` converts the assets, which are hosted as
`objects/rj45` of the same dataset, whose README records the one mesh repair the conversion makes (a
non-watertight socket whose SDF traps the latch).

## How entities pick the Mochi solver

As everywhere in Eden, the *material* decides the solver and the backing entity class:

| Material options | Entity class | Morph |
| --- | --- | --- |
| `MochiRigidMaterialOptions` | `MochiRigidEntity` | any rigid morph (primitive, mesh, URDF, MJCF) |
| `MochiElasticMaterialOptions` | `MochiSoftEntity` | Box / Sphere / Cylinder / Mesh (tetrahedralized by tetgen), or a tetgen `.node` / legacy ASCII `.vtk` file |
| `MochiShellMaterialOptions` | `MochiSoftEntity` | a surface Mesh (or primitive) — its triangles are the elements |
| `MochiRodMaterialOptions` | `MochiSoftEntity` | `RodOptions` only (a polyline centerline) |

Solver-level tuning is `EnvOptions.mochi_options=MochiOptions(...)`: `n_newton_iterations`,
`integrator` (`backward_euler` / `bdf2`), `boundary_element_type` (contact samples per triangle),
`linear_solver`, `equality_stiffness`, `implicit_normal_force_for_dissipation`, and the contact-capacity
knobs. Leave `MochiOptions.dt` alone — the solver substep is `sim_dt / sim_substeps`, like every other
Eden solver.

Contact parameters live on the *material* (combined per pair by geometric mean, except
`penalty_threshold`, `penalty_smoothing_half_distance` and `max_alignment_normals`, which come from the
collider side):

| Parameter | Default | Meaning |
| --- | --- | --- |
| `penalty_coefficient` | 1e9 Pa/m | contact pressure per unit penetration once the ramp is fully active |
| `penalty_threshold` | 1 mm | signed distance at which pressure starts to build |
| `penalty_smoothing_half_distance` | 5 mm | half-width of the smooth pressure ramp |
| `friction` | 0.5 | Coulomb friction coefficient |
| `friction_falloff_vel` | 0.01 m/s | sliding speed below which friction is regularized |
| `normal_viscous_damping` | 0 s/m | impact damping; the restitution knob |

Every recipe states `gravity=(0, 0, -9.8)` on its `MochiOptions`: that is what the fork's examples and
the mochi benchmark scenes run at, while Eden's `SimOptions` defaults to -9.81, and the 0.1% difference is
enough to settle a draped cloth into a different fold (`cloth_p1q6.py`'s box-edge metric moves from
-7.5 mm to -5.1 mm). `MochiOptions` carries a per-solver gravity; `EnvOptions` has no gravity field.

## Requirements

- **Stock Genesis.** `MochiSolver` is not in upstream Genesis; Eden hosts it (`eden.engine.solvers.mochi`,
  ported from [Kashu7100/Genesis@feat/mochi-solver](https://github.com/Kashu7100/Genesis/tree/feat/mochi-solver))
  and installs it on the scene of the first Mochi entity, so the recipes run on the upstream `genesis-world` pin.
- **Double precision.** The default 1e9 Pa/m contact stiffness makes the Newton system ill-conditioned in
  single precision, so every recipe calls `en.init(precision="64")` (hence `hardware: gpu-fp64`). fp32 works
  with a lower `penalty_coefficient` — light cloth uses 1e7 either way.
- **A Mochi scene is all-Mochi.** The solver assembles every body into one Newton system and refuses
  to run beside another physics solver, so every entity — the ground plane included — carries a Mochi
  material. `SceneOptions` rejects a mixed scene with a message naming the offending entities, instead of
  letting the build fail late. Kinematic (visual-only) entities are the one exception.

## Notes and current gaps

- **Mounting a soft body on a link is pre-build work.** `MochiSoftEntity.attach_to_link`,
  `set_visual_mesh` and `set_rest_positions` change what the solver assembles, and an attachment anchors
  each vertex where it stands *at build time*, so they run from an `env.register_pre_build_hook(...)`
  callback — the point where the entities exist and the scene is not yet built. See `finray_gripper.py`.
  Order matters inside the hook: `set_visual_mesh` places the render mesh through the morph transform, so
  it runs *before* `set_rest_positions` (and raises after it). A declarative options field for the common
  cases is a tracked follow-up.
- **An articulated entity must name its DOFs.** Eden resolves `dofs_idx_local` and every per-DOF default
  from `dofs_name`, so `is_articulated=True` with no `dofs_name` leaves an *empty* index tensor and every
  `control_dofs_*` call on the entity addresses nothing. The two FinRay recipes declare the gripper's
  `["finger_joint_1", "finger_joint_2"]` and drive the first by name; `loop_closure.py` and
  `newton_cradle.py` do the same for their hinges.
- **Fields the solver never reads are rejected.** `default_dofs_frictionloss`,
  `default_dofs_force_limits` and the constraint-solver parameters raise on a `MochiRigidMaterialOptions`
  entity: Mochi does not clamp drive forces, model joint friction loss, or have a constraint solver to tune.
  Support surfaces are rejected too (they walk collision AABBs the solver does not expose).
- **Link mass is fixed at build.** `set_mass` raises; set the material's `rho` instead. `_verify_build`
  turns a sub-gram Mochi body into an error rather than silently patching the mass as it does for the rigid
  solver.
- **No differentiable mode**, and `EnvOptions.requires_grad=True` with `mochi_options` is rejected up front.
- **Batching works** (`num_envs > 1`); note that per-env DOF info is `MochiOptions.batch_dofs_info`, not
  `EnvOptions.batch_dofs_info` (which configures the rigid solver).
- **A close camera needs `cam_near`.** Genesis' default near plane is sized for a robot, not for a 25 mm
  connector: any camera closer than ~0.1 m renders an empty frame, with no error. `CameraOptions` takes
  `cam_near` / `cam_far` (rasterizer and batch renderer); `contacts_rj45_plug_nyx.py` sets 2 mm.
- **A rod collides through its centerline.** Rod contact samples sit on the polyline with no
  cross-section offset (the rod radius is only what *other* point-cloud bodies feel, through its node
  spheres), so a cable laid on the ground rests with its tube half buried. The threshold of a contact pair
  comes from the collider side, so giving the ground a `penalty_threshold` of one rod radius restores the
  cable's thickness — `contacts_rj45_plug.py` does exactly that.
- **Rod self-contact is off by default, and the two knobs that turn it on are coupled.** With
  `self_contact=True` a rod's samples meet its own node spheres, closing at
  `radius + penalty_threshold` — one radius short of where two tubes touch, so the threshold wants to be
  nearly a radius rather than the 1 mm default. But nodes within
  `radius * self_contact_exclusion_ratio + penalty_threshold` of each other *in the rest pose* are excluded
  for the whole run, and the ratio must exceed 1, so the exclusion radius always covers the closing
  distance: **a crossing that is already tight at rest can never be resolved.** Set the threshold just under
  the rest separation of the crossing you care about, and the ratio barely above 1. Measured on
  `contacts_rj45_plug.py`'s cable (3.25 mm radius, strands 6.31 mm apart at rest), worst-case separation
  between strands over the run, against the 6.50 mm where two tubes touch: 0.54 mm with self-contact off,
  0.74 mm at a full radius and the default ratio of 1.5 (the exclusion swallows the crossing, so it is
  barely better than off), 5.05 mm at 2.9 mm and 1.03. Stiffness is not the lever — raising the rod's
  `penalty_coefficient` from 1e9 to 1e11 moves the settled gap by 0.03 mm.
- **Contact stiffness is per link, and against a static collider it is the sample side's alone.** Mochi
  combines a pair's `penalty_coefficient` as a geometric mean, *except* when the collider is a static body,
  where the sample side's value is used unchanged. `MochiRigidEntity.set_contact_params(links=[...])` then
  tunes one link of an articulated entity against a fixed collider independently of its siblings.
  `contacts_rj45_plug.py` needs it: the latch presses on the socket through a ~5 mm² tab, and at the
  scene-wide 1e9 Pa/m its 0.15 N·m/rad return spring pushes the tab 0.6 mm *through* the socket instead of
  swinging the latch (0.009 rad of deflection instead of 0.155).
- **An MJCF whose root body carries explicit joints still needs `is_fixed_base=True`.** Eden sizes
  `qs_idx_local` from the *config* flag, not from the parsed model, so without it `get_qpos` reads seven
  phantom root qs past the entity's range — a CUDA device-side assert, not a message. Genesis warns that it
  cannot forward the flag to `gs.morphs.MJCF`; that part is harmless, since its parser only adds a free
  joint to a body that declares none. `loop_closure.py` and `contacts_rj45_plug.py` both pass it.
