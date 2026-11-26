---
name: eden-asset-audit
description: Audit and fix a robot or object asset (URDF, MJCF, USD, mesh, or an Eden entity preset) for Genesis physics problems with `eden audit`, then fix what it finds in the asset or the entity options. Covers missing collision, massless or very light links, bad mass ratios, the armature 0.1 fill, empty joint ranges, self-contact and pairs Genesis drops, objects that sink or jitter, and arms that drift under their gains. Use when importing a new asset, adding a robot preset, or when an entity explodes, sinks, jitters or sags in simulation. Don't use for MuJoCo-only MJCF authoring with no Eden entity in view.
allowed-tools: Read, Grep, Glob, Bash, Edit, Write
---

# Auditing an asset with `eden audit`

`eden audit` builds the entity alone on a ground plane, on the CPU, and checks what Genesis made of it. The full
description of each check is in `docs/reference/cli.md` ("eden audit") and the module docstring of
`eden/utils/asset_audit.py`.

```bash
eden audit <RobotClass-or-key>                      # a preset, as tasks build it
eden audit path/to/arm.urdf --fixed-base            # a URDF arm: pin it so the hold check runs
eden audit path/to/mug.obj                          # a mesh object: dropped and watched
eden audit path/to/model.xml --json audit.json      # MJCF: the base comes from the file
```

It exits 1 when a check fails. Composite rigs are refused: audit the arm and the gripper separately.

## Order of work

1. Run the audit as the task will build the entity (the preset, with its options).
2. Fix the failures, then the warnings, one at a time, re-running after each fix.
3. Look at the collision geometry: `eden view <Entity> -c`.
4. For a preset, pin what you fixed in `tests/options/robots/test_<module>.py` (or the object family's test).

## Fixing what it finds

| Check | Fix |
|---|---|
| Build raises on inertia | The asset's inertia tensor is not physical. Fix it in the asset, or set `recompute_inertia=True` to derive it from the collision geometry and the material density. |
| Build raises on `<inertial><density>` | Genesis' URDF parser needs explicit `<mass>` and `<inertia>`. |
| `collision` | No collision shapes: add them to the asset, or check that `collision=False` and `collision_link_patterns` are not hiding them. |
| `mass` | Give massless links a `<mass>`. A link lighter than 1e-4 kg with stiff contacts (finger pads) can jitter; raise it if nothing depends on it. |
| `mass_ratio` | Over 100x across a joint slows the solver and jitters. Lump tiny links into their parent, or make the masses realistic. |
| `armature` | Set `default_dofs_armature` from the source asset or the motor datasheet, or `default_armature=0.0` (`docs/guides/entities.md`, "Armature defaults and the URDF 0.1 trap"). For a URDF every hinge gets the 0.1 fill. |
| `joint_limits` | An empty range is a fixed joint authored as a hinge: make it fixed. |
| `self_contact`, live pairs | Two links touch at the default pose: trim the shapes, or move `default_dofs_pos`. |
| `self_contact`, dropped pairs | Genesis never collides pairs that overlap at the asset's zero pose. Usually intended (adjacent shells). If the pair must collide, fix the overlap in the asset, or set `EnvOptions.enable_neutral_collision=True` scene-wide. |
| `settle` explodes or sinks | Check mass and inertia first, then thin or concave collision meshes (`convexify`, `decompose_object_error_threshold`, `coacd_threshold`). Stiffen one object with `default_geoms_sol_params`, never a global `constraint_timeconst`. |
| `settle` still moving | A round body may just roll. Jitter in place points at contacts or inertia. |
| `hold` skipped | No gains: set `default_dofs_kp` / `default_dofs_kd` (from the source stack: Isaac Lab's `stiffness` is Eden's kp). |
| `hold` drifts | The gains do not hold the arm against gravity, or the default pose sits on a limit. Compare with the source stack's gains and armature; a closed-loop gripper also needs `max_substep_dt`. |

The thresholds are `eden.utils.asset_audit.AuditThresholds`; call `audit_entity(options, thresholds=...)` from
Python for other limits or a longer `seconds`.

## What it does not check

Visual appearance, textures and materials; whether the asset matches the real robot's dimensions; actuator
models beyond PD. For a policy ported from another stack, also read `docs/guides/policy-porting.md` and audit the
built robot's gains, friction and control rate against the source.
