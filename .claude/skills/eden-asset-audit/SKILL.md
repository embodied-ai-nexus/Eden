---
name: eden-asset-audit
description: Audit and fix a robot or object asset (URDF, MJCF, USD, mesh, an Eden entity preset, or a task's scene entity) for Genesis physics problems with `eden audit`, then fix what it finds in the asset or the entity options. Covers missing collision, massless or very light links, bad mass ratios, the armature 0.1 fill, empty joint ranges, self-contact and dropped collision pairs, objects that sink or jitter, and arms that drift under their gains. Use when importing a new asset, adding a robot preset, or when an entity explodes, sinks, jitters or sags in simulation. Don't use for MuJoCo-only MJCF authoring with no Eden entity in view.
allowed-tools: Read, Grep, Glob, Bash, Edit, Write
---

# Auditing an asset with `eden audit`

Read the command reference (`docs/reference/cli.md`, "eden audit") for what each check measures and every flag,
and `docs/guides/entities.md`, "Auditing a built entity", for what to do about each finding. This skill is the order
of work.

## Order of work

1. Audit the entity as the task builds it: `eden audit --task <task> [--entity <name>]` (the task's variant flags
   and `--override` work as in `eden smoke`). For an asset no task uses yet, name the file or preset and add
   `--task <task>` for the physics it will run at: `hold` and `settle` depend on the timestep and integrator.
2. Fix the failures, then the warnings, one at a time, re-running the audit after each fix. Fix the asset or the
   entity options; never loosen `AuditThresholds` to make a finding go away.
3. Look at the collision geometry: `eden view <Entity> -c`.
4. For a preset, pin what you fixed in `tests/options/robots/test_<module>.py` (or the object family's test).
5. Report which checks still warn and why that is acceptable, quoting the audit line.

## Checklist before calling an asset done

- No `fail` line, and every `warn` either fixed or explained.
- `armature` names no DOF, or the 0.1 is what the source asset or datasheet says.
- `hold` passes at the task's physics for an arm; `settle` passes or only warns that a round body still rolls.
- `dropped_pairs` lists no pair that must collide in the task (fingers against each other, a hand against the arm).

## What it does not check

Visual appearance, textures and materials; whether the asset matches the real robot's dimensions; actuator
models beyond PD. For a policy ported from another stack, also read `docs/guides/policy-porting.md` and audit the
built robot's gains, friction and control rate against the source.
