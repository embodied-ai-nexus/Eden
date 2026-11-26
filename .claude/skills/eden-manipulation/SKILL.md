---
name: eden-manipulation
description: Scripted grasping, pick-and-place and collision-aware arm motion in Eden simulation. Covers choosing a grasp (GraspGen/GraspGenX or by hand), IK and task-space control, motion planning with cuRobo or OMPL, gripper physics (closed-loop Robotiq substeps, friction, per-object stiffness), grasp modes beyond top-down, verifying a grasp, and scripted demonstrators with eden datagen. Use when writing or debugging a scripted manipulation controller, a demonstrator, or a manipulation task's reset and success logic. Don't use for RL reward design (use eden-rl-task) or for driving real hardware (use eden-hardware-bringup).
allowed-tools: Read, Grep, Glob, Bash, Edit, Write, Agent
---

<!--
The grasping heuristics and the grasp modes are adapted from google-deepmind/robotics-antigravity-plugin,
plugins/robotics/skills/pick-and-place/SKILL.md, references/grasp_modes.md and skills/motion-planning/SKILL.md
(Copyright 2026 Google LLC, Apache License 2.0). Modified for Eden: Eden's planners, grasp samplers, action terms
and demonstrators replace the plugin's scripts, and the Genesis gripper physics was added.
-->

# Scripted manipulation in Eden

Work in stages and measure each one (the `eden-iterate` skill). Start from the closest existing piece of code
rather than from scratch:

| Need | Start from |
|---|---|
| A pick, place or stack demonstrator | `eden datagen --task robosuite_stack` and `docs/guides/scripted-demos.md` (keyframe and tool-routine demonstrators, `eden/extensions/scripted_demos/`) |
| Grasp poses for an object | `docs/guides/grasp-sampling.md`, `examples/recipes/planning/grasp_sampling_graspgenx.py` |
| A collision-free joint path | `docs/guides/planning-and-sysid.md`: `CuroboPlanner` (end-effector goal, batched) or `OMPLPlanner` (joint goal) |
| Arm control in task space | `docs/guides/actions.md`, "Task-space controllers": `DifferentialIKController`, `IKPoseController`, `OperationalSpaceController` |
| A gripper that holds things | `docs/guides/entities.md`, "Closed-loop grippers and constraint fidelity"; `examples/recipes/collision/robotiq_2f85_grasp.py` |

## Gripper physics first

Most "the grasp slips" bugs are physics, not planning:

- A closed-loop gripper (Robotiq 2F-85) needs an effective substep `sim_dt / sim_substeps` of at most 1 ms. Its
  options declare `max_substep_dt`.
- Stiffen the grasped object with its own `default_geoms_sol_params`. Never set a global
  `EnvOptions.constraint_timeconst`: it ejects mesh objects from the table.
- Contact friction combines as `max` of the two geoms. Torsional friction is off unless the scene enables it, and
  an MJCF geom without `condim >= 4` parses it as 0, so a pinched object spins out of the jaws.
- Audit a new gripper or object with `eden audit` (the `eden-asset-audit` skill) before blaming the controller.

## Perception in simulation

Read object poses with `get_pos(relative=True)` / `get_quat(relative=True)` (the authored frame). For a
camera-only controller, use the task camera's RGB-D and segmentation (`docs/guides/sensors-and-rendering.md`);
`eden datagen --camera_only` runs the demonstrators that support it this way. Localize a movable place target
(bin, plate, stack base) before picking, while the gripper is empty and out of view.

## Moving the arm

- Compute IK with `robot.inverse_kinematics(link_name, pos, quat)` (wxyz) or let an action term do it. Verify
  the solution with forward kinematics before moving.
- Plan collision-free motion only between pre-contact poses 5 to 8 cm clear of everything. Make the last approach,
  the lift and the retreat straight Cartesian moves, not two-point joint interpolations.
- Plan only up to the next safe checkpoint you can stop at (lifted hover, object placed, arm parked), execute it,
  then re-observe. A multi-stage plan solved up front breaks when the first stage lands differently.
- Before committing to a grasp, check that the place pose is reachable from it. If coupled planning fails twice,
  place the object on a free surface, release, re-observe and regrasp instead of tuning the plan.
- Respect the wrist range: rotate the grasp heading by the object's symmetry (a quarter turn for a cube, a half
  turn for a parallel gripper) into the range the last joint can reach. The demonstrators' `yaw_window()` does this.

## Grasping

- Keep `z_top` (the object's top, for perception) apart from `z_grasp` (where the jaws close). Closing at `z_top`
  pinches air. For a top-down jaw grasp, descend to 1 to 2 mm above the support surface, unless the object is
  taller than the finger pads or flanged, so the pads cover as much of it as they can. On a raised tray or plate,
  use that surface's height.
- Keep the approach vertical on top-down grasps whenever IK reaches it: a 10 to 15 degree tilt puts one jaw on the
  table and the other in the air. Close the jaws across the object's narrow width.
- Hold the gripper command for about half a second after closing or opening before lifting or retracting.
- Verify the grasp twice: after closing (finger opening within the object's width) and after lifting (the object
  rose with the hand; the demonstrators' rule is at least half the hand's rise). On a failed check, open, retreat,
  re-observe and retry, at most three times.
- Place by descending vertically, opening, and retreating vertically before any lateral move. Confirm the
  placement from the settled pose.

## Grasp modes beyond top-down

- **Blocked jaw clearance** (object against a wall or a neighbor, under 1 cm free): close the jaws into a narrow
  finger, nudge the object 2 to 4 cm toward open space at 5 cm/s or slower, retreat, re-observe, then grasp.
- **Tall objects** (height over twice the width, or a top wider than the jaw opening): approach from the side, or
  at 45 to 75 degrees over a bin wall, near mid-height; lift vertically before moving sideways.
- **Bowls, wide cups, trays**: pinch the rim or the handle with the closing axis radial to the wall, opened to the
  wall thickness plus 1 to 2 cm, one finger inside and one outside.
- **Suction**: approach along the surface normal of a flat face; `SuctionCupWelding` is the vacuum bit.
  **Multi-finger hands**: thumb-index pinch for small items, a power wrap for heavy cylinders and spheres; see the
  dexterous tasks (`reorient`, `manipulation_dex_lift`) and `docs/guides/dex-retargeting.md`.
