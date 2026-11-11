# Manipulation examples

Demos for the `manipulation_reach_droid` / `manipulation_lift_box` / `manipulation_dex_lift` tasks
(`eden/tasks/manipulation/`). `manipulation_dex_lift` is the dexterous variant of `manipulation_lift_box`
(Franka + XHand1_L instead of the two-finger Robotiq gripper).

## Scripts

### `pickup.py` — the whole pick scene's collision setup
Loads a registered manipulation task (default `manipulation_lift_box`) and renders **every**
entity in `vis_mode="collision"` (Franka arm + Robotiq gripper + table + cube +
ground), printing each entity's per-link collision flags.

```bash
uv run python examples/recipes/rendering/pickup.py                 # manipulation_lift_box
uv run python examples/recipes/rendering/pickup.py --task manipulation_reach_droid
uv run python examples/recipes/rendering/pickup.py --vis           # interactive viewer (needs a display)
```

Shows the full setup: the arm links + table + cube all collide, while the
gripper collides only at `left_pad` / `right_pad` (see
[`../../recipes/collision/`](../../recipes/collision/) for the per-geom
collision-mask feature this relies on).

## Task variants

`manipulation_lift_box` can swap the arm embodiment while keeping the same Robotiq 2F85 gripper
and reward structure, via its `--manipulator` [TaskMod](../../../docs/guides/task-mods.md):

```bash
eden train --task manipulation_lift_box                       # droid (hand-less Panda + Robotiq 2F85) — default
eden train --task manipulation_lift_box --manipulator ur5e_2f85    # Universal Robots UR5e
eden train --task manipulation_lift_box --manipulator ur10e_2f85   # Universal Robots UR10e
```

Each arm is a composite robot from `COMPOSITE_ROBOT_REGISTRY` (`Droid`,
`UR5eRobotiq`, `UR10eRobotiq` — see
[Robot Presets](../../../docs/api/options/robots.md#composite-robots)). The
observation dim follows the arm's DOF count (38-d for the 6-DOF URs vs 40-d for
the 7-DOF Franka), so checkpoints do not transfer across embodiments.
