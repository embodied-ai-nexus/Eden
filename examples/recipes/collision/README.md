# Per-geom collision mask

Examples for `EntityOptions.collision_link_patterns` / `collision_friction` — a
declarative way to restrict an entity's collisions to a subset of its links.

When set, only geoms on links whose name matches one of the patterns keep
collision; every other link's geoms are disabled (`contype = conaffinity = 0`),
applied in `RigidEntity.pre_build()` before `scene.build()` (Genesis bakes the
collision-pair list at build time). `collision_friction` optionally overrides
the friction of the kept geoms.

The motivating use case: a parallel-jaw gripper (Robotiq 2F85) should grasp
objects at its **fingertip pads**, not wedge them against its inner
linkage/palm. `collision_link_patterns=[".*pad"]` enforces exactly that.

```python
en.robots.Robotiq2f85(
    collision_link_patterns=[".*pad"],   # only left_pad / right_pad collide
    collision_friction=2.0,              # firmer pad contact for stable grasps
)
```

> The whole-pick-scene collision view (`vis_mode="collision"` over the arm +
> gripper + table + cube) now lives at
> [`examples/recipes/rendering/pickup.py`](../rendering/pickup.py).

## Scripts

### `visualize_mask.py` — see the gripper collision geometry + which links collide
Renders the gripper with `vis_mode="collision"` and prints each link's
collision flags.

```bash
uv run python examples/recipes/collision/visualize_mask.py          # default: every link collides
uv run python examples/recipes/collision/visualize_mask.py --mask    # pad-only: only the pads collide
uv run python examples/recipes/collision/visualize_mask.py --mask --vis   # interactive viewer (needs a display)
```

Compare the printed flags: without `--mask` the gripper uses its MJCF's own
per-geom groups (most links `COLLIDES`); with `--mask` only `left_pad` /
`right_pad` read `COLLIDES` and every other link reads `disabled`. PNGs are
saved under `logs/collision/`.

### `drop_test.py` — behavioral proof
Drops a cube onto the gripper body.

```bash
uv run python examples/recipes/collision/drop_test.py          # default: cube rests ON the body
uv run python examples/recipes/collision/drop_test.py --mask    # pad-only: cube falls THROUGH the body
```

Without the mask the cube lands on the gripper; with the mask the inner linkage
no longer collides, so the cube passes through and drops to the ground. Saves
`drop_{all,padonly}_{start,end}.png` under `logs/collision/`.

### `robotiq_2f85_grasp.py` — the payoff: teleop-grasp with the mask
The pad-only mask (plus per-object contact stiffness and a substep dt small enough to
keep the 2F85's tuned solref) lets the closed-loop gripper actually grasp, instead of
wedging/ejecting the object with its inner body. Plane + robot + object, driven by
keyboard teleop (SpaceMouse via `--spacemouse`); compare the closed-loop Droid against
the loop-free Franka Hand, on a box or any Polaris mesh asset.

```bash
uv run python examples/recipes/collision/robotiq_2f85_grasp.py --vis                  # keyboard teleop
uv run python examples/recipes/collision/robotiq_2f85_grasp.py --vis --object tomato  # a mesh object
uv run python examples/recipes/collision/robotiq_2f85_grasp.py --vis --robot panda    # loop-free control case
uv run python examples/recipes/collision/robotiq_2f85_grasp.py --cpu -s 150           # headless smoke
```

WASD/QE moves the end-effector; the gripper key toggles open/close. The hands-off grasp
regression is pinned in `tests/options/robots/test_robotiq_2f85.py`.
