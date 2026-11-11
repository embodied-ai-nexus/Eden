# Velocity Tracking: Design Notes

Why `eden/tasks/velocity_tracking/base_config.py` is set up the way it is. The config file holds what you need to
change a value; this page holds the reasons, the provenance and the hardware findings behind those values. Read it
before you add a robot, change what the policy observes, or retune a penalty. For how to train the tasks, see
[Locomotion tasks](../tasks/locomotion.md#velocity-tracking).

## Layout

Two tasks share one base config:

| Task | Config class | Adds to the base |
|---|---|---|
| `velocity_tracking_flat` | `VelocityTrackingFlatConfig` (`flat/config.py`) | A plain ground plane. |
| `velocity_tracking_terrain` | `VelocityTrackingTerrainConfig` (`terrain/config.py`) | A procedural terrain grid, a height scan in both observation groups, an `out_of_terrain_bounds` termination, 128 collision pairs. |

Both register two TaskMods, `VelocityTrackingRobotMod` (`--robot`) and `VelocityTrackingObsMod` (`--obs_preset`).
Build a config through the registry so the mods run:

```python
from eden.tasks import TASK_REGISTRY

config = TASK_REGISTRY.build("velocity_tracking_flat")                           # G1 29-DOF (default)
config = TASK_REGISTRY.build("velocity_tracking_flat", robot="oli")              # LimX Oli 31-DOF
config = TASK_REGISTRY.build("velocity_tracking_terrain", obs_preset="privileged")
```

Control runs at 50 Hz (`sim_dt=0.005`, `decimation=4`), episodes last 20 s, and the runner is rsl_rl PPO with a
`[512, 256, 128]` ELU actor and critic, 24 steps per env per iteration, 30 000 iterations.

## Robot binding

The task is robot-parametric. Everything in `VelocityTrackingBaseConfig` is robot-agnostic except a handful of
slots, which `VelocityTrackingRobotMod` fills at build time from `VelocityTrackingRobotMod._PRESETS` and from the
robot options themselves:

| Slot | Filled from |
|---|---|
| `scene_options.robot` | The preset's robot class (`G1Dofs29` or `Oli`). |
| Every joint position controller on `robot`: `dofs_name`, `scale` | The robot's `dofs_name`; the preset's whole-body action-scale dict (`G1_ACTION_SCALE`, `OLI_ACTION_SCALE`). |
| `Upright.link_name`, `BodyAngVelXYPenalty.link_name` | The preset's torso link. |
| `RandomizeComShift.links_name` | `[torso link]`. |
| `FeetClearance.feet_link_names`, `FeetSlip.feet_link_names`, `RandomizeFrictionRatio.links_name` | The robot's `RobotOptions.feet_links_name`. |
| `FeetClearance.target_height` | The preset's swing-foot apex target. |
| `runner_options.experiment_name` | Prefixed with the robot key: `velocity_tracking_flat` → `g1_velocity_tracking_flat`. |

These slots ship as empty sentinels (`""`, `[]`, `0.0`, and a controller `scale` of `None`), so a link name from one
robot can never leak into another. `VelocityTrackingBaseConfig.check_bindings()` raises `ValueError`, naming every
field still at its sentinel and the mod that fills it; `EnvBase.from_config` calls it before constructing anything.
That is what catches a config made without the mod: `TASK_REGISTRY.get("velocity_tracking_flat")()`, the class
itself, a subclass registered with `modifiers=()`, or a term added after the mod ran.

### Found by type, filled only when unset

The mod finds the slots by term **type** on the `robot` entity, never by term key, and fills a slot only while it
still holds its sentinel. So a variant may rename, drop or split terms in its own class body — two joint position
controllers for the legs and the upper body, say, each keeping the robot's action scale — and may pin any slot
itself. The term classes the mod binds are listed in `_ROBOT_BOUND_FIELDS`, and reward groups are searched too. A
manager group that a config sets to `None` (`action_options` and `reward_options` are both optional) has nothing to
bind and nothing to report.

For the joint position controllers:

- **`dofs_name` left empty** gets every joint of the robot, in the robot's order. Only one *DOF-owning action
  term* may then exist on the robot (`_robot_dof_owners`): a whole-robot controller collides with any other term
  that commands a joint. A joint position controller counts as DOF-owning even with an empty `dofs_name`, because
  the mod is about to fill it; any other action term (`NullJointAction`, the velocity controllers,
  `BinaryJointController`, the task-space terms) counts once it names DOFs of its own. Terms that split the robot
  must each name their joints.
- **`scale` left at its `None` default**, with no `scale_ratio`, gets the robot's whole-body action-scale dict with
  `dict_keys_scope="entity"`, so a controller that owns only the legs takes the leg entries and skips the rest. An
  authored `scale=1.0` is kept, and a controller that sets `scale_ratio` counts as scaled.
- **No joint position controller at all** on `robot` raises `ValueError`, listing the action terms the config has.

The mod copies every mutable value it assigns. Assignment on an options model is not validated, so pydantic does not
copy, and one shared list would let an edit to one term silently rewrite another (or hand a caller a handle on the
module-level `*_ACTION_SCALE` dict). `TaskRegistry.build` runs each mod once on a fresh config, so the
read-modify-write on `experiment_name` is safe.

`push_robot` (`PushByVelocity`) needs no binding: with its default `base_joint_name=None` it finds the free base joint
by type, which covers both the MJCF `<freejoint>` a G1 declares and the root joint Genesis synthesizes for Oli's
URDF. `encoder_bias` (`RandomizeDofsPosOffset`) leaves `action_term_name=None`, so it covers every joint position
controller on the robot, and a split controller keeps the bias on all of its joints.

### Adding a robot

Everything else — observations, commands, terminations, curriculum, PPO settings and the posture tolerance maps — is
robot-agnostic, so adding a humanoid is one entry in `_PRESETS` plus its key in the `robot` `Literal` of
`VelocityTrackingRobotMod.__init__`. The current presets, from `base_config.py`:

```python
_PRESETS: dict[str, dict] = {
    "g1": {
        "robot_cls": en.robots.G1Dofs29,
        "action_scale": en.robots.G1_ACTION_SCALE,
        "torso_link": "torso_link",
        # Swing-foot apex target [m]; mjlab's G1 value.
        "foot_clearance_height": 0.10,
    },
    "oli": {
        "robot_cls": en.robots.Oli,
        "action_scale": en.robots.OLI_ACTION_SCALE,
        # No ``torso_link`` in Oli's URDF: the last waist link is the trunk.
        "torso_link": "waist_pitch_link",
        # G1's 0.10 scaled by standing height (1.65 m / 1.32 m).
        "foot_clearance_height": 0.125,
    },
}
```

The robot must set `feet_links_name`, and its joint names must follow the `<side>_<body part>_<axis>_joint`
convention the posture maps match (see [Posture tolerances](#posture-tolerances)).

Oli's URDF has no `torso_link`. Its waist is a three-joint serial chain, and `waist_pitch_link`, the last body of that
chain, is the trunk that carries the arms and head — the equivalent of G1's `torso_link`.

## Observations

The observation is split by what a deployed policy can **measure**. Every policy term comes off the robot's own
joint encoders or IMU:

| Term | Policy | Critic | Noise (uniform) |
|---|---|---|---|
| `base_lin_vel` (body frame) | — | yes | ±0.5 |
| `base_ang_vel` (body frame) | yes | yes | ±0.2 |
| `projected_gravity` | yes | yes | ±0.05 |
| `dofs_pos` (offset from default) | yes | yes | ±0.01 |
| `dofs_vel`, `scale=0.05` | yes | yes | ±1.5 |
| `actions` (last action) | yes | yes | — |
| `commands` (velocity command) | yes | yes | — |
| `height_scan` (terrain task only), `scale=0.2` | yes | yes | ±0.1 |

`base_lin_vel` has no sensor on a humanoid. On hardware the policy would get either zeros or a contact-aided Kalman
estimate, and neither is what training produced. It is privileged information, so it belongs to the critic, which
only ever runs in simulation — the standard asymmetric actor-critic split, and one fewer thing that can be wrong on
the robot. The critic has no history: it estimates a value from a state it sees fully. The policy group sets
`enable_corruption=True` and the critic `enable_corruption=False`, so the critic reads the same terms without noise.

The policy stacks **5 frames** (`history_length=5`). From a single frame the policy cannot tell a real velocity from
a one-step sensor glitch; with history it can average one out.

| Robot | Policy, per frame | Policy, 5 frames | Terrain policy, 5 frames | `privileged` policy |
|---|---|---|---|---|
| G1 29-DOF | 96 | 480 | (96 + 187) × 5 = 1415 | 99 |
| Oli 31-DOF | 102 | 510 | (102 + 187) × 5 = 1445 | 105 |

The terrain variant stacks its 187-dot height scan into the same 5-frame window. That is the cost of having one
history mechanism rather than two.

### Why `dofs_vel` is scaled by 0.05

Joint velocity is the noisiest channel a real robot reports, and a single bad joint can dominate an unscaled
31-vector. On one deployed Oli, `waist_roll` was measured reporting a velocity about 10× its own position derivative
and uncorrelated with it, which at scale 1.0 reached the policy 2–4× outside the ±1.5 rad/s range it trained against.
Scaling by 0.05 (the legged_gym convention) keeps the channel informative while bounding what one broken encoder
can do. Observation modifiers apply in the order raw → noise → clip → scale, so the noise stays in physical rad/s
and only the network input shrinks.

### `--obs_preset`

`VelocityTrackingObsMod` changes only the policy group; the critic keeps `base_lin_vel` and stays single-frame under
both presets.

- **`proprioceptive`** (default) — the table above: sensor-only, 5 frames. Deploys with no state estimator. It is
  identical to the unmodified config, so applying it is a no-op, and a config that round-trips through it is
  bit-identical to one that never did.
- **`privileged`** — the recipe the task shipped before the split. The policy also reads `base_lin_vel` (first in
  the vector), in a single frame, with `dofs_vel` unscaled in both groups. The three changes revert together because
  they are one recipe: reverting only the group membership would build a config that loads an old checkpoint at the
  right width and then feeds it joint velocities 20× larger than it trained on. The mod states `terms_order`
  explicitly, because a newly assigned term would otherwise land last; the terrain variant's `height_scan` keeps its
  place at the end. A policy trained this way needs `eden deploy --estimate_lin_vel` on hardware.

## Commands and curriculum

`UniformVelocityCommand` resamples every 3–8 s. 10 % of envs are commanded to stand still and 30 % track a heading
(`heading_control_stiffness=0.5`). The initial ranges are ±1 m/s forward and lateral and ±0.5 rad/s yaw.
`VelocityCommandCurriculum` widens them in stages counted in env steps (`5000 * 24` is iteration 5000 at 24 steps
per iteration). A stage that does not name a range keeps the previous one:

| From env step | `lin_vel_x` (m/s) | `ang_vel_z` (rad/s) |
|---|---|---|
| 0 | (−1.0, 1.0) | (−0.5, 0.5) |
| 5000 × 24 | (−1.5, 2.0) | (−0.7, 0.7) |
| 10000 × 24 | (−2.0, 3.0) | (−0.7, 0.7), unchanged |

## Rewards

### Posture tolerances

`VariablePosture` holds the robot near its default pose with a per-joint tolerance that depends on the commanded
speed: `STD_STANDING` below `walking_threshold=0.05`, `STD_WALKING` up to `running_threshold=1.5`, `STD_RUNNING`
above. The maps are regular expressions over joint names, so every humanoid that follows the
`<side>_<body part>_<axis>_joint` convention (G1, Oli) shares them.

Every DOF must be matched: `resolve_per_dof_std` raises `ValueError` naming any joint the map does not cover. (It
used to fall back to 0.1 silently, which made an uncovered joint train against a tolerance nobody chose.) The
`head` entries cover Oli's head joints and stay in the maps for robots without a head (G1), so adding a robot with
one is a preset entry and not also a std-map edit.

### Smoothness penalties

Three penalties on successive derivatives work together against jitter:

```text
action_rate_l2        = ||a_t - a_{t-1}||^2                    (1st difference)
action_rate_l2_smooth = above + ||a_t - 2a_{t-1} + a_{t-2}||^2 (adds the 2nd difference)
dofs_acc_l2           = ||v_t - v_{t-1}||^2                    (joint acceleration)
```

The first difference alone does not distinguish a fast smooth motion from one that reverses every control step. In
one hardware run, an Oli trained with only the first difference reversed its `waist_roll` command on 81 % of
consecutive steps at 3 rad/s mean, and the resulting ≥ 25 Hz torque was audible; the second-difference term
penalizes exactly that alternation.

`DofsAccL2` sums `(v - v_prev)^2`, a velocity **difference** that is not divided by `dt`. legged_gym's weight of
`-2.5e-7` applies to a true acceleration, so it has to be scaled by `1/dt^2` to carry the same weight. The config
writes it as `-2.5e-7 / 0.02**2` rather than folding it into a literal, so the provenance stays visible
(`eden/tasks/husky/config.py` does the same).

### The full reward set

| Term | Weight | Notes |
|---|---|---|
| `track_linear_velocity` | 2.0 | `track_lin_vel_with_z_penalty`, std √0.25 |
| `track_angular_velocity` | 2.0 | `track_ang_vel_with_xy_penalty`, std √0.5 |
| `upright` | 1.0 | `Upright` on the torso link, std √0.2 |
| `pose` | 1.0 | `VariablePosture`, see above |
| `body_ang_vel` | −0.05 | `BodyAngVelXYPenalty` on the torso link |
| `joint_pos_limits` | −1.0 | `dofs_pos_limits` |
| `action_rate_l2` | −0.1 | |
| `action_rate_l2_smooth` | −0.05 | `ActionRateL2Smooth` |
| `dofs_acc_l2` | −2.5e-7 / 0.02² | `DofsAccL2` |
| `foot_clearance` | −2.0 | `FeetClearance`, active above a 0.05 command |
| `foot_slip` | −0.1 | `FeetSlip`, active above a 0.05 command |
| `self_collisions` | −1.0 | `self_collision_cost` |

Episodes end on time-out or when the robot tilts past 70° (`fell_over`). The terrain task also ends an episode,
counted as a time-out, when the robot comes within 3 m of the terrain edge (`out_of_terrain_bounds`).

## Domain randomization

| Event | Mode | Range |
|---|---|---|
| `reset_base` | reset | xy ±0.5 m, z +0.01 to +0.05 m, yaw ±3.14 rad |
| `push_robot` | interval, every 1–3 s | linear ±0.5 m/s in x and y, ±0.4 in z; angular ±0.52 rad/s roll and pitch, ±0.78 yaw |
| `randomize_friction` | startup | ratio 0.3–1.2 on the feet links |
| `randomize_com` | startup | torso CoM shift ±2.5 cm in x and y, ±3 cm in z |
| `encoder_bias` | startup | ±0.015 rad joint-position offset |

`RandomizeComShift` writes per-env link CoMs, which needs `EnvOptions.batch_links_info=True`. The terrain variant
replaces the whole `env_options` block, so it restates that flag; without it the term raises at build.

## The terrain grid

`_make_terrain_grid()` in `terrain/config.py` builds 10 rows of increasing difficulty by 10 columns, approximating
mjlab's `ROUGH_TERRAINS_CFG` mix: 20 % flat, 20 % pyramid stairs, 20 % inverted pyramid stairs, 10 % pyramid slope,
10 % inverted pyramid slope, 10 % random uniform bumps, 10 % waves. Each sub-terrain is 8 × 8 m at a 0.1 m horizontal
and 0.005 m vertical scale, and the grid is centred on the world origin. The height scan is `GridScanDots`, a
17 × 11 grid read from the terrain's `get_height()`, matching mjlab's 1.6 × 1.0 m ray-cast at 0.1 m resolution.

## Not yet ported from mjlab

The config docstrings list what the mjlab velocity tasks have and these do not: critic-only foot observations
(height, air time, contact, contact forces) and the contact-based rewards (`feet_air_time`, `foot_swing_height`,
`soft_landing`), the `angular_momentum` reward, the `rel_forward_envs` command option and, for the terrain task, the
terrain-level curriculum.

## See also

- [Locomotion tasks](../tasks/locomotion.md) — how to train and replay these tasks.
- [TaskMods](task-mods.md) — the robot-binding pattern this task is the worked example of.
- [Observations](observations.md) — groups, history, noise and modifier order.
- [Deployment](deployment.md) — `eden deploy`, including `--estimate_lin_vel`.
