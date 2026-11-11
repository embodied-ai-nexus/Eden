# Scripted demonstrations (`eden datagen`)

`eden datagen` rolls a task's scripted demonstrator out in parallel envs and records every episode to HDF5 — the
same layout as a teleoperated recording, so `eden lerobot convert` reads either. The demonstrators are a port of
[OCBench](https://github.com/seohongpark/ocbench)'s scripted policies (`eden/extensions/scripted_demos/`), built to
produce data that moves like a person — smooth, randomly paced, with retries and deliberate mistakes — rather than
a rule-based controller with noise on top. This page covers running the generator, what it records, and how to
write a demonstrator for a new task.

## Run it

```bash
eden datagen --task robosuite_stack --num_episodes 1000 --num_envs 32 --output_dir data/demos
eden datagen --task robosuite_stack --robot ur5e_2f85 --export succeeded   # OCBench's own arm, keep successes
eden datagen --task robosuite_lift --p_mistake 0.0                          # no deliberate mistakes
eden datagen --task robosuite_pick_place --camera_only                      # sees through the task camera
eden datagen --task robosuite_door --latch no_latch                         # a task variant flag
eden datagen --task simpler_put_carrot_on_plate \
    --override 'env_options: {use_gjk_collision: false}'                    # SIMPLER: see below
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--task` | — | A task with a registered demonstrator (below). Its variant flags (`--robot`, `--latch`, `--object`, …) are accepted too; `eden datagen --task <name> --help` lists them. |
| `--num_episodes` | `100` | Episodes to end before stopping. |
| `--num_envs` | `16` | Parallel envs. |
| `--output_dir` / `--filename` | `data/demos` / the task name | The dataset is `<output_dir>/<filename>.hdf5`. An existing file of that name is **replaced**. |
| `--export` | `all` | `all` keeps every episode, tagged with its success; `succeeded` keeps only successes; `separate` writes failures to `<filename>_failed.hdf5`. |
| `--seed` | `0` | Seeds the env and the demonstrator. |
| `--p_mistake` | the demonstrator's own | Probability that a subgoal is a deliberate mistake. |
| `--episode_length_s` | the demonstrator's own, else 90 | The generator's episode time limit. |
| `--camera_only` | off | Run without the simulator's object state (demonstrators that support it). |
| `--show_viewer`, `--cpu` | off | Open the Genesis viewer; use the CPU backend. |
| `--config`, `--override` | — | Layer a config file or inline YAML onto the task config, as for `eden train`. |

The run ends by printing a report: episodes, successes, how many were cut short, steps, wall time, and subgoals
started by kind (`goal`, `clear`, `mistake`) — or, for a tool routine, motions ended by `<tool>:<status>`. For a
tool routine the trace of the first unsuccessful episode follows; it is the place to start when the success rate
is low. Logs go under `logs/datagen/<task>/`.

## Tasks with a demonstrator

[`SCRIPTED_DEMO_REGISTRY`](../api/extensions/scripted-demos.md) (`eden/extensions/scripted_demos/registry.py`) maps a task name to its demonstrator
factory. A task without one is refused with the list of tasks that have one.

| Task | Demonstrator | Variant flags, camera-only |
| --- | --- | --- |
| `robosuite_lift`, `robosuite_stack` | keyframe (`LiftController`, `TowerController`) | `--robot` |
| `robosuite_pick_place` | tool routine (`eden/tasks/robosuite/pick_place/routine.py`) | `--object`; camera-only |
| `robosuite_door` | tool routine (`door/routine.py`) | `--latch` |
| `robosuite_wipe` | tool routine (`wipe/routine.py`) | |
| `robosuite_tool_hang` | tool routine (`tool_hang/routine.py`) | |
| `robosuite_two_arm_transport` | two-arm routines (`two_arm_transport/routine.py`) | |
| `simpler_stack_green_cube_on_yellow_cube` | keyframe (`TowerController`) | |
| `simpler_put_carrot_on_plate`, `simpler_put_spoon_on_table_cloth`, `simpler_put_eggplant_in_basket` | tool routine (`eden/extensions/scripted_demos/put_on_routine.py`) | camera-only |
| `polaris_block_stack_kitchen_batched`, `polaris_food_bussing_batched`, `polaris_pan_clean_batched`, `polaris_move_latte_cup_batched`, `polaris_organize_tools_batched`, `polaris_tape_into_container_batched` | tool routine (put-on, bound in `eden/tasks/polaris/scripted_demos.py`) | |
| `whiteboard_write_text`, `whiteboard_write_in_colour`, `whiteboard_draw_table` | the task's own `ScriptedDrawer` | |
| `art_calligraphy_dai`, `art_calligraphy_san`, `art_calligraphy_ei`, `art_calligraphy_lettering_minimum` | the task's own `ScriptedWriter` | |
| `art_painting_sunset`, `art_painting_fuji_study`, `art_painting_fuji` | the task's own `ScriptedPainter` | |

The whiteboard and art demonstrators move a held tool through the arm's IK alone, with no gripper action and no
randomness; they accept only `--p_mistake 0`. See [Whiteboard](../tasks/whiteboard.md) and [Art](../tasks/art.md).

## What an episode records

The generator replaces the task's recorder options with this set, written with `HDF5FileHandler`:

| Key | Recorder | Content |
| --- | --- | --- |
| `obs/<group>` | `ObservationRecorder` | Every observation group at the start of each step — what the step's action was chosen from. |
| `action` | `ActionRecorder` | The action sent to the task's own action terms, so it replays as-is. |
| `state/{qpos,dofs_vel,pos,quat}` | `StateRecorder` | The arm's state; a two-arm task records both arms, concatenated in scene order. `eden lerobot convert` reads this as `observation.state`. |
| `reward`, `done`, `terminated`, `timeout`, `timestamp` | `RLStepRecorder` | Per-step RL buffers. |
| `initial_state/entities/...` | `InitialStateRecorder` | Every entity's state after reset, for replay. |
| attribute `success` | — | The task's success metric when the episode ended. |

An episode ends when the demonstrator has finished the task (its success is the task's success metric at that
step), when the task ends it (a termination, e.g. an object off the table), or at the generator's time limit — the
generator enforces that limit itself, because a task need not declare a time-out. Episodes still running when
`--num_episodes` is reached are dropped, not written cut short. The env is built in eval mode. The layout and its
conversion are described in [Datasets and recording](datasets-and-recording.md).

The generator refuses a task with no success metric (no metric term with `contributes_to_success=True`), since
its episodes could not be labelled, and it refuses `export_mode=EXPORT_NONE`.

!!! note "SIMPLER tasks record no observations"
    SIMPLER tasks are evaluated from the camera and declare no observation groups, so their episodes have no
    `obs/*` keys (and no images).

## What makes the data demo-like

| Property | Where | What it does |
| --- | --- | --- |
| Smooth, non-Markovian motion | `trajectory.py` | Each subgoal is a keyframe plan executed on a per-env clock: Hermite splines that come to rest at *stop* keyframes (before closing, at the place, at the end) and pass through the others; rotation and gripper ease with a quintic smoothstep. |
| Human speed profile | `trajectory.py` | Segments are stretched so a minimum-jerk profile never exceeds a peak tool speed: `T >= 1.875 d / v_peak`. |
| Speed varies per episode | `controller.py` | One `segment_dt` per episode, uniform in `speed_range` (default `(0.7, 1.5)` s). Tool routines draw a speed factor from their own `speed_range` (default `(0.8, 1.25)`). |
| Varied grasps and paths | `primitives.py` | Grasp face (usually the nearest), wrist tilt up to `max_tilt` (25° by default), approach heights, contact offsets, an optional pause before lifting, when the fingers close, and a carry midpoint that bows sideways and sometimes overshoots. |
| Recovers from failure | `controller.py` | A pick whose object does not follow the hand is re-planned from wherever the hand is (`max_pick_retries=2`), every random offset scaled by `0.75` per retry. |
| Suboptimal high-level plans | `block_world.py` | With probability `p_mistake` (default `0.1`) a subgoal is replaced by a wrong one — a block moved to a random free spot or onto another — and the demonstrator works its way back. |
| Unsaturated actions | `rig.py` + the task binding | See [Speed and saturation](#speed-and-saturation). |

## The robot it drives

A demonstrator drives an arm with a two-finger gripper through the task's own action terms. It reads what it
needs from the scene entry holding the robot (`gripper_rig()` in `rig.py`); two shapes qualify:

| Scene entry | Arm / fingers | IK link | Tool point |
| --- | --- | --- | --- |
| a `ManipulatorOptions` composite (robosuite's rigs, DROID, UR5e + Robotiq) | the arm entity / the welded gripper entity | the composite's `ee_link` | the gripper's `tcp_offset`, carried through the weld |
| a `RobotOptions` whose asset has its own fingers (SIMPLER's `WidowX250`) | every actuated DOF except those of `open_dofs_pos` / the DOFs of `open_dofs_pos` | `ee_links_name[0]` | the robot's `tcp_offset`, in that link's frame |

A robot of the second shape must declare `open_dofs_pos`, `ee_links_name` and `tcp_offset`; a missing one is
refused by name. `tcp_offset` is the pad midpoint with the fingers open — measure it off the built finger pads
(the WidowX's is `(0.0935, 0, 0)` in its IK link). Its direction is the approach axis: `+z` for a Franka Hand or
a Robotiq mount, `+x` for the WidowX.

The action terms are found by what they drive, not by name (`find_rig_actions()`): one
`DifferentialIKController` on the arm's IK link and one `BinaryJointController` on the fingers. robosuite names
them `robot_arm` / `robot_gripper`, SIMPLER and PolaRiS `ee_controller` / `gripper_controller`. Every action term
of the task must be one of the two (for one arm, or for each arm of a two-arm task).

## How it drives the robot

The demonstrator plans the **tool point** and converts it to the task's action every step (`RigDriver` in
`rig.py`): a 6-D world-frame delta of the IK link's pose, at most `max_step_translation` per axis (default
`0.05` m) and `max_step_rotation` (default `0.3` rad) per step — OCBench's `ee_action_delta` clip — and `2g − 1`
for the binary gripper. The IK term squashes its input with `tanh`, so the recorded arm action is
`atanh(delta / scale)`, with the ratio capped at `0.9` (`MAX_COMMAND_RATIO`).

Object poses come from the scene in their authored frames (`get_pos(relative=True)`), never from the noisy policy
observations. The keyframe controller also handles what OCBench did not need on Eden's robots:

- **Headings respect the wrist.** A Panda's last joint turns ±2.9 rad and the task's damped-least-squares IK has
  no posture term, so the arm drifts in its null space. Every heading is moved by a symmetry of what it grasps (a
  quarter turn for a cube, a half turn for the gripper) into the range the last joint can still reach
  (`yaw_window()`).
- **Plans start from the hand's actual orientation**, not its heading alone (the robosuite Panda's home pose is
  tilted).
- **The plan waits at the grasp and the release.** Contact gates hold the plan clock until the hand is within
  `contact_gate_tolerance` (1 cm) of the gate — or has come to rest against contact within 4 cm of it (slower than
  1 cm/s for 0.6 s: `SETTLE_SPEED`, `SETTLE_TIME`, `SETTLE_REACH`), the way a person lowers a cube onto another
  until it stops.
- **A pick is judged by whether the object follows the hand**: the check waits until the hand has risen
  three quarters of the way to its hover and fails the pick only if the object rose less than half as much
  (`PICK_CHECK_RISE`, `PICK_FOLLOW_FRACTION`).
- **The fingertips, not the tool point, stay above the table.** The workspace floor is for the fingertips: the
  driver measures how far the fingers reach past the tool point on the built rig (`tip_reach`) and raises the
  tool's floor by it.
- **The gripper never closes during the descent**: closing ramps over the contact segment or the pause before
  lifting only, which suits a binary gripper.

### Speed and saturation

The task's action realizes only part of each command per step (damped IK into a position-PD arm), so an arm
asked to follow a fast plan lags it, and the command saturates — the bang-bang data the demonstrators exist to
avoid. Each binding sizes its peak speed and step clip to what its action realizes:

- robosuite and SIMPLER bindings use a peak speed of `0.12` m/s, PolaRiS `0.15` m/s; OCBench's own default
  (`PickPlaceParams.peak_speed`) is `0.3`.
- SIMPLER's WidowX has a generous IK scale (0.5 m per action unit) but trails its plan at 5 Hz, so its binding
  lifts `max_step_translation` to `0.1` m, caps the wrist tilt at `0.1` rad, and uses Bridge-scale hover heights
  (`_DESCENT_HEIGHT = (0.04, 0.12)`).

A new binding should measure how often its arm commands sit at the clip before choosing `peak_speed` and
`max_step_translation`.

## Writing a demonstrator

Register a factory under the task's name, next to the task (`eden/tasks/<suite>/scripted_demos.py`), and list
that module in `SCRIPTED_DEMO_REGISTRY`'s `populated_by`. A factory is `(env, *, seed, p_mistake) ->
demonstrator`; one that also takes `privileged` has a camera-only mode, and one whose episodes need longer than
90 s declares it as an attribute (`my_factory.episode_length_s = 200.0`).

### Keyframe demonstrators

A keyframe demonstrator is a batched state machine: fast, but a new kind of task means a new subclass. Block
worlds reuse `TowerController` or `LiftController` (`block_world.py`):

```python
from eden.extensions.scripted_demos.block_world import TowerController
from eden.extensions.scripted_demos.primitives import PickPlaceParams
from eden.extensions.scripted_demos.registry import SCRIPTED_DEMO_REGISTRY


@SCRIPTED_DEMO_REGISTRY.register(name="my_three_block_tower")
def my_three_block_tower(env, *, seed=0, p_mistake=0.1):
    return TowerController(
        env, ["base", "middle", "top"], table_z=0.8,
        spot_bounds=((-0.12, -0.15), (0.12, 0.15)),
        workspace=((-0.35, -0.35, 0.805), (0.35, 0.35, 1.25)),
        effector_bounds=((-0.2, -0.15, 0.92), (0.05, 0.15, 1.1)),
        p_mistake=p_mistake, seed=seed, params=PickPlaceParams(peak_speed=0.12),
    )
```

Any other pick-and-place task subclasses `ScriptedPickPlaceController` and implements
`choose_subgoals(env_ids, objects) -> list[Subgoal | None]`, returning `None` for an env whose task is complete.

### Tool routines

A *tool routine* is a scripted policy written for **one** robot as a plain Python generator, against a handful
of robot tools — the form a person or a coding agent writes a demonstrator in. `RoutineDemonstrator`
(`routines.py`) runs one instance per env, batched:

```python
import math

from eden.extensions.scripted_demos.registry import SCRIPTED_DEMO_REGISTRY
from eden.extensions.scripted_demos.routines import RoutineDemonstrator


def pick_and_lift(tools):
    for attempt in range(3):
        cube = tools.object_footprint("cube")                    # or tools.locate("cube") through a camera
        yaw = cube.yaw + math.pi / 2 - tools.finger_axis_yaw    # close across the short side
        yield tools.open_gripper()
        yield tools.move_to(cube.top_center + (0, 0, 0.10), yaw)
        yield tools.move_to((*cube.top_center[:2], cube.top_z - 0.02), yaw)
        yield tools.close_gripper()
        yield tools.move_by((0, 0, 0.15))
        if tools.holding():
            return
        tools.note(f"attempt {attempt} missed (closure {tools.gripper_closure():.2f})")


@SCRIPTED_DEMO_REGISTRY.register(name="my_lift")
def my_lift(env, *, seed=0, p_mistake=0.0):
    return RoutineDemonstrator(env, pick_and_lift, seed=seed, workspace=((-0.35, -0.35, 0.805), (0.35, 0.35, 1.25)))
```

A motion is `yield`ed, and the routine gets an `Outcome` back when the motion has ended. Its `status` is one of
`reached`, `contact`, `done`, `timeout`, `blocked` or `no_contact`, and `outcome.ok` is true for the first three,
so checks and retries are ordinary Python. Routines resume only at those boundaries; every motion in flight is
evaluated as batched torch and turned into the task's actions by the same `RigDriver` the keyframe controller
uses.

| Tool (`tools.py`) | Kind | What it does or reads |
| --- | --- | --- |
| `move_to(pos, yaw)`, `move_by(offset)` | motion | Straight, minimum-jerk path at the episode's speed; `yaw` is turned by a half turn into the range the last joint reaches. Ends `reached` within `tolerance` (default 1 cm) or `timeout` (`timeout_s`, default 2 s past the path's own duration). |
| `move_to(pos, rot=tools.aim(approach, closing))` | motion | The same to a full effector rotation (a hand that leans or points sideways), judged on the whole rotation. |
| `move_until_contact(direction, max_distance)` | motion | Constant speed until a finger presses on something that is not the robot (`contact`), the hand is held back without one (`blocked`), or the distance is used up (`no_contact`). |
| `close_gripper()`, `open_gripper()`, `wait(s)` | motion | Hold the arm, set the fingers; `done`. |
| `tool_pos()`, `tool_yaw()`, `tool_rotation()`, `gripper_closure()`, `holding()` | robot | The robot's own state. `holding()`: commanded closed and stopped short of closing (`holding_closure`, default 0.9). |
| `heading_window()` | robot | The lowest and highest heading the last joint can turn the hand to from here. |
| `object_footprint(name)`, `grasped(name)` | privileged | Off the simulator: the footprint of the collision mesh; two finger links touching the object. |
| `link_pose(name, link=None)` | privileged | A link's (or the root's) pose in the asset's authored frame — for an articulated fixture's parts, or a kinematic entity with no collision mesh. |
| `locate(name, camera=None)` | camera | The footprint of the pixels showing `name`, back-projected through depth: what the camera sees, occlusion included. The segmentation stands in for a perfect text-prompted segmenter. |
| `shared`, `rng`, `note(text)` | episode | A per-episode dict (shared between arms), the episode's random stream, and a line in the trace. |

A `Footprint` is the same whichever way it was measured — `center`, `yaw` (of the long axis), `half_length`,
`half_width`, `top_z`, `bottom_z`, and `top_center` (where a hand coming from above meets it, off the centre when
the object leans) — so a routine written against it runs privileged or camera-only unchanged.

**Camera-only.** `RoutineDemonstrator(..., privileged=False, cameras=("agentview",))` refuses `object_footprint`
and `grasped` with `PrivilegedToolError`, naming the camera tool to use instead, so a demonstrator meant for
deployment cannot read simulator state by accident. With several envs each camera must render every env:
`eden datagen --camera_only` sets `env_options.env_separate_rigid` for you; built by hand, the demonstrator refuses
a shared camera. Camera-only runs render every env's view at each routine boundary, so they are slower.

**The trace.** Every ended motion and every `tools.note(...)` is a line of `demo.trace(env_id)`:

```text
 36.75 s  # bread -> goal quadrant (0.149, 0.083)
 37.25 s  move_to(0.214, -0.311, 0.953, yaw=-1.06) -> reached in 1.50 s (off target 10 mm)
 38.75 s  move_to(0.214, -0.311, 0.844, yaw=-1.06) -> timeout in 3.50 s (off target 185 mm)
 47.05 s  # bread: pick 1 missed (closure 1.00)
```

A routine written for one task lives with that task, in `eden/tasks/<suite>/<task>/routine.py`, as a factory that
returns the routine; the suite's `scripted_demos.py` binds it to the scene. A routine several suites share lives
in `eden/extensions/scripted_demos/` (`put_on_routine.py`, used by SIMPLER and PolaRiS). Each routine module's
docstring records the design decisions its rollouts forced — read it before changing the routine.

### Two-arm routines

A two-arm task is two routines, one per arm, each written against its own `Tools` and run side by side by
`MultiArmDemonstrator` (`multi_arm.py`). Each arm resumes its routine when its own motion ends; they coordinate
through `tools.shared`, a dict both routines of an env share for the episode:

```python
from eden.extensions.scripted_demos.multi_arm import MultiArmDemonstrator


def giver(tools):
    ...  # carry the object to the hand-off pose
    tools.shared["offered"] = True
    while not tools.shared.get("taken"):
        yield tools.wait(0.1)
    yield tools.open_gripper()


def taker(tools):
    while not tools.shared.get("offered"):
        yield tools.wait(0.1)
    ...  # close on the object
    tools.shared["taken"] = True


demo = MultiArmDemonstrator(
    env, {"robot0": giver, "robot1": taker},
    workspace={"robot0": ((-0.45, -0.75, 0.805), (0.45, 0.2, 1.3)), "robot1": ((-0.45, -0.2, 0.805), (0.45, 0.75, 1.3))},
)
```

Options that differ per arm are dicts keyed by every arm's scene entry; other options apply to both. A
single-arm `RoutineDemonstrator` on a two-arm task is refused, naming the other arm's terms. An env's task is
complete when both routines have returned; an arm that finishes first holds still. The trace interleaves both
arms' lines, each prefixed with its arm, and the report counts motions per arm (`robot0/move_to:reached`).

### Running a demonstrator by hand

Every demonstrator exposes `reset(envs_idx=None)`, `update_subgoals()` (returns a per-env "done" mask) and
`act()` (returns the action dict). The generator's loop, reduced:

```python
from eden.extensions.scripted_demos.registry import SCRIPTED_DEMO_REGISTRY

demo = SCRIPTED_DEMO_REGISTRY.get("robosuite_lift")(env, seed=0)
demo.reset()
while True:
    done = demo.update_subgoals()
    if done.any():
        env.reset(done)
        demo.reset(done)
    env.step(demo.act())
    if env.reset_buf.any():       # the env ended some episodes itself
        demo.reset(env.reset_buf)
```

Reset the demonstrator for every env the env resets: a routine that keeps running after the env has reset the
scene under it acts on a scene it no longer knows. `generate_demos()` in `eden/extensions/scripted_demos/generate.py`
is the full version, with the recorder, success labels and the time limit.

## Suite notes

These are the constraints found while making each suite's demonstrators succeed; the routine module docstrings
carry the full reasoning.

### robosuite

- **PickPlace** carries objects over the tallest thing in the way, not just the bin rims; places on the robot's
  side of the target cell (the far cell centres are at the edge of the Panda's reach); grasps up to 4.5 cm deep
  (`GRASP_DEPTH`) so the milk carton is not held by its gable top; and waits for the reset's drop to settle before
  looking.
- **Door** leans the hand 30° away from the panel (`LEAN`: a hand pointing straight down has its palm in the
  panel), presses the handle down until the latch clears the frame, then pulls with the rigid back finger and
  turns the hand with the door. Door positions are read with `link_pose`, since the door is re-posed every reset.
- **Wipe** feels for the table with `move_until_contact` at low speed and sweeps just above the touch height: the
  pad is position-controlled, so commanding it into the table pushes back with tens of newtons per millimetre,
  and robosuite's excess-force penalty starts at 60 N. It levels the hand first, and stops at every marker.
- **ToolHang** pinches each object above its centre of mass and expects it to turn in the pinch (there is no
  torsional friction on the pads), re-measuring the grasp after each turn. The wrench's contacts are stiffened
  (`GRASPED_SOL_PARAMS`).
- **TwoArmTransport** hands the hammer across in the air, because neither arm reaches both the start bin and the
  target bin. robot0 lets go only after robot1's fingers have closed (`taken` on `tools.shared`). The robosuite
  Panda rig uses robosuite's gripper actuation, and the carried objects carry stiffened
  `default_geoms_sol_params` so they do not creep out of the pinch. The DROID rig (`--robot droid`) is not covered
  by this routine.

### SIMPLER

- **Run with MPR collision**: `--override 'env_options: {use_gjk_collision: false}'`. Genesis's GJK narrowphase
  can report a non-finite contact between thin convex geoms (the WidowX's finger pads are thin boxes), which a
  long scripted rollout eventually hits. The generator stops with this fix in its message; catching the error and
  resetting does not help, because every env's solver state is invalid after it.
- Every SIMPLER task enables torsional friction with a 2 cm contact patch on the fingers: without it, a spoon
  pinched at its thin handle spins about the line between the pads.
- `RoutineDemonstrator` keeps commanding a move's target through the waits and gripper motions that follow it,
  because the WidowX's damped IK closes the last centimetre slowly and the pads would otherwise close high.
- The put-on routine looks until the object has stopped before picking: eggplants roll across the basin after
  the reset.

### PolaRiS

- The demonstrators are registered on the `*_batched` tasks: `eden datagen` needs the RL surface and many envs,
  and the single-env PolaRiS tasks are teleop and inference scenes.
- Success is PolaRiS's own rubric (`eden/tasks/polaris/rubric.py`, the `rubric` metric of every PolaRiS task):
  reach, lift and placement criteria mapped onto `EeNearEntity`, `object_bottom_above_height` and
  `object_a_within_b_xy`, with `RubricMetricTerm` keeping PolaRiS's bookkeeping.
- Objects wider than the Robotiq 2F-85's stroke (the latte cup, the tape roll) are gripped by their wall
  (`grasp="rim"`). A thin wall barely stops the fingers, so the PolaRiS demonstrators use `holding_closure=0.97`.
- `Goal(container=True)` drops the object anywhere inside a container, avoiding what already lies in it (`avoid=`)
  and staying within reach (`max_reach=`); `clutter=` names entities a pinch keeps its fingers off.
- The batched tasks render with the batch renderer, which privileged datagen does not use. On a GPU it cannot
  launch on, swap in the rasterizer before building:
  `config.renderer_options = genesis.options.renderers.RendererOptions()`. The PolaRiS demonstrators are
  privileged only: their camera-only mode has not been run.

## See also

- [Datasets and recording](datasets-and-recording.md) — the HDF5 layout and `eden lerobot convert`.
- [Teleoperation](teleoperation.md) — recording human demonstrations of the same tasks.
- [Grasp sampling](grasp-sampling.md) — learned grasp poses.
- [robosuite](../tasks/robosuite.md), [SIMPLER](../tasks/simpler.md), [PolaRiS](../tasks/polaris.md) — the suites
  these demonstrators script.
