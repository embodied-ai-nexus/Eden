# Porting a Pretrained Policy

A *frozen* policy trained in another stack (IsaacGym, Isaac Lab, mjlab, MuJoCo, rl_games, an exported ONNX graph)
runs in Eden for inference and sim-to-sim as a **policy extension**, not as a task. The policy keeps its weights and
its observation and action contract; Eden supplies the robot, the scene and the control loop. This page is the
checklist for reproducing such a policy faithfully, and the order in which to verify it. How to package the result
(the class, its files, the registry) is in [Adding a Policy to the Zoo](extending-policies.md).

!!! tip "Training a new policy is a different job"
    This page is for reproducing a policy someone else trained. To train in Eden, write a task config instead: see
    [Task Configuration](task-config.md).

The ports that ship with Eden are worked examples, all under `eden/extensions/policies/`:

- `husky` (G1 skateboarding, ONNX) declares its observation and action through the training task's own terms: the
  model to follow for a new port.
- `holosoma`, `homie`, `sonic` assemble their input from raw simulator state, which is what lets them ride along as
  extra slots in a deployment.
- `twist2`, `omnixtreme`, `humanoid_gpt` (ONNX motion trackers), `protomotionv3` (a torch actor loaded from a
  training checkpoint), and `simtoolreal` (a 29-DOF arm and hand with an rl_games LSTM actor reimplemented in torch).

[The Policy Zoo](policy-zoo.md) lists every one with what it needs.

## The shape of a port

A port is one `PolicyBase` subclass whose classmethods describe the world the policy was trained in:

| Member | What to put there |
|--------|-------------------|
| `robot_options()` | The robot, as an `eden.robots` options object (`G1Dofs29(...)`, `KukaSharpa(...)`) with the source's default pose, gains and armature. |
| `scene_entity_options()` | Every other entity the policy reads: the object it manipulates, the board it rides, a goal marker, or a `plane` entity that replaces the flat ground (for example with terrain). |
| `env_options()` | `sim_dt`, `decimation` and solver settings: the control rate and physics the policy expects. |
| `action_options()` | The action terms its output drives. |
| `observation_options()` | The observation groups it reads. |
| `command_options()` | The command terms it consumes, if any (a velocity command, a motion reference). |
| `step(obs)` | One inference call, returning `{action_term_name: tensor}`. |
| `reset(envs_idx)` | Clear recurrent state and per-episode buffers. |

`robot_options()` and `scene_entity_options()` together **are** the scene: `eden inference --policy <name>` without a
`--task` builds an empty scene from exactly those two. A policy whose observation is defined against something other
than its robot (SimToolReal's tool on its table, HUSKY's skateboard) must declare that entity, or a task-less run
spawns a lone robot and the policy fails resolving the entity. Resolve each entity with `self.require_entity(name)`
in `__init__` so a scene that lacks it fails at construction, and keep reading it through `self.env.entities[name]`
in `step()`, because deployment swaps the entity for a live-robot stand-in.

### Declare the observation; don't assemble it in `step()`

A policy that owns its env (the task-less `eden inference --policy` case) should state its input as
`observation_options()`, with `command_options()` for any command or clock it observes and `action_options()` scaled
so that `last_action` is the actor's raw output. It then sets `reads_env_observations = True`, and its `step()` just
feeds `obs[<group>]` to the network. The layout is explicit and reusable, and it gets the observation manager's
history, scale, NaN and reset handling for free.

`husky` is the worked example. Its `observation_options()` and `action_options()` reuse the training task's actor
group and action term (`eden/tasks/husky/config.py`), so the released checkpoint and a policy trained in Eden share
one layout, and a held `ConstantSkateCommand` carries the command and the phase clock.
`tests/extensions/policies/husky/test_policy.py` pins that env against a hand-written formula of the source's
observation.

!!! warning "A command term that ticks must tick in `pre_physics_advance`"
    A deployment never runs `post_physics_refresh`, so a command term a frozen policy observes (a phase clock, a
    held command) must advance its state in a hook that runs on hardware too. HUSKY's `ConstantSkateCommand` ticks in
    `pre_physics_advance` for that reason.

Assembling the input from raw sim state inside `step()` is for policies that have to **ride along** as extra slots on
someone else's env: the Holosoma safety policy `eden deploy` inserts on a G1, and controllers stacked next to it.
One env publishes one observation layout, so only one env-reading policy can be loaded at a time (see
[Deployment](deployment.md#policy-slots)). Such a policy's `observation_options()` may return `{}`: no groups at all
builds and steps fine, as `holosoma` does. What is **not** allowed is declaring a group and leaving it empty: a group
with `concatenate_terms=True` and no terms fails at construction with
``Unable to concatenate observation terms in group `<name>`.`` Declare one real term or no group.

## The porting checklist

Match the source **exactly**, in this order. Each step's failure is invisible to the steps before it, which is why
the order matters.

### 1. The I/O contract

Reproduce the observation layout and order, any input normalization, and the action transform bit for bit.

- **Index every DOF and link by name.** Genesis' native DOF order differs from the source's. Resolve the names once
  with `robot.find_named_dofs(POLICY_JOINT_NAMES)`, then hand its `.columns` to tensor reads (`default_dofs_*`,
  full-width `get_dofs_*()`) and its `.dofs_idx_local` to Genesis calls (`dofs_idx_local=`). Never index a
  `default_dofs_*` tensor with local indices: they are a different index space (see
  [DOF ordering](entities.md#dof-ordering)).
- **Check the built robot's frozen constants by name, in every env.**
  `robot.get_default_dofs_pos(POLICY_JOINT_NAMES)` returns `(num_envs, n)` in the policy's order; compare it against
  the source's default pose at construction, as `HuskyPolicy.__init__` does.
- **Quaternions are wxyz in Eden**, while many sources put xyzw in their observation vector.
- **Mind world vs. base frame** for every absolute term.
- **Read object and free-body poses with `get_pos(relative=True)` / `get_quat(relative=True)`.** The default returns
  Genesis' solver frame, re-originated at the center of mass, not the declared root pose the source reports (see
  [Which frame?](entities.md#which-frame-relativefalse-vs-relativetrue)). This one fails *quietly*.

### 2. The network

Use `OnnxModule` from `eden/extensions/policies/onnx.py` for feedforward ONNX actors (`twist2`, `sonic` and
`humanoid_gpt` wrap it). Reimplement exotic backends in plain torch, as `simtoolreal/network.py` does for an
rl_games LSTM. Carry recurrent state across `step()` and clear it in `reset()`. When you load weights, confirm there
are no missing or unexpected keys.

### 3. The dynamics

Match the robot's base pose, PD gains (`default_dofs_kp` / `default_dofs_kd`), **armature**
(`default_dofs_armature`: Eden fills `armature=0.1` on every URDF joint whose asset armature is zero, often far too
high for a low-stiffness finger; see [Armature defaults](entities.md#armature-defaults-and-the-urdf-01-trap)), joint
friction, control rate (`sim_dt × decimation`) and gravity compensation. The objects and the scene must match the
world the policy trained in.

### 4. Verify, in layers

This is what catches the bugs.

1. **Observation parity.** Feed the same state `(q, q̇, …)` to the source's observation function and to your
   assembly, and diff field by field (expect differences around `1e-7`). This validates the math *given* a state,
   so synthetic poses never reveal that you read the sim in the wrong frame: also diff a few **sim-measured** states
   against the source's convention.
2. **Action parity.** Feed the same observation to the source's policy and to yours, and diff the action. A faithful
   network reimplementation is bit-identical.
3. **Dynamics audit.** Dump the **built** robot's per-DOF `kp`, `kd` and armature (`get_dofs_armature()`), plus object
   mass, placement and control rate, and compare them with the source. Passing observation and action parity does
   **not** mean the rollout matches: mismatched dynamics make the trajectory diverge while both parity checks stay
   green. When a ported policy underperforms with parity passing, look here first.
4. **Sim-to-sim rollout.** Only after the above, run the closed loop and render it. Two rungs:
   `eden inference --policy <name> --async` replays the loop in Genesis with deployment-style timing, and
   `eden deploy --policy <name> --mujoco --menagerie` replays it against MuJoCo, which catches dynamics assumptions
   only Genesis satisfies (see [Sim-to-sim validation](deployment.md#sim-to-sim-validation-with-mujoco)).

!!! warning "Why the closed loop cannot be the first check"
    A policy servos whatever pose it is shown, so a frame or gain mismatch still *converges*: it just converges to the
    wrong place. A constant offset between object and goal is the classic symptom of the `relative=` mistake. A
    rollout that "mostly works" is not evidence of a correct port.

## See also

- [Adding a Policy to the Zoo](extending-policies.md): the class, its files and how it is discovered
- [Entities](entities.md): pose frames, DOF ordering, armature, contacts
- [Deployment](deployment.md): running a policy on a robot, and validating it in MuJoCo first
