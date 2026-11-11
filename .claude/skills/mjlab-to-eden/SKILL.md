---
name: mjlab-to-eden
description: Converts mjlab (MuJoCo Warp) RL task configurations to Eden (Genesis) format. Covers architecture, config, observations, rewards, terminations, events, commands, actions, coordinate systems, and common pitfalls.
allowed-tools: Read, Grep, Glob, Bash, Edit, Write, Agent
argument-hint: [mjlab-task-path]
---

You are an expert at converting mjlab RL task configurations to the Eden framework.
Below is a comprehensive reference for performing this conversion. When the user asks you to convert an mjlab task, follow this guide precisely.

If `$ARGUMENTS` is provided, it is the path to the mjlab task directory or config file to convert.

---

# mjlab-to-Eden Conversion Guide

## 1. Architectural Overview

### mjlab (source)
- Built on **MuJoCo Warp** for GPU-accelerated simulation.
- Config system uses Python **dataclasses** (`@dataclass`).
- Task config is a plain function returning `ManagerBasedRlEnvCfg`.
- Entity state accessed via `Entity.data.*` properties (e.g., `entity.data.root_link_lin_vel_b`).
- Sensors are explicit objects (IMU, ContactSensor, RayCast) registered in `SceneCfg.sensors`.
- Terms are plain functions receiving `(env, **params)`.
- Entity/link selection uses `SceneEntityCfg("robot", body_names=..., joint_names=..., geom_names=...)` with regex patterns.
- Line length: 88 columns.

### Eden (target)
- Built on **Genesis** for GPU-accelerated simulation.
- Config system uses **Pydantic** models (`EdenRLConfig` for RL tasks, `EdenMimicConfig` for mimic, `EdenConfig` base).
- Task config is a **class** inheriting `EdenRLConfig` (RL tasks) or `EdenConfig` (non-RL) with typed fields.
- Entity state accessed via method calls (e.g., `entity.get_vel(frame="body")`).
- Sensors are available via Genesis (`gs.sensors.*`): `Contact`, `ContactForce`, `IMU`, `Lidar`, `DepthCamera`. Configured through `SensorsOptions` / `SensorOptions` and attached to entity links. Terrain height scanning also available via `GridScanDots` (uses `terrain.get_height()` directly, no raycast needed).
- Terms can be **functions** (`@REGISTRY.register()`) or **classes** (inherit `RewardTerm`, `TerminationTerm`, etc.).
- Entity/link selection uses explicit `entity_name` + `links_name` params (no SceneEntityCfg wrapper).
- All config fields use `*_options` suffix: `env_options`, `scene_options`, `observation_options`, `action_options`, `reward_options`, `termination_options`, `event_options`, `command_options`, `curriculum_options`, `runner_options`, `motion_options`. Exception: `cameras_options` and `sensors_options` use plural.
- Line length: 120 columns.
- **Always use absolute imports** in non-`__init__.py` modules.
- Use `TYPE_CHECKING` guards for heavy imports that cause circular dependencies.

---

## 2. Config Structure Mapping

### mjlab
```python
def make_task_env_cfg() -> ManagerBasedRlEnvCfg:
    observations = {"actor": ObservationGroupCfg(...), "critic": ObservationGroupCfg(...)}
    actions = {"joint_pos": JointPositionActionCfg(...)}
    commands = {"twist": UniformVelocityCommandCfg(...)}
    events = {"push_robot": EventTermCfg(...)}
    rewards = {"reward_name": RewardTermCfg(...)}
    terminations = {"time_out": TerminationTermCfg(...)}
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(...), observations=observations, actions=actions,
        commands=commands, events=events, rewards=rewards, terminations=terminations,
        sim=SimulationCfg(...), decimation=4, episode_length_s=20.0,
    )
```

### Eden equivalent
```python
class MyTaskConfig(EdenRLConfig):
    env_options: EnvOptions = EnvOptions(
        num_envs=4096, episode_length_s=20.0, sim_dt=0.005,
        decimation=4, iterations=10, ls_iterations=20, ...
    )
    scene_options: SceneOptions = en.scenes.Empty(robot=RobotOptions(...))

    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        policy=ObservationGroupOptions(...),
        critic=ObservationGroupOptions(...),
    )
    reward_options: RewardManagerOptions = RewardManagerOptions(...)
    termination_options: TerminationManagerOptions = TerminationManagerOptions(...)
    event_options: EventManagerOptions = EventManagerOptions(...)
    action_options: ActionManagerOptions = ActionManagerOptions(...)
```

Or use `__init__` when the scene/actions/commands depend on constructor arguments:

```python
class MyTaskConfig(EdenRLConfig):
    env_options: EnvOptions = EnvOptions(...)
    # ...fields that don't depend on constructor args...

    def __init__(self, robot_options=None, **data):
        super().__init__(**data)
        if robot_options is None:
            robot_options = G1Dofs29()
        self.scene_options = en.scenes.Empty(robot=robot_options)
        self.action_options = ActionManagerOptions(
            dofs_pos_controller=actions.ExplicitPDController.configure(
                entity_name="robot", dofs_name=robot_options.dofs_name, ...
            )
        )
```

### Key Mapping Table

| mjlab | Eden |
|-------|------|
| `ManagerBasedRlEnvCfg` | `EdenRLConfig` subclass (or `EdenConfig` for non-RL) |
| `SceneCfg(terrain=..., entities=..., sensors=...)` | `scene_options: SceneOptions = en.scenes.Empty(robot=...)` as field or in `__init__` |
| `SimulationCfg(mujoco=MujocoCfg(timestep=...))` | `env_options: EnvOptions = EnvOptions(sim_dt=..., iterations=..., ls_iterations=...)` |
| `decimation=4` | `EnvOptions(decimation=4)` (inside `env_options`) |
| `episode_length_s=20.0` | `EnvOptions(episode_length_s=20.0)` (inside `env_options`) |
| `ObservationGroupCfg(terms=dict, ...)` | `ObservationGroupOptions(**terms, concatenate_terms=True, ...)` |
| `ObservationTermCfg(func=..., params=..., noise=...)` | `ObsTerm.configure(func=..., params=..., noise=UniformNoise.configure(...))` |
| `ObservationTermCfg(..., delay_min_lag=k, delay_max_lag=k)` | `ObsTerm.configure(..., modifiers=[ObservationDelay.configure(control_steps=k)])` (from `eden.managers.modifiers.observations`). Fixed lag only: Eden has no per-env random observation lag, so `min < max` has no equivalent. |
| `ObservationGroupCfg(nan_policy=...)` | `ObservationGroupOptions(nan_policy=...)`, with the same four values and the same placement (each term's output after clip and scale). Eden always checks per term (there is no `nan_check_per_term`), `"warn"` logs once per term rather than on every step, and `"error"` raises `FloatingPointError` where mjlab raises `ValueError`. |
| `RewardTermCfg(func=..., weight=..., params=...)` | `RewardTerm.configure(func=..., weight=..., params=...)` |
| `TerminationTermCfg(func=..., time_out=...)` | `DoneTerm.configure(func=..., time_out=...)` |
| `EventTermCfg(func=..., mode="reset")` | `EventTerm.configure(func=..., mode="reset", ...)` or `EventClass.configure(mode=en.EventMode.RESET, ...)` |
| `CommandTermCfg` (commands dict) | `command_options: CommandManagerOptions` with command terms |
| `JointPositionActionCfg(entity_name=..., scale=...)` | `actions.ExplicitPDController.configure(entity_name=..., dofs_name=..., scale=...)` |
| `SceneEntityCfg("robot", joint_names=...)` | `params={"entity_name": "robot"}` directly |

---

## 3. Observation Term Conversion

### 3.1 Built-in Observations

| mjlab | Eden | Notes |
|-------|------|-------|
| `mdp.base_lin_vel` (via `entity.data.root_link_lin_vel_b`) | `observations.base_lin_vel` with `frame="body"` | **CRITICAL**: mjlab always returns body-frame velocity. Eden defaults to `frame="world"` -- you MUST specify `frame="body"` |
| `mdp.base_ang_vel` (via `entity.data.root_link_ang_vel_b`) | `observations.base_ang_vel` with `frame="body"` | Same body-frame issue |
| `mdp.builtin_sensor(sensor_name="robot/imu_lin_vel")` | `observations.base_lin_vel` with `frame="body"` | mjlab's IMU sensor returns body-frame velocity; Eden also has `gs.sensors.IMU` but the built-in obs with `frame="body"` is simpler |
| `mdp.builtin_sensor(sensor_name="robot/imu_ang_vel")` | `observations.base_ang_vel` with `frame="body"` | Same as above |
| `mdp.joint_pos_rel(biased=True)` | `observations.DofsPos` with `offset_from_default=True` | mjlab's `biased=True` adds encoder bias (from DR); Eden uses `offset_from_default=True` for the default offset |
| `mdp.joint_vel_rel` | `observations.DofsVel` | Eden returns raw velocity (mjlab subtracts default, but default is typically zero) |
| `mdp.last_action` | `observations.last_action` | Direct mapping |
| `mdp.generated_commands(command_name="...")` | `observations.generated_commands` with `params={"command_name": "..."}` | Direct mapping -- command name must match the key in `CommandManagerOptions` |
| `mdp.projected_gravity` | `observations.projected_gravity` | Direct mapping |
| `envs_mdp.height_scan(sensor_name=...)` | `GridScanDots.configure(entity_name=..., terrain_entity_name=...)` | mjlab uses RayCast sensor; Eden uses `GridScanDots` (class-based obs term using `terrain.get_height()`) or `gs.sensors.Lidar` for general raycast |
| Sensor reading (any) | `observations.sensor_reading` with `params={"sensor_name": "..."}` | Generic sensor obs; reads from `env.sensors["name"]` |

### 3.2 Custom Observation Conversion Pattern

**mjlab pattern** (positional params, accesses command/entity via scene):
```python
def my_obs(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = cast(MyCommand, env.command_manager.get_term(command_name))
    asset: Entity = env.scene["robot"]
    return some_computation(asset.data.root_link_pos_w, command.target_pos)
```

**Eden pattern** (keyword-only params, accesses entity via `env.entities`):
```python
@OBSERVATION_TERM_REGISTRY.register()
def my_obs(env: EnvBase, *, entity_name: str = "robot") -> torch.Tensor:
    entity = env.entities[entity_name]
    robot_pos = entity.get_pos()
    return some_computation(robot_pos)
```

### 3.3 Key Differences in Custom Observations

1. **Entity access**: mjlab uses `env.scene["robot"]` -> Eden uses `env.entities["robot"]`.
2. **State access**: mjlab reads `entity.data.body_link_pos_w[:, idx]` -> Eden calls `entity.get_links_pos(ls_idx_local=idx)`.
3. **Link indexing**: mjlab uses `entity.find_bodies(names)` -> Eden uses `entity.find_named_links_idx_local(names, preserve_order=True)`.
4. **Registration**: Eden functions need `@OBSERVATION_TERM_REGISTRY.register()`.
5. **Keyword-only params**: Eden term functions use `*` separator for keyword-only arguments (no positional params after env).
6. **Noise**: mjlab uses `UniformNoiseCfg(n_min=..., n_max=...)` -> Eden uses `UniformNoise.configure(n_min=..., n_max=...)`.

### 3.4 Observation Group Mapping

| mjlab | Eden |
|-------|------|
| `"actor"` group | `"policy"` group |
| `"critic"` group | `"critic"` group |
| `ObservationGroupCfg(terms={...}, concatenate_terms=True, enable_corruption=True)` | `ObservationGroupOptions(**terms, concatenate_terms=True, enable_corruption=True)` |

**IMPORTANT**: mjlab calls the actor observation group `"actor"`, Eden calls it `"policy"`.

---

## 4. Reward Term Conversion

### 4.1 Built-in Rewards

| mjlab | Eden |
|-------|------|
| `mdp.action_rate_l2` | `rewards.action_rate_l2` |
| `mdp.joint_pos_limits` with `SceneEntityCfg("robot", joint_names=...)` | `rewards.dofs_pos_limits` with `params={"entity_name": "robot"}` |
| `mdp.self_collision_cost` with `sensor_name=..., force_threshold=...` | `rewards.self_collision_cost` with `params={"entity_name": "robot"}` |
| `mdp.track_linear_velocity` | `rewards.track_lin_vel_with_z_penalty` | Note: Eden adds a 2x multiplier on z-error |
| `mdp.track_angular_velocity` | `rewards.track_ang_vel_with_xy_penalty` | Note: Eden adds a 0.05x multiplier on xy-error |

### 4.2 Custom Reward Conversion

**mjlab**: Rewards are functions or callable classes.

Function:
```python
def my_reward(env, command_name: str, std: float) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    asset: Entity = env.scene["robot"]
    ...
```

Callable class (stateful, with `__init__` and `__call__`):
```python
class my_reward:
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.cached_data = ...
    def __call__(self, env, ...) -> torch.Tensor:
        ...
    def reset(self, env_ids): ...
```

**Eden**: Simple rewards are functions; stateful rewards are classes inheriting `RewardTerm`.

Function-based:
```python
@REWARD_TERM_REGISTRY.register()
def my_reward(env: EnvBase, *, entity_name: str, std: float) -> torch.Tensor:
    robot = env.entities[entity_name]
    ...
```

Class-based:
```python
@REWARD_TERM_REGISTRY.register()
class MyReward(RewardTerm):
    entity_name: str = "robot"
    std: float = 0.1
    _cached_idx: torch.Tensor | None = None

    def compute(self, envs_idx: torch.Tensor | None = None) -> torch.Tensor:
        ...
```

Key differences:
- mjlab callable classes use `__init__(cfg, env)` + `__call__(env, **params)` -> Eden classes use Pydantic fields + `compute(envs_idx)`.
- mjlab `env.scene["robot"]` -> Eden `self.resolve_entity("robot")` (class — resolve it in `compute()`, not cached in `build()`; see the `writing-eden-terms` skill) or `env.entities["robot"]` (function).
- mjlab `env.command_manager.get_command(name)` -> Eden `env.command_manager.get_term(name).command`.

---

## 5. Command Term Conversion

### 5.1 Velocity Commands

mjlab's `UniformVelocityCommandCfg` maps to Eden's `UniformVelocityCommand`:

| mjlab | Eden |
|-------|------|
| `UniformVelocityCommandCfg(entity_name="robot", ranges=Ranges(lin_vel_x=(...), ...), heading_command=True, ...)` | `UniformVelocityCommand.configure(entity_name="robot", lin_vel_x_range=(...), ..., heading_command=True, ...)` |
| `resampling_time_range=(3.0, 8.0)` | `resampling_time_range=(3.0, 8.0)` |
| `rel_standing_envs=0.1` | `rel_standing_envs=0.05` |
| `rel_heading_envs=0.3` | `rel_heading_envs=0.25` |

Note: mjlab nests velocity ranges inside a `Ranges` dataclass; Eden flattens them as direct fields (`lin_vel_x_range`, `lin_vel_y_range`, `ang_vel_z_range`, `heading_range`).

### 5.2 Custom Commands

mjlab `CommandTerm` has:
- `cfg` attribute, `_env` attribute
- `command` property -> the tensor returned to observations
- `_resample_command(env_ids)` -> called on reset
- `_update_command()` -> called each step
- `_update_metrics()` -> called each step for logging

Eden `CommandTerm` has:
- Pydantic fields for configuration
- `command` property -> same
- `_resample_command(envs_idx)` -> same role, but `envs_idx` is a `(num_envs,)` bool mask, not an `env_ids` index
  tensor: `len(envs_idx)` is `num_envs`, so draw full-batch and keep the selected envs with `torch.where`
  (idiom in the `CommandTerm._resample_command` docstring)
- `_update_command()` -> same
- `build()` -> one-time setup after environment is constructed

---

## 6. Termination Term Conversion

### 6.1 Built-in Terminations

| mjlab | Eden |
|-------|------|
| `mdp.time_out` with `time_out=True` | `terminations.time_out` with `time_out=True` |
| `mdp.bad_orientation(limit_angle=...)` | `terminations.fell_over(entity_name=..., limit_angle=...)` |

### 6.2 Custom Termination Patterns

**mjlab**: Function-based, receives `SceneEntityCfg` for entity access.
```python
def bad_orientation(env, limit_angle: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    projected_gravity = asset.data.projected_gravity_b
    return torch.acos(-projected_gravity[:, 2]).abs() > limit_angle
```

**Eden**: Function-based or class-based, uses `entity_name` parameter.
```python
@TERMINATION_TERM_REGISTRY.register()
def fell_over(env: EnvBase, *, entity_name: str, limit_angle: float) -> torch.Tensor:
    robot: Entity = env.entities[entity_name]
    inv_base_quat = inv_quat(robot.get_quat())
    grav = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, -1)
    projected_gravity = transform_by_quat(grav, inv_base_quat)
    cos_angle = -projected_gravity[:, 2]
    cos_limit = torch.cos(torch.tensor(limit_angle, device=env.device))
    return cos_angle < cos_limit
```

### 6.3 Gravity Source

- mjlab: `asset.data.projected_gravity_b` or `asset.data.gravity_vec_w`
- Eden: compute manually via `inv_quat(robot.get_quat())` + `transform_by_quat(grav, inv_quat)`, or `env.rigid_solver.get_gravity()`

---

## 7. Event Term Conversion

### 7.1 Event Modes

| mjlab | Eden |
|-------|------|
| `mode="startup"` | `mode=en.EventMode.STARTUP` or `mode="startup"` |
| `mode="reset"` | `mode=en.EventMode.RESET` or `mode="reset"` |
| `mode="interval"` with `interval_range_s=(...)` | `mode=en.EventMode.INTERVAL` with `interval_range_s=(...)` |
| `mode="step"` | `mode=en.EventMode.STEP` or `mode="step"` |

### 7.2 Built-in Events

| mjlab | Eden | Notes |
|-------|------|-------|
| `mdp.push_by_setting_velocity` | `events.PushByVelocity.configure(...)` | Class-based in Eden; needs `entity_name` and `velocity_range` dict |
| `dr.body_com_offset` | `events.RandomizeComShift.configure(...)` | Different param names: `links_name`, `com_shift_{x,y,z}_range` |
| `dr.encoder_bias` | `events.RandomizeDofsPosOffset.configure(...)` | Approximate equivalent; set `dofs_pos_range`. `action_term_name` defaults to `None` = every joint position controller on `entity_name` (or name one / a list) |
| `dr.geom_friction` | `events.RandomizeFrictionRatio.configure(...)` | Uses `links_name` and `friction_range` instead of `geom_names` and `ranges` |
| `mdp.reset_root_state_uniform` | `en.events.reset_base_state_uniform` | Different param names: `pos_{x,y,z}_range`, `roll_range` / `pitch_range` / `yaw_range`. Eden composes the rotation in the world frame (`R_rpy @ R_default`), mjlab in the body frame (`R_default @ R_rpy`) — same for an identity default quat |
| `mdp.reset_joints_by_offset` | `en.events.reset_dofs_by_offset` | Uses `dofs_pos_range` and `dofs_vel_range` |

### 7.3 Domain Randomization Parameter Mapping

| mjlab | Eden |
|-------|------|
| `asset_cfg=SceneEntityCfg("robot", body_names=("torso_link",))` | `entity_name="robot", links_name=["torso_link"]` |
| `ranges={0: (-0.025, 0.025), 1: (-0.05, 0.05), 2: (-0.05, 0.05)}` | `com_shift_x_range=(-0.025, 0.025), com_shift_y_range=(-0.05, 0.05), com_shift_z_range=(-0.05, 0.05)` |
| `asset_cfg=SceneEntityCfg("robot", geom_names=r"^(left\|right)_foot.*$")` | `entity_name="robot", links_name=["*_ankle_roll_link"]` |
| `ranges=(0.3, 1.2)` | `friction_range=(0.3, 1.2)` |

### 7.4 Friction Domain Randomization (max-combine × absolute ratios × low-friction ground)

Porting a `dr.geom_friction` setup is NOT a field-name mapping — three semantic differences compose,
and missing any one of them makes the DR silently ineffective:

1. **Genesis combines pair friction as `max(a, b)`** (MuJoCo uses geom `priority` / elliptic
   combine). A randomized coefficient only takes effect when it is the larger side of the pair.
2. **`RandomizeFrictionRatio` is absolute.** It pins the entity's base friction to 1.0 entity-wide
   (clobbering any authored `default_geoms_friction`) and then writes per-link ratios — so
   `friction_range` is an absolute coefficient range. Convert a source "scale × authored value"
   range into the resulting absolute range before porting.
3. **The ground must not win the max.** mjlab scenes give feet `priority=1`, so the foot coefficient
   is used against any ground. The Eden equivalent: author the plane with low friction so the
   randomized robot side always wins — `plane=PlaneOptions(default_geoms_friction={"*": 0.01})`.

Overlapping link sets (whole-robot pass + a tighter feet range) are ordered by the events'
`priority` field: startup events run in ascending priority, so give the feet event the higher
number. Worked, training-validated example: `eden/tasks/husky/config.py` (plane + `robot_friction`
priority=0 + `foot_friction` priority=10). Full recipe: `docs/guides/entities.md`, "Porting a
friction-randomization setup". One more gap to check: MuJoCo `friction[1]`/`friction[2]`
(torsional/rolling) DR has no effect unless the scene opts in (`EnvOptions.enable_*_friction`) —
Genesis multiplies the same ratio into those axes only where they exist.

---

## 8. Action Term Conversion

| mjlab | Eden |
|-------|------|
| `JointPositionActionCfg(entity_name="robot", actuator_names=(".*",), scale=0.5, use_default_offset=True)` | `actions.ExplicitPDController.configure(entity_name="robot", dofs_name=robot_options.dofs_name, scale=G1_ACTION_SCALE, reference_source=ReferenceSource.DEFAULT)` (import from `eden.constants`) |

**Key difference**: mjlab uses regex `actuator_names` to match actuators; Eden requires an explicit `dofs_name` list. Get this from the robot options (e.g., `G1Dofs29().dofs_name`).

**Action scale**: mjlab uses a single `scale=0.5`; Eden can use per-DOF scaling via a dict (e.g., `G1_ACTION_SCALE`).

---

## 9. Coordinate System and Convention Pitfalls

### 9.1 Quaternion Convention
Both frameworks use **wxyz** format (w, x, y, z). No conversion needed.

### 9.2 Velocity Frame (CRITICAL PITFALL)

**mjlab**: `base_lin_vel` and `base_ang_vel` observations **always** return body-frame velocities.
- `entity.data.root_link_lin_vel_b` = `quat_apply_inverse(quat_w, lin_vel_w)`
- IMU sensors also return body-frame data.

**Eden**: `observations.base_lin_vel` and `observations.base_ang_vel` default to **world-frame**.
- You MUST pass `frame="body"` to get body-frame velocities.
- Forgetting this will silently produce wrong observations and the policy will fail to train.

```python
# WRONG (world frame by default):
base_lin_vel=ObsTerm.configure(func=observations.base_lin_vel, params={"entity_name": "robot"})

# CORRECT (body frame, matching mjlab):
base_lin_vel=ObsTerm.configure(func=observations.base_lin_vel, params={"entity_name": "robot", "frame": "body"})
```

### 9.3 Joint Velocity Default Offset

**mjlab**: `joint_vel_rel` subtracts `default_joint_vel` (typically zero).
**Eden**: `DofsVel` returns raw velocity without subtracting defaults.
In practice, since default joint velocity is zero for most robots, this is a non-issue.

### 9.4 Link Velocities

Both frameworks provide link velocities in **world frame** by default.
- mjlab: `entity.data.body_link_lin_vel_w[:, idx]` / `entity.data.body_link_ang_vel_w[:, idx]`
- Eden: `entity.get_links_vel(ls_idx_local=idx)` / `entity.get_links_ang(ls_idx_local=idx)`

### 9.5 Orientation Representation

Both frameworks can represent orientation as the **first 2 columns of the 3x3 rotation matrix** (6D representation, more stable than Euler angles):
```python
mat = matrix_from_quat(quat)    # (N, 3, 3)
ori_6d = mat[..., :2].reshape(N, -1)  # (N, 6)
```

---

## 10. Step Loop Differences

### mjlab step order:
```
1. action_manager.process_action(action)
2. for _ in decimation: apply_action -> write_to_sim -> sim.step -> scene.update
3. termination_manager.compute()
4. reward_manager.compute()
5. _reset_idx(reset_env_ids)
6. sim.forward()              <-- single forward call for all envs
7. command_manager.compute()
8. event_manager.apply("interval")
9. sim.sense()
10. observation_manager.compute()
```

### Eden step order (RLEnvBase):
```
1. action_manager.compute(action)
2. command_manager.pre_physics_advance()      <-- time-advancing refs (e.g. motion clips) advance here
3. for _ in decimation: apply_actions -> scene.step
4. command_manager.post_physics_refresh()     <-- per-step caches refresh on the post-physics state
5. metric_manager.compute()
6. termination_manager.compute()
7. reward_manager.compute()
8. _reset_idx(reset_buf):
   a. scene.reset()
   b. event_manager.compute(RESET)
   c. every manager's reset()
9. curriculum_manager.compute() (training only)
10. command_manager.compute() (resample / update)
11. event_manager.compute(INTERVAL)
12. observation_manager.compute()
```

**Key differences**:
- Eden has no explicit `sim.forward()` / `sim.sense()` calls -- Genesis handles this internally.
- Both frameworks compute observations **after** reset, so returned observations reflect the fresh post-reset state.
- Reset order: mjlab resets scene then applies events; Eden does the same.
- A custom-env step loop that refreshes shared state after physics ports onto
  `CommandTerm.post_physics_refresh()` (step 4) — but note it runs BEFORE resets (step 8), so the
  term's `reset()` must rebuild any cache that observations (step 12) will read. See
  `docs/guides/terms.md`, "Term state lifecycle".

---

## 11. Entity and Scene Configuration

### mjlab
```python
cfg.scene = SceneCfg(
    terrain=TerrainEntityCfg(terrain_type="plane"),
    entities={"robot": get_g1_robot_cfg()},
    sensors=(ContactSensorCfg(...),),
    num_envs=1,
)
```
Entities and sensors are separate scene members.

### Eden
```python
scene_options: SceneOptions = en.scenes.Empty(robot=G1Dofs29())
```
Robot is passed as a kwarg to `en.scenes.Empty`. Self-collision and collision settings are on `EnvOptions`:
```python
env_options: EnvOptions = EnvOptions(
    enable_self_collision=False,
    max_collision_pairs=64,
    ...
)
```

---

## 12. Sensor Conversion

### 12.1 Sensor Infrastructure

mjlab registers sensors inside `SceneCfg.sensors` as dedicated sensor objects (`ContactSensorCfg`, `RayCastSensorCfg`, `TerrainHeightSensorCfg`).

Eden configures sensors via the `sensors_options` field on the config class, using Genesis sensor primitives:

```python
from eden.options.sensors import SensorOptions, SensorsOptions
import genesis as gs

class MyTaskConfig(EdenRLConfig):
    # ... other fields ...

    sensors_options: SensorsOptions = SensorsOptions(
        contact_left_foot=SensorOptions(
            sensor=gs.sensors.Contact(),
            attach_entity_name="robot",
            attach_link_name="left_ankle_roll_link",
        ),
        contact_force_right_foot=SensorOptions(
            sensor=gs.sensors.ContactForce(),
            attach_entity_name="robot",
            attach_link_name="right_ankle_roll_link",
        ),
        imu=SensorOptions(
            sensor=gs.sensors.IMU(delay=0.01, acc_noise=(0.01, 0.01, 0.01)),
            attach_entity_name="robot",
        ),
        lidar=SensorOptions(
            sensor=gs.sensors.Lidar(pattern=gs.sensors.GridPattern()),
            attach_entity_name="robot",
        ),
    )
```

Reading sensor data: `env.sensors["contact_left_foot"].read()` or via observation terms `observations.sensor_reading` / `observations.sensor_ground_truth`.

### 12.2 Sensor Type Mapping

| mjlab | Eden | Notes |
|-------|------|-------|
| `ContactSensorCfg(primary=..., secondary=..., fields=("found", "force"))` | `SensorOptions(sensor=gs.sensors.Contact(), attach_link_name=...)` | `gs.sensors.Contact` returns boolean; `gs.sensors.ContactForce` returns 3D force vector |
| `RayCastSensorCfg(frame=..., pattern=GridPatternCfg(...))` | `gs.sensors.Lidar(pattern=gs.sensors.GridPattern())` or `GridScanDots` (for terrain) | `GridScanDots` uses `terrain.get_height()` directly (faster, no raycast); `gs.sensors.Lidar` for general-purpose raycast |
| `TerrainHeightSensorCfg(frame=...)` | `GridScanDots.configure(entity_name=..., terrain_entity_name=...)` | Uses terrain height field interpolation |
| IMU (`builtin_sensor("robot/imu_*")`) | `gs.sensors.IMU(acc_noise=..., delay=...)` or built-in obs `base_lin_vel`/`base_ang_vel` with `frame="body"` | Built-in obs is simpler for velocity; use `gs.sensors.IMU` if you need configurable noise/delay |

### 12.3 Terrain Height Scanning

Eden's `GridScanDots` observation term scans terrain height at a grid of XY points in the robot's base frame (see `examples/recipes/sensors/sensors_lidar_go2.py` for lidar alternative):

```python
from eden.managers.terms.observations.locomotion import GridScanDots

# In observation group:
height_scan=GridScanDots.configure(
    entity_name="robot",
    terrain_entity_name="terrain",
    measured_points_x=[-0.8, -0.7, ..., 0.7, 0.8],  # 17 points
    measured_points_y=[-0.5, -0.4, ..., 0.4, 0.5],   # 11 points
    offset_from_base=True,
    noise=UniformNoise.configure(n_min=-0.1, n_max=0.1),
)
```

mjlab equivalent: `height_scan` obs with `RayCastSensorCfg` + `GridPatternCfg(size=(1.6, 1.0), resolution=0.1)`.

### 12.4 Contact-Based Observations and Rewards

mjlab's contact sensor provides rich data (found, force, air_time, first_contact). In Eden, build contact-based features on top of Genesis sensors:

- **Contact state**: `gs.sensors.Contact()` returns boolean per link per step
- **Contact force**: `gs.sensors.ContactForce()` returns 3D force per link per step (supports noise, bias, jitter, min/max clamping)
- **Air time / first contact**: Not built-in; track in a custom class-based term (accumulate time between contact transitions)
- **Foot height**: Use link z-position (`entity.get_links_pos()`) or `GridScanDots` per-foot

See `examples/recipes/sensors/sensors_imu_contact_go1.py` for a working contact sensor setup.

---

## 13. Learning / Runner Configuration

mjlab and Eden both support RSL-RL, but Eden configures it inline in the task config:

```python
runner_options: RslRlOnPolicyRunnerOptions = RslRlOnPolicyRunnerOptions(
    num_steps_per_env=24,
    max_iterations=30000,
    actor=RslRlActorOptions(hidden_dims=[512, 256, 128], activation="elu"),
    critic=RslRlCriticOptions(hidden_dims=[512, 256, 128], activation="elu"),
    algorithm=RslRlPpoAlgorithmOptions(learning_rate=0.001, ...),
)
```

mjlab keeps RL config in separate `rl_cfg.py` files.

---

## 14. Helper Utilities

### Link Index Caching

mjlab resolves indices via `SceneEntityCfg.resolve()` at manager init time. The Eden equivalent is a
**class term** that resolves in `build()` with the shared `resolve_link_idx` helper:

```python
from eden.managers.terms.utils import resolve_link_idx

@REWARD_TERM_REGISTRY.register()
class MyFeetReward(RewardTerm):
    entity_name: str = "robot"
    feet_link_names: list[str] = []

    def build(self) -> None:                       # once, after the scene exists
        robot = self.resolve_entity(self.entity_name)
        self._ls_idx = resolve_link_idx(robot, self.feet_link_names, self.device)

    def compute(self) -> torch.Tensor:             # hot path
        pos = self.resolve_entity(self.entity_name).get_links_pos(ls_idx_local=self._ls_idx)
        ...
```

**Never** cache on the env (`setattr(env, f"_ls_idx_{entity_name}", ...)`) and never resolve names
lazily in the first `compute()`: an env stash leaks across terms, has no reset semantics, and forces
a `hasattr` branch on the hot path. Resolve *names* in `build()`, resolve the *entity handle* in
`compute()` — the `writing-eden-terms` skill (§2, §3, §3.1) and `docs/guides/terms.md` are
authoritative here.

---

## 15. Common Pitfalls Checklist

When converting any mjlab task to Eden, verify each of these:

- [ ] **Velocity frame**: `base_lin_vel` and `base_ang_vel` must use `frame="body"` to match mjlab
- [ ] **Observation group name**: mjlab `"actor"` -> Eden `"policy"`
- [ ] **Registry decorators**: All custom terms need `@REGISTRY.register()`
- [ ] **Keyword-only args**: Eden term functions use `*, entity_name: str` pattern (keyword-only after `*`)
- [ ] **Absolute imports**: Use absolute imports in all non-`__init__` files
- [ ] **`TYPE_CHECKING` guards**: Import env types (e.g., `EnvBase`, `RLEnvBase`) only under `TYPE_CHECKING`
- [ ] **Entity access**: mjlab `env.scene["robot"]` -> Eden `env.entities["robot"]` in a function term, `self.resolve_entity("robot")` inside a class term's `compute()` (never cached in `build()`)
- [ ] **Link methods**: mjlab `entity.data.body_link_pos_w[:, idx]` -> Eden `entity.get_links_pos(ls_idx_local=idx)`
- [ ] **DOF indexing**: Eden requires `find_named_dofs_idx_local` for Genesis tree order indices
- [ ] **Self-collision config**: mjlab uses `ContactSensorCfg`; Eden uses `EnvOptions(enable_self_collision=True)` and built-in `rewards.self_collision_cost`
- [ ] **Sensor conversion**: mjlab `ContactSensorCfg` -> `SensorOptions(sensor=gs.sensors.Contact())` or `gs.sensors.ContactForce()`; mjlab `RayCastSensorCfg` -> `GridScanDots` (terrain) or `gs.sensors.Lidar` (general); mjlab `TerrainHeightSensorCfg` -> `GridScanDots`
- [ ] **No `set_vel`**: Eden Entity wrapper has no `set_vel` -- use `_entity.set_vel()` or `set_dofs_vel` for velocity setting
- [ ] **Action scale**: Eden uses per-DOF scale dict (e.g., `G1_ACTION_SCALE`); mjlab uses scalar
- [ ] **Event priority**: Eden events have a `priority` field for ordering within a mode; mjlab doesn't
- [ ] **SceneEntityCfg removal**: mjlab wraps entity selection in `SceneEntityCfg`; Eden passes `entity_name`, `links_name`, etc. as direct params

---

## 16. File Organization Template

For a new Eden task converted from mjlab:

```
eden/tasks/<task_name>/
    config.py              # EdenRLConfig subclass (or EdenMimicConfig for mimic tasks)
    custom_terms/
        __init__.py
        observations.py    # @OBSERVATION_TERM_REGISTRY.register()
        rewards.py         # @REWARD_TERM_REGISTRY.register()
        terminations.py    # @TERMINATION_TERM_REGISTRY.register()
        events.py          # @EVENT_TERM_REGISTRY.register()
```

Or for simpler tasks that reuse only built-in terms:

```
eden/tasks/<task_name>/
    config.py              # Everything in one file
```

---

## 17. Quick Reference: Math Utilities

Eden has no `isaac_math` module. The quaternion / frame math
lives in two places: ops with an exact torch equivalent in
`genesis.utils.geom` are called there directly (note the **vec-first** argument
order on the apply ops, and that genesis `quat_to_R` replaces `matrix_from_quat`);
everything else lives in `eden.utils.geom`. All quaternions are wxyz.

| mjlab (`mjlab.utils.lab_api.math`) | Eden |
|-------------------------------------|------|
| `quat_mul(q1, q2)` | `eden.utils.geom.quat_mul(q1, q2)` |
| `quat_inv(q)` | `eden.utils.geom.quat_inv(q)` |
| `quat_apply(q, v)` | `genesis.utils.geom.transform_by_quat(v, q)` |
| `quat_apply_inverse(q, v)` | `genesis.utils.geom.inv_transform_by_quat(v, q)` |
| `quat_error_magnitude(q1, q2)` | `eden.utils.geom.quat_error_magnitude(q1, q2)` |
| `quat_from_euler_xyz(r, p, y)` | `eden.utils.geom.quat_from_euler_xyz(r, p, y)` |
| `matrix_from_quat(q)` | `genesis.utils.geom.quat_to_R(q)` |
| `subtract_frame_transforms(t1, q1, t2, q2)` | `eden.utils.geom.subtract_frame_transforms(t1, q1, t2, q2)` |
| `yaw_quat(q)` | `eden.utils.geom.yaw_quat(q)` |
| `sample_uniform(lo, hi, size, device)` | `eden.utils.sample.sample_uniform(lo, hi, size, device)` |
| `wrap_to_pi(angle)` | `eden.utils.geom.wrap_to_pi(angle)` |

---

## Appendix A: Motion Tracking Tasks (Mimic)

This appendix covers conversions specific to motion-tracking / mimic tasks. Skip this if converting a velocity-command or manipulation task.

### A.1 CommandManager vs MotionManager

This is the biggest architectural difference for mimic tasks.

**mjlab**: Motion tracking is implemented as a `MotionCommand` (extends `CommandTerm`) inside the `CommandManager`.
- Motion data loaded from `.npz` files containing pre-computed kinematics: `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`.
- `MotionCommand.command` returns `cat(joint_pos, joint_vel)` (consumed by `generated_commands`).
- Relative body poses computed in `update_relative_body_poses()`, stored as `body_pos_relative_w`, `body_quat_relative_w`.
- Reset writes state to sim directly via `_resample_command()` -> `_write_reference_state_to_sim()`.
- Adaptive sampling logic lives inside the command.

**Eden**: Motion tracking uses a dedicated `MotionManager` (separate from `CommandManager`).
- Motion data loaded from dataset classes (e.g., `BvhCsvG1Dataset`).
- Dataset provides `BatchMotionSequence`; forward velocity kinematics computed by `_forward_velocity_kinematics()`.
- `MotionReferenceData` accessed via `env.motion_manager.data[entity_name]` with shape `(num_envs, num_frames, ...)`.
- Reset handled by `ReferenceStateInitialization` event term (not by command).
- `MimicRLEnvBase.step()` calls `self.motion_manager.compute(dt)` **before** `super().step()`.
- Adaptive sampling configured via `MotionTerm.configure(adaptive_sampling=True)`.

### A.2 Conversion Steps

1. Replace `MotionCommandCfg` with `motion_options: MotionManagerOptions` + `MotionTerm.configure()`.
2. Create or use a dataset class instead of `MotionLoader`.
3. Replace `generated_commands(command_name="motion")` with a custom obs term like `motion_dofs_reference`.
4. Replace `MotionCommand._resample_command()` with `ReferenceStateInitialization` event.
5. All terms that accessed `command.property` now access `env.motion_manager.data[entity_name]` and `env.entities[entity_name]`.
6. Use `EdenMimicConfig` (not `EdenRLConfig`) as the config base class and `MimicRLEnvBase` (not `RLEnvBase`) as the environment base class.

### A.3 Motion Data Access

mjlab reads from `MotionCommand` properties:
```python
command = cast(MotionCommand, env.command_manager.get_term("motion"))
command.anchor_pos_w          # (num_envs, 3)
command.robot_anchor_pos_w    # (num_envs, 3)
command.body_lin_vel_w        # (num_envs, num_bodies, 3)
```

Eden reads from `MotionReferenceData` (note the extra frame dimension):
```python
ref = env.motion_manager.data["robot"]
ref.links_pos_w[:, 0, anchor_idx]      # (num_envs, 3)  -- [:, 0, ...] for current frame
ref.links_lin_vel_w[:, 0, body_idx]    # (num_envs, num_bodies, 3)
```

Robot state in Eden:
```python
entity = env.entities["robot"]
entity.get_links_pos(ls_idx_local=idx)  # (num_envs, num_links, 3)
entity.get_links_vel(ls_idx_local=idx)  # (num_envs, num_links, 3)
```

### A.4 Relative Body Pose Computation

Both frameworks compute relative body poses using yaw-only alignment:

1. `delta_ori = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(motion_anchor_quat)))`
2. `delta_pos[:, 2] = motion_anchor_pos[:, 2]` (keep Z from motion, don't track height)
3. `body_pos_rel = delta_pos + quat_apply(delta_ori, motion_body_pos - motion_anchor_pos)`

mjlab precomputes this in `MotionCommand.update_relative_body_poses()`.
Eden computes on-the-fly in each reward/termination term.

### A.5 Motion Data Quaternion Convention

**mjlab .npz files**: `body_quat_w` is already in wxyz format.
**Eden CSV files**: May use xyzw -- the dataset must convert during loading:
```python
base_quat_wxyz = torch.cat([base_quat_xyzw[:, 3:4], base_quat_xyzw[:, 0:3]], dim=-1)
```

### A.6 Mimic-Specific Pitfalls

- [ ] **Motion data frame dimension**: Eden motion data has `(envs, frames, ...)` -- always index `[:, 0, ...]` for current frame
- [ ] **Motion data access**: Replace `command.property` -> `env.motion_manager.data[entity_name].property[:, 0, ...]`
- [ ] **Env base class**: Use `MimicRLEnvBase` (not `RLEnvBase`) for motion tracking tasks
- [ ] **Dataset + MotionManager**: Replace mjlab's `MotionLoader` + `MotionCommand` with Eden's dataset + `MotionManager`
- [ ] **Quaternion format in data files**: Check source quaternion convention (xyzw vs wxyz) and convert if needed
- [ ] **Link velocity in world frame**: Both frameworks track link velocities in world frame for reward computation -- do NOT convert to body frame
