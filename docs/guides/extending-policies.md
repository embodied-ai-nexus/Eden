# Adding a Policy to the Zoo

A policy extension packages a frozen controller (an ONNX graph, a TorchScript module, a torch network, or a client for
a remote server) so that `eden inference --policy <name>` and `eden deploy --policy <name>` can run it. This page
covers the package layout, the `PolicyBase` contract, how weights are fetched, and the mistakes that produce a policy
that builds and then acts on the wrong input. Faithfully reproducing a policy from another stack is its own checklist:
[Porting a Pretrained Policy](policy-porting.md). Server-backed policies add a transport layer on top:
[Remote Policy Clients](remote-policies.md).

## What a policy is

A policy is any object that takes the env's observations and returns actions. In Eden every policy:

- subclasses [`PolicyBase`](../api/extensions/policies.md) (`eden.extensions.policies.base`) and is registered with `@POLICY_REGISTRY.register()`;
- declares the world it was trained in through classmethods: the robot, the other scene entities, the env options, and
  the observation, action and command terms;
- implements `step(obs) -> {action_term_name: tensor}`, and `reset(envs_idx)` if it holds per-env state.

The registry key is `to_snake_case(ClassName)` (`Twist2Policy` → `twist2_policy`). `--policy` also accepts the key
without `_policy` and the package name, as long as the package registers one policy. Policy packages are imported on
demand: the CLI imports `eden.extensions.policies.<pkg>.policy` when it resolves a name, so a package with optional
dependencies does not slow down or break the others.

## Package layout

One package per policy under `eden/extensions/policies/<name>/`:

| File | Contents |
|------|----------|
| `policy.py` | The registered `PolicyBase` subclass. Required: the CLI discovers a package by this file. |
| `constants.py` | The frozen I/O contract: joint-name order, observation scales, default pose, gains. |
| `onnx_wrapper.py` / `network.py` / `modules.py` | The actor, when it is not a plain ONNX graph. |
| `custom_terms/` | Observation, action or command terms only this policy needs. |
| `setups.py` | Per-embodiment setups, for a server-backed client that serves several robots. |

Keep the package to the policy, its network and its contract. When Eden also has a training task for the policy, reuse
that task's terms and scene from `eden/tasks/<name>/` rather than copying them: `husky` builds its observation and
action from `eden/tasks/husky/config.py`, and `omnireset` its scene from `eden/tasks/omnireset/`. A released checkpoint
and a policy trained in Eden then share one layout.

Add a runnable example under `examples/policies/<name>/main.py` with the metadata header described in
`examples/AGENTS.md` (`tier: policy`, its `capabilities`, `hardware`, `optional_deps` and `run` command), and tests
under `tests/extensions/policies/<name>/`.

## A worked example: TWIST2

`eden/extensions/policies/twist2/policy.py` is a compact, complete policy that reads the env's observations. Its
skeleton, with the class constants and the `__init__` checks trimmed:

```python
from eden import actions, observations
from eden.constants import ReferenceSource
from eden.extensions.policies.base import POLICY_REGISTRY, PolicyBase, PolicyOptions
from eden.managers.terms.commands.mimic import MotionCommand
from eden.options.managers.observations import ObservationGroupOptions
from eden.options.robots.unitree_g1 import G1Dofs29
from eden.terms import ObsTerm

# Importing the custom terms registers them before any env is built.
from eden.extensions.policies.twist2.custom_terms.observations import (
    twist2_imu_roll_pitch, twist2_motion_dofs_pos, twist2_motion_root,
)
from eden.extensions.policies.twist2.onnx_wrapper import Twist2Actor
# POLICY_JOINT_NAMES, KNEES_BENT_DOFS_POS, JOINT_SCALES, ... come from eden.extensions.policies.twist2.constants


@POLICY_REGISTRY.register()
class Twist2Policy(PolicyBase):
    """TWIST2 whole-body motion-tracking controller (G1 29-DOF, ONNX backend)."""

    reads_env_observations = True          # step() reads the observation dict the env publishes
    entity_name: str = "robot"
    checkpoint_path: str = "twist2/pretrained.onnx"
    md5_hash: str | None = "a045b72bd49ce0993aa145f8958a5564"

    def __init__(self, env, options: PolicyOptions) -> None:
        super().__init__(env, options)      # resolves and verifies checkpoint_path
        robot = self.require_entity(self.entity_name)
        # ... check robot.dofs_name, the built default pose and the observation width here ...
        self._actor = Twist2Actor(self.checkpoint_path, device=str(env.device))

    @classmethod
    def robot_options(cls) -> G1Dofs29:
        return G1Dofs29(default_dofs_pos=dict(KNEES_BENT_DOFS_POS), default_root_pos=(0.0, 0.0, 0.76))

    @classmethod
    def env_options(cls) -> dict:
        return dict(sim_dt=0.01, sim_substeps=2, decimation=2, ...)   # 50 Hz control

    @classmethod
    def action_options(cls, entity_name: str = "robot") -> dict:
        return dict(
            dofs_pos_controller=actions.ImplicitPDController.configure(
                entity_name=entity_name,
                dofs_name=list(POLICY_JOINT_NAMES),
                scale=dict(zip(POLICY_JOINT_NAMES, JOINT_SCALES)),
                reference_source=ReferenceSource.DEFAULT,
            ),
        )

    @classmethod
    def command_options(cls, motion_file: str = "motion.npz", entity_name: str = "robot", ...) -> dict:
        return dict(motion=MotionCommand.configure(motion_file=motion_file, entity_name=entity_name, ...))

    @classmethod
    def observation_options(cls, entity_name: str = "robot") -> dict:
        entity = {"entity_name": entity_name}
        return {
            "policy": ObservationGroupOptions(
                enable_corruption=False,
                history_length=ACTOR_HISTORY_LENGTH,
                flatten_history_dim=True,
                motion_root=ObsTerm.configure(func=twist2_motion_root),
                motion_dofs_pos=ObsTerm.configure(func=twist2_motion_dofs_pos),
                base_ang_vel=ObsTerm.configure(
                    func=observations.base_ang_vel, params={**entity, "frame": "body"}, scale=BASE_ANG_VEL_SCALE
                ),
                imu_roll_pitch=ObsTerm.configure(func=twist2_imu_roll_pitch, params=entity),
                dofs_pos=observations.DofsPos.configure(entity_name=entity_name, offset_from_default=True),
                dofs_vel=observations.DofsVel.configure(entity_name=entity_name, scale=list(JOINT_VEL_SCALES)),
                last_action=ObsTerm.configure(func=observations.last_action),
            )
        }

    def step(self, obs):
        history = obs["policy"]             # (N, 11 * 127), oldest frame first
        actor_obs = torch.cat([history[:, -ACTOR_CURRENT_DIM:], history], dim=-1)
        return {"dofs_pos_controller": self._actor(actor_obs)}
```

## The `PolicyBase` contract

### Class attributes

| Attribute | Default | Meaning |
|-----------|---------|---------|
| `checkpoint_path` | `None` | The main checkpoint, relative to the policy's Hugging Face dataset. `__init__` replaces it with the local path, downloading on first use. |
| `md5_hash` | `None` | The MD5 the downloaded file must match; `None` skips the check. |
| `registry` | Eden's policy repository owner | The Hugging Face owner of the dataset (a class constant). |
| `dataset` | `"eden_policies"` | The Hugging Face dataset holding the weights (a class constant). |
| `local_dir` | `None` | The cache directory; `None` means `assets_subdir` under the asset root (a class constant). |
| `assets_subdir` | `"policies"` | The subdirectory of the asset root an unset `local_dir` means. |
| `reads_env_observations` | `False` | Whether `step()` consumes the env's observation dict (see below). |

Any other annotated class attribute (`entity_name`, `hand_name`, `pair`, …) is a configurable parameter:
`MyPolicy.configure(entity_name="g1")` sets it, and the spec classmethods can read it (next section).

### Classmethods: the world the policy was trained in

| Classmethod | Required | Returns |
|-------------|----------|---------|
| `robot_options()` | no | The robot options with the training pose, gains and armature, or `None` to use `--robot`. Takes no arguments. |
| `scene_entity_options()` | no | `{name: entity options}` for every non-robot entity `step()` or the observations read. A `plane` entity replaces the flat ground. |
| `env_options()` | no | `EnvOptions` fields: at least `sim_dt` and `decimation`. Takes no arguments. |
| `action_options()` | **yes** | `{term_name: action term options}`. |
| `observation_options()` | **yes** | `{group_name: ObservationGroupOptions}`. May be `{}`. |
| `command_options()` | no | `{term_name: command term options}`. |

In `action_options`, `observation_options` and `scene_entity_options`, a parameter named after one of the policy's
configurable attributes is **filled from that attribute**: declare `hand_name: str = "hand"` on the class and
`action_options(cls, entity_name="robot", hand_name="hand")` receives `cls.hand_name`. So a name is declared once,
instead of restated as a literal that can drift from it. `entity_name` is the caller's (`--entity_name` wins), and any
other parameter keeps its default, which means a knob that must keep its literal default needs a name no attribute
uses.

`command_options` is filled by the CLI instead: `entity_name`, `motion_file` (from `--motion_file`, and **required**
whenever the signature takes it), and `clip_name` (from `--clip_name`, when the signature takes it).

### `env_options`: match the control rate

`sim_dt × decimation` is the time per policy step, and it must match the training setup. A policy that does not
override `env_options()` runs at the env's defaults and will very likely perform poorly. The shipped values:

| Policy | `sim_dt` | `decimation` | Control rate |
|--------|----------|--------------|--------------|
| `amo` | 0.002 s | 10 | 50 Hz |
| `gmt` | 0.001 s | 20 | 50 Hz |
| `holosoma`, `husky` | 0.005 s | 4 | 50 Hz |
| `sonic`, `twist2`, `protomotionv3`, `humanoid_gpt` | 0.01 s | 2 | 50 Hz |
| `omnixtreme` | 0.004 s | 5 | 50 Hz |
| `metamotivo` | 1/450 s | 15 | 30 Hz |

Solver settings (`iterations`, `substeps`, contact options) belong here too when the training simulator's differ.

### `step()` and `reset()`

- `obs` keys are the group names from `observation_options()`. A group with `concatenate_terms=False` arrives as a
  dict of its terms (`obs["wrist_image"]["image"]`), which is how camera images are delivered.
- The returned dict's keys must be the term names from `action_options()`. An unknown key raises a `KeyError`. A term
  the dict leaves out is **not** an error: it keeps its previous action, silently. Return every term on every step.
- Return tensors on `self.env.device`, and run inference under `torch.no_grad()` (or `torch.inference_mode()`).
- Override `reset(envs_idx)` when the policy holds per-env state across steps (a recurrent state, a last-action
  history it builds itself, an action-chunk queue). `eden inference` calls it with the envs that just ended an episode;
  `envs_idx=None` means all envs.

## Observations: declare them, or read raw state

`reads_env_observations` says which side of the env boundary `step()` sits on, and `eden deploy` arranges a run around
the answer, so treat it as a contract:

- **`True`: the policy owns the env.** The env is built from its observation groups and action terms, and its action
  goes through the env's action manager, which keeps env-computed observations such as `last_action` in step with what
  the robot was commanded. This is the preferred form: the layout is explicit, and the observation manager handles
  history, scale, NaN policy and resets. Only one such policy can be loaded at a time, and not alongside a task, since
  one env publishes one observation layout.
- **`False`: the policy reads raw state.** `step()` reads the robot's state off the entity and builds the input itself
  (`holosoma`, `homie`, `sonic`). Such a policy can ride along as an extra slot on another policy's env, converting its
  own actions; its `observation_options()` may return `{}`.

Rules that keep an observation correct:

- **The term order is the layout.** Terms concatenate in declaration order (or `terms_order`, when set), and the
  network only sees numbers: a wrong order is dimensionally valid and silently wrong.
- **History is oldest first.** With `history_length=H` and `flatten_history_dim=True`, a group of frame width `D`
  arrives as `(N, H * D)` with the newest frame last (`obs[g][:, -D:]`). With `flatten_history_dim=False` it is
  `(N, H, D)`. At an episode start the window is filled with the first frame when `backfill=True` (the default), or
  with zeros when `backfill=False`.
- **Turn training noise off.** Pass `enable_corruption=False` on a group whose term definitions carry noise (for
  example terms shared with a training task), as `husky` and `twist2` do.
- **A declared group needs at least one term.** No groups at all builds fine; a `concatenate_terms=True` group with
  no terms fails at construction with ``Unable to concatenate observation terms in group `<name>`.``

## Entities and commands

- **Resolve every entity in `__init__` with `self.require_entity(name)`.** A scene without it then fails at
  construction with a message that says where the entity should come from (the policy's `scene_entity_options()` in a
  task-less run, the task's scene with `--task`). In `step()`, read `self.env.entities[name]` instead of caching the
  handle: deployment swaps the entity for a live-robot stand-in.
- **Check the built robot against the contract in `__init__`.** The shipped ports refuse a robot whose `dofs_name` is
  not the policy's joint order, or whose built default pose differs from the training keyframe in any env
  (`robot.get_default_dofs_pos(POLICY_JOINT_NAMES)` is `(num_envs, n)` in that order), and an env whose observation
  group has the wrong width. Copy those checks; they turn a silent wrong-input rollout into an error.
- **A command-driven policy defines `set_command(command)` and a `_commands` buffer.** `--base_command VX VY WZ` is
  padded with zeros or truncated to the width of `policy._commands` (HOMIE's is 4-D, HUSKY's 2-D) before
  `set_command` is called. A policy whose command lives on a command term exposes `_commands` as a property that reads
  the term, as `HuskyPolicy` does.
- **A command term a frozen policy observes must tick in `pre_physics_advance`**, never in `post_physics_refresh`,
  which a deployment does not run.

## Custom terms

A term only the policy needs goes in `custom_terms/`, registered like any other term (see [Writing Terms](terms.md)):

```python
# eden/extensions/policies/mypolicy/custom_terms/observations.py
import torch

from eden.envs.base import EnvBase
from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY


@OBSERVATION_TERM_REGISTRY.register()
def mypolicy_root_height(env: EnvBase, *, entity_name: str = "robot") -> torch.Tensor:
    return env.entities[entity_name].get_pos()[:, 2:3]
```

The built-in term registries find their terms by importing the modules they list, and a policy's `custom_terms/` is
not among them, so **import every custom term module in `policy.py`** (at module level). The CLI imports `policy.py`
before it builds the env, and the import registers the terms. Prefix a policy-only term's name with the policy's
(`twist2_imu_roll_pitch`): term keys share one namespace across Eden.

A custom action term is needed when the robot's actuators differ from Eden's controllers (MetaMotivo's SMPL humanoid
uses MuJoCo-style affine actuators, `MujocoSMPLActuator`); a custom command term when the policy consumes a signal no
built-in command produces (`AMOCommand`, GMT's `GMTMotionLibCommand`).

## Weights

`PolicyBase.__init__` fetches `checkpoint_path` with the class's `fetch_asset()`, which downloads from the
`registry`/`dataset` Hugging Face dataset into `resolve_local_dir()` (`local_dir`, else `assets_subdir` under the asset
root), and checks `md5_hash`. Fetch every further file the same way, with `self.fetch_asset(path)`:

- Never call `get_asset_path(..., local_dir=self.local_dir)` yourself. An unset `local_dir` is `None` and means
  "`assets_subdir` under the asset root", resolved at load time; passing the raw `None` fetches into the asset root
  instead.
- `registry`, `dataset`, `local_dir` and `assets_subdir` are class constants, not options: passing one to
  `configure()` raises. That keeps a lookup made before any instance exists (a clip catalog read by
  `command_options()`, a file an observation term loads) on the same path as `__init__`: call
  `MyPolicy.fetch_asset(path)` on the class.

| Scenario | What to do |
|----------|------------|
| One `.pt` / `.onnx` file | Set `checkpoint_path` and `md5_hash`. |
| Several files | Keep the main one in `checkpoint_path`; name the others `<prefix>checkpoint_path` / `<prefix>md5_hash` (`decoder_checkpoint_path`, `residual_checkpoint_path`) and fetch and verify them in `__init__`. The naming lets `use_local_checkpoint(path, checkpoint_attr=...)` redirect any of them. |
| A model loaded from Hugging Face by its own library (`metamotivo`) | Set `checkpoint_path = None`, `registry = None`, `dataset = None`, and load in `__init__`. |
| Weights from another dataset (`beyondmimic_moves`) | Override `registry` and `dataset` on the class. |
| No released weights (`limx_oli_loco`) | `checkpoint_path = None`, `registry = None`, `dataset = None`, and raise in `__init__` when no local file was given. |

To run local weights, call `MyPolicy.use_local_checkpoint(path)` before constructing the policy: it sets the path
and clears the MD5 pinned to the released file together. To fetch from a mirror, export `EDEN_ASSETS_DIR=/data/mirror`
(policies then cache under `/data/mirror/policies/`), or set `MyPolicy.local_dir = "/data/mirror"` on the class before
anything reads it.

## Run it

```bash
eden inference --policy mypolicy --vis                 # task-less: the policy's own world
eden inference --policy mypolicy --async --vis         # deployment-style timing
eden deploy --policy mypolicy --mujoco --vis           # MuJoCo sim-to-sim (see Deployment)
```

From Python, build the env from the policy's spec the way a task-less run does:

```python
import eden as en
import genesis as gs

from eden.envs.base import RLEnvBase
from eden.extensions.policies.twist2.policy import Twist2Policy
from eden.options.envs import EnvOptions
from eden.options.managers.actions import ActionManagerOptions
from eden.options.managers.commands import CommandManagerOptions
from eden.options.managers.observations import ObservationManagerOptions
from eden.utils.configs import EdenRLConfig

en.init(backend=gs.gpu)

cfg = EdenRLConfig(
    env_options=EnvOptions(num_envs=1, **Twist2Policy.env_options()),
    scene_options=en.scenes.Empty(robot=Twist2Policy.robot_options(), **Twist2Policy.scene_entity_options()),
    action_options=ActionManagerOptions(**Twist2Policy.action_options()),
    observation_options=ObservationManagerOptions(**Twist2Policy.observation_options()),
    command_options=CommandManagerOptions(**Twist2Policy.command_options(motion_file="motion.npz")),
)
env = RLEnvBase.from_config(cfg, eval_mode=True)
env.build()
policy = Twist2Policy(env=env, options=Twist2Policy.configure())

obs, _ = env.reset()
policy.reset()
for _ in range(1000):
    obs, *_ = env.step(policy.step(obs))
```

The CLI's version of this is `_build_extension_env` in `eden/utils/cli/tasks.py`, which the policy tests use too.
Calling the spec classmethods directly skips the CLI's attribute filling, so pass any parameter that names a
configured attribute yourself.

## Checklist

- [ ] `PolicyBase` subclass in `eden/extensions/policies/<name>/policy.py`, decorated with
      `@POLICY_REGISTRY.register()`.
- [ ] `checkpoint_path` and `md5_hash` set, or the weight source handled as in [Weights](#weights).
- [ ] `robot_options()` with the training pose, gains and armature; `scene_entity_options()` for every other entity
      the policy reads.
- [ ] `env_options()` with the training `sim_dt` and `decimation` (and solver settings that differ).
- [ ] `action_options()` keys equal the keys `step()` returns, on every step.
- [ ] Observation groups in the training order, history oldest first, training noise off.
- [ ] `reads_env_observations` set to match what `step()` reads.
- [ ] `__init__` resolves every entity with `require_entity` and checks the built robot and the observation width.
- [ ] Every custom term module imported in `policy.py`.
- [ ] An example in `examples/policies/<name>/main.py` with its metadata header, and tests in
      `tests/extensions/policies/<name>/` that pin the observation and action contract.
- [ ] Verified layer by layer as in [Porting a Pretrained Policy](policy-porting.md#4-verify-in-layers).

## See also

- [The Policy Zoo](policy-zoo.md): every shipped policy
- [Porting a Pretrained Policy](policy-porting.md): the fidelity checklist
- [Remote Policy Clients](remote-policies.md): server-backed policies
- [Writing Terms](terms.md): term registration and lifecycle
- [Deployment](deployment.md): policy slots on hardware
