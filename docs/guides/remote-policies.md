# Remote Policy Clients

Some policies do not run inside the Eden process: a server (on another GPU, in another framework or another
repository) computes the actions, and Eden sends it observations. This page covers the client policies Eden ships for
such servers, how to run one in an evaluation loop, and the machinery they share (transports, action chunking,
per-embodiment setups, gripper conventions), which is also what you build on to wire a new server in.

Eden ships six clients under `eden/extensions/policies/`, for vision-language-action models (OpenPI `pi`, GR00T
`gr00t`, GalaxeaVLA `galaxeavla`, Octo `octo`, MolmoAct2 `molmoact2`) and a world action model (Cosmos3 `cosmos3`). The
shared layer is family-neutral: nothing in it assumes images, language or grippers.

Install the client dependencies (ZMQ, msgpack, WebSockets, `json-numpy`, Pillow) with
`pip install 'eden-world[policy-client]'`, or `uv sync --inexact --extra policy-client` in a checkout. Each transport
imports its own dependencies only when it is used.

## The shipped clients

| Policy | Server | `transport` | Default `host:port` | Setups (`policy_setup`) |
|--------|--------|-------------|---------------------|--------------------------|
| `pi` (`PiPolicy`) | OpenPI: pi0, pi0-FAST, pi0.5 | `openpi_websocket` | `localhost:8000` | `widowx_bridge` (default), `google_robot`, `libero`, `droid`, `droid_jointpos`, `aloha` |
| `gr00t` (`Gr00tPolicy`) | NVIDIA Isaac-GR00T | `zmq_msgpack` | `localhost:5555` | `simpler_env_widowx` (default), `libero_sim`, `oxe_droid_relative_eef_relative_joint` |
| `galaxeavla` (`GalaxeaVlaPolicy`) | GalaxeaVLA (G0.5) | `galaxea_websocket` | `localhost:8765` | `galaxea_r1lite` (default), `galaxea_r1pro`, `droid_franka` |
| `cosmos3` (`Cosmos3Policy`) | Cosmos3-Nano-Policy-DROID | `openpi_websocket` | `localhost:8000` | — (DROID) |
| `molmoact2` (`MolmoAct2Policy`) | MolmoAct2 DROID server, absolute joint control | `http_json_numpy` | `localhost:8000` | — (DROID) |
| `octo` (`OctoPolicy`) | Octo, behind an HTTP server | `http_json` | `localhost:9010` | — (WidowX) |

A setup is the per-robot half of a client whose server serves several embodiments; it decides the observation groups
and action terms the env must provide (see [Per-embodiment setups](#per-embodiment-setups)).

## Wire a client into an evaluation loop

A client needs an env whose observation groups and action terms are the ones its setup reads and writes, so build the
task config, replace its observation and action options with the client's, and then construct the client:

```python
import eden as en
import genesis as gs

from eden.envs.base import env_class_for_config
from eden.extensions.policies.gr00t.policy import Gr00tPolicy
from eden.options.managers.actions import ActionManagerOptions
from eden.options.managers.observations import ObservationManagerOptions
from eden.tasks import TASK_REGISTRY

en.init(backend=gs.gpu)

setup = "simpler_env_widowx"
cfg = TASK_REGISTRY.build("simpler_put_carrot_on_plate")
cfg.env_options.num_envs = 1
cfg.observation_options = ObservationManagerOptions(**Gr00tPolicy.observation_options(policy_setup=setup))
cfg.action_options = ActionManagerOptions(**Gr00tPolicy.action_options(policy_setup=setup))
env = env_class_for_config(cfg).from_config(cfg, show_viewer=False, eval_mode=True)
env.build()

policy = Gr00tPolicy(env, Gr00tPolicy.configure(host="localhost", port=5555, policy_setup=setup))
policy.set_task(cfg.language_instructions[0])

for episode in range(5):
    obs, _ = env.reset()
    policy.reset_server()
    for _ in range(100):
        obs, *_ = env.step(policy.step(obs))
```

`observation_options(policy_setup=...)` and `action_options(policy_setup=...)` are classmethods, so the env is wired
without a live server. Constructing the client connects to the server and checks it (below); a wrong setup, a missing
gripper term or a control rate the checkpoint was not trained at fails there, before the first step.

`eden inference --policy gr00t` also runs a client, but builds it with `configure()`'s defaults (the address and setup
in the table, no instruction), so use the loop above to choose a server, a setup or an instruction.

## How a client works: `ClientMixin`

A client is a `PolicyBase` whose `step()` comes from [`ClientMixin`](../api/extensions/policies.md) (`eden.extensions.policies.client`):

```python
from eden.extensions.policies.base import POLICY_REGISTRY, PolicyBase
from eden.extensions.policies.client import ClientMixin
from eden.options.extensions.policies import RemotePolicyOptions
from eden.utils.common import ConfigurableMixin


@POLICY_REGISTRY.register()
class MyClientPolicy(ClientMixin, PolicyBase, ConfigurableMixin[RemotePolicyOptions]):
    transport = "http_json"          # a TRANSPORT_REGISTRY key
    response_action_key = "action"   # where the reply carries the chunk

    def pack_observations(self, obs): ...   # Eden observations -> server payload
    def unpack_actions(self, raw): ...      # one (D,) row -> {action_term: tensor}

    @classmethod
    def observation_options(cls, **kwargs): ...
    @classmethod
    def action_options(cls, **kwargs): ...
```

`ClientMixin` sits **left** of `PolicyBase` so its concrete `step()` satisfies the abstract one, and the third base
re-parameterizes `ConfigurableMixin` with `RemotePolicyOptions` (the policy fields plus the client fields), so
`configure()` accepts both.

Every step runs the same pipeline:

```text
obs -> observe(obs)                                   per-step hooks (frame history, measured pose)
    -> [if needs_inference]
          payload = pack_request(obs)                 default: pack_observations(obs)
          reply   = transport.infer(payload, request_options())
          chunk   = decode_response(reply)            -> (num_envs, T, D)
          scheduler.accept(chunk)
    -> row = scheduler.pop()                          -> (num_envs, D)
    -> unpack_actions(row)                            -> {action_term: tensor}
```

The mixin owns three things:

| Piece | Where | What it does |
|-------|-------|--------------|
| Transport | `eden/utils/policy_client/transport.py` | Moves a payload to the server and decodes the reply; selected by `transport="<key>"`. |
| Scheduler | `ActionChunkScheduler` in `client.py` | Queues server chunks, applies `pred_action_horizon` / `exec_horizon`, optional temporal ensembling. |
| Lifecycle | `ClientMixin` in `client.py` | `set_task`, `reset_server`, `reset(envs_idx)`: each clears the queue and local state, then tells the server. |

`reset(envs_idx)` resets the server only when it covers every env the client serves (with one env, every reset). A
partial reset of a batched client raises `NotImplementedError`, because the server's episode state is shared by all
of them. Everything the server returns besides the action chunk is kept on `policy.last_response_info` (GR00T's
`info`, a world model's predictions), so recorders and evaluators can log it, and `policy.action_metadata` reports
where the last executed row sat in its chunk.

### Running inference asynchronously

`step()` is also available as its stages: `observe`, `needs_inference`, `pack_request`, `infer_payload`,
`accept_chunk` and `pop_action`. An asynchronous evaluator can run the blocking round-trip on a worker thread while the
simulator keeps stepping, so the robot moves on during the wait as much as the latency would let it. Two rules:

- `observe` must still run on **every** step: it records the frame histories and measured poses the next request
  reads.
- Keep one request in flight: accept or discard its chunk before the next `pack_request`, and do not call `reset`,
  `reset_server` or `set_task` while a request is in flight. Nothing in the client locks the transport.

## Transports and serializers

Both are chosen by registry **key**, a plain string that survives config serialization, never by passing instances
through `configure()`.

| `transport` | Wire | Default serializer | Used by |
|-------------|------|--------------------|---------|
| `http_json` | `requests` POST, JSON body and reply | requests' JSON | `octo` |
| `http_json_numpy` | POST with a `json_numpy` text body | `json_numpy` | `molmoact2` |
| `http_multipart` | multipart form; the policy's `pack_request` returns `(data, files)` | — | — |
| `openpi_websocket` | msgpack WebSocket with OpenPI's numpy hook | `msgpack_openpi` | `pi`, `cosmos3` |
| `galaxea_websocket` | as above, with str-keyed records and a `{"__reset__": True}` message | `msgpack_galaxea` | `galaxeavla` |
| `zmq_msgpack` | ZMQ REQ/REP with an `{"endpoint", "data"}` envelope | `msgpack_numpy` (or `msgpack_npy_blob`) | `gr00t` |

A transport is built from a `TransportConfig` that `ClientMixin.transport_config()` fills from the policy's
attributes: the address, timeouts, endpoints, and the transport-specific **knobs** each transport declares, read from
policy attributes of the same name:

| Transport | Knobs (defaults) |
|-----------|------------------|
| `http_json`, `http_json_numpy`, `http_multipart` | `set_task_as_query` (`False`), `reset_with_payload` (`True`) |
| `openpi_websocket` | `api_key` (`None`), `ws_ping_timeout` (`20.0`), `wait_for_server_s` (`600.0`) |
| `galaxea_websocket` | `api_key` (`None`), `ws_ping_timeout` (`None`), `ws_timeout_s` (`60.0`) |
| `zmq_msgpack` | `timeout_ms` (`60000`), `api_token` (`None`) |

### Set-task and reset endpoints

`set_task_endpoint` and `reset_endpoint` name the server's set-task and reset API and default to `None`. With `None`
the server has no such API (it takes the task with every request), so `set_task`, `reset_server` and an env reset
only clear local state. A policy declares one only when its official server exposes it:

| Policy | `set_task_endpoint` | `reset_endpoint` |
|--------|---------------------|------------------|
| `octo` | `"/set_task"` (HTTP) | `"/reset"` (HTTP) |
| `gr00t` | — | `"reset"` (a ZMQ endpoint name) |
| `galaxeavla` | — | `"__reset__"` (the reset message key) |
| `pi`, `cosmos3`, `molmoact2` | — | — |

A transport whose protocol has no such message rejects a non-`None` value at construction; `openpi_websocket`, for
example, has no reset message.

### The server address

The address is `host` plus `port`, joined by `eden.utils.policy_client.server_url`:

- A bare hostname such as `localhost` gets the transport's scheme: `http`, `ws` or `tcp`.
- `host` may be a full URL whose scheme fits the transport, such as `wss://example.com/ws` or `https://node/api`.
  Pass `port=None` to use the port in the URL, or the scheme's default when the URL has none.
- A port in the URL that disagrees with `port` raises at construction.
- ZMQ (`tcp://`) needs a port and no path.

Errors map onto one hierarchy in `eden.utils.policy_client.errors`: an unreachable server or a timeout is a
`TransportError` (also a `ConnectionError`; a timeout is the subclass `TransportTimeoutError`), a server-side failure
is a `ServerError`, and a payload that cannot be encoded or decoded is a `SerializationError`. All derive from
`ClientError` (a `RuntimeError`).

## Action chunking

Servers usually return a chunk of `T` actions per request. The scheduler's rules are client options:

| Option | Meaning |
|--------|---------|
| `chunk_mode` | `client`: buffer the chunk and re-query when it drains. `server`: the server caches the chunk and serves one row per call (GalaxeaVLA). `single`: one action per call (Octo). |
| `pred_action_horizon` | The expected chunk length; `None` accepts any. |
| `exec_horizon` | Rows executed open-loop before re-querying; `None` executes the whole chunk. |
| `short_chunk` | `error`: a chunk shorter than `pred_action_horizon` is a truncated prediction and raises. `accept`: use it. |
| `chunk_temporal_ensemble_steps` / `chunk_temporal_ensemble_dims` | Blend the previous chunk's unused tail into the new head (OpenPI's optional smoothing); `0` disables. |
| `response_action_key` | The key that holds the chunk in a dict reply (`"action"` or `"actions"`). |

## Per-embodiment setups

A server that serves several robots (OpenPI: WidowX, Google Robot, LIBERO, DROID, ALOHA; GR00T: SimplerEnv, LIBERO,
DROID; GalaxeaVLA: R1 Lite, R1 Pro, a DROID Franka) uses `SetupClientMixin` and a `SETUP_REGISTRY` of
`PolicySetupBase` subclasses, selected by `policy_setup`:

```python
from typing import ClassVar

from eden.extensions.policies.client import PolicySetupBase, SetupClientMixin


class MySetup(PolicySetupBase):
    pred_action_horizon: ClassVar[int | None] = 8   # checkpoint defaults, used when the policy leaves them None
    exec_horizon: ClassVar[int | None] = 4
    default_image_size: ClassVar[tuple[int, int] | None] = (224, 224)

    @classmethod
    def observation_options(cls, **kwargs): ...
    @classmethod
    def action_options(cls, **kwargs): ...
    def pack_observations(self, obs): ...
    def decode_response(self, response): ...   # optional: reply -> (T, D)
    def observe(self, obs): ...                # optional: per-step state (a frame history)
    def unpack_actions(self, raw_action): ...  # optional: default is self.adapter(row, env=...)


@POLICY_REGISTRY.register()
class MyPolicy(SetupClientMixin, PolicyBase, ConfigurableMixin[RemotePolicyOptions]):
    SETUP_REGISTRY = {"my_robot": MySetup}
    policy_setup = "my_robot"
    transport = "zmq_msgpack"
```

`MyPolicy.observation_options(policy_setup="my_robot")` and `action_options` forward to the setup.

### Server configuration checks

The upstream servers own checkpoint and embodiment selection (`serve_policy.py --env` for OpenPI, `--embodiment-tag`
for Isaac-GR00T, a per-request `embodiment_type` for GalaxeaVLA), and the client's `policy_setup` has to agree with
them. Right after the transport connects, `SetupClientMixin` hands `transport.server_config()` to the setup's
`validate_server_config`:

- GR00T setups declare a `ModalitySchema` (video, state and action keys, action horizon, video window) and compare it
  with the server's `get_modality_config` reply. A server started with another embodiment tag fails at construction,
  with both sides named.
- OpenPI servers advertise free-form metadata (kept on `transport.metadata`). Every setup also declares `action_dim`,
  so a chunk of the wrong width is rejected before it is queued.

## Observations

Manipulation servers consume some composition of three core observation terms (all in `eden.observations`), so setups declare their groups from
these instead of writing per-policy functions. The terms resolve link and DOF names once in `build()` (a wrong name
fails at env construction) and read entities through `resolve_entity`, so a deployment's entity swap works.

| Term | Output | Options |
|------|--------|---------|
| `EndEffectorPose` | `[pos(3), rotation]` of a link in the base or world frame | `rotation`: `quat` / `euler_xyz` / `axis_angle` / `rot6d`; a virtual tool offset (`offset_pos`, `offset_quat`); `frame_correction_quat` (for example DROID's rot6d correction) |
| `DofsPos` | Joint positions of the named DOFs, in the declared order | `entity_name`, `dofs_name`, `offset_from_default` |
| `GripperState` | Raw, openness or closedness of the gripper DOFs | `normalize`, `open_pos`, `close_pos`, `scale` |

Terms in one group concatenate in declaration order: an 8-D DROID state is `DofsPos` plus `GripperState`. Builders in
`eden/extensions/policies/adapters.py` cover the recurring layouts: `make_ee_pose_group` (`[pos, quat_wxyz,
openness]`), `make_droid_joint_state_group` (`[q(7), gripper]`) and `make_image_groups`. Camera images, resizing and
channel order go through `eden.utils.image` (`to_uint8_hwc`, `to_chw_uint8`, `resize_uint8`, `resize_with_pad`).

## Actions

`eden/extensions/policies/adapters.py` holds the reusable action decoders for manipulation. Each is an
`ActionAdapter`: a setup assigns one to `self.adapter`, and the policy calls `adapter.bind(env)` once the setup is
built, so an adapter that reads the env fails at construction on a bad name.

| Adapter | Row layout | Drives |
|---------|------------|--------|
| `EEDeltaAdapter` | `[dxyz, drot(3), gripper]`, Euler or axis-angle rotation, handed to the IK controller as a rotation vector | `ee_controller` (differential IK), `gripper_controller` |
| `JointPositionAdapter` | `[q(n), gripper]` absolute targets | `ee_controller` (implicit PD, zero reference), `gripper_controller` |
| `JointVelocityAdapter` | `[qd(n), gripper]` in rad/s, optional lag, compensation and integration | `ee_controller` (implicit velocity or PD), `gripper_controller` |
| `BimanualJointPositionAdapter` | `[left(n), right(m)]` | `left_arm_controller`, `right_arm_controller` |

A policy that emits accumulated Cartesian targets drives the core `IKPoseController` (`eden.actions`) with a 7-D
base-frame pose. Other policy families add their own decoder module beside `adapters.py`.

### Gripper conventions

Gripper conventions differ per server and are the most common porting bug. The client does **not** decide open or
close: `unpack_actions` forwards the server's gripper scalar raw (one column; `NaN` means "no command this step, hold
the previous target"), and the env's `GripperDecisionController` (a `BinaryJointController` in `eden.actions`) applies
the rule declared in the policy's `action_options()`:

| `input` | Rule | Used by |
|---------|------|---------|
| `sign` (default) | `raw > 0` closes | RL policies |
| `open_sign` | `raw > 0` opens | `octo` |
| `openness` | `raw > threshold` opens | OpenPI Bridge, GR00T SimplerEnv WidowX |
| `closedness` | `raw > threshold` closes | DROID-style servers (OpenPI DROID, `cosmos3`, `molmoact2`, GalaxeaVLA) |

Optional knobs, each matching a reference client:

| Knob | Behavior |
|------|----------|
| `open_threshold` / `close_threshold` with `hold_steps` | Hysteresis, then hold the new state. |
| `sticky_repeats` | SimplerEnv's Google Robot latch on openness: a jump beyond ±0.5 is repeated for N steps, otherwise the small relative command is sent, integrated into a continuous finger target (OpenPI `google_robot`). |
| `zero_holds` | With a sign input, exactly `0` keeps the state (robosuite-style sign commands). |
| `continuous` with `snap_threshold` | Continuous closedness, snapped to closed above the threshold (GalaxeaVLA's DROID Franka). |
| `on_missing="measured"` | A `NaN` commands the measured closedness (continuous mode only). The default `hold` keeps the previous target, seeded from the measured position on a first `NaN`. |

`build()` rejects settings the chosen mode would ignore (for example `hold_steps` without hysteresis), and the term's
`reset(envs_idx)` clears its state per env. Every client that forwards a raw gripper scalar calls
`adapters.check_env_gripper_term(env, "gripper_controller", ...)` at construction: it raises when the env has no such
term, when the term is not a `GripperDecisionController` (a task's own `BinaryJointController` would silently misread
the scalar), or when its settings disagree with the policy's. `make_droid_gripper_action` builds the Robotiq
`[0, close_pos]` term every DROID-family setup uses.

The LIBERO setups (OpenPI `libero`, GR00T `libero_sim`) do not use this term: they drive the stock `libero` task's own
stateful `LiberoPandaGripperController` (`+1` closes, `-1` opens, `0` holds, ramping the fingers as robosuite does),
which construction requires by exact type. OpenPI forwards robosuite's command as it is; GR00T's `[0, 1]` openness
becomes `-sign(2g - 1)`, as Isaac-GR00T's LIBERO client maps it.

## Writing a new client

The `/wire-vla` skill (`.claude/skills/wire-vla/SKILL.md`) walks a coding agent through analyzing a policy server's
repository and generating the client files. By hand, the checklist is:

- [ ] `class MyPolicy(ClientMixin, PolicyBase, ConfigurableMixin[RemotePolicyOptions])` (or `SetupClientMixin` for
      several embodiments), decorated with `@POLICY_REGISTRY.register()`, in `eden/extensions/policies/<name>/policy.py`.
- [ ] Pick `transport` (and `serializer` if the default is wrong); declare the knobs the transport reads as class
      attributes.
- [ ] Set the chunk rules: `chunk_mode`, `pred_action_horizon`, `exec_horizon`, `short_chunk`, `response_action_key`.
- [ ] Implement `pack_observations` and `unpack_actions`. Override `decode_response` if the chunk is not under one
      key, `observe` for per-step history, and `request_options` / `reset_options` for server sessions.
- [ ] Implement `observation_options` / `action_options` from the core terms, the `adapters.py` builders and
      `eden.utils.image` before writing new code.
- [ ] Check the env at construction: the gripper term (`check_env_gripper_term`) and the control rate the checkpoint
      was trained at (`check_control_rate`), both in `adapters.py`.
- [ ] State the gripper convention in a comment, and verify it against the server's reference client.
- [ ] Pin the wire contract with a golden case in `tests/extensions/policies/<name>/test_policy.py` (payload, decoded
      actions, inference-call count for a fixed observation sequence), using `record_golden` /
      `assert_matches_golden` from `tests/extensions/policies/conftest.py` and a fake server from
      `tests/support/policy_servers.py`.

## See also

- [The Policy Zoo](policy-zoo.md): every shipped policy
- [Adding a Policy to the Zoo](extending-policies.md): the `PolicyBase` contract
- [Actions](actions.md): IK and binary gripper controllers
- Benchmarks these clients are evaluated on: [SIMPLER](../tasks/simpler.md), [LIBERO](../tasks/libero.md),
  [PolaRiS](../tasks/polaris.md)
