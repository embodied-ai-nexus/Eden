---
name: wire-vla
description: Wire a new server-backed policy (VLA, world action model, remote controller) into Eden as a ClientMixin client. Analyzes the policy repo, selects the transport key, setups and adapters, and generates all required integration files.
allowed-tools: Read, Grep, Glob, Bash, Edit, Write, Agent
argument-hint: <policy-name> [<repo-path>]
---

You are an expert at wiring server-backed policies — VLAs, world action models (cosmos3), remote RL controllers, diffusion policies — into the Eden robotics simulation framework. Every such policy is a `ClientMixin` client (`eden/extensions/policies/client.py`, wire plumbing in `eden/utils/policy_client/`, guide: `docs/guides/remote-policies.md`) that picks a **transport key**, packs observations, and decodes action rows. You know the six existing clients (pi, gr00t, galaxeavla, cosmos3, octo, molmoact2) and can rapidly assess a new policy repo to determine the correct transport, serializer, observation schema, action space, and adapters.

---

# Wire-VLA Skill — Complete Integration Guide

If `$ARGUMENTS` is provided, parse it as: `<policy-name> [<repo-path>]`. If no repo path is given, ask the user for it.

Follow the workflow below **in order**. Do not skip steps. Fill in the fact sheet completely before generating any code.

---

## Step 1: Parse Arguments and Determine Paths

1. Extract `policy_name` as the first token of `$ARGUMENTS` (convert to snake_case if needed).
2. Extract `repo_path` as the second token if provided, otherwise ask the user for the path.
3. Set the target directory to: `eden/extensions/policies/<policy_name>/`
4. Confirm the paths with the user before proceeding.

---

## Step 2: Fill the Fact Sheet — Read and Analyze the Policy Repo

Before writing a single line of code, you must answer all of the questions below by reading the target repo. If a fact is ambiguous or unclear, ask a clarifying question instead of guessing.

### Transport Protocol

**How to discover:** Grep the server entrypoint (e.g., `server.py`, `main.py`, `run.py`, `serve_policy.py`) for keywords.

- **FastAPI/Flask indicator:** `@app.post`, `@app.get`, `uvicorn.run`, `flask.Flask`
- **WebSocket indicator:** `websocket`, `asyncio.websockets`, `server.WebSocket`
- **ZMQ indicator:** `zmq.Context`, `zmq.REQ`, `socket.connect`, port typically 5555
- **Request HTTP client:** `requests.Session`, `requests.post`, `httpx`

**Question:** Is the transport HTTP (FastAPI/Flask), WebSocket, or ZMQ?

### Serialization Format

**How to discover:** Grep the request/response code for serialization patterns.

- **json_numpy:** `json_numpy.dumps()`, `json_numpy.loads()`, `import json_numpy`
- **msgpack:** `msgpack_numpy`, `msgpack.packb`, `msgpack.unpackb`
- **Standard JSON:** `json.dumps()`, `requests.json` parameter, `.json()` call

**Question:** Does the policy use json_numpy, msgpack, or standard JSON serialization?

### Server Endpoints and Payloads

**How to discover:** Read the server code (FastAPI route definitions or ZMQ endpoint handlers).

For **HTTP (FastAPI):**
- Primary inference endpoint: e.g., `/act`, `/action`, `/infer`
- Task-setting endpoint: e.g., `/set_task` (or none if stateless)
- Reset endpoint: e.g., `/reset` (or none)
- Expected request body schema: list the keys (e.g., `{"instruction": str, "image": bytes, "proprio": array}`)
- Expected response schema: list the keys (e.g., `{"actions": array}` or `{"action": list}`)

For **WebSocket:**
- Does it use an external client library (e.g., `openpi_client.websocket_client_policy.WebsocketClientPolicy`)? Name it.
- What are the message formats (msgpack, JSON, etc.)?

For **ZMQ:**
- What are the endpoint names? (e.g., `get_action`, `reset`, `ping`)
- Does it use msgpack or custom serialization?

**Questions:**
- What are the primary inference endpoint(s)?
- What are the request and response schemas?
- Does the server have `/set_task` and/or `/reset` endpoints?

### Observation Schema: Camera and Proprioceptive State

**How to discover:** Read the inference function signature and any docstrings. Look for `preprocess_observation`, `pack_observation`, or similar wrapper functions.

**Camera configuration:**

- **How many cameras?** 1, 2 (arm + wrist), 3+ (multi-view)?
- **Camera keys in the payload?** (e.g., `image`, `external_cam`, `wrist_cam`, `exterior_image_1_left`)
- **Image format:** HxWx3 uint8 RGB (standard), or CHW, or other?
- **Expected resolution:** e.g., 224x224, 256x256, 512x512

**Proprioceptive state:**

This is **critical** — get it exactly right.

- **Type of state:** Is it EE (end-effector) pose, joint positions, joint velocities, or a mix?
  - **EE delta pose:** 7-D `[dx, dy, dz, drx, dry, drz, gripper]` or similar
  - **EE absolute pose:** 7-D `[x, y, z, qw, qx, qy, qz]` + gripper
  - **Joint positions:** N-D vector of absolute joint targets (including gripper)
  - **Joint velocities:** N-D vector of desired joint velocities
  
- **Dimension and layout:** e.g., 8-D for WidowX (7 joints + 1 gripper), 14-D for bimanual (7+7)
- **Gripper representation:** Is the gripper included as a separate DOF (yes for most), or separate?
- **Same entity or separate?** Are arm and gripper on the same entity, or does gripper have a separate entity?

**Questions:**
- How many cameras does the policy expect, and what are their key names?
- What is the proprioceptive state type (EE delta, joint positions, joint velocities)?
- How many DOFs (arm+gripper) does the state encode?
- Is the gripper on the same entity as the arm, or a separate entity?

### Action Space: Type, Layout, and Gripper Convention

**How to discover:** Read the `model.predict_action()` or `infer()` output and any normalization/decoding logic.

**Action type:**

- **EE delta:** Relative end-effector motion `[dx, dy, dz, drx, dry, drz, gripper_cmd]`
- **Joint position:** Absolute joint targets `[q1, q2, ..., q7, gripper]`
- **Joint velocity:** Desired joint velocities `[dq1, dq2, ..., dq7, gripper_dq]`
- **Bimanual:** Two arms worth of actions (14-D total for dual 7-DOF arms)

**Action dimension and layout:** Count how many components the action has and list their order.
Example: `[dx, dy, dz, drx_axisangle, dry_axisangle, drz_axisangle, gripper_closedness]` = 7-D with axis-angle rotation.

**Rotation representation (if EE delta):** Euler XYZ, axis-angle, or 6D rotation matrix representation?

**Gripper convention — THIS IS CRITICAL:**
- Raw output range: e.g., `[-1, 1]`, `[0, 1]`, float32 raw
- Interpretation: Does `output > threshold` mean OPEN or CLOSE?
- Example: "If raw > 0.5 then gripper is OPEN; else CLOSE" (typical for OpenPI zero)
- Or: "If raw > 0.5 then gripper is CLOSE; else OPEN" (alternative)
- **Where the decision lives:** do NOT binarize or sign-flip on the client. Forward the raw
  scalar and declare the rule on the env term: `GripperDecisionController.configure(..., input=...)`
  with `input` = `sign` (>0 closes), `open_sign` (>0 opens), `openness` (>threshold opens) or
  `closedness` (>threshold closes); `open_threshold`/`close_threshold`/`hold_steps` for
  hysteresis, `sticky_repeats` for the Google-robot latch, `continuous`/`snap_threshold` for
  continuous drive. Emit `NaN` when the server sends no gripper this step (hold previous).

**Action chunk length:**
- Does the server return a single action `(D,)` per call, or a chunk `(T, D)` (multiple timesteps)?
- If a chunk, what is the typical `T`?

**Questions:**
- What is the action type (EE delta, joint position, joint velocity, or bimanual)?
- What is the exact action layout (order of components)?
- If rotation is EE delta, is it Euler XYZ or axis-angle?
- What is the gripper convention: raw `> threshold` means OPEN or CLOSE? Any sign flip?
- Does the server return single steps or action chunks?

### Checkpoint and Configuration

**How to discover:** Read the server startup code and config files.

- **Checkpoint loading:** Is there a local checkpoint file path, or is the model downloaded at runtime (HuggingFace hub, etc.), or is it server-only?
- **If local path:** Where is it stored? How is it specified (CLI arg, config file, hardcoded)?
- **Multi-robot support:** Does the policy support multiple embodiments (e.g., WidowX, ALOHA, DROID)? If yes, how are they configured?

**Questions:**
- Does the policy require a local checkpoint, or is it purely remote server?
- If local, how is the checkpoint path specified?
- Does the policy support multiple robot configurations?

---

## Step 3: Select the Transport Key and Base Class

Every generated policy is `class <Name>Policy(ClientMixin, PolicyBase, ConfigurableMixin[RemotePolicyOptions])`
(or `SetupClientMixin` instead of `ClientMixin` when the server serves several embodiments).
**Never** write a private WebSocket/ZMQ client class or override `step()`: pick a transport from
`TRANSPORT_REGISTRY` and, if the reply does not carry the chunk under one key, override `decode_response()`.

| Fact sheet | `transport=` | `serializer=` | Notes |
|---|---|---|---|
| HTTP, plain JSON body | `"http_json"` | default | `set_task_as_query=True` / `reset_with_payload=False` for query-param / bare-POST servers (Octo) |
| HTTP, `json_numpy` body | `"http_json_numpy"` | default | reply decoded with `json_numpy`; `{"error": ...}` raises `ServerError` |
| HTTP, multipart / file upload | `"http_multipart"` | — | override `pack_request(obs) -> (data, files)` |
| WebSocket, OpenPI msgpack (`openpi_client` server) | `"openpi_websocket"` | default | `api_key`, `ws_ping_timeout` knobs; `response_action_key="actions"` for pi0 servers |
| WebSocket, GalaxeaVLA-style (str-keyed records, `__reset__`) | `"galaxea_websocket"` | default | `chunk_mode="server"` |
| ZMQ REQ/REP `{"endpoint","data"}` (Isaac-GR00T family) | `"zmq_msgpack"` | `msgpack_numpy` (default) or `msgpack_npy_blob` (RLDX-1) | reply is `[action_dict, info]`; concatenate keys in `decode_response()` |
| anything else | add a `TransportBase` subclass to `eden/utils/policy_client/transport.py` + a test in `tests/utils/policy_client/test_transport.py` | | do not embed it in the policy |

Set the chunk rules from the fact sheet: `chunk_mode` (`client` / `server` / `single`),
`pred_action_horizon`, `exec_horizon`, `short_chunk` (`error` if the server's horizon is fixed, `accept` otherwise),
`response_action_key`. `set_task_endpoint` / `reset_endpoint` default to `None` (no such server API — the
server takes the task with every request, so an env reset only clears local state). Set them **only** when the
official server exposes the route: an HTTP path (`"/reset"`), a ZMQ endpoint name (`"reset"`) or a GalaxeaVLA
reset key (`"__reset__"`); `openpi_websocket` has no reset message and rejects one.

**Multi-embodiment?** Use `SetupClientMixin` + a `SETUP_REGISTRY` of `PolicySetupBase` subclasses (`setups.py`),
each carrying `pred_action_horizon` / `exec_horizon` / `default_image_size` and the per-robot `pack_observations`,
`decode_response`, `unpack_actions`, `observation_options`, `action_options`. Examples: `pi/setups.py`, `gr00t/setups.py`.

---

## Step 4: Select Reusable Eden Components — Adapters and Observation Terms

### 4.1 Select the Action Adapter (if needed)

Based on the **action type** from your fact sheet:

| Action Type | Adapter | Import | How to use |
|-------------|---------|--------|-----------|
| EE delta, Euler rotation | `EEDeltaAdapter(rotation="euler")` | `from eden.extensions.policies.adapters import EEDeltaAdapter` | Converts 7-D EE delta to `{ee_controller, gripper_controller}`; gripper column forwarded raw |
| EE delta, Axis-angle rotation | `EEDeltaAdapter(rotation="axis_angle")` | Same import | Same as above, different rotation format |
| Joint velocity | `JointVelocityAdapter(n_arm=7, arm_dofs_name=..., mode="velocity")` | `from eden.extensions.policies.adapters import JointVelocityAdapter` | Converts N-D joint velocity to `{ee_controller, gripper_controller}` (supports integration modes) |
| Joint position absolute | `JointPositionAdapter(n_arm=7)` | `from eden.extensions.policies.adapters import JointPositionAdapter` | Converts N-D absolute joint targets to controller dict |
| Bimanual (14-D split) | `BimanualJointPositionAdapter` | `from eden.extensions.policies.adapters import BimanualJointPositionAdapter` | Splits 14-D into two 7-D controllers (left + right) |
| Custom / mixed | **Write inline `unpack_actions()`** | N/A | Manually slice the action array and return controller dict |

If your action space doesn't fit a standard adapter, **write the `unpack_actions()` method inline** to manually decode the action. Example snippet:
```python
def unpack_actions(self, raw_action: torch.Tensor) -> dict[str, torch.Tensor]:
    # raw_action shape: (1, D) or (N, D) for batched policies
    return {
        "arm_controller": raw_action[:, :7],      # first 7 are joint targets
        "gripper_controller": raw_action[:, 7:8],  # last 1 is gripper, RAW: the env term decides
    }
```

The gripper decision belongs to the env term, declared in `action_options()`:
```python
from eden.extensions.policies.adapters import make_droid_gripper_action  # Robotiq [0, close_pos]
# or, for any gripper:
from eden.managers.terms.actions import GripperDecisionController
GripperDecisionController.configure(entity_name=..., dofs_name=[...], open_action=..., close_action=...,
                            input="closedness", threshold=0.5)
```

### 4.2 Compose Observation Terms

Proprioception is composed from three **core** class terms in
`eden.managers.terms.observations.manipulation` (indices resolved once in `build()`). Terms in
one `ObservationGroupOptions` concatenate in declaration order. Never import another policy's
`custom_terms`.

| State layout | Compose |
|---|---|
| `[pos(3), quat_wxyz(4), gripper_openness(1)]` | `make_ee_pose_group(...)` from `eden.extensions.policies.adapters` |
| `[q(N), gripper(1)]` (DROID style) | `make_droid_joint_state_group(...)` |
| EE pose in another rotation format / frame | `EndEffectorPose.configure(rotation="euler_xyz" \| "axis_angle" \| "rot6d", frame_correction_quat=..., offset_pos=...)` |
| Arm joints only | `DofsPos.configure(entity_name, dofs_name)` (`observations.DofsPos`) |
| Gripper raw / openness / closedness | `GripperState.configure(entity_name, dofs_name, normalize=..., open_pos=..., close_pos=...)` |
| Bimanual | two `DofsPos` (+ two `GripperState`) in one group |
| Genuinely new quantity | new term in `custom_terms/observations.py`, class-based, `build()` resolves names, `compute()` reads via `resolve_entity`, geometry via `eden.utils.geom` |

### 4.3 Camera Groups in Observations

Based on the **camera configuration** from your fact sheet:

| Camera Setup | Obs Group Pattern |
|--------------|-------------------|
| Single camera | `image=ObsTerm.configure(func=observations.rgb_image, ...)` |
| Two cameras (external + wrist) | Use `_make_image_groups()` helper or manual dict group with sub-keys `external` and `wrist` |
| 3+ cameras | Build a dict-based `ObservationGroupOptions` with named sub-keys per camera |

Reference: see `pi/setups.py` for the `_make_image_groups()` helper pattern.

---

## Step 5: Generate Files

Generate files in **this exact order** in the target directory `eden/extensions/policies/<policy_name>/`.

### 5.1 Generate `custom_terms/__init__.py`

- If no custom observation or action terms are needed: empty file (just a pass statement or comment)
- If custom terms exist: re-export them, e.g.:
  ```python
  from eden.extensions.policies.<policy_name>.custom_terms.observations import my_custom_obs
  from eden.extensions.policies.<policy_name>.custom_terms.actions import MyCustomAction

  __all__ = ["my_custom_obs", "MyCustomAction"]
  ```

### 5.2 Generate `custom_terms/observations.py` (only if needed)

**When to create this file:**
- If you selected "Write new function" in step 4.2 (novel obs term)

**File structure:**
```python
from typing import TYPE_CHECKING

import torch

from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY

if TYPE_CHECKING:
    from eden.managers.base_manager import EnvBase


@OBSERVATION_TERM_REGISTRY.register()
def my_obs_term(env: "EnvBase", *, entity_name: str, ...) -> torch.Tensor:
    """Docstring describing the observation."""
    entity = env.entities[entity_name]
    # ... implementation ...
    return result_tensor
```

**IMPORTANT:** 
- All function parameters **after the `*` separator** are keyword-only
- Use absolute imports (e.g., `from eden.managers...`, not `from ...`)
- Import `EnvBase` only inside `TYPE_CHECKING` block
- Return a tensor of shape `(num_envs, obs_dim)`

### 5.3 Generate `custom_terms/actions.py` (only if needed)

**When to create this file:**
- If you selected "Write inline" for the adapter AND the policy uses a custom action term class (rare)
- No shipped client needs one; `MujocoSMPLActuator` (`eden/extensions/policies/metamotivo/custom_terms/actions.py`) shows the shape of a custom `ActionTerm` subclass. Absolute pose targets use the core `IKPoseController` instead of a new term

**File structure:**
```python
from typing import TYPE_CHECKING

import torch

from eden.managers.action_manager import ActionTerm, ACTION_TERM_REGISTRY

if TYPE_CHECKING:
    from eden.managers.base_manager import EnvBase


@ACTION_TERM_REGISTRY.register()
class MyCustomActionTerm(ActionTerm):
    """Custom action term class."""
    
    entity_name: str = "robot"
    # ... other config fields ...

    def compute(self, action: torch.Tensor) -> torch.Tensor:
        """Process the action tensor."""
        # ... implementation ...
        return processed_action

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset any internal state."""
        pass

    def apply_actions(self, actions: torch.Tensor) -> None:
        """Apply the action to the environment."""
        # ... implementation ...
```

### 5.4 Generate `setups.py` (only if multi-embodiment)

Reference: `eden/extensions/policies/gr00t/setups.py` (ZMQ, per-key concat) or `pi/setups.py` (WebSocket, adapters).

```python
"""Per-embodiment setups for <PolicyName>Policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from eden.extensions.policies.adapters import DROID_ARM_DOFS, JointPositionAdapter, make_image_groups  # or EEDeltaAdapter, ...
from eden.extensions.policies.client import PolicySetupBase
from eden.utils.image import prepare_image_resized
from eden.utils.torch import to_numpy_unbatched

if TYPE_CHECKING:
    from eden.options.managers.actions import ActionTermOptions
    from eden.options.managers.observations import ObservationGroupOptions


class <Robot>Setup(PolicySetupBase):
    pred_action_horizon = 16
    exec_horizon = 8
    default_image_size = (224, 224)

    def __init__(self, policy) -> None:
        super().__init__(policy)
        # The gripper scalar is forwarded raw; the env's GripperDecisionController makes the decision.
        self.adapter = JointPositionAdapter(n_arm=7)

    @classmethod
    def observation_options(cls, camera_name: str = "camera", **_) -> dict[str, "ObservationGroupOptions"]:
        groups = make_image_groups(camera_name, use_wrist_image=False, wrist_camera_name="")
        # + proprio group built from existing observation terms
        return groups

    @classmethod
    def action_options(cls, **_) -> dict[str, "ActionTermOptions"]:
        ...

    def pack_observations(self, obs: dict) -> dict:
        image = prepare_image_resized(obs["image"], self.policy.image_size)
        state = to_numpy_unbatched(obs["proprio"]).astype(np.float32)
        return {"image": image, "state": state, "prompt": self.policy.task_description}

    # GR00T-family servers: declare the schema and it is checked against get_modality_config at connect
    # video_keys / state_keys / action_keys / language_key / video_horizon / video_stride + action_dim

    # optional: reply -> (T, D) when the server spreads the chunk over several keys
    def decode_response(self, response) -> np.ndarray | None:
        return None

    # unpack_actions(row) defaults to self.adapter(row, env=self.policy.env)


SETUP_REGISTRY: dict[str, type[PolicySetupBase]] = {"<robot>": <Robot>Setup}
```

### 5.5 Generate `policy.py` — The Main Policy Class

One template; the transport key does the rest. Reference implementations: `molmoact2/policy.py`
(`http_json_numpy`), `octo/policy.py` (`http_json`, `single` chunk mode, `decode_response` override),
`cosmos3/policy.py` (`openpi_websocket`, single embodiment, adapter), `pi/policy.py` + `gr00t/policy.py` +
`galaxeavla/policy.py` (`SetupClientMixin`). No shipped client overrides the `observe()` hook; its base
implementations are in `eden/extensions/policies/client.py`.

```python
"""<PolicyName> remote policy client.

<One paragraph: server repo, transport, payload keys, action layout, gripper convention.>
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from eden.extensions.policies.base import POLICY_REGISTRY, PolicyBase
from eden.extensions.policies.client import ClientMixin           # or: SetupClientMixin
from eden.utils.image import to_uint8_hwc
from eden.utils.torch import to_numpy_unbatched
from eden.options.extensions.policies import RemotePolicyOptions
from eden.utils.common import ConfigurableMixin

if TYPE_CHECKING:
    from eden.options.managers.actions import ActionTermOptions
    from eden.options.managers.observations import ObservationGroupOptions


@POLICY_REGISTRY.register()          # digit-suffixed names: register(name="<pkg>_policy")
class <PolicyName>Policy(ClientMixin, PolicyBase, ConfigurableMixin[RemotePolicyOptions]):
    """<PolicyName>: <one line>."""

    # -- connection / wire --------------------------------------------
    host: str = "localhost"
    port: int | None = 8000                    # None: take the port from a full-URL host
    transport: str = "<transport key from Step 3>"
    # serializer: str | None = None            # only when the transport default is wrong
    # reset_endpoint: str | None = "/reset"    # only when the official server has a reset API
    response_action_key: str = "action"        # or "actions"

    # -- chunking -------------------------------------------------------
    chunk_mode: str = "client"
    pred_action_horizon: int | None = None
    exec_horizon: int | None = None
    short_chunk: str = "accept"

    # -- policy knobs ---------------------------------------------------
    task_description: str = "pick up the object"
    gripper_threshold: float = 0.5

    def pack_observations(self, obs: dict) -> dict:
        image = to_uint8_hwc(obs["image"]["rgb"])
        state = to_numpy_unbatched(obs["proprio"]).astype(np.float32)
        return {"instruction": self.task_description, "image": image, "state": state}

    def unpack_actions(self, raw_action: np.ndarray) -> dict[str, torch.Tensor]:
        # raw_action: (1, D). Gripper: <state the server convention here and how it maps to Eden>.
        device = self.env.device
        arm = torch.as_tensor(raw_action[:, :7], dtype=torch.float32, device=device)
        gripper = torch.as_tensor((raw_action[:, 7:8] > self.gripper_threshold).astype(np.float32), device=device)
        return {"arm_controller": arm, "gripper_controller": gripper}

    # optional hooks: decode_response(reply) -> (N, T, D); observe(obs); request_options(); _reset_local()

    @classmethod
    def action_options(cls, entity_name: str = "robot", **_) -> dict[str, "ActionTermOptions"]:
        ...

    @classmethod
    def observation_options(cls, camera_name: str = "camera", **_) -> dict[str, "ObservationGroupOptions"]:
        ...
```

### 5.6 Generate `__init__.py` — Package Initialization

```python
from eden.extensions.policies.<policy_name>.policy import <PolicyName>Policy

__all__ = ["<PolicyName>Policy"]
```

---

## Step 6: Validate Generated Code

- [ ] Class is `(ClientMixin | SetupClientMixin, PolicyBase, ConfigurableMixin[RemotePolicyOptions])`, mixin **first**, decorated with `@POLICY_REGISTRY.register()`
- [ ] No `step()` override, no private WebSocket/ZMQ client class, no `requests` import in the policy — the transport key covers it
- [ ] `transport` (and `serializer` if needed) is a registered key; transport knobs the policy uses are declared class attributes
- [ ] Chunk rules set: `chunk_mode`, `pred_action_horizon`, `exec_horizon`, `short_chunk`, `response_action_key`
- [ ] `pack_observations` / `unpack_actions` implemented; `decode_response` / `observe` / `request_options` only when the server needs them
- [ ] `action_options` / `observation_options` implemented; keys returned by `unpack_actions` match the action term names
- [ ] Gripper convention stated in a comment inside `unpack_actions`
- [ ] Images via `eden.utils.image` (`to_uint8_hwc`, `prepare_image_padded`, `prepare_image_resized`); no local float-to-255 heuristics
- [ ] Geometry via `eden.utils.geom` / `genesis.utils.geom`, never re-implemented
- [ ] Absolute imports; heavy types under `TYPE_CHECKING`; custom terms decorated with their registry
- [ ] `__init__.py` re-exports the policy class (the CLI sweep needs it)
- [ ] A golden case in `tests/extensions/policies/<name>/test_policy.py` (a fake server from `tests/support/policy_servers.py`, fixed obs -> payload + actions, via `record_golden` / `assert_matches_golden` in `tests/extensions/policies/conftest.py`), recorded under `tests/data/policy_client_goldens/` with `EDEN_REGEN_GOLDENS=1`
- [ ] `uv run pytest tests/extensions/policies/<name>/test_policy.py` and `uv run pytest tests/extensions/policies/test_client.py` pass

---

## Step 7: Report to User

Summarize what you have created:

1. **Files generated:**
   - List each file path
   - Brief description of what each does

2. **Key decisions made:**
   - Which template was selected (HTTP + json_numpy, HTTP + JSON, WebSocket, ZMQ)
   - Which adapters or obs terms were reused
   - Whether a setup registry was needed
   - Any custom obs/action terms created

3. **Manual follow-up steps (if any):**
   - [ ] Verify robot DOF names match the URDF / entity definition
   - [ ] Ensure server is running before testing with the policy
   - [ ] If using a checkpoint file, download/place it at the expected path
   - [ ] Test the policy with a single inference call before running full training
   - [ ] If multi-setup, test each setup independently

4. **Next steps:**
   - Import the policy: `from eden.extensions.policies.<policy_name>.policy import <PolicyName>Policy`
   - Wire into an environment config (see examples in `eden/tasks/`)
   - Test with a single environment before scaling

---

## Reference: Critical Patterns and Conventions

### MRO for client policies

```python
class MyPolicy(ClientMixin, PolicyBase, ConfigurableMixin[RemotePolicyOptions]):
```

`ClientMixin` must resolve `step()` before `PolicyBase`; the third base gives `configure()` the merged option fields.

### Step pipeline (what the mixin does for you)

```
observe(obs) -> [needs_inference: pack_observations -> transport.infer -> decode_response -> scheduler.accept]
             -> scheduler.pop() -> unpack_actions(row)
```

`set_task` / `reset_server` / `reset` clear the scheduler and `_reset_local()` state, then call the transport.
Whatever the server returns besides the chunk is on `policy.last_response_info`.

### Gripper convention examples

```python
# OpenPI / Bridge servers: raw > 0.5 means OPEN; Eden BinaryJointController closes on > 0 -> sign flip
gripper = torch.where(raw > 0.5, -1.0, 1.0)

# DROID-style servers: raw is closedness in [0, 1]; ImplicitPD gripper scaled by close position
gripper = (raw > threshold).float()
```

### Fail fast

Validate `policy_setup`, horizons and required observation keys in `_post_configure()` / `pack_observations()`
with messages naming the offending value and the valid options. Do not `getattr(..., default)` around missing
policy knobs — declare them.

---

## Examples to Reference

| Task | Reference |
|------|-----------|
| HTTP + `json_numpy` | `eden/extensions/policies/molmoact2/policy.py` |
| HTTP, one action per call, multi-key reply | `eden/extensions/policies/octo/policy.py` |
| HTTP multipart (no shipped client) | `HttpMultipartTransport` in `eden/utils/policy_client/transport.py`, `tests/utils/policy_client/test_transport.py` |
| WebSocket, single embodiment (world action model) | `eden/extensions/policies/cosmos3/policy.py` |
| WebSocket + setups | `eden/extensions/policies/pi/policy.py`, `pi/setups.py` |
| ZMQ + setups, per-key concat | `eden/extensions/policies/gr00t/policy.py`, `gr00t/setups.py` |
| Server-side chunking, omitted action parts | `eden/extensions/policies/galaxeavla/setups.py` |
| Transports / serializers | `eden/utils/policy_client/{transport,serializer}.py` |
| Mixin / setups / scheduler | `eden/extensions/policies/client.py` |
| Adapters | `eden/extensions/policies/adapters.py` |
| Golden behaviour pins | `tests/extensions/policies/conftest.py` (harness), `tests/data/policy_client_goldens/*.json` |

---

## Checklist: Before Declaring Success

- [ ] All fact sheet questions answered
- [ ] Transport template selected and justified
- [ ] Adapters and obs terms identified (reused or new)
- [ ] All files generated in the correct directory
- [ ] Code passes validation checklist (Step 6)
- [ ] Gripper convention explicitly documented
- [ ] No circular imports
- [ ] All imports are absolute
- [ ] TYPE_CHECKING guards in place for heavy imports
- [ ] Report provided to user with key decisions and manual follow-ups
