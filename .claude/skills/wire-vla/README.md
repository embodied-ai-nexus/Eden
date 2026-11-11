# wire-vla Skill — VLA Policy Integration Assistant

A Claude skill that guides you through integrating a new server-backed policy (VLA, world action model, remote controller) into the Eden robotics framework.

## Quick Start

From the Eden repository, invoke the skill with:

```bash
/wire-vla <policy-name> [<repo-path>]
```

**Examples:**

```bash
# Integrate openpi-based policy from a repo
/wire-vla my_vla_policy ~/src/my-vla-repo

# Same policy, let Claude ask for the repo path
/wire-vla my_vla_policy

# If in plan mode
/wire-vla some_policy some/path/to/repo
```

## What It Does

The skill walks you through a **7-step process** to integrate any VLA policy:

1. **Parse arguments** — determine policy name and repo path
2. **Fill fact sheet** — analyze the policy repo to extract transport, serialization, obs schema, action layout, gripper convention, etc.
3. **Select transport template** — choose the right base pattern (HTTP+json_numpy, HTTP+JSON, WebSocket, or ZMQ)
4. **Select reusable components** — pick adapters and observation terms that exist in Eden, or mark new ones to write
5. **Generate files** — write all required files in the target directory
6. **Validate code** — check for correctness, imports, registry decorators, etc.
7. **Report results** — summarize what was created and note any manual follow-ups

## Key Concepts

### Transport Patterns

Every client policy is `ClientMixin` (`eden/extensions/policies/client.py`) + a transport key from `eden/utils/policy_client` (see `docs/guides/remote-policies.md`):

| `transport` | Wire | Example |
|-----------|---------------|---------|
| `http_json` | HTTP, JSON body | octo |
| `http_json_numpy` | HTTP, json_numpy body | molmoact2 |
| `http_multipart` | HTTP multipart upload | none shipped (see `tests/utils/policy_client/test_transport.py`) |
| `openpi_websocket` | WebSocket, OpenPI msgpack | pi (OpenPI pi0), cosmos3 |
| `galaxea_websocket` | WebSocket, str-keyed msgpack | galaxeavla |
| `zmq_msgpack` | ZMQ REQ/REP, msgpack_numpy / npy blobs | gr00t |

### Adapters

Eden provides reusable action adapters for common action spaces:

- **EEDeltaAdapter** — converts 7-D EE delta (xyz + rotation + gripper) to controllers
- **JointVelocityAdapter** — converts joint velocities to controllers
- **JointPositionAdapter** — converts absolute joint targets to controllers
- **BimanualJointPositionAdapter** — splits 14-D bimanual into two controllers

### Observation Terms

Eden provides reusable observation functions for common state representations:

- **pi_ee_pose_in_base** — EE pose in base frame (xyz + quat + gripper)
- **pi_joint_state** — joint positions from named DOFs
- **pi_joint_state_with_gripper** — joint positions + separate gripper
- **gr00t_eef_9d** — 9-D rot6d EE pose (DROID-corrected frame)
- **pi_bimanual_joint_state** — bimanual (14-D split)

If none of these match your policy, the skill guides you through writing a custom observation term.

## Generated Files

For a policy named `<policy_name>`, the skill creates:

```
eden/extensions/policies/<policy_name>/
├── __init__.py                  # Re-exports the policy class
├── policy.py                    # Main policy class (one of 4 templates)
├── setups.py                    # (only if multi-robot) Setup registry
└── custom_terms/
    ├── __init__.py              # Empty or re-exports custom terms
    ├── observations.py          # (only if custom obs term needed)
    └── actions.py               # (only if custom action term needed)
```

## Fact Sheet — What the Skill Asks You

Before generating any code, the skill requires answers to:

| Fact | Purpose |
|------|---------|
| **Transport** | HTTP, WebSocket, or ZMQ? | Selects base class template |
| **Serialization** | json_numpy, msgpack, or standard JSON? | Determines step() override needed |
| **Server endpoints** | `/act`, `/set_task`, `/reset` paths and payloads? | Wires ClientMixin/custom transport |
| **Camera schema** | How many cameras? What keys? What format? | Defines observation group |
| **Proprio type** | EE delta, joint pos, joint vel? Same/separate gripper? | Selects obs term function |
| **Action type** | EE delta, joint absolute, joint velocity, bimanual? | Selects adapter or inline unpack |
| **Action layout** | Order of action components (e.g., `[dx,dy,dz,drx,dry,drz,gripper]`)? | Implements action slicing |
| **Gripper convention** | Does raw `> threshold` mean OPEN or CLOSE? Sign flip? | Implements gripper decode logic |
| **Chunk length** | Single action `(D,)` or chunk `(T, D)`? | Determines queue usage |
| **Checkpoint** | Local file or server-only? | Sets PolicyOptions |
| **Multi-robot** | Multiple embodiments supported? | Determines if setups.py needed |

If any fact is ambiguous, the skill asks a clarifying question instead of guessing.

## Template Selection Logic

The skill uses this decision tree:

```
Is transport HTTP?
├─ YES → json_numpy? → YES: MolmoAct2 template
│                   → NO:  OpenPiZero template
└─ NO  → WebSocket? → YES: PiPolicy template (with optional setups.py)
         └─ NO  → ZMQ? → YES: Gr00tPolicy template (with optional setups.py)
                   └─ NO  → ERROR: unsupported transport
```

## Edge Cases Handled

The skill explicitly handles:

- ✅ No `/set_task` or `/reset` endpoints → the default (`None`): set_task / reset only clear local state
- ✅ Single action per call (no chunks) → action queue still works
- ✅ Custom obs term needed → write with `@OBSERVATION_TERM_REGISTRY.register()`
- ✅ Custom action term needed → subclass `ActionTerm` with registry decorator
- ✅ Gripper convention ambiguity → explicit comment in code + convention statement
- ✅ Bimanual policies → uses `BimanualJointPositionAdapter` and dual controllers
- ✅ Multi-robot embodiments → generates `setups.py` with `SetupBase` subclasses and `SETUP_REGISTRY`

## Example Usage

### Example 1: Integrating a Simple HTTP+JSON Policy

```
User: /wire-vla simple_vla ~/src/simple-vla-repo

Skill will:
1. Read the repo and detect HTTP FastAPI + standard JSON
2. Ask: "What are the observation keys in the server payload?"
3. Ask: "What are the action components and their order?"
4. Ask: "What gripper convention does the policy use?"
5. Select: OpenPiZeroPolicy template (ClientMixin base)
6. Select: EEDeltaAdapter for action unpacking
7. Select: pi_ee_pose_in_base for observations
8. Generate: policy.py, __init__.py, custom_terms/__init__.py
9. Report: Files created, next steps (verify DOF names, test single env)
```

### Example 2: Integrating a WebSocket Policy with Multiple Setups

```
User: /wire-vla multi_robot_vla ~/src/multi-robot-vla

Skill will:
1. Read the repo and detect WebSocket + msgpack + multiple configs
2. Extract setup names (e.g., "widowx_bridge", "droid")
3. Ask: "What are the observation schemas for each setup?"
4. Ask: "What are the action spaces for each setup?"
5. Select: PiPolicy template (PolicyBase with WebSocket client)
6. Generate: policy.py, setups.py, custom_terms/__init__.py
7. For each setup, generate SetupBase subclass in setups.py
8. Report: Files created, remind user to test each setup
```

## Manual Follow-Ups

After the skill finishes, you may need to:

- [ ] Verify robot DOF names match your URDF/entity definition
- [ ] Ensure the server is running before testing
- [ ] If using a local checkpoint, download/place it at the expected path
- [ ] Test the policy with a single inference call before training
- [ ] If multi-setup, test each setup independently
- [ ] Adjust observation grouping if the server expects a different layout

## Reusing Existing Policies

For reference, examine these files:

- **HTTP + json_numpy:** `eden/extensions/policies/molmoact2/policy.py`
- **HTTP + JSON:** `eden/extensions/policies/octo/policy.py`
- **WebSocket (single embodiment):** `eden/extensions/policies/cosmos3/policy.py`
- **WebSocket (multi-setup):** `eden/extensions/policies/pi/policy.py` + `pi/setups.py`
- **ZMQ (multi-setup):** `eden/extensions/policies/gr00t/policy.py` + `gr00t/setups.py`

## Validation Checklist

Before running your generated policy, the skill checks:

- ✅ All abstract methods implemented (`step`, `action_options`, `observation_options`)
- ✅ `ClientMixin` appears before `PolicyBase` in inheritance (if used)
- ✅ All imports are absolute (e.g., `from eden.managers...`)
- ✅ Heavy imports guarded by `TYPE_CHECKING`
- ✅ Custom obs/action terms have registry decorators
- ✅ Gripper convention explicitly documented in code
- ✅ No circular imports

## Troubleshooting

**Q: The skill asked for clarification on gripper convention, but I'm not sure.**

A: Look at the server code for the inference output post-processing. Find comments like "close gripper if output > 0.5" or similar. If unsure, test with the server directly using a simple script to see what command closes vs. opens the gripper.

**Q: Can I use an adapter from `policies/adapters.py` with a policy that's not pi?**

A: Yes! Adapters are generic and can be reused. `EEDeltaAdapter`, `JointVelocityAdapter`, etc. work with any policy. Just import them: `from eden.extensions.policies.adapters import EEDeltaAdapter`. Adapters forward the gripper scalar raw; the env's `GripperDecisionController` (declared in `action_options()`) makes the open/close decision.

**Q: Do I need to write a custom observation term?**

A: Only if your state representation doesn't match one of the built-in terms. For most policies (WidowX, DROID, ALOHA), `pi_joint_state_with_gripper` or `pi_ee_pose_in_base` will work. If your policy uses a novel state (e.g., force/torque, depth, wrist proprioception), write a custom term.

**Q: The skill generated setups.py but I only have one robot config.**

A: Delete `setups.py` and move `pack_observations()` and `unpack_actions()` directly to the policy class. Then remove the `policy_setup` field and any `SETUP_REGISTRY` imports.

## Next Steps

After the skill finishes:

1. **Review generated files** — ensure they match your expected policy behavior
2. **Test single environment** — create a small test script that:
   - Builds an Eden env with your new policy
   - Calls `env.step(action)` a few times
   - Checks that observations and actions are sensible
3. **Verify DOF names** — cross-check the DOF names in `action_options()` against your URDF
4. **Run full training** — once single-env testing passes, launch RL training

## Questions or Issues?

Refer to the existing policy integrations in `Eden/eden/extensions/policies/` for patterns, or consult the skill's embedded reference sections (Templates A–D, decision trees, adapter/obs term tables).
