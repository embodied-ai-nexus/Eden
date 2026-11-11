# Policies

A policy extension packages a frozen policy so Eden can run it: a local actor (a checkpoint or an ONNX graph with its
observation and action contract) or a client of a policy server (a VLA or world action model). Every policy is a
`PolicyBase` subclass registered in `POLICY_REGISTRY`, run with `eden inference --policy <name>`. The
[policy zoo guide](../../guides/policy-zoo.md) lists what each one needs, the
[remote policies guide](../../guides/remote-policies.md) explains server-backed clients, and
[extending policies](../../guides/extending-policies.md) shows how to add one.

```python
from eden.extensions.policies.base import POLICY_REGISTRY, PolicyBase
from eden.options.extensions.policies import PolicyOptions, RemotePolicyOptions
```

## Base class and registry

::: eden.extensions.policies.base
    options:
      show_root_heading: false

## Options

::: eden.options.extensions.policies
    options:
      show_root_heading: false

## ONNX runtime helpers

::: eden.extensions.policies.onnx
    options:
      show_root_heading: false

## Server-backed clients

A server-backed policy mixes in `ClientMixin`: it builds a request from the env's observations, sends it through a
transport and serializer, and schedules the returned action chunk. The adapters turn a server's action format into
the env's action terms.

::: eden.extensions.policies.client
    options:
      show_root_heading: false

### Action adapters

::: eden.extensions.policies.adapters
    options:
      show_root_heading: false

### Transports, serializers and errors

::: eden.utils.policy_client.transport
    options:
      show_root_heading: false

::: eden.utils.policy_client.serializer
    options:
      show_root_heading: false

::: eden.utils.policy_client.errors
    options:
      show_root_heading: false

## Local policies

### Locomotion

::: eden.extensions.policies.holosoma.policy.HolosomaLocomotionPolicy
    options:
      show_bases: true
      inherited_members: false

::: eden.extensions.policies.homie.policy.HomiePolicy
    options:
      show_bases: true
      inherited_members: false

::: eden.extensions.policies.limx_oli_loco.policy.OliVelocityLocoPolicy
    options:
      show_bases: true
      inherited_members: false

::: eden.extensions.policies.husky.policy.HuskyPolicy
    options:
      show_bases: true
      inherited_members: false

### Whole-body control and motion tracking

::: eden.extensions.policies.amo.policy.AMOPolicy
    options:
      show_bases: true
      inherited_members: false

::: eden.extensions.policies.beyondmimic_moves.policy.BeyondMimicMovesPolicy
    options:
      show_bases: true
      inherited_members: false

::: eden.extensions.policies.gmt.policy.GMTPolicy
    options:
      show_bases: true
      inherited_members: false

::: eden.extensions.policies.humanoid_gpt.policy.HumanoidGptPolicy
    options:
      show_bases: true
      inherited_members: false

::: eden.extensions.policies.metamotivo.policy.MetaMotivoPolicy
    options:
      show_bases: true
      inherited_members: false

::: eden.extensions.policies.omnixtreme.policy.OmniXtremePolicy
    options:
      show_bases: true
      inherited_members: false

::: eden.extensions.policies.protomotionv3.policy.ProtoMotionV3Policy
    options:
      show_bases: true
      inherited_members: false

::: eden.extensions.policies.sonic.policy.SonicPolicy
    options:
      show_bases: true
      inherited_members: false

::: eden.extensions.policies.twist2.policy.Twist2Policy
    options:
      show_bases: true
      inherited_members: false

### Manipulation

::: eden.extensions.policies.omnireset.policy.OmniResetPolicy
    options:
      show_bases: true
      inherited_members: false

::: eden.extensions.policies.simtoolreal.policy.SimToolRealPolicy
    options:
      show_bases: true
      inherited_members: false

## Remote policy clients

::: eden.extensions.policies.pi.policy.PiPolicy
    options:
      show_bases: true
      inherited_members: false

::: eden.extensions.policies.gr00t.policy.Gr00tPolicy
    options:
      show_bases: true
      inherited_members: false

::: eden.extensions.policies.cosmos3.policy.Cosmos3Policy
    options:
      show_bases: true
      inherited_members: false

::: eden.extensions.policies.galaxeavla.policy.GalaxeaVlaPolicy
    options:
      show_bases: true
      inherited_members: false

::: eden.extensions.policies.molmoact2.policy.MolmoAct2Policy
    options:
      show_bases: true
      inherited_members: false

::: eden.extensions.policies.octo.policy.OctoPolicy
    options:
      show_bases: true
      inherited_members: false

## See also

- [Policy zoo guide](../../guides/policy-zoo.md)
- [Remote policies guide](../../guides/remote-policies.md)
- [Extending policies](../../guides/extending-policies.md)
