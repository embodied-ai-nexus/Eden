# Learning

`eden.learning` holds learning-side code that is not a task: policy export, and algorithms that extend an RL
library. A task selects an algorithm in its `runner_options` through an options class whose `class_name` names the
implementation, so `eden train --task <name>` trains it and a saved `config.json` resumes through it. The
[training runs guide](../guides/training-runs.md) explains runners and custom algorithms; the runner options
themselves are on the [learning options page](options/learning.md).

## Policy export

`eden export` writes `policy.onnx`, `policy.pt` (TorchScript) and `policy_spec.json` for a runtime outside Eden.
See the [policy export guide](../guides/policy-export.md).

::: eden.learning.export
    options:
      show_root_heading: false

## rsl-rl-lib algorithms

`eden.learning.wrappers.rsl_rl` holds one subpackage per algorithm. Each has an `options` module (what a task
configures) next to the implementation.

::: eden.learning.wrappers.rsl_rl
    options:
      show_root_heading: false
      members:
        - MIN_RSL_RL_VERSION

### SimBaV2 actor-critic

::: eden.learning.wrappers.rsl_rl.simba.options
    options:
      show_root_heading: false

::: eden.learning.wrappers.rsl_rl.simba.model.SimBaModel
    options:
      show_bases: true
      inherited_members: false

### PPO with Muon

::: eden.learning.wrappers.rsl_rl.muon_ppo.options
    options:
      show_root_heading: false

::: eden.learning.wrappers.rsl_rl.muon_ppo.ppo.MuonPPO
    options:
      show_bases: true
      inherited_members: false

### PPO with an adversarial motion prior

::: eden.learning.wrappers.rsl_rl.amp_ppo.options
    options:
      show_root_heading: false

::: eden.learning.wrappers.rsl_rl.amp_ppo.ppo.AmpPPO
    options:
      show_bases: true
      inherited_members: false

::: eden.learning.wrappers.rsl_rl.amp_ppo.amp
    options:
      show_root_heading: false

### RMA teacher and depth-student distillation

::: eden.learning.wrappers.rsl_rl.rma.options
    options:
      show_root_heading: false

::: eden.learning.wrappers.rsl_rl.rma.model.RmaActorModel
    options:
      show_bases: true
      inherited_members: false

::: eden.learning.wrappers.rsl_rl.rma.ppo.RmaPPO
    options:
      show_bases: true
      inherited_members: false

::: eden.learning.wrappers.rsl_rl.rma.distillation
    options:
      show_root_heading: false

### HIMLoco

::: eden.learning.wrappers.rsl_rl.him.options
    options:
      show_root_heading: false

::: eden.learning.wrappers.rsl_rl.him.ppo.HIMPPO
    options:
      show_bases: true
      inherited_members: false

::: eden.learning.wrappers.rsl_rl.him.actor_critic.HIMActorCritic
    options:
      show_bases: true
      inherited_members: false

### FlashSAC

Off-policy SAC from the external `rsl-rl-flashsac` package, which is not on PyPI: install it with
`uv sync --inexact --extra flashsac --group flashsac-git` in a checkout.

::: eden.learning.wrappers.rsl_rl.flashsac.options
    options:
      show_root_heading: false

::: eden.learning.wrappers.rsl_rl.flashsac.runner
    options:
      show_root_heading: false

## ConTrack CMDP-PPO

Constrained-MDP PPO trained through the external `rsl_rl_contrack` fork of rsl_rl.

::: eden.learning.wrappers.rsl_rl_contrack.options
    options:
      show_root_heading: false

::: eden.learning.wrappers.rsl_rl_contrack.runner
    options:
      show_root_heading: false

::: eden.learning.wrappers.rsl_rl_contrack.env
    options:
      show_root_heading: false

## See also

- [Training runs guide](../guides/training-runs.md)
- [Learning options](options/learning.md)
- [Policy export guide](../guides/policy-export.md)
