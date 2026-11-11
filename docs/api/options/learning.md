# Learning Options

Typed options for the RL runners. A task's `runner_options` holds one; Eden translates it into the configuration
dict the installed library version expects, so a task on rsl_rl's own algorithms trains on every rsl-rl-lib version
the pin admits (`>=3.0`). Some of Eden's own algorithms need newer: AMP-PPO and RMA need 5.5.1
(`MIN_RSL_RL_VERSION` in `eden.learning.wrappers.rsl_rl`), and the `flashsac` extra 5.4.2, while HIM adapts itself to
3.x (see
[Across rsl-rl-lib versions](../../guides/training-runs.md#custom-algorithms-algorithmclass_name)).
The [training runs guide](../../guides/training-runs.md) explains runners, checkpoints and resuming; the algorithms
Eden adds on top of a library are on the [learning page](../learning.md).

```python
from eden.options.learning.rsl_rl.rl import RslRlOnPolicyRunnerOptions, RslRlPpoAlgorithmOptions
```

## rsl_rl

### Models, distributions and PPO

::: eden.options.learning.rsl_rl.rl
    options:
      show_root_heading: false

### Distillation

::: eden.options.learning.rsl_rl.distillation
    options:
      show_root_heading: false

### Random network distillation

::: eden.options.learning.rsl_rl.rnd
    options:
      show_root_heading: false

### Symmetry augmentation

::: eden.options.learning.rsl_rl.symmetry
    options:
      show_root_heading: false

### Spatial-softmax CNN

::: eden.options.learning.rsl_rl.spatial_softmax
    options:
      show_root_heading: false

### Version translation

Helpers that detect the installed rsl-rl-lib version and translate the typed options into the runner dict it
expects.

::: eden.options.learning.rsl_rl.translate
    options:
      show_root_heading: false

## rl_games

::: eden.options.learning.rl_games.rl
    options:
      show_root_heading: false

::: eden.options.learning.rl_games.translate
    options:
      show_root_heading: false

## Action clipping

The `clip_actions` bound that a runner's options and every env wrapper accept.

::: eden.options.learning.clip_actions
    options:
      show_root_heading: false

## See also

- [Training runs guide](../../guides/training-runs.md)
- [Learning: algorithms and export](../learning.md)
- [Other RL libraries](../../guides/rl-libraries.md)
