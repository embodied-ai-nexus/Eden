# Reward Terms

Reward terms return one value per env per step; the reward manager multiplies each by its `weight`, sums them and
scales the sum by the control step `dt`. The
[rewards and terminations guide](../../guides/rewards-and-terminations.md) explains weights, groups and logging; the [term catalog](catalog.md#reward-terms) lists every parameter.

```python
from eden import rewards
```

The base classes `RewardTerm` and `RewardTermFuncWrapper` are on the [managers page](../managers.md#rewardmanager).

## Common penalties

Action rate and magnitude, joint-limit, torque and contact penalties.

::: eden.managers.terms.rewards.common
    options:
      show_root_heading: false

## Locomotion

::: eden.managers.terms.rewards.locomotion
    options:
      show_root_heading: false

## Motion tracking

Rewards that score the robot against `env.motion_manager` reference data.

::: eden.managers.terms.rewards.tracking
    options:
      show_root_heading: false

## Contact matching

::: eden.managers.terms.rewards.contact
    options:
      show_root_heading: false

## See also

- [Rewards and terminations guide](../../guides/rewards-and-terminations.md)
- [Reward manager options](../options/managers.md#rewards)
