# Termination Terms

Termination terms return a boolean per env; any true term ends the episode. A term configured with `time_out=True`
marks a truncation rather than a failure. The
[rewards and terminations guide](../../guides/rewards-and-terminations.md) explains both; the
[term catalog](catalog.md#termination-terms) lists every parameter.

```python
from eden import terminations
from eden.terms import DoneTerm

DoneTerm.configure(func=terminations.time_out, time_out=True)
```

The base classes `TerminationTerm` and `TerminationTermFuncWrapper` are on the
[managers page](../managers.md#terminationmanager).

## Common

Time-out, metric success and failure, illegal contact and limit violations.

::: eden.managers.terms.terminations.common
    options:
      show_root_heading: false

## Locomotion

::: eden.managers.terms.terminations.locomotion
    options:
      show_root_heading: false

## Velocity tracking

::: eden.managers.terms.terminations.velocity_tracking
    options:
      show_root_heading: false

## Manipulation

::: eden.managers.terms.terminations.manipulation
    options:
      show_root_heading: false

## Motion tracking

::: eden.managers.terms.terminations.tracking
    options:
      show_root_heading: false

## See also

- [Rewards and terminations guide](../../guides/rewards-and-terminations.md)
- [Termination manager options](../options/managers.md#terminations)
