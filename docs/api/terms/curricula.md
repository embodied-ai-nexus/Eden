# Curriculum Terms

Curriculum terms adapt the task while it trains: they widen command ranges, ramp reward weights and tighten
tolerances, usually keyed off average episode length or success. The curriculum manager computes them only in
training; at eval the curricula are frozen. The [curricula guide](../../guides/curricula.md) covers sharing state
between terms; the [term catalog](catalog.md#curriculum-terms) lists every parameter.

```python
from eden import curricula
```

The base classes `CurriculumTerm` and `CurriculumTermFuncWrapper` are on the
[managers page](../managers.md#curriculummanager).

## Episode-length tracking

::: eden.managers.terms.curricula.locomotion
    options:
      show_root_heading: false

## Reward weights

::: eden.managers.terms.curricula.rewards
    options:
      show_root_heading: false

## Velocity commands

::: eden.managers.terms.curricula.velocity
    options:
      show_root_heading: false

## Motion tracking

::: eden.managers.terms.curricula.motion
    options:
      show_root_heading: false

## Success Guided Sampling

::: eden.managers.terms.curricula.success_guided
    options:
      show_root_heading: false

## See also

- [Curricula guide](../../guides/curricula.md)
- [Curriculum manager options](../options/managers.md#curricula)
