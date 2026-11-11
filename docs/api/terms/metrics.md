# Metric Terms

Metric terms grade a rollout: each reports a value per env with a direction (`MetricDirection`, higher or lower is
better), a mode (`MetricMode`) and a `success_threshold`, and the metric manager combines the ones that contribute
to success. Rewards and terminations can read a metric instead of recomputing it. The
[metrics and recorders guide](../../guides/metrics-and-recorders.md) explains them; the
[term catalog](catalog.md#metric-terms) lists every parameter.

```python
from eden import metrics
from eden.constants import MetricDirection, MetricMode
```

The base classes `MetricTerm` and `MetricTermFuncWrapper` are on the [managers page](../managers.md#metricmanager).

## Proximity

::: eden.managers.terms.metrics.proximity
    options:
      show_root_heading: false

## Containment

::: eden.managers.terms.metrics.containment
    options:
      show_root_heading: false

## Grasping

::: eden.managers.terms.metrics.grasp
    options:
      show_root_heading: false

## Articulation

::: eden.managers.terms.metrics.articulation
    options:
      show_root_heading: false

## Velocity-tracking error

::: eden.managers.terms.metrics.error
    options:
      show_root_heading: false

## Boolean combinators

::: eden.managers.terms.metrics.boolean
    options:
      show_root_heading: false

## Sequences and rubrics

::: eden.managers.terms.metrics.sequence
    options:
      show_root_heading: false

## See also

- [Metrics and recorders guide](../../guides/metrics-and-recorders.md)
- [Metric manager options](../options/managers.md#metrics)
