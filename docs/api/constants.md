# Constants and Types

Shared enums and type aliases. The enums configure terms: `EventMode` is an event's `mode`, `MetricDirection` and
`MetricMode` describe a metric, `ReferenceSource` selects what an action term's target is relative to,
`NoiseOperation` says whether a noise model adds to or scales a value, and `DatasetExportMode` chooses which recorded
episodes the recorder manager writes.

```python
from eden.constants import EventMode, MetricDirection, MetricMode
```

## Enums

::: eden.constants
    options:
      show_root_heading: false

## Type aliases

::: eden.types
    options:
      show_root_heading: false

## See also

- [Event terms](terms/events.md), [metric terms](terms/metrics.md), [modifiers](terms/modifiers.md)
