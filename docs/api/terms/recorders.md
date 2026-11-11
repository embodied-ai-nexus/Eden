# Recorder Terms

Recorder terms capture rollout data (actions, observations, RL transitions, entity states, sensor readings,
system-identification signals) for the recorder manager to write to an episode file. The
[metrics and recorders guide](../../guides/metrics-and-recorders.md) and the
[datasets and recording guide](../../guides/datasets-and-recording.md) explain the file layout; the
[term catalog](catalog.md#recorder-terms) lists every parameter.

```python
from eden import recorders
```

The base class `RecorderTerm` is on the [managers page](../managers.md#recordermanager).

## Actions

::: eden.managers.terms.recorders.actions
    options:
      show_root_heading: false

## Observations

::: eden.managers.terms.recorders.observations
    options:
      show_root_heading: false

## RL transitions

::: eden.managers.terms.recorders.rl
    options:
      show_root_heading: false

## Entity states

::: eden.managers.terms.recorders.states
    options:
      show_root_heading: false

## Sensors

::: eden.managers.terms.recorders.sensors
    options:
      show_root_heading: false

## System identification

::: eden.managers.terms.recorders.sysid
    options:
      show_root_heading: false

## See also

- [Recorder manager options](../options/managers.md#recorders)
- [File handlers](../utils/io.md#file-handlers)
