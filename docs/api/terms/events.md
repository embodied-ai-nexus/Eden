# Event Terms

Event terms change the simulation outside the policy's control: they reset state, place objects and randomize
dynamics. Each runs in one mode from `EventMode` (`STARTUP`, `RESET` or `INTERVAL`, see [constants](../constants.md)).
The [events guide](../../guides/events.md) explains modes, ordering and what each randomizer needs from the scene;
the [term catalog](catalog.md#event-terms) lists every parameter.

```python
from eden import events
from eden.constants import EventMode
```

The base classes `EventTerm` and `EventTermFuncWrapper` are on the [managers page](../managers.md#eventmanager).

## Resets

::: eden.managers.terms.events.resets
    options:
      show_root_heading: false

## Domain randomization

Pushes, external wrenches, DOF position and offset randomization, actuator randomization (motor strength, gains,
backlash, deadband, torque noise) and mass, CoM and friction randomization.

::: eden.managers.terms.events.domain_rand
    options:
      show_root_heading: false

## Placement

Place objects on a support entity's surfaces, with constant, range, uniform or Gaussian samplers, or from a fixed
list of layouts that place several entities together without overlap (`PlaceFromLayouts`).

::: eden.managers.terms.events.placement
    options:
      show_root_heading: false

## Restoring saved states

::: eden.managers.terms.events.state_buffer
    options:
      show_root_heading: false

## Contact solver parameters

::: eden.managers.terms.events.solver
    options:
      show_root_heading: false

## Motion-reference initialization

`RESET` events that start each episode from a state of the motion reference (reference state initialization).

::: eden.managers.terms.events.motion
    options:
      show_root_heading: false

## See also

- [Events guide](../../guides/events.md)
- [Event manager options](../options/managers.md#events)
