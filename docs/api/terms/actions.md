# Action Terms

Action terms turn the policy's action vector into commands for an entity: joint-space PD and velocity controllers,
task-space controllers (differential IK, absolute-pose IK, operational-space control), floating-base pose control,
grippers and welds. The [actions guide](../../guides/actions.md) explains how to choose and configure them; the
[term catalog](catalog.md#action-terms) lists every parameter.

```python
from eden import actions

# excerpted from eden/tasks/benchmark/reacher/config.py
actions.ImplicitPDController.configure(entity_name="robot", dofs_name="joint*", scale=0.05)
```

The action-term base class `ActionTerm` is on the [managers page](../managers.md#actionmanager).

## Joint-space controllers

`ImplicitPDController` lets Genesis integrate the PD law (position mode); `ExplicitPDController` computes the
torque itself (`control_dofs_force`), so its damping is explicit under every integrator.

::: eden.managers.terms.actions.joint_actions
    options:
      show_root_heading: false

## Task-space controllers

::: eden.managers.terms.actions.task_space_actions
    options:
      show_root_heading: false

## Floating-base pose control

::: eden.managers.terms.actions.free_base_actions
    options:
      show_root_heading: false

## Grippers

::: eden.managers.terms.actions.binary_actions
    options:
      show_root_heading: false

## Welding

::: eden.managers.terms.actions.welding_actions
    options:
      show_root_heading: false

## See also

- [Actions guide](../../guides/actions.md)
- [Action modifiers](modifiers.md#action-modifiers)
- [Action manager options](../options/managers.md#actions)
