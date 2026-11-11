"""Base classes and registries for manager term modifiers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.constants import NoiseOperation
from eden.managers.base import ConfigurableTermBase
from eden.options.managers.modifiers import ActionModifierOptions, NoiseOptions, ObservationModifierOptions
from eden.utils.registry import TermRegistry

if TYPE_CHECKING:
    from eden.entities.base import Entity


class NoiseModel(ConfigurableTermBase[NoiseOptions]):
    """Base class for noise models applied to observation terms."""

    operation: NoiseOperation = NoiseOperation.ADD


# TermRegistry (not plain Registry) so NoiseOptions gets the registry stamp for
# construction-time parameter-name validation. Noise models are class-based only.
NOISE_MODEL_REGISTRY = TermRegistry("NOISE_MODEL", NoiseModel, populated_by=("eden.managers.modifiers.observations",))


class ActionModifier(ConfigurableTermBase[ActionModifierOptions]):
    """Base class for composable action modifiers.

    Action modifiers intercept and transform processed actions (target
    positions/velocities) and/or control torques. They are designed to be
    composed together using :class:`~eden.managers.modifiers.actions.actuators.Compose`.

    Subclasses override any of three hooks, which run at two different rates:

    - ``modify_processed_action``: once per **control step**, from the controller's
      ``compute()``. Its output is ``processed_action`` — the command a deployment sends.
    - ``modify_applied_target``: once per **physics step** (``sim_dt``), from the controller's
      ``apply_actions()``, on the command the actuator acts on (e.g. a delay).
    - ``modify_ctrl_torque``: once per physics step, on the torque an explicit controller computed.
    """

    def build(
        self,
        num_envs: int,
        device: str,
        entity: "Entity" | None = None,
        dofs_idx_local: torch.Tensor | None = None,
    ) -> None:
        """Initialize internal buffers, called by the owning action term after the entity is resolved.

        Parameters
        ----------
        num_envs : int
            Number of parallel environments.
        device : str
            Torch device string.
        entity : Entity, optional
            The entity being controlled. Needed by state-dependent modifiers.
        dofs_idx_local : torch.Tensor, optional
            Local DOF indices being controlled.
        """
        pass

    def compute(self) -> None:
        """No-op by default. Override if the modifier needs per-step computation."""
        pass

    def modify_processed_action(self, processed_action: torch.Tensor) -> torch.Tensor:
        """Transform target positions/velocities before PD control. Default is identity.

        Runs once per control step, from the controller's ``compute()``; what it returns becomes
        ``processed_action``.
        """
        return processed_action

    def modify_applied_target(self, target: torch.Tensor) -> torch.Tensor:
        """Transform the command the actuator acts on during this physics step. Default is identity.

        Runs once per physics step (``sim_dt``) — ``decimation`` times per control step — from the
        controller's ``apply_actions()``, on the controller's whole command (``processed_action``,
        both halves of a position+velocity one), after every ``modify_processed_action`` and before
        the control law. A stateful override may count on exactly one call per physics step.

        ``processed_action`` itself is left alone, and a deployment sends that and never calls
        ``apply_actions()``, so what this hook does stays in simulation — the right place for a
        model of the robot's own latency, which the hardware already has.
        """
        return target

    def modify_ctrl_torque(
        self,
        ctrl_torque: torch.Tensor,
        dofs_vel: torch.Tensor,
        pos_err: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Transform computed torques after PD control. Default is identity.

        ``pos_err`` (target - current joint position) is supplied by PD-position
        controllers for modifiers that need it (e.g. position deadbands); it is
        ``None`` for velocity-only controllers.
        """
        return ctrl_torque

    def get(self, cls: type) -> ActionModifier | None:
        """Return self if it is an instance of ``cls``, else ``None``.

        :class:`~eden.managers.modifiers.actions.actuators.Compose` overrides this
        to search its children. Callers that require the modifier to exist should
        check for ``None`` and raise their own actionable error.
        """
        return self if isinstance(self, cls) else None


# TermRegistry for the same reason as NOISE_MODEL_REGISTRY above.
ACTION_MODIFIER_REGISTRY = TermRegistry(
    "ACTION_MODIFIER", ActionModifier, populated_by=("eden.managers.modifiers.actions",)
)


class ObservationModifier(ConfigurableTermBase[ObservationModifierOptions]):
    """Base class for a transform the observation manager applies to one term's output, frame by frame.

    Listed on an observation term as ``modifiers=[...]``. Each frame of the term goes through its
    modifiers in list order, then noise, clip and scale, then the group's ``nan_policy``, and last the
    history window. Unlike noise, a modifier runs in every mode (training, eval, deployment). Each term
    instance owns its modifier instances, so the same modifier on a ``policy`` and a ``critic`` copy of
    a term keeps two independent states.

    **``compute(frame)`` is a read, the same rule as for an observation term.** The manager calls it
    every time the observation is computed, which can be several times per control step: from
    ``EnvBase.reset()``, at phase 19, at phase 12 for the final-observation snapshot, from
    ``get_observations()``, and once at build to check the output. It returns the output for
    ``frame`` given the modifier's current state and changes nothing.

    **``advance(frame)`` is the commit.** A stateful modifier (a filter, a delay) folds ``frame`` into
    its state here. The manager calls it right after ``compute(frame)``, with the same frame, on the
    pass that also pushes the history windows (``compute(update_history=True)``). That pass runs once
    per control step (phase 19 of ``RLEnvBase.step``, ``EnvBase.step``, each deploy tick) and once per
    ``EnvBase.reset()``. A partial ``env.reset(envs_idx)`` therefore advances every env, the same way
    it pushes every env's history window. So a modifier's state moves together with the history window,
    and a delay of ``k`` is ``k`` pushes. A read that comes *after* the push and before the next step
    (``get_observations()`` between steps) applies the current frame once more, as the history
    window's read does. With backfill, right after a reset, that gives the same value; otherwise it
    runs one push ahead. A stateless modifier leaves ``advance`` as the no-op it is here.

    Two rules keep reads and commits apart:

    - ``compute`` returns a tensor it does not write afterwards. The manager uses the output after
      calling ``advance``, so a view of state that ``advance`` updates in place would change under it.
      Return a fresh tensor, or ``.clone()`` the state.
    - Neither method writes into ``frame``, and ``advance`` copies what it keeps
      (``self._last.copy_(frame)``, not ``self._last = frame``). ``frame`` may be the raw output that
      another group's copy of the term also reads (the manager's cross-group cache), the term's own
      buffer, or a view of simulator state.

    The other hooks: ``build(frame)`` allocates the state from a sample frame, before the env steps
    (allocate here, then write in place: see "Term state lifecycle" in ``docs/guides/terms.md``).
    ``reset(envs_idx)`` restarts the rows of the envs whose episode restarts. It is called with the
    same indices as the term's history window: ``None`` for every env, otherwise an index or
    boolean-mask tensor to index with.

    A modifier returns a frame of the shape, dtype and device it received. The manager checks this at
    build, because ``scale``, history and concatenation are all sized from the term's raw output.
    """

    def build(self, frame: torch.Tensor) -> None:
        """Allocate per-env state sized like ``frame``. No-op by default.

        Parameters
        ----------
        frame : torch.Tensor
            A sample of the frames this modifier will receive, ``(num_envs, ...)``: the term's output
            from the manager's build-time shape probe, passed through the modifiers listed before this
            one. Size state from it (``torch.zeros_like(frame)`` gets shape, dtype and device right).
            Do not seed state from its values: they are read from the scene before any reset.
        """

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        """Restart the state of the envs in ``envs_idx`` (``None``: every env). No-op by default."""

    def compute(self, frame: torch.Tensor) -> torch.Tensor:
        """Return this frame's output given the current state, without changing the state or ``frame``."""
        raise NotImplementedError(f"{type(self).__name__} must implement compute(frame).")

    def advance(self, frame: torch.Tensor) -> None:
        """Fold ``frame`` into the state: called once per control step, right after ``compute(frame)``.

        No-op by default, which is right for a stateless modifier.
        """


# TermRegistry for the same reason as NOISE_MODEL_REGISTRY above.
OBSERVATION_MODIFIER_REGISTRY = TermRegistry(
    "OBSERVATION_MODIFIER", ObservationModifier, populated_by=("eden.managers.modifiers.observations",)
)
