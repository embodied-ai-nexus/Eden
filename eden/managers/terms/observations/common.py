"""Common observation terms (commands, last action, episode phase)."""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY, ObservationTerm
from eden.managers.terms.utils import resolve_metric_manager

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase
    from eden.managers.action_manager import ActionTerm


@OBSERVATION_TERM_REGISTRY.register()
def generated_commands(
    env: RLEnvBase,
    *,
    command_name: str,
) -> torch.Tensor:
    return env.command_manager.get_command(command_name)


@OBSERVATION_TERM_REGISTRY.register()
def last_action(
    env: RLEnvBase,
    *,
    action_name: str | None = None,
) -> torch.Tensor:
    if action_name is not None:
        action_term: ActionTerm = env.action_manager.get_term(action_name)
        return action_term.action
    return env.action_manager.action


@OBSERVATION_TERM_REGISTRY.register()
def episode_phase(
    env: RLEnvBase,
    *,
    centered: bool = False,
) -> torch.Tensor:
    """Return normalized episode time as a ``(num_envs, 1)`` tensor.

    ``centered=False`` (default): ``t / max_episode_length`` in ``[0, 1]``.
    ``centered=True``: ``2 * t / max_episode_length - 1`` in ``[-1, 1]``.
    """
    phase = env.episode_length_buf.float() / env.max_episode_length
    if centered:
        phase = 2.0 * phase - 1.0
    return phase.unsqueeze(-1)


@OBSERVATION_TERM_REGISTRY.register()
class MetricObservation(ObservationTerm):
    """Expose a :class:`~eden.managers.metric_manager.MetricTerm`'s cached value as a ``(num_envs, 1)`` observation.

    Pulls the per-env value the metric manager computed this step (``metric_manager`` runs
    before the observation manager in ``env.step()``). Same rationale as ``metric_reward``:
    compute a shared quantity once as a metric and read it here by name rather than
    recomputing it or stashing it on ``env``. The standard obs modifiers (scale/clip/noise/
    history) still apply. Observations are computed after env resets, so an environment that
    reset this step reads the metric's post-reset value (0 until its next compute),
    consistent with Eden's post-reset observation convention.

    A missing metric manager or an unset or unknown ``metric_name`` (the error lists the active
    metrics) raises in :meth:`build`, i.e. at env construction. :meth:`compute` is a pure read of
    the manager-owned buffer, so the extra calls an observation term receives (build-time shape
    probe, ``reset()``, the final-observation snapshot) are harmless.
    """

    # Was the function term ``metric_observation(env, *, metric_name)``; a config saved then carries it in
    # ``params``, which load as fields.
    _function_era_params = frozenset({"metric_name"})

    metric_name: str = ""

    def build(self) -> None:
        resolve_metric_manager(self._env, term="metric_observation", metric_name=self.metric_name)

    def compute(self) -> torch.Tensor:
        return self._env.metric_manager.get_value(self.metric_name).unsqueeze(-1)
