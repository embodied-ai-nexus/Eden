"""Recorder term capturing the observations each action was chosen from."""

from __future__ import annotations

import torch

from eden.managers.recorder_manager import RECORDER_TERM_REGISTRY, RecorderTerm


@RECORDER_TERM_REGISTRY.register()
class ObservationRecorder(RecorderTerm):
    """Record observation groups at the start of each step, as ``obs/<group>``.

    The start of a step is when the policy — or a scripted demonstrator — reads the observations
    and picks the action :class:`~eden.managers.terms.recorders.actions.ActionRecorder` records for
    that same step, so row ``t`` of ``obs`` and ``action`` form one ``(o_t, a_t)`` pair: the shape
    a behavior-cloning dataset needs.

    Parameters
    ----------
    groups : list[str] | None
        Observation groups to record; every group the env computes when None.
    """

    groups: list[str] | None = None

    def build(self) -> None:
        available = list(self._env.observation_manager.group_term_names)
        unknown = [group for group in self.groups or [] if group not in available]
        if unknown:
            raise KeyError(
                f"ObservationRecorder.groups names observation groups {unknown} that this env does not have; "
                f"available groups: {available}."
            )

    def record_pre_step(self) -> dict[str, torch.Tensor | dict]:
        obs = self._env.obs_buf
        if obs is None:
            raise RuntimeError(
                "ObservationRecorder recorded a step before the env computed any observations; reset the env "
                "(env.reset()) before stepping it."
            )
        groups = self.groups if self.groups is not None else list(obs)
        return {"obs": {group: _clone(obs[group]) for group in groups}}


def _clone(value: torch.Tensor | dict) -> torch.Tensor | dict:
    if isinstance(value, dict):
        return {key: _clone(sub) for key, sub in value.items()}
    return value.clone()
