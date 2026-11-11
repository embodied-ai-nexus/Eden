"""Recorder terms capturing initial and per-step entity states."""

from __future__ import annotations

import torch

from eden.managers import RECORDER_TERM_REGISTRY, RecorderTerm


@RECORDER_TERM_REGISTRY.register()
class InitialStateRecorder(RecorderTerm):
    """Record per-episode initial entity states right after reset."""

    include_qpos: bool = True
    include_dofs_vel: bool = True
    include_pos_quat: bool = True

    def record_post_reset(self, envs_idx=None) -> dict[str, torch.Tensor | dict]:
        if not self._env.entities:
            return {}

        entities_state: dict[str, dict[str, torch.Tensor]] = {}
        for entity_name, entity in self._env.entities.items():
            entity_state: dict[str, torch.Tensor] = {}

            if self.include_qpos:
                entity_state["qpos"] = entity.get_qpos(qs_idx_local=entity.qs_idx_local, envs_idx=envs_idx).clone()
            if self.include_dofs_vel:
                entity_state["dofs_vel"] = entity.get_dofs_vel(envs_idx=envs_idx).clone()
            if self.include_pos_quat:
                entity_state["pos"] = entity.get_pos(envs_idx=envs_idx).clone()
                entity_state["quat"] = entity.get_quat(envs_idx=envs_idx).clone()

            if entity_state:
                entities_state[entity_name] = entity_state

        if not entities_state:
            return {}

        return {"initial_state": {"entities": entities_state}}


@RECORDER_TERM_REGISTRY.register()
class StateRecorder(RecorderTerm):
    """Record the state of the environment at the beginning of each step.

    ``entity_name`` is one entity, or several — the two arms of a two-arm task — whose states are
    concatenated in that order along the last axis of each key (``qpos``, ``dofs_vel``, ``pos``,
    ``quat``), so a dataset keeps one ``state/<key>`` per key whatever the number of arms.
    """

    entity_name: str | list[str] = "robot"
    include_qpos: bool = True
    include_dofs_vel: bool = True
    include_pos_quat: bool = True

    def build(self) -> None:
        names = [self.entity_name] if isinstance(self.entity_name, str) else list(self.entity_name)
        if not names:
            raise ValueError("StateRecorder.entity_name lists no entity; name the entity (or entities) to record.")
        for name in names:
            self.resolve_entity(name)
        self._entity_names = names

    def record_pre_step(self) -> dict[str, torch.Tensor | dict]:
        states = [self._entity_state(self.resolve_entity(name)) for name in self._entity_names]
        return {"state": {key: torch.cat([state[key] for state in states], dim=-1) for key in states[0]}}

    def _entity_state(self, entity) -> dict[str, torch.Tensor]:
        entity_state: dict[str, torch.Tensor] = {}
        if self.include_qpos:
            entity_state["qpos"] = entity.get_qpos(qs_idx_local=entity.qs_idx_local).clone()
        if self.include_dofs_vel:
            entity_state["dofs_vel"] = entity.get_dofs_vel().clone()
        if self.include_pos_quat:
            entity_state["pos"] = entity.get_pos().clone()
            entity_state["quat"] = entity.get_quat().clone()
        return entity_state
