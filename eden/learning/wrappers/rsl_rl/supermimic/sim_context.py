"""The per-env simulation parameters the world model is conditioned on, read from the task's randomization events."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.utils.string import term_subject

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase
    from eden.learning.wrappers.rsl_rl.supermimic.options import SimContextOptions

#: A range half-width at or below this is constant: dividing by it would amplify float rounding.
_DEGENERATE_HALF_WIDTH = 1e-3


class SimContext(torch.nn.Module):
    """Each env's foot friction, centre-of-mass offset and encoder bias, normalized to their randomization ranges.

    A value maps to ``(value - centre) / half_width`` of its range, so a context lies in ``[-1, 1]``. Columns:
    friction (1), centre-of-mass offset (3), encoder bias (one per joint), each present when its event is named.
    """

    def __init__(self, env: RLEnvBase, options: SimContextOptions, action_term: str):
        super().__init__()
        self._env = env
        self._friction = self._com = self._bias = None
        ranges: list[tuple[float, float]] = []
        if options.foot_friction_event is not None:
            term = self._event(options.foot_friction_event, "RandomizeFrictionRatio")
            self._friction = (term.entity_name, term.ls_idx_local[:1])
            ranges.append(tuple(term.friction_range))
        if options.com_event is not None:
            term = self._event(options.com_event, "RandomizeComShift")
            if len(term.ls_idx_local) != 1:
                raise ValueError(
                    f"sim_context reads one link's centre of mass, but {options.com_event!r} shifts "
                    f"{len(term.ls_idx_local)} links ({term.links_name})."
                )
            self._com = (term.entity_name, term.ls_idx_local)
            ranges += [tuple(term.com_shift_x_range), tuple(term.com_shift_y_range), tuple(term.com_shift_z_range)]
        if options.encoder_bias_event is not None:
            term = self._event(options.encoder_bias_event, "RandomizeDofsPosOffset")
            controller = env.action_manager.get_term(action_term)
            entity = env.entities[controller.entity_name]
            default = entity.get_default_dofs_pos(entity.find_named_dofs(list(controller.dofs_name)))
            self._bias = (controller, default)
            # The controller's offset is q0 - b for an encoder bias b, so the event draws -b in dofs_pos_range.
            lo, hi = term.dofs_pos_range
            ranges += [(-hi, -lo)] * len(controller.dofs_name)
        if not ranges:
            raise ValueError("sim_context names no randomization event; name at least one in SimContextOptions.")
        centre = torch.tensor([(lo + hi) / 2 for lo, hi in ranges], device=env.device)
        half = torch.tensor([(hi - lo) / 2 for lo, hi in ranges], device=env.device)
        self.register_buffer("centre", centre)
        self.register_buffer("scale", torch.where(half > _DEGENERATE_HALF_WIDTH, half, torch.ones_like(half)))

    def _event(self, name: str, kind: str):
        term = self._env.event_manager.get_term(name)
        if type(term).__name__ != kind:
            raise ValueError(
                f"sim_context reads event {name!r} as a {kind}, but it is a {type(term).__name__} "
                f"({term_subject(term)}). Name a {kind} event, or set the option to None."
            )
        return term

    @property
    def dim(self) -> int:
        return self.centre.numel()

    def read(self) -> torch.Tensor:
        """``(num_envs, dim)`` normalized context of every env."""
        entities, parts = self._env.entities, []
        if self._friction is not None:
            name, links = self._friction
            parts.append(entities[name].get_friction_ratio(ls_idx_local=links))
        if self._com is not None:
            name, link = self._com
            parts.append(entities[name].get_COM_shift(ls_idx_local=link)[:, 0])
        if self._bias is not None:
            controller, default = self._bias
            parts.append((default - controller.effective_offset).expand(self._env.num_envs, -1))
        return (torch.cat(parts, dim=-1) - self.centre) / self.scale
