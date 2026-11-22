"""The per-env simulation parameters the world model is conditioned on, read from the task's randomization events."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.managers.terms.events.domain_rand import RandomizeComShift, RandomizeDofsPosOffset, RandomizeFrictionRatio
from eden.utils.string import term_subject

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase
    from eden.learning.wrappers.rsl_rl.supermimic.options import SimContextOptions

#: A range half-width at or below this is constant: dividing by it would amplify float rounding.
_DEGENERATE_HALF_WIDTH = 1e-3


class SimContext(torch.nn.Module):
    """Each env's foot friction, centre-of-mass offset and encoder bias, normalized to their randomization ranges.

    A value maps to ``(value - centre) / half_width`` of its range, so a context lies in ``[-1, 1]``. Columns:
    friction (one per link of the friction event, which draws each link's value on its own), centre-of-mass offset
    (3), encoder bias (one per joint), each present when its event is named.
    """

    def __init__(self, env: RLEnvBase, options: SimContextOptions, action_term: str):
        super().__init__()
        self._env = env
        self._friction = self._com = self._bias = None
        self._bias_columns: slice | None = None
        ranges: list[tuple[float, float]] = []
        if options.foot_friction_event is not None:
            term = self._event("foot_friction_event", options.foot_friction_event, RandomizeFrictionRatio)
            self._friction = (term.entity_name, term.ls_idx_local)
            ranges += [tuple(term.friction_range)] * len(term.ls_idx_local)
        if options.com_event is not None:
            term = self._event("com_event", options.com_event, RandomizeComShift)
            if len(term.ls_idx_local) != 1:
                raise ValueError(
                    f"sim_context reads one link's centre of mass, but {options.com_event!r} shifts "
                    f"{len(term.ls_idx_local)} links ({term.links_name})."
                )
            self._com = (term.entity_name, term.ls_idx_local)
            ranges += [tuple(term.com_shift_x_range), tuple(term.com_shift_y_range), tuple(term.com_shift_z_range)]
        if options.encoder_bias_event is not None:
            term = self._event("encoder_bias_event", options.encoder_bias_event, RandomizeDofsPosOffset)
            controller = env.action_manager.get_term(action_term)
            entity = env.entities[controller.entity_name]
            default = entity.get_default_dofs_pos(entity.find_named_dofs(list(controller.dofs_name)))
            self._bias = (controller, default)
            # The controller's offset is q0 - b for an encoder bias b, so the event draws -b in dofs_pos_range.
            lo, hi = term.dofs_pos_range
            self._bias_columns = slice(len(ranges), len(ranges) + len(controller.dofs_name))
            ranges += [(-hi, -lo)] * len(controller.dofs_name)
        if not ranges:
            raise ValueError("sim_context names no randomization event; name at least one in SimContextOptions.")
        centre = torch.tensor([(lo + hi) / 2 for lo, hi in ranges], device=env.device)
        half = torch.tensor([(hi - lo) / 2 for lo, hi in ranges], device=env.device)
        self.register_buffer("centre", centre)
        self.register_buffer("scale", torch.where(half > _DEGENERATE_HALF_WIDTH, half, torch.ones_like(half)))

    def _event(self, option: str, name: str, kind: type):
        events = self._env.event_manager
        if name not in events.active_terms:
            raise ValueError(
                f"runner_options.sim_context.{option}={name!r}, but the task has no such event term. Its event "
                f"terms: {list(events.active_terms)}. Name a {kind.__name__} event, or set the option to None."
            )
        term = events.get_term(name)
        if not isinstance(term, kind):
            raise TypeError(
                f"runner_options.sim_context.{option}={name!r} must name a {kind.__name__} event, but it is a "
                f"{type(term).__name__} ({term_subject(term)}). Name a {kind.__name__} event, or set the option to "
                "None."
            )
        return term

    @property
    def dim(self) -> int:
        return self.centre.numel()

    @property
    def has_encoder_bias(self) -> bool:
        """Whether the context carries an encoder bias (an ``encoder_bias_event`` is named)."""
        return self._bias_columns is not None

    def encoder_bias(self, sim_ctx: torch.Tensor) -> torch.Tensor | None:
        """Each joint's encoder bias (rad) in the normalized context ``sim_ctx``; ``None`` without a bias event.

        The simulator's joint reading is ``q - q0`` plus this bias, so the world model adds it to the reading it
        synthesizes.
        """
        if self._bias_columns is None:
            return None
        columns = self._bias_columns
        return sim_ctx[..., columns] * self.scale[columns] + self.centre[columns]

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
