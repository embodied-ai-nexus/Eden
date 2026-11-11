"""Termination manager for computing done signals."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import torch

from eden.managers.base import ConfigurableTermBase, ConfigurableTermFuncWrapper, FlatManagerBase
from eden.options.managers.terminations import (
    TerminationManagerOptions,
    TerminationTermOptions,
)
from eden.utils.common import ConfigurableMixin
from eden.utils.misc import sum_over_envs
from eden.utils.registry import TermRegistry

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


class _TerminationTermFields(ConfigurableMixin):
    """The configurable fields every termination term has, class or function."""

    # time_out=True marks this term as a TRUNCATION: it lands in `truncated`
    # (not `terminated`) and rsl_rl/rl_games bootstrap V-targets for it via
    # extras["time_outs"]. Declare a deadline term with time_out=False to have
    # it treated as a true terminal instead (no bootstrapping).
    time_out: bool = False


class TerminationTerm(ConfigurableTermBase[TerminationTermOptions], _TerminationTermFields):
    """Base class for termination terms."""


class TerminationTermFuncWrapper(ConfigurableTermFuncWrapper[TerminationTermOptions], _TerminationTermFields):
    """Base class for termination terms defined as a function."""


TerminationTermLike: TypeAlias = TerminationTerm | TerminationTermFuncWrapper
"""Union of a class-based termination term and its function-wrapper form."""


TERMINATION_TERM_REGISTRY = TermRegistry(
    "TERMINATION_TERM",
    TerminationTerm,
    TerminationTermFuncWrapper,
    # ``.metrics`` as well as ``.terminations``: seven terms there are dual-registered as
    # terminations (``object_lifted``, ``object_in_region``, ``entity_near_target``, ...) — a
    # success condition is the natural place to end an episode, so the term is written once and
    # registered into both kinds. ``tests/managers/test_catalog.py`` derives this from the catalog
    # rather than trusting the declaration.
    populated_by=("eden.managers.terms.terminations", "eden.managers.terms.metrics"),
)


#: ``Episode_Termination/`` key under which the envs Genesis flagged as corrupt are counted.
SOLVER_ERROR_KEY = "solver_error"


class TerminationManager(FlatManagerBase[TerminationManagerOptions]):
    _registry = TERMINATION_TERM_REGISTRY

    def __init__(self, env: EnvBase, options: TerminationManagerOptions):
        super().__init__(env=env, options=options)

        self._term_dones: dict[str, torch.Tensor] = {}
        for term_name in self._term_names:
            self._term_dones[term_name] = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._truncated_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._terminated_buf = torch.zeros_like(self._truncated_buf)
        self._solver_error_buf = torch.zeros_like(self._truncated_buf)

    def summary(self) -> str:
        return self._format_summary_table(
            title="Active Termination Terms",
            field_names=["Index", "Name", "Time Out"],
            rows=([index, name, term.time_out] for index, (name, term) in enumerate(self._terms.items())),
            align={"Name": "l"},
        )

    @property
    def dones(self) -> torch.Tensor:
        return self._truncated_buf | self._terminated_buf

    @property
    def timeouts(self) -> torch.Tensor:
        return self._truncated_buf

    @property
    def terminated(self) -> torch.Tensor:
        return self._terminated_buf

    @property
    def solver_errors(self) -> torch.Tensor:
        """Per-env mask of the envs Genesis flagged on the last :meth:`compute` (see
        ``EnvBase.get_solver_error_envs_mask``, which also says how a misconfiguration of the scene still raises).

        They are reset as *truncated* (:attr:`timeouts`), never terminated, and logged only as
        ``Episode_Termination/solver_error``: the simulator failed, not the task, so the learner bootstraps
        the step instead of reading it as a terminal outcome, and no term's verdict on the corrupt state
        counts.
        """
        return self._solver_error_buf

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None) -> dict[str, torch.Tensor]:
        if envs_idx is None:
            envs_idx = slice(None)
        extras = {}
        # Kept on the device: a count read back with ``.item()`` is a GPU->CPU sync per term on every
        # auto-reset step, and the loggers take a 0-dim tensor as readily as an int.
        for key, dones in self._term_dones.items():
            extras["Episode_Termination/" + key] = sum_over_envs(dones, envs_idx)
        extras["Episode_Termination/" + SOLVER_ERROR_KEY] = sum_over_envs(self._solver_error_buf, envs_idx)
        for term in self._reset_terms:
            term.reset(envs_idx=envs_idx)
        return extras

    def compute(self) -> torch.Tensor:
        self._truncated_buf[:] = False
        self._terminated_buf[:] = False
        # NOTE: reset corrupted state (call set_qpos to reset), per
        # https://github.com/Genesis-Embodied-AI/Genesis/issues/2239; truncated, not terminated, and without
        # any term's verdict (see `solver_errors`).
        self._solver_error_buf[:] = self._env.get_solver_error_envs_mask()
        healthy = ~self._solver_error_buf
        # One pass for `compute_once` (ManagerBase): terms share an intermediate within it.
        with self.compute_pass():
            for name, term in self._terms.items():
                dones = torch.logical_and(term.compute(), healthy, out=self._term_dones[name])  # written in place
                if term.time_out:
                    self._truncated_buf |= dones
                else:
                    self._terminated_buf |= dones
        self._truncated_buf |= self._solver_error_buf
        return self._truncated_buf | self._terminated_buf

    def get_term_dones(self, name: str) -> torch.Tensor:
        """Return the per-env done mask most recently computed for term ``name``.

        The term object itself is available via the inherited :meth:`get_term`.
        """
        return self._term_dones[name]

    def _prepare_terms(self):
        if SOLVER_ERROR_KEY in self._options.term_keys():
            raise ValueError(
                f"Termination term name {SOLVER_ERROR_KEY!r} is reserved: the manager logs the envs Genesis flags "
                f"as corrupt under 'Episode_Termination/{SOLVER_ERROR_KEY}'. Rename the term."
            )
        super()._prepare_terms()
