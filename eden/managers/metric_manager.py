"""Metric manager and MetricTerm base for task success / evaluation signals."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import torch

from eden.constants import MetricDirection, MetricMode
from eden.managers.base import ConfigurableTermBase, ConfigurableTermFuncWrapper, FlatManagerBase
from eden.options.managers.metrics import MetricManagerOptions, MetricTermOptions
from eden.utils.common import ConfigurableMixin
from eden.utils.misc import assign_over_envs, sum_over_envs
from eden.utils.registry import TermRegistry

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


class SuccessCheckerMixin(ConfigurableMixin):
    """The configurable fields every metric term has, class or function, and the success check they drive.

    Parameters
    ----------
    success_threshold: float
        The threshold for success. The threshold is inclusive. Defaults to 1.0.
    direction: MetricDirection
        The metric direction. Defaults to ``MetricDirection.HIB`` (``"hib"``).
    metric_mode: MetricMode
        The metric success mode. Defaults to ``MetricMode.INTERVAL`` (``"interval"``).
    is_cumulative: bool
        Whether the metric is cumulative. Defaults to False.
    contributes_to_success: bool
        Whether this term participates in task-success accounting (``success_buf``,
        ``Metrics/Success``, ``Metrics/Progress``, and the per-episode success label the
        recorder stamps). Defaults to True. Set False for a term that exists only to share a
        computed quantity with other terms (via ``metric_manager.get_value`` — see
        ``metric_reward`` / ``metric_observation``) and has no notion of succeeding: such a
        term would otherwise pin ``Metrics/Success`` to its own trivially-true (or
        trivially-false) verdict for the whole task. ``is_success`` is never called on a
        non-contributing term, so its ``success_threshold`` / ``direction`` are unused.

    Notes
    -----
    These five settings are fixed once the metric manager is built: it reads them into its
    batched ``reset()`` then, so a later write to one (from a curriculum, say) would not take
    effect. Set them through ``configure``.
    """

    success_threshold: float = 1.0
    direction: MetricDirection = MetricDirection.HIB
    metric_mode: MetricMode = MetricMode.INTERVAL
    is_cumulative: bool = False
    contributes_to_success: bool = True

    if TYPE_CHECKING:
        _env: EnvBase

    def is_success(self, value: torch.Tensor, envs_idx: torch.Tensor | None = None) -> torch.Tensor:
        """Check whether a metric is successful (default implementation).

        This method can be overridden by subclasses to provide a custom implementation.

        Parameters
        ----------
        value: torch.Tensor
            The value of the metric.
        envs_idx: torch.Tensor, optional
            The indices of the environments to evaluate. If None, all environments are used. Defaults to None.

        Returns
        -------
        torch.Tensor
            A boolean tensor indicating whether the metric is successful.
        """
        if envs_idx is None:
            envs_idx = slice(None)
        if self.is_cumulative:
            value_norm = value[envs_idx] / torch.clamp_min(self._env.episode_length_buf[envs_idx], 1)
        else:
            value_norm = value[envs_idx]
        if self.direction == MetricDirection.HIB:
            return value_norm >= self.success_threshold
        elif self.direction == MetricDirection.LIB:
            return value_norm <= self.success_threshold
        else:
            raise ValueError(f"Invalid metric direction: {self.direction}")


class MetricTerm(ConfigurableTermBase[MetricTermOptions], SuccessCheckerMixin):
    """Base class for metric terms. The fields every metric term has are on :class:`SuccessCheckerMixin`."""


class MetricTermFuncWrapper(ConfigurableTermFuncWrapper[MetricTermOptions], SuccessCheckerMixin):
    """Base class for metric terms defined as function. See :class:`SuccessCheckerMixin` for the shared fields."""


MetricTermLike: TypeAlias = MetricTerm | MetricTermFuncWrapper
"""Union of a class-based metric term and its function-wrapper form."""


METRIC_TERM_REGISTRY = TermRegistry(
    "METRIC_TERM", MetricTerm, MetricTermFuncWrapper, populated_by=("eden.managers.terms.metrics",)
)


class MetricManager(FlatManagerBase[MetricManagerOptions]):
    """Manager for computing metrics.

    Every term's value lives in one ``(num_envs, num_terms)`` buffer (``get_value`` returns its
    column), so ``reset()``, which the auto-reset runs every step, handles all terms at once. The
    terms' ``success_threshold``, ``direction``, ``metric_mode``, ``is_cumulative`` and
    ``contributes_to_success`` are read at construction and fixed from then on: configure them,
    never write them at runtime.
    """

    _registry = METRIC_TERM_REGISTRY

    def __init__(self, env: EnvBase, options: MetricManagerOptions):
        super().__init__(env=env, options=options)
        # One (num_envs, num_terms) buffer, each term's `_step_metric` entry a column view of it (stable
        # identity, so `get_value` callers may still hold on to theirs). `reset()`, which the auto-reset
        # runs every step, then normalizes, logs, checks and clears every term at once rather than
        # issuing a dozen small ops per term.
        self._values = torch.zeros(self.num_envs, len(self._term_names), dtype=torch.float, device=self.device)
        self._step_metric = {name: self._values[:, i] for i, name in enumerate(self._term_names)}
        terms = [self._terms[name] for name in self._term_names]
        self._cumulative = torch.tensor([term.is_cumulative for term in terms], dtype=torch.bool, device=self.device)
        # The RESET-mode success checks. Terms keeping the default `is_success` are evaluated together
        # from their columns; one that overrides it keeps its own call.
        reset_success = [
            (i, term)
            for i, term in enumerate(terms)
            if term.metric_mode == MetricMode.RESET and term.contributes_to_success
        ]
        batched = [(i, term) for i, term in reset_success if type(term).is_success is SuccessCheckerMixin.is_success]
        self._reset_success_cols = torch.tensor([i for i, _ in batched], dtype=torch.long, device=self.device)
        self._reset_success_threshold = torch.tensor(
            [term.success_threshold for _, term in batched], dtype=torch.float, device=self.device
        )
        self._reset_success_hib = torch.tensor(
            [term.direction == MetricDirection.HIB for _, term in batched], dtype=torch.bool, device=self.device
        )
        batched_cols = {i for i, _ in batched}
        self._reset_success_custom = [(self._term_names[i], term) for i, term in reset_success if i not in batched_cols]

        self._success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._progress_buf = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        # False for an env Genesis flagged as corrupt on the last `compute()` (see there).
        self._healthy_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        # Each env's last finished episode, as `reset()` logged it: the per-env rows its "Metrics/*" means average.
        self._last_episode_keys = [f"Metrics/{name}" for name in self._term_names]
        if self._num_success_terms > 0:
            self._last_episode_keys += ["Metrics/Success", "Metrics/Progress"]
        self._last_episode = torch.zeros(
            self.num_envs, len(self._last_episode_keys), dtype=torch.float, device=self.device
        )
        self._last_episode_healthy = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def last_episode(self) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Each env's last finished episode: its ``Metrics/<name>`` values (and ``Metrics/Success`` /
        ``Metrics/Progress`` when a term contributes to success), and whether Genesis left the env healthy.

        ``reset()`` logs the mean of these rows over the envs that reset (the healthy ones); an evaluation that has to
        pick *which* episodes it counts (each env's first ``k``, say) reads the rows instead. An env that has not
        finished an episode yet reads zeros, unhealthy.

        Returns
        -------
        tuple[dict[str, torch.Tensor], torch.Tensor]
            ``(num_envs,)`` values per key, and the ``(num_envs,)`` bool healthy mask. The manager's own buffers:
            read them before the next reset.
        """
        values = {key: self._last_episode[:, i] for i, key in enumerate(self._last_episode_keys)}
        return values, self._last_episode_healthy

    def get_value(self, name: str) -> torch.Tensor:
        """Return the latest cached per-env value buffer for metric ``name``.

        The returned tensor is the manager-owned buffer updated in place every
        :meth:`compute` (stable identity), so callers — e.g. the ``metric_reward`` /
        ``metric_observation`` pull terms — may grab the reference once and reuse it.
        For a cumulative metric this is the running episode sum; otherwise the
        current-step value. ``metric_manager`` runs before the reward and observation
        managers in ``env.step()``, so the value reflects the current step. Treat it
        as read-only — mutating it corrupts the metric.

        Raises
        ------
        KeyError
            If ``name`` is not an active metric term.
        """
        if name not in self._step_metric:
            raise KeyError(f"Metric '{name}' not found. Active metrics: {list(self._step_metric)}")
        return self._step_metric[name]

    def is_cumulative(self, name: str) -> bool:
        """Whether metric ``name`` accumulates over the episode.

        When True, :meth:`get_value` returns the running episode sum (reset per-env on
        episode reset); when False, the current-step value. Pull terms use this to reject
        a cumulative metric where a per-step value is expected (see ``metric_reward``).

        Raises
        ------
        KeyError
            If ``name`` is not an active metric term.
        """
        if name not in self._terms:
            raise KeyError(f"Metric '{name}' not found. Active metrics: {list(self._terms)}")
        return self._terms[name].is_cumulative

    def is_current_step_success(self, name: str) -> torch.Tensor:
        """Per-env success mask for an INTERVAL metric evaluated on the current step's value.

        Thin public accessor over the metric's :meth:`is_success` check applied to the
        current-step value buffer — used by the ``on_metric_success`` termination so it
        need not reach into the manager's private term/value dicts.

        Raises
        ------
        KeyError
            If ``name`` is not an active metric term.
        ValueError
            If ``name`` is not an INTERVAL-mode metric (current-step success is undefined
            for cumulative/RESET-mode metrics).
        """
        if name not in self._terms:
            raise KeyError(f"Metric '{name}' not found. Active metrics: {list(self._terms)}")
        term = self._terms[name]
        if term.metric_mode != MetricMode.INTERVAL:
            raise ValueError(f"is_current_step_success requires an INTERVAL metric; '{name}' is {term.metric_mode}.")
        if not term.contributes_to_success:
            raise ValueError(
                f"metric '{name}' is declared contributes_to_success=False — it is a shared-quantity term "
                f"with no success notion, so its current-step success is undefined. Gate on a metric that "
                f"does define success."
            )
        return term.is_success(self._step_metric[name]) & self._healthy_buf

    @property
    def success_buf(self) -> torch.Tensor:
        return self._success_buf

    @property
    def num_step_success_terms(self) -> int:
        """How many INTERVAL metrics contribute to :attr:`success_buf` — with none, it never turns True."""
        return self._num_interval_terms

    @property
    def progress_buf(self) -> torch.Tensor:
        return self._progress_buf

    def summary(self) -> str:
        return self._format_summary_table(
            title="Metrics Terms",
            field_names=[
                "Index",
                "Name",
                "Success Threshold",
                "Direction",
                "Success Mode",
                "Cumulative",
                "Counts For Success",
            ],
            rows=(
                [
                    index,
                    name,
                    term.success_threshold,
                    term.direction,
                    term.metric_mode,
                    term.is_cumulative,
                    term.contributes_to_success,
                ]
                for index, (name, term) in enumerate(self._terms.items())
            ),
            align={"Name": "l"},
        )

    @torch.inference_mode()
    def reset(self, envs_idx: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if envs_idx is None:
            envs_idx = slice(None)
        extras = {}
        # Every read below is over all envs and reduced / applied through `envs_idx`, never a gather of
        # the selected rows: on the auto-reset path `envs_idx` is the bool `reset_buf`, and each gather
        # with it is a GPU->CPU sync — three per term per step.
        # A cumulative metric logs, and is judged on, its per-step mean over the episode.
        num_healthy = sum_over_envs(self._healthy_buf, envs_idx).clamp(min=1)
        episode_length = torch.clamp_min(self._env.episode_length_buf, 1).unsqueeze(1)
        normalized = torch.where(self._cumulative, self._values / episode_length, self._values)
        for name, mean in zip(self._term_names, self._mean_over_healthy(normalized, envs_idx, num_healthy).unbind()):
            extras[f"Metrics/{name}"] = mean
        # NOTE: check if the episode maintained the success status on the reset — the default check
        # (`SuccessCheckerMixin.is_success`) for every term keeping it at once, then any override.
        if self._reset_success_cols.numel():
            judged = normalized.index_select(1, self._reset_success_cols)
            threshold = self._reset_success_threshold
            succeeded = torch.where(self._reset_success_hib, judged >= threshold, judged <= threshold)
            count = succeeded.sum(dim=1, dtype=self._progress_buf.dtype)
            assign_over_envs(self._progress_buf, envs_idx, self._progress_buf + count)
        for name, term in self._reset_success_custom:
            is_success = term.is_success(self._step_metric[name]).to(self._progress_buf.dtype)
            assign_over_envs(self._progress_buf, envs_idx, self._progress_buf + is_success)
        self._values[envs_idx] = 0.0

        # NOTE: compute the success and progress metrics based on the maintenance and attainment terms.
        # Only success-contributing terms count, and with none of them there is no task success to
        # report: omit the keys rather than log a 0/0 that reads as a real 0% success rate.
        if self._num_success_terms > 0:
            success_val = (self._progress_buf >= self._num_success_terms).float()
            progress_val = self._progress_buf / self._num_success_terms
            extras["Metrics/Success"] = self._mean_over_healthy(success_val, envs_idx, num_healthy)
            extras["Metrics/Progress"] = self._mean_over_healthy(progress_val, envs_idx, num_healthy)
            normalized = torch.cat([normalized, success_val[:, None], progress_val[:, None]], dim=1)
        assign_over_envs(self._last_episode, envs_idx, normalized)
        assign_over_envs(self._last_episode_healthy, envs_idx, self._healthy_buf)
        self._progress_buf[envs_idx] = 0
        self._success_buf[envs_idx] = 0.0
        for terms in self._reset_terms:
            terms.reset(envs_idx=envs_idx)
        return extras

    @torch.inference_mode()
    def compute(self) -> dict[str, torch.Tensor]:
        if len(self._terms) == 0:
            return self._step_metric
        self._progress_buf.zero_()
        self._success_buf.zero_()
        # An env Genesis flagged as corrupt read garbage this step, and is reset in it. Its value is 0 so
        # no NaN reaches a buffer or a cumulative sum, but 0 is the *best* value of a lower-is-better metric: it
        # is kept out of every success check here, in `is_current_step_success` and in `reset()`'s logs.
        torch.logical_not(self._env.get_solver_error_envs_mask(), out=self._healthy_buf)
        # One pass for `compute_once` (ManagerBase): terms share an intermediate within it.
        with self.compute_pass():
            for name, term in self._terms.items():
                value = torch.where(self._healthy_buf, term.compute().float(), 0.0)
                if term.is_cumulative:
                    self._step_metric[name] += value
                else:
                    self._step_metric[name][:] = value

                if term.metric_mode == MetricMode.INTERVAL and term.contributes_to_success:
                    is_success = term.is_success(self._step_metric[name]) & self._healthy_buf
                    self._progress_buf += is_success.long()
        if self._num_interval_terms > 0:
            self._success_buf[:] = self._progress_buf >= self._num_interval_terms
        return self._step_metric

    def _mean_over_healthy(
        self, values: torch.Tensor, envs_idx: slice | torch.Tensor, num_healthy: torch.Tensor
    ) -> torch.Tensor:
        """:func:`mean_over_envs` over the selected envs Genesis did not flag; 0 when it flagged them all.

        Sync-free for any ``envs_idx``: one reduction over all envs.
        """
        kept = torch.where(self._healthy_buf.view(-1, *([1] * (values.ndim - 1))), values, 0)
        return sum_over_envs(kept, envs_idx) / num_healthy

    def _prepare_terms(self):
        # First pass: build all terms normally
        super()._prepare_terms()
        success_terms = [t for t in self._terms.values() if t.contributes_to_success]
        self._num_success_terms = len(success_terms)
        self._num_interval_terms = sum(t.metric_mode == MetricMode.INTERVAL for t in success_terms)

        # Second pass: resolve refs for terms that need it (e.g. SequentialMetricTerm)
        for term in self._terms.values():
            if hasattr(term, "resolve_refs"):
                term.resolve_refs(self._options)
