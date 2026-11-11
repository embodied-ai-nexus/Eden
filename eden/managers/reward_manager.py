"""Reward manager for computing reward signals."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, TypeAlias

import torch

import eden as en
from eden.managers.base import ConfigurableTermBase, ConfigurableTermFuncWrapper, FlatManagerBase
from eden.options.managers.rewards import RewardGroupOptions, RewardManagerOptions, RewardTermOptions
from eden.utils.common import ConfigurableMixin
from eden.utils.misc import mean_over_envs
from eden.utils.registry import TermRegistry

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase


#: Group that flat (ungrouped) reward terms belong to. Present in
#: :attr:`RewardManager.group_term_names` only when at least one flat term is
#: declared — fully-grouped configs have no ``"default"`` entry, so generic
#: consumers should iterate ``group_term_names`` rather than key on this constant.
DEFAULT_REWARD_GROUP = "default"


class _RewardTermFields(ConfigurableMixin):
    """The configurable fields every reward term has, class or function."""

    range_s: list[tuple[float, float]] | None = None
    weight: float = 1.0
    tags: list[str] = []

    @property
    def is_temporal_dependent(self) -> bool:
        """Whether the term is active only within ``range_s``."""
        return self.range_s is not None


class RewardTerm(ConfigurableTermBase[RewardTermOptions], _RewardTermFields):
    """Base class for reward terms."""


class RewardTermFuncWrapper(ConfigurableTermFuncWrapper[RewardTermOptions], _RewardTermFields):
    """Base class for reward terms defined as a function."""


RewardTermLike: TypeAlias = RewardTerm | RewardTermFuncWrapper
"""Union of a class-based reward term and its function-wrapper form."""


REWARD_TERM_REGISTRY = TermRegistry(
    "REWARD_TERM", RewardTerm, RewardTermFuncWrapper, populated_by=("eden.managers.terms.rewards",)
)


class RewardManager(FlatManagerBase[RewardManagerOptions]):
    """Computes the scalar step reward as the weighted sum of all active terms.

    Terms are declared flat (:class:`RewardTermOptions`) or inside named groups
    (:class:`RewardGroupOptions`); flat terms implicitly belong to
    :data:`DEFAULT_REWARD_GROUP`. Grouping never changes the scalar reward the
    environment returns — :meth:`compute` always sums over all groups — it adds a
    per-group readout (:meth:`get_group_reward`, :meth:`get_group_episode_sum`)
    for multi-channel consumers (CMDP-style task/style/penalty critics,
    per-group logging). Groups partition the active terms, so summing
    :meth:`get_group_reward` over :attr:`group_term_names` reconstructs the
    value :meth:`compute` returned for the same step (up to floating-point
    summation order).

    Unlike observation groups (which may share a term across groups), reward
    groups are a strict partition of one flat term set — duplicate term names
    across groups raise at construction. That is why this stays a
    :class:`FlatManagerBase`: the terms, buffers, and the flat API
    (``terms``, ``active_terms``, ``get_term``, ``iter_terms``,
    ``get_episode_sum``) are unchanged; groups are a labeling on top.
    """

    _registry = REWARD_TERM_REGISTRY

    #: The weights live on the device (``_weights``), mirrored from each term's ``weight`` by
    #: :meth:`sync_device_fields`, so a curriculum moving one needs no re-capture of a graded-phase CUDA graph.
    device_synced_fields: ClassVar[frozenset[str]] = frozenset({"weight"})

    def __init__(self, env: RLEnvBase, options: RewardManagerOptions):
        super().__init__(env=env, options=options)
        num_terms = len(self._term_names)

        # Whether reset() emits the per-term reward log keys. None auto-selects:
        # off when explicit groups give a group-sum readout, on otherwise (a flat config has
        # nothing to summarize to). _has_explicit_groups is set by _prepare_terms above.
        log_pref = self._options.log_per_term_rewards
        self._log_per_term = (not self._has_explicit_groups) if log_pref is None else bool(log_pref)
        if not self._log_per_term and log_pref is None:
            # Declaring the first reward group silently flips this default, so every
            # ``Episode_Reward/<term>`` key a flat config was logging disappears. Say so once at
            # build rather than leaving it to be noticed as a gap in wandb.
            en.logger.info(
                f"RewardManager: per-term reward log keys (Episode_Reward/<term>) are off because this config "
                f"declares reward groups; logging Episode_Reward_Group/<group> for "
                f"{list(self._group_term_names)} instead. Set `log_per_term_rewards=True` to keep both."
            )

        # 2D episode sums for vectorized accumulation.
        # ``_episode_sums`` is the weighted+masked accumulation; ``_episode_raw_sums``
        # mirrors the raw (unweighted) term outputs for diagnostic logging.
        self._episode_sums = torch.zeros((self.num_envs, num_terms), dtype=torch.float, device=self.device)
        self._episode_raw_sums = torch.zeros((self.num_envs, num_terms), dtype=torch.float, device=self.device)
        self._reward_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._step_reward = torch.zeros((self.num_envs, num_terms), dtype=torch.float, device=self.device)
        # Unweighted (but temporally masked) term outputs of the last step: compute() fills it and
        # multiplies it into _step_reward; get_term_value() reads it.
        self._step_raw_reward = torch.zeros((self.num_envs, num_terms), dtype=torch.float, device=self.device)

        # Per-group column indices into the flat term buffers.
        self._group_col_idx: dict[str, torch.Tensor] = {
            group: torch.tensor([self._term_name_to_idx[name] for name in names], dtype=torch.long, device=self.device)
            for group, names in self._group_term_names.items()
        }
        # dt of the last compute() call; 0.0 until then so the group readout is
        # zeros before the first step, matching the zero-filled buffers.
        self._last_dt = 0.0
        # Constant zero scalar: sum-start for the per-group reset logs so empty groups
        # stay tensor-typed without a per-reset allocation. Read-only by contract —
        # it is returned as-is as the log value of an empty group.
        self._log_zero = torch.zeros((), dtype=torch.float, device=self.device)

        # Pre-compute weights vector for vectorized multiply. ``_weights_host`` is the same list on
        # the host: compute() compares against it and uploads only on a change (a curriculum moving
        # a weight), since writing a Python float into a CUDA tensor is a GPU->CPU sync.
        self._weights_host = [term.weight for term in self._terms.values()]
        self._weights = torch.tensor(self._weights_host, dtype=torch.float, device=self.device)

        # Pre-allocate temporal mask (1.0 for non-temporal terms, updated per-step for temporal).
        # ``range_s`` is in seconds; convert to integer step bounds once here so the per-step
        # compare uses ``episode_length_buf`` directly (rounding keeps both ends inclusive).
        self._temporal_mask = torch.ones((self.num_envs, num_terms), dtype=torch.float, device=self.device)
        self._temporal_term_info: list[tuple[int, list[tuple[int, int]]]] = []
        for term_idx, term in enumerate(self._terms.values()):
            if term.is_temporal_dependent:
                step_range = [(round(lo / self._env.dt), round(hi / self._env.dt)) for lo, hi in term.range_s]
                self._temporal_term_info.append((term_idx, step_range))

    def summary(self) -> str:
        if self._has_explicit_groups:
            group_of = {name: group for group, names in self._group_term_names.items() for name in names}
            return self._format_summary_table(
                title="Active Reward Terms",
                field_names=["Index", "Name", "Group", "Weight"],
                rows=(
                    [index, name, group_of[name], term.weight] for index, (name, term) in enumerate(self._terms.items())
                ),
                align={"Name": "l", "Group": "l", "Weight": "r"},
            )
        return self._format_summary_table(
            title="Active Reward Terms",
            field_names=["Index", "Name", "Weight"],
            rows=([index, name, term.weight] for index, (name, term) in enumerate(self._terms.items())),
            align={"Name": "l", "Weight": "r"},
        )

    @property
    def group_term_names(self) -> dict[str, list[str]]:
        """The active term names keyed by group, in declaration order (a copy, safe to mutate)."""
        return {group: list(names) for group, names in self._group_term_names.items()}

    def iter_terms(self):
        """Yield ``(name, term)`` pairs for every active reward term, flat across all groups.

        Public accessor for sibling managers (e.g. curricula) that need to scan terms
        without reaching into the private ``_terms`` dict.
        """
        return self._terms.items()

    def get_episode_sum(self, name: str, raw: bool = True) -> torch.Tensor:
        """Return the per-env running reward sum for term ``name`` as a ``(num_envs,)`` view.

        ``raw=True`` (default) returns the unweighted, mask-applied accumulation;
        ``raw=False`` returns the weighted post-curriculum sum. Both buffers are
        reset to zero per env in :meth:`reset`. Raises ``KeyError`` (with the
        active term list) if ``name`` is not registered. See :meth:`iter_terms`
        for full-buffer scanning.
        """
        buf = self._episode_raw_sums if raw else self._episode_sums
        return buf[:, self._term_col(name)]

    def get_term_value(self, name: str) -> torch.Tensor:
        """Return term ``name``'s output at the last computed step as a ``(num_envs,)`` view.

        The value is the term's ``compute()`` output, temporally masked and zero for an env Genesis flagged as
        corrupt, with no weight and no ``dt`` — what an off-policy learner that weights its own reward channels
        stores per transition (e.g. FB-CPR's auxiliary rewards). The weighted, ``dt``-scaled share of the scalar
        reward is :meth:`get_group_reward`'s. Like ``MetricManager.get_value``, the view is the manager's
        buffer: :meth:`compute` overwrites it in place, so treat it as read-only and clone it to keep a step. It
        reads zeros before the first :meth:`compute` and still reads the pre-reset step after an auto-reset.
        Raises ``KeyError`` (with the active term list) if ``name`` is not registered.
        """
        return self._step_raw_reward[:, self._term_col(name)]

    def _term_col(self, name: str) -> int:
        """Column of term ``name`` in the flat buffers; ``KeyError`` for an inactive term."""
        idx = self._term_name_to_idx.get(name)
        if idx is None:
            message = f"reward term '{name}' not active. Active terms: {self._term_names}"
            path = self._unbuilt_term_paths.get(name)
            if path is not None:
                message += (
                    f". {path} has weight 0.0, so it is never built and has no readout: give it a nonzero weight "
                    f"to read it, which also adds it to the scalar reward compute() returns"
                )
            raise KeyError(message)
        return idx

    def _group_cols(self, group: str) -> torch.Tensor:
        """Column indices of ``group``'s terms in the flat buffers; ``KeyError`` for an unknown group."""
        idx = self._group_col_idx.get(group)
        if idx is None:
            raise KeyError(f"reward group '{group}' not found. Active groups: {list(self._group_col_idx)}")
        return idx

    def get_group_reward(self, group: str) -> torch.Tensor:
        """Return the reward of group ``group`` for the last computed step as a ``(num_envs,)`` tensor.

        The value is weighted, temporally masked, and dt-scaled — the group's
        contribution to the scalar reward :meth:`compute` returned, so summing
        this over all groups reconstructs that scalar (up to floating-point
        summation order). Valid until the next :meth:`compute` call (it reads
        the per-term step buffer that call refills); before the first compute
        it returns zeros, and after :meth:`reset` it still reads the pre-reset
        step — consistent with the scalar :meth:`compute` returned for that
        step (auto-reset runs after reward). Raises ``KeyError`` (with the
        active group list) for an unknown group.
        """
        return self._step_reward.index_select(1, self._group_cols(group)).sum(dim=1) * self._last_dt

    def get_group_episode_sum(self, group: str, *, raw: bool) -> torch.Tensor:
        """Return the per-env running reward sum of group ``group`` as a ``(num_envs,)`` tensor.

        ``raw=False`` sums the weighted post-curriculum accumulation — the
        quantity whose per-group parts add up to the episode's scalar reward
        (e.g. the CMDP task-return signal); ``raw=True`` sums the unweighted,
        mask-applied accumulation for diagnostics. ``raw`` is deliberately
        keyword-only with no default: the per-term :meth:`get_episode_sum`
        defaults to the raw view, so a silent (opposite) default here would
        flip semantics on consumers porting per-term code. Raises ``KeyError``
        (with the active group list) for an unknown group.
        """
        buf = self._episode_raw_sums if raw else self._episode_sums
        return buf.index_select(1, self._group_cols(group)).sum(dim=1)

    @staticmethod
    def is_timestep_in_range(timestep: torch.Tensor, ranges: list[tuple[float, float]]) -> torch.Tensor:
        """Check whether a batch of timesteps falls within any of the given intervals.

        Parameters
        ----------
        timestep : torch.Tensor
            ``(batch_size,)`` tensor of timesteps.
        ranges : list of tuple of float
            List of ``(lower, upper)`` tuples defining valid intervals, in the same
            units as ``timestep`` (step counts, not seconds).

        Returns
        -------
        torch.Tensor
            ``(batch_size,)`` boolean tensor indicating membership.
        """
        is_valid = torch.zeros_like(timestep, dtype=torch.bool, device=timestep.device)
        for lower, upper in ranges:
            is_valid |= (timestep >= lower) & (timestep <= upper)
        return is_valid

    def reset(self, envs_idx: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if envs_idx is None:
            envs_idx = slice(None)
        extras = {}
        # Per-term episode averages are computed for every term (the group readout below is
        # derived from them), but only surfaced as log keys when _log_per_term is set —
        # otherwise a term-heavy config's per-iteration log is dominated by them.
        # One reduction over every term's column rather than a gather per term: on the auto-reset
        # path ``envs_idx`` is the bool ``reset_buf``, and each gather with it is a GPU->CPU sync.
        max_episode_length_s = self._env.max_episode_length_s
        episode_means = mean_over_envs(self._episode_sums, envs_idx) / max_episode_length_s
        per_term: dict[str, torch.Tensor] = dict(zip(self._term_names, episode_means.unbind()))
        if self._log_per_term:
            raw_episode_means = mean_over_envs(self._episode_raw_sums, envs_idx) / max_episode_length_s
            for key, raw_mean in zip(self._term_names, raw_episode_means.unbind()):
                extras["Episode_Reward/" + key] = per_term[key]
                extras["Episode_Reward_Raw/" + key] = raw_mean
        # Always cleared, even when not logged — the accumulators must reset every episode.
        self._episode_sums[envs_idx] = 0.0
        self._episode_raw_sums[envs_idx] = 0.0
        # Per-group episode averages, derived from the per-term scalars above: groups
        # partition the terms, so a group's average is exactly the sum of its members' —
        # no extra buffer indexing on the auto-reset hot path, and one normalization
        # convention instead of two. Only when groups are declared — ungrouped envs
        # log no group keys. Group sums are emitted regardless of
        # _log_per_term (they are the terse default readout, not per-term detail).
        if self._has_explicit_groups:
            for group, names in self._group_term_names.items():
                extras["Episode_Reward_Group/" + group] = sum((per_term[name] for name in names), self._log_zero)
        for terms in self._reset_terms:
            terms.reset(envs_idx=envs_idx)
        return extras

    @torch.inference_mode()
    def compute(self, dt: float) -> torch.Tensor:
        # _step_raw_reward is filled and masked in-place through phases 1-2a; phase 2c writes its
        # weighted copy into _step_reward. After compute() returns, _step_raw_reward holds
        # compute() * mask and _step_reward holds compute() * mask * weight.
        self._last_dt = dt

        # Phase 1: Fill raw term outputs and sync weights (curriculum may modify them).
        # _step_raw_reward[:, i] = term_i.compute()
        # One pass for `compute_once` (ManagerBase): terms share an intermediate within it.
        with self.compute_pass():
            for term_idx, term in enumerate(self._terms.values()):
                self._step_raw_reward[:, term_idx] = term.compute()
        # An env Genesis flagged as corrupt this step is reset in the same step, but its terms just read that state
        # (a NaN contact force, say): the transition carries no signal, and a NaN would end the run in the learner's
        # NaN check and poison the episode sums. ``masked_fill_``, not a multiply: NaN * 0 is NaN.
        self._step_raw_reward.masked_fill_(self._env.get_solver_error_envs_mask()[:, None], 0.0)
        self.sync_device_fields()

        # Phase 2a: Apply temporal mask in-place. Raw episode sums respect this
        # mask too — accumulating before masking would log term outputs from
        # timesteps where range_s makes the term inactive.
        if self._temporal_term_info:
            for idx, step_range in self._temporal_term_info:
                self._temporal_mask[:, idx] = self.is_timestep_in_range(
                    self._env.episode_length_buf, step_range
                ).float()
            # _step_raw_reward *= mask  →  now holds compute() * mask
            self._step_raw_reward.mul_(self._temporal_mask)

        # Phase 2b: Accumulate raw episode sums (compute() * mask, no weights).
        self._episode_raw_sums.add_(self._step_raw_reward, alpha=dt)

        # Phase 2c: Apply per-term weights into the weighted buffer; the raw one is kept for
        # get_term_value(). An out-of-place multiply, so keeping it costs no extra copy.
        # _step_reward = _step_raw_reward * weights  →  compute() * mask * weight
        torch.mul(self._step_raw_reward, self._weights, out=self._step_reward)

        # Phase 3: Reduce to scalar reward and accumulate episode sums.
        torch.sum(self._step_reward, dim=1, out=self._reward_buf)
        self._reward_buf.mul_(dt)
        self._episode_sums.add_(self._step_reward, alpha=dt)
        return self._reward_buf

    def sync_device_fields(self) -> None:
        """Copy the terms' current ``weight`` onto the device, when one changed since the last call.

        A curriculum may move a term's weight between steps. ``compute()`` calls this; the CUDA graph of the
        graded phase (``EnvOptions.cuda_graph_options``) calls it before each replay instead, since a host-side
        comparison is exactly what a replay skips.
        """
        weights = [term.weight for term in self._terms.values()]
        if weights != self._weights_host:
            self._weights.copy_(torch.tensor(weights, dtype=torch.float))
            self._weights_host = weights

    def _prepare_terms(self):
        self._group_term_names: dict[str, list[str]] = dict()
        # Weight-0 (never built) term name → its config path, so a readout of one can say why it is missing.
        self._unbuilt_term_paths: dict[str, str] = {}
        self._has_explicit_groups = False
        has_flat_terms = False

        for entry_name in self._options.term_keys():
            entry = getattr(self._options, entry_name)
            if isinstance(entry, RewardGroupOptions):
                self._has_explicit_groups = True
                group_name = entry_name
                term_items = [(term_name, getattr(entry, term_name)) for term_name in entry.term_keys()]
                path_prefix = f"reward_options.{group_name}."
            else:
                # Flat term: implicit member of the default group.
                has_flat_terms = True
                group_name = DEFAULT_REWARD_GROUP
                term_items = [(entry_name, entry)]
                path_prefix = "reward_options."
            group_terms = self._group_term_names.setdefault(group_name, [])
            for term_name, term_option in term_items:
                if term_option.weight == 0.0:
                    en.logger.warning(
                        f"{path_prefix}{term_name} has weight 0.0, so the term is not built and contributes nothing."
                    )
                    self._unbuilt_term_paths.setdefault(term_name, f"{path_prefix}{term_name}")
                    continue
                # Uniqueness is enforced over *active* terms only — a weight-0
                # (never built) term occupies no buffer column or log key, so it
                # doesn't reserve its name.
                if term_name in self._terms:
                    raise ValueError(
                        f"Duplicate reward term name '{term_name}' (in group '{group_name}'): active reward term "
                        f"names must be unique across all groups — they share one flat buffer and one log namespace."
                    )
                term: RewardTermLike = self._build_term(term_name, term_option)
                self._term_names.append(term_name)
                self._terms[term_name] = term
                group_terms.append(term_name)

        if has_flat_terms and isinstance(getattr(self._options, DEFAULT_REWARD_GROUP, None), RewardGroupOptions):
            en.logger.warning(
                f"Reward group '{DEFAULT_REWARD_GROUP}' is declared explicitly alongside flat terms; "
                f"both feed the same '{DEFAULT_REWARD_GROUP}' readout and log key."
            )

        # name → column-index map; used by the per-term readouts (get_episode_sum, get_term_value)
        # so per-step callers don't pay an O(num_terms) list scan on every call.
        self._term_name_to_idx: dict[str, int] = {n: i for i, n in enumerate(self._term_names)}
