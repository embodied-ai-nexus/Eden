"""Reward manager, group, and reward-term configuration options."""

from pydantic import field_validator

from eden.options.managers.base import FunctionTermOptions, ManagerOptions


class RewardTermOptions(FunctionTermOptions):
    """
    Reward term specification.

    Parameters
    ----------
    range_s: tuple[float, float] | list[tuple[float, float]] | None
        The time window(s) in seconds where the reward is active, as ``(start_s, end_s)`` or a list of such tuples.
        Both ends are inclusive. If None, the reward is active for the entire episode. Default is None.
    weight: float
        A reward weight for the term. Default is 1.0.
    tags: list[str]
        Optional labels for grouping reward terms (e.g. ``["penalty"]`` or ``["tracking", "feet"]``).
        Curricula and other consumers can filter terms by tag instead of hard-coding term names.
    """

    range_s: tuple[float, float] | list[tuple[float, float]] | None = None
    weight: float = 1.0
    tags: list[str] = []

    @field_validator("range_s", mode="before")
    @classmethod
    def _wrap_single_range(cls, v):
        # accept a single (start_s, end_s) tuple as shorthand for [(start_s, end_s)]
        return [v] if isinstance(v, tuple) else v


class RewardGroupOptions(ManagerOptions[RewardTermOptions]):
    """
    A named group of reward terms.

    Groups give downstream consumers a per-group reward readout (multi-channel
    critics, CMDP task/style/penalty splits, per-group logging) without changing
    the scalar reward the environment returns — the reward manager always sums
    over all groups. Unlike observation groups, reward groups partition one flat
    term set: active term names must be unique across the whole reward manager,
    groups included.

    Extras type-checking, ``term_keys()``, and ``disable_term()`` all come from
    :class:`ManagerOptions` via the generic parameter.

    Parameters
    ----------
    <reward_term_name>: RewardTermOptions
        The reward terms belonging to this group.
    """


class RewardManagerOptions(ManagerOptions[RewardTermOptions | RewardGroupOptions]):
    """
    Reward manager options.

    Entries are flat reward terms, named groups of reward terms, or a mix of
    both. Flat terms implicitly belong to the manager's default group.

    Parameters
    ----------
    <reward_term_name>: RewardTermOptions
        A flat reward term configuration.
    <group_name>: RewardGroupOptions
        A named group of reward terms.
    log_per_term_rewards: bool | None
        Whether ``reset()`` emits the per-term reward log keys (``Episode_Reward/<term>``
        and the diagnostic ``Episode_Reward_Raw/<term>``) each iteration, which dominate
        per-iteration log volume on term-heavy configs. ``None`` (default)
        auto-selects: **off** when the config declares explicit reward groups — the
        ``Episode_Reward_Group/<group>`` sums are the intended readout — and **on**
        otherwise, since a fully-flat config has no group readout to fall back to. Set
        ``True`` to always emit per-term keys or ``False`` to always
        suppress them. Group-sum keys are always emitted when groups are declared and are
        unaffected by this flag; it changes only logging, never the reward itself.
    """

    # A declared config knob, not a term: it lands in ``__dict__`` and so is excluded from
    # ``term_keys()`` (like ``RecorderManagerOptions.dataset_filename``).
    log_per_term_rewards: bool | None = None

    def _groups_containing(self, name: str) -> list[str]:
        """Names of the groups that declare a term called ``name``, in declaration order."""
        hits = []
        for group_name in self.term_keys():
            group = getattr(self, group_name)
            if isinstance(group, RewardGroupOptions) and name in group.term_keys():
                hits.append(group_name)
        return hits

    def _entry_type_message(self, key, value) -> str:
        # The generic check's most likely trigger deserves an actionable message: a
        # flat-name partial override of a grouped term ({"reward_options": {"track": {...}}}
        # where `track` lives inside a group) deep-merges as a metadata-less top-level dict.
        # Name the owning group so the user can re-address the override.
        owners = self._groups_containing(key)
        if owners:
            return (
                f"Reward entry '{key}' is not a term or group, but a term named '{key}' exists inside "
                f"group '{owners[0]}' — address it through the group: {{'{owners[0]}': {{'{key}': ...}}}}."
            )
        return super()._entry_type_message(key, value)

    def disable_term(self, name: str) -> None:
        """Remove a flat term, a whole group, or a term nested inside exactly one group, by name.

        The name must resolve unambiguously. If it matches more than one entry (a term
        declared in several groups, or a top-level entry and a nested term), this raises
        ``ValueError`` — disable through the owning group instead:
        ``getattr(options, group).disable_term(name)``. Ambiguity here is name-based, not
        activity-based: a weight-0 (inert) duplicate still counts as a match, unlike the
        manager's build-time uniqueness check, because options can't know final weights.
        """
        owner_groups = self._groups_containing(name)
        places = (["top level"] if name in self.term_keys() else []) + [f"group '{g}'" for g in owner_groups]
        if len(places) > 1:
            raise ValueError(
                f"Name '{name}' is ambiguous ({' and '.join(places)}). To remove a nested term, call "
                f"disable_term on its group: getattr(options, group).disable_term({name!r})."
            )
        if owner_groups:
            getattr(self, owner_groups[0]).disable_term(name)
            return
        # Top-level hit (flat term or whole group), or the canonical KeyError.
        self._remove_named_child(name, noun="term")
