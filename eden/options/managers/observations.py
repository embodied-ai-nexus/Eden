"""Observation manager, group, and term configuration options."""

import difflib
from collections.abc import Collection
from typing import Any, ClassVar, Literal, TYPE_CHECKING

from genesis.typing import Vec2FType
from eden.options._metadata import OPTION_METADATA_KEYS
from eden.options.options import DynamicConfigurableOptions
from eden.options.managers.base import FunctionTermOptions, ManagerOptions
from eden.options.managers.modifiers import NoiseOptions, ObservationModifierOptions

#: "no entry under that name", distinct from an entry whose value is ``None``.
_UNSET = object()


def _near_misses(names: Collection[str], candidates: Collection[str]) -> dict[str, str]:
    """Pair each of ``names`` with its closest ``candidates`` entry, each candidate claimed at most once.

    One-to-one because each pairing becomes a rewrite of ``terms_order``: two names mapped onto one
    candidate would hand back an order that lists that term twice.
    """
    pairs: dict[str, str] = {}
    for name in names:
        pool = [candidate for candidate in candidates if candidate not in pairs.values()]
        close = difflib.get_close_matches(name, pool, n=1)
        if close:
            pairs[name] = close[0]
    return pairs


class ObservationTermOptions(FunctionTermOptions):
    """
    Observation term specification.

    The manager processes each frame of the term's output in this order: ``modifiers``, ``noise``,
    ``clip``, ``scale``, then the group's ``nan_policy``, and last the history window.

    Parameters
    ----------
    modifiers: list[ObservationModifierOptions], optional
        Transforms applied to the term's output in list order, before noise (default=[], none). Each entry
        is a registered :class:`~eden.managers.modifiers.base.ObservationModifier` configured with
        ``<ModifierClass>.configure(...)``, e.g. ``ExponentialMovingAverage.configure(alpha=0.2)`` or
        ``ObservationDelay.configure(control_steps=2)``. Unlike noise, modifiers run in every mode:
        training, eval and deployment. A modifier must return the shape, dtype and device it receives.
    noise: NoiseOptions or None, optional
        Noise model applied to the observation, in training only (default=None, no noise).
    clip: array-like[float, float] or None, optional
        Range to clip the observation values (default=None, no clipping).
    scale: float or list[float], optional
        Scaling factor applied to the observation (default=1.0). A list gives a per-component
        scale matched against the term's feature dim — the last dimension of its raw
        ``compute()`` output — validated at env-construction time and applied before history
        stacking.
    history_length: int, optional
        Number of previous observations to include (default=0, no history).
    flatten_history_dim: bool, optional
        Whether to flatten the history into the feature axis, giving ``(num_envs, history_length * dim)``
        instead of ``(num_envs, history_length, dim)`` (default=True, the same as the group's).
    backfill: bool, optional
        Whether to backfill the history with the first observation (default=True).
        If False, the history will be filled with zeros.
    """

    modifiers: list[ObservationModifierOptions] = []
    noise: NoiseOptions | None = None
    clip: Vec2FType | None = None
    scale: float | list[float] = 1.0
    history_length: int = 0
    # Matches ``ObservationGroupOptions.flatten_history_dim``: a term whose history equals its group's
    # is then the same history whether or not the term spelled out the flag.
    flatten_history_dim: bool = True
    backfill: bool = True

    #: Single source of truth for the post-compute modifier fields — those applied by the manager *after*
    #: ``term.compute()`` (so they don't change a term's raw output). Co-located with the field declarations
    #: above so the set can't drift from the schema. Used by the observation manager to (a) exclude these
    #: from the cross-group dedup signature and (b) treat them as options-only params not mirrored as term
    #: attributes. Keep in sync with the fields above when editing.
    POST_COMPUTE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"modifiers", "noise", "clip", "scale", "history_length", "flatten_history_dim", "backfill"}
    )


class ObservationGroupOptions(DynamicConfigurableOptions):
    """
    Observation group options.

    Parameters
    ----------
    concatenate_terms: bool, optional
        Whether to concatenate the terms (default=True).
    concatenate_dim: int, optional
        The dimension to concatenate the terms (default=-1).
    terms_order: list[str] | None, optional
        The order of the terms in the group (default=None).
    enable_corruption: bool, optional
        Whether to allow adding noise to each term in the group (default=True). ``False`` is a
        group-level master switch: it disables the ``noise`` of *every* term in the group, whatever
        each term configured, so one term dict can serve a noisy ``policy`` and a clean ``critic``.
        The terms keep their ``noise`` in the config — the group does not edit what it adopts — and
        the observation manager drops it when it builds the group, logging one info line that names
        the group and the terms, so noise that does not reach the policy is stated rather than
        silent.
    history_length: int | None, optional
        The history length enforced for all terms in the group (default=None).
    flatten_history_dim: bool, optional
        Whether to flatten the history dimension (default=True).
    backfill: bool, optional
        Whether to backfill the history with the first observation (default=True).
        If False, the history will be filled with zeros.
    nan_policy: {"disabled", "warn", "sanitize", "error"}, optional
        What to do when a term delivers a non-finite value (NaN or ±Inf) (default="disabled", no
        check). The check is on each term's frame after its modifiers, noise, clip and scale, just
        before the history window, so a report names the term. A ``clip`` that bounds an infinity is
        therefore never reported. ``"sanitize"`` replaces each non-finite entry with 0.0 in what the
        policy receives. ``"warn"`` does the same and also logs the first occurrence for each term,
        naming the envs. ``"error"`` raises ``FloatingPointError``. The check runs in every mode
        (training, eval, deployment) and on every computation, including the final-observation
        snapshot. It cannot clean a stateful modifier upstream of it: a NaN that reaches an
        ``ExponentialMovingAverage`` stays in the average until the env resets. ``"sanitize"``
        stays on the device. ``"error"``, and ``"warn"`` for terms that have not warned yet, cost one GPU
        sync per group per observation compute.
    <term_name>: ObservationTermOptions, optional
        The observation term configuration to be used for the given group.
    """

    concatenate_terms: bool = True
    concatenate_dim: int = -1
    terms_order: list[str] | None = None
    enable_corruption: bool = True
    history_length: int | None = None
    flatten_history_dim: bool = True
    backfill: bool = True
    nan_policy: Literal["disabled", "warn", "sanitize", "error"] = "disabled"

    if TYPE_CHECKING:
        # NOTE: these will be set automatically in the observation manager
        terms: dict[str, ObservationTermOptions]

    def _adopt_terms(self, only: Collection[str] | None = None) -> None:
        """Take ownership of the term options handed to this group.

        ``only`` restricts the adoption to those entry names — the ones a ``model_copy(update=...)``
        just brought in; every other term was adopted when the source group was constructed.

        One term dict is routinely splatted into two groups — the usual ``policy`` / ``critic``
        pair. Keeping the caller's objects would make both groups share them, so editing one
        group's term silently edits the other's; this is the same aliasing
        :class:`~eden.utils.common.ConfigurableMixin` avoids by deep-copying mutable class
        defaults. Copying here means the plain constructor is safe and a task author has nothing
        to remember.

        A term written as a plain ``dict`` — the spelling :meth:`_validate_group` accepts and the
        observation manager builds with ``ObservationTermOptions(**value)`` — is built here
        instead, for the same reason: a dict handed to two groups is shared by both *and* by the
        caller, so the de-aliasing this method exists for would not cover it. It is then copied
        like any other term, because building alone leaves the new options holding the caller's
        own nested ``params`` dict. Building it here also moves the term registry's param check
        from manager build to this line. An assignment onto a constructed group ends
        here too (:meth:`__setattr__`), so every way a term enters a group runs this — and
        the observation manager can take each entry as an ``ObservationTermOptions`` the group
        owns rather than re-building what it finds.

        Adoption copies and builds; it applies none of the group's settings. A group that runs
        uncorrupted keeps the ``noise`` it is handed, and ``enable_corruption=False`` is applied
        where every other group-level override is: in the observation manager, as it builds the
        group. Nulling it here instead would delete the setting before the manager's info line
        could report it, and only for the entries of the ``model_copy(update=...)`` that carried
        it, so which terms the build named would depend on how the author split their edits
        across calls. A shared dict still does not have to be noise-free to be
        reused by a critic — the switch disables it either way, and keeping it in the config
        means flipping the switch back on restores it.
        """
        extra = self.__pydantic_extra__
        if not extra:
            return
        for name, term in extra.items():
            if only is not None and name not in only:
                continue
            if isinstance(term, dict) and self._is_term_entry(name):
                if term.get("name") is None and not any(key in term for key in OPTION_METADATA_KEYS):
                    # Any mapping builds — ``ObservationTermOptions`` is ``extra="allow"`` and ``name``
                    # defaults to ``None`` — so without this a misspelled group field written as a
                    # dict (``nan_polciy={"mode": "warn"}``) would be a term, and die at build as
                    # ``KeyError: None`` naming neither the group, the entry nor the field.
                    # A dict carrying the type metadata is a *serialized* term, nameless or not —
                    # the placeholder ``ObservationTermOptions()`` round-trips that way — and a
                    # nameless one is refused where it can first be, at build.
                    raise TypeError(
                        self._not_a_term_message(
                            name,
                            f"a term written as a dict names its registered term with 'name' "
                            f"({{'name': 'base_lin_vel', ...}}, or ObsTerm.configure(...)), and this one has "
                            f"{sorted(term) or 'no keys'}",
                        )
                    )
                try:
                    term = ObservationTermOptions(**term)
                except Exception as e:
                    # Composed into the message, not an add_note: this runs inside
                    # ``model_post_init``, so pydantic re-raises as a ValidationError for the
                    # *group* and only the text survives. Without the entry name, a group with a
                    # dozen terms reports an invalid param and not which term wrote it.
                    raise ValueError(f"observation term '{name}', written as a dict, is invalid: {e}") from e
            if not isinstance(term, ObservationTermOptions):
                continue  # non-term extras (the _option_* type metadata strings, the resolved terms dict)
            # Both spellings end here, and both need the deep copy: pydantic shares the caller's
            # nested values by reference, so a freshly built term still holds the very ``params``
            # dict the caller wrote — two groups splatted from one dict-spelled term would edit
            # each other's entity_name. ``clone()`` is what makes "the group owns its terms" true
            # of the whole tree rather than of the top-level object.
            extra[name] = term.clone()

    def model_post_init(self, context) -> None:
        super().model_post_init(context)
        self._adopt_terms()
        self._validate_group()

    def _validate_copy_update(self, update) -> None:
        """Hold ``model_copy(update=...)`` to the construction rules.

        A group is ``extra="allow"`` — its terms *are* the extras — so pydantic's unvalidated
        ``update=`` could add a term the ``terms_order`` does not list, park a misspelled group
        field (``terms_ordr``) as if it were a term, and hand the copy a term object the caller
        still owns. Same adoption, same checks as the constructor.
        """
        self._adopt_terms(only=update.keys())
        self._validate_group()

    def __setattr__(self, name: str, value: Any) -> None:
        """Adopt and check a term *assigned* onto a constructed group, as construction does.

        ``group.<term> = ObsTerm.configure(...)`` is the counterpart of :meth:`disable_term` and
        the natural way to add a term to an inherited config, but it is the one entry into the
        extras that passes through neither ``model_post_init`` nor :meth:`_validate_copy_update`.
        Without this the value would land in ``__pydantic_extra__`` exactly as written: two groups
        assigned one dict would share it — and the ``params`` inside it with each other and with the
        caller, all the way into the terms the manager builds from it — and a misspelled group
        field parked as a term would be found at build instead of on the line that wrote it.

        Only a write that lands in the extras as a term is adopted; a declared group field, a
        private attribute and the bookkeeping extras (:meth:`_is_term_entry`) are left to pydantic.
        A rejected assignment is rolled back, so a group that refuses a term is left exactly as it
        was — as a refused construction leaves no object at all.

        The checks are :meth:`_validate_term_entries` and :meth:`_validate_terms_listed` — the
        rules a term write can break, and only those. The invariants between the group's declared
        fields (:meth:`_validate_group_settings`) and "every name in ``terms_order`` is a term the
        group has, once" (:meth:`_validate_terms_order`) belong to the writes that can break *them*, and
        an assignment is not one: it touches no declared field, and adding a term can only ever
        satisfy a name the order already carries. Running either here would refuse a sound term and
        name a rule the line never touched — over a ``history_length`` / ``concatenate_terms``
        pair written two lines earlier, or over the *first* of two terms an ordered group is
        being given, because the order names the second, which is the shape
        ``docs/guides/task-mods.md`` teaches for adding terms at all.
        """
        if not name.startswith("_") and name not in type(self).model_fields and not self._locked:
            # Ahead of the write: a name the class already has goes to the instance ``__dict__``,
            # not the extras, so the checks below would never see it. A locked group raises its
            # own RuntimeError in ``super().__setattr__`` instead.
            self._refuse_shadowing_child(name, noun="term", assigned=True)
        extra = getattr(self, "__pydantic_extra__", None)
        previous = _UNSET if extra is None else extra.get(name, _UNSET)

        super().__setattr__(name, value)  # the lock guard lives here, ahead of any mutation

        extra = getattr(self, "__pydantic_extra__", None)
        if extra is None or name not in extra or not self._is_term_entry(name):
            return
        try:
            self._adopt_terms(only=(name,))
            self._validate_term_entries()
            self._validate_terms_listed()
        except Exception as e:
            if previous is _UNSET:
                del extra[name]
                # pydantic records an extras write as a set field, and a name left behind there is
                # a field the group does not have; drop it with the entry.
                self.__pydantic_fields_set__.discard(name)
            else:
                extra[name] = previous
            # Said once, for every way this write can be refused: the checks above describe what is
            # wrong, none of them knows that the group has since been put back.
            e.add_note(
                f"The assignment of '{name}' was refused, so the group is unchanged: it has "
                f"{', '.join(repr(key) for key in self.term_keys()) or 'no terms'}."
            )
            raise

    @staticmethod
    def _is_term_entry(name: str) -> bool:
        """Whether an extras key names a term, as opposed to the bookkeeping parked beside them.

        ``terms`` is the resolved dict the observation manager stashes on its build-time copy;
        ``_option_*`` is the type metadata serialization writes. Everything else in the extras is
        a term, or a misspelled group field that :meth:`_validate_group` rejects as one.
        """
        return name != "terms" and not name.startswith("_option_")

    def _term_entries(self) -> dict[str, Any]:
        """The extras that are this group's terms (see :meth:`_is_term_entry`)."""
        extra = self.__pydantic_extra__ or {}
        return {key: value for key, value in extra.items() if self._is_term_entry(key)}

    def term_keys(self) -> tuple[str, ...]:
        """Return the names of this group's terms (the counterpart of ``ManagerOptions.term_keys``)."""
        return tuple(self._term_entries())

    def _validate_group(self) -> None:
        """The whole construction rule: the group's own settings, then its terms.

        Also what the observation manager asks before it builds the group, since a write to a declared
        field is pydantic's and unchecked.
        """
        self._validate_group_settings()
        self._validate_terms()

    def _validate_group_settings(self) -> None:
        """The invariants *between the group's declared fields*.

        Split from :meth:`_validate_terms` so a term write can be held to the term rules without
        also being held to these (see :meth:`__setattr__`): assignment leaves declared fields to
        pydantic, so a term write cannot have broken one, and re-checking them there refuses a
        sound term and names a rule its line never touched.
        """
        if self.history_length is not None:
            if self.history_length <= 0:
                raise ValueError("History length must be greater than 0!")
            if not self.concatenate_terms:
                raise ValueError("Concatenate terms must be True if history length is enforced for the group!")

    def _validate_terms(self) -> None:
        """All three term rules: the entries are terms, and ``terms_order`` names exactly them.

        Run wherever the group is made whole — construction, ``model_copy(update=...)`` — and once
        more (as part of :meth:`_validate_group`) by the observation manager as it builds the group,
        because ``terms_order`` is a *declared* field: a bare ``group.terms_order = [...]`` reaches a
        build with the order and the terms out of step, and it can break either direction.

        The order of the three is the order of causes. A name misspelled in ``terms_order`` breaks
        both directions at once — it names no term, and it leaves the term it meant unlisted — so
        the half that can recognise a misspelling runs first, or the author is told to append the
        real name *beside* the typo and fails again on the fix they were handed.
        """
        self._validate_term_entries()
        self._validate_terms_order()
        self._validate_terms_listed()

    def _validate_term_entries(self) -> None:
        """Every entry in the extras is a term, not a misspelled group field parked among them."""
        for name, value in self._term_entries().items():
            # Every way into the extras runs ``_adopt_terms`` first, and it builds the ``dict``
            # spelling into an ``ObservationTermOptions``, so a term entry is one by the time it
            # gets here. Anything else is a misspelled group field that landed in the extras —
            # unchecked, it would reach the manager and die there as ``ObservationTermOptions(**["a"])``.
            self._refuse_shadowing_child(name, noun="term")
            if not isinstance(value, ObservationTermOptions):
                raise TypeError(
                    self._not_a_term_message(
                        name, f"a term must be an ObservationTermOptions, got {type(value).__name__}"
                    )
                )

    def _not_a_term_message(self, name: str, reason: str) -> str:
        """Message for an entry that is no term — most often a misspelled group field — with its likely fix."""
        if name in ObservationTermOptions.model_fields:
            # ``modifiers`` / ``clip`` / ``noise`` written on the group: plausible, because
            # ``nan_policy`` and ``history_length`` are group fields.
            hint = f" '{name}' is set per term: ObsTerm.configure(..., {name}=...)."
        else:
            close = difflib.get_close_matches(name, type(self).model_fields, n=1)
            hint = f" Did you mean the group field '{close[0]}'?" if close else ""
        return f"'{name}' is neither a field of {type(self).__name__} nor an observation term: {reason}.{hint}"

    def _validate_terms_order(self) -> None:
        """Every name in ``terms_order`` is a term this group has, and is named there once.

        The rules a ``terms_order`` write can break — never a term write, which can only ever
        *satisfy* a name the order already carries and cannot repeat one, so an assignment leaves
        this one alone (:meth:`__setattr__`). It is also the half that has to run on a group holding
        *no* terms: an order declared over nothing would otherwise reach the manager's
        ``group_options.terms[term_name]`` as a bare ``KeyError`` naming neither the group, the
        order, nor the fix.

        The duplicate check runs first, and counts only names the group *has*. Dropping a repeat of
        a real term never changes which terms the order names, so it cannot invalidate the corrected
        order the second check offers; the reverse is not true — a corrected order handed out while
        a repeat is still in the list carries the repeat forward into the fix the author types. A
        repeated name that is *no* term inverts that: it is a symptom of the misspelling the second
        check reports, nothing builds it twice, and reporting it as a repeat explains it with a
        consequence that cannot happen and offers an order that still carries the real fault. So it
        falls through, and one corrected order resolves both.

        This runs only where the group must be whole, so every name it reports is a fault: a
        misspelling of a term the order leaves unlisted, or a name that matches no term at all.
        """
        if self.terms_order is None:
            return
        entries = self._term_entries()

        counts: dict[str, int] = {}
        for name in self.terms_order:
            counts[name] = counts.get(name, 0) + 1
        repeated = [f"{name!r} ({count} times)" for name, count in counts.items() if count > 1 and name in entries]
        if repeated:
            # Set membership alone accepts a repeat: both directions below are satisfied by
            # ``['a', 'a']`` over a group holding ``a``, and the manager then walks the order and
            # builds the term once per entry. ``list(counts)`` de-duplicates the whole order, not
            # just the reported names — dropping a repeat is this message's fix, and a name that
            # is no term is the *next* message's, which the author reaches by typing this one.
            raise ValueError(
                f"This group's terms_order names {', '.join(repeated)}, but it must name each term exactly "
                f"once. The order is the layout of the group's observation vector, so a repeated name builds "
                f"that term a second time and concatenates its output again — a wider group, with no error and "
                f"no log line to say why. terms_order = {list(counts)} names each once."
            )

        # De-duplicated: a repeated name that is no term gets here (the check above counts only the
        # group's own), and listing it twice reads as two faults and takes the spelling fix below
        # with it, since that is offered for a single missing name.
        missing = list(dict.fromkeys(name for name in self.terms_order if name not in entries))
        if not missing:
            return
        # Two ways to get here, and the group's own terms tell them apart on sight: a name
        # misspelled in the order — close to a term the order leaves unlisted — and a name that
        # matches no term. Each gets the fix that belongs to it, per name: offering
        # "assign group.base_lin_vell" for a typo would add a second term under the misspelling and
        # leave the real one unlisted. Only *unlisted* terms are candidates, so a correction never
        # names a term the order already carries — the duplicate check above would refuse it.
        one = len(missing) == 1
        names = ", ".join(repr(name) for name in missing)
        have = ", ".join(repr(name) for name in entries) if entries else "no terms"
        lead = (
            f"Term{'' if one else 's'} {names} {'is' if one else 'are'} named in this group's terms_order "
            f"but {'is not a term' if one else 'are not terms'} of the group, which has {have}."
        )
        corrections = _near_misses(missing, [name for name in entries if name not in self.terms_order])
        unmatched = [name for name in missing if name not in corrections]
        # De-duplicated because a misspelling may itself be repeated — every occurrence becomes the
        # one correct name, and an order that then listed it twice is a fix that fails above.
        corrected = list(dict.fromkeys(corrections.get(name, name) for name in self.terms_order))
        if not unmatched:
            raise ValueError(
                f"{lead} terms_order = {corrected} spells {'it' if one else 'them'} the way the group does; "
                f"terms_order must name exactly the group's terms."
            )
        # ``terms_order = None`` rather than an empty list when nothing is left: the field's own
        # "fall back to declaration order", and a group ordering an empty list reads like a typo.
        kept = [name for name in corrected if name not in unmatched]
        drop = f"terms_order = {kept}" if kept else "terms_order = None"
        pronoun = "it" if len(unmatched) == 1 else "them"
        # A name close to a term the order *already* lists is the one the group cannot read: a
        # misspelled repeat (drop it) or a term still to be written (assign it). Unqualified, "assign
        # group.dofs_poss" for ``['dofs_pos', 'dofs_poss']`` adds a second term under the typo, so the
        # message says which listed name it is near, offers the assignment for a name that matches
        # nothing ahead of one that echoes a listed term, and for the echo only as a condition.
        listed = [name for name in entries if name in self.terms_order]
        echoes = {name: close[0] for name in unmatched if (close := difflib.get_close_matches(name, listed, n=1))}
        strangers = [name for name in unmatched if name not in echoes]
        near = "".join(
            f" {name!r} is a near spelling of {term!r}, which terms_order already names, so it may be a misspelled "
            f"repeat."
            for name, term in echoes.items()
        )
        if strangers:
            target, whom = strangers[0], pronoun if not echoes else ", ".join(repr(name) for name in strangers)
            condition = ""
        else:
            target, whom = unmatched[0], pronoun
            condition = f", if {target!r} is a term still to be written,"
        dropped = ", ".join(repr(name) for name in unmatched)
        # With a name to assign beside an echo, the order that keeps it must still drop the echo.
        keep = [name for name in corrected if name not in echoes] if strangers else corrected
        if corrections:
            # A misspelling beside a name that matches nothing: correct the one, and the "drop or
            # assign" fix is offered for the other alone — never for the typo.
            typos = "; ".join(f"{typo!r} looks like {term!r} misspelled" for typo, term in corrections.items())
            if strangers:
                others = ", ".join(repr(name) for name in strangers)
                typos += f"; {others} {'matches' if len(strangers) == 1 else 'match'} no term"
            raise ValueError(
                f"{lead} {typos}.{near} terms_order must name exactly the group's terms: {drop} corrects the "
                f"spelling and drops {dropped}, or{condition} terms_order = {keep} corrects it and keeps "
                f"{whom} for group.{target} = ObsTerm.configure(...)."
            )
        if echoes and strangers:
            repeats = ", ".join(repr(name) for name in echoes)
            raise ValueError(
                f"{lead}{near} terms_order must name exactly the group's terms: {drop} drops {pronoun}, or "
                f"terms_order = {keep} drops {repeats} and keeps {whom} for group.{target} = ObsTerm.configure(...)."
            )
        raise ValueError(
            f"{lead}{near} terms_order must name exactly the group's terms: {drop} drops {pronoun}, or{condition} "
            f"assign {whom}: group.{target} = ObsTerm.configure(...)."
        )

    def _validate_terms_listed(self) -> None:
        """Every term the group has is named in ``terms_order``.

        With :meth:`_validate_term_entries`, the rules a *term* write can break, so this is what an
        assignment runs (:meth:`__setattr__`). Wherever the group must be whole,
        :meth:`_validate_terms_order` has run first, so every name the order carries is a term.
        """
        if self.terms_order is None:
            return
        entries = self._term_entries()
        unlisted = [name for name in entries if name not in self.terms_order]
        if not unlisted:
            return
        # Reached from an assignment, the order may carry names the group does not have *yet* —
        # which is allowed (see :meth:`__setattr__`) — and one of them can look like a term left
        # unlisted. That is either a misspelling in the order or a second term still to come
        # (``foot_pos_l`` / ``foot_pos_r``), and nothing here can tell which, so both fixes are
        # offered. Picking one would hand half of the authors the wrong fix: "the typo" drops a
        # term they meant, and "append it" keeps a typo the build then refuses.
        waiting = list(dict.fromkeys(name for name in self.terms_order if name not in entries))
        near = _near_misses(unlisted, waiting)
        if near:
            by_waiting = {name: term for term, name in near.items()}
            corrected = [by_waiting.get(name, name) for name in self.terms_order]
            corrected += [name for name in unlisted if name not in near]
            appended = self.terms_order + unlisted
            if len(near) == 1:
                ((term, name),) = near.items()
                message = (
                    f"Term {term!r} is not listed in this group's terms_order, which names {name!r} — not a "
                    f"term of the group, and a near spelling of {term!r}. If {name!r} misspells it, "
                    f"terms_order = {corrected} corrects it; if {name!r} is another term still to be "
                    f"assigned, terms_order = {appended} lists both."
                )
            else:
                message = (
                    f"Terms {', '.join(repr(term) for term in near)} are not listed in this group's "
                    f"terms_order, which names {', '.join(repr(name) for name in near.values())} — not "
                    f"terms of the group, and near spellings of those. If they are misspellings, "
                    f"terms_order = {corrected} corrects them; if they are other terms still to be "
                    f"assigned, terms_order = {appended} lists all of them."
                )
            raise ValueError(
                f"{message} An appended term lands last; the order is the layout of the group's observation "
                f"vector, so move it if it belongs elsewhere."
            )
        # *Every* missing name, not the first: a group can already be missing one when an
        # assignment adds another, and a message naming only the pre-existing one would send the
        # author back for a second round without ever mentioning the term they were writing.
        # Appending is a guess only the author can make — the order is the layout of the group's
        # observation vector — so the order to type is offered, not applied.
        names = ", ".join(repr(name) for name in unlisted)
        subject = f"Term {names} is" if len(unlisted) == 1 else f"Terms {names} are"
        placement = "puts it last" if len(unlisted) == 1 else "puts them last"
        raise ValueError(
            f"{subject} not listed in this group's terms_order, which must name every term in the "
            f"group. terms_order is currently {self.terms_order}; terms_order = "
            f"{self.terms_order + unlisted} {placement}, or terms_order = None falls back to "
            f"declaration order."
        )

    def disable_term(self, name: str) -> None:
        """Remove a term from this group by name, and from ``terms_order`` when the group declares one.

        ``terms_order`` must list exactly the group's terms, so a name left behind there would fail
        the next load or the observation manager's build, far from this call.

        Raises ``KeyError`` (listing the terms that exist) for an unknown name and
        ``RuntimeError`` on a locked config.
        """
        self._remove_named_child(name, noun="term")
        if self.terms_order is not None:
            # Rebuilt, not edited in place: assignment on an options model is unvalidated, so the
            # list may be shared with another group or with the code that assigned it.
            self.terms_order = [term for term in self.terms_order if term != name]


class ObservationManagerOptions(ManagerOptions[ObservationGroupOptions]):
    """
    Observation manager options.

    Parameters
    ----------
    <group_name>: ObservationGroupOptions
        The observation terms configuration to be used for the given group.
    """

    _entry_noun: ClassVar[str] = "group"
