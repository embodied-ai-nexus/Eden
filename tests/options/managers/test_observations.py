"""Tests for ``eden.options.managers.observations``: ``ObservationGroupOptions`` and ``ObservationTermOptions``,
their validated fields, the post-compute field set, ``disable_term``, adoption of the terms a group is given, and ``model_copy`` of a group that
holds its terms as extras. No scene.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from eden import observations

import eden.terms  # noqa: F401 — stamps the observation registry onto its options class
from eden.managers.modifiers.observations import ObservationDelay
from eden.managers.modifiers.observations.noise import UniformNoise
from eden.options.managers.modifiers import NoiseOptions
from eden.options.managers.observations import ObservationGroupOptions, ObservationTermOptions
from eden.terms import ObsTerm


def _term() -> ObservationTermOptions:
    return ObservationTermOptions(name="base_lin_vel", params={"entity_name": "robot"}, noise=NoiseOptions())


def test_a_misspelled_nan_policy_is_refused_where_it_is_written():
    with pytest.raises(ValidationError, match="nan_policy"):
        ObservationGroupOptions(nan_policy="sanitise")


def test_modifiers_written_on_the_group_point_at_the_term():
    with pytest.raises(TypeError, match=r"'modifiers' is set per term: ObsTerm.configure\(\.\.\., modifiers=\.\.\.\)"):
        ObservationGroupOptions(x=_term(), modifiers=[ObservationDelay.configure(control_steps=1)])


def test_the_post_compute_fields_are_real_schema_fields():
    """The manager excludes these from the cross-group cache key; a stale name would silently stop excluding."""
    assert ObservationTermOptions.POST_COMPUTE_FIELDS <= set(ObservationTermOptions.model_fields)


def test_a_misspelled_group_field_is_named_at_construction_on_copy_and_on_assignment():
    """``terms_ordr`` used to land in the extras as a term and die in the manager as ``ObservationTermOptions(**["a"])``."""
    with pytest.raises(TypeError, match=r"'terms_ordr'.*Did you mean the group field 'terms_order'\?"):
        ObservationGroupOptions(a=_term(), terms_ordr=["a"])

    group = ObservationGroupOptions(a=_term())
    with pytest.raises(TypeError, match=r"'terms_ordr'.*Did you mean the group field 'terms_order'\?"):
        group.model_copy(update={"terms_ordr": ["a"]})
    with pytest.raises(TypeError, match="'b' is neither a field"):
        group.updated(b=0.0)
    with pytest.raises(TypeError, match=r"'terms_ordr'.*Did you mean the group field 'terms_order'\?"):
        group.terms_ordr = ["a"]


def test_a_group_copy_checks_its_terms_order():
    group = ObservationGroupOptions(a=_term(), terms_order=["a"])

    with pytest.raises(ValueError, match=r"Term 'b' is not listed in this group's terms_order"):
        group.model_copy(update={"b": _term()})
    with pytest.raises(ValueError, match=r"Term 'zzz' is named in this group's terms_order"):
        group.model_copy(update={"terms_order": ["zzz"]})
    added = group.model_copy(update={"b": _term(), "terms_order": ["b", "a"]})
    assert (added.terms_order, group.terms_order) == (["b", "a"], ["a"])


def test_a_group_copy_owns_an_added_term_and_leaves_its_noise_alone():
    """Adoption is a copy and nothing else: the noise stays whatever ``enable_corruption`` says."""
    group = ObservationGroupOptions(a=_term(), enable_corruption=False)
    term = _term()

    copied = group.model_copy(update={"b": term})

    assert copied.b is not term
    assert copied.b.noise == term.noise is not None
    assert copied.a is group.a  # shallow: untouched entries are shared, as in pydantic


def test_disable_term_keeps_the_declared_order_consistent_without_touching_the_callers_list():
    group = ObservationGroupOptions(
        dofs_pos=ObservationTermOptions(), dofs_vel=ObservationTermOptions(), last_action=ObservationTermOptions()
    )
    order = ["dofs_pos", "dofs_vel", "last_action"]
    group.terms_order = order  # unvalidated assignment: the group now shares the caller's list

    group.disable_term("dofs_vel")

    assert group.term_keys() == ("dofs_pos", "last_action")
    assert group.terms_order == ["dofs_pos", "last_action"]
    assert order == ["dofs_pos", "dofs_vel", "last_action"]
    # Reconstruction re-runs the group check a dangling terms_order entry used to fail.
    assert ObservationGroupOptions.model_validate(group.model_dump()).terms_order == ["dofs_pos", "last_action"]


def test_a_group_without_a_declared_order_keeps_not_having_one():
    group = ObservationGroupOptions(a=ObservationTermOptions(), b=ObservationTermOptions())

    group.disable_term("a")

    assert group.terms_order is None


@pytest.mark.parametrize(
    "make_term",
    [
        lambda: ObsTerm.configure(func=observations.base_lin_vel, params={"entity_name": "robot"}),
        lambda: {"name": "base_lin_vel", "params": {"entity_name": "robot"}},
    ],
    ids=["obs-term", "dict"],
)
@pytest.mark.parametrize("given_by", ["construction", "assignment"])
def test_groups_given_one_term_own_their_copies(make_term, given_by):
    """Pydantic shares a caller's nested dict by reference, so ``params`` must be copied, not just the term.

    Assignment is how a task mod adds a term to an inherited config, so it must not keep the caller's object either.
    """
    shared = {"dofs_pos": make_term()}
    if given_by == "construction":
        policy = ObservationGroupOptions(**shared, enable_corruption=True)
        critic = ObservationGroupOptions(**shared, enable_corruption=False)
    else:
        policy, critic = ObservationGroupOptions(), ObservationGroupOptions()
        policy.dofs_pos = shared["dofs_pos"]
        critic.dofs_pos = shared["dofs_pos"]

    assert isinstance(policy.dofs_pos, ObservationTermOptions)
    policy.dofs_pos.scale = 7.0
    policy.dofs_pos.params["entity_name"] = "MUTATED"

    assert critic.dofs_pos.scale != 7.0
    assert critic.dofs_pos.params["entity_name"] == "robot"
    source = shared["dofs_pos"]
    if isinstance(source, dict):
        assert source == {"name": "base_lin_vel", "params": {"entity_name": "robot"}}
    else:
        assert source.scale != 7.0
        assert source.params["entity_name"] == "robot"


def test_updated_adopts_a_dict_spelled_term_too():
    """``updated()`` runs the same adoption on the entries it brings in."""
    group = ObservationGroupOptions(
        a=ObsTerm.configure(func=observations.base_lin_vel, params={"entity_name": "robot"})
    )
    raw = {"name": "base_ang_vel", "params": {"entity_name": "robot"}}

    copied = group.updated(b=raw)

    assert isinstance(copied.b, ObservationTermOptions)
    copied.b.params["entity_name"] = "MUTATED"
    assert raw["params"]["entity_name"] == "robot"


def test_adoption_keeps_noise_whatever_enable_corruption_says():
    """Whether noise applies is the observation manager's call at build, not adoption's."""
    noisy = {
        "dofs_pos": ObsTerm.configure(
            func=observations.base_lin_vel,
            params={"entity_name": "robot"},
            noise=UniformNoise.configure(n_min=-1, n_max=1),
        )
    }
    corrupted = ObservationGroupOptions(**noisy, enable_corruption=True)
    uncorrupted = ObservationGroupOptions(**noisy, enable_corruption=False)

    assert corrupted.dofs_pos.noise == uncorrupted.dofs_pos.noise == noisy["dofs_pos"].noise
    assert noisy["dofs_pos"].noise is not None


@pytest.mark.parametrize("given_by", ["construction", "assignment"])
def test_an_invalid_dict_spelled_term_names_the_entry_it_came_from(given_by):
    """The term registry's param check runs at the config line, and the message names the entry."""
    good = {"name": "base_lin_vel", "params": {"entity_name": "robot"}}
    bad = {"name": "base_lin_vel", "params": {"bogus_param": 1}}
    with pytest.raises(ValueError, match=r"(?s)'bad_term', written as a dict.*bogus_param.*accepted params"):
        if given_by == "construction":
            ObservationGroupOptions(good_term=good, bad_term=bad)
        else:
            ObservationGroupOptions(good_term=good).bad_term = bad


# ---------------------------------------------------------------------------
# A term assigned onto a constructed group is adopted and checked like a constructed one
# ---------------------------------------------------------------------------


def test_a_refused_term_assignment_leaves_the_group_as_it_was_and_says_so():
    group = ObservationGroupOptions(a=_term(), terms_order=["a"])
    before = group.a

    with pytest.raises(ValueError, match=r"Term 'b' is not listed in this group's terms_order") as excinfo:
        group.b = _term()
    assert excinfo.value.__notes__ == ["The assignment of 'b' was refused, so the group is unchanged: it has 'a'."]
    with pytest.raises(TypeError, match="'a' is neither a field"):
        group.a = 0.0

    assert group.term_keys() == ("a",) and group.a is before
    # A rolled-back entry must not stay "set", or model_dump(exclude_unset=True) would name a term that is gone.
    assert "b" not in group.__pydantic_fields_set__
    with pytest.raises(TypeError) as excinfo:
        ObservationGroupOptions().bad = 0.0
    assert excinfo.value.__notes__ == [
        "The assignment of 'bad' was refused, so the group is unchanged: it has no terms."
    ]


def test_a_declared_field_write_is_left_to_pydantic():
    group = ObservationGroupOptions(a=_term(), terms_order=["a"])

    group.enable_corruption = False
    group.terms_order = ["a", "b"]

    assert (group.enable_corruption, group.terms_order) == (False, ["a", "b"])


@pytest.mark.parametrize("name", ["b", "clone"], ids=["a-new-term", "a-name-the-class-has"])
def test_a_locked_group_refuses_a_term_write_before_anything_else(name):
    group = ObservationGroupOptions(a=_term())
    group.lock()

    with pytest.raises(RuntimeError, match="because it is locked"):
        setattr(group, name, _term())
    assert group.term_keys() == ("a",)


@pytest.mark.parametrize(
    "group_fields, declared_writes",
    [
        (dict(history_length=3), dict(concatenate_terms=False)),
        (dict(terms_order=["a"]), dict(terms_order=["a", "b", "c"])),
        (dict(terms_order=["a"]), dict(terms_order=["a", "b", "b"])),
    ],
    ids=["group-settings", "a-name-still-to-come", "a-repeat-in-the-order"],
)
def test_a_term_write_is_held_only_to_the_rules_a_term_write_can_break(group_fields, declared_writes):
    """What the declared fields break is the build's to answer; two new terms take one terms_order line, not one each."""
    group = ObservationGroupOptions(a=_term(), **group_fields)
    for field, value in declared_writes.items():
        setattr(group, field, value)

    group.b = _term()

    assert group.term_keys() == ("a", "b")


def test_a_group_history_needs_concatenated_terms():
    with pytest.raises(ValueError, match="Concatenate terms must be True"):
        ObservationGroupOptions(a=_term(), history_length=3, concatenate_terms=False)


@pytest.mark.parametrize(
    "terms, order, assigned, says, never, fix",
    [
        (
            ("a", "b"),
            ["a"],
            "c",
            [
                "Terms 'b', 'c' are not listed",
                "terms_order is currently ['a']",
                "terms_order = ['a', 'b', 'c'] puts them",
            ],
            [],
            ["a", "b", "c"],
        ),
        (
            ("a",),
            ["a", "milk_poss"],
            "milk_pos",
            [
                "which names 'milk_poss' — not a term of the group, and a near spelling of 'milk_pos'",
                "If 'milk_poss' misspells it, terms_order = ['a', 'milk_pos'] corrects it",
                "terms_order = ['a', 'milk_poss', 'milk_pos'] lists both",
            ],
            [],
            ["a", "milk_pos"],
        ),
        (
            ("a",),
            ["a", "foot_pos_r"],
            "foot_pos_l",
            ["terms_order = ['a', 'foot_pos_r', 'foot_pos_l'] lists both"],
            ["spells it the way"],
            ["a", "foot_pos_r", "foot_pos_l"],
        ),
        (
            ("a",),
            ["a", "dofs_poss", "c_future"],
            "dofs_pos",
            ["terms_order = ['a', 'dofs_pos', 'c_future'] corrects it"],
            ["group.dofs_poss"],
            ["a", "dofs_pos", "c_future"],
        ),
    ],
    ids=["every-unlisted-term", "near-miss", "left-right-pair", "typo-beside-a-name-to-come"],
)
def test_an_unlisted_assigned_term_gets_an_order_that_takes_the_write(terms, order, assigned, says, never, fix):
    """A near miss in the order may be a typo or a term still to come, so it gets both fixes; either one lands."""
    group = ObservationGroupOptions(**{name: _term() for name in terms}, terms_order=list(terms))
    group.terms_order = order

    with pytest.raises(ValueError) as excinfo:
        setattr(group, assigned, _term())
    message = str(excinfo.value)
    for text in says:
        assert text in message, message
    for text in never:
        assert text not in message, message

    group.terms_order = fix
    for name in fix:
        if name not in group.term_keys():
            setattr(group, name, _term())
    assert group.term_keys() == (*terms, *(name for name in fix if name not in terms))


@pytest.mark.parametrize(
    "terms, order, says, never, fix",
    [
        (
            ("a", "d"),
            ["a", "b", "c", "d"],
            ["Terms 'b', 'c' are named in this group's terms_order", "drops them"],
            [],
            ["a", "d"],
        ),
        ((), ["a"], ["which has no terms", "terms_order = None drops it"], [], None),
        (("dofs_pos",), ["dofs_poss"], ["spells it the way the group does"], ["ObsTerm"], ["dofs_pos"]),
        (("dofs_pos",), ["dofs_poss", "dofs_poss"], ["spells it the way the group does"], [], ["dofs_pos"]),
        (
            ("dofs_pos",),
            ["dofs_pos", "dofs_poss"],
            [
                "'dofs_poss' is a near spelling of 'dofs_pos', which terms_order already names",
                "drops it, or, if 'dofs_poss' is a term still to be written, assign it",
            ],
            ["spells it", "drops it, or assign it"],
            ["dofs_pos"],
        ),
        (
            ("dofs_pos", "x"),
            ["dofs_pos", "dofs_poss", "c", "x"],
            ["drops them, or terms_order = ['dofs_pos', 'c', 'x'] drops 'dofs_poss' and keeps 'c' for group.c"],
            ["group.dofs_poss"],
            ["dofs_pos", "x"],
        ),
        (
            ("a", "dofs_pos"),
            ["a", "dofs_poss", "c"],
            [
                "'dofs_poss' looks like 'dofs_pos' misspelled; 'c' matches no term",
                "corrects the spelling and drops 'c'",
                "terms_order = ['a', 'dofs_pos', 'c'] corrects it and keeps it for group.c",
            ],
            ["group.dofs_poss"],
            ["a", "dofs_pos"],
        ),
        (
            ("a", "dofs_pos", "milk_pos"),
            ["a", "dofs_poss", "milk_poss"],
            ["spells them the way"],
            [],
            ["a", "dofs_pos", "milk_pos"],
        ),
        (("a", "b"), ["a", "b", "a"], ["terms_order names 'a' (2 times)", "names each once"], [], ["a", "b"]),
        (("a", "b"), ["b", "a", "b", "b"], ["'b' (3 times)", "names each once"], [], ["b", "a"]),
        (("dofs_pos",), ["dofs_pos", "dofs_pos", "dofs_poss"], ["names 'dofs_pos' (2 times)"], ["spells it"], None),
        (
            ("a",),
            ["a", "zz", "zz"],
            ["Term 'zz' is named in this group's terms_order", "drops it"],
            ["names each once", "Terms 'zz', 'zz'"],
            ["a"],
        ),
    ],
    ids=[
        "names-the-group-lacks",
        "no-terms-at-all",
        "a-typo",
        "a-repeated-typo",
        "a-typo-repeating-a-listed-name",
        "a-repeating-typo-beside-a-name-matching-nothing",
        "a-typo-beside-a-name-matching-nothing",
        "two-typos",
        "a-repeat",
        "repeats-keep-first-mention-order",
        "repeats-are-named-before-typos",
        "a-repeated-name-that-is-no-term",
    ],
)
def test_a_terms_order_the_group_cannot_satisfy_is_refused_with_an_order_to_type(terms, order, says, never, fix):
    """A misspelling is fixed in the order, never by assigning the typo; a repeat would build the term twice.

    ``fix`` is the order the message offers first, so the test also constructs it back.
    """
    with pytest.raises(ValueError) as excinfo:
        ObservationGroupOptions(**{name: _term() for name in terms}, terms_order=order)
    message = str(excinfo.value)
    for text in says:
        assert text in message, message
    for text in never:
        assert text not in message, message

    if fix is not None:
        assert f"terms_order = {fix}" in message, message
        assert ObservationGroupOptions(**{name: _term() for name in terms}, terms_order=fix).terms_order == fix


# ---------------------------------------------------------------------------
# A dict that names no term, and a term named like a member of the group's class
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("given_by", ["construction", "assignment", "copy"])
def test_a_dict_that_names_no_term_is_the_misspelled_group_field_it_is(given_by):
    """A mapping without ``name`` was built into a nameless term, which died at build as ``KeyError: None``."""
    group = ObservationGroupOptions(a=_term())
    with pytest.raises(TypeError) as excinfo:
        if given_by == "construction":
            ObservationGroupOptions(a=_term(), nan_polciy={"mode": "warn"})
        elif given_by == "assignment":
            group.nan_polciy = {"mode": "warn"}
        else:
            group.model_copy(update={"nan_polciy": {"mode": "warn"}})
    message = str(excinfo.value)
    assert "'nan_polciy' is neither a field of ObservationGroupOptions nor an observation term" in message, message
    assert "names its registered term with 'name'" in message, message
    assert "Did you mean the group field 'nan_policy'?" in message, message
    if given_by == "assignment":
        assert group.term_keys() == ("a",)


def test_a_term_field_written_on_the_group_as_a_dict_is_pointed_at_the_term():
    with pytest.raises(TypeError, match=r"'noise' is set per term"):
        ObservationGroupOptions(noise={"operation": "add"})


def test_a_serialized_nameless_term_still_loads():
    """The type metadata marks a *saved* placeholder, which the observation manager refuses at build instead."""
    group = ObservationGroupOptions(placeholder=ObservationTermOptions().model_dump())

    assert isinstance(group.placeholder, ObservationTermOptions) and group.placeholder.name is None


@pytest.mark.parametrize("name", ["lock", "clone", "keys", "term_keys"])
def test_a_term_cannot_take_a_name_the_group_class_already_has(name):
    """Constructed, the term hid the member (``config.lock()`` then called the term); assigned, it was never a term."""
    with pytest.raises(TypeError) as excinfo:
        ObservationGroupOptions(**{name: _term()})
    message = str(excinfo.value)
    assert f"'{name}' cannot name a term of ObservationGroupOptions" in message, message
    assert "hides it on this object" in message, message
    assert f"rename it, e.g. '{name}_term'" in message, message

    group = ObservationGroupOptions(a=_term())
    with pytest.raises(TypeError, match="never adds a term"):
        setattr(group, name, _term())
    assert group.term_keys() == ("a",)
    assert not isinstance(getattr(group, name), ObservationTermOptions), "the refused write must not shadow it"
