"""Tests for ``eden.options.managers.rewards``: which entries ``RewardManagerOptions`` / ``RewardGroupOptions``
accept, and ``disable_term`` across flat terms and groups. No scene; what a manager built from them
computes is pinned in ``tests/managers/test_reward_manager.py``.
"""

from __future__ import annotations

import pytest

from eden.managers.reward_manager import REWARD_TERM_REGISTRY
from eden.options.managers.rewards import RewardGroupOptions, RewardManagerOptions, RewardTermOptions


@REWARD_TERM_REGISTRY.register()
def reward_options_test_const(env, value: float = 1.0):
    return value


def _const(weight: float = 1.0) -> RewardTermOptions:
    return RewardTermOptions(name="reward_options_test_const", weight=weight)


# ---------------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------------


def test_manager_options_take_terms_or_groups_and_groups_take_terms_only():
    with pytest.raises(TypeError, match="RewardTermOptions or RewardGroupOptions"):
        RewardManagerOptions(bogus=3)
    with pytest.raises(TypeError, match="RewardTermOptions"):
        RewardGroupOptions(bogus=RewardGroupOptions())


def test_a_flat_override_of_a_grouped_term_names_the_group():
    """``track={...}`` where ``track`` lives in a group is the pre-groups override syntax: a hard error on
    every construction path, since an entry that failed type recovery cannot be kept.
    """
    with pytest.raises(TypeError, match="inside group 'rg'"):
        RewardManagerOptions(rg=RewardGroupOptions(track=_const()), track={"weight": 5.0})


def test_a_raw_dict_naming_no_nested_term_gets_the_generic_type_error():
    with pytest.raises(TypeError, match="must be an instance of RewardTermOptions or RewardGroupOptions"):
        RewardManagerOptions(rg={"track": {"weight": 1.0}})


# ---------------------------------------------------------------------------
# disable_term
# ---------------------------------------------------------------------------


def test_disable_term_reaches_into_groups_and_removes_a_whole_group_by_name():
    options = RewardManagerOptions(
        alive=_const(), rg=RewardGroupOptions(track=_const(), rot=_const()), rs=RewardGroupOptions(style=_const())
    )

    options.disable_term("track")  # nested in a group
    options.disable_term("alive")  # flat
    options.disable_term("rs")  # an unambiguous top-level hit removes the whole group

    assert options.term_keys() == ("rg",)
    assert options.rg.term_keys() == ("rot",)
    with pytest.raises(KeyError):
        options.disable_term("missing")


def test_disable_term_refuses_a_name_several_entries_answer_to():
    """First-match removal could drop an inert weight-0 copy while the active one keeps training."""
    duplicated = RewardManagerOptions(rg=RewardGroupOptions(track=_const(0.0)), rs=RewardGroupOptions(track=_const()))
    with pytest.raises(ValueError, match="ambiguous.*rg.*rs"):
        duplicated.disable_term("track")
    duplicated.rs.disable_term("track")  # disambiguated through the owning group
    assert (duplicated.rs.term_keys(), duplicated.rg.term_keys()) == ((), ("track",))

    shadowed = RewardManagerOptions(style=RewardGroupOptions(a=_const()), rg=RewardGroupOptions(style=_const()))
    with pytest.raises(ValueError, match="ambiguous.*top level.*rg"):
        shadowed.disable_term("style")


def test_disable_term_never_touches_serialization_metadata():
    options = RewardManagerOptions(alive=_const())

    with pytest.raises(KeyError):
        options.disable_term("_option_module_")
    assert "_option_module_" in (options.__pydantic_extra__ or {})


def test_a_locked_manager_refuses_a_nested_disable():
    """``lock()`` recurses into the groups."""
    options = RewardManagerOptions(rg=RewardGroupOptions(track=_const()))
    options.lock()

    with pytest.raises(RuntimeError, match="locked"):
        options.disable_term("track")
    assert "track" in options.rg.term_keys()
