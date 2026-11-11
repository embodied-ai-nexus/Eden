"""Tests for ``eden.utils.string``: exact column lookup, glob/regex name matching, and name-list rendering."""

import numpy as np
import pytest

from eden.utils.string import (
    columns_by_name,
    first_few,
    key_selects,
    name_matches,
    resolve_matching_names,
    resolve_matching_names_values,
    term_subject,
)


def test_columns_by_name_is_exact_and_order_preserving():
    avail = ["pelvis", "torso_link", "left_foot", "right_foot"]
    assert columns_by_name(avail, ["right_foot", "pelvis"]) == [3, 0]


def test_columns_by_name_coerces_numpy_strings():
    # npz string arrays load as numpy str_, not python str
    assert columns_by_name(np.array(["a", "b", "c"]), ["c", "a"]) == [2, 0]


def test_columns_by_name_names_what_it_searched():
    with pytest.raises(ValueError, match=r"^BVH 'a\.bvh' joints is missing \['knee'\]; available: \['hip'\]$"):
        columns_by_name(["hip"], ["knee"], what="BVH 'a.bvh' joints")


def test_columns_by_name_matches_literally_not_as_glob():
    assert columns_by_name(["a*b", "ab", "axb"], ["a*b"]) == [0]


@pytest.mark.parametrize(
    ("names", "kwargs", "expected"),
    [
        (["j0", "j1"], {}, "['j0', 'j1']"),
        ([f"f{i}" for i in range(8)], {}, "['f0', 'f1', 'f2', 'f3', 'f4'] (+3 more)"),
        (["a", "b", "c"], {"limit": 3}, "['a', 'b', 'c']"),
        (["a", "b", "c"], {"limit": 2}, "['a', 'b'] (+1 more)"),
        (("j0", "j1"), {}, "['j0', 'j1']"),
        ({"j0": 0, "j1": 1}.keys(), {}, "['j0', 'j1']"),
        ([], {}, "[]"),
    ],
    ids=["short", "truncated", "exactly-limit", "one-hidden", "tuple", "dict-keys", "empty"],
)
def test_first_few_renders_any_sequence_as_a_truncated_list(names, kwargs, expected):
    assert first_few(names, **kwargs) == expected


@pytest.mark.parametrize(
    ("pattern", "name", "expected"),
    [
        ("left_*", "left_knee_joint", True),  # glob
        (".*_knee_.*", "left_knee_joint", True),  # regex
        ("knee", "left_knee_joint", False),  # a regex must match the whole name
        ("left_hip_(pitch|roll)_joint", "left_hip_yaw_joint", False),
        ("joint_[23]", "joint_3", True),
    ],
)
def test_name_matches_agrees_with_resolve_matching_names(pattern, name, expected):
    """``name_matches`` answers what ``resolve_matching_names`` would match, without raising on a miss."""
    assert name_matches(pattern, name) is expected
    if expected:
        assert resolve_matching_names(pattern, [name]) == ([0], [name])
    else:
        with pytest.raises(ValueError, match="Not all query patterns are matched"):
            resolve_matching_names(pattern, [name])


def test_name_matches_honors_a_forced_mode():
    assert name_matches("a.c", "abc", use_glob=False)
    assert not name_matches("a.c", "abc", use_glob=True)


# ``pin.1`` is also a regex matching ``pin_1``; ``arm(2)`` is one that does not match itself.
_PATTERN_LIKE_NAMES = ["pin.1", "pin_1", "arm(2)"]


@pytest.mark.parametrize(("key", "index"), [("pin.1", 0), ("pin_1", 1), ("arm(2)", 2)])
def test_a_key_that_is_exactly_a_name_selects_that_name_alone(key, index):
    assert resolve_matching_names(key, _PATTERN_LIKE_NAMES) == ([index], [key])
    assert resolve_matching_names_values({key: 1.0}, _PATTERN_LIKE_NAMES) == ([index], [key], [1.0])


def test_a_key_that_names_nothing_exactly_is_still_a_pattern():
    assert resolve_matching_names("pin.*", _PATTERN_LIKE_NAMES) == ([0, 1], ["pin.1", "pin_1"])
    assert resolve_matching_names_values({"pin?1": 2.0}, _PATTERN_LIKE_NAMES) == (
        [0, 1],
        ["pin.1", "pin_1"],
        [2.0, 2.0],
    )


def test_exact_keys_can_spell_out_every_name_without_overlapping():
    """Read as patterns, ``pin.1`` and ``pin_1`` both select ``pin_1``, which raised as a multiple match."""
    assert resolve_matching_names(_PATTERN_LIKE_NAMES, _PATTERN_LIKE_NAMES, preserve_order=True) == (
        [0, 1, 2],
        _PATTERN_LIKE_NAMES,
    )


@pytest.mark.parametrize(
    ("key", "selected"),
    [("pin.1", ["pin.1"]), ("arm(2)", ["arm(2)"]), ("pin?1", ["pin.1", "pin_1"]), ("pin.*", ["pin.1", "pin_1"])],
    ids=["exact-regex-like", "exact-unmatchable-as-regex", "glob", "regex"],
)
def test_key_selects_is_the_resolvers_test_one_name_at_a_time(key, selected):
    names = set(_PATTERN_LIKE_NAMES)
    assert [name for name in _PATTERN_LIKE_NAMES if key_selects(key, name, names)] == selected
    assert resolve_matching_names(key, _PATTERN_LIKE_NAMES)[1] == selected


def test_key_selects_decides_literalness_against_every_name_it_is_given():
    """``pin.1`` is literal among names that hold it, so it does not fall back to a pattern on a subset without it."""
    assert not key_selects("pin.1", "pin_1", set(_PATTERN_LIKE_NAMES))
    assert key_selects("pin.1", "pin_1", {"pin_1"})


def test_a_scoped_resolve_judges_literalness_against_the_names_it_is_cut_from():
    """``pin.1`` outside the scope selects nothing, rather than its in-scope neighbour ``pin_1`` as a regex."""
    assert resolve_matching_names("pin.1", ["pin_1"]) == ([0], ["pin_1"])
    with pytest.raises(ValueError, match="Not all query patterns are matched"):
        resolve_matching_names("pin.1", ["pin_1"], literal_names=_PATTERN_LIKE_NAMES)
    assert resolve_matching_names("pin?1", ["pin_1"], literal_names=_PATTERN_LIKE_NAMES) == ([0], ["pin_1"])


def test_term_subject_names_the_entity_from_the_term_or_as_given():
    class Reach:
        entity_name = "robot"

    class Writing:  # a term that names its entity in a field of its own (tool_name)
        tool_name = "brush"

    assert term_subject(Reach()) == "Reach on entity 'robot'"
    assert term_subject(Writing(), Writing.tool_name) == "Writing on entity 'brush'"
