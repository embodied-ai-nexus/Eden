"""String helpers: glob name-matching, and rendering name lists and term subjects for a log line."""

# Modified from isaaclab/utils/string.py

from collections.abc import Collection, Iterable, Sequence
from typing import Any
import fnmatch
import re


_GLOB_WILDCARD_PATTERN = re.compile(r"(?<!\\)(\*|\?)")
_REGEX_INDICATOR_PATTERN = re.compile(r"(?<!\\)[\(\)\{\}\|\+\^\$]")
_DOT_STAR_PATTERN = re.compile(r"(?<!\\)\.\*")


def is_glob_pattern(pattern: str) -> bool:
    """Best-effort heuristic to determine whether the provided pattern looks like a glob."""
    if not isinstance(pattern, str):
        return False
    has_glob_wildcards = bool(_GLOB_WILDCARD_PATTERN.search(pattern))
    if not has_glob_wildcards:
        return False
    if _REGEX_INDICATOR_PATTERN.search(pattern):
        return False
    if _DOT_STAR_PATTERN.search(pattern):
        return False
    return True


def name_matches(pattern: str, name: str, use_glob: bool | None = None) -> bool:
    """Whether ``pattern``, read as a pattern, matches ``name``.

    A glob (per :func:`is_glob_pattern`, unless ``use_glob`` forces the mode) is matched with
    :func:`fnmatch.fnmatchcase`; anything else is a regular expression that must match the whole
    name. This is the pattern half of :func:`key_selects` only: it sees one name, so it cannot take a
    key that is exactly one of the names literally. To ask what a config key selects, use
    :func:`key_selects`.

    Parameters
    ----------
    pattern: str
        The glob or regular expression.
    name: str
        The candidate name.
    use_glob: bool | None
        ``True``/``False`` forces glob/regex matching; ``None`` infers it from ``pattern``.

    Returns
    -------
    bool
        ``True`` when ``pattern`` matches all of ``name``.
    """
    glob = is_glob_pattern(pattern) if use_glob is None else use_glob
    if glob:
        return fnmatch.fnmatchcase(name, pattern)
    return re.fullmatch(pattern, name) is not None


def key_selects(key: str, name: str, names: Collection[str], use_glob: bool | None = None) -> bool:
    """Whether the config key ``key`` selects ``name``, one of ``names``.

    A key that is exactly one of ``names`` selects that name alone; any other key is a glob or a regex
    (:func:`name_matches`). A name can carry a pattern metacharacter (``pin.1``, ``arm(2)``, an
    exporter's ``link.001``): read as a regex, ``pin.1`` would also select ``pin_1``, and ``arm(2)``
    would select nothing, not even itself. :func:`resolve_matching_names` and
    :func:`resolve_matching_names_values` select with this test, so a caller matching keys one name at
    a time agrees with them.

    Parameters
    ----------
    key: str
        The name, glob or regex.
    name: str
        The candidate name.
    names: Collection[str]
        Every name the key could select, which decides whether ``key`` is one of them. Pass a set when
        asking about many names.
    use_glob: bool | None
        ``True``/``False`` forces glob/regex matching of a pattern key; ``None`` infers it.

    Returns
    -------
    bool
        ``True`` when ``key`` selects ``name``.
    """
    if key in names:
        return key == name
    return name_matches(key, name, use_glob=use_glob)


def first_few(names: Iterable[str], limit: int = 5) -> str:
    """Render ``names`` for a log line, truncating a long list to its first few plus a count.

    For messages that name the joints/DOFs/links behind a diagnostic: a 20-DOF hand or a
    29-DOF humanoid would otherwise wrap the console, and past the first few entries the
    list stops being read anyway. The count of what is hidden is kept, so the reader can
    tell "a couple of joints" from "the whole robot" without the full list.

    Parameters
    ----------
    names:
        The names to render, in the order they should appear. Any iterable: callers pass
        whatever the entity handed them, which is as often ``dict.keys()`` as a list.
    limit:
        How many names to show before truncating.

    Returns
    -------
    str
        ``"['a', 'b']"`` when the list fits, ``"['a', 'b'] (+3 more)"`` when it does not.
    """
    names = list(names)
    if len(names) <= limit:
        return f"{names}"
    return f"{names[:limit]} (+{len(names) - limit} more)"


def term_subject(term: Any, entity_name: str | None = None) -> str:
    """Render the subject an entity-bound term opens its errors with: ``"<Class> on entity '<name>'"``.

    The same class can be configured twice against different entities (one reward per arm,
    one randomization per object), so the class alone does not say which of them failed.
    Class and entity only: ``ManagerBase._build_term`` already adds the config key as a note,
    and a term that restated it would print it twice.

    Parameters
    ----------
    term:
        A term that declares ``entity_name``. A term with no entity (``time_out``, most
        commands) names its class alone and does not call this.
    entity_name:
        The entity, for a term that names it in a field of its own (a writing metric's
        ``tool_name``, a whiteboard metric's ``marker_name``); ``term.entity_name`` by default.

    Returns
    -------
    str
        ``"FeetAirTime on entity 'robot'"``.
    """
    return f"{type(term).__name__} on entity '{term.entity_name if entity_name is None else entity_name}'"


def resolve_matching_names(
    keys: str | Sequence[str],
    list_of_strings: Sequence[str],
    preserve_order: bool = False,
    use_glob: bool | None = None,
    *,
    literal_names: Collection[str] | None = None,
) -> tuple[list[int], list[str]]:
    """Match a list of query patterns against a list of strings and return the matched indices and names.

    When a list of query regular expressions or glob patterns is provided, the function checks each target string
    against each query and returns the indices of the matched strings and the matched strings.

    If the :attr:`preserve_order` is True, the ordering of the matched indices and names follows the order of
    the query patterns: all matches for the first pattern come first, then the second, and so on.

    If the :attr:`preserve_order` is False, the ordering of the matched indices and names is the same as the
    order of the provided list of strings.

    For example, consider the list of strings is ['a', 'b', 'c', 'd', 'e'] and the patterns are ['a|c', 'b'].
    If :attr:`preserve_order` is False, then the function will return the indices of the matched strings and the
    strings as: ([0, 1, 2], ['a', 'b', 'c']). When :attr:`preserve_order` is True, it will return them as:
    ([0, 2, 1], ['a', 'c', 'b']).

    A key that is exactly one of the strings selects that string alone, even when it contains a pattern
    character: ``pin.1`` does not also select ``pin_1``, and ``arm(2)`` selects ``arm(2)`` (:func:`key_selects`).

    Note:
        The function does not sort the indices. It returns the indices in the order they are found.

    Parameters
    ----------
    keys: str | Sequence[str]
        A regular expression, glob pattern, or a list of either to match the strings in the list.
    list_of_strings: Sequence[str]
        A list of strings to match.
    preserve_order: bool
        Whether to preserve the order of the query keys in the returned values. Defaults to False.
    use_glob: bool | None
        Optional override for the matching strategy. When True, treat all keys as Unix shell-style glob
        patterns; when False, treat them as regular expressions. When None, automatically infer the mode per key
        using :func:`is_glob_pattern`. Defaults to None.
    literal_names: Collection[str] | None
        The names a key is taken literally against, when ``list_of_strings`` is a scope cut from a larger set
        (an entity's controlled joints out of all its joints). A key that is exactly one of them selects that
        name or, outside the scope, nothing; it never falls back to a pattern that selects a neighbour
        (``pin.1`` passive, ``pin_1`` in scope). Defaults to ``list_of_strings``.

    Returns
    -------
    tuple[list[int], list[str]]
        A tuple of lists containing the matched indices and names.

    Raises
    ------
    ValueError
        When multiple matches are found for a string in the list.
    ValueError
        When not all query patterns are matched.
    """
    # resolve name keys
    if isinstance(keys, str):
        keys = [keys]
    # find matching patterns
    index_list = []
    names_list = []
    key_idx_list = []
    # book-keeping to check that we always have a one-to-one mapping
    # i.e. each target string should match only one pattern
    target_strings_match_found = [None for _ in range(len(list_of_strings))]
    keys_match_found = [[] for _ in range(len(keys))]
    names = set(list_of_strings if literal_names is None else literal_names)
    # loop over all target strings
    for target_index, potential_match_string in enumerate(list_of_strings):
        for key_index, re_key in enumerate(keys):
            if key_selects(re_key, potential_match_string, names, use_glob=use_glob):
                # check if match already found
                if target_strings_match_found[target_index]:
                    raise ValueError(
                        f"Multiple matches for '{potential_match_string}':"
                        f" '{target_strings_match_found[target_index]}' and '{re_key}'!"
                    )
                # add to list
                target_strings_match_found[target_index] = re_key
                index_list.append(target_index)
                names_list.append(potential_match_string)
                key_idx_list.append(key_index)
                # add for pattern key
                keys_match_found[key_index].append(potential_match_string)
    # reorder keys if they should be returned in order of the query keys
    if preserve_order:
        reordered_index_list = [None] * len(index_list)
        global_index = 0
        for key_index in range(len(keys)):
            for key_idx_position, key_idx_entry in enumerate(key_idx_list):
                if key_idx_entry == key_index:
                    reordered_index_list[key_idx_position] = global_index
                    global_index += 1
        # reorder index and names list
        index_list_reorder = [None] * len(index_list)
        names_list_reorder = [None] * len(index_list)
        for idx, reorder_idx in enumerate(reordered_index_list):
            index_list_reorder[reorder_idx] = index_list[idx]
            names_list_reorder[reorder_idx] = names_list[idx]
        # update
        index_list = index_list_reorder
        names_list = names_list_reorder
    # check that all query patterns are matched
    if not all(keys_match_found):
        # make this print nicely aligned for debugging
        msg = "\n"
        for key, value in zip(keys, keys_match_found):
            msg += f"\t{key}: {value}\n"
        msg += f"Available strings: {list_of_strings}\n"
        # raise error
        raise ValueError(
            f"Not all query patterns are matched! Please check that the provided patterns are correct: {msg}"
        )
    # return
    return index_list, names_list


def columns_by_name(available_names: Sequence[Any], wanted_names: Sequence[str], *, what: str = "names") -> list[int]:
    """Column indices of ``wanted_names`` within ``available_names`` (exact match, query order).

    Decodes a motion file's columns by the names it stores beside them — a motion ``.npz``'s ``body_names`` /
    ``dof_names``, a BVH's joints — or by a dataset's documented column order, so the mapping is independent of the
    writer's column order and of Genesis's internal link/DOF ordering (which changed breadth-first -> depth-first in
    genesis 1.2.0). ``available_names`` entries are coerced to ``str`` (npz string arrays load as numpy ``str_``).

    Unlike :func:`resolve_matching_names`, this is a plain exact-name lookup (no glob/regex), so
    names containing pattern metacharacters are matched literally.

    Parameters
    ----------
    available_names:
        The column labels, in column order (e.g. the npz ``body_names``).
    wanted_names:
        The names to locate, in the desired output order.
    what:
        What ``available_names`` are, for the error message only (e.g. ``"motion npz 'body_names'"``).

    Returns
    -------
    list[int]
        One column index per entry in ``wanted_names``, in that order.

    Raises
    ------
    ValueError
        If any of ``wanted_names`` is absent from ``available_names``.
    """
    available = [str(n) for n in available_names]
    position = {name: i for i, name in enumerate(available)}
    missing = [n for n in wanted_names if n not in position]
    if missing:
        raise ValueError(f"{what} is missing {missing}; available: {available}")
    return [position[n] for n in wanted_names]


def resolve_matching_names_values(
    data: dict[str, Any],
    list_of_strings: Sequence[str],
    preserve_order: bool = False,
    use_glob: bool | None = None,
    *,
    literal_names: Collection[str] | None = None,
) -> tuple[list[int], list[str], list[Any]]:
    """Match query patterns against a list of strings, returning matched indices, names, and values.

    If the :attr:`preserve_order` is True, the ordering of the matched indices, names, and values follows the
    order of the query patterns (the dictionary keys): all matches for the first key come first, then the
    second, and so on. The returned values are NOT positionally aligned with the target list — scatter them
    through the returned indices instead.

    If the :attr:`preserve_order` is False, the ordering of the matched indices and names is the same as the
    order of the provided list of strings.

    As in :func:`resolve_matching_names`, a key that is exactly one of the strings selects that string alone.

    For example, consider the dictionary is {"a|d|e": 1, "b|c": 2}, the list of strings is
    ['a', 'b', 'c', 'd', 'e']. If :attr:`preserve_order` is False, then the function will return the indices of the
    matched strings, the matched strings, and the values as: ([0, 1, 2, 3, 4], ['a', 'b', 'c', 'd', 'e'],
    [1, 2, 2, 1, 1]). When :attr:`preserve_order` is True, it will return them as: ([0, 3, 4, 1, 2],
    ['a', 'd', 'e', 'b', 'c'], [1, 1, 1, 2, 2]).

    Parameters
    ----------
    data: dict[str, Any]
        A dictionary mapping query patterns (regular expressions or globs) to values to match against the list.
    list_of_strings: Sequence[str]
        A list of strings to match.
    preserve_order: bool
        Whether to preserve the order of the query keys in the returned values. Defaults to False.
    use_glob: bool | None
        Optional override for the matching strategy. When True, treat all dictionary keys as glob patterns;
        when False, treat them as regular expressions. When None, automatically infer the mode per key using
        :func:`is_glob_pattern`. Defaults to None.
    literal_names: Collection[str] | None
        The names a key is taken literally against, as in :func:`resolve_matching_names`. Defaults to
        ``list_of_strings``.

    Returns
    -------
    tuple[list[int], list[str], list[Any]]
        A tuple of lists containing the matched indices, names, and values.

    Raises
    ------
    TypeError
        When the input argument :attr:`data` is not a dictionary.
    ValueError
        When multiple matches are found for a string in the dictionary.
    ValueError
        When not all query patterns in the data keys are matched.
    """
    # check valid input
    if not isinstance(data, dict):
        raise TypeError(f"Input argument `data` should be a dictionary. Received: {data}")
    # find matching patterns
    index_list = []
    names_list = []
    values_list = []
    key_idx_list = []
    # book-keeping to check that we always have a one-to-one mapping
    # i.e. each target string should match only one pattern
    target_strings_match_found = [None for _ in range(len(list_of_strings))]
    keys_match_found = [[] for _ in range(len(data))]
    names = set(list_of_strings if literal_names is None else literal_names)
    # loop over all target strings
    for target_index, potential_match_string in enumerate(list_of_strings):
        for key_index, (re_key, value) in enumerate(data.items()):
            if key_selects(re_key, potential_match_string, names, use_glob=use_glob):
                # check if match already found
                if target_strings_match_found[target_index]:
                    raise ValueError(
                        f"Multiple matches for '{potential_match_string}':"
                        f" '{target_strings_match_found[target_index]}' and '{re_key}'!"
                    )
                # add to list
                target_strings_match_found[target_index] = re_key
                index_list.append(target_index)
                names_list.append(potential_match_string)
                values_list.append(value)
                key_idx_list.append(key_index)
                # add for pattern key
                keys_match_found[key_index].append(potential_match_string)
    # reorder keys if they should be returned in order of the query keys
    if preserve_order:
        reordered_index_list = [None] * len(index_list)
        global_index = 0
        for key_index in range(len(data)):
            for key_idx_position, key_idx_entry in enumerate(key_idx_list):
                if key_idx_entry == key_index:
                    reordered_index_list[key_idx_position] = global_index
                    global_index += 1
        # reorder index and names list
        index_list_reorder = [None] * len(index_list)
        names_list_reorder = [None] * len(index_list)
        values_list_reorder = [None] * len(index_list)
        for idx, reorder_idx in enumerate(reordered_index_list):
            index_list_reorder[reorder_idx] = index_list[idx]
            names_list_reorder[reorder_idx] = names_list[idx]
            values_list_reorder[reorder_idx] = values_list[idx]
        # update
        index_list = index_list_reorder
        names_list = names_list_reorder
        values_list = values_list_reorder
    # check that all query patterns are matched
    if not all(keys_match_found):
        # make this print nicely aligned for debugging
        msg = "\n"
        for key, value in zip(data.keys(), keys_match_found):
            msg += f"\t{key}: {value}\n"
        msg += f"Available strings: {list_of_strings}\n"
        # raise error
        raise ValueError(
            f"Not all query patterns are matched! Please check that the provided patterns are correct: {msg}"
        )
    # return
    return index_list, names_list, values_list
