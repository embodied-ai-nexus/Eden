"""Tests for ``eden.utils.cli.registry.Command``: contradictory command shapes are refused at construction."""

import pytest

from eden.utils.cli.registry import Command


def _noop(_):
    return None


_LEAF = Command("leaf", "h", handler=_noop, forwards_extra=True)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"handler": _noop, "subcommands": (_LEAF,)}, "only dispatches to children"),
        ({"configure": _noop, "subcommands": (_LEAF,)}, "only dispatches to children"),
        ({"forwards_extra": True, "subcommands": (_LEAF,)}, "only dispatches to children"),
        ({}, "needs a handler"),
        ({"handler": _noop, "forwards_extra": True, "configure": _noop}, "parses its own args"),
    ],
    ids=[
        "group-with-handler",
        "group-with-configure",
        "group-forwarding",
        "leaf-without-handler",
        "forwarding-configured",
    ],
)
def test_a_contradictory_shape_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Command("x", "h", **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"handler": _noop},
        {"handler": _noop, "configure": _noop},
        {"handler": _noop, "forwards_extra": True},
        {"subcommands": (_LEAF,)},
    ],
    ids=["no-arg", "self-parsed", "forwarding", "group"],
)
def test_the_four_valid_shapes_construct(kwargs):
    Command("x", "h", **kwargs)
