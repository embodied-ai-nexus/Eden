"""Tests for ``eden/__main__.py``: the ``COMMANDS`` table, ``main()``'s dispatch, and ``--log-theme``.

``main()`` runs in-process over a patched ``sys.argv``, on commands cheap enough to run for real.
"""

import argparse
import json
import os
import sys

import pytest

import eden as en
import eden.__main__ as m
from eden._logging import LOG_THEME_ENV_VAR, resolve_theme
from eden.utils.cli.registry import _add_command


@pytest.fixture(autouse=True)
def _restore_theme(monkeypatch):
    """``main()`` writes ``os.environ`` and re-themes ``eden.logger`` itself; register both restores first."""
    monkeypatch.setenv(LOG_THEME_ENV_VAR, "agent")
    previous = en.logger.theme
    yield
    en.logger.set_theme(previous)


def _every_command(commands):
    """Every command in the tree, groups and the children of nested groups included."""
    for cmd in commands:
        yield cmd
        yield from _every_command(cmd.subcommands)


def _run_main(monkeypatch, *argv):
    """Run ``eden <argv>`` in-process and return the ``SystemExit`` code, or ``None`` if it returned."""
    monkeypatch.setattr(sys, "argv", ["eden", *argv])
    try:
        m.main()
    except SystemExit as exit_:
        return exit_.code
    return None


class TestCommandTable:
    def test_names_are_unique_and_the_whole_table_builds_onto_argparse(self):
        names = [c.name for c in m.COMMANDS]
        assert len(names) == len(set(names)), f"duplicate command names: {names}"

        parsers: dict = {}
        subparsers = argparse.ArgumentParser().add_subparsers(dest="command")
        for cmd in m.COMMANDS:
            _add_command(subparsers, cmd, parsers)
        assert len(parsers) == len(list(_every_command(m.COMMANDS)))

    def test_exactly_these_commands_forward_leftover_args(self):
        """Hand-maintained on purpose: a new name here means a command now swallows typos instead of rejecting them."""
        assert {c.name for c in m.COMMANDS if c.consumes_extra} == {
            "train",
            "inference",
            "export",
            "deploy",
            "preprocess",
            "smoke",
            "datagen",
            "examples",
            "gello",
            "lerobot",
            "metaquest",
            "pico",
            "pose",
            "retarget",
            "teleop",
            "wuji",
        }

    def test_group_names_are_unique_across_the_tree(self):
        """A group's chosen child lands in ``<name>_command``: two groups sharing a name at any depth would collide."""
        groups = [cmd.name for cmd in _every_command(m.COMMANDS) if cmd.subcommands]
        assert len(groups) == len(set(groups)), f"duplicate group names: {groups}"

    @pytest.mark.parametrize(
        ("group", "children"),
        [("retarget", {"omniretarget", "otretarget", "soma", "dex", "flashdex"}), ("pose", {"gvhmr"})],
    )
    def test_backend_groups_forward_to_one_child_per_backend(self, group, children):
        """Each backend parses its own args, so ``--help`` works without its optional extra installed; a backend with
        several commands (``retarget flashdex import|kinematic``) is a group of such children.
        """
        by_name = {c.name: c for c in next(c for c in m.COMMANDS if c.name == group).subcommands}
        assert set(by_name) == children
        assert all(leaf.forwards_extra for leaf in _every_command(by_name.values()) if not leaf.subcommands)


class TestDispatch:
    def test_a_group_without_a_subcommand_prints_its_help_and_exits_zero(self, monkeypatch, capsys):
        assert _run_main(monkeypatch, "gello") == 0
        assert "gello subcommands" in capsys.readouterr().out

    def test_a_self_parsed_child_of_a_group_gets_its_namespace(self, monkeypatch, capsys):
        """The regression: the group used to call every child with the leftover args, not the namespace."""
        assert _run_main(monkeypatch, "examples", "list", "--tier", "project", "--json") is None
        records = json.loads(capsys.readouterr().out)
        assert records and {record["tier"] for record in records} == {"project"}

    def test_a_forwarding_command_gets_the_leftover_args(self, monkeypatch, capsys):
        assert _run_main(monkeypatch, "train", "--help") == 0
        assert "Train a registered task" in capsys.readouterr().out

    def test_a_forwarding_child_of_a_group_gets_the_leftover_args(self, monkeypatch, capsys):
        assert _run_main(monkeypatch, "retarget", "soma", "--help") == 0
        assert "--body-model" in capsys.readouterr().out

    def test_a_nested_group_dispatches_to_its_forwarding_leaf(self, monkeypatch, capsys):
        """``eden teleop sonic robot``: two group levels, then the leaf parses its own args."""
        assert _run_main(monkeypatch, "teleop", "sonic", "robot", "--help") == 0
        assert "usage: eden teleop sonic robot" in capsys.readouterr().out

    def test_a_nested_group_without_a_subcommand_prints_its_own_help(self, monkeypatch, capsys):
        assert _run_main(monkeypatch, "teleop", "sonic") == 0
        out = capsys.readouterr().out
        assert "sonic subcommands" in out and "operator" in out

    def test_a_non_forwarding_command_rejects_unknown_args(self, monkeypatch, capsys):
        assert _run_main(monkeypatch, "list", "--collison") == 2
        assert "unrecognized arguments: --collison" in capsys.readouterr().err


class TestLogThemeFlag:
    """``eden --log-theme``: the CLI spelling of ``EDEN_LOG_THEME``."""

    def test_it_sets_the_variable_the_rest_of_eden_reads(self, monkeypatch, capsys):
        """The theme has three consumers (``eden.init``, help formatters, subprocesses) and no one call site."""
        monkeypatch.delenv(LOG_THEME_ENV_VAR)

        assert _run_main(monkeypatch, "--log-theme", "human") == 0

        assert os.environ[LOG_THEME_ENV_VAR] == "human"
        assert resolve_theme() == "human"

    @pytest.mark.parametrize(("argv", "expected"), [(["--log-theme", "agent"], "agent"), ([], "human")])
    def test_a_flag_beats_an_inherited_variable_and_its_absence_leaves_it_alone(
        self, monkeypatch, capsys, argv, expected
    ):
        monkeypatch.setenv(LOG_THEME_ENV_VAR, "human")

        assert _run_main(monkeypatch, *argv) == 0

        assert os.environ[LOG_THEME_ENV_VAR] == expected

    @pytest.mark.parametrize(
        "argv",
        [
            ["list", "--log-theme", "human"],
            ["train", "--task", "reacher", "--log-theme=human"],
            ["list", "--log-th", "human"],
            ["train", "--task", "reacher", "--log=human"],
        ],
        ids=["plain-subcommand", "forwarding-subcommand", "abbreviated", "abbreviated-with-equals"],
    )
    def test_the_flag_after_the_subcommand_says_where_it_goes(self, monkeypatch, capsys, argv):
        """After the subcommand it belongs to that subparser; argparse accepts any unambiguous prefix."""
        assert _run_main(monkeypatch, *argv) != 0

        message = capsys.readouterr().err
        assert "came after the subcommand" in message
        assert f"eden --log-theme <auto|human|agent> {argv[0]}" in message
        assert LOG_THEME_ENV_VAR in message

    def test_a_bare_double_dash_is_not_mistaken_for_it(self, monkeypatch, capsys):
        """``--`` is a prefix of every long option, so a naive prefix check claims it."""
        _run_main(monkeypatch, "list", "--", "extra")

        assert "came after the subcommand" not in capsys.readouterr().err

    def test_an_unknown_theme_is_refused_by_argparse(self, monkeypatch, capsys):
        assert _run_main(monkeypatch, "--log-theme", "humam", "list") == 2
        assert "invalid choice: 'humam'" in capsys.readouterr().err
