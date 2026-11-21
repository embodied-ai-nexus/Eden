"""Tests for ``eden.utils.cli.maintenance``: ``eden list`` and its ``--tag`` filter."""

from __future__ import annotations

import subprocess
import sys

import pytest

from eden.__main__ import main as eden_main


def _table_rows(out: str) -> list[list[str]]:
    """The cells of each body row of an agent-theme table (``  cell | cell``)."""
    return [[cell.strip() for cell in line.split("|")] for line in out.splitlines() if line.startswith("  ")]


def test_eden_list_filters_by_tag(monkeypatch, capsys):
    monkeypatch.setenv("EDEN_LOG_THEME", "agent")  # one "name | source | tags | modifiers" line per task
    monkeypatch.setattr(sys, "argv", ["eden", "list", "--tag", "manipulation", "--tag", "dexterous"])
    eden_main()
    cells = {row[0]: row[1:] for row in _table_rows(capsys.readouterr().out)}

    assert cells["reorient"] == ["eden", "manipulation, dexterous", "HandSideMod"]
    assert "reacher" not in cells  # manipulation, but no multi-fingered hand
    assert all("dexterous" in tags.split(", ") for _, tags, _ in cells.values())


def test_eden_list_takes_a_repeated_tag_as_one(monkeypatch, capsys):
    monkeypatch.setenv("EDEN_LOG_THEME", "agent")
    monkeypatch.setattr(sys, "argv", ["eden", "list", "--tag", "deformable", "--tag", "deformable"])
    eden_main()
    names = sorted(row[0] for row in _table_rows(capsys.readouterr().out))
    assert names == ["mochi_fold_the_towel", "mochi_route_the_cable", "mochi_wipe_the_bar"]


def test_eden_list_says_when_no_task_carries_every_tag(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["eden", "list", "--tag", "locomotion", "--tag", "deformable"])
    eden_main()
    assert capsys.readouterr().out.strip() == "No task carries every tag of ['locomotion', 'deformable']."


def test_eden_list_rejects_an_unknown_tag(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["eden", "list", "--tag", "locomtion"])
    with pytest.raises(SystemExit):
        eden_main()
    assert "invalid choice: 'locomtion'" in capsys.readouterr().err


def test_building_the_cli_parser_imports_no_torch():
    """``eden list``'s ``--tag`` choices are read while the parser of every command is built, ``eden --help`` included."""
    code = (
        "import sys\n"
        "from eden.__main__ import main\n"
        "sys.argv = ['eden', 'clean', '--help']\n"
        "try:\n"
        "    main()\n"
        "except SystemExit:\n"
        "    pass\n"
        "print(sorted(m for m in ('torch', 'genesis') if m in sys.modules))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip().splitlines()[-1] == "[]"
