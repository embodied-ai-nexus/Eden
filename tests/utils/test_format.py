"""Tests for ``eden.utils.format``: ANSI output follows Eden's log theme, the one switch for a command."""

import pytest

from eden._logging import LOG_THEME_ENV_VAR
from eden.utils.format import ANSI_GREEN, colorize, format_table


@pytest.mark.parametrize(
    ("theme", "colored"),
    [("human", True), ("agent", False), ("auto", True)],
    ids=["human", "agent", "auto-with-force-color"],
)
def test_colorize_follows_the_theme(monkeypatch, theme, colored):
    """It used to read the TTY alone, so ``--help`` on a pty was colored under ``agent``."""
    monkeypatch.setenv(LOG_THEME_ENV_VAR, theme)
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)

    assert (colorize("x", ANSI_GREEN) != "x") is colored


def test_a_theme_broken_after_import_renders_plain_instead_of_raising(monkeypatch):
    """``colorize`` feeds ``parser.error(...)``; a raise there would replace the user's own argparse error.

    ``import eden`` already refuses a bad variable, so only a mutation after import reaches this.
    """
    monkeypatch.setenv(LOG_THEME_ENV_VAR, "humam")

    assert colorize("x", ANSI_GREEN) == "x"


def test_format_table_drops_the_box_under_agent_and_is_prettytable_under_human(monkeypatch):
    """Borders and padding are half a boxed table's tokens; the human rendering is PrettyTable's, byte for byte."""
    from prettytable import PrettyTable

    rows = [[0, "reaching_task", 1.0], [1, "action_rate", -0.01]]

    monkeypatch.setenv(LOG_THEME_ENV_VAR, "agent")
    assert format_table("Active Reward Terms", ["Index", "Name", "Weight"], rows) == (
        "Active Reward Terms: Name | Weight\n  reaching_task | 1.0\n  action_rate | -0.01"
    )
    assert format_table("Examples", ["No.", "Example", "Capabilities"], [[1, "ex", "a\nb"]]) == (
        "Examples: Example | Capabilities\n  ex | a, b"
    )

    monkeypatch.setenv(LOG_THEME_ENV_VAR, "human")
    table = PrettyTable()
    table.title = "Active Reward Terms"
    table.field_names = ["Index", "Name", "Weight"]
    table.align["Name"] = "l"
    for row in rows:
        table.add_row(row)
    assert (
        format_table("Active Reward Terms", ["Index", "Name", "Weight"], rows, align={"Name": "l"})
        == table.get_string()
    )
