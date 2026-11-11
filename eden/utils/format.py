"""Terminal styling and formatting helpers (ANSI colors, argparse help formatter).

Whether anything here emits an escape is decided by Eden's log theme — the same
``eden.init(logger_theme=...)`` / ``EDEN_LOG_THEME`` / ``eden --log-theme`` setting that governs
the logger — so one choice covers the whole of a command's output. The default theme is
``agent``, which emits none.

The environment's own reading — ``NO_COLOR`` / ``FORCE_COLOR`` / ``isatty`` — is not the switch:
on its own it would color ``eden train --task X --help`` on a pty for a run that asked, through the
theme, for no escapes. It is what the ``auto`` theme means; :func:`color_enabled` is that reading,
and :func:`eden._logging.resolve_theme` calls it.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Iterable
from typing import Any


ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_ITALIC = "\033[3m"
ANSI_UNDERLINE = "\033[4m"

#: Move the cursor to column 1 of the previous line -- the one escape that repaints an already
#: emitted record in place (``eden._logging.TimeElapser``'s ticker). Human-theme-only by nature:
#: the agent rendering promises no ANSI anywhere, and only a terminal acts on this.
ANSI_PREV_LINE = "\033[1F"

ANSI_BLACK = "\033[30m"
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_BLUE = "\033[34m"
ANSI_MAGENTA = "\033[35m"
ANSI_CYAN = "\033[36m"
ANSI_WHITE = "\033[37m"


def color_enabled() -> bool:
    """Return whether the *environment* wants color — the ``auto`` theme's reading.

    Honors the de-facto standards:

    - ``NO_COLOR`` (any value) disables color.
    - ``FORCE_COLOR`` (any value) enables color regardless of TTY.
    - Otherwise, color is enabled only when ``sys.stdout`` is a TTY.

    This is one *input* to the decision, not the decision: what actually gets emitted is
    :func:`color_output_enabled`, which asks the log theme. A caller reaching for this one wants
    "would a person be watching?", which is a different question from "does this run print
    escapes?" — and the answer to the second is no by default, on a TTY included.
    """
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


def color_output_enabled() -> bool:
    """Return whether this process emits ANSI escapes, per Eden's resolved log theme.

    ``human`` yes, ``agent`` no, ``auto`` per :func:`color_enabled`. One switch for the whole of a
    command's output, so the logger and everything here (``--help`` included) agree. The switch is
    ``EDEN_LOG_THEME``, which both ``eden.init(logger_theme=...)`` and ``eden --log-theme``
    export, so a theme asked for either way governs this and not just the logger.

    Returns
    -------
    bool
        Whether to emit escapes.
    """
    # Deferred: ``eden._logging`` imports this module at module scope for :func:`color_enabled`,
    # so the edge has to run one way only.
    from eden._logging import DEFAULT_LOGGER_THEME, resolve_theme

    try:
        theme = resolve_theme()
    except ValueError:
        # Not a swallowed misconfiguration: ``import eden`` resolves the theme unconditionally
        # and raises there, so a value this bad has already taken the process down -- the only
        # way to reach it here is an ``os.environ`` mutation *after* import. And this function is
        # reached from ``ColorHelpFormatter``, i.e. from ``parser.error(...)``, where raising
        # would replace the user's actual argparse error with a secondary one about a variable
        # they changed mid-process. Message-building code does not raise.
        theme = DEFAULT_LOGGER_THEME
    return theme == "human"


#: Leading columns that only restate a row's position -- dropped from the agent rendering, where
#: the row order already says it.
_INDEX_COLUMNS = ("Index", "No.")


def format_table(
    title: str,
    field_names: list[str],
    rows: Iterable[Iterable[Any]],
    *,
    align: dict[str, str] | None = None,
    max_width: dict[str, int] | None = None,
) -> str:
    """Render a titled table in the resolved log theme.

    ``human`` gets a boxed :class:`prettytable.PrettyTable`. ``agent`` gets the same cells with
    none of the box: a ``title: col | col`` header, then one ``  cell | cell`` line per row. The
    borders and the column padding are about half of a boxed table's tokens and carry nothing a
    reader of a transcript uses. Under ``agent`` a leading ``Index`` / ``No.`` column is dropped,
    since it restates the row order, and a multi-line cell is joined with ``", "``, since a row
    has to be one line to be read as one.

    Parameters
    ----------
    title : str
        The table's title.
    field_names : list[str]
        The column headers.
    rows : Iterable[Iterable[Any]]
        One iterable of cells per row, in ``field_names`` order.
    align : dict[str, str] | None
        PrettyTable alignment (``"l"``, ``"c"``, ``"r"``) per column name; ``human`` only.
    max_width : dict[str, int] | None
        PrettyTable wrap width per column name; ``human`` only.

    Returns
    -------
    str
        The rendered table, without a trailing newline.
    """
    if color_output_enabled():
        from prettytable import PrettyTable

        table = PrettyTable()
        table.title = title
        table.field_names = field_names
        for col, a in (align or {}).items():
            table.align[col] = a
        for col, width in (max_width or {}).items():
            table.max_width[col] = width
        for row in rows:
            table.add_row(list(row))
        return table.get_string()

    skip = 1 if field_names and field_names[0] in _INDEX_COLUMNS else 0
    lines = [f"{title}: {' | '.join(field_names[skip:])}"]
    for row in rows:
        cells = [str(cell).replace("\n", ", ") for cell in list(row)[skip:]]
        lines.append(f"  {' | '.join(cells)}")
    return "\n".join(lines)


def colorize(text: str, *codes: str) -> str:
    """Wrap ``text`` with the given ANSI escape ``codes`` and a reset suffix.

    Returns ``text`` unchanged when :func:`color_output_enabled` is False, so it's safe to
    use unconditionally. Multiple codes are concatenated (e.g. ``ANSI_BOLD + ANSI_CYAN``).
    """
    if not codes or not color_output_enabled():
        return text
    return f"{''.join(codes)}{text}{ANSI_RESET}"


class ColorHelpFormatter:
    """Argparse help formatter that ANSI-colorizes section headings and ``--flag`` tokens.

    Suppresses the auto-generated ``usage:`` block in full ``--help`` output (where it
    duplicates the ``options:`` section) but **keeps** it on parse errors so
    ``parser.error(...)`` still produces a useful diagnostic. Wraps
    :class:`argparse.RawDescriptionHelpFormatter` so descriptions and epilogs preserve
    newlines.

    Subclassing is deferred to ``__new__`` so ``argparse`` is not imported at module load.
    """

    def __new__(cls, prog, *args, **kwargs):
        from argparse import RawDescriptionHelpFormatter

        class _Impl(RawDescriptionHelpFormatter):
            def start_section(self, heading):
                return super().start_section(colorize(heading, ANSI_BOLD, ANSI_YELLOW))

            def _format_action_invocation(self, action):
                text = super()._format_action_invocation(action)
                return re.sub(
                    r"(-{1,2}[A-Za-z][\w-]*)",
                    lambda m: colorize(m.group(1), ANSI_GREEN),
                    text,
                )

            def add_usage(self, usage, actions, groups, prefix=None):
                # `ArgumentParser.format_help` and `ArgumentParser.format_usage` both call
                # `formatter.add_usage(...)`. We only want to suppress the usage line in
                # the former (full --help), not the latter (used by `print_usage` /
                # `parser.error`). Distinguish by inspecting the caller frame.
                caller = sys._getframe(1).f_code.co_name
                if caller == "format_help":
                    return None
                return super().add_usage(usage, actions, groups, prefix)

        return _Impl(prog, *args, **kwargs)
