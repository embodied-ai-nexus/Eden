"""Tests for ``eden._logging``: theme resolution, the two renderings, ``Logger.timer``, log capture, and the collapsing of repeated Genesis warnings."""

import logging
import os
import re
import sys
import time

import genesis as gs
import pytest
from genesis.styles import THEME, style

import eden as en
from eden._logging import (
    _COLORS,
    DEFAULT_LOGGER_THEME,
    LOG_THEME_ENV_VAR,
    EdenFormatter,
    Logger,
    resolve_theme,
    collapse_repeated_genesis_warnings,
    sync_genesis_theme,
)


def _record(level=logging.INFO, msg="hello"):
    return logging.LogRecord("eden", level, "x.py", 1, msg, None, None)


@pytest.fixture
def genesis_theme():
    """Set Genesis' theme (``genesis.styles.style.theme``) for one test and put back whatever was there.

    Genesis' only public setter is ``gs.init``, which would re-initialize the simulator.

    Yields
    ------
    collections.abc.Callable
        Called with a theme name, or ``None`` to model a process where ``gs.init()`` has not run.
    """
    previous = style.theme
    try:
        yield lambda theme: setattr(style, "theme", None if theme is None else THEME[theme])
    finally:
        style.theme = previous


@pytest.fixture
def no_ambient_theme(monkeypatch):
    """Start from "nobody has chosen", whatever the shell running the suite set."""
    for variable in (LOG_THEME_ENV_VAR, "NO_COLOR", "FORCE_COLOR"):
        monkeypatch.delenv(variable, raising=False)


class TestEdenCaplog:
    """``en.logger`` does not propagate, so the root conftest's ``eden_caplog`` is what can see it."""

    @pytest.fixture
    def _eden_logger_muted(self):
        """Raise the ``eden`` logger above INFO; requested before ``eden_caplog``, so it tears down last."""
        eden_logger = logging.getLogger("eden")
        previous = eden_logger.level
        eden_logger.setLevel(logging.CRITICAL)
        yield
        assert eden_logger.level == logging.CRITICAL, "eden_caplog did not restore the level it found"
        eden_logger.setLevel(previous)

    def test_it_captures_records_below_the_loggers_own_level(self, _eden_logger_muted, eden_caplog):
        assert logging.getLogger("eden").level == logging.DEBUG
        en.logger.info("sentinel-below-the-loggers-level")
        assert "sentinel-below-the-loggers-level" in eden_caplog.text


class TestEdenFormatter:
    """Eden's rendering is Eden's own: no state of Genesis' theme may change it."""

    def test_an_unmapped_level_is_uncolored_not_fatal(self):
        """``Logger.log`` takes any level; an unmapped one used to put "None" in the banner and then raise."""
        out = EdenFormatter(verbose_time=False, theme="human").format(_record(level=25, msg="custom"))

        assert "custom" in out
        assert "None" not in out

    @pytest.mark.parametrize("theme", [None, "dark", "light", "raw"], ids=["unset", "dark", "light", "raw"])
    def test_the_human_palette_is_edens_own_whatever_genesis_is_set_to(self, genesis_theme, theme):
        genesis_theme(theme)

        out = EdenFormatter(verbose_time=False, theme="human").format(_record(logging.WARNING, "a ~<word>~ here"))

        assert "word" in out
        assert _COLORS["YELLOW"] in out
        assert _COLORS["MINT"] in out
        assert "~<" not in out and ">~" not in out

    def test_a_live_formatter_is_unaffected_by_the_theme_changing_under_it(self, genesis_theme):
        """``gs.init()`` moves Genesis' theme while Eden's logger is running."""
        genesis_theme("dark")
        fmt = EdenFormatter(verbose_time=False, theme="human")
        before = fmt.format(_record(msg="same"))

        genesis_theme(None)

        assert fmt.format(_record(msg="same")) == before

    def test_the_palette_is_still_byte_identical_to_genesis_dark(self, genesis_theme):
        """``_COLORS`` copies Genesis' ``dark`` codes so both render alike; nothing else notices drift."""
        genesis_theme("dark")

        assert {name: getattr(gs.styles.colors, name) for name in _COLORS} == _COLORS

    def test_the_agent_theme_emits_no_escape_even_under_genesis_dark(self, genesis_theme):
        genesis_theme("dark")

        out = EdenFormatter(verbose_time=False, theme="agent").format(_record(logging.ERROR, "see ~~<a/path>~~ now"))

        assert "\x1b[" not in out
        assert "see a/path now" in out
        assert "~" not in out

    def test_only_edens_theme_silences_the_color(self, genesis_theme):
        """Genesis' ``raw`` theme asks for no ANSI in *Genesis'* output, not Eden's."""
        genesis_theme("raw")

        colored = EdenFormatter(verbose_time=False, theme="human").format(_record(logging.ERROR, "a ~~<w>~~ b"))

        assert "\x1b[" in colored

    @pytest.mark.parametrize(("theme", "banner"), [("agent", "[Eden "), ("human", "[Eden 🍏]")])
    def test_the_banner(self, theme, banner):
        out = EdenFormatter(verbose_time=False, theme=theme).format(_record())

        assert banner in out
        if theme == "agent":
            assert out.startswith(banner) and "🍏" not in out

    def test_agent_info_lines_carry_no_level_and_every_other_level_names_its_own(self):
        """The prefix is paid on every line, so INFO spells nothing; an agent greps for ``WARNING``."""
        formatter = EdenFormatter(verbose_time=False, theme="agent")

        assert re.fullmatch(r"\[Eden \d\d:\d\d:\d\d\] hello", formatter.format(_record(logging.INFO)))
        for level in (logging.DEBUG, logging.WARNING, logging.ERROR, logging.CRITICAL):
            out = formatter.format(_record(level))
            assert re.fullmatch(rf"\[Eden \d\d:\d\d:\d\d {logging.getLevelName(level)}\] hello", out), out

    def test_agent_strips_pictographs_and_keeps_status_marks(self):
        """An emoji is decoration; a check mark can be the whole message."""
        msg = "🍎 Eden ver: abc, 🌏 Genesis ver: def. 💽 Log dir: x ✓ ok ✕ bad"
        out = EdenFormatter(verbose_time=False, theme="agent").format(_record(msg=msg))

        assert out.endswith("] Eden ver: abc, Genesis ver: def. Log dir: x ✓ ok ✕ bad"), out


@pytest.mark.usefixtures("no_ambient_theme")
class TestResolveTheme:
    """Which rendering a process gets, and who decides."""

    @pytest.mark.parametrize("requested", ["human", "agent"])
    def test_an_explicit_theme_is_honored(self, requested):
        assert resolve_theme(requested) == requested

    def test_the_default_is_agent_even_on_a_terminal(self, monkeypatch):
        """A pty attached by an agent harness passes a TTY check, so ``auto`` is not the default."""
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)

        assert resolve_theme() == DEFAULT_LOGGER_THEME == "agent"

    @pytest.mark.parametrize(
        ("isatty", "variable", "expected"),
        [(True, None, "human"), (False, None, "agent"), (True, "NO_COLOR", "agent"), (False, "FORCE_COLOR", "human")],
        ids=["tty", "pipe", "no-color", "force-color"],
    )
    def test_auto_follows_the_terminal_and_the_color_variables(self, monkeypatch, isatty, variable, expected):
        monkeypatch.setattr(sys.stdout, "isatty", lambda: isatty, raising=False)
        if variable:
            monkeypatch.setenv(variable, "1")

        assert resolve_theme("auto") == expected

    @pytest.mark.parametrize(
        ("variable", "argument", "expected"),
        [("human", None, "human"), ("auto", None, "human"), ("human", "agent", "agent")],
    )
    def test_the_environment_variable_sets_the_default_and_an_argument_beats_it(
        self, monkeypatch, variable, argument, expected
    ):
        monkeypatch.setenv(LOG_THEME_ENV_VAR, variable)
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)

        assert resolve_theme(argument) == expected

    @pytest.mark.parametrize("raw", ["", "   ", "\t\n"], ids=["empty", "spaces", "whitespace"])
    def test_an_exported_but_empty_variable_means_unset(self, monkeypatch, raw):
        """``ENV EDEN_LOG_THEME=`` in a Dockerfile used to make ``import eden`` itself raise."""
        monkeypatch.setenv(LOG_THEME_ENV_VAR, raw)

        assert resolve_theme() == DEFAULT_LOGGER_THEME

    def test_an_explicitly_empty_argument_is_still_refused(self):
        with pytest.raises(ValueError, match="Invalid logger_theme: ''"):
            resolve_theme("")

    @pytest.mark.parametrize("source", ["argument", "environment"])
    def test_an_unknown_theme_names_its_source(self, monkeypatch, source):
        """The fix differs: edit the call, or unset the variable."""
        if source == "environment":
            monkeypatch.setenv(LOG_THEME_ENV_VAR, "quiet")
        with pytest.raises(
            ValueError, match=rf"Invalid {LOG_THEME_ENV_VAR if source == 'environment' else 'logger_theme'}: 'quiet'"
        ):
            resolve_theme("quiet" if source == "argument" else None)

    @pytest.mark.parametrize("raw", ["HUMAM", "  humam  ", "Agnet\t"])
    def test_the_message_quotes_the_value_as_it_was_set(self, monkeypatch, raw):
        """Validation normalises the name; the message must still quote what the reader wrote."""
        monkeypatch.setenv(LOG_THEME_ENV_VAR, raw)

        with pytest.raises(ValueError) as caught:
            resolve_theme()

        assert repr(raw) in str(caught.value), str(caught.value)


class TestLogger:
    @pytest.mark.usefixtures("no_ambient_theme")
    def test_it_resolves_and_exposes_its_theme(self):
        """``eden.init`` reads ``logger.theme`` to decide whether the greeting box is worth printing."""
        assert Logger(logging.INFO, theme="agent").theme == "agent"
        assert Logger(logging.INFO, theme="human").theme == "human"

    def test_it_can_be_rethemed_after_it_is_built(self, capsys, monkeypatch):
        """``eden.logger`` exists from ``import eden``, i.e. before ``--log-theme`` is parsed."""
        monkeypatch.delenv(LOG_THEME_ENV_VAR, raising=False)
        logger = Logger(logging.INFO, theme="agent")

        logger.set_theme("human")
        logger.info("rethemed")

        assert logger.theme == "human"
        assert "[Eden 🍏]" in capsys.readouterr().out

    def test_retheming_exports_the_request_so_tables_colorize_and_genesis_follow(self, monkeypatch, genesis_theme):
        """One switch: ``set_theme`` used to move the log lines alone, around agent-rendered tables."""
        from eden.utils.format import ANSI_GREEN, colorize, format_table

        monkeypatch.delenv(LOG_THEME_ENV_VAR, raising=False)
        genesis_theme("raw")
        logger = Logger(logging.INFO, theme="agent")

        logger.set_theme("human")

        assert os.environ[LOG_THEME_ENV_VAR] == "human"
        assert format_table("T", ["Name"], [["a"]]).startswith("+")
        assert colorize("x", ANSI_GREEN) != "x"
        assert style.theme is THEME.dark

        logger.set_theme("auto")  # the request is exported, so ``auto`` stays a per-process decision downstream
        assert os.environ[LOG_THEME_ENV_VAR] == "auto"


class TestTimer:
    """``Logger.timer``: Eden's own ``TimeElapser``, theme-correct in both renderings."""

    def test_the_agent_theme_emits_no_escape_and_reports_the_whole_message_and_time(self, capsys):
        """Genesis' version wrote a literal cursor-up, and sliced four characters off the message."""
        logger = Logger(logging.INFO, theme="agent")

        with logger.timer("abcdefgh", refresh_rate=100, end_msg="(cached)"):
            pass

        out = capsys.readouterr().out
        assert "\x1b[" not in out, out
        assert "abcdefgh" in out
        assert "done in 0." in out and "(cached)" in out

    def test_the_human_theme_keeps_the_line_colored_and_closes_it(self, capsys):
        logger = Logger(logging.INFO, theme="human")

        with logger.timer("Loading the thing", refresh_rate=100):
            time.sleep(0.05)

        out = capsys.readouterr().out
        assert "\x1b[1F" in out, "the human rendering has a line to step back onto"
        assert "✅" in out
        assert out.endswith("\x1b[0m\n"), repr(out[-20:])

    def test_a_non_positive_refresh_rate_is_refused_before_anything_is_logged(self, capsys):
        """The opening record would claim a step started that never did."""
        logger = Logger(logging.INFO, theme="agent")
        capsys.readouterr()

        with pytest.raises(ValueError, match="refresh_rate must be positive"):
            logger.timer("Loading the thing", refresh_rate=0)

        assert "Loading the thing" not in capsys.readouterr().out

    def test_a_timer_on_a_logger_that_filters_info_repaints_nothing(self, capsys):
        """``last_output`` still holds the last record that got through, which a ticker must not repaint."""
        logger = Logger(logging.WARNING, theme="human")
        logger.warning("an earlier line")
        capsys.readouterr()

        with logger.timer("not logged at this level"):
            time.sleep(0.02)

        assert capsys.readouterr().out == ""


class TestSyncGenesisTheme:
    """A theme chosen after ``gs.init()`` (``Logger.set_theme``) has to move Genesis' output along with Eden's."""

    def test_it_repoints_a_theme_gs_init_already_chose(self, genesis_theme, monkeypatch):
        genesis_theme("dark")
        monkeypatch.setenv(LOG_THEME_ENV_VAR, "agent")

        sync_genesis_theme()

        assert style.theme is THEME.raw


# ---------------------------------------------------------------------------
# collapse_repeated_genesis_warnings: a G1 build printed one "not watertight" line per mesh
# ---------------------------------------------------------------------------


@pytest.fixture
def genesis_caplog(caplog):
    """``caplog`` on the ``genesis`` logger, which ``eden.init`` sets not to propagate."""
    genesis_logger = logging.getLogger("genesis")
    previous_level = genesis_logger.level
    genesis_logger.setLevel(logging.DEBUG)
    genesis_logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        genesis_logger.removeHandler(caplog.handler)
        genesis_logger.setLevel(previous_level)


def _messages(caplog, level: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelname == level]


@pytest.mark.parametrize(
    ("logged", "expected"),
    [
        (["Mesh is not watertight."] * 9, ["Mesh is not watertight. (9 times while adding entity 'robot')"]),
        (["only once"], ["only once"]),
        (["b", "a", "b"], ["b (2 times while adding entity 'robot')", "a"]),  # first-seen order
    ],
    ids=["repeats", "single", "distinct"],
)
def test_repeated_warnings_collapse_to_one_counted_line(genesis_caplog, logged, expected):
    with collapse_repeated_genesis_warnings("entity 'robot'"):
        for message in logged:
            gs.logger.warning(message)

    assert _messages(genesis_caplog, "WARNING") == expected


def test_other_levels_pass_through_as_they_happen(genesis_caplog):
    with collapse_repeated_genesis_warnings("entity 'robot'"):
        gs.logger.info("progress")
        gs.logger.error("broken")
        assert (_messages(genesis_caplog, "INFO"), _messages(genesis_caplog, "ERROR")) == (["progress"], ["broken"])
        assert _messages(genesis_caplog, "WARNING") == []


def test_held_warnings_are_flushed_when_the_block_raises(genesis_caplog):
    """A warning logged just before a load error often explains it."""
    with pytest.raises(RuntimeError, match="load failed"), collapse_repeated_genesis_warnings("entity 'robot'"):
        gs.logger.warning("the reason")
        raise RuntimeError("load failed")

    assert _messages(genesis_caplog, "WARNING") == ["the reason"]


def test_the_filter_is_removed_on_exit(genesis_caplog):
    with collapse_repeated_genesis_warnings("entity 'robot'"):
        pass
    gs.logger.warning("after")
    gs.logger.warning("after")

    assert logging.getLogger("genesis").filters == []
    assert _messages(genesis_caplog, "WARNING") == ["after", "after"]
