"""Logging setup: Eden's two log themes, its formatter, and the rank-aware logger."""

# Code modified from: https://github.com/Genesis-Embodied-AI/Genesis/blob/main/genesis/logging/logger.py
# Copyright (c) 2025, Genesis-Embodied-AI
# Apache License 2.0

import math
import os
import re
import sys
import logging
import threading
import time
from contextlib import contextmanager
from typing import Literal

from eden.utils.format import (
    ANSI_BOLD,
    ANSI_ITALIC,
    ANSI_PREV_LINE,
    ANSI_RESET,
    ANSI_UNDERLINE,
    color_enabled,
)


# Marks the handlers this module installs on the "eden" logger, so a later Logger()
# can retire its predecessor's without touching handlers the caller attached.
_EDEN_OWNED_HANDLER = "_eden_owned_handler"

#: What a caller may ask for. ``"human"`` and ``"agent"`` are the two renderings; ``"auto"``
#: picks between them per process (see :func:`resolve_theme`).
LoggerTheme = Literal["auto", "human", "agent"]

#: The two renderings ``"auto"`` resolves to.
ResolvedLoggerTheme = Literal["human", "agent"]

#: Sets the theme for a whole process tree without touching any call site -- the case an agent
#: harness is in, since it launches ``uv run python train.py`` rather than calling
#: :func:`eden.init` itself. Chiefly how a *person* asks for ``"human"`` back.
LOG_THEME_ENV_VAR = "EDEN_LOG_THEME"

#: What a process gets when nobody chooses: plain output, no escapes, no greeting box. Eden's
#: console output is read by a machine far more often than by a person -- a transcript, a CI log,
#: an agent's context window -- so the plain rendering is the default and color is the request,
#: rather than the other way around. ``"auto"`` (TTY-sensing) remains available by name.
DEFAULT_LOGGER_THEME: ResolvedLoggerTheme = "agent"

#: Markers :meth:`EdenFormatter.extra_fmt` renders as emphasis, longest first so ``~~~~<`` is
#: consumed before the ``~~<`` inside it. Openers and closers are stripped together under the
#: agent theme -- leaving a literal ``~~<path>~~`` in the text would be worse than the color.
_EMPHASIS_MARKERS = ("~~~~<", "~~~<", "~~<", "~<", ">~~~~", ">~~~", ">~~", ">~")

#: Pictographic emoji (with any variation selector / joiner and the space after it) -- what the
#: agent theme strips from a message. Each costs one to three tokens and carries nothing a reader
#: of a transcript uses. The range stops short of the dingbats (U+2700 block), so a status mark
#: such as ``✓`` / ``✕`` that *is* the content survives.
_PICTOGRAPH = re.compile("[\U0001f000-\U0001faff][\ufe0f\u200d\U0001f000-\U0001faff]* ?")


def resolve_theme(theme: LoggerTheme | None = None) -> ResolvedLoggerTheme:
    """Pick the rendering for this process: colored for a person, plain for a machine.

    ``"human"`` is the colored terminal rendering; ``"agent"`` drops the ANSI escapes *and* the
    greeting box, for output that lands in a transcript, a CI log or an agent's context window
    (escapes can be roughly a third of a redirected log's bytes).

    Precedence is explicit-argument, then ``EDEN_LOG_THEME``, then
    :data:`DEFAULT_LOGGER_THEME` -- which is ``"agent"``: the plain rendering is what a process
    gets unless someone asks for color, including on a terminal. ``"auto"`` follows
    :func:`eden.utils.format.color_enabled` (``NO_COLOR`` off, ``FORCE_COLOR`` on, a TTY
    otherwise) -- the one to set if you want color only when a person is watching.

    Parameters
    ----------
    theme : {'auto', 'human', 'agent'} | None
        The requested theme. ``None`` (the default) reads ``EDEN_LOG_THEME``, falling back to
        :data:`DEFAULT_LOGGER_THEME`.

    Returns
    -------
    {'human', 'agent'}
        The rendering to use.

    Raises
    ------
    ValueError
        If ``theme`` (or the environment variable) is not one of the three names.
    """
    if theme is not None:
        raw = theme
    else:
        raw = os.environ.get(LOG_THEME_ENV_VAR, DEFAULT_LOGGER_THEME)
        # Exported-but-empty means *unset*, not "invalid". ``ENV EDEN_LOG_THEME=`` in a
        # Dockerfile or ``EDEN_LOG_THEME= uv run python train.py`` is the ordinary way to clear a
        # variable, and ``os.environ.get(var, default)`` returns ``""`` for it -- which would fail
        # the check below and make ``import eden`` itself raise over a setting that cannot affect
        # a result. ``eden.utils.format.color_enabled`` reads ``NO_COLOR`` the same way. An
        # explicit ``logger_theme=""`` -- a caller passed it -- still raises.
        if not raw.strip():
            raw = DEFAULT_LOGGER_THEME
    requested = str(raw).strip().lower()
    if requested not in ("auto", "human", "agent"):
        source = "logger_theme" if theme is not None else LOG_THEME_ENV_VAR
        # ``raw!r``, not the normalized spelling: the value is stripped and lower-cased before the
        # comparison, so quoting the normalized form hides the two mistakes it papers over --
        # ``EDEN_LOG_THEME=" human "`` is accepted while ``HUMAM`` would be reported as 'humam',
        # which is not what the reader has to go and find.
        raise ValueError(
            f"Invalid {source}: {raw!r}. Valid themes are: agent (the default -- never color, "
            "and no greeting box), human (always color), auto (color when stdout is a TTY and "
            "NO_COLOR is unset)."
        )
    if requested != "auto":
        return requested
    return "human" if color_enabled() else "agent"


#: Genesis' theme for each Eden theme -- what ``genesis.styles.style.theme`` has to hold for Genesis'
#: own output, and every ``Options.__repr__``, to render the way Eden's records do. ``"raw"`` is the
#: plain, compact rendering https://github.com/Genesis-Embodied-AI/Genesis/pull/3452 put in place of
#: ``"dumb"``. ``gs.init()`` is what sets it; until then Genesis renders plain, a config ``repr``
#: included, whatever Eden's theme.
_GENESIS_THEMES: dict[str, str] = {"human": "dark", "agent": "raw"}


def sync_genesis_theme() -> None:
    """Point Genesis' theme (``genesis.styles.style.theme``) at Eden's resolved one.

    For a theme chosen after the fact -- :meth:`Logger.set_theme` -- so Genesis' own output, and
    every ``Options.__repr__``, follows the switch rather than keeping what ``gs.init()`` chose.

    A no-op when genesis is not imported: asking for a theme must never be the thing that drags
    it in, and ``eden.init()`` hands Genesis the theme when it does import it.
    """
    if "genesis" not in sys.modules:
        return
    from genesis.styles import THEME, style

    style.theme = THEME[_GENESIS_THEMES[resolve_theme()]]


#: Eden's console palette -- what the ``"human"`` theme renders a record in. The five codes are
#: Genesis' ``dark`` theme, so Eden's lines and Genesis' match in one terminal; they are *owned
#: here* rather than read out of ``genesis.styles.colors``, for two reasons. That import would put
#: the whole of ``import genesis`` (~2.8 s) on ``import eden``'s path for five strings. And those
#: properties are keyed on Genesis' global theme (``genesis.styles.style.theme``), so Eden's
#: rendering would be decided by Genesis' palette: unset, every property returns ``None``, and a
#: ``Logger("INFO", theme="human")`` built without ``eden.init()`` would render uncolored. Eden's
#: theme decides Eden's output; ``eden.init`` sets Genesis' theme so Genesis' own output follows
#: the same choice.
_COLORS = {
    "GREEN": "\x1b[38;5;119m",
    "BLUE": "\x1b[38;5;159m",
    "YELLOW": "\x1b[38;5;226m",
    "RED": "\x1b[38;5;9m",
    "MINT": "\x1b[38;5;121m",
}

# Level -> its :data:`_COLORS` key. A level with no entry here renders uncolored rather than
# raising: ``Logger.log(level, ...)`` takes any level, including ones ``logging`` does not name.
_LEVEL_COLOR_NAMES = {
    logging.DEBUG: "GREEN",
    logging.INFO: "BLUE",
    logging.WARNING: "YELLOW",
    logging.ERROR: "RED",
    logging.CRITICAL: "RED",
}


@contextmanager
def collapse_repeated_genesis_warnings(subject: str):
    """Hold back Genesis' warnings for the block, then emit each distinct one once, with its count.

    Loading one asset can make Genesis repeat a per-mesh warning verbatim: a G1 prints
    "Mesh is not watertight. ..." nine times, once per collision mesh, on every build, and
    the repeats bury whatever else the build reports. Only identical text merges and
    nothing is dropped: the count goes on the line, and a warning seen once is re-emitted
    unchanged. Other levels pass straight through. The held warnings are flushed even when
    the block raises, because a warning logged just before a load error often explains it.
    A Genesis logger quieted below WARNING never hands the filter a record, so a user who
    silenced these gets none back.

    Parameters
    ----------
    subject : str
        What the block loads, named on each collapsed line (e.g. ``"entity 'robot'"``).
    """
    import genesis as gs

    counts: dict[str, int] = {}  # insertion-ordered, so the lines keep their first-seen order

    class _HoldWarnings(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if record.levelno != logging.WARNING:
                return True
            message = record.getMessage()
            counts[message] = counts.get(message, 0) + 1
            return False

    hold = _HoldWarnings()
    genesis_logger = logging.getLogger("genesis")
    genesis_logger.addFilter(hold)
    try:
        yield
    finally:
        genesis_logger.removeFilter(hold)
        for message, count in counts.items():
            gs.logger.warning(message if count == 1 else f"{message} ({count} times while adding {subject})")


class EdenFormatter(logging.Formatter):
    """Renders a record in one of the two :func:`resolve_theme` themes.

    Parameters
    ----------
    verbose_time : bool
        Include the date and milliseconds in the timestamp.
    theme : {'human', 'agent'}
        Already resolved, not ``"auto"``: a formatter is built once and reused for the process,
        so re-deciding per record would let a redirected stream change mid-run. :class:`Logger`
        resolves it and always passes it; the default here only matters to a direct construction,
        and matches :data:`DEFAULT_LOGGER_THEME` so there is one default rather than two.
    """

    def __init__(self, verbose_time=True, theme: ResolvedLoggerTheme = DEFAULT_LOGGER_THEME):
        super().__init__()

        self.theme = theme

        if verbose_time:
            self.TIME = "%(asctime)s.%(msecs)03d"
            self.DATE_FORMAT = "%y-%m-%d %H:%M:%S"
            self.INFO_length = 41
        else:
            self.TIME = "%(asctime)s"
            self.DATE_FORMAT = "%H:%M:%S"
            self.INFO_length = 28

        self.LEVEL = "%(levelname)s"
        self.MESSAGE = "%(message)s"

        self.last_output = ""
        self.last_color = ""

    def colored_fmt(self, color):
        self.last_color = color
        return f"{color}[Eden 🍏] [{self.TIME}] [{self.LEVEL}] {self.MESSAGE}{ANSI_RESET}"

    def agent_fmt(self, levelno: int) -> str:
        """The agent theme's line: ``[Eden 12:17:11] msg``, the level named only when it is not INFO.

        This rendering is read out of a file, a CI log or a context window, where the prefix is
        paid on every line: ``[Eden] [12:17:11] [INFO] `` is 13 tokens, this is 9. INFO is the
        level nearly every line has, so only the exceptions spell theirs -- and a ``grep
        WARNING`` still finds every warning.
        """
        if levelno == logging.INFO:
            return f"[Eden {self.TIME}] {self.MESSAGE}"
        return f"[Eden {self.TIME} {self.LEVEL}] {self.MESSAGE}"

    def extra_fmt(self, msg):
        if self.theme == "agent":
            for marker in _EMPHASIS_MARKERS:
                msg = msg.replace(marker, "")
            return _PICTOGRAPH.sub("", msg)
        mint = _COLORS["MINT"]
        msg = msg.replace("~~~~<", mint + ANSI_BOLD + ANSI_ITALIC)
        msg = msg.replace("~~~<", mint + ANSI_ITALIC)
        msg = msg.replace("~~<", mint + ANSI_UNDERLINE)
        msg = msg.replace("~<", mint)

        msg = msg.replace(">~~~~", ANSI_RESET + self.last_color)
        msg = msg.replace(">~~~", ANSI_RESET + self.last_color)
        msg = msg.replace(">~~", ANSI_RESET + self.last_color)
        msg = msg.replace(">~", ANSI_RESET + self.last_color)

        return msg

    def format(self, record):
        # ``.get`` with a default, because ``Logger.log(level, ...)`` takes any level: an
        # unmapped one renders uncolored rather than as a literal "None" color code.
        if self.theme == "agent":
            log_fmt = self.agent_fmt(record.levelno)
        else:
            log_fmt = self.colored_fmt(_COLORS.get(_LEVEL_COLOR_NAMES.get(record.levelno), ""))
        formatter = logging.Formatter(log_fmt, datefmt=self.DATE_FORMAT)
        msg = self.extra_fmt(formatter.format(record))
        self.last_output = msg
        return msg


class Logger:
    """Eden's process-singleton logger over the ``"eden"`` stdlib logger.

    Parameters
    ----------
    logging_level : int | str
        A ``logging`` level, by value or by name.
    verbose_time : bool
        Include the date and milliseconds in each line's timestamp.
    theme : {'auto', 'human', 'agent'} | None
        How to render. ``None`` (the default) reads ``EDEN_LOG_THEME`` and otherwise falls back
        to :data:`DEFAULT_LOGGER_THEME`, which is ``"agent"`` — see :func:`resolve_theme`. The
        resolved value is on :attr:`theme`, which is what ``eden.init`` reads to decide whether
        the greeting box is worth printing.
    """

    def __init__(self, logging_level: int | str, verbose_time: bool = False, theme: LoggerTheme | None = None):
        self.theme: ResolvedLoggerTheme = resolve_theme(theme)
        if isinstance(logging_level, str):
            logging_level_str = logging_level.upper()
            logging_level = getattr(logging, logging_level_str, None)
            if logging_level is None:
                raise ValueError(
                    f"Invalid logging level: '{logging_level_str}'. "
                    f"Valid levels are: DEBUG, INFO, WARNING, ERROR, CRITICAL"
                )
        elif not isinstance(logging_level, int):
            raise ValueError(f"Logging level must be a string or integer, got: {type(logging_level)}")

        self._logger = logging.getLogger("eden")
        # Constructing a Logger rebinds the process-singleton "eden" logger: drop the
        # handlers left by any previous instance (the module-import seed in
        # eden/__init__.py, or a repeated init()) so no log line is ever emitted twice.
        # StreamHandler.close() does not close sys.stdout.
        #
        # Only *our* handlers, identified by the marker set below. Anything else on
        # this logger belongs to the caller — an app routing Eden's output to a file,
        # or pytest's capture handler — and removing (let alone closing) it would break
        # that silently the next time someone calls eden.init().
        for stale in list(self._logger.handlers):
            if getattr(stale, _EDEN_OWNED_HANDLER, False):
                self._logger.removeHandler(stale)
                stale.close()
        self._logger.setLevel(logging_level)
        self._logger.propagate = False

        self._formatter = EdenFormatter(verbose_time, self.theme)

        self._handler = logging.StreamHandler(sys.stdout)
        setattr(self._handler, _EDEN_OWNED_HANDLER, True)
        self._handler.setLevel(logging_level)
        self._handler.setFormatter(self._formatter)
        self._logger.addHandler(self._handler)

        self._stream = self._handler.stream
        self._is_new_line = True

        self.timer_lock = threading.Lock()

    def addFilter(self, filter):
        self._logger.addFilter(filter)

    def removeFilter(self, filter):
        self._logger.removeFilter(filter)

    def removeHandler(self, handler):
        self._logger.removeHandler(handler)

    @property
    def INFO_length(self):
        return self._formatter.INFO_length

    @contextmanager
    def log_wrapper(self):
        self.timer_lock.acquire()

        # swap with timer output
        if not self._is_new_line and not self._stream.closed:
            self._stream.write("\r")
        try:
            yield
        finally:
            self._is_new_line = True
            self.timer_lock.release()

    @contextmanager
    def lock_timer(self):
        self.timer_lock.acquire()
        try:
            yield
        finally:
            self.timer_lock.release()

    def log(self, level, msg, *args, **kwargs):
        with self.log_wrapper():
            self._logger.log(level, msg, *args, **kwargs)

    def debug(self, message):
        with self.log_wrapper():
            self._logger.debug(message)

    def info(self, message):
        with self.log_wrapper():
            self._logger.info(message)

    def warning(self, message):
        with self.log_wrapper():
            self._logger.warning(message)

    def error(self, message):
        with self.log_wrapper():
            self._logger.error(message)

    def critical(self, message):
        with self.log_wrapper():
            self._logger.critical(message)

    def raw(self, message):
        self._stream.write(self._formatter.extra_fmt(message))
        self._stream.flush()
        if message.endswith("\n"):
            self._is_new_line = True
        else:
            self._is_new_line = False

    def set_theme(self, theme: LoggerTheme | None = None) -> None:
        """Switch the process to ``theme``, from the next record on.

        For the case where the theme is chosen *after* the logger exists: ``eden.logger`` is built
        at ``import eden``, which is long before ``eden --log-theme ...`` has been parsed, so
        without this the flag would reach only the records emitted past ``eden.init()``.

        The theme has consumers besides this logger -- :func:`eden.utils.format.format_table` and
        :func:`~eden.utils.format.colorize` read ``EDEN_LOG_THEME``, and so does every subprocess
        -- so an explicit ``theme`` is exported there, the way ``eden.init(logger_theme=...)``
        does, and Genesis' palette is re-pointed; retheming only the logger would leave the
        summary tables and ``--help`` in the old theme next to log lines in the new one.

        Parameters
        ----------
        theme : {'auto', 'human', 'agent'} | None
            The requested theme, resolved by :func:`resolve_theme`. ``None`` re-reads the
            environment. The *request* is exported, not its resolution, so ``auto`` stays a
            per-process decision in a subprocess.
        """
        self.theme = resolve_theme(theme)
        self._formatter.theme = self.theme
        if theme is not None:
            os.environ[LOG_THEME_ENV_VAR] = theme
        sync_genesis_theme()

    def timer(self, msg, refresh_rate=10, end_msg=""):
        """Log ``msg`` and return a ``with``-block that reports how long the block took.

        Parameters
        ----------
        msg : str
            Logged immediately, at INFO, and repeated on the completion line.
        refresh_rate : float
            Ticker updates per second under the ``"human"`` theme. It also sets the number of
            decimals on the completion line (10/s -> one decimal), which the ``"agent"`` theme
            renders too -- only the ticker itself is human-only.
        end_msg : str
            Appended to the completion line, after the elapsed time.

        Returns
        -------
        TimeElapser
            The context manager. Calling this without ``with`` logs ``msg`` and nothing else.

        Raises
        ------
        ValueError
            If ``refresh_rate`` is not positive. Checked before ``msg`` is logged, so a rejected
            call leaves no record claiming a step started that never did.
        """
        # Constructed first so an invalid `refresh_rate` raises *before* `msg` is logged: a
        # rejected call must not leave a record claiming a step started that never did, and in
        # the `with` form never reports finishing. `TimeElapser` reads `last_output` at entry
        # rather than at construction for exactly this ordering.
        elapser = TimeElapser(self, msg, refresh_rate, end_msg)
        self.info(msg)
        return elapser

    @property
    def handler(self):
        return self._handler

    @property
    def last_output(self):
        return self._formatter.last_output

    @property
    def level(self):
        return self._logger.level


class TimeElapser:
    r"""``with``-block that reports how long the block took, in the theme the logger resolved.

    Built by :meth:`Logger.timer`, which has already logged the opening message. Under the
    ``"human"`` theme a background thread repaints that line in place with a live elapsed-time
    counter; under ``"agent"`` there is no ticker at all and the block emits one completion
    record when it exits. Both themes end with the total, so a transcript still says how long the
    step took.

    Eden's own rather than ``genesis.logging.time_elapser.TimeElapser``, which is wrong under the
    agent theme in two ways and would put ``import genesis`` on ``import eden``'s cold path. It
    slices ``logger.last_output[:-4]`` for an ANSI reset the agent banner never emits -- eating
    four real characters of the message -- and its ``run()`` writes a literal ``\x1b[1F`` to move
    the cursor up, in the rendering whose whole contract is no ANSI anywhere. The reset is
    *checked for* here rather than assumed, and the cursor is only moved on the path that has one
    to move.

    Parameters
    ----------
    logger : Logger
        The logger to write through; its resolved theme picks the rendering.
    msg : str
        The message :meth:`Logger.timer` already logged, repeated on the agent completion line.
    refresh_rate : float
        Ticker updates per second, and the source of the elapsed time's precision.
    end_msg : str
        Appended after the elapsed time on the completion line.
    """

    def __init__(self, logger: "Logger", msg: str, refresh_rate: float, end_msg: str) -> None:
        if refresh_rate <= 0:
            raise ValueError(f"refresh_rate must be positive, got {refresh_rate!r}.")
        self._logger = logger
        self._msg = msg
        self._end_msg = end_msg
        self._dt = 1.0 / refresh_rate
        # One decimal at 10 Hz, two at 100 Hz: never print digits the ticker cannot move.
        self._decimals = max(0, math.ceil(math.log10(refresh_rate)))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t_start = 0.0

        # Nothing was emitted at all if the logger filters INFO out, and `logger.last_output`
        # then still holds some *earlier* record -- which is the line the ticker would repaint,
        # and what Genesis' version did. A timer on a quiet logger stays quiet end to end.
        self._silent = logger.level > logging.INFO
        self._start_msg = ""
        self._reset = ""

    def __enter__(self) -> "TimeElapser":
        # Read at entry, not at construction: `Logger.timer` validates `refresh_rate` by building
        # this object *before* it logs `msg`, so that a rejected call leaves no record behind --
        # which means construction runs one record too early to see the line the ticker repaints.
        #
        # Under "human" that record ends with the formatter's reset; strip it so the counter is
        # appended *inside* the colored line, and put it back on the closing one. Under "agent"
        # there is no reset to strip -- the check is the fix, Genesis' unconditional `[:-4]` ate
        # four characters of the message instead.
        last_output = self._logger.last_output
        if last_output.endswith(ANSI_RESET):
            self._start_msg = last_output[: -len(ANSI_RESET)]
            self._reset = ANSI_RESET
        else:
            self._start_msg = last_output
            self._reset = ""

        self._t_start = time.perf_counter()
        # The ticker repaints one terminal line in place, which takes cursor escapes and a
        # terminal to obey them. The agent theme has neither by definition, and a line per tick
        # would be the worst of both -- so it gets the completion record only.
        if self._logger.theme == "human" and not self._silent:
            # Step back onto the line `Logger.timer` just emitted, *here* rather than inside the
            # ticker, so every write after this one is a plain carriage return and the state is
            # the same whether or not a tick got in before the block finished.
            self._logger.raw(ANSI_PREV_LINE)
            self._stop.clear()
            # Daemon: a `with` block left by an unhandled exception must not keep the
            # interpreter alive on a progress indicator.
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        elapsed = time.perf_counter() - self._t_start
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._silent:
            return
        suffix = f" {self._end_msg}" if self._end_msg else ""
        if self._logger.theme == "human":
            self._logger.raw(f"\r{self._start_msg} {self._format(elapsed)} \u2705{suffix}{self._reset}\n")
        else:
            # A record, not a raw write: off a terminal this line is the only trace the step
            # leaves, so it carries the banner and the timestamp like every other one.
            self._logger.info(f"{self._msg} done in {self._format(elapsed)}{suffix}")

    def _format(self, elapsed: float) -> str:
        return f"{elapsed:.{self._decimals}f}s"

    def _run(self) -> None:
        """Repaint the opening line with a live counter until :meth:`__exit__` stops it."""
        while not self._stop.is_set():
            with self._logger.lock_timer():
                elapsed = time.perf_counter() - self._t_start
                self._logger.raw(f"\r{self._start_msg} {self._format(elapsed)}{self._reset}")
            self._stop.wait(self._dt)
