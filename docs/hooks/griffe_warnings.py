"""MkDocs hook: demote griffe's docstring warnings so that ``mkdocs build --strict`` fails on everything else.

The API pages render about 2.7k pre-existing griffe warnings (mostly NumPy ``Parameters`` sections on Pydantic models
that do not match their ``__init__`` signatures). Under ``--strict`` they would fail every build, so the build could
not also fail on what matters for the site: a link to a missing page or anchor, or a page left out of the nav
(``validation:`` in ``mkdocs.yml``). This hook lowers only griffe's records to DEBUG, which ``--strict`` does not
count, and prints how many it lowered. ``mkdocs build -v`` shows them.

So that lowering them cannot hide new ones, the count is a ratchet: a full build with more than
``MAX_GRIFFE_WARNINGS`` logs a warning, which fails ``--strict``. When you fix docstrings, lower the ceiling to the
new count in the same change.

Delete this hook, and its ``hooks:`` entry in ``mkdocs.yml``, once the docstrings are clean.
"""

import logging

#: The griffe warning count of a full build on ``main``, with the toolchain pinned in
#: ``.github/actions/docs-toolchain/constraints.txt``. Lower it as docstrings are fixed. Raise it only in a change that
#: bumps those pins and adds warnings by doing so, and say so in the commit message.
MAX_GRIFFE_WARNINGS = 2611

# griffe logs through mkdocstrings' logger factory, under the ``mkdocs.plugins`` namespace.
_GRIFFE_LOGGER = logging.getLogger("mkdocs.plugins.griffe")
_MKDOCS_LOGGER = logging.getLogger("mkdocs")
_log = logging.getLogger("mkdocs.hooks.griffe_warnings")


class _DemoteToDebug(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno != logging.WARNING:
            return True
        record.levelno, record.levelname = logging.DEBUG, "DEBUG"
        self.count += 1
        # The console handler sits on the ``mkdocs`` logger with no level of its own, so a lowered record would still
        # print. Keep it only when that logger is at DEBUG (``mkdocs build -v``).
        return _MKDOCS_LOGGER.isEnabledFor(logging.DEBUG)


_filter = _DemoteToDebug()


#: Whether every build of this process renders every page: ``mkdocs build`` / ``gh-deploy`` without ``--dirty``. A
#: ``serve`` rebuild can reuse what an earlier one collected, so its count is no measure to lower the ceiling to.
_full_build = False


def on_startup(command, dirty) -> None:
    global _full_build
    _full_build = command in ("build", "gh-deploy") and not dirty
    # A logger's filters run before its handlers and before the record propagates to the ``mkdocs`` logger, where
    # the strict-mode counter and the console handler sit.
    if _filter not in _GRIFFE_LOGGER.filters:
        _GRIFFE_LOGGER.addFilter(_filter)


def on_pre_build(config) -> None:
    _filter.count = 0  # ``mkdocs serve`` rebuilds in the same process


def on_post_build(config) -> None:
    # mkdocs checks the strict-mode count after this hook, so a warning logged here fails ``--strict``.
    if _filter.count > MAX_GRIFFE_WARNINGS:
        _log.warning(
            f"{_filter.count} griffe docstring warnings, more than the {MAX_GRIFFE_WARNINGS} allowed "
            f"(MAX_GRIFFE_WARNINGS in docs/hooks/griffe_warnings.py). Run `mkdocs build -v` and fix the new ones "
            f"in the docstrings your change touched, rather than raising the ceiling."
        )
    elif _filter.count:
        _log.info(f"{_filter.count} griffe docstring warnings shown as DEBUG (see docs/hooks/griffe_warnings.py)")
        if _full_build and _filter.count < MAX_GRIFFE_WARNINGS:
            _log.info(f"Lower MAX_GRIFFE_WARNINGS in docs/hooks/griffe_warnings.py to {_filter.count}.")
