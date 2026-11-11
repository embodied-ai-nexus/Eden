"""Tests for ``eden/resources/term_catalog.md``: it is current, it ships in the wheel, and it stays an index.

Regenerate the checked-in pages with ``uv run python -m eden.managers.catalog``.
"""

from __future__ import annotations

import importlib
import re
import tomllib
from fnmatch import fnmatch
from functools import reduce
from pathlib import Path

import pytest

import eden
from eden.managers.catalog import (
    CATALOG_PAGE,
    DOCS_CATALOG_PAGE,
    INDEX_PAGE_BUDGET,
    KINDS,
    REPO_ROOT,
    builtin_catalog,
    kind_shared_specs,
    render_index,
    render_markdown,
)

# Only the shipped terms: another test module in this session may have registered its own.
ENTRIES = builtin_catalog()


@pytest.mark.parametrize(
    ("page", "render"),
    [(CATALOG_PAGE, render_index), (DOCS_CATALOG_PAGE, render_markdown)],
    ids=["shipped-index", "docs-catalog"],
)
def test_the_checked_in_page_is_current(page, render):
    assert page.exists(), f"{page} is missing — run: uv run python -m eden.managers.catalog"
    assert page.read_text() == render(ENTRIES), (
        f"{page.relative_to(REPO_ROOT)} is stale — run: uv run python -m eden.managers.catalog"
    )


def test_the_index_is_packaged_for_the_wheel():
    """Shipped, not merely current. ``pip install`` carries ``eden/`` but not ``docs/``.

    Checks the build config rather than building a wheel: the page must sit under the package and match a
    ``[tool.setuptools.package-data]`` pattern.
    """
    eden_pkg = Path(eden.__file__).resolve().parent
    page = CATALOG_PAGE.resolve()
    assert page.is_relative_to(eden_pkg), f"{page} is outside the eden package — packages.find won't ship it"

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    package_data = pyproject.get("tool", {}).get("setuptools", {}).get("package-data", {})
    owning_pkg = ".".join(["eden", *page.relative_to(eden_pkg).parent.parts])
    patterns = [*package_data.get("*", []), *package_data.get(owning_pkg, [])]
    assert any(fnmatch(page.name, pat) for pat in patterns), (
        f"[tool.setuptools.package-data] ships no pattern for '{owning_pkg}' (or '*') that matches "
        f"{page.name} — the wheel would omit the term catalog"
    )


class TestShippedIndex:
    """An agent reads the installed page whole and keeps it in context, so it lists terms without their parameters."""

    INDEX = CATALOG_PAGE.read_text()

    @classmethod
    def _lines_by_term(cls) -> dict[tuple[str, str, str], str]:
        """Each term line, keyed by the ``(kind, module, name)`` its sections place it under.

        Keyed by section rather than searched by name: seven names are registered in two kinds.
        """
        lines: dict[tuple[str, str, str], str] = {}
        kind = module = None
        for line in cls.INDEX.splitlines():
            if heading := re.fullmatch(r"## (\w+) terms \(\d+\)", line):
                kind, module = heading[1], None
            elif heading := re.fullmatch(r"`([\w.]+)`:", line):
                module = heading[1]
            elif term := re.match(r"- `(\w+)`", line):
                assert kind and module, f"term line outside a kind or module section: {line}"
                lines[(kind, module, term[1])] = line
        return lines

    def test_it_names_every_term_and_the_class_to_import_from_its_section_module(self):
        lines = self._lines_by_term()
        missing = [(e.kind, e.name) for e in ENTRIES if (e.kind, e.module, e.name) not in lines]
        assert not missing, f"the shipped index omits {missing}"
        for entry in ENTRIES:
            if entry.form == "function":
                continue
            assert lines[(entry.kind, entry.module, entry.name)].startswith(f"- `{entry.name}` (`{entry.qualname}`)")
            # The recipe says to import the class from the module its section names.
            assert reduce(getattr, entry.qualname.split("."), importlib.import_module(entry.module))

    def test_it_carries_no_parameter_types_or_defaults(self):
        spelled = {f"`{p.name}: " for e in ENTRIES for p in e.params}
        spelled |= {f"`{p.name}: " for kind in KINDS for p in kind_shared_specs(kind)}
        leaked = sorted(spec for spec in spelled if spec in self.INDEX)
        assert not leaked, f"the shipped index renders parameters: {leaked[:5]}"

    def test_it_keeps_the_env_options_a_term_requires(self):
        """A precondition no parameter can supply: dropping it lets a config written from the index fail at build."""
        needing = [e for e in ENTRIES if e.requires_env_options]
        assert needing, "no term declares requires_env_options any more; drop this test"
        lines = self._lines_by_term()
        for entry in needing:
            line = lines[(entry.kind, entry.module, entry.name)]
            assert all(flag in line for flag in entry.requires_env_options), line

    def test_it_points_at_where_the_parameters_are(self):
        assert "eden terms --grep <name>" in self.INDEX

    def test_it_stays_within_its_budget(self):
        size = len(self.INDEX.encode())
        assert size <= INDEX_PAGE_BUDGET, (
            f"the shipped index is {size:,} bytes, over its {INDEX_PAGE_BUDGET:,}-byte budget. It is read whole "
            "and re-sent on every later turn of an agent session, so each byte is paid many times. "
            "Shorten a term summary (the first docstring line) rather than raising the budget."
        )
