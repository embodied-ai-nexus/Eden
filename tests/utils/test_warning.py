"""Tests for ``eden.utils.warning.warn_once``: one warning per distinct message text."""

from __future__ import annotations

import pytest

from eden.utils.warning import warn_once


@pytest.fixture(autouse=True)
def _clear_cache():
    warn_once.cache_clear()
    yield
    warn_once.cache_clear()


def test_identical_text_warns_once(eden_caplog):
    for _ in range(3):
        warn_once("[policy] 'Foo' has no _commands buffer")

    assert len([r for r in eden_caplog.records if "_commands" in r.message]) == 1


def test_differing_text_warns_each_time(eden_caplog):
    warn_once("[policy] 'Foo' has no _commands buffer")
    warn_once("[policy] 'Bar' has no _commands buffer")

    assert len([r for r in eden_caplog.records if "_commands" in r.message]) == 2
