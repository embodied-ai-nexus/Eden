"""Tests for ``eden terms``: what lands on stdout and stderr, and the exit code, run through ``eden.__main__.main``."""

import json
import sys

import pytest

import eden.__main__ as m


def _eden_terms(monkeypatch, *argv):
    """Run ``eden terms <argv>`` in-process and return the ``SystemExit`` code, or ``None`` if it returned."""
    monkeypatch.setattr(sys, "argv", ["eden", "terms", *argv])
    try:
        m.main()
    except SystemExit as exit_:
        return exit_.code
    return None


def test_no_match_under_json_is_an_empty_document_not_prose(monkeypatch, capsys):
    """``--json`` is for tooling: a sentence on stdout is one a pipe cannot parse."""
    assert _eden_terms(monkeypatch, "--grep", "definitely_not_a_term_anywhere", "--json") == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out) == []
    assert "No term matches" in captured.err


def test_no_match_explains_on_stderr(monkeypatch, capsys):
    assert _eden_terms(monkeypatch, "--grep", "definitely_not_a_term_anywhere") == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "searched" in captured.err


@pytest.mark.parametrize(
    ("argv", "expected"),
    [(["--grep", "dofs[0"], "not a valid regular expression"), (["--kind", "rewards"], "unknown term kind")],
    ids=["bad-regex", "bad-kind"],
)
def test_a_usage_error_goes_to_stderr_and_exits_two(monkeypatch, capsys, argv, expected):
    assert _eden_terms(monkeypatch, *argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert expected in captured.err


def test_a_match_prints_the_filtered_catalog_on_stdout(monkeypatch, capsys):
    assert _eden_terms(monkeypatch, "--kind", "reward", "--grep", "height") is None
    captured = capsys.readouterr()
    assert "matching 'height'" in captured.out
    assert captured.err == ""
