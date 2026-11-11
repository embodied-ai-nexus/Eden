"""Tests for ``eden.utils.cli.pose``: ``eden pose gvhmr`` parses its args before importing the optional backend."""

import pytest

from eden.utils.cli.pose import _run_gvhmr


def test_help_reaches_argparse_without_the_backend_installed():
    with pytest.raises(SystemExit) as exc:
        _run_gvhmr(["--help"])
    assert exc.value.code == 0


@pytest.mark.parametrize("argv", [[], ["--src", "clip.mp4"], ["--out", "out/"]], ids=["none", "no-out", "no-src"])
def test_a_missing_required_flag_is_a_usage_error(argv):
    with pytest.raises(SystemExit) as exc:
        _run_gvhmr(argv)
    assert exc.value.code == 2
