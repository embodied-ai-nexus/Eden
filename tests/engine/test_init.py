"""``eden.engine``: the package stays off ``import eden``'s path."""

from __future__ import annotations

import subprocess
import sys


def test_import_eden_does_not_import_the_engine_package():
    """A sensor is paid for only by the scene that declares it."""
    code = "import sys, eden; sys.exit(int(any(m.startswith('eden.engine') for m in sys.modules)))"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
