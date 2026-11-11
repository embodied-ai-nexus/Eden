"""Unitree Go1 legacy SDK setup helper (``robot_interface`` UDP module).

Unlike the Go2/G1 backends (which use the DDS-based ``unitree_sdk2py``), the
Go1 ships the *legacy* ``unitree_legged_sdk`` whose Python binding is a single
pybind11 module named ``robot_interface`` (synchronous ``UDP.Send``/``Recv``,
``LowCmd``/``LowState`` structs, ``Safety``). This shim guards the import so a
missing SDK leaves the backend merely unavailable instead of breaking package
import, and gives a helpful build/install message via :func:`ensure_go1_sdk`.
"""

from __future__ import annotations

import os
import sys

# The legacy SDK does NOT export the control-level constants; the SDK examples
# hardcode them Python-side (see ``example_py/example_position.py``).
LOWLEVEL = 0xFF
HIGHLEVEL = 0xEE

# Optionally make an *uninstalled* build importable: point EDEN_GO1_SDK_PATH at
# the directory holding ``robot_interface.*.so`` (e.g. the SDK repo root).
_sdk_path = os.environ.get("EDEN_GO1_SDK_PATH")
if _sdk_path and _sdk_path not in sys.path:
    sys.path.insert(0, _sdk_path)

IS_GO1_SDK_AVAILABLE = True
_GO1_SDK_IMPORT_ERROR: Exception | None = None
try:
    import robot_interface as robot_interface
except Exception as _exc:  # ImportError, or a native-load failure
    IS_GO1_SDK_AVAILABLE = False
    _GO1_SDK_IMPORT_ERROR = _exc
    robot_interface = None


__all__ = [
    "HIGHLEVEL",
    "IS_GO1_SDK_AVAILABLE",
    "LOWLEVEL",
    "ensure_go1_sdk",
    "robot_interface",
]


def ensure_go1_sdk() -> None:
    if not IS_GO1_SDK_AVAILABLE:
        cause = _GO1_SDK_IMPORT_ERROR
        detail = f"{type(cause).__name__}: {cause}" if cause is not None else "import failed"
        raise RuntimeError(
            "Unitree Go1 SDK (the legacy `robot_interface` module) failed to import.\n"
            "Build and install it from the unitree_legged_sdk python wrapper "
            "(`python -m build --wheel` then `pip install`), or point "
            "EDEN_GO1_SDK_PATH at the directory containing `robot_interface.*.so`.\n"
            f"Underlying error: {detail}"
        ) from cause
