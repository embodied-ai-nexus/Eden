"""Fakes of the XRoboToolkit SDK boundary, shared by the PICO client and bridge tests.

``xrobotoolkit_sdk`` is a pybind module built against the PC Service, which talks to a headset, so the
tests replace it at the import system: :class:`FakeSdk` is its surface with a scripted headset,
:func:`write_fake_sdk_module` turns that into an importable ``xrobotoolkit_sdk.py`` for a subprocess, and
:class:`SdkImportFinder` reproduces how the real module fails to import until the service's shared
library is loaded into the process.
"""

from __future__ import annotations

import ctypes
import importlib
import importlib.abc
import importlib.util
import inspect
import os
import shutil
import sys
from pathlib import Path

import pytest

from eden.utils.devices.pico.xr_client import N_BODY_JOINTS, PicoXrClient

SDK_MODULE = "xrobotoolkit_sdk"
PXREA_LIB = "libPXREARobotSDK.so"  # the PC Service library the pybind module links against


class FakeSdk:
    """The ``xrobotoolkit_sdk`` extension module's surface, with a scripted headset, trackers and body."""

    def __init__(self, trackers=(), body=False):
        self._trackers = list(trackers)  # each: (serial, pose7, vel6, acc6)
        self._body = body

    # lifecycle / scalars ------------------------------------------------
    def init(self):
        pass

    def close(self):
        pass

    def get_time_stamp_ns(self):
        return 123

    # a device on the floor: identity-ish head so `ready` is True
    def get_headset_pose(self):
        return [0.0, 0.0, 1.6, 0.0, 0.0, 0.0, 1.0]

    def get_left_controller_pose(self):
        return [0.1, 0.2, 1.0, 0.0, 0.0, 0.0, 1.0]

    def get_right_controller_pose(self):
        return [-0.1, 0.2, 1.0, 0.0, 0.0, 0.0, 1.0]

    def get_left_trigger(self):
        return 0.0

    def get_right_trigger(self):
        return 0.0

    def get_left_grip(self):
        return 0.0

    def get_right_grip(self):
        return 0.0

    def get_left_axis(self):
        return [0.0, 0.0]

    def get_right_axis(self):
        return [0.0, 0.0]

    def _false(self, *_):
        return False

    get_X_button = get_Y_button = get_A_button = get_B_button = _false
    get_left_menu_button = get_right_menu_button = _false
    get_left_axis_click = get_right_axis_click = _false
    get_left_hand_is_active = get_right_hand_is_active = _false

    def get_left_hand_tracking_state(self):
        return [[0.0] * 7 for _ in range(26)]

    get_right_hand_tracking_state = get_left_hand_tracking_state

    # motion trackers ----------------------------------------------------
    def num_motion_data_available(self):
        return len(self._trackers)

    def get_motion_tracker_pose(self):
        return [t[1] for t in self._trackers]

    def get_motion_tracker_velocity(self):
        return [t[2] for t in self._trackers]

    def get_motion_tracker_acceleration(self):
        return [t[3] for t in self._trackers]

    def get_motion_tracker_serial_numbers(self):
        return [t[0] for t in self._trackers]

    # full body ----------------------------------------------------------
    def is_body_data_available(self):
        return self._body

    def get_body_joints_pose(self):
        return [[0.0, 0.0, float(i) * 0.1, 0.0, 0.0, 0.0, 1.0] for i in range(N_BODY_JOINTS)]

    def get_body_joints_velocity(self):
        return [[0.0] * 6 for _ in range(N_BODY_JOINTS)]

    def get_body_joints_acceleration(self):
        return [[0.0] * 6 for _ in range(N_BODY_JOINTS)]


# The module a subprocess imports as ``xrobotoolkit_sdk``: FakeSdk's own source, one instance serving the
# module-level functions, and init/close appended to ``sdk_events.log`` next to the module so a test can
# read back what the process did to the SDK.
_FAKE_SDK_MODULE = '''"""Fake xrobotoolkit_sdk, written by tests/utils/devices/pico/conftest.py."""

import os

from eden.utils.devices.pico.xr_client import N_BODY_JOINTS

{fake_sdk_source}

_SDK = FakeSdk()
EVENTS_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sdk_events.log")


def _record(event):
    with open(EVENTS_LOG, "a") as f:
        f.write(event + "\\n")


def init():
    _SDK.init()
    _record("init")


def close():
    _SDK.close()
    _record("close")


def __getattr__(name):
    return getattr(_SDK, name)
'''


def write_fake_sdk_module(directory: Path) -> Path:
    """Write an importable ``xrobotoolkit_sdk.py`` into ``directory``; returns its events-log path."""
    directory.mkdir(parents=True, exist_ok=True)
    source = _FAKE_SDK_MODULE.format(fake_sdk_source=inspect.getsource(FakeSdk))
    (directory / f"{SDK_MODULE}.py").write_text(source)
    return directory / "sdk_events.log"


def library_is_loaded(path: Path) -> bool:
    """Whether the shared library at ``path`` is mapped into this process (``dlopen`` with ``RTLD_NOLOAD``)."""
    try:
        ctypes.CDLL(str(path), mode=os.RTLD_NOLOAD)
    except OSError:
        return False
    return True


def copy_loadable_library(directory: Path) -> Path:
    """A fresh, loadable stand-in for ``libPXREARobotSDK.so`` in ``directory``, not yet loaded by anyone.

    A copy of a small stdlib extension module: ``dlopen`` treats a new inode as a new library, so the copy
    starts out unloaded and only the code under test can load it.
    """
    for name in ("_bisect", "_json", "_csv", "_struct"):
        source = getattr(importlib.import_module(name), "__file__", None)
        if source and source.endswith(".so"):
            directory.mkdir(parents=True, exist_ok=True)
            return Path(shutil.copyfile(source, directory / PXREA_LIB))
    pytest.skip("no stdlib extension module ships as a .so in this interpreter to stand in for the PC library")


class SdkImportFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """``import xrobotoolkit_sdk`` as the real pybind build behaves: it links only against a loaded library.

    ``links_against`` is the ``libPXREARobotSDK.so`` this build was linked with; until that file is loaded
    into the process each import raises the dynamic linker's ImportError. ``None`` is a build that never
    links (e.g. against a different service version). ``errors`` holds the ImportError of every failed
    attempt and ``attempts`` counts them all.
    """

    def __init__(self, links_against: Path | None) -> None:
        self.links_against = links_against
        self.attempts = 0
        self.errors: list[ImportError] = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname != SDK_MODULE:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        self.attempts += 1
        if self.links_against is None or not library_is_loaded(self.links_against):
            error = ImportError(
                f"{PXREA_LIB}: cannot open shared object file: No such file or directory (attempt {self.attempts})"
            )
            self.errors.append(error)
            raise error
        # Linked: fall through to the default module creation.

    def exec_module(self, module):
        pass


@pytest.fixture
def sdk_import(monkeypatch):
    """Route ``import xrobotoolkit_sdk`` through a finder: ``sdk_import(finder)`` installs it first in line."""
    # Record the current entry (absent included) so a successful import is undone at teardown:
    # ``delitem`` alone records nothing for a missing key.
    monkeypatch.setitem(sys.modules, SDK_MODULE, None)
    monkeypatch.delitem(sys.modules, SDK_MODULE)

    def install(finder: SdkImportFinder) -> SdkImportFinder:
        monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
        return finder

    return install


@pytest.fixture
def fake_client(monkeypatch):
    """Build a PicoXrClient over a fake SDK, installed where ``import xrobotoolkit_sdk`` finds it."""

    def make(sdk, tracker_roles=None):
        monkeypatch.setitem(sys.modules, SDK_MODULE, sdk)
        return PicoXrClient(tracker_roles=tracker_roles)

    return make
