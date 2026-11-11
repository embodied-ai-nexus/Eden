"""Fakes of the two input-device libraries, installed where the device modules look them up.

``pynput`` needs a local display and ``inputs`` a plugged-in pad, so both are replaced by small fakes
that let a test press keys or move sticks; the Eden device classes are constructed for real on top.
"""

from __future__ import annotations

import sys
import time
import types
from collections import deque
from types import SimpleNamespace

import pytest

from eden.utils.devices import gamepad as gamepad_module
from eden.utils.devices import keyboard as keyboard_module


class _SpecialKey:
    """A ``pynput.keyboard.Key`` member: no ``.char``."""

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"Key.{self.name}"


@pytest.fixture
def pynput(monkeypatch):
    """A fake ``pynput``: ``pynput.press(key)`` / ``pynput.release(key)`` reach every started listener."""
    listeners: list = []

    class Listener:
        def __init__(self, on_press, on_release):
            self.on_press, self.on_release = on_press, on_release
            self.daemon = False
            self.running = False
            listeners.append(self)

        def start(self):
            self.running = True

        def stop(self):
            self.running = False

    key = SimpleNamespace(**{name: _SpecialKey(name) for name in ("up", "down", "left", "right", "space")})
    keyboard_mod = types.ModuleType("pynput.keyboard")
    keyboard_mod.Listener, keyboard_mod.Key = Listener, key
    root = types.ModuleType("pynput")
    root.keyboard = keyboard_mod
    monkeypatch.setitem(sys.modules, "pynput", root)
    monkeypatch.setitem(sys.modules, "pynput.keyboard", keyboard_mod)
    # The device module bound pynput at its own import, which failed here (no display, no package).
    monkeypatch.setattr(keyboard_module, "IS_PYNPUT_INSTALLED", True)
    monkeypatch.setattr(keyboard_module, "keyboard", keyboard_mod, raising=False)
    monkeypatch.setattr(keyboard_module, "Key", key, raising=False)

    def _event(k):
        return SimpleNamespace(char=k) if isinstance(k, str) else k

    def press(k):
        for listener in listeners:
            if listener.running:
                listener.on_press(_event(k))

    def release(k):
        for listener in listeners:
            if listener.running:
                listener.on_release(_event(k))

    return SimpleNamespace(Key=key, press=press, release=release, listeners=listeners)


class FakePad:
    """A gamepad on ``inputs``: ``send(code=state, ...)`` queues one event batch; ``unplug()`` pulls it out."""

    def __init__(self):
        self.batches: deque = deque()
        self.unplugged = False

    def send(self, **states):
        self.batches.append([SimpleNamespace(code=code, state=state) for code, state in states.items()])

    def unplug(self):
        self.unplugged = True

    def get_gamepad(self):
        if self.unplugged:
            raise OSError("device disconnected")
        if self.batches:
            return self.batches.popleft()
        time.sleep(0.002)
        return []

    def drained(self, timeout: float = 2.0) -> bool:
        """Wait until the device's reader thread has consumed every queued batch."""
        deadline = time.monotonic() + timeout
        while self.batches and time.monotonic() < deadline:
            time.sleep(0.002)
        time.sleep(0.01)  # the batch just popped is applied under the device lock
        return not self.batches


@pytest.fixture
def pad(monkeypatch):
    fake = FakePad()
    monkeypatch.setattr(gamepad_module, "IS_INPUTS_INSTALLED", True)
    monkeypatch.setattr(gamepad_module, "input_devices", SimpleNamespace(gamepads=[fake]), raising=False)
    monkeypatch.setattr(gamepad_module, "get_gamepad", fake.get_gamepad, raising=False)
    return fake
