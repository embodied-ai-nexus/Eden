"""Robot-agnostic operator command vocabulary and keyboard input for deploy state machines.

Every deployment backend runs the same operator state machine, driven by the same
discrete intents — START / STOP / INIT / KILL / policy switching — while a robot with
a physical remote (Unitree G1, LimX Oli, ...) supplies its own reader for it.
This module holds the shared vocabulary and the line-based keyboard reader;
the per-robot remote readers live next to their backends.
"""

from __future__ import annotations

import sys
import threading
from collections import deque

# --- Operator state-machine commands ------------------------------------------
# Discrete operator intents, decoupled from the physical input that produced
# them (remote button/chord vs. keyboard key). Mirrors Holosoma's
# ``StateCommand`` enum but kept dependency-free.
CMD_START = "start"  # enable policy actions
CMD_STOP = "stop"  # hold current measured pose with PD
CMD_INIT = "init"  # interpolate smoothly to the ``init_target`` pose
CMD_KILL = "kill"  # soft-damp and exit
CMD_NEXT_POLICY = "next_policy"  # cycle to the next loaded policy
# Sim-only rig controls: crank the simulated gantry up / down. Backends with no gantry
# (every real robot) ignore them — the vocabulary is shared, the capability is not.
CMD_GANTRY_UP = "gantry_up"
CMD_GANTRY_DOWN = "gantry_down"
# Jump to an explicit 1-based policy slot, e.g. "switch_policy:2". Resolved to a
# 0-based index by the deploy loop, which owns the policy list (and its bounds).
SWITCH_PREFIX = "switch_policy:"

# Keyboard keys → commands. Line-based (type the key, then Enter), so it works
# over plain SSH without raw-tty handling or extra dependencies.
KEYBOARD_COMMANDS: dict[str, str] = {
    "s": CMD_START,
    "x": CMD_STOP,
    "i": CMD_INIT,
    "k": CMD_KILL,
    "n": CMD_NEXT_POLICY,
    "+": CMD_GANTRY_UP,
    "-": CMD_GANTRY_DOWN,
    # Digits 1-9 jump directly to that policy slot (mirrors Holosoma's
    # SWITCH_POLICY_1..9 keyboard binding).
    **{str(d): f"{SWITCH_PREFIX}{d}" for d in range(1, 10)},
}


class KeyboardInput:
    """Line-based stdin command reader (type a key, then Enter).

    A daemon thread drains ``sys.stdin`` so the control loop never blocks.
    ``start()`` returns False when there is no TTY (e.g. headless), letting
    the caller fall back to the remote only.
    """

    def __init__(self) -> None:
        self._queue: deque[str] = deque()
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        if not sys.stdin.isatty():
            return False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def _run(self) -> None:
        try:
            for line in sys.stdin:
                tok = line.strip().lower()
                if tok:
                    self._queue.append(tok)
        except Exception:
            pass

    def poll(self) -> list[str]:
        cmds: list[str] = []
        while self._queue:
            tok = self._queue.popleft()
            cmd = KEYBOARD_COMMANDS.get(tok)
            if cmd is not None:
                cmds.append(cmd)
        return cmds
