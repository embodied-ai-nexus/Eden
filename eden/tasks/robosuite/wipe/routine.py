"""A tool routine that wipes robosuite's table clean (``eden datagen --task robosuite_wipe``).

Written against :class:`~eden.extensions.scripted_demos.tools.Tools`, for the Panda, as the
routines of :mod:`eden.extensions.scripted_demos.routines` are: close the fingers into a flat pad,
lower it onto the first dirt marker, sweep it over every marker in the order the dirt path lays them
down, and lift off. With probability ``p_mistake`` the hand lifts off partway, as if done, then sees
the dirt still left and comes back for it.

The markers are kinematic (no collision geometry, so no footprint): their positions are read with
the privileged :meth:`~eden.extensions.scripted_demos.tools.Tools.link_pose`, once, after the reset
has laid the path. The table top is the table's footprint.

What the task asks of the hand, found in rollouts:

* **Press lightly, and level.** The pad is position-controlled: commanded into the table it pushes
  back at tens of newtons per millimetre, and robosuite's excess-force penalty starts at 60 N. So
  the pad feels for the table (:meth:`~eden.extensions.scripted_demos.tools.Tools.move_until_contact`)
  rather than trusting the fingertip reach measured on the tilted home pose (a few millimetres long,
  so a level pad aimed by it never touched), and sweeps :data:`SWEEP_LIFT` above where it touched,
  square to the table — tilted as the home pose leaves it, one corner dug in to 60 N.
* **Stop at every marker.** A marker counts once the pad midpoint passes within 2 cm of it; the
  markers are 3 cm apart along a path that turns, and a move that skips one cuts the corner past it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from eden.extensions.scripted_demos.tools import CONTACT
from eden.tasks.robosuite.wipe.config import MARKER_NAMES

if TYPE_CHECKING:
    from collections.abc import Generator

    from eden.extensions.scripted_demos.routines import Routine
    from eden.extensions.scripted_demos.tools import Motion, Outcome, Tools

#: The table the markers lie on.
TABLE = "table"
#: How far [m] above the height it touched the table at the pad sweeps: the arm sags onto the table
#: by a millimetre or two that varies across it, and swept at the touch height the pad pressed with
#: up to 65 N.
SWEEP_LIFT = 0.0015
#: Height [m] above the table the pad descends to before feeling for it, the speed [m/s] it feels at,
#: and how far [m] past the expected surface it may go. Felt for at 3 cm/s, the hand trailed its
#: command by ~2.7 cm and met the table at up to 89 N.
TOUCH_FROM = 0.01
TOUCH_SPEED = 0.005
TOUCH_REACH = 0.02
#: Hover [m] of the fingertips above the table before lowering onto a marker, drawn per descent.
HOVER = (0.04, 0.08)
#: How far [m] off each marker's centre the pad aims, drawn per marker: inside the 2 cm the task counts.
AIM_JITTER = 0.006
#: Tolerance [m] of the sweep's moves: the pad drags on the table, and the next marker is 3 cm on.
SWEEP_TOLERANCE = 0.012
#: How long [s] the hand pauses above the table after the deliberate early lift-off, before coming back.
MISTAKE_PAUSE = (0.5, 1.0)
#: The earliest and latest marker the deliberate lift-off happens before.
MISTAKE_RANGE = (4, len(MARKER_NAMES) - 3)


def wipe_routine(*, p_mistake: float, markers: tuple[str, ...] = MARKER_NAMES) -> Routine:
    """The routine that sweeps the pad over ``markers``, in order."""

    def wipe(tools: Tools) -> Generator[Motion, Outcome, None]:
        rng = tools.rng
        surface = tools.object_footprint(TABLE).top_z
        path = [tools.link_pose(name)[0] for name in markers]
        lift_before = int(rng.integers(*MISTAKE_RANGE)) if rng.random() < p_mistake else None
        yield tools.close_gripper()
        k = 0
        while k < len(path):
            sweep_z = yield from _lower_onto(tools, path[k], surface)
            end = lift_before if lift_before is not None and lift_before > k else len(path)
            for marker in path[k + 1 : end]:
                yield tools.move_to(_aim(rng, marker, sweep_z), tolerance=SWEEP_TOLERANCE)
            if end < len(path):
                tools.note(f"mistake: lifting off with {len(path) - end} markers left")
                yield tools.move_by((0.0, 0.0, rng.uniform(*HOVER)))
                yield tools.wait(rng.uniform(*MISTAKE_PAUSE))
                tools.note("dirt left on the table: back to it")
            lift_before = None
            k = end
        yield tools.move_by((0.0, 0.0, rng.uniform(*HOVER)))

    wipe.entity_names = (TABLE, *markers)
    return wipe


def _lower_onto(tools: Tools, marker: np.ndarray, surface: float) -> Generator[Motion, Outcome, float]:
    """Hover over ``marker``, lower the pad until it touches the table, square to it; returns the sweep height."""
    rng = tools.rng
    above = _aim(rng, marker, surface + tools.tip_reach + rng.uniform(*HOVER))
    # A heading levels the hand (a move without one keeps the home pose's ~7 degree tilt, and a tilted
    # pad digs one corner in: 60 N at 1.5 mm); the later moves hold it.
    yield tools.move_to(above, tools.tool_yaw())
    yield tools.move_to((above[0], above[1], surface + tools.tip_reach + TOUCH_FROM))
    touch = yield tools.move_until_contact((0.0, 0.0, -1.0), TOUCH_FROM + TOUCH_REACH, speed=TOUCH_SPEED)
    if touch.status != CONTACT:
        tools.note(f"the pad did not feel the table ({touch.status}); sweeping where the table should be")
        return surface + tools.tip_reach + SWEEP_LIFT
    return float(tools.tool_pos()[2]) + SWEEP_LIFT


def _aim(rng: np.random.Generator, marker: np.ndarray, z: float) -> tuple[float, float, float]:
    """A point within :data:`AIM_JITTER` of ``marker``'s centre, at height ``z``."""
    dx, dy = rng.uniform(-AIM_JITTER, AIM_JITTER, size=2)
    return float(marker[0] + dx), float(marker[1] + dy), float(z)
