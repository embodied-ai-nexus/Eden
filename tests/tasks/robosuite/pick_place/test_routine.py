"""The grocery-sorting routine: where it puts things, whose cell an object is in, and a real run on the PickPlace bins."""

from __future__ import annotations

# isort: off
import eden  # noqa: F401 - isolate the DDP rank before importing Torch
# isort: on

import numpy as np
import pytest

from eden.extensions.scripted_demos.tools import Footprint
from eden.tasks.robosuite.pick_place.config import BIN_TOP_Z, OBJECT_NAMES, target_region
from eden.tasks.robosuite.pick_place.routine import PLACE_JITTER, _place_xy, _quadrant_of, pick_place_routine


def _footprint(x, y, *, bottom=BIN_TOP_Z, half=(0.025, 0.025)) -> Footprint:
    return Footprint(
        center=np.array([x, y, bottom + 0.04]),
        yaw=0.0,
        half_length=half[0],
        half_width=half[1],
        top_z=bottom + 0.08,
        bottom_z=bottom,
        top_center=np.array([x, y, bottom + 0.08]),
        num_points=100,
    )


class _Rng:
    """The routine's ``tools`` as far as _place_xy reads it: a random stream."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)


@pytest.mark.parametrize("cell", OBJECT_NAMES)
def test_an_object_is_lowered_inside_the_cell_on_the_robots_side(cell):
    low, high = target_region(cell)
    can = _footprint(0.0, 0.0)
    inset = np.hypot(0.025, 0.025)
    for seed in range(20):
        x, y = _place_xy(_Rng(seed), cell, can)
        # Its whole footprint inside the cell, and in the cell's near (low x, low y) half: the far
        # cells' centres are at the edge of the Panda's reach with the hand pointing down.
        assert low[0] + inset <= x <= (low[0] + high[0]) / 2
        assert low[1] + inset <= y <= (low[1] + high[1]) / 2
        assert x - (low[0] + inset) <= PLACE_JITTER + 0.0151


def test_an_object_is_in_the_cell_its_centre_rests_in_and_in_none_while_lifted():
    low, high = target_region("can")
    centre = ((low[0] + high[0]) / 2, (low[1] + high[1]) / 2)
    assert _quadrant_of(_footprint(*centre)) == "can"
    assert _quadrant_of(_footprint(*centre, bottom=BIN_TOP_Z + 0.1)) is None  # held above it
    assert _quadrant_of(_footprint(0.1, -0.25)) is None  # in bin1


def test_an_object_the_bins_have_no_cell_for_is_refused():
    with pytest.raises(KeyError, match=r"No bin2 quadrant for objects \['mug'\]"):
        pick_place_routine(("can", "mug"), p_mistake=0.0, camera=None)
