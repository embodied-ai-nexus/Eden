"""``omnireset.retarget``: which UWLab rows count as holding the object, and what the retargeted cache is named after."""

from __future__ import annotations

import torch

from eden.tasks.omnireset.constants import PAIRS
from eden.tasks.omnireset.retarget import held_by_object


def test_a_uwlab_gripper_counts_as_holding_only_when_stopped_partway():
    finger = torch.tensor([0.0, 0.02, 0.3, 0.52, 0.76, 0.78, 0.785])
    assert held_by_object(finger).tolist() == [False, False, True, True, True, False, False]


def test_the_retargeted_buffer_is_named_after_the_uwlab_buffer_it_is_made_from(monkeypatch):
    import dataclasses

    from eden.tasks.omnireset import retarget

    first = retarget.retargeted_path("peg")
    # A re-pinned centre of mass re-converts UWLab's buffer, so the rows retargeted from the old one must not be read.
    monkeypatch.setitem(PAIRS, "peg", dataclasses.replace(PAIRS["peg"], insertive_com=(0.0, 0.0, 0.01)))
    moved_com = retarget.retargeted_path("peg")
    assert moved_com != first
    monkeypatch.setattr(retarget, "RETARGET_VERSION", retarget.RETARGET_VERSION + 1)
    assert retarget.retargeted_path("peg") not in (first, moved_com)


def test_the_retargeted_buffer_follows_the_robot_it_was_settled_on(monkeypatch):
    from eden.tasks.omnireset import constants, retarget

    before = retarget.retargeted_path("peg")
    monkeypatch.setattr(
        constants, "ROBOT_FRICTION", constants.ROBOT_FRICTION + 0.1
    )  # the pads the gripper settles with
    assert retarget.retargeted_path("peg") != before
