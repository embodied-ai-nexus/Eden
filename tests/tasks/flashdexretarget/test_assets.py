"""Tests for ``eden.tasks.flashdexretarget.assets``: what the task reads off a FlashDexRetarget robot."""

from __future__ import annotations

from pathlib import Path

import pytest

from eden.tasks.flashdexretarget.assets import _urdf_parent_link


def test_only_the_palm_body_may_take_the_urdfs_renamed_hand_link():
    """The converter renames each palm body to <side>_hand_link; any other unmatched body (a fingertip's, say) is
    refused rather than read at the palm.
    """
    links = {"right_hand_link", "right_hand_index_rota_link2"}
    urdf = Path("xhand_right_spec.urdf")
    assert _urdf_parent_link(links, urdf, "R_hand_palm", "right", palm=True) == "right_hand_link"
    assert _urdf_parent_link(links, urdf, "right_hand_index_rota_link2", "right") == "right_hand_index_rota_link2"
    with pytest.raises(ValueError, match=r"No link of xhand_right_spec\.urdf matches MJCF body 'R_index_tip'"):
        _urdf_parent_link(links, urdf, "R_index_tip", "right")
