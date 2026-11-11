"""Tests for ``eden.extensions.retargeting.flashdexretarget.quality``: the flags of a retargeted clip's replay."""

from __future__ import annotations

import pytest

from eden.extensions.retargeting.flashdexretarget import scene as S
from eden.extensions.retargeting.flashdexretarget.convex import decompose_objects
from eden.extensions.retargeting.flashdexretarget.quality import flags
from eden.options.extensions.flashdexretarget import FDRCoacdConvexOptions, FDRRetargetOptions
from tests.extensions.retargeting.flashdexretarget.conftest import self_fk_clip, write_box


def test_a_robot_without_a_fingertip_site_is_refused_rather_than_scored_on_another_frame(tmp_path, monkeypatch):
    """MuJoCo gives -1 for a missing name, which would read the last site's position into the tip error."""
    write_box(tmp_path / "objects" / "box" / "visual.obj")
    decompose_objects(tmp_path, ["box"], FDRCoacdConvexOptions(), 1)
    clip, qpos = self_fk_clip(8)
    clip["qpos"] = qpos.astype("float32")
    built = S.build

    def without_the_thumb_tip(*args, **kwargs):
        spec = built(*args, **kwargs)
        spec.delete(spec.site("right_thumb_tip"))
        return spec

    monkeypatch.setattr(S, "build", without_the_thumb_tip)
    with pytest.raises(ValueError, match=r"No site named 'right_thumb_tip' in the IK model of .*right\.xml"):
        flags(clip, "xhand", tmp_path, FDRRetargetOptions(), float(clip["fps"]))
