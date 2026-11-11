"""Tests for ``eden.extensions.retargeting.flashdexretarget.convex``: CoACD parts and CoACD's merge of GPU parts."""

from __future__ import annotations

import numpy as np
import trimesh

import pytest

from eden.extensions.retargeting.flashdexretarget.convex import (
    _write,
    convex_parts,
    decompose_objects,
    has_parts,
    merge_parts,
)
from eden.options.extensions.flashdexretarget import FDRCoacdConvexOptions
from tests.extensions.retargeting.flashdexretarget.conftest import write_box


def _cube(lo, hi) -> np.ndarray:
    return trimesh.creation.box(bounds=(lo, hi)).vertices


def test_merge_joins_two_halves_of_one_box_and_keeps_separate_boxes_apart():
    halves = [_cube((0, 0, 0), (0.5, 1, 1)), _cube((0.5, 0, 0), (1, 1, 1))]
    merged = merge_parts(halves, threshold=0.05, max_parts=32)
    assert len(merged) == 1
    np.testing.assert_allclose(merged[0].min(0), (0, 0, 0))
    np.testing.assert_allclose(merged[0].max(0), (1, 1, 1))
    apart = [_cube((0, 0, 0), (1, 1, 1)), _cube((3, 0, 0), (4, 1, 1))]
    assert len(merge_parts(apart, threshold=0.05, max_parts=32)) == 2


def test_merge_goes_down_to_max_parts():
    row = [_cube((2 * i, 0, 0), (2 * i + 1, 1, 1)) for i in range(4)]
    assert len(merge_parts(row, threshold=0.05, max_parts=2)) == 2


def test_coacd_writes_the_parts_of_a_box_once(tmp_path):
    write_box(tmp_path / "objects" / "box" / "visual.obj")
    cfg = FDRCoacdConvexOptions()
    decompose_objects(tmp_path, ["box"], cfg, workers=1)
    parts = convex_parts(tmp_path / "objects" / "box")
    assert len(parts) == 1
    np.testing.assert_allclose(trimesh.load(parts[0]).extents, (0.1, 0.1, 0.1), atol=2e-3)
    stamp = parts[0].stat().st_mtime_ns
    decompose_objects(tmp_path, ["box"], cfg, workers=1)  # already decomposed: left alone
    assert parts[0].stat().st_mtime_ns == stamp


def test_an_empty_decomposition_is_refused_and_the_old_parts_kept(tmp_path):
    obj = tmp_path / "objects" / "box"
    write_box(obj / "visual.obj")
    decompose_objects(tmp_path, ["box"], FDRCoacdConvexOptions(), workers=1)
    with pytest.raises(ValueError, match=r"has no part .*collide with nothing"):
        _write(obj, [], {})
    assert len(convex_parts(obj)) == 1


def test_parts_written_halfway_are_not_taken_for_a_decomposition(tmp_path):
    """A run killed while writing leaves convex.<pid>.tmp/, which is not the convex/ that marks an object done."""
    obj = tmp_path / "objects" / "box"
    (obj / "convex.123.tmp").mkdir(parents=True)
    trimesh.creation.box().export(obj / "convex.123.tmp" / "0.obj")
    assert not has_parts(obj / "convex")


def test_parts_are_redone_when_the_mesh_or_a_setting_changes_and_kept_otherwise(tmp_path):
    obj = tmp_path / "objects" / "box"
    write_box(obj / "visual.obj")
    decompose_objects(tmp_path, ["box"], FDRCoacdConvexOptions(), workers=1)
    stamp = (obj / "convex" / "0.obj").stat().st_mtime_ns

    def redone(cfg) -> bool:
        nonlocal stamp
        decompose_objects(tmp_path, ["box"], cfg, workers=1)
        new = (obj / "convex" / "0.obj").stat().st_mtime_ns
        changed, stamp = new != stamp, new
        return changed

    assert not redone(FDRCoacdConvexOptions())
    assert redone(FDRCoacdConvexOptions(seed=2))  # another setting
    write_box(obj / "visual.obj", extents=(0.2, 0.2, 0.2))  # another mesh under the same id
    assert redone(FDRCoacdConvexOptions(seed=2))
    np.testing.assert_allclose(trimesh.load(convex_parts(obj)[0]).extents, (0.2, 0.2, 0.2), atol=4e-3)
    assert not redone(FDRCoacdConvexOptions(seed=2))
    (obj / "convex" / "source.json").unlink()  # parts made before their source was recorded
    assert redone(FDRCoacdConvexOptions(seed=2))
    (obj / "convex" / "source.json").write_text("{")  # a damaged record
    assert redone(FDRCoacdConvexOptions(seed=2))
    assert not redone(FDRCoacdConvexOptions(seed=2, force=False))  # force and gpus are not part of the source
