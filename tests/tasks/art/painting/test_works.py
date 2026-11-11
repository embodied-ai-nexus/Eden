"""Tests for the paintings the tasks copy: their plans stay on the sheet, paint with the brush's sizes, and dip."""

import hashlib
import os
import typing
from pathlib import Path

import numpy as np
import pytest
import torch

from eden.tasks.art.painting.paint import RoundBrush
from eden.tasks.art.painting.planner import Layer
from eden.tasks.art.painting.references import render_painting
from eden.tasks.art.painting.works import FUJI, FUJI_STUDY, WORKS, PlannedPainting, WorkName, fetch_picture, painting
from eden.tasks.art.plans import STOP
from eden.utils.warning import warn_once
from tests.tasks.art.painting.conftest import picture_file


@pytest.mark.parametrize(("name", "strokes", "paints"), [("sunset", 7, 3), ("fuji_study", 116, 5), ("fuji", 333, 10)])
def test_each_work_has_its_strokes_and_paints(name, strokes, paints):
    work = painting(name)
    assert (len(work.strokes), len(work.palette)) == (strokes, paints)
    assert {s.paint for s in work.strokes} == set(range(paints)), "every paint is used"


@pytest.mark.parametrize("name", sorted(WORKS))
def test_a_plan_touches_the_paper_only_on_the_sheet_and_dips_only_in_wells(name):
    work = painting(name)
    path, owner = work.path()
    on_paper = (path[:, 2] > 0.0) & (owner >= 0)
    assert (np.abs(path[on_paper, :2]) < np.asarray(work.sheet) / 2).all()
    dips = path[(owner == STOP) & (path[:, 2] > 0.0), :2]
    assert np.linalg.norm(dips[:, None] - work.wells[None], axis=-1).min(axis=1).max() < 1e-9
    depth = path[on_paper, 2].max()
    assert depth <= RoundBrush().full_press, "no stroke presses the ferrule into the paper"


def test_the_fuji_is_painted_with_robopaints_four_acrylic_brush_sizes():
    brush = RoundBrush()
    depths = {round(max(p[2] for p in s.stroke.points), 5) for s in painting("fuji").strokes}
    sizes = sorted(round(float(brush.diameter_at(torch.tensor(d))) * 1000, 1) for d in depths)
    assert sizes == [3.0, 5.2, 9.4, 16.0]


@pytest.mark.parametrize("name", sorted(WORKS))
def test_a_target_has_paint_and_its_plan_dips_at_least_once_per_paint(name):
    work = painting(name)
    rendering = render_painting(work)
    assert bool(rendering.painted.any()) and rendering.dips >= len(work.palette)


def test_the_work_names_a_config_takes_are_the_works():
    assert list(typing.get_args(WorkName)) == list(WORKS)


SHEET, CELL = (0.06, 0.06), 0.0005


def _square(picture: str, **recipe) -> PlannedPainting:
    """A red square on blue, planned in two paints and one layer on a small sheet."""
    recipe = {"paints": 2, "layers": (Layer(0.016, 6.0),), "sheet": SHEET, "cell": CELL} | recipe
    return PlannedPainting("square", "a red square", picture=picture, crop=(60, 20, 180, 140), **recipe)


def test_a_planned_work_is_planned_once_then_read_from_where_it_is_kept(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    work = _square(picture_file(tmp_path / "p.png"), rule={"de_mean": 1.0})
    first = work.painting()
    assert len(first.strokes) > 0 and first.rule["de_mean"] == 1.0
    kept = os.stat(work.cache_path()).st_mtime_ns
    assert work.painting() == first and os.stat(work.cache_path()).st_mtime_ns == kept, "read, not planned again"
    assert _square(work.picture, paints=3).cache_path() != work.cache_path(), "another recipe is another plan"


def test_a_plan_unlike_the_published_one_is_used_with_a_warning(tmp_path, monkeypatch, eden_caplog):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    warn_once.cache_clear()
    work = _square(picture_file(tmp_path / "p.png"), plan_sha256="0" * 64)
    assert len(work.painting().strokes) > 0
    assert any("is not the published one (plan_sha256 0000000000000000...)" in r.message for r in eden_caplog.records)


def test_a_re_pinned_work_is_planned_again(tmp_path, monkeypatch):
    """A deliberate re-plan is published by a new plan_sha256: a machine that kept the old plan plans afresh."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    picture = picture_file(tmp_path / "p.png")
    old, new = _square(picture, plan_sha256="a" * 64), _square(picture, plan_sha256="b" * 64)
    old.painting()
    assert new.cache_path() != old.cache_path() and not os.path.exists(new.cache_path())
    new.painting()
    assert os.path.exists(new.cache_path())


def test_a_picture_that_cannot_be_fetched_is_named_with_the_dataset_and_the_way_out(monkeypatch):
    import huggingface_hub

    def offline(*args, **kwargs):  # the network boundary
        raise huggingface_hub.errors.LocalEntryNotFoundError("no connection and not in the local cache")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    with pytest.raises(
        RuntimeError,
        match=r"Could not fetch the picture 'painting/x\.jpg' of the Hugging Face dataset "
        r"Kashu7100/eden_art at revision c516699e62a7 \(LocalEntryNotFoundError: no connection.*"
        r"give the recipe an absolute path to a local copy",
    ):
        fetch_picture("painting/x.jpg")


def test_a_kept_plan_of_a_dataset_picture_is_read_offline(tmp_path, monkeypatch):
    """A kept plan is found by the dataset's pinned revision: once planned, a run offline never fetches the picture."""
    import huggingface_hub

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    local = picture_file(tmp_path / "p.png")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *args, **kwargs: local)  # the network boundary
    planned = _square("painting/square.png").painting()

    def offline(*args, **kwargs):
        raise huggingface_hub.errors.LocalEntryNotFoundError("no connection and not in the local cache")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    assert _square("painting/square.png").painting() == planned


@pytest.mark.slow  # fetches the print once, and plans both works (about 30 s)
@pytest.mark.parametrize("work", [FUJI_STUDY, FUJI], ids=lambda work: work.name)
def test_the_code_still_plans_each_hokusai_work_as_published(work, tmp_path, monkeypatch):
    """Planned afresh (not read from the cache): a change to the planner or the paint model that changes a work fails
    here, naming the new digest to publish with the documented counts.
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    work.painting()
    planned = hashlib.sha256(Path(work.cache_path()).read_bytes()).hexdigest()
    assert planned == work.plan_sha256, f"{work.name} now plans as {planned}"
