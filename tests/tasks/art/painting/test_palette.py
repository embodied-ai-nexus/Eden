"""Tests for the palette: where its wells are, and the tray drawn with a well of each paint."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from eden.tasks.art.painting.palette import WELL_PITCH, WELL_RADIUS, PaletteOptions, palette_file, well_centres
from eden.tasks.art import workcell


def test_the_wells_stand_in_columns_of_five_left_of_the_sheet():
    centres = well_centres(7, (0.18, 0.18))
    assert (centres[:, 0] < -0.09 - WELL_RADIUS).all()  # clear of the sheet's left edge
    np.testing.assert_allclose(centres[:5, 1], [0.06, 0.03, 0.0, -0.03, -0.06])
    np.testing.assert_allclose(centres[5:, 0], centres[0, 0] - WELL_PITCH)
    distance = np.linalg.norm(centres[:, None] - centres[None], axis=-1) + np.eye(7)
    assert distance.min() > 2 * WELL_RADIUS


def test_the_tray_s_face_is_the_paper_s_and_each_well_has_its_paint():
    paints = [(0.9, 0.1, 0.1), (0.1, 0.9, 0.1)]
    body = ET.parse(palette_file(paints, (0.18, 0.18))).getroot().find("worldbody/body")
    tray = body.find("geom[@name='tray']")
    z, half = float(tray.get("pos").split()[2]), float(tray.get("size").split()[2])
    assert z + half == pytest.approx(workcell.SHEET_CENTER[2])
    wells = body.findall("geom[@contype='0']")
    assert [tuple(float(c) for c in w.get("rgba").split()[:3]) for w in wells] == paints


def test_a_palette_is_a_work_s_and_an_unknown_work_is_refused_naming_the_paintings():
    assert PaletteOptions(work="sunset").work == "sunset"
    with pytest.raises(
        ValueError, match=r"work='sunrise' is not a painting; the paintings are \['sunset', 'fuji_study'"
    ):
        PaletteOptions(work="sunrise")


def test_a_palette_with_a_pose_of_its_own_is_refused():
    with pytest.raises(ValueError, match=r"default_root_pos=\(0\.1, 0\.0, 0\.0\).*takes no pose of its own"):
        PaletteOptions(work="sunset", default_root_pos=(0.1, 0.0, 0.0))
