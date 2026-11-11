"""Tests for the whiteboard references: the original's text layout, its table, and the reference raster."""

import numpy as np
import pytest

from eden.tasks.whiteboard.strokes import DRAWINGS, raster, sample_strokes, strokes_bounds, table_strokes, text_strokes


def test_text_spans_the_given_width_centred_on_the_drawing_centre():
    # The original's layout: 1.1 cells a glyph, so HELLO spans 5.2 cells; the 0.8-wide glyph cell sets both ends.
    u_min, v_min, u_max, v_max = strokes_bounds(text_strokes("HELLO", 0.60))
    assert (u_min, u_max) == pytest.approx((-0.30, 0.30))
    assert (v_min, v_max) == pytest.approx((-0.60 / 5.2 / 2, 0.60 / 5.2 / 2))


def test_text_reads_left_to_right_as_the_robot_faces_the_board():
    # u runs to the robot's right, so the first letter is the leftmost: no mirroring.
    strokes = text_strokes("LO", 0.2)
    first = np.asarray([p for name, _, path in strokes if name == "letter 0" for p in path])
    second = np.asarray([p for name, _, path in strokes if name == "letter 1" for p in path])
    assert first[:, 0].max() < second[:, 0].min()


def test_a_character_outside_the_alphabet_is_refused_by_name():
    with pytest.raises(ValueError, match=r"no glyph for \['a', 'l', 'o'\] in 'Hallo'"):
        text_strokes("Hallo", 0.6)


def test_the_table_is_the_originals_seven_lines():
    lines = {name: (weight, path) for name, weight, path in table_strokes(0.60)}
    assert set(lines) == {"top", "bottom", "left", "right", "header", "column one", "column two"}
    assert lines["top"] == ("thick", ((-0.30, 0.18), (0.30, 0.18)))
    assert lines["column one"][0] == "normal"
    assert np.asarray(lines["header"][1]) == pytest.approx(np.array([[-0.30, 0.12], [0.30, 0.12]]))


def test_raster_draws_the_lines_their_width_wide():
    # A 3 mm "normal" line on 1 mm cells is three cells across; a 6 mm "thick" line, six.
    line = (("l", "normal", ((0.0, -0.02), (0.0, 0.02))),)
    image = raster(line, origin=(-0.01, 0.03), cell=0.001, shape=(60, 20))
    assert image[30].sum() == 3
    thick = (("l", "thick", ((0.0, -0.02), (0.0, 0.02))),)
    assert raster(thick, origin=(-0.01, 0.03), cell=0.001, shape=(60, 20))[30].sum() == 6


def test_samples_follow_each_stroke_in_drawing_order():
    points, stroke = sample_strokes(DRAWINGS["table"], 0.002)
    assert np.all(np.diff(stroke) >= 0)  # in drawing order, stroke by stroke
    top = points[stroke == 0]
    assert top[0] == pytest.approx((-0.30, 0.18)) and top[-1] == pytest.approx((0.30, 0.18))
    assert np.diff(top[:, 0]).max() <= 0.002 + 1e-12
