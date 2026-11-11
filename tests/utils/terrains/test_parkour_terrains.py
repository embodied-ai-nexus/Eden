"""``eden.utils.terrains.parkour_terrains``: parkour courses, their goal walk, and the lane-fitting helpers.

The courses draw from ``np.random``, so each property is checked over a range of seeds. Goal spacing on the
shipped ``extreme_parkour`` lanes is pinned by that task's tests; the square-strip figures here are looser.
"""

from __future__ import annotations

import numpy as np
import pytest
from genesis.ext.isaacgym import terrain_utils as tu

from eden.utils.terrains import TERRAIN_GENERATOR_REGISTRY
from eden.utils.terrains.parkour_terrains import (
    MIN_GOAL_SPACING_M,
    _fit_final_goal_x,
    _goal_margin_cells,
    _set_height_block,
    _validate_goals,
)

pytestmark = pytest.mark.requires_genesis  # SubTerrain reads gs.np_float

HS, VS = 0.25, 0.005
FULL_COURSE_SIZE_M = 32.0  # holds every shipped default course untouched (the longest ends ~29 m downrange)
LANE_SIZE_M = (16.0, 4.0)  # the extreme_parkour teacher lane, at its resolution
LANE_HS = 0.08
# Cells reserved at the far end: MIN_GOAL_MARGIN_M (0.5 m) at each resolution, which exceeds the 0.1 m border pad.
MARGIN_CELLS_AT_HS = 2
MARGIN_CELLS_AT_LANE_HS = 6
GOAL_REACHED_THRESHOLD_M = 0.2  # within this, extreme_parkour's ParkourCommand counts a goal reached

# One record per course generator:
#   min_size_m      smallest square strip (at HS) that holds an ordered course with the shipped defaults, over 256
#                   seeds; below it the generator raises.
#   pull_back_m     lane length (at LANE_HS) where the walk overruns often enough to exercise the final-goal
#                   pull-back, with no seed raising or relocating off the corridor over 256 seeds (64 are used here).
#   min_goal_gap_m  closest consecutive goals the generator lays out at min_size_m, over 64 seeds.
#   lane_options    course-length knobs that fit the shipped ~29 m course into LANE_SIZE_M.
PARKOUR_CASES = {
    "parkour_terrain": {
        "min_size_m": 24.0,
        "pull_back_m": 10.0,
        "min_goal_gap_m": 1.00,
        "lane_options": {"num_stones": 6, "x_range": (-0.1, 0.2), "stone_len": (0.9, 1.0), "stone_width": 1.0},
    },
    "parkour_gap_terrain": {
        "min_size_m": 22.0,
        "pull_back_m": 13.0,
        "min_goal_gap_m": 1.25,
        "lane_options": {"num_gaps": 6, "x_range": (0.8, 1.5)},
    },
    "parkour_hurdle_terrain": {
        "min_size_m": 20.0,
        "pull_back_m": 14.8,
        "min_goal_gap_m": 1.00,
        "lane_options": {"num_stones": 6, "x_range": (1.2, 2.2)},
    },
    "parkour_step_terrain": {
        "min_size_m": 8.0,
        "pull_back_m": 11.2,
        "min_goal_gap_m": 0.50,
        "lane_options": {"num_stones": 6, "x_range": (0.3, 1.5)},
    },
    "parkour_demo_terrain": {
        "min_size_m": 12.0,
        "pull_back_m": 10.4,
        "min_goal_gap_m": 0.25,
        "lane_options": {},  # fixed demo layout, no course-length knobs
    },
}


def _gen(name, **options):
    cls = TERRAIN_GENERATOR_REGISTRY.get(name)
    return cls(options=cls.configure(**options))


def _lane_gen(name):
    return _gen(name, **PARKOUR_CASES[name]["lane_options"])


def _field(width=40, length=40, hs=HS):
    return tu.SubTerrain(width=width, length=length, vertical_scale=VS, horizontal_scale=hs)


def _last_goal_cell(subterrain, margin_cells):
    return subterrain.height_field_raw.shape[0] - margin_cells - 1


# ======================================= course generators =======================================


@pytest.mark.parametrize(
    ("name", "size_m"),
    [
        (n, s)
        for n, case in PARKOUR_CASES.items()
        for s in (6.0, 8.0, 12.0, 16.0, 20.0, 24.0, 32.0)
        if s >= case["min_size_m"]
    ],
)
def test_course_field_is_finite_and_shape_preserving_at_or_above_its_floor(name, size_m, subterrain):
    gen = _gen(name)
    for seed in range(8):
        np.random.seed(seed)
        st = subterrain(size_m)
        shape = st.height_field_raw.shape
        out = gen.compute(st)
        assert out.height_field_raw.shape == shape
        assert np.isfinite(out.height_field_raw).all()


@pytest.mark.parametrize("name", PARKOUR_CASES)
def test_course_carves_relief(name, subterrain):
    """Regression: a zero-cell ``pad_width`` made ``[:, -0:]`` flood the field; sub-cell hurdles drew nothing."""
    np.random.seed(0)
    hf = _gen(name).compute(subterrain(FULL_COURSE_SIZE_M)).height_field_raw
    assert hf.max() > hf.min()


@pytest.mark.parametrize("at", ["full_course", "floor"])
@pytest.mark.parametrize("name", PARKOUR_CASES)
def test_goals_are_ordered_and_on_the_walkable_strip(name, at, subterrain):
    """The invariant a goal-tracking command needs: goals advance strictly and stay short of the far-end clearance.

    On the full-course strip it holds untouched; at the generator's floor the fit check is most stressed.
    """
    size_m = FULL_COURSE_SIZE_M if at == "full_course" else PARKOUR_CASES[name]["min_size_m"]
    gen = _gen(name)
    for seed in range(64):
        np.random.seed(seed)
        st = subterrain(size_m)
        limit_m = _last_goal_cell(st, MARGIN_CELLS_AT_HS) * HS
        goals_x = gen.compute(st).goals[:, 0]
        assert ((0.0 <= goals_x) & (goals_x <= limit_m)).all(), f"seed {seed}: {goals_x.tolist()} off [0, {limit_m}]"
        assert (np.diff(goals_x) > 0).all(), f"seed {seed}: goals not ordered: {goals_x.tolist()}"


@pytest.mark.parametrize("name", PARKOUR_CASES)
def test_goal_spacing_at_the_floor_stays_resolvable(name, subterrain):
    """Interior spacing is the course's own layout, not the fit check's.

    Below the goal-reached threshold a command would satisfy two waypoints at once.
    """
    case = PARKOUR_CASES[name]
    gen = _gen(name)
    worst_gap = float("inf")
    for seed in range(64):
        np.random.seed(seed)
        worst_gap = min(worst_gap, float(np.diff(gen.compute(subterrain(case["min_size_m"])).goals[:, 0]).min()))
    assert worst_gap == pytest.approx(case["min_goal_gap_m"], abs=1e-6), "course retuned? update PARKOUR_CASES"
    assert worst_gap > GOAL_REACHED_THRESHOLD_M


@pytest.mark.parametrize("name", PARKOUR_CASES)
def test_a_strip_too_short_for_the_course_raises(name, subterrain):
    """Only the last goal can be pulled back; once the one before it is off the strip too, nothing is left to save.

    Regression: ``ParkourTerrain`` lacked the check (its last goal landed up to ~20 m past an 8 m strip) and
    ``ParkourDemoTerrain`` had no guard at all.
    """
    gen = _gen(name)
    for seed in range(16):
        np.random.seed(seed)
        with pytest.raises(ValueError, match="does not fit"):
            gen.compute(subterrain(6.0))


@pytest.mark.parametrize("name", PARKOUR_CASES)
def test_a_lane_with_its_axes_swapped_raises(name, subterrain):
    """The course runs along dim0; a generator measuring it against dim1 would lay 16 m of obstacles into 4 m."""
    gen = _lane_gen(name)
    for seed in range(4):
        np.random.seed(seed)
        with pytest.raises(ValueError, match="does not fit"):
            gen.compute(subterrain(LANE_SIZE_M[1], LANE_SIZE_M[0], hs=LANE_HS))


@pytest.mark.parametrize("size_xy", [LANE_SIZE_M, (32.0, 8.0)])
@pytest.mark.parametrize("name", PARKOUR_CASES)
def test_course_stays_on_a_non_square_lane(name, size_xy, subterrain):
    """Goals stay within the lane laterally: swapped axes would spread them over the course extent instead."""
    gen = _lane_gen(name)
    for seed in range(4):
        np.random.seed(seed)
        st = subterrain(*size_xy, hs=LANE_HS)
        shape = st.height_field_raw.shape
        out = gen.compute(st)
        assert out.height_field_raw.shape == shape
        assert np.isfinite(out.height_field_raw).all() and np.isfinite(out.goals).all()
        assert (out.goals[:, 1] >= 0.0).all() and (out.goals[:, 1] <= size_xy[1]).all()


@pytest.mark.parametrize(
    ("name", "options", "size_xy"),
    [("parkour_demo_terrain", {}, (12.0, 0.8)), ("parkour_terrain", {"stone_width": 3.0}, (32.0, 2.0))],
    ids=["demo_slopes_on_a_0.8m_lane", "stones_3x_the_lane"],
)
def test_a_feature_wider_than_its_lane_is_clipped(name, options, size_xy, subterrain):
    """Clipped, not a broadcast error or a wrapped write; the course still fits along x, so goals stay valid."""
    gen = _gen(name, **options)
    for seed in range(8):
        np.random.seed(seed)
        out = gen.compute(subterrain(*size_xy))
        assert np.isfinite(out.height_field_raw).all()
        assert out.height_field_raw.max() > out.height_field_raw.min(), "the clipped feature vanished"
        assert (np.diff(out.goals[:, 0]) > 0).all()


@pytest.mark.parametrize("name", PARKOUR_CASES)
def test_pull_back_fires_on_an_overrun_lane_and_keeps_the_course_valid(name, subterrain):
    """The ``extreme_parkour`` regime: a walk ending just past the far edge gets its last goal pulled back.

    Raising there would fail a 200-cell teacher grid ~99% of the time. The test asserts the pull-back fired, or it
    would silently duplicate the ordered-goals test.
    """
    gen = _lane_gen(name)
    pulled_back = 0
    for seed in range(64):
        np.random.seed(seed)
        st = subterrain(PARKOUR_CASES[name]["pull_back_m"], LANE_SIZE_M[1], hs=LANE_HS)
        limit_m = _last_goal_cell(st, MARGIN_CELLS_AT_LANE_HS) * LANE_HS
        goals_x = gen.compute(st).goals[:, 0]
        assert (goals_x < limit_m + 1e-9).all(), f"seed {seed}: goal past the limit: {goals_x.tolist()}"
        assert (np.diff(goals_x) > 0).all(), f"seed {seed}: goals not ordered: {goals_x.tolist()}"
        pulled_back += abs(goals_x[-1] - limit_m) < 1e-9
    assert pulled_back > 0, "no seed overran the lane; retune pull_back_m"


@pytest.mark.parametrize("name", PARKOUR_CASES)
def test_a_relocated_goal_lands_on_the_courses_corridor(name, subterrain):
    """Not inside a gap or on a hurdle. Heights are read before roughness, which is added after the goal is placed."""
    gen = _lane_gen(name)
    for seed in range(64):
        np.random.seed(seed)
        st = subterrain(PARKOUR_CASES[name]["pull_back_m"], LANE_SIZE_M[1], hs=LANE_HS)
        out = gen.compute(st)
        goal_x, goal_y = (round(v / LANE_HS) for v in out.goals[-1])
        assert out.height_field_raw[goal_x, goal_y] == gen.corridor_height_raw(st), f"seed {seed}"


@pytest.mark.parametrize(
    ("name", "options", "expected_raw"),
    [
        ("parkour_gap_terrain", {}, 0),
        ("parkour_hurdle_terrain", {}, 0),
        ("parkour_step_terrain", {}, 0),
        ("parkour_demo_terrain", {}, 0),
        ("parkour_terrain", {}, 0),  # platform_height=0
        ("parkour_terrain", {"platform_height": 0.4}, round(0.4 / VS)),
    ],
)
def test_corridor_height_is_declared_per_generator(name, options, expected_raw):
    """Declared, not read off the field: no single reading of it is right for every course.

    Corridor courses carve into a zero field, so zero is ground; ``ParkourTerrain`` floods the lane with its pit and
    paints platforms on top, so its ground is ``platform_height``.
    """
    assert _gen(name, **options).corridor_height_raw(_field()) == expected_raw


@pytest.mark.parametrize(
    ("options", "hs", "seed"),
    [
        # goals came out at [2.0, -0.25, 0.0, 0.0] m: behind the strip origin, and the last two on one cell.
        ({"num_stones": 2, "x_range": (-1.0, 3.0)}, HS, 110),
        # extreme_parkour's own stepping stones at difficulty 1.0, at a resolution where the advance rounds to 1 cell.
        ({"num_stones": 6, "x_range": (-0.1, 0.4), "stone_len": (0.6, 0.8), "stone_width": 1.0}, 0.5, 2),
    ],
    ids=["negative_x_range", "coarse_horizontal_scale"],
)
def test_a_walk_that_steps_backwards_is_rejected(options, hs, seed, subterrain):
    """``_fit_final_goal_x`` returns early when the final goal fits, so the whole sequence is checked as well."""
    np.random.seed(seed)
    with pytest.raises(ValueError, match="not a usable goal sequence"):
        _gen("parkour_terrain", **options).compute(subterrain(FULL_COURSE_SIZE_M, hs=hs))


def test_roughness_perturbs_the_field_and_scales_with_difficulty(subterrain):
    def build(**options):
        np.random.seed(3)
        gen = _gen(
            "parkour_hurdle_terrain",
            flat=True,
            pad_height=0.0,
            **PARKOUR_CASES["parkour_hurdle_terrain"]["lane_options"],
            **options,
        )
        return gen.compute(subterrain(*LANE_SIZE_M, hs=LANE_HS)).height_field_raw

    smooth = build(apply_roughness=False)
    gentle = build(apply_roughness=True, difficulty=0.0) - smooth
    rough = build(apply_roughness=True, difficulty=1.0) - smooth
    assert rough.any()
    # At difficulty 0 the sampled range collapses to [-noise_range[0], 0]; spline upsampling may overshoot slightly.
    assert gentle.std() < rough.std()
    assert np.abs(gentle).max() <= 3 * int(0.02 / VS)


# ====================================== other parkour-module generators ======================================


def test_gap_parkour_feature_is_centred_on_a_non_square_strip(subterrain):
    """Regression: dim0 was indexed with the dim1 extent, pushing the feature off-centre on strips."""
    np.random.seed(0)
    hf = _gen("gap_parkour_terrain", platform_size=2.0).compute(subterrain(16.0, 4.0, hs=LANE_HS)).height_field_raw
    center = np.argwhere(hf != 0).mean(axis=0)
    assert abs(center[0] - hf.shape[0] // 2) < 4 and abs(center[1] - hf.shape[1] // 2) < 4


def test_half_sloped_slope_starts_past_the_x_middle_on_a_non_square_strip(subterrain):
    """Regression: the start was placed from the dim1 extent."""
    hf = _gen("half_sloped_terrain", start2center=0.7).compute(subterrain(16.0, 4.0, hs=LANE_HS)).height_field_raw
    first_x = np.argwhere((hf != 0).any(axis=1)).min()
    assert abs(first_x - (hf.shape[0] // 2 + round(0.7 / LANE_HS))) <= 1


# ========================================= lane-fitting helpers =========================================


@pytest.mark.parametrize(
    ("rows", "cols", "surviving"),
    [
        ((6, 10), (2, 6), np.s_[0:2, :]),  # off the far x edge: the block's first two rows land
        ((-2, 2), (2, 6), np.s_[2:4, :]),  # off the near x edge: its last two rows
        ((2, 6), (6, 10), np.s_[:, 0:2]),  # off the far y edge
        ((2, 6), (-2, 2), np.s_[:, 2:4]),  # off the near y edge
    ],
    ids=["far_x", "near_x", "far_y", "near_y"],
)
def test_set_height_block_writes_the_in_bounds_window_in_the_blocks_own_frame(rows, cols, surviving):
    """``values`` is indexed in the unclipped block's frame; a wrong window is not a crash but a shifted feature."""
    values = np.arange(16, dtype=np.int16).reshape(4, 4)
    st = _field(8, 8)
    _set_height_block(st, *rows, *cols, values)
    window = st.height_field_raw[max(rows[0], 0) : min(rows[1], 8), max(cols[0], 0) : min(cols[1], 8)]
    np.testing.assert_array_equal(window, values[surviving])


def test_set_height_block_entirely_outside_is_a_no_op():
    """Not a wrapped negative-index write."""
    st = _field(8, 8)
    values = np.ones((4, 4), dtype=np.int16)
    _set_height_block(st, 20, 24, 2, 6, values)
    _set_height_block(st, -10, -6, 2, 6, values)
    assert (st.height_field_raw == 0).all()


@pytest.mark.parametrize(
    ("hs", "pad_width", "expected_cells"),
    [
        (HS, 0.1, MARGIN_CELLS_AT_HS),
        # The shipped 0.1 m pad is one cell at 0.08 m: the 0.5 m floor must win, or the goal sits 8 cm from a wall.
        (LANE_HS, 0.1, MARGIN_CELLS_AT_LANE_HS),
        (LANE_HS, 1.0, 12),  # a pad wider than the floor wins instead: the wall is what must be cleared
        (LANE_HS, None, MARGIN_CELLS_AT_LANE_HS),  # no pad still keeps the clearance
        (1.0, None, 1),  # round(0.5 / hs) floors to zero at hs >= 1 m; one cell is kept
        (2.0, 0.1, 1),
    ],
)
def test_goal_margin_is_the_larger_of_the_pad_and_a_clearance_floor(hs, pad_width, expected_cells):
    assert _goal_margin_cells(_field(64, 64, hs=hs), pad_width) == expected_cells


@pytest.mark.parametrize("name", ["parkour_terrain", "parkour_demo_terrain"], ids=["with_pad", "without_pad"])
@pytest.mark.parametrize("overshoot", [37, 38, 39, 40, 45])
def test_fit_final_goal_x_keeps_the_far_end_clear(name, overshoot):
    """A course exactly filling the strip would end on the border (a wall where one is raised, the edge otherwise).

    The last admissible cell of a 40-cell strip is 37; anything past it is pulled back there.
    """
    gen = _gen(name)
    assert _fit_final_goal_x(_field(), overshoot, prev_goal_x=30, goal_y=20, generator=gen) == 37


@pytest.mark.parametrize("prev_goal_x", [35, 36, 37, 50])
def test_fit_final_goal_x_needs_room_ahead_of_the_previous_goal(prev_goal_x):
    """The pulled-back goal must land ``MIN_GOAL_SPACING_M`` (2 cells here) ahead of its predecessor.

    Otherwise a command reaches both at once. From 35 there is exactly that room before the limit at 37.
    """
    gen = _gen("parkour_terrain")
    assert round(MIN_GOAL_SPACING_M / HS) == 2
    if prev_goal_x == 35:
        assert _fit_final_goal_x(_field(), 60, prev_goal_x=prev_goal_x, goal_y=20, generator=gen) == 37
    else:
        with pytest.raises(ValueError, match="does not fit"):
            _fit_final_goal_x(_field(), 60, prev_goal_x=prev_goal_x, goal_y=20, generator=gen)


@pytest.mark.parametrize(
    ("obstacle_raw", "band", "prev_goal_x", "expected"),
    [
        (-160, (34, 38), 20, 33),  # a pit covering the bare limit: step back onto the corridor
        (60, (34, 38), 20, 33),  # a hurdle covering it
        (None, None, 20, 37),  # a clean lane is placed exactly at the limit
        (-160, (20, 40), 25, 37),  # no corridor cell in reach: degrade to the bare limit rather than raise
    ],
    ids=["pit", "hurdle", "clean_lane", "whole_tail_is_obstacle"],
)
def test_pull_back_steps_off_an_obstacle_covering_the_limit(obstacle_raw, band, prev_goal_x, expected):
    """A relocated goal lands on the run-out, not in a gap (0.99 m deep on ``ParkourGapTerrain``) or on a hurdle."""
    st = _field()
    if obstacle_raw is not None:
        st.height_field_raw[band[0] : band[1], :] = obstacle_raw
    assert _fit_final_goal_x(st, 60, prev_goal_x=prev_goal_x, goal_y=20, generator=_gen("parkour_terrain")) == expected


@pytest.mark.parametrize(
    "goals",
    [
        [[5.0, 20.0], [-1.0, 20.0], [10.0, 20.0]],  # behind the strip origin
        [[5.0, 20.0], [10.0, 20.0], [10.0, 20.0]],  # last two on one cell
        [[5.0, 20.0], [20.0, 20.0], [10.0, 20.0]],  # the walk steps backwards
        [[5.0, 20.0], [10.0, 20.0], [38.0, 20.0]],  # final goal inside the far-end clearance
    ],
    ids=["behind_origin", "tied", "backwards", "in_clearance"],
)
def test_validate_goals_checks_the_whole_sequence(goals):
    gen = _gen("parkour_terrain")
    _validate_goals(_field(), np.array([[5.0, 20.0], [10.0, 20.0], [37.0, 20.0]]), gen)  # ordered and on-strip
    with pytest.raises(ValueError, match="not a usable goal sequence"):
        _validate_goals(_field(), np.array(goals), gen)
