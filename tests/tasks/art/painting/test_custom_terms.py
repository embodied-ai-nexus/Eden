"""Tests for the painting terms on a built ``art_painting_sunset``: the dip, the paint laid, the hand's count and the
grade.
"""

import pytest
import torch

from eden.tasks.art.painting.custom_terms import PaintingCopy
from eden.tasks.art.painting.scripted_demos import ScriptedPainter
from eden.tasks.art import workcell
from eden.tasks.art.terms import TipPose
from tests.tasks.art.painting.conftest import build_task

pytestmark = pytest.mark.requires_genesis


@pytest.fixture(scope="module")
def env():
    return build_task("art_painting_sunset")


def _hold(env, steps):
    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
    for _ in range(steps):
        env.step(zero)


def _paint_until(env, painter, condition, steps=900):
    for _ in range(steps):
        env.step(painter.act())
        if bool(condition()):
            return True
    return False


def test_the_brush_starts_upright_above_the_sheets_centre_with_no_paint(env):
    env.reset()
    copy = env.metric_manager.get_term("painting")
    sheet = copy.tip_on_sheet()
    # HOME puts the tip 30 mm above the centre, give or take the arm's reset noise.
    torch.testing.assert_close(sheet, torch.tensor([[0.0, 0.0, -workcell.HOME_STANDOFF]] * 2), atol=0.025, rtol=0)
    assert copy.paint.tolist() == [-1, -1] and copy.canvas.load.tolist() == [0.0, 0.0]


def test_the_observations_right_after_a_reset_are_finite_and_have_their_widths(env):
    obs, _ = env.reset()
    assert bool(torch.isfinite(obs["policy"]).all()) and bool(torch.isfinite(obs["critic"]).all())
    manager = env.observation_manager
    widths = {name: manager.get_term(name).compute().shape for name in ("brush_paint", "next_paint", "next_points")}
    assert widths == {"brush_paint": (2, 4), "next_paint": (2, 6), "next_points": (2, 16)}
    assert manager.get_term("strokes_left").compute().tolist() == [[10.0], [10.0]]  # 1.3 times 7, up


def test_a_dip_fills_the_brush_with_its_well_s_paint_and_is_no_stroke(env):
    env.reset()
    copy = env.metric_manager.get_term("painting")
    painter = ScriptedPainter(env)
    assert _paint_until(env, painter, lambda: (copy.dips > 0).all())
    first = copy.work_spec.strokes[0].paint
    assert copy.paint.tolist() == [first, first] and copy.canvas.load.tolist() == [1.0, 1.0]
    torch.testing.assert_close(copy.canvas.brush_rgb(), copy.palette[[first, first]], atol=1e-4, rtol=0.0)
    assert copy.strokes.tolist() == [0, 0] and not bool(copy.canvas.painted_cells().any())


def test_painting_lays_the_brush_s_paint_near_the_tip_uses_paint_and_counts_a_stroke(env):
    env.reset()
    copy = env.metric_manager.get_term("painting")
    painter = ScriptedPainter(env)
    assert _paint_until(env, painter, lambda: (copy.canvas.painted_cells().flatten(1).sum(1) > 500).all())
    assert copy.strokes.tolist() == [1, 1] and bool((copy.canvas.load < 1.0).all())
    cells = copy.canvas.painted_cells()[0]
    rows, cols = torch.nonzero(cells, as_tuple=True)
    u = copy.grid.u0 + (cols.float() + 0.5) * copy.grid.cell
    v = copy.grid.v0 - (rows.float() + 0.5) * copy.grid.cell
    tip = copy.tip_on_sheet()[0]
    assert float(torch.hypot(u - tip[0], v - tip[1]).min()) < 0.01
    # Where the brush covered a cell fully it has the paint's own colour; round the mark's soft edge, a blend.
    full = copy.canvas.cover[0, : copy.canvas.pad].view(copy.grid.shape) > 0.999
    assert int(full.sum()) > 100
    colours = copy.canvas.rgb()[0][full]
    torch.testing.assert_close(colours, copy.palette[copy.paint[0]].expand_as(colours), atol=0.01, rtol=0.0)
    assert bool((copy.de_drop >= 0).all()) and bool((copy.de < copy.bare_de_mean).all())


def test_paint_without_a_dip_lays_nothing(env):
    """A brush that was never dipped is dry: pressed on the sheet it makes a stroke, but no paint."""
    env.reset()
    copy = env.metric_manager.get_term("painting")
    uv = torch.zeros(2, 3, 2)
    uv[:, :, 0] = torch.tensor([0.0, 0.001, 0.002])
    copy.advance(uv, torch.full((2, 3), 0.005), torch.zeros(2, 3, 2))
    assert copy.strokes.tolist() == [1, 1] and not bool(copy.canvas.painted_cells().any())
    env.reset()


def test_a_there_and_back_stroke_is_a_scribble_and_a_sweep_is_not(env):
    env.reset()
    copy = env.metric_manager.get_term("painting")
    u = torch.cat([torch.linspace(0.0, 0.02, 21), torch.linspace(0.02, 0.0, 21)])  # 4 cm of path, back where it began
    uv = torch.stack([u, torch.zeros_like(u)], dim=-1).expand(2, -1, -1).clone()
    uv[1, :, 0] = torch.linspace(0.0, 0.04, 42)  # env 1 sweeps the same length one way
    copy.advance(uv, torch.full((2, 42), 0.005), torch.zeros(2, 42, 2))
    share = copy.scribble_share()
    assert float(share[0]) == pytest.approx(1.0) and float(share[1]) == 0.0
    env.reset()


def test_a_copy_that_breaks_a_hand_rule_is_graded_just_under_success(env):
    # The target painted on with no brush: the picture is perfect, but over the stroke budget (counted as if the hand
    # had made 11 strokes), so the grade is held strictly under the SSIM threshold.
    env.reset()
    copy = env.metric_manager.get_term("painting")
    copy.canvas.paint(
        copy.reference.rgb.expand(env.num_envs, -1, -1, -1), copy.reference.painted.expand(env.num_envs, -1, -1)
    )
    copy.strokes.fill_(11)
    _hold(env, 1)
    assert bool(copy.measures["passed"].all()) and not bool(copy.passed.any())
    assert bool((copy.score < copy.success_threshold).all())
    assert not bool(env.termination_manager.get_term_dones("painted").any())
    env.reset()


def test_resetting_one_env_wipes_only_its_copy_and_empties_its_brush(env):
    env.reset()
    copy = env.metric_manager.get_term("painting")
    painter = ScriptedPainter(env)
    assert _paint_until(env, painter, lambda: (copy.canvas.painted_cells().flatten(1).sum(1) > 500).all())
    kept = copy.canvas.rgb()[1].clone(), copy.canvas.load[1].clone(), copy.paint[1].clone()
    copy.reset(torch.tensor([0]))
    assert not bool(copy.canvas.painted_cells()[0].any())
    assert (int(copy.paint[0]), float(copy.canvas.load[0]), int(copy.dips[0])) == (-1, 0.0, 0)
    torch.testing.assert_close(copy.canvas.rgb()[1], kept[0])
    assert (float(copy.canvas.load[1]), int(copy.paint[1])) == (float(kept[1]), int(kept[2]))
    env.reset()


def test_a_success_threshold_other_than_the_ssim_is_refused(env):
    term = PaintingCopy(env=env, options=PaintingCopy.configure(work="sunset", success_threshold=0.9))
    with pytest.raises(
        ValueError,
        match=r"Configure success_threshold=0.92, equal to ssim_l: configure\(\) sets it so when it is not given",
    ):
        term.build()


def test_the_painted_plane_is_the_paint_of_the_envs_asked_for_in_its_colours(env):
    env.reset()
    copy = env.metric_manager.get_term("painting")
    rgb = copy.canvas.rgb().clone()
    rgb[1, 10:20, 30:40] = torch.tensor((0.2, 0.4, 0.6))
    painted = torch.zeros(2, *copy.grid.shape, dtype=torch.bool)
    painted[1, 10:20, 30:40] = True
    copy.canvas.paint(rgb, painted)
    plane = copy.painted_plane(torch.tensor([1]))
    assert torch.equal(plane.weight, painted[1:])
    assert plane.grid == copy.grid and plane.weight.shape == (1, *copy.grid.shape)
    torch.testing.assert_close(plane.rgb[0, 15, 35], torch.tensor((0.2, 0.4, 0.6)))
    torch.testing.assert_close(plane.origin, torch.tensor(workcell.SHEET_CENTER))
    env.reset()


def test_a_tool_that_is_no_held_paintbrush_is_refused_naming_both(env):
    term = PaintingCopy(env=env, options=PaintingCopy.configure(work="sunset", tool_name="palette"))
    with pytest.raises(
        ValueError, match=r"work='sunset' is made with the 'paintbrush', but the scene's 'palette' is 'PaletteOptions'"
    ):
        term.build()


def test_a_work_whose_palette_the_scene_lacks_is_refused_naming_both(env):
    """The sunset's scene holds the sunset's palette; the Fuji study's wells would be ones the scene does not show."""
    term = PaintingCopy(env=env, options=PaintingCopy.configure(work="fuji_study"))
    with pytest.raises(
        ValueError,
        match=r"work='fuji_study' dips in its own palette, but the scene's 'palette' is the palette of work='sunset' "
        r"\(the scene's palettes, by work: \{'palette': 'sunset'\}\)\. Put PaletteOptions\(work='fuji_study'\)",
    ):
        term.build()


def test_a_palette_name_that_is_no_palette_is_refused(env):
    term = PaintingCopy(env=env, options=PaintingCopy.configure(work="sunset", palette_name="sheet"))
    with pytest.raises(
        ValueError,
        match=r"the scene's 'sheet' is a BoxOptions \(.*\)\. Set palette_name='palette', the scene's palette of",
    ):
        term.build()


def test_a_dip_in_one_env_and_a_stroke_in_another_in_the_same_step(env):
    """Env 0 dips in the second well while env 1, never dipped, makes a stroke: one run for both, no split."""
    env.reset()
    copy = env.metric_manager.get_term("painting")
    uv, depth = torch.zeros(2, 4, 2), torch.zeros(2, 4)
    uv[0] = copy.wells[1]
    depth[0] = torch.tensor([-0.003, 0.004, 0.004, -0.003])
    uv[1, :, 0] = torch.tensor([0.0, 0.001, 0.002, 0.003])
    depth[1] = 0.005
    copy.advance(uv, depth, torch.zeros(2, 4, 2))
    assert copy.dips.tolist() == [1, 0] and copy.paint.tolist() == [1, -1] and copy.strokes.tolist() == [0, 1]
    assert copy.canvas.load.tolist() == [1.0, 0.0] and not bool(copy.canvas.painted_cells().any())
    torch.testing.assert_close(copy.canvas.brush_rgb()[0], copy.palette[1], atol=1e-4, rtol=0.0)
    env.reset()


def test_strokes_begun_within_one_step_are_each_judged_a_scribble_or_not(env):
    """Env 0 scribbles (2 cm there and back), lifts, and sweeps 4 cm; env 1 makes the same two strokes the other way
    round. A scribble counts once it has ended; the stroke being made is counted as it stands.
    """
    env.reset()
    copy = env.metric_manager.get_term("painting")
    there_and_back = torch.cat([torch.linspace(0.0, 0.02, 21), torch.linspace(0.02, 0.0, 21)])
    scribble = torch.stack([there_and_back, torch.zeros(42)], dim=-1)
    sweep = torch.stack([torch.linspace(0.03, 0.07, 42), torch.full((42,), 0.02)], dim=-1)
    uv = torch.stack([torch.cat([scribble, sweep[:1], sweep]), torch.cat([sweep, scribble[:1], scribble])])
    depth = torch.full((2, 85), 0.005)
    depth[:, 42] = -0.003  # clear of the paper: the next touch is a new stroke
    copy.advance(uv, depth, torch.zeros(2, 85, 2))
    assert copy.strokes.tolist() == [2, 2]
    torch.testing.assert_close(copy.scribble_path, torch.tensor([0.04, 0.0]), atol=1e-5, rtol=0.0)
    torch.testing.assert_close(copy.scribble_share(), torch.tensor([0.5, 0.5]), atol=1e-4, rtol=0.0)
    env.reset()


def test_next_paint_is_the_paint_of_the_plan_s_next_point_and_where_its_well_is(env):
    env.reset()
    copy = env.metric_manager.get_term("painting")
    paint = copy.work_spec.strokes[0].paint
    observed = env.observation_manager.get_term("next_paint").compute()
    well = copy.wells[paint] - copy.tip_on_sheet()[:, :2]
    expected = torch.cat([copy.palette[paint].expand(2, 3), well, torch.ones(2, 1)], dim=-1)  # the brush holds none
    torch.testing.assert_close(observed, expected)


def test_leaving_the_sheet_pressed_and_coming_back_is_a_new_stroke(env):
    """Past the paper's edge the brush is off the paper, which stands a paper's thickness above the desk."""
    env.reset()
    copy = env.metric_manager.get_term("painting")
    u = torch.tensor([0.080, 0.085, 0.089, 0.095, 0.098, 0.089, 0.085])  # out past 0.09 and back, pressed throughout
    uv = torch.stack([u, torch.zeros_like(u)], dim=-1).expand(2, -1, -1).clone()
    copy.advance(uv, torch.full((2, 7), 0.002), torch.zeros(2, 7, 2))
    assert copy.strokes.tolist() == [2, 2]
    env.reset()


def test_a_brush_pressed_past_the_sheet_s_edge_paints_the_part_of_its_mark_on_the_paper(env):
    env.reset()
    copy = env.metric_manager.get_term("painting")
    copy.canvas.dip(torch.ones(2, dtype=torch.bool), copy.palette[[0, 0]])
    u = torch.tensor([0.093, 0.094, 0.095])  # the tip 3-5 mm past the edge, its mark 16 mm across
    uv = torch.stack([u, torch.zeros_like(u)], dim=-1).expand(2, -1, -1).clone()
    copy.advance(uv, torch.full((2, 3), copy.canvas.round_brush.depth_for(0.016)), torch.zeros(2, 3, 2))
    painted = copy.canvas.painted_cells()
    assert int(painted[0].sum()) > 50, "the mark's paper side is painted"
    assert copy.strokes.tolist() == [0, 0] and copy.path_length.tolist() == [0.0, 0.0], "but the tip never touched"
    env.reset()


def test_a_reset_env_s_picture_measures_are_the_bare_sheet_s(env):
    env.reset()
    copy = env.metric_manager.get_term("painting")
    copy.canvas.paint(
        copy.reference.rgb.expand(env.num_envs, -1, -1, -1), copy.reference.painted.expand(env.num_envs, -1, -1)
    )
    copy.strokes.fill_(11)  # over the budget, so the perfect picture does not end the episode (and reset it)
    _hold(env, 1)
    assert copy.measures["de_mean"].tolist() == pytest.approx([0.0, 0.0], abs=1e-3)
    copy.reset(torch.tensor([0]))
    assert float(copy.measures["de_mean"][0]) == pytest.approx(copy.bare_de_mean)
    assert float(copy.measures["de_mean"][1]) == pytest.approx(0.0, abs=1e-3)
    env.reset()


def _trace_first_points(copy, count=5):
    """Lay the plan's first ``count`` guide points (the sky's first sweep) in every env, at their depth."""
    points = copy.guide.points[:count]
    copy.advance(
        points[None, :, :2].expand(2, -1, -1).clone(),
        points[None, :, 2].expand(2, -1).clone(),
        torch.zeros(2, count, 2),
    )


def _load(copy, envs, paint):
    copy.canvas.dip(envs, copy.palette[torch.full((2,), paint)])
    copy.paint[envs] = paint


def test_plan_points_are_reached_only_with_their_paint_on_the_brush(env):
    """A dry brush (env 0) traces the plan for nothing, and the guide still leads it to the plan's first point; the
    brush holding the sky's paint (env 1) reaches the points it traces.
    """
    env.reset()
    copy = env.metric_manager.get_term("painting")
    sky = int(copy.stroke_paint[copy.guide.stroke[0]])
    _load(copy, torch.tensor([False, True]), sky)
    _trace_first_points(copy)
    assert copy.guide.reached[0].sum() == 0 and float(copy.guide.newly_reached[0]) == 0.0
    reached = copy.guide.reached[1]
    assert bool(reached[:5].all()), "the points traced, and any within guide_radius of them"
    assert float(copy.guide.newly_reached[1]) == pytest.approx(int(reached.sum()) / copy.guide.num_points)
    assert int(copy.guide.next_index()[0]) == 0 and int(copy.guide.next_index()[1]) >= 5
    env.reset()


def test_plan_points_passed_with_another_paint_or_a_brush_running_dry_stay_unreached(env):
    env.reset()
    copy = env.metric_manager.get_term("painting")
    sky = int(copy.stroke_paint[copy.guide.stroke[0]])
    _load(copy, torch.tensor([True, True]), (sky + 1) % len(copy.palette))
    _load(copy, torch.tensor([False, True]), sky)
    copy.canvas.load[1] = 0.1  # env 1 holds the sky's paint, running dry: it lays it at half opacity
    _trace_first_points(copy)
    assert not bool(copy.guide.reached.any())
    env.reset()


def test_a_reset_by_env_mask_mid_stroke_leaves_the_other_env_s_stroke_going(env):
    """An auto-reset hands a bool mask: the env reset starts afresh, the other keeps its stroke, paint and canvas."""
    env.reset()
    copy = env.metric_manager.get_term("painting")
    _load(copy, torch.tensor([True, True]), 0)
    u = torch.linspace(-0.02, 0.0, 6)
    copy.advance(
        torch.stack([u, torch.zeros_like(u)], -1).expand(2, -1, -1).clone(),
        torch.full((2, 6), 0.005),
        torch.zeros(2, 6, 2),
    )
    kept = copy.canvas.rgb()[1].clone(), float(copy.canvas.load[1]), float(copy._stroke_path[1])
    copy.reset(torch.tensor([True, False]))
    assert copy._in_stroke.tolist() == [False, True] and copy.touching.tolist() == [False, True]
    assert not bool(copy.canvas.painted_cells()[0].any()) and float(copy.canvas.load[0]) == 0.0
    torch.testing.assert_close(copy.canvas.rgb()[1], kept[0])
    assert (float(copy.canvas.load[1]), float(copy._stroke_path[1])) == kept[1:]
    assert float(copy.score[0]) == pytest.approx(float(copy._bare_score))
    env.reset()


def test_the_grade_observed_holds_the_bare_sheet_s_from_a_reset_to_the_first_step(env):
    """No jump on an episode's first step: a reset env observes its bare sheet's grade, not 0."""
    env.reset()
    copy = env.metric_manager.get_term("painting")
    grade = env.observation_manager.get_term("grade")
    reset = grade.compute()[:, 0].clone()
    torch.testing.assert_close(reset, copy.score)
    _hold(env, 1)
    torch.testing.assert_close(grade.compute()[:, 0], reset)
    assert bool((reset > 0.0).all()) and bool((reset < copy.success_threshold).all())


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"blur": 0.0}, r"PaintingRule: blur=0\.0; it is a standard deviation \(m\) and must be positive"),
        ({"de_p90": -1.0}, r"PaintingRule: de_p90=-1\.0; it is a ΔE and must be positive"),
        ({"paint_iou": 1.2}, r"PaintingRule: paint_iou=1\.2; it is at most 1 for any copy, so no copy would pass"),
        (
            {"stroke_allowance": 0.0},
            r"stroke_allowance=0\.0; it is a budget relative to the plan's and must be positive",
        ),
        ({"path_allowance": -1.0}, r"path_allowance=-1\.0; it is a budget"),
        ({"max_scribble_share": 1.5}, r"max_scribble_share=1\.5; it is a share of the path, in \[0, 1\]"),
        ({"guide_spacing": 0.0}, r"guide_spacing=0\.0; it is a distance on the paper \(m\) and must be positive"),
        ({"guide_radius": -0.001}, r"guide_radius=-0\.001; it is a distance"),
    ],
)
def test_a_rule_or_budget_no_copy_could_meet_is_refused_at_build(env, params, message):
    term = PaintingCopy(env=env, options=PaintingCopy.configure(work="sunset", **params))
    with pytest.raises(ValueError, match=message):
        term.build()


def test_a_copy_at_double_precision_is_refused_at_build(env, monkeypatch):
    import genesis as gs

    monkeypatch.setattr(gs, "tc_float", torch.float64)  # as eden.init(precision="64") sets it
    term = PaintingCopy(env=env, options=PaintingCopy.configure(work="sunset"))
    with pytest.raises(ValueError, match=r"initialized at precision='64', but the art tasks' .* run in float32"):
        term.build()


def test_the_first_plan_point_is_flagged_as_a_stroke_s_start(env):
    env.reset()
    points = env.observation_manager.get_term("next_points").compute().view(2, 4, 4)
    assert points[:, 0, 3].tolist() == [1.0, 1.0]


def test_an_art_term_with_no_metric_name_is_refused_naming_the_copy_metrics(env):
    term = TipPose(env=env, options=TipPose.configure())
    with pytest.raises(ValueError, match=r"metric_name is not set\. .* one of the CopyMetric metrics: \['painting'\]"):
        term.build()


@pytest.fixture(scope="module")
def recorded():
    """``art_painting_sunset`` with the ``rec`` camera ``eden inference --rec`` adds."""
    return build_task("art_painting_sunset", recording=True)


def test_the_rec_camera_paints_the_paint_of_the_env_it_shows_in_its_colours(recorded):
    """The paint is the metric's, not geometry in the scene: the ``rec`` camera's overlay is what shows it."""
    env = recorded
    copy, rec = env.metric_manager.get_term("painting"), env.cameras["rec"]
    try:
        clean = torch.tensor(rec.render_rgb(), dtype=torch.float32)
    except Exception as e:  # noqa: BLE001 - an offscreen-render failure is a skip, not a test failure
        pytest.skip(f"offscreen rendering unavailable: {e}")
    target, painted = copy.reference.rgb, copy.reference.painted
    bare = torch.zeros_like(painted)
    copy.canvas.paint(torch.stack([copy.canvas.rgb()[0], target]), torch.stack([bare, painted]))
    assert torch.equal(torch.tensor(rec.render_rgb(), dtype=torch.float32), clean), "rec shows env 0, unpainted"

    copy.canvas.paint(torch.stack([target, target]), torch.stack([painted, painted]))
    shown = torch.tensor(rec.render_rgb(), dtype=torch.float32)
    changed = (shown - clean).abs().sum(dim=-1) > 30
    assert int(changed.sum()) > 1000, "the sea and the sky should cover thousands of pixels of the sheet"
    points, _ = rec.render_pointcloud(world_frame=True)
    on_sheet = copy.to_sheet(torch.as_tensor(points[changed.numpy()], dtype=torch.float32))
    assert bool((on_sheet[:, :2].abs() <= copy.half_sheet + 0.002).all()), "painted off the sheet"
    # The sea's deep blue shows, shaded as the sheet is: some painted pixel is the clean sheet's pixel times sea / paper
    # colour, which no paper pixel is.
    sea = clean[changed] * torch.tensor(copy.work_spec.palette[1]) / torch.tensor(workcell.PAPER_COLOR)
    assert float((shown[changed] - sea).norm(dim=-1).min()) < 3.0
    assert float((clean[changed] - sea).norm(dim=-1).min()) > 40.0
