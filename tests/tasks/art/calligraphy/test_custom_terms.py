"""Tests for the calligraphy terms on a built ``art_calligraphy_dai``: the held brush, the ink gate, the hand's count."""

import numpy as np
import pytest
import torch

from eden.tasks.art.calligraphy.grading import INK_COLOR
from eden.tasks.art.calligraphy.custom_terms import CalligraphyWriting
from eden.tasks.art.calligraphy.scripted_demos import ScriptedWriter
from eden.tasks.art import workcell
from tests.tasks.art.calligraphy.conftest import build_task

pytestmark = pytest.mark.requires_genesis


@pytest.fixture(scope="module")
def env():
    return build_task("art_calligraphy_dai")


def _hold(env, steps):
    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
    for _ in range(steps):
        env.step(zero)


def _write_until(env, writer, condition, steps=600):
    for _ in range(steps):
        env.step(writer.act())
        if bool(condition()):
            return True
    return False


def test_the_brush_starts_upright_above_the_sheets_centre_and_stays_in_the_hand(env):
    env.reset()
    writing = env.metric_manager.get_term("writing")
    tip, quat = writing.tip_pose()
    sheet = writing.to_sheet(tip)
    # HOME puts the tip 30 mm above the centre, give or take the arm's reset noise (0.02 rad a joint swings the
    # tip, 30 cm out from the wrist, by up to ~2.5 cm).
    torch.testing.assert_close(sheet, torch.tensor([[0.0, 0.0, -workcell.HOME_STANDOFF]] * 2), atol=0.025, rtol=0)
    axis = writing.to_sheet(tip + 0.01 * torch.nn.functional.normalize(tip - env.entities["tool"].get_pos(), dim=-1))
    assert bool((axis[:, 2] > sheet[:, 2]).all())  # the tool points into the paper
    hand = env.entities["robot_gripper"].get_link("hand")
    before = writing.tip_pose()[0] - hand.get_pos()
    _hold(env, 40)  # 2 s
    torch.testing.assert_close(writing.tip_pose()[0] - hand.get_pos(), before, atol=1e-4, rtol=0)


def test_the_observations_right_after_a_reset_are_finite(env):
    # The tool is a welded body with no DOF of its own: its pose must be valid before the first physics step.
    obs, _ = env.reset()
    assert bool(torch.isfinite(obs["policy"]).all()) and bool(torch.isfinite(obs["critic"]).all())


def test_a_brush_in_the_air_inks_nothing(env):
    env.reset()
    _hold(env, 10)
    writing = env.metric_manager.get_term("writing")
    assert int(writing.ink.inked.sum()) == 0 and int(writing.strokes.sum()) == 0
    assert float(writing.score.sum()) == 0.0


def test_writing_inks_near_the_tip_and_counts_a_stroke(env):
    env.reset()
    writing = env.metric_manager.get_term("writing")
    writer = ScriptedWriter(env)
    assert _write_until(env, writer, lambda: (writing.ink.inked.flatten(1).sum(1) > 50).all())
    assert writing.strokes.tolist() == [1, 1]
    rows, cols = torch.nonzero(writing.ink.inked[0], as_tuple=True)
    u = writing.grid.u0 + (cols.float() + 0.5) * writing.grid.cell
    v = writing.grid.v0 - (rows.float() + 0.5) * writing.grid.cell
    tip = writing.to_sheet(writing.tip_pose()[0])[0]
    # Within the brush's largest mark (its root up to Offset + Drag behind the tip) of where the tip is.
    assert float(torch.hypot(u - tip[0], v - tip[1]).max()) < 0.04
    assert bool((writing.coverage > 0).all()) and bool((writing.guide.reached.any(dim=1)).all())


def test_resetting_one_env_wipes_only_its_copy(env):
    env.reset()
    writing = env.metric_manager.get_term("writing")
    writer = ScriptedWriter(env)
    assert _write_until(env, writer, lambda: (writing.ink.inked.flatten(1).sum(1) > 50).all())
    kept = [t[1].clone() for t in (writing.ink.inked, writing.guide.reached, writing.strokes, writing.path_length)]
    writing.reset(torch.tensor([0]))
    assert int(writing.ink.inked[0].sum()) == 0 and not bool(writing.guide.reached[0].any())
    assert int(writing.strokes[0]) == 0 and float(writing.path_length[0]) == 0.0
    now = (writing.ink.inked, writing.guide.reached, writing.strokes, writing.path_length)
    for before, after in zip(kept, now):
        torch.testing.assert_close(after[1], before)
    env.reset()


def test_a_copy_that_breaks_a_hand_rule_is_graded_just_under_success(env):
    # The target painted on with no tool: the picture is perfect, but no stroke was written, so the mark-width rule
    # (a mean over no path) fails. The grade is held strictly under the IoU threshold, in the score's float32.
    env.reset()
    writing = env.metric_manager.get_term("writing")
    writing.ink.paint(writing.judge.target.expand(env.num_envs, -1, -1))
    _hold(env, 1)
    assert bool(writing.measures["passed"].all()) and not bool(writing.passed.any())
    assert bool((writing.score < writing.success_threshold).all())
    assert not bool(env.termination_manager.get_term_dones("written").any())
    env.reset()


def test_a_reset_clears_the_tool_mark_of_the_envs_it_resets_only(env):
    """The auto-reset runs before the observations, so a reset env's first ToolMark must not be its last episode's."""
    env.reset()
    writing = env.metric_manager.get_term("writing")
    mark = writing.last_footprint
    mark.live[:] = True  # as if the episode's last control step inked, in both envs
    mark.width[:] = 0.008
    mark.length[:] = 0.009
    env.reset(torch.tensor([1]))
    assert mark.live.tolist() == [True, False]
    assert mark.width.tolist() == [pytest.approx(0.008), 0.0] and mark.length.tolist() == [pytest.approx(0.009), 0.0]
    assert torch.equal(mark.root[1], torch.zeros(2)) and torch.equal(mark.along[1], torch.zeros(2))


def test_the_observations_have_their_documented_widths(env):
    env.reset()
    _hold(env, 1)
    manager = env.observation_manager
    widths = {name: manager.get_term(name).compute().shape for name in ("tip_pose", "tool_mark", "next_points")}
    assert widths == {"tip_pose": (2, 6), "tool_mark": (2, 7), "next_points": (2, 16)}
    assert manager.get_term("strokes_left").compute().tolist() == [[3.0], [3.0]]


def test_a_work_for_another_tool_is_refused_naming_both(env):
    term = CalligraphyWriting(env=env, options=CalligraphyWriting.configure(work="minimum"))
    with pytest.raises(ValueError, match=r"work='minimum' is made with the 'pen', but the scene's 'tool' is 'brush'"):
        term.build()


def test_a_success_threshold_other_than_the_iou_is_refused(env):
    term = CalligraphyWriting(env=env, options=CalligraphyWriting.configure(work="dai", success_threshold=0.9))
    with pytest.raises(
        ValueError,
        match=r"Configure success_threshold=0.8, equal to iou: configure\(\) sets it so when it is not given",
    ):
        term.build()


def test_the_painted_plane_is_the_ink_of_the_envs_asked_for_on_the_sheet(env):
    env.reset()
    writing = env.metric_manager.get_term("writing")
    mark = torch.zeros(2, *writing.grid.shape, dtype=torch.bool)
    mark[1, 10:20, 30:40] = True
    writing.ink.paint(mark)
    plane = writing.painted_plane(torch.tensor([1]))
    assert torch.equal(plane.weight, mark[1:])
    assert plane.grid == writing.grid
    torch.testing.assert_close(plane.origin, torch.tensor(workcell.SHEET_CENTER))


@pytest.fixture(scope="module")
def recorded():
    """``art_calligraphy_dai`` with the ``rec`` camera ``eden inference --rec`` adds."""
    return build_task("art_calligraphy_dai", recording=True)


def test_the_rec_camera_paints_the_ink_of_the_env_it_shows_on_the_sheet(recorded):
    """The ink is the metric's, not geometry in the scene: the ``rec`` camera's overlay is what shows it."""
    env = recorded
    writing, rec = env.metric_manager.get_term("writing"), env.cameras["rec"]
    try:
        clean = torch.tensor(rec.render_rgb(), dtype=torch.float32)
    except Exception as e:  # noqa: BLE001 - an offscreen-render failure is a skip, not a test failure
        pytest.skip(f"offscreen rendering unavailable: {e}")
    target = torch.as_tensor(writing.reference.target, dtype=torch.bool)

    writing.ink.paint(torch.stack([torch.zeros_like(target), target]))
    assert torch.equal(torch.tensor(rec.render_rgb(), dtype=torch.float32), clean), "rec shows env 0, which has no ink"

    writing.ink.paint(torch.stack([target, target]))
    inked = torch.tensor(rec.render_rgb(), dtype=torch.float32)
    changed = (inked - clean).abs().sum(dim=-1) > 30
    assert int(changed.sum()) > 1000, "大 should cover thousands of pixels of the sheet"
    points, _ = rec.render_pointcloud(world_frame=True)
    on_sheet = writing.to_sheet(torch.as_tensor(points[changed.numpy()], dtype=torch.float32))
    half = torch.tensor(writing.work_spec.sheet) / 2
    assert bool((on_sheet[:, :2].abs() <= half + 0.002).all()), "painted off the sheet"
    # The ink is shaded as the sheet is: a pixel a stroke covers whole is the clean sheet's pixel in the ink's colour,
    # the sheet's rendered light (the tool's shadow included) times ink / sheet colour.
    # The colour the renderer draws the sheet in, read off the built entity as the overlay reads it.
    sheet = torch.tensor(
        np.asarray(env.entities["sheet"].entity.vgeoms[0].surface.get_rgba().color, dtype=np.float32)[:3]
    )
    ink = clean[changed] * torch.tensor(INK_COLOR) / sheet
    # Every painted pixel moved toward that, and where a stroke covers the whole pixel it is that.
    assert bool(((inked[changed] - ink).norm(dim=-1) < (clean[changed] - ink).norm(dim=-1)).all())
    assert float((inked[changed] - ink).abs().max(dim=-1).values.min()) <= 1.0
