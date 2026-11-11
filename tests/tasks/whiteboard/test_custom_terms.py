"""Tests for the whiteboard terms on a built ``whiteboard_write_text``: the grasp, the ink gate and the drop."""

import math

import numpy as np
import pytest
import torch

from eden.tasks.whiteboard import common
from eden.tasks.whiteboard.custom_terms import WhiteboardDrawing
from eden.tasks.whiteboard.scripted_demos import ScriptedDrawer
from eden.tasks.whiteboard.strokes import DRAWINGS
from tests.tasks.whiteboard.conftest import build_task

pytestmark = pytest.mark.requires_genesis


@pytest.fixture(scope="module")
def env():
    return build_task("whiteboard_write_text")


def _hold(env, steps):
    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
    for _ in range(steps):
        _, reward, terminated, truncated, _ = env.step(zero)
    return reward, terminated, truncated


def test_the_marker_starts_in_the_hand_and_stays_there_by_friction(env):
    env.reset()
    marker, gripper = env.entities["marker"], env.entities["robot_gripper"]
    pads = (gripper.get_link("left_gripper_pad").get_pos() + gripper.get_link("right_gripper_pad").get_pos()) / 2

    def grip_point():
        drawing = env.metric_manager.get_term("drawing")
        tip, quat = drawing.nib_pose()
        axis = torch.nn.functional.normalize(tip - marker.get_pos(relative=True), dim=-1)
        return marker.get_pos(relative=True) + common.GRIP_Z * axis

    torch.testing.assert_close(grip_point(), pads, atol=1e-3, rtol=0)
    _, terminated, _ = _hold(env, 40)  # 2 s of holding still
    assert not bool(terminated.any())
    pads = (gripper.get_link("left_gripper_pad").get_pos() + gripper.get_link("right_gripper_pad").get_pos()) / 2
    assert float((grip_point() - pads).norm(dim=-1).max()) < 3e-3


def test_a_hovering_nib_draws_nothing(env):
    env.reset()
    _hold(env, 20)
    drawing = env.metric_manager.get_term("drawing")
    gap = drawing.to_board(drawing.nib_pose()[0])[:, 2]
    assert bool((gap > 0.015).all())
    assert float(drawing.ink.pigment.sum()) == 0.0 and float(drawing.nib_force.abs().sum()) == 0.0


def test_ink_is_laid_where_the_nib_presses_and_nowhere_else(env):
    env.reset()
    drawing = env.metric_manager.get_term("drawing")
    drawer = ScriptedDrawer(env)
    pressed = None
    for _ in range(400):
        env.step(drawer.act())
        if bool((drawing.nib_force > 0.2).all()):
            pressed = drawing.to_board(drawing.nib_pose()[0])
            break
    assert pressed is not None, "the drawer never pressed the nib on the board"
    assert bool((pressed[:, 2].abs() < 1e-3).all())  # pressed onto the face
    weight = drawing.ink.weight
    inked = weight > 0.5
    assert bool(inked.flatten(1).any(dim=1).all())
    # Every inked cell is within the chisel's reach of where the nib went, here the first stroke's first point.
    rows, cols = torch.nonzero(inked[0], as_tuple=True)
    u = drawing.grid.u0 + (cols.float() + 0.5) * drawing.grid.cell
    v = drawing.grid.v0 - (rows.float() + 0.5) * drawing.grid.cell
    start = drawing.guide.points[0]
    assert float(torch.hypot(u - start[0], v - start[1]).max()) < 0.02


def test_resetting_one_env_wipes_only_its_drawing(env):
    env.reset()
    drawing = env.metric_manager.get_term("drawing")
    drawer = ScriptedDrawer(env)
    for _ in range(600):
        env.step(drawer.act())
        if bool((drawing.guide.covered.sum(dim=1) > 5).all()):
            break
    assert bool((drawing.ink.total > 0).all()), "the drawer inked nothing to reset"
    kept = [t[1].clone() for t in (drawing.ink.pigment, drawing.judge.reach, drawing.guide.covered, drawing.score)]
    drawing.reset(torch.tensor([0]))
    assert float(drawing.ink.pigment[0].sum()) == 0.0 and float(drawing.ink.total[0]) == 0.0
    assert float(drawing.judge.reach[0].sum()) == 0.0 and not bool(drawing.guide.covered[0].any())
    for before, now in zip(kept, (drawing.ink.pigment, drawing.judge.reach, drawing.guide.covered, drawing.score)):
        torch.testing.assert_close(now[1], before)


@pytest.fixture(scope="module")
def recorded():
    """``whiteboard_write_text`` with the ``rec`` camera ``eden inference --rec`` adds."""
    return build_task("whiteboard_write_text", recording=True)


def test_the_rec_camera_paints_the_ink_of_the_env_it_shows_in_the_markers_colour(recorded):
    """The ink is the metric's, not geometry in the scene: the ``rec`` camera's overlay is what shows it."""
    env = recorded
    drawing, rec = env.metric_manager.get_term("drawing"), env.cameras["rec"]
    try:
        clean = torch.tensor(rec.render_rgb(), dtype=torch.float32)
    except Exception as e:  # noqa: BLE001 - an offscreen-render failure is a skip, not a test failure
        pytest.skip(f"offscreen rendering unavailable: {e}")
    reference = torch.tensor(drawing.grid.raster(DRAWINGS[drawing.drawing]), dtype=torch.float32)

    drawing.ink.paint(torch.stack([torch.zeros_like(reference), reference]))
    assert torch.equal(torch.tensor(rec.render_rgb(), dtype=torch.float32), clean), "rec shows env 0, which has no ink"

    drawing.ink.paint(torch.stack([reference, reference]))
    inked = torch.tensor(rec.render_rgb(), dtype=torch.float32)
    changed = (inked - clean).abs().sum(dim=-1) > 30
    assert int(changed.sum()) > 300, "HELLO should cover hundreds of pixels of the board"
    # The ink is shaded as the board is: a pixel a stroke covers whole is the clean board's pixel in the ink's colour,
    # the board's rendered light (its shadows included) times ink / board colour.
    # The colour the renderer draws the board in, read off the built entity as the overlay reads it.
    board = torch.tensor(
        np.asarray(env.entities["board"].entity.vgeoms[0].surface.get_rgba().color, dtype=np.float32)[:3]
    )
    shaded = clean[changed] * torch.tensor(common.COLORS["black"]) / board
    # Every painted pixel moved toward that, and where a stroke covers the whole pixel it is that.
    assert bool(((inked[changed] - shaded).norm(dim=-1) < (clean[changed] - shaded).norm(dim=-1)).all())
    assert float((inked[changed] - shaded).abs().max(dim=-1).values.min()) <= 1.0


def test_a_dropped_marker_ends_the_episode(env):
    env.reset()
    marker = env.entities["marker"]
    away = marker.get_pos(relative=True).clone()
    away[0, 2] -= 0.2
    marker.set_pos(away[:1], envs_idx=torch.tensor([0]), relative=True)
    _, terminated, _ = _hold(env, 1)
    assert terminated.tolist() == [True, False]
    assert env.termination_manager.get_term_dones("marker_dropped").tolist() == [True, False]


def test_a_misnamed_nib_geom_fails_the_build_naming_the_marker_geoms(env):
    term = WhiteboardDrawing(env=env, options=WhiteboardDrawing.configure(nib_geom="tip"))
    with pytest.raises(
        ValueError, match=r"has 0 geoms named nib_geom='tip' \(its geoms: \['barrel', 'collar', 'nib'\]\)"
    ):
        term.build()


def test_a_table_graded_short_of_1_is_refused(env):
    term = WhiteboardDrawing(env=env, options=WhiteboardDrawing.configure(drawing="table", success_threshold=0.85))
    with pytest.raises(ValueError, match=r"success_threshold=0.85 would count an unfinished table as drawn"):
        term.build()


def test_the_next_points_lead_from_the_nib_along_the_reference(env):
    env.reset()
    drawing = env.metric_manager.get_term("drawing")
    values = env.observation_manager.get_term("next_points").compute()
    first = drawing.guide.points[0] - drawing.to_board(drawing.nib_pose()[0])[:, :2]
    torch.testing.assert_close(values[:, :2], first)
    assert values.shape == (env.num_envs, 12)
    assert math.isclose(float(values[0, 2]), 0.0)


def test_a_drawing_term_the_metric_manager_does_not_compute_is_harmless(env):
    # Built on its own, its sampling callback stays registered with the env for good: stepping must not fail on it
    # (it sees several control steps of samples with no compute), and the task's own metric still grades.
    env.reset()
    spare = WhiteboardDrawing(env=env, options=WhiteboardDrawing.configure())
    spare.build()
    drawing = env.metric_manager.get_term("drawing")
    drawer = ScriptedDrawer(env)
    for _ in range(300):
        env.step(drawer.act())
        if bool((drawing.ink.total > 0).all()):
            break
    assert bool((drawing.ink.total > 0).all()), "the task's metric stopped inking"
    # Computed late, it draws its latest control step's samples.
    late = spare.compute()
    assert late.shape == (env.num_envs,) and bool(torch.isfinite(late).all())


def test_a_sample_the_solver_failed_on_draws_nothing_and_stays_finite(env, monkeypatch):
    # A NaN press read off a failed env (env 0 here) is dropped: no ink, a finite press, grade and stray penalty.
    env.reset()
    drawing = env.metric_manager.get_term("drawing")
    real_press = drawing._press

    def press():
        values = real_press().clone()
        values[0] = float("nan")
        return values

    monkeypatch.setattr(drawing, "_press", press)
    _hold(env, 3)
    assert bool(torch.isfinite(drawing.nib_force).all()) and float(drawing.nib_force[0]) == 0.0
    assert bool(torch.isfinite(drawing.score).all()) and bool(torch.isfinite(drawing.new_stray_m2).all())
    assert float(drawing.ink.total[0]) == 0.0


def test_the_first_next_point_flags_a_stroke_to_start(env):
    # With HELLO's first stroke (H's left post) inked, the next point to ink starts the next stroke: lift there.
    env.reset()
    drawing = env.metric_manager.get_term("drawing")
    weight = torch.tensor(drawing.grid.raster(DRAWINGS["hello"][:1]), dtype=torch.float32).expand(env.num_envs, -1, -1)
    drawing.guide.update(drawing.ink.paint(weight))
    values = env.observation_manager.get_term("next_points").compute()
    assert bool((drawing.guide.stroke[drawing.guide.next_index()] == 1).all())
    assert values[:, 2].tolist() == [1.0] * env.num_envs
    env.reset()
