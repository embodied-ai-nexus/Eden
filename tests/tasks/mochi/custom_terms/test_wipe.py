"""Tests for the grab-bar stripe (Preston wear) and RoboSimStudio's sponge latch, on ``mochi_wipe_the_bar``."""

import math

import pytest
import torch

from eden.tasks.mochi import custom_terms
from eden.tasks.mochi.wipe_the_bar import config as wipe_config
from tests.tasks.mochi.conftest import action, build_task, place_tcp


@pytest.fixture(scope="module")
def env():
    return build_task("mochi_wipe_the_bar")


def _sponge_on_crown(env, u: torch.Tensor, lift: torch.Tensor) -> None:
    """Put the rest-shape sponge centred over the bar at ``u`` along it, its wiping face ``lift`` over the crown."""
    sponge = env.entities["sponge"]
    rest = torch.as_tensor(sponge.init_positions, dtype=torch.float, device=u.device)
    rest = rest - rest.mean(dim=0)
    bottom = rest[:, 2].min()
    crown = wipe_config.BAR_Z + wipe_config.TUBE_RADIUS
    centre = torch.stack([torch.full_like(u, wipe_config.BAR_X), wipe_config.BAR_Y + u, crown - bottom + lift], dim=-1)
    sponge.set_vertex_positions(rest[None] + centre[:, None, :])


@pytest.mark.requires_genesis
@pytest.mark.slow
def test_the_recording_camera_paints_each_cell_s_ink_where_the_cell_lies_on_the_crown():
    """The stripe is the metric's, not geometry: the overlay paints it, cell by cell, as dark as its ink.

    Only the cells toward the near block (u < 0) and toward +x (angle > 0) keep their ink, so every painted pixel's world
    point must lie on that quarter of the crown.
    """
    env = build_task("mochi_wipe_the_bar", recording=True)
    wipe, rec = env.metric_manager.get_term("cleaned"), env.cameras["rec"]
    wipe.ink.zero_()
    try:
        clean = rec.render_rgb().astype(int)
    except Exception as e:  # noqa: BLE001 - an offscreen-render failure is a skip, not a test failure
        pytest.skip(f"offscreen rendering unavailable: {e}")
    kept = (wipe.cells[:, 0] < 0) & (wipe.cells[:, 1] > 0)
    wipe.ink[:, kept] = 1.0
    painted = rec.render_rgb().astype(int)

    changed = abs(painted - clean).sum(axis=-1) > 30
    points, _ = rec.render_pointcloud(world_frame=True)
    assert changed.sum() > 30, "test premise: the camera sees the stripe"
    margin = wipe_config.STRIPE_HALF / 40  # a pixel's footprint at the stripe and more
    assert (points[changed][:, 1] < wipe_config.BAR_Y + margin).all(), "painted beyond u < 0"
    assert (points[changed][:, 0] > wipe_config.BAR_X - margin / 4).all(), "painted on the crown's -x side"
    colour = torch.tensor(wipe_config.STRIPE_COLOR) * 255
    darkest = painted[changed][(abs(painted[changed] - colour.numpy())).max(axis=-1).argmin()]
    assert abs(darkest - colour.numpy()).max() <= 1, "a fully inked pixel is the stripe's colour"


@pytest.mark.requires_genesis
@pytest.mark.slow
def test_the_stripe_is_243_cells_on_the_crown(env):
    """RoboSimStudio's stripe: 2.5 mm cells over the middle 200 mm of the run and 9 degrees either side of the top."""
    wipe = env.metric_manager.get_term("cleaned")
    cells = wipe.cell_positions()
    assert cells.shape == (243, 3)
    axis = torch.tensor([wipe_config.BAR_X, wipe_config.BAR_Z], device=cells.device)
    radial = (cells[:, (0, 2)] - axis).norm(dim=-1)
    torch.testing.assert_close(radial, torch.full_like(radial, wipe_config.TUBE_RADIUS))
    along = cells[:, 1] - wipe_config.BAR_Y
    assert float(along.min()) == pytest.approx(-0.1) and float(along.max()) == pytest.approx(0.1)
    assert float(wipe.cells[:, 1].abs().max()) <= math.radians(12.0)  # one ring of cells either side of the top


@pytest.mark.requires_genesis
@pytest.mark.slow
def test_a_sliding_press_wears_the_cells_under_the_sponge_and_no_others(env):
    """Env 0's sponge sits pressed into the crown sliding along the bar; env 1's slides 5 cm above it."""
    env.reset()
    sponge = env.entities["sponge"]
    wipe = env.metric_manager.get_term("cleaned")
    u = torch.tensor([-0.05, -0.05], device=env.device)
    _sponge_on_crown(env, u, torch.tensor([-0.004, 0.05], device=env.device))
    sponge.set_vertex_velocities(torch.tensor([0.0, 0.2, 0.0], device=env.device).expand(2, sponge.n_vertices, 3))
    env.step(action(env))
    worn = wipe.ink < 1.0
    assert not worn[1].any()  # pressing nothing wears nothing
    assert worn[0].any()
    # Only cells under the 75 mm sponge (with a vertex's reach to spare) wore.
    along = wipe.cells[:, 0]
    under = (along - u[0]).abs() <= 0.5 * wipe_config.SPONGE_SIZE[1] + wipe.reach
    assert not worn[0, ~under].any()
    assert float(env.metric_manager.get_value("press")[0]) > 0.0
    assert float(env.metric_manager.get_value("press")[1]) == 0.0

    env.reset()
    assert (wipe.ink == 1.0).all()


@pytest.mark.requires_genesis
@pytest.mark.slow
def test_the_latch_takes_a_settled_grasp_on_the_sponge_and_not_a_close_on_air(env):
    """Env 0 closes on the sponge's middle; env 1 closes 10 cm above it. Only env 0 latches, then carries it."""
    env.reset()
    sponge = env.entities["sponge"]
    latch = env.action_manager.get_term("gripper")
    centroid = sponge.get_vertex_positions().mean(dim=1)
    place_tcp(env, "right", centroid + torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.10]], device=env.device))
    for _ in range(round(1.0 / wipe_config.CONTROL_DT)):
        env.step(action(env, close={"gripper": True}))
    assert latch.is_latched.tolist() == [True, False]
    assert float(env.metric_manager.get_value("grip")[0]) >= latch.min_press
    # What it pinned is the foam under the pads: every pinned vertex within reach of a pad's face.
    pinned = latch._pinned[0][latch._pinned[0] >= 0].long()
    distance = latch._face_distance(sponge.get_vertex_positions())[0, pinned]
    assert len(pinned) > 0 and float(distance.max()) <= latch.reach + 1e-4

    before = sponge.get_vertex_positions()[:, :, 2].mean(dim=-1)
    for _ in range(30):
        env.step(action(env, close={"gripper": True}, move={"hand": torch.tensor([0.0, 0.0, 1.0], device=env.device)}))
    after = sponge.get_vertex_positions()[:, :, 2].mean(dim=-1)
    assert float(after[0] - before[0]) > 0.05  # carried up in the jaws
    assert abs(float(after[1] - before[1])) < 0.002  # still lying on the mat

    env.step(action(env, close={"gripper": False}))
    assert latch.is_latched.tolist() == [False, False]


def _wipe_config():
    from eden.tasks import TASK_REGISTRY

    config = TASK_REGISTRY.build("mochi_wipe_the_bar")
    config.env_options.num_envs = 1
    return config


@pytest.mark.requires_genesis
@pytest.mark.slow
@pytest.mark.parametrize(
    ("break_config", "error", "match"),
    [
        pytest.param(
            lambda c: setattr(
                c.reward_options,
                "reach_ink",
                custom_terms.ReachDirtyCell.configure(entity_name="sponge", grip_metric_name="gripp", weight=0.2),
            ),
            KeyError,
            r"Metric 'gripp' not found",
            id="reach_dirty_cell-grip-metric",
        ),
        pytest.param(
            lambda c: setattr(
                c.metric_options,
                "press",
                custom_terms.BarPress.configure(metric_name="grip", contributes_to_success=False),
            ),
            ValueError,
            r"metric_name='grip', which is not a BarStripeWipe metric \(BarStripeWipe metrics: \['cleaned'\]",
            id="bar_press-source",
        ),
        pytest.param(
            lambda c: setattr(
                c,
                "metric_options",
                type(c.metric_options)(
                    press=c.metric_options.press,
                    **{k: getattr(c.metric_options, k) for k in c.metric_options.term_keys() if k != "press"},
                ),
            ),
            ValueError,
            r"'press' is declared before its BarStripeWipe metric 'cleaned'.*Declare 'press' after",
            id="bar_press-order",
        ),
    ],
)
def test_a_misnamed_or_misordered_metric_handle_raises_at_build(break_config, error, match):
    """Grip gates and the press read sibling metrics: a bad handle fails the build, not the first step."""
    from eden.envs.base import RLEnvBase

    config = _wipe_config()
    break_config(config)
    with pytest.raises(error, match=match):
        RLEnvBase.from_config(config).build()
