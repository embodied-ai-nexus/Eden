"""The Extreme Parkour teacher task: its terrain grid and goals (numpy level), and spawn/step on a real 2x2 env."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from genesis.ext.isaacgym import terrain_utils as tu

from eden.tasks import TASK_REGISTRY
from eden.tasks.extreme_parkour.teacher.config import (
    _COLS,
    _ROWS,
    ExtremeParkourTeacherConfig,
    ExtremeParkourTerrainSamplerMod,
    _make_parkour_grid,
)
from tests.support.envs import build_env
from tests.tasks.extreme_parkour.conftest import GRID_COLS, GRID_ROWS, NUM_ENVS
from eden.utils.terrains import TERRAIN_GENERATOR_REGISTRY
from eden.utils.terrains.base import generate_terrain_fields

pytestmark = pytest.mark.requires_genesis

_SPAWN_OFFSET_X = 7.0


def test_config_wiring():
    cfg = TASK_REGISTRY.build("extreme_parkour_teacher")
    assert list(cfg.observation_options.term_keys()) == ["policy"]
    # Trained by the RMA stack (stock PPO does not learn the task); the mirror loss is the task's own.
    assert cfg.runner_options.actor.class_name == "eden.learning.wrappers.rsl_rl.rma.model:RmaActorModel"
    assert cfg.runner_options.algorithm.class_name == "eden.learning.wrappers.rsl_rl.rma.ppo:RmaPPO"
    assert cfg.runner_options.algorithm.mirror_class_name == "eden.tasks.extreme_parkour.symmetry:ParkourMirror"
    # The spawn event references the terrain-level curriculum by name.
    assert "terrain_level" in cfg.curriculum_options.term_keys()
    assert set(cfg.termination_options.term_keys()) == {
        "time_out",
        "roll_limit",
        "pitch_limit",
        "base_height",
        "all_goals_reached",
    }
    for name in cfg.termination_options.term_keys():
        assert getattr(cfg.termination_options, name).time_out is True, name


def test_full_size_grid_generates_eight_goals_per_cell():
    """Row-0 and row-9 difficulty params must not crash any generator."""
    grid = _make_parkour_grid()
    assert len(grid) == _ROWS and all(len(row) == _COLS for row in grid)
    fields = generate_terrain_fields(grid, (16.0, 4.0), (_ROWS, _COLS), 0.08, 0.005, randomize=True)
    assert np.isfinite(fields.height_field).all()
    assert fields.goals.shape[:2] == (_ROWS, _COLS)
    # num_gaps / num_stones = 6 -> exactly 8 goals in every cell.
    assert (fields.goals_count == 8).all()


# One pass over the shipped grid gives 200 cells; the tight courses only show up across repeats.
_GOAL_PIN_PASSES = 8
#: Tightest spacing (m) between consecutive goals in those 1600 cells (``parkour_terrain``, hardest row).
_TIGHTEST_GOAL_GAP_M = 0.24
#: Farthest a goal reaches down a 16 m lane, once the final-goal clearance is kept.
_FARTHEST_GOAL_M = 15.52


def test_lane_goals_are_ordered_reachable_and_resolvable():
    """Goals advance strictly down each 16 x 4 m lane, stay in it, and stay farther apart than the command's
    ``next_goal_threshold``. The tightest gap and farthest goal are recorded pins, so retuning a course comes past here.
    """
    threshold = TASK_REGISTRY.build("extreme_parkour_teacher").command_options.parkour_command.next_goal_threshold
    lane_x, lane_y = 16.0, 4.0
    hs, vs = 0.08, 0.005
    num_x, num_y = int(lane_x / hs) + 1, int(lane_y / hs) + 1
    grid = _make_parkour_grid()

    tightest_gap, farthest_goal = float("inf"), 0.0
    for pass_i in range(_GOAL_PIN_PASSES):
        for i, row in enumerate(grid):
            for j, term_options in enumerate(row):
                np.random.seed(pass_i * 1000 + i * len(row) + j)
                subterrain = tu.SubTerrain(width=num_x, length=num_y, vertical_scale=vs, horizontal_scale=hs)
                goals = (
                    TERRAIN_GENERATOR_REGISTRY.get(term_options.name)(options=term_options).compute(subterrain).goals
                )
                where = f"{term_options.name} row {i} col {j} pass {pass_i}"

                assert (np.diff(goals[:, 0]) > 0).all(), f"{where}: goals not ordered: {goals[:, 0].tolist()}"
                assert (goals[:, 0] >= 0).all() and (goals[:, 0] < lane_x).all(), f"{where}: goal off the lane"
                tightest_gap = min(tightest_gap, float(np.diff(goals[:, 0]).min()))
                farthest_goal = max(farthest_goal, float(goals[:, 0].max()))

    assert tightest_gap > threshold, f"goals {tightest_gap:.3f} m apart are within the {threshold} m threshold"
    assert tightest_gap == pytest.approx(_TIGHTEST_GOAL_GAP_M, abs=5e-3)
    assert farthest_goal == pytest.approx(_FARTHEST_GOAL_M, abs=5e-3)


def test_policy_group_width_is_753(parkour_env):
    assert parkour_env.observation_manager.group_obs_dim["policy"] == (753,)


def test_env_steps_with_finite_command_buffers(parkour_env):
    env = parkour_env
    env.reset()
    assert env.entities["terrain"].goals.shape[:2] == (2, 2)

    for _ in range(3):
        obs, reward, *_ = env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device))
    assert torch.isfinite(obs["policy"]).all()
    assert torch.isfinite(reward).all()

    command = env.command_manager.get_term("parkour_command")
    for buf in (command.command, command.target_pos_rel, command.target_yaw, command.dis_to_start):
        assert torch.isfinite(buf).all()
    assert ((command.cur_goal_idx >= 0) & (command.cur_goal_idx < command.env_goals.shape[1])).all()
    # Env i walks column floor(i / (num_envs / cols)); column 1 is the flat (obstacle-suppressed) course.
    assert command.env_is_flat.tolist() == [False, False, True, True]


def test_reset_places_the_robot_at_its_cells_spawn(parkour_env):
    env = parkour_env
    env.reset()
    command = env.command_manager.get_term("parkour_command")
    curriculum = env.curriculum_manager.get_term("terrain_level")
    robot = env.entities["robot"]

    # The level curriculum is the spawn event's sampler over every cell; it moves envs between rows only,
    # so each env keeps the column dealt at build and the command applies the levels it decided.
    assert curriculum.num_configs == GRID_ROWS * GRID_COLS
    assert command.terrain_types.tolist() == [0, 0, 1, 1]
    assert torch.equal(command.terrain_levels, curriculum.terrain_levels)
    assert curriculum.assignment.tolist() == (command.terrain_levels * GRID_COLS + command.terrain_types).tolist()

    origins = env.entities["terrain"].subterrain_origins[command.terrain_levels, command.terrain_types]
    expected_xy = origins[:, :2].clone()
    expected_xy[:, 0] -= _SPAWN_OFFSET_X  # spawn = cell center - 7 m along the course
    torch.testing.assert_close(command.spawn_pos, expected_xy, atol=1e-5, rtol=0)
    torch.testing.assert_close(robot.get_pos()[:, :2], expected_xy, atol=1e-3, rtol=0)
    torch.testing.assert_close(
        robot.get_pos()[:, 2], torch.full((env.num_envs,), 0.4, device=env.device), atol=1e-3, rtol=0
    )
    assert (command.cur_goal_idx == 0).all()


def test_terrain_levels_round_trip_through_the_state_dict(parkour_env):
    env = parkour_env
    curriculum = env.curriculum_manager.get_term("terrain_level")
    command = env.command_manager.get_term("parkour_command")
    original = curriculum.state_dict()
    try:
        loaded = [1, 0] * (NUM_ENVS // 2)
        curriculum.load_state_dict(original | {"terrain_levels": torch.tensor(loaded)})
        assert curriculum.state_dict()["terrain_levels"].tolist() == loaded
        # The next reset is a first spawn: it places each env on its loaded level, whatever it walked before.
        assert curriculum.first_spawn.all()
        assert curriculum.compute()["mean_terrain_level"] == 0.5  # published before that reset, too
        env.reset()
        assert command.terrain_levels.tolist() == loaded

        # Saved under another env count: every env draws one of the saved levels.
        curriculum.load_state_dict(original | {"terrain_levels": torch.ones(NUM_ENVS + 3, dtype=torch.long)})
        assert curriculum.state_dict()["terrain_levels"].tolist() == [1] * NUM_ENVS
    finally:
        curriculum.load_state_dict(original)
        env.reset()


@pytest.mark.parametrize(
    ("state", "match"),
    [
        ({"num_configs": 99}, r"saved for a set of 99 .* over the 2 x 2 terrain grid bound this one to 4"),
        # Same cell count, another shape: the base's size check passes, the curriculum's row check does not.
        ({"num_levels": 10}, "10-row terrain grid"),
        ({"terrain_levels": torch.zeros(0, dtype=torch.long)}, "hold 0 levels"),
        ({"terrain_levels": torch.tensor([0, GRID_ROWS])}, r"levels \[0, 2\]"),
    ],
    ids=["other-set", "other-grid", "empty", "out-of-range"],
)
def test_a_terrain_level_state_from_elsewhere_is_refused(parkour_env, state, match):
    curriculum = parkour_env.curriculum_manager.get_term("terrain_level")
    before = curriculum.state_dict()["terrain_levels"]
    with pytest.raises(ValueError, match=match):
        curriculum.load_state_dict(curriculum.state_dict() | state)
    assert torch.equal(curriculum.state_dict()["terrain_levels"], before)


def test_goal_progress_reads_the_command_and_reaches_one_at_all_goals_reached(parkour_env):
    """Read from the command, so every --terrain_sampler logs it; 1 is the all_goals_reached termination."""
    env = parkour_env
    env.reset()
    command = env.command_manager.get_term("parkour_command")
    command.cur_goal_idx[0] = command.env_goals_count[0]  # env 0 is past its last goal
    env.metric_manager.compute()

    progress = env.metric_manager.get_value("parkour_goal_progress")
    assert progress[0] == 1.0 and (progress[1:] < 1.0).all()
    env.reset()


def test_cells_decode_to_the_level_and_type_they_were_numbered_from(parkour_env):
    command = parkour_env.command_manager.get_term("parkour_command")
    levels, types = torch.meshgrid(torch.arange(GRID_ROWS), torch.arange(GRID_COLS), indexing="ij")
    cells = command.cell_index(levels.flatten(), types.flatten())
    assert sorted(cells.tolist()) == list(range(command.num_cells))
    assert command.cell_level(cells).tolist() == levels.flatten().tolist()
    assert command.cell_type(cells).tolist() == types.flatten().tolist()


def test_a_course_with_no_goals_is_complete(parkour_env):
    """The all_goals_reached termination fired at once on such a cell before goal_progress; it still does."""
    env = parkour_env
    env.reset()
    command = env.command_manager.get_term("parkour_command")
    try:
        command.env_goals_count[0] = 0
        assert command.goal_progress[0] == 1.0
        assert command.goal_progress[1:].lt(1.0).all()
    finally:
        env.reset()


def test_a_level_curriculum_on_another_command_fails_the_build(small_teacher_config):
    """Equal cell counts would let it bind, and then follow the other command's walked distance."""
    cfg = small_teacher_config()
    cfg.command_options.other_parkour_command = cfg.command_options.parkour_command.model_copy()
    cfg.curriculum_options.terrain_level.command_name = "other_parkour_command"
    with pytest.raises(ValueError, match="command term 'other_parkour_command'.*command_name='parkour_command'"):
        build_env(cfg, reset=False)


# ---------------------------------------------------------------------------
# --terrain_sampler
# ---------------------------------------------------------------------------


def test_the_default_terrain_sampler_leaves_the_shipped_task_unchanged():
    assert TASK_REGISTRY.build("extreme_parkour_teacher").model_dump() == ExtremeParkourTeacherConfig().model_dump()


@pytest.mark.parametrize("task", ["extreme_parkour_teacher", "extreme_parkour_student"])
@pytest.mark.parametrize(
    ("terrain_sampler", "curricula", "sampler"),
    [("uniform", [], None), ("sgs", ["terrain_sampler"], "terrain_sampler")],
)
def test_a_terrain_sampler_replaces_the_level_curriculum(task, terrain_sampler, curricula, sampler):
    cfg = TASK_REGISTRY.build(task, terrain_sampler=terrain_sampler)
    assert list(cfg.curriculum_options.term_keys()) == curricula
    assert cfg.event_options.reset_parkour_root_state.sampler == sampler
    if terrain_sampler == "sgs":
        assert cfg.curriculum_options.terrain_sampler.success_term == "all_goals_reached"


@pytest.fixture(scope="module")
def sgs_parkour_env(small_teacher_config):
    return build_env(ExtremeParkourTerrainSamplerMod("sgs").apply(small_teacher_config()), reset=True)


def test_sgs_places_each_env_on_the_cell_it_drew(sgs_parkour_env):
    """The configuration set is every cell, row-major, so the sampler moves envs across columns too."""
    env = sgs_parkour_env
    env.reset()
    sampler = env.curriculum_manager.get_term("terrain_sampler")
    command = env.command_manager.get_term("parkour_command")

    assert sampler.num_configs == GRID_ROWS * GRID_COLS
    assert command.terrain_levels.tolist() == (sampler.assignment // GRID_COLS).tolist()
    assert command.terrain_types.tolist() == (sampler.assignment % GRID_COLS).tolist()
    origins = env.entities["terrain"].subterrain_origins[command.terrain_levels, command.terrain_types]
    expected_xy = origins[:, :2].clone()
    expected_xy[:, 0] -= _SPAWN_OFFSET_X
    torch.testing.assert_close(env.entities["robot"].get_pos()[:, :2], expected_xy, atol=1e-3, rtol=0)


def test_sgs_grades_an_episode_that_reached_every_goal_as_a_success_on_its_cell(sgs_parkour_env):
    env = sgs_parkour_env
    env.reset()
    sampler = env.curriculum_manager.get_term("terrain_sampler")
    command = env.command_manager.get_term("parkour_command")
    started_from = sampler.assignment.clone()
    counts_before = sampler.num_outcomes.clone()
    successes_before = (sampler.success_rate * counts_before).round()

    command.cur_goal_idx[0] = command.env_goals_count[0]  # env 0 is past its last goal: all_goals_reached
    env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device))

    ended = env.termination_manager.dones.clone()
    succeeded = env.termination_manager.get_term_dones("all_goals_reached").float()
    assert ended[0] and succeeded[0] == 1.0
    num_cells = GRID_ROWS * GRID_COLS
    counts = counts_before + torch.bincount(started_from[ended], minlength=num_cells)
    successes = successes_before + torch.bincount(started_from[ended], weights=succeeded[ended], minlength=num_cells)
    assert sampler.num_outcomes.tolist() == counts.tolist()
    torch.testing.assert_close(sampler.success_rate, successes / counts.clamp(min=1))
