"""The Extreme Parkour student config: what it adds to the teacher, the camera mount, and its terrain mix.

No env is built: the student's gs-madrona batch renderer compiles kernels for minutes on a first build. The depth
term's cadence and reset semantics are pinned in ``custom_terms/test_observations.py`` against a real env.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from genesis.utils.geom import quat_to_R

from eden.tasks import TASK_REGISTRY
from eden.tasks.extreme_parkour.student.config import _STUDENT_HORIZONTAL_SCALE, _make_student_grid
from eden.tasks.extreme_parkour.teacher.config import _COLS, _ROWS
from eden.utils.terrains.base import generate_terrain_fields

pytestmark = pytest.mark.requires_genesis


def test_config_wiring():
    cfg = TASK_REGISTRY.build("extreme_parkour_student")
    teacher = TASK_REGISTRY.build("extreme_parkour_teacher")

    # The owner (policy/ParkourProprio) must be declared first: the siblings find it at build.
    assert list(cfg.observation_options.term_keys()) == ["policy", "depth_camera", "delta_yaw_ok"]
    # Distilled from the teacher, whose actor architecture the student's runner rebuilds to load it.
    assert cfg.runner_options.class_name.endswith("rma.distillation:RmaDepthDistillationRunner")
    assert cfg.runner_options.actor == teacher.runner_options.actor
    depth_term = cfg.observation_options.depth_camera.depth
    assert cfg.runner_options.algorithm.update_interval == depth_term.update_interval
    assert cfg.runner_options.num_steps_per_env % depth_term.update_interval == 0
    assert cfg.reward_options.term_keys() == teacher.reward_options.term_keys()
    assert cfg.termination_options.term_keys() == teacher.termination_options.term_keys()

    # The source's fixed 20 ms action delay is one control step (ActionDelay counts physics steps).
    modifier = cfg.action_options.dofs_pos_controller.modifier
    assert modifier.name == "action_delay"
    assert tuple(modifier.delay_range_s) == (0.02, 0.02)
    assert modifier.delay_range_s[0] == pytest.approx(cfg.env_options.sim_dt * cfg.env_options.decimation)

    # Batch renderer in rasterizer mode (depth needs no lighting); camera-mount DR present.
    assert type(cfg.renderer_options).__name__ == "BatchRendererOptions"
    assert cfg.renderer_options.use_rasterizer is True
    assert "randomize_camera_mount" in cfg.event_options.term_keys()


def test_depth_camera_matches_the_source_sensor():
    """106x60 at an 87 deg *horizontal* FOV; Genesis takes the vertical one, so recover it from the aspect."""
    cam = TASK_REGISTRY.build("extreme_parkour_student").cameras_options.depth
    assert tuple(cam.cam_resolution) == (106, 60)
    width, height = cam.cam_resolution
    hfov = math.degrees(2 * math.atan(math.tan(math.radians(cam.cam_fov / 2)) * width / height))
    assert hfov == pytest.approx(87.0, abs=1e-6)
    assert (cam.attach_entity_name, cam.attach_link_name) == ("robot", "base")
    assert tuple(cam.attach_offset) == (0.33, 0.0, 0.08)

    # OpenGL camera looking +x pitched 20 deg down, image-right = -y.
    rot = quat_to_R(np.array(cam.attach_quat))
    pitch = math.radians(20.0)
    np.testing.assert_allclose(rot[:, 0], [0.0, -1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(-rot[:, 2], [math.cos(pitch), 0.0, -math.sin(pitch)], atol=1e-6)
    np.testing.assert_allclose(rot[:, 1], [math.sin(pitch), 0.0, math.cos(pitch)], atol=1e-6)


def test_student_grid_mix():
    """Per row: 4 gap | 4 hurdle | 1 flat (a hurdle with flat=True) | 4 step | 4 stones | 3 demo."""
    grid = _make_student_grid()
    assert len(grid) == _ROWS and all(len(row) == _COLS for row in grid)
    for row in grid:
        names = [g.name for g in row]
        assert names.count("parkour_gap_terrain") == 4
        assert names.count("parkour_hurdle_terrain") == 5
        assert sum(bool(getattr(g, "flat", False)) for g in row) == 1
        assert names.count("parkour_step_terrain") == 4
        assert names.count("parkour_terrain") == 4
        assert names.count("parkour_demo_terrain") == 3
        # Corridor courses narrowed for the camera; the stepping-stones course keeps its own range.
        for g in row:
            if g.name in ("parkour_gap_terrain", "parkour_hurdle_terrain", "parkour_step_terrain"):
                assert tuple(g.y_range) == (-0.1, 0.1)


def test_flat_and_demo_courses_generate_eight_goals_at_both_difficulty_extremes():
    """Columns 8 (flat) and 17 (demo), the courses the teacher grid does not have, at rows 0 and 9."""
    grid = _make_student_grid()
    small = [[grid[0][8], grid[0][17]], [grid[-1][8], grid[-1][17]]]
    fields = generate_terrain_fields(small, (16.0, 4.0), (2, 2), _STUDENT_HORIZONTAL_SCALE, 0.005, randomize=True)
    assert np.isfinite(fields.height_field).all()
    assert (fields.goals_count == 8).all()
