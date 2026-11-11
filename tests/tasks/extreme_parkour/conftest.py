"""A real Extreme Parkour teacher env on a 2x2 terrain grid, shared by the teacher and observation tests."""

from __future__ import annotations

import pytest
import torch

from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY
from eden.options import ObservationGroupOptions
from eden.options.camera import CameraOptions, CamerasOptions
from eden.tasks.extreme_parkour.custom_terms.observations import (
    _DEPTH_RAW_RESOLUTION,
    ParkourDeltaYawOk,
    ParkourDepthImage,
    ParkourScanDots,
)
from eden.tasks.extreme_parkour.teacher.config import ExtremeParkourTeacherConfig, _make_parkour_grid
from tests.support.envs import build_env

NUM_ENVS = 4
GRID_ROWS, GRID_COLS = 2, 2


@OBSERVATION_TERM_REGISTRY.register()
class CountingDepthImage(ParkourDepthImage):
    """``ParkourDepthImage`` whose render is a counter: render #k is a frame filled with k.

    The renderer is the one boundary stubbed here: the student's gs-madrona batch renderer compiles for
    minutes, and the rasterizer renders one env. Everything else (cadence, slots, reset) is the real term.
    """

    def build(self) -> None:
        super().build()
        self.render_count = 0

    def _render_frame(self) -> torch.Tensor:
        self.render_count += 1
        h, w = self.resized_hw
        return torch.full((self.num_envs, h, w), float(self.render_count), device=self.device)


def _small_teacher_config() -> ExtremeParkourTeacherConfig:
    """The teacher on rows 0 (easiest) and 9 (hardest), columns [hurdle, flat], so the flat-cell mask is exercised.

    On top of the shipped task: the student's two extra groups (depth frame, delta-yaw gate), a
    ``ParkourScanDots`` probe that refreshes every step (the reference the rate-limited scan is compared to),
    a never-rendered depth camera, and no interval push (it would perturb the frames the tests compare).
    """
    cfg = ExtremeParkourTeacherConfig()
    cfg.env_options.num_envs = NUM_ENVS
    full_grid = _make_parkour_grid()
    cfg.scene_options.terrain.terrain_generators = [full_grid[0][1:3], full_grid[-1][1:3]]
    cfg.scene_options.terrain.default_root_pos = (-GRID_ROWS * 16.0 / 2.0, -GRID_COLS * 4.0 / 2.0, 0.0)
    cfg.cameras_options = CamerasOptions(depth=CameraOptions(cam_resolution=_DEPTH_RAW_RESOLUTION))
    cfg.event_options.disable_term("push_robot")
    groups = cfg.observation_options
    groups.depth_camera = ObservationGroupOptions(
        depth=CountingDepthImage.configure(update_interval=5), enable_corruption=False
    )
    groups.delta_yaw_ok = ObservationGroupOptions(ok=ParkourDeltaYawOk.configure(threshold=0.6))
    groups.scan_probe = ObservationGroupOptions(
        scan=ParkourScanDots.configure(entity_name="robot", terrain_entity_name="terrain", update_interval=1)
    )
    return cfg


@pytest.fixture(scope="session")
def small_teacher_config():
    """The factory behind ``parkour_env``, for a test that needs its own (e.g. unbuilt) env."""
    return _small_teacher_config


@pytest.fixture(scope="package")
def parkour_env():
    return build_env(_small_teacher_config(), reset=True)
