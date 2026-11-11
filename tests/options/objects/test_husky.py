"""``HuskySkateboard``: the board HUSKY's policy rides, built with the values of the compiled training scene."""

from __future__ import annotations

import pytest

from eden.options import EnvOptions, SceneOptions
from eden.options.objects.husky import HuskySkateboard
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env


@pytest.fixture(scope="module")
def board():
    env = build_env(
        EdenConfig(env_options=EnvOptions(num_envs=1), scene_options=SceneOptions(skateboard=HuskySkateboard()))
    )
    return env.entities["skateboard"]


def test_the_board_articulates_as_tilt_then_trucks_and_wheels(board):
    """1 tilt + 2 trucks + 4 wheels, in MJCF declaration order."""
    assert board.dofs_name == [
        "board_tilt_joint",
        *("front_truck_roll_joint", "front_left_wheel_joint", "front_right_wheel_joint"),
        *("rear_truck_roll_joint", "rear_left_wheel_joint", "rear_right_wheel_joint"),
    ]


def test_the_wheels_roll_freely(board):
    """Genesis's 0.1 default armature would stop the wheels from rolling."""
    assert board.get_dofs_armature().abs().max().item() == 0.0


def test_the_centering_servos_match_the_training_scene(board):
    stiffness = dict(zip(board.dofs_name, board.get_dofs_stiffness()[0].tolist()))
    assert stiffness["board_tilt_joint"] == pytest.approx(14.67)
    assert stiffness["front_truck_roll_joint"] == stiffness["rear_truck_roll_joint"] == pytest.approx(10.0)
    assert all(stiffness[name] == 0.0 for name in board.dofs_name if name.endswith("_wheel_joint"))
