"""ToolHang: robosuite's ``_check_frame_assembled`` and ``_check_tool_on_frame``, on objects posed into and out of the goal."""

from __future__ import annotations

# isort: off
import eden  # noqa: F401 - isolate the DDP rank before importing Torch
# isort: on

import math

import pytest
import torch

from eden.managers.base import build_term
from eden.managers.metric_manager import METRIC_TERM_REGISTRY
from eden.tasks import TASK_REGISTRY
from eden.tasks.robosuite.common import axis_angle_quat
from eden.tasks.robosuite.tool_hang.custom_terms import ToolHoleOnBar
from eden.utils.geom import transform_by_quat
from tests.support.envs import build_env

#: In the stand's root frame: the socket axis and the top of the base slab (``tool_hang_stand.xml``).
_SOCKET_XY = (0.0, 0.045)
_BASE_TOP_Z = -0.07
#: In the frame's root frame: the rod axis, its tip, and the hook bar's free end (``tool_hang_frame.xml``).
_ROD_X = 0.04375
_TIP_Z = -0.13445
_HANG_SITE = (-0.0475, 0.0, 0.08625)
_BAR_LENGTH = 0.09125
#: In the wrench's root frame: the centre of the ``hole1`` ring (``tool_hang_wrench.xml``).
_HOLE_X = -0.0935328


@pytest.fixture(scope="module")
def env():
    config = TASK_REGISTRY.build("robosuite_tool_hang")
    config.env_options.num_envs = 2
    return build_env(config, reset=True)


def _rows(env, values) -> torch.Tensor:
    return torch.tensor([values] * env.num_envs, dtype=torch.float32, device=env.device)


def _assembled_frame_pos(env) -> torch.Tensor:
    """Root position of an upright frame whose rod stands on the socket axis, its tip 1 mm above the base slab."""
    stand = env.entities["stand"].get_pos(relative=True)
    offset = _rows(env, (_SOCKET_XY[0] - _ROD_X, _SOCKET_XY[1], _BASE_TOP_Z + 0.001 - _TIP_Z))
    return stand + offset


def _pose(entity, pos: torch.Tensor, quat: torch.Tensor) -> None:
    entity.set_quat(quat, relative=True)
    entity.set_pos(pos, relative=True)


#: The wrench turned a quarter turn about +y: its body z (the ring's axis) along world x, the bar's direction.
_ACROSS_THE_BAR = axis_angle_quat(math.pi / 2, (0.0, 1.0, 0.0))
_FLAT = (1.0, 0.0, 0.0, 0.0)


def _pose_wrench(env, frame_pos: torch.Tensor, *, along: float, below: float, quat) -> None:
    """Put ``hole1``'s centre ``along`` of the way down the bar from its free end, ``below`` [m] under the bar axis."""
    hole = frame_pos + _rows(env, _HANG_SITE) + _rows(env, (along * _BAR_LENGTH, 0.0, -below))
    quat = _rows(env, quat)
    _pose(env.entities["tool"], hole - transform_by_quat(_rows(env, (_HOLE_X, 0.0, 0.0)), quat), quat)


@pytest.mark.requires_genesis
@pytest.mark.parametrize(
    "shift,assembled",
    [
        ((0.0, 0.0, 0.0), True),  # tip 1 mm over the base slab, rod on the socket axis
        ((0.0, 0.0, 0.03), False),  # lifted 3 cm: the tip is 5.8 cm from the base slab's centre
        ((0.01, 0.0, 0.0), False),  # 1 cm sideways: the rod is outside one pair of walls
        ((0.0, 0.006, 0.0), False),  # 6 mm along y: past wall 2's centre plane (5.5 mm off the axis)
    ],
)
def test_the_frame_counts_assembled_only_with_its_tip_down_between_both_wall_pairs(env, shift, assembled):
    env.reset(seed=0)
    _pose(env.entities["frame"], _assembled_frame_pos(env) + _rows(env, shift), _rows(env, (1.0, 0.0, 0.0, 0.0)))
    assert env.metric_manager.get_term("frame_assembled").compute().tolist() == [float(assembled)] * env.num_envs


@pytest.mark.requires_genesis
@pytest.mark.parametrize(
    "along,below,quat,on_bar",
    [
        (0.5, 0.006, _ACROSS_THE_BAR, True),  # hanging half way along, the ring's inner face on the bar
        (-0.1, 0.006, _ACROSS_THE_BAR, False),  # past the bar's free end: not threaded on
        (0.5, 0.010, _ACROSS_THE_BAR, False),  # 1 cm under the bar axis, more than the 6.75 mm the ring allows
        (0.5, 0.0, _FLAT, False),  # lying flat along the bar: centred on it, but the bar does not pass through
    ],
)
def test_the_ring_counts_on_the_bar_only_when_threaded_onto_it(env, along, below, quat, on_bar):
    env.reset(seed=1)
    frame_pos = _assembled_frame_pos(env)
    _pose(env.entities["frame"], frame_pos, _rows(env, (1.0, 0.0, 0.0, 0.0)))
    _pose_wrench(env, frame_pos, along=along, below=below, quat=quat)
    # The config uses ToolHoleOnBar only inside the success AND; build it on its own to read it alone.
    on = build_term(METRIC_TERM_REGISTRY, env, ToolHoleOnBar.configure(), label="ToolHoleOnBar").compute()
    assert on.tolist() == [float(on_bar)] * env.num_envs


@pytest.mark.requires_genesis
def test_a_wrench_hanging_on_the_assembled_frame_is_a_success(env):
    """The full ``_check_success``, end to end: its contact halves need the physics to run, so it steps.

    The arm is swung aside first — at its home pose the gripper reaches the hanging wrench, and any
    gripper contact with the wrench fails ``_check_tool_on_frame``.
    """
    env.reset(seed=2)
    arm = env.entities["robot"]
    base_yaw = arm.find_named_dofs([arm.dofs_name[0]])
    arm.set_dofs_pos(_rows(env, (1.2,)), dofs_idx_local=base_yaw.dofs_idx_local)
    frame_pos = _assembled_frame_pos(env)
    _pose(env.entities["frame"], frame_pos, _rows(env, (1.0, 0.0, 0.0, 0.0)))
    _pose_wrench(env, frame_pos, along=0.5, below=0.006, quat=_ACROSS_THE_BAR)

    success = []
    for _ in range(20):
        env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device))
        assert not env.reset_buf.any()
        success.append(env.metric_manager.get_value("success").tolist())
    assert success[-5:] == [[1.0] * env.num_envs] * 5, success
