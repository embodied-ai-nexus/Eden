"""``DropFromHeight`` resets to a uniform draw of the soft joint limits, so it refuses a DOF with no finite range.

``sample_uniform`` over -/+inf is NaN, which the term would write straight into the sim; the guard names the
offending DOFs at build time instead. Tested against a real built getup scene (Go1).
"""

from __future__ import annotations

import pytest
import torch

from eden.envs.base import RLEnvBase
from eden.options.managers.events import EventTermOptions
from eden.tasks.mujoco_playground.getup.custom_terms import DropFromHeight

pytestmark = pytest.mark.requires_genesis


def _build_term(env, entity_name: str = "robot") -> DropFromHeight:
    term = DropFromHeight(env, EventTermOptions(entity_name=entity_name))
    term.build()
    return term


@pytest.fixture
def robot_limits(env):
    """The Go1's real soft limits, restored after a test that edits them."""
    robot = env.entities["robot"]
    original = robot.soft_dofs_pos_limits
    yield robot
    robot.soft_dofs_pos_limits = original


def test_the_shipped_go1_is_fully_bounded_and_resets_to_finite_joints(env):
    """Guards the entity/term build order too: the limits must be a tensor by the time the term builds."""
    robot = env.entities["robot"]
    assert torch.is_tensor(robot.soft_dofs_pos_limits)
    assert torch.isfinite(robot.soft_dofs_pos_limits).all()

    env.reset()
    assert torch.isfinite(robot.get_dofs_pos()).all()


@pytest.mark.parametrize(
    "env_row, dof, side",
    [(0, 1, slice(None)), (0, 2, 1), (1, 0, 0)],
    ids=["unbounded", "half_bounded", "unbounded_in_one_env_only"],
)
def test_a_dof_with_an_infinite_limit_is_named_at_build(env, robot_limits, env_row, dof, side):
    limits = robot_limits.soft_dofs_pos_limits.expand(env.num_envs, -1, -1).clone()
    limits[env_row, dof, side] = torch.inf
    robot_limits.soft_dofs_pos_limits = limits

    name = robot_limits.dofs_name[dof]
    with pytest.raises(ValueError, match=rf"DOFs \['{name}'\].*no range to sample"):
        _build_term(env)


def test_an_entity_without_dofs_is_named(env):
    """A DOF-less entity carries ``None`` limits; the guard names it rather than raising an AttributeError."""
    with pytest.raises(ValueError, match="needs a simulated articulated entity.*carries NoneType"):
        _build_term(env, entity_name="plane")


def test_an_unbuilt_entity_is_named(getup_config):
    """An unbuilt (or kinematic) entity still holds the mirrored options dict, not ``None``."""
    unbuilt = RLEnvBase.from_config(getup_config)
    with pytest.raises(ValueError, match="needs a simulated articulated entity.*carries dict"):
        _build_term(unbuilt)
