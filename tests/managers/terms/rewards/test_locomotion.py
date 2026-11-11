"""The locomotion reward terms in ``eden/managers/terms/rewards/locomotion.py``."""

from __future__ import annotations

import pytest
import math

import torch

from eden.managers.command_manager import CommandManager
from eden.managers.terms.commands.velocity import UniformVelocityCommand
from eden.managers.terms.observations.locomotion import base_height
from eden.envs.base import env_class_for_config
from eden.managers.terms.rewards.locomotion import (
    CloseFeetYawProjectedPenalty,
    FeetOrientationPenalty,
    StumblePenalty,
    base_height_l2,
    constant_base_height,
)
from eden.options import BoxOptions, EnvOptions, PlaneOptions, SceneOptions
from eden.options.robots.unitree_g1 import G1Dofs29
from eden.utils.configs import EdenRLConfig
from eden.options.managers.commands import CommandManagerOptions

pytestmark = pytest.mark.requires_genesis


def test_base_height_l2_squares_the_error_of_the_observed_height(sloped_env):
    """The penalty is the squared error of exactly the number the policy sees, in either framing."""
    for terrain_name in (None, "terrain"):
        observed = base_height(sloped_env, entity_name="box", terrain_entity_name=terrain_name)[:, 0]
        penalty = base_height_l2(sloped_env, entity_name="box", target_height=0.5, terrain_entity_name=terrain_name)
        torch.testing.assert_close(penalty, (observed - 0.5).square())
    world = base_height_l2(sloped_env, entity_name="box", target_height=0.5)
    torch.testing.assert_close(world, torch.full((2,), (2.0 - 0.5) ** 2))


def test_constant_base_height_is_base_height_l2_without_the_command_gate(sloped_env):
    for terrain_name in (None, "terrain"):
        alias = constant_base_height(sloped_env, entity_name="box", target_height=0.5, terrain_entity_name=terrain_name)
        full = base_height_l2(sloped_env, entity_name="box", target_height=0.5, terrain_entity_name=terrain_name)
        assert torch.equal(alias, full), terrain_name


def test_base_height_l2_amplifies_the_penalty_only_while_commanded_to_stand(sloped_env, install_manager):
    manager = install_manager(
        sloped_env,
        "command_manager",
        CommandManager(sloped_env, CommandManagerOptions(vel=UniformVelocityCommand.configure(entity_name="box"))),
    )
    manager.get_term("vel").vel_command_b[:] = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # stand, walk
    gated = base_height_l2(
        sloped_env,
        entity_name="box",
        target_height=1.0,
        command_name="vel",
        zero_vel_penalty_scale=5.0,
        zero_vel_threshold=0.1,
    )
    torch.testing.assert_close(gated, torch.tensor([5.0, 1.0]))  # (2.0 - 1.0)^2 = 1, x5 for the standing env


def test_feet_orientation_penalty_sums_each_resolved_links_tilt(sloped_env, make_term, restore_state):
    """``sin(tilt)`` per foot, summed; one ``.*`` pattern resolves to both of the probe's links."""
    term = make_term(FeetOrientationPenalty, sloped_env, entity_name="probe", feet_link_names=[".*"])
    torch.testing.assert_close(term.compute(), torch.zeros(2))  # upright

    probe = restore_state(sloped_env.entities["probe"])
    half = math.radians(30.0) / 2.0
    probe.set_quat(torch.tensor([[math.cos(half), math.sin(half), 0.0, 0.0]] * 2))
    torch.testing.assert_close(term.compute(), torch.full((2,), 2 * math.sin(math.radians(30.0))))


# ---------------------------------------------------------------------------
# Foot terms on a G1
# ---------------------------------------------------------------------------

G1_ENVS = 3
FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]
FOOT_HALF_SPAN = 0.1185  # each ankle-roll link's y offset from the base at the default pose
FOOT_DROP = 1.2508 - 2.0  # their z offset from the base


@pytest.fixture(scope="module")
def g1_env():
    """A G1 (``robot``) held 2 m over a plane, and a fixed wall whose face is at y = 2.9, at foot height."""
    config = EdenRLConfig(
        env_options=EnvOptions(num_envs=G1_ENVS),
        scene_options=SceneOptions(
            plane=PlaneOptions(),
            robot=G1Dofs29(default_root_pos=(0.0, 0.0, 2.0)),  # in the air
            wall=BoxOptions(size=(0.4, 0.2, 0.4), default_root_pos=(0.0, 3.0, 1.25), fixed=True),  # face at y = 2.9
        ),
    )
    built = env_class_for_config(config).from_config(config, show_viewer=False)
    built.build()
    built.reset()
    return built


@pytest.fixture
def robot(g1_env):
    """The G1, put back at its default pose after the test."""
    yield g1_env.entities["robot"]
    g1_env.reset()


def _feet_term(env, cls, **kwargs):
    term = cls(env=env, options=cls.configure(entity_name="robot", feet_link_names=FEET, **kwargs))
    term.build()
    return term


def test_stumble_fires_on_a_lateral_foot_contact_only(g1_env, robot):
    """Env 0 in the air, env 1 standing on the plane (vertical contact), env 2's left foot into the wall's side."""
    pos = robot.get_pos().clone()
    pos[1, 2] = -FOOT_DROP + 0.03
    pos[2, 1] = 2.9 - FOOT_HALF_SPAN + 0.04
    robot.set_pos(pos)
    g1_env.scene.step()

    # The wall contact is ~1850 N sideways against ~500 N vertical, the stance ~780 N vertical.
    stumble = _feet_term(g1_env, StumblePenalty, horizontal_to_vertical_ratio=2.0)

    assert stumble.compute().tolist() == [0.0, 0.0, 1.0]


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2], ids=["yaw-0", "yaw-90"])
@pytest.mark.parametrize("threshold, close", [(0.05, 0.0), (0.5, 1.0)])
def test_close_feet_measures_the_lateral_gap_in_the_bases_yaw_frame(g1_env, robot, yaw, threshold, close):
    """The default stance is 0.237 m wide. Yawed 90 degrees the feet line up along world x, where a
    world-frame y gap would read 0 and call them close.
    """
    robot.set_quat(torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]] * G1_ENVS))

    penalty = _feet_term(g1_env, CloseFeetYawProjectedPenalty, threshold=threshold)

    assert penalty.compute().tolist() == [close] * G1_ENVS


def test_close_feet_needs_exactly_two_feet(g1_env):
    term = CloseFeetYawProjectedPenalty(
        env=g1_env, options=CloseFeetYawProjectedPenalty.configure(entity_name="robot", feet_link_names=FEET[:1])
    )
    with pytest.raises(ValueError, match="expects exactly 2 feet, got 1"):
        term.build()
