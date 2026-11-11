"""The reorient task config: the ``--hand`` swap and the settings a training run depends on."""

import pytest

from eden.options.materials import KinematicMaterialOptions
from eden.tasks import TASK_REGISTRY


@pytest.mark.parametrize("side", ["right", "left"])
def test_hand_swap_reaches_every_side_dependent_value(side):
    """The port's patterns are side-agnostic, but three explicit name lists are not, and a missed one is silent."""
    cfg = TASK_REGISTRY.build("reorient", hand=side)
    robot = cfg.scene_options.robot
    assert robot.metadata.side == side

    assert cfg.action_options.joint_pos.dofs_name == list(robot.dofs_name)
    assert all(name.startswith(side) for name in cfg.action_options.joint_pos.dofs_name)
    # ``set_contact_sol_params`` matches link names literally, not by pattern.
    stiffened = cfg.event_options.stiffen_contact.params["entity_links"]["robot"]
    assert len(stiffened) == 5 and all(name.startswith(side) for name in stiffened)

    # The cube spawn leans toward the thumb and the goal marker hangs over it, so both flip with the hand.
    toward_thumb = 1.0 if side == "right" else -1.0
    assert cfg.event_options.reset_cube.pos_in_palm[1] * toward_thumb > 0
    assert cfg.command_options.reorient.goal_marker_in_palm[1] * toward_thumb > 0


def test_rejects_unknown_hand():
    with pytest.raises(ValueError, match="third"):
        TASK_REGISTRY.build("reorient", hand="third")


def test_the_goal_marker_wears_the_cube_asset_and_touches_no_physics():
    scene = TASK_REGISTRY.build("reorient").scene_options
    marker = scene.cube_goal
    assert marker.file == scene.cube.file
    assert marker.surface is None  # a surface would override the cube's face texture
    assert isinstance(marker.material, KinematicMaterialOptions)
    assert marker.collision is False


def test_no_runner_level_action_clip():
    """A runner clip truncates the raw action ``ActionRateL2Smooth`` reads, so the penalty goes blind outside ±1."""
    assert TASK_REGISTRY.build("reorient").runner_options.clip_actions is None
