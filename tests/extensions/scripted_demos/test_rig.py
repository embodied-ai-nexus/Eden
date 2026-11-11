"""Reading an arm + two-finger gripper out of a scene: a composite rig, or a robot that carries its own fingers."""

from __future__ import annotations

# isort: off
import eden  # noqa: F401 - isolate the DDP rank before importing Torch
# isort: on

import pytest

import eden as en
from eden.extensions.scripted_demos.rig import find_gripper_rig, find_gripper_rigs, gripper_rig
from eden.options.entities import BoxOptions, SceneOptions
from eden.tasks import TASK_REGISTRY


def test_a_composite_rig_reads_its_arm_gripper_and_tool_point_through_the_weld():
    rig = gripper_rig(TASK_REGISTRY.build("robosuite_stack").scene_options, "robot")
    assert (rig.arm_name, rig.gripper_name, rig.ee_link) == ("robot", "robot_gripper", "attachment")
    assert rig.arm_dofs == tuple(f"joint{i}" for i in range(1, 8))
    assert rig.gripper_dofs == ("finger_joint1", "finger_joint2")
    assert rig.tcp_offset == pytest.approx((0.0, 0.0, 0.1034))


def test_a_robot_with_its_own_fingers_splits_its_dofs_into_arm_and_fingers():
    """SIMPLER's WidowX250: one entity, the fingers are ``open_dofs_pos``, the tool point is the robot's own."""
    rig = gripper_rig(TASK_REGISTRY.build("simpler_stack_green_cube_on_yellow_cube").scene_options, "robot")
    assert rig.arm_name == rig.gripper_name == "robot"
    assert rig.ee_link == "gripper_link"
    assert rig.arm_dofs == ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")
    assert rig.gripper_dofs == ("left_finger", "right_finger")
    assert rig.tcp_offset == pytest.approx((0.0935, 0.0, 0.0))


def test_a_robot_that_does_not_declare_its_tool_point_is_refused_naming_the_missing_declaration():
    """The Panda with its hand declares the finger targets but no tool point."""
    scene = SceneOptions(robot=en.robots.FrankaEmikaPanda())
    with pytest.raises(TypeError, match=r"FrankaEmikaPanda.*tcp_offset \(pad midpoint"):
        gripper_rig(scene, "robot")


def test_a_scene_entry_that_is_not_a_robot_is_refused():
    scene = SceneOptions(box=BoxOptions(size=(0.1, 0.1, 0.1)))
    with pytest.raises(TypeError, match=r"'box' is a BoxOptions"):
        gripper_rig(scene, "box")
    with pytest.raises(KeyError, match=r"no entry 'robot'.*\['box'\]"):
        gripper_rig(scene, "robot")


@pytest.mark.parametrize(
    ("task", "slot"), [("robosuite_stack", "robot"), ("simpler_stack_green_cube_on_yellow_cube", "robot")]
)
def test_the_one_rig_in_a_scene_is_found_without_naming_it(task, slot):
    assert find_gripper_rig(TASK_REGISTRY.build(task).scene_options).slot == slot


def test_every_rig_of_a_two_arm_scene_is_found_in_scene_order_and_the_one_rig_lookup_refuses_it():
    scene = TASK_REGISTRY.build("robosuite_two_arm_transport").scene_options
    assert [rig.slot for rig in find_gripper_rigs(scene)] == ["robot0", "robot1"]
    with pytest.raises(TypeError, match=r"found 2 \(\['robot0', 'robot1'\]\); pass the slot explicitly"):
        find_gripper_rig(scene)


def test_a_scene_without_a_rig_names_what_it_found():
    with pytest.raises(TypeError, match=r"exactly one arm \+ gripper in the scene, found 0"):
        find_gripper_rig(SceneOptions(box=BoxOptions(size=(0.1, 0.1, 0.1))))
