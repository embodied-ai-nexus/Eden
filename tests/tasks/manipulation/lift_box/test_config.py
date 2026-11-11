"""Tests for ``eden.tasks.manipulation.lift_box.config``: the ``manipulation_lift_box`` family's field-default factories
and gates.

Config-level only; no Genesis build.
"""

from __future__ import annotations

import pytest

from eden import rewards
from eden.options.robots import Droid, UR5eRobotiq, UR10eRobotiq
from eden.tasks import TASK_REGISTRY
from eden.tasks.registry import cli_fields
from eden.tasks.manipulation.lift_box.config import _lift_env_options, _lift_rewards
from eden.terms import RewardTerm

LIFT_TASKS = ("manipulation_lift_box", "manipulation_lift_box_rgb", "manipulation_lift_ycb_rgb")


def _a_reward_term():
    return RewardTerm.configure(func=rewards.action_rate_l2, weight=-0.5)


def test_lift_rewards_override_replaces_a_term_in_place():
    retargeted = _lift_rewards(lift_height=_a_reward_term())
    assert retargeted.lift_height.weight == -0.5
    assert set(retargeted.term_keys()) == set(_lift_rewards().term_keys())


def test_lift_rewards_add_introduces_a_new_term():
    extended = _lift_rewards(add={"extra_penalty": _a_reward_term()})
    assert set(extended.term_keys()) == set(_lift_rewards().term_keys()) | {"extra_penalty"}


def test_lift_rewards_rejects_a_misnamed_override_or_a_colliding_addition():
    """``RewardManagerOptions`` allows extras, so a typo would otherwise add a brand-new term."""
    with pytest.raises(ValueError, match=r"unknown reward term\(s\) \['lift_hieght'\]"):
        _lift_rewards(lift_hieght=_a_reward_term())
    with pytest.raises(ValueError, match="already defines"):
        _lift_rewards(add={"lift_height": _a_reward_term()})


def test_lift_env_options_varies_only_num_envs():
    default, small = _lift_env_options(), _lift_env_options(num_envs=8)
    assert (default.num_envs, small.num_envs) == (4096, 8)
    assert default.model_dump(exclude={"num_envs"}) == small.model_dump(exclude={"num_envs"})


@pytest.mark.parametrize("task_name", LIFT_TASKS)
def test_env_and_reward_overrides_survive_reconstruction(task_name):
    """A ``--config`` override of a factory-built field takes effect instead of being reverted by ``__init__``."""
    merged = TASK_REGISTRY.build(task_name).with_overrides_from_dict(
        {"env_options": {"num_envs": 8}, "reward_options": {"lift_height": {"weight": 99.0}}}
    )
    assert merged.env_options.num_envs == 8
    assert merged.reward_options.lift_height.weight == 99.0


def test_each_lift_task_has_its_own_experiment_name():
    """Distinct tasks must not share a ``logs/<experiment_name>`` tree."""
    names = {t: TASK_REGISTRY.build(t).runner_options.experiment_name for t in LIFT_TASKS}
    assert len(set(names.values())) == len(names), f"experiment_name collision: {names}"


@pytest.mark.parametrize("task_name", LIFT_TASKS)
def test_lift_gates_declare_exactly_one_rest_z_source(task_name):
    """The resolver is a function term with no ``build()``, so a bad pair would only die on the first step.

    Found by params rather than term name, so a renamed or newly added gate is covered too.
    """
    reward_options = TASK_REGISTRY.build(task_name).reward_options
    gates = {
        name: getattr(reward_options, name).params
        for name in reward_options.term_keys()
        if {"rest_z", "rest_z_metric"} & set(getattr(reward_options, name).params or {})
    }
    assert gates, f"{task_name}: no reward term references a resting height"
    for name, params in gates.items():
        sources = [key for key in ("rest_z", "rest_z_metric") if params.get(key) is not None]
        assert len(sources) == 1, f"{task_name}.{name}: expected exactly one resting-height source, got {sources}"


def test_lift_box_rest_z_is_where_the_cube_rests_on_the_table():
    cfg = TASK_REGISTRY.build("manipulation_lift_box")
    table, cube = cfg.scene_options.table, cfg.scene_options.cube
    table_top = table.default_root_pos[2] + table.size[2] / 2
    rest_z = cfg.reward_options.lift_height.params["rest_z"]
    assert rest_z == pytest.approx(table_top + cube.size[2] / 2)
    assert cube.default_root_pos[2] == pytest.approx(rest_z)


def _cli_flags(task: str) -> set[str]:
    return {mod.flag_for(field[0]) for mod in TASK_REGISTRY.get_modifiers(task) for field in cli_fields(type(mod))}


def test_only_lift_box_exposes_the_manipulator_mod():
    """The RGB variants' wrist camera is calibrated to the DROID flange: no ``--manipulator`` flag.

    Their other mods (``manipulation_lift_ycb_rgb``'s ``--object_properties``) are fine, so the flag is what is
    checked.
    """
    assert "manipulator" in _cli_flags("manipulation_lift_box")
    assert "manipulator" not in _cli_flags("manipulation_lift_box_rgb")
    assert "manipulator" not in _cli_flags("manipulation_lift_ycb_rgb")


def test_the_droid_manipulator_is_the_default():
    default, droid = (
        TASK_REGISTRY.build("manipulation_lift_box"),
        TASK_REGISTRY.build("manipulation_lift_box", manipulator="droid"),
    )
    assert isinstance(default.scene_options.robot, Droid)
    assert droid.model_dump() == default.model_dump()


@pytest.mark.parametrize("name, cls", [("ur5e_2f85", UR5eRobotiq), ("ur10e_2f85", UR10eRobotiq)])
def test_a_ur_manipulator_repoints_the_ik_and_keeps_the_gripper_controller(name, cls):
    config = TASK_REGISTRY.build("manipulation_lift_box", manipulator=name)
    robot = config.scene_options.robot
    assert isinstance(robot, cls)
    assert tuple(robot.arm.default_root_pos) == (0.0, 0.0, robot.base_pos[2])  # raised to table height
    ee_ik = config.action_options.ee_ik
    assert (ee_ik.ee_link_name, ee_ik.dofs_name) == ("ee_link", robot.arm_dofs)
    assert config.action_options.gripper.dofs_name == list(robot.gripper_dofs)  # same Robotiq on every arm
    assert config.runner_options.experiment_name == f"lift_box_{name}"


def test_the_reset_buffer_mod_leaves_lift_box_unchanged_by_default():
    assert (
        TASK_REGISTRY.build("manipulation_lift_box", reset_buffer=None).model_dump()
        == TASK_REGISTRY.build("manipulation_lift_box").model_dump()
    )


def test_a_reset_buffer_replaces_the_randomized_resets_with_one_restore():
    events = TASK_REGISTRY.build("manipulation_lift_box", reset_buffer="states.pt").event_options
    assert set(events.term_keys()) == {"stiffen_contact", "restore_reset_state"}
    assert events.restore_reset_state.path == "states.pt"
    assert events.restore_reset_state.sampler is None  # uniform rows
