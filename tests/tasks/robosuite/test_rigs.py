"""The robosuite rigs and ``--robot`` (``eden/tasks/robosuite/rigs.py``).

Every ``robosuite_*`` task must stay swappable to any composite arm + two-finger gripper. That rests on a
convention (terms name the rig's *slot*, never its links or DOFs), so it is pinned over the whole registry.
"""

from __future__ import annotations

# isort: off
import eden  # noqa: F401 - isolate the DDP rank before importing Torch
# isort: on

import math

import pytest

from eden.options.managers.base import ManagerTermOptions
from eden.options.options import DynamicConfigurableOptions
from eden.options.robots import ManipulatorOptions
from eden.tasks import TASK_REGISTRY
from eden.tasks.robosuite.common import ROBOT_BASE_Z
from eden.tasks.robosuite.rigs import DEFAULT_ROBOT, PANDA_HOME, ROBOT_NAMES, make_rig, panda_rig, rig_name, rig_of

ROBOSUITE_TASKS = sorted(name for name in TASK_REGISTRY.list_tasks() if name.startswith("robosuite_"))
OTHER_ROBOTS = [name for name in ROBOT_NAMES if name != DEFAULT_ROBOT]


def _iter_term_options(node):
    """Yield every :class:`ManagerTermOptions` reachable from a manager options tree."""
    if isinstance(node, ManagerTermOptions):
        yield node
        for value in (node.__pydantic_extra__ or {}).values():
            yield from _iter_term_options(value)
        for value in (node.params or {}).values():
            yield from _iter_term_options(value)
    elif isinstance(node, DynamicConfigurableOptions):
        for value in (node.__pydantic_extra__ or {}).values():
            yield from _iter_term_options(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from _iter_term_options(value)


def _term_param_items(cfg):
    """Yield ``(term, key, value)`` for every parameter of every term of every manager."""
    managers = (
        cfg.observation_options,
        cfg.reward_options,
        cfg.termination_options,
        cfg.metric_options,
        cfg.curriculum_options,
        cfg.event_options,
        cfg.command_options,
    )
    for manager in managers:
        if manager is None:
            continue
        for term in _iter_term_options(manager):
            for key, value in (term.params or {}).items():
                yield term, key, value
            for key, value in (term.__pydantic_extra__ or {}).items():
                if not key.startswith("_option_") and not isinstance(value, (ManagerTermOptions, list, tuple)):
                    yield term, key, value


def _mentions(value, vocabulary: set[str]) -> bool:
    if isinstance(value, str):
        return value in vocabulary
    if isinstance(value, (list, tuple)):
        return any(_mentions(v, vocabulary) for v in value)
    if isinstance(value, dict):
        return any(_mentions(k, vocabulary) or _mentions(v, vocabulary) for k, v in value.items())
    return False


def _terms_carrying_rig_facts(cfg) -> list[tuple[str, str, object]]:
    """Every ``(term, key, value)`` of ``cfg`` that names a rig fact (link, DOF, gripper) instead of its slot."""
    vocabulary: set[str] = set()
    for slot in type(cfg).robot_slots:
        rig = rig_of(cfg, slot)
        # ``arm_name`` is deliberately absent: it is the slot, which is what a term is supposed to name.
        vocabulary |= {rig.ee_link, rig.gripper_name, *rig.grasp_links, *rig.gripper_dofs, *rig.arm_dofs}
    return [(term.name, key, value) for term, key, value in _term_param_items(cfg) if _mentions(value, vocabulary)]


# ── the rigs ─────────────────────────────────────────────────────────────────


def test_robot_names_are_the_registered_two_finger_rigs():
    assert DEFAULT_ROBOT in ROBOT_NAMES and {"droid", "ur5e_2f85", "ur10e_2f85"} <= set(ROBOT_NAMES)


@pytest.mark.parametrize("robot", ROBOT_NAMES)
def test_make_rig_names_the_rig_after_its_slot(robot):
    rig = make_rig(robot, "robot1", (0.0, 0.81, ROBOT_BASE_Z), (0.0, 0.0, 0.0, 1.0))
    assert (rig.arm_name, rig.gripper_name) == ("robot1", "robot1_gripper")
    assert tuple(rig.arm.default_root_pos) == (0.0, 0.81, ROBOT_BASE_Z)
    assert tuple(rig.arm.default_root_quat) == (0.0, 0.0, 0.0, 1.0)
    assert rig_name(rig) == robot
    assert rig.stroke > 0 and rig.tcp_offset[2] > 0.05  # the gripper declares its tool point


def test_make_rig_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="Unknown robosuite robot"):
        make_rig("not_a_rig", "robot", (0.0, 0.0, ROBOT_BASE_Z))


def test_panda_rig_is_robosuites_home_pose():
    rig = panda_rig("robot", (-0.56, 0.0, ROBOT_BASE_Z))
    assert rig_name(rig) == DEFAULT_ROBOT
    assert rig.arm.default_dofs_pos["joint4"] == pytest.approx(-math.pi / 2 - math.pi / 3)
    assert rig.arm.default_dofs_pos["joint6"] == pytest.approx(math.pi - 0.2)
    assert set(rig.arm_dofs) == set(PANDA_HOME)
    assert rig.gripper_open == {"finger_joint1": 0.04, "finger_joint2": 0.04}


def test_panda_gripper_is_robosuites_panda_gripper():
    """``panda_gripper.xml``: a pure position servo (kp 1000, no velocity gain) clipped to 20 N, the 100 damping on the joint."""
    gripper = panda_rig("robot", (0.0, 0.0, ROBOT_BASE_Z)).gripper
    fingers = ("finger_joint1", "finger_joint2")
    assert gripper.default_dofs_kp == dict.fromkeys(fingers, 1000.0)
    assert gripper.default_dofs_kd == dict.fromkeys(fingers, 0.0)
    assert gripper.default_dofs_damping == dict.fromkeys(fingers, 100.0)
    assert gripper.default_dofs_force_limits == dict.fromkeys(fingers, 20.0)


@pytest.mark.parametrize("robot", OTHER_ROBOTS)
def test_other_rigs_keep_their_registry_gripper(robot):
    rig = make_rig(robot, "robot", (0.0, 0.0, ROBOT_BASE_Z))
    assert rig.gripper.default_dofs_damping == type(rig.gripper)().default_dofs_damping


@pytest.mark.parametrize("robot", ROBOT_NAMES)
def test_hold_width_interpolates_between_open_and_closed(robot):
    rig = make_rig(robot, "robot", (0.0, 0.0, ROBOT_BASE_Z))
    assert rig.hold_width(rig.stroke, clearance=0.0) == rig.gripper_open
    assert rig.hold_width(0.0, clearance=0.0) == rig.gripper_close
    half_open = rig.hold_width(rig.stroke / 2, clearance=0.0)
    for dof in rig.gripper_dofs:
        assert half_open[dof] == pytest.approx((rig.gripper_open[dof] + rig.gripper_close[dof]) / 2)
    # Wider than the stroke clamps to open rather than extrapolating past the joint's range.
    assert rig.hold_width(2 * rig.stroke) == rig.gripper_open


def test_panda_hold_width_is_half_the_object_per_finger():
    rig = panda_rig("robot", (0.0, 0.0, ROBOT_BASE_Z))
    assert rig.hold_width(0.04, clearance=0.0) == {
        "finger_joint1": pytest.approx(0.02),
        "finger_joint2": pytest.approx(0.02),
    }
    rig.preset_hold(0.02)
    assert rig.gripper.default_dofs_pos["finger_joint1"] == pytest.approx(0.012)


def test_rig_of_rejects_a_rig_built_for_another_slot():
    """The terms find their rig by scene key, so the key and the rig's ``arm_name`` cannot disagree."""
    cfg = TASK_REGISTRY.build("robosuite_lift")
    cfg.scene_options.robot = make_rig(DEFAULT_ROBOT, "elsewhere", (0.0, 0.0, ROBOT_BASE_Z))
    with pytest.raises(ValueError, match="arm_name"):
        rig_of(cfg, "robot")


# ── --robot ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("task_name", ROBOSUITE_TASKS)
def test_no_mdp_term_carries_a_rig_fact(task_name):
    """A term that copies a link, DOF or entity name keeps pointing at the Panda after a swap."""
    carried = _terms_carrying_rig_facts(TASK_REGISTRY.build(task_name))
    assert not carried, f"{task_name}: MDP terms carry rig facts instead of the rig slot: {carried}"


@pytest.mark.parametrize("robot", OTHER_ROBOTS)
@pytest.mark.parametrize("task_name", ROBOSUITE_TASKS)
def test_robot_mod_swaps_the_scene_and_the_actions(task_name, robot):
    """After ``--robot <other rig>`` no action, weld or scene slot still points at the Panda."""
    cfg = TASK_REGISTRY.build(task_name, robot=robot)
    for slot in type(cfg).robot_slots:
        rig = getattr(cfg.scene_options, slot)
        assert isinstance(rig, ManipulatorOptions), f"{task_name}: slot {slot} still holds {type(rig).__name__}"
        assert rig_name(rig) == robot and rig.arm_name == slot
        # A composite is one scene entry that expands into arm + gripper entities at build time.
        assert dict(rig.entity_parts()).keys() == {slot, f"{slot}_gripper"}
        arm = getattr(cfg.action_options, f"{slot}_arm")
        assert (arm.entity_name, arm.ee_link_name, list(arm.dofs_name)) == (slot, rig.ee_link, list(rig.arm_dofs))
        gripper = getattr(cfg.action_options, f"{slot}_gripper", None)
        if gripper is not None:
            assert gripper.entity_name == rig.gripper_name
            assert list(gripper.dofs_name) == list(rig.gripper_dofs)
        # Anything welded onto a rig (PegInHole's peg / plate) must hang off the new rig's tool link.
        for child, spec in cfg.scene_options.attachments_dict.items():
            if spec[0] == slot:
                assert spec[1] == rig.ee_link, f"{task_name}: {child} still welded to link {spec[1]!r}"
    assert not _terms_carrying_rig_facts(cfg), f"{task_name} --robot {robot}: a term names the embodiment"
    assert cfg.runner_options.experiment_name.endswith(f"_{robot}")


@pytest.mark.parametrize("task_name", ROBOSUITE_TASKS)
def test_robot_mod_panda_is_a_no_op(task_name):
    plain = TASK_REGISTRY.build(task_name).model_dump(mode="json")
    explicit = TASK_REGISTRY.build(task_name, robot=DEFAULT_ROBOT).model_dump(mode="json")
    assert plain == explicit


def test_robot_mod_rejects_an_unknown_robot():
    with pytest.raises(ValueError):
        TASK_REGISTRY.build("robosuite_lift", robot="not_a_robot")


@pytest.mark.parametrize("robot", ROBOT_NAMES)
def test_robot_mod_fits_the_substep_to_the_gripper(robot):
    """The Franka Hand keeps robosuite's 5 ms / 2-substep step; a closed-loop gripper gets its ``max_substep_dt``.

    Genesis clamps a ``solref`` ``timeconst`` to ``2 * substep_dt``: above the Robotiq's 1 ms the four-bar
    softens and the gripper slides off what it holds while the scene still builds and steps.
    """
    rig = make_rig(robot, "robot", (0.0, 0.0, ROBOT_BASE_Z))
    options = TASK_REGISTRY.build("robosuite_lift", robot=robot).env_options
    if robot == DEFAULT_ROBOT:
        assert rig.max_substep_dt is None, "the Franka Hand is rigid; it needs no finer substep"
    if rig.max_substep_dt is None:
        assert (options.sim_dt, options.sim_substeps) == (0.005, 2)
    else:
        substep = options.sim_dt / options.sim_substeps
        assert substep <= rig.max_substep_dt + 1e-12, f"{robot}: substep {substep} exceeds {rig.max_substep_dt}"
