"""Every ``robosuite_*`` task config and scene, through the shared base in ``eden/tasks/robosuite/common.py``.

The config-level tests enumerate the registry, so a task added later is covered without editing this file.
"""

from __future__ import annotations

# isort: off
import eden  # noqa: F401 - isolate the DDP rank before importing Torch
# isort: on

import pytest
import torch

from eden.options.robots import PandaFrankaHand
from eden.tasks import TASK_REGISTRY
from eden.tasks.robosuite.common import ROBOT_BASE_Z, TABLE_TOP_Z, table_options
from eden.tasks.robosuite.rigs import PANDA_HOME, rig_of
from eden.utils.smoke import build_and_step

ROBOSUITE_TASKS = sorted(name for name in TASK_REGISTRY.list_tasks() if name.startswith("robosuite_"))


def test_the_port_registers_its_tasks():
    assert {"robosuite_lift", "robosuite_stack", "robosuite_two_arm_handover"} <= set(ROBOSUITE_TASKS)


@pytest.mark.parametrize("task_name", ROBOSUITE_TASKS)
def test_task_is_authored_around_the_panda_at_robosuites_home_pose(task_name):
    """The home pose must also survive ``TwoArmArrangementMod``, which re-places both arms on every build."""
    cfg = TASK_REGISTRY.build(task_name)
    for slot in type(cfg).robot_slots:
        rig = rig_of(cfg, slot)
        assert isinstance(rig, PandaFrankaHand), f"{task_name}: slot {slot} holds {type(rig).__name__}"
        assert tuple(rig.arm.default_root_pos)[2] == ROBOT_BASE_Z
        assert rig.arm.default_dofs_pos == pytest.approx(PANDA_HOME), f"{task_name}/{slot} is not at init_qpos"
        assert getattr(cfg.action_options, f"{slot}_arm").entity_name == slot
    # The whole action is the arm(s) + gripper(s): nothing else may sneak in.
    action_names = set(cfg.action_options.keys())
    expected = {f"{slot}_{part}" for slot in type(cfg).robot_slots for part in ("arm", "gripper")}
    assert action_names <= expected, f"{task_name}: unexpected action terms {action_names - expected}"


def test_parallel_arrangement_re_places_the_arms_at_their_home_pose():
    cfg = TASK_REGISTRY.build("robosuite_two_arm_lift", arrangement="parallel")
    rigs = [rig_of(cfg, slot) for slot in type(cfg).robot_slots]
    assert [rig.arm.default_dofs_pos for rig in rigs] == [pytest.approx(PANDA_HOME)] * 2
    assert tuple(rigs[0].arm.default_root_pos) != tuple(rigs[1].arm.default_root_pos)
    opposed = [rig_of(TASK_REGISTRY.build("robosuite_two_arm_lift"), slot) for slot in type(cfg).robot_slots]
    assert [tuple(r.arm.default_root_pos) for r in rigs] != [tuple(r.arm.default_root_pos) for r in opposed]


@pytest.mark.parametrize("task_name", ROBOSUITE_TASKS)
def test_object_pose_observations_read_the_authored_frame(task_name):
    """``<obj>_pos`` / ``<obj>_quat`` must be ``relative=True``, the frame every reward and event reads.

    Left at the solver-frame default they are off by the body's centre-of-mass offset (6.8 cm on the
    hammer, rotating with the object) with nothing failing loudly.
    """
    cfg = TASK_REGISTRY.build(task_name)
    offenders = []
    for group_name in cfg.observation_options.term_keys():
        group = getattr(cfg.observation_options, group_name)
        for key, term in (group.__pydantic_extra__ or {}).items():
            if (
                getattr(term, "name", None) in ("base_pos", "base_quat")
                and (term.params or {}).get("relative") is not True
            ):
                offenders.append(f"{group_name}.{key}")
    assert not offenders, (
        f"{task_name}: object-pose observations left in the solver frame: {offenders}. Build them with "
        f"common.object_pose_observations(), which passes relative=True."
    )


def test_table_top_is_where_robosuite_puts_it():
    table = table_options()
    assert table.default_root_pos[2] + table.size[2] / 2 == pytest.approx(TABLE_TOP_Z)


# ── built scenes ─────────────────────────────────────────────────────────────


@pytest.mark.requires_genesis
@pytest.mark.slow
@pytest.mark.parametrize("task_name", [t for t in ROBOSUITE_TASKS if t != "robosuite_two_arm_handover"])
def test_task_builds_and_steps(task_name):
    env = build_and_step(task_name, num_envs=2, num_steps=2)
    assert env.num_envs == 2


@pytest.mark.requires_genesis
@pytest.mark.slow
def test_two_arm_handover_builds_and_its_object_observables_agree():
    """``hammer_pos`` and ``eef_pos + gripper0_to_hammer`` are one point read through two terms.

    The hammer has the largest centre-of-mass offset, which is what makes the agreement meaningful: on an
    asset whose CoM sits at its origin either frame would pass.
    """
    env = build_and_step("robosuite_two_arm_handover", num_envs=2, num_steps=2)
    hammer = env.entities["hammer"]
    gap = (hammer.get_pos() - hammer.get_pos(relative=True)).norm(dim=-1)
    assert gap.min() > 0.05, f"the hammer no longer has a centre-of-mass offset ({gap.tolist()}) — pick another asset"

    obs = env.observation_manager
    hammer_pos = obs.get_term("hammer_pos").compute()
    reconstructed = obs.get_term("robot0_eef_pos").compute() + obs.get_term("gripper0_to_hammer").compute()
    torch.testing.assert_close(hammer_pos, reconstructed, atol=1e-4, rtol=0)
    torch.testing.assert_close(hammer_pos, hammer.get_pos(relative=True), atol=1e-6, rtol=0)
    # robosuite reports handle_xpos alongside hammer_pos; in hammer.xml they coincide, so must here.
    torch.testing.assert_close(obs.get_term("handle_pos").compute(), hammer_pos, atol=1e-6, rtol=0)


#: One build per variant axis: a config-level test cannot tell a variant scene that will not build (a rig
#: whose links the terms cannot resolve, a dropped object some term still names, re-placed welds).
_VARIANTS = [
    ("robosuite_lift", {"robot": "ur5e_2f85"}),  # six arm DOFs instead of seven, and a welded Robotiq
    ("robosuite_two_arm_lift", {"robot": "droid"}),  # the swap on two slots at once
    ("robosuite_door", {"latch": "no_latch"}),  # a different door asset, one DOF fewer, a reward term dropped
    ("robosuite_nut_assembly", {"nut": "square"}),  # drop_object: one nut and every term naming it
    ("robosuite_pick_place", {"object": "can"}),
    ("robosuite_two_arm_peg_in_hole", {"arrangement": "parallel"}),  # re-placed arms, re-welded objects
]


@pytest.mark.requires_genesis
@pytest.mark.slow
@pytest.mark.parametrize(
    "task_name,modifiers",
    _VARIANTS,
    ids=[f"{task}-{','.join(f'{k}={v}' for k, v in mods.items())}" for task, mods in _VARIANTS],
)
def test_each_variant_axis_builds_and_steps(task_name, modifiers):
    env = build_and_step(TASK_REGISTRY.build(task_name, **modifiers), num_envs=2, num_steps=2)
    assert env.num_envs == 2
