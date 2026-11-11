"""The Weave G1-Inspire-HOI task: its data mod, and the task built on one and on two objects.

A single object builds a plain object entity and no per-object clip groups; two build a heterogeneous object whose
variant decides which clips an env may play. Both must reset the robot and the object exactly onto the reference.
"""

from __future__ import annotations

import pytest
import torch

from eden.datasets.weave_dataset import WEAVE_OBJECTS
from eden.options.entities import GroupedEntityOptions
from eden.tasks import TASK_REGISTRY
from tests.tasks.weave_hoi.conftest import build_weave_env

pytestmark = [pytest.mark.slow, pytest.mark.requires_genesis]

#: Observation group widths. Weave's are 310 / 315 / 221 / 153 / 273; the object groups here carry each hand's
#: base link too (its contact label five frames ahead, and its contact flag), 10 + 2 dims.
_GROUP_DIMS = {
    "ref_motion_body": 310,
    "ref_motion_object": 325,
    "object_state": 223,
    "robot_proprio": 153,
    "robot_privileged": 273,
}


# ── the data mod (no build) ────────────────────────────────────────────────────────────────────────


def test_an_unknown_object_is_refused_listing_the_choices():
    with pytest.raises(ValueError, match=r"--objects 'smallbox,mug' names unknown object\(s\) \['mug'\]"):
        TASK_REGISTRY.build("weave_hoi", objects="smallbox,mug")


def test_a_split_other_than_train_or_test_is_refused():
    with pytest.raises(ValueError, match="--split must be 'train' or 'test', got 'val'"):
        TASK_REGISTRY.build("weave_hoi", split="val")


def test_the_default_is_all_nine_objects_of_the_test_split_as_weaves_train_yaml(weave_root):
    """Weave's training config lists the nine test files; ``--split train`` is the paper's train split."""
    cfg = TASK_REGISTRY.build("weave_hoi", raw_data_path=weave_root)
    dataset = cfg.motion_options.default.dataset_options
    assert (dataset.split, dataset.objects) == ("test", list(WEAVE_OBJECTS))
    groups = cfg.motion_options.default.env_motion_groups
    assert groups.entity_name == "object" and groups.variant_labels == list(WEAVE_OBJECTS)
    assert isinstance(cfg.scene_options.object, GroupedEntityOptions)
    assert [e.align for e in cfg.scene_options.object.grouped_entities] == [False] * 9
    # largebox is Weave's light object; each variant carries its own density, the group none to fall back on.
    rhos = [e.material.rho for e in cfg.scene_options.object.grouped_entities]
    assert rhos == [25.0 if o == "largebox" else 200.0 for o in WEAVE_OBJECTS]
    assert cfg.scene_options.object.material is None


@pytest.mark.parametrize("objects", ["all", "smallbox"])
def test_the_robot_stands_in_weaves_posture_whatever_the_objects(objects):
    """The posture is the joint action's offset: the stock G1 one would shift every target a policy commands."""
    body = TASK_REGISTRY.build("weave_hoi", objects=objects).scene_options.robot.robot
    assert body.file == "humanoids/unitree_g1/g1_29dof_rev_1_0_bare_wrist.urdf"
    assert tuple(body.default_root_pos) == (0.0, 0.0, 0.76)
    assert body.default_dofs_pos["*_knee_joint"] == pytest.approx(0.669)
    assert body.default_dofs_pos["right_shoulder_roll_joint"] == pytest.approx(-0.2)


@pytest.mark.parametrize(
    "objects, experiment_name",
    [
        ("all", "weave_hoi"),
        (",".join(WEAVE_OBJECTS), "weave_hoi"),
        ("smallbox", "weave_hoi_smallbox"),
        ("tripod,smallbox", "weave_hoi_tripod-smallbox"),
    ],
)
def test_a_run_on_other_objects_logs_under_its_own_experiment(objects, experiment_name):
    """So ``--resume latest`` on a specialist never picks up the joint policy's run, nor the other way round."""
    assert TASK_REGISTRY.build("weave_hoi", objects=objects).runner_options.experiment_name == experiment_name


# ── built ──────────────────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def single(weave_root):
    return build_weave_env(weave_root, "smallbox", num_envs=2)


@pytest.fixture(scope="module")
def pair(weave_root):
    return build_weave_env(weave_root, "smallbox,largebox", num_envs=4)


def test_the_built_robot_rests_and_acts_around_weaves_posture(pair):
    robot = pair.entities["robot"]
    knee = robot.dofs_name.index("left_knee_joint")
    assert float(robot.default_dofs_pos[..., knee].flatten()[0]) == pytest.approx(0.669, abs=1e-6)


@pytest.mark.parametrize(
    ("name", "low", "high"),
    [("robot", 0.65, 1.3), ("left_hand", 0.65, 1.3), ("right_hand", 0.65, 1.3), ("object", 0.7, 1.1)],
)
def test_each_randomized_friction_is_the_one_its_ground_contacts_use(pair, name, low, high):
    """Genesis takes max(a, b) per contact pair: at Weave's 1.0 the ground overrode every sampled value below it."""
    ground = max(float(geom.get_friction()) for geom in pair.entities["plane"]._entity.geoms)
    entity = pair.entities[name]
    base = {float(geom.get_friction()) for geom in entity._entity.geoms}
    friction = entity.get_friction_ratio() * base.pop()
    assert not base  # one base coefficient for the whole entity, which the ratio scales
    assert low - 1e-6 <= float(friction.min()) and float(friction.max()) <= high + 1e-6
    assert ground < low  # so max(friction, ground) == friction for every draw


def test_one_object_builds_a_plain_object_and_no_clip_groups(single):
    term = next(iter(single.motion_manager._terms.values()))
    assert not single.entities["object"].is_heterogeneous
    assert term.env_motion_groups is None


@pytest.mark.parametrize("which", ["single", "pair"])
def test_the_policy_sees_weaves_five_groups_and_acts_on_41_dofs(which, request):
    env = request.getfixturevalue(which)
    obs, _ = env.reset()
    assert {group: tuple(v.shape) for group, v in obs.items()} == {g: (env.num_envs, d) for g, d in _GROUP_DIMS.items()}
    assert env.action_manager.total_action_dim == 41  # 29 body joints + 6 finger drivers per hand


@pytest.mark.parametrize("which", ["single", "pair"])
def test_reset_puts_the_pelvis_the_hands_and_the_object_on_the_reference(which, request):
    env = request.getfixturevalue(which)
    env.reset()
    data = env.motion_manager.data
    obj = env.entities["object"]
    torch.testing.assert_close(obj.get_pos(), data["object"].base_pos_w[:, 0], atol=1e-5, rtol=0)
    quat_dot = (obj.get_quat() * data["object"].base_quat_w[:, 0]).sum(-1).abs()
    torch.testing.assert_close(quat_dot, torch.ones_like(quat_dot), atol=1e-5, rtol=0)
    robot = env.entities["robot"]
    _, pelvis = robot.find_named_links_idx_local(["pelvis"], preserve_order=True)
    torch.testing.assert_close(
        robot.get_links_pos(ls_idx_local=pelvis)[:, 0], data["robot"].links_pos_w[:, 0, 0], atol=1e-5, rtol=0
    )
    # The reference strays a little past some finger stops; the reset clips to the soft limits, as Weave's does.
    for name in ("robot", "left_hand", "right_hand"):
        entity = env.entities[name]
        soft = entity.soft_dofs_pos_limits.expand(env.num_envs, -1, -1)
        expected = data[name].dofs_pos[:, 0].clamp(soft[..., 0], soft[..., 1])
        torch.testing.assert_close(entity.get_dofs_pos(), expected, atol=1e-5, rtol=0)


def test_every_env_plays_a_clip_of_the_object_it_holds_across_resets(pair):
    term = next(iter(pair.motion_manager._terms.values()))
    labels = term.data_source.motion_labels("objects")
    variants = pair.entities["object"].env_variant_idx.tolist()
    assert sorted(set(variants)) == [0, 1]
    for _ in range(3):
        pair.reset()
        clips = term._assigned_env_motion_selection.tolist()
        assert [labels[c] for c in clips] == [("smallbox", "largebox")[v] for v in variants]


def test_the_finger_followers_are_commanded_from_their_drivers(pair):
    """The software mimic: each hand term's follower targets are its driver targets times the ratio."""
    pair.reset()
    action = torch.rand(pair.num_envs, 41, device=pair.device) - 0.5
    pair.step(action)
    for name in ("left_hand_dofs_pos", "right_hand_dofs_pos"):
        term = pair.action_manager.get_term(name)
        processed = dict(zip(term.dofs_name, term.processed_action.unbind(-1), strict=True))
        for follower, (driver, mult, offset) in term.mimic_dofs.items():
            torch.testing.assert_close(processed[follower], processed[driver] * mult + offset)


def test_the_task_steps_with_every_term_computing(pair):
    pair.reset()
    for _ in range(3):
        step = pair.step(torch.zeros(pair.num_envs, 41, device=pair.device))
    rewards = step[1]
    assert torch.isfinite(rewards).all()
