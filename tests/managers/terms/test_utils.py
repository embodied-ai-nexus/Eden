"""The shared term helpers in ``eden/managers/terms/utils.py``."""

from __future__ import annotations

import math

import pytest
import torch

from eden.managers.terms.utils import (
    gravity_vec,
    parallel_jaw_is_grasping,
    projected_gravity,
    resolve_link_idx,
    resolve_metric_term,
    resolve_per_dof_std,
    soft_dof_pos_violation,
)
from eden.options import BoxOptions, EnvOptions, SceneOptions
from eden.options.robots import FrankaHand
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env

pytestmark = pytest.mark.requires_genesis

ARM, FINGERS = 7, 2  # the Panda's DOFs: joint1..joint7, then finger_joint1..2


@pytest.mark.parametrize(
    ("std", "default", "expected"),
    [
        ({r"joint[1-4]": 0.3, r"joint[5-7]": 0.35, r"finger.*": 0.2}, None, [0.3] * 4 + [0.35] * 3 + [0.2] * 2),
        ({".*": 0.05, "joint2": 0.5}, None, [0.05, 0.5] + [0.05] * 7),  # a later pattern overrides
        # A pattern matching no DOF is tolerated: one map may carry entries for DOFs only some robots have.
        ({".*": 0.05, r".*wrist.*": 0.3}, None, [0.05] * 9),
        ({"joint1": 0.3}, 0.1, [0.3] + [0.1] * 8),  # an explicit default fills the rest without raising
    ],
    ids=["full-coverage", "later-overrides", "dead-pattern", "explicit-default"],
)
def test_resolve_per_dof_std_assigns_pattern_values(panda_env, std, default, expected):
    robot = panda_env.entities["robot"]
    torch.testing.assert_close(resolve_per_dof_std(std, robot, "cpu", default=default), torch.tensor(expected))


def test_resolve_per_dof_std_names_the_dofs_it_does_not_cover(panda_env):
    """A map written for another robot raises at build instead of training on a silent default."""
    robot = panda_env.entities["robot"]
    with pytest.raises(ValueError) as exc:
        resolve_per_dof_std({"joint.*": 0.3}, robot, "cpu", field="VariablePosture.std_walking")
    msg = str(exc.value)
    assert "VariablePosture.std_walking" in msg and f"{FINGERS} of {ARM + FINGERS}" in msg
    assert "finger_joint1" in msg and "joint.*" in msg  # the uncovered DOFs and the patterns tried
    assert "'.*'" in msg  # the catch-all fix
    with pytest.raises(ValueError, match=f"{ARM + FINGERS} of {ARM + FINGERS}"):
        resolve_per_dof_std({}, robot, "cpu")


def test_soft_dof_pos_violation_is_the_distance_outside_the_band():
    # Below-lower, at-lower, inside, at-upper, above-upper; limits (num_dofs, 2) broadcast over the batch.
    soft = torch.tensor([[-1.0, 1.0]] * 5)
    per_dof = soft_dof_pos_violation(torch.tensor([[-2.5, -1.0, 0.0, 1.0, 2.0]]), soft)
    torch.testing.assert_close(per_dof, torch.tensor([[1.5, 0.0, 0.0, 0.0, 1.0]]))
    assert soft_dof_pos_violation(torch.zeros(1, 5), soft).sum() == 0.0


def test_projected_gravity_expresses_down_in_the_body_frame():
    grav = gravity_vec(2, torch.device("cpu"))
    torch.testing.assert_close(grav, torch.tensor([[0.0, 0.0, -1.0]] * 2))
    h = math.pi / 4  # half of a 90 deg roll about x: the body's +y axis points up, so down is body -y
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0], [math.cos(h), math.sin(h), 0.0, 0.0]])
    torch.testing.assert_close(projected_gravity(quat, grav), torch.tensor([[0.0, 0.0, -1.0], [0.0, -1.0, 0.0]]))


def test_resolve_link_idx_returns_local_indices_in_the_order_asked(panda_env):
    """``build()`` resolves names, ``compute()`` indexes per-link readings: both are ``links``' order."""
    robot = panda_env.entities["robot"]
    by_name = {link.name: i for i, link in enumerate(robot.links)}
    for names in (["hand"], ["link1", "link2"], ["right_finger", "link0"]):  # the last one out of tree order
        idx = resolve_link_idx(robot, names, robot.device)
        assert idx.tolist() == [by_name[n] for n in names]


def test_resolve_link_idx_rejects_an_empty_selection(panda_env):
    """Unlabelled callers (the ``feet_link_names`` rewards) get the guard too; a label names the field."""
    robot = panda_env.entities["robot"]
    with pytest.raises(ValueError, match=r"links_name=\[\] matched no links on entity 'robot'") as exc:
        resolve_link_idx(robot, [], robot.device)
    assert "a penalty that reads zero for a whole training run" in str(exc.value)  # why an empty selection raises
    with pytest.raises(ValueError, match=r"FeetAirTime: feet_link_names=\[\] matched no links"):
        resolve_link_idx(robot, [], robot.device, term="FeetAirTime", field="feet_link_names")


def test_resolve_link_idx_rejects_an_entity_without_links():
    """A particle entity has no named links; it is refused by type, not with an ``AttributeError``.

    Constructed but never built: the check reads the entity's type and name only.
    """
    import genesis as gs

    from eden.envs.base import EnvBase
    from eden.options import BoxOptions, EnvOptions, SceneOptions
    from eden.options.materials import PBDParticleMaterialOptions
    from eden.utils.configs import EdenConfig

    pbd = gs.options.PBDOptions(lower_bound=(-1.0, -1.0, 0.0), upper_bound=(1.0, 1.0, 1.0))
    config = EdenConfig(
        env_options=EnvOptions(num_envs=1, pbd_options=pbd),
        scene_options=SceneOptions(
            goo=BoxOptions(
                size=(0.1, 0.1, 0.1), default_root_pos=(0.0, 0.0, 0.5), material=PBDParticleMaterialOptions()
            )
        ),
    )
    goo = EnvBase.from_config(config, show_viewer=False).entities["goo"]
    with pytest.raises(TypeError, match="entity 'goo' is a ParticleEntity, which has no named links"):
        resolve_link_idx(goo, [".*"], torch.device("cpu"))


_HAND_Z = 0.5


@pytest.fixture(scope="module")
def grasp():
    """``(hand, contacts, left_pad, right_pad)`` after the fingers have squeezed the cube for 60 steps."""
    env = build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=2),
            scene_options=SceneOptions(
                hand=FrankaHand(default_root_pos=(0.0, 0.0, _HAND_Z)),
                # Centred between the open pads, at the tool point (FrankaHand.tcp_offset).
                cube=BoxOptions(size=(0.03, 0.03, 0.03), fixed=False, default_root_pos=(0.0, 0.0, _HAND_Z + 0.1034)),
            ),
        )
    )
    hand = env.entities["hand"]
    for _ in range(60):
        hand.control_dofs_pos(torch.zeros(env.num_envs, 2, device=hand.device))
        env.scene.step()
    contacts = hand.get_contacts(exclude_self_contact=True)
    return hand, contacts, hand.get_link("left_gripper_pad"), hand.get_link("right_gripper_pad")


def _is_grasping(grasp, threshold, contacts=None, **kwargs):
    hand, all_contacts, left, right = grasp
    contacts = all_contacts if contacts is None else contacts
    return parallel_jaw_is_grasping(
        hand, contacts, left.idx, right.idx, [left.idx_local, right.idx_local], threshold, **kwargs
    )


def _pad_partner(contacts, pad_idx) -> int:
    """The one link a pad touches (the cube, in the ``grasp`` fixture)."""
    a, b, valid = contacts["link_a"], contacts["link_b"], contacts["valid_mask"]
    partners = torch.cat([b[(a == pad_idx) & valid], a[(b == pad_idx) & valid]]).unique()
    assert partners.numel() == 1, f"pad {pad_idx} touches links {partners.tolist()}, expected only the cube"
    return int(partners[0])


def test_parallel_jaw_closed_pads_on_an_object_are_a_grasp(grasp):
    """The per-env verdict broadcasts over a contact count (50 slots) that is not the env count."""
    assert _is_grasping(grasp, threshold=1.0).tolist() == [True, True]


def test_parallel_jaw_squeeze_below_the_force_threshold_is_not_a_grasp(grasp):
    assert _is_grasping(grasp, threshold=1.0e3).tolist() == [False, False]


def test_parallel_jaw_one_loaded_pad_is_not_a_grasp(grasp):
    """Drop the right pad's contacts from the real set: the left pad alone does not make a grasp."""
    _, contacts, _, right = grasp
    touches_right = (contacts["link_a"] == right.idx) | (contacts["link_b"] == right.idx)
    assert touches_right.any(), "the cube is not touching the right pad; the grasp fixture no longer grasps"
    left_only = {**contacts, "valid_mask": contacts["valid_mask"] & ~touches_right}
    assert _is_grasping(grasp, threshold=1.0, contacts=left_only).tolist() == [False, False]


def test_parallel_jaw_pads_on_two_different_objects_are_a_grasp_only_without_an_object_filter(grasp):
    """Relabel the right pad's contacts as touching a second object: each pad is loaded, but by a different
    object. Without ``object_link_idx`` the forces still combine into a grasp; naming the cube rejects it.
    """
    _, contacts, left, right = grasp
    cube = _pad_partner(contacts, left.idx)
    assert _is_grasping(grasp, threshold=1.0, object_link_idx=cube).tolist() == [True, True]
    other = cube + 1
    on_right = contacts["valid_mask"] & ((contacts["link_a"] == right.idx) | (contacts["link_b"] == right.idx))
    split = {
        **contacts,
        "link_a": torch.where(on_right & (contacts["link_a"] == cube), other, contacts["link_a"]),
        "link_b": torch.where(on_right & (contacts["link_b"] == cube), other, contacts["link_b"]),
    }
    assert _is_grasping(grasp, threshold=1.0, contacts=split).tolist() == [True, True]
    assert _is_grasping(grasp, threshold=1.0, contacts=split, object_link_idx=cube).tolist() == [False, False]


PULL_TERMS = ["metric_reward", "metric_observation", "on_metric_success", "metric_above"]


def _pull_term(key):
    from eden.managers.terms.observations.common import MetricObservation
    from eden.managers.terms.rewards.common import MetricReward
    from eden.managers.terms.terminations.common import MetricAbove, OnMetricSuccess

    return {
        "metric_reward": MetricReward,
        "metric_observation": MetricObservation,
        "on_metric_success": OnMetricSuccess,
        "metric_above": MetricAbove,
    }[key]


@pytest.mark.parametrize("key", PULL_TERMS)
def test_a_metric_pull_term_needs_a_metric_manager(panda_env, make_term, key):
    """Checked in build() (``resolve_metric_manager``), so a pull term fails at env construction."""
    assert panda_env.metric_manager is None  # a plain EnvBase with no metric terms
    with pytest.raises(ValueError, match="requires a metric term"):
        make_term(_pull_term(key), panda_env, metric_name="score")


@pytest.mark.parametrize("key", PULL_TERMS)
@pytest.mark.parametrize(
    ("metric_name", "error", "match"),
    [
        ("does_not_exist", KeyError, r"Active metrics: \['score'\]"),
        # "" is the class default: say the name is unset, not "'' not found".
        ("", ValueError, r"metric_name is not set .* Active metrics: \['score'\]"),
    ],
    ids=["unknown", "unset"],
)
def test_a_metric_pull_term_names_the_active_metrics(
    panda_env, make_term, score_metric, key, metric_name, error, match
):
    score_metric(panda_env)
    with pytest.raises(error, match=match):
        make_term(_pull_term(key), panda_env, metric_name=metric_name)


def test_a_metric_term_is_resolved_only_as_the_class_its_reader_expects(panda_env, score_metric):
    """A term that reads a task metric's own attributes resolves it by class, so a wrong metric_name fails in build()
    naming the metrics that would do, not with an AttributeError on the first step.
    """
    from eden.managers.metric_manager import MetricTerm, MetricTermFuncWrapper

    class InkMetric(MetricTerm):  # a metric class the env does not have
        pass

    score_metric(panda_env)
    found = resolve_metric_term(panda_env, term="Reader", metric_name="score", expected=MetricTermFuncWrapper)
    assert found is panda_env.metric_manager.get_term("score")
    with pytest.raises(
        TypeError,
        match=r"Reader reads a InkMetric metric, but metric 'score' is a MetricTermFuncWrapper\. "
        r"Set metric_name to one of the InkMetric metrics: \[\]",
    ):
        resolve_metric_term(panda_env, term="Reader", metric_name="score", expected=InkMetric)
