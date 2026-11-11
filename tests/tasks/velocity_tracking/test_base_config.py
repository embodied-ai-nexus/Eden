"""The velocity-tracking tasks' TaskMods: the per-robot binding, the observation presets, and ``check_bindings``.

The robot binding lives in ``VelocityTrackingRobotMod``, not the config class, so these build real configs through
``TASK_REGISTRY`` (flat and terrain, G1 and Oli) and variants of them in a throwaway ``TaskRegistry``.
"""

from __future__ import annotations

import re

import pytest
import torch

import eden as en
from eden import actions, rewards
from eden.constants import ReferenceSource
from eden.envs.base import RLEnvBase
from eden.options.robots.limx_oli import Oli
from eden.options.robots.unitree_g1 import G1Dofs29
from eden.tasks import TASK_REGISTRY
from eden.tasks.registry import TaskRegistry
from eden.tasks.velocity_tracking.base_config import (
    _CRITIC_OBSERVATIONS,
    STD_RUNNING,
    STD_WALKING,
    VelocityTrackingObsMod,
    VelocityTrackingRobotMod,
)
from eden.tasks.velocity_tracking.flat.config import VelocityTrackingFlatConfig
from eden.utils.configs import ActionManagerOptions, RewardManagerOptions
from eden.utils.smoke import build_and_step
from eden.utils.string import name_matches

_TASKS = ("velocity_tracking_flat", "velocity_tracking_terrain")
_ROBOTS = sorted(VelocityTrackingRobotMod._PRESETS)
_G1_FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]


# ---------------------------------------------------------------------------
# VelocityTrackingRobotMod on the shipped tasks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", _TASKS)
def test_build_defaults_to_the_g1(task):
    cfg = TASK_REGISTRY.build(task)
    assert isinstance(cfg.scene_options.robot, G1Dofs29)
    assert cfg.action_options.dofs_pos_controller.dofs_name == G1Dofs29().dofs_name

    r, e = cfg.reward_options, cfg.event_options
    assert r.upright.link_name == r.body_ang_vel.link_name == "torso_link"
    assert r.foot_clearance.feet_link_names == r.foot_slip.feet_link_names == _G1_FEET
    assert r.foot_clearance.target_height > 0.0
    assert e.randomize_com.links_name == ["torso_link"]
    assert e.randomize_friction.links_name == _G1_FEET
    # Not a mod-managed slot: push_robot finds the free base joint by type.
    assert e.push_robot.base_joint_name is None


@pytest.mark.parametrize("task", _TASKS)
def test_build_with_oli_rewires_every_slot(task):
    """No G1 identifier may survive an ``--robot oli`` build."""
    cfg = TASK_REGISTRY.build(task, robot="oli")
    oli = Oli()

    assert isinstance(cfg.scene_options.robot, Oli)
    controller = cfg.action_options.dofs_pos_controller
    assert controller.dofs_name == oli.dofs_name
    assert len(controller.dofs_name) == 31
    assert set(controller.scale) == set(oli.default_dofs_kp)
    # Oli's URDF has no ``torso_link``; ``waist_pitch_link`` is the trunk.
    assert cfg.reward_options.upright.link_name == cfg.reward_options.body_ang_vel.link_name == "waist_pitch_link"
    assert cfg.event_options.randomize_com.links_name == ["waist_pitch_link"]
    assert cfg.event_options.push_robot.base_joint_name is None


@pytest.mark.parametrize("task", _TASKS)
def test_experiment_name_is_robot_prefixed(task):
    variant = task.removeprefix("velocity_tracking_")
    assert TASK_REGISTRY.build(task).runner_options.experiment_name == f"g1_velocity_tracking_{variant}"
    assert TASK_REGISTRY.build(task, robot="oli").runner_options.experiment_name == f"oli_velocity_tracking_{variant}"


def test_unknown_robot_raises():
    with pytest.raises(ValueError, match="robot"):
        TASK_REGISTRY.build("velocity_tracking_flat", robot="not_a_robot")


@pytest.mark.parametrize("robot_key", _ROBOTS)
def test_posture_std_maps_cover_every_joint(robot_key):
    """``resolve_per_dof_std`` raises at ``env.build()`` on an uncovered DOF; fail here, naming it."""
    dofs_name = VelocityTrackingRobotMod._PRESETS[robot_key]["robot_cls"]().dofs_name
    for std_map, label in ((STD_WALKING, "STD_WALKING"), (STD_RUNNING, "STD_RUNNING")):
        unmatched = [n for n in dofs_name if not any(re.match(p, n) for p in std_map)]
        assert not unmatched, f"{label} does not cover {robot_key} joints: {unmatched}"


def test_back_to_back_builds_do_not_alias():
    cfg_g1 = TASK_REGISTRY.build("velocity_tracking_flat")
    TASK_REGISTRY.build("velocity_tracking_flat", robot="oli")
    assert cfg_g1.reward_options.upright.link_name == "torso_link"
    assert cfg_g1.event_options.randomize_com.links_name == ["torso_link"]


def test_mod_copies_every_mutable_it_assigns():
    """Assignment on an options model is unvalidated, so pydantic does not copy for us."""
    cfg = TASK_REGISTRY.build("velocity_tracking_flat")
    r, e = cfg.reward_options, cfg.event_options

    assert cfg.action_options.dofs_pos_controller.scale is not en.robots.G1_ACTION_SCALE
    assert cfg.action_options.dofs_pos_controller.dofs_name is not cfg.scene_options.robot.dofs_name
    r.foot_clearance.feet_link_names.append("sentinel_link")
    assert "sentinel_link" not in r.foot_slip.feet_link_names
    assert "sentinel_link" not in e.randomize_friction.links_name


def test_terrain_variant_adds_the_height_scan_and_bounds_termination():
    cfg = TASK_REGISTRY.build("velocity_tracking_terrain", robot="oli")
    assert "height_scan" in (cfg.observation_options.policy.model_extra or {})
    assert "out_of_terrain_bounds" in cfg.termination_options.term_keys()


# ---------------------------------------------------------------------------
# VelocityTrackingObsMod: the policy observation recipe
# ---------------------------------------------------------------------------


def _policy_terms(cfg) -> list[str]:
    """Policy-group term names, in the order the observation manager will build them."""
    policy = cfg.observation_options.policy
    return policy.terms_order or [n for n in (policy.model_extra or {}) if not n.startswith("_option_")]


@pytest.mark.parametrize("task", _TASKS)
def test_default_policy_obs_is_proprioceptive(task):
    """The policy sees only what the robot can measure; base_lin_vel stays with the critic."""
    cfg = TASK_REGISTRY.build(task, robot="oli")
    policy, critic = cfg.observation_options.policy, cfg.observation_options.critic

    assert "base_lin_vel" not in _policy_terms(cfg)
    assert policy.history_length == 5
    assert policy.dofs_vel.scale == 0.05
    # The critic runs only in sim: it keeps the privileged channel and no history.
    assert "base_lin_vel" in (critic.model_extra or {})
    assert critic.history_length is None


@pytest.mark.parametrize("task", _TASKS)
def test_proprioceptive_preset_is_a_no_op(task):
    assert TASK_REGISTRY.build(task, robot="oli", obs_preset="proprioceptive").observation_options.dict() == (
        TASK_REGISTRY.build(task, robot="oli").observation_options.dict()
    )


@pytest.mark.parametrize("task", _TASKS)
def test_privileged_preset_restores_the_pre_split_recipe(task):
    """base_lin_vel leads (with the noise the task shipped), single frame, dofs_vel unscaled."""
    cfg = TASK_REGISTRY.build(task, robot="oli", obs_preset="privileged")
    policy, critic = cfg.observation_options.policy, cfg.observation_options.critic

    # Leading, not appended: a new pydantic extra lands last, so the mod must state terms_order.
    assert _policy_terms(cfg)[0] == "base_lin_vel"
    assert (policy.base_lin_vel.noise.n_min, policy.base_lin_vel.noise.n_max) == (-0.5, 0.5)
    assert policy.history_length is None
    assert policy.dofs_vel.scale == critic.dofs_vel.scale == 1.0


def test_privileged_preset_keeps_the_height_scan_last_on_terrain():
    cfg = TASK_REGISTRY.build("velocity_tracking_terrain", robot="oli", obs_preset="privileged")
    assert _policy_terms(cfg) == [
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "dofs_pos",
        "dofs_vel",
        "actions",
        "commands",
        "height_scan",
    ]


def test_privileged_preset_does_not_alias_the_critic_term():
    cfg = TASK_REGISTRY.build("velocity_tracking_flat", obs_preset="privileged")
    policy, critic = cfg.observation_options.policy, cfg.observation_options.critic

    policy.base_lin_vel.scale = 0.5
    assert critic.base_lin_vel.scale == 1.0
    assert _CRITIC_OBSERVATIONS["base_lin_vel"].scale == 1.0


def test_unknown_obs_preset_raises():
    with pytest.raises(ValueError, match="obs_preset"):
        TASK_REGISTRY.build("velocity_tracking_flat", obs_preset="not_a_preset")
    # Direct construction bypasses the registry's Literal check, so apply() guards too.
    with pytest.raises(ValueError, match="obs_preset"):
        VelocityTrackingObsMod(obs_preset="not_a_preset").apply(VelocityTrackingFlatConfig())


# ---------------------------------------------------------------------------
# Name couplings: variants rename, drop and split terms in their own class body
# ---------------------------------------------------------------------------

_LEGS = [r".*_hip_.*", r".*_knee_.*", r".*_ankle_.*"]
_UPPER = [r"waist_.*", r".*_shoulder_.*", r".*_elbow_.*", r".*_wrist_.*"]


def _registry_with(variant: type, name: str = "variant", **register_kwargs) -> TaskRegistry:
    """A throwaway registry holding ``velocity_tracking_flat`` (its real mods) and ``variant`` under ``name``.

    Registering on the global ``TASK_REGISTRY`` would leak the variant into every later test.
    """
    reg = TaskRegistry()
    reg.register(
        VelocityTrackingFlatConfig,
        name="velocity_tracking_flat",
        modifiers=TASK_REGISTRY.get_modifiers("velocity_tracking_flat"),
    )
    reg.register(variant, name=name, **register_kwargs)
    return reg


def _split_action_options(**upper_fields) -> ActionManagerOptions:
    return ActionManagerOptions(
        legs=actions.ImplicitPDController.configure(
            entity_name="robot", dofs_name=_LEGS, reference_source=ReferenceSource.DEFAULT
        ),
        upper=actions.ImplicitPDController.configure(
            entity_name="robot", dofs_name=_UPPER, reference_source=ReferenceSource.DEFAULT, **upper_fields
        ),
    )


class _SplitConfig(VelocityTrackingFlatConfig):
    """The issue's variant: the shipped controller replaced by two, nothing else."""

    action_options: ActionManagerOptions = _split_action_options()


def test_a_plain_subclass_inherits_the_robot_binding():
    """``class C(VelocityTrackingFlatConfig): pass`` must not build against the empty sentinels."""

    class _Plain(VelocityTrackingFlatConfig):
        pass

    reg = _registry_with(_Plain, name="g1_nomod")
    assert [type(m) for m in reg.get_modifiers("g1_nomod")] == [VelocityTrackingRobotMod, VelocityTrackingObsMod]
    cfg = reg.build("g1_nomod", robot="oli")
    assert cfg.reward_options.upright.link_name == "waist_pitch_link"
    cfg.check_bindings()


@pytest.mark.parametrize("robot", _ROBOTS)
def test_split_controllers_each_get_the_robots_scale_and_keep_their_joints(robot):
    """Both controllers are bound by type; the whole-body scale is matched per controller."""
    cfg = _registry_with(_SplitConfig).build("variant", robot=robot)
    preset_scale = VelocityTrackingRobotMod._PRESETS[robot]["action_scale"]
    for key, joints in (("legs", _LEGS), ("upper", _UPPER)):
        controller = getattr(cfg.action_options, key)
        assert controller.dofs_name == joints
        assert controller.scale == preset_scale
        assert controller.scale is not preset_scale
        assert controller.dict_keys_scope == "entity"
    cfg.check_bindings()


def test_encoder_bias_names_no_action_term():
    """The shipped event covers every position controller on the robot, so a split needs no copy of it."""
    cfg = TASK_REGISTRY.build("velocity_tracking_flat")
    assert cfg.event_options.encoder_bias.action_term_name is None


def test_authored_reward_values_are_kept():
    """Only unset sentinels are filled: an authored link or foot target survives the mod."""
    authored_rewards = VelocityTrackingFlatConfig.model_fields["reward_options"].default.clone()
    authored_rewards.upright.link_name = "pelvis"
    authored_rewards.foot_clearance.target_height = 0.07

    class _Authored(VelocityTrackingFlatConfig):
        reward_options: RewardManagerOptions = authored_rewards

    cfg = _registry_with(_Authored).build("variant")
    assert cfg.reward_options.upright.link_name == "pelvis"
    assert cfg.reward_options.foot_clearance.target_height == 0.07
    # ...while the slots it left alone are still bound.
    assert cfg.reward_options.body_ang_vel.link_name == "torso_link"
    assert cfg.reward_options.foot_clearance.feet_link_names == _G1_FEET


@pytest.mark.parametrize(
    "upper_fields, expected_scale",
    [({"scale": 0.1}, 0.1), ({"scale": 1.0}, 1.0), ({"scale_ratio": 0.25}, None)],
    ids=["authored_scale", "authored_unit_scale", "scale_ratio"],
)
def test_an_authored_controller_scale_is_kept(upper_fields, expected_scale):
    """``scale=1.0`` is a value, not the sentinel; ``scale_ratio`` wins at build, so no dead scale dict beside it."""

    class _Authored(VelocityTrackingFlatConfig):
        action_options: ActionManagerOptions = _split_action_options(**upper_fields)

    cfg = _registry_with(_Authored).build("variant")
    assert cfg.action_options.upper.scale == expected_scale
    assert cfg.action_options.upper.dict_keys_scope == "term"
    assert cfg.action_options.legs.scale == dict(en.robots.G1_ACTION_SCALE)
    cfg.check_bindings()


def test_terms_are_bound_by_type_whatever_their_key():
    """A variant may rename a robot-bound term or drop one; the mod neither misses nor trips over it."""
    base = VelocityTrackingFlatConfig.model_fields["reward_options"].default
    kept = {k: getattr(base, k).clone() for k in base.term_keys() if k not in ("foot_slip", "upright")}

    class _Renamed(VelocityTrackingFlatConfig):
        reward_options: RewardManagerOptions = RewardManagerOptions(
            **kept,
            stay_upright=rewards.Upright.configure(weight=1.0, entity_name="robot", link_name="", std=0.45),
        )

    cfg = _registry_with(_Renamed).build("variant", robot="oli")
    assert cfg.reward_options.stay_upright.link_name == "waist_pitch_link"
    assert "foot_slip" not in cfg.reward_options.term_keys()
    cfg.check_bindings()


def test_mod_raises_naming_itself_when_the_robot_has_no_position_controller():
    class _VelocityOnly(VelocityTrackingFlatConfig):
        action_options: ActionManagerOptions = ActionManagerOptions(
            vel=actions.ImplicitVelocityController.configure(entity_name="robot", dofs_name=[".*"]),
        )

    with pytest.raises(ValueError, match=r"(?s)VelocityTrackingRobotMod.*joint position controller.*'vel'"):
        _registry_with(_VelocityOnly).build("variant")


@pytest.mark.parametrize(
    "colliding, match",
    [
        (
            ActionManagerOptions(
                legs=actions.ImplicitPDController.configure(entity_name="robot", dofs_name=_LEGS),
                rest=actions.ImplicitPDController.configure(entity_name="robot"),
            ),
            r"(?s)\['rest'\] leave dofs_name empty.*2 terms that command DOFs",
        ),
        (
            ActionManagerOptions(
                dofs_pos_controller=actions.ImplicitPDController.configure(entity_name="robot"),
                arms=actions.NullJointAction.configure(entity_name="robot", dofs_name=[r".*_shoulder_.*"]),
            ),
            r"(?s)\['dofs_pos_controller'\] leave dofs_name empty.*2 terms that command DOFs.*'arms'",
        ),
    ],
    ids=["split_position_controllers", "any_dof_owning_term"],
)
def test_a_whole_robot_controller_beside_another_dof_owner_raises(colliding, match):
    """Counting every DOF-owning term, not just position controllers: a ``NullJointAction`` collides too."""

    class _Colliding(VelocityTrackingFlatConfig):
        action_options: ActionManagerOptions = colliding

    with pytest.raises(ValueError, match=match):
        _registry_with(_Colliding).build("variant")


def test_a_zero_dof_term_on_the_robot_does_not_trip_the_guard():
    """The rule is "will command DOFs", not "is on the robot": a welding term owns none."""

    class _Welded(VelocityTrackingFlatConfig):
        action_options: ActionManagerOptions = ActionManagerOptions(
            dofs_pos_controller=actions.ImplicitPDController.configure(entity_name="robot"),
            weld=actions.SuctionCupWelding.configure(entity_name="robot", ee_link_name="left_ankle_roll_link"),
        )

    cfg = _registry_with(_Welded).build("variant")
    assert cfg.action_options.dofs_pos_controller.dofs_name == list(en.robots.G1Dofs29().dofs_name)
    cfg.check_bindings()


def test_an_unbound_config_fails_at_env_construction_naming_the_mod():
    """The bare class reaches ``from_config`` and is refused there, before any Genesis work."""
    with pytest.raises(ValueError, match=r"(?s)VelocityTrackingRobotMod binds per robot.*upright\.link_name = ''"):
        RLEnvBase.from_config(VelocityTrackingFlatConfig())


def test_opting_out_of_the_mods_hits_the_named_check():
    reg = _registry_with(_SplitConfig, modifiers=())
    cfg = reg.build("variant")
    with pytest.raises(ValueError, match=r"(?s)action_options\.legs\.scale = None.*modifiers=\(\)"):
        cfg.check_bindings()


@pytest.mark.parametrize("task", _TASKS)
@pytest.mark.parametrize("robot", _ROBOTS)
def test_every_shipped_build_passes_the_binding_check(task, robot):
    TASK_REGISTRY.build(task, robot=robot).check_bindings()


@pytest.mark.parametrize("group", ["reward_options", "action_options"])
def test_the_binding_check_tolerates_a_dropped_manager_group(group):
    """Both groups are declared ``| None``; a dropped one must not become an AttributeError in ``from_config``."""
    cfg = TASK_REGISTRY.build("velocity_tracking_flat")
    setattr(cfg, group, None)
    cfg.check_bindings()


@pytest.mark.slow
@pytest.mark.requires_genesis
def test_split_controller_variant_builds_and_steps():
    """End to end on G1: two controllers, the preset scale per joint, the encoder bias on both."""
    cfg = _registry_with(_SplitConfig).build("variant")
    env = build_and_step(cfg, num_envs=2, num_steps=2)
    manager = env.action_manager
    assert manager.active_terms == ["legs", "upper"]
    assert manager.total_action_dim == 29

    for key in manager.active_terms:
        term = manager.get_term(key)
        expected = [
            next(v for k, v in reversed(en.robots.G1_ACTION_SCALE.items()) if name_matches(k, dof))
            for dof in term.dofs_name
        ]
        assert torch.allclose(term.effective_scale.cpu(), torch.tensor([expected], device="cpu"))

    encoder_bias = next(term for _mode, name, term in env.event_manager.iter_terms() if name == "encoder_bias")
    controllers = [manager.get_term("legs"), manager.get_term("upper")]
    # Private reads: the resolved targets and per-env offsets are the only observable form of the binding.
    assert encoder_bias._targets == controllers
    # The startup draw landed in each controller's own per-env offset, one column per controlled joint.
    for term in controllers:
        assert term._dofs_pos_offset.shape == (2, len(term.dofs_name))
        assert (term._dofs_pos_offset.abs() <= 0.015).all()
