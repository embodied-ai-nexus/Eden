"""The HUSKY task config and data bundles against the frozen policy contract."""

from __future__ import annotations

import fnmatch

import numpy as np

from eden.extensions.policies.husky.constants import ACTOR_OBS_DIM, HUSKY_G1_DOFS_SPEC, NUM_JOINTS, POLICY_JOINT_NAMES
from eden.options.managers.observations import ObservationTermOptions
from eden.tasks import TASK_REGISTRY
from eden.tasks.husky.custom_terms.commands import TRACKED_LINK_NAMES
from eden.tasks.husky.custom_terms.rewards import STEER_DOFS_POS
from eden.utils.assets import get_asset_path


def _declared_terms(group) -> list[str]:
    return [k for k, v in group.__pydantic_extra__.items() if isinstance(v, ObservationTermOptions)]


def test_ref_poses_cover_every_tracked_link_with_unit_quaternions():
    data = np.load(get_asset_path("objects/skateboard/husky_skate_ref_poses.npz"))
    names = [str(n) for n in data["link_names"]]
    assert len(TRACKED_LINK_NAMES) == 14
    assert set(TRACKED_LINK_NAMES) <= set(names)
    for key in ("push_pos", "steer_pos"):
        assert data[key].shape == (len(names), 3)
    for key in ("push_quat", "steer_quat"):
        assert data[key].shape == (len(names), 4)
        np.testing.assert_allclose(np.linalg.norm(data[key], axis=-1), 1.0, atol=1e-3)


def test_amp_motions_are_23_dofs_in_policy_order_at_50_fps():
    data = np.load(get_asset_path("objects/skateboard/husky_amp_push_motions.npz"))
    assert [str(n) for n in data["joint_names"]] == list(POLICY_JOINT_NAMES)
    assert float(data["fps"]) == 50.0
    for i in range(int(data["num_motions"])):
        assert data[f"motion_{i}"].shape[1] == NUM_JOINTS
    assert len(STEER_DOFS_POS) == NUM_JOINTS


def test_config_matches_the_frozen_policy_contract():
    cfg = TASK_REGISTRY.build("husky")

    # Actor obs = per-term 5-step history blocks over 8 terms -> 395 dims.
    policy = cfg.observation_options.policy
    terms = ["command", "heading", "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "phase"]
    assert _declared_terms(policy) == terms
    for name in terms:
        assert getattr(policy, name).history_length == 5, name
        assert getattr(policy, name).flatten_history_dim, name
    assert ACTOR_OBS_DIM == 395

    # AmpPPO slices the amp group by POSITION (window first, flag last), so the order is load-bearing.
    amp = cfg.observation_options.amp
    assert _declared_terms(amp) == ["joint_pos", "push"]
    assert amp.joint_pos.history_length == 5
    assert cfg.runner_options.obs_groups == {"actor": ["policy"], "critic": ["critic"]}
    # The source's AMP-augmented PPO, so `eden train --task husky` is the full recipe.
    assert cfg.runner_options.algorithm.class_name == "eden.learning.wrappers.rsl_rl.amp_ppo.ppo:AmpPPO"
    assert cfg.runner_options.algorithm.amp_obs_group == "amp"
    assert cfg.runner_options.algorithm.amp_num_frames == amp.joint_pos.history_length

    # The action scale is the source's 0.25 * effort / kp per joint.
    scale = cfg.action_options.dofs_pos_controller.scale
    specs = [
        next(s for p, s in HUSKY_G1_DOFS_SPEC.items() if fnmatch.fnmatch(joint, p)) for joint in POLICY_JOINT_NAMES
    ]
    expected = [0.25 * spec.EFFORT_LIMIT / spec.RAIBERT_HEURISTIC_KP() for spec in specs]
    np.testing.assert_allclose([scale[joint] for joint in POLICY_JOINT_NAMES], expected, rtol=1e-12)


def test_a_config_json_saved_by_the_old_husky_example_reloads_as_amp_ppo(tmp_path):
    from eden.learning.wrappers.rsl_rl.amp_ppo import RslRlAmpPpoAlgorithmOptions
    from eden.options.learning.rsl_rl import RslRlPpoAlgorithmOptions
    from eden.utils.configs import EdenConfig

    # What `husky_skate/main.py train` wrote before AmpPPO moved into eden/learning: the stock algorithm options
    # with only class_name pointing at the example's own module, and no amp_* field.
    cfg = TASK_REGISTRY.build("husky")
    fields = {
        k: v for k, v in cfg.runner_options.algorithm.dict().items() if k in RslRlPpoAlgorithmOptions.model_fields
    }
    old = RslRlPpoAlgorithmOptions(**{**fields, "class_name": "amp_training.ppo:AmpPPO", "gamma": 0.98})
    cfg.runner_options.algorithm = old
    cfg.save_as_json(str(tmp_path / "config.json"), lock_config=False)

    algorithm = EdenConfig.load_from_file(str(tmp_path / "config.json")).runner_options.algorithm
    assert type(algorithm) is RslRlAmpPpoAlgorithmOptions
    assert algorithm.class_name == "eden.learning.wrappers.rsl_rl.amp_ppo.ppo:AmpPPO"
    assert algorithm.amp_motion_file == "objects/skateboard/husky_amp_push_motions.npz"
    assert algorithm.gamma == 0.98  # the run's own PPO fields survive


def test_a_config_json_saved_with_plain_ppo_reloads_as_plain_ppo(tmp_path):
    from eden.options.learning.rsl_rl import RslRlPpoAlgorithmOptions
    from eden.utils.configs import EdenConfig

    # A husky run trained without AMP (the task's default before this config selected AmpPPO) replays as what it was.
    cfg = TASK_REGISTRY.build("husky")
    fields = {
        k: v for k, v in cfg.runner_options.algorithm.dict().items() if k in RslRlPpoAlgorithmOptions.model_fields
    }
    cfg.runner_options.algorithm = RslRlPpoAlgorithmOptions(**{**fields, "class_name": "PPO"})
    cfg.save_as_json(str(tmp_path / "config.json"), lock_config=False)

    algorithm = EdenConfig.load_from_file(str(tmp_path / "config.json")).runner_options.algorithm
    assert type(algorithm) is RslRlPpoAlgorithmOptions and algorithm.class_name == "PPO"
