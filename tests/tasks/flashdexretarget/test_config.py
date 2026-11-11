"""The ``flashdexretarget`` task built on a two-object motion file: frames, resets, object variants, actions, assist."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch

from eden.extensions.retargeting.flashdexretarget.clips import FINGERS
from eden.extensions.retargeting.flashdexretarget.scene import robot_xml
from eden.tasks import TASK_REGISTRY
from eden.tasks.flashdexretarget.assets import motion_summary
from tests.tasks.flashdexretarget.conftest import build_task

pytestmark = [pytest.mark.slow, pytest.mark.requires_genesis]


def test_the_task_refuses_to_build_without_a_motion_file():
    with pytest.raises(ValueError, match="pass --motion <dataset>/<robot>/motion.pt"):
        TASK_REGISTRY.build("flashdexretarget")


def test_flashsac_learner_takes_flashdexretargets_hyperparameters(motion_file):
    from eden.learning.wrappers.rsl_rl.flashsac import RslRlFlashSacRunnerOptions

    cfg = TASK_REGISTRY.build("flashdexretarget", motion=str(motion_file), learner="flashsac")
    runner = cfg.runner_options
    assert isinstance(runner, RslRlFlashSacRunnerOptions)
    alg = runner.algorithm
    assert (alg.gamma, alg.n_steps, alg.num_mini_batches, alg.mini_batch_size) == (0.98, 5, 2, 4096)
    assert (runner.critic.hidden_dim, runner.actor.hidden_dim) == (1024, 128)
    # 500M env steps at one step per iteration, sized for the env count the run is built with (--num_envs is
    # applied after the task's modifiers).
    assert runner.num_env_steps == 500_000_000
    assert runner.for_num_envs(1024, world_size=1).max_iterations == 500_000_000 // 1024
    assert alg.learning_rate_decay_steps is None  # FlashSAC spans the decay over the sized run
    assert runner.obs_groups == {"actor": ["policy"], "critic": ["policy"]}
    assert cfg.env_options.record_final_observations  # FlashSAC bootstraps time-outs from them


def test_only_flashsac_records_the_final_observations(motion_file):
    """PPO never reads them, and recording them computes every observation term twice a step. Built after a FlashSAC
    config, so a shared default the FlashSAC build had changed would show here too.
    """
    flashsac = TASK_REGISTRY.build("flashdexretarget", motion=str(motion_file), learner="flashsac")
    ppo = TASK_REGISTRY.build("flashdexretarget", motion=str(motion_file))
    assert flashsac.env_options.record_final_observations
    assert not ppo.env_options.record_final_observations


def test_a_flashsac_runner_swapped_in_without_the_learner_flag_is_pointed_at_it(motion_file):
    """``runner_options`` switched to FlashSAC by hand (an ``--override``) leaves PPO's final observations off; the
    task names ``--learner flashsac`` before any env is built, rather than the wrapper's generic advice.
    """
    from eden.learning.wrappers.rsl_rl.flashsac.runner import FlashSacVecEnvWrapper
    from eden.tasks.flashdexretarget.config import flashsac_runner_options

    cfg = TASK_REGISTRY.build("flashdexretarget", motion=str(motion_file))
    cfg.runner_options = flashsac_runner_options()
    with pytest.raises(ValueError, match="Select FlashSAC with `--learner flashsac`"):
        FlashSacVecEnvWrapper.from_config(cfg)
    TASK_REGISTRY.build("flashdexretarget", motion=str(motion_file), learner="flashsac").check_bindings()


def test_the_experiment_is_named_after_the_dataset_the_robot_and_the_learner(motion_file):
    """``<dataset>/<robot>/motion.pt``: two datasets retargeted to the same robot get different names."""
    dataset = motion_file.parent.parent.name
    for learner in ("ppo", "flashsac"):
        cfg = TASK_REGISTRY.build("flashdexretarget", motion=str(motion_file), learner=learner)
        assert cfg.runner_options.experiment_name == f"flashdexretarget_{dataset}_xhand_{learner}"


def test_a_flashsac_iteration_override_must_clear_the_env_step_length_it_replaces(motion_file):
    """The run is sized in env steps, so ``runner_options.max_iterations`` alone would be ignored: it is refused,
    and the override the error names sets the iterations.
    """
    cfg = TASK_REGISTRY.build("flashdexretarget", motion=str(motion_file), learner="flashsac")
    with pytest.raises(ValueError, match=r"num_env_steps: null"):
        cfg.with_overrides_from_dict({"runner_options": {"max_iterations": 1000}})
    cfg = cfg.with_overrides_from_dict({"runner_options": {"max_iterations": 1000, "num_env_steps": None}})
    assert cfg.runner_options.for_num_envs(4096, world_size=1).max_iterations == 1000


def test_the_robot_is_the_one_the_motion_was_retargeted_to(motion_file):
    """motion.pt names its robot: --robot may be left out, and another one is refused rather than read with the wrong
    joint columns.
    """
    cfg = TASK_REGISTRY.build("flashdexretarget", motion=str(motion_file))
    assert cfg.motion_options.default.dataset_options.robot == "xhand"
    with pytest.raises(ValueError, match=r"--robot sharpa but .*motion\.pt was retargeted to xhand"):
        TASK_REGISTRY.build("flashdexretarget", motion=str(motion_file), robot="sharpa")


def test_an_unknown_learner_is_refused(motion_file):
    with pytest.raises(ValueError, match="--learner must be 'ppo' or 'flashsac', got 'sac'"):
        TASK_REGISTRY.build("flashdexretarget", motion=str(motion_file), learner="sac")


@pytest.fixture(scope="module")
def env(motion_file):
    return build_task(motion_file)


@pytest.fixture(scope="module")
def state(env):
    return env.reward_manager._terms["wrist_trans"].state  # the shared reader every hand term builds


def test_palm_and_fingertips_are_the_mjcf_sites_the_retargeting_solved_for(env, state):
    """The URDF's own fingertip frames sit up to 10 mm off the MJCF sites; the task must read the sites."""
    hand = env.action_manager.get_term("hand")
    q = env.entities["robot"].get_dofs_pos(dofs_idx_local=hand.genesis_dofs_idx)[0].cpu().numpy()
    m = mujoco.MjModel.from_xml_path(str(robot_xml("xhand", ("right",))))
    d = mujoco.MjData(m)
    for j in range(m.njnt):
        d.qpos[m.jnt_qposadr[j]] = q[hand.dofs_name.index(m.joint(j).name)]
    mujoco.mj_kinematics(m, d)
    palm_pos, palm_quat = state.palm_pose()
    site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
    site_quat = np.zeros(4)
    mujoco.mju_mat2Quat(site_quat, d.site_xmat[site])
    np.testing.assert_allclose(palm_pos[0, 0].cpu().numpy(), d.site_xpos[site], atol=1e-5)
    assert abs(np.dot(palm_quat[0, 0].cpu().numpy(), site_quat)) == pytest.approx(1.0, abs=1e-5)
    tips = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"track_hand_right_{f}_tip") for f in FINGERS]
    np.testing.assert_allclose(state.tips_pos()[0, 0].cpu().numpy(), d.site_xpos[tips], atol=1e-5)


def test_reset_puts_the_hand_and_the_object_exactly_on_the_reference(env, state):
    hand = env.action_manager.get_term("hand")
    q = env.entities["robot"].get_dofs_pos(dofs_idx_local=hand.genesis_dofs_idx)
    torch.testing.assert_close(q, hand._motion_ref_offset(), atol=1e-5, rtol=0)
    pos, quat = state.object_pose()
    ref = state.ref_object(0)
    torch.testing.assert_close(pos, ref["pos"], atol=1e-5, rtol=0)
    torch.testing.assert_close(quat.abs(), ref["quat"].abs(), atol=1e-5, rtol=0)


def test_each_env_plays_only_clips_of_the_object_it_holds(env, motion_file):
    labels = motion_summary(str(motion_file)).slot_labels
    obj = env.entities["object_right"]
    assert obj.num_variants == 2
    term = env.motion_manager._terms["default"]
    clip = term._assigned_env_motion_selection  # the clip each env plays: the only form of the assignment
    clip_labels = [term.data_source.motion_labels("objects")[int(i)] for i in term._local_to_global[clip]]
    assert clip_labels == [labels[int(v)] for v in obj.env_variant_idx]


def test_a_zero_action_holds_the_wrist_on_the_reference_and_opens_the_fingers_to_mid_range(env):
    hand = env.action_manager.get_term("hand")
    env.step(torch.zeros(env.num_envs, hand.action_dim))
    target, ref = hand.processed_action, hand.reference_used
    wrist = hand._residual
    torch.testing.assert_close(target[:, wrist], ref[:, wrist])
    mid = 0.5 * (hand._lower + hand._upper)
    torch.testing.assert_close(target[:, ~wrist], mid[~wrist].expand_as(target[:, ~wrist]))


def test_the_assist_pulls_a_displaced_object_back_toward_its_reference(motion_file):
    """Released 20 cm up and 5 cm aside, the object falls freely without the assist (keeping its 5 cm), and the
    assist's PD at its starting gains pulls it sideways back under 2 cm within 10 steps (still in the air).
    """
    no_resets = {
        # From the clip's first frame, so the clip outlasts the test, and no early reset on the poor tracking.
        "motion_options.default.motion_start_from_middle_range": (0.0, 0.0),
        "termination_options.task_reward_early_reset.params": {
            "reward_term": "object_pcd",
            "per_step_threshold": -1.0,
            "check_interval": 5,
        },
    }
    offset = {}
    for kp in (80.0, 0.0):
        env = build_task(motion_file, num_envs=1, **no_resets)
        env.action_manager.get_term("assist").set_gains(
            torch.full((1,), kp), torch.full((1,), 5.0 if kp else 0.0), 50.0
        )
        obj = env.entities["object_right"]
        obj.set_pos(obj.get_pos() + torch.tensor([0.05, 0.0, 0.2]))
        state = env.reward_manager._terms["wrist_trans"].state
        hand = env.action_manager.get_term("hand")
        for _ in range(10):
            env.step(torch.zeros(1, hand.action_dim))
        offset[kp] = float((obj.get_pos() - state.ref_object(0)["pos"][:, 0])[0, 0])
    assert abs(offset[80.0]) < 0.02
    assert offset[0.0] == pytest.approx(0.05, abs=0.005)


def test_the_fused_observations_equal_the_flashdexretarget_pairs_they_replace(env):
    group = env.observation_manager
    contact = group.get_term("robot_body_contact_wrist")
    torch.testing.assert_close(
        contact.compute(),
        torch.cat(
            [
                contact._robot_body_contact_wrench_wrist().flatten(1),
                contact._robot_body_contact_trans_wrist().flatten(1),
            ],
            dim=-1,
        ),
    )
    sdf = group.get_term("robot_links_obj_sdf_normal_wrist")
    torch.testing.assert_close(
        sdf.compute(),
        torch.cat([sdf._robot_links_obj_sdf().flatten(1), sdf._robot_links_obj_normal_wrist().flatten(1)], dim=-1),
    )


def test_every_term_shares_one_copy_of_the_object_geometry(env):
    """The SDF grids are 33.5 MB per object slot; a copy per term (15 of them) would not fit a large dataset."""
    readers = [env.observation_manager.get_term("obj_point_cloud_wrist"), env.reward_manager._terms["object_pcd"]]
    grids = {id(t.state.geometry["right"]["sdf"]) for t in readers}
    assert len(grids) == 1
    assert env.action_manager.get_term("assist").state.geometry == {}  # it needs the inertia only


def test_the_sdf_reads_each_envs_box_along_the_axes_the_grid_was_baked_in(env, motion_file):
    """Points 5 cm beyond each face of each env's own box (a 6 cm cube or a 10 x 8 x 12 cm box), the box tilted:
    the SDF is 5 cm and its gradient the face normal, so the grid axes and the per-env slot are read right.
    """
    from eden.utils.geom import quat_from_axis_angle, transform_by_quat
    from tests.tasks.flashdexretarget.conftest import BOXES

    state = env.reward_manager._terms["object_pcd"].state  # a term that loads the object geometry
    labels = motion_summary(str(motion_file)).slot_labels  # slot order: "<id>*<scale>"
    half = torch.tensor([BOXES[label.split("*")[0]] for label in labels], dtype=torch.float32) / 2
    half = half[state.slot["right"].cpu()].to(env.device)  # (N, 3) each env's box
    assert half[:, 0].unique().numel() == env.num_envs  # the two envs hold different boxes
    N = env.num_envs
    obj_pos = torch.tensor([0.3, -0.2, 0.5], device=env.device).expand(N, 3)
    obj_quat = quat_from_axis_angle(torch.tensor([[0.1, 0.2, 0.3]], device=env.device)).expand(N, 4)
    axes = torch.eye(3, device=env.device)
    local = torch.cat([axes * (half[:, None] + 0.05), -axes * (half[:, None] + 0.05)], dim=1)  # (N, 6, 3)
    points = obj_pos[:, None] + transform_by_quat(local, obj_quat[:, None].expand(N, 6, 4))
    d, grad = state.sdf(0, points, obj_pos, obj_quat)
    torch.testing.assert_close(d, torch.full_like(d, 0.05), atol=2e-3, rtol=0)
    normals = transform_by_quat(torch.cat([axes, -axes]).expand(N, 6, 3), obj_quat[:, None].expand(N, 6, 4))
    torch.testing.assert_close(grad, normals, atol=2e-2, rtol=0)
