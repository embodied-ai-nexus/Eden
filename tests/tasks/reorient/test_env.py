"""The reorient task built and stepped: observations, the action bounds, the goal machine and the cage.

One training-mode right-hand env and one eval-mode left-hand env (eval is what draws the goal marker) serve the
whole module; every test starts from ``env.reset()``.
"""

import pytest
import torch

from eden.envs.base import RLEnvBase
from eden.tasks import TASK_REGISTRY
from tests.support.envs import build_env

pytestmark = [pytest.mark.requires_genesis, pytest.mark.slow]


def _build(side: str, *, eval_mode: bool = False) -> RLEnvBase:
    cfg = TASK_REGISTRY.build("reorient", hand=side)
    cfg.env_options.num_envs = 8
    env = RLEnvBase.from_config(cfg, show_viewer=False, eval_mode=eval_mode)
    env.build()
    return env


@pytest.fixture(scope="module")
def right_env():
    return _build("right")


@pytest.fixture(scope="module")
def left_env():
    return _build("left", eval_mode=True)


@pytest.fixture(params=["right", "left"])
def env(request):
    return request.getfixturevalue(f"{request.param}_env")


def _zero(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)


def test_observation_groups_are_flat_finite_and_keep_privileged_terms_off_the_actor(env):
    obs, _ = env.reset()
    assert env.action_manager.total_action_dim == 20
    assert env.dt == pytest.approx(0.05)  # 20 Hz, as upstream
    # Flat, because rsl-rl's ActorCritic rejects a history axis. The critic's privileged half was once dropped by
    # a refactor and the env still trained, blind: 5 policy terms x 3-step history = 207, the critic 403.
    assert obs["policy"].shape == (env.num_envs, 207)
    assert obs["critic"].shape == (env.num_envs, 403)
    privileged = {
        "true_dofs_pos",
        "true_dofs_vel",
        "fingertip_positions",
        "cube_lin_vel",
        "cube_ang_vel",
        "palm_rot_6d",
        "command_state",
        "cage_counter",
        "perturbation_dir",
        "perturbation_vel",
        "dr_params",
    }
    groups = env.observation_manager.group_term_names
    assert privileged <= set(groups["critic"]), privileged - set(groups["critic"])
    assert not privileged & set(groups["policy"])

    for _ in range(5):
        obs, reward, _, _, _ = env.step(_zero(env))
    for name, value in obs.items():
        assert torch.isfinite(value).all(), name
    assert torch.isfinite(reward).all()


def test_holding_the_default_pose_does_not_drop_the_cube(env):
    """A zero action commands the grasp keyframe; this is what caught the left-hand mirroring bug."""
    env.reset()
    dropped = 0
    for _ in range(60):
        _, _, terminated, _, _ = env.step(_zero(env))
        dropped += int(terminated.sum().item())
    assert dropped == 0
    assert env.metric_manager.get_term("hand_object_contact").distal_contact.float().mean().item() > 0.5


def test_privileged_dr_observation_reports_ratios_inside_the_randomization_ranges(right_env):
    """The baselines are captured in ``build()``, before startup DR; a wrong one leaves a channel near zero."""
    right_env.reset()
    right_env.step(_zero(right_env))
    armature, damping, kp, kd, mass = right_env.observation_manager.get_term("dr_params").compute().unbind(dim=-1)
    assert ((armature > 0.7) & (armature < 1.4)).all(), armature
    assert ((kp > 0.7) & (kp < 1.6)).all(), kp
    assert ((kd > 0.4) & (kd < 2.1)).all(), kd
    assert ((mass > 0.35) & (mass < 1.7)).all(), mass
    assert ((damping >= 0.0) & (damping < 0.01)).all(), damping


def test_the_action_term_bounds_the_target_without_clipping_the_raw_action(right_env):
    """The runner does not clip, so the term must keep the target in range while the raw action stays whole."""
    env = right_env
    env.reset()
    term = env.action_manager.get_term("joint_pos")
    for _ in range(int(term.warmup_time_s / env.dt) + 3):  # past the warmup window, so the target is live
        env.step(torch.full_like(_zero(env), 5.0))

    torch.testing.assert_close(term.action, torch.full_like(term.action, 5.0))
    entity = env.entities["robot"]
    # By name column: ``soft_dofs_pos_limits`` is columned by ``dofs_name``, not by Genesis-local index.
    soft = entity.soft_dofs_pos_limits[:, entity.find_named_dofs(term.dofs_name).columns, :]
    assert (term.processed_action >= soft[..., 0] - 1e-5).all()
    assert (term.processed_action <= soft[..., 1] + 1e-5).all()


def test_the_goal_machine_banks_a_goal_and_opens_the_success_window(right_env):
    env = right_env
    env.reset()
    command, cube = env.command_manager.get_term("reorient"), env.entities["cube"]
    assert command.goal_reach_count.sum().item() == 0
    opened = False
    for _ in range(command.success_hold_steps + 4):
        command.goal_quat_w = cube.get_quat().clone()  # keep the orientation error inside the threshold
        env.step(_zero(env))
        opened = opened or bool(command.in_success_window.any().item())
    assert command.goal_reach_count.sum().item() > 0
    assert opened


def test_hold_escalation_pays_its_peak_before_the_goal_switch_clears_it(right_env):
    """Graded at phase 9 from state measured at phase 5; the switch at phase 17 clears it afterwards.

    Read back through the reward manager's raw episode sum, because the value under test no longer exists by the
    time ``step()`` returns.
    """
    env = right_env
    env.reset()
    command, cube = env.command_manager.get_term("reorient"), env.entities["cube"]
    paid, banked = [], torch.zeros(env.num_envs, device=env.device)
    for _ in range(command.success_hold_steps + command.goal_switch_delay + 6):
        command.goal_quat_w = cube.get_quat().clone()
        _, _, terminated, truncated, _ = env.step(_zero(env))
        assert not bool((terminated | truncated)[0].item()), "episode ended mid-cycle"
        total = env.reward_manager.get_episode_sum("hold_escalation", raw=True)
        paid.append(float(((total - banked) / env.dt)[0].item()))
        banked = total.clone()

    # Integer counters recovered by differencing a float32 running sum; the bug moves the peak by 1.0.
    assert max(paid) == pytest.approx(command.goal_switch_delay - 1, abs=1e-3), paid
    rises = [b - a for a, b in zip(paid, paid[1:]) if b > a]
    assert rises and all(rise == pytest.approx(1.0, abs=1e-3) for rise in rises), paid


def test_the_goal_marker_tracks_the_goal_orientation_at_eval(left_env):
    """Only an eval env calls ``command_manager.draw_vis()``; a training build would leave the marker at spawn."""
    env = left_env
    env.reset()
    env.step(_zero(env))
    marker = env.entities["cube_goal"]
    torch.testing.assert_close(
        marker.get_quat(), env.command_manager.get_term("reorient").goal_quat_w, atol=1e-5, rtol=0.0
    )
    above = marker.get_pos()[:, 2] - env.entities["cube"].get_pos()[:, 2]
    assert (above > 0.05).all(), above.tolist()


def test_the_cage_counter_escalates_and_terminates_when_the_cube_leaves_the_hand(right_env):
    env = right_env
    env.reset()
    cage, cube = env.metric_manager.get_term("cage_escape"), env.entities["cube"]
    far = cube.get_pos().clone()
    far[:, 2] += 0.5
    terminated_any, counters = False, []
    for _ in range(cage.max_outside_steps + 3):
        cube.set_pos(far)
        _, _, terminated, _, _ = env.step(_zero(env))
        counters.append(cage.penalty_counter.max().item())
        terminated_any = terminated_any or bool(terminated.any().item())
    assert counters[2] > counters[0], counters
    assert terminated_any, f"cage_drop never fired; counter reached {max(counters)}"
    env.reset()


def test_overlapping_reset_patterns_are_rejected_at_build():
    """``set_dofs_pos`` takes the index list as given, so a DOF matched twice would silently take the later range."""
    cfg = TASK_REGISTRY.build("reorient", hand="right")
    cfg.env_options.num_envs = 1
    cfg.event_options.reset_hand_dofs.dofs_pos_range = {".*_joint[12]": (-0.1, 0.0), ".*_joint2": (0.0, 0.1)}
    with pytest.raises(ValueError, match="matched by both"):
        build_env(cfg)
