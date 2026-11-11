"""The homie task config, built: the observation layout HIMPPO's mirror maps index, and the reset state."""

from __future__ import annotations

import torch

from eden.tasks.homie.symmetry import ACTION_DIM, CRITIC_DIM, FRAME_DIM, POLICY_DIM


def test_observation_layout_is_the_one_the_mirror_maps_index(homie_env):
    obs = homie_env.observation_manager
    frame = ["command", "base_ang_vel", "projected_gravity", "dofs_pos", "dofs_vel", "actions"]
    assert obs.group_term_names["policy"] == frame
    assert obs.group_term_names["critic"] == [*frame, "base_lin_vel"]
    assert [d[-1] for d in obs.group_obs_term_dim["critic"]] == [4, 3, 3, 29, 29, 12, 3]
    assert sum(d[-1] for d in obs.group_obs_term_dim["critic"][:-1]) == FRAME_DIM

    out = obs.compute()
    assert out["policy"].shape[-1] == POLICY_DIM
    assert out["critic"].shape[-1] == CRITIC_DIM
    assert homie_env.action_manager.total_action_dim == ACTION_DIM


def test_reset_starts_the_base_moving_within_openhomie_range(homie_env):
    homie_env.reset()
    robot = homie_env.entities["robot"]
    vel = torch.cat([robot.get_vel(), robot.get_ang(frame="body")], dim=-1)
    # Every axis is drawn from U(-0.5, 0.5) on top of the pose reset's zero velocity.
    assert (vel.abs() <= 0.5 + 1e-5).all()
    assert (vel.abs().amax(dim=-1) > 0.0).all()


def test_the_ground_never_wins_the_friction_max_combine():
    from eden.tasks import TASK_REGISTRY

    cfg = TASK_REGISTRY.build("homie")
    ground = max(cfg.scene_options.plane.default_geoms_friction.values())
    # Genesis resolves a contact pair's friction as max(a, b): a ground above the robot's lowest
    # draw would clip the randomized range from below.
    assert ground < cfg.event_options.randomize_friction.friction_range[0]


def test_the_runner_wrapper_clips_actions_before_the_env_sees_them():
    from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper
    from eden.tasks import TASK_REGISTRY

    cfg = TASK_REGISTRY.build("homie")
    cfg.env_options.num_envs = 2
    env = RslRlVecEnvWrapper.from_config(cfg)
    actions = torch.full((2, env.num_actions), 1.0e6, device=env.device)
    actions[1] = -1.0e6
    env.step(actions)
    # OpenHomie clips at 100: an action mean that runs away cannot blow up the quadratic action penalties.
    applied = env.unwrapped.action_manager.action
    assert torch.equal(applied[0], torch.full_like(applied[0], 100.0))
    assert torch.equal(applied[1], torch.full_like(applied[1], -100.0))
