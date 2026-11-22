"""The SuperMimic runner on the supermimic task: its world model grades as the simulator does, and it trains."""

import pytest
import torch

from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper, make_rsl_rl_runner
from eden.learning.wrappers.rsl_rl.supermimic.runner import SuperMimicRunner
from tests.support.supermimic import supermimic_config, write_lafan1_library

pytestmark = [pytest.mark.requires_genesis]


def _small(cfg):
    o = cfg.runner_options
    o.device, o.logger, o.num_steps_per_env, o.buffer_size = "cpu", "tensorboard", 4, 32
    o.world_model.hidden_dims, o.world_model.batch_size, o.world_model.num_batches = (32,), 8, 2
    o.policy.actor_hidden_dims, o.policy.critic_hidden_dims, o.policy.batch_size = (32,), (16,), 8
    o.policy.horizon, o.replay_ratio = 4, 64.0
    return cfg


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    return write_lafan1_library(tmp_path_factory.mktemp("lafan1"))


@pytest.fixture(scope="module")
def runner(library, tmp_path_factory):
    env = RslRlVecEnvWrapper.from_config(_small(supermimic_config(library)))
    return make_rsl_rl_runner(env, log_dir=str(tmp_path_factory.mktemp("run")))


def test_the_generic_factory_builds_the_runner_without_training_state(runner):
    assert isinstance(runner, SuperMimicRunner)
    assert runner.replay_buffer is None


def test_the_world_model_reward_and_termination_grade_a_real_transition_as_the_task_does(runner):
    env, rep, wm_env = runner.env, runner.representation, runner.wm_env
    obs = env.get_observations()
    torch.manual_seed(0)
    for _ in range(6):
        action = 0.5 * torch.randn(env.num_envs, env.num_actions)
        last = env.unwrapped.action_manager.action.clone()
        obs, reward, dones, _ = env.step(action)
        kept = ~dones.bool()
        state = obs["full_state"]
        target = wm_env.reference_bodies(env.unwrapped.motion_manager.get_library_frame("default"))
        wm_reward = rep.reward(rep.aux_state(state), rep.represent(state), action, last, target)
        torch.testing.assert_close(wm_reward[kept], reward[kept], atol=1e-5, rtol=1e-4)


def test_it_trains_and_a_checkpoint_restores_the_policy(runner, tmp_path):
    runner.learn(num_learning_iterations=3)
    assert runner.current_learning_iteration == 3
    assert runner.replay_buffer.num_transitions == 12
    path = str(tmp_path / "model_3.pt")
    runner.save(path)
    obs = runner.env.get_observations()
    policy = runner.get_inference_policy()
    before = policy(obs)
    for p in runner.actor.parameters():
        p.data.add_(1.0)
    runner.load(path)
    torch.testing.assert_close(runner.get_inference_policy()(obs), before)
    assert runner.current_learning_iteration == 3


def test_a_checkpoint_restores_the_return_sampler_without_the_optimizers(runner, tmp_path):
    path = str(tmp_path / "model.pt")
    runner.save(path)
    value = runner.sampler.value.clone()
    runner.sampler.value.add_(1.0)
    runner.load(path, load_optimizer=False)
    torch.testing.assert_close(runner.sampler.value, value)


def test_a_checkpoint_of_the_other_algorithm_is_refused(runner, tmp_path):
    path = str(tmp_path / "model.pt")
    runner.save(path)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["critic"] = None
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="trained with policy.algorithm='bptt'"):
        runner.load(path)


def test_the_generic_factory_loads_a_checkpoint_through_the_runner(runner, tmp_path):
    path = str(tmp_path / "model.pt")
    runner.save(path)
    expected = {k: v.clone() for k, v in runner.actor.state_dict().items()}
    for p in runner.actor.parameters():
        p.data.add_(1.0)
    restored = make_rsl_rl_runner(runner.env, checkpoint=path, for_inference=True)
    for key, value in restored.actor.state_dict().items():
        torch.testing.assert_close(value, expected[key])


def test_a_non_finite_collected_value_is_named_at_the_end_of_its_iteration(runner):
    runner._raise_if_not_finite()
    mean = runner.dynamics.action_normalizer.mean
    kept = mean.clone()
    mean[0] = float("nan")
    try:
        with pytest.raises(ValueError, match=r"Non-finite values in the collected \['action'\]"):
            runner._raise_if_not_finite()
    finally:
        mean.copy_(kept)


def test_the_reward_is_read_off_the_tasks_terms(runner):
    spec = runner.representation.reward_spec
    assert spec.tracking == ((0.5, 0.3), (0.5, 0.4), (1.0, 0.3), (1.0, 0.4), (1.0, 1.0), (1.0, 3.14))
    assert (spec.action_rate, spec.joint_limit) == (-0.1, -2.0)


def _reward_spec(env, **renamed):
    from eden.learning.wrappers.rsl_rl.supermimic.options import RewardTermsOptions
    from eden.learning.wrappers.rsl_rl.supermimic.task_terms import task_reward_spec

    links = env.unwrapped.motion_manager.get_term("default").fk.link_names
    return task_reward_spec(env.unwrapped, RewardTermsOptions(**renamed), links)


def test_a_reward_term_of_another_kind_is_refused_and_named(runner):
    with pytest.raises(ValueError, match=r"'motion_body_ori' \(runner_options.reward_terms.body_pos\) as a"):
        _reward_spec(runner.env, body_pos="motion_body_ori", body_ori="motion_body_pos")


def test_a_reward_term_the_task_lacks_lists_the_tasks_terms(runner):
    with pytest.raises(ValueError, match=r"reward_terms.joint_limit='limits', but the task's RewardManager has no"):
        _reward_spec(runner.env, joint_limit="limits")


@pytest.mark.parametrize(
    "change, message",
    [
        (
            lambda cfg: setattr(cfg.reward_options.motion_body_lin_vel, "links_name", ["pelvis", "torso_link"]),
            r"'motion_body_lin_vel' .* differs from what the world model computes: links_name=",
        ),
        (
            lambda cfg: setattr(cfg.motion_options, "data_start_from", "one_frame_interval"),
            r"motion_options.data_start_from='one_frame_interval'; set it to 'current_time'",
        ),
        (
            lambda cfg: setattr(cfg.motion_options, "frame_interval_s_range", (0.04, 0.04)),
            r"frame_interval_s_range=\(0.04, 0.04\); set both ends to the env dt 0.02",
        ),
        (
            lambda cfg: cfg.termination_options.anchor_pos.params.update(threshold_adaptive=True),
            r"'anchor_pos' \(runner_options.anchor_pos_termination\) sets threshold_adaptive=True",
        ),
        (
            lambda cfg: setattr(cfg.motion_options.default, "freeze_at_timestep_zero_prob", 0.3),
            r"freeze_at_timestep_zero_prob=0.3, which holds some envs on their first frame; set it to 0",
        ),
        (
            lambda cfg: setattr(cfg.motion_options, "resync_on_motion_end", True),
            r"resync_on_motion_end=True, which restarts a finished clip from the base term's start sampling",
        ),
        (
            lambda cfg: cfg.motion_options.default.target_links_name.update(
                g1=["torso_link", *[n for n in cfg.motion_options.default.target_links_name["g1"] if n != "torso_link"]]
            ),
            r"must list the base link 'pelvis' first in target_links_name, got 'torso_link'",
        ),
    ],
    ids=[
        "reward_links",
        "data_start_from",
        "frame_interval",
        "adaptive_threshold",
        "freeze",
        "resync",
        "anchor_not_base",
    ],
)
def test_a_task_the_world_model_would_grade_differently_fails_at_construction(library, change, message):
    cfg = _small(supermimic_config(library))
    cfg.env_options.num_envs = 2
    change(cfg)
    env = RslRlVecEnvWrapper.from_config(cfg)
    with pytest.raises(ValueError, match=message):
        make_rsl_rl_runner(env)


def test_the_imagined_proprioception_is_the_simulators_reading_encoder_bias_included(runner):
    env, rep, context = runner.env, runner.representation, runner.sim_context
    obs = env.get_observations()
    torch.manual_seed(1)
    for _ in range(3):
        obs, _, _, _ = env.step(0.3 * torch.randn(env.num_envs, env.num_actions))
    bias = context.encoder_bias(context.read())
    assert bias.abs().max() > 1e-4  # the task randomizes it, so the comparison below has something to miss
    last_action = env.unwrapped.action_manager.action
    imagined = rep.proprio(rep.represent(obs["full_state"]), last_action, bias)
    torch.testing.assert_close(imagined, obs["policy"], atol=1e-4, rtol=0)
    unbiased = rep.proprio(rep.represent(obs["full_state"]), last_action)
    assert (unbiased - obs["policy"]).abs().max() > 1e-4


def test_the_inference_policy_follows_the_motion_goal_on_every_env_without_noise(library):
    cfg = _small(supermimic_config(library))
    cfg.runner_options.policy.observation = "proprio"
    cfg.runner_options.policy.history_length = 2
    cfg.runner_options.policy.slot_mix = {"motion": 0.5, "velocity": 0.5}
    env = RslRlVecEnvWrapper.from_config(cfg)
    runner = make_rsl_rl_runner(env)
    obs = env.get_observations()
    first = runner.get_inference_policy()(obs)
    torch.testing.assert_close(runner.get_inference_policy()(obs), first)  # no sensor or command noise
    runner.actor.set_slot_mix({"motion": 1.0})
    state, sensor, frames, _ = runner._observe(obs)
    runner._seed_history(obs, training=False)
    motion_only = runner.actor(runner._policy_input(state, sensor, frames, training=False))
    torch.testing.assert_close(first, motion_only)


def test_the_context_holds_the_friction_of_each_link_the_event_randomizes(runner):
    base = runner.env.unwrapped
    feet = base.event_manager.get_term("foot_friction")
    assert runner.sim_context.dim == len(feet.links_name) + 3 + runner.env.num_actions
    friction = base.entities["robot"].get_friction_ratio(ls_idx_local=feet.ls_idx_local)
    lo, hi = feet.friction_range
    expected = (friction - (lo + hi) / 2) / ((hi - lo) / 2)
    torch.testing.assert_close(runner.sim_context.read()[:, : len(feet.links_name)], expected)
    assert (friction[:, 0] != friction[:, 1]).any()  # each link has its own draw


def test_a_runner_device_other_than_the_envs_is_refused(library):
    cfg = _small(supermimic_config(library))
    cfg.runner_options.device = "cuda:0"
    env = RslRlVecEnvWrapper.from_config(cfg)
    if torch.device(env.device).type != "cpu":
        pytest.skip("needs a CPU simulator to name a different device")
    with pytest.raises(ValueError, match=r"runner_options.device resolves to 'cuda:0', the env is on 'cpu'"):
        make_rsl_rl_runner(env)


def test_a_checkpoint_from_another_library_replays_and_refuses_to_train(runner, library, tmp_path):
    from tests.support.motion import write_lafan1_csv_library

    path = str(tmp_path / "model.pt")
    runner.save(path)
    longer = write_lafan1_csv_library(tmp_path / "lafan1", [90, 45, 200])
    other = make_rsl_rl_runner(RslRlVecEnvWrapper.from_config(_small(supermimic_config(longer))), checkpoint=path)
    assert other.sampler.bin_count != runner.sampler.bin_count
    obs = other.env.get_observations()
    assert other.get_inference_policy()(obs).shape == (other.env.num_envs, other.env.num_actions)
    with pytest.raises(ValueError, match=r"return sampler holds \d+ reset bins, but this run's motion library has"):
        other.learn(num_learning_iterations=1)


def test_a_loaded_checkpoint_hands_its_reset_distribution_to_the_motion_term(runner, tmp_path):
    runner.sampler.effort += 1.0
    runner.sampler.value[0] = 0.0  # the lowest return: resets focus on bin 0
    path = str(tmp_path / "model.pt")
    runner.save(path)
    runner.motion.set_reset_distribution(torch.ones(runner.motion.bin_count))
    runner.load(path)
    torch.testing.assert_close(runner.motion.reset_distribution, runner.sampler.distribution())
    assert runner.motion.reset_distribution.argmax() == 0


def test_a_second_learn_call_ends_the_stored_episodes_before_it_resets_the_envs(library, tmp_path):
    env = RslRlVecEnvWrapper.from_config(_small(supermimic_config(library)))
    runner = make_rsl_rl_runner(env, log_dir=str(tmp_path))
    runner.learn(num_learning_iterations=1)
    buffer = runner.replay_buffer
    newest = (buffer._step - 1) % buffer.buffer_size
    runner.learn(num_learning_iterations=1)
    # The last step of the first call is followed by a reset, so no sampled window may run across it.
    assert buffer._storage["truncated"][:, newest].all()


def _drop_actor_term(cfg):
    del cfg.observation_options.policy.last_action


def _swap_state_term(cfg):
    from eden.managers.terms import observations

    cfg.observation_options.full_state.root_joint_state = observations.DofsVel.configure(entity_name="robot")


@pytest.mark.parametrize(
    "change, message",
    [
        (_swap_state_term, r"group 'full_state' .* must hold one RootJointState term on that entity, got terms"),
        (_drop_actor_term, r"group 'policy' .* as the proprioception .* The group holds terms \['projected_gravity'"),
        (
            lambda cfg: setattr(cfg.runner_options, "actor_observation_group", "actor"),
            r"actor_observation_group='actor', but the task has no such observation group. Its groups: \['policy'",
        ),
    ],
    ids=["state_term", "actor_terms", "missing_group"],
)
def test_an_observation_group_the_runner_would_misread_fails_at_construction(library, change, message):
    cfg = _small(supermimic_config(library))
    cfg.env_options.num_envs = 2
    change(cfg)
    env = RslRlVecEnvWrapper.from_config(cfg)
    with pytest.raises(ValueError, match=message):
        make_rsl_rl_runner(env)


def test_a_checkpoint_of_other_network_sizes_names_the_options_that_differ(runner, library, tmp_path):
    path = str(tmp_path / "model.pt")
    runner.save(path)
    cfg = _small(supermimic_config(library))
    cfg.runner_options.policy.goal_frames = 4
    cfg.runner_options.world_model.hidden_dims = (16,)
    env = RslRlVecEnvWrapper.from_config(cfg)
    with pytest.raises(
        ValueError,
        match=r"runner_options.policy.goal_frames: checkpoint 10, this run 4; "
        r"runner_options.world_model.hidden_dims: checkpoint \(32,\), this run \(16,\)",
    ):
        make_rsl_rl_runner(env, checkpoint=path)
