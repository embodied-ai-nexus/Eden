"""The SuperMimic runner on the supermimic task: its world model grades as the simulator does, and it trains."""

import pathlib
import re

import pytest
import torch

from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper, make_rsl_rl_runner
from eden.learning.wrappers.rsl_rl.supermimic.runner import SuperMimicRunner
from eden.learning.wrappers.rsl_rl.supermimic.world_model_env import reference_action, sample_starts
from tests.learning.wrappers.rsl_rl.supermimic.conftest import paper_learner, small_learner
from tests.support.supermimic import supermimic_config, write_lafan1_library

pytestmark = [pytest.mark.requires_genesis]


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    return write_lafan1_library(tmp_path_factory.mktemp("lafan1"))


@pytest.fixture(scope="module")
def paper_runner(library, tmp_path_factory):
    """A runner with the recipe's options off (:func:`paper_learner`)."""
    env = RslRlVecEnvWrapper.from_config(paper_learner(small_learner(supermimic_config(library))))
    return make_rsl_rl_runner(env, log_dir=str(tmp_path_factory.mktemp("paper")))


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


def test_it_trains_and_a_checkpoint_restores_the_averaged_policy(runner, tmp_path):
    runner.learn(num_learning_iterations=3)
    assert runner.current_learning_iteration == 3
    assert runner.replay_buffer.num_transitions == 12
    path = str(tmp_path / "model_3.pt")
    runner.save(path)
    obs = runner.env.get_observations()
    before = runner.get_inference_policy()(obs)
    for p in runner.actor_ema.parameters():
        p.data.add_(1.0)
    assert (runner.get_inference_policy()(obs) - before).abs().max() > 1e-3
    runner.load(path)
    torch.testing.assert_close(runner.get_inference_policy()(obs), before)
    assert runner.current_learning_iteration == 3
    assert runner.policy_learner.average_started


def test_with_the_recipes_options_off_it_trains_and_a_checkpoint_restores_the_actor(paper_runner, tmp_path):
    runner = paper_runner
    assert (runner.actor_ema, runner.policy_learner.lr_scale, runner.world_model_learner.mirror) == (None, None, None)
    runner.learn(num_learning_iterations=3)
    assert runner.sampler.fall_rate.sum() == 0.0
    path = str(tmp_path / "model_3.pt")
    runner.save(path)
    obs = runner.env.get_observations()
    before = runner.get_inference_policy()(obs)
    for p in runner.actor.parameters():
        p.data.add_(1.0)
    assert (runner.get_inference_policy()(obs) - before).abs().max() > 1e-3
    runner.load(path)
    torch.testing.assert_close(runner.get_inference_policy()(obs), before)


def test_a_collected_step_acts_on_the_statistics_before_it_and_folds_itself_in_once(runner):
    if runner.replay_buffer is None:
        runner.replay_buffer = runner._make_replay_buffer()
    obs = runner.env.get_observations()
    with torch.inference_mode():
        state, sensor, frames, _ = runner._observe(obs)
        expected = runner.actor.eval()(runner._policy_input(state, sensor, frames))
    folded = [runner.dynamics.state_normalizer, runner.dynamics.action_normalizer, runner.actor.obs_normalizer]
    counts = [int(normalizer.count) for normalizer in folded]
    runner._collect(obs)
    buffer = runner.replay_buffer
    stored = buffer._storage["action"][:, (buffer._step - 1) % buffer.buffer_size]
    torch.testing.assert_close(stored, expected)
    assert [int(normalizer.count) for normalizer in folded] == [count + runner.num_envs for count in counts]


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
    checkpoint["layout"]["runner_options.policy.algorithm"] = "bptt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match=r"policy.algorithm: checkpoint 'bptt', this run 'actor_critic'"):
        runner.load(path)
    # A checkpoint from before the layout record: the algorithm shows in whether it holds a critic.
    del checkpoint["layout"]
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
    from eden.learning.wrappers.rsl_rl.supermimic.task_contract import task_reward_spec

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
    cfg = small_learner(supermimic_config(library))
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
    cfg = small_learner(supermimic_config(library))
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
    cfg = small_learner(supermimic_config(library))
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
    other = make_rsl_rl_runner(
        RslRlVecEnvWrapper.from_config(small_learner(supermimic_config(longer))), checkpoint=path
    )
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
    env = RslRlVecEnvWrapper.from_config(small_learner(supermimic_config(library)))
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


def _swap_gravity_and_ang_vel(cfg):
    policy = cfg.observation_options.policy
    policy.projected_gravity, policy.base_ang_vel = policy.base_ang_vel, policy.projected_gravity


def _absolute_joint_term_without_bias(cfg):
    from eden.managers.terms import observations

    cfg.runner_options.sim_context.encoder_bias_event = None
    cfg.observation_options.policy.dofs_pos = observations.DofsPos.configure(entity_name="robot")


def _unbiased_joint_term(cfg):
    from eden.managers.terms import observations

    cfg.observation_options.policy.dofs_pos = observations.DofsVel.configure(entity_name="robot")


@pytest.mark.parametrize(
    "change, message",
    [
        (_swap_state_term, r"group 'full_state' .* must hold one RootJointState term on that entity, got terms"),
        (_drop_actor_term, r"group 'policy' .* as the proprioception .* The group holds terms \['projected_gravity'"),
        (
            lambda cfg: setattr(cfg.runner_options, "actor_observation_group", "actor"),
            r"actor_observation_group='actor', but the task has no such observation group. Its groups: \['policy'",
        ),
        (
            lambda cfg: setattr(cfg.runner_options.sim_context, "encoder_bias_event", None),
            r"term 'dofs_pos' \(joint_pos\) is a DofsPosBiased, which reads the encoder bias, but "
            r"runner_options.sim_context.encoder_bias_event is None",
        ),
        (_unbiased_joint_term, r"term 'dofs_pos' \(joint_pos\) is a 'dofs_vel' term, which reads no encoder bias"),
        (_swap_gravity_and_ang_vel, r"term 'projected_gravity' \(projected_gravity\) is a 'base_ang_vel' term"),
        (
            lambda cfg: cfg.observation_options.policy.base_ang_vel.params.update(frame="world"),
            r"term 'base_ang_vel' \(base_ang_vel\) has frame='world'; set frame='body'",
        ),
        (
            lambda cfg: setattr(cfg.observation_options.policy.dofs_vel, "scale", 0.25),
            r"term 'dofs_vel' sets \{'scale': 0.25\}, which the world model does not apply",
        ),
        (
            lambda cfg: setattr(cfg.observation_options.policy, "enable_corruption", True),
            r"group 'policy' .*the group has enable_corruption=True, which adds the terms' noise in the simulator",
        ),
        (
            lambda cfg: setattr(cfg.observation_options.full_state.root_joint_state, "clip", (-5.0, 5.0)),
            r"group 'full_state' .* from the raw state: term 'root_joint_state' sets \{'clip':",
        ),
        (_absolute_joint_term_without_bias, r"term 'dofs_pos' \(joint_pos\) is a 'dofs_pos' term with "),
    ],
    ids=[
        "state_term",
        "actor_terms",
        "missing_group",
        "bias_without_event",
        "event_without_bias",
        "swapped_terms",
        "world_frame_ang_vel",
        "scaled_term",
        "corrupted_group",
        "clipped_state",
        "absolute_joints_without_bias",
    ],
)
def test_an_observation_group_the_runner_would_misread_fails_at_construction(library, change, message):
    cfg = small_learner(supermimic_config(library))
    cfg.env_options.num_envs = 2
    change(cfg)
    env = RslRlVecEnvWrapper.from_config(cfg)
    with pytest.raises(ValueError, match=message):
        make_rsl_rl_runner(env)


def test_a_checkpoint_of_other_network_sizes_names_the_options_that_differ(runner, library, tmp_path):
    path = str(tmp_path / "model.pt")
    runner.save(path)
    cfg = small_learner(supermimic_config(library))
    cfg.runner_options.policy.goal_frames = 4
    cfg.runner_options.world_model.hidden_dims = (16,)
    env = RslRlVecEnvWrapper.from_config(cfg)
    with pytest.raises(
        ValueError,
        match=r"runner_options.policy.goal_frames: checkpoint 10, this run 4; "
        r"runner_options.world_model.hidden_dims: checkpoint \(32,\), this run \(16,\)",
    ):
        make_rsl_rl_runner(env, checkpoint=path)


def test_a_task_without_an_encoder_bias_reads_the_joints_relative_to_the_default_pose(library):
    from eden.managers.terms import observations

    cfg = small_learner(supermimic_config(library))
    cfg.env_options.num_envs = 2
    cfg.runner_options.sim_context.encoder_bias_event = None
    cfg.observation_options.policy.dofs_pos = observations.DofsPos.configure(
        entity_name="robot", offset_from_default=True
    )
    runner = make_rsl_rl_runner(RslRlVecEnvWrapper.from_config(cfg))
    assert not runner.sim_context.has_encoder_bias


def _clone_state(module):
    return {key: value.clone() for key, value in module.state_dict().items()}


def test_replay_runs_the_average_of_the_actor_and_collection_the_actor(runner):
    obs = runner.env.get_observations()
    replayed = runner.get_inference_policy()(obs)
    kept = _clone_state(runner.actor)
    try:
        for p in runner.actor.parameters():
            p.data.add_(1.0)
        torch.testing.assert_close(runner.get_inference_policy()(obs), replayed)
        state, sensor, frames, _ = runner._observe(obs)
        collected = runner.actor(runner._policy_input(state, sensor, frames, training=False))
        assert (collected - replayed).abs().max() > 1e-3
    finally:
        runner.actor.load_state_dict(kept)


def test_an_actor_step_moves_the_average_by_one_minus_the_decay(runner):
    runner.learn(num_learning_iterations=1)
    policy = runner.options.policy
    average = _clone_state(runner.actor_ema)
    runner.policy_learner.update(
        sample_starts(runner.replay_buffer, policy.history_length, policy.batch_size) for _ in range(1)
    )
    live = runner.actor.state_dict()
    for key, value in runner.actor_ema.state_dict().items():
        if value.is_floating_point():
            expected = average[key] + (1.0 - policy.ema_decay) * (live[key] - average[key])
            torch.testing.assert_close(value, expected, atol=1e-7, rtol=1e-5)
        else:
            assert torch.equal(value, live[key])  # a normalizer's sample count
    moved = [(value - average[key]).abs().max() for key, value in runner.actor_ema.named_parameters()]
    assert max(moved) > 0  # the step changed the actor, so the comparison above had a difference to miss


def test_a_checkpoint_without_the_average_starts_it_from_the_actor(runner, tmp_path):
    path = str(tmp_path / "model.pt")
    runner.save(path)
    checkpoint = torch.load(path, weights_only=False)
    del checkpoint["actor_ema"]  # a checkpoint trained without policy.ema_decay
    torch.save(checkpoint, path)
    for p in runner.actor_ema.parameters():
        p.data.add_(1.0)
    runner.load(path)
    for key, value in runner.actor_ema.state_dict().items():
        torch.testing.assert_close(value, checkpoint["actor"][key])
    assert not runner.policy_learner.average_started


def _write_as_before_the_recipes_options(runner, path):
    """A checkpoint of ``runner`` without the entries the recipe's options added, as the first release wrote it."""
    runner.save(path)
    checkpoint = torch.load(path, weights_only=False)
    del checkpoint["actor_ema"], checkpoint["lr_decay_start"], checkpoint["return_sampler"]["fall_rate"]
    del checkpoint["layout"]["runner_options.policy.reference_action"]
    torch.save(checkpoint, path)


def test_a_checkpoint_written_before_the_recipes_options_loads_with_them_off(paper_runner, tmp_path):
    runner, path = paper_runner, str(tmp_path / "model.pt")
    obs = runner.env.get_observations()
    before = runner.get_inference_policy()(obs)
    _write_as_before_the_recipes_options(runner, path)
    for p in runner.actor.parameters():
        p.data.add_(1.0)
    runner.load(path)
    torch.testing.assert_close(runner.get_inference_policy()(obs), before)


def test_a_checkpoint_trained_without_the_reference_action_is_refused_with_the_override_that_loads_it(
    paper_runner, runner, tmp_path
):
    path = str(tmp_path / "model.pt")
    message = (
        r"runner_options.policy.reference_action: checkpoint False, this run True\)\. To load it, set the options "
        r"it trained with: --override '\{runner_options: \{policy: \{reference_action: false\}\}\}'"
    )
    paper_runner.save(path)
    with pytest.raises(ValueError, match=message):
        runner.load(path)
    _write_as_before_the_recipes_options(paper_runner, path)
    with pytest.raises(ValueError, match=message):
        runner.load(path)
    checkpoint = torch.load(path, weights_only=False)
    del checkpoint["layout"]  # a checkpoint written before the layout record
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match=message):
        runner.load(path)
    paper_runner.load(path)


def test_a_checkpoint_saved_before_the_first_policy_update_holds_no_average(runner, tmp_path):
    path, started = str(tmp_path / "model.pt"), runner.policy_learner.average_started
    runner.policy_learner.average_started = False  # as before the first update, whatever earlier tests trained
    try:
        runner.save(path)
    finally:
        runner.policy_learner.average_started = started
    assert torch.load(path, weights_only=False)["actor_ema"] is None


def test_a_run_config_written_before_the_recipes_options_replays_under_the_override_the_error_names(
    paper_runner, library, tmp_path
):
    """Such a config lacks the recipe's options, which take today's defaults when it is loaded."""
    import json

    from eden.utils.configs import EdenConfig, overrides_from_str

    checkpoint, saved = str(tmp_path / "model.pt"), str(tmp_path / "config.json")
    _write_as_before_the_recipes_options(paper_runner, checkpoint)
    paper_learner(small_learner(supermimic_config(library))).save_as_file(saved)
    data = json.loads(pathlib.Path(saved).read_text())
    options = data["config"]["runner_options"]
    for group, names in {
        "policy": ("ema_decay", "reference_action", "lr_multiplier", "lr_decay_return", "lr_decay_deadline"),
        "world_model": ("mirror_share",),
        "return_sampler": ("failure", "failure_lead_s"),
    }.items():
        for name in names:
            del options[group][name]
    pathlib.Path(saved).write_text(json.dumps(data))

    cfg = EdenConfig.load_from_file(saved)
    assert cfg.runner_options.policy.reference_action
    with pytest.raises(ValueError) as error:
        make_rsl_rl_runner(RslRlVecEnvWrapper.from_config(cfg), checkpoint=checkpoint, for_inference=True)
    override = re.search(r"--override '([^']*)'", str(error.value)).group(1)
    assert override == "{runner_options: {policy: {reference_action: false}}}"
    cfg = EdenConfig.load_from_file(saved).with_overrides_from_dict(overrides_from_str(override))
    replayed = make_rsl_rl_runner(RslRlVecEnvWrapper.from_config(cfg), checkpoint=checkpoint, for_inference=True)
    # It replays the checkpoint's actor: the average this config keeps starts from it.
    for key, value in paper_runner.actor.state_dict().items():
        torch.testing.assert_close(replayed.actor_ema.state_dict()[key], value)


def test_a_checkpoint_with_an_average_warns_a_runner_that_keeps_none(paper_runner, runner, tmp_path, eden_caplog):
    path = str(tmp_path / "model.pt")
    paper_runner.save(path)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["actor_ema"] = checkpoint["actor"]
    torch.save(checkpoint, path)
    paper_runner.load(path)
    assert "holds an average of the actor" in eden_caplog.text
    assert "runner_options.policy.ema_decay=0" in eden_caplog.text


def _zero_decoder_output(actor):
    last = actor.decoder[-1]
    last.weight.data.zero_()
    last.bias.data.zero_()


def test_a_zero_network_output_commands_the_next_reference_pose_in_replay_collection_and_imagination(runner):
    runner.learn(num_learning_iterations=1)
    library, dynamics = runner.motion.library, runner.dynamics
    kept = _clone_state(runner.actor), _clone_state(runner.actor_ema)
    try:
        _zero_decoder_output(runner.actor)
        _zero_decoder_output(runner.actor_ema)
        obs = runner.env.get_observations()
        state, sensor, frames, _ = runner._observe(obs)
        following = torch.minimum(frames + 1, library.last_frame(frames))
        expected = library.joint_pos[following]
        assert (expected - library.joint_pos[frames]).abs().max() > 1e-4  # the next frame differs from the current

        replayed = runner.get_inference_policy()(obs)
        torch.testing.assert_close(dynamics.target(replayed), expected, atol=1e-5, rtol=0)
        runner.actor.eval()
        collected = runner.actor(runner._policy_input(state, sensor, frames))
        torch.testing.assert_close(collected, replayed, atol=1e-6, rtol=0)

        policy = runner.options.policy
        start = sample_starts(runner.replay_buffer, policy.history_length, policy.batch_size)
        imagined = runner.actor(runner.wm_env.reset(start))
        torch.testing.assert_close(imagined, reference_action(library, start["frame"], dynamics), atol=1e-6, rtol=0)
        next_obs, _, _ = runner.wm_env.step(imagined)
        advanced = torch.minimum(start["frame"] + 1, library.last_frame(start["frame"]))
        torch.testing.assert_close(
            runner.actor(next_obs), reference_action(library, advanced, dynamics), atol=1e-6, rtol=0
        )
    finally:
        runner.actor.load_state_dict(kept[0])
        runner.actor_ema.load_state_dict(kept[1])


def test_the_reference_action_is_recorded_in_the_checkpoint_layout(paper_runner, runner):
    assert paper_runner._layout()["runner_options.policy.reference_action"] is False
    assert runner._layout()["runner_options.policy.reference_action"] is True


def test_the_normalized_return_is_the_mean_step_reward_of_ended_episodes_over_the_largest(runner):
    runner._take_normalized_return()
    runner._episode_reward.zero_()
    runner._episode_steps.zero_()
    best = runner.max_step_reward
    no_end = torch.zeros(runner.num_envs, dtype=torch.bool)
    first_env = torch.zeros(runner.num_envs, dtype=torch.bool)
    first_env[0] = True
    runner._record_returns(torch.full((runner.num_envs,), 0.25 * best), no_end)
    assert runner._take_normalized_return() is None  # no episode ended
    runner._record_returns(torch.full((runner.num_envs,), 0.75 * best), first_env)
    assert runner._take_normalized_return() == pytest.approx(0.5)  # (0.25 + 0.75) / 2 steps
    runner._record_returns(torch.full((runner.num_envs,), best), ~no_end)
    # Env 0 started a new episode of one step; the others end after three.
    expected = (1.0 + (runner.num_envs - 1) * (0.25 + 0.75 + 1.0) / 3) / runner.num_envs
    assert runner._take_normalized_return() == pytest.approx(expected)


def test_the_learning_rate_ramp_starts_when_the_return_reaches_its_threshold(runner):
    policy = runner.options.policy
    assert (policy.lr_multiplier, policy.lr_decay_return, policy.lr_decay_iterations) == (4.0, 0.75, 75)
    started = runner._lr_decay_start  # training in an earlier test may have started the ramp
    runner._lr_decay_start = None
    try:
        runner._start_lr_decay_if_due(7, 0.74)
        runner._start_lr_decay_if_due(8, None)
        assert runner._lr_multiplier(8) == 4.0
        runner._start_lr_decay_if_due(9, 0.76)
        runner._start_lr_decay_if_due(20, 0.9)  # the ramp has started and keeps its start
        assert [runner._lr_multiplier(it) for it in (9, 9 + 25, 9 + 75, 9 + 500)] == pytest.approx([4.0, 3.0, 1.0, 1.0])
    finally:
        runner._lr_decay_start = started


def test_a_multiplier_of_one_has_no_ramp_to_start(paper_runner, eden_caplog):
    policy = paper_runner.options.policy
    assert (policy.lr_multiplier, policy.lr_decay_deadline) == (1.0, 350)
    paper_runner._start_lr_decay_if_due(400, 0.9)
    assert paper_runner._lr_decay_start is None
    assert "ramps" not in eden_caplog.text


def test_the_learning_rate_ramp_starts_at_its_deadline_and_survives_a_resume(library, tmp_path):
    cfg = small_learner(supermimic_config(library))
    policy = cfg.runner_options.policy
    policy.lr_decay_return, policy.lr_decay_deadline, policy.lr_decay_iterations = None, 1, 2
    env = RslRlVecEnvWrapper.from_config(cfg)
    runner = make_rsl_rl_runner(env, log_dir=str(tmp_path))
    runner.learn(num_learning_iterations=3)
    # Iterations 0 and 1 step at 4x (the ramp starts at iteration 1), iteration 2 halfway down.
    assert runner.policy_learner.lr_scale.item() == pytest.approx(2.5)
    path = str(tmp_path / "model_3.pt")
    runner.save(path)
    resumed = make_rsl_rl_runner(env, checkpoint=path, log_dir=str(tmp_path / "resumed"))
    assert resumed.policy_learner.lr_scale.item() == pytest.approx(4.0)
    resumed.learn(num_learning_iterations=1)
    assert resumed.policy_learner.lr_scale.item() == pytest.approx(1.0)


def test_a_foot_randomized_on_one_side_only_cannot_be_mirrored(library):
    cfg = small_learner(supermimic_config(library))
    cfg.env_options.num_envs = 2
    cfg.event_options.foot_friction.links_name = ["left_ankle_roll_link"]
    env = RslRlVecEnvWrapper.from_config(cfg)
    with pytest.raises(
        ValueError, match=r"\[\('friction', 'left_ankle_roll_link'\)\] have no partner among the columns"
    ):
        make_rsl_rl_runner(env)


def test_a_lateral_offset_randomized_off_centre_cannot_be_mirrored(library):
    cfg = small_learner(supermimic_config(library))
    cfg.env_options.num_envs = 2
    cfg.event_options.base_com.com_shift_y_range = (0.0, 0.05)
    env = RslRlVecEnvWrapper.from_config(cfg)
    with pytest.raises(
        ValueError, match=r"randomization ranges of columns \[\('com_y', 'torso_link'\)\] are not the mirror"
    ):
        make_rsl_rl_runner(env)


def test_an_untrained_actor_with_the_reference_action_commands_nearly_the_reference_pose(library):
    cfg = small_learner(supermimic_config(library))
    cfg.env_options.num_envs = 2
    env = RslRlVecEnvWrapper.from_config(cfg)
    runner = make_rsl_rl_runner(env)
    assert runner.options.policy.reference_action
    state, sensor, frames, _ = runner._observe(env.get_observations())
    actor_in = runner._policy_input(state, sensor, frames, training=False)
    correction = runner.actor(actor_in, interface="motion") - actor_in["reference_action"]
    assert 0 < correction.abs().max() < 0.05
