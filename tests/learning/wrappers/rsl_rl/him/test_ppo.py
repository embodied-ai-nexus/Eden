"""HIMPPO through Eden's stock rsl_rl runner, on the homie task — the path ``eden train --task homie`` takes.

Version-agnostic: the same tests pass on rsl-rl-lib 3, 4 and 5
(``uv run --with rsl-rl-lib==3.3.0 pytest <this file>``).
"""

from __future__ import annotations

import math

import pytest
import torch

from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper, make_rsl_rl_runner
from eden.learning.wrappers.rsl_rl.him import HIMPPO, HIMPPOAlgorithmOptions
from eden.options.learning.rsl_rl.translate import detect_rsl_rl_major_version
from eden.tasks import TASK_REGISTRY
from eden.tasks.homie.symmetry import ACTION_DIM


@pytest.fixture(scope="module")
def wrapped_env():
    cfg = TASK_REGISTRY.build("homie")
    cfg.env_options.num_envs = 4
    cfg.runner_options.num_steps_per_env = 4
    cfg.runner_options.algorithm.num_mini_batches = 2
    cfg.runner_options.logger = "tensorboard"
    return RslRlVecEnvWrapper.from_config(cfg)


@pytest.mark.slow
def test_runner_trains_saves_and_resumes_the_iteration(wrapped_env, tmp_path):
    runner = make_rsl_rl_runner(wrapped_env, log_dir=str(tmp_path))
    assert isinstance(runner.alg, HIMPPO)
    runner.learn(num_learning_iterations=2)
    checkpoint = tmp_path / "model_1.pt"
    assert checkpoint.exists()

    resumed = make_rsl_rl_runner(wrapped_env, checkpoint=str(checkpoint), log_dir=str(tmp_path / "resumed"))
    assert resumed.current_learning_iteration == 1
    for key, value in runner.alg.actor_critic.state_dict().items():
        torch.testing.assert_close(resumed.alg.actor_critic.state_dict()[key], value, msg=key)

    policy = resumed.get_inference_policy(device=wrapped_env.device)
    actions = policy(wrapped_env.get_observations())
    assert actions.shape == (wrapped_env.num_envs, ACTION_DIM)
    assert torch.isfinite(actions).all()


def test_a_layout_the_task_mirror_does_not_cover_is_refused_at_construction():
    """Five history frames instead of six: HIM's own layout check passes, the homie mirror's does not."""
    cfg = TASK_REGISTRY.build("homie")
    cfg.env_options.num_envs = 2
    cfg.observation_options.policy.history_length = 5
    cfg.runner_options.logger = "tensorboard"
    env = RslRlVecEnvWrapper.from_config(cfg)
    with pytest.raises(ValueError, match="400-wide tensor has no mirror map"):
        make_rsl_rl_runner(env)


_MIRROR = "eden.tasks.homie.symmetry:mirror_observations_and_actions"


@pytest.mark.parametrize(
    "field",
    [{"symmetry_cfg": {"use_data_augmentation": True}}, {"normalize_advantage_per_mini_batch": True}],
    ids=lambda f: next(iter(f)),
)
def test_options_refuse_ppo_features_himppo_lacks(field):
    with pytest.raises(ValueError, match=next(iter(field))):
        HIMPPOAlgorithmOptions(data_augmentation_func=_MIRROR, **field)


def test_options_require_the_task_mirror_as_an_import_path():
    with pytest.raises(ValueError, match="data_augmentation_func"):
        HIMPPOAlgorithmOptions()
    with pytest.raises(ValueError, match="'module:function' import path"):
        HIMPPOAlgorithmOptions(data_augmentation_func="mirror_observations_and_actions")


def test_building_him_leaves_torch_distributions_callable_and_unvalidated(wrapped_env, tmp_path):
    """OpenHomie assigns ``Normal.set_default_validate_args = False``, which breaks every later rsl-rl model."""
    make_rsl_rl_runner(wrapped_env, log_dir=str(tmp_path))
    torch.distributions.Normal.set_default_validate_args(False)  # what rsl-rl's own models call
    assert torch.distributions.Normal._validate_args is False


def test_actor_mirror_loss_pulls_only_the_policy_on_the_mirrored_observation(wrapped_env, tmp_path):
    alg = make_rsl_rl_runner(wrapped_env, log_dir=str(tmp_path)).alg
    obs = torch.randn_like(wrapped_env.get_observations()[alg.actor_obs_group])
    params = [p for p in alg.actor_critic.parameters() if p.requires_grad]

    def grads(loss):
        return torch.autograd.grad(loss, params, allow_unused=True, retain_graph=True)

    mirrored = alg.actor_critic.act_inference(alg._mirror_obs(alg.actor_obs_group, obs))
    action = alg.actor_critic.act_inference(obs)
    detached = alg.symmetry_scale * torch.square(mirrored - alg._mirror_actions(action).detach()).sum(-1).mean()
    both_ends = alg.symmetry_scale * torch.square(mirrored - alg._mirror_actions(action)).sum(-1).mean()

    got = grads(alg._actor_symmetry_loss(obs))
    for g, want in zip(got, grads(detached)):
        assert (g is None) == (want is None)
        if g is not None:
            torch.testing.assert_close(g, want)
    assert any(g is not None and not torch.allclose(g, w) for g, w in zip(got, grads(both_ends)) if w is not None), (
        "the target branch must carry no gradient"
    )


def test_an_env_that_fell_over_on_the_step_it_timed_out_is_not_bootstrapped(wrapped_env, tmp_path):
    """HIMPPO adds ``gamma * V`` to the reward of every env in ``extras["time_outs"]`` whatever ``dones`` says, so an
    env the wrapper also reported as timed out would have its fall valued as if the episode went on.
    Env 0 is tipped past homie's ``fell_over`` angle and every env times out: only envs 1-3 are bootstrapped.
    """
    alg = make_rsl_rl_runner(wrapped_env, log_dir=str(tmp_path)).alg
    env = wrapped_env.unwrapped
    on_its_side = torch.tensor([[math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]], device=env.device)

    # One rollout step as the runner takes it, under inference mode: once a rollout has run there, the env's control
    # buffers are inference tensors that cannot be written outside it. The reset starts every env upright, wherever
    # earlier tests left the shared env, so only env 0 falls.
    with torch.inference_mode():
        wrapped_env.reset()
        actions = alg.act(wrapped_env.get_observations())
        values = alg.transition.values.clone()
        env.entities["robot"].set_quat(on_its_side, envs_idx=torch.tensor([0], device=env.device))
        env.episode_length_buf[:] = env.max_episode_length - 1
        obs, rewards, dones, extras = wrapped_env.step(actions)
        assert dones.tolist() == [1, 1, 1, 1] and extras["time_outs"].tolist() == [False, True, True, True]
        alg.process_env_step(obs, rewards, dones, extras)

    bootstrapped = torch.tensor([0.0, 1.0, 1.0, 1.0], device=env.device)
    stored = alg.storage.rewards[0, :, 0]  # the step's own transition; its mirror is row 1
    torch.testing.assert_close(stored, rewards + alg.gamma * values[:, 0] * bootstrapped)


_RSL_MAJOR = detect_rsl_rl_major_version()


def _needs(major: int, why: str):
    return pytest.mark.skipif(_RSL_MAJOR < major, reason=f"rsl-rl-lib {_RSL_MAJOR}: {why}")


def _runner_cfg(**edits):
    cfg = TASK_REGISTRY.build("homie")
    cfg.env_options.num_envs = 2
    cfg.runner_options.logger = "tensorboard"
    for path, value in edits.items():
        obj, *attrs, last = path.split(".")
        target = getattr(cfg, obj)
        for attr in attrs:
            target = getattr(target, attr)
        setattr(target, last, value)
    return cfg


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        pytest.param(
            {"runner_options.actor.distribution_cfg.std_type": "log"},
            r"std_type='log'.*supports only 'scalar'",
            id="std_type",
        ),
        pytest.param(
            {"runner_options.actor.distribution_cfg.std_range": (0.05, 2.0)},
            r"std_range=\(0.05, 2.0\)",
            id="std_range",
            marks=_needs(5, "the v3/v4 translators drop std_range, so no algorithm sees it"),
        ),
        pytest.param(
            {"runner_options.critic.activation": "tanh"},
            r"critic.activation='tanh'.*supports only 'elu'",
            id="critic_activation",
            marks=_needs(4, "v3 has one activation for actor and critic"),
        ),
        pytest.param(
            {"env_options.record_final_observations": False},
            r"record_final_observations=True",
            id="final_observations",
            marks=_needs(4, "v3 hands the algorithm no env; the first rollout step refuses it instead"),
        ),
    ],
)
def test_settings_himppo_cannot_honour_are_refused_at_construction(edit, message):
    env = RslRlVecEnvWrapper.from_config(_runner_cfg(**edit))
    with pytest.raises(ValueError, match=message):
        make_rsl_rl_runner(env)


@_needs(4, "v3's runner loads model_state_dict itself, and Eden's v3 format error is the right one there")
def test_a_checkpoint_that_is_not_this_himppos_fails_loudly_through_edens_loader(wrapped_env, tmp_path):
    runner = make_rsl_rl_runner(wrapped_env, log_dir=str(tmp_path))
    good = runner.alg.save()

    stock = tmp_path / "stock_ppo.pt"
    torch.save({"actor_state_dict": {}, "critic_state_dict": {}, "iter": 0}, stock)
    with pytest.raises(ValueError, match="Not a HIMPPO checkpoint"):
        make_rsl_rl_runner(wrapped_env, checkpoint=str(stock), log_dir=str(tmp_path / "a"))

    # Eden's loader passes strict=False for stock PPO's legacy layouts; HIM still refuses a partial match.
    partial = dict(
        good, actor_critic_state_dict={k: v for k, v in good["actor_critic_state_dict"].items() if k != "std"}
    )
    with pytest.raises(ValueError, match=r"missing \['std'\]"):
        runner.alg.load(partial, strict=False)
