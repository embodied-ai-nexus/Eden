"""FlashSAC behind Eden's rsl_rl wrapper: the env contract it needs, and a few real off-policy iterations."""

from __future__ import annotations

import sys

import pytest
import torch

from eden.learning.wrappers.rsl_rl.flashsac import (
    RslRlFlashSacAlgorithmOptions,
    RslRlFlashSacCriticOptions,
    RslRlFlashSacRunnerOptions,
)
from eden.learning.wrappers.rsl_rl.flashsac.runner import _import_runner
from eden.tasks import TASK_REGISTRY
from eden.utils.distribution import pip_install_extra, uv_sync_extra
from tests.envs.conftest import PENDULUM_MJCF
from tests.envs.wrappers.conftest import ending, gym_pendulum_config

# rsl-rl-flashsac is not on PyPI, so the `flashsac` extra does not bring it: these skip unless the group is synced.
NEEDS_FLASHSAC = f"rsl_rl_flashsac is not installed: {uv_sync_extra('flashsac')}"

pytestmark = [pytest.mark.requires_genesis]

NUM_ENVS = 4


def _reacher_config(record_final_observations: bool = True, **runner):
    """The reacher on FlashSAC, with 0.2 s episodes so time-outs come quickly."""
    cfg = TASK_REGISTRY.build("reacher")
    cfg.env_options.num_envs = NUM_ENVS
    cfg.env_options.episode_length_s = 0.2
    cfg.env_options.record_final_observations = record_final_observations
    cfg.runner_options = RslRlFlashSacRunnerOptions(
        device="cpu",
        log_interval=1,
        logger="tensorboard",
        critic=RslRlFlashSacCriticOptions(hidden_dim=32),
        algorithm=RslRlFlashSacAlgorithmOptions(
            replay_buffer_size=64, buffer_min_length=8, mini_batch_size=8, use_compile=False, use_amp=False
        ),
        **{"max_iterations": 3, **runner},
    )
    return cfg


def _reacher(record_final_observations: bool = True, **runner):
    from eden.learning.wrappers.rsl_rl.flashsac.runner import FlashSacVecEnvWrapper

    return FlashSacVecEnvWrapper.from_config(_reacher_config(record_final_observations, **runner))


@pytest.fixture(scope="module")
def env():
    return _reacher()


def test_an_env_that_does_not_record_final_observations_is_refused():
    with pytest.raises(ValueError, match="set `env_options.record_final_observations=True`"):
        _reacher(record_final_observations=False)


def test_a_timed_out_env_hands_flashsac_its_last_observation_not_the_next_episodes_first(env):
    env.reset()
    for _ in range(50):
        obs, _, _, extras = env.step(torch.zeros(NUM_ENVS, env.num_actions))
        timed_out = extras["time_outs"].bool()
        if timed_out.any():
            break
    assert timed_out.any(), "no env timed out within 50 steps of a 0.2 s episode"
    last = extras["time_outs_obs"]["policy"][timed_out]
    assert not torch.allclose(last, obs["policy"][timed_out])  # obs is already the reset one
    torch.testing.assert_close(last, extras["final_observations"]["policy"][timed_out])


@pytest.fixture(scope="module")
def pendulum_file(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("assets") / "pendulum.xml"
    path.write_text(PENDULUM_MJCF)
    return str(path)


def test_an_env_that_failed_on_the_step_it_timed_out_is_stored_as_terminated_not_truncated(pendulum_file, tmp_path):
    """FlashSAC stores ``terminated = dones - time_outs`` and bootstraps a truncated transition from its final
    observation, so an env the wrapper also reported as timed out would have its failure valued as if the episode
    went on. Env 0 fails and both time out: only env 1 is truncated.
    """
    pytest.importorskip("rsl_rl_flashsac", reason=NEEDS_FLASHSAC)
    from eden.learning.wrappers.rsl_rl.flashsac.runner import FlashSacVecEnvWrapper, make_flashsac_runner

    config = gym_pendulum_config(pendulum_file, num_envs=2)
    config.env_options.record_final_observations = True
    config.runner_options = RslRlFlashSacRunnerOptions(
        device="cpu",
        logger="tensorboard",
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        critic=RslRlFlashSacCriticOptions(hidden_dim=32),
        # One-step returns: the step is stored as soon as it is processed, under its own flags.
        algorithm=RslRlFlashSacAlgorithmOptions(
            n_steps=1, replay_buffer_size=8, buffer_min_length=2, mini_batch_size=2, use_compile=False, use_amp=False
        ),
        max_iterations=1,
    )
    env = FlashSacVecEnvWrapper.from_config(config)
    alg = make_flashsac_runner(env, log_dir=str(tmp_path)).alg

    actions = alg.act(env.get_observations())
    with ending(env.unwrapped, [True, False]):
        env.unwrapped.episode_length_buf[:] = env.max_episode_length - 1
        obs, rew, dones, extras = env.step(actions)
    alg.process_env_step(obs, rew, dones, extras)

    stored = alg.replay_buffer.sample(sample_idxs=torch.arange(2))
    assert stored["terminated"].tolist() == [1.0, 0.0]
    assert stored["truncated"].tolist() == [0.0, 1.0]
    # The truncated env bootstraps from its last observation, not the next episode's first.
    torch.testing.assert_close(stored["next_observation"]["policy"][1], extras["final_observations"]["policy"][1])


def test_actions_are_bounded_to_the_unit_box_by_default(env):
    torch.testing.assert_close(env.action_bias, torch.zeros(env.num_actions))
    torch.testing.assert_close(env.action_scale, torch.ones(env.num_actions))


def test_the_off_policy_runner_fills_its_buffer_and_updates(env, tmp_path):
    pytest.importorskip("rsl_rl_flashsac", reason=NEEDS_FLASHSAC)
    from eden.learning.wrappers.rsl_rl.flashsac.runner import make_flashsac_runner

    runner = make_flashsac_runner(env, log_dir=str(tmp_path))
    actor_before = [p.detach().clone() for p in runner.alg.actor.parameters()]
    runner.learn(num_learning_iterations=12)
    assert len(runner.alg.replay_buffer) >= 12 * NUM_ENVS - 8  # every step stored (n-step returns hold back a few)
    changed = any(not torch.equal(a, b) for a, b in zip(actor_before, runner.alg.actor.parameters()))
    assert changed, "the actor never updated once the buffer passed buffer_min_length"
    assert list(tmp_path.glob("model_*.pt"))


def test_the_run_length_in_env_steps_is_sized_for_the_env_count_and_actions_are_scaled_to_the_bound():
    """``num_env_steps`` becomes iterations only once the env count is known (``--num_envs`` is applied after the
    task's modifiers); the policy's [-1, 1] actions reach the env scaled by ``action_bound_scale``.
    """
    env = _reacher(max_iterations=None, num_env_steps=40, action_bound_scale=0.5)
    assert env.runner_options.max_iterations == 40 // NUM_ENVS
    env.reset()
    env.step(torch.ones(NUM_ENVS, env.num_actions))
    torch.testing.assert_close(env.unwrapped.action_manager.action, torch.full((NUM_ENVS, env.num_actions), 0.5))


def test_inference_loads_a_flashsac_checkpoint_into_a_runner_without_the_training_replay_buffer(env, tmp_path):
    """``eden inference`` builds the off-policy runner for a FlashSAC config (not rsl_rl's on-policy one) and loads
    the checkpoint strictly, with a replay buffer of one row per env instead of ``replay_buffer_size``.
    """
    pytest.importorskip("rsl_rl_flashsac", reason=NEEDS_FLASHSAC)
    from eden.learning.wrappers.rsl_rl.flashsac.runner import FlashSacVecEnvWrapper, make_flashsac_runner
    from eden.utils.cli.tasks import _rsl_rl_env_and_runner

    trained = make_flashsac_runner(env, log_dir=str(tmp_path))
    trained.learn(num_learning_iterations=12)
    checkpoint = tmp_path / "trained.pt"
    trained.save(str(checkpoint))

    play_env, runner = _rsl_rl_env_and_runner(_reacher_config(), str(checkpoint), inference=True, eval_mode=True)
    assert isinstance(play_env, FlashSacVecEnvWrapper)
    assert runner.alg.replay_buffer._max_length == NUM_ENVS
    for loaded, saved in zip(runner.alg.actor.parameters(), trained.alg.actor.parameters()):
        torch.testing.assert_close(loaded, saved)
    obs = play_env.get_observations()
    policy = runner.get_inference_policy(device=play_env.device)
    assert policy(obs).abs().max() <= 1.0


def test_the_flashsac_runner_keeps_the_newest_checkpoints_and_writes_progress(tmp_path):
    """The features Eden adds to every rsl_rl runner reach FlashSAC's: ``keep_last_checkpoints`` (a 500M-step run
    saving every 5,000 iterations would otherwise keep all 24 checkpoints, 598 MB each) and ``progress.json``.
    """
    pytest.importorskip("rsl_rl_flashsac", reason=NEEDS_FLASHSAC)
    import json

    from eden.learning.wrappers.rsl_rl.flashsac.runner import make_flashsac_runner

    runner = make_flashsac_runner(_reacher(save_interval=2, keep_last_checkpoints=1), log_dir=str(tmp_path))
    runner.learn(num_learning_iterations=8)
    assert sorted(p.name for p in tmp_path.glob("model_*.pt")) == ["model_7.pt"]  # saved at 2, 4, 6 and 7
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert (progress["status"], progress["iteration"], progress["total_iterations"]) == ("completed", 7, 8)


def test_a_flashsac_checkpoint_is_refused_by_eden_deploy(env):
    """The deploy loop bypasses the wrapper that scales FlashSAC's tanh actions, so it would drive the robot with
    unscaled ones.
    """
    from eden.utils.cli.tasks import _load_checkpoint_slots

    with pytest.raises(SystemExit, match="Deploying a FlashSAC policy"):
        _load_checkpoint_slots(env, None, ["model_7.pt"])


def _mimic_config(library, **env_options):
    """A G1 mimic config on FlashSAC whose one motion term mines hard motions."""
    import eden as en
    from eden.options.envs import EnvOptions
    from eden.options.robots.unitree_g1 import G1Dofs29
    from eden.utils.configs import EdenMimicConfig
    from tests.support.motion import manager_options, term_options

    return EdenMimicConfig(
        env_options=EnvOptions(num_envs=NUM_ENVS, record_final_observations=True, **env_options),
        scene_options=en.scenes.Empty(robot=G1Dofs29()),
        motion_options=manager_options(
            terms={"default": term_options(library, entities=("robot",), adaptive_motion_weights=True)}
        ),
        runner_options=RslRlFlashSacRunnerOptions(device="cpu", max_iterations=3),
    )


@pytest.fixture(scope="module")
def mining_library(tmp_path_factory):
    from tests.support.motion import write_library

    return write_library(tmp_path_factory.mktemp("mining"), [50, 50])


def test_training_on_motions_mined_by_a_curriculum_is_refused_before_the_env_is_built(mining_library, monkeypatch):
    """FlashSAC's runner never steps the motion-weight curriculum, so the weights would stay at their initial values."""
    from eden.envs import base
    from eden.learning.wrappers.rsl_rl.flashsac.runner import FlashSacVecEnvWrapper

    monkeypatch.setattr(base, "env_class_for_config", lambda cfg: pytest.fail("an env was made before the refusal"))
    with pytest.raises(ValueError, match=r"Set `motion_options\.default\.adaptive_motion_weights=False`"):
        FlashSacVecEnvWrapper.from_config(_mimic_config(mining_library))


def test_inference_runs_a_config_whose_motions_were_mined(mining_library):
    """No curriculum runs at inference, so a checkpoint trained with one still replays (``eden inference``)."""
    from eden.learning.wrappers.rsl_rl.flashsac.runner import FlashSacVecEnvWrapper

    play_env = FlashSacVecEnvWrapper.from_config(_mimic_config(mining_library), eval_mode=True)
    assert play_env.unwrapped.motion_manager.has_adaptive_motion_weights


def test_a_missing_backend_names_both_install_commands(monkeypatch):
    """rsl-rl-flashsac is not on PyPI, so `pip install 'eden-world[flashsac]'` alone does not bring it."""
    for module in ("rsl_rl_flashsac", "rsl_rl_flashsac.runners"):  # an earlier test may have imported both
        monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(ImportError) as info:
        _import_runner()
    assert pip_install_extra("flashsac") in str(info.value)
    assert uv_sync_extra("flashsac") in str(info.value)
