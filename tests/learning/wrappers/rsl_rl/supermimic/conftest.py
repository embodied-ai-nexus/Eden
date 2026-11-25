"""The suite's supermimic library and env (``tests/support/supermimic.py``) under the names these tests use, and
runners with networks small enough to train on the CPU.
"""

import pytest

from tests.support.supermimic import supermimic_config


def small_learner(cfg):
    """``cfg`` with the learner on the CPU and its networks, batches and horizons cut to a few units."""
    o = cfg.runner_options
    o.device, o.logger, o.num_steps_per_env, o.buffer_size = "cpu", "tensorboard", 4, 32
    o.world_model.hidden_dims, o.world_model.batch_size, o.world_model.num_batches = (32,), 8, 2
    o.policy.actor_hidden_dims, o.policy.critic_hidden_dims, o.policy.batch_size = (32,), (16,), 8
    o.policy.horizon, o.replay_ratio = 4, 64.0
    return cfg


@pytest.fixture(scope="session")
def lafan1_library(supermimic_library):
    return supermimic_library


@pytest.fixture(scope="session")
def env(supermimic_env):
    return supermimic_env


def paper_learner(cfg):
    """``cfg`` with the options that default to BlinkMimic's recipe at the SuperMimic paper's values: no actor
    average, reference action, mirrored batches, failure resets or learning-rate multiplier.
    """
    o = cfg.runner_options
    o.policy.ema_decay, o.policy.reference_action, o.policy.lr_multiplier = 0.0, False, 1.0
    o.world_model.mirror_share = 0.0
    o.return_sampler.failure, o.return_sampler.explore, o.return_sampler.keep = 0.0, 0.5, 0.1
    return cfg


@pytest.fixture(scope="module")
def runner(supermimic_library, tmp_path_factory):
    """A runner of the task's default options, with small networks."""
    from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper, make_rsl_rl_runner

    cfg = small_learner(supermimic_config(supermimic_library))
    return make_rsl_rl_runner(RslRlVecEnvWrapper.from_config(cfg), log_dir=str(tmp_path_factory.mktemp("run")))
