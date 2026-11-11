"""``SkatePushFlag`` says out loud when a training run has no AMP trainer to read it.

``build()`` reads only ``env.training`` and the algorithm block of ``env.config``, so an unbuilt env is enough.
"""

from __future__ import annotations

import pytest

from eden.envs.base import RLEnvBase
from eden.tasks import TASK_REGISTRY
from eden.tasks.husky.custom_terms.observations import SkatePushFlag


@pytest.fixture(scope="module")
def unbuilt_env():
    cfg = TASK_REGISTRY.build("husky")
    cfg.env_options.num_envs = 2
    return RLEnvBase.from_config(cfg)


@pytest.fixture
def build_push_flag(unbuilt_env):
    """``build(class_name, training)`` builds a ``SkatePushFlag`` against an env with that algorithm and mode."""
    algorithm = unbuilt_env.config.runner_options.algorithm  # live: the env's own source config
    original = (algorithm.class_name, unbuilt_env.training)

    def build(class_name, training):
        algorithm.class_name = class_name
        unbuilt_env.training = training
        SkatePushFlag(unbuilt_env, SkatePushFlag.configure()).build()

    yield build
    algorithm.class_name, unbuilt_env.training = original


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]


def test_a_training_run_with_plain_ppo_warns_and_names_the_way_out(build_push_flag, eden_caplog):
    build_push_flag("PPO", training=True)

    (message,) = _warnings(eden_caplog)
    assert "AMP style reward is OFF" in message
    assert "algorithm=RslRlAmpPpoAlgorithmOptions(amp_motion_file=...)" in message
    assert "not class_name alone" in message  # AmpPPO under plain PPO options would miss its amp_* fields


@pytest.mark.parametrize(
    "class_name, training",
    [
        ("eden.learning.wrappers.rsl_rl.amp_ppo.ppo:AmpPPO", True),  # the task's default
        ("my_project.trainers:AmpPPO", True),  # a peer trainer by qualified name
        ("PPO", False),  # eval build: no algorithm is constructed
        (None, True),  # a runner block with no class_name at all: never guess
    ],
    ids=["default_amp_trainer", "qualified_peer", "eval_build", "no_class_name"],
)
def test_the_flag_stays_quiet_when_nothing_is_wrong(build_push_flag, eden_caplog, class_name, training):
    build_push_flag(class_name, training)
    assert _warnings(eden_caplog) == []
