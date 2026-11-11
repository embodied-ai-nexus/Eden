"""Runner options small enough to train the ``tests/envs`` pendulum task for an iteration in a test."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from tests.envs.conftest import trace_config


def rsl_rl_runner_options(**overrides):
    """PPO runner options: an 8-unit MLP, 4 steps per env, CPU, tensorboard."""
    from eden.options.learning.rsl_rl import (
        RslRlActorOptions,
        RslRlCriticOptions,
        RslRlGaussianDistributionOptions,
        RslRlOnPolicyRunnerOptions,
        RslRlPpoAlgorithmOptions,
    )

    fields = dict(
        device="cpu",
        num_steps_per_env=4,
        logger="tensorboard",
        # rsl-rl's NaN check cannot read the task's non-concatenated (nested) critic group.
        check_for_nan=False,
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        actor=RslRlActorOptions(hidden_dims=[8], distribution_cfg=RslRlGaussianDistributionOptions(std_type="log")),
        critic=RslRlCriticOptions(hidden_dims=[8]),
        algorithm=RslRlPpoAlgorithmOptions(num_learning_epochs=1, num_mini_batches=1),
    )
    return RslRlOnPolicyRunnerOptions(**{**fields, **overrides})


@pytest.fixture
def rl_config(pendulum_file, tmp_path):
    """``rl_config(runner_options)``: the pendulum tracing task carrying ``runner_options``."""

    def make(runner_options):
        config = trace_config(pendulum_file, str(tmp_path / "recordings"))
        config.runner_options = runner_options
        return config

    return make


def gym_pendulum_config(pendulum_file: str, *, num_envs: int = 3):
    """A pendulum task for the gymnasium wrappers: ``policy`` concatenates its terms, ``critic`` does not.

    The reward is the action's squared norm, so a test can drive the return; the ``ends`` termination
    ends the envs a test marks (``EnvsTraceTermination.ending``) and ``timeout`` the rest after
    ``max_episode_length`` steps. The curriculum holds the reward weight at 1, and logs it as a float.
    """
    from eden import observations, rewards, terminations
    from eden.managers.terms.actions.joint_actions import ImplicitPDController
    from eden.managers.terms.curricula.rewards import StageRewardWeightCurriculum
    from eden.options import (
        ActionManagerOptions,
        CurriculumManagerOptions,
        EnvOptions,
        ObservationGroupOptions,
        ObservationManagerOptions,
        PlaneOptions,
        RewardManagerOptions,
        SceneOptions,
        TerminationManagerOptions,
    )
    from eden.terms import DoneTerm, RewardTerm
    from eden.utils.configs import EdenRLConfig
    from tests.envs.conftest import EnvsTraceTermination, pendulum_options

    def group(concatenate: bool):
        return ObservationGroupOptions(
            concatenate_terms=concatenate,
            dofs_pos=observations.DofsPos.configure(entity_name="pendulum"),
            dofs_vel=observations.DofsVel.configure(entity_name="pendulum"),
        )

    return EdenRLConfig(
        env_options=EnvOptions(num_envs=num_envs, decimation=2, episode_length_s=0.1),
        scene_options=SceneOptions(plane=PlaneOptions(), pendulum=pendulum_options(pendulum_file)),
        action_options=ActionManagerOptions(
            hinge=ImplicitPDController.configure(entity_name="pendulum", dofs_name=["hinge"])
        ),
        observation_options=ObservationManagerOptions(policy=group(True), critic=group(False)),
        reward_options=RewardManagerOptions(action=RewardTerm.configure(func=rewards.action_l2, weight=1.0)),
        curriculum_options=CurriculumManagerOptions(
            action_weight=StageRewardWeightCurriculum.configure(reward_term_name="action", weight_stages=[(0, 1.0)])
        ),
        termination_options=TerminationManagerOptions(
            ends=EnvsTraceTermination.configure(),
            timeout=DoneTerm.configure(func=terminations.time_out, time_out=True),
        ),
    )


def built_gym_pendulum_env(pendulum_file: str, *, num_envs: int = 3, record_final_observations: bool = True):
    """``gym_pendulum_config``'s env, built, for a wrapper to wrap; the caller destroys it."""
    from eden.envs.base import RLEnvBase

    config = gym_pendulum_config(pendulum_file, num_envs=num_envs)
    config.env_options.record_final_observations = record_final_observations
    env = RLEnvBase.from_config(config)
    env.build()
    return env


def unbuildable_gym_pendulum_config(tmp_path):
    """``gym_pendulum_config`` with a missing pendulum file: its build raises ``FileNotFoundError``.

    So a ``from_config`` that raises anything else on it raised before the build.
    """
    return gym_pendulum_config(str(tmp_path / "missing.urdf"), num_envs=1)


@contextmanager
def ending(env, mask: list[bool]):
    """End the episodes of the envs ``mask`` marks, through ``gym_pendulum_config``'s ``ends`` term, until exit."""
    import torch

    ends = env.termination_manager.get_term("ends")
    ends.ending[:] = torch.tensor(mask)
    try:
        yield
    finally:
        ends.ending[:] = False
