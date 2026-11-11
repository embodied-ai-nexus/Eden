"""``ContrackVecEnvWrapper``, the CMDP-PPO seam, against a real built ConTrack env (no backend needed)."""

import pytest
import torch

from eden.learning.wrappers.rsl_rl_contrack import RslRlContrackRunnerOptions
from eden.learning.wrappers.rsl_rl_contrack.env import CMDP_GROUPS, ContrackVecEnvWrapper
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import BoxOptions
from eden.options.managers.rewards import RewardGroupOptions, RewardManagerOptions, RewardTermOptions
from eden.tasks import TASK_REGISTRY
from eden.utils.configs import EdenRLConfig
from tests.support.envs import build_env

pytestmark = pytest.mark.requires_genesis


@pytest.fixture(scope="module")
def wrapper():
    cfg = TASK_REGISTRY.build("contrack")
    cfg.env_options.num_envs = 2
    return ContrackVecEnvWrapper(build_env(cfg), RslRlContrackRunnerOptions())


@pytest.mark.slow
def test_step_forwards_each_cmdp_group_reward_and_they_sum_to_the_reward(wrapper):
    _, reward, _, extras = wrapper.step(torch.zeros(wrapper.num_envs, wrapper.num_actions, device=wrapper.device))

    reward_manager = wrapper.unwrapped.reward_manager
    for group in CMDP_GROUPS:
        torch.testing.assert_close(extras[group], reward_manager.get_group_reward(group))
    torch.testing.assert_close(sum(extras[group] for group in CMDP_GROUPS), reward, rtol=1e-5, atol=1e-6)
    # The runner's object-tracking logs advanced from the real object and its reference.
    assert wrapper.it_step_count == 1
    assert torch.isfinite(wrapper.it_obj_pos_err_sum) and torch.isfinite(wrapper.it_obj_rot_err_sum)


def test_rejects_reward_terms_outside_exactly_the_cmdp_groups():
    """A flat term would be invisible to the CMDP critic, and ``rs``/``rp`` are missing."""
    height = RewardTermOptions(name="constant_base_height", params={"entity_name": "box", "target_height": 0.5})
    env = build_env(
        EdenRLConfig(
            env_options=EnvOptions(num_envs=1),
            scene_options=SceneOptions(box=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.0, 0.0, 0.5))),
            reward_options=RewardManagerOptions(rg=RewardGroupOptions(height=height), alive=height),
        )
    )
    with pytest.raises(
        ValueError, match=r"exactly the groups \('rg', 'rs', 'rp'\); the env declares \['default', 'rg'\]"
    ):
        ContrackVecEnvWrapper(env, RslRlContrackRunnerOptions())


@pytest.mark.slow
def test_check_for_nan_names_the_cmdp_group_that_went_nan(wrapper, monkeypatch):
    """The fork's runner has no NaN check, so the wrapper applies runner_options.check_for_nan itself."""
    reward_manager = wrapper.unwrapped.reward_manager
    real = reward_manager.get_group_reward

    def rs_goes_nan(group):
        value = real(group)
        return torch.full_like(value, float("nan")) if group == "rs" else value

    monkeypatch.setattr(reward_manager, "get_group_reward", rs_goes_nan)
    actions = torch.zeros(wrapper.num_envs, wrapper.num_actions, device=wrapper.device)
    with pytest.raises(ValueError, match=r"NaN in the env output \[\"extras\['rs'\]\"\]"):
        wrapper.step(actions)

    monkeypatch.setattr(wrapper.runner_options, "check_for_nan", False)
    wrapper.step(actions)  # off: the fork gets the NaN, as asked
