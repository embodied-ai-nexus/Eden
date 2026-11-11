"""``OmniResetPolicy``: a pair with no released expert is refused rather than run with the peg's."""

from __future__ import annotations

import pytest

from eden.extensions.policies.omnireset.policy import OmniResetPolicy


@pytest.mark.requires_genesis
def test_a_pair_with_no_released_expert_is_refused_before_anything_downloads():
    from eden.options import EnvOptions, SceneOptions
    from eden.utils.configs import EdenConfig
    from tests.support.envs import build_env

    env = build_env(EdenConfig(env_options=EnvOptions(num_envs=1), scene_options=SceneOptions()))
    with pytest.raises(ValueError, match=r"no released expert for pair 'drawer'; pairs: \['peg'"):
        OmniResetPolicy(env=env, options=OmniResetPolicy.configure(pair="drawer"))


def test_a_task_less_scene_for_an_unknown_pair_names_the_pairs():
    with pytest.raises(ValueError, match=r"no object pair 'drawer'; pairs: \['peg'"):
        OmniResetPolicy.scene_entity_options(pair="drawer")
