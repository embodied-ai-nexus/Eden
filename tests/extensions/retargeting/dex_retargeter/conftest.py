"""One scene with a fixed-base XHand1 (``robot``) and a MANO hand (``mano``) for the dex-retargeter tests."""

import pytest

import eden as en
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import PlaneOptions
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env


@pytest.fixture(scope="module")
def env():
    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=1),
            scene_options=SceneOptions(
                robot=en.robots.XHand1_R(is_fixed_base=True, default_root_pos=(0.0, 0.0, 0.1)),
                mano=en.robots.ManoHand_R(is_fixed_base=True, default_root_pos=(0.5, 0.0, 0.1)),
                floor=PlaneOptions(),
            ),
        ),
        reset=True,
    )
