"""Tests for ``eden.extensions.sysid.optimizers.scipy_ls``."""

from __future__ import annotations

import pytest

from eden.extensions.sysid.optimizers.scipy_ls import SciPyLeastSquares
from eden.extensions.sysid.parameter import Parameter, ParameterSet
from eden.options import BoxOptions, EnvOptions, SceneOptions
from eden.options.extensions.sysid import SystemIdentificationOptions
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env


@pytest.fixture(scope="module")
def env():
    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=3),
            scene_options=SceneOptions(robot=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.0, 0.0, 0.6))),
        )
    )


def test_parallel_fd_refuses_more_envs_than_candidates_before_any_rollout(env):
    # One scalar parameter: the batched Jacobian writes 1 + 1 = 2 candidates, one per env, so 3 envs
    # must be refused here rather than inside the batched rollout on the optimizer's first Jacobian call.
    params = ParameterSet([Parameter("damping", "damping", dof_names=["joint"], nominal=0.5)])
    sysid = SciPyLeastSquares(env=env, options=SystemIdentificationOptions(parallel_fd=True, verbose=False))
    with pytest.raises(ValueError, match=r"requires env\.num_envs == 2 \(got 3\)"):
        sysid.fit(params, trajectories=[])
