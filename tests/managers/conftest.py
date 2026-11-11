"""Shared scenes for the manager tests: one free box in a real, built RL env.

Managers are constructed against a real env. A test that needs a manager with its own options
constructs it standalone against ``box_env`` (``RewardManager(env=box_env, options=...)``); a test that
needs the manager wired into the env's own step builds a module-scoped env with ``make_box_env``.
"""

from __future__ import annotations

import pytest
import torch

NUM_ENVS = 4


def _build_box_env(*, num_envs: int = NUM_ENVS, eval_mode: bool = False, **config_options):
    from eden.envs.base import env_class_for_config
    from eden.options import BoxOptions, EnvOptions, SceneOptions
    from eden.utils.configs import EdenRLConfig

    env_options = config_options.pop("env_options", None) or EnvOptions(num_envs=num_envs)
    config = EdenRLConfig(
        env_options=env_options,
        scene_options=SceneOptions(box=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.0, 0.0, 0.5))),
        **config_options,
    )
    env = env_class_for_config(config).from_config(config, show_viewer=False, eval_mode=eval_mode)
    env.build()
    env.reset()
    return env


@pytest.fixture(scope="session")
def make_box_env():
    """Return ``make(*, num_envs=4, eval_mode=False, **config_options)`` -> a built, reset RL env.

    ``config_options`` are ``EdenRLConfig`` fields (``observation_options=...``, ``env_options=...``).
    Call it from a ``scope="module"`` fixture: every call is a scene build.
    """
    return _build_box_env


@pytest.fixture(scope="module")
def box_env():
    """A built RL env (training) with one free 10 cm box named ``box`` and no manager terms."""
    return _build_box_env()


@pytest.fixture(scope="module")
def eval_box_env():
    """The same scene built with ``eval_mode=True``, so ``env.training`` is False."""
    return _build_box_env(eval_mode=True)


@pytest.fixture
def set_box_heights():
    """Return ``set(env, heights)``: teleport env ``i``'s box to ``z = heights[i]`` through the real setter."""

    def set_heights(env, heights) -> None:
        pos = torch.zeros(env.num_envs, 3, device=env.device)
        pos[:, 2] = torch.as_tensor(heights, dtype=pos.dtype, device=env.device)
        env.entities["box"].set_pos(pos)

    return set_heights


class _NoErrorSolver:
    """Genesis' per-env errno getter, all clear: what ``rigid_solver.get_error_envs_mask()`` returns on a healthy
    scene. The real getter is a sync-free device compare of the flags latched at the end of each step.
    """

    def __init__(self, num_envs: int, device: torch.device):
        self._errno_latched = torch.zeros(num_envs, dtype=torch.int32, device=device)

    def get_error_envs_mask(self) -> torch.Tensor:
        return self._errno_latched > 0


class _CudaEnv:
    """The env surface a manager or term under a GPU-sync budget reads (``num_envs``, ``device``,
    ``entities``, ``training``, ``episode_length_buf``, ``dt``, ``max_episode_length_s``,
    ``get_solver_error_envs_mask()``, ``register_post_build_hook``, which queues the hook for the test to run,
    and the ``compute_once`` memo its term loop opens), on CUDA.

    A stand-in, not a built env, and only for tests that count syncs (``docs/guides/testing.md``): the
    suite initialises Genesis on the CPU (``tests/conftest.py``), so no built env has a CUDA device, and
    a GPU->CPU sync exists only on one. What the unit computes belongs to a real-env test beside it.
    """

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.device = torch.device("cuda")
        self.entities: dict = {}
        self.training = True
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.dt = 0.02
        self.max_episode_length_s = 20.0
        self.rigid_solver = _NoErrorSolver(num_envs, self.device)
        self.post_build_hooks: list = []

    def register_post_build_hook(self, hook) -> None:
        self.post_build_hooks.append(hook)

    def get_solver_error_envs_mask(self) -> torch.Tensor:
        return self.rigid_solver.get_error_envs_mask()

    # The env's own memo, not a stand-in for it: a manager's term loop runs inside it.
    _pass_cache = None

    def compute_once(self, key, producer):
        from eden.envs.base import EnvBase

        return EnvBase.compute_once(self, key, producer)

    def compute_pass(self):
        from eden.envs.base import EnvBase

        return EnvBase.compute_pass(self)


@pytest.fixture
def cuda_stand_in_env():
    """A ``_CudaEnv`` with ``NUM_ENVS`` envs; skips without CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("a GPU-sync budget needs CUDA")
    return _CudaEnv(NUM_ENVS)


@pytest.fixture
def flag_solver_error():
    """Return ``flag(env, envs)``: mark those envs as Genesis marks a corrupt one once a step is over (here
    ``INVALID_CONTACT_NAN``), which ``rigid_solver.get_error_envs_mask()`` reports. Cleared on teardown.
    """
    from genesis.utils.array_class import ErrorCode

    flagged = []

    def flag(env, envs) -> None:
        env.rigid_solver.flag_errno(int(ErrorCode.INVALID_CONTACT_NAN), envs)
        flagged.append(env)

    yield flag
    for env in flagged:
        env.rigid_solver.clear_errno()
