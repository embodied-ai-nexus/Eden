"""Build real environments for tests.

Tests exercise Eden against real built scenes rather than stand-in envs (see
``docs/guides/testing.md``). A scene build costs seconds, so build once per module
with a ``scope="module"`` fixture and share it; a test that mutates the env either
restores what it changed or calls ``env.reset()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from eden.envs.base import EnvBase
    from eden.utils.configs import EdenConfig


def build_env(config: EdenConfig, *, reset: bool = False, eval_mode: bool = False) -> EnvBase:
    """Construct the env class ``config`` pairs with, build it, and optionally reset it.

    Parameters
    ----------
    config : EdenConfig
        Any config leaf; ``env_class_for_config`` picks the matching env class.
    reset : bool
        Call ``env.reset()`` after ``build()`` — needed before ``step()`` on an RL env.
    eval_mode : bool
        Build as ``eden inference`` does (``env.training`` False).

    Returns
    -------
    EnvBase
        The built env, viewer off.
    """
    from eden.envs.base import env_class_for_config

    env = env_class_for_config(config).from_config(config, show_viewer=False, eval_mode=eval_mode)
    env.build()
    if reset:
        env.reset()
    return env


def control_force_over_one_step(env: EnvBase, entity) -> torch.Tensor:
    """``entity``'s control force over one physics step from the current state, the rigid state put back after it.

    Genesis reports the control force the last step applied (genesis-world#3642), so reading what a target commands
    takes a step. The coordinates and velocities of every rigid body are restored, so a shared fixture env stays where
    its setters put it, and through the setters rather than ``set_state``, which would also zero every control. The
    step still counts on the scene's clock.
    """
    solver = env.rigid_solver
    state = solver.get_state()
    env.scene.step()
    force = entity.get_dofs_control_force()
    solver.set_qpos(state.qpos)
    solver.set_dofs_velocity(state.dofs_vel)
    return force
