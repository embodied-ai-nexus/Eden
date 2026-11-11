"""The locomotion observation terms in ``eden/managers/terms/observations/locomotion.py``."""

from __future__ import annotations

import pytest
import torch

from eden.managers.terms.observations.locomotion import base_height
from eden.envs.base import env_class_for_config
from eden.managers.command_manager import COMMAND_TERM_REGISTRY, CommandTerm
from eden.managers.terms.observations.locomotion import VelocityTrackingErrorIntegral
from eden.options import BoxOptions, EnvOptions, SceneOptions
from eden.options.managers.commands import CommandManagerOptions
from eden.utils.configs import EdenRLConfig

pytestmark = pytest.mark.requires_genesis


def _ground_under(env, name):
    xy = env.entities[name].get_pos()[:, :2]
    return env.entities["terrain"].get_height(xy.reshape(-1, 1, 1, 2)).reshape(-1)


def test_base_height_is_the_world_z_without_a_terrain(sloped_env):
    torch.testing.assert_close(base_height(sloped_env, entity_name="box"), torch.full((2, 1), 2.0))


def test_base_height_reads_the_frame_relative_asks_for(sloped_env):
    """The solver frame sits on the probe's CoM, (1.0 kg * 0.1 m + 0.1 kg * 0.3 m) / 1.1 kg above its origin."""
    authored = base_height(sloped_env, entity_name="probe", relative=True)
    solver = base_height(sloped_env, entity_name="probe")
    torch.testing.assert_close(authored, torch.full((2, 1), 2.0))
    torch.testing.assert_close(solver - authored, torch.full((2, 1), 0.13 / 1.1))


def test_base_height_with_a_terrain_is_the_height_above_the_ground_under_the_base(sloped_env):
    world = base_height(sloped_env, entity_name="box")
    relative = base_height(sloped_env, entity_name="box", terrain_entity_name="terrain")
    torch.testing.assert_close(relative, world - _ground_under(sloped_env, "box").unsqueeze(-1))
    assert (relative - world).abs().min() > 0.5  # the slope makes the two framings disagree


@pytest.mark.parametrize(
    ("terrain_entity_name", "error", "match"),
    [
        ("nope", KeyError, "No scene entity named 'nope'.*terrain"),
        # The plausible mistake: an entity that is not a Terrain fails by name, not as an AttributeError.
        ("box", TypeError, "terrain_entity_name='box'.*no get_height.*must be a Terrain"),
    ],
)
def test_base_height_rejects_a_bad_terrain_entity(sloped_env, terrain_entity_name, error, match):
    with pytest.raises(error, match=match):
        base_height(sloped_env, entity_name="box", terrain_entity_name=terrain_entity_name)


# ---------------------------------------------------------------------------
# VelocityTrackingErrorIntegral: the worked example of a stateful observation term
# ---------------------------------------------------------------------------

VEL_ENVS = 3


@COMMAND_TERM_REGISTRY.register()
class ScriptedVelocityCommand(CommandTerm):
    """A ``[vx, vy, wz]``-shaped command (``dim`` wide) the test writes; never resampled."""

    dim: int = 3
    resampling_time_range = (1.0e9, 1.0e9)

    def build(self) -> None:
        self._command = torch.zeros(self.num_envs, self.dim, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, envs_idx) -> None: ...

    def _update_command(self) -> None: ...


def _scene() -> SceneOptions:
    return SceneOptions(
        robot=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.0, 0.0, 0.5)),
        spare=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(1.0, 0.0, 0.5)),
    )


def _build(config, **kwargs):
    env = env_class_for_config(config).from_config(config, show_viewer=False, **kwargs)
    env.build()
    env.reset()
    return env


@pytest.fixture(scope="module")
def rl_env():
    """Two boxes (``robot``, ``spare``) and two scripted commands, three columns wide and two."""
    return _build(
        EdenRLConfig(
            env_options=EnvOptions(num_envs=VEL_ENVS),
            scene_options=_scene(),
            command_options=CommandManagerOptions(
                velocity_command=ScriptedVelocityCommand.configure(), narrow=ScriptedVelocityCommand.configure(dim=2)
            ),
        )
    )


def _integral(env, **kwargs) -> VelocityTrackingErrorIntegral:
    term = VelocityTrackingErrorIntegral(env=env, options=VelocityTrackingErrorIntegral.configure(**kwargs))
    term.build()
    return term


@pytest.fixture
def command(rl_env):
    """The scripted velocity command, zeroed after the test."""
    buffer = rl_env.command_manager.get_term("velocity_command").command
    yield buffer
    buffer.zero_()


def test_the_velocity_error_integral_integrates_per_tick_clamps_and_resets_per_env(rl_env, command):
    """At rest the error is the command itself: ``command * dt`` per tick, up to ``max_abs``."""
    command[:] = torch.tensor([1.0, 0.0, 0.5])
    term = _integral(rl_env, max_abs=0.05)

    for k in range(1, 4):
        term.post_physics_refresh()
        expected = (torch.tensor([1.0, 0.0, 0.5]) * k * rl_env.dt).clamp(-0.05, 0.05)
        torch.testing.assert_close(term.compute(), expected.expand(VEL_ENVS, 3), msg=f"tick {k}")

    for _ in range(20):  # anti-windup: a persistent error saturates
        term.post_physics_refresh()
    torch.testing.assert_close(term.compute(), torch.tensor([[0.05, 0.0, 0.05]]).expand(VEL_ENVS, 3))

    term.reset(envs_idx=torch.tensor([1]))
    assert term.compute()[1].tolist() == [0.0, 0.0, 0.0]
    torch.testing.assert_close(term.compute()[[0, 2]], torch.tensor([[0.05, 0.0, 0.05]]).expand(2, 3))


def test_the_velocity_error_integral_reads_whatever_entity_the_env_holds_at_each_tick(rl_env, command):
    """Resolved in the hook, not cached at build: a deployment swaps ``rl_env.entities['robot']`` under it."""
    term = _integral(rl_env)
    spare = rl_env.entities["spare"]
    velocity = torch.zeros(VEL_ENVS, 6)
    velocity[:, 0] = 2.0
    spare.set_dofs_vel(velocity)
    original = rl_env.entities["robot"]
    rl_env.entities["robot"] = spare
    try:
        term.post_physics_refresh()
    finally:
        rl_env.entities["robot"] = original
        rl_env.reset()

    torch.testing.assert_close(term.compute(), torch.tensor([[-2.0 * rl_env.dt, 0.0, 0.0]]).expand(VEL_ENVS, 3))


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        (dict(command_name="nope"), KeyError, "nope"),
        (dict(command_name="narrow"), ValueError, r"publishes 2 component\(s\).*needs at least 3"),
        (dict(max_abs=0.0), ValueError, "max_abs must be positive"),
    ],
    ids=["unknown-command", "narrow-command", "non-positive-clamp"],
)
def test_a_bad_setup_fails_at_build(rl_env, kwargs, error, message):
    """A narrow command would otherwise surface on the first tick as a broadcasting error."""
    with pytest.raises(error, match=message):
        _integral(rl_env, **kwargs)


def test_an_env_without_a_command_manager_is_refused_at_build(sloped_env):
    """``None`` on an RL env with no command terms; no attribute at all on a plain ``EnvBase``: one message."""
    rl_without_commands = _build(EdenRLConfig(env_options=EnvOptions(num_envs=VEL_ENVS), scene_options=_scene()))
    for env in (rl_without_commands, sloped_env):
        with pytest.raises(ValueError, match="requires a command manager"):
            _integral(env)
