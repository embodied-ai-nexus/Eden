"""Tests for ``CommandManager`` and the ``CommandTerm`` base: the resampling clock and reset stats,
``post_reset``, and ``swap_term`` / ``restore_term`` (the post-build substitution a deployment's
``command_overrides`` rides on).

The base-class tests use the tests' own ``CommandTerm`` subclasses, recording what reaches their hooks,
on the shared ``box_env``; the swap tests use a one-hinge arm driven by a real velocity command. The GPU
sync budgets build their term against ``cuda_stand_in_env`` (see ``tests/managers/conftest.py``).
"""

from __future__ import annotations

import pytest
import torch

from eden.managers.command_manager import COMMAND_TERM_REGISTRY, CommandManager, CommandTerm
from eden.managers.terms.commands.oracle import UniformSE3Command
from eden.managers.terms.commands.velocity import ConstantVelocityCommand, UniformVelocityCommand
from eden.options.managers.commands import CommandManagerOptions, CommandTermOptions
from tests.support.envs import build_env

pytestmark = pytest.mark.requires_genesis

ARM_MJCF = """
<mujoco model="arm">
  <worldbody>
    <body name="base" pos="0 0 1.0">
      <geom type="sphere" size="0.05" mass="1.0"/>
      <body name="link1">
        <joint name="j1" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="0.5"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

_RANGES = {"lin_vel_x_range": (-1.0, 1.0), "lin_vel_y_range": (-1.0, 1.0), "ang_vel_z_range": (-0.5, 0.5)}


@COMMAND_TERM_REGISTRY.register()
class DeviceOwningPoseCommand(UniformSE3Command):
    """A 7-wide command that owns something to release, as ``OperatorVelocityCommand`` owns a keyboard grab."""

    _closes = 0

    def close(self) -> None:
        type(self)._closes += 1


@COMMAND_TERM_REGISTRY.register()
class RecordingResampleCommand(CommandTerm):
    """Records every selection handed to ``_resample_command``."""

    def __init__(self, env, options):
        super().__init__(env, options)
        self.resample_calls: list[torch.Tensor] = []

    @property
    def command(self):
        return torch.zeros(self._env.num_envs, 1, device=self.device)

    def _resample_command(self, envs_idx):
        self.resample_calls.append(envs_idx)

    def _update_command(self):
        pass


@COMMAND_TERM_REGISTRY.register()
class RecordingPostResetCommand(RecordingResampleCommand):
    """Records what its ``post_reset`` hook receives, and whether it ran under inference mode."""

    def __init__(self, env, options):
        super().__init__(env, options)
        self.post_reset_calls: list = []
        self.saw_inference_mode: bool | None = None

    def post_reset(self, envs_idx):
        self.post_reset_calls.append(envs_idx)
        self.saw_inference_mode = torch.is_inference_mode_enabled()


# ---------------------------------------------------------------------------
# CommandTerm: the resampling clock, reset stats, post_reset
# ---------------------------------------------------------------------------


def _recording(env):
    return RecordingResampleCommand(env, CommandTermOptions(resampling_time_range=(1.0, 2.0)))


@pytest.mark.parametrize(
    "selector",
    [slice(None), slice(1, 3), [0, 2], torch.tensor([0, 2]), torch.tensor([True, False, True, False])],
    ids=["full-slice", "slice", "list", "index-tensor", "bool-mask"],
)
def test_resample_hands_resample_command_a_fresh_bool_mask(box_env, selector):
    """Every selector reaches ``_resample_command`` as a new ``(num_envs,)`` mask the caller's buffer is safe from."""
    term = _recording(box_env)
    expected = torch.zeros(4, dtype=torch.bool)
    expected[selector] = True
    term._resample(selector)
    (received,) = term.resample_calls
    assert received.dtype == torch.bool and torch.equal(received, expected)
    if isinstance(selector, torch.Tensor):
        before = selector.clone()
        received.logical_not_()
        assert torch.equal(selector, before)


def test_resample_skips_an_empty_selection(box_env):
    term = _recording(box_env)
    term.time_left.fill_(0.5)
    term._resample(torch.zeros(4, dtype=torch.bool))
    assert term.resample_calls == []
    assert torch.equal(term.time_left, torch.full((4,), 0.5))
    assert torch.equal(term.command_counter, torch.zeros(4, dtype=torch.long))


def test_resample_redraws_the_clock_and_counts_only_the_selected_envs(box_env):
    term = _recording(box_env)
    term.time_left.fill_(-0.1)
    mask = torch.tensor([True, False, True, False])
    term._resample(mask)
    term._resample(mask)
    assert term.command_counter.tolist() == [2, 0, 2, 0]
    assert ((term.time_left[mask] >= 1.0) & (term.time_left[mask] <= 2.0)).all()
    assert torch.equal(term.time_left[~mask], torch.full((2,), -0.1))


def test_reset_logs_each_stat_as_the_mean_over_the_reset_envs_and_clears_it(box_env):
    term = _recording(box_env)
    values = torch.tensor([0.0, 1.0, 2.0, 4.0])
    term.stats = {"err": values.clone()}
    extras = term.reset(torch.tensor([True, False, True, True]))
    torch.testing.assert_close(extras["err"], torch.tensor(2.0))
    assert term.stats["err"].tolist() == [0.0, 1.0, 0.0, 0.0]


def test_post_reset_reaches_every_term_under_inference_mode(box_env):
    """Sanitized, so it selects exactly the envs asked for; inference-moded, so hooks mutate buffers in place."""
    options = CommandManagerOptions(
        rec=RecordingPostResetCommand.configure(resampling_time_range=(1.0, 1.0)),
        plain=RecordingResampleCommand.configure(resampling_time_range=(1.0, 1.0)),  # the inherited no-op hook
    )
    manager = CommandManager(env=box_env, options=options)
    term = manager.get_term("rec")
    manager.post_reset(None)
    manager.post_reset(torch.tensor([True, False, True, False]))
    assert [torch.zeros(4)[idx].shape[0] for idx in term.post_reset_calls] == [4, 2]
    assert term.saw_inference_mode is True


def _cuda_term(env, stats=None):
    term = RecordingResampleCommand(env, CommandTermOptions(resampling_time_range=(1.0, 2.0)))
    if stats is not None:
        term.stats = stats
    return term


def test_resample_costs_one_gpu_sync_per_bool_mask(cuda_stand_in_env, count_cuda_syncs):
    """The empty-selection count is the only sync on the resample path, which runs every control step."""
    term = _cuda_term(cuda_stand_in_env)
    mask = torch.tensor([True, False, True, False], device="cuda")
    assert count_cuda_syncs(lambda: term._resample(mask)) == 1
    assert count_cuda_syncs(lambda: term._resample(slice(None))) == 0  # a slice is counted on the host
    term.time_left.copy_(torch.tensor([0.01, 5.0, 0.01, 5.0], device="cuda"))  # two clocks expire
    calls = len(term.resample_calls)
    assert count_cuda_syncs(lambda: term.compute(dt=0.02)) == 1
    assert len(term.resample_calls) == calls + 1  # the non-empty path, not the early return
    assert count_cuda_syncs(lambda: term.compute(dt=0.02)) == 1  # nothing expires: the early return
    assert len(term.resample_calls) == calls + 1


def test_reset_stats_add_no_gpu_sync_to_the_resample(cuda_stand_in_env, count_cuda_syncs):
    """The auto-reset's mask costs the resample its one sync (pinned above); the stats' means add none."""
    term = _cuda_term(cuda_stand_in_env, {"a": torch.rand(4, device="cuda"), "b": torch.rand(4, device="cuda")})
    mask = torch.tensor([True, False, True, True], device="cuda")
    assert count_cuda_syncs(lambda: term.reset(mask)) == 1


# ---------------------------------------------------------------------------
# swap_term / restore_term
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def arm_env(tmp_path_factory):
    """A one-hinge arm under a real ``UniformVelocityCommand`` named ``velocity_command``."""
    from eden import actions
    from eden.options.entities import RobotOptions
    from eden.utils.configs import ActionManagerOptions, CommandManagerOptions, EdenRLConfig, EnvOptions, SceneOptions

    asset = tmp_path_factory.mktemp("arm") / "arm.xml"
    asset.write_text(ARM_MJCF)
    return build_env(
        EdenRLConfig(
            env_options=EnvOptions(num_envs=1),
            scene_options=SceneOptions(
                robot=RobotOptions(
                    file=str(asset),
                    dofs_name=["j1"],
                    default_dofs_pos={"j1": 0.0},
                    default_dofs_kp={".*": 10.0},
                    default_dofs_kd={".*": 1.0},
                    is_fixed_base=True,
                )
            ),
            action_options=ActionManagerOptions(
                joint_pos=actions.ImplicitPDController.configure(entity_name="robot", dofs_name=["j1"])
            ),
            command_options=CommandManagerOptions(
                velocity_command=UniformVelocityCommand.configure(
                    entity_name="robot", resampling_time_range=(3.0, 8.0), **_RANGES
                )
            ),
        ),
        reset=True,
    )


@pytest.fixture
def manager(arm_env):
    """The env's command manager, with its trained term put back after the test."""
    trained = arm_env.command_manager.get_term("velocity_command")
    yield arm_env.command_manager
    arm_env.command_manager.restore_term("velocity_command", trained)


def _held(**overrides):
    return ConstantVelocityCommand.configure(entity_name="robot", **{**_RANGES, **overrides})


def test_swap_installs_the_replacement_and_hands_back_the_old_term(manager):
    trained = manager.get_term("velocity_command")

    old = manager.swap_term("velocity_command", _held(command_value=(0.4, 0.0, 0.2)))

    assert old is trained
    assert isinstance(manager.get_term("velocity_command"), ConstantVelocityCommand)
    torch.testing.assert_close(manager.get_command("velocity_command")[0], torch.tensor([0.4, 0.0, 0.2]))


def test_the_replacement_is_an_ordinary_term_the_manager_keeps_computing(manager):
    manager.swap_term("velocity_command", _held(command_value=(0.7, 0.0, 0.0)))
    for _ in range(5):
        manager.compute(dt=0.02)
    torch.testing.assert_close(manager.get_command("velocity_command")[0], torch.tensor([0.7, 0.0, 0.0]))


def test_restore_puts_the_exact_original_back(manager):
    old = manager.swap_term("velocity_command", _held())
    manager.restore_term("velocity_command", old)
    assert manager.get_term("velocity_command") is old


def test_a_term_that_is_not_active_cannot_be_swapped(manager):
    with pytest.raises(KeyError, match="No active term named 'nope'"):
        manager.swap_term("nope", _held())


def test_a_replacement_of_another_width_is_refused_and_released(manager, monkeypatch):
    """The policy's observation width came from the replaced term; and ``build()`` already opened the device."""
    monkeypatch.setattr(DeviceOwningPoseCommand, "_closes", 0)
    trained = manager.get_term("velocity_command")

    with pytest.raises(ValueError, match=r"commands \(1, 7\) but UniformVelocityCommand commands \(1, 3\)"):
        manager.swap_term("velocity_command", DeviceOwningPoseCommand.configure(resampling_time_range=(3.0, 8.0)))

    assert DeviceOwningPoseCommand._closes == 1
    assert manager.get_term("velocity_command") is trained
