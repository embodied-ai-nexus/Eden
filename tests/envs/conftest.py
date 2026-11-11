"""Shared scenes and term fixtures for the ``eden.envs`` tests.

``trace_config`` builds a small RL task whose every manager holds one tracing term: each hook a
step or a reset reaches appends a label to :data:`TRACE`, and a ``physics`` label marks each
Genesis sub-step that ran between two hooks. The step-order and reset-order tests read that trace.
"""

from __future__ import annotations

import pytest
import torch

from eden.managers import (
    ACTION_TERM_REGISTRY,
    COMMAND_TERM_REGISTRY,
    CURRICULUM_TERM_REGISTRY,
    EVENT_TERM_REGISTRY,
    METRIC_TERM_REGISTRY,
    OBSERVATION_TERM_REGISTRY,
    RECORDER_TERM_REGISTRY,
    REWARD_TERM_REGISTRY,
    TERMINATION_TERM_REGISTRY,
)
from eden.managers.command_manager import CommandTerm
from eden.managers.curriculum_manager import CurriculumTerm
from eden.managers.event_manager import EventTerm
from eden.managers.metric_manager import MetricTerm
from eden.managers.motion_manager import MOTION_TERM_REGISTRY, MotionTerm
from eden.managers.observation_manager import ObservationTerm
from eden.managers.recorder_manager import RecorderTerm
from eden.managers.reward_manager import RewardTerm
from eden.managers.termination_manager import TerminationTerm
from eden.managers.terms.actions.joint_actions import ImplicitPDController

#: One hinge on a fixed base: the cheapest scene an actuated action term can drive.
PENDULUM_MJCF = """\
<mujoco model="pendulum">
  <worldbody>
    <body name="base">
      <geom type="box" size="0.05 0.05 0.05" mass="1"/>
      <body name="arm" pos="0 0 0.1">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.02" mass="0.1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

#: The hook labels, in call order, of every tracing term since the last ``TRACE.clear()``.
TRACE: list[str] = []
_last_physics_step = [0]


def _trace(env, label: str) -> None:
    step = env.scene.sim.cur_step_global
    if step > _last_physics_step[0]:
        TRACE.extend(["physics"] * (step - _last_physics_step[0]))
    _last_physics_step[0] = step
    TRACE.append(label)


def clear_trace(env) -> None:
    """Empty :data:`TRACE` and restart the physics count at ``env``'s current sub-step."""
    TRACE.clear()
    _last_physics_step[0] = env.scene.sim.cur_step_global


class _PostResetTraced:
    """Mixin: trace ``reset`` and ``post_reset`` under the term's ``_trace_name``.

    Also keeps the ``episode_length_buf`` each of the two saw, as ``lengths_at_reset`` and
    ``lengths_at_post_reset``.
    """

    _trace_name = ""

    def reset(self, envs_idx=None):
        _trace(self._env, f"{self._trace_name}.reset")
        self.lengths_at_reset = getattr(self._env, "episode_length_buf", torch.zeros(0)).clone()
        return super().reset(envs_idx=envs_idx)

    def post_reset(self, envs_idx):
        _trace(self._env, f"{self._trace_name}.post_reset")
        self.lengths_at_post_reset = self._env.episode_length_buf.clone()
        super().post_reset(envs_idx)


@ACTION_TERM_REGISTRY.register()
class EnvsTraceAction(_PostResetTraced, ImplicitPDController):
    _trace_name = "action"

    def compute(self, actions: torch.Tensor) -> None:
        _trace(self._env, "action.compute")
        super().compute(actions)

    def apply_actions(self) -> None:
        _trace(self._env, "action.apply_actions")
        super().apply_actions()


@COMMAND_TERM_REGISTRY.register()
class EnvsTraceCommand(_PostResetTraced, CommandTerm):
    _trace_name = "command"
    resampling_time_range: tuple[float, float] = (100.0, 100.0)
    debug_vis: bool = True

    def build(self) -> None:
        self._command = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, envs_idx: torch.Tensor) -> None:
        pass

    def _update_command(self) -> None:
        _trace(self._env, "command.compute")

    def pre_physics_advance(self) -> None:
        _trace(self._env, "command.pre_physics_advance")

    def post_physics_refresh(self) -> None:
        _trace(self._env, "command.post_physics_refresh")

    def draw_vis(self) -> None:
        _trace(self._env, "command.draw_vis")


@METRIC_TERM_REGISTRY.register()
class EnvsTraceMetric(_PostResetTraced, MetricTerm):
    """Reports success on every env, so an ending episode is a successful one."""

    _trace_name = "metric"

    def compute(self) -> torch.Tensor:
        _trace(self._env, "metric.compute")
        return torch.ones(self.num_envs, device=self.device)


@OBSERVATION_TERM_REGISTRY.register()
class EnvsTraceObservation(_PostResetTraced, ObservationTerm):
    _trace_name = "observation"

    def compute(self) -> torch.Tensor:
        _trace(self._env, "observation.compute")
        return torch.zeros(self.num_envs, 1, device=self.device)

    def post_physics_refresh(self) -> None:
        _trace(self._env, "observation.post_physics_refresh")


@TERMINATION_TERM_REGISTRY.register()
class EnvsTraceTermination(_PostResetTraced, TerminationTerm):
    """Ends the episodes of the envs a test marks in :attr:`ending`.

    A test that needs a step to fail sets :attr:`raise_at_step` to a ``common_step_counter`` value.
    """

    _trace_name = "termination"

    def build(self) -> None:
        self.ending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.raise_at_step: int | None = None

    def compute(self) -> torch.Tensor:
        _trace(self._env, "termination.compute")
        if self.raise_at_step is not None and self._env.common_step_counter >= self.raise_at_step:
            raise RuntimeError(f"step {self._env.common_step_counter} fails, as the test asked")
        return self.ending.clone()


@REWARD_TERM_REGISTRY.register()
class EnvsTraceReward(_PostResetTraced, RewardTerm):
    _trace_name = "reward"

    def compute(self) -> torch.Tensor:
        _trace(self._env, "reward.compute")
        return torch.zeros(self.num_envs, device=self.device)


@CURRICULUM_TERM_REGISTRY.register()
class EnvsTraceCurriculum(_PostResetTraced, CurriculumTerm):
    _trace_name = "curriculum"

    def compute(self) -> dict:
        _trace(self._env, "curriculum.compute")
        return {"level": 0.0}


@EVENT_TERM_REGISTRY.register()
class EnvsTraceEvent(_PostResetTraced, EventTerm):
    """Traces its firing as ``<label>.compute``; configure one per mode with a distinct label."""

    label: str = "event"

    @property
    def _trace_name(self) -> str:
        return self.label

    def compute(self, envs_idx=None) -> None:
        _trace(self._env, f"{self.label}.compute")


@RECORDER_TERM_REGISTRY.register()
class EnvsTraceRecorder(_PostResetTraced, RecorderTerm):
    _trace_name = "recorder"

    def record_pre_step(self) -> dict:
        _trace(self._env, "recorder.record_pre_step")
        return {}

    def record_post_step(self) -> dict:
        _trace(self._env, "recorder.record_post_step")
        return {"step": torch.zeros(self.num_envs, device=self.device)}  # an empty episode is never exported

    def record_pre_reset(self, envs_idx=None) -> dict:
        _trace(self._env, "recorder.record_pre_reset")
        return {}

    def record_post_reset(self, envs_idx=None) -> dict:
        _trace(self._env, "recorder.record_post_reset")
        return {}


@MOTION_TERM_REGISTRY.register()
class EnvsTraceMotion(MotionTerm):
    """A real motion term tracing as ``<label>.*``; ``fill`` is one reference fill of one entity.

    Keeps what its ``post_reset`` saw: the envs it was handed, and the manager's robot reference buffer.
    """

    label: str = "motion"

    def reset(self, envs_idx, symmetric_augmentation_mask_buffer=None):
        _trace(self._env, f"{self.label}.reset")
        super().reset(envs_idx, symmetric_augmentation_mask_buffer)

    def compute(self, motion_sequence_key, envs_idx, sample_timestamp, env_origins, data_buffer):
        _trace(self._env, f"{self.label}.fill")
        super().compute(motion_sequence_key, envs_idx, sample_timestamp, env_origins, data_buffer)

    def post_reset(self, envs_idx):
        _trace(self._env, f"{self.label}.post_reset")
        self.post_reset_envs_idx = envs_idx.clone()
        self.robot_reference_at_post_reset = self._env.motion_manager.data["robot"].dofs_pos.clone()
        super().post_reset(envs_idx)


@pytest.fixture(scope="session")
def pendulum_file(tmp_path_factory) -> str:
    """Path to :data:`PENDULUM_MJCF` on disk."""
    path = tmp_path_factory.mktemp("envs_assets") / "pendulum.xml"
    path.write_text(PENDULUM_MJCF)
    return str(path)


def pendulum_options(pendulum_file: str):
    """The pendulum as an actuated, fixed-base entity."""
    from eden.options import EntityOptions

    return EntityOptions(
        file=pendulum_file,
        is_fixed_base=True,
        is_articulated=True,
        is_actuated=True,
        dofs_name=["hinge"],
        default_dofs_kp={"hinge": 10.0},
        default_dofs_kd={"hinge": 1.0},
    )


def trace_config(pendulum_file: str, export_dir: str, *, without: tuple[str, ...] = (), decimation: int = 2):
    """An RL config with one tracing term per manager; ``without`` names managers to leave empty.

    ``without`` takes ``"command"``, ``"metric"``, ``"curriculum"`` and ``"interval_event"``.
    """
    from eden.options import (
        ActionManagerOptions,
        CommandManagerOptions,
        CurriculumManagerOptions,
        EnvOptions,
        EventManagerOptions,
        MetricManagerOptions,
        ObservationGroupOptions,
        ObservationManagerOptions,
        PlaneOptions,
        RecorderManagerOptions,
        RewardManagerOptions,
        SceneOptions,
        TerminationManagerOptions,
    )
    from eden import observations, terminations
    from eden.terms import DoneTerm
    from eden.utils.configs import EdenRLConfig

    events = {"reset_event": EnvsTraceEvent.configure(mode="reset", label="event_reset")}
    if "interval_event" not in without:
        events["interval_event"] = EnvsTraceEvent.configure(
            mode="interval", interval_range_s=(0.0, 0.0), label="event_interval"
        )
    return EdenRLConfig(
        env_options=EnvOptions(num_envs=2, decimation=decimation, episode_length_s=100.0),
        scene_options=SceneOptions(plane=PlaneOptions(), robot=pendulum_options(pendulum_file)),
        action_options=ActionManagerOptions(pd=EnvsTraceAction.configure(entity_name="robot", dofs_name=["hinge"])),
        command_options=CommandManagerOptions(
            **({} if "command" in without else {"command": EnvsTraceCommand.configure()})
        ),
        metric_options=MetricManagerOptions(**({} if "metric" in without else {"metric": EnvsTraceMetric.configure()})),
        observation_options=ObservationManagerOptions(
            policy=ObservationGroupOptions(
                obs=EnvsTraceObservation.configure(),
                dofs_pos=observations.DofsPos.configure(entity_name="robot"),
                history_length=3,
            ),
            critic=ObservationGroupOptions(
                concatenate_terms=False,
                dofs_pos=observations.DofsPos.configure(entity_name="robot"),
                dofs_vel=observations.DofsVel.configure(entity_name="robot"),
            ),
        ),
        termination_options=TerminationManagerOptions(
            ends=EnvsTraceTermination.configure(),
            timeout=DoneTerm.configure(func=terminations.time_out, time_out=True),
        ),
        reward_options=RewardManagerOptions(reward=EnvsTraceReward.configure()),
        curriculum_options=CurriculumManagerOptions(
            **({} if "curriculum" in without else {"curriculum": EnvsTraceCurriculum.configure()})
        ),
        event_options=EventManagerOptions(**events),
        recorder_options=RecorderManagerOptions(
            dataset_export_dir_path=export_dir, recorder=EnvsTraceRecorder.configure()
        ),
    )


def trace_base_config(pendulum_file: str, export_dir: str, *, decimation: int = 2):
    """The :func:`trace_config` task cut down to the managers a plain ``EnvBase`` owns."""
    from eden.utils.configs import EdenConfig

    rl = trace_config(pendulum_file, export_dir, decimation=decimation)
    return EdenConfig(**{name: getattr(rl, name) for name in EdenConfig.model_fields})
