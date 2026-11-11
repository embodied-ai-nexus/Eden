"""A small RL task whose terms charge a known cost to a fake clock, for the profiler tests.

The profilers read ``perf_counter`` through ``eden.utils.profiling.core``; the ``clock`` fixture swaps
that one binding for :data:`CLOCK`, and each ``ProfilingCost*`` term advances it by its ``cost_us``.
"""

from __future__ import annotations

import pytest
import torch

from eden.managers import (
    EVENT_TERM_REGISTRY,
    OBSERVATION_TERM_REGISTRY,
    REWARD_TERM_REGISTRY,
    TERMINATION_TERM_REGISTRY,
)
from eden.managers.event_manager import EventTerm
from eden.managers.observation_manager import ObservationTerm
from eden.managers.reward_manager import RewardTerm
from eden.managers.termination_manager import TerminationTerm
from tests.envs.conftest import PENDULUM_MJCF, pendulum_options


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def read(self) -> float:
        return self.t

    def advance_us(self, us: float) -> None:
        self.t += us * 1e-6


CLOCK = FakeClock()


@OBSERVATION_TERM_REGISTRY.register()
class ProfilingCostObservation(ObservationTerm):
    cost_us: float = 0.0

    def compute(self) -> torch.Tensor:
        CLOCK.advance_us(self.cost_us)
        return torch.zeros(self.num_envs, 1, device=self.device)


@REWARD_TERM_REGISTRY.register()
class ProfilingCostReward(RewardTerm):
    cost_us: float = 0.0

    def compute(self) -> torch.Tensor:
        CLOCK.advance_us(self.cost_us)
        return torch.zeros(self.num_envs, device=self.device)


@TERMINATION_TERM_REGISTRY.register()
class ProfilingCostTermination(TerminationTerm):
    cost_us: float = 0.0

    def compute(self) -> torch.Tensor:
        CLOCK.advance_us(self.cost_us)
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)


@EVENT_TERM_REGISTRY.register()
def profiling_noop_event(env, envs_idx=None) -> None:
    """An INTERVAL event with nothing to do."""


@EVENT_TERM_REGISTRY.register()
class ProfilingCostResetEvent(EventTerm):
    """A RESET event whose kind fires adjacent terms as one ``compute_batch``, each charging its ``cost_us``."""

    cost_us: float = 0.0

    def compute(self, envs_idx=None) -> None:
        type(self).compute_batch([self], envs_idx)

    @classmethod
    def compute_batch(cls, terms, envs_idx) -> None:
        for term in terms:
            CLOCK.advance_us(term.cost_us)


@pytest.fixture
def clock(monkeypatch) -> FakeClock:
    """The profilers' clock, replaced by :data:`CLOCK` for the test."""
    from eden.utils.profiling import core

    monkeypatch.setattr(core, "perf_counter", CLOCK.read)
    return CLOCK


@pytest.fixture(scope="module")
def profiled_env(tmp_path_factory):
    """Pendulum task: obs 1000 us (policy group), a 0 us critic group, reward 2000 us, termination 500 us,
    an INTERVAL event, two RESET events of 300 and 400 us that fire as one batch, and a PD action term at
    decimation 2. Built and reset.
    """
    from eden import observations
    from eden.envs.base import RLEnvBase
    from eden.managers.terms.actions.joint_actions import ImplicitPDController
    from eden.options import (
        ActionManagerOptions,
        EnvOptions,
        EventManagerOptions,
        ObservationGroupOptions,
        ObservationManagerOptions,
        PlaneOptions,
        RewardManagerOptions,
        SceneOptions,
        TerminationManagerOptions,
    )
    from eden.terms import EventTerm
    from eden.utils.configs import EdenRLConfig

    pendulum = tmp_path_factory.mktemp("profiling") / "pendulum.xml"
    pendulum.write_text(PENDULUM_MJCF)
    config = EdenRLConfig(
        env_options=EnvOptions(num_envs=2, decimation=2, episode_length_s=100.0),
        scene_options=SceneOptions(plane=PlaneOptions(), robot=pendulum_options(str(pendulum))),
        action_options=ActionManagerOptions(
            pd=ImplicitPDController.configure(entity_name="robot", dofs_name=["hinge"])
        ),
        observation_options=ObservationManagerOptions(
            policy=ObservationGroupOptions(costly=ProfilingCostObservation.configure(cost_us=1000.0)),
            critic=ObservationGroupOptions(dofs_pos=observations.DofsPos.configure(entity_name="robot")),
        ),
        reward_options=RewardManagerOptions(
            costly=ProfilingCostReward.configure(cost_us=2000.0), free=ProfilingCostReward.configure()
        ),
        termination_options=TerminationManagerOptions(costly=ProfilingCostTermination.configure(cost_us=500.0)),
        event_options=EventManagerOptions(
            noop=EventTerm.configure(func=profiling_noop_event, mode="interval", interval_range_s=(1.0, 1.0)),
            seed_a=ProfilingCostResetEvent.configure(mode="reset", cost_us=300.0),
            seed_b=ProfilingCostResetEvent.configure(mode="reset", cost_us=400.0),
        ),
    )
    env = RLEnvBase.from_config(config)
    env.build()
    env.reset()
    return env
