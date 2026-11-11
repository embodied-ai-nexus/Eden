"""``eden.envs.base``: construction from a config, the build, the step loop and resets.

The step and reset order are read off a real env whose managers each hold a tracing term
(``tests/envs/conftest.py``), and checked against the phase table in ``docs/guides/environments.md``.
"""

from __future__ import annotations

import importlib
import json
import os
import pathlib
import re
import subprocess
import sys
from collections import Counter

import genesis as gs
import pytest
import torch
from torch.overrides import TorchFunctionMode
from genesis.utils.array_class import ErrorCode
from torch.utils._device import _device_constructors

import eden

from eden.constants import EventMode
from eden.entities.particle import ParticleEntity
from eden.envs.base import EnvBase, MimicRLEnvBase, RLEnvBase, _entity_class_for_options, env_class_for_config
from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper
from eden.managers.curriculum_manager import CURRICULUM_TERM_REGISTRY, CurriculumTerm
from eden.managers.event_manager import EVENT_TERM_REGISTRY, EventTerm
from eden.managers.metric_manager import METRIC_TERM_REGISTRY, MetricTerm
from eden.managers.modifiers.observations import ExponentialMovingAverage, ObservationDelay
from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY, ObservationTerm
from eden.managers.recorder_manager import RECORDER_TERM_REGISTRY, RecorderTerm
from eden.managers.reward_manager import REWARD_TERM_REGISTRY, RewardTerm
from eden.managers.termination_manager import TERMINATION_TERM_REGISTRY, TerminationTerm
from eden.options import (
    BoxOptions,
    CurriculumManagerOptions,
    EnvOptions,
    EventManagerOptions,
    MetricManagerOptions,
    MotionManagerOptions,
    ObservationGroupOptions,
    ObservationManagerOptions,
    PlaneOptions,
    RecorderManagerOptions,
    RewardManagerOptions,
    SceneOptions,
    SphereOptions,
    TerminationManagerOptions,
)
from eden.options.learning.rsl_rl.rl import RslRlBaseRunnerOptions
from eden.options.materials import (
    MPMElasticMaterialOptions,
    MPMElastoPlasticMaterialOptions,
    MPMLiquidMaterialOptions,
    MPMSandMaterialOptions,
    PBDClothMaterialOptions,
    PBDElasticMaterialOptions,
    PBDLiquidMaterialOptions,
    PBDParticleMaterialOptions,
    SPHLiquidMaterialOptions,
)
from eden.options.options import ConfigurableOptions
from eden.options.sensors import SensorOptions, SensorsOptions
from eden.options.splat import NyxSplatOptions, SplatOptions
from eden.terms import ObsTerm
from eden.utils.configs import EdenConfig, EdenMimicConfig, EdenRLConfig
from tests.envs.conftest import TRACE, clear_trace, trace_base_config, trace_config
from tests.support.envs import build_env

_GUIDE = pathlib.Path(__file__).resolve().parents[2] / "docs" / "guides" / "environments.md"
_TINY = EnvOptions(num_envs=2)

# ---------------------------------------------------------------------------
# The step order: declaration, guide, and the real step
# ---------------------------------------------------------------------------

#: Every phase of ``RLEnvBase.step`` in order, as the guide's table names them, for the maximal step:
#: an action passed, every optional manager present, a recording on with an episode ending, final
#: observations on, an INTERVAL event, and eval mode (so ``draw_vis`` fires).
RL_STEP_ORDER: tuple[str, ...] = (
    "action_manager.compute",  # 1
    "recorder_manager._on_step_started",  # 2
    "command_manager.pre_physics_advance",  # 3
    "_scene_step_with_sensor_gating",  # 4 (xdecimation, with apply_actions)
    "command_manager.post_physics_refresh",  # 5
    "metric_manager.compute",  # 6
    "observation_manager.post_physics_refresh",  # 7
    "termination_manager.compute",  # 8
    "reward_manager.compute",  # 9
    "recorder_manager.set_episode_success",  # 10
    "recorder_manager._on_step_finished",  # 11
    "_record_final_observations",  # 12
    "_reset_idx",  # 13
    "_dispatch_post_reset",  # 14
    "recorder_manager._on_reset_finished",  # 15
    "curriculum_manager.compute",  # 16
    "command_manager.compute",  # 17
    "event_manager.compute",  # 18
    "observation_manager.compute",  # 19
    "command_manager.draw_vis",  # 20
)

#: Conditional phase -> every switch it needs. The vocabulary of the guide table's "runs when" column.
PHASE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "action_manager.compute": ("action",),
    "command_manager.pre_physics_advance": ("command_manager",),
    "command_manager.post_physics_refresh": ("command_manager",),
    "command_manager.compute": ("command_manager",),
    "command_manager.draw_vis": ("command_manager", "eval_mode"),
    "metric_manager.compute": ("metric_manager",),
    "curriculum_manager.compute": ("curriculum_manager",),
    "recorder_manager.set_episode_success": ("metric_manager", "recording", "episode_ends"),
    "_record_final_observations": ("record_final_observations",),
    "event_manager.compute": ("interval_events",),
}
SWITCHES = tuple(dict.fromkeys(s for switches in PHASE_REQUIREMENTS.values() for s in switches))

#: The switches that change which managers exist, as the ``without=`` of ``trace_config``.
_BUILD_SWITCHES = {
    "command_manager": "command",
    "metric_manager": "metric",
    "curriculum_manager": "curriculum",
    "interval_events": "interval_event",
}

#: Tracing term names per manager, in the order each manager walks them.
_TRACED = {
    "action_manager": ("action",),
    "command_manager": ("command",),
    "metric_manager": ("metric",),
    "termination_manager": ("termination",),
    "reward_manager": ("reward",),
    "curriculum_manager": ("curriculum",),
    "event_manager": ("event_reset", "event_interval"),
    "observation_manager": ("observation",),
    "recorder_manager": ("recorder",),
}


def _guide_phase_table() -> list[tuple[int, str, tuple[str, ...]]]:
    """The guide's step-order table as ``(number, call, conditions)`` rows."""
    match = re.search(r"<!-- step-order-table -->\n(.*?)\n<!-- /step-order-table -->", _GUIDE.read_text(), re.DOTALL)
    assert match, f"{_GUIDE.name} lost its <!-- step-order-table --> block; restore it rather than this test"
    rows = []
    for line in match.group(1).splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 4 or not cells[0].isdigit():
            continue
        call = re.fullmatch(r"`([^`]+)`", cells[2])
        assert call, f"row {cells[0]} of the step-order table has no backticked call: {cells[2]!r}"
        conditions = () if cells[3] == "always" else tuple(re.findall(r"`([^`]+)`", cells[3]))
        assert cells[3] == "always" or conditions, f"row {cells[0]}: unreadable 'runs when' cell {cells[3]!r}"
        rows.append((int(cells[0]), call.group(1), conditions))
    return rows


def _guide_manager_order(env_cls_name: str) -> tuple[str, ...]:
    """The guide's "Manager order" row for ``env_cls_name``, as manager attribute names."""
    match = re.search(rf"^\| `{env_cls_name}` \| ([a-z →]+) \|$", _GUIDE.read_text(), re.MULTILINE)
    assert match, f"{_GUIDE.name} has no manager-order row for {env_cls_name}"
    return tuple(f"{name.strip()}_manager" for name in match.group(1).split("→"))


def test_the_guide_phase_table_is_the_declared_order():
    """Calls, numbering and the "runs when" column: the prose around the table addresses all three."""
    rows = _guide_phase_table()

    assert [call for _, call, _ in rows] == list(RL_STEP_ORDER)
    assert [number for number, _, _ in rows] == list(range(1, len(RL_STEP_ORDER) + 1))
    assert {call: set(c) for _, call, c in rows if c} == {p: set(s) for p, s in PHASE_REQUIREMENTS.items()}


def _expected_trace(off: frozenset[str], *, decimation: int = 2) -> list[str]:
    """The tracing terms' hook sequence for one step, derived from :data:`RL_STEP_ORDER`.

    At term level a reset reaches only the envs that end, and curricula are frozen at eval, so
    ``episode_ends`` and ``eval_mode`` gate a few more events than the manager-level table says.
    """
    on = {switch: switch not in off for switch in (*SWITCHES, "episode_ends")}
    evaluating, ends, recording = on["eval_mode"], on["episode_ends"], on["recording"]

    def terms_of(manager: str) -> tuple[str, ...]:
        if not on.get(manager, True):
            return ()
        names = _TRACED[manager]
        return tuple(n for n in names if n != "event_interval" or on["interval_events"])

    order = _guide_manager_order("RLEnvBase")
    resets = [f"{n}.reset" for m in order for n in terms_of(m) if not (n == "curriculum" and evaluating)]
    post_resets = [f"{n}.post_reset" for m in order for n in terms_of(m)]
    apply = ["action.apply_actions"] if on["action"] else []

    events = {
        "action_manager.compute": ["action.compute"],
        "recorder_manager._on_step_started": ["recorder.record_pre_step"] if recording else [],
        "command_manager.pre_physics_advance": ["command.pre_physics_advance"],
        "_scene_step_with_sensor_gating": [*apply, "physics"] * decimation,
        "command_manager.post_physics_refresh": ["command.post_physics_refresh"],
        "metric_manager.compute": ["metric.compute"],
        "observation_manager.post_physics_refresh": ["observation.post_physics_refresh"],
        "termination_manager.compute": ["termination.compute"],
        "reward_manager.compute": ["reward.compute"],
        # Leaves no trace; test_an_ending_episode_is_exported_with_its_success_flag pins it.
        "recorder_manager.set_episode_success": [],
        "recorder_manager._on_step_finished": (
            ["recorder.record_post_step", *(["recorder.record_pre_reset", "recorder.reset"] if ends else [])]
            if recording
            else []
        ),
        "_record_final_observations": ["observation.compute"],
        "_reset_idx": ["event_reset.compute", *resets] if ends else [],
        "_dispatch_post_reset": post_resets if ends else [],
        "recorder_manager._on_reset_finished": ["recorder.record_post_reset"] if recording and ends else [],
        "curriculum_manager.compute": [] if evaluating else ["curriculum.compute"],
        "command_manager.compute": ["command.compute"],
        "event_manager.compute": ["event_interval.compute"],
        "observation_manager.compute": ["observation.compute"],
        "command_manager.draw_vis": ["command.draw_vis"],
    }
    trace = []
    for phase in RL_STEP_ORDER:
        if all(on[s] for s in PHASE_REQUIREMENTS.get(phase, ())):
            trace.extend(events[phase])
    return trace


@pytest.fixture(scope="module")
def trace_envs(pendulum_file, tmp_path_factory):
    """``trace_envs(without=(), eval_mode=True)``: a built tracing RL env, one per variant, cached."""
    cache: dict = {}

    def get(without: tuple[str, ...] = (), eval_mode: bool = True) -> RLEnvBase:
        key = (tuple(sorted(without)), eval_mode)
        if key not in cache:
            config = trace_config(pendulum_file, str(tmp_path_factory.mktemp("recordings")), without=without)
            env = RLEnvBase.from_config(config, eval_mode=eval_mode)
            env.build()
            env.reset()
            cache[key] = env
        return cache[key]

    return get


def _traced_step(env: RLEnvBase, off: frozenset[str] = frozenset()) -> list[str]:
    """Step ``env`` once with the runtime switches in ``off`` turned off; return the trace."""
    ending = env.termination_manager.get_term("ends").ending
    ending[:] = False
    if "episode_ends" not in off:
        ending[0] = True
    env.env_options.record_final_observations = "record_final_observations" not in off
    if "recording" not in off:
        env.recorder_manager.start_recording()
    clear_trace(env)
    try:
        env.step(None if "action" in off else torch.zeros(env.num_envs, 1, device=env.device))
        return list(TRACE)
    finally:
        env.recorder_manager.end_recording()
        env.env_options.record_final_observations = False
        ending[:] = False


def _env_for(trace_envs, off: frozenset[str]) -> RLEnvBase:
    without = tuple(_BUILD_SWITCHES[s] for s in off if s in _BUILD_SWITCHES)
    return trace_envs(without=without, eval_mode="eval_mode" not in off)


def test_rl_step_runs_the_declared_phase_order(trace_envs):
    assert _traced_step(_env_for(trace_envs, frozenset())) == _expected_trace(frozenset())


@pytest.mark.parametrize("switch", [*SWITCHES, "episode_ends"])
def test_each_switch_removes_exactly_its_phases(trace_envs, switch):
    """Equality, not a subsequence: turning one condition off must not take an unrelated phase along."""
    off = frozenset({switch})

    assert _traced_step(_env_for(trace_envs, off), off) == _expected_trace(off)


def test_an_ending_episode_is_exported_with_its_success_flag(trace_envs):
    """Phase 10 tags the episode before phase 11 exports it; the other way round it would export as failed."""
    env = _env_for(trace_envs, frozenset())
    before = env.recorder_manager.exported_successful_episode_count

    _traced_step(env)

    assert env.recorder_manager.exported_successful_episode_count == before + 1


def test_reset_sees_the_finished_episode_length_and_post_reset_the_zeroed_one(trace_envs):
    """``episode_length_buf`` is zeroed after every manager's ``reset()`` and before any ``post_reset()``."""
    env = _env_for(trace_envs, frozenset())
    term = env.termination_manager.get_term("ends")
    for _ in range(3):
        env.step(torch.zeros(env.num_envs, 1, device=env.device))
    lengths = env.episode_length_buf.clone()

    _traced_step(env)

    assert term.lengths_at_reset[0] == lengths[0] + 1
    assert term.lengths_at_post_reset[0] == 0
    assert term.lengths_at_post_reset[1] == lengths[1] + 1  # the env still running keeps counting


def test_reset_runs_both_halves_then_observes(trace_envs):
    """``EnvBase.reset`` is the path with no ``step`` around it; it dispatches ``post_reset`` too."""
    env = _env_for(trace_envs, frozenset())
    order = _guide_manager_order("RLEnvBase")
    terms = [n for m in order for n in _TRACED[m]]
    clear_trace(env)

    env.reset()

    resets = [f"{n}.reset" for n in terms if n != "curriculum"]  # eval env: curricula frozen
    assert TRACE == ["event_reset.compute", *resets, *(f"{n}.post_reset" for n in terms), "observation.compute"]


def test_base_env_step_is_the_rl_order_without_the_rl_phases(pendulum_file, tmp_path):
    """``EnvBase.step`` transcribes the same order: it closes the recorded step before the INTERVAL
    events and the observation, which a recorder reading post-step state depends on.
    """
    env = EnvBase.from_config(trace_base_config(pendulum_file, str(tmp_path)))
    env.build()
    env.reset()
    env.recorder_manager.start_recording()
    clear_trace(env)

    env.step(torch.zeros(env.num_envs, 1, device=env.device))

    assert TRACE == [
        "action.compute",
        "recorder.record_pre_step",
        *["action.apply_actions", "physics"] * 2,
        "metric.compute",
        "observation.post_physics_refresh",
        "recorder.record_post_step",
        "event_interval.compute",
        "observation.compute",
    ]
    assert _guide_manager_order("EnvBase") == env._manager_order()


# ---------------------------------------------------------------------------
# MimicRLEnvBase: the motion manager brackets the step and both reset halves
# ---------------------------------------------------------------------------


def _mimic_env(tmp_path_factory, *labels: str):
    """A real mimic env (G1 + box over synthetic clips) with one tracing motion term per label."""
    from eden.options import EventManagerOptions, ObservationGroupOptions, ObservationManagerOptions
    from eden.options import TerminationManagerOptions
    from tests.envs.conftest import EnvsTraceEvent, EnvsTraceMotion, EnvsTraceObservation, EnvsTraceTermination
    from tests.support.motion import build_mimic_env, manager_options, term_options, write_library

    root = write_library(tmp_path_factory.mktemp("clips"), [20, 30, 40])
    base = term_options(root)
    fields = {k: getattr(base, k) for k in ("dataset_options", "entities_map", "target_dofs_name", "target_links_name")}
    terms = {label: EnvsTraceMotion.configure(label=label, **fields) for label in labels}
    return build_mimic_env(
        root,
        motion_options=manager_options(terms=terms),
        termination_options=TerminationManagerOptions(ends=EnvsTraceTermination.configure()),
        event_options=EventManagerOptions(reset_event=EnvsTraceEvent.configure(mode="reset", label="event_reset")),
        observation_options=ObservationManagerOptions(
            policy=ObservationGroupOptions(obs=EnvsTraceObservation.configure())
        ),
    )


@pytest.fixture(scope="module")
def mimic_env(tmp_path_factory):
    return _mimic_env(tmp_path_factory, "motion")


def _step_ending(env, ending_envs: list[int]) -> list[str]:
    ending = env.termination_manager.get_term("ends").ending
    ending[ending_envs] = True
    clear_trace(env)
    try:
        env.step(None)
        return list(TRACE)
    finally:
        ending[:] = False


def test_the_mimic_step_refreshes_the_motion_reference_first_and_brackets_the_reset_with_it(mimic_env):
    """The reference is advanced before physics; on reset, motion samples the new clip before the scene reset
    (RSI events read it), refills the buffer once the RESET events are done and before every manager's
    ``reset()``, and runs ``post_reset`` first.
    """
    decimation = mimic_env.env_options.decimation
    trace = _step_ending(mimic_env, [0])

    fill = ["motion.fill"] * 2  # one per tracked entity: robot, object
    assert trace == [
        *fill,
        *["physics"] * decimation,
        "observation.post_physics_refresh",
        "termination.compute",
        "motion.reset",
        "event_reset.compute",
        *fill,
        "termination.reset",
        "event_reset.reset",
        "observation.reset",
        "motion.post_reset",
        "termination.post_reset",
        "event_reset.post_reset",
        "observation.post_reset",
        "observation.compute",
    ]


def test_post_reset_sees_the_refilled_motion_reference(mimic_env):
    """``fill_data`` runs before the manager loop, so before ``post_reset``: a term rebuilding a cache there
    reads the new episode's frames, not the finished episode's.
    """
    for _ in range(3):
        mimic_env.step(None)
    before = mimic_env.motion_manager.data["robot"].dofs_pos.clone()

    _step_ending(mimic_env, [0])

    term = mimic_env.motion_manager.get_term("motion")
    after = mimic_env.motion_manager.data["robot"].dofs_pos
    torch.testing.assert_close(term.robot_reference_at_post_reset[0], after[0])
    assert not torch.allclose(after[0], before[0])


def test_a_manager_reset_reading_the_motion_reference_raises(mimic_env, monkeypatch):
    """The window holds the new episode by the manager resets, so a ``reset()`` reading it to grade the
    ended episode would grade the wrong one: it raises, naming the API that keeps the ended episode.
    """
    term = mimic_env.termination_manager.get_term("ends")
    reset = term.reset

    def grading_reset(envs_idx=None):
        mimic_env.motion_manager.data["robot"].validity[:, 0]
        return reset(envs_idx=envs_idx)

    monkeypatch.setattr(term, "reset", grading_reset)
    with pytest.raises(RuntimeError, match=r"read inside a manager reset\(\).*ended_episode_validity\(<entity>\)"):
        mimic_env._reset_idx(torch.tensor([0], device=mimic_env.device))
    monkeypatch.undo()

    mimic_env.motion_manager.data  # the scope closed with the error
    mimic_env.reset()


def test_motion_post_reset_hands_each_term_only_its_own_envs(tmp_path_factory):
    """Two terms split the four envs (0-1, 2-3); the auto-reset mask reaches each with the other's envs cleared."""
    env = _mimic_env(tmp_path_factory, "left", "right")

    _step_ending(env, [0, 3])

    left, right = (env.motion_manager.get_term(name) for name in ("left", "right"))
    assert left.post_reset_envs_idx.tolist() == [True, False, False, False]
    assert right.post_reset_envs_idx.tolist() == [False, False, False, True]


# ---------------------------------------------------------------------------
# The manager order is validated at build
# ---------------------------------------------------------------------------


def test_an_out_of_sync_manager_order_fails_the_build():
    """The validator runs inside ``build()`` and names each problem: a manager the order skips, and
    ``event_manager`` ahead of ``action_manager`` (reset-mode DR would be overwritten by the action reset).
    """

    class _DriftedEnv(EnvBase):
        def _manager_order(self):
            order = list(super()._manager_order())
            order.remove("event_manager")
            return ("event_manager", *order)

        def _load_managers(self):
            super()._load_managers()
            self.telemetry_manager = object()

    env = _DriftedEnv.from_config(EdenConfig(env_options=_TINY, scene_options=SceneOptions(plane=PlaneOptions())))

    with pytest.raises(RuntimeError) as excinfo:
        env.build()
    assert "never reset or dispatched to: telemetry_manager" in str(excinfo.value)
    assert "event_manager must come after action_manager" in str(excinfo.value)


@OBSERVATION_TERM_REGISTRY.register()
class EnvsBuildCountingObservation(ObservationTerm):
    def build(self) -> None:
        self.builds = getattr(self, "builds", 0) + 1

    def compute(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 3, device=self.device)


@EVENT_TERM_REGISTRY.register()
class EnvsBuildCountingEvent(EventTerm):
    def build(self) -> None:
        self.builds = getattr(self, "builds", 0) + 1

    def compute(self, envs_idx=None) -> None:
        pass


def test_the_build_builds_every_term_once_and_orders_each_event_mode_by_priority():
    """Lower ``priority`` runs first."""
    from eden.options import EventManagerOptions, ObservationGroupOptions, ObservationManagerOptions

    env = EnvBase.from_config(
        EdenConfig(
            env_options=EnvOptions(num_envs=1),
            scene_options=SceneOptions(plane=PlaneOptions()),
            observation_options=ObservationManagerOptions(
                policy=ObservationGroupOptions(
                    a=EnvsBuildCountingObservation.configure(), b=EnvsBuildCountingObservation.configure()
                ),
                critic=ObservationGroupOptions(a=EnvsBuildCountingObservation.configure()),
            ),
            event_options=EventManagerOptions(
                late=EnvsBuildCountingEvent.configure(mode="reset", priority=10),
                early=EnvsBuildCountingEvent.configure(mode="reset", priority=0),
                boot=EnvsBuildCountingEvent.configure(mode="startup", priority=0),
            ),
        )
    )
    env.build()

    builds = {f"{g}.{n}": t.builds for g, n, t in env.observation_manager.iter_terms()}
    builds |= {n: t.builds for _, n, t in env.event_manager.iter_terms()}
    assert builds == {"policy.a": 1, "policy.b": 1, "critic.a": 1, "late": 1, "early": 1, "boot": 1}
    assert env.event_manager.mode_term_names[EventMode.RESET] == ["early", "late"]


# ---------------------------------------------------------------------------
# Post-build hooks
# ---------------------------------------------------------------------------


def test_post_build_hooks_run_once_whenever_they_are_queued_before_the_drain():
    """Registered before build, from a manager's construction (``_load_managers``), or by another hook
    during the drain: all run. After the drain registering is an error, since the hook could never run.
    """
    ran = []

    class _HookEnv(EnvBase):
        def _load_managers(self):
            super()._load_managers()
            self.register_post_build_hook(lambda: ran.append("from _load_managers"))

    env = _HookEnv.from_config(EdenConfig(env_options=_TINY, scene_options=SceneOptions(plane=PlaneOptions())))
    env.register_post_build_hook(
        lambda: (ran.append("before build"), env.register_post_build_hook(lambda: ran.append("queued by a hook")))
    )

    env.build()

    assert ran == ["before build", "from _load_managers", "queued by a hook"]
    with pytest.raises(RuntimeError, match="would never run"):
        env.register_post_build_hook(lambda: None)


# ---------------------------------------------------------------------------
# Construction: from_config, the config leaf pairing and option forwarding
# ---------------------------------------------------------------------------


class _TaskConfig(EdenRLConfig):
    env_options: EnvOptions = _TINY


class _TaskEnv(RLEnvBase):
    pass


def test_a_task_subclass_on_either_side_is_accepted():
    """The pairing is asked of the config leaves, not the exact classes."""
    env = _TaskEnv.from_config(_TaskConfig())

    assert isinstance(env, _TaskEnv)
    assert env.config.env_options.num_envs == 2


@pytest.mark.parametrize(
    ("env_cls", "config", "named"),
    [
        # Regression: motion_options were dropped and training saw no reference.
        (RLEnvBase, EdenMimicConfig(env_options=_TINY, motion_options=MotionManagerOptions()), "MimicRLEnvBase"),
        (MimicRLEnvBase, EdenRLConfig(env_options=_TINY), "EdenMimicConfig"),
        (EnvBase, EdenRLConfig(env_options=_TINY), "RLEnvBase"),
    ],
    ids=["rl-env-mimic-config", "mimic-env-rl-config", "base-env-rl-config"],
)
def test_a_config_from_another_leaf_is_refused(env_cls, config, named):
    with pytest.raises(TypeError, match=named) as excinfo:
        env_cls.from_config(config)
    assert type(config).__name__ in str(excinfo.value)


def test_a_config_leaf_mismatch_points_at_env_class_for_config():
    with pytest.raises(TypeError, match="env_class_for_config"):
        RLEnvBase.from_config(EdenMimicConfig(env_options=_TINY, motion_options=MotionManagerOptions()))


def test_an_env_subclass_is_told_which_config_to_pass_before_which_env_to_swap_to():
    """``env_class_for_config`` cannot recover a user's env subclass, so the config is the advice that leads."""

    class _SubclassConfig(EdenRLConfig):
        pass

    class _SubclassEnv(RLEnvBase, EnvBase[_SubclassConfig]):
        pass

    with pytest.raises(TypeError) as excinfo:
        _SubclassEnv.from_config(EdenRLConfig(env_options=_TINY))
    message = str(excinfo.value)
    assert message.index("_SubclassConfig") < message.index("RLEnvBase")


def test_binding_a_leaf_whose_fields_the_env_cannot_take_fails_at_class_creation():
    """``EnvBase[TaskConfig]`` would forward ``grasp_threshold`` into ``gs.Scene``; the plain subclass does not."""

    class _TaskFieldConfig(EdenRLConfig):
        grasp_threshold: float = 0.03

    with pytest.raises(TypeError, match="grasp_threshold") as excinfo:

        class _TaskFieldEnv(RLEnvBase, EnvBase[_TaskFieldConfig]):
            pass

    assert "_TaskFieldConfig" in str(excinfo.value)

    class _OrdinaryTaskEnv(RLEnvBase):
        pass

    assert _OrdinaryTaskEnv.from_config(_TaskFieldConfig(env_options=_TINY)).config.grasp_threshold == 0.03


def test_a_non_config_type_argument_is_rejected_at_class_creation():
    with pytest.raises(TypeError, match="must be an EdenConfig subclass"):

        class _NotAConfig(EnvBase[int]):  # type: ignore[type-var]
            pass


def test_a_plain_subclass_inherits_its_parents_config_leaf():
    class _UserEnv(RLEnvBase):
        pass

    assert env_class_for_config(_UserEnv._CONFIG_CLASS()) is RLEnvBase


_LEAVES = [(EnvBase, EdenConfig), (RLEnvBase, EdenRLConfig), (MimicRLEnvBase, EdenMimicConfig)]


@pytest.mark.parametrize(("env_cls", "cfg_cls"), _LEAVES, ids=lambda obj: obj.__name__)
def test_every_leaf_accepts_its_own_config_and_only_the_two_keywords(env_cls, cfg_cls):
    """A misspelled keyword (``eval_model=True``) must raise, not be ignored and train in training mode."""
    config = cfg_cls(env_options=_TINY)
    assert env_class_for_config(config) is env_cls

    env = env_cls.from_config(config, show_viewer=False, eval_mode=True)
    assert type(env) is env_cls
    assert env.training is False and env.headless is True
    with pytest.raises(TypeError, match="eval_model"):
        env_cls.from_config(config, eval_model=True)


def _splat_env(module_path: str, env_name: str, *construction_deps: str):
    module = pytest.importorskip(module_path, reason=f"{env_name} needs optional renderer deps")
    for dep in construction_deps:
        pytest.importorskip(dep, reason=f"constructing {env_name} needs {dep}")
    return getattr(module, env_name)


#: Every config leaf's env class, with the ``*_options`` its config leaves ``None`` by default and the
#: third-party modules its ``__init__`` reaches (skipped without them: the ``nyx`` extra is optional).
_ENV_LEAVES = [
    ("eden.envs.base", "EnvBase", {}, ()),
    ("eden.envs.base", "RLEnvBase", {}, ()),
    ("eden.envs.base", "MimicRLEnvBase", {"motion_options": MotionManagerOptions}, ()),
    ("eden.envs.splat", "SplatEnvBase", {"splat_options": SplatOptions}, ()),
    ("eden.envs.splat", "SplatRLEnvBase", {"splat_options": SplatOptions}, ()),
    ("eden.envs.nyx_splat", "NyxSplatEnvBase", {"splat_options": NyxSplatOptions}, ("gs_nyx", "gs_nyx_plugin")),
]


@pytest.mark.parametrize(("module_path", "env_name", "unset", "deps"), _ENV_LEAVES, ids=[e[1] for e in _ENV_LEAVES])
def test_from_config_forwards_every_option_to_the_env(module_path, env_name, unset, deps):
    """Identity, not equality: a dropped group is replaced by an equal-looking fresh default."""
    env_cls = _splat_env(module_path, env_name, *deps)
    config = env_cls._CONFIG_CLASS(env_options=EnvOptions(num_envs=1), **{k: v() for k, v in unset.items()})

    env = env_cls.from_config(config)

    fields = env_cls._runtime_option_fields()
    assert fields, "no runtime option fields: the check below would pass vacuously"
    assert [name for name in fields if getattr(env, name) is not getattr(config, name)] == []


@pytest.mark.parametrize(("module_path", "env_name", "unset", "deps"), _ENV_LEAVES, ids=[e[1] for e in _ENV_LEAVES])
def test_every_env_option_parameter_is_a_config_field(module_path, env_name, unset, deps):
    """An ``*_options`` parameter no config field names is an option no config can reach, silently."""
    import inspect

    env_cls = _splat_env(module_path, env_name)
    params = {name for name in inspect.signature(env_cls.__init__).parameters if name.endswith("_options")}

    assert params == set(env_cls._runtime_option_fields())


@pytest.mark.parametrize(
    ("module", "wrapper_name", "runner"),
    [
        ("eden.envs.wrappers.rsl_rl_env", "RslRlVecEnvWrapper", "rsl_rl"),
        ("eden.envs.wrappers.rl_games_env", "RlGamesVecEnvWrapper", "rl_games"),  # the opt-in rl-games extra
    ],
    ids=["rsl_rl", "rl_games"],
)
def test_the_training_wrappers_reject_an_unknown_keyword(module, wrapper_name, runner):
    """They forward ``show_viewer``/``eval_mode`` to whichever leaf ``env_class_for_config`` picks."""
    pytest.importorskip(runner, exc_type=ImportError)
    wrapper = getattr(importlib.import_module(module), wrapper_name)
    with pytest.raises(TypeError, match="eval_model"):
        wrapper.from_config(EdenRLConfig(env_options=_TINY), eval_model=True)


def test_no_from_config_override_in_the_tree_swallows_unknown_keywords():
    """Every ``from_config`` defined under ``eden/`` or ``examples/`` on a class reaching the contract.

    Read by parsing: the overrides most likely to drift live in example scripts a test cannot import
    (two ``RslRlVecEnvWrapper`` subclasses there kept ``**kwargs``).
    """
    import ast

    repo = pathlib.Path(__file__).resolve().parents[2]
    classes = [
        (path, node)
        for tree in ("eden", "examples")
        for path in sorted((repo / tree).rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ClassDef)
    ]

    def base_name(node):
        node = node.value if isinstance(node, ast.Subscript) else node
        return node.attr if isinstance(node, ast.Attribute) else getattr(node, "id", None)

    in_scope = {"EnvBase", "RslRlVecEnvWrapper", "RlGamesVecEnvWrapper"}
    while grown := {n.name for _, n in classes if {base_name(b) for b in n.bases} & in_scope} - in_scope:
        in_scope |= grown
    overrides = [
        (f"{path.relative_to(repo)}::{node.name}", fn)
        for path, node in classes
        if node.name in in_scope
        for fn in node.body
        if isinstance(fn, ast.FunctionDef) and fn.name == "from_config"
    ]

    assert {"EnvBase", "RslRlVecEnvWrapper", "RlGamesVecEnvWrapper"} <= {w.rsplit("::", 1)[1] for w, _ in overrides}
    assert {"MimicRLEnvBase", "SplatRLEnvBase"} <= in_scope, "the scan must follow subscripted EnvBase[Config] bases"
    bad = [
        where
        for where, fn in overrides
        if fn.args.kwarg is not None or {"show_viewer", "eval_mode"} - {a.arg for a in fn.args.kwonlyargs}
    ]
    assert not bad, f"these from_config overrides take **kwargs or miss show_viewer/eval_mode: {bad}"


@pytest.mark.parametrize(
    ("module_path", "env_name", "foreign"),
    [
        ("eden.envs.splat", "SplatEnvBase", NyxSplatOptions),
        ("eden.envs.splat", "SplatRLEnvBase", NyxSplatOptions),
        ("eden.envs.nyx_splat", "NyxSplatEnvBase", SplatOptions),
    ],
    ids=["SplatEnvBase", "SplatRLEnvBase", "NyxSplatEnvBase"],
)
def test_a_splat_env_refuses_missing_or_foreign_splat_options(module_path, env_name, foreign):
    """Checked in ``__init__``, so direct construction is covered as well as ``from_config``.

    Assigning ``cfg.splat_options`` after construction is not validated, so the env checks the class too.
    """
    env_cls = _splat_env(module_path, env_name)

    with pytest.raises(ValueError, match="splat_options"):
        env_cls.from_config(env_cls._CONFIG_CLASS(env_options=_TINY))
    with pytest.raises(ValueError, match="splat_options"):
        env_cls(env_options=_TINY)

    config = env_cls._CONFIG_CLASS(env_options=_TINY)
    config.splat_options = foreign()
    with pytest.raises(TypeError, match=foreign.__name__):
        env_cls.from_config(config)
    with pytest.raises(TypeError, match=foreign.__name__):
        env_cls(env_options=_TINY, splat_options=foreign())


def test_splat_configs_dispatch_to_their_own_env_classes():
    """``EdenSplatRLConfig`` subclasses ``EdenRLConfig``, so it must be matched before the generic branch."""
    from eden.utils.configs import EdenNyxSplatConfig, EdenSplatConfig, EdenSplatRLConfig

    splat = pytest.importorskip("eden.envs.splat", reason="splat env needs optional renderer deps")
    nyx = pytest.importorskip("eden.envs.nyx_splat", reason="nyx splat env needs optional renderer deps")

    assert env_class_for_config(EdenSplatConfig()) is splat.SplatEnvBase
    assert env_class_for_config(EdenSplatRLConfig()) is splat.SplatRLEnvBase
    assert env_class_for_config(EdenNyxSplatConfig()) is nyx.NyxSplatEnvBase


def test_attaching_to_a_missing_entity_fails_at_construction():
    scene = SceneOptions(child=BoxOptions(size=(0.1, 0.1, 0.1)), attachments_dict={"child": ("ghost", "link_0")})

    with pytest.raises(ValueError, match="nonexistent entities"):
        EnvBase.from_config(EdenConfig(env_options=_TINY, scene_options=scene))


def test_env_separate_rigid_reaches_genesis_as_split_envs():
    """Eden passes the current ``split_envs`` keyword, not the deprecated one."""
    env = EnvBase.from_config(
        EdenConfig(env_options=EnvOptions(num_envs=2, env_separate_rigid=True), scene_options=SceneOptions())
    )
    assert env.scene.options.vis.split_envs is True


@pytest.mark.parametrize(
    "integrator, expected",
    [
        ("approximate_implicitfast", gs.integrator.approximate_implicitfast),
        ("implicitfast", gs.integrator.implicitfast),
        ("euler", gs.integrator.Euler),
    ],
)
def test_mujoco_port_solver_options_reach_the_rigid_solver(integrator, expected):
    env = EnvBase.from_config(
        EdenConfig(
            env_options=EnvOptions(num_envs=2, integrator=integrator, impratio=2.0, enable_neutral_collision=True),
            scene_options=SceneOptions(),
        )
    )
    # Private Genesis attributes: the solver exposes these three settings no other way.
    solver = env.rigid_solver
    assert solver._integrator == expected
    assert solver._options.impratio == 2.0
    assert solver._enable_neutral_collision is True


@pytest.mark.parametrize("impratio", [0.0, -1.0])
def test_a_non_positive_impratio_is_rejected_by_the_config(impratio):
    with pytest.raises(ValueError, match="impratio"):
        EnvOptions(impratio=impratio)


def test_optional_managers_read_none_on_an_unbuilt_env():
    """Consumers test ``env.<x>_manager is not None``; that must hold before build too.

    The managers every env constructs have no such default, so reading one early stays a loud mistake.
    """
    env = RLEnvBase.from_config(EdenRLConfig(env_options=_TINY))

    assert (env.metric_manager, env.command_manager, env.curriculum_manager) == (None, None, None)
    for name in ("action_manager", "observation_manager", "event_manager", "recorder_manager"):
        with pytest.raises(AttributeError):
            getattr(env, name)


# ---------------------------------------------------------------------------
# env.config: a round trip of the source config
# ---------------------------------------------------------------------------


class _RunnerOptions(ConfigurableOptions):
    run_name: str = "a-training-run"


class SourceConfig(EdenRLConfig):
    """A task config: module level, so a save/load round trip can import it by name."""

    env_options: EnvOptions = EnvOptions(num_envs=2)
    runner_options: ConfigurableOptions | None = _RunnerOptions()
    language_instructions: list[str] = ["put the cube on the shelf"]
    task_specific_field: str = "declared by the task, not by the env"


def test_config_is_the_source_class_with_the_fields_no_env_parameter_takes():
    config = SourceConfig()
    env = RLEnvBase.from_config(config)

    reconstructed = env.config

    assert type(reconstructed) is SourceConfig
    assert reconstructed.runner_options is config.runner_options
    assert reconstructed.task_specific_field == "declared by the task, not by the env"


def test_config_reads_runtime_options_off_the_live_env():
    """A group the config left ``None`` reports the default the env substituted, not ``None``."""
    config = SourceConfig()
    config.reward_options = None
    env = RLEnvBase.from_config(config)
    env.env_options.num_envs = 77

    assert env.config.env_options.num_envs == 77
    assert env.config.reward_options is env.reward_options is not None
    assert env.config.env_options is env.env_options  # live children, documented: clone() to decouple
    assert env.config.clone().env_options is not env.env_options


def test_config_round_trips_through_a_generic_load(tmp_path):
    """Loaded through ``EdenConfig``, as ``eden train --config <saved config.json>`` does."""
    env = RLEnvBase.from_config(SourceConfig())
    env.env_options.num_envs = 5
    path = str(tmp_path / "config.json")

    env.config.save_as_json(path, lock_config=False)
    loaded = EdenConfig.load_from_file(path)

    assert type(loaded) is SourceConfig
    assert loaded.env_options.num_envs == 5
    assert loaded.language_instructions == ["put the cube on the shelf"]
    assert loaded.runner_options.run_name == "a-training-run"


def test_config_does_not_rerun_a_custom_init_nor_inherit_the_sources_lock():
    """A clobbering ``__init__`` would revert the caller's edit; a locked source must not lock the result."""

    class ClobberingConfig(EdenRLConfig):
        env_options: EnvOptions = EnvOptions(num_envs=1)
        marker: str = "default"

        def __init__(self, **data):
            super().__init__(**data)
            self.marker = "clobbered by __init__"

    config = ClobberingConfig()
    config.marker = "set by the caller"
    env = RLEnvBase.from_config(config)
    config.lock()

    reconstructed = env.config
    assert reconstructed.marker == "set by the caller"
    reconstructed.marker = "relabelled"  # raises on a locked root


def test_config_of_a_directly_constructed_env_is_a_synthesized_snapshot():
    env = RLEnvBase(env_options=EnvOptions(num_envs=3), scene_options=SceneOptions())

    assert type(env.config) is EdenRLConfig
    assert env.config.env_options.num_envs == 3
    assert env.config.runner_options is None


# ---------------------------------------------------------------------------
# Final observations (EnvOptions.record_final_observations)
# ---------------------------------------------------------------------------


def test_final_observations_hold_the_pre_reset_frame_of_the_ending_envs_only(trace_envs):
    """The snapshot is taken before the reset and survives the post-reset observation (#history view).

    Covers a history-bearing concatenated group and a non-concatenated (nested dict) one.
    """
    env = _env_for(trace_envs, frozenset())
    robot = env.entities["robot"]
    ending = env.termination_manager.get_term("ends").ending
    env.env_options.record_final_observations = True
    try:
        robot.set_dofs_pos(torch.tensor([[0.7], [0.3]], device=env.device))
        ending[:] = torch.tensor([True, False])
        obs, _, _, _, extras = env.step(None)
    finally:
        ending[:] = False
        env.env_options.record_final_observations = False

    final = extras["final_observations"]
    pre_reset = final["critic"]["dofs_pos"][0].clone()
    assert pre_reset.abs().item() > 0.1  # the swung pendulum, not the default pose
    assert torch.equal(final["critic"]["dofs_pos"][1], torch.zeros(1, device=env.device))
    assert torch.equal(final["critic"]["dofs_vel"][1], torch.zeros(1, device=env.device))
    # The returned observation already reflects the reset env's fresh episode.
    assert not torch.allclose(obs["critic"]["dofs_pos"][0], pre_reset)
    # policy = [trace obs, dofs_pos] x history 3: the newest frame of the ending env is the pre-reset pose.
    assert final["policy"].shape == obs["policy"].shape
    assert torch.equal(final["policy"][1], torch.zeros_like(final["policy"][1]))
    torch.testing.assert_close(final["policy"][0].view(3, 2)[-1, 1], pre_reset[0])


def test_final_observations_are_absent_unless_requested(trace_envs):
    env = _env_for(trace_envs, frozenset())
    assert "final_observations" not in env.step(None)[4]


# ---------------------------------------------------------------------------
# No Eden code relies on a torch default device
# ---------------------------------------------------------------------------


_EDEN_DIR = os.path.dirname(eden.__file__) + os.sep
_TORCH_DIR = os.path.dirname(torch.__file__) + os.sep


class _DevicelessFactoryCalls(TorchFunctionMode):
    """Count Eden-side tensor-factory calls that pass no ``device=``, keyed by call site.

    Neither ``genesis.init`` (since https://github.com/Genesis-Embodied-AI/Genesis/pull/3431) nor ``eden.init`` sets a
    torch default device, so such a tensor lands on the CPU whatever the backend. This suite runs on ``gs.cpu``,
    where that is invisible, so the call is checked instead: ``_device_constructors()`` is the exact set the
    default-device mode rewrites.
    """

    def __init__(self) -> None:
        super().__init__()
        self._constructors = _device_constructors()
        self.sites: Counter[tuple[str, int, str, str]] = Counter()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        # `torch.tensor(t)` / `torch.as_tensor(t)` of an existing tensor keep `t`'s device: not device-less.
        if (
            func in self._constructors
            and kwargs.get("device") is None
            and not (args and isinstance(args[0], torch.Tensor))
        ):
            frame = sys._getframe(1)
            while frame is not None and frame.f_code.co_filename.startswith(_TORCH_DIR):
                frame = frame.f_back
            if frame is not None and frame.f_code.co_filename.startswith(_EDEN_DIR):
                path = os.path.relpath(frame.f_code.co_filename, os.path.dirname(_EDEN_DIR.rstrip(os.sep)))
                self.sites[(path, frame.f_lineno, frame.f_code.co_name, func.__name__)] += 1
        return func(*args, **kwargs)

    def report(self) -> str:
        return "; ".join(
            f"{path}:{line} in {fn}() calls torch.{name} with no device= ({n}x)"
            for (path, line, fn, name), n in sorted(self.sites.items())
        )


def test_the_deviceless_call_detector_sees_an_eden_frame():
    """Guards the check below against a vacuous pass (a torch rename would record nothing).

    Compiled under a nonexistent path inside the ``eden`` package, which is all attribution reads.
    """
    code = compile(
        "torch.zeros(3)\ntorch.zeros(3, device='cpu')\ntorch.as_tensor(existing)\n",
        os.path.join(_EDEN_DIR, "_implicit_device_probe.py"),
        "exec",
    )
    with _DevicelessFactoryCalls() as mode:
        exec(code, {"torch": torch, "existing": torch.ones(2, device="cpu")})  # noqa: S102 -- a fixed literal
    assert [(site[1], site[3]) for site in mode.sites] == [(1, "zeros")], mode.sites


@pytest.mark.requires_genesis
@pytest.mark.parametrize(
    "task",
    [
        "velocity_tracking_flat",  # floating base, curriculum, push and reset events
        "beyondmimic",  # motion reference, RSI, the motion command's per-step buffers
        "manipulation_lift_box",  # fixed-base arm, a free object and its placement
    ],
)
def test_building_resetting_and_stepping_a_task_passes_a_device_everywhere(task):
    from eden.utils.smoke import build_and_step

    # Eager, so `@compile_model` bodies run op by op through the mode rather than as a traced graph.
    with torch.compiler.set_stance("force_eager"), _DevicelessFactoryCalls() as mode:
        env = build_and_step(task, num_envs=2, num_steps=3)
        env.reset()
        for _ in range(3):
            env.step(None)
    assert not mode.sites, (
        f"Building, resetting and stepping '{task}' made tensors with no device=, which are CPU tensors after "
        f"eden.init() on every backend: {mode.report()}. Pass device=self.device (or env.device, or the "
        "device of the tensor it is combined with), derive it with torch.zeros_like / x.new_zeros, or pass "
        'device="cpu" if it is deliberately host-side. Do not add a .to(device) after the call instead: it builds '
        "on the CPU and copies, and this check would still flag the construction."
    )


# ---------------------------------------------------------------------------
# The observation manager inside the real step: the phase-7 tick, the phase-12 snapshot, the auto-reset
# and the order _load_managers walks. A free box with probe terms per manager.
# ---------------------------------------------------------------------------


_PROBE_NUM_ENVS = 4
_RL_ORDER = RLEnvBase._manager_order(object.__new__(RLEnvBase))


# ---------------------------------------------------------------------------
# Terms
# ---------------------------------------------------------------------------


@OBSERVATION_TERM_REGISTRY.register()
class StepClockObs(ObservationTerm):
    """``k * dt`` after ``k`` refreshes, zero on reset: the smallest stateful observation term."""

    def build(self) -> None:
        self._clock = torch.zeros(self.num_envs, 1, device=self.device)

    def post_physics_refresh(self) -> None:
        self._clock.add_(self._env.dt)

    def reset(self, envs_idx=None) -> None:
        self._clock[slice(None) if envs_idx is None else envs_idx] = 0.0

    def compute(self) -> torch.Tensor:
        return self._clock


@OBSERVATION_TERM_REGISTRY.register()
def episode_step(env) -> torch.Tensor:
    """The env's episode step count: a frame whose value says which step produced it."""
    return env.episode_length_buf.to(torch.float32).unsqueeze(-1)


@OBSERVATION_TERM_REGISTRY.register()
def odd_env_ratio(env) -> torch.Tensor:
    """``x / x`` with ``x = env index % 2``: NaN in the even envs, 1.0 in the odd ones."""
    x = (torch.arange(env.num_envs, device=env.device) % 2).to(torch.float32).unsqueeze(-1)
    return x / x


def _existing_managers(env) -> list[str]:
    return [name for name in _RL_ORDER if getattr(env, name, None) is not None]


@TERMINATION_TERM_REGISTRY.register()
class ForcedDone(TerminationTerm):
    """Ends an env's episode on the step a test sets its row of ``force``; records the managers built before it."""

    def build(self) -> None:
        self.force = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.managers_before = _existing_managers(self._env)

    def compute(self) -> torch.Tensor:
        return self.force.clone()


@OBSERVATION_TERM_REGISTRY.register()
class OrderProbeObs(ObservationTerm):
    def build(self) -> None:
        self.managers_before = _existing_managers(self._env)

    def compute(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 1, device=self.device)


@REWARD_TERM_REGISTRY.register()
class OrderProbeReward(RewardTerm):
    def build(self) -> None:
        self.managers_before = _existing_managers(self._env)

    def compute(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)


@METRIC_TERM_REGISTRY.register()
class OrderProbeMetric(MetricTerm):
    def build(self) -> None:
        self.managers_before = _existing_managers(self._env)

    def compute(self, envs_idx=None) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)


@CURRICULUM_TERM_REGISTRY.register()
class OrderProbeCurriculum(CurriculumTerm):
    def build(self) -> None:
        self.managers_before = _existing_managers(self._env)

    def compute(self) -> dict:
        return {"level": 0.0}


@RECORDER_TERM_REGISTRY.register()
class OrderProbeRecorder(RecorderTerm):
    def build(self) -> None:
        self.managers_before = _existing_managers(self._env)


@EVENT_TERM_REGISTRY.register()
class OrderProbeStartup(EventTerm):
    """A STARTUP event: records the managers built before its own, and those present when it fires."""

    def build(self) -> None:
        self.managers_before = _existing_managers(self._env)
        self.managers_at_startup = None

    def compute(self, envs_idx=None) -> None:
        self.managers_at_startup = _existing_managers(self._env)


def _rl_config(*, record_final_observations: bool) -> EdenRLConfig:
    return EdenRLConfig(
        env_options=EnvOptions(
            num_envs=_PROBE_NUM_ENVS, decimation=2, record_final_observations=record_final_observations
        ),
        scene_options=SceneOptions(box=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.0, 0.0, 0.5))),
        observation_options=ObservationManagerOptions(
            policy=ObservationGroupOptions(
                clock=StepClockObs.configure(),
                now=ObsTerm.configure(func=episode_step),
                delayed=ObsTerm.configure(func=episode_step, modifiers=[ObservationDelay.configure(control_steps=2)]),
                averaged=ObsTerm.configure(
                    func=episode_step, modifiers=[ExponentialMovingAverage.configure(alpha=0.2, backfill=False)]
                ),
            ),
            guarded=ObservationGroupOptions(r=ObsTerm.configure(func=odd_env_ratio), nan_policy="sanitize"),
            probe=ObservationGroupOptions(order_probe=OrderProbeObs.configure()),
        ),
        termination_options=TerminationManagerOptions(forced=ForcedDone.configure()),
        reward_options=RewardManagerOptions(probe=OrderProbeReward.configure()),
        metric_options=MetricManagerOptions(probe=OrderProbeMetric.configure()),
        curriculum_options=CurriculumManagerOptions(probe=OrderProbeCurriculum.configure()),
        recorder_options=RecorderManagerOptions(probe=OrderProbeRecorder.configure()),
        event_options=EventManagerOptions(probe=OrderProbeStartup.configure(mode=EventMode.STARTUP)),
    )


def _build_probe(config):
    env = env_class_for_config(config).from_config(config, show_viewer=False)
    env.build()
    return env


@pytest.fixture(scope="module")
def probe_env():
    """The RL env with the final-observation snapshot on."""
    return _build_probe(_rl_config(record_final_observations=True))


@pytest.fixture(scope="module")
def probe_env_without_snapshot():
    return _build_probe(_rl_config(record_final_observations=False))


@pytest.fixture(scope="module")
def probe_base_env():
    """A plain ``EnvBase``: no commands, no auto-reset."""
    return _build_probe(
        EdenConfig(
            env_options=EnvOptions(num_envs=_PROBE_NUM_ENVS, decimation=2),
            scene_options=SceneOptions(box=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.0, 0.0, 0.5))),
            observation_options=ObservationManagerOptions(
                policy=ObservationGroupOptions(clock=StepClockObs.configure())
            ),
        )
    )


def _forced_step(env, *, done=()):
    """One control step with no action; ``done`` names the envs the step ends."""
    forced = env.termination_manager.get_term("forced").force
    forced[list(done)] = True
    try:
        return env.step(torch.zeros(_PROBE_NUM_ENVS, 0, device=env.device))
    finally:
        forced[:] = False


def _column(obs, name: str) -> torch.Tensor:
    return obs["policy"][:, ["clock", "now", "delayed", "averaged"].index(name)]


# ---------------------------------------------------------------------------
# The phase-7 tick and the snapshot
# ---------------------------------------------------------------------------


def test_a_stateful_term_reads_zero_after_reset_and_k_dt_after_k_steps(probe_env):
    obs, _ = probe_env.reset()
    assert _column(obs, "clock").tolist() == [0.0] * _PROBE_NUM_ENVS

    for k in range(1, 6):
        obs, *_ = _forced_step(probe_env)
        torch.testing.assert_close(
            _column(obs, "clock"), torch.full((_PROBE_NUM_ENVS,), k * probe_env.dt), msg=f"step {k}"
        )


def test_reads_between_steps_do_not_advance_state(probe_env):
    """rsl_rl's ``get_observations()`` between steps is a read, for a stateful term and a delay alike."""
    wrapper = RslRlVecEnvWrapper(probe_env, RslRlBaseRunnerOptions())  # resets the env, as rsl_rl expects
    reset_obs = probe_env.observation_manager.compute()
    torch.testing.assert_close(wrapper.get_observations()["policy"], reset_obs["policy"])

    for t in range(1, 6):
        obs, *_ = _forced_step(probe_env)
        assert _column(obs, "now").tolist() == [float(t)] * _PROBE_NUM_ENVS
        assert _column(obs, "delayed").tolist() == [float(max(t - 2, 0))] * _PROBE_NUM_ENVS
        wrapper.get_observations()
        wrapper.get_observations()

    torch.testing.assert_close(
        wrapper.get_observations()["policy"][:, 0], torch.full((_PROBE_NUM_ENVS,), 5 * probe_env.dt)
    )


def test_an_auto_reset_env_restarts_while_the_others_keep_counting(probe_env):
    """Phase 7 is above the reset: the snapshot carries the final increment, the new episode starts at zero."""
    probe_env.reset()
    for _ in range(3):
        _forced_step(probe_env)

    obs, _, _, _, extras = _forced_step(probe_env, done=[2])

    expected = torch.full((_PROBE_NUM_ENVS,), 4 * probe_env.dt)
    expected[2] = 0.0
    torch.testing.assert_close(_column(obs, "clock"), expected)
    final = extras["final_observations"]["policy"]
    assert final[2, 0] == pytest.approx(4 * probe_env.dt)  # the step that ended it is in there
    assert torch.equal(final[[0, 1, 3]], torch.zeros(3, final.shape[1]))  # non-done rows are zero by contract
    # The delay restarts for probe_env 2 only; its terminal observation is its step-4 frame, delayed as usual.
    assert final[2, 1:3].tolist() == [4.0, 2.0]
    assert _column(obs, "delayed").tolist() == [2.0, 2.0, 0.0, 2.0]

    obs, *_ = _forced_step(probe_env)
    expected = torch.full((_PROBE_NUM_ENVS,), 5 * probe_env.dt)
    expected[2] = probe_env.dt
    torch.testing.assert_close(_column(obs, "clock"), expected)
    assert _column(obs, "delayed").tolist() == [3.0, 3.0, 0.0, 3.0]


def test_the_snapshot_flag_does_not_change_what_the_policy_observes(probe_env, probe_env_without_snapshot):
    """A ``compute()``-time stamp used to advance pre-reset with the snapshot on and post-reset with it off."""
    delivered = []
    for built in (probe_env, probe_env_without_snapshot):
        built.reset()
        _forced_step(built)
        _forced_step(built, done=[1])
        obs, *_ = _forced_step(built)
        delivered.append(obs["policy"])

    torch.testing.assert_close(delivered[0], delivered[1])
    torch.testing.assert_close(delivered[0][:, 0], torch.tensor([3.0, 1.0, 3.0, 3.0]) * probe_env.dt)
    # The EMA follows its recurrence from zero: 0.2, then 0.2 * 2 + 0.8 * 0.2, then 0.2 * 3 + 0.8 * 0.56.
    assert delivered[0][0, 3].item() == pytest.approx(1.048)


def test_the_nan_guard_covers_both_the_observation_and_the_final_observation(probe_env):
    obs, _ = probe_env.reset()
    assert obs["guarded"][:, 0].tolist() == [0.0, 1.0, 0.0, 1.0]

    obs, _, _, _, extras = _forced_step(probe_env, done=[0])

    assert obs["guarded"][:, 0].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert torch.isfinite(extras["final_observations"]["guarded"]).all()


def test_env_base_step_ticks_a_stateful_term_once_too(probe_base_env):
    probe_base_env.reset()
    for k in range(1, 4):
        obs, *_ = probe_base_env.step()
        torch.testing.assert_close(obs["policy"][:, 0], torch.full((_PROBE_NUM_ENVS,), k * probe_base_env.dt))


# ---------------------------------------------------------------------------
# Manager order: declared once, walked by _load_managers
# ---------------------------------------------------------------------------


def _splat_env_classes() -> list[type]:
    """The splat leaves, when the ``batch-splat`` extra is installed (``eden.envs.splat`` needs ``plyfile``)."""
    try:
        from eden.envs.splat import SplatEnvBase, SplatRLEnvBase
    except ModuleNotFoundError:
        return []
    return [SplatEnvBase, SplatRLEnvBase]


ENV_CLASSES = pytest.mark.parametrize(
    "env_cls", [EnvBase, RLEnvBase, MimicRLEnvBase, *_splat_env_classes()], ids=lambda c: c.__name__
)


@ENV_CLASSES
def test_the_factories_and_the_order_declare_the_same_managers(env_cls):
    """On an unconstructed instance: both methods are declarations and read no state.

    A factory outside the order is never called; an ordered name with no factory cannot be built; a
    bracketed manager is built by the override that brackets the loop; ``_load_managers`` only drops an
    optional manager it walks past.
    """
    bare = object.__new__(env_cls)
    factories, order = set(env_cls._manager_factories(bare)), env_cls._manager_order(bare)

    assert factories == set(order)
    assert not factories & env_cls._BRACKETED_MANAGERS
    assert env_cls._OPTIONAL_MANAGERS <= set(order)


def test_the_rl_order_puts_every_build_time_producer_ahead_of_its_consumer():
    """Each pair is a handle an in-tree term resolves in ``build()``; the old order broke three."""
    pairs = [
        ("action_manager", "reward_manager"),  # ActionRateL2Smooth
        ("command_manager", "reward_manager"),  # TrackingLinVel & co.
        ("command_manager", "observation_manager"),
        ("command_manager", "curriculum_manager"),  # VelocityCommandCurriculum
        ("metric_manager", "reward_manager"),
        ("metric_manager", "termination_manager"),
        ("metric_manager", "observation_manager"),
        ("reward_manager", "curriculum_manager"),  # StageRewardWeightCurriculum, PenaltyCurriculum
        ("termination_manager", "curriculum_manager"),  # MotionBasePoseThresholdCurriculum
    ]
    for producer, consumer in pairs:
        assert _RL_ORDER.index(producer) < _RL_ORDER.index(consumer), f"{producer} must be built before {consumer}"


def test_each_manager_is_constructed_after_every_manager_ahead_of_it_in_the_order(probe_env):
    """Every probe term records, in its own ``build()``, which managers already existed."""
    present = _existing_managers(probe_env)
    probes = {
        "metric_manager": probe_env.metric_manager.get_term("probe"),
        "termination_manager": probe_env.termination_manager.get_term("forced"),
        "reward_manager": probe_env.reward_manager.get_term("probe"),
        "curriculum_manager": probe_env.curriculum_manager.get_term("probe"),
        "event_manager": next(term for _, _, term in probe_env.event_manager.iter_terms()),
        "observation_manager": probe_env.observation_manager.get_term("order_probe"),
        "recorder_manager": probe_env.recorder_manager.get_term("probe"),
    }

    for manager, probe in probes.items():
        assert probe.managers_before == present[: present.index(manager)], manager


def test_startup_events_fire_after_every_manager_is_constructed(probe_env):
    """STARTUP DR moves the scene and may reach for any manager, so it runs below the whole walk."""
    probe = next(term for _, _, term in probe_env.event_manager.iter_terms())

    assert probe.managers_at_startup == _existing_managers(probe_env)


def test_an_optional_manager_with_no_terms_is_none_and_the_others_always_exist(probe_env, probe_base_env):
    assert probe_env.command_manager is None  # optional, and this config declares no command term
    assert probe_base_env.metric_manager is None
    assert (
        probe_base_env.action_manager is not None and probe_env.action_manager is not None
    )  # not optional, even empty


@pytest.mark.parametrize(
    "edit, message",
    [
        (
            lambda factories: {k: v for k, v in factories.items() if k != "reward_manager"},
            r"no factory for: reward_manager",
        ),
        (lambda factories: factories | {"ghost_manager": object}, r"_manager_order\(\) would call: ghost_manager"),
    ],
    ids=["missing-factory", "unordered-factory"],
)
def test_a_factory_map_that_disagrees_with_the_order_raises_before_building_anything(edit, message):
    """``_load_managers.__wrapped__`` skips the ``@gs.assert_built`` guard, which is about the scene, not the order."""

    class Edited(RLEnvBase):
        def _manager_factories(self):
            return edit(super()._manager_factories())

    with pytest.raises(RuntimeError, match=message):
        EnvBase._load_managers.__wrapped__(object.__new__(Edited))


# ---------------------------------------------------------------------------
# Sensor updates on decimation sub-steps, and the entity class a material dispatches to.
# The sub-step cases wrap Genesis' ``SensorManager.step`` (a Genesis object) to record whether it carried
# the per-sensor dispatch; nothing in a sensor's read can tell a skipped update from a real one.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sensor", "every_substep", "expected"),
    [
        # The default: the heavy per-class update runs once per env.step(), on the last sub-step.
        (gs.sensors.Contact(), False, [False, False, False, True]),
        # Opt-in, for Genesis-level delay/history rings that must be sample-aligned to the physics rate.
        (gs.sensors.Contact(), True, [True, True, True, True]),
        # A Genesis-level delay allocates such rings itself, so the skip turns itself off.
        (gs.sensors.IMU(delay=0.01), False, [True, True, True, True]),
    ],
    ids=["default_skip", "opt_in", "delay_forces_every_substep"],
)
def test_sensor_dispatch_per_substep(sensor, every_substep, expected):
    env = build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=1, decimation=4, update_sensors_every_substep=every_substep),
            scene_options=SceneOptions(robot=BoxOptions(size=(0.2, 0.2, 0.2))),
            sensors_options=SensorsOptions(s=SensorOptions(sensor=sensor, attach_entity_name="robot")),
        )
    )
    try:
        manager = env.scene.sim._sensor_manager
        real_step, dispatched = manager.step, []

        def recording_step():
            dispatched.append(bool(manager._sensors_by_type))
            real_step()

        manager.step = recording_step
        env.step()
        assert dispatched == expected
    finally:
        env.destroy()


@pytest.mark.parametrize(
    "material",
    [
        MPMElasticMaterialOptions,
        MPMElastoPlasticMaterialOptions,
        MPMLiquidMaterialOptions,
        MPMSandMaterialOptions,
        SPHLiquidMaterialOptions,
        PBDLiquidMaterialOptions,
        PBDClothMaterialOptions,
        PBDElasticMaterialOptions,
        PBDParticleMaterialOptions,
    ],
    ids=lambda c: c.__name__,
)
def test_particle_materials_dispatch_to_particle_entity(material):
    """Dispatch keys off the material, so a ``BoxOptions`` carrier stands for every morph."""
    assert _entity_class_for_options(BoxOptions(material=material())) is ParticleEntity


# ---------------------------------------------------------------------------
# get_entities / get_entity name matching
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def named_env():
    """Name shapes under test: a shared ``robot_*`` prefix, an exact ``robot``, digit suffixes of two lengths."""
    cube = BoxOptions(size=(0.2, 0.2, 0.2))
    names = ["robot", "robot_arm", "robot_base", "robot_1", "robot_2", "robot_10", "box1", "box2"]
    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=1),
            scene_options=SceneOptions(**{name: cube for name in names}, sphere=SphereOptions(radius=0.1)),
        )
    )


@pytest.mark.parametrize(
    ("patterns", "expected"),
    [
        ("box.*", {"box1", "box2"}),  # a search over the whole name
        ("^robot$", {"robot"}),  # anchors make it exact
        (["box.*", "sphere"], {"box1", "box2", "sphere"}),  # a list unions its patterns
        (r"robot_\d$", {"robot_1", "robot_2"}),  # one trailing digit: robot_10 has two
        (r"robot_\d+$", {"robot_1", "robot_2", "robot_10"}),
        ("nonexistent.*", set()),  # no match is an empty list, not an error
    ],
)
def test_get_entities_matches_names_without_duplicates(named_env, patterns, expected):
    matched = named_env.get_entities(patterns)
    assert sorted(entity.name for entity in matched) == sorted(expected)


def test_get_entity_returns_the_named_entity_or_lists_the_real_ones(named_env):
    """The name came from a config or a term param, so the failure lists what the scene has."""
    assert named_env.get_entity("robot").name == "robot"
    with pytest.raises(KeyError) as excinfo:
        named_env.get_entity("robot_3")
    assert all(f"'{name}'" in str(excinfo.value) for name in ("robot_3", "robot_1", "sphere"))


# ---------------------------------------------------------------------------
# GPU->CPU syncs in an auto-resetting step
# ---------------------------------------------------------------------------

#: Syncs per ``RLEnvBase.step`` that resets some envs, keyed ``(task, wrap)``: the reset's one shared
#: ``nonzero()`` (``sharing_env_indices``), Genesis' per-step ``check_errno``, the contact read-back of
#: the contact terms (``RigidEntity.get_contacts``, twice on ConTrack), and — velocity — the command
#: clock's empty-resample count. ``wrap`` steps also push two envs past their clip's end, so
#: BeyondMimic's motion-end resync resamples and teleports them: through the wrap mask, every step,
#: so no sync of its own. Gathering through the mask instead cost 8 on velocity and BeyondMimic, 10 on a
#: BeyondMimic wrap step.
#: The art tasks: the reset's shared ``nonzero()`` only. Their metrics (one ``CopyMetric`` base, sampling the tool
#: every sim step and laying and grading the marks every control step) have none: 19 a step on painting, 14 with the
#: brush and 10 with the pen before their constants were built once. Their step path has no Python branch on data, so
#: the zero-action steps counted here run the same GPU work as a stroke; one task stands for each mark model (the
#: paint, the brush's ink, the pen's).
STEP_SYNC_BUDGETS = {
    ("velocity_tracking_flat", False): 4,
    ("beyondmimic", False): 3,
    ("beyondmimic", True): 3,
    ("contrack", False): 4,
    ("art_painting_sunset", False): 1,
    ("art_calligraphy_ei", False): 1,
    ("art_calligraphy_lettering_minimum", False): 1,
}


@pytest.mark.slow
@pytest.mark.requires_genesis
@pytest.mark.skipif(not torch.cuda.is_available(), reason="counting GPU->CPU syncs needs CUDA")
@pytest.mark.parametrize(
    ("task", "wrap", "budget"),
    [(task, wrap, budget) for (task, wrap), budget in STEP_SYNC_BUDGETS.items()],
    ids=[f"{task}-wrap" if wrap else task for task, wrap in STEP_SYNC_BUDGETS],
)
def test_an_auto_resetting_step_stays_within_its_sync_budget(task, wrap, budget):
    """Each sync stalls the host until the GPU queue drains, so Eden's managers cannot overlap physics.

    Counted in a process of its own (``tests/support/step_syncs.py``): this one holds Genesis on gs.cpu,
    where nothing syncs. So it runs only on a CUDA machine — CI has none: run
    ``uv run pytest tests/envs/test_base.py -k sync_budget`` before merging a change to a step or reset
    path, or ``uv run python -m tests.support.step_syncs TASK [--wrap]`` for the per-site counts. Over
    budget: the failure lists the sites; a bool mask used as an index, a scalar written through int ids,
    or a ``.item()`` is the usual new one.
    """
    repo = pathlib.Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "tests.support.step_syncs", task, *(["--wrap"] if wrap else [])],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    line = next((line for line in result.stdout.splitlines() if line.startswith("RESULT ")), None)
    assert line is not None, f"step_syncs {task} printed no RESULT line; stderr tail:\n{result.stderr[-3000:]}"
    measured = json.loads(line.removeprefix("RESULT "))
    assert max(measured["counts"]) <= budget, (
        f"{task}{' (wrap)' if wrap else ''}: {measured['counts']} syncs per step, budget {budget}; "
        f"per site over 5 steps: {measured['sites']}"
    )


# ---------------------------------------------------------------------------
# A solver error inside the decimation loop
# ---------------------------------------------------------------------------

_DECIMATION = 4


@pytest.fixture(scope="module")
def decimated_box_env():
    """A free box in each of four envs at ``decimation=4``, with the solver-error truncation of every RL env."""
    return build_env(
        EdenRLConfig(
            env_options=EnvOptions(num_envs=4, decimation=_DECIMATION),
            scene_options=SceneOptions(box=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.0, 0.0, 0.5))),
        ),
        reset=True,
    )


def _corrupt_env_1_in_sim_step(env, sim_step: int, code: int = ErrorCode.INVALID_CONTACT_NAN):
    """Register a callback writing, in the given sim step, the flags a rigid kernel sets in env 1's errno (by default
    those of a NaN contact force).
    """
    from genesis.utils.misc import qd_to_torch

    solver = env.rigid_solver
    sim_steps = []

    def corrupt_env_1() -> None:
        sim_steps.append(None)
        if len(sim_steps) == sim_step + 1:
            errno = qd_to_torch(solver._errno).clone()
            errno[1] = int(code)
            solver._errno.from_torch(errno)

    env.register_substep_callback(corrupt_env_1, before=True)
    return corrupt_env_1


@pytest.mark.parametrize("flagged_sim_step", range(_DECIMATION))
def test_a_solver_error_in_any_sim_step_of_a_control_step_resets_its_env_while_the_others_go_on(
    decimated_box_env, flagged_sim_step
):
    """Genesis raises the flags it latched at the start of its next ``scene.step()``, and a control step makes
    ``decimation`` of them before the termination manager resets the flagged env: an error in any sim step but the
    last raised for the whole scene, unless the env defers the check to its next control step.
    """
    env = decimated_box_env
    corrupt_env_1 = _corrupt_env_1_in_sim_step(env, flagged_sim_step)
    try:
        env.reset()
        env.step()
        assert env.termination_manager.solver_errors.tolist() == [False, True, False, False]
        assert env.episode_length_buf.tolist() == [1, 0, 1, 1]  # env 1 reset, the others carried on
        assert not env.get_solver_error_envs_mask().any()
        for _ in range(3):  # the next control steps check the flags again, and find none
            env.step()
    finally:
        env._pre_substep_callbacks.remove(corrupt_env_1)


def test_a_solver_error_no_reset_clears_raises_at_the_next_control_step():
    """Deferring the check past the first sim step of a control step hides nothing: a plain env resets nothing."""
    env = build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=4, decimation=_DECIMATION),
            scene_options=SceneOptions(box=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.0, 0.0, 0.5))),
        )
    )
    _corrupt_env_1_in_sim_step(env, 0)
    env.step()  # the flag is latched; the rest of this control step does not check it
    with pytest.raises(gs.GenesisException, match="not finite"):
        env.step()


@pytest.mark.parametrize(
    ("code", "message"),
    [
        (ErrorCode.OVERFLOW_CONTACTS, "Exceeding max number of post-pruning contact points"),
        (ErrorCode.OVERFLOW_CONTACTS | ErrorCode.INVALID_CONTACT_NAN, "Exceeding max number of post-pruning contact"),
        (ErrorCode.UNSUPPORTED_COLLISION_PAIR, "Collision between a non-convex geom and a terrain is not supported"),
    ],
    ids=["overflow", "overflow-and-nan", "unsupported-pair"],
)
def test_a_misconfigured_scene_raises_at_the_next_control_step_though_its_env_is_reset(
    decimated_box_env, code, message
):
    """A misconfiguration is fixed in the options, not by a reset that would hide it. The flagged env is truncated and
    reset like a corrupt one, so no value it read reaches the learner, but the env keeps its flag across the reset, and
    the next control step raises it with the option to change.
    """
    env = decimated_box_env
    corrupt_env_1 = _corrupt_env_1_in_sim_step(env, 0, code)
    try:
        env.reset()
        env.step()
        # Truncated like a corrupt env, so the reward and metric managers drop its values (tests/managers)
        assert env.termination_manager.solver_errors.tolist() == [False, True, False, False]
        assert env.episode_length_buf.tolist() == [1, 0, 1, 1]
        with pytest.raises(gs.GenesisException, match=message):
            env.step()
    finally:
        env._pre_substep_callbacks.remove(corrupt_env_1)
        env.rigid_solver.clear_errno()


@pytest.mark.parametrize("before", [False, True], ids=["after-step", "before-step"])
def test_a_misconfiguration_a_substep_callback_clears_raises_at_the_next_control_step(before):
    """A substep callback whose setter clears env 1's flags (as the MochiSolver's do) runs at every sim step, the first
    one of the next control step included: the env hands the kept flags back after it, right before the step that
    checks them, and on CPU marks the check as pending, which the deferred sim steps after the setter left unset.
    """
    env = build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=4, decimation=_DECIMATION),
            scene_options=SceneOptions(box=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.0, 0.0, 0.5))),
        )
    )
    _corrupt_env_1_in_sim_step(env, 0, ErrorCode.OVERFLOW_CONTACTS)
    env_1 = torch.tensor([1], device=env.device)
    calls = []

    def clear_env_1() -> None:
        calls.append(None)
        if not (before and len(calls) == 1):  # not between the corruption and the step that latches it
            env.rigid_solver.clear_errno(env_1)

    env.register_substep_callback(clear_env_1, before=before)
    env.step()
    assert not env.rigid_solver.get_error_envs_mask().any()  # the setter cleared the flag; the env kept it
    with pytest.raises(gs.GenesisException, match="Exceeding max number of post-pruning contact points"):
        env.step()


def test_a_flag_no_solver_registered_is_reset_like_a_corrupt_state(decimated_box_env):
    """A bit without a message never raises, so carrying it past the reset would reset its env at every control step."""
    env = decimated_box_env
    unregistered = 1 << 20
    assert not any(int(code) & unregistered for code in env.rigid_solver._error_messages)
    corrupt_env_1 = _corrupt_env_1_in_sim_step(env, 0, unregistered)
    try:
        env.reset()
        env.step()
        assert env.termination_manager.solver_errors.tolist() == [False, True, False, False]
        for _ in range(3):
            env.step()
        assert not env.termination_manager.solver_errors.any()
    finally:
        env._pre_substep_callbacks.remove(corrupt_env_1)


@pytest.mark.parametrize("as_mask", [False, True], ids=["ids", "mask"])
def test_a_reset_env_reports_no_control_force_until_its_next_step(decimated_box_env, as_mask):
    """``get_dofs_control_force`` reads the force the last step applied, which a reset leaves at the last force of the
    episode that ended: the first observation of the next one would read it.
    """
    env = decimated_box_env
    box = env.entities["box"]
    free_dofs = range(6)  # a free base's DOFs, which a controlled-DOF read (`dofs_name`) leaves out
    env.reset()
    box.control_dofs_force(torch.full((env.num_envs, 6), 0.1), dofs_idx_local=free_dofs)
    try:
        env.step()
        torch.testing.assert_close(box.get_dofs_control_force(free_dofs), torch.full((env.num_envs, 6), 0.1))

        envs_idx = torch.tensor([False, True, False, True]) if as_mask else torch.tensor([1, 3])
        env.reset(envs_idx=envs_idx.to(env.device))
        force = box.get_dofs_control_force(free_dofs)
        assert (force[[1, 3]] == 0.0).all()
        torch.testing.assert_close(force[[0, 2]], torch.full((2, 6), 0.1))
    finally:
        box.control_dofs_force(torch.zeros(env.num_envs, 6), dofs_idx_local=free_dofs)
