"""Tests for ``eden.managers.base``: the flat-term API, ``build_term`` and the build note naming a config entry,
the fields a kind's class term and function wrapper share, the ``_requires_env_options`` precondition a term
declares, a function term's registered defaults reaching the built term, ``resolve_entity``, the
``term_subject`` an entity-bound term's errors open with, and the ``compute_once`` passes a step shares
across managers.
"""

from __future__ import annotations

import inspect
import math
import pathlib
import re

import pytest
import torch

from eden.constants import MetricDirection
from eden.managers.base import build_term
from eden.managers.curriculum_manager import CurriculumTerm, CurriculumTermFuncWrapper
from eden.managers.event_manager import (
    EVENT_TERM_REGISTRY,
    EventManager,
    EventTerm,
    EventTermFuncWrapper,
    _EventTermFields,
)
from eden.managers.metric_manager import METRIC_TERM_REGISTRY, MetricTerm, MetricTermFuncWrapper, SuccessCheckerMixin
from eden.managers.observation_manager import (
    OBSERVATION_TERM_REGISTRY,
    ObservationManager,
    ObservationTerm,
    ObservationTermFuncWrapper,
    _ObservationTermFields,
)
from eden.managers.reward_manager import (
    REWARD_TERM_REGISTRY,
    RewardManager,
    RewardTerm,
    RewardTermFuncWrapper,
    _RewardTermFields,
)
from eden.managers.termination_manager import (
    TERMINATION_TERM_REGISTRY,
    TerminationTerm,
    TerminationTermFuncWrapper,
    _TerminationTermFields,
)
from eden.managers.terms.events.domain_rand import ApplyExternalForce
from eden.managers.terms.observations.locomotion import ProjectedGravity
from eden.managers.terms.observations.manipulation import GripperState
from eden.managers.terms.rewards.locomotion import WeightedPosePenalty
from eden.options import BoxOptions, PlaneOptions, SceneOptions
from eden.options.envs import EnvOptions
from eden.options.managers.curricula import CurriculumTermOptions
from eden.options.managers.events import EventManagerOptions, EventTermOptions
from eden.options.managers.metrics import MetricManagerOptions, MetricTermOptions
from eden.options.managers.observations import (
    ObservationGroupOptions,
    ObservationManagerOptions,
    ObservationTermOptions,
)
from eden.options.managers.rewards import RewardManagerOptions, RewardTermOptions
from eden.options.managers.terminations import TerminationManagerOptions, TerminationTermOptions
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env

# 90 deg about +x: world down (0, 0, -1) reads as (0, -1, 0) in the rotated body frame.
_ROLL90 = (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)


@REWARD_TERM_REGISTRY.register()
def base_test_reward(env) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


@EVENT_TERM_REGISTRY.register()
class RequiresBatchedLinks(EventTerm):
    """Declares a scene-wide precondition; never computed here."""

    _requires_env_options = ("batch_links_info",)

    def compute(self, envs_idx=None):
        raise AssertionError("compute() must not run in these tests")


@pytest.fixture(scope="module")
def batched_links_env(make_box_env):
    return make_box_env(env_options=EnvOptions(num_envs=4, batch_links_info=True))


@pytest.fixture(scope="module")
def rolled_env():
    """Two fixed cubes, ``robot`` upright and ``rolled`` rolled 90 degrees."""
    cube = dict(size=(0.1, 0.1, 0.1), fixed=True)
    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=2),
            scene_options=SceneOptions(
                plane=PlaneOptions(),
                robot=BoxOptions(**cube, default_root_pos=(0.0, 0.0, 0.5)),
                rolled=BoxOptions(**cube, default_root_pos=(1.0, 0.0, 0.5), default_root_quat=_ROLL90),
            ),
        )
    )


def _rewards(env) -> RewardManager:
    return RewardManager(
        env=env,
        options=RewardManagerOptions(
            alive=RewardTermOptions(name="base_test_reward"), upright=RewardTermOptions(name="base_test_reward")
        ),
    )


# ---------------------------------------------------------------------------
# The flat-term API
# ---------------------------------------------------------------------------


def test_get_term_and_get_term_options_resolve_every_active_term(box_env):
    manager = _rewards(box_env)

    assert manager.active_terms == ["alive", "upright"]
    for name in manager.active_terms:
        assert manager.terms[name] is manager.get_term(name)
        assert manager.get_term_options(name) is manager.get_term(name).options


def test_a_get_term_miss_lists_the_active_terms(box_env):
    """A bare ``KeyError: 'dofs_pos_controller'`` would say nothing about what the manager holds."""
    manager = _rewards(box_env)

    with pytest.raises(
        KeyError, match=r"No active term named 'aliv' in RewardManager. Available options: alive, upright"
    ):
        manager.get_term("aliv")
    with pytest.raises(ValueError, match="Term 'aliv' not found in active terms"):
        manager.get_term_options("aliv")


@pytest.mark.parametrize(
    "manager_cls, absent",
    [(EventManager, ("get_term_options", "terms")), (ObservationManager, ("get_term_options", "terms"))],
    ids=["event", "observation"],
)
def test_the_hierarchical_managers_do_not_inherit_the_flat_api(manager_cls, absent):
    """Events by mode and observations by group would have inherited a flat API they never populate.

    ``EventManager.get_term`` is its own mode-agnostic lookup, not the flat one.
    """
    for attribute in absent:
        assert not hasattr(manager_cls, attribute), attribute


def test_a_term_that_fails_to_build_is_noted_with_its_config_entry(box_env):
    """The note names the entry: a task can hold a dozen entries of one term class."""
    with pytest.raises(ValueError) as excinfo:
        EventManager(env=box_env, options=EventManagerOptions(com_dr=RequiresBatchedLinks.configure(mode="startup")))

    assert "while building EventManager term 'com_dr' (name='requires_batched_links')" in excinfo.value.__notes__


def test_a_nameless_term_is_refused_at_build_naming_the_entry(box_env):
    """``RewardTermOptions()`` is a legal placeholder; the build refuses it instead of passing ``name=None`` on."""
    with pytest.raises(ValueError, match=r"RewardManager term 'unnamed' names no registered term \(name=None\)"):
        RewardManager(
            env=box_env,
            options=RewardManagerOptions(alive=RewardTermOptions(name="base_test_reward"), unnamed=RewardTermOptions()),
        )


def test_build_term_returns_the_built_term(box_env):
    term = build_term(REWARD_TERM_REGISTRY, box_env, RewardTermOptions(name="base_test_reward"), label="Composite.term")

    assert isinstance(term, RewardTermFuncWrapper)
    assert term.name == "base_test_reward"


def test_build_term_refuses_a_nameless_term_opening_with_its_label(box_env):
    """The label opens the message; the fix names both ways to give the entry a name."""
    with pytest.raises(ValueError) as excinfo:
        build_term(REWARD_TERM_REGISTRY, box_env, RewardTermOptions(), label="Composite.terms[1]")

    message = str(excinfo.value)
    assert message.startswith("Composite.terms[1] names no registered term (name=None)"), message
    assert "Write it as <TermClass>.configure(...), which sets name, or give it name='<registry key>'" in message


def test_build_term_notes_a_build_failure_with_its_label(box_env):
    """The type and message are the term's own; only the note is added."""
    with pytest.raises(ValueError, match=r"requires `EnvOptions\(batch_links_info=True\)`") as excinfo:
        build_term(EVENT_TERM_REGISTRY, box_env, RequiresBatchedLinks.configure(mode="startup"), label="Composite.term")

    assert excinfo.value.__notes__ == ["while building Composite.term (name='requires_batched_links')"]


# ---------------------------------------------------------------------------
# A kind's class term and its function wrapper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "term_cls, wrapper_cls, options_cls",
    [
        (CurriculumTerm, CurriculumTermFuncWrapper, CurriculumTermOptions),
        (EventTerm, EventTermFuncWrapper, EventTermOptions),
        (MetricTerm, MetricTermFuncWrapper, MetricTermOptions),
        (ObservationTerm, ObservationTermFuncWrapper, ObservationTermOptions),
        (RewardTerm, RewardTermFuncWrapper, RewardTermOptions),
        (TerminationTerm, TerminationTermFuncWrapper, TerminationTermOptions),
    ],
    ids=["curriculum", "event", "metric", "observation", "reward", "termination"],
)
def test_a_kinds_class_term_and_function_wrapper_accept_the_same_fields(term_cls, wrapper_cls, options_cls):
    """Declared once for both: the options class reaches each through its generic base, and the catalog's
    per-kind shared fields (read off the wrapper) are exactly what a class term of the kind accepts too.
    """
    assert term_cls._options_class_ is options_cls
    assert wrapper_cls._options_class_ is options_cls
    assert term_cls.get_configure_specs() == tuple(
        spec for spec in wrapper_cls.get_configure_specs() if spec.name != "params"
    )


@pytest.mark.parametrize(
    "term_cls, mixin",
    [
        (EventTerm, _EventTermFields),
        (MetricTerm, SuccessCheckerMixin),
        (ObservationTerm, _ObservationTermFields),
        (RewardTerm, _RewardTermFields),
        (TerminationTerm, _TerminationTermFields),
    ],
    ids=["event", "metric", "observation", "reward", "termination"],
)
def test_the_api_page_renders_every_field_a_kinds_term_inherits(term_cls, mixin):
    """The fields live on a mixin the API page does not walk, so ``docs/api/managers.md`` names them per module."""
    assert issubclass(term_cls, mixin)
    annotations = inspect.get_annotations(mixin)
    properties = [name for name, value in vars(mixin).items() if isinstance(value, property)]
    fields = {name for name in [*annotations, *properties] if not name.startswith("_")}

    page = (pathlib.Path(__file__).parents[2] / "docs" / "api" / "managers.md").read_text()
    block = re.search(rf"::: {re.escape(term_cls.__module__)}\n((?:    .*\n)*)", page)
    listed = re.search(r"inherited_members: \[(.*)\]", block.group(1)) if block else None

    assert listed is not None, f"the {term_cls.__module__} block lists no inherited_members"
    assert fields <= {name.strip() for name in listed.group(1).split(",")}


# ---------------------------------------------------------------------------
# _requires_env_options
# ---------------------------------------------------------------------------


def test_an_unset_env_flag_raises_at_construction_naming_the_term_and_the_flag(box_env, batched_links_env):
    """A ``ValueError``, not the ``assert`` it was: ``python -O`` stripped it, and DR then randomized nothing."""
    with pytest.raises(ValueError, match=r"RequiresBatchedLinks requires `EnvOptions\(batch_links_info=True\)`"):
        RequiresBatchedLinks(box_env, EventTermOptions())

    RequiresBatchedLinks(batched_links_env, EventTermOptions())  # the negative space


@pytest.mark.parametrize(
    "declared, error, message",
    [
        ("batch_links_info", TypeError, "bare string"),  # iterating it would report 'b' as the field name
        (("batch_dof_info",), ValueError, "not EnvOptions fields"),
    ],
    ids=["bare-string", "unknown-flag"],
)
def test_a_malformed_declaration_is_refused_at_class_definition(declared, error, message):
    """The catalog renders the declaration without instantiating the term, so a typo would ship."""
    with pytest.raises(error, match=message):
        type("Declared", (EventTerm,), {"_requires_env_options": declared, "compute": lambda self, envs_idx=None: None})


def test_a_function_terms_declaration_is_validated_at_registration():
    """It declares through its registration, which synthesizes a real wrapper subclass."""
    with pytest.raises(ValueError, match="not EnvOptions fields"):

        @EVENT_TERM_REGISTRY.register(_requires_env_options=("batch_dof_info",))
        def bad_declaration_term(env, envs_idx=None):
            """Probe term."""


# ---------------------------------------------------------------------------
# A function term's registered defaults
# ---------------------------------------------------------------------------
# Each kind whose function wrapper has a term-level field. The observation and curriculum wrappers have
# none (an observation's scale/noise/clip are options-only), so there is nothing for register() to default.


@REWARD_TERM_REGISTRY.register(weight=2.5)
def weighted_at_registration(env) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


@TERMINATION_TERM_REGISTRY.register(time_out=True)
def timeout_at_registration(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


@EVENT_TERM_REGISTRY.register(priority=7)
def prioritized_at_registration(env, envs_idx=None) -> None:
    """Probe term."""


@METRIC_TERM_REGISTRY.register(direction=MetricDirection.LIB, success_threshold=0.2)
def lower_is_better_at_registration(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)


_REGISTERED_DEFAULTS = [
    pytest.param(RewardTermFuncWrapper, RewardTermOptions, weighted_at_registration, {"weight": 2.5}, id="reward"),
    pytest.param(
        TerminationTermFuncWrapper,
        TerminationTermOptions,
        timeout_at_registration,
        {"time_out": True},
        id="termination",
    ),
    pytest.param(EventTermFuncWrapper, EventTermOptions, prioritized_at_registration, {"priority": 7}, id="event"),
    pytest.param(
        MetricTermFuncWrapper,
        MetricTermOptions,
        lower_is_better_at_registration,
        {"direction": MetricDirection.LIB, "success_threshold": 0.2},
        id="metric",
    ),
]


@pytest.mark.parametrize("path", ["configure", "options"])
@pytest.mark.parametrize("wrapper, options_cls, func, registered", _REGISTERED_DEFAULTS)
def test_a_function_terms_registered_defaults_reach_the_built_term(
    box_env, path, wrapper, options_cls, func, registered
):
    """They lived only on the per-registration wrapper subclass, and the kind's defaults overwrote them at build."""
    options = wrapper.configure(func=func) if path == "configure" else options_cls(name=func.__name__)
    term = build_term(type(options)._term_registry_, box_env, options, label="test")

    assert {field: getattr(options, field) for field in registered} == registered  # so the saved config says so
    assert {field: getattr(term, field) for field in registered} == registered
    assert all(getattr(wrapper, field) != value for field, value in registered.items())  # the kind's are untouched


@pytest.mark.parametrize("path", ["configure", "options"])
@pytest.mark.parametrize("wrapper, options_cls, func, registered", _REGISTERED_DEFAULTS)
def test_a_field_set_explicitly_wins_over_the_registered_default(box_env, path, wrapper, options_cls, func, registered):
    explicit = {field: getattr(wrapper, field) for field in registered}  # the kind's defaults, spelled out
    if path == "configure":
        options = wrapper.configure(func=func, **explicit)
    else:
        options = options_cls(name=func.__name__, **explicit)

    term = build_term(type(options)._term_registry_, box_env, options, label="test")
    assert {field: getattr(term, field) for field in registered} == explicit


# ---------------------------------------------------------------------------
# resolve_entity: resolved lazily, so a swap of env.entities[name] is seen
# ---------------------------------------------------------------------------


def test_a_term_resolving_lazily_reads_the_entity_swapped_in_after_build(rolled_env):
    """What deployment does (sim -> ``RobotStateEntity``); a cached handle would keep reading the sim."""
    term = ProjectedGravity(rolled_env, options=ProjectedGravity.configure())
    term.build()
    sim = rolled_env.entities["robot"]
    torch.testing.assert_close(term.compute(), torch.tensor([[0.0, 0.0, -1.0]] * 2, device=rolled_env.device))

    rolled_env.entities["robot"] = rolled_env.entities["rolled"]  # the deployment swap
    try:
        torch.testing.assert_close(
            term.compute(), torch.tensor([[0.0, -1.0, 0.0]] * 2, device=rolled_env.device), atol=1e-5, rtol=0
        )
    finally:
        rolled_env.entities["robot"] = sim


def test_an_unknown_entity_name_lists_the_entities_that_exist(rolled_env):
    term = ProjectedGravity(rolled_env, options=ProjectedGravity.configure(entity_name="cube"))
    term.build()
    with pytest.raises(KeyError) as excinfo:
        term.compute()
    assert "'cube'" in str(excinfo.value) and "'robot'" in str(excinfo.value) and "'rolled'" in str(excinfo.value)


@pytest.mark.parametrize(
    ("term_cls", "options", "rest"),
    [
        (GripperState, dict(dofs_name=[]), " needs at least one gripper DOF name."),
        (ApplyExternalForce, dict(mode="reset", prob=2.0), ": prob must be in [0.0, 1.0], got 2.0."),
        (WeightedPosePenalty, dict(pose_weights=[1.0] * 99), " expected 6 pose weights, got 99."),
    ],
    ids=["observation", "event", "reward"],
)
def test_a_configuration_error_names_the_term_class_and_its_entity(box_env, term_cls, options, rest):
    """Every kind opens the error with the same ``term_subject``, ``"<Class> on entity '<name>'"``.

    Not the config key: the build note ``ManagerBase._build_term`` adds already names it.
    """
    term = term_cls(box_env, options=term_cls.configure(entity_name="box", **options))
    with pytest.raises(ValueError) as excinfo:
        term.build()
    assert str(excinfo.value) == f"{term_cls.__name__} on entity 'box'{rest}"


# ---------------------------------------------------------------------------
# compute_once across managers
# ---------------------------------------------------------------------------

#: One entry per production of the shared intermediate below, so a test can count them.
_PRODUCTIONS: list[str] = []


def _graded_shared(env, manager) -> torch.Tensor:
    """A fresh random draw per production, recorded, so sharing is visible both as a count and as equality."""

    def producer() -> torch.Tensor:
        _PRODUCTIONS.append(type(manager).__name__)
        return torch.rand(env.num_envs, device=env.device)

    # The build probes an observation term's shape before ``env.observation_manager`` exists.
    return producer() if manager is None else manager.compute_once(("graded_test_shared",), producer)


@METRIC_TERM_REGISTRY.register()
def graded_shared_metric(env) -> torch.Tensor:
    return _graded_shared(env, env.metric_manager)


@TERMINATION_TERM_REGISTRY.register()
def graded_shared_done(env) -> torch.Tensor:
    return _graded_shared(env, env.termination_manager) > 2.0  # never: only the read matters


@REWARD_TERM_REGISTRY.register()
def graded_shared_reward(env) -> torch.Tensor:
    return _graded_shared(env, env.reward_manager)


@pytest.fixture(scope="module")
def graded_sharing_env(make_box_env):
    return make_box_env(
        metric_options=MetricManagerOptions(shared=MetricTermOptions(name="graded_shared_metric")),
        termination_options=TerminationManagerOptions(shared=TerminationTermOptions(name="graded_shared_done")),
        reward_options=RewardManagerOptions(shared=RewardTermOptions(name="graded_shared_reward")),
    )


def test_a_step_grades_metrics_terminations_and_rewards_off_one_production(graded_sharing_env):
    """They read one post-physics state, so the env opens one pass over its whole graded phase."""
    env = graded_sharing_env
    _PRODUCTIONS.clear()
    with torch.inference_mode():
        env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device))

    assert _PRODUCTIONS == ["MetricManager"]
    torch.testing.assert_close(env.reward_manager.get_term_value("shared"), env.metric_manager.get_value("shared"))


def test_a_manager_computed_on_its_own_opens_a_pass_of_its_own(graded_sharing_env):
    env = graded_sharing_env
    _PRODUCTIONS.clear()
    with torch.inference_mode():
        env.metric_manager.compute()
        env.reward_manager.compute(dt=env.dt)

    assert _PRODUCTIONS == ["MetricManager", "RewardManager"]


def test_a_pass_opened_inside_another_joins_it_and_leaves_it_open(box_env):
    """A manager's term loop inside the env's graded pass shares the env's memo: it belongs to the pass."""
    manager = RewardManager(env=box_env, options=RewardManagerOptions())
    with box_env.compute_pass():
        with manager.compute_pass():
            assert manager.compute_once("key", lambda: "first") == "first"
        assert box_env.compute_once("key", lambda: "second") == "first"
    assert manager.compute_once("key", lambda: "outside") == "outside"


@OBSERVATION_TERM_REGISTRY.register()
def obs_shared_first(env) -> torch.Tensor:
    return _graded_shared(env, getattr(env, "observation_manager", None))[:, None]


@OBSERVATION_TERM_REGISTRY.register()
def obs_shared_doubled(env) -> torch.Tensor:
    return 2.0 * _graded_shared(env, getattr(env, "observation_manager", None))[:, None]


def test_an_observation_compute_shares_an_intermediate_across_its_terms_and_groups(make_box_env):
    """The critic's body-position and body-orientation observations read one anchor-relative transform."""
    env = make_box_env(
        observation_options=ObservationManagerOptions(
            policy=ObservationGroupOptions(first=ObservationTermOptions(name="obs_shared_first")),
            critic=ObservationGroupOptions(doubled=ObservationTermOptions(name="obs_shared_doubled")),
        )
    )
    _PRODUCTIONS.clear()
    with torch.inference_mode():
        obs = env.observation_manager.compute()

    assert _PRODUCTIONS == ["ObservationManager"]
    torch.testing.assert_close(obs["critic"], 2.0 * obs["policy"])
