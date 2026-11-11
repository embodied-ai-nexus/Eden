"""Tests for ``eden.options.managers.base``: term options validate their parameter names against the
registered term at construction, on assignment and through ``model_copy(update=)``, even before the
manager is imported; a field left out defaults to the registered term's value; a function-era save loads
onto a class term's fields; and manager options hold only term entries.
No scene needed.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap

import pytest
from pydantic import ValidationError

import eden.terms  # noqa: F401 — the wrapper aliases load every manager, stamping each kind's registry
from eden.constants import EventMode, MetricDirection, MetricMode
from eden.managers.catalog import KINDS
from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY
from eden.options.managers.actions import ActionManagerOptions, ActionTermOptions
from eden.options.managers.base import _TERM_REGISTRY_MODULES, FunctionTermOptions, ManagerTermOptions
from eden.options.managers.commands import CommandTermOptions
from eden.options.managers.curricula import CurriculumTermOptions
from eden.options.managers.events import EventTermOptions
from eden.options.managers.metrics import MetricTermOptions
from eden.options.managers.modifiers import ActionModifierOptions, NoiseOptions
from eden.options.managers.motions import MotionTermOptions
from eden.options.managers.observations import ObservationTermOptions
from eden.options.managers.recorders import RecorderTermOptions
from eden.options.managers.rewards import RewardManagerOptions, RewardTermOptions
from eden.options.managers.terminations import TerminationTermOptions


@OBSERVATION_TERM_REGISTRY.register()
def kwargs_probe_obs(env, **kwargs):
    return kwargs


@pytest.mark.parametrize(
    "build, message",
    [
        (lambda: RewardTermOptions(name="upright", wieght=2.0), r"wieght.*'upright'"),
        (
            lambda: ObservationTermOptions(name="base_lin_vel", params={"entity_nmae": "robot"}),
            r"entity_nmae.*'base_lin_vel'",
        ),
        (
            lambda: CommandTermOptions(name="uniform_velocity_command", lin_vel_x_rnage=(-1.0, 1.0)),
            r"lin_vel_x_rnage.*'uniform_velocity_command'",
        ),
        (lambda: NoiseOptions(name="gaussian_noise", stdd=1.0), r"stdd.*'gaussian_noise'"),
    ],
    ids=["class-term-field", "function-params-key", "class-term-extra", "noise-model"],
)
def test_a_misspelled_parameter_of_a_named_term_raises_at_construction(build, message):
    with pytest.raises(ValueError, match=message):
        build()


def test_correct_parameters_construct_and_params_is_kept_raw_by_identity():
    """``params`` follows the ``morph`` escape-hatch pattern: ``None`` default, the dict kept as passed."""
    passed = {"entity_name": "robot"}

    assert ObservationTermOptions(name="base_lin_vel", params=passed, scale=2.0).params is passed
    assert RewardTermOptions().params is None
    command = CommandTermOptions(
        name="uniform_velocity_command",
        entity_name="robot",
        lin_vel_x_range=(-1.0, 1.0),
        resampling_time_range=(2.0, 4.0),
    )
    assert command.lin_vel_x_range == (-1.0, 1.0)
    assert NoiseOptions(name="gaussian_noise", mean=0.0, std=0.5).std == 0.5
    with pytest.raises(ValueError, match="params must be a dict"):
        RewardTermOptions(params=[("entity_name", "robot")])


def test_params_on_a_class_term_raises_rather_than_being_ignored():
    """A command options class has no ``params`` field; a reward one does, but ``upright`` is a class term."""
    with pytest.raises(ValueError, match=r"params"):
        CommandTermOptions(name="uniform_velocity_command", params={"a": 1})
    with pytest.raises(ValueError, match=r"class-based"):
        RewardTermOptions(name="upright", params={"a": 1})


def test_a_kwargs_function_accepts_any_params_key():
    assert ObservationTermOptions(name="kwargs_probe_obs", params={"anything": 1}).params == {"anything": 1}


def test_an_unregistered_name_defers_validation_to_build():
    assert RewardTermOptions(name="definitely_not_registered", some_param=1).some_param == 1


def test_only_the_function_capable_kinds_carry_a_params_field():
    """On a class-only kind a ``params`` field would be dead: silently ignored."""
    function_capable = (
        ObservationTermOptions,
        RewardTermOptions,
        TerminationTermOptions,
        EventTermOptions,
        CurriculumTermOptions,
        MetricTermOptions,
    )
    class_only = (
        ActionTermOptions,
        CommandTermOptions,
        RecorderTermOptions,
        MotionTermOptions,
        NoiseOptions,
        ActionModifierOptions,
    )
    assert all(issubclass(cls, FunctionTermOptions) for cls in function_capable)
    assert not any(issubclass(cls, FunctionTermOptions) for cls in class_only)


def test_an_assignment_is_validated_like_construction_and_a_rejected_one_rolls_back():
    options = RewardTermOptions(name="upright")
    with pytest.raises(ValueError, match="wieght"):
        options.wieght = 2.0
    assert "wieght" not in (options.__pydantic_extra__ or {})
    options.std = 0.5  # a legitimate class-term parameter still flows
    assert options.std == 0.5

    observation = ObservationTermOptions(name="base_lin_vel", params={"entity_name": "robot"})
    with pytest.raises(ValueError, match="entity_nmae"):
        observation.params = {"entity_nmae": "robot"}
    assert observation.params == {"entity_name": "robot"}


def test_assigning_the_name_last_validates_the_extras_accumulated_before_it():
    options = RewardTermOptions()
    options.bogus_param = 1  # no name yet: nothing to validate against

    with pytest.raises(ValueError, match="bogus_param"):
        options.name = "upright"

    assert (options.name, options.bogus_param) == (None, 1)


def test_dumps_carry_name_and_params_only_when_they_say_something():
    """A dump writes ``name`` and ``params`` only when set: no ``params: {}`` on a class-based term."""
    bare = RewardTermOptions().model_dump()
    assert "name" not in bare and "params" not in bare

    class_term = CommandTermOptions(name="uniform_velocity_command", entity_name="robot").model_dump()
    assert class_term["name"] == "uniform_velocity_command" and "params" not in class_term

    assert ObservationTermOptions(name="base_lin_vel", params={"entity_name": "robot"}).model_dump()["params"] == {
        "entity_name": "robot"
    }


# ---------------------------------------------------------------------------
# A field left out is the registered term's default, not the kind's
# ---------------------------------------------------------------------------


def test_a_function_terms_registration_defaults_fill_the_fields_left_out():
    """``error_vel_xy`` is registered cumulative, lower-is-better, scored at reset: the kind says none of that."""
    options = MetricTermOptions(name="error_vel_xy", success_threshold=0.5)

    assert (options.is_cumulative, options.direction, options.metric_mode) == (
        True,
        MetricDirection.LIB,
        MetricMode.RESET,
    )
    assert options.success_threshold == 0.5  # what the caller wrote is kept
    assert MetricTermOptions.model_validate(options.model_dump()).direction == MetricDirection.LIB  # and saved


def test_a_class_terms_own_class_attribute_fills_the_field_left_out():
    """``configure()`` always read it off the class; naming the term in the options did not."""
    assert EventTermOptions(name="push_by_velocity").mode == EventMode.INTERVAL
    assert EventTermOptions(name="push_by_velocity", mode="reset").mode == EventMode.RESET


def test_an_unregistered_or_absent_name_keeps_the_kinds_defaults():
    assert MetricTermOptions().direction == MetricDirection.HIB
    assert MetricTermOptions(name="definitely_not_registered").direction == MetricDirection.HIB


def test_a_required_field_is_not_filled_from_a_placeholder_class_attribute():
    """An action term's ``entity_name = ""`` is a placeholder; filling it would mask the missing field."""
    with pytest.raises(ValidationError, match="entity_name"):
        ActionTermOptions(name="implicit_pd_controller")


# ---------------------------------------------------------------------------
# model_copy(update=): pydantic's unvalidated path, validated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "options, typo",
    [
        (RewardTermOptions(name="upright", weight=2.0), "weigth"),
        (ObservationTermOptions(name="base_lin_vel", params={"entity_name": "robot"}), "sclae"),
        (CommandTermOptions(name="uniform_velocity_command", entity_name="robot"), "lin_vel_x_rnage"),
    ],
    ids=["reward", "observation", "command"],
)
def test_a_typo_in_a_copy_update_raises_at_the_call(options, typo):
    """It used to be parked as an extra while the real field kept its value, until ``env.build()``."""
    with pytest.raises(ValueError, match=rf"{typo}.*'{options.name}'"):
        options.model_copy(update={typo: 0.0})
    with pytest.raises(ValueError, match=rf"{typo}.*'{options.name}'"):
        options.updated(**{typo: 0.0})


def test_a_copy_update_validates_params_keys_and_passes_correct_keys():
    with pytest.raises(ValueError, match="entity_nmae"):
        ObservationTermOptions(name="base_lin_vel", params={"entity_name": "robot"}).model_copy(
            update={"params": {"entity_nmae": "robot"}}
        )

    source = RewardTermOptions(name="upright", weight=2.0)
    assert source.model_copy(update={"weight": 0.0}).weight == 0.0
    assert source.model_copy(update={"std": 0.5}).std == 0.5
    assert (source.model_copy().weight, source.weight) == (2.0, 2.0)


def test_manager_options_copies_can_add_and_replace_terms_and_refuse_a_non_term():
    """The copy enumerates its own children: a stale ``keys`` would hide an added term from the manager."""
    manager_options = RewardManagerOptions(upright=RewardTermOptions(name="upright", weight=2.0))
    manager_options.keys()  # materialize the cached property a shallow copy used to carry over

    added = manager_options.model_copy(update={"alive": RewardTermOptions(name="upright", weight=1.0)})
    assert added.term_keys() == ("upright", "alive")
    assert "alive" in added.keys() and "alive" not in manager_options.keys()

    replaced = manager_options.model_copy(update={"upright": manager_options.upright.updated(weight=0.0)})
    assert (replaced.upright.weight, manager_options.upright.weight) == (0.0, 2.0)

    with pytest.raises(TypeError, match="'uprihgt' must be an instance of"):
        manager_options.model_copy(update={"uprihgt": 0.0})


# ---------------------------------------------------------------------------
# Manager options: entries and disable_term
# ---------------------------------------------------------------------------


def test_manager_options_refuse_an_entry_that_is_not_term_options():
    with pytest.raises(TypeError, match="must be an instance of ActionTermOptions"):
        ActionManagerOptions(term_a={"entity_name": "robot", "dofs_name": ["dof1"]})


def test_an_entry_assigned_onto_manager_options_is_type_checked_and_a_refused_one_rolls_back():
    """Construction and ``model_copy`` checked their entries; an assignment was accepted and died at build."""
    options = RewardManagerOptions(alive=RewardTermOptions(name="upright", weight=1.0))
    original = options.alive

    with pytest.raises(TypeError, match=r"Term 'bogus' must be an instance of RewardTermOptions or RewardGroupOptions"):
        options.bogus = "x"
    with pytest.raises(TypeError, match="'alive' must be an instance of"):
        options.alive = 0.0

    assert options.term_keys() == ("alive",) and options.alive is original
    assert "bogus" not in options.model_dump(exclude_unset=True)
    options.upright = RewardTermOptions(name="upright", weight=0.5)
    assert options.term_keys() == ("alive", "upright")


def test_a_declared_manager_field_write_is_left_to_pydantic():
    options = RewardManagerOptions(alive=RewardTermOptions(name="upright", weight=1.0))

    options.log_per_term_rewards = False

    assert options.log_per_term_rewards is False and options.term_keys() == ("alive",)


def test_a_misspelled_manager_field_is_named_as_the_field_it_looks_like():
    with pytest.raises(TypeError, match=r"Did you mean the field 'log_per_term_rewards'\?"):
        RewardManagerOptions().log_per_term_reward = False


@pytest.mark.parametrize(
    "given_by, name, message",
    [
        (
            "construction",
            "keys",
            r"RewardManagerOptions.keys is already a property, and a term under that name hides it",
        ),
        ("construction", "lock", r"RewardManagerOptions.lock is already a method, and a term under that name hides it"),
        ("assignment", "keys", r"RewardManagerOptions.keys is already a property, .* never adds a term"),
        ("assignment", "lock", r"RewardManagerOptions.lock is already a method, .* never adds a term"),
    ],
)
def test_manager_options_refuse_a_term_named_like_a_member_of_their_class(given_by, name, message):
    """A constructed ``lock`` term hid ``lock()`` (the lock then called the term); an assigned one was never a term."""
    term = RewardTermOptions(name="upright", weight=1.0)
    options = RewardManagerOptions()
    with pytest.raises(TypeError, match=rf"'{name}' cannot name a term of RewardManagerOptions: {message}"):
        if given_by == "construction":
            RewardManagerOptions(**{name: term})
        else:
            setattr(options, name, term)
    if given_by == "assignment":
        assert options.term_keys() == ()
        assert not isinstance(getattr(options, name), RewardTermOptions), "the refused write must not shadow it"


def test_disable_term_removes_a_term_and_names_an_unknown_one():
    options = ActionManagerOptions(
        term_a=ActionTermOptions(entity_name="robot", dofs_name=["dof1"]),
        term_b=ActionTermOptions(entity_name="robot", dofs_name=["dof2"]),
    )

    options.disable_term("term_a")

    assert options.term_keys() == ("term_b",)
    with pytest.raises(KeyError, match="Term 'term_a' not found"):
        options.disable_term("term_a")


def test_a_locked_manager_options_refuses_disable_term():
    options = ActionManagerOptions(term_a=ActionTermOptions(entity_name="robot", dofs_name=["dof1"]))
    options.lock()

    with pytest.raises(RuntimeError, match="locked"):
        options.disable_term("term_a")


# ---------------------------------------------------------------------------
# The registry a term's options validate against, resolved without the manager imported
# ---------------------------------------------------------------------------


def _in_fresh_interpreter(body: str) -> str:
    """Run ``body`` in a new process and return its stdout, failing the test on a non-zero exit.

    Parameters
    ----------
    body : str
        Python source, dedented before it is run.

    Returns
    -------
    str
        The script's stdout.
    """
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(body)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.mark.parametrize(
    ("options_module", "options_class", "term", "kind"),
    [
        ("eden.options.managers.rewards", "RewardTermOptions", "action_l2", "reward"),
        ("eden.options.managers.observations", "ObservationTermOptions", "base_lin_vel", "observation"),
        ("eden.options.managers.events", "EventTermOptions", "push_by_velocity", "event"),
    ],
)
def test_term_options_validate_their_parameters_without_the_manager_imported(options_module, options_class, term, kind):
    """A saved config rebuilt through the options class alone used to find no registry and skip the check."""
    out = _in_fresh_interpreter(
        f"""
        import sys
        from {options_module} import {options_class}
        assert "eden.managers" not in sys.modules, "the options module already pulled the manager in"
        try:
            {options_class}(name="{term}", nonsense_parameter=1)
        except Exception as error:
            print("raised", "eden terms --kind {kind}" in str(error))
        else:
            print("silent")
        """
    )
    assert out.strip() == "raised True"


def _construct_while_eden_managers_imports(construct: str) -> str:
    """Evaluate ``construct`` from inside ``import eden.managers``, as module-scope code there would run.

    An import hook fires when ``eden.managers`` asks for ``reward_manager``, before that module has
    constructed (and so stamped) ``REWARD_TERM_REGISTRY``.

    Parameters
    ----------
    construct : str
        An expression over ``RewardTermOptions`` / ``FunctionTermOptions`` / ``ManagerTermOptions``.

    Returns
    -------
    str
        ``"returned <repr>"`` or ``"raised <message>"``.
    """
    return _in_fresh_interpreter(
        f"""
        import importlib.abc, sys
        from eden.options.managers.base import FunctionTermOptions, ManagerTermOptions
        from eden.options.managers.rewards import RewardTermOptions

        class OutOfTreeTermOptions(ManagerTermOptions):
            pass

        class Hook(importlib.abc.MetaPathFinder):
            outcome = "hook never fired"

            def find_spec(self, name, path, target=None):
                if name == "eden.managers.reward_manager":
                    sys.meta_path.remove(self)
                    try:
                        Hook.outcome = "returned " + repr({construct})
                    except Exception as error:
                        Hook.outcome = "raised " + str(error)
                return None

        sys.meta_path.insert(0, Hook())
        import eden.managers
        print(Hook.outcome)
        """
    ).strip()


def test_a_kinds_options_built_mid_import_raise_rather_than_skip_the_check():
    """A module still importing on this thread comes back half-built; ``None`` there would lose the check silently."""
    outcome = _construct_while_eden_managers_imports('RewardTermOptions(name="action_l2", weight=1.0)')

    assert outcome.startswith("raised "), outcome
    assert "still being imported on this thread" in outcome
    assert "eden.managers" in outcome
    assert "at module scope inside eden.managers.*" in outcome
    assert "try/except" in outcome  # pre-empts the wrong fix, per docs/guides/error-messages.md


@pytest.mark.parametrize("cls", ["FunctionTermOptions", "OutOfTreeTermOptions"])
def test_a_class_no_registry_could_claim_resolves_to_none_even_mid_import(cls):
    """The unfinished module is shared by nine kinds; keying the raise on it alone would fire for these too."""
    assert _construct_while_eden_managers_imports(f"{cls}._resolve_term_registry()") == "returned None"


def test_which_classes_a_registry_claims_matches_where_they_are_declared():
    """``_stamped_by_a_registry`` answers from the declaring module; pin it against the actual stamps."""
    for module_name in _TERM_REGISTRY_MODULES:
        importlib.import_module(module_name)  # construct every registry, so every stamp has landed

    def subclasses(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from subclasses(sub)

    in_tree = sorted(
        {sub for sub in subclasses(ManagerTermOptions) if sub.__module__.startswith("eden.")},
        key=lambda cls: (cls.__module__, cls.__name__),
    )
    assert in_tree
    disagreed = [
        f"{cls.__module__}.{cls.__name__}: declared-in says {cls._stamped_by_a_registry()}, "
        f"stamp says {cls.__dict__.get('_term_registry_') is not None}"
        for cls in in_tree
        if cls._stamped_by_a_registry() != (cls.__dict__.get("_term_registry_") is not None)
    ]
    assert not disagreed, "\n  ".join(["_stamped_by_a_registry disagrees with the actual stamp:", *disagreed])


def test_an_options_class_no_registry_claims_still_constructs_quietly():
    """An unclaimed class skips the construction-time check, leaving it to env build: never is not yet."""

    class _UnclaimedTermOptions(ManagerTermOptions):
        """A kind no ``TermRegistry`` has ever been constructed for."""

    assert _UnclaimedTermOptions._resolve_term_registry() is None
    assert _UnclaimedTermOptions(name="whatever", not_a_real_parameter=1).not_a_real_parameter == 1


def test_the_term_registry_module_list_covers_every_kind():
    """A kind whose registry lives in a module the tuple misses would lose its parameter check silently."""
    assert set(_TERM_REGISTRY_MODULES) == {kind.module for kind in KINDS}


# ---------------------------------------------------------------------------
# A function-era save loads onto the class term's fields: a load shim only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("options_cls", "key", "module"),
    [
        (RewardTermOptions, "metric_reward", "rewards"),
        (ObservationTermOptions, "metric_observation", "observations"),
        (TerminationTermOptions, "on_metric_success", "terminations"),
    ],
)
def test_a_function_era_save_loads_onto_the_class_fields(options_cls, key, module):
    saved = {
        "name": key,
        "params": {"metric_name": "score"},
        "_option_module_": f"eden.options.managers.{module}",
        "_option_class_": options_cls.__name__,
    }
    options = options_cls.model_validate(saved)
    assert options.metric_name == "score" and options.params is None
    assert "params" not in options.model_dump()


def test_every_declared_key_is_hoisted_and_the_term_fields_are_kept():
    options = RewardTermOptions.model_validate(
        {"name": "metric_reward", "weight": 0.5, "params": {"metric_name": "score", "allow_cumulative": True}}
    )
    assert (options.metric_name, options.allow_cumulative, options.weight) == ("score", True, 0.5)


@pytest.mark.parametrize(
    "params",
    [
        {"metric_nam": "score"},  # a key the function never took
        {"metric_name": "score", "allow_cumulativ": True},  # one bad key spoils the whole dict
        {"weight": 3.0},  # a term-level field, which never lived in params
    ],
)
def test_params_outside_the_function_era_keys_are_not_hoisted(params):
    with pytest.raises(ValueError, match="class-based"):
        RewardTermOptions(name="metric_reward", params=params)


def test_params_colliding_with_a_top_level_field_raise():
    with pytest.raises(ValueError, match="collide with top-level options fields"):
        RewardTermOptions(name="metric_reward", metric_name="other", params={"metric_name": "score"})


def test_assigning_params_on_a_class_term_raises():
    options = RewardTermOptions(name="metric_reward", metric_name="score")
    with pytest.raises(ValueError, match="only used by function-based terms"):
        options.params = {"metric_name": "other"}
    assert options.params is None
