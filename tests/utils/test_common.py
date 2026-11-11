"""Tests for ``eden.utils.common``: ``ConfigurableMixin`` / ``ConfigurableFuncWrapperMixin`` parameter discovery,
``configure()`` and its validation, the build-time backstop, copies (never aliases) of mutable values, and the
configuration fingerprint (``frozen_value`` / ``configured_fingerprint``).
"""

from __future__ import annotations

import dataclasses
from types import MappingProxyType

import genesis as gs
import numpy as np
import pytest
import torch
from genesis.options.surfaces import Surface
from pydantic import ValidationError

from eden import metrics, observations, rewards
from eden.constants import MetricDirection, MetricMode
from eden.options.managers.observations import ObservationTermOptions
from eden.options.managers.rewards import RewardTermOptions
from eden.options.options import ConfigurableOptions, DynamicConfigurableOptions
from eden.terms import MetricTerm, ObsTerm, RewardTerm
from eden.utils.common import ConfigurableMixin, frozen_value


class SimpleOptions(DynamicConfigurableOptions):
    test_param: str = "default"
    number_param: int = 42
    list_param: list = []


class SimpleConfigurable(ConfigurableMixin[SimpleOptions]):
    test_param: str = "default"
    number_param: int = 42
    list_param: list = []


class ForbidOptions(ConfigurableOptions):
    """``extra="forbid"`` with no ``name`` field: the Camera / Sensor / Entity pattern."""

    test_param: str = "default"


class ForbidConfigurable(ConfigurableMixin[ForbidOptions]):
    test_param: str = "default"


class ForbidNamedOptions(ConfigurableOptions):
    name: str | None = None
    test_param: str = "default"


class ForbidNamedConfigurable(ConfigurableMixin[ForbidNamedOptions]):
    test_param: str = "default"


# ---------------------------------------------------------------------------
# Parameters and configure()
# ---------------------------------------------------------------------------


def test_the_parameters_are_exactly_the_public_plain_class_attributes_inherited_ones_included():
    """An exact set, so a newly-leaked category (a method, a property) fails; a leak would make
    ``configure(my_method=...)`` silently accepted.
    """

    class Subject(SimpleConfigurable):
        declared: str = "yes"
        _private: str = "no"

        @property
        def computed(self):
            return 42

        def my_method(self):
            return "result"

    class Empty(ConfigurableMixin):
        pass

    assert set(Subject.get_parameter_names()) == {"declared", "test_param", "number_param", "list_param"}
    assert Empty.get_parameter_names() == []


def test_configure_takes_the_given_values_and_stamps_the_snake_case_name():
    class OverriddenConfigurable(SimpleConfigurable):
        test_param: str = "overridden"

    options = SimpleConfigurable.configure(test_param="custom", number_param=100)

    assert (options.test_param, options.number_param, options.name) == ("custom", 100, "simple_configurable")
    assert OverriddenConfigurable.configure().test_param == "overridden"  # a subclass default reaches configure


def test_configure_reserves_name_and_lists_the_allowed_parameters_for_an_unknown_one():
    with pytest.raises(ValueError, match="reserved"):
        SimpleConfigurable.configure(name="custom_name")

    with pytest.raises(ValueError) as exc_info:
        SimpleConfigurable.configure(invalid_param="value")
    message = str(exc_info.value)
    assert "invalid_param" in message and "allowed parameters" in message.lower()
    assert all(name in message for name in ("test_param", "number_param", "list_param"))


def test_a_forbid_options_class_gets_the_name_only_if_it_declares_one():
    """Managers resolve terms by the injected name; a forbid class without the field must not trip on it."""
    options = ForbidConfigurable.configure(test_param="custom")

    assert options.test_param == "custom" and "name" not in options.model_dump()
    assert ForbidNamedConfigurable.configure().name == "forbid_named_configurable"


def test_camera_and_sensor_configure_cleanly_and_keep_their_typo_protection():
    from eden.entities.camera import Camera
    from eden.entities.sensor import Sensor

    camera_options = Camera.configure(cam_fov=60.0)

    assert camera_options.cam_fov == 60.0 and "name" not in camera_options.model_dump()
    assert "name" not in Sensor.configure().model_dump()
    with pytest.raises(ValidationError, match="extra_forbidden"):
        type(camera_options)(fov=60.0)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_an_instance_takes_its_parameters_from_the_options_and_keeps_them():
    options = SimpleOptions(test_param="from_options", number_param=123)
    instance = SimpleConfigurable(options=options)
    options.test_param = "modified"

    assert (instance.test_param, instance.number_param) == ("from_options", 123)
    assert instance.options is options
    assert instance.name == "simple_configurable"
    with pytest.raises(AttributeError):
        instance.name = "new_name"
    assert instance.uid != SimpleConfigurable(options=SimpleOptions()).uid


def test_the_build_time_backstop_lists_the_allowed_parameters_with_types_and_defaults():
    with pytest.raises(ValueError) as exc_info:
        SimpleConfigurable(options=SimpleOptions(invalid_param="value"))

    message = str(exc_info.value)
    assert "invalid_param" in message and "allowed parameters" in message
    assert "test_param: str = 'default'" in message and "number_param: int = 42" in message


def test_the_backstop_names_the_term_and_catches_what_construction_could_not():
    """A nameless typo, a typo slipped past every authoring check, an in-place ``params`` edit."""
    from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY
    from eden.managers.reward_manager import REWARD_TERM_REGISTRY

    upright = REWARD_TERM_REGISTRY.get("upright")
    with pytest.raises(ValueError, match=r"wieght"):
        upright(env=None, options=RewardTermOptions(wieght=2.0))  # no name: construction could not resolve a term

    options = RewardTermOptions(name="upright")
    options.__pydantic_extra__["wieght"] = 2.0
    with pytest.raises(ValueError, match=r"wieght.*for term 'upright' \(Upright\)"):
        upright(env=None, options=options)

    options = ObservationTermOptions(name="base_lin_vel", params={"entity_name": "robot"})
    options.params["entity_nmae"] = "robot"  # kept by identity, so invisible to assignment validation
    with pytest.raises(ValueError, match=r"entity_nmae.*'base_lin_vel'"):
        OBSERVATION_TERM_REGISTRY.get("base_lin_vel")(env=None, options=options)


# ---------------------------------------------------------------------------
# Mutable class-level defaults
# ---------------------------------------------------------------------------


class PartialOptions(DynamicConfigurableOptions):
    """Declares ``str_param`` only, so the mutable parameters fall back to the class defaults."""

    str_param: str = "default"


class MutableDefaults(ConfigurableMixin[PartialOptions]):
    str_param: str = "default"
    dict_param: dict = {}
    list_param: list = []
    nested_dict_param: dict = {"key": [1, 2, 3]}


def test_instances_get_independent_deep_copies_of_mutable_class_defaults():
    first = MutableDefaults(options=PartialOptions(str_param="a"))
    first.dict_param["injected"] = "surprise"
    first.list_param.append("surprise")
    first.nested_dict_param["key"].append(999)

    second = MutableDefaults(options=PartialOptions(str_param="b"))

    assert (second.dict_param, second.list_param, second.nested_dict_param) == ({}, [], {"key": [1, 2, 3]})
    assert (MutableDefaults.dict_param, MutableDefaults.list_param) == ({}, [])


def test_configure_returns_independent_copies_of_mutable_class_defaults():
    first, second = MutableDefaults.configure(), MutableDefaults.configure()
    first.dict_param["injected"] = True
    first.list_param.append("injected")

    assert (second.dict_param, second.list_param) == ({}, [])
    assert (MutableDefaults.dict_param, MutableDefaults.list_param) == ({}, [])


# ---------------------------------------------------------------------------
# configure(func=...): function terms
# ---------------------------------------------------------------------------


def test_a_registered_function_with_its_required_params_configures():
    """``env`` (injected) and ``envs_idx`` (defaulted) are not flagged; options-only fields still pass."""
    options = ObsTerm.configure(
        func=observations.base_lin_vel, params={"entity_name": "robot"}, history_length=3, scale=0.5
    )

    assert (options.name, options.params, options.history_length, options.scale) == (
        "base_lin_vel",
        {"entity_name": "robot"},
        3,
        0.5,
    )


def test_configure_defaults_a_function_term_from_its_registration_not_the_shared_wrapper():
    """``MetricTerm`` is every metric function's wrapper; ``register(**defaults)`` is recorded per function."""
    options = MetricTerm.configure(func=metrics.error_vel_xy)

    assert (options.is_cumulative, options.direction, options.metric_mode) == (
        True,
        MetricDirection.LIB,
        MetricMode.RESET,
    )
    assert MetricTerm.configure(func=metrics.error_vel_xy, direction="hib").direction == MetricDirection.HIB
    assert (MetricTerm.is_cumulative, MetricTerm.direction) == (False, MetricDirection.HIB)


def test_an_unregistered_function_is_refused_at_configure():
    def my_unregistered_obs(env, *, foo):
        return foo

    with pytest.raises(ValueError) as exc_info:
        ObsTerm.configure(func=my_unregistered_obs, params={"foo": 1})

    message = str(exc_info.value)
    assert "my_unregistered_obs" in message and "not registered" in message
    assert "OBSERVATION_TERM_REGISTRY.register()" in message


def test_a_bad_params_key_lists_the_accepted_params_with_types():
    """``configure(params=...)``: the site every observed failure came from."""
    with pytest.raises(ValueError) as exc_info:
        ObsTerm.configure(func=observations.base_lin_vel, params={"entity": "robot"})

    message = str(exc_info.value)
    assert "['entity']" in message and "base_lin_vel" in message  # the term, not a FuncWrapper class repr
    assert "entity_name: str" in message and "frame: Literal['world', 'body'] = 'world'" in message


def test_a_missing_required_param_is_named_with_the_accepted_set():
    with pytest.raises(ValueError) as exc_info:
        ObsTerm.configure(func=observations.base_lin_vel)

    message = str(exc_info.value)
    assert "missing required param(s) ['entity_name']" in message
    assert "Accepted params:" in message and "frame: Literal['world', 'body'] = 'world'" in message


def test_configure_and_construction_word_a_bad_params_key_identically():
    """They fire at different times against the same term; a caller hitting both must see one complaint."""
    with pytest.raises(ValueError) as at_configure:
        ObsTerm.configure(func=observations.base_lin_vel, params={"entity_name": "robot", "typo": 1})
    with pytest.raises(Exception) as at_construction:  # pydantic wraps it in a ValidationError
        ObservationTermOptions(name="base_lin_vel", params={"entity_name": "robot", "typo": 1})

    shared = "The following params keys are invalid: ['typo'] for function term 'base_lin_vel'"
    assert shared in str(at_configure.value) and shared in str(at_construction.value)


def test_a_misspelled_term_level_field_raises_naming_the_term_and_the_allowed_fields():
    """It was silently dropped: ``wieght=3.0`` trained with the default weight."""
    with pytest.raises(ValueError) as exc_info:
        RewardTerm.configure(func=rewards.dofs_pos_limits, params={"entity_name": "robot"}, wieght=3.0)

    message = str(exc_info.value)
    assert "wieght" in message and "weight: float" in message
    assert "dofs_pos_limits" in message and "FuncWrapper" not in message


def test_the_options_only_observation_fields_are_listed_as_allowed():
    with pytest.raises(ValueError) as exc_info:
        ObsTerm.configure(func=observations.base_lin_vel, params={"entity_name": "robot"}, noize=1)

    message = str(exc_info.value)
    assert "noise:" in message and "clip:" in message and "history_length: int = 0" in message
    assert "FieldInfo" not in message  # the pydantic Annotated soup is not actionable


def test_a_term_name_where_the_function_belongs_raises_a_typed_error():
    """The natural mistake from a saved config, which spells the term ``name: base_lin_vel``."""
    with pytest.raises(TypeError) as exc_info:
        ObsTerm.configure(func="base_lin_vel", params={"entity_name": "robot"})

    message = str(exc_info.value)
    assert "func must be the term function itself" in message and "got str" in message and "base_lin_vel" in message
    assert "__name__" not in message


class _KeysButNotADict:
    """What a stripped ``assert`` accepted: ``.keys()`` and an empty, silently wrong parameter set."""

    def keys(self):
        return []


@pytest.mark.parametrize(
    "bad_params",
    [["entity_name"], "entity_name", 42, None, _KeysButNotADict()],
    ids=["list", "str", "int", "none", "keys-but-not-a-dict"],
)
def test_params_must_be_a_real_dict_even_under_python_o(bad_params):
    """This was an ``assert``, stripped under ``-O``."""
    with pytest.raises(TypeError) as exc_info:
        ObsTerm.configure(func=observations.base_lin_vel, params=bad_params)

    message = str(exc_info.value)
    assert "params must be a dict" in message and "base_lin_vel" in message and type(bad_params).__name__ in message


def _injected_names_keyword_only(*, env, envs_idx, entity_name):
    return env


def _positional_or_keyword(env, entity_name):
    return env


@pytest.mark.parametrize("func", [_injected_names_keyword_only, _positional_or_keyword], ids=lambda f: f.__name__)
def test_the_required_function_parameters_exclude_the_injected_ones(func):
    """A positional-or-keyword author param is required too; only KEYWORD_ONLY was considered."""
    assert ObsTerm.get_required_function_parameter_names(func) == {"entity_name"}


# ---------------------------------------------------------------------------
# Values mirrored from the options are copies, except built Genesis objects
# ---------------------------------------------------------------------------


class _WidgetOptions(ConfigurableOptions):
    """Declares ``tags`` but not ``table``, so a widget takes both mirroring branches."""

    tags: list[str] = []


class _Widget(ConfigurableMixin[_WidgetOptions]):
    tags: list[str] = []
    table: dict[str, float] = {}


def test_a_parameter_the_options_declare_is_a_copy_of_the_options_value():
    options = _WidgetOptions(tags=["a"])
    a, b = _Widget(options), _Widget(options)
    a.tags.append("leaked")
    assert b.tags == ["a"], "two objects built from one options instance share state"
    assert options.tags == ["a"], "mutating an instance rewrote the config it was built from"


class _SurfaceOptions(ConfigurableOptions):
    surface: object = None


class _SurfaceWidget(ConfigurableMixin[_SurfaceOptions]):
    surface: object = None


def test_a_built_genesis_surface_is_aliased_not_deep_copied():
    """It has no reconstruction path, so a deep copy would clone decoded textures or raise on a handle."""
    surface = gs.surfaces.Default()
    assert _SurfaceWidget(_SurfaceOptions(surface=surface)).surface is surface

    class _UncopyableSurface(Surface):
        def __deepcopy__(self, memo):
            raise RuntimeError("surface holds an uncopyable handle")

    uncopyable = _UncopyableSurface()
    assert _SurfaceWidget(_SurfaceOptions(surface=uncopyable)).surface is uncopyable


# ---------------------------------------------------------------------------
# The configuration fingerprint (frozen_value, configured_fingerprint)
# ---------------------------------------------------------------------------


def test_a_fingerprint_sees_a_changed_value_or_a_rebound_tensor_but_not_an_in_place_write():
    tensor = torch.zeros(2)
    fingerprint = frozen_value({"std": 0.3, "links": ["a", "b"], "buffer": tensor})
    assert frozen_value({"std": 0.3, "links": ["a", "b"], "buffer": tensor}) == fingerprint
    tensor.add_(1.0)  # read where it lives: the same configuration
    assert frozen_value({"std": 0.3, "links": ["a", "b"], "buffer": tensor}) == fingerprint
    assert frozen_value({"std": 0.4, "links": ["a", "b"], "buffer": tensor}) != fingerprint
    assert frozen_value({"std": 0.3, "links": ["a", "b"], "buffer": tensor.clone()}) != fingerprint


def test_a_fingerprint_sees_inside_an_options_model_a_dataclass_and_a_set():
    """A curriculum editing a nested object in place keeps its identity: its fields are the configuration."""

    @dataclasses.dataclass
    class Band:
        low: float
        high: float

    ranges = SimpleOptions()
    band = Band(0.0, 1.0)
    names = {"a", "b"}
    fingerprint = frozen_value({"ranges": ranges, "band": band, "names": names})
    ranges.number_param = 3
    band.high = 2.0
    names.add("c")
    changed = frozen_value({"ranges": ranges, "band": band, "names": names})
    assert changed != fingerprint
    assert [old != new for (_, old), (_, new) in zip(fingerprint, changed, strict=True)] == [True, True, True]


def test_a_fingerprint_reads_a_numpy_array_by_its_values():
    """A host array is read as a constant, so editing one in place is another configuration (unlike a tensor)."""
    gains = np.array([1.0, 2.0])
    fingerprint = frozen_value({"gains": gains, "scale": np.float32(0.5)})
    assert frozen_value({"gains": gains.copy(), "scale": np.float32(0.5)}) == fingerprint
    gains[1] = 3.0
    assert frozen_value({"gains": gains, "scale": np.float32(0.5)}) != fingerprint
    assert frozen_value({"gains": gains.astype(np.float32), "scale": 0.5}) != frozen_value(
        {"gains": gains, "scale": 0.5}
    )


def test_a_configured_fingerprint_pairs_every_parameter_by_name_and_leaves_out_the_excluded_ones():
    configurable = SimpleConfigurable(SimpleConfigurable.configure(number_param=7, list_param=[1, 2]))
    assert configurable.configured_fingerprint() == (
        ("list_param", (1, 2)),
        ("number_param", 7),
        ("test_param", "default"),
    )
    assert configurable.configured_fingerprint(exclude={"list_param"}) == (
        ("number_param", 7),
        ("test_param", "default"),
    )
    other = SimpleConfigurable(SimpleConfigurable.configure(number_param=7, list_param=[1, 2]))
    assert other.configured_fingerprint() == configurable.configured_fingerprint()  # configured alike


def test_a_configured_fingerprint_reads_names_resolved_once_as_it_would_resolve_them():
    """A caller fingerprinting on every step resolves the names once (the MRO walk) and passes them back."""
    configurable = SimpleConfigurable(SimpleConfigurable.configure(number_param=7, list_param=[1, 2]))
    names = SimpleConfigurable.fingerprint_names(exclude={"list_param"})

    assert names == ("number_param", "test_param")
    assert configurable.configured_fingerprint(names=names) == configurable.configured_fingerprint(
        exclude={"list_param"}
    )
    with pytest.raises(ValueError, match=r"takes exclude or names, not both.*fingerprint_names\(exclude="):
        configurable.configured_fingerprint(exclude={"list_param"}, names=names)
    # an exclude that yields nothing excludes nothing, generator or not
    assert configurable.configured_fingerprint(exclude=(n for n in ()), names=names) == (
        configurable.configured_fingerprint(names=names)
    )


def test_retired_parameter_notes_accumulate_over_the_mro():
    """A subclass retiring a name of its own keeps its base's notes, as ``_non_parameter_names`` accumulates."""

    class Base(SimpleConfigurable):
        _retired_params = MappingProxyType({"old": "base note", "shared": "base wording"})

    class Child(Base):
        _retired_params = MappingProxyType({"older": "child note", "shared": "child wording"})

    class Grandchild(Child):
        pass

    assert dict(Grandchild._retired_params) == {"old": "base note", "older": "child note", "shared": "child wording"}
    with pytest.raises(ValueError, match=r"Retired \['old'\]: base note"):
        Grandchild(SimpleOptions(old=1))
