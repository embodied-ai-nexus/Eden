"""Tests for ``eden.utils.params``: the annotation and default rendering behind the term catalog and error messages."""

from typing import Literal

import pytest

from pydantic import BaseModel, Field

from eden.utils.common import ConfigurableMixin
from eden.utils.params import (
    ParamInfo,
    render_annotation,
    render_default,
    unknown_parameter_names,
    validated_parameter_values,
)


class _Fields(BaseModel):
    mode: Literal["compound", "snapshot"] = "snapshot"
    opt_mode: Literal["a", "b"] | None = None
    shape: tuple[float, ...] = ()
    labels: list[str] = Field(default_factory=list)


def test_render_annotation_gives_source_an_author_could_paste_back():
    """Quote-stripped Literals read as undefined names; ``...`` must not come out as ``Ellipsis``."""
    assert render_annotation(Literal["world", "body"]) == "Literal['world', 'body']"
    assert render_annotation(tuple[float, ...]) == "tuple[float, ...]"


def test_render_annotation_through_live_pydantic_field_annotations():
    """The path ``get_configure_specs`` takes for an options-only field, ``Optional[Literal]`` included."""
    fields = _Fields.model_fields
    assert render_annotation(fields["mode"].annotation) == "Literal['compound', 'snapshot']"
    assert render_annotation(fields["opt_mode"].annotation) == "Literal['a', 'b'] | None"
    assert render_annotation(fields["shape"].annotation) == "tuple[float, ...]"


def test_render_default_collapses_line_breaks_only():
    assert render_default("a  b") == "'a  b'"
    assert render_default([1, 2, 3]) == "[1, 2, 3]"
    assert render_default(_Fields.model_fields["labels"].get_default(call_default_factory=True)) == "[]"


def test_paraminfo_renders_no_dangling_equals_without_a_default():
    assert str(ParamInfo("labels")) == "labels"
    assert str(ParamInfo("labels", "list[str]")) == "labels: list[str]"
    assert str(ParamInfo("x", "str", "''")) == "x: str = ''"  # a real empty-string default still renders
    assert str(ParamInfo("x", "int", required=True)) == "x: int"


def test_unknown_parameter_names_passes_parameters_options_only_fields_metadata_and_tolerated_names():
    """The one check behind term options and camera-overlay options: what a configurable takes, and nothing else."""
    from eden.managers.terms.observations import MetricObservation

    extras = {"metric_name": "drawing", "noise": None, "_option_module_": "m", "class_name": "c", "metric_nme": "x"}
    assert unknown_parameter_names(extras, MetricObservation, tolerated=("class_name",)) == {"metric_nme"}
    assert unknown_parameter_names(extras, MetricObservation) == {"metric_nme", "class_name"}


def test_validated_parameter_values_converts_to_declared_types_and_names_a_value_it_cannot():
    from eden.entities.camera_overlays import PlanePaintOverlay

    values = {"color": [0, 1, 0], "max_gap": "0.003", "name": "plane_paint_overlay"}
    validated = validated_parameter_values(PlanePaintOverlay, values, subject="PlanePaintOverlay")
    assert validated == {"color": (0.0, 1.0, 0.0), "max_gap": 0.003, "name": "plane_paint_overlay"}
    with pytest.raises(
        ValueError,
        match=r"^PlanePaintOverlay needs supersample as int; got None \(NoneType\): .*"
        r"E\.g\. supersample=4 \(the default\)\.$",
    ):
        validated_parameter_values(PlanePaintOverlay, {"supersample": None}, subject="PlanePaintOverlay")


class _TypeCheckingOnlyAttribute(ConfigurableMixin):
    width: float = 1.0
    _camera: "Camera"  # noqa: F821 - imported only under TYPE_CHECKING in a real module


class _TypeCheckingOnlyParameter(ConfigurableMixin):
    camera: "Camera" = None  # noqa: F821


class _TypeCheckingOnlyNestedParameter(ConfigurableMixin):
    cameras: "list['Camera']" = []  # noqa: F821


class _Opaque:
    pass


class _OpaqueParameter(ConfigurableMixin):
    thing: _Opaque = None


class _Flags(ConfigurableMixin):
    strength: float = 0.0
    enabled: bool = False


def test_validated_parameter_values_resolves_only_the_parameters_annotations():
    """A non-parameter annotating a type that exists only for type checking must not break the parameters' validation;
    a parameter that does is named, with the fix.
    """
    validated = validated_parameter_values(_TypeCheckingOnlyAttribute, {"width": "2"}, subject="probe")
    assert validated == {"width": 2.0}
    with pytest.raises(
        TypeError,
        match=r"^_TypeCheckingOnlyParameter's parameter `camera: Camera` \(declared on .*\) names a type that does "
        r"not exist at runtime \(NameError: .*\).*not under `if TYPE_CHECKING:`\.$",
    ):
        validated_parameter_values(_TypeCheckingOnlyParameter, {"camera": 1}, subject="probe")


def test_a_quoted_type_nested_in_a_parameters_annotation_is_named_too():
    with pytest.raises(
        TypeError,
        match=r"^_TypeCheckingOnlyNestedParameter's parameter `cameras: list\['Camera'\]`.*does not exist at runtime",
    ):
        validated_parameter_values(_TypeCheckingOnlyNestedParameter, {"cameras": []}, subject="probe")


def test_a_parameter_type_pydantic_cannot_validate_is_named_with_the_types_it_can():
    with pytest.raises(
        TypeError,
        match=r"^_OpaqueParameter's parameter `thing: .*_Opaque` .* cannot be validated \(PydanticSchemaGenerationError: Unable to generate pydantic-core schema for .*_Opaque'>\)\. A parameter is written to",
    ):
        validated_parameter_values(_OpaqueParameter, {"thing": _Opaque()}, subject="probe")


def test_a_class_outside_any_imported_module_resolves_its_parameters_against_its_own_names():
    """A class built by ``type()`` or ``exec`` names a module that is not in ``sys.modules``: builtins still resolve."""
    elsewhere = type(
        "_Elsewhere",
        (ConfigurableMixin,),
        {"__module__": "not_an_imported_module", "__annotations__": {"width": "float"}, "width": 1.0},
    )
    assert validated_parameter_values(elsewhere, {"width": "2"}, subject="probe") == {"width": 2.0}


def test_a_bool_is_a_number_only_where_the_type_admits_a_bool():
    """Lax conversion would take True as 1.0; a bool for a number is a mistake (``strength: yes`` in YAML)."""
    assert validated_parameter_values(_Flags, {"strength": 1, "enabled": True}, subject="probe") == {
        "strength": 1.0,
        "enabled": True,
    }
    with pytest.raises(ValueError, match=r"^probe needs strength as float; got True \(bool\)"):
        validated_parameter_values(_Flags, {"strength": True}, subject="probe")
