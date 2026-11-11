"""Tests for ``eden.utils.ndarray_field.FloatArray``, the fixed-shape float array config field."""

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from eden.options.camera import CameraOptions
from eden.options.options import ConfigurableOptions
from eden.utils.configs import serialize_obj_with_metadata, to_options
from eden.utils.ndarray_field import FloatArray


class _Matrix(BaseModel):
    m: FloatArray[3, 3] | None = None
    v: FloatArray[2] | None = None


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_a_float_array_is_kept_as_is(dtype):
    """Same object and dtype back: no copy, no silent widening of a float32 matrix."""
    value = np.eye(3, dtype=dtype)
    assert _Matrix(m=value).m is value


@pytest.mark.parametrize(
    "value", [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], ((1, 0, 0), (0, 1, 0), (0, 0, 1)), np.eye(3, dtype=int)]
)
def test_integer_input_becomes_a_float64_array(value):
    """``K = [[600, 0, 320], ...]`` is a valid intrinsic, not a type error (numpydantic rejected it)."""
    m = _Matrix(m=value).m
    assert isinstance(m, np.ndarray) and m.dtype == np.float64 and m.shape == (3, 3)


def test_a_1d_shape_takes_a_bare_int():
    assert _Matrix(v=[1.0, 2.0]).v.shape == (2,)
    assert FloatArray[2] == FloatArray[(2,)]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.eye(4), "expected shape (3, 3), got shape (4, 4)"),
        ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], "expected shape (3, 3), got shape (2, 3)"),
        ([[1.0, 0.0], [1.0]], "expected a 3x3 array, got a ragged sequence"),
        ("abc", "expected a 3x3 array of real numbers, got dtype <U3 (from a str)"),
        (np.eye(3, dtype=bool), "expected a 3x3 array of real numbers, got dtype bool (from a ndarray)"),
        (np.eye(3, dtype=complex), "expected a 3x3 array of real numbers, got dtype complex128"),
    ],
)
def test_a_bad_value_is_rejected_naming_the_field_and_the_shape(value, message):
    with pytest.raises(ValidationError) as exc_info:
        _Matrix(m=value)
    text = str(exc_info.value)
    assert "\nm\n" in text and message in text, text


def test_camera_options_reject_a_wrong_shape_matrix_when_the_config_is_built():
    """Config-time, not camera-build-time: a 4x4 given as the intrinsic fails in ``CameraOptions(...)``."""
    with pytest.raises(ValidationError, match=r"cam_intrinsic\n.*expected shape \(3, 3\), got shape \(4, 4\)"):
        CameraOptions(cam_intrinsic=np.eye(4))
    with pytest.raises(ValidationError, match=r"cam_extrinsic\n.*expected shape \(4, 4\), got shape \(3, 3\)"):
        CameraOptions(cam_extrinsic=np.eye(3))


def test_dumps_an_array_in_python_mode_and_nested_lists_in_json_mode():
    model = _Matrix(m=np.eye(3))
    assert isinstance(model.model_dump()["m"], np.ndarray)
    assert model.model_dump(mode="json")["m"] == np.eye(3).tolist()
    assert _Matrix.model_validate_json(model.model_dump_json()).m.shape == (3, 3)


def test_json_schema_is_nested_fixed_length_arrays_of_numbers():
    schema = _Matrix.model_json_schema()["properties"]["m"]["anyOf"][0]
    row = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
    assert schema == {"type": "array", "items": row, "minItems": 3, "maxItems": 3}


class _RequiredMatrixOptions(ConfigurableOptions):
    required: FloatArray[3, 3] = np.eye(3)
    optional: FloatArray[4, 4] | None = None


def test_a_saved_matrix_loads_back_as_an_array_whether_or_not_the_field_is_optional():
    """Pydantic strips a non-Optional field's ``Annotated`` metadata, so loading must not route on the annotation."""
    opts = _RequiredMatrixOptions(required=np.diag([1.0, 2.0, 3.0]), optional=np.eye(4))
    loaded = to_options(serialize_obj_with_metadata(opts), _RequiredMatrixOptions)
    for name, shape in (("required", (3, 3)), ("optional", (4, 4))):
        value = getattr(loaded, name)
        assert isinstance(value, np.ndarray) and value.shape == shape and value.dtype.kind == "f", name
    assert np.array_equal(loaded.required, opts.required)
    with pytest.raises(ValidationError, match=r"required\n.*expected shape \(3, 3\), got shape \(2, 2\)"):
        to_options({"required": [[1.0, 0.0], [0.0, 1.0]]}, _RequiredMatrixOptions)
