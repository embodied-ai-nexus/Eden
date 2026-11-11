"""A fixed-shape float ``np.ndarray`` pydantic field type, e.g. ``FloatArray[3, 3]`` for a camera intrinsic.

Eden's own replacement for ``numpydantic.NDArray[Shape["3, 3"], float]``, which imports ``cv2``
(through numpydantic's video interface) and so would pull it into every task config for two camera-matrix
annotations. Only numpy and pydantic here, so :mod:`eden.options.camera` can import it without an
``eden.utils`` <-> ``eden.options`` cycle.

A saved matrix loads back as an ``np.ndarray`` because ``EdenConfig.load_from_file`` builds every
``ConfigurableOptions`` with ``model_validate``, which runs :class:`FloatArrayShape` on the nested
list. Declare a ``FloatArray`` field on a ``ConfigurableOptions`` subclass only: a genesis-native
``Options`` loads through ``model_construct``, so nothing would validate the field.
"""

from dataclasses import dataclass
from typing import Annotated, Any

import numpy as np
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema


@dataclass(frozen=True)
class FloatArrayShape:
    """``Annotated`` metadata that validates a value into a float ``np.ndarray`` of ``shape``.

    Accepts an ``np.ndarray`` or a nested sequence of real numbers. A float array is kept as-is
    (same object, same float dtype); anything else is converted with ``np.asarray`` and, if it
    holds integers, cast to ``float64``. Serializes to nested lists in JSON mode and stays an
    ``np.ndarray`` in Python mode.

    Parameters
    ----------
    shape : tuple[int, ...]
        The exact shape a value must have.
    """

    shape: tuple[int, ...]

    def _validate(self, value: Any) -> np.ndarray:
        # ValueError, not AssertionError: pydantic wraps it into a ValidationError naming the field.
        if isinstance(value, np.ndarray):
            array = value
        else:
            try:
                array = np.asarray(value)
            except ValueError as e:  # ragged nesting, e.g. [[1.0, 0.0], [1.0]]
                raise ValueError(f"expected a {self._describe()} array, got a ragged sequence ({e})") from None
        if array.dtype.kind == "f":
            pass
        elif array.dtype.kind in "iu":
            array = array.astype(np.float64)
        else:
            raise ValueError(
                f"expected a {self._describe()} array of real numbers, got dtype {array.dtype} "
                f"(from a {type(value).__name__})"
            )
        if array.shape != self.shape:
            raise ValueError(f"expected shape {self.shape}, got shape {array.shape}")
        return array

    def _describe(self) -> str:
        return "x".join(str(n) for n in self.shape)

    def __get_pydantic_core_schema__(self, source: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        """Validate with :meth:`_validate`; dump nested lists in JSON mode, the array in Python mode."""
        return core_schema.no_info_plain_validator_function(
            self._validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda array: array.tolist(), when_used="json"
            ),
        )

    def __get_pydantic_json_schema__(
        self, schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """Nested fixed-length ``array`` schemas of ``number``, outermost dimension first."""
        json_schema: JsonSchemaValue = {"type": "number"}
        for n in reversed(self.shape):
            json_schema = {"type": "array", "items": json_schema, "minItems": n, "maxItems": n}
        return json_schema


class FloatArray:
    """``FloatArray[3, 3]`` is ``Annotated[np.ndarray, FloatArrayShape((3, 3))]``.

    Examples
    --------
    >>> class Opts(pydantic.BaseModel):
    ...     intrinsic: FloatArray[3, 3] | None = None
    """

    def __class_getitem__(cls, shape: int | tuple[int, ...]) -> Any:
        """Build the annotation for ``shape``."""
        shape = shape if isinstance(shape, tuple) else (shape,)
        return Annotated[np.ndarray, FloatArrayShape(tuple(int(n) for n in shape))]
