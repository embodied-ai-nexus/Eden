"""Option-metadata helpers for :class:`~eden.options.options.ConfigurableOptions`.

1. :func:`option_metadata` provides the ``_option_module_`` / ``_option_class_`` pair so
   the serialization layer (:mod:`eden.utils.configs`) can recover the concrete type on
   load. Genesis-native options don't carry it live — ``serialize_obj_with_metadata``
   stamps it at dump time from ``__module__``/``__class__``.
2. :func:`getattribute_with_extra` surfaces dynamic ``__pydantic_extra__`` entries
   (named children / terms on ``extra="allow"`` options) as plain attributes.
"""

#: Metadata keys stamped onto every extra-allowing options instance for type recovery.
#: Excluded from ``keys()`` / ``dict()`` output so they stay an implementation detail.
OPTION_METADATA_KEYS = ("_option_module_", "_option_class_")


def option_metadata(obj) -> dict[str, str]:
    """The ``_option_module_`` / ``_option_class_`` metadata for ``obj``, for type recovery.

    Returns an empty dict for models that forbid extras (the config root classes), which
    carry their type metadata through the serialization envelope instead. Safe to call
    before ``BaseModel.__init__`` — it only touches the class-level ``model_config`` and
    ``type(obj)``.
    """
    if obj.model_config.get("extra") == "forbid":
        return {}
    return {"_option_module_": type(obj).__module__, "_option_class_": type(obj).__name__}


def getattribute_with_extra(obj, item: str, fallback):
    """``__getattribute__`` body that surfaces ``__pydantic_extra__`` entries as attributes.

    ``fallback`` is the normal lookup to fall through to (``super().__getattribute__``);
    it is called with ``item`` only.

    Hot path: this backs ``__getattribute__`` on every options object, so it runs on *every*
    attribute access (not just dynamic children). Keep it cheap — the common case is a miss
    that falls straight through to ``fallback``.
    """
    try:
        extra = object.__getattribute__(obj, "__pydantic_extra__")
        if extra is not None and item in extra:
            return extra[item]
    except AttributeError:
        pass  # __pydantic_extra__ not yet initialized
    return fallback(item)
