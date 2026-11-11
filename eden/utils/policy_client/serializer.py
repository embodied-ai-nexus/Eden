"""Wire serialisers for request/reply clients.

A serialiser turns a payload dict into the bytes (or text) a server expects and decodes
the reply. They are selected by ``SERIALIZER_REGISTRY`` key (``TransportConfig.serializer``)
so the choice is a plain string that survives config serialisation. Each one imports its
optional dependency lazily and reports the extra to install when it is missing.

=====================  ==========================================================
Key                    Wire format
=====================  ==========================================================
``json``               ``json.dumps`` / ``json.loads``
``json_numpy``         ``json_numpy`` (numpy arrays as base64 inside JSON text)
``msgpack_openpi``     msgpack with OpenPI's numpy hook (``b"__ndarray__"`` records)
``msgpack_galaxea``    same record layout with *str* keys, as the GalaxeaVLA server speaks
``msgpack_numpy``      ``msgpack_numpy`` (Isaac-GR00T ``PolicyServer``)
``msgpack_npy_blob``   msgpack with arrays tunnelled as ``np.save`` blobs (RLDX-1)
=====================  ==========================================================
"""

from __future__ import annotations

import functools
import io
import json
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

from eden.utils.distribution import install_hint
from eden.utils.policy_client.errors import SerializationError
from eden.utils.registry import Registry

SERIALIZER_REGISTRY = Registry("SERIALIZER")


def require_module(module: str, extra: str = "policy-client"):
    """Import an optional client dependency, or raise naming the extra that provides it."""
    try:
        return __import__(module)
    except ImportError as e:  # pragma: no cover - depends on the install
        raise ImportError(f"The '{module}' package is required here. Install it with {install_hint(extra)}.") from e


def _map_errors(method, action: str):
    """Report a codec's ``TypeError`` / ``ValueError`` / ``KeyError`` as :class:`SerializationError`."""

    @functools.wraps(method)
    def wrapper(self, data):
        try:
            return method(self, data)
        except (TypeError, ValueError, KeyError) as e:  # msgpack/json errors are ValueErrors; KeyError: bad record
            raise SerializationError(f"{type(self).__name__} could not {action}: {e}") from e

    return wrapper


class SerializerBase(ABC):
    """Encode a payload for the wire and decode a server reply.

    Subclasses' ``encode`` / ``decode`` are wrapped so codec failures surface as
    :class:`SerializationError`.
    """

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        for name, action in (("encode", "encode the payload"), ("decode", "decode the reply")):
            if name in cls.__dict__:
                setattr(cls, name, _map_errors(cls.__dict__[name], action))

    @abstractmethod
    def encode(self, data: Any) -> bytes | str:
        """Serialise ``data`` (dict of str / numbers / numpy arrays) for transmission."""

    @abstractmethod
    def decode(self, raw: bytes | str) -> Any:
        """Deserialise a server reply."""


@SERIALIZER_REGISTRY.register(name="json")
class JsonSerializer(SerializerBase):
    """Plain JSON text; numpy arrays must already be converted to lists by the policy."""

    def encode(self, data: Any) -> str:
        return json.dumps(data)

    def decode(self, raw: bytes | str) -> Any:
        return json.loads(raw)


@SERIALIZER_REGISTRY.register(name="json_numpy")
class JsonNumpySerializer(SerializerBase):
    """``json_numpy`` text: numpy arrays are embedded as base64 records."""

    def __init__(self) -> None:
        self._json_numpy = require_module("json_numpy")

    def encode(self, data: Any) -> str:
        return self._json_numpy.dumps(data)

    def decode(self, raw: bytes | str) -> Any:
        return self._json_numpy.loads(raw)


_UNSUPPORTED_KINDS = ("V", "O", "c")  # void, object, complex


@SERIALIZER_REGISTRY.register(name="msgpack_openpi")
class MsgpackOpenPiSerializer(SerializerBase):
    """msgpack with OpenPI's numpy hook, byte-for-byte what ``openpi_client.msgpack_numpy`` produces.

    Array records use *bytes* keys (``b"__ndarray__"``) because that is what the OpenPI
    ``WebsocketPolicyServer`` looks for when unpacking. Bytes keys survive msgpack's
    ``raw=False`` decoding unchanged (they are packed as ``bin``), so the same hook decodes
    the server's replies.
    """

    #: Whether array records are keyed with bytes (OpenPI servers) or str (GalaxeaVLA).
    binary_keys: bool = True

    def __init__(self) -> None:
        self._msgpack = require_module("msgpack")

    def _key(self, name: str) -> bytes | str:
        return name.encode() if self.binary_keys else name

    def _pack_default(self, obj: Any) -> Any:
        if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in _UNSUPPORTED_KINDS:
            raise SerializationError(f"Unsupported dtype for msgpack transport: {obj.dtype}")
        if isinstance(obj, np.ndarray):
            return {
                self._key("__ndarray__"): True,
                self._key("data"): obj.tobytes(),
                self._key("dtype"): obj.dtype.str,
                self._key("shape"): obj.shape,
            }
        if isinstance(obj, np.generic):
            return {self._key("__npgeneric__"): True, self._key("data"): obj.item(), self._key("dtype"): obj.dtype.str}
        return obj

    @staticmethod
    def _str_keys(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {(k.decode() if isinstance(k, bytes) else k): v for k, v in obj.items()}
        return obj

    def _object_hook(self, obj: dict) -> Any:
        norm = self._str_keys(obj)
        if "__ndarray__" in norm:
            return np.ndarray(buffer=norm["data"], dtype=np.dtype(norm["dtype"]), shape=tuple(norm["shape"]))
        if "__npgeneric__" in norm:
            return np.dtype(norm["dtype"]).type(norm["data"])
        return norm if not self.binary_keys else obj

    def encode(self, data: Any) -> bytes:
        return self._msgpack.packb(data, default=self._pack_default)

    def decode(self, raw: bytes | str) -> Any:
        return self._msgpack.unpackb(raw, object_hook=self._object_hook)


@SERIALIZER_REGISTRY.register(name="msgpack_galaxea")
class MsgpackGalaxeaSerializer(MsgpackOpenPiSerializer):
    """OpenPI record layout with *str* keys, matching the GalaxeaVLA server's codec.

    Replies are normalised recursively so nested dicts come back with str keys.
    """

    binary_keys = False


def _modality_config_dict(as_json: Any) -> Any:
    """The plain-dict form of a tunnelled ``ModalityConfig``'s ``as_json`` field.

    Servers send it either as a dict or as the JSON text of one (pydantic's
    ``model_dump_json()``); both decode to the same dict.
    """
    return json.loads(as_json) if isinstance(as_json, (str, bytes)) else as_json


@SERIALIZER_REGISTRY.register(name="msgpack_numpy")
class MsgpackNumpySerializer(SerializerBase):
    """``msgpack_numpy`` as used by Isaac-GR00T's ZMQ ``PolicyServer``.

    GR00T's ``MsgSerializer`` additionally tunnels ``ModalityConfig`` objects as
    ``{"__ModalityConfig__": True, "as_json": ...}`` records (the ``get_modality_config``
    reply), with ``as_json`` a dict or its JSON text. They are decoded to the plain dict so
    clients can read ``modality_keys`` / ``delta_indices`` without importing ``gr00t``.
    """

    def __init__(self) -> None:
        self._mnp = require_module("msgpack_numpy")

    def _decode_hook(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            if "__ModalityConfig__" in obj:
                return _modality_config_dict(obj["as_json"])
            if b"__ModalityConfig__" in obj:
                return _modality_config_dict(obj[b"as_json"])
        return self._mnp.decode(obj)

    def encode(self, data: Any) -> bytes:
        return self._mnp.packb(data)

    def decode(self, raw: bytes | str) -> Any:
        return self._mnp.unpackb(raw, object_hook=self._decode_hook, raw=False)


@SERIALIZER_REGISTRY.register(name="msgpack_npy_blob")
class MsgpackNpyBlobSerializer(SerializerBase):
    """Client-side port of RLDX-1's ``MsgSerializer``: arrays travel as ``np.save`` byte blobs.

    Torch tensors are moved to CPU (half precision widened to float32) and encoded like
    arrays. RLDX tunnels ``ModalityConfig`` objects (the ``get_modality_config`` reply) as
    ``{"__ModalityConfig_class__": True, "as_json": ...}`` records; they are decoded to the
    plain dict (from a dict or its JSON text), as :class:`MsgpackNumpySerializer` does for
    GR00T's tag.
    """

    def __init__(self) -> None:
        self._msgpack = require_module("msgpack")

    @staticmethod
    def _encode_default(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            t = obj.detach().cpu()
            if t.dtype in (torch.bfloat16, torch.float16):
                t = t.float()
            obj = t.numpy()
        if isinstance(obj, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
        if isinstance(obj, np.generic):
            # The reference serializer only handles arrays; numpy scalars are plain numbers on the wire.
            return obj.item()
        return obj

    @staticmethod
    def _decode_hook(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "__ndarray_class__" in obj:
                return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
            if "__ModalityConfig_class__" in obj:
                return _modality_config_dict(obj["as_json"])
        return obj

    def encode(self, data: Any) -> bytes:
        return self._msgpack.packb(data, default=self._encode_default)

    def decode(self, raw: bytes | str) -> Any:
        return self._msgpack.unpackb(raw, object_hook=self._decode_hook, raw=False)


def make_serializer(key: str) -> SerializerBase:
    """Instantiate the registered serialiser ``key``; unknown keys list the available ones."""
    if key not in SERIALIZER_REGISTRY.keys():
        raise ValueError(f"Unknown serializer {key!r}. Available: {sorted(SERIALIZER_REGISTRY.keys())}")
    return SERIALIZER_REGISTRY.get(key)()
