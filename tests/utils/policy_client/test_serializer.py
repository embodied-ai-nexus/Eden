"""Unit tests for the remote-policy client serialisers (wire-format parity with the reference codecs)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from eden.utils.policy_client import SERIALIZER_REGISTRY, make_serializer


def _payload():
    return {
        "prompt": "pick",
        "observation/state": np.arange(8, dtype=np.float32),
        "observation/image": np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3),
        "nested": {"scalar": np.float32(1.5), "flag": True},
    }


def _assert_roundtrip(decoded, original):
    assert decoded["prompt"] == original["prompt"]
    np.testing.assert_array_equal(decoded["observation/state"], original["observation/state"])
    np.testing.assert_array_equal(decoded["observation/image"], original["observation/image"])
    assert decoded["nested"]["flag"] is True
    assert float(decoded["nested"]["scalar"]) == 1.5


@pytest.mark.parametrize(
    ("key", "module"),
    [
        ("json_numpy", "json_numpy"),
        ("msgpack_openpi", "msgpack"),
        ("msgpack_galaxea", "msgpack"),
        ("msgpack_numpy", "msgpack_numpy"),
        ("msgpack_npy_blob", "msgpack"),
    ],
)
def test_serializers_round_trip_numpy_payloads(key, module):
    pytest.importorskip(module)  # each codec lives in the policy-client extra
    s = make_serializer(key)
    _assert_roundtrip(s.decode(s.encode(_payload())), _payload())


def test_msgpack_openpi_is_byte_identical_to_openpi_client():
    """The OpenPI server unpacks with ``b"__ndarray__"`` keys; our bytes must match its client's."""
    ref = pytest.importorskip("openpi_client.msgpack_numpy")
    s = make_serializer("msgpack_openpi")
    assert s.encode(_payload()) == ref.packb(_payload())
    decoded = s.decode(ref.packb(_payload()))
    _assert_roundtrip(decoded, _payload())


def test_msgpack_numpy_matches_reference_library():
    mnp = pytest.importorskip("msgpack_numpy")
    s = make_serializer("msgpack_numpy")
    assert s.encode(_payload()) == mnp.packb(_payload())


def test_msgpack_npy_blob_encodes_torch_tensors_and_widens_half_precision():
    pytest.importorskip("msgpack")
    s = make_serializer("msgpack_npy_blob")
    out = s.decode(s.encode({"t": torch.arange(3, dtype=torch.float16), "shape": [1, 2]}))
    assert out["t"].dtype == np.float32
    np.testing.assert_array_equal(out["t"], [0, 1, 2])
    assert out["shape"] == [1, 2]


def test_unknown_serializer_lists_available_keys():
    with pytest.raises(ValueError, match="Available"):
        make_serializer("protobuf")
    assert {"json", "json_numpy", "msgpack_openpi", "msgpack_galaxea", "msgpack_numpy", "msgpack_npy_blob"} <= set(
        SERIALIZER_REGISTRY.keys()
    )


# A server tunnels a ModalityConfig's ``as_json`` either as a dict or as its JSON text (pydantic's
# ``model_dump_json()``); both must decode to the dict a client reads ``modality_keys`` from.
_AS_JSON_FORMS = pytest.mark.parametrize("as_json", [lambda d: d, json.dumps], ids=["dict", "json-text"])


@_AS_JSON_FORMS
def test_msgpack_numpy_decodes_gr00t_modality_config_records_as_dicts(as_json):
    mnp = pytest.importorskip("msgpack_numpy")
    record = {
        "video": {"__ModalityConfig__": True, "as_json": as_json({"modality_keys": ["image_0"], "delta_indices": [0]})},
        "arr": np.arange(3, dtype=np.float32),
    }
    decoded = make_serializer("msgpack_numpy").decode(mnp.packb(record))
    assert decoded["video"] == {"modality_keys": ["image_0"], "delta_indices": [0]}
    np.testing.assert_array_equal(decoded["arr"], [0, 1, 2])


@_AS_JSON_FORMS
def test_msgpack_npy_blob_decodes_rldx_modality_config_records_as_dicts(as_json):
    msgpack = pytest.importorskip("msgpack")
    # RLDX-1's server_client.encode_custom_classes tags ModalityConfig with __ModalityConfig_class__.
    record = {
        "video": {
            "__ModalityConfig_class__": True,
            "as_json": as_json({"modality_keys": ["exterior"], "delta_indices": [0]}),
        },
        "state": {
            "__ModalityConfig_class__": True,
            "as_json": as_json({"modality_keys": ["joints"], "delta_indices": [0]}),
        },
    }
    decoded = make_serializer("msgpack_npy_blob").decode(msgpack.packb(record))
    assert decoded["video"] == {"modality_keys": ["exterior"], "delta_indices": [0]}
    assert decoded["state"]["modality_keys"] == ["joints"]


def test_codec_failures_are_serialization_errors():
    from eden.utils.policy_client import SerializationError

    with pytest.raises(SerializationError, match="decode the reply"):
        make_serializer("json").decode("{not json")
    with pytest.raises(SerializationError, match="encode the payload"):
        make_serializer("json").encode({"x": object()})
    pytest.importorskip("msgpack")
    with pytest.raises(SerializationError):
        make_serializer("msgpack_npy_blob").decode(b"\xc1")  # 0xc1 is never valid msgpack
