"""Tests for ``eden.utils.policy_client.transport``: every transport against a local server speaking its wire
protocol (tests/support/policy_servers.py), and the address joining every transport shares.
"""

from __future__ import annotations

import socket
import time

import numpy as np
import pytest

from eden.utils.policy_client import (
    TRANSPORT_REGISTRY,
    ServerError,
    TransportError,
    TransportTimeoutError,
    make_transport,
)
from eden.utils.policy_client.errors import SerializationError
from eden.utils.policy_client.transport import TransportConfig, server_url
from tests.support.policy_servers import HttpPolicyServer, WebsocketPolicyServer, ZmqPolicyServer


def _transport(key: str, server, **config):
    return make_transport(key, TransportConfig(host=server.host, port=server.port, **config))


def _skip_without(*modules: str) -> None:
    """Skip unless every module a transport and its serializer need (the policy-client extra) imports."""
    for module in modules:
        pytest.importorskip(module)


def _free_port() -> int:
    """A port nothing listens on: bound, read, released."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


# ---------------------------------------------------------------------------
# registry / server address
# ---------------------------------------------------------------------------


def test_registry_has_every_documented_transport():
    assert {
        "http_json",
        "http_json_numpy",
        "http_multipart",
        "openpi_websocket",
        "galaxea_websocket",
        "zmq_msgpack",
    } <= set(TRANSPORT_REGISTRY.keys())


def test_unknown_transport_key_lists_the_available_ones():
    with pytest.raises(ValueError, match="Available"):
        make_transport("carrier_pigeon", TransportConfig())


@pytest.mark.parametrize(
    ("host", "port", "schemes", "expected"),
    [
        ("localhost", 8000, ("ws", "wss"), "ws://localhost:8000"),
        ("wsnode01", 8000, ("ws", "wss"), "ws://wsnode01:8000"),  # a bare host starting with "ws"
        ("wss://example.com/ws", None, ("ws", "wss"), "wss://example.com/ws"),
        ("wss://example.com/ws", 8443, ("ws", "wss"), "wss://example.com:8443/ws"),  # port goes before the path
        ("ws://node:9000", None, ("ws", "wss"), "ws://node:9000"),
        ("ws://node:9000", 9000, ("ws", "wss"), "ws://node:9000"),
        ("node:9000", None, ("http", "https"), "http://node:9000"),
        ("https://example.com/api", None, ("http", "https"), "https://example.com/api"),
        ("[::1]", 5555, ("tcp",), "tcp://[::1]:5555"),
        ("tcp://10.0.0.1:5555", None, ("tcp",), "tcp://10.0.0.1:5555"),
    ],
)
def test_server_url_joins_host_and_port(host, port, schemes, expected):
    assert server_url(host, port, schemes=schemes) == expected


@pytest.mark.parametrize(
    ("host", "port", "schemes", "match"),
    [
        ("ws://node:9000", 8000, ("ws", "wss"), "embeds port 9000, but port=8000"),
        ("http://node", 1, ("ws", "wss"), "scheme 'http'"),
        ("ws://node:notaport", None, ("ws", "wss"), "invalid port"),
        ("h", None, ("tcp",), "needs a port"),
        ("tcp://h:5555/x", None, ("tcp",), "needs a port and no path"),
    ],
)
def test_server_url_rejects_conflicting_or_invalid_addresses(host, port, schemes, match):
    with pytest.raises(ValueError, match=match):
        server_url(host, port, schemes=schemes)


def test_transports_without_a_reset_or_set_task_message_reject_one():
    for key, field in (
        ("openpi_websocket", "reset_endpoint"),
        ("openpi_websocket", "set_task_endpoint"),
        ("galaxea_websocket", "set_task_endpoint"),
        ("zmq_msgpack", "set_task_endpoint"),
    ):
        with pytest.raises(ValueError, match=f"Set {field}=None"):
            make_transport(key, TransportConfig(host="h", port=1, **{field: "/x"}))


@pytest.mark.parametrize("key", ["http_json", "http_multipart"])
def test_a_transport_that_builds_its_own_body_rejects_a_serializer(key):
    with pytest.raises(ValueError, match="takes no serializer. Set serializer=None"):
        make_transport(key, TransportConfig(host="h", port=1, serializer="msgpack_numpy"))


@pytest.mark.parametrize(
    ("key", "field", "value", "knob"),
    [
        ("zmq_msgpack", "request_timeout", 5.0, "timeout_ms"),
        ("zmq_msgpack", "act_endpoint", "/predict", "timeout_ms"),
        ("openpi_websocket", "request_timeout", 120.0, "ws_ping_timeout"),
        ("galaxea_websocket", "reset_payload_key", "instruction", "ws_timeout_s"),
    ],
)
def test_a_transport_rejects_an_http_setting_it_would_ignore(key, field, value, knob):
    # Checked before any connection is attempted, so the address need not be served.
    with pytest.raises(ValueError, match=rf"does not read {field} \(got {value!r}\).*knobs \[.*'{knob}'"):
        make_transport(key, TransportConfig(host="127.0.0.1", port=_free_port(), **{field: value}))


@pytest.mark.parametrize("field", ["act_endpoint", "set_task_endpoint", "reset_endpoint"])
def test_an_http_route_without_a_leading_slash_is_rejected_at_construction(field):
    with pytest.raises(ValueError, match=rf"{field}='act' is not an HTTP route.*Set {field}='/act'"):
        make_transport("http_json", TransportConfig(host="h", port=1, **{field: "act"}))


# ---------------------------------------------------------------------------
# http_json
# ---------------------------------------------------------------------------


def test_http_json_set_task_reset_and_infer_use_their_routes_and_payload_keys():
    with HttpPolicyServer(lambda call, record: {"action": [[1.0, 2.0]]}) as server:
        transport = _transport(
            "http_json", server, set_task_endpoint="/set_task", reset_endpoint="/reset", reset_payload_key="instruction"
        )
        transport.set_task("do it")
        transport.reset()
        reply = transport.infer({"x": 1})

    assert server.records == [
        {"url": "set_task", "json": {"task_description": "do it"}},
        {"url": "reset", "json": {"instruction": "do it"}},
        {"url": "act", "json": {"x": 1}},
    ]
    assert reply == {"action": [[1.0, 2.0]]}


def test_http_json_query_param_task_and_bare_reset_knobs():
    """Octo's server reads the task from the query string and resets on a bare POST."""
    with HttpPolicyServer() as server:
        transport = _transport(
            "http_json",
            server,
            set_task_endpoint="/set_task",
            reset_endpoint="/reset",
            knobs={"set_task_as_query": True, "reset_with_payload": False},
        )
        transport.set_task("t")
        transport.reset()

    assert server.records == [{"url": "set_task", "params": {"task_description": "t"}}, {"url": "reset"}]


def test_a_server_without_a_reset_or_set_task_api_is_sent_nothing():
    # Servers that take the task with every request have no /reset or /set_task route; the
    # transport must not invent one (an env reset would otherwise POST /reset and 404).
    with HttpPolicyServer() as server:
        transport = _transport("http_json", server)
        transport.set_task("t")
        transport.reset()
    assert server.records == []


def test_http_json_sends_a_non_ascii_task_intact():
    with HttpPolicyServer() as server:
        _transport("http_json", server, set_task_endpoint="/set_task").set_task("把杯子拿起来")
    assert server.records == [{"url": "set_task", "json": {"task_description": "把杯子拿起来"}}]


# ---------------------------------------------------------------------------
# HTTP error mapping
# ---------------------------------------------------------------------------


def test_an_unreachable_server_is_a_transport_error_that_is_a_connection_error():
    transport = make_transport("http_json", TransportConfig(host="127.0.0.1", port=_free_port()))
    with pytest.raises(TransportError, match="Cannot reach policy server") as raised:
        transport.infer({})
    assert isinstance(raised.value, ConnectionError)


def test_an_http_error_status_is_a_server_error_carrying_the_server_body():
    with HttpPolicyServer(lambda call, record: "Traceback: model exploded", status=500) as server:
        with pytest.raises(ServerError, match="model exploded") as raised:
            _transport("http_json", server).infer({})
    assert isinstance(raised.value, RuntimeError)


def test_a_non_json_reply_is_a_serialization_error():
    with HttpPolicyServer(lambda call, record: "<html>502 Bad Gateway</html>") as server:
        with pytest.raises(SerializationError, match="non-JSON reply"):
            _transport("http_json", server).infer({})


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (float("nan"), "Strict JSON has no NaN or inf"),
        (np.zeros(2), "not JSON serializable.*transport='http_json_numpy'"),
    ],
    ids=["nan", "ndarray"],
)
def test_a_payload_json_cannot_carry_is_a_serialization_error_sent_nowhere(value, match):
    with HttpPolicyServer(lambda call, record: {}) as server:
        with pytest.raises(SerializationError, match=match):
            _transport("http_json", server).infer({"x": value})
    assert server.records == []


def test_a_slow_reply_is_a_timeout_naming_request_timeout():
    def slow(call, record):
        time.sleep(0.5)
        return {}

    with HttpPolicyServer(slow) as server:
        with pytest.raises(TransportTimeoutError, match=r"did not answer within request_timeout=0.1 s.*raise"):
            _transport("http_json", server, request_timeout=0.1).infer({})


# ---------------------------------------------------------------------------
# http_json_numpy / http_multipart
# ---------------------------------------------------------------------------


def test_http_json_numpy_encodes_the_body_and_decodes_the_text_reply():
    json_numpy = pytest.importorskip("json_numpy")
    reply = lambda call, record: json_numpy.dumps({"actions": np.ones((2, 3), np.float32), "dt_ms": 1.0})  # noqa: E731
    with HttpPolicyServer(reply, decode_body=json_numpy.loads) as server:
        out = _transport("http_json_numpy", server).infer({"arr": np.arange(3)})

    np.testing.assert_array_equal(server.records[0]["json"]["arr"], np.arange(3))
    np.testing.assert_array_equal(out["actions"], np.ones((2, 3), np.float32))
    assert out["dt_ms"] == 1.0


def test_http_json_numpy_server_error_field_raises_server_error():
    json_numpy = pytest.importorskip("json_numpy")
    with HttpPolicyServer(lambda call, record: json_numpy.dumps({"error": "bad prompt"})) as server:
        with pytest.raises(ServerError, match="bad prompt"):
            _transport("http_json_numpy", server).infer({})


def test_http_multipart_posts_the_form_fields_and_files():
    with HttpPolicyServer(lambda call, record: {"action": [[0.5]]}) as server:
        out = _transport("http_multipart", server).infer(
            ({"instruction": "x"}, {"side_image": ("side.jpg", b"jpeg", "image/jpeg")})
        )

    assert server.records[0]["data"] == {"instruction": "x"}
    assert server.records[0]["files"] == {"side_image": ("side.jpg", b"jpeg", "image/jpeg")}
    assert out == {"action": [[0.5]]}


# ---------------------------------------------------------------------------
# zmq_msgpack
# ---------------------------------------------------------------------------


def test_zmq_transport_wraps_requests_in_the_endpoint_envelope():
    _skip_without("zmq", "msgpack_numpy")
    with ZmqPolicyServer("msgpack_numpy", lambda call: {"x": np.zeros((1, 2, 1), np.float32)}) as server:
        transport = _transport("zmq_msgpack", server, reset_endpoint="reset", knobs={"api_token": "tok"})
        action, info = transport.infer({"o": 1})
        transport.reset()
        transport.close()

    np.testing.assert_array_equal(action["x"], np.zeros((1, 2, 1), np.float32))
    assert info == {}
    assert server.requests == [
        {"endpoint": "get_action", "data": {"observation": {"o": 1}, "options": None}, "api_token": "tok"},
        {"endpoint": "reset", "data": {"options": None}, "api_token": "tok"},
    ]


def test_zmq_transport_without_a_reset_endpoint_sends_no_reset():
    _skip_without("zmq", "msgpack_numpy")
    with ZmqPolicyServer("msgpack_numpy", lambda call: {}) as server:
        transport = _transport("zmq_msgpack", server)
        transport.reset()
        transport.close()
    assert server.requests == []


def test_zmq_get_action_reply_shape_is_checked():
    _skip_without("zmq", "msgpack_numpy")
    bare = lambda call: {"action.x": np.zeros(2, np.float32)}  # noqa: E731 — a bare dict, not [action, info]
    with ZmqPolicyServer("msgpack_numpy", bare, wrap_reply=False) as server:
        transport = _transport("zmq_msgpack", server)
        try:
            with pytest.raises(ServerError, match=r"\[action_dict, info\]"):
                transport.infer({"o": 1})
        finally:
            transport.close()


def test_zmq_timeout_names_the_knob_and_the_next_request_still_works():
    _skip_without("zmq", "msgpack_numpy")

    def first_call_is_slow(call):
        if call == 1:
            time.sleep(0.3)
        return {"call": call}

    with ZmqPolicyServer("msgpack_numpy", first_call_is_slow) as server:
        transport = _transport("zmq_msgpack", server, knobs={"timeout_ms": 100})
        try:
            with pytest.raises(TransportTimeoutError, match=r"timeout_ms=100.*knobs=\{'timeout_ms'"):
                transport.infer({"o": 1})
            time.sleep(0.4)  # let the server finish the abandoned request
            action, _ = transport.infer({"o": 2})  # the wedged REQ socket was replaced
        finally:
            transport.close()
    assert action == {"call": 2}


# ---------------------------------------------------------------------------
# WebSocket (OpenPI / GalaxeaVLA)
# ---------------------------------------------------------------------------


def test_openpi_websocket_reads_the_handshake_and_round_trips_an_inference():
    _skip_without("websockets", "msgpack")
    with WebsocketPolicyServer(
        "msgpack_openpi", lambda call: {"actions": np.ones((2, 3), np.float32)}, metadata={"model": "pi0"}
    ) as server:
        transport = _transport("openpi_websocket", server)  # keepalive pings on, as OpenPI's client
        try:
            reply = transport.infer({"state": np.arange(3, dtype=np.float32), "prompt": "pick"})
            metadata = transport.server_config()
        finally:
            transport.close()

    assert metadata == {"model": "pi0"}
    np.testing.assert_array_equal(reply["actions"], np.ones((2, 3), np.float32))
    np.testing.assert_array_equal(server.records[0]["state"], np.arange(3, dtype=np.float32))


def test_openpi_websocket_stops_waiting_for_a_server_after_wait_for_server_s():
    _skip_without("websockets", "msgpack")
    config = TransportConfig(host="127.0.0.1", port=_free_port(), knobs={"wait_for_server_s": 0.0})
    with pytest.raises(TransportError, match=r"still refuses connections.*knobs=\{'wait_for_server_s'"):
        make_transport("openpi_websocket", config)


def test_a_websocket_closed_by_the_server_is_a_transport_error():
    _skip_without("websockets", "msgpack")

    def crash(call):
        raise RuntimeError("server crashed mid-request")

    with WebsocketPolicyServer("msgpack_openpi", crash) as server:
        transport = _transport("openpi_websocket", server)
        try:
            with pytest.raises(TransportError, match="was closed"):
                transport.infer({"x": 1})
        finally:
            transport.close()


@pytest.mark.parametrize(
    ("control_reply", "error"),
    [
        (None, None),  # acknowledged with the same message
        (lambda message: {"error": {"code": 500, "message": "boom"}}, "code=500"),
        (lambda message: {"status": "?"}, "did not acknowledge"),
    ],
    ids=["acknowledged", "server-error", "not-acknowledged"],
)
def test_galaxea_reset_checks_the_server_acknowledgement(control_reply, error):
    _skip_without("websockets", "msgpack")
    with WebsocketPolicyServer(
        "msgpack_galaxea", lambda call: {}, control_keys=("__reset__",), control_reply=control_reply
    ) as server:
        transport = _transport("galaxea_websocket", server, reset_endpoint="__reset__")
        try:
            if error is None:
                transport.reset()
            else:
                with pytest.raises(ServerError, match=error):
                    transport.reset()
        finally:
            transport.close()
    assert server.control == [{"__reset__": True}]


def test_galaxea_without_a_reset_endpoint_sends_no_control_message():
    _skip_without("websockets", "msgpack")
    with WebsocketPolicyServer("msgpack_galaxea", lambda call: {}, control_keys=("__reset__",)) as server:
        transport = _transport("galaxea_websocket", server)  # reset_endpoint=None: the server has no reset
        transport.reset()
        transport.set_task("t")
        transport.close()
    assert server.control == [] and server.records == []
