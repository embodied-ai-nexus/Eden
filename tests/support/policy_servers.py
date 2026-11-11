"""Local fake policy servers: the network boundary for client-policy and transport tests.

Each server binds ``127.0.0.1`` on an OS-assigned port, runs in a daemon thread, speaks the real
wire protocol of one family of policy servers, records what it received and answers through a
``respond`` callable. Use them as context managers; ``host``/``port`` go into the client's options.

* :class:`HttpPolicyServer` — ``requests`` clients (``http_json``, ``http_json_numpy``,
  ``http_multipart``). Each POST is recorded as ``{"url": <route without the leading "/">}`` plus
  ``"params"`` (query), ``"json"`` (a decoded body) or ``"data"`` + ``"files"`` (multipart form).
* :class:`WebsocketPolicyServer` — OpenPI / GalaxeaVLA msgpack WebSocket servers: sends the
  metadata handshake, then answers one message per inference.
* :class:`ZmqPolicyServer` — Isaac-GR00T ``PolicyServer`` (and RLDX-1): ZMQ REP with the
  ``{"endpoint", "data"}`` envelope.

Only inference requests count as ``infer_calls``. Control traffic (a WebSocket reset message, a ZMQ
``ping`` / ``reset`` / ``get_modality_config``) is answered and listed in ``control`` but not in
``records``, so ``records`` is exactly the inference payloads.
"""

from __future__ import annotations

import json
import threading
from email.parser import BytesParser
from email.policy import HTTP as _HTTP_POLICY
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import pytest

from eden.utils.policy_client import make_serializer

Respond = Callable[..., Any]


class _ServerBase:
    host = "127.0.0.1"

    def __init__(self, respond: Respond | None):
        self.respond = respond
        self.records: list[Any] = []
        self.control: list[Any] = []
        self.infer_calls = 0
        self._lock = threading.Lock()

    def _next_call(self) -> int:
        with self._lock:
            self.infer_calls += 1
            return self.infer_calls


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class HttpPolicyServer(_ServerBase):
    """A ``requests``-facing policy server.

    Parameters
    ----------
    respond
        ``respond(call, record) -> reply`` for inference POSTs (``call`` counts from 1). A ``str`` /
        ``bytes`` reply is sent verbatim; anything else as JSON. Other POSTs get ``{}``.
    decode_body
        How a non-multipart body becomes ``record["json"]`` (default ``json.loads``; pass
        ``json_numpy.loads`` for a ``json_numpy`` client).
    act_suffix
        A POST whose route ends with this is an inference call.
    status
        HTTP status of every reply (set it to test error mapping).
    """

    def __init__(
        self,
        respond: Respond | None = None,
        *,
        decode_body: Callable[[str], Any] = json.loads,
        act_suffix: str = "act",
        status: int = 200,
    ):
        super().__init__(respond)
        self.decode_body = decode_body
        self.act_suffix = act_suffix
        self.status = status
        self._httpd = ThreadingHTTPServer((self.host, 0), self._handler())
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def _record(self, path: str, content_type: str, body: bytes) -> dict:
        url = urlparse(path)
        record: dict[str, Any] = {"url": url.path.lstrip("/")}
        if url.query:
            record["params"] = {k: v[0] if len(v) == 1 else v for k, v in parse_qs(url.query).items()}
        if content_type.startswith("multipart/form-data"):
            message = BytesParser(policy=_HTTP_POLICY).parsebytes(
                b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body
            )
            data, files = {}, {}
            for part in message.iter_parts():
                name = part.get_param("name", header="content-disposition")
                payload = part.get_payload(decode=True)
                if part.get_filename() is None:
                    data[name] = payload.decode()
                else:
                    files[name] = (part.get_filename(), payload, part.get_content_type())
            record["data"], record["files"] = data, files
        elif body:
            record["json"] = self.decode_body(body.decode())
        return record

    def _handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 — http.server's spelling
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                record = server._record(self.path, self.headers.get("Content-Type", ""), body)
                server.records.append(record)
                if record["url"].endswith(server.act_suffix) and server.respond is not None:
                    reply = server.respond(server._next_call(), record)
                else:
                    reply = {}
                payload = reply if isinstance(reply, (str, bytes)) else json.dumps(reply)
                payload = payload.encode() if isinstance(payload, str) else payload
                self.send_response(server.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        return Handler

    def __enter__(self) -> HttpPolicyServer:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


# ---------------------------------------------------------------------------
# WebSocket (OpenPI / GalaxeaVLA)
# ---------------------------------------------------------------------------


class WebsocketPolicyServer(_ServerBase):
    """An OpenPI-protocol WebSocket server (``openpi_websocket`` / ``galaxea_websocket`` clients).

    Parameters
    ----------
    serializer
        ``SERIALIZER_REGISTRY`` key the client uses (``msgpack_openpi`` / ``msgpack_galaxea``).
    respond
        ``respond(call) -> reply`` for each inference message.
    metadata
        The handshake the server sends on connect.
    control_keys
        Single-key messages ``{key: True}`` that are control traffic (GalaxeaVLA's ``__reset__``).
    control_reply
        ``control_reply(message) -> reply`` for a control message; by default it is acknowledged
        with the same message.

    A ``respond`` that raises makes the server drop the connection (close code 1011).
    """

    def __init__(
        self,
        serializer: str,
        respond: Respond,
        *,
        metadata: dict | None = None,
        control_keys: tuple[str, ...] = (),
        control_reply: Callable[[dict], Any] | None = None,
    ):
        super().__init__(respond)
        # Both of its serialisers are msgpack codecs, in the policy-client extra with websockets: without the extra a
        # test serving a client is skipped, as the extra-less suite skips every other optional runtime.
        pytest.importorskip("websockets", exc_type=ImportError)
        pytest.importorskip("msgpack", exc_type=ImportError)
        import websockets.sync.server as ws_server

        self.serializer = make_serializer(serializer)
        self.metadata = {} if metadata is None else metadata
        self.control_keys = control_keys
        self.control_reply = control_reply if control_reply is not None else (lambda message: message)
        self._server = ws_server.serve(self._handle, self.host, 0, compression=None, max_size=None)
        self.port = self._server.socket.getsockname()[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def _handle(self, conn) -> None:
        conn.send(self.serializer.encode(self.metadata))
        for message in conn:
            request = self.serializer.decode(message)
            if isinstance(request, dict) and len(request) == 1 and next(iter(request)) in self.control_keys:
                self.control.append(request)
                conn.send(self.serializer.encode(self.control_reply(request)))
                continue
            self.records.append(request)
            conn.send(self.serializer.encode(self.respond(self._next_call())))

    def __enter__(self) -> WebsocketPolicyServer:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()


# ---------------------------------------------------------------------------
# ZMQ (Isaac-GR00T PolicyServer / RLDX-1)
# ---------------------------------------------------------------------------


class ZmqPolicyServer(_ServerBase):
    """A GR00T-style ZMQ REP server (``zmq_msgpack`` client).

    ``get_action`` requests are recorded as their ``{"observation", "options"}`` data and answered
    ``[respond(call), info]`` (``respond(call)`` alone with ``wrap_reply=False``, a malformed reply);
    ``get_modality_config`` returns ``modality_config``; ``ping`` and ``reset`` are acknowledged.
    ``requests`` keeps every decoded ``{"endpoint", "data", ...}`` envelope, control traffic included.
    """

    def __init__(
        self,
        serializer: str,
        respond: Respond,
        *,
        modality_config: dict | None = None,
        info: dict | None = None,
        wrap_reply: bool = True,
    ):
        super().__init__(respond)
        import zmq

        self._zmq = zmq
        self.serializer = make_serializer(serializer)
        self.modality_config = {} if modality_config is None else modality_config
        self.info = {} if info is None else info
        self.wrap_reply = wrap_reply
        self.requests: list[dict] = []
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self.port = self._socket.bind_to_random_port(f"tcp://{self.host}")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def _reply(self, request: dict) -> Any:
        self.requests.append(request)
        endpoint = request.get("endpoint")
        if endpoint == "get_action":
            self.records.append(request["data"])
            reply = self.respond(self._next_call())
            return [reply, self.info] if self.wrap_reply else reply
        self.control.append(endpoint)
        if endpoint == "get_modality_config":
            return self.modality_config
        if endpoint == "ping":
            return {"status": "ok"}
        if endpoint == "reset":
            return {}
        return {"error": f"unknown endpoint {endpoint!r}"}

    def _serve(self) -> None:
        poller = self._zmq.Poller()
        poller.register(self._socket, self._zmq.POLLIN)
        while not self._stop.is_set():
            if not poller.poll(50):
                continue
            request = self.serializer.decode(self._socket.recv())
            self._socket.send(self.serializer.encode(self._reply(request)))

    def __enter__(self) -> ZmqPolicyServer:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._socket.close(linger=0)
        self._context.term()
