"""Request/reply transports to an inference (or any request/reply) server.

A transport moves a payload to a server and returns the decoded reply. It is built from
an explicit :class:`TransportConfig` (address, timeout, endpoints, serializer key and
transport-specific ``knobs``) and selected by ``TRANSPORT_REGISTRY`` key, so nothing here
depends on who the caller is. Optional dependencies are imported lazily.

Every transport maps failures onto :mod:`eden.utils.policy_client.errors`: unreachable server or
timeout -> :class:`TransportError` (a ``ConnectionError``), server-side failure ->
:class:`ServerError` (a ``RuntimeError``).

====================  =============================================  ==================
Key                   Wire                                           Default serializer
====================  =============================================  ==================
``http_json``         ``requests`` POST, JSON body / JSON reply       (strict JSON)
``http_json_numpy``   POST with a ``json_numpy`` text body            ``json_numpy``
``http_multipart``    POST multipart form; payload is ``(data, files)`` none
``openpi_websocket``  WebSocket, msgpack + OpenPI numpy hook          ``msgpack_openpi``
``galaxea_websocket`` as above, str-keyed records, ``__reset__`` msg  ``msgpack_galaxea``
``zmq_msgpack``       ZMQ REQ/REP ``{"endpoint", "data"}`` envelope   ``msgpack_numpy``
====================  =============================================  ==================
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar
from urllib.parse import urlsplit, urlunsplit

from eden.utils.registry import Registry
from eden.utils.policy_client.errors import ServerError, SerializationError, TransportError, TransportTimeoutError
from eden.utils.policy_client.serializer import SerializerBase, make_serializer, require_module

TRANSPORT_REGISTRY = Registry("TRANSPORT")

_LOG = logging.getLogger(__name__)


@dataclass
class TransportConfig:
    """Everything a transport needs to know, independent of the caller.

    Parameters
    ----------
    host, port
        Server address, joined by :func:`server_url`: ``host`` is a bare hostname or a full
        URL (``wss://example.com/ws``, ``http://node:8000``). A port embedded in the URL and
        ``port`` must agree; pass ``port=None`` to use the URL's.
    request_timeout: float
        Request timeout in seconds (HTTP transports).
    act_endpoint: str
        HTTP route for inference, starting with ``/``.
    set_task_endpoint, reset_endpoint: str | None
        The server's set-task / reset API: an HTTP route, a ZMQ endpoint name or a GalaxeaVLA
        reset message key. ``None`` (the default) means the server has none (it takes the
        task with each request), so ``set_task`` / ``reset`` send nothing. A transport whose
        protocol has no such message rejects a non-``None`` value.
    set_task_payload_key, reset_payload_key: str
        JSON keys carrying the task description (HTTP transports).
    task_description: str | None
        Initial task text; :meth:`TransportBase.set_task` updates it.
    serializer: str | None
        ``SERIALIZER_REGISTRY`` key; ``None`` takes the transport's default.
    knobs: dict[str, Any]
        Transport-specific settings; each transport declares the ones it reads (with
        defaults) in its ``knobs`` class attribute.

    ``request_timeout``, ``act_endpoint`` and the two payload keys are HTTP settings: a
    transport that does not read one rejects a value other than its default rather than
    ignoring it (the ZMQ timeout, for one, is ``knobs={"timeout_ms": ...}``).
    """

    host: str = "localhost"
    port: int | None = None
    request_timeout: float = 30.0
    act_endpoint: str = "/act"
    set_task_endpoint: str | None = None
    reset_endpoint: str | None = None
    set_task_payload_key: str = "task_description"
    reset_payload_key: str = "task_description"
    task_description: str | None = None
    serializer: str | None = None
    knobs: dict[str, Any] = field(default_factory=dict)


#: :class:`TransportConfig` fields only some transports read (``TransportBase.config_fields``).
_OPTIONAL_CONFIG_FIELDS = frozenset({"request_timeout", "act_endpoint", "set_task_payload_key", "reset_payload_key"})


def server_url(host: str, port: int | None, *, schemes: tuple[str, ...]) -> str:
    """Join ``host`` and ``port`` into ``scheme://host[:port][/path]``.

    ``host`` is either a bare hostname (``localhost``, ``10.0.0.1``, ``node:8000``), which gets
    ``schemes[0]``, or a full URL whose scheme must be one of ``schemes``. Its path is kept,
    so ``host="wss://example.com/ws", port=None`` addresses the server as given. A port in
    ``host`` and ``port`` must agree. Pass ``port=None`` to use the URL's port, or the
    scheme's default port when the URL has none. ``tcp`` (ZMQ) needs a port and no path.
    """
    parts = urlsplit(host if "://" in host else f"{schemes[0]}://{host}")
    scheme = parts.scheme.lower()
    if scheme not in schemes:
        raise ValueError(f"Policy server host {host!r} uses scheme {scheme!r}; this transport accepts {list(schemes)}.")
    if not parts.hostname:
        raise ValueError(f"Policy server host {host!r} has no hostname.")
    try:
        embedded = parts.port
    except ValueError as e:
        raise ValueError(f"Policy server host {host!r} has an invalid port.") from e
    if embedded is not None and port is not None and embedded != port:
        raise ValueError(
            f"Policy server host {host!r} embeds port {embedded}, but port={port}. Pass port=None to use the "
            "URL's port, or drop the port from the URL."
        )
    resolved = embedded if embedded is not None else port
    if scheme == "tcp" and (resolved is None or parts.path not in ("", "/")):
        raise ValueError(f"A tcp:// policy server address needs a port and no path; got host={host!r}, port={port}.")
    netloc = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    if parts.username is not None:
        userinfo = parts.username if parts.password is None else f"{parts.username}:{parts.password}"
        netloc = f"{userinfo}@{netloc}"
    if resolved is not None:
        netloc = f"{netloc}:{resolved}"
    return urlunsplit((scheme, netloc, parts.path, parts.query, parts.fragment))


def make_transport(key: str, config: TransportConfig) -> TransportBase:
    """Instantiate the transport registered under ``key``; unknown keys list the available ones."""
    if key not in TRANSPORT_REGISTRY.keys():
        raise ValueError(f"Unknown transport {key!r}. Available: {sorted(TRANSPORT_REGISTRY.keys())}")
    return TRANSPORT_REGISTRY.get(key)(config)


class TransportBase(ABC):
    """Request/reply channel to a server.

    Subclasses declare in ``knobs`` the optional settings they read from
    ``config.knobs`` (with defaults), so a transport's configuration surface is visible
    in one place.
    """

    #: Serializer used when ``config.serializer`` is ``None``. ``None`` marks a transport that builds its own
    #: request body (strict JSON, multipart form) and therefore rejects an explicit serializer.
    default_serializer: ClassVar[str | None] = None
    #: Optional settings this transport reads from ``config.knobs``, with their defaults.
    knobs: ClassVar[dict[str, Any]] = {}
    #: Whether the protocol can carry a server reset / set-task call (``config.reset_endpoint`` /
    #: ``config.set_task_endpoint``). A set endpoint on a transport that cannot is rejected.
    supports_reset_endpoint: ClassVar[bool] = False
    supports_set_task_endpoint: ClassVar[bool] = False
    #: The optional :class:`TransportConfig` fields (of ``_OPTIONAL_CONFIG_FIELDS``) this transport
    #: reads. One it does not read must keep its default, so a setting cannot silently do nothing.
    config_fields: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, config: TransportConfig) -> None:
        unknown = set(config.knobs) - set(self.knobs)
        if unknown:
            raise ValueError(
                f"{type(self).__name__} does not accept knobs {sorted(unknown)}; known: {sorted(self.knobs)}"
            )
        defaults = {f.name: f.default for f in fields(TransportConfig)}
        for name in sorted(_OPTIONAL_CONFIG_FIELDS - self.config_fields):
            value = getattr(config, name)
            if value != defaults[name]:
                raise ValueError(
                    f"{type(self).__name__} does not read {name} (got {value!r}), so it would have no effect. "
                    f"Leave it at its default {defaults[name]!r}; this transport's own settings are knobs "
                    f"{sorted(self.knobs)}."
                )
        for name, supported in (
            ("reset_endpoint", self.supports_reset_endpoint),
            ("set_task_endpoint", self.supports_set_task_endpoint),
        ):
            if not supported and getattr(config, name) is not None:
                raise ValueError(
                    f"{type(self).__name__} has no {name.removesuffix('_endpoint')} message; its servers take "
                    f"the task with each request. Set {name}=None (got {getattr(config, name)!r})."
                )
        if self.default_serializer is None and config.serializer is not None:
            raise ValueError(
                f"{type(self).__name__} builds its own request body and takes no serializer. "
                f"Set serializer=None (got {config.serializer!r}), or pick a transport that encodes with one."
            )
        self.config = config
        key = config.serializer if config.serializer is not None else self.default_serializer
        self.serializer: SerializerBase | None = make_serializer(key) if key is not None else None
        self.metadata: dict = {}

    def knob(self, name: str) -> Any:
        """Read a declared knob, falling back to its default."""
        if name not in self.knobs:
            raise KeyError(f"{type(self).__name__} does not declare knob {name!r}")
        return self.config.knobs.get(name, self.knobs[name])

    @abstractmethod
    def infer(self, payload: Any, options: dict | None = None) -> Any:
        """Send one payload and return the decoded reply."""

    def set_task(self, task_description: str) -> None:
        """Record a new task text; servers with a set_task route also get told."""
        self.config.task_description = task_description

    def reset(self, options: dict | None = None) -> None:
        """Reset server-side episode state through ``config.reset_endpoint``; default: no such API."""

    def server_config(self) -> dict:
        """Whatever the server advertises about itself (handshake metadata by default)."""
        return dict(self.metadata)

    def close(self) -> None:
        """Release the connection."""


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class HttpTransportBase(TransportBase):
    """Shared ``requests.Session`` plumbing and error mapping for the HTTP transports.

    ``knobs``: ``set_task_as_query`` (send the task as a URL query parameter instead of a
    JSON body; Octo) and ``reset_with_payload`` (send ``{reset_payload_key:
    task_description}`` on reset, else a bare POST).
    """

    knobs = {"set_task_as_query": False, "reset_with_payload": True}
    supports_reset_endpoint = True
    supports_set_task_endpoint = True
    config_fields = _OPTIONAL_CONFIG_FIELDS

    def __init__(self, config: TransportConfig) -> None:
        super().__init__(config)
        import requests

        for name in ("act_endpoint", "set_task_endpoint", "reset_endpoint"):
            route = getattr(config, name)
            if route is not None and not (isinstance(route, str) and route.startswith("/")):
                raise ValueError(
                    f"{type(self).__name__} {name}={route!r} is not an HTTP route: it is appended to the "
                    f"server URL, so it must start with '/'. Set {name}={'/' + str(route).lstrip('/')!r}."
                )
        self.session = requests.Session()
        self.base_url = server_url(config.host, config.port, schemes=("http", "https")).rstrip("/")

    def post(self, endpoint: str, **kwargs):
        """POST to ``endpoint`` and return the raw response, mapping request failures."""
        import requests

        url = f"{self.base_url}{endpoint}"
        timeout = self.config.request_timeout
        try:
            response = self.session.post(url, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response
        except requests.ConnectTimeout as e:
            raise TransportTimeoutError(
                f"Connecting to policy server at {self.base_url} timed out after request_timeout={timeout} s. "
                "Is it running, and is that address reachable from this machine?"
            ) from e
        except requests.Timeout as e:  # connected, but no reply in time (ReadTimeout)
            raise TransportTimeoutError(
                f"Policy server at {url} did not answer within request_timeout={timeout} s. If the server is "
                "just slow on this request (e.g. compiling on its first call), raise request_timeout."
            ) from e
        except requests.ConnectionError as e:
            raise TransportError(f"Cannot reach policy server at {self.base_url}. Is it running?") from e
        except requests.HTTPError as e:
            body = e.response.text[:2000] if e.response is not None else ""
            raise ServerError(f"Policy server request to {url} failed ({e}):\n{body}") from e
        except requests.RequestException as e:
            raise TransportError(f"Policy server request to {url} failed: {e}") from e

    @staticmethod
    def _json(response) -> Any:
        """Parse a JSON reply; a non-JSON body is a :class:`SerializationError` naming the URL."""
        try:
            return response.json()
        except ValueError as e:  # requests' JSONDecodeError derives from ValueError
            raise SerializationError(
                f"Policy server at {response.url} returned a non-JSON reply: {response.text[:500]!r}"
            ) from e

    def _json_body(self, payload: Any, endpoint: str) -> bytes:
        """Encode ``payload`` as strict JSON (no NaN/inf, as ``requests``' ``json=`` would)."""
        try:
            return json.dumps(payload, allow_nan=False).encode()
        except TypeError as e:
            raise SerializationError(
                f"Cannot send the payload to {self.base_url}{endpoint} as JSON: {e}. Convert numpy arrays and "
                "tensors with .tolist() first, or use transport='http_json_numpy', which carries arrays."
            ) from e
        except ValueError as e:
            raise SerializationError(
                f"Cannot send the payload to {self.base_url}{endpoint} as JSON: {e}. Strict JSON has no NaN or "
                "inf; a non-finite observation usually means the simulation diverged before this request."
            ) from e

    def post_json(self, endpoint: str, payload: dict | None = None) -> dict:
        """POST a JSON body (none when ``payload`` is ``None``) and return the parsed JSON reply."""
        if payload is None:
            return self._json(self.post(endpoint))
        body = self._json_body(payload, endpoint)
        return self._json(self.post(endpoint, data=body, headers={"Content-Type": "application/json"}))

    def set_task(self, task_description: str) -> None:
        super().set_task(task_description)
        endpoint = self.config.set_task_endpoint
        if endpoint is None:
            return
        if self.knob("set_task_as_query"):
            self.post(endpoint, params={self.config.set_task_payload_key: task_description})
        else:
            self.post_json(endpoint, {self.config.set_task_payload_key: task_description})

    def reset(self, options: dict | None = None) -> None:
        endpoint = self.config.reset_endpoint
        if endpoint is None:
            return
        if self.knob("reset_with_payload"):
            self.post_json(endpoint, {self.config.reset_payload_key: self.config.task_description})
        else:
            self.post(endpoint)

    def close(self) -> None:
        self.session.close()


@TRANSPORT_REGISTRY.register(name="http_json")
class HttpJsonTransport(HttpTransportBase):
    """JSON body in, JSON reply out (RobotArena-style servers: open-pi-zero, Octo)."""

    def infer(self, payload: Any, options: dict | None = None) -> Any:
        return self.post_json(self.config.act_endpoint, payload)


@TRANSPORT_REGISTRY.register(name="http_json_numpy")
class HttpJsonNumpyTransport(HttpTransportBase):
    """``json_numpy``-encoded text body and reply (MolmoAct2's ``host_server_droid.py``)."""

    default_serializer = "json_numpy"

    def infer(self, payload: Any, options: dict | None = None) -> Any:
        body = self.serializer.encode(payload)
        response = self.post(self.config.act_endpoint, data=body, headers={"Content-Type": "application/json"})
        result = self.serializer.decode(response.text)
        if isinstance(result, dict) and "error" in result:
            raise ServerError(f"Policy server returned error: {result['error']}")
        return result


@TRANSPORT_REGISTRY.register(name="http_multipart")
class HttpMultipartTransport(HttpTransportBase):
    """Multipart form upload; the payload is a ``(data, files)`` pair as ``requests`` expects."""

    def infer(self, payload: Any, options: dict | None = None) -> Any:
        try:
            data, files = payload
        except (TypeError, ValueError) as e:
            raise TypeError("http_multipart expects the payload as a (data, files) tuple") from e
        return self._json(self.post(self.config.act_endpoint, data=data, files=files))


# ---------------------------------------------------------------------------
# WebSocket (msgpack)
# ---------------------------------------------------------------------------


@TRANSPORT_REGISTRY.register(name="openpi_websocket")
class OpenPiWebsocketTransport(TransportBase):
    """Port of ``openpi_client.websocket_client_policy.WebsocketClientPolicy`` (no ``openpi_client`` needed).

    Connects to ``ws://host:port`` (``host`` may be a full ``ws://``/``wss://`` URL, see
    :func:`server_url`), reads the server's metadata handshake, then exchanges one msgpack
    message per inference. A str reply is the server reporting an error.

    ``knobs``:

    * ``api_key``: sent as an ``Authorization: Api-Key …`` header.
    * ``ws_ping_timeout``: keepalive pings every 20 s, each answered within this many seconds,
      as the OpenPI client does by default; a dead server then closes the connection instead
      of blocking :meth:`infer` (which has no timeout of its own) forever. ``None`` disables
      them: a server whose inference blocks its event loop (slow JIT compilation, diffusion
      sampling) cannot answer pings in time, and the client would drop a live connection.
    * ``wait_for_server_s``: a refused connection is retried every 5 s, as the OpenPI client
      waits for its server to come up, for at most this many seconds (``None``: forever).
    """

    default_serializer = "msgpack_openpi"
    knobs = {"api_key": None, "ws_ping_timeout": 20.0, "wait_for_server_s": 600.0}
    #: Retry every 5 s while the connection is refused (up to the ``wait_for_server_s`` knob), instead of failing.
    wait_for_server: ClassVar[bool] = True

    def __init__(self, config: TransportConfig) -> None:
        super().__init__(config)
        require_module("websockets")  # a missing policy-client extra fails here, naming it
        from websockets import exceptions as ws_exc
        from websockets.sync import client as ws_sync

        self._ws_exc, self._ws_sync = ws_exc, ws_sync
        self.uri = server_url(config.host, config.port, schemes=("ws", "wss"))
        self._conn = None
        self.connect()

    def _headers(self) -> dict[str, str] | None:
        api_key = self.knob("api_key")
        return {"Authorization": f"Api-Key {api_key}"} if api_key else None

    def _open_timeout(self) -> float | None:
        return None

    def connect(self) -> None:
        ws_exc, ws_sync = self._ws_exc, self._ws_sync
        wait = self.wait_for_server
        started = time.monotonic()
        while True:
            try:
                conn = ws_sync.connect(
                    self.uri,
                    compression=None,
                    max_size=None,
                    additional_headers=self._headers(),
                    ping_interval=None if self.knob("ws_ping_timeout") is None else 20,
                    ping_timeout=self.knob("ws_ping_timeout"),
                    open_timeout=self._open_timeout(),
                )
            except ConnectionRefusedError as e:
                if not wait:
                    raise TransportError(f"Cannot reach policy server at {self.uri}. Is it running?") from e
                limit = self.knob("wait_for_server_s")
                waited = time.monotonic() - started
                if limit is not None and waited >= limit:
                    raise TransportError(
                        f"Policy server at {self.uri} still refuses connections after {waited:.0f} s of waiting. "
                        "Start the server or fix host/port; if it is still loading, wait longer with "
                        "knobs={'wait_for_server_s': <seconds>} (None waits forever)."
                    ) from e
                _LOG.info("Waiting for policy server at %s (%.0f s so far) ...", self.uri, waited)
                time.sleep(5 if limit is None else min(5.0, limit - waited))
                continue
            except TimeoutError as e:  # TCP connect or handshake open_timeout
                raise TransportTimeoutError(f"Connecting to policy server at {self.uri} timed out.") from e
            except OSError as e:
                raise TransportError(f"Cannot reach policy server at {self.uri}. Is it running?") from e
            except ws_exc.InvalidStatus as e:
                # The server answered the handshake with an HTTP error (e.g. a rejected api_key).
                raise ServerError(f"Policy server at {self.uri} rejected the connection: {e}") from e
            except ws_exc.WebSocketException as e:
                raise TransportError(f"WebSocket handshake with {self.uri} failed: {e}") from e
            self._conn = conn
            self.metadata = self._recv()
            return

    def _send(self, message: Any) -> None:
        data = self.serializer.encode(message)
        try:
            self._conn.send(data)
        except self._ws_exc.ConnectionClosed as e:
            raise TransportError(f"Connection to policy server at {self.uri} was closed: {e}") from e

    def _recv(self) -> Any:
        try:
            response = self._conn.recv()
        except self._ws_exc.ConnectionClosed as e:
            raise TransportError(f"Connection to policy server at {self.uri} was closed: {e}") from e
        if isinstance(response, str):
            raise ServerError(f"Error in inference server:\n{response}")
        return self.serializer.decode(response)

    def infer(self, payload: Any, options: dict | None = None) -> Any:
        self._send(payload)
        return self._recv()

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None


@TRANSPORT_REGISTRY.register(name="galaxea_websocket")
class GalaxeaWebsocketTransport(OpenPiWebsocketTransport):
    """GalaxeaVLA ``scripts/serve_policy.py``: OpenPI-style socket with str-keyed records and a reset message.

    Always sends the ``Authorization`` header (empty key allowed), does not retry on
    refused connections, and clears the server's chunk cache with ``{reset_endpoint: True}``
    (``serve_policy.py``'s ``"__reset__"``) on both ``reset`` and ``set_task``; with
    ``reset_endpoint=None`` neither sends anything. A dict reply carrying ``error`` is a
    server failure.

    ``knobs``: ``api_key``, ``ws_ping_timeout``, ``ws_timeout_s`` (connection open timeout).
    """

    default_serializer = "msgpack_galaxea"
    knobs = {"api_key": None, "ws_ping_timeout": None, "ws_timeout_s": 60.0}
    wait_for_server = False
    supports_reset_endpoint = True

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Api-Key {self.knob('api_key') or ''}"}

    def _open_timeout(self) -> float | None:
        return self.knob("ws_timeout_s")

    def _recv(self) -> Any:
        result = super()._recv()
        if isinstance(result, dict) and "error" in result:
            err = result["error"]
            code = err.get("code") if isinstance(err, dict) else None
            message = err.get("message") if isinstance(err, dict) else err
            raise ServerError(f"GalaxeaVLA server error (code={code}): {message}")
        return result

    def set_task(self, task_description: str) -> None:
        TransportBase.set_task(self, task_description)
        self.reset()

    def reset(self, options: dict | None = None) -> None:
        key = self.config.reset_endpoint
        if key is None:
            return
        self._send({key: True})
        # serve_policy.py acknowledges with {"__reset__": True}; errors come back as {"error": ...}.
        reply = self._recv()
        if not (isinstance(reply, dict) and reply.get(key)):
            raise ServerError(f"GalaxeaVLA server did not acknowledge {key}; got {reply!r}")


# ---------------------------------------------------------------------------
# ZMQ (msgpack)
# ---------------------------------------------------------------------------


@TRANSPORT_REGISTRY.register(name="zmq_msgpack")
class ZmqMsgpackTransport(TransportBase):
    """ZMQ REQ/REP client for Isaac-GR00T's ``PolicyServer`` and its derivatives (RLDX-1).

    Requests are ``{"endpoint": name, "data": {...}, "api_token"?: str}``; ``get_action``
    returns ``[action_dict, info]`` and ``reset`` a dict. A dict reply with ``error`` is a
    server failure. After a timeout the REQ socket is wedged, so it is recreated before
    the timeout is re-raised. Select the serializer per server: ``msgpack_numpy`` (GR00T,
    default) or ``msgpack_npy_blob`` (RLDX-1).

    ``knobs``: ``timeout_ms`` (send/receive timeout) and ``api_token``. ``config.reset_endpoint``
    names the server's reset endpoint (``"reset"`` for GR00T / RLDX-1); ``None`` sends nothing.
    """

    default_serializer = "msgpack_numpy"
    knobs = {"timeout_ms": 60000, "api_token": None}
    supports_reset_endpoint = True

    def __init__(self, config: TransportConfig) -> None:
        super().__init__(config)
        self._zmq = require_module("zmq")
        self._context = self._zmq.Context.instance()
        self._socket = None
        self._init_socket()

    @property
    def address(self) -> str:
        return server_url(self.config.host, self.config.port, schemes=("tcp",))

    def _init_socket(self) -> None:
        self._socket = self._context.socket(self._zmq.REQ)
        timeout_ms = int(self.knob("timeout_ms"))
        self._socket.setsockopt(self._zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(self._zmq.SNDTIMEO, timeout_ms)
        self._socket.connect(self.address)

    def call_endpoint(self, endpoint: str, data: dict | None, requires_input: bool = True) -> Any:
        """Round-trip one ``{"endpoint": …}`` request."""
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        api_token = self.knob("api_token")
        if api_token:
            request["api_token"] = api_token
        try:
            self._socket.send(self.serializer.encode(request))
            message = self._socket.recv()
        except self._zmq.error.Again as e:
            # REQ socket is wedged after a timeout — recreate it before re-raising.
            self._socket.close(linger=0)
            self._init_socket()
            raise TransportTimeoutError(
                f"Policy server at {self.address} did not answer {endpoint!r} within timeout_ms="
                f"{self.knob('timeout_ms')}. Is it running at that address? If it is just slow on this request, "
                "raise it with knobs={'timeout_ms': <ms>}."
            ) from e
        response = self.serializer.decode(message)
        if isinstance(response, dict) and "error" in response:
            raise ServerError(f"Policy server error: {response['error']}")
        return response

    def infer(self, payload: Any, options: dict | None = None) -> Any:
        response = self.call_endpoint("get_action", {"observation": payload, "options": options})
        # Server returns [action_dict, info_dict]; msgpack decodes to a list.
        if not (isinstance(response, (list, tuple)) and len(response) == 2 and isinstance(response[0], dict)):
            raise ServerError(
                f"Policy server at {self.address} get_action reply must be [action_dict, info]; "
                f"got {type(response).__name__} {str(response)[:200]}"
            )
        return tuple(response)

    def reset(self, options: dict | None = None) -> None:
        if self.config.reset_endpoint is None:
            return
        self.call_endpoint(self.config.reset_endpoint, {"options": options})

    def server_config(self) -> dict:
        """The server's ``get_modality_config`` reply (GR00T family), or ``{}`` when it has none."""
        result = self.call_endpoint("get_modality_config", None, requires_input=False)
        return result if isinstance(result, dict) else {}

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
