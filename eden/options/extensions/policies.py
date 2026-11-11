"""Configuration options for policy extensions (local and remote clients)."""

from __future__ import annotations

from typing import Literal

from eden.options.options import DynamicConfigurableOptions

#: How a client executes the server's action chunk (see ``ActionChunkScheduler``).
ChunkMode = Literal["client", "server", "single"]
#: What a client does with a chunk shorter than ``pred_action_horizon``.
ShortChunk = Literal["error", "accept"]


class PolicyOptions(DynamicConfigurableOptions):
    """
    Options for policy extensions.

    Parameters
    ----------
    checkpoint_path: str | None
        The path to the checkpoint file.
    """

    checkpoint_path: str | None = None


class ClientOptions(DynamicConfigurableOptions):
    """
    Options for policy client extensions (``ClientMixin``).

    Parameters
    ----------
    host: str
        Policy server hostname, or a full URL (``wss://example.com/ws``, ``http://node:8000``)
        whose scheme matches the transport.
    port: int | None
        Policy server port. ``None`` uses the port embedded in ``host`` (or the scheme's
        default when there is none); a port in ``host`` that disagrees with ``port`` raises.
    request_timeout: float
        Request timeout in seconds for HTTP transports.
    set_task_endpoint: str | None
        The server's set-task API (HTTP route); ``None`` (default) when the server has none
        and takes the task with each request.
    reset_endpoint: str | None
        The server's reset API: an HTTP route, a ZMQ endpoint name or a GalaxeaVLA reset
        message key; ``None`` (default) when the server has none, so a reset only clears
        local state.
    act_endpoint: str
        HTTP endpoint for requesting actions.
    task_description: str | None
        Language instruction sent on set_task / reset. Optional: policies without a
        language input leave it ``None``.
    set_task_payload_key: str
        JSON key carrying the task description on set_task.
    reset_payload_key: str
        JSON key carrying the task description on reset.
    transport: str
        ``TRANSPORT_REGISTRY`` key selecting the wire protocol
        (``http_json``, ``http_json_numpy``, ``http_multipart``, ``openpi_websocket``,
        ``galaxea_websocket``, ``zmq_msgpack``). Defaults to ``http_json``.
    serializer: str | None
        ``SERIALIZER_REGISTRY`` key; ``None`` uses the transport's default.
    chunk_mode: ChunkMode
        ``client`` (buffer the server's chunk and re-query when drained), ``server`` (the
        server caches the chunk and serves one row per call) or ``single`` (one action
        per call).
    pred_action_horizon: int | None
        Expected action-chunk length; ``None`` accepts any length.
    exec_horizon: int | None
        Rows executed open-loop before re-querying; ``None`` executes the whole chunk.
    short_chunk: ShortChunk
        ``error`` raises when the server returns fewer than ``pred_action_horizon`` rows;
        ``accept`` queues what came back.
    chunk_temporal_ensemble_steps: int
        Temporal ensembling of the previous chunk's unused tail into the new head; zero
        keeps plain open-loop execution.
    chunk_temporal_ensemble_dims: int | None
        Restrict ensembling to the first N action dims (e.g. to spare a binary gripper).
    response_action_key: str
        Key holding the action chunk in a dict reply (``action`` or ``actions``).
    """

    host: str = "localhost"
    port: int | None = 9010
    request_timeout: float = 30.0
    set_task_endpoint: str | None = None
    reset_endpoint: str | None = None
    act_endpoint: str = "/act"
    task_description: str | None = None
    set_task_payload_key: str = "task_description"
    reset_payload_key: str = "task_description"
    transport: str = "http_json"
    serializer: str | None = None
    chunk_mode: ChunkMode = "client"
    pred_action_horizon: int | None = None
    exec_horizon: int | None = None
    short_chunk: ShortChunk = "error"
    chunk_temporal_ensemble_steps: int = 0
    chunk_temporal_ensemble_dims: int | None = None
    response_action_key: str = "action"


class RemotePolicyOptions(PolicyOptions, ClientOptions):
    """Merged options for policies that combine ``ClientMixin`` and ``PolicyBase``.

    Use as the type argument when subclassing both: ``class FooPolicy(
    ClientMixin, PolicyBase, ConfigurableMixin[RemotePolicyOptions]):``. Without
    this third base, ``__init_subclass__`` would resolve ``_options_class_`` to
    whichever parent appears first in MRO, hiding the other parent's fields.
    """
