"""Server-backed policies: ``ClientMixin`` and the per-embodiment setup machinery.

The counterpart of :mod:`eden.extensions.policies.onnx` (local actors): a client policy is
``class FooPolicy(ClientMixin, PolicyBase, ConfigurableMixin[RemotePolicyOptions])`` whose
inference happens on a server. The layer is policy-family neutral (VLA, world action model,
remote RL controller, ...) and owns three things:

* a transport from :mod:`eden.utils.policy_client`, selected by key (``transport="http_json"``,
  ``"openpi_websocket"``, ``"zmq_msgpack"``, ...),
* an :class:`ActionChunkScheduler` that turns server action chunks into one
  ``(num_envs, D)`` row per step,
* a uniform lifecycle: :meth:`ClientMixin.set_task`, :meth:`ClientMixin.reset_server`,
  :meth:`ClientMixin.reset`.

Every step runs the same pipeline::

    obs -> observe(obs)                                   per-step hooks (frame history, measured pose)
        -> [if scheduler.needs_inference]
              payload = pack_request(obs)                 default: pack_observations(obs)
              reply   = transport.infer(payload, request_options())
              chunk   = decode_response(reply)            -> (num_envs, T, D)
              scheduler.accept(chunk)
        -> row = scheduler.pop()                          -> (num_envs, D)
        -> unpack_actions(row)                            -> {action_term: tensor}

``step`` is also exposed as its stages ``observe`` (every step: frame histories, measured
poses) / ``pack_request`` / ``infer_payload`` / ``accept_chunk`` / ``pop_action`` (plus
:attr:`ClientMixin.needs_inference`). An evaluator can then run the blocking round-trip on a
worker thread while it keeps stepping the simulation, so the robot moves on during the
wait as much as the inference latency would let it. The evaluator must still call
``observe`` on every step. The contract is one request in flight: accept or discard its
chunk before the next ``pack_request``. Do not call ``reset`` / ``reset_server`` /
``set_task`` while a request is in flight, because nothing here locks or fences the
transport.

Servers that serve several embodiments use :class:`SetupClientMixin` with a
``SETUP_REGISTRY`` of :class:`PolicySetupBase` strategies selected by ``policy_setup``.
"""

from __future__ import annotations

import collections
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch

from eden.options.extensions.policies import ChunkMode, ClientOptions, ShortChunk
from eden.utils.buffers.circular_buffer import CircularBuffer
from eden.utils.common import ConfigurableMixin
from eden.utils.policy_client import TRANSPORT_REGISTRY, TransportBase, TransportConfig, make_transport
from eden.utils.policy_client.transport import HttpTransportBase

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.extensions.policies.adapters import ActionAdapter
    from eden.options.extensions.policies import PolicyOptions
    from eden.options.managers.actions import ActionTermOptions
    from eden.options.managers.observations import ObservationGroupOptions


# ---------------------------------------------------------------------------
# Action-chunk scheduling
# ---------------------------------------------------------------------------


class ActionChunkScheduler:
    """Queue server action chunks and hand them out one row per env per step.

    Parameters
    ----------
    num_envs: int
        Number of envs; chunks are ``(num_envs, T, D)`` and rows are ``(num_envs, D)``.
    mode: {"client", "server", "single"}
        ``client``: this side owns the chunk and re-queries only when its queue drains.
        ``server``: the server caches the chunk and serves one row per call, so every
        step is an inference call (GalaxeaVLA). ``single``: the server returns one
        action per call (Octo). The latter two never buffer: a multi-row chunk raises.
    pred_action_horizon: int | None
        Expected chunk length ``T``. ``None`` accepts any length.
    exec_horizon: int | None
        Rows executed before re-querying. ``None`` executes the whole chunk. Must not
        exceed ``pred_action_horizon`` when both are set.
    short_chunk: {"error", "accept"}
        What to do when the server returns fewer than ``pred_action_horizon`` rows:
        raise (the prediction is truncated relative to the trained horizon) or queue
        what came back.
    ensemble_steps: int
        Temporal ensembling of the previous chunk's unused tail into the new chunk's
        head: for ``K`` steps the old plan is weighted ``K/(K+1) … 1/(K+1)``. Zero
        disables it (bit-exact open-loop execution). Requires both horizons and
        ``ensemble_steps <= pred_action_horizon - exec_horizon``.
    ensemble_dims: int | None
        Only the first ``ensemble_dims`` action dims are ensembled (e.g. to keep a
        binary gripper channel untouched). ``None`` ensembles every dim.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        mode: ChunkMode = "client",
        pred_action_horizon: int | None = None,
        exec_horizon: int | None = None,
        short_chunk: ShortChunk = "error",
        ensemble_steps: int = 0,
        ensemble_dims: int | None = None,
    ) -> None:
        if mode not in ("client", "server", "single"):
            raise ValueError(f"chunk_mode must be 'client', 'server' or 'single', got {mode!r}")
        if short_chunk not in ("error", "accept"):
            raise ValueError(f"short_chunk must be 'error' or 'accept', got {short_chunk!r}")
        if pred_action_horizon is not None and pred_action_horizon <= 0:
            raise ValueError(f"pred_action_horizon must be positive, got {pred_action_horizon}")
        if exec_horizon is not None and exec_horizon <= 0:
            raise ValueError(f"exec_horizon must be positive, got {exec_horizon}")
        if pred_action_horizon is not None and exec_horizon is not None and exec_horizon > pred_action_horizon:
            raise ValueError(f"exec_horizon ({exec_horizon}) must be <= pred_action_horizon ({pred_action_horizon})")
        if ensemble_steps < 0:
            raise ValueError("chunk_temporal_ensemble_steps must be non-negative")
        if ensemble_steps > 0:
            if pred_action_horizon is None or exec_horizon is None:
                raise ValueError("temporal ensembling requires both pred_action_horizon and exec_horizon")
            available_overlap = pred_action_horizon - exec_horizon
            if ensemble_steps > available_overlap:
                raise ValueError(
                    f"chunk_temporal_ensemble_steps ({ensemble_steps}) exceeds the unused "
                    f"prediction tail ({available_overlap})"
                )
        if ensemble_dims is not None and ensemble_dims <= 0:
            raise ValueError("chunk_temporal_ensemble_dims must be positive or None")

        self.num_envs = int(num_envs)
        self.mode = mode
        self.pred_action_horizon = pred_action_horizon
        self.exec_horizon = exec_horizon
        self.short_chunk = short_chunk
        self.ensemble_steps = int(ensemble_steps)
        self.ensemble_dims = ensemble_dims

        self.queues: list[collections.deque[np.ndarray]] = [collections.deque() for _ in range(self.num_envs)]
        self.raw_queues: list[collections.deque[np.ndarray]] = [collections.deque() for _ in range(self.num_envs)]
        self._previous_unused_tail: list[np.ndarray | None] = [None] * self.num_envs
        self._last_take = 0
        # Diagnostics
        self.chunk_index = -1
        self.chunk_action_index = -1
        self.last_prediction_chunk_length = 0
        self.last_temporal_ensemble_steps = 0
        self.last_raw_row: np.ndarray | None = None

    # ------------------------------------------------------------------

    @property
    def needs_inference(self) -> bool:
        """True when the next :meth:`pop` needs a fresh server chunk first."""
        if self.mode != "client":
            return True
        return not self.queues[0]

    def clear(self) -> None:
        """Drop buffered actions and ensembling state (task change / episode boundary)."""
        for q in self.queues:
            q.clear()
        for q in self.raw_queues:
            q.clear()
        self._previous_unused_tail = [None] * self.num_envs
        self._last_take = 0
        self.chunk_index = -1
        self.chunk_action_index = -1
        self.last_prediction_chunk_length = 0
        self.last_temporal_ensemble_steps = 0
        self.last_raw_row = None

    def accept(self, chunk: np.ndarray, raw_chunk: np.ndarray | None = None) -> None:
        """Queue the first ``exec_horizon`` rows of a ``(num_envs, T, D)`` chunk.

        ``raw_chunk`` (same shape) is an optional untouched copy queued alongside for
        diagnostics, e.g. before a setup latched the gripper column.
        """
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 3 or chunk.shape[0] != self.num_envs:
            raise ValueError(f"Expected an action chunk of shape ({self.num_envs}, T, D), got {chunk.shape}")
        T = chunk.shape[1]
        if T == 0:
            raise ValueError("Policy server returned an empty action chunk.")
        if self.mode != "client" and T != 1:
            raise ValueError(
                f"chunk_mode={self.mode!r} expects one action per inference call, but the server returned "
                f"{T} rows. Use chunk_mode='client' if this server returns action chunks."
            )
        if raw_chunk is None:
            raw_chunk = chunk
        else:
            raw_chunk = np.asarray(raw_chunk, dtype=np.float32)
            if raw_chunk.shape != chunk.shape:
                raise ValueError(f"raw_chunk shape {raw_chunk.shape} does not match chunk shape {chunk.shape}")

        self.last_prediction_chunk_length = int(T)
        if self.pred_action_horizon is not None and T < self.pred_action_horizon and self.short_chunk == "error":
            raise RuntimeError(
                f"Policy server returned {T} actions but pred_action_horizon={self.pred_action_horizon}. "
                "The prediction chunk is truncated — check the server's action_horizon before "
                "open-loop execution."
            )
        take = T if self.exec_horizon is None else min(self.exec_horizon, T)
        if self.pred_action_horizon is not None:
            take = min(take, self.pred_action_horizon)

        self.last_temporal_ensemble_steps = 0
        for i in range(self.num_envs):
            env_chunk = chunk[i]
            previous_tail = self._previous_unused_tail[i]
            if self.ensemble_steps > 0 and previous_tail is not None:
                overlap = min(self.ensemble_steps, previous_tail.shape[0], take)
                dims = env_chunk.shape[1] if self.ensemble_dims is None else min(self.ensemble_dims, env_chunk.shape[1])
                if overlap > 0:
                    env_chunk = env_chunk.copy()
                    # For K steps, old-plan weights are K/(K+1), ..., 1/(K+1): the new
                    # observation-conditioned prediction takes over completely after K rows.
                    old_weights = np.arange(overlap, 0, -1, dtype=np.float32) / float(self.ensemble_steps + 1)
                    env_chunk[:overlap, :dims] = (
                        old_weights[:, None] * previous_tail[:overlap, :dims]
                        + (1.0 - old_weights[:, None]) * env_chunk[:overlap, :dims]
                    )
                    self.last_temporal_ensemble_steps = overlap
            self._previous_unused_tail[i] = chunk[i][take:].copy()
            self.queues[i].extend(env_chunk[:take])
            self.raw_queues[i].extend(raw_chunk[i][:take])

        self._last_take = int(take)
        self.chunk_index += 1

    def pop(self) -> np.ndarray:
        """Return the next ``(num_envs, D)`` row; raises if nothing is queued."""
        if not self.queues[0]:
            raise RuntimeError("No queued action: call accept() with a server chunk before pop().")
        self.chunk_action_index = self._last_take - len(self.queues[0])
        rows = np.stack([q.popleft() for q in self.queues], axis=0)
        self.last_raw_row = np.stack([q.popleft() for q in self.raw_queues], axis=0)
        return rows

    @property
    def action_metadata(self) -> dict[str, int]:
        """Where the last popped row sat in its chunk (for recorders / evaluators)."""
        return {
            "chunk_index": int(self.chunk_index),
            "chunk_action_index": int(self.chunk_action_index),
            "predicted_chunk_length": int(self.last_prediction_chunk_length),
            "executed_chunk_length": int(self._last_take),
        }


# ---------------------------------------------------------------------------
# Reply helpers
# ---------------------------------------------------------------------------


def concat_action_streams(action_dict: dict[str, np.ndarray], keys: tuple[str, ...]) -> np.ndarray:
    """Concatenate per-key ``(1, T, d_k)`` action streams in ``keys`` order into one ``(T, D)`` chunk.

    Servers of the Isaac-GR00T family answer with one array per modality key. A ``(T, d)``
    array (batch already squeezed) is accepted; a batch larger than one is rejected because
    the client serves a single env.
    """
    cols = []
    for key in keys:
        if key not in action_dict:
            raise KeyError(f"Server reply lacks action key {key!r}; got {sorted(action_dict)}")
        arr = np.asarray(action_dict[key], dtype=np.float32)
        if arr.ndim == 2:  # (T, d) — server already squeezed the batch axis
            arr = arr[np.newaxis]
        if arr.ndim != 3:
            raise ValueError(f"Action key {key!r} must be (B, T, d); got shape {arr.shape}")
        if arr.shape[0] != 1:
            raise ValueError(f"Remote clients serve num_envs=1; action key {key!r} has batch={arr.shape[0]}")
        cols.append(arr[0])  # (T, d)
    return np.concatenate(cols, axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Server modality-config validation (Isaac-GR00T family)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModalitySchema:
    """What a setup expects the server's ``get_modality_config`` to say.

    Isaac-GR00T and its derivatives select the checkpoint's embodiment server-side
    (``--embodiment-tag``) and advertise the resulting per-modality ``modality_keys`` and
    ``delta_indices``. A setup declares the schema it packs and decodes; the two are
    compared once at connect time so a mismatch fails there, with both sides named,
    instead of as a ``KeyError`` on the server or a silently wrong video window.
    """

    video_keys: tuple[str, ...]
    state_keys: tuple[str, ...]
    action_keys: tuple[str, ...]
    language_key: str
    action_horizon: int | None = None
    video_horizon: int = 1
    video_stride: int = 1


def validate_modality_config(config: dict, schema: ModalitySchema, *, setup_name: str) -> None:
    """Raise ``ValueError`` when ``config`` (a ``get_modality_config`` reply) contradicts ``schema``.

    An empty ``config`` means the server advertises nothing and passes. Keys are compared
    as sets for video / state (payloads are keyed dicts) and as an ordered list for action
    (the client concatenates streams in that order); horizons come from ``delta_indices``.
    """
    if not config:
        return
    problems: list[str] = []

    def modality(name: str) -> dict | None:
        entry = config.get(name)
        if not isinstance(entry, dict) or "modality_keys" not in entry:
            problems.append(f"server config has no {name!r} modality with modality_keys (got {sorted(config)})")
            return None
        return entry

    video, state, action, language = (modality(k) for k in ("video", "state", "action", "language"))
    if video is not None:
        served = tuple(video["modality_keys"])
        if set(served) != set(schema.video_keys):
            problems.append(f"video keys: setup packs {sorted(schema.video_keys)}, server serves {sorted(served)}")
        delta = list(video.get("delta_indices", []))
        if delta and len(delta) != schema.video_horizon:
            problems.append(
                f"video horizon: setup sends {schema.video_horizon} frame(s), server expects {len(delta)} ({delta})"
            )
        elif len(delta) > 1:
            strides = {b - a for a, b in zip(delta[:-1], delta[1:])}
            if strides != {schema.video_stride}:
                problems.append(
                    f"video stride: setup samples every {schema.video_stride} step(s), server expects {delta}"
                )
    if state is not None:
        served = tuple(state["modality_keys"])
        if set(served) != set(schema.state_keys):
            problems.append(f"state keys: setup packs {sorted(schema.state_keys)}, server serves {sorted(served)}")
    if action is not None:
        served = tuple(action["modality_keys"])
        if served != tuple(schema.action_keys):
            problems.append(
                f"action keys (order matters): setup decodes {list(schema.action_keys)}, server serves {list(served)}"
            )
        delta = list(action.get("delta_indices", []))
        if schema.action_horizon is not None and delta and len(delta) != schema.action_horizon:
            problems.append(f"action horizon: setup expects {schema.action_horizon} rows, server predicts {len(delta)}")
    if language is not None and schema.language_key not in tuple(language["modality_keys"]):
        problems.append(
            f"language key: setup sends {schema.language_key!r}, server accepts {list(language['modality_keys'])}"
        )

    if problems:
        raise ValueError(
            f"Setup {setup_name!r} does not match the served checkpoint's modality config:\n  - "
            + "\n  - ".join(problems)
        )


# ---------------------------------------------------------------------------
# Frame history (temporal observation windows)
# ---------------------------------------------------------------------------


class FrameHistory:
    """Per-step frame buffer that samples a strided temporal window the way ``delta_indices`` do.

    Servers trained with a video window (GR00T ``delta_indices=[-15, 0]``, RLDX-1
    ``[-6, -4, -2, 0]``) expect ``horizon`` frames spaced ``stride`` control steps apart,
    ending at the current one. :meth:`append` records one frame per control step and
    :meth:`window` returns ``(horizon, ...)`` in chronological order; missing history is
    padded with the oldest available frame, as the reference ``MultiStepWrapper`` does by
    seeding its deque with copies of the first observation.

    Built on :class:`~eden.utils.buffers.circular_buffer.CircularBuffer` (batch size 1,
    backfilled), so no per-key list bookkeeping is needed.
    """

    def __init__(self, horizon: int, stride: int) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self.horizon = int(horizon)
        self.stride = int(stride)
        self._max_len = (self.horizon - 1) * self.stride + 1
        self._buffer: CircularBuffer | None = None

    def reset(self) -> None:
        self._buffer = None

    def append(self, frame: np.ndarray | torch.Tensor) -> None:
        tensor = torch.as_tensor(frame)
        if self._buffer is None:
            self._buffer = CircularBuffer(self._max_len, batch_size=1, device=str(tensor.device))
        self._buffer.append(tensor.unsqueeze(0))

    def window(self) -> np.ndarray:
        """``(horizon, ...)`` frames at ``stride`` spacing, newest last."""
        if self._buffer is None:
            raise RuntimeError("No frames recorded: call append() once per control step before window().")
        history = self._buffer.buffer[0]  # (max_len, ...) oldest -> newest, backfilled
        offsets = [(self.horizon - 1 - i) * self.stride for i in range(self.horizon)]
        return np.stack([history[self._max_len - 1 - off].cpu().numpy() for off in offsets], axis=0)


# ---------------------------------------------------------------------------
# ClientMixin
# ---------------------------------------------------------------------------


class ClientMixin(ConfigurableMixin[ClientOptions]):
    """Mixin for policy extensions that delegate inference to a remote policy server.

    Concrete subclasses implement :meth:`pack_observations` and :meth:`unpack_actions`
    (the serialization contract between Eden's observation/action format and the server's
    API) and pick a ``transport``. Everything else — connection handling, chunk queuing,
    horizon rules, task/reset lifecycle — is shared. Override :meth:`decode_response` when
    the reply does not carry the chunk under a single key, :meth:`observe` for per-step
    state the policy needs even on steps that do not query the server, and
    :meth:`request_options` for per-request server options (sessions, memory flags).

    Two requirements must both be met when combining with ``PolicyBase``:

    1. ``ClientMixin`` must appear **before** ``PolicyBase`` so that its concrete
       :meth:`step` takes precedence over the abstract one.
    2. The subclass must re-parameterize ``ConfigurableMixin`` with the merged options
       type ``RemotePolicyOptions``; otherwise ``__init_subclass__`` inherits
       ``_options_class_`` from whichever parent is first in MRO and :meth:`configure`
       rejects the other parent's fields.

    Parameters
    ----------
    host, port
        Policy server address. ``host`` may be a full URL (``wss://example.com/ws``); use
        ``port=None`` to take the URL's port. See :func:`eden.utils.policy_client.server_url`.
    request_timeout: float
        Request timeout in seconds (HTTP transports).
    set_task_endpoint, reset_endpoint: str | None
        The server's set-task / reset API: an HTTP route, a ZMQ endpoint name or a
        GalaxeaVLA reset message key. ``None`` (the default) means the server has none,
        because it takes the task with each request, so :meth:`set_task` / :meth:`reset_server`
        only update local state. Declare one only when the official server exposes it.
    act_endpoint: str
        HTTP route for inference.
    task_description: str | None
        Language instruction sent on ``set_task`` / ``reset``; optional for policies with
        no language input.
    set_task_payload_key, reset_payload_key: str
        JSON keys carrying the task description.
    transport: str
        ``TRANSPORT_REGISTRY`` key. Default ``"http_json"``.
    serializer: str | None
        ``SERIALIZER_REGISTRY`` key; ``None`` takes the transport's default.
    chunk_mode: {"client", "server", "single"}
        Who owns the action chunk: this client (buffer and re-query when drained), the
        server (it caches the chunk and serves one row per call), or nobody (one action
        per call).
    pred_action_horizon: int | None
        Expected chunk length from the server; ``None`` accepts any length.
    exec_horizon: int | None
        Rows executed open-loop before re-querying; ``None`` executes the whole chunk.
    short_chunk: {"error", "accept"}
        Whether a chunk shorter than ``pred_action_horizon`` is an error.
    chunk_temporal_ensemble_steps, chunk_temporal_ensemble_dims
        Temporal ensembling of overlapping chunks; zero disables.
    response_action_key: str
        Key holding the action chunk in a dict reply (``"action"`` or ``"actions"``).
    """

    #: ``step(obs)`` packs the env's observation groups into the server request.
    reads_env_observations = True

    # -- connection -----------------------------------------------------
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

    # -- chunk scheduling ----------------------------------------------
    chunk_mode: ChunkMode = "client"
    pred_action_horizon: int | None = None
    exec_horizon: int | None = None
    short_chunk: ShortChunk = "error"
    chunk_temporal_ensemble_steps: int = 0
    chunk_temporal_ensemble_dims: int | None = None
    response_action_key: str = "action"

    #: Subclass opts in when the remote server accepts batched requests with a
    #: leading (N, ...) axis. When False, num_envs must be 1.
    supports_batch: ClassVar[bool] = False
    #: Width ``D`` of one action row, when the server's schema fixes it; a chunk of another width
    #: is rejected before it is queued. ``None`` skips the check. Setup-driven clients take their
    #: setup's ``action_dim`` instead.
    action_dim: ClassVar[int | None] = None
    # Class-level facts fixed by the subclass: never configure() parameters.
    _non_parameter_names: ClassVar[frozenset[str]] = frozenset({"supports_batch", "action_dim"})

    def __init__(self, env: EnvBase, options: PolicyOptions):
        super().__init__(env, options)
        if not self.supports_batch and env.num_envs != 1:
            raise ValueError(
                f"{type(self).__name__} has supports_batch=False and only supports "
                f"num_envs=1 (got {env.num_envs}). Set supports_batch=True on the "
                f"subclass if the remote server accepts batched requests."
            )
        if env.num_envs > 1 and type(self).reset is ClientMixin.reset:
            # The env resets its envs one by one; the default reset can only reset the shared
            # server state of all of them, so it would raise at the first episode end.
            raise ValueError(
                f"{type(self).__name__} serves {env.num_envs} envs but keeps ClientMixin.reset, which "
                "refuses a partial reset because the server's episode state is shared by every env. Fix: "
                "override reset(envs_idx) with a per-env scheme (e.g. one server session per env), or "
                "run with num_envs=1."
            )
        self._num_envs: int = env.num_envs
        self._post_configure()
        self._scheduler = ActionChunkScheduler(
            num_envs=env.num_envs,
            mode=self.chunk_mode,
            pred_action_horizon=self.pred_action_horizon,
            exec_horizon=self.exec_horizon,
            short_chunk=self.short_chunk,
            ensemble_steps=self.chunk_temporal_ensemble_steps,
            ensemble_dims=self.chunk_temporal_ensemble_dims,
        )
        self.last_response_info: dict[str, Any] = {}
        self._transport: TransportBase = self.make_transport()
        self._post_connect()

    # ------------------------------------------------------------------
    # Construction hooks
    # ------------------------------------------------------------------

    def _post_configure(self) -> None:
        """Runs after options are mirrored onto ``self`` and before the scheduler/transport exist.

        Override to resolve ``None`` chunk settings from a selected embodiment setup or to
        validate cross-field constraints; raise ``ValueError`` naming the offending value.
        """

    def transport_config(self) -> TransportConfig:
        """Collect the connection settings and the selected transport's knobs from this policy."""
        if self.transport not in TRANSPORT_REGISTRY.keys():
            raise ValueError(
                f"{type(self).__name__}.transport={self.transport!r} is not a registered transport. "
                f"Available: {sorted(TRANSPORT_REGISTRY.keys())}"
            )
        transport_cls = TRANSPORT_REGISTRY.get(self.transport)
        knobs = {name: getattr(self, name) for name in transport_cls.knobs if hasattr(self, name)}
        return TransportConfig(
            host=self.host,
            port=self.port,
            request_timeout=self.request_timeout,
            act_endpoint=self.act_endpoint,
            set_task_endpoint=self.set_task_endpoint,
            reset_endpoint=self.reset_endpoint,
            set_task_payload_key=self.set_task_payload_key,
            reset_payload_key=self.reset_payload_key,
            task_description=self.task_description,
            serializer=self.serializer,
            knobs=knobs,
        )

    def make_transport(self) -> TransportBase:
        """Build the transport named by ``self.transport``; override to inject a custom one."""
        return make_transport(self.transport, self.transport_config())

    def _post_connect(self) -> None:
        """Runs once the transport exists: check what the server advertises against this client.

        The default does nothing. Setup-driven clients hand ``transport.server_config()`` to
        the selected setup's ``validate_server_config`` so a checkpoint served with another
        embodiment's schema fails here instead of as a ``KeyError`` on the server.
        """

    # ------------------------------------------------------------------
    # Queue / chunk accessors (diagnostics)
    # ------------------------------------------------------------------

    @property
    def _action_queues(self) -> tuple[tuple[np.ndarray, ...], ...]:
        """Read-only snapshot of the per-env queued actions (queue through ``accept_chunk``)."""
        return tuple(tuple(q) for q in self._scheduler.queues)

    @property
    def _action_plan(self) -> tuple[np.ndarray, ...]:
        """Read-only snapshot of env 0's queued actions (what single-env evaluators read)."""
        return tuple(self._scheduler.queues[0])

    @property
    def last_prediction_chunk_length(self) -> int:
        """Length ``T`` of the most recent server chunk."""
        return self._scheduler.last_prediction_chunk_length

    @property
    def last_temporal_ensemble_steps(self) -> int:
        """How many rows of the most recent chunk were temporally ensembled."""
        return self._scheduler.last_temporal_ensemble_steps

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def set_task(self, task_description: str) -> None:
        """Set the task description locally and on the server, clearing pending actions.

        Should be called after construction and before the first :meth:`step`.
        """
        self.task_description = task_description
        self._scheduler.clear()
        self._reset_local()
        self._transport.set_task(task_description)

    def reset_server(self) -> None:
        """Reset per-episode state locally, and on the server when it has a reset API (``reset_endpoint``)."""
        self._scheduler.clear()
        self._reset_local()
        # The task text may have been assigned directly; reset routes that carry it must send the current one.
        self._transport.config.task_description = self.task_description
        self._transport.reset(self.reset_options())

    def close(self) -> None:
        """Release the connection to the policy server (WebSocket / ZMQ socket, HTTP session)."""
        self._transport.close()

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None) -> None:
        """``PolicyBase`` hook: reset the server once every env this client serves is being reset.

        The server's episode state (history, chunk cache, recurrent memory) belongs to this
        client's connection / session as a whole, so a reset that covers every env maps to
        :meth:`reset_server`; with ``num_envs == 1`` that is every non-empty reset. A partial
        reset of a batched client would wipe the other envs' state, so it raises until the
        client implements a per-env scheme (e.g. one server session per env).
        """
        covered = self._reset_coverage(envs_idx)
        if covered == 0:
            return
        if covered < self.env.num_envs:
            raise NotImplementedError(
                f"{type(self).__name__} cannot reset {covered} of {self.env.num_envs} envs: its server-side "
                "episode state is shared by all of them. Implement a per-env reset (e.g. one session per env)."
            )
        self.reset_server()

    def _reset_coverage(self, envs_idx: (slice | torch.Tensor) | None) -> int:
        """How many distinct envs ``envs_idx`` names (``None`` / ``slice(None)`` = all); invalid indices raise."""
        n = self.env.num_envs
        if envs_idx is None:
            return n
        if isinstance(envs_idx, slice):
            return len(range(n)[envs_idx])
        idx = torch.as_tensor(envs_idx).reshape(-1)
        if idx.dtype == torch.bool:
            if idx.numel() != n:
                raise ValueError(f"reset mask has {idx.numel()} entries for {n} envs.")
            return int(idx.sum())
        if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= n):
            raise ValueError(f"reset envs_idx {idx.tolist()} out of range for {n} envs.")
        return int(torch.unique(idx).numel())

    def _reset_local(self) -> None:
        """Clear policy-specific per-episode state (adapters, pose trackers, frame history)."""

    def reset_options(self) -> dict | None:
        """Options sent with the server ``reset`` (e.g. session ids); ``None`` by default."""
        return None

    def request_options(self) -> dict | None:
        """Options sent with every inference request (e.g. memory flags); ``None`` by default."""
        return None

    # ------------------------------------------------------------------
    # Inference: observe, pack_request, infer_payload, accept_chunk, pop_action (= step)
    # ------------------------------------------------------------------

    @property
    def needs_inference(self) -> bool:
        """True when the next :meth:`step` will query the server."""
        return self._scheduler.needs_inference

    def observe(self, obs: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> None:
        """Per-step hook called before scheduling, on every step (default: no-op).

        Use it for state that must track every control step regardless of when the server
        is queried: rolling frame histories, the measured pose a delta is applied to, ...
        """

    def pack_request(self, obs: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> Any:
        """Build the server payload from an observation (main thread; reads sim state)."""
        return self.pack_observations(obs)

    def infer_payload(self, payload: Any) -> np.ndarray:
        """Blocking server round-trip -> ``(num_envs, T, D)`` chunk.

        Safe on a worker thread while the main thread keeps stepping the simulation, as long
        as it is the only call using the transport, i.e. no ``reset`` / ``reset_server`` /
        ``set_task`` until it returns.
        """
        response = self._transport.infer(payload, self.request_options())
        self._record_response_info(response)
        return self.decode_response(response)

    def accept_chunk(self, chunk: np.ndarray) -> None:
        """Validate a predicted chunk and queue its executable prefix."""
        chunk = self._normalize_action_shape(np.asarray(chunk, dtype=np.float32))
        self._check_action_width(chunk, self.action_dim, schema=type(self).__name__)
        self._scheduler.accept(chunk)

    @staticmethod
    def _check_action_width(chunk: np.ndarray, expected: int | None, *, schema: str) -> None:
        if expected is not None and chunk.shape[-1] != expected:
            raise ValueError(
                f"{schema} expects {expected}-D action rows, the server returned {chunk.shape[-1]}-D. "
                "The served checkpoint does not match this client."
            )

    def pop_action(self) -> dict[str, torch.Tensor]:
        """Unpack and return the next queued ``(num_envs, D)`` row (main thread)."""
        return self.unpack_actions(self._scheduler.pop())

    def step(self, obs: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Query the policy server when needed and return the next action in Eden format."""
        self.observe(obs)
        if self.needs_inference:
            self.accept_chunk(self.infer_payload(self.pack_request(obs)))
        return self.pop_action()

    @property
    def action_metadata(self) -> dict[str, int]:
        """Chunk position of the last executed action."""
        return self._scheduler.action_metadata

    # ------------------------------------------------------------------
    # Serialization hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def pack_observations(self, obs: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> Any:
        """Convert Eden observations to the server payload.

        When ``supports_batch=True``, observation fields in the returned payload must
        carry a leading batch axis of size ``env.num_envs``.
        """
        raise NotImplementedError

    def decode_response(self, response: Any) -> np.ndarray:
        """Extract the action chunk from a server reply as ``(num_envs, T, D)``.

        Default: ``response[response_action_key]`` normalized by
        :meth:`_normalize_action_shape`. Override for replies that spread the action over
        several keys.
        """
        if not isinstance(response, dict) or self.response_action_key not in response:
            raise KeyError(
                f"{type(self).__name__}: server reply has no {self.response_action_key!r} key "
                f"(got {sorted(response) if isinstance(response, dict) else type(response).__name__}). "
                "Set response_action_key or override decode_response()."
            )
        return self._normalize_action_shape(np.asarray(response[self.response_action_key], dtype=np.float32))

    @abstractmethod
    def unpack_actions(self, raw_action: np.ndarray) -> dict[str, torch.Tensor]:
        """Convert a ``(num_envs, D)`` action row from the server to an Eden action dict.

        For non-batched subclasses this is always ``(1, D)``.
        """
        raise NotImplementedError

    def _record_response_info(self, response: Any) -> None:
        """Keep everything the server returned besides the action chunk (WAM predictions, timing, traces).

        A dict reply keeps its other keys; a ``(actions, info)`` pair (the GR00T-family ZMQ reply)
        keeps ``info``.
        """
        if isinstance(response, dict):
            self.last_response_info = {k: v for k, v in response.items() if k != self.response_action_key}
        elif isinstance(response, (tuple, list)) and len(response) == 2 and isinstance(response[1], dict):
            self.last_response_info = dict(response[1])
        else:
            self.last_response_info = {}

    def _normalize_action_shape(self, actions: np.ndarray) -> np.ndarray:
        """Normalize a server chunk to ``(N, chunk_len, D)``.

        Accepts 1D/2D shapes only when ``num_envs == 1``. A 2D response with
        ``num_envs > 1`` is rejected because ``(N, D)`` and ``(chunk_len, D)`` are
        indistinguishable.
        """
        if actions.ndim == 1:
            if self._num_envs != 1:
                raise ValueError(
                    f"Server returned 1D action {actions.shape} but num_envs={self._num_envs}; "
                    f"expected (N, chunk_len, D)."
                )
            return actions[np.newaxis, np.newaxis, :]
        if actions.ndim == 2:
            if self._num_envs != 1:
                raise ValueError(
                    f"Server returned 2D action {actions.shape} with num_envs={self._num_envs}; "
                    f"ambiguous between (N, D) and (chunk_len, D). Require (N, chunk_len, D)."
                )
            return actions[np.newaxis, :, :]
        if actions.ndim == 3:
            if actions.shape[0] != self._num_envs:
                raise ValueError(f"Server returned batch size {actions.shape[0]}, expected {self._num_envs}.")
            return actions
        raise ValueError(f"Unexpected action ndim={actions.ndim}, shape={actions.shape}.")

    # ------------------------------------------------------------------
    # HTTP helper
    # ------------------------------------------------------------------

    def _send_request(self, endpoint: str, payload: dict | None = None) -> dict:
        """POST JSON to the server and return the parsed reply (HTTP transports only)."""
        if not isinstance(self._transport, HttpTransportBase):
            raise TypeError(
                f"{type(self).__name__}._send_request posts JSON over HTTP; transport={self.transport!r} is "
                f"{type(self._transport).__name__}. Call the transport's own request method instead."
            )
        return self._transport.post_json(endpoint, payload)


# ---------------------------------------------------------------------------
# Per-embodiment setups
# ---------------------------------------------------------------------------


class PolicySetupBase(ABC):
    """Strategy object for one server embodiment / schema.

    Class-level defaults (``pred_action_horizon``, ``exec_horizon``, ``default_image_size``)
    fill the policy's ``pred_action_horizon`` / ``exec_horizon`` / ``image_size`` when the
    policy leaves them ``None``, so each setup can carry the horizon and resolution its
    checkpoint was trained with.

    Concrete setups own ``observation_options`` / ``action_options`` (classmethod
    factories so the env can be built without a live policy), ``pack_observations``, and
    ``unpack_actions``; they optionally override ``decode_response`` (reply -> ``(T, D)``
    chunk), ``observe`` (per-step state) and ``reset``.
    """

    pred_action_horizon: ClassVar[int | None] = None
    exec_horizon: ClassVar[int | None] = None
    default_image_size: ClassVar[tuple[int, int] | None] = None

    #: Width ``D`` of one action row, when the schema fixes it; chunks with another width are
    #: rejected before they are queued. ``None`` skips the check.
    action_dim: ClassVar[int | None] = None

    #: Optional row decoder (an :class:`~eden.extensions.policies.adapters.ActionAdapter`); the
    #: policy binds it to the env once the setup is built.
    adapter: ActionAdapter | None = None
    #: ``(T, D)`` copy of the next chunk *before* any setup post-processing (latching, ...),
    #: for setups that want the untouched server rows in their diagnostics. Handed to the
    #: scheduler with that chunk, then cleared, so it is never paired with a later one.
    last_raw_action_chunk: np.ndarray | None = None

    def __init__(self, policy: SetupClientMixin) -> None:
        self.policy = policy

    def reset(self) -> None:
        """Clear per-episode state (adapter latches, frame history, cached indices)."""
        self.last_raw_action_chunk = None
        if self.adapter is not None:
            self.adapter.reset()

    # -- env wiring ----------------------------------------------------

    @classmethod
    @abstractmethod
    def observation_options(cls, **kwargs) -> dict[str, ObservationGroupOptions]:
        """Observation group options to register with the Eden env."""

    @classmethod
    @abstractmethod
    def action_options(cls, **kwargs) -> dict[str, ActionTermOptions]:
        """Action term options to register with the Eden env."""

    # -- per-step ------------------------------------------------------

    def observe(self, obs: dict) -> None:
        """Called on every step before scheduling (default: nothing)."""

    def validate_server_config(self, config: dict) -> None:
        """Check the server's advertised configuration against this setup's schema.

        ``config`` is ``transport.server_config()`` — the handshake metadata (WebSocket) or
        the GR00T-family ``get_modality_config`` reply (ZMQ); ``{}`` when the server
        advertises nothing, in which case there is nothing to check. Raise ``ValueError``
        naming the expected and served values on a mismatch.
        """

    @abstractmethod
    def pack_observations(self, obs: dict) -> Any:
        """Convert Eden observation tensors to the server payload."""

    def decode_response(self, response: Any) -> np.ndarray | None:
        """Reply -> ``(T, D)`` chunk, or ``None`` to use the policy's default key lookup."""
        return None

    def set_raw_server_action(self, action: np.ndarray) -> None:
        """Receive the untouched server row matching the row about to be unpacked (default: ignored)."""

    def unpack_actions(self, raw_action: np.ndarray) -> dict[str, torch.Tensor]:
        """One ``(D,)`` action row -> ``{action_term: tensor}``; delegates to ``adapter`` by default."""
        if self.adapter is None:
            raise NotImplementedError(f"{type(self).__name__} has no adapter set and does not override unpack_actions")
        return self.adapter(raw_action, env=self.policy.env)


class SetupClientMixin(ClientMixin):
    """``ClientMixin`` whose serialization contract is delegated to a ``policy_setup``.

    Subclasses set ``SETUP_REGISTRY`` (name -> :class:`PolicySetupBase` subclass) and the
    default ``policy_setup``. ``observation_options`` / ``action_options`` take
    ``policy_setup=`` and forward the remaining kwargs to the setup.

    Parameters
    ----------
    policy_setup: str
        Key in ``SETUP_REGISTRY`` selecting the embodiment.
    image_size: tuple[int, int] | None
        Image size sent to the server; ``None`` takes the setup's training default.
    """

    SETUP_REGISTRY: ClassVar[dict[str, type[PolicySetupBase]]] = {}
    # The setup registry is a table of classes, not a configure() parameter (nor serialisable).
    _non_parameter_names: ClassVar[frozenset[str]] = frozenset({"SETUP_REGISTRY"})

    policy_setup: str = ""
    image_size: tuple[int, int] | None = None

    def _post_configure(self) -> None:
        self._setup: PolicySetupBase = self._setup_class(self.policy_setup)(self)
        if self._setup.adapter is not None:
            self._setup.adapter.bind(self.env)
        # Instance overrides win; otherwise take the setup's checkpoint defaults.
        if self.image_size is None:
            self.image_size = self._setup.default_image_size
        if self.pred_action_horizon is None:
            self.pred_action_horizon = self._setup.pred_action_horizon
        if self.exec_horizon is None:
            self.exec_horizon = self._setup.exec_horizon

    # -- env wiring ----------------------------------------------------

    @classmethod
    def _setup_class(cls, policy_setup: str) -> type[PolicySetupBase]:
        if policy_setup not in cls.SETUP_REGISTRY:
            raise ValueError(
                f"{cls.__name__}.policy_setup={policy_setup!r} is not a setup of this client. "
                f"Available: {sorted(cls.SETUP_REGISTRY)}"
            )
        return cls.SETUP_REGISTRY[policy_setup]

    @classmethod
    def observation_options(cls, policy_setup: str | None = None, **kwargs) -> dict[str, ObservationGroupOptions]:
        """Observation group options for ``policy_setup`` (default: the class default)."""
        return cls._setup_class(policy_setup or cls.policy_setup).observation_options(**kwargs)

    @classmethod
    def action_options(cls, policy_setup: str | None = None, **kwargs) -> dict[str, ActionTermOptions]:
        """Action term options for ``policy_setup`` (default: the class default)."""
        return cls._setup_class(policy_setup or cls.policy_setup).action_options(**kwargs)

    # -- delegation ----------------------------------------------------

    def _reset_local(self) -> None:
        self._setup.reset()

    def observe(self, obs: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> None:
        self._setup.observe(obs)

    def pack_observations(self, obs: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> Any:
        return self._setup.pack_observations(obs)

    def decode_response(self, response: Any) -> np.ndarray:
        chunk = self._setup.decode_response(response)
        if chunk is None:
            return super().decode_response(response)
        return self._normalize_action_shape(np.asarray(chunk, dtype=np.float32))

    def _post_connect(self) -> None:
        self._setup.validate_server_config(self._transport.server_config())

    def accept_chunk(self, chunk: np.ndarray) -> None:
        chunk = self._normalize_action_shape(np.asarray(chunk, dtype=np.float32))
        self._check_action_width(
            chunk, type(self._setup).action_dim, schema=f"{type(self).__name__} setup {self.policy_setup!r}"
        )
        raw, self._setup.last_raw_action_chunk = self._setup.last_raw_action_chunk, None
        raw_chunk = None if raw is None else self._normalize_action_shape(np.asarray(raw, dtype=np.float32))
        self._scheduler.accept(chunk, raw_chunk=raw_chunk)

    def pop_action(self) -> dict[str, torch.Tensor]:
        row = self._scheduler.pop()
        self._setup.set_raw_server_action(self._scheduler.last_raw_row[0])
        return self.unpack_actions(row)

    def unpack_actions(self, raw_action: np.ndarray) -> dict[str, torch.Tensor]:
        raw_action = np.asarray(raw_action)
        if raw_action.ndim == 2:
            if raw_action.shape[0] != 1:
                raise ValueError(
                    f"{type(self).__name__} setups decode one env at a time; got a batched row {raw_action.shape}."
                )
            raw_action = raw_action[0]
        return self._setup.unpack_actions(raw_action)
