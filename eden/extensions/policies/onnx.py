"""Shared ONNX-runtime infrastructure for policy extensions.

Provides the session layer (provider selection, thread policy, CPU fallback) and a generic
``torch.nn.Module`` wrapper so policies don't each re-implement session
creation + inference. Single-input / single-output graphs use :class:`OnnxModule`
directly (see the sonic / holosoma ports); multi-input or IO-binding graphs
subclass :class:`_OnnxWrapperBase` (see omnixtreme).
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Any

import numpy as np
import torch
from torch.utils.dlpack import to_dlpack

from eden.utils.distribution import install_hint
from eden.utils.torch import to_numpy

# onnxruntime is bound on first session build, not at import. Importing it here would
# make merely *importing* this module — or anything that reaches it, e.g. the sonic
# policy package — fail outright when the optional wheel is missing or broken, which
# takes down the graph-patching and verifier logic below that needs no runtime at all
# (their tests mock the session). Fail at use instead, with the same install hint.
ort: Any = None

# Shared by every "no usable onnxruntime" path below: absent wheel and broken wheel alike.
_ORT_INSTALL_HINT = (
    f"Install the GPU wheel with {install_hint('onnx-gpu')}; or the CPU wheel with {install_hint('onnx')}. "
    "The two wheels share the `onnxruntime/` directory, and each command puts its own files on top of the other's. "
    "Don't uninstall either wheel: that deletes files the other one needs. The GPU wheel needs a matching "
    "CUDA/cuDNN. See https://onnxruntime.ai/getting-started"
)


def _load_ort() -> Any:
    """Bind the module-global ``ort`` on first use.

    Every later ``ort.`` reference is on a path that can only be reached once a session
    exists, so this runs off the hot path.
    """
    global ort
    if ort is None:
        try:
            import onnxruntime
        except ImportError as exc:
            raise ImportError(f"onnxruntime is not installed. {_ORT_INSTALL_HINT}") from exc
        # Both wheels own the same `onnxruntime/` tree, so installing one over the other and
        # then uninstalling either deletes their shared files — `__init__.py` included —
        # leaving a directory that still *imports*, as an empty namespace package. Without
        # this check the absence surfaces only at the first attribute access, as a bare
        # `AttributeError: module 'onnxruntime' has no attribute 'SessionOptions'` deep in
        # session setup, which names neither the real cause nor the fix.
        if not hasattr(onnxruntime, "InferenceSession"):
            where = getattr(onnxruntime, "__file__", None) or list(getattr(onnxruntime, "__path__", []))
            raise ImportError(
                f"onnxruntime imported from {where} but exposes no InferenceSession — the install is "
                "broken, typically by uninstalling one of the two wheels, which deletes files both share. "
                f"Reinstalling the wheel you want puts them back. {_ORT_INSTALL_HINT}"
            )
        ort = onnxruntime
    return ort


def _env_bool(name: str, default: str = "0") -> bool:
    """Read an environment variable as a boolean flag."""
    return os.environ.get(name, default).lower() not in ("0", "false", "no", "")


def _build_providers() -> list:
    """Build ONNX Runtime execution providers list respecting env vars."""
    if _env_bool("ONNX_CPU"):
        return ["CPUExecutionProvider"]

    _load_ort()

    want_trt = _env_bool("TENSORRT_EP") or _env_bool("ONNX_TRT") or _env_bool("ORT_TRT")
    avail = set(getattr(ort, "get_available_providers", lambda: [])())
    providers: list = []
    if torch.cuda.is_available():
        if want_trt and "TensorrtExecutionProvider" in avail:
            trt_opts = {
                "trt_fp16_enable": _env_bool("ORT_TRT_FP16", "1"),
                "trt_int8_enable": _env_bool("ORT_TRT_INT8"),
                "trt_engine_cache_enable": _env_bool("ORT_TRT_CACHE", "1"),
                "trt_engine_cache_path": os.environ.get("ORT_TRT_CACHE_PATH", ".ort_trt_cache"),
            }
            ws = os.environ.get("ORT_TRT_WORKSPACE")
            if ws is not None:
                try:
                    trt_opts["trt_max_workspace_size"] = int(ws)
                except Exception:
                    pass
            providers.append(("TensorrtExecutionProvider", trt_opts))
        if "CUDAExecutionProvider" in avail:
            providers.append(("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}))
    providers.append("CPUExecutionProvider")
    return providers


#: Environment variable overriding the per-session intra-op thread count (a positive integer).
INTRA_OP_THREADS_ENV = "ONNX_INTRA_OP_THREADS"
#: Intra-op threads per session unless :data:`INTRA_OP_THREADS_ENV` says otherwise.
DEFAULT_INTRA_OP_THREADS = 1


def _intra_op_threads() -> int:
    """Resolve the per-session intra-op thread count (see :func:`_session_options`)."""
    raw = os.environ.get(INTRA_OP_THREADS_ENV)
    if raw is None:
        return DEFAULT_INTRA_OP_THREADS
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise ValueError(
            f"{INTRA_OP_THREADS_ENV}={raw!r} is not a positive integer. It is the number of intra-op threads "
            f"each ONNX session may use; unset it for the default of {DEFAULT_INTRA_OP_THREADS}."
        )
    return value


def _session_options() -> "ort.SessionOptions":
    """Session options every Eden ONNX session is built with.

    ONNX Runtime's own default is ``intra_op_num_threads=0`` -- one thread per CPU core -- and
    those threads **spin** between ops, per session. Eden runs several sessions in one process
    (SONIC alone is an encoder and a decoder; ``eden deploy`` adds the Holosoma base) next to
    torch's pool and Genesis, so on a 24-core deploy box that was 72 busy-waiting threads
    competing with the control thread. Measured on the SONIC deploy loop: the policy step took a
    median 13-18 ms and 70-89% of 50 Hz ticks overran; with one non-spinning thread per session
    it took 3.7 ms and no tick overran. The graphs are small enough that one thread runs them in
    about 1 ms, so the pool was never buying anything at batch 1.

    The thread count is :data:`DEFAULT_INTRA_OP_THREADS` unless :data:`INTRA_OP_THREADS_ENV`
    overrides it (a large batched CPU rollout may want more); spinning stays off either way.
    """
    _load_ort()
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = _intra_op_threads()
    sess_opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return sess_opts


def _make_session(onnx_path: str, label: str = "ONNX") -> "ort.InferenceSession":
    """Create an ONNX Runtime session with automatic CPU fallback."""
    sess_opts = _session_options()
    providers = _build_providers()
    debug = _env_bool("ONNX_DEBUG")
    try:
        sess = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)
    except Exception as e:
        warnings.warn(f"[{label}] Session with providers={providers} failed: {e}; falling back to CPU.", stacklevel=2)
        sess = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])
        return sess  # the warning above already says why it is on CPU
    if debug:
        print(f"[{label}] Requested providers: {providers}; session providers: {sess.get_providers()}")
    # Diagnose the silent CPU fallback: torch sees CUDA but the session landed on
    # CPU (wheel is the CPU-only `onnxruntime`, or a CUDA/cuDNN mismatch made the
    # CUDA provider unavailable). Without this the only symptom is ~10-50x slower
    # inference. Fires once per session build, never changes behavior.
    if torch.cuda.is_available() and not _env_bool("ONNX_CPU"):
        gpu_eps = {"CUDAExecutionProvider", "TensorrtExecutionProvider"}
        if not (gpu_eps & set(sess.get_providers())):
            warnings.warn(
                f"[{label}] torch sees CUDA but the ONNX session is running on {sess.get_providers()} "
                "(CPU). Install onnxruntime-gpu matching your CUDA/cuDNN (e.g. "
                "`pip install --no-deps onnxruntime-gpu`) to run the policy on GPU.",
                stacklevel=2,
            )
    return sess


def _rebuild_session_cpu(onnx_path: str) -> "ort.InferenceSession":
    """Rebuild an ONNX session with CPU-only provider."""
    # Options first: the callee ``ort.InferenceSession`` is evaluated before its arguments, and
    # ``ort`` may still be the unbound sentinel until ``_session_options`` has run ``_load_ort``.
    sess_opts = _session_options()
    return ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])


def _run_with_cpu_fallback(
    obj,
    inputs: dict,
    *,
    session_attr: str = "session",
    path_attr: str = "_onnx_path",
    flag_attr: str = "_cpu_fallback",
    output_names: list[str] | None = None,
    label: str = "ONNX",
) -> list:
    """Run ONNX session with automatic CPU fallback on CUBLAS/CUDA errors.

    Generic helper that works for any object holding an ONNX session.
    """
    session = getattr(obj, session_attr)
    try:
        return session.run(output_names, inputs)
    except Exception as e:
        if not getattr(obj, flag_attr, False):
            warnings.warn(f"[{label}] CUDA failed ({type(e).__name__}), rebuilding with CPUExecutionProvider")
            new_session = _rebuild_session_cpu(getattr(obj, path_attr))
            setattr(obj, session_attr, new_session)
            setattr(obj, flag_attr, True)
            return new_session.run(output_names, inputs)
        raise


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr[None, :]
    return arr


class _OnnxWrapperBase(torch.nn.Module):
    """Common base for ONNX policy wrappers."""

    def __init__(self, onnx_path: str, device: str | None = None, label: str = "ONNX", dynamic_batch: bool = False):
        super().__init__()
        # Some released graphs are exported with a static batch of 1, which rejects
        # num_envs>1 feeds; patch to a dynamic leading axis first. The original
        # asset's md5 is verified by the policy before this point, so integrity is
        # unaffected (the patched graph is a cached derivative of the verified file).
        if dynamic_batch:
            onnx_path = dynamic_batch_onnx(onnx_path)
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"ONNX file does not exist: {onnx_path}")
        self._onnx_path = onnx_path
        self._label = label
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.session = _make_session(onnx_path, label=label)
        self._input_names: list[str] = [i.name for i in self.session.get_inputs()]
        self._output_names: list[str] = [o.name for o in self.session.get_outputs()]
        self.eval()

    def _run(self, inputs: dict, label: str | None = None, output_names: list[str] | None = None) -> torch.Tensor:
        """Run inference with CPU fallback, return the first requested output on self.device.

        ``output_names`` defaults to all graph outputs; pass a single name to fetch
        exactly that output (as :class:`OnnxModule` does).
        """
        valid = set(self._input_names)
        inputs = {k: v for k, v in inputs.items() if k in valid}
        names = self._output_names if output_names is None else output_names
        outputs = _run_with_cpu_fallback(self, inputs, output_names=names, label=label or self._label)
        return torch.from_numpy(outputs[0]).to(self.device).float()

    def to(self, device: Any):
        self.device = torch.device(device)
        return self

    def _can_iobind(self) -> bool:
        """True when GPU IO-binding is usable: a CUDA device, an active GPU
        provider, and not in a CPU-forced / CPU-fallback state.
        """
        if _env_bool("ONNX_CPU") or getattr(self, "_cpu_fallback", False):
            return False
        try:
            prov = set(self.session.get_providers())
        except Exception:
            prov = set()
        return self.device.type == "cuda" and bool(prov & {"CUDAExecutionProvider", "TensorrtExecutionProvider"})


def _nonbatch_numel(session: "ort.InferenceSession", name: str) -> int | None:
    """Flattened element count of the non-leading (feature) dims of a tensor, or
    None if any feature dim is dynamic. This is the per-row feature count used by
    the swapped-graph guard — correct for both a static-batch-1 graph and a
    dynamic-batch graph whose leading axis is symbolic but feature dims static.
    """
    shape = next((i.shape for i in [*session.get_inputs(), *session.get_outputs()] if i.name == name), None)
    if shape is None:
        return None
    tail = shape[1:]
    if not all(isinstance(d, int) for d in tail):
        return None
    return math.prod(tail) if tail else 1


def dynamic_batch_onnx(src: str) -> str:
    """Patch a static-batch-1 ONNX graph to accept an arbitrary leading batch.

    Rewrites every ``Reshape`` shape target whose leading dim is a constant ``1``
    to ``-1`` and marks each graph input/output's dim 0 as the symbolic
    ``"batch"``. The patched graph is cached next to ``src`` as
    ``*.dynbatch.onnx`` (falling back to a temp dir if ``src``'s directory is
    read-only) and the cached path is returned; an existing cache is reused.

    Needed because some released graphs (e.g. SONIC) are exported with a static
    batch of 1, which rejects ``num_envs > 1`` feeds outright. The patch is a
    heuristic, so the freshly built graph is checked by :func:`_verify_dynamic_batch`
    (accepts a batched feed + permutation-equivariant rows) before it is used — a
    corrupt patch fails loudly here instead of silently returning wrong rows.
    """
    # ``splitext`` (not ``str.replace``) so a path with ``.onnx`` in a parent dir
    # isn't corrupted, and an extensionless path can't collapse ``dst`` onto
    # ``src`` and overwrite the verified original.
    dst = os.path.splitext(src)[0] + ".dynbatch.onnx"

    # Reuse the cache only if it is at least as new as the source: a re-downloaded
    # / updated checkpoint at the same path (newer mtime) forces a rebuild, so we
    # never run a stale patch derived from older weights.
    def _cache_fresh(path: str) -> bool:
        return os.path.exists(path) and os.path.getmtime(path) >= os.path.getmtime(src)

    if _cache_fresh(dst):
        return dst
    import onnx
    from onnx import numpy_helper

    model = onnx.load(src)
    # Names of the shape tensors feeding every Reshape (constants + initializers).
    shape_inputs = {n.input[1] for n in model.graph.node if n.op_type == "Reshape"}

    def _lead_one_to_minus_one(t) -> bool:
        arr = numpy_helper.to_array(t).copy()
        # Only a leading literal 1 with no other inferred (-1) dim: a Reshape
        # target may carry at most one -1, so never introduce a second one
        # (transformer attention reshapes already use -1 for the flattened tail).
        if arr.ndim == 1 and arr.size and arr[0] == 1 and -1 not in arr.tolist():
            arr[0] = -1
            t.CopyFrom(numpy_helper.from_array(arr, t.name))
            return True
        return False

    for node in model.graph.node:
        if node.op_type == "Constant" and node.output[0] in shape_inputs:
            for attr in node.attribute:
                if attr.name == "value":
                    _lead_one_to_minus_one(attr.t)
    for init in model.graph.initializer:
        if init.name in shape_inputs:
            _lead_one_to_minus_one(init)
    for vi in list(model.graph.input) + list(model.graph.output):
        vi.type.tensor_type.shape.dim[0].dim_param = "batch"

    try:
        onnx.save(model, dst)
    except OSError:
        import hashlib
        import tempfile

        cache = os.path.join(tempfile.gettempdir(), "eden_onnx_dynbatch")
        os.makedirs(cache, exist_ok=True)
        dst = os.path.join(cache, hashlib.md5(src.encode()).hexdigest() + ".dynbatch.onnx")
        if not _cache_fresh(dst):
            onnx.save(model, dst)
    # Just-built graph: confirm the patch is numerically equivalent to looped
    # batch-1 before any policy runs it. Cache hits above skip this (already
    # verified when first built).
    _verify_dynamic_batch(dst)
    return dst


def _verify_dynamic_batch(path: str, n: int = 4, atol: float = 1e-4) -> None:
    """Assert the patched graph treats the leading axis as an independent batch.

    Two properties, both checked at a *fixed* batch size ``n`` so they hold exactly
    even for graphs whose quantized / rounded outputs are sensitive to the batch-
    size-dependent matmul reduction order (e.g. SONIC's VQ encoder, where batched
    vs looped-batch-1 legitimately differ by whole quantization bins — so an exact
    batched-vs-looped check would false-positive on a perfectly correct graph):

    1. the graph accepts an ``n``-row feed at all — a graph that bakes ``batch=1``
       too deeply (the un-batchable omnixtreme exports) raises in ORT here;
    2. permuting the input rows permutes every output row identically — a patch that
       folded the batch axis into a feature / sequence dim is not permutation-
       equivariant.

    A failure raises rather than letting a mis-batched graph silently corrupt a
    rollout. (A pure permutation-*invariant* coupling — e.g. a mean over the batch
    axis — is not detected, but deployed single-env policy exports don't contain
    one, and the real un-batchable graphs are caught by property 1.) Inputs with a
    non-static feature dim are skipped (no probe can be built).
    """
    sess = _rebuild_session_cpu(path)
    rng = np.random.RandomState(0)
    feeds: dict[str, np.ndarray] = {}
    for inp in sess.get_inputs():
        feat = inp.shape[1:]
        if not all(isinstance(d, int) for d in feat):
            return  # symbolic feature dim — cannot construct a probe input
        feeds[inp.name] = rng.standard_normal((n, *feat)).astype(np.float32)

    perm = np.roll(np.arange(n), 1)  # non-identity permutation of the rows
    try:
        out = sess.run(None, feeds)
        out_perm = sess.run(None, {name: x[perm] for name, x in feeds.items()})
    except Exception as e:
        raise RuntimeError(
            f"dynamic_batch_onnx: patched graph {path} rejects an {n}-row batched feed "
            f"({type(e).__name__}: {e}); it bakes batch=1 too deeply for a leading-axis "
            "rewrite — re-export it with a dynamic batch axis."
        ) from e
    for j, (o, o_perm) in enumerate(zip(out, out_perm)):
        o = np.asarray(o)
        o_perm = np.asarray(o_perm)
        # Only outputs whose leading axis IS the batch (length n) are row-permutable. An output
        # with a different leading axis is not batched — e.g. a fixed per-feature parameter like a
        # policy's ``std_param`` (shape ``(num_dofs,)``), or a scalar — so permuting the input rows
        # cannot permute it, and indexing it with the row permutation would raise a spurious
        # broadcast error. Skip those (a permutation-invariant coupling is out of scope, as noted
        # above); the batched outputs still enforce equivariance.
        if o.shape[:1] != (n,):
            continue
        ref = o[perm]
        if not np.allclose(o_perm, ref, atol=atol, rtol=atol):
            raise RuntimeError(
                f"dynamic_batch_onnx: patched graph {path} is not batch-permutation-equivariant "
                f"(output {j} max|Δ|={np.abs(o_perm - ref).max():.3e}); the patch mixed the batch "
                "axis into features/sequence — refusing to use it."
            )


class OnnxModule(_OnnxWrapperBase):
    """Generic single-input / single-output ONNX graph as a torch Module.

    Covers the common policy shape: one named float32 input (e.g. ``actor_obs``
    or ``obs_dict``) and one named float32 output (e.g. ``action``). ``forward``
    accepts a tensor or array, runs the graph with automatic CPU fallback, and
    returns a float32 tensor on ``self.device``.

    Parameters
    ----------
    onnx_path : str
        Path to the ONNX file.
    input_name, output_name : str
        Names of the graph's single input / output tensor (validated against
        the loaded graph).
    device : str | None
        Device the returned tensor is placed on.
    label : str
        Human-readable tag used in log / fallback messages.
    expected_input_dim, expected_output_dim : int | None
        When set and the corresponding tensor has a static shape, the flattened
        dim is validated and a ``ValueError`` is raised on mismatch — a guard
        against a swapped / re-exported graph.
    enable_iobinding : bool
        Use the GPU IO-binding fast path when the device is CUDA, a GPU provider is
        active and the output is float32 with static non-batch dims; otherwise the
        numpy path runs.
    dynamic_batch : bool
        Patch a static-batch-1 graph to a dynamic leading axis first
        (:func:`dynamic_batch_onnx`), so ``num_envs > 1`` feeds are accepted.
    """

    def __init__(
        self,
        onnx_path: str,
        input_name: str,
        output_name: str,
        *,
        device: str | None = None,
        label: str = "ONNX",
        expected_input_dim: int | None = None,
        expected_output_dim: int | None = None,
        enable_iobinding: bool = False,
        dynamic_batch: bool = False,
    ) -> None:
        super().__init__(onnx_path, device=device, label=label, dynamic_batch=dynamic_batch)
        if input_name not in self._input_names:
            raise ValueError(f"{label} at {onnx_path} has no input '{input_name}'; got {sorted(self._input_names)}")
        if output_name not in self._output_names:
            raise ValueError(f"{label} at {onnx_path} has no output '{output_name}'; got {sorted(self._output_names)}")
        self._input_name = input_name
        self._output_name = output_name
        # Per-row (feature) element count, used by the swapped-graph guard. Works
        # whether the leading axis is a literal 1 or the symbolic "batch" that
        # dynamic_batch leaves behind.
        self.input_dim = _nonbatch_numel(self.session, input_name)
        self.output_dim = _nonbatch_numel(self.session, output_name)
        self._validate_dim("input", self.input_dim, expected_input_dim)
        self._validate_dim("output", self.output_dim, expected_output_dim)

        # Per-row output shape (dims after the dynamic batch axis), used to
        # pre-allocate the GPU output for IO-binding. None if any non-batch dim
        # is dynamic — then IO-binding is disabled and we use the numpy path.
        out = next((o for o in self.session.get_outputs() if o.name == output_name), None)
        out_shape = out.shape if out is not None else None
        tail = list(out_shape[1:]) if out_shape else []
        self._out_tail = tuple(int(d) for d in tail) if tail and all(isinstance(d, int) for d in tail) else None
        # IO-binding pre-allocates a float32 output buffer, so it's only sound
        # when the graph output is float32 (matching the numpy path, which casts
        # to float32). A non-float32 output (e.g. float16 / int) falls back to the
        # numpy path rather than writing wrong-width bytes into a float32 buffer.
        out_is_float32 = out is not None and out.type == "tensor(float)"
        self._iobind = bool(enable_iobinding) and self._out_tail is not None and out_is_float32

    def _validate_dim(self, which: str, actual: int | None, expected: int | None) -> None:
        if expected is not None and actual is not None and actual != expected:
            raise ValueError(
                f"{self._label} at {self._onnx_path} expects {which} dim {expected}, but the graph has {actual}."
            )

    def _forward_iobind(self, obs: torch.Tensor) -> torch.Tensor:
        """GPU IO-binding fast path: bind torch CUDA tensors directly so the
        graph's input and output never touch host memory. Numerically identical
        to the numpy path (same graph, same provider — only the data location
        differs).
        """
        t = obs if isinstance(obs, torch.Tensor) else torch.as_tensor(obs)
        t = t.to(self.device, dtype=torch.float32)
        if t.ndim == 1:
            t = t[None, :]
        t = t.contiguous()
        io = self.session.io_binding()
        io.bind_ortvalue_input(self._input_name, ort.OrtValue.from_dlpack(to_dlpack(t)))
        out_t = torch.empty((t.shape[0], *self._out_tail), device=self.device, dtype=torch.float32)
        io.bind_ortvalue_output(self._output_name, ort.OrtValue.from_dlpack(to_dlpack(out_t)))
        # ORT runs on its own CUDA stream: without this fence it can read the
        # bound input while the torch kernels producing it are still in flight,
        # returning stale/garbage rows. Fires under load (large batches, a busy
        # sim stream) — observed as rare corrupted actions that diverge
        # otherwise-identical batched envs.
        torch.cuda.synchronize(self.device)
        self.session.run_with_iobinding(io)
        return out_t

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self._iobind and self._can_iobind():
            try:
                return self._forward_iobind(obs)
            except Exception as e:
                if _env_bool("ONNX_DEBUG"):
                    warnings.warn(f"[{self._label}] IO-binding failed ({type(e).__name__}: {e}); using numpy path")
        feed = {self._input_name: _ensure_2d(to_numpy(obs, np.float32))}
        return self._run(feed, output_names=[self._output_name])
