"""Torch helpers: backend configuration, model compilation, checkpoint inspection, config values, numpy<->torch."""

# Adapted from: https://github.com/mujocolab/mjlab/blob/main/src/mjlab/utils/torch.py
import torch
import functools
import threading
import warnings
from typing import Annotated, Any, Callable, TypeVar
from packaging.version import parse

import numpy as np
from pydantic import StringConstraints

T = TypeVar("T", bound=torch.nn.Module | Callable)


# Pydantic-validated device string. Accepts the ``"auto"`` sentinel (resolved
# lazily by ``resolve_device``), the unindexed forms ``"cpu" / "mps" / "cuda"``,
# and any indexed accelerator form ``"mps:N" / "cuda:N"``. Rejects typos like
# ``"cdua:0"`` or aliases like ``"gpu"`` at config-load time so the failure is
# a clear ValidationError instead of a confusing torch error several frames in.
DeviceStr = Annotated[
    str,
    StringConstraints(pattern=r"^(auto|cpu|(mps|cuda)(:\d+)?)$"),
]


def resolve_device(device: DeviceStr) -> str:
    """Resolve the sentinel ``"auto"`` to the active Genesis device.

    Allows runner config defaults (e.g. ``RslRlBaseRunnerOptions.device``) to
    avoid hardcoding ``"cuda:0"`` — which crashes on CPU-only Macs and
    Windows boxes without CUDA. Any non-``"auto"`` string is returned
    unchanged so explicit user choices still win.
    """
    if device != "auto":
        return device
    import genesis as gs

    if gs.device is None:
        raise RuntimeError(
            "runner device set to 'auto' but Genesis is not initialised yet — "
            "call `eden.init(...)` (which runs `gs.init`) before constructing the runner."
        )
    return str(gs.device)


def configure_torch_backends(allow_tf32: bool = True, deterministic: bool = False):
    """Configure PyTorch CUDA and cuDNN backends for performance/reproducibility.

    Parameters
    ----------
    allow_tf32: bool
        If True, use TF32 precision for faster computation on Ampere+ GPUs. If
        False, use standard IEEE FP32 precision.
    deterministic: bool
        Currently ignored here: :func:`eden.init` enables deterministic algorithms through
        ``gs.init(use_deterministic_algorithms=...)`` instead, and no cuDNN setting is touched.

    Notes
    -----
    TF32 uses reduced precision (10-bit mantissa vs 23-bit for FP32) for internal
    matrix multiplications providing a speedup with minimal impact on accuracy.

    See https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere for details.
    """
    torch_version = parse(torch.__version__.split("+")[0])  # Handle e.g., "2.9.0+cu118".
    if torch_version >= parse("2.9.0"):
        _configure_29(allow_tf32)
    else:
        _configure_pre29(allow_tf32)


def _configure_29(allow_tf32: bool):
    """Configure PyTorch CUDA and cuDNN backends for PyTorch 2.9+.

    The matmul precision goes through ``torch.set_float32_matmul_precision``, not the new
    ``torch.backends.cuda.matmul.fp32_precision`` alone: a matmul precision set only through the new API
    makes ``torch.get_float32_matmul_precision()`` raise ("mix of the legacy and new APIs"; seen on torch
    2.10 and 2.14), and Inductor reads the legacy precision when it lowers a matmul, so a ``torch.compile``
    of a network would die on its first call after ``eden.init()``. The legacy setter also sets
    ``cuda.matmul.fp32_precision`` (``high`` -> ``tf32``, ``highest`` -> ``ieee``), so both getters agree.

    cuDNN takes both APIs, legacy first, for the same reason and one more. The new
    ``cudnn.fp32_precision`` alone leaves the legacy ``cudnn.allow_tf32`` at its default ``True``, so with
    ``allow_tf32=False`` its getter raises on torch 2.14 (Inductor reads it when it lowers a convolution). On
    torch 2.10 it does not reach ``cudnn.conv`` / ``cudnn.rnn`` at all, so their convolutions would stay TF32. The
    legacy setter clears the per-op precisions, after which the new one sets all three.
    """
    precision = "tf32" if allow_tf32 else "ieee"
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.fp32_precision = precision  # type: ignore


def _configure_pre29(allow_tf32: bool):
    """Configure PyTorch CUDA and cuDNN backends for PyTorch <2.9."""
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32


def compile_model(model: T, mode: str = "reduce-overhead") -> T:
    """Safely compile a model or function if ``torch.compile`` is available.

    Worth it only for a function whose eager cost exceeds what a compiled call costs by itself: ~25 us of
    Dynamo guard evaluation and CUDA-graph replay per call (L40S / RTX 3080, measured at 4096 envs).
    A one- to five-op reward kernel runs in 6-22 us eager and in ~25 us compiled, which is why the generic
    reward helpers in ``eden.managers.terms.rewards`` are not compiled; a fused helper with a dozen ops (the
    getup / handstand terms, 42 -> 31 us) is.

    Parameters
    ----------
    model : T
        The ``torch.nn.Module`` or function to compile.
    mode : str, optional
        The compilation mode. ``"reduce-overhead"`` is recommended for RL inference.

    Returns
    -------
    T
        The compiled model/function, or the original if compilation fails/is unavailable. A
        ``torch.nn.Module`` comes back as a ``Module``; a function comes back as a function whose
        ``inspect.signature`` and ``__wrapped__`` are the original's.
    """
    if not hasattr(torch, "compile"):
        return model

    if isinstance(model, torch.nn.Module):
        # Eagerly, and deliberately. Deferring means returning a closure, which for a Module
        # would drop ``.parameters()``, ``.to()``, ``.state_dict()``, ``.eval()`` and
        # ``isinstance(x, nn.Module)`` on the way out -- silently, since the caller holds
        # something that is still callable. ``torch.compile`` on a Module returns an
        # ``OptimizedModule`` (still an ``nn.Module``), and *building* it is the part that
        # imports ``torch._dynamo``, so there is nothing to defer here without giving up the
        # type. No in-tree caller takes this branch -- every ``@compile_model`` target is a
        # function -- so ``import eden`` pays nothing for it.
        try:
            return torch.compile(model, mode=mode)
        except Exception as e:  # noqa: BLE001 — degrade to eager on any failure, as the lazy path does
            warnings.warn(f"torch.compile failed, using eager mode. Error: {e}")
            return model

    # ``torch.compile`` does not trace here -- it builds a wrapper and traces on first call -- but
    # building that wrapper imports ``torch._dynamo``, ~170 MB and ~0.5 s, and as a decorator it would
    # run on ``import eden``, for every run whether or not the function is ever called. Defer it to
    # the first call, which is where the tracing happens anyway, so nothing is paid twice.
    compiled: Callable | None = None
    lock = threading.Lock()

    @functools.wraps(model)
    def _lazily_compiled(*args, **kwargs):
        nonlocal compiled
        if compiled is None:
            with lock:
                if compiled is None:  # another thread may have won the race while we waited
                    try:
                        compiled = torch.compile(model, mode=mode)
                    except Exception as e:
                        warnings.warn(f"torch.compile failed, using eager mode. Error: {e}")
                        compiled = model
        return compiled(*args, **kwargs)

    return _lazily_compiled


def numpy_to_tensor(
    array: Any, *, dtype: torch.dtype | None = None, device: torch.device | str | None = None
) -> torch.Tensor:
    """A tensor of a numpy array (or a tensor), sharing its memory where it can.

    A tensor passes through. An array goes through ``np.ascontiguousarray`` first, because
    ``torch.from_numpy`` refuses negative strides (a flipped image, ``x[::-1]``). Then ``.to(dtype,
    device)`` when either is given, which copies only when it converts: a contiguous array already in
    ``dtype`` on the CPU stays shared with the tensor.
    """
    tensor = array if isinstance(array, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(array))
    if dtype is None and device is None:
        return tensor
    return tensor.to(device=device, dtype=dtype)


def to_numpy(value: Any, dtype: np.dtype | type | None = None) -> np.ndarray:
    """A numpy array of ``value``: a tensor is detached and moved to the host, anything else goes via ``np.asarray``.

    ``dtype`` converts (``np.asarray(..., dtype=dtype)``), copying only when the dtype differs — so a CPU
    tensor already in ``dtype`` comes back as a view of its storage. Copy the result if it must outlive
    writes to the tensor.
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


@functools.lru_cache(maxsize=512)
def _device_constant(values: tuple, device: str, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype, device=device)


def device_constant(values, device: torch.device | str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``torch.tensor(values, dtype=dtype, device=device)``, built once per distinct ``values`` and reused.

    For constants a function term builds from its (host) parameters on every call: each build is a
    blocking host->device copy, a GPU->CPU sync on the step path. ``values`` is a number or a
    (nested) sequence of numbers — whatever ``torch.tensor`` takes. The tensor is **shared** between
    callers, so treat it as read-only: derive from it (``.unsqueeze``, arithmetic), never write into it.
    """

    def freeze(v):
        return (
            tuple(freeze(x) for x in v) if isinstance(v, (list, tuple)) else float(v) if dtype.is_floating_point else v
        )

    return _device_constant(freeze(values), str(device), dtype)


def to_numpy_unbatched(t: torch.Tensor | np.ndarray) -> np.ndarray:
    """Move a per-env tensor to CPU numpy and drop the leading ``num_envs`` axis.

    A ``(1, ...)`` input becomes ``(...)``; a 1-D input is returned unchanged. A leading
    axis larger than one raises, since silently taking row 0 would hide a batching error
    (single-env consumers such as remote policy servers).
    """
    arr = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    if arr.ndim >= 2:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected a single-env tensor with leading axis 1, got shape {arr.shape}.")
        arr = arr[0]
    return arr


def broadcast_to_envs(t: torch.Tensor, num_envs: int) -> torch.Tensor:
    """``(D,)`` -> ``(num_envs, D)`` expanded view (no copy)."""
    return t.unsqueeze(0).expand(num_envs, -1)


def resolve_scalar_or_vector(
    value: float | list[float] | tuple[float, ...],
    expected_dim: int,
    device: torch.device | str,
    *,
    what: str,
    dtype: torch.dtype = torch.float32,
) -> float | torch.Tensor:
    """Resolve a scalar or fixed-length vector into its runtime form at ``build()`` time.

    A scalar stays a Python float. A sequence is length-checked against ``expected_dim`` and becomes
    a 1-D tensor on ``device``.

    Parameters
    ----------
    value: float or list[float] or tuple[float, ...]
        The configured value.
    expected_dim: int
        Required vector length.
    device: torch.device or str
        Device for the resolved tensor.
    what: str
        Subject of the error message, naming the offending term (e.g. ``"Observation term 'x' in
        group 'policy'"``).
    dtype: torch.dtype, optional
        Dtype of the resolved tensor; defaults to ``torch.float32``.

    Returns
    -------
    float or torch.Tensor
        The scalar itself, or a 1-D tensor of length ``expected_dim``.

    Raises
    ------
    TypeError
        If ``value`` is neither a real scalar nor a sequence.
    ValueError
        If ``value`` is a vector whose length is not ``expected_dim``.
    """
    # Class-term extras reach ``build()`` without annotation validation. Exclude bool because it is
    # numerically scalar in Python but is not a meaningful scale or gain.
    if isinstance(value, bool) or not isinstance(value, (int, float, list, tuple)):
        raise TypeError(f"{what} must be a scalar or a sequence of {expected_dim} values; got {type(value).__name__}")
    if isinstance(value, (int, float)):
        return float(value)
    if len(value) != expected_dim:
        raise ValueError(f"{what} must be a scalar or contain {expected_dim} values; got {len(value)}")
    return torch.tensor(value, dtype=dtype, device=device)


def read_checkpoint_index(path: str) -> dict[str, Any] | None:
    """The checkpoint's top-level mapping, with tensor data left on disk.

    For callers that read key *names* only — the rsl_rl legacy-format probe and key
    audit in :mod:`eden.envs.wrappers.rsl_rl_env`, the CLI's checkpoint framework
    detection — a full ``torch.load`` deserializes every weight just to throw it
    away. ``torch.load(mmap=True)`` maps the storages lazily instead, so an
    inspection costs a directory read rather than a copy of the weights, and the
    result can be dropped without ever having held them in RAM. The mapping's
    tensor values are still usable (paged in on first touch) — but a caller that
    needs values eagerly, e.g. to rewrite them into a new file, should do its own
    full ``torch.load`` so its output never depends on a live mapping of the input.

    Parameters
    ----------
    path : str
        Path to the checkpoint file.

    Returns
    -------
    dict | None
        The mapping, or ``None`` if the file does not deserialize to one. Raises whatever
        ``torch.load`` raises for an unreadable file — each caller decides what that means,
        and one of them (the rsl_rl legacy converter) must stay fail-loud about a missing
        checkpoint.
    """
    try:
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError:
        # Written before torch 1.6's zipfile format, which cannot be mapped. Rare, and
        # such checkpoints predate the weights that make a full read expensive.
        state = torch.load(path, map_location="cpu", weights_only=False)
    return state if isinstance(state, dict) else None
