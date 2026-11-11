"""Eden package entry point: per-rank GPU isolation for DDP and runtime initialization.

Importing :mod:`eden` performs per-rank GPU isolation **before** CUDA is initialized,
so Eden must be imported before ``genesis`` and ``torch`` (importing those first would
pin every rank to the same device). :func:`init` then configures the torch backends,
determinism, and Genesis logging.

``import eden`` imports neither ``genesis`` nor ``torch``: between them they would be ~2.8 s of
the import, paid by ``eden --help``, by ``eden list``, and by every script that only reads a
config. Both are imported by :func:`init`, which every process that builds a scene calls anyway.
The isolation runs at the top of this module, so any import order that starts with
``import eden`` is correct by construction -- but a module that wants ``genesis`` must import it
itself rather than relying on Eden having pulled it in.

Common entry points:

- :func:`init` — set up the runtime (TF32, determinism, per-rank logging).
- :func:`get_log_dir` (a.k.a. ``eden.log_dir``) — this run's log directory.
- ``eden.envs.base.RLEnvBase.from_config`` — build a vectorized environment from a config.
- :data:`eden.tasks.TASK_REGISTRY` — look up built-in task configs by name.
"""

import importlib
import logging as _stdlib_logging
import os
import secrets
import string
from typing import Any, Literal


def _isolate_gpu_for_ddp() -> None:
    """Restrict ``CUDA_VISIBLE_DEVICES`` to one GPU per DDP rank.

    Must execute before any import that may initialize CUDA.
    No-op when ``WORLD_SIZE <= 1`` (single-GPU).
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size <= 1:
        return
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [d.strip() for d in visible.split(",") if d.strip()]
        if local_rank >= len(devices):
            raise RuntimeError(
                f"CUDA_VISIBLE_DEVICES has {len(devices)} entries but LOCAL_RANK={local_rank} "
                f"(WORLD_SIZE={world_size}). Ensure CUDA_VISIBLE_DEVICES lists at least "
                f"WORLD_SIZE GPUs or adjust --nproc_per_node."
            )
        gpu = devices[local_rank]
    else:
        gpu = str(local_rank)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["QD_VISIBLE_DEVICE"] = gpu
    os.environ["TI_VISIBLE_DEVICE"] = gpu

    # Give each rank its own Quadrants kernel cache to avoid lock contention.
    # Without this, all ranks fight over the same ticache.lock file and crash.
    default_cache = os.path.join(
        os.path.expanduser("~"), ".cache", "quadrants", "qdcache", "kernel_compilation_manager"
    )
    base_cache = os.environ.get("QD_OFFLINE_CACHE_FILE_PATH", default_cache)
    os.environ["QD_OFFLINE_CACHE_FILE_PATH"] = os.path.join(base_cache, f"rank_{local_rank}")


_isolate_gpu_for_ddp()

from eden._logging import _GENESIS_THEMES, LOG_THEME_ENV_VAR, Logger, resolve_theme
from eden.constants import EventMode

# `genesis` and `torch` are imported by `init()`, not here (see the module docstring).
#
# The built-in term modules and the options namespaces are not imported here either: eagerly they
# would cost `import eden` ~2.4 s over `import genesis`, paid by `eden --help`, `eden list` and any
# script that only reads a config. They are re-exported lazily (`__getattr__` below), and each
# registry names the modules that fill it (`Registry._populate`), so a lookup still sees a complete
# registry -- the import moves to the first *use*, it does not become the caller's problem.
#
# `eden.<name>` and `from eden import <name>` both go through the module `__getattr__`, so this
# is invisible at every call site. But the modules are not imported at a fixed point, so anything
# relying on `import eden` alone to register an out-of-tree term through one of them must import
# it itself.
_LAZY_SUBMODULES = {
    # `en.distributed` is a re-export like the rest, and `eden.utils.distributed` imports
    # `torch.distributed` -- i.e. torch -- at module scope for `dist`. Every caller of it is
    # already inside a run.
    "distributed": "eden.utils.distributed",
    "actions": "eden.managers.terms.actions",
    "commands": "eden.managers.terms.commands",
    "curricula": "eden.managers.terms.curricula",
    "events": "eden.managers.terms.events",
    "metrics": "eden.managers.terms.metrics",
    "observations": "eden.managers.terms.observations",
    "recorders": "eden.managers.terms.recorders",
    "rewards": "eden.managers.terms.rewards",
    "terminations": "eden.managers.terms.terminations",
    "materials": "eden.options.materials",
    "objects": "eden.options.objects",
    "robots": "eden.options.robots",
    "scenes": "eden.options.scenes",
    "surfaces": "eden.options.surfaces",
}

# `eden._logging` is the submodule (underscore avoids shadowing stdlib `logging`);
# `eden.logger` is the Logger instance. It is seeded here so `en.logger` is
# usable before `init()` (library code may log from device threads or helpers
# that run pre-init); `init()` rebinds it with the configured level/format/theme.
logger = Logger(_stdlib_logging.INFO)

# This run's log directory, resolved by init() and created by get_log_dir() on first
# use. None until init() runs. Never bind a module global named `log_dir`: the lazy
# `eden.log_dir` alias goes through the module __getattr__ below, which PEP 562 only
# consults when normal attribute lookup fails.
_log_dir: str | None = None
# The two parts of the log-dir basename, kept as the authoritative source for run
# provenance (the run manifest) so nothing has to re-parse the basename — which would
# break the moment the basename gains a decoration (e.g. an optional run_name segment).
# None until init() runs; set on every rank (broadcast under DDP) so they always agree.
_run_timestamp: str | None = None
_run_id: str | None = None

from eden.utils.smoke import is_smoke  # noqa: F401


#: ``init(backend=...)``'s "you did not choose" value. A sentinel rather than ``gs.cpu``, which
#: would have to be evaluated when this module is imported and so would put the whole of
#: ``import genesis`` back on ``import eden``'s cold path for one default argument.
#: ``None`` is not a backend Genesis accepts, so nothing can be passing it deliberately.
_DEFAULT_BACKEND = None


def init(
    backend=_DEFAULT_BACKEND,
    initialize_genesis: bool = True,
    debug: bool = False,
    logging_level: int | None = None,
    logger_verbose_time: bool = False,
    log_root_path: str | None = None,
    run_name: str = "",
    allow_tf32: bool = True,
    deterministic: bool = False,
    performance_mode: bool = False,
    precision: Literal["32", "64"] = "32",
    eps: float = 1e-8,
    # Keyword-only, and last: inserted into the middle of this list it would silently re-bind
    # every caller that passes `log_root_path` and beyond positionally -- `eden.init(gs.gpu,
    # True, False, None, False, "./runs")` would hand "./runs" to `logger_theme` and raise on a
    # perfectly good log root. The `*` keeps the same trap from being set for whatever is added
    # next.
    *,
    logger_theme: Literal["auto", "human", "agent"] | None = None,
):
    """
    Initialize the Eden environment.

    Parameters
    ----------
    backend: genesis backend constant | None
        The backend device to use for the simulation (``gs.cpu``, ``gs.gpu``, ...). ``None`` --
        the default -- means ``gs.cpu``, resolved inside this call: spelling it in the signature
        would evaluate ``gs.cpu`` at import and drag ``import genesis`` back onto
        ``import eden``.
    initialize_genesis: bool
        Whether to initialize the Genesis here if not already initialized.
    debug: bool
        Whether to run in debug mode.
    logging_level: int | str | None
        The logging level to use.
    logger_verbose_time: bool
        Whether to log the time in the logger.
    log_root_path: str | None
        The root path to use for the logs.
        Defaults to ``./logs``. The run directory under it is *resolved* here but only
        created on the first :func:`get_log_dir` / ``eden.log_dir`` access.
    run_name: str
        Optional human-facing run name folded into the log-dir basename as
        ``<timestamp>-<run_name>-<id>`` (sanitized). Because rsl_rl derives the wandb run
        name from the log-dir basename, this names the run in the directory and in wandb;
        empty (the default) yields ``<timestamp>-<id>``.
    allow_tf32: bool
        Whether to use TF32 precision for faster computation on Ampere+ GPUs.
    deterministic: bool
        Whether to use deterministic algorithms for reproducibility.
    performance_mode: bool
        Whether to use performance mode for the simulation.
    precision: {'32', '64'}
        Floating-point precision passed to ``gs.init``. Default '32'.
        Some solvers (e.g. the SAP coupler) require '64' for numerical stability.
    eps: float
        Global numerical epsilon passed to ``gs.init``. Default 1e-8 (chosen for
        rigid-solver stability). Lower it (e.g. 1e-12) when a coupler uses tighter
        convergence tolerances than 1e-8 — the SAP coupler's default 1e-10 tolerances
        are corrupted by a coarser eps, which manifests as contacts exploding.
    logger_theme: {'auto', 'human', 'agent'} | None
        Keyword-only. How Eden (and Genesis) render their console output. ``'agent'`` — **the
        default** — drops every ANSI escape and the greeting box, for output that will be read
        out of a file, a CI log or a model's context window, which is where most of it goes.
        ``'human'`` is the colored, box-greeted terminal rendering, and is now something you ask
        for: pass it here, or set ``EDEN_LOG_THEME=human`` to get it for a whole process tree
        (exported empty counts as unset). ``'auto'`` is the middle setting — ``'human'`` on a
        TTY and ``'agent'`` otherwise, honoring ``NO_COLOR`` / ``FORCE_COLOR``.
        Passing this exports ``EDEN_LOG_THEME``, because the theme has consumers this call never
        touches: ``colorize``, the argparse help formatters, and every subprocess a run spawns.
        The *request* is exported rather than the resolution, so an explicit ``'auto'`` stays a
        per-process TTY decision downstream instead of an inherited one.
        This also sets Genesis' theme (``genesis.styles.style.theme``: ``THEME.raw`` for
        ``'agent'``, ``THEME.dark`` for ``'human'``) so the escapes are gone from the whole run's
        output rather than from Eden's share of it — including under
        ``initialize_genesis=False``. Assign ``genesis.styles.style.theme`` after this call to
        choose a different Genesis theme. Nothing sets it before this call — genesis is off
        ``import eden``'s path entirely — so a config ``repr``, which Genesis colorizes
        through it, prints plain until then under either theme. Eden's own rendering does not
        read it at all.
    """
    # Declared up front: the double-init warning below reads the module ``logger`` before the
    # rebind further down reassigns it, and Python forbids using a name ahead of its ``global``.
    global logger

    # Deferred so `import eden` costs neither -- this is the first point in a process
    # that genuinely needs them, and `_isolate_gpu_for_ddp()` ran at module import, long before.
    import genesis as gs
    import torch

    from eden.utils import distributed
    from eden.utils.torch import configure_torch_backends

    if backend is _DEFAULT_BACKEND:
        backend = gs.cpu

    configure_torch_backends(allow_tf32=allow_tf32, deterministic=deterministic)

    # Resolved before anything can log or print a colorized repr. Eden's theme drives Genesis'
    # palette too: "agent" means no escapes in the *run's* output, not just in Eden's share of it.
    if logger_theme is not None:
        # Exported, because the theme has consumers this call never touches: `colorize` and the
        # argparse help formatters read EDEN_LOG_THEME, as does every subprocess (a DDP worker,
        # a spawned viewer); otherwise `--help` and every colorized repr would stay plain. The
        # *request* goes in the variable, not the resolution, so an explicit "auto" stays a
        # per-process TTY decision downstream.
        os.environ[LOG_THEME_ENV_VAR] = logger_theme
    theme = resolve_theme(logger_theme)
    gs_theme = _GENESIS_THEMES[theme]

    # Suppress Genesis logging on non-main ranks to avoid noisy output.
    gs_log_level = logging_level
    if distributed.is_distributed() and not distributed.is_main_process():
        gs_log_level = "warning"

    if initialize_genesis:
        if not gs._initialized:
            gs.init(
                backend=backend,
                precision=precision,
                theme=gs_theme,
                logging_level=gs_log_level,
                # NOTE: this is for the numerical stability of the simulation
                eps=eps,
                # A placeholder, not the run's seed: gs.init() seeds numpy/torch, and a
                # constant here alone would make an unseeded training run come out *identical*
                # for every configured seed. `eden.utils.random.apply_run_seed` overrides it from runner_options.seed
                # before the env is built. Varies by *local* rank so a DDP run that never
                # reaches that call still differs between ranks on a node (two nodes' local
                # rank 0 share it — another reason not to treat this as the run's seed).
                seed=int(os.environ.get("LOCAL_RANK", 0)),
                performance_mode=performance_mode,
                use_deterministic_algorithms=deterministic,
            )
        else:
            # Loud, not a quiet info: re-initializing Eden/Genesis in one already-initialized
            # process is a common cause of the opaque CUDA "unknown error" that costs training
            # attempts. Genesis is not re-initialized (the guard above holds), but
            # the rest of init() below still re-runs — NCCL setup and, notably, the run's log
            # directory is re-resolved to a new timestamp, orphaning the earlier one. Warn so the
            # double-init is visible; the fix is to call eden.init() exactly once per process, or
            # pass initialize_genesis=False when a second init() is deliberate (e.g. reconfiguring
            # logging or the log directory).
            logger.warning(
                "eden.init() called again in a process where Genesis is already initialized. "
                "Genesis will not be re-initialized, but re-running init() in one process is a "
                "common source of an opaque CUDA 'unknown error' and it re-resolves the log "
                "directory. Call eden.init() exactly once per process (one process per run)."
            )
    # Also for the paths the gs.init() above did not take -- Genesis already initialized, or
    # initialize_genesis=False. Unconditional, so one rule holds everywhere: after eden.init()
    # Genesis' theme is the one Eden's asked for. A caller who wants Genesis' "light" assigns
    # `genesis.styles.style.theme` afterwards.
    from genesis.styles import THEME, style

    style.theme = THEME[gs_theme]

    # Up to https://github.com/Genesis-Embodied-AI/Genesis/pull/3431, `gs.init()` also set torch's default device
    # and dtype. Eden does not restore the device: torch implements it as a Python TorchFunctionMode that every torch
    # call is dispatched through, 1.7-2.0 ms per velocity-tracking control step and 4 ms on BeyondMimic, so a tensor
    # built without `device=` is a CPU tensor, as in plain torch. The dtype is a C-level setting with no per-call
    # cost, and it is what makes a dtype-less `torch.zeros(n)` match Genesis' own tensors under `precision="64"` --
    # where a float32 buffer would silently downcast an assignment and make an indexed write or a matmul against a
    # Genesis tensor raise -- so Eden sets it on every path that leaves Genesis initialized.
    if gs._initialized:
        torch.set_default_dtype(gs.tc_float)

    # ── Initialize NCCL right after Genesis, before scene building ──
    #   gs.init() → torch.cuda.set_device(0) → dist.init_process_group("nccl")
    # Doing it here (before any scene.build) ensures the CUDA context is
    # in a clean state for NCCL communicator creation. Device 0 is this rank's GPU:
    # `_isolate_gpu_for_ddp` narrowed CUDA_VISIBLE_DEVICES to it.
    if distributed.is_distributed():
        torch.cuda.set_device(0)
        distributed.setup("nccl")

    if logging_level is None:
        logging_level = _stdlib_logging.DEBUG if debug else _stdlib_logging.INFO
    logger = Logger(logging_level, logger_verbose_time, theme)
    # Apply Eden's level to the third-party stdlib "genesis" logger here (runtime setup), not in
    # Logger.__init__ — a bare `import eden` must not mutate third-party logging state. This also
    # covers a Genesis initialized before eden.init(). Genesis' logger stops propagating to the root
    # logger on its own since https://github.com/Genesis-Embodied-AI/Genesis/pull/3427.
    _stdlib_logging.getLogger("genesis").setLevel(logging_level)

    # Only show greeting and version info on the main process.
    if distributed.is_main_process():
        from eden.utils.misc import _display_greeting, format_provenance, get_eden_provenance, get_package_provenance

        # The box is three lines of terminal decoration, and off a TTY it cannot even be sized
        # (os.get_terminal_size raises, and the 80-column fallback is the piped case) — so the
        # agent theme skips it rather than rendering it blind. The version line below stays:
        # it is provenance, not decoration.
        if theme != "agent":
            _display_greeting(logger.INFO_length)

        eden_version = format_provenance(get_eden_provenance())
        genesis_version = format_provenance(get_package_provenance("genesis-world"))
        logger.info(f"🍎 Eden ver: ~~<{eden_version}>~~, 🌏 Genesis ver: ~~<{genesis_version}>~~.")

    if distributed.is_distributed():
        logger.info(
            f"🚀 DDP enabled: rank {distributed.get_global_rank()}/{distributed.get_world_size()}"
            f" (local_rank={distributed.get_local_rank()})"
        )

    # NOTE: resolve the log directory (leaf name ``<timestamp>-<run id>``; see
    # _new_run_dirname for why the random id, not the timestamp, carries uniqueness).
    # Under DDP, rank 0 generates the name and broadcasts it so every rank agrees on one
    # path. The directory is *not* created here; get_log_dir() creates it on first use.
    global _log_dir, _run_timestamp, _run_id
    base_log_dir = log_root_path or "./logs"
    if distributed.is_distributed():
        # _new_run_dirname() also sets _run_timestamp/_run_id (left-to-right eval), so we
        # broadcast all three together — rank 0's parts, not each rank re-deriving them.
        payload = (
            [_new_run_dirname(run_name), _run_timestamp, _run_id]
            if distributed.is_main_process()
            else [None, None, None]
        )
        torch.distributed.broadcast_object_list(payload, src=0)
        run_dirname, _run_timestamp, _run_id = payload
    else:
        run_dirname = _new_run_dirname(run_name)
    _log_dir = os.path.abspath(os.path.join(base_log_dir, run_dirname))
    logger.info(f"💽 Log directory: ~~<{_log_dir}>~~ (created on first use)")


# Run ids are 8-char base-36 (lowercase letters + digits) strings, as wandb uses for its
# run ids. Kept path/URL/shell-safe by construction (no separators or special chars).
_RUN_ID_ALPHABET = string.ascii_lowercase + string.digits
_RUN_ID_LENGTH = 8


def _generate_run_id() -> str:
    """Return a random 8-character base-36 run id (lowercase letters + digits).

    Uniqueness of a run directory rests on this id, not on the per-second ``get_now()``
    timestamp, so two runs launched in the same second still resolve to distinct
    directories with no filesystem check and no inter-process coordination — closing the
    same-second / concurrent-launch race that a timestamp-only name cannot. There are
    ``36**8 ≈ 2.8e12`` ids, so even a few thousand same-second launches have a negligible
    birthday-collision probability. Drawn from :mod:`secrets` (a CSPRNG) so ids are
    independent of any seeded global RNG and of the wall clock.

    Returns
    -------
    str
        An 8-character base-36 string.
    """
    return "".join(secrets.choice(_RUN_ID_ALPHABET) for _ in range(_RUN_ID_LENGTH))


def _new_run_dirname(run_name: str = "") -> str:
    """Return this run's log-directory leaf name, ``<timestamp>[-<run_name>]-<id>``.

    The ``get_now()`` timestamp is a human-readable, chronologically-sortable prefix (so
    "newest run under ``logs/``" lookups keep working), an optional sanitized ``run_name``
    is folded in for human/wandb legibility, and the :func:`_generate_run_id` suffix is
    what actually makes the name unique. See :func:`_generate_run_id` for why this is
    collision-safe across same-second and concurrent launches without the
    existence-check-and-retry dance a timestamp-only name would require. The id stays the
    last ``-``-delimited segment, and ``_run_timestamp`` / ``_run_id`` are recorded
    separately, so a decorated name never has to be re-parsed to recover either part.

    Parameters
    ----------
    run_name : str
        Optional human-facing run name; sanitized and folded in as the middle segment.
        Empty (the default) yields ``<timestamp>-<id>``.

    Returns
    -------
    str
        The ``<timestamp>[-<run_name>]-<id>`` directory basename for this run.
    """
    global _run_timestamp, _run_id
    # Deferred with the rest: `eden.utils.misc` imports numpy and torch at module scope,
    # and this is the first thing on the `init()` path that reaches it.
    from eden.utils.misc import get_now, sanitize_run_name

    _run_timestamp = get_now()
    _run_id = _generate_run_id()
    safe_name = sanitize_run_name(run_name)
    middle = f"-{safe_name}" if safe_name else ""
    return f"{_run_timestamp}{middle}-{_run_id}"


def get_run_id() -> str | None:
    """Return this run's base-36 id (the uniqueness primitive of the log-dir basename).

    ``None`` before :func:`init` has resolved a run. Provenance metadata only — a caller
    that needs a directory should use :func:`get_log_dir` / ``eden.log_dir``.
    """
    return _run_id


def get_run_timestamp() -> str | None:
    """Return this run's creation timestamp (the ``get_now()`` prefix of the basename).

    ``None`` before :func:`init` has resolved a run.
    """
    return _run_timestamp


def get_log_dir() -> str:
    """Return this run's log directory, creating it if it does not exist yet.

    ``eden.log_dir`` is an alias for this call, so ``f"{en.log_dir}/video.mp4"`` and
    ``os.path.join(en.log_dir, "config.json")`` keep working unchanged.

    Creation is deferred to the first access rather than done in :func:`init` so that a
    process which never logs anything — an evaluation, export, or viewer script, all of
    which must still call :func:`init` before they can build a scene — leaves no empty
    ``logs/<timestamp>/`` directory behind. That matters beyond tidiness: the usual way
    to find a trained run is to take the newest directory under ``logs/``, and an
    eagerly-created directory makes that resolve to the current process's own empty one.

    Returns
    -------
    str
        Absolute path to the log directory.

    Raises
    ------
    RuntimeError
        If :func:`init` has not been called yet, so no log directory is resolved.
    """
    if _log_dir is None:
        raise RuntimeError("Eden is not initialized: call `eden.init()` before using the log directory.")
    # Unconditional (not main-rank-only): with lazy creation any rank may be the first to
    # write, and makedirs(exist_ok=True) is safe to race.
    os.makedirs(_log_dir, exist_ok=True)
    return _log_dir


def __getattr__(name: str) -> Any:
    """Resolve ``eden.log_dir``, ``eden.__version__`` (``eden-world``'s installed version, ``""`` when unknown) and the
    term/options namespaces lazily (PEP 562).

    Anything else is a normal ``AttributeError``: this resolves the names :data:`__all__`
    promises, not an arbitrary ``eden.<submodule>``, so a typo still fails where it is written
    rather than importing something surprising.
    """
    if name == "log_dir":
        return get_log_dir()
    if name == "__version__":
        # Read on access, not at import: importlib.metadata walks sys.path.
        from eden.utils.distribution import get_eden_version

        return get_eden_version()
    module_path = _LAZY_SUBMODULES.get(name)
    if module_path is not None:
        module = importlib.import_module(module_path)
        # Bind it, so this runs once per name and `eden.rewards is eden.rewards` holds.
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include the lazily-resolved names, which ``globals()`` alone does not yet hold."""
    return sorted({*globals(), *_LAZY_SUBMODULES, "log_dir", "__version__"})


__all__ = [
    "init",
    "get_log_dir",
    "get_run_id",
    "get_run_timestamp",
    "distributed",
    "EventMode",
    "robots",
    "scenes",
    "actions",
    "commands",
    "curricula",
    "events",
    "metrics",
    "observations",
    "recorders",
    "rewards",
    "terminations",
    "objects",
    "materials",
    "surfaces",
]
