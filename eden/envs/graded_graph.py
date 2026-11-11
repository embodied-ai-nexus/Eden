"""CUDA-graph replay of the graded phase of ``RLEnvBase.step`` (``EnvOptions.cuda_graph_options``).

Between the physics step and the auto-reset, ``RLEnvBase.step`` grades the post-physics state: the metric
manager, the observation terms' ``post_physics_refresh`` tick, the termination manager and the reward
manager. That phase reads the simulation (through Genesis' zero-copy views, plain torch indexing) and writes
only the managers' own buffers, but it does so in hundreds of small kernels, and on a host-bound step their
launches are what it costs: ~3 ms of host time on BeyondMimic at 4096 envs, against ~1 ms of device time.
Replayed from a CUDA graph the same kernels launch in ~5 µs, and the checks before each replay (below)
take ~70 µs.

A graph replays kernels, not Python, so it is only right for code that keeps its state in tensors it writes
in place, and reads no value back to the host — which is the term contract (``docs/guides/terms.md``).
:class:`GradedPhaseGraph` does not take that on trust (its settings are
:class:`~eden.options.envs.CudaGraphOptions`):

* It captures after ``warmup_steps`` eager steps, once every term has built its lazy caches (building one
  inside a capture is a host-to-device copy, which a capture rejects).
* The capture step and the next ``validation_steps`` replays are each also run eagerly and compared bit for
  bit with the graph: every tensor the phase leaves behind (each one bound to an object the walk follows that
  it writes or rebinds, and each one it writes that no such binding reaches) and the phase's outputs. A
  mismatch raises, naming the tensors that differ. The two runs go in a different order on each kind of step:

  - On the capture step the eager run goes first (a capture records kernels without running them). It is
    undone so the capture starts from the same state. Every tensor that existed before it is put back,
    reached by a binding or not, so the capture's replay does not advance it a second time. One it allocated
    is not: it may already sit in a cache that outlives the phase (Genesis' contact query), and rolling it
    back would hand the cache's next reader stale values.
  - On a validated replay the graph goes first and is undone (the same tensors), and the eager run's result
    is the one kept. Run the other way round, work a capture recorded without being able to replay (a kernel
    launched outside torch's stream runs once, at capture time) would be redone by the eager run just before
    the replay read it, and agree. Graph-first, the replay reads the stale value and disagrees.
* Before every replay it checks what the graph baked in, and re-captures (and re-validates) when it changed:
  the graded terms' configured fields (a curriculum moving a termination threshold), the env's per-run
  constants a term may read as Python values (:data:`_ENV_CONSTANTS`), and every binding on the way to a
  tensor the capture read (a term or ``load_state_dict`` rebinding the tensor, or replacing the dict or
  helper that holds it, leaves the graph reading the old address). The fields a manager mirrors on the device
  (:attr:`~eden.managers.base.ManagerBase.device_synced_fields`, the reward weights) are the exception: the
  manager's ``sync_device_fields`` runs before every replay, so a weight curriculum needs no re-capture.
* ``env.common_step_counter`` changes on every step, so it cannot be watched: reading it during a capture
  raises :class:`BakedHostReadError`, naming the term's line. Any other Python value a term reads from outside
  its own configured fields (another manager's term, a module global) is not watched either, and the graph
  replays what it read at capture. Keep such a value in a tensor written in place.
* More than ``max_recaptures_per_window`` re-captures within ``recapture_window`` steps raise, naming what
  changed: re-capturing on nearly every step costs more than the eager phase it replaces. A one-off change,
  such as a curriculum switching stage, re-captures once.

The walk that finds those bindings follows every object reachable from the env through attributes and
containers, however deep, except three kinds: the env itself (every term's back-reference to it), the standard
library's objects, and the objects of :data:`_UNWALKED_PACKAGES` (the simulator and torch), which hold the
simulation's own state and per-step caches rather than a term's.
"""

from __future__ import annotations

import functools
import os
import re
import sys
import traceback
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from eden.utils.common import frozen_value
from eden.utils.genesis_runtime import clear_rigid_query_caches

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase
    from eden.managers.base import ManagerBase
    from eden.options.envs import CudaGraphOptions

#: The env attributes a graded term may read as Python values, which a capture bakes in as constants.
_ENV_CONSTANTS = ("dt", "max_episode_length", "training")

#: The top-level packages whose objects the binding walk never enters: the simulator, the kernel backend it runs
#: on, and torch. Their objects hold the simulation's state and its per-step query caches (which Genesis empties
#: every step), not a term's: a term reads the simulation through Eden's handles, which the walk follows. Matched
#: by the name the classes are defined under, so an editable checkout of one is skipped as well.
_UNWALKED_PACKAGES = frozenset({"genesis", "quadrants", "torch"})

#: The ``eden`` package directory, and where term code lives inside it (relative to it), for naming the line a
#: failed capture started at.
_EDEN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TERM_CODE = re.compile(r"(managers[/\\]terms|tasks|extensions)[/\\]")

#: Differing tensors named in a validation error before it summarizes the rest.
_SHOWN_DIFFERENCES = 8


class BakedHostReadError(RuntimeError):
    """A host value that changes on every step was read while the graded phase was being captured.

    Raised by ``RLEnvBase.common_step_counter`` during a capture: the graph would replay the count it read on
    every later step, so a cadence gated on it (``counter % n == 0``) would stay on the branch the capture took.
    """


class GradedPhaseGraph:
    """Run the graded phase from a CUDA graph, validated against eager runs of it.

    Parameters
    ----------
    env : RLEnvBase
        The built env whose graded phase this replays.
    compute : Callable[[], tuple[torch.Tensor, torch.Tensor]]
        The eager graded phase, returning ``(reset_buf, reward_buf)``: ``RLEnvBase._compute_graded``.
    options : CudaGraphOptions
        How to capture, validate and re-capture (``EnvOptions.cuda_graph_options``).
    """

    def __init__(
        self,
        env: RLEnvBase,
        compute: Callable[[], tuple[torch.Tensor, torch.Tensor]],
        options: CudaGraphOptions,
    ) -> None:
        if torch.device(env.device).type != "cuda":
            raise ValueError(
                f"EnvOptions.cuda_graph_options needs a CUDA device, but this env runs on '{env.device}'. Set "
                "cuda_graph_options=None, or run on an NVIDIA GPU with en.init(backend=gs.cuda); gs.gpu is not "
                "enough, as it picks Metal on a Mac."
            )
        self._env = env
        self._compute = compute
        self._options = options
        self._graph: torch.cuda.CUDAGraph | None = None
        self._outputs: tuple[torch.Tensor, torch.Tensor] | None = None
        self._fingerprint: tuple[tuple, tuple] | None = None
        self._steps = 0
        self._validations_left = 0
        # The graded steps at which the most recent re-captures ran (not the first capture).
        self._recaptures: list[int] = []
        # The managers the phase runs, and every term whose configured fields a capture bakes in, as
        # ``(label, term, the names of those fields)``: the per-step fingerprint reads them. Resolved once, as
        # the terms are: walking each term class's MRO for its field names on every step cost more host time
        # than the replay saves of it (~150 µs a step on BeyondMimic). A field its manager mirrors on the device
        # is left out.
        self._managers = _graded_managers(env)
        self._terms = self._resolve_terms()
        # The bindings on the way to each tensor the capture read (see :meth:`_watch`): one that now holds
        # another object leaves the graph reading the old address. Set by each capture.
        self._links: list[tuple[str, Any, Any, Any]] = []
        self._leaves: list[tuple[str, Any, Any, tuple]] = []
        # The paths of the tensors the phase leaves behind (each bound tensor it writes, each binding it
        # replaces): what a validated replay compares and undoes. Set by each capture.
        self._state_paths: frozenset[str] = frozenset()
        # The tensors the phase writes that no binding the walk follows reaches (a helper's buffer below the
        # walk, a closure's), held directly: what a validated replay compares and undoes beside the paths. Set by each
        # capture.
        self._unreached: list[torch.Tensor] = []

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The graded phase of this step: ``(reset_buf, reward_buf)``, as the eager phase returns them.

        Every query the phase makes takes its fixed-shape form (``env.fixed_shape_query_scope()``) on every
        step, the eager ones included: a capture cannot record a host read, and the eager runs a validation
        compares with the graph must reduce over the same shapes.
        """
        with self._env.fixed_shape_query_scope():
            return self._graded_step()

    def _graded_step(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._steps += 1
        for manager in self._managers:
            manager.sync_device_fields()
        replaced = self._replaced_terms()
        fingerprint = self._baked_fingerprint()
        if self._graph is None:
            if self._steps <= self._options.warmup_steps:
                return self._compute()
            return self._capture(fingerprint)
        # A replaced term list has nothing to compare field by field against: the term change is the change.
        changed = replaced or self._changed_fields(fingerprint)
        changed += self._rebound_inputs()
        if changed:
            window = self._options.recapture_window
            recent = [step for step in self._recaptures if self._steps - step < window]
            if len(recent) >= self._options.max_recaptures_per_window:
                raise RuntimeError(self._recapture_storm(changed, recent))
            self._recaptures = [*recent, self._steps]
            return self._capture(fingerprint)
        if self._validations_left:
            self._validations_left -= 1
            return self._validated_replay()
        self._graph.replay()
        return self._result()

    def _capture(self, fingerprint: tuple[tuple, tuple]) -> tuple[torch.Tensor, torch.Tensor]:
        # Drop the previous graph (and its memory pool) before capturing its replacement.
        self._graph, self._outputs = None, None
        graph = torch.cuda.CUDAGraph()
        reads = _ReadRecorder()

        def capture_and_replay() -> None:
            # Genesis serves a query it already answered this step from a per-step cache (contacts, equality
            # constraints). Served to the capture, those cached tensors would be read by every replay at an
            # address the graph does not own: drop the cache so the capture records the query itself.
            clear_rigid_query_caches(self._env.rigid_solver)
            try:
                with self._env.graded_capture_scope(), torch.cuda.graph(graph), reads:
                    self._outputs = self._compute()
            except RuntimeError as exc:
                raise _capture_error(exc) from exc
            # Capturing records the kernels without running them: this step's values come from the replay.
            graph.replay()

        result, self._state_paths, self._unreached = self._validated_capture(capture_and_replay)
        self._graph, self._fingerprint = graph, fingerprint
        self._watch(reads.storages)
        self._validations_left = self._options.validation_steps
        return result

    def _watch(self, read_storages: set[int]) -> None:
        """Watch every binding on the way from the env to each tensor the capture read.

        The tensor's own binding is watched by address (rebinding the tensor), and each container and object
        above it by identity (replacing a dict, a list or a helper that holds it), so a re-capture follows any
        of them.
        """
        links: dict[tuple[int, Any], tuple[str, Any, Any, Any]] = {}
        leaves = []
        for path, binding in _reachable_tensors(self._env).items():
            if _storage(binding.tensor) not in read_storages:
                continue
            leaves.append((path, binding.container, binding.key, _address(binding.tensor)))
            for container, key, child in binding.chain:
                links.setdefault((id(container), key), (path, container, key, child))
        self._links, self._leaves = list(links.values()), leaves

    def _resolve_terms(self) -> list[tuple[str, Any, tuple[str, ...]]]:
        """Every graded term, as ``(label, term, the names of the configured fields a capture bakes in)``."""
        return [
            (_term_label(kind, name), term, term.fingerprint_names(exclude=manager.device_synced_fields))
            for kind, name, manager, term in _graded_terms(self._env)
        ]

    def _replaced_terms(self) -> list[str]:
        """Re-resolve the graded terms if one was added, removed or replaced since they were resolved; say so.

        The graph replays the terms it captured, and the fingerprint reads the fields of the terms it resolved:
        a term swapped into a manager afterwards would be neither replayed nor watched.
        """
        current = [term for _, _, _, term in _graded_terms(self._env)]
        if len(current) == len(self._terms) and all(now is then for now, (_, then, _) in zip(current, self._terms)):
            return []
        self._terms = self._resolve_terms()
        return ["the graded terms (one was added, removed or replaced in its manager)"]

    def _changed_fields(self, fingerprint: tuple[tuple, tuple]) -> list[str]:
        """The baked-in values that differ from what the current graph captured."""
        (terms_now, env_now), (terms_then, env_then) = fingerprint, self._fingerprint
        changed = [
            f"{label} field '{name}'"
            for (label, _, _), now, then in zip(self._terms, terms_now, terms_then)
            if now != then
            for (name, value_now), (_, value_then) in zip(now, then)
            if value_now != value_then
        ]
        changed += [
            f"env attribute '{name}'" for name, now, then in zip(_ENV_CONSTANTS, env_now, env_then) if now != then
        ]
        return changed

    def _rebound_inputs(self) -> list[str]:
        """The bindings on the way to a tensor the graph reads that now hold another object (or none)."""
        rebound = [
            f"{path} (a container or object on the way to it was replaced)"
            for path, container, key, child in self._links
            if _value_at(container, key) is not child
        ]
        rebound += [
            path
            for path, container, key, address in self._leaves
            if not isinstance(value := _value_at(container, key), torch.Tensor) or _address(value) != address
        ]
        return list(dict.fromkeys(rebound))

    def _recapture_storm(self, changed: list[str], recent: list[int]) -> str:
        shown = ", ".join(changed[:_SHOWN_DIFFERENCES])
        more = f" and {len(changed) - _SHOWN_DIFFERENCES} more" if len(changed) > _SHOWN_DIFFERENCES else ""
        return (
            f"EnvOptions.cuda_graph_options: {shown}{more} changed again at graded step {self._steps}, after "
            f"re-captures at step(s) {', '.join(map(str, recent))}, and CudaGraphOptions allows at most "
            f"{self._options.max_recaptures_per_window} re-captures within {self._options.recapture_window} steps. "
            "A capture bakes in every configured field of the graded terms and the address of every tensor it "
            "reads, so something that changes one on (nearly) every step, typically a curriculum, would re-capture "
            "and re-validate every step, which is slower than running the phase eagerly. Make it write in place "
            "instead (keep the value in a tensor allocated once, and update it with copy_), or set "
            "cuda_graph_options=None; catching this error would leave the graph replaying the old value."
        )

    def _validated_capture(
        self, capture_and_replay: Callable[[], None]
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], frozenset[str], list[torch.Tensor]]:
        """The capture step's result, checked bit for bit against an eager run, and what the phase leaves.

        The eager run goes first, since a capture records kernels without running them, and is undone so the
        capture starts from the same state: every tensor that existed before it is put back, whether or not an
        Eden attribute reaches it, so the capture's replay does not advance it a second time. A tensor the
        eager run allocated is not (see :class:`_WriteRecorder`).

        Returns the outputs, the paths of the bound tensors the phase leaves (each one it writes, each binding
        it replaces), and the tensors it writes that no path reaches, which a validated replay compares as well.
        """
        bound = _reachable_tensors(self._env)
        with _WriteRecorder() as writes:
            expected_outputs = tuple(out.clone() for out in self._compute())
        after = _reachable_tensors(self._env)
        # What the phase leaves: each bound tensor it wrote, each binding it replaced or added, and each tensor
        # it wrote that no binding reaches.
        written = writes.written()
        written_storages = {_storage(tensor) for tensor in written}
        bound_storages = {_storage(binding.tensor) for binding in bound.values()}
        state_paths = {path for path, binding in bound.items() if _storage(binding.tensor) in written_storages}
        state_paths |= {
            path for path, binding in after.items() if path not in bound or bound[path].tensor is not binding.tensor
        }
        unreached = [tensor for tensor in written if _storage(tensor) not in bound_storages]
        expected = {path: after[path].tensor.clone() for path in state_paths if path in after}
        expected_unreached = [tensor.clone() for tensor in unreached]
        # Undo it: put the replaced bindings back, then every written value, last write first.
        for binding in bound.values():
            _rebind(binding.container, binding.key, binding.tensor)
        _write_back(reversed(writes.saved))

        capture_and_replay()
        outputs = self._result()
        self._check(
            "the capture step", expected, expected_outputs, outputs, unreached=zip(unreached, expected_unreached)
        )
        return outputs, frozenset(state_paths), unreached

    def _validated_replay(self) -> tuple[torch.Tensor, torch.Tensor]:
        """This step from the graph, checked bit for bit against an eager run; the eager result is kept.

        The graph goes first and is undone, so the eager run reads nothing a replay produced. The other way
        round, work the capture recorded without being able to replay (a kernel launched outside torch's stream
        ran once, at capture time) would be redone by the eager run just before the replay read it, and agree.
        Undoing it puts back every tensor the capture step found the phase writing, reached by an Eden
        attribute or not, so the eager run does not advance any of them a second time.
        """
        bound = _reachable_tensors(self._env)
        state = {path: bound[path].tensor for path in self._state_paths if path in bound}
        snapshot = [*state.values(), *self._unreached]
        before = [tensor.clone() for tensor in snapshot]
        self._graph.replay()
        replay_outputs = tuple(out.clone() for out in self._outputs)
        replayed = {path: tensor.clone() for path, tensor in state.items()}
        replayed_unreached = [tensor.clone() for tensor in self._unreached]
        _write_back(zip(snapshot, before))

        with _WriteRecorder() as writes:
            outputs = self._compute()
        # A tensor the eager phase writes but the capture's eager run did not: the graph's value of it is
        # unknown, and a phase whose writes change from step to step is not one a graph replays.
        watched = {_storage(tensor) for tensor in snapshot}
        unwatched = [
            f"{_describe(tensor, bound)} (written eagerly, not on the capture step)"
            for tensor in writes.written()
            if _storage(tensor) not in watched
        ]
        unwatched += [f"{path} (bound on the capture step, gone now)" for path in self._state_paths - state.keys()]
        now = self._check(
            "a replay", replayed, replay_outputs, outputs, unwatched, unreached=zip(self._unreached, replayed_unreached)
        )
        # Keep the graph's bindings, holding the eager values: the next replay writes where the capture bound.
        for path, tensor in state.items():
            current = now[path]
            if current.tensor is not tensor:
                _write_back([(tensor, current.tensor)])
                _rebind(current.container, current.key, tensor)
        return outputs

    def _check(
        self,
        which: str,
        expected: dict[str, torch.Tensor],
        expected_outputs: tuple[torch.Tensor, ...],
        outputs: tuple[torch.Tensor, ...],
        differ: Iterable[str] = (),
        unreached: Iterable[tuple[torch.Tensor, torch.Tensor]] = (),
    ) -> dict[str, _Binding]:
        """Raise unless the env's tensors at ``expected``'s paths and ``outputs`` hold the same bits as expected.

        ``unreached`` pairs each written tensor no path reaches with its expected value. ``differ`` lists
        differences the caller already found. Returns the env's reachable tensors, as checked.
        """
        now = _reachable_tensors(self._env)
        differ = list(differ)
        differ += [
            path for path, value in expected.items() if path not in now or not _same_bits(value, now[path].tensor)
        ]
        differ += [_describe(tensor, now) for tensor, value in unreached if not _same_bits(value, tensor)]
        differ += [
            name for name, a, b in zip(("reset_buf", "reward_buf"), expected_outputs, outputs) if not _same_bits(a, b)
        ]
        differ = list(dict.fromkeys(differ))
        if differ:
            shown = ", ".join(differ[:_SHOWN_DIFFERENCES])
            more = f" and {len(differ) - _SHOWN_DIFFERENCES} more" if len(differ) > _SHOWN_DIFFERENCES else ""
            raise RuntimeError(
                f"EnvOptions.cuda_graph_options: on {which}, the CUDA graph of the graded phase disagrees with "
                f"an eager run of the same step in {len(differ)} tensor(s): {shown}{more}. A term in the metric, "
                "termination or reward manager (or an observation term's post_physics_refresh) depends on "
                "something a replay does not redo: Python state outside its configured fields, a tensor it "
                "rebinds instead of writing in place, or work launched outside torch's stream (it runs once, at "
                "capture time). Set cuda_graph_options=None (the eager step is unaffected) or fix the term; do "
                "not catch this error, the replayed values are wrong."
            )
        return now

    def _result(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._outputs is not None
        reset_buf, reward_buf = self._outputs
        # The eager termination pass returns a fresh tensor every step, which a caller may keep; the graph's
        # lives in its memory pool and is overwritten by the next replay.
        return reset_buf.clone(), reward_buf

    def _baked_fingerprint(self) -> tuple[tuple, tuple]:
        """What a capture bakes in: ``(each graded term's configured fields, the env's constants)``."""
        terms = tuple(term.configured_fingerprint(names=names) for _, term, names in self._terms)
        return terms, tuple(frozen_value(getattr(self._env, name, None)) for name in _ENV_CONSTANTS)


def _capture_error(exc: RuntimeError) -> RuntimeError:
    """The error to raise for a capture that failed with ``exc``, naming the term line it started at."""
    root, where = _first_failure(exc)
    if isinstance(root, BakedHostReadError):
        # Its own message says what was read and what to do instead.
        return RuntimeError(f"EnvOptions.cuda_graph_options: capturing the graded phase failed at {where}: {root}")
    cause = str(root).splitlines()[0] if str(root) else type(root).__name__
    return RuntimeError(
        f"EnvOptions.cuda_graph_options: capturing the graded phase (metric, termination and "
        f"reward compute) failed at {where}: {cause}. That operation reads a value back to the host "
        "(.item(), nonzero(), a bool-mask gather, a Python branch on a tensor) or copies from host "
        "memory, which a CUDA graph cannot record. Set cuda_graph_options=None, or make that term stay "
        "on the device; catching this error would leave the phase unreplayable."
    )


def _first_failure(exc: BaseException) -> tuple[BaseException, str]:
    """The error a failed capture started with, and the term source line that raised it, for a message.

    A capture that fails is invalidated, and leaving the capture context raises again: the error that names
    the offending operation is the first in the ``__context__`` chain. The line reported is the innermost one
    that is term code — anything but torch, Genesis and the like, or Eden's own plumbing (:func:`_is_plumbing`)
    — so a term defined in a task, an extension or a user's package is named by its own line.
    """
    root = exc
    while isinstance(root.__context__, Exception):
        root = root.__context__
    frames = traceback.extract_tb(root.__traceback__)
    term_frames = [frame for frame in frames if not _is_plumbing(frame.filename)]
    frame = (term_frames or frames or [None])[-1]
    where = "an unknown line" if frame is None else f"{_shown_path(frame.filename)}:{frame.lineno} ({frame.name})"
    return root, where


def _is_plumbing(filename: str) -> bool:
    """Whether ``filename`` is library code rather than a term's: an installed dependency or an Eden module that
    runs terms (``eden/envs``, ``eden/entities``, ``eden/utils``, the managers themselves).
    """
    if filename.startswith(_EDEN_ROOT):
        relative = os.path.relpath(filename, _EDEN_ROOT)
        return not _TERM_CODE.match(relative)
    return "site-packages" in filename or "dist-packages" in filename or filename.startswith("<")


def _shown_path(filename: str) -> str:
    """``filename`` relative to the checkout holding Eden when it is inside it, else as given."""
    checkout = os.path.dirname(_EDEN_ROOT)
    return os.path.relpath(filename, checkout) if filename.startswith(checkout + os.sep) else filename


def _graded_managers(env: RLEnvBase) -> list[ManagerBase]:
    """The managers the graded phase runs: the metric (when the task has one), termination, reward and observation."""
    managers = (env.metric_manager, env.termination_manager, env.reward_manager, env.observation_manager)
    return [manager for manager in managers if manager is not None]


def _graded_terms(env: RLEnvBase) -> Iterable[tuple[str, str, ManagerBase, Any]]:
    """Every term the graded phase runs, as ``(kind, name, its manager, term)`` (see :func:`_term_label`).

    The metric, termination and reward terms, and the observation terms ``post_physics_refresh`` ticks (named
    ``group.term``). Checked on every step, so the label is left to the caller.
    """
    for kind in ("metric", "termination", "reward"):
        manager = getattr(env, f"{kind}_manager")
        if manager is not None:
            for name, term in manager.terms.items():
                yield kind, name, manager, term
    observations = env.observation_manager
    for group, name, term in observations.iter_post_physics_terms():
        yield "observation", f"{group}.{name}", observations, term


def _term_label(kind: str, name: str) -> str:
    """How an error message names a graded term: ``reward term 'track_pos'``."""
    return f"{kind} term '{name}'"


class _Binding(NamedTuple):
    """Where the walk found a tensor: ``container[key]`` (an item) or ``getattr(container, key)`` (an attribute).

    ``chain`` is every ``(container, key, child)`` step from the env down to ``container``, so a watch can tell
    when any of them was replaced. A tensor inside a tuple cannot be rebound (:func:`_rebind` skips it); the
    step that holds the tuple can be.
    """

    container: Any
    key: Any
    tensor: torch.Tensor
    chain: tuple[tuple[Any, Any, Any], ...]


def _reachable_tensors(env: RLEnvBase) -> dict[str, _Binding]:
    """Every CUDA tensor reachable from ``env``, as ``path -> binding``, through the objects the walk follows.

    Containers (dict, list, tuple) are followed, and every object :func:`_walked` accepts, however deep: a
    binding the walk stopped short of would go unwatched. Each object is entered once, by the first path that
    reaches it (depth first, in attribute and item order), and the env only at the root.
    """
    found: dict[str, _Binding] = {}
    seen: set[int] = set()
    # (value, path, container, key, chain): a stack rather than recursion, so no object graph is too deep.
    pending: list[tuple[Any, str, Any, Any, tuple]] = [(env, "env", None, None, ())]
    while pending:
        value, path, container, key, chain = pending.pop()
        if isinstance(value, torch.Tensor):
            if value.is_cuda:
                found[path] = _Binding(container, key, value, chain)
            continue
        if id(value) in seen:
            continue
        if isinstance(value, dict):
            items = list(value.items())
        elif isinstance(value, (list, tuple)):
            items = list(enumerate(value))
        elif value is env or _walked(value):
            items = list(vars(value).items())
        else:
            continue
        seen.add(id(value))
        below = chain if container is None else (*chain, (container, key, value))
        # Pushed last first, so they are taken in order.
        pending.extend(
            (item, path + _step(value, item_key), value, item_key, below) for item_key, item in reversed(items)
        )
    return found


def _step(container: Any, key: Any) -> str:
    """How a path names ``key`` of ``container``: ``['key']`` in a dict, ``[index]`` in a sequence, else ``.name``."""
    if isinstance(container, dict):
        return f"[{key!r}]"
    if isinstance(container, (list, tuple)):
        return f"[{key}]"
    return f".{key}"


def _walked(value: Any) -> bool:
    """Whether the walk descends into ``value``'s attributes.

    Not into the env (every term's back-reference to it), this graph, a class, an object without attributes, or
    an object of the standard library or of :data:`_UNWALKED_PACKAGES`.
    """
    from eden.envs.base import EnvBase

    if isinstance(value, type) or not hasattr(value, "__dict__") or isinstance(value, (EnvBase, GradedPhaseGraph)):
        return False
    return _walked_class(type(value))


@functools.cache
def _walked_class(cls: type) -> bool:
    """Whether ``cls`` is defined outside the standard library and :data:`_UNWALKED_PACKAGES`."""
    top = cls.__module__.partition(".")[0]
    return top not in _UNWALKED_PACKAGES and top not in sys.stdlib_module_names and top != "builtins"


class _WriteRecorder(TorchDispatchMode):
    """Record, before each op runs, the value of every tensor it writes (in place, or through ``out=``) that
    existed before the recording began.

    ``saved`` lists ``(tensor, value before the write)`` in write order, so writing the values back last
    first (:func:`_write_back`) restores even a tensor written twice, or through two overlapping views. That
    covers every tensor the phase found in place, whether or not an Eden attribute reaches it (a helper's
    buffer below the walk, a closure's). A tensor the recorded run allocated itself is left out: undoing a
    write to it would only hand stale values to whoever keeps it, such as Genesis' per-step contact cache,
    which pads a fresh tensor in place and serves it to every later query of the step.
    """

    def __init__(self) -> None:
        super().__init__()
        self.saved: list[tuple[torch.Tensor, torch.Tensor]] = []
        # The storages the recorded run allocated: an op's output that shares no storage with its inputs.
        self._fresh: set[int] = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        for index, argument in enumerate(func._schema.arguments):
            if argument.alias_info is None or not argument.alias_info.is_write:
                continue
            target = args[index] if index < len(args) else kwargs.get(argument.name)
            if isinstance(target, torch.Tensor) and _storage(target) not in self._fresh:
                self.saved.append((target, target.clone()))
        out = func(*args, **kwargs)
        inputs = {_storage(leaf) for leaf in tree_leaves((args, kwargs)) if isinstance(leaf, torch.Tensor)}
        self._fresh.update(
            _storage(leaf)
            for leaf in tree_leaves(out)
            if isinstance(leaf, torch.Tensor) and _storage(leaf) not in inputs
        )
        return out

    def written(self) -> list[torch.Tensor]:
        """Every tensor written, once each (two writes through the same view count once), in first-write order."""
        return list({_address(tensor): tensor for tensor, _ in self.saved}.values())


def _write_back(values: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> None:
    """Copy each ``(tensor, value)`` pair's value into its tensor, in the order given.

    Inside ``torch.inference_mode``: a tensor allocated under it (anything ``RewardManager.compute``, which runs
    under it, allocates and keeps) refuses an in-place write outside it, while a normal tensor accepts one inside.
    """
    with torch.inference_mode():
        for tensor, value in values:
            tensor.copy_(value)


class _ReadRecorder(TorchDispatchMode):
    """Record the storage of every tensor an op takes: what a capture reads, for :meth:`GradedPhaseGraph._watch`."""

    def __init__(self) -> None:
        super().__init__()
        self.storages: set[int] = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        for leaf in tree_leaves((args, kwargs)):
            if isinstance(leaf, torch.Tensor):
                self.storages.add(_storage(leaf))
        return func(*args, **kwargs)


def _storage(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def _address(tensor: torch.Tensor) -> tuple:
    """Where and how a graph reads ``tensor``: another tensor with the same address reads the same memory."""
    return tensor.data_ptr(), tuple(tensor.shape), tensor.stride(), tensor.dtype


def _value_at(container: Any, key: Any) -> Any:
    """What is bound at ``container[key]`` (an item) or the attribute ``key`` of ``container`` now, or None."""
    if isinstance(container, dict):
        return container.get(key)
    if isinstance(container, (list, tuple)):
        return container[key] if key < len(container) else None
    return getattr(container, key, None)


def _rebind(container: Any, key: Any, tensor: torch.Tensor) -> None:
    """Bind ``tensor`` back at ``container[key]`` (or the attribute ``key``) if something replaced it."""
    if isinstance(container, tuple):  # immutable: its items were restored in place
        return
    if isinstance(container, (dict, list)):
        if container[key] is not tensor:
            container[key] = tensor
    elif getattr(container, key) is not tensor:
        setattr(container, key, tensor)


def _describe(tensor: torch.Tensor, bound: dict[str, _Binding]) -> str:
    """A path the env reaches ``tensor``'s storage by, for an error message."""
    storage = _storage(tensor)
    unnamed = f"<a {tuple(tensor.shape)} {tensor.dtype} tensor no binding the walk follows reaches>"
    return next((path for path, binding in bound.items() if _storage(binding.tensor) == storage), unnamed)


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether ``a`` and ``b`` hold the same bytes — NaN payloads included, which ``torch.equal`` calls unequal."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.numel() == 0:
        return True
    return torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))
