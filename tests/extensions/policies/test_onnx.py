"""``eden.extensions.policies.onnx``: the lazy runtime bind, session options, and ``dynamic_batch_onnx``."""

from __future__ import annotations

import ast
import os
import pathlib
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from eden.extensions.policies import onnx as onnx_mod
from eden.extensions.policies.onnx import _verify_dynamic_batch, dynamic_batch_onnx

_EDEN_ROOT = pathlib.Path(onnx_mod.__file__).resolve().parents[2]
_RNG = np.random.RandomState(0)


def _run_fresh(code: str) -> None:
    """Run ``code`` in a fresh interpreter (the state a deploy starts in), failing on a non-zero exit."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(_EDEN_ROOT.parent), env.get("PYTHONPATH", "")])
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


# ---------------------------------------------------------------------------
# lazy onnxruntime bind
# ---------------------------------------------------------------------------


def test_module_and_its_consumers_import_without_onnxruntime(tmp_path):
    """The graph tools and the sonic package import with no wheel; the runtime is demanded at use."""
    _run_fresh(
        f"""
        import sys
        sys.modules["onnxruntime"] = None  # an absent wheel: ``import onnxruntime`` raises ImportError
        import eden.extensions.policies.sonic.policy
        from eden.extensions.policies import onnx
        from eden.extensions.policies.omnixtreme import onnx_wrapper

        # The FK session fails on the missing file, not on a runtime it believes absent (the
        # from-imported ``ort`` sentinel once made this report "onnxruntime is not installed").
        try:
            onnx_wrapper._make_fk_onnx_session({str(tmp_path / "absent.onnx")!r})
        except FileNotFoundError as exc:
            assert "FK ONNX file does not exist" in str(exc)
        else:
            raise AssertionError("expected FileNotFoundError")
        try:
            onnx._load_ort()
        except ImportError as exc:
            assert "onnxruntime is not installed" in str(exc), exc
        else:
            raise AssertionError("expected ImportError")
        """
    )


def test_no_module_from_imports_the_lazy_ort_global():
    """``ort`` is rebound by ``_load_ort()``, so a from-imported copy stays ``None`` for ever.

    A lint over every module under ``eden/``: the behavioural test above covers the one consumer
    that shipped the bug; this keeps the next one from repeating it.
    """
    offenders = [
        f"{path.relative_to(_EDEN_ROOT.parent)}:{node.lineno}"
        for path in sorted(_EDEN_ROOT.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.ImportFrom) and any(alias.name == "ort" for alias in node.names)
    ]
    assert not offenders, f"reference the runtime as `onnx_mod.ort` instead: {offenders}"


def test_load_ort_publishes_the_runtime_on_the_module(onnx_runtime):
    loaded = onnx_mod._load_ort()
    assert loaded is onnx_mod.ort is onnx_runtime
    assert onnx_mod._load_ort() is loaded


def test_load_ort_rejects_a_half_removed_wheel(tmp_path):
    """A leftover ``onnxruntime/`` without ``__init__.py`` imports as an empty namespace package."""
    shell = tmp_path / "onnxruntime"
    shell.mkdir()
    _run_fresh(
        f"""
        import importlib.machinery, importlib.util, sys
        # What the import system builds for a directory with no __init__.py; installed ahead of the
        # real wheel, which would otherwise win (a regular package beats a namespace portion).
        spec = importlib.machinery.ModuleSpec("onnxruntime", None, is_package=True)
        spec.submodule_search_locations = [{str(shell)!r}]
        sys.modules["onnxruntime"] = importlib.util.module_from_spec(spec)

        from eden.extensions.policies import onnx
        try:
            onnx._load_ort()
        except ImportError as exc:
            assert "exposes no InferenceSession" in str(exc), exc
        else:
            raise AssertionError("expected ImportError")
        assert onnx.ort is None, "the broken shell was published as a live runtime"
        """
    )


# ---------------------------------------------------------------------------
# session options
# ---------------------------------------------------------------------------


@pytest.fixture
def identity_graph(tmp_path, write_onnx, onnx_runtime, monkeypatch):
    # ONNX_CPU keeps the "torch sees CUDA but the session is on CPU" warning out of these tests.
    monkeypatch.setenv("ONNX_CPU", "1")
    monkeypatch.delenv(onnx_mod.INTRA_OP_THREADS_ENV, raising=False)
    return write_onnx(tmp_path / "identity.onnx", [("Identity", ["x"], ["y"], {})], [("x", [1, 3])], [("y", [1, 3])])


@pytest.mark.parametrize("build", [onnx_mod._make_session, onnx_mod._rebuild_session_cpu])
def test_sessions_get_one_non_spinning_thread_by_default(identity_graph, onnx_runtime, build):
    """ORT's default is a spinning thread per core per session; that overran 70-89% of 50 Hz deploy ticks."""
    session = build(identity_graph)
    options = session.get_session_options()
    assert options.intra_op_num_threads == 1
    assert options.get_session_config_entry("session.intra_op.allow_spinning") == "0"
    assert options.graph_optimization_level == onnx_runtime.GraphOptimizationLevel.ORT_ENABLE_ALL
    assert session.get_providers() == ["CPUExecutionProvider"]


def test_env_var_raises_the_thread_count_but_never_re_enables_spinning(identity_graph, monkeypatch):
    monkeypatch.setenv(onnx_mod.INTRA_OP_THREADS_ENV, "4")
    options = onnx_mod._make_session(identity_graph).get_session_options()
    assert options.intra_op_num_threads == 4
    assert options.get_session_config_entry("session.intra_op.allow_spinning") == "0"


@pytest.mark.parametrize("bad", ["0", "-2", "all", "", "1.5"])
def test_invalid_thread_count_raises_naming_the_variable(identity_graph, monkeypatch, bad):
    monkeypatch.setenv(onnx_mod.INTRA_OP_THREADS_ENV, bad)
    with pytest.raises(ValueError, match=onnx_mod.INTRA_OP_THREADS_ENV):
        onnx_mod._make_session(identity_graph)


# ---------------------------------------------------------------------------
# dynamic_batch_onnx
# ---------------------------------------------------------------------------


@pytest.fixture
def static_batch1_mlp(tmp_path, write_onnx, onnx_runtime):
    """``obs(1,8) -> MatMul -> Reshape([1,4]) -> MatMul -> actions(1,3)``: the batch baked into a Reshape."""

    def make(directory=tmp_path):
        return write_onnx(
            pathlib.Path(directory) / "policy.onnx",
            [
                ("MatMul", ["obs", "W1"], ["h1"], {}),
                ("Reshape", ["h1", "shape1"], ["h1r"], {}),
                ("MatMul", ["h1r", "W2"], ["actions"], {}),
            ],
            [("obs", [1, 8])],
            [("actions", [1, 3])],
            {
                "W1": _RNG.randn(8, 4).astype(np.float32),
                "shape1": np.array([1, 4], np.int64),
                "W2": _RNG.randn(4, 3).astype(np.float32),
            },
        )

    return make


def _cpu_session(path):
    import onnxruntime

    return onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])


def test_patched_graph_accepts_a_batch_and_matches_looped_batch1(static_batch1_mlp):
    src = static_batch1_mlp()
    x = _RNG.randn(5, 8).astype(np.float32)
    with pytest.raises(Exception):  # the premise: the released graph rejects num_envs > 1
        _cpu_session(src).run(None, {"obs": x})

    session = _cpu_session(dynamic_batch_onnx(src))
    batched = session.run(None, {"obs": x})[0]
    looped = np.concatenate([session.run(None, {"obs": x[i : i + 1]})[0] for i in range(5)])
    assert batched.shape == (5, 3)
    np.testing.assert_allclose(batched, looped, atol=1e-5)


def test_cache_path_replaces_only_the_extension(tmp_path, static_batch1_mlp):
    """A parent directory containing ``.onnx`` must survive the cache-path derivation."""
    directory = tmp_path / "v1.onnx.d"
    directory.mkdir()
    assert dynamic_batch_onnx(static_batch1_mlp(directory)) == str(directory / "policy.dynbatch.onnx")


def test_cache_is_reused_until_the_source_is_newer(static_batch1_mlp):
    src = static_batch1_mlp()
    cached = dynamic_batch_onnx(src)
    os.utime(cached, (1_000.0, 1_000.0))
    os.utime(src, (500.0, 500.0))
    assert dynamic_batch_onnx(src) == cached
    assert os.path.getmtime(cached) == 1_000.0, "a fresh cache was rebuilt"

    os.utime(src, (2_000.0, 2_000.0))  # a re-downloaded checkpoint at the same path
    dynamic_batch_onnx(src)
    assert os.path.getmtime(cached) > 2_000.0, "a stale cache was reused"


def _batch_graph(write_onnx, tmp_path, name, nodes, initializers, outputs=(("actions", ["batch", 4]),)):
    return write_onnx(tmp_path / f"{name}.onnx", nodes, [("obs", ["batch", 4])], list(outputs), initializers)


def test_verify_rejects_rows_coupled_across_the_batch(tmp_path, write_onnx, onnx_runtime):
    """``CumSum`` over the batch axis is not permutation-equivariant."""
    path = _batch_graph(
        write_onnx, tmp_path, "coupled", [("CumSum", ["obs", "axis"], ["actions"], {})], {"axis": np.array(0, np.int64)}
    )
    with pytest.raises(RuntimeError, match="permutation-equivariant"):
        _verify_dynamic_batch(path)


def test_verify_accepts_a_quantized_row_independent_graph(tmp_path, write_onnx, onnx_runtime):
    """Batched and looped-batch-1 differ by rounding bins here (SONIC's VQ encoder); the check must not care."""
    path = _batch_graph(
        write_onnx,
        tmp_path,
        "quantized",
        [("MatMul", ["obs", "W"], ["h"], {}), ("Round", ["h"], ["actions"], {})],
        {"W": _RNG.randn(4, 4).astype(np.float32)},
    )
    _verify_dynamic_batch(path)


def test_verify_skips_an_output_whose_leading_axis_is_not_the_batch(tmp_path, write_onnx, onnx_runtime):
    """A fixed ``std_param`` output of shape ``(num_dofs,)`` used to crash the verifier on a broadcast."""
    path = _batch_graph(
        write_onnx,
        tmp_path,
        "with_std",
        [("Identity", ["obs"], ["actions"], {}), ("Identity", ["std_param"], ["std"], {})],
        {"std_param": np.arange(29, dtype=np.float32)},
        outputs=(("actions", ["batch", 4]), ("std", [29])),
    )
    _verify_dynamic_batch(path)
