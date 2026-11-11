"""Tests for ``eden.utils.torch``: lazy ``compile_model``, ``read_checkpoint_index``, ``device_constant`` and the
numpy<->torch conversions.

``compile_model`` must not compile at decoration time: building the ``torch.compile`` wrapper imports
``torch._dynamo`` (~170 MB), which ran on every ``import eden``. ``torch.compile`` is replaced only to make it
fail or to keep a test off dynamo.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest
import torch

from eden.utils.torch import (
    compile_model,
    device_constant,
    numpy_to_tensor,
    read_checkpoint_index,
    to_numpy,
    to_numpy_unbatched,
)


def _raise(*args, **kwargs):
    raise RuntimeError("nope")


def test_neither_importing_eden_nor_decorating_imports_dynamo_but_the_first_call_does():
    """In a fresh interpreter: anything earlier in this session may already have compiled something."""
    script = textwrap.dedent(
        """
        import sys, torch
        import eden  # noqa: F401
        assert "torch._dynamo" not in sys.modules, "importing eden imported torch._dynamo"
        from eden.utils.torch import compile_model

        @compile_model
        def double(x):
            return x * 2

        assert "torch._dynamo" not in sys.modules, "decorating with compile_model compiled eagerly"
        assert torch.equal(double(torch.arange(4.0)), torch.arange(4.0) * 2)
        assert "torch._dynamo" in sys.modules, "calling a compiled function should have compiled it"
        print("OK")
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)

    assert result.returncode == 0 and "OK" in result.stdout, result.stderr


def test_the_compiled_function_returns_what_the_eager_one_does_and_keeps_its_identity():
    def rowwise_square_sum(x):
        """Doc."""
        return torch.sum(torch.square(x), dim=1)

    compiled = compile_model(rowwise_square_sum)
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    torch.testing.assert_close(compiled(x), torch.tensor([5.0, 50.0, 149.0, 302.0]))
    assert (compiled.__name__, compiled.__doc__, compiled.__wrapped__) == (
        "rowwise_square_sum",
        "Doc.",
        rowwise_square_sum,
    )


def test_a_failed_compile_warns_at_the_call_and_runs_eagerly(monkeypatch):
    compiled = compile_model(lambda x: x + 1)  # nothing compiled yet, so nothing to fail
    monkeypatch.setattr(torch, "compile", _raise)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.testing.assert_close(compiled(torch.ones(3)), torch.full((3,), 2.0))

    assert any("torch.compile failed" in str(w.message) for w in caught)


def test_a_module_comes_back_as_a_module(monkeypatch):
    """A deferred closure would drop ``.parameters()``, ``.to()``, ``.state_dict()`` and ``isinstance``."""

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.full((3,), 2.0))

        def forward(self, x):
            return x * self.scale

    monkeypatch.setattr(torch, "compile", lambda model, **kw: model)  # keeps this test off dynamo
    compiled = compile_model(Tiny())

    assert isinstance(compiled, torch.nn.Module)
    assert [name for name, _ in compiled.named_parameters()] == ["scale"]
    assert compiled.eval() is compiled
    torch.testing.assert_close(compiled(torch.ones(3)), torch.full((3,), 2.0))


def test_a_module_whose_compile_fails_is_returned_unchanged_with_a_warning(monkeypatch):
    module = torch.nn.Linear(2, 2)
    monkeypatch.setattr(torch, "compile", _raise)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert compile_model(module) is module

    assert any("torch.compile failed" in str(w.message) for w in caught)


def test_a_pre_zipfile_checkpoint_is_indexed_by_a_full_read(tmp_path):
    """Pre-1.6 checkpoints cannot be memory-mapped, so the index falls back to reading them."""
    path = str(tmp_path / "ancient.pt")
    torch.save({"model_state_dict": {"actor.0.weight": torch.zeros(4)}}, path, _use_new_zipfile_serialization=False)

    with pytest.raises(RuntimeError, match="mmap can only be used"):
        torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    assert sorted(read_checkpoint_index(path)) == ["model_state_dict"]


def test_device_constant_is_built_once_per_value(count_cuda_syncs):
    first = device_constant([0.1, 0.2], "cuda")
    assert device_constant((0.1, 0.2), "cuda") is first
    assert count_cuda_syncs(lambda: device_constant([0.1, 0.2], "cuda")) == 0
    assert device_constant([0.1, 0.3], "cuda") is not first


@pytest.mark.parametrize(("allow_tf32", "legacy", "cuda"), [(True, "high", "tf32"), (False, "highest", "ieee")])
def test_backend_configuration_leaves_both_precision_getters_readable(allow_tf32, legacy, cuda):
    """Inductor reads the legacy precisions when it lowers a matmul or a convolution; a precision set through the new
    API alone made those getters raise (matmul: torch 2.10 and 2.14; cuDNN with ``allow_tf32=False``: 2.14), so a
    ``torch.compile`` of a network died on its first call after ``eden.init()``. On 2.10 the new cuDNN field alone
    also left ``cudnn.conv`` / ``cudnn.rnn`` at TF32.
    """
    from eden.utils.torch import configure_torch_backends

    before_matmul = torch.get_float32_matmul_precision()
    before_cudnn = (torch.backends.cudnn.allow_tf32, torch.backends.cudnn.fp32_precision)
    try:
        configure_torch_backends(allow_tf32=allow_tf32)
        assert torch.get_float32_matmul_precision() == legacy
        assert torch.backends.cuda.matmul.fp32_precision == cuda
        assert torch.backends.cudnn.allow_tf32 is allow_tf32
        assert torch.backends.cudnn.conv.fp32_precision == cuda
        assert torch.backends.cudnn.rnn.fp32_precision == cuda
    finally:
        # The rest of the session runs on these globals: restore both cuDNN APIs, legacy first, as the code under test.
        torch.set_float32_matmul_precision(before_matmul)
        torch.backends.cudnn.allow_tf32, torch.backends.cudnn.fp32_precision = before_cudnn


def test_numpy_to_tensor_shares_a_contiguous_array_and_copies_a_flipped_one():
    """``torch.from_numpy`` refuses negative strides, which is how the rasterizer hands back a flipped image."""
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    shared = numpy_to_tensor(array)
    array[0, 0] = 42.0
    assert shared[0, 0] == 42.0

    flipped = numpy_to_tensor(array[:, ::-1])
    assert flipped.tolist() == [[2.0, 1.0, 42.0], [5.0, 4.0, 3.0]]


def test_numpy_to_tensor_converts_only_when_asked():
    array = np.arange(3, dtype=np.float64)
    assert numpy_to_tensor(array).dtype == torch.float64
    assert numpy_to_tensor(array, dtype=torch.float32).dtype == torch.float32
    tensor = torch.arange(3)
    assert numpy_to_tensor(tensor) is tensor
    assert numpy_to_tensor(tensor, dtype=torch.float32).dtype == torch.float32


def test_to_numpy_detaches_a_tensor_and_converts_array_likes():
    leaf = torch.ones(2, requires_grad=True)
    assert to_numpy(leaf * 2).tolist() == [2.0, 2.0]  # ``.numpy()`` alone raises on a tensor that requires grad
    assert to_numpy(torch.ones(2), np.float64).dtype == np.float64
    assert to_numpy([1, 2], np.float32).dtype == np.float32
    array = np.zeros(2)
    assert to_numpy(array) is array


def test_to_numpy_unbatched_drops_only_a_unit_batch_axis():
    np.testing.assert_array_equal(to_numpy_unbatched(torch.tensor([[1.0, 2.0]])), [1.0, 2.0])
    np.testing.assert_array_equal(to_numpy_unbatched(torch.tensor([1.0, 2.0])), [1.0, 2.0])
    with pytest.raises(ValueError, match="leading axis 1"):
        to_numpy_unbatched(torch.zeros(2, 3))
