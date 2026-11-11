"""Tensor/array assertions that accept torch, numpy and nested sequences alike."""

import numpy as np
import torch
from typing import Type


def tensor_to_cpu(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
    return x


def tensor_to_array(x: torch.Tensor, dtype: Type[np.generic] | None = None) -> np.ndarray:
    return np.asarray(tensor_to_cpu(x), dtype=dtype)


def assert_allclose(actual, desired, *, atol=None, rtol=None, tol=None, err_msg=None):
    # Determine absolute and relative tolerance from input arguments
    assert (tol is not None) ^ (atol is not None or rtol is not None)
    if tol is not None:
        atol = tol
        rtol = tol
    if rtol is None:
        rtol = 0.0
    if atol is None:
        atol = 0.0

    # Convert input arguments as numpy arrays
    args = [actual, desired]
    for i, arg in enumerate(args):
        if isinstance(arg, (tuple, list)):
            arg = np.stack([tensor_to_array(val) for val in arg], axis=0)
        args[i] = tensor_to_array(arg)

    # Early return without checking anything is both arrays are empty (0D arrays have size 1).
    if all(e.size == 0 for e in args):
        return

    # Try to make sure both arrays have the exact same shape.
    # First, try to broadcast both matrices. Then it is does not work, squeeze them before trying again.
    try:
        args = np.broadcast_arrays(*args)
    except ValueError:
        args = np.broadcast_arrays(*map(np.squeeze, args))

    np.testing.assert_allclose(*args, atol=atol, rtol=rtol, err_msg=err_msg)


def assert_array_equal(actual, desired, *, err_msg=None):
    assert_allclose(actual, desired, atol=0.0, rtol=0.0, err_msg=err_msg)
