"""ConcatBatchTensor: store unequal-length tensors in a single concatenated tensor."""

from __future__ import annotations

import torch
from typing import Sequence


class _ConcatBatchTensorMaskView:
    def __init__(self, parent: "ConcatBatchTensor"):
        self._parent = parent

    def __getitem__(self, idx):
        return self._parent.get_with_mask(idx)


class ConcatBatchTensor:
    """Store a batch of unequal-length tensors concatenated along dim 0 into a single tensor.

    The tensors share their trailing shape; each one's length (its batch size) is kept, so every tensor can be
    recovered. An element is addressed by ``(batch_idx, data_idx)``: which tensor, and the position within it.

    Parameters
    ----------
    tensors : list[torch.Tensor] | None
        The tensors to concatenate. If ``None``, ``batch_sizes`` and ``data_shape`` allocate an uninitialised
        buffer instead.
    batch_sizes : Sequence[int] | None
        The length of each tensor (used only when ``tensors`` is ``None``).
    data_shape : tuple | None
        The trailing shape every tensor shares (used only when ``tensors`` is ``None``).
    dtype : torch.dtype, optional
        The dtype of the concatenated tensor.
    device : torch.device | str, optional
        The device of the concatenated tensor.
    """

    def __init__(
        self,
        tensors: list[torch.Tensor] | None = None,
        batch_sizes: Sequence[int] | None = None,
        data_shape: tuple | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ):
        self._device = torch.device(device)
        if tensors is None:
            if batch_sizes is None or data_shape is None:
                raise ValueError("Either tensors or batch_sizes and data_shape should be provided.")

            self._set_batch_sizes(torch.as_tensor(batch_sizes, dtype=torch.int64, device=self._device))
            self._data_shape = torch.Size(data_shape)
            total_batch = int(torch.sum(self._batch_sizes).item())
            self._concatenated_tensor = torch.empty((total_batch,) + self._data_shape, dtype=dtype, device=self._device)
        else:
            self._set_batch_sizes(torch.tensor([t.shape[0] for t in tensors], dtype=torch.int64, device=self._device))
            self._data_shape = tensors[0].shape[1:]
            for i, t in enumerate(tensors):
                if t.shape[1:] != self._data_shape:
                    raise ValueError(
                        f"Tensor at index {i} has shape {t.shape}, but expected {self._data_shape} after batch dim."
                    )

            # Efficiently concatenate and move to device
            self._concatenated_tensor = torch.cat(tensors, dim=0).to(device=self._device, dtype=dtype)

    def _set_batch_sizes(self, batch_sizes: torch.Tensor) -> None:
        """Adopt ``batch_sizes`` (int64, on this buffer's device) and derive the per-batch bookkeeping from it."""
        self._batch_sizes = batch_sizes
        batch_cuts = torch.cumsum(batch_sizes, dim=0)
        self._batch_starts = torch.zeros_like(batch_cuts)
        self._batch_starts[1:] = batch_cuts[:-1]
        self._batch_ends = batch_cuts
        self._with_mask_view = _ConcatBatchTensorMaskView(self)

    @classmethod
    def from_concatenated(
        cls,
        concatenated_tensor: torch.Tensor,
        batch_sizes: torch.Tensor | Sequence[int],
        device: torch.device | str | None = None,
    ) -> "ConcatBatchTensor":
        """Build a ConcatBatchTensor that wraps an already-concatenated tensor.

        Skips the empty-buffer allocation done by ``__init__(batch_sizes=...,
        data_shape=...)`` and avoids the ``torch.cat`` done by
        ``__init__(tensors=...)``. Used by the cached-load path so a mmap'd
        tensor can become a CBT with zero copies.

        ``device`` is only a label for the wrapped tensor's actual device — if
        provided it must match ``concatenated_tensor.device``. We refuse a
        mismatch rather than silently allocating index buffers on a different
        device than the data, which would surface later as opaque indexing
        errors. To wrap-then-move, do ``CBT.from_concatenated(t, ...).to(dev)``.

        The caller is responsible for ensuring
        ``concatenated_tensor.shape[0] == sum(batch_sizes)``.
        """
        resolved_device = torch.device(device) if device is not None else concatenated_tensor.device
        if resolved_device != concatenated_tensor.device:
            raise ValueError(
                f"device={resolved_device} does not match "
                f"concatenated_tensor.device={concatenated_tensor.device}. "
                "Wrap then call .to(device) instead."
            )
        new = cls._from_parts(
            concatenated_tensor,
            torch.as_tensor(batch_sizes, dtype=torch.int64, device=resolved_device),
            resolved_device,
        )
        if int(new._batch_sizes.sum().item()) != concatenated_tensor.shape[0]:
            raise ValueError(
                f"sum(batch_sizes)={int(new._batch_sizes.sum().item())} does not match "
                f"concatenated_tensor.shape[0]={concatenated_tensor.shape[0]}"
            )
        return new

    @classmethod
    def _from_parts(
        cls, concatenated_tensor: torch.Tensor, batch_sizes: torch.Tensor, device: torch.device
    ) -> "ConcatBatchTensor":
        """A buffer around ``concatenated_tensor``, cut into ``batch_sizes`` (int64), both already on ``device``.

        No check, copy or GPU->CPU sync: the per-batch bookkeeping is derived from ``batch_sizes`` on its device.
        """
        new = cls.__new__(cls)
        new._device = device
        new._concatenated_tensor = concatenated_tensor
        new._data_shape = concatenated_tensor.shape[1:]
        new._set_batch_sizes(batch_sizes)
        return new

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._concatenated_tensor.dtype

    @property
    def shape(self) -> torch.Size:
        return self._concatenated_tensor.shape

    @property
    def concatenated_tensor(self) -> torch.Tensor:
        return self._concatenated_tensor

    @property
    def batch_starts(self) -> torch.Tensor:
        return self._batch_starts

    @property
    def batch_sizes(self) -> torch.Tensor:
        return self._batch_sizes

    @property
    def longest_batch(self) -> int:
        """The largest batch size, read back once (a GPU->CPU sync) and remembered.

        Keyed on the sizes tensor itself, which is only ever replaced (:meth:`load`), never written.
        """
        cached = self.__dict__.get("_longest_batch")
        if cached is None or cached[0] is not self._batch_sizes:
            value = int(self._batch_sizes.max().item()) if len(self._batch_sizes) else 0
            cached = self.__dict__["_longest_batch"] = (self._batch_sizes, value)
        return cached[1]

    @property
    def with_mask(self) -> _ConcatBatchTensorMaskView:
        return self._with_mask_view

    def __len__(self) -> int:
        return len(self._batch_sizes)

    def _as_int64_tensor(self, idx, length: int = 0) -> torch.Tensor:
        if isinstance(idx, slice):
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else length
            step = idx.step if idx.step is not None else 1
            if start < 0:
                start += length
            if stop < 0:
                stop += length
            return torch.arange(start, stop, step, dtype=torch.int64, device=self._device)
        if isinstance(idx, torch.Tensor):
            if idx.device == self._device and idx.dtype == torch.int64:
                return idx
            return idx.to(device=self._device, dtype=torch.int64)
        return torch.as_tensor(idx, dtype=torch.int64, device=self._device)

    def _get_batch_indices(self, batch_indices: torch.Tensor) -> torch.Tensor:
        starts = self._batch_starts[batch_indices]
        sizes = self._batch_sizes[batch_indices]
        total_size = int(torch.sum(sizes).item())

        if total_size == 0:
            return torch.empty(0, dtype=torch.int64, device=self._device)

        full_indices = torch.repeat_interleave(starts, sizes)
        cumsum_sizes = torch.cumsum(sizes, dim=0)
        offsets = torch.zeros(len(sizes), dtype=torch.int64, device=self._device)
        offsets[1:] = cumsum_sizes[:-1]

        relative_indices = torch.arange(total_size, device=self._device) - torch.repeat_interleave(offsets, sizes)
        return full_indices + relative_indices

    def _padded_slice_info(self, b_tensor: torch.Tensor, data_idx: slice, *, pad_to_longest_batch: bool = False):
        """Compute indexing info for multi-batch + slice access.

        Uses the max batch size among selected batches as the padded length — reading it back is a
        GPU->CPU sync — or, with ``pad_to_longest_batch``, the buffer's :attr:`longest_batch`, which is
        known on the host. Invalid positions (beyond each batch's actual size) are clamped to index 0.

        Returns
        -------
        d_tensor : torch.Tensor
            Data index tensor of shape (slice_len,).
        valid_mask : torch.Tensor
            Boolean mask of shape (num_batches, slice_len). True where data is real.
        flat_indices : torch.Tensor
            Flat indices into _concatenated_tensor, shape (num_batches, slice_len).
            Invalid entries are clamped to 0.
        """
        sizes = self._batch_sizes[b_tensor]
        if len(b_tensor) == 0:
            max_size = 0
        elif pad_to_longest_batch:
            max_size = self.longest_batch
        else:
            max_size = int(sizes.max().item())
        d_tensor = self._as_int64_tensor(data_idx, max_size)

        valid_mask = d_tensor.unsqueeze(0) < sizes.unsqueeze(1)
        batch_starts = self._batch_starts[b_tensor]
        flat_indices = batch_starts.unsqueeze(1) + d_tensor.unsqueeze(0)
        flat_indices = torch.where(valid_mask, flat_indices, torch.zeros_like(flat_indices))

        return d_tensor, valid_mask, flat_indices

    def _full_mask(self, shape: tuple | torch.Size) -> torch.Tensor:
        """Create an all-True boolean mask of the given shape."""
        return torch.ones(shape, dtype=torch.bool, device=self._device)

    def get_with_mask(self, idx, *, pad_to_longest_batch: bool = False):
        """Return (data, valid_mask) for all indexing patterns.

        The valid_mask indicates which entries in the result contain real data
        vs zero-padding. For non-padded access patterns the mask is all True.
        Prefer cbt.with_mask[...] when using indexing syntax.

        A padded result (several batches, whole) is as long as its longest *selected* batch, which is
        read back from the device, a GPU->CPU sync. ``pad_to_longest_batch=True`` pads to the
        buffer's :attr:`longest_batch` instead, for a hot path that would rather carry a few more
        padding columns than pay the sync.

        Mask shapes by indexing pattern:
            cbt[int]                  -> (batch_size_i,)
            cbt[int, int]             -> () (scalar True)
            cbt[int, slice]           -> (slice_len,)
            cbt[0d_tensor, slice]     -> (slice_len,)
            cbt[tensor, slice]        -> (num_batches, max_slice_len) — may contain False
            cbt[tensor, tensor]       -> (n,) or () if both 0-d
            cbt[slice] / cbt[list]    -> (num_batches, max_batch_len) — may contain False
        """
        if isinstance(idx, tuple):
            batch_idx = idx[0]
            data_idx = idx[1]

            # Optimization: both are int
            if isinstance(batch_idx, int) and isinstance(data_idx, int):
                res = self._concatenated_tensor[self._batch_starts[batch_idx] + data_idx]
                if len(idx) > 2:
                    res = res[idx[2:]]
                return res, self._full_mask(())

            # Optimization: batch_idx is int, data_idx is slice
            if isinstance(batch_idx, int) and isinstance(data_idx, slice):
                start = self._batch_starts[batch_idx]
                size = self._batch_sizes[batch_idx].item()
                s_start = data_idx.start if data_idx.start is not None else 0
                s_stop = data_idx.stop if data_idx.stop is not None else size
                s_step = data_idx.step if data_idx.step is not None else 1
                if s_start < 0:
                    s_start += size
                if s_stop < 0:
                    s_stop += size

                res = self._concatenated_tensor[start + s_start : start + s_stop : s_step]
                valid_mask = self._full_mask((res.shape[0],))
                if len(idx) > 2:
                    res = res[(slice(None),) + idx[2:]]
                return res, valid_mask

            # General case for tuple
            b_tensor = self._as_int64_tensor(batch_idx, len(self._batch_sizes))

            if isinstance(data_idx, slice):
                if b_tensor.ndim == 0:
                    d_len = self._batch_sizes[b_tensor].item()
                    d_tensor = self._as_int64_tensor(data_idx, d_len)
                    res = self._concatenated_tensor[self._batch_starts[b_tensor] + d_tensor]
                    valid_mask = self._full_mask((len(d_tensor),))
                else:
                    # Multi-batch + slice: pad to max batch size
                    d_tensor, valid_mask, flat_indices = self._padded_slice_info(b_tensor, data_idx)
                    num_batches = len(b_tensor)
                    slice_len = len(d_tensor)

                    # Gather and zero out invalid entries
                    res = self._concatenated_tensor[flat_indices.reshape(-1)]
                    res = res.reshape(num_batches, slice_len, *self._data_shape)
                    mask_ex = valid_mask.reshape(num_batches, slice_len, *([1] * len(self._data_shape)))
                    res = res * mask_ex

                    if len(idx) > 2:
                        res = res[(slice(None), slice(None)) + idx[2:]]
                    return res, valid_mask
            else:
                d_tensor = self._as_int64_tensor(data_idx, 0)
                if b_tensor.ndim == 0 and d_tensor.ndim > 0:
                    b_tensor = b_tensor.expand_as(d_tensor)
                elif d_tensor.ndim == 0 and b_tensor.ndim > 0:
                    d_tensor = d_tensor.expand_as(b_tensor)
                if b_tensor.ndim > 0:
                    sizes = self._batch_sizes[b_tensor]
                    d_tensor = torch.where(d_tensor < 0, sizes + d_tensor, d_tensor)
                    valid_mask = (d_tensor >= 0) & (d_tensor < sizes)
                    flat_indices = self._batch_starts[b_tensor] + torch.where(
                        valid_mask, d_tensor, torch.zeros_like(d_tensor)
                    )
                    res = self._concatenated_tensor[flat_indices]
                    if len(self._data_shape) > 0:
                        mask_ex = valid_mask.reshape(valid_mask.shape + (1,) * len(self._data_shape))
                        res = res * mask_ex
                    else:
                        res = res * valid_mask
                else:
                    size = self._batch_sizes[b_tensor]
                    d_resolved = torch.where(d_tensor < 0, size + d_tensor, d_tensor)
                    valid_mask = (d_resolved >= 0) & (d_resolved < size)
                    safe_d = torch.where(valid_mask, d_resolved, torch.zeros_like(d_resolved))
                    res = self._concatenated_tensor[self._batch_starts[b_tensor] + safe_d]
                    res = res * valid_mask

            if len(idx) > 2:
                if b_tensor.ndim == 0 and (not isinstance(data_idx, slice) and d_tensor.ndim == 0):
                    res = res[idx[2:]]
                else:
                    res = res[(slice(None),) + idx[2:]]
            return res, valid_mask

        if isinstance(idx, int):
            res = self._concatenated_tensor[self._batch_starts[idx] : self._batch_ends[idx]]
            return res, self._full_mask((res.shape[0],))

        if isinstance(idx, slice):
            b_tensor = self._as_int64_tensor(idx, len(self._batch_sizes))
            if b_tensor.ndim == 0:
                res = self._concatenated_tensor[self._batch_starts[b_tensor] : self._batch_ends[b_tensor]]
                return res, self._full_mask((res.shape[0],))
            d_tensor, valid_mask, flat_indices = self._padded_slice_info(
                b_tensor, slice(None), pad_to_longest_batch=pad_to_longest_batch
            )
            num_batches = len(b_tensor)
            slice_len = len(d_tensor)
            res = self._concatenated_tensor[flat_indices.reshape(-1)]
            res = res.reshape(num_batches, slice_len, *self._data_shape)
            mask_ex = valid_mask.reshape(num_batches, slice_len, *([1] * len(self._data_shape)))
            res = res * mask_ex
            return res, valid_mask

        if isinstance(idx, (list, torch.Tensor)):
            b_tensor = self._as_int64_tensor(idx)
            if b_tensor.ndim == 0:
                res = self._concatenated_tensor[self._batch_starts[b_tensor] : self._batch_ends[b_tensor]]
                return res, self._full_mask((res.shape[0],))
            d_tensor, valid_mask, flat_indices = self._padded_slice_info(
                b_tensor, slice(None), pad_to_longest_batch=pad_to_longest_batch
            )
            num_batches = len(b_tensor)
            slice_len = len(d_tensor)
            res = self._concatenated_tensor[flat_indices.reshape(-1)]
            res = res.reshape(num_batches, slice_len, *self._data_shape)
            mask_ex = valid_mask.reshape(num_batches, slice_len, *([1] * len(self._data_shape)))
            res = res * mask_ex
            return res, valid_mask

        raise TypeError(f"Invalid index type: {type(idx)}")

    def __getitem__(self, idx):
        """Return data only. Use get_with_mask() if mask is needed."""
        data, _ = self.get_with_mask(idx)
        return data

    def __setitem__(self, idx, data):
        if isinstance(idx, tuple):
            batch_idx, data_idx = idx[0], idx[1]
            if isinstance(batch_idx, int) and isinstance(data_idx, int):
                self._concatenated_tensor[self._batch_starts[batch_idx] + data_idx] = data
            elif isinstance(batch_idx, int) and isinstance(data_idx, slice):
                start = self._batch_starts[batch_idx]
                size = self._batch_sizes[batch_idx].item()
                s_start = data_idx.start if data_idx.start is not None else 0
                s_stop = data_idx.stop if data_idx.stop is not None else size
                s_step = data_idx.step if data_idx.step is not None else 1
                if s_start < 0:
                    s_start += size
                if s_stop < 0:
                    s_stop += size
                self._concatenated_tensor[start + s_start : start + s_stop : s_step] = data
            elif isinstance(data_idx, slice):
                # Non-int batch + slice
                b_tensor = self._as_int64_tensor(batch_idx)
                if b_tensor.ndim == 0:
                    # Single batch (0-d tensor) + slice: direct range write
                    d_len = int(self._batch_sizes[b_tensor].item())
                    d_tensor = self._as_int64_tensor(data_idx, d_len)
                    self._concatenated_tensor[self._batch_starts[b_tensor] + d_tensor] = data
                else:
                    # Multi-batch + slice: write only valid (non-padded) entries
                    d_tensor, valid_mask, flat_indices = self._padded_slice_info(b_tensor, data_idx)
                    valid_flat = flat_indices[valid_mask]
                    if (
                        isinstance(data, torch.Tensor)
                        and data.dim() >= 2
                        and data.shape[:2] == (len(b_tensor), len(d_tensor))
                    ):
                        # Padded data (num_batches, slice_len, ...): extract valid entries
                        self._concatenated_tensor[valid_flat] = data[valid_mask]
                    else:
                        self._concatenated_tensor[valid_flat] = data
            else:
                # Element-wise indexing: (tensor, tensor), (tensor, int), etc.
                b_tensor = self._as_int64_tensor(batch_idx)
                d_tensor = self._as_int64_tensor(data_idx)
                if b_tensor.ndim == 0 and d_tensor.ndim > 0:
                    b_tensor = b_tensor.expand_as(d_tensor)
                elif d_tensor.ndim == 0 and b_tensor.ndim > 0:
                    d_tensor = d_tensor.expand_as(b_tensor)
                self._concatenated_tensor[self._batch_starts[b_tensor] + d_tensor] = data
            return

        if isinstance(idx, int):
            self._concatenated_tensor[self._batch_starts[idx] : self._batch_ends[idx]] = data
        elif isinstance(idx, slice):
            self[(idx, slice(None))] = data
        elif isinstance(idx, (list, torch.Tensor)):
            if isinstance(idx, torch.Tensor) and idx.ndim == 0:
                self[int(idx.item())] = data
            else:
                self[(idx, slice(None))] = data
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    def fill_data(self, data: torch.Tensor):
        """Fill all the data dimension with the given tensor."""
        if data.shape != self._data_shape:
            raise ValueError(f"Expected shape {self._data_shape}, got {data.shape}")
        self._concatenated_tensor[:] = data.unsqueeze(0)

    def contiguous(self):
        return self._concatenated_tensor.contiguous()

    def select_batches(self, batch_indices: torch.Tensor | list[int] | slice) -> "ConcatBatchTensor":
        """Select a subset of batches and return a new compact ConcatBatchTensor.

        Parameters
        ----------
        batch_indices : torch.Tensor | list[int] | slice
            Indices of batches to select.

        Returns
        -------
        ConcatBatchTensor
            A new ConcatBatchTensor containing only the selected batches.
        """
        b_tensor = self._as_int64_tensor(batch_indices, len(self._batch_sizes))
        if b_tensor.ndim == 0:
            b_tensor = b_tensor.unsqueeze(0)

        new_sizes = self._batch_sizes[b_tensor]
        flat_indices = self._get_batch_indices(b_tensor)

        return ConcatBatchTensor._from_parts(self._concatenated_tensor[flat_indices], new_sizes, self._device)

    def unwrap_flattened_idx(self, flattened_idx: torch.Tensor):
        """Unwrap a flattened index into the original batch index and data index.

        Recommend the flattened_idx to have only 1 dimension.
        E.g.
            batch_starts =  [0, 3, 6, 9]
            batch_ends =    [3, 6, 9, 12]
            flattened_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            batch_idx =     [0, 0, 0, 1, 1, 1, 2, 2, 2, 3]
            data_idx =      [0, 1, 2, 0, 1, 2, 0, 1, 2, 0].
        """
        if flattened_idx.device != self._device:
            flattened_idx = flattened_idx.to(self._device)
        batch_idx = torch.searchsorted(self._batch_ends, flattened_idx, side="right")
        data_idx = flattened_idx - self._batch_starts[batch_idx]
        return batch_idx, data_idx

    def save(self, filepath: str):
        torch.save(
            {
                "concatenated_tensor": self._concatenated_tensor,
                "batch_sizes": self._batch_sizes,
            },
            filepath,
        )

    def load(self, filepath: str):
        data = torch.load(filepath, map_location=self._device, weights_only=True)
        self._concatenated_tensor = data["concatenated_tensor"]
        self._set_batch_sizes(data["batch_sizes"])
        self._data_shape = self._concatenated_tensor.shape[1:]

    def to(self, device: torch.device | str) -> "ConcatBatchTensor":
        """Move to device. Returns self if already on target device, a new object otherwise."""
        device = torch.device(device)
        if device == self._device:
            return self
        return ConcatBatchTensor._from_parts(self._concatenated_tensor.to(device), self._batch_sizes.to(device), device)

    def __repr__(self) -> str:
        return f"ConcatBatchTensor(num_batches={len(self)}, total_elements={self.shape[0]}, data_shape={self._data_shape}, device={self._device})"
