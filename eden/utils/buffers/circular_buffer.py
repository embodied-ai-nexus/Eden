# Adapted from https://github.com/mujocolab/mjlab/blob/main/src/mjlab/utils/buffers/circular_buffer.py

"""Fixed-length circular buffer for batched tensor history.

Stores ``max_len`` frames per environment with shape
``(max_len, batch_size, ...)`` internally (time-first for pointer arithmetic)
and exposes a ``(batch_size, max_len, ...)`` view via the :attr:`buffer`
property (batch-first, chronological oldest-to-newest).

When backfill is enabled (default), the first append after construction or
reset copies the new value into every history slot so that downstream
consumers always see valid data instead of zeros.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from eden.utils.misc import is_env_mask


class CircularBuffer:
    """Fixed-length circular buffer for batched tensor history.

    Stores history with shape ``(max_len, batch_size, ...)`` internally.
    The :attr:`buffer` property returns chronologically ordered data
    (oldest to newest) with shape ``(batch_size, max_len, ...)``.

    LIFO retrieval via ``__getitem__``::

        buffer[0]  # most recent frame
        buffer[1]  # one step back
        buffer[2]  # two steps back (oldest if max_len=3)

    Per-batch :meth:`reset` zeroes the specified rows and marks them
    for backfill on the next :meth:`append`.

    Use :meth:`peek_buffer` to obtain the chronological view that *would*
    result from appending a frame, without mutating internal state — useful
    when a caller needs a hypothetical post-append snapshot but the
    canonical advance is owned by another code path (e.g. the observation
    manager's ``update_history=False`` snapshot pass during
    ``record_final_observations``).

    Parameters
    ----------
    max_len : int
        Maximum number of historical frames to retain.
    batch_size : int
        Size of the batch dimension (number of environments).
    device : str
        Torch device for storage.
    backfill : bool, optional
        If True (default), the first append after construction or reset
        copies the new value into every history slot.
    """

    def __init__(self, max_len: int, batch_size: int, device: str, backfill: bool = True) -> None:
        if max_len < 1:
            raise ValueError(f"Buffer size must be >= 1, got {max_len}")

        self._max_len = max_len
        self._batch_size = batch_size
        self._device = device
        self._pointer: int = -1
        self._buffer: torch.Tensor | None = None
        self._all_indices = torch.arange(batch_size, device=device)
        self._num_pushes = torch.zeros(batch_size, dtype=torch.long, device=device)
        self._max_len_tensor = torch.full((batch_size,), max_len, dtype=torch.long, device=device)
        self._backfill = backfill
        self._has_unfilled_rows = True  # CPU-side flag to avoid GPU sync in append()
        self._has_frames = False  # CPU-side flag, same reason: see :attr:`has_frames`

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> str:
        return self._device

    @property
    def max_length(self) -> int:
        return self._max_len

    @property
    def current_length(self) -> torch.Tensor:
        """Per-batch count of valid frames. Shape: (batch_size,)."""
        return torch.minimum(self._num_pushes, self._max_len_tensor)

    @property
    def is_initialized(self) -> bool:
        """Check if the buffer has been initialized with at least one append.

        Storage question — whether :attr:`buffer` can be read. Not the same as :attr:`has_frames`:
        :meth:`reset` zeroes rows in place and keeps the allocation, so this stays ``True`` across a
        reset. Callers asking "does this window hold real data" want :attr:`has_frames`.
        """
        return self._buffer is not None

    @property
    def has_frames(self) -> bool:
        """Whether any frame appended since the last full :meth:`reset` is still in the window.

        Content question, as opposed to :attr:`is_initialized`'s storage question. ``False`` on a
        fresh buffer and after a ``batch_ids=None`` reset (every row zeroed, and backfill will refill
        the whole window from the next frame); ``True`` after any append, including after a *partial*
        reset, because the un-reset rows still hold their frames.

        A CPU-side flag rather than a read of ``_num_pushes``, for the reason ``_has_unfilled_rows``
        is: this must stay sync-free so it is safe to consult from anywhere. It is deliberately
        conservative — an explicit reset listing every index leaves it ``True``.
        """
        return self._has_frames

    @property
    def buffer(self) -> torch.Tensor:
        """History in chronological order (oldest to newest).

        Returns
        -------
        torch.Tensor
            Shape ``(batch_size, max_len, ...)``, index 0 is oldest.
        """
        if self._buffer is None:
            raise RuntimeError("Buffer not initialized. Call append() first.")

        start = (self._pointer + 1) % self._max_len
        idx = (torch.arange(self._max_len, device=self._device) + start) % self._max_len
        buf = self._buffer.index_select(0, idx)  # (max_len, batch, ...)
        return buf.transpose(0, 1)  # (batch, max_len, ...)

    def reset(self, batch_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Zero out values and counters for specified batch rows.

        Parameters
        ----------
        batch_ids : Sequence[int] | torch.Tensor | None, optional
            Batch indices to reset, or ``None`` to reset all.
        """
        if batch_ids is None:
            # Every row cleared, so the window holds nothing a caller could still be relying on.
            # A partial reset leaves the flag set: the un-reset rows keep their frames.
            self._has_frames = False
        if is_env_mask(batch_ids):
            # A mask — what auto-reset passes, every step — is applied with ``masked_fill_``: indexing
            # with it (``self._buffer[:, mask]``) is a ``nonzero``, a GPU sync.
            self._num_pushes.masked_fill_(batch_ids, 0)
            if self._buffer is not None:
                self._buffer.masked_fill_(self._rows(batch_ids), 0.0)
        else:
            ids: Sequence[int] | torch.Tensor | slice = slice(None) if batch_ids is None else batch_ids
            self._num_pushes[ids] = 0
            if self._buffer is not None:
                self._buffer[:, ids] = 0.0
        if self._backfill:
            self._has_unfilled_rows = True

    def _rows(self, mask: torch.Tensor) -> torch.Tensor:
        """``mask`` of shape ``(batch_size,)`` shaped to broadcast over the ``(max_len, batch_size, ...)`` storage."""
        return mask.view(1, -1, *([1] * (self._buffer.ndim - 2)))

    def append(self, data: torch.Tensor) -> None:
        """Append a new frame for all batch elements.

        Parameters
        ----------
        data : torch.Tensor
            Tensor of shape ``(batch_size, ...)``.
        """
        if data.shape[0] != self._batch_size:
            raise ValueError(f"Expected batch size {self._batch_size}, got {data.shape[0]}")

        data = data.to(self._device)
        self._has_frames = True

        if self._buffer is None:
            self._pointer = -1
            # Allocate the storage OUTSIDE inference mode so it is a normal tensor even when the
            # first append happens inside the rsl_rl rollout (which wraps env.step in
            # torch.inference_mode). Otherwise the storage would become an inference tensor and a
            # later reset()/append() outside inference mode would raise "Inplace update to inference
            # tensor outside InferenceMode is not allowed". Writing normal-buffer
            # cells from inference-tensor data is always allowed, so only the alloc needs the guard.
            with torch.inference_mode(False):
                self._buffer = torch.zeros((self._max_len, *data.shape), dtype=data.dtype, device=self._device)

        self._pointer = (self._pointer + 1) % self._max_len
        self._buffer[self._pointer] = data

        # Backfill entire history with first frame for newly initialized batches.
        # Use CPU-side flag to avoid torch.any() GPU→CPU sync every step, and ``where`` rather than
        # indexing with the mask, which would be one too.
        if self._backfill and self._has_unfilled_rows:
            is_first_push = self._rows(self._num_pushes == 0)
            self._buffer.copy_(torch.where(is_first_push, data.unsqueeze(0), self._buffer))
            self._has_unfilled_rows = False

        self._num_pushes += 1

    def peek_buffer(self, data: torch.Tensor) -> torch.Tensor:
        """Return the buffer that *would* result from ``append(data)``, without mutating state.

        Used by the observation manager when snapshotting a history-bearing
        group with ``update_history=False``: the freshly-computed frame must
        appear in the most-recent slot of the returned view, but the underlying
        buffer state is owned by the once-per-step canonical advance elsewhere.

        Parameters
        ----------
        data : torch.Tensor
            Tensor of shape ``(batch_size, ...)``, same shape as ``append``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch_size, max_len, ...)``, oldest to newest, with
            ``data`` occupying the newest slot. Independent of internal storage.
        """
        if data.shape[0] != self._batch_size:
            raise ValueError(f"Expected batch size {self._batch_size}, got {data.shape[0]}")

        data = data.to(self._device)

        if self._buffer is None:
            # Hypothetical first append on an uninitialized buffer. Stand in
            # with zero storage at pointer = -1; the per-row backfill below
            # then reproduces append()'s ``backfill`` semantics.
            storage = torch.zeros((self._max_len, *data.shape), dtype=data.dtype, device=self._device)
            next_pointer = 0
        else:
            storage = self._buffer
            next_pointer = (self._pointer + 1) % self._max_len

        start = (next_pointer + 1) % self._max_len
        idx = (torch.arange(self._max_len, device=self._device) + start) % self._max_len
        buf = storage.index_select(0, idx)  # (max_len, batch, ...) — independent copy
        buf[-1] = data  # most-recent slot

        # Mirror append()'s per-row backfill for batches still on their first push.
        if self._backfill and self._has_unfilled_rows:
            is_first_push = (self._num_pushes == 0).view(1, -1, *([1] * (buf.ndim - 2)))
            buf = torch.where(is_first_push, data.unsqueeze(0), buf)

        return buf.transpose(0, 1)  # (batch, max_len, ...)

    def peek_oldest(self, data: torch.Tensor) -> torch.Tensor:
        """Return ``peek_buffer(data)[:, 0]``, the oldest frame of the window after ``append(data)``, alone.

        That is the frame appended ``max_len - 1`` pushes before ``data``, under the same backfill
        rules: a row still on its first push gets ``data``, and a row with too few frames gets zeros
        when backfill is off. Reading the one slot rather than building the whole window is what makes
        a delay of ``max_len - 1`` frames cost one frame per read
        (:class:`~eden.managers.modifiers.observations.temporal.ObservationDelay`).

        Parameters
        ----------
        data : torch.Tensor
            Tensor of shape ``(batch_size, ...)``, same shape as ``append``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch_size, ...)``. Independent of internal storage, so a later ``append`` does not
            change it.
        """
        if data.shape[0] != self._batch_size:
            raise ValueError(f"Expected batch size {self._batch_size}, got {data.shape[0]}")

        data = data.to(self._device)
        if self._max_len == 1:
            return data.clone()  # the window after the append holds only ``data``

        if self._buffer is None:
            oldest = torch.zeros_like(data)  # as peek_buffer: zero storage standing in for the first append
        else:
            # The append would write slot ``pointer + 1``; the window then starts at the slot after that.
            oldest = self._buffer[(self._pointer + 2) % self._max_len].clone()

        if self._backfill and self._has_unfilled_rows:
            is_first_push = (self._num_pushes == 0).view(-1, *([1] * (data.ndim - 1)))
            oldest = torch.where(is_first_push, data, oldest)
        return oldest

    def __getitem__(self, key: torch.Tensor | int) -> torch.Tensor:
        """Retrieve lagged frames per batch (LIFO).

        Parameters
        ----------
        key : torch.Tensor | int
            Per-batch lags of shape ``(batch_size,)`` or a shared scalar lag. A lag past the frames pushed
            since the row's last reset returns its oldest pushed frame.

        Returns
        -------
        torch.Tensor
            Lagged frames with shape ``(batch_size, ...)``.
        """
        if self._buffer is None:
            raise RuntimeError("Buffer not initialized. Call append() first.")

        if isinstance(key, int):
            key = torch.full((self._batch_size,), key, dtype=torch.long, device=self._device)
        else:
            if key.ndim == 0:
                key = key.expand(self._batch_size)
            key = key.to(device=self._device, dtype=torch.long)

        if key.numel() != self._batch_size:
            raise ValueError(f"Expected {self._batch_size} lags, got {key.numel()}")

        pushes = self._num_pushes.clamp_min(1)
        valid = torch.minimum(key, pushes - 1).clamp_min(0)

        # Always gather: an all-lags-zero shortcut needs a ``torch.all`` host read, a GPU sync on
        # every call — and ``ActionDelay`` calls this once per physics step.
        idx = torch.remainder(self._pointer - valid, self._max_len)
        return self._buffer[idx, self._all_indices]


def latest_ring_slots(sampled: int, size: int) -> list[int]:
    """The slots, oldest first, of the latest ``min(sampled, size)`` of ``sampled`` samples written in turn to a ring
    of ``size`` slots starting at slot 0 (sample ``k`` at slot ``k % size``).
    """
    n = min(sampled, size)
    return [(sampled - n + j) % size for j in range(n)]
