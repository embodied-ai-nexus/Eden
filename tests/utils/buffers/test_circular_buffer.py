"""Tests for ``eden.utils.buffers.circular_buffer.CircularBuffer``, the per-env history window."""

import pytest
import torch

from eden.utils.buffers.circular_buffer import CircularBuffer, latest_ring_slots


def _column(*values):
    """A ``(batch, 1)`` frame, one value per row."""
    return torch.tensor([[float(v)] for v in values])


def test_append_returns_history_oldest_first():
    buffer = CircularBuffer(max_len=3, batch_size=2, device="cpu")
    for t in (1, 2, 3):
        buffer.append(_column(t, 10 * t))

    assert buffer.buffer.squeeze(-1).tolist() == [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]


def test_append_past_capacity_overwrites_the_oldest():
    buffer = CircularBuffer(max_len=2, batch_size=1, device="cpu")
    for t in (1, 2, 3):
        buffer.append(_column(t))

    assert buffer.buffer.squeeze(-1).tolist() == [[2.0, 3.0]]


@pytest.mark.parametrize(("backfill", "expected"), [(True, [1.0, 1.0, 1.0]), (False, [0.0, 0.0, 1.0])])
def test_first_append_backfills_the_history_only_when_asked(backfill, expected):
    buffer = CircularBuffer(max_len=3, batch_size=1, device="cpu", backfill=backfill)
    buffer.append(_column(1))

    assert buffer.buffer.squeeze(-1).tolist() == [expected]
    assert buffer.current_length.tolist() == [1]


def test_current_length_counts_per_row_and_clamps_to_max_len():
    buffer = CircularBuffer(max_len=4, batch_size=3, device="cpu")
    frame = torch.arange(3.0).unsqueeze(-1)
    buffer.append(frame)
    buffer.append(frame)
    buffer.reset(batch_ids=[1])
    assert buffer.current_length.tolist() == [2, 0, 2]

    buffer.append(frame)
    assert buffer.current_length.tolist() == [3, 1, 3]

    for _ in range(5):
        buffer.append(frame)
    assert buffer.current_length.tolist() == [4, 4, 4]


def test_a_row_reset_zeroes_it_and_its_next_append_backfills_it():
    buffer = CircularBuffer(max_len=3, batch_size=2, device="cpu")
    buffer.append(_column(1, 10))
    buffer.append(_column(2, 20))

    buffer.reset(batch_ids=torch.tensor([1]))
    assert buffer.buffer.squeeze(-1).tolist() == [[1.0, 1.0, 2.0], [0.0, 0.0, 0.0]]

    buffer.append(_column(3, 99))
    assert buffer.buffer.squeeze(-1).tolist() == [[1.0, 2.0, 3.0], [99.0, 99.0, 99.0]]


def test_reset_all_zeroes_every_row_and_counter():
    buffer = CircularBuffer(max_len=3, batch_size=2, device="cpu")
    buffer.append(_column(1, 2))
    buffer.append(_column(3, 4))

    buffer.reset()

    assert buffer.current_length.tolist() == [0, 0]
    assert torch.count_nonzero(buffer.buffer) == 0


def test_getitem_reads_a_per_row_lag_clamped_to_the_oldest_frame():
    buffer = CircularBuffer(max_len=3, batch_size=2, device="cpu")
    for t in (1, 2, 3):
        buffer.append(_column(t, 10 * t))

    assert buffer[torch.tensor([0, 2])].squeeze(-1).tolist() == [3.0, 10.0]
    assert buffer[torch.tensor([99, 1])].squeeze(-1).tolist() == [1.0, 20.0]


def test_errors():
    buffer = CircularBuffer(max_len=2, batch_size=2, device="cpu")
    with pytest.raises(ValueError):
        buffer.append(_column(1))  # batch of 1 into a batch-2 buffer
    with pytest.raises(ValueError):
        buffer.peek_buffer(_column(1))
    with pytest.raises(RuntimeError):
        _ = buffer.buffer  # nothing appended yet
    with pytest.raises(RuntimeError):
        _ = buffer[torch.tensor([0, 0])]

    buffer.append(_column(1, 2))
    with pytest.raises(ValueError):
        _ = buffer[torch.tensor([0])]  # one lag for two rows


def test_first_append_inside_inference_mode_leaves_a_resettable_buffer():
    """rsl_rl wraps ``env.step`` in ``torch.inference_mode``; a later reset must not hit an inference tensor."""
    buffer = CircularBuffer(max_len=3, batch_size=2, device="cpu")
    with torch.inference_mode():
        buffer.append(torch.ones(2, 4))

    buffer.reset()
    buffer.append(torch.zeros(2, 4))
    assert buffer.buffer.shape == (2, 3, 4)


def _peek_scenarios():
    """Named ``(max_len, batch_size, backfill, setup)`` states to peek from."""

    def filled(buf):
        for v in (1.0, 2.0, 3.0):
            buf.append(torch.full((buf.batch_size, 1), v) + torch.arange(float(buf.batch_size)).unsqueeze(-1))

    def wrapped(buf):
        for v in (1.0, 2.0, 3.0, 4.0, 5.0):
            buf.append(torch.full((buf.batch_size, 1), v))

    def row_reset(buf):
        filled(buf)
        buf.reset(batch_ids=[1])

    return {
        "filling": (4, 2, True, filled),
        "wrapped": (3, 1, True, wrapped),
        "unallocated": (3, 2, True, lambda buf: None),
        "row-reset": (3, 2, True, row_reset),
        "no-backfill": (3, 1, False, lambda buf: None),
    }


@pytest.mark.parametrize("scenario", list(_peek_scenarios()))
def test_peek_buffer_is_the_view_append_would_give_without_mutating(scenario):
    max_len, batch_size, backfill, setup = _peek_scenarios()[scenario]
    peeked_from = CircularBuffer(max_len=max_len, batch_size=batch_size, device="cpu", backfill=backfill)
    appended_to = CircularBuffer(max_len=max_len, batch_size=batch_size, device="cpu", backfill=backfill)
    setup(peeked_from)
    setup(appended_to)
    frame = torch.full((batch_size, 1), 42.0) + torch.arange(float(batch_size)).unsqueeze(-1)
    before = (peeked_from.buffer.clone() if peeked_from.is_initialized else None, peeked_from.current_length.clone())

    peeked = peeked_from.peek_buffer(frame)
    appended_to.append(frame)

    assert torch.equal(peeked, appended_to.buffer)
    if before[0] is None:
        assert not peeked_from.is_initialized
    else:
        assert torch.equal(peeked_from.buffer, before[0])
    assert torch.equal(peeked_from.current_length, before[1])


def test_peek_buffer_after_wraparound_is_chronological():
    buffer = CircularBuffer(max_len=3, batch_size=1, device="cpu")
    for v in (1, 2, 3, 4, 5):
        buffer.append(_column(v))
    assert buffer.peek_buffer(_column(6)).squeeze(-1).tolist() == [[4.0, 5.0, 6.0]]


def test_has_frames_tracks_content_while_is_initialized_tracks_storage():
    """``ObservationManager`` keys its history guard on ``has_frames``; reset keeps the storage but empties it."""
    buffer = CircularBuffer(max_len=3, batch_size=4, device="cpu")
    assert not buffer.has_frames and not buffer.is_initialized

    buffer.append(torch.ones(4, 2))
    assert buffer.has_frames and buffer.is_initialized

    buffer.reset(batch_ids=[0, 1])
    assert buffer.has_frames, "the un-reset rows still hold frames"

    buffer.reset()
    assert buffer.is_initialized and not buffer.has_frames


def _drive(buffer: CircularBuffer, reset_rows) -> dict[str, torch.Tensor]:
    """Append, reset ``reset_rows``, append again, then read back everything a caller can observe."""
    buffer.append(torch.arange(12.0).reshape(4, 3))
    buffer.append(torch.arange(12.0).reshape(4, 3) + 100)
    buffer.reset(batch_ids=reset_rows)
    peeked = buffer.peek_buffer(torch.full((4, 3), -7.0))
    buffer.append(torch.full((4, 3), -1.0))
    return {"buffer": buffer.buffer, "lag1": buffer[1], "current_length": buffer.current_length, "peeked": peeked}


@pytest.mark.parametrize("backfill", [True, False])
def test_reset_by_mask_matches_reset_by_index(backfill):
    """Auto-reset passes a bool mask, which takes a sync-free ``masked_fill_`` path instead of indexing."""
    mask = torch.tensor([True, False, True, False])
    by_mask = _drive(CircularBuffer(max_len=3, batch_size=4, device="cpu", backfill=backfill), mask)
    by_index = _drive(CircularBuffer(max_len=3, batch_size=4, device="cpu", backfill=backfill), [0, 2])
    for key, expected in by_index.items():
        torch.testing.assert_close(by_mask[key], expected, msg=key)


def test_the_per_step_reset_peek_append_cycle_is_sync_free(count_cuda_syncs):
    """Indexing with the auto-reset mask would be a ``nonzero`` — a GPU sync per step per history buffer."""
    buffer = CircularBuffer(max_len=4, batch_size=8, device="cuda")
    frame = torch.randn(8, 5, device="cuda")
    buffer.append(frame)  # the first append allocates; measure the steady state
    mask = torch.arange(8, device="cuda") % 3 == 0

    def step():
        buffer.reset(batch_ids=mask)
        buffer.peek_buffer(frame)
        buffer.append(frame)
        assert buffer.has_frames

    assert count_cuda_syncs(step) == 0


@pytest.mark.parametrize("max_len", [1, 2, 4])
@pytest.mark.parametrize("backfill", [True, False])
def test_peek_oldest_is_the_oldest_slot_of_peek_buffer(max_len, backfill):
    """Every state a delay passes through: unallocated, filling, wrapped, and after a mask and an index reset."""
    buffer = CircularBuffer(max_len=max_len, batch_size=4, device="cpu", backfill=backfill)
    frames = [torch.full((4, 3), float(i)) + torch.arange(4.0).unsqueeze(-1) for i in range(1, 9)]

    def check(frame):
        torch.testing.assert_close(buffer.peek_oldest(frame), buffer.peek_buffer(frame)[:, 0])

    check(frames[0])
    for i, frame in enumerate(frames):
        buffer.append(frame)
        if i == 4:
            buffer.reset(batch_ids=torch.tensor([True, False, True, False]))
        if i == 6:
            buffer.reset(batch_ids=torch.tensor([3]))
        check(frames[(i + 1) % len(frames)])


def test_peek_oldest_does_not_alias_the_storage():
    buffer = CircularBuffer(max_len=3, batch_size=2, device="cpu")
    for i in range(3):
        buffer.append(torch.full((2, 1), float(i)))

    oldest = buffer.peek_oldest(torch.full((2, 1), 3.0))
    buffer.append(torch.full((2, 1), 9.0))
    buffer.append(torch.full((2, 1), 9.0))
    assert oldest.flatten().tolist() == [1.0, 1.0]


def test_the_latest_control_step_is_read_off_the_sample_ring_oldest_first():
    assert latest_ring_slots(10, 10) == list(range(10))
    assert latest_ring_slots(4, 10) == [0, 1, 2, 3]  # a step cut short with nothing after it
    # A step cut short after 5 samples, then a full one: its 10 samples are the latest, from slot 5 round to 4.
    assert latest_ring_slots(15, 10) == [5, 6, 7, 8, 9, 0, 1, 2, 3, 4]
