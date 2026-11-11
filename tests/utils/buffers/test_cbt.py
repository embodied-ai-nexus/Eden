"""Tests for ``eden.utils.buffers.cbt.ConcatBatchTensor``: ragged batches in one concatenated tensor."""

import pytest
import torch

from eden.utils.buffers.cbt import ConcatBatchTensor

# Three batches of 3, 5 and 2 frames, each frame 4 wide, values distinct across the whole tensor.
TENSORS = [
    torch.arange(12.0).reshape(3, 4),
    100 + torch.arange(20.0).reshape(5, 4),
    200 + torch.arange(8.0).reshape(2, 4),
]


@pytest.fixture
def cbt():
    return ConcatBatchTensor(tensors=[t.clone() for t in TENSORS])


def _padded(indices, length):
    """The selected batches zero-padded to ``length`` frames, and the matching validity mask."""
    data = torch.zeros(len(indices), length, 4)
    mask = torch.zeros(len(indices), length, dtype=torch.bool)
    for row, i in enumerate(indices):
        n = TENSORS[i].shape[0]
        data[row, :n], mask[row, :n] = TENSORS[i], True
    return data, mask


def test_construction_concatenates_the_batches(cbt):
    assert len(cbt) == 3
    assert cbt.shape == (10, 4)
    assert cbt.batch_sizes.tolist() == [3, 5, 2]
    assert torch.equal(cbt.concatenated_tensor, torch.cat(TENSORS))


@pytest.mark.parametrize(
    ("idx", "expected"),
    [
        (1, TENSORS[1]),
        ((0, 1), TENSORS[0][1]),
        ((1, 4), TENSORS[1][4]),
        ((0, slice(0, 2)), TENSORS[0][0:2]),
        ((1, slice(None)), TENSORS[1]),
        (
            (torch.tensor([0, 1, 2, 1]), torch.tensor([0, 4, 1, 0])),
            torch.stack([TENSORS[0][0], TENSORS[1][4], TENSORS[2][1], TENSORS[1][0]]),
        ),
        ((torch.tensor(0), torch.tensor(-1)), TENSORS[0][-1]),  # negative index resolves within the batch
        ((torch.tensor(2), torch.tensor(-1)), TENSORS[2][-1]),
        ((slice(0, 2), 1), torch.stack([TENSORS[0][1], TENSORS[1][1]])),
        ((slice(0, 2), slice(0, 2)), torch.stack([TENSORS[0][0:2], TENSORS[1][0:2]])),
        (([0, 1], slice(0, 2)), torch.stack([TENSORS[0][0:2], TENSORS[1][0:2]])),
    ],
    ids=[
        "int",
        "int-int",
        "int-int-last",
        "int-slice",
        "int-full",
        "tensor-tensor",
        "neg-first",
        "neg-last",
        "slice-int",
        "slice-slice",
        "list-slice",
    ],
)
def test_an_in_range_selection_is_the_data_with_an_all_true_mask(cbt, idx, expected):
    data, mask = cbt.with_mask[idx]
    assert torch.equal(data, expected)
    assert mask.shape == expected.shape[: mask.ndim] and mask.all()


def test_an_out_of_range_negative_index_is_zeroed_and_masked_out(cbt):
    data, mask = cbt.with_mask[torch.tensor(2), torch.tensor(-3)]  # batch 2 has 2 frames
    assert not mask.item()
    assert torch.equal(data, torch.zeros(4))


@pytest.mark.parametrize(
    ("idx", "indices", "length"),
    [
        (slice(0, 2), [0, 1], 5),
        ([0, 2], [0, 2], 3),
        (([0, 1, 2], slice(None)), [0, 1, 2], 5),
        (([0, 2], slice(None)), [0, 2], 3),
    ],
)
def test_a_multi_batch_selection_pads_to_the_longest_with_zeros(cbt, idx, indices, length):
    data, mask = cbt.with_mask[idx]
    expected_data, expected_mask = _padded(indices, length)
    assert torch.equal(data, expected_data)
    assert torch.equal(mask, expected_mask)


@pytest.mark.parametrize("idx", [[0, 2], torch.tensor([2, 0]), slice(0, 1)], ids=["list", "tensor", "slice"])
def test_pad_to_longest_batch_pads_to_the_buffers_longest_not_the_selections(cbt, idx):
    """[0, 2] alone would pad to 3 frames; the buffer's longest batch has 5."""
    indices = list(range(3))[idx] if isinstance(idx, slice) else [int(i) for i in idx]
    data, mask = cbt.get_with_mask(idx, pad_to_longest_batch=True)
    expected_data, expected_mask = _padded(indices, 5)
    assert cbt.longest_batch == 5
    assert torch.equal(data, expected_data)
    assert torch.equal(mask, expected_mask)


def test_pad_to_longest_batch_is_sync_free_once_the_length_is_read(count_cuda_syncs):
    cbt = ConcatBatchTensor(tensors=[t.clone() for t in TENSORS], device="cuda")
    idx = torch.tensor([0, 2], device="cuda")
    assert count_cuda_syncs(lambda: cbt.get_with_mask(idx)) >= 1  # the selection's longest batch, read back
    cbt.get_with_mask(idx, pad_to_longest_batch=True)
    assert count_cuda_syncs(lambda: cbt.get_with_mask(idx, pad_to_longest_batch=True)) == 0


def test_indexing_a_trailing_dim_keeps_the_frame_mask(cbt):
    data, mask = cbt.with_mask[[0, 1, 2], :, 0]
    expected_data, expected_mask = _padded([0, 1, 2], 5)
    assert torch.equal(data, expected_data[..., 0])
    assert torch.equal(mask, expected_mask)


def test_equal_batches_are_all_valid():
    data, mask = ConcatBatchTensor(tensors=[torch.ones(4, 3)] * 3).with_mask[[0, 1, 2], :]
    assert data.shape == (3, 4, 3) and mask.all()


@pytest.mark.parametrize(
    ("idx", "new_data"),
    [(1, torch.full((5, 4), -1.0)), ((1, slice(1, 3)), torch.full((2, 4), -1.0))],
    ids=["int", "int-slice"],
)
def test_setitem_writes_exactly_the_selected_frames(cbt, idx, new_data):
    cbt[idx] = new_data
    assert torch.equal(cbt[idx], new_data)
    assert torch.equal(cbt[0], TENSORS[0]) and torch.equal(cbt[2], TENSORS[2])


@pytest.mark.parametrize(("idx", "indices"), [(slice(0, 2), [0, 1]), ([0, 2], [0, 2])])
def test_setitem_over_several_batches_takes_the_flat_valid_frames(cbt, idx, indices):
    new_data = -1 - torch.arange(float(sum(TENSORS[i].shape[0] for i in indices) * 4)).reshape(-1, 4)
    cbt[idx] = new_data
    data, mask = cbt.with_mask[idx]
    assert torch.equal(data[mask], new_data)
    assert not data[~mask].any()


def test_a_padded_read_written_back_round_trips(cbt):
    data, mask = cbt.with_mask[[0, 1, 2], :]
    cbt[[0, 1, 2], :] = data * 2.0
    data_back, mask_back = cbt.with_mask[[0, 1, 2], :]
    assert torch.equal(mask_back, mask)
    assert torch.equal(data_back[mask], (data * 2.0)[mask])


def test_unwrap_flattened_idx_gives_batch_and_frame(cbt):
    batch_idx, data_idx = cbt.unwrap_flattened_idx(torch.tensor([0, 2, 3, 7, 8, 9]))
    assert batch_idx.tolist() == [0, 0, 1, 1, 2, 2]
    assert data_idx.tolist() == [0, 2, 0, 4, 0, 1]


@pytest.mark.parametrize(
    ("selection", "indices"),
    [
        ([0, 2], [0, 2]),
        (torch.tensor([1, 2]), [1, 2]),
        (slice(0, 2), [0, 1]),
        ([1], [1]),
        ([2, 1, 0], [2, 1, 0]),
        (torch.tensor(1), [1]),
        ([0, 0], [0, 0]),
    ],
    ids=["list", "tensor", "slice", "single", "reversed", "scalar-tensor", "duplicates"],
)
def test_select_batches_returns_a_compact_copy_of_those_batches(cbt, selection, indices):
    selected = cbt.select_batches(selection)
    assert selected.batch_sizes.tolist() == [TENSORS[i].shape[0] for i in indices]
    for row, i in enumerate(indices):
        assert torch.equal(selected[row], TENSORS[i])
    assert torch.equal(selected.with_mask[len(indices) - 1, :][0], TENSORS[indices[-1]])


def test_select_batches_does_not_alias_the_original(cbt):
    selected = cbt.select_batches([0])
    selected[0] = torch.zeros(3, 4)
    assert torch.equal(cbt[0], TENSORS[0])


def test_save_then_load_round_trips(cbt, tmp_path):
    path = tmp_path / "cbt.pt"
    cbt.save(str(path))

    loaded = ConcatBatchTensor(tensors=[torch.zeros(1, 4)])
    loaded.load(str(path))

    assert loaded.batch_sizes.tolist() == [3, 5, 2]
    for i, expected in enumerate(TENSORS):
        assert torch.equal(loaded[i], expected)


class TestFromConcatenated:
    """The zero-copy wrap the mmap'd dataset-cache load uses."""

    def test_wraps_the_storage_without_copying_and_indexes_like_a_fresh_cbt(self, cbt):
        wrapped = ConcatBatchTensor.from_concatenated(cbt.concatenated_tensor, batch_sizes=cbt.batch_sizes)
        for i in range(3):
            assert torch.equal(wrapped[i], cbt[i])
        cbt.concatenated_tensor[0, 0] = 1234.0
        assert wrapped.concatenated_tensor[0, 0].item() == 1234.0

    def test_a_size_mismatch_raises(self):
        with pytest.raises(ValueError, match="does not match"):
            ConcatBatchTensor.from_concatenated(torch.randn(5, 4), batch_sizes=torch.tensor([3, 5, 2]))

    def test_a_device_other_than_the_tensors_raises(self):
        """Index buffers on another device than the data would surface later as opaque indexing errors."""
        with pytest.raises(ValueError, match="does not match"):
            ConcatBatchTensor.from_concatenated(torch.randn(5, 4), batch_sizes=torch.tensor([3, 2]), device="meta")
