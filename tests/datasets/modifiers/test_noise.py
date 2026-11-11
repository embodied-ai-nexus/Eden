"""Tests for ``eden.datasets.modifiers.noise.Jitter``."""

import pytest
import torch

from eden.datasets.modifiers import Jitter


def test_jitter_adds_noise_of_the_configured_std(make_sequence):
    torch.manual_seed(0)
    seq = make_sequence(buffer_lengths=(50, 50))
    before = seq.dofs_pos.concatenated_tensor.clone()

    Jitter(options=Jitter.configure(fields=("dofs_pos",), std=0.1)).apply_to_sequence(seq)

    assert (seq.dofs_pos.concatenated_tensor - before).std().item() == pytest.approx(0.1, rel=0.2)


def test_jitter_with_zero_std_is_a_noop(make_sequence):
    seq = make_sequence()
    before = seq.dofs_pos.concatenated_tensor.clone()
    Jitter(options=Jitter.configure(fields=("dofs_pos",), std=0.0)).apply_to_sequence(seq)
    assert torch.equal(seq.dofs_pos.concatenated_tensor, before)


@pytest.mark.parametrize(
    ("field", "message"),
    [("framerate", "ConcatBatchTensor"), ("does_not_exist", "not populated")],
    ids=["per-clip-field", "missing-field"],
)
def test_jitter_refuses_a_field_it_cannot_perturb_per_frame(make_sequence, field, message):
    with pytest.raises(ValueError, match=message):
        Jitter(options=Jitter.configure(fields=(field,), std=0.1)).apply_to_sequence(make_sequence())
