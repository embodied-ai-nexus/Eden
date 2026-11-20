"""Replay-buffer windows respect episode ends."""

import pytest
import torch
from tensordict import TensorDict

from eden.learning.wrappers.rsl_rl.supermimic.replay_buffer import FieldSpec, ReplayBuffer

SCHEMA = {
    "t": FieldSpec(dtype=torch.long),
    "terminated": FieldSpec(dtype=torch.bool),
    "truncated": FieldSpec(dtype=torch.bool),
}


def _filled(steps: int, ends: set[int], buffer_size: int = 16) -> ReplayBuffer:
    buffer = ReplayBuffer(SCHEMA, buffer_size, num_envs=1, device="cpu")
    for t in range(steps):
        done = torch.tensor([t in ends])
        buffer.insert(TensorDict({"t": torch.tensor([t]), "terminated": done, "truncated": torch.tensor([False])}, [1]))
    return buffer


def test_windows_are_consecutive_and_never_hold_an_end_before_their_last_step():
    buffer = _filled(steps=20, ends={6, 11})
    windows = buffer.sample(sequence_length=4, batch_size=500)
    t, done = windows["t"], windows["terminated"]
    assert torch.equal(t[:, 1:] - t[:, :-1], torch.ones_like(t[:, 1:]))
    assert not done[:, :-1].any()
    assert t.min() >= 4  # the 16-slot buffer holds steps 4..19


def test_a_causal_window_repeats_the_episodes_first_step_before_it_starts():
    buffer = _filled(steps=10, ends={4})
    torch.manual_seed(0)
    windows, padded = buffer.sample_causal(sequence_length=4, batch_size=500)
    newest = windows["t"][:, -1]
    starts_after_end = newest == 6
    assert starts_after_end.any()
    assert torch.equal(
        windows["t"][starts_after_end], torch.tensor([[5, 5, 5, 6]]).expand(int(starts_after_end.sum()), 4)
    )
    assert torch.equal(
        padded[starts_after_end], torch.tensor([[True, True, False, False]]).expand_as(padded[starts_after_end])
    )


def test_a_window_longer_than_the_buffer_is_refused():
    with pytest.raises(ValueError, match="exceeds buffer_size 16"):
        _filled(steps=4, ends=set()).sample(sequence_length=17, batch_size=1)
