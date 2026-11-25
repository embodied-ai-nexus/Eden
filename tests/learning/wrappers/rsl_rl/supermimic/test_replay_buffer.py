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


def test_a_buffer_whose_every_window_holds_an_episode_end_reports_no_window():
    buffer = _filled(steps=8, ends={2, 5})
    assert buffer.has_window(3) and not buffer.has_window(4)
    with pytest.raises(RuntimeError, match=r"no window of 4 steps .* Check has_window\(4\)"):
        buffer.sample(sequence_length=4, batch_size=2)


def test_ending_the_episodes_keeps_windows_from_crossing_the_newest_step():
    buffer = _filled(steps=6, ends=set())
    buffer.end_episodes()
    for t in range(6, 12):
        buffer.insert(
            TensorDict(
                {"t": torch.tensor([t]), "terminated": torch.tensor([False]), "truncated": torch.tensor([False])}, [1]
            )
        )
    t = buffer.sample(sequence_length=3, batch_size=300)["t"]
    # Step 5 ends its episode: a window may end on it, and none holds it before its last step.
    assert not (t[:, :-1] == 5).any()
    assert (t[:, -1] == 5).any() and (t[:, 0] == 6).any()


def _step(t: int, terminated: bool = False) -> TensorDict:
    return TensorDict(
        {"t": torch.tensor([t]), "terminated": torch.tensor([terminated]), "truncated": torch.tensor([False])}, [1]
    )


def test_the_windows_counted_before_a_write_are_counted_again_after_it():
    buffer = _filled(steps=2, ends=set())
    assert not buffer.has_window(3)
    buffer.insert(_step(2, terminated=True))
    assert buffer.has_window(3)
    buffer.insert(_step(3))
    # Step 2 ends its episode, so the one window of three steps is still the first.
    assert torch.equal(buffer.sample(sequence_length=3, batch_size=8)["t"], torch.tensor([[0, 1, 2]]).expand(8, 3))
    buffer.end_episodes()
    buffer.insert(_step(4))
    buffer.insert(_step(5))
    assert not (buffer.sample(sequence_length=2, batch_size=200)["t"] == torch.tensor([3, 4])).all(1).any()


def test_samples_between_two_writes_draw_as_from_a_buffer_never_sampled():
    sampled, fresh = _filled(steps=14, ends={3, 9}), _filled(steps=14, ends={3, 9})
    torch.manual_seed(0)
    sampled.sample(sequence_length=3, batch_size=5)
    state = torch.get_rng_state()
    again = sampled.sample(sequence_length=3, batch_size=50)["t"]
    torch.set_rng_state(state)
    assert torch.equal(again, fresh.sample(sequence_length=3, batch_size=50)["t"])


def test_naming_the_fields_to_return_draws_the_same_windows():
    buffer = _filled(steps=14, ends={3, 9})
    torch.manual_seed(0)
    whole = buffer.sample(sequence_length=3, batch_size=50)
    torch.manual_seed(0)
    part = buffer.sample(sequence_length=3, batch_size=50, fields=("t",))
    assert set(part.keys()) == {"t"}
    assert torch.equal(part["t"], whole["t"])


def test_several_batches_at_once_are_the_batches_successive_calls_draw():
    buffer = _filled(steps=14, ends={3, 9})
    torch.manual_seed(0)
    one_by_one = torch.stack([buffer.sample(sequence_length=3, batch_size=20)["t"] for _ in range(4)])
    torch.manual_seed(0)
    together = buffer.sample(sequence_length=3, batch_size=20, num_batches=4)
    assert together.shape == (4, 20, 3)
    assert torch.equal(together["t"], one_by_one)
