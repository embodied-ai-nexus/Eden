"""The learners' optimizer handling."""

import torch

from eden.learning.wrappers.rsl_rl.supermimic.learners import build_optimizers, load_optimizer_state


def _stepped(seed: int):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    optimizers = build_optimizers([net], lr=1e-2)
    for _ in range(2):
        net(torch.randn(5, 3)).square().mean().backward()
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad()
    return optimizers


def _state_tensors(optimizers):
    return [t for o in optimizers for s in o.state.values() for t in s.values() if torch.is_tensor(t)]


def test_a_saved_optimizer_state_is_copied_into_the_tensors_the_optimizer_holds():
    saved, current = _stepped(0), _stepped(1)
    before = _state_tensors(current)
    addresses = [t.data_ptr() for t in before]
    for optimizer, source in zip(current, saved, strict=True):
        load_optimizer_state(optimizer, source.state_dict())
    after = _state_tensors(current)
    assert [t.data_ptr() for t in after] == addresses  # a recorded CUDA graph still reads these
    for restored, expected in zip(after, _state_tensors(saved), strict=True):
        torch.testing.assert_close(restored, expected)


def test_an_optimizer_that_has_not_stepped_takes_the_saved_state():
    saved = _stepped(0)
    torch.manual_seed(1)
    net = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    fresh = build_optimizers([net], lr=1e-2)
    for optimizer, source in zip(fresh, saved, strict=True):
        load_optimizer_state(optimizer, source.state_dict())
    for restored, expected in zip(_state_tensors(fresh), _state_tensors(saved), strict=True):
        torch.testing.assert_close(restored, expected)
