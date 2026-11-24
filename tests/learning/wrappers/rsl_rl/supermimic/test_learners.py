"""The learners' optimizer handling."""

import pytest
import torch
from tensordict import TensorDict

from eden.learning.wrappers.rsl_rl.supermimic.learners import (
    CapturedUpdate,
    build_optimizers,
    load_optimizer_state,
    optimize,
)


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


def _net_after_steps(lr: float, lr_scale: torch.Tensor | None, steps: int = 3) -> torch.nn.Sequential:
    """A three-layer net, so Adam steps its outer layers and Muon its hidden weight, after ``steps`` updates."""
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Linear(8, 8), torch.nn.Linear(8, 2))
    optimizers = build_optimizers([net], lr=lr)
    assert len(optimizers) == 2
    for _ in range(steps):
        loss = net(torch.randn(16, 3)).square().mean()
        optimize(optimizers, loss, net.parameters(), max_norm=1.0, lr_scale=lr_scale)
    return net


def test_a_scaled_step_is_the_step_at_the_scaled_learning_rate():
    scaled = _net_after_steps(lr=1e-3, lr_scale=torch.tensor(4.0))
    at_the_rate = _net_after_steps(lr=4e-3, lr_scale=None)
    unscaled = _net_after_steps(lr=1e-3, lr_scale=None)
    for got, expected, other in zip(scaled.parameters(), at_the_rate.parameters(), unscaled.parameters(), strict=True):
        torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-4)
        assert (got - other).abs().max() > 1e-4  # the scale changed the step


def test_a_scale_of_one_leaves_the_step_as_it_is():
    scaled = _net_after_steps(lr=1e-3, lr_scale=torch.tensor(1.0))
    plain = _net_after_steps(lr=1e-3, lr_scale=None)
    for got, expected in zip(scaled.parameters(), plain.parameters(), strict=True):
        torch.testing.assert_close(got, expected, atol=1e-7, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="a CUDA graph needs a CUDA device")
def test_an_update_with_its_own_pool_leaves_the_outputs_of_the_updates_recorded_after_it():
    device, rows = torch.device("cuda:0"), 1024
    weight = torch.ones(rows, device=device)

    def scratch_then_sum(inputs):
        scratch = inputs["x"] * 2.0  # freed when the recording ends, for the next recording of its pool to take
        return {"sum": (scratch + weight).sum()}

    def shifted(inputs):
        return {"y": inputs["x"] + weight}

    def batch(value):
        return TensorDict({"x": torch.full((rows,), value, device=device)}, batch_size=[rows])

    acting = CapturedUpdate(scratch_then_sum, lambda: [weight], warmup=0, own_pool=True)
    learning = CapturedUpdate(shifted, lambda: [weight], warmup=0)
    acting(batch(1.0))
    kept = learning(batch(3.0))["y"]
    assert float(acting(batch(7.0))["sum"]) == rows * 15.0
    torch.testing.assert_close(kept, torch.full((rows,), 4.0, device=device))
