"""The return sampler's bin values and the reset distribution they give."""

import pytest
import torch

from eden.learning.wrappers.rsl_rl.supermimic.options import ReturnSamplerOptions
from eden.learning.wrappers.rsl_rl.supermimic.reset_sampler import ReturnSampler, bin_frame_range, frame_bins


def _sampler(**options) -> ReturnSampler:
    return ReturnSampler(
        ReturnSamplerOptions(horizon=4, **options),
        bin_count=2,
        frame_count=20,
        num_envs=2,
        max_step_reward=1.0,
        device="cpu",
    )


def test_a_fall_concentrates_resets_on_its_bin():
    sampler = _sampler()
    frames = torch.tensor([2, 12])  # env 0 in bin 0, env 1 in bin 1
    dist = None
    for step in range(4):
        fell = torch.tensor([step == 3, False])
        dist = sampler.record(
            frames + step, torch.ones(2), fell, torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)
        )
    assert sampler.value[0] < sampler.value[1]
    assert dist[0] > dist[1]
    assert dist.sum() == pytest.approx(1.0)


def test_the_value_of_a_bin_is_the_discounted_return_after_a_fall():
    sampler = _sampler(gamma=0.5)
    no = torch.zeros(2, dtype=torch.bool)
    for step in range(4):
        sampler.record(torch.tensor([0, 0]), torch.ones(2), torch.tensor([step == 3] * 2), no, no)
    # Four rewards of 1 then a fall: the mean of 1.875, 1.75, 1.5 and 1.
    assert sampler.value[0].item() == pytest.approx((1.875 + 1.75 + 1.5 + 1.0) / 4)


def test_shares_that_leave_no_focus_are_refused():
    with pytest.raises(ValueError, match="must leave a share for the focus"):
        ReturnSamplerOptions(explore=0.6, keep=0.4)


@pytest.mark.parametrize("frame_count, bin_count", [(1000, 101), (135, 14), (20, 21), (7, 3)])
def test_every_frame_lies_in_the_range_of_the_bin_it_is_scored_in(frame_count, bin_count):
    frames = torch.arange(frame_count)
    bins = frame_bins(frames, bin_count, frame_count)
    first, end = bin_frame_range(bins, bin_count, frame_count)
    assert ((first <= frames) & (frames < end)).all()
    first, end = bin_frame_range(torch.arange(bin_count), bin_count, frame_count)
    assert first[0] == 0 and end[-1] == frame_count and torch.equal(first[1:], end[:-1])


def test_a_saved_state_with_another_bin_count_names_both_libraries():
    state = _sampler().state_dict()
    other = ReturnSampler(ReturnSamplerOptions(horizon=4), 5, 50, num_envs=2, max_step_reward=1.0, device="cpu")
    with pytest.raises(ValueError, match=r"holds 2 reset bins, but this run's motion library has 5 \(50 frames\)"):
        other.load_state_dict(state)
