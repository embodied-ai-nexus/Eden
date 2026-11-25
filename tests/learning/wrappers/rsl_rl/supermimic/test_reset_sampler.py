"""The return sampler's bin values and the reset distribution they give."""

import pytest
import torch

from eden.learning.wrappers.rsl_rl.supermimic.options import ReturnSamplerOptions
from eden.learning.wrappers.rsl_rl.supermimic.reset_sampler import ReturnSampler, bin_frame_range, frame_bins
from eden.utils.motion_reference.library import MotionLibrary


def _sampler(**options) -> ReturnSampler:
    """A sampler without the failure share, whose shares are then explore, focus and keep."""
    return ReturnSampler(
        ReturnSamplerOptions(horizon=4, failure=0.0, **options),
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
    options = ReturnSamplerOptions(horizon=4, failure=0.0)
    other = ReturnSampler(options, 5, 50, num_envs=2, max_step_reward=1.0, device="cpu")
    with pytest.raises(ValueError, match=r"holds 2 reset bins, but this run's motion library has 5 \(50 frames\)"):
        other.load_state_dict(state)


def _library(*clip_frames: int) -> MotionLibrary:
    """A library timeline of clips with these frame counts at 50 Hz; the sampler reads only its clip bounds."""
    ends = torch.tensor(clip_frames).cumsum(0)
    empty = torch.zeros(int(ends[-1]), 0)
    return MotionLibrary(
        empty, empty, empty, empty, empty, empty, starts=ends - torch.tensor(clip_frames), ends=ends, fps=50.0
    )


def _failure_sampler(failure: float = 0.3, lead_s: float = 0.4, **options) -> ReturnSampler:
    """Two clips of 60 frames in 12 bins of 10 frames; the failure share reaches 2 bins (0.4 s) back."""
    return ReturnSampler(
        ReturnSamplerOptions(horizon=4, failure=failure, failure_lead_s=lead_s, **options),
        bin_count=12,
        frame_count=120,
        num_envs=2,
        max_step_reward=1.0,
        device="cpu",
        library=_library(60, 60),
        bin_frames=10,
    )


def _record(sampler, frames, fell, ended=None):
    fell = torch.tensor(fell)
    timed_out = torch.zeros_like(fell) if ended is None else torch.tensor(ended) & ~fell
    return sampler.record(torch.tensor(frames), torch.ones(len(frames)), fell, torch.zeros_like(fell), timed_out)


def test_a_fall_is_credited_to_the_bins_within_the_lead_before_it():
    sampler = _failure_sampler()
    _record(sampler, [35, 100], [True, False])  # env 0 falls at frame 35 (bin 3); env 1 goes on
    assert sampler.fall_rate.nonzero().flatten().tolist() == [1, 2]
    torch.testing.assert_close(sampler.fall_rate[[1, 2]], torch.tensor([0.1, 0.1]))


def test_a_fall_near_the_start_of_its_clip_stays_in_the_clip():
    sampler = _failure_sampler()
    _record(sampler, [72, 64], [True, True])  # the second clip starts at frame 60, in bin 6
    # Frame 72 reaches bin 6 only; frame 64 lies within one bin of the clip's start and is credited to its first bin.
    assert sampler.fall_rate.nonzero().flatten().tolist() == [6]
    assert sampler.fall_rate[:6].sum() == 0.0


def test_the_fall_rate_of_a_bin_is_the_share_of_the_episodes_ending_after_it_that_fell():
    sampler = _failure_sampler()
    _record(sampler, [35, 35], [True, False], ended=[True, True])  # one fall and one time-out at the same frame
    torch.testing.assert_close(sampler.fall_rate[2], torch.tensor(0.1 * 0.5))
    before = sampler.fall_rate[2].clone()
    _record(sampler, [80, 81], [False, False], ended=[True, False])  # an episode ends elsewhere without a fall
    torch.testing.assert_close(sampler.fall_rate[2], before * 0.999)


def test_the_failure_share_of_the_resets_goes_to_the_bins_before_recent_falls():
    sampler = _failure_sampler(failure=0.3)
    dist = None
    for step in range(4):
        dist = _record(sampler, [32 + step, 100 + step], [step == 3, False])
    assert dist.sum().item() == pytest.approx(1.0)
    before_the_fall = dist[[1, 2]].sum().item()
    without = _failure_sampler(failure=0.3)
    without.value, without.effort = sampler.value, sampler.effort
    assert before_the_fall == pytest.approx(0.7 * without.distribution()[[1, 2]].sum().item() + 0.3)


def test_until_a_fall_is_recorded_the_failure_share_goes_to_the_other_shares_in_proportion():
    with_failure = _failure_sampler(failure=0.3, explore=0.35, keep=0.07)
    plain = _failure_sampler(failure=0.0, explore=0.5, keep=0.1)
    for sampler in (with_failure, plain):
        for step in range(4):
            dist = _record(sampler, [32 + step, 100 + step], [False, False])
        assert dist.sum().item() == pytest.approx(1.0)
    torch.testing.assert_close(with_failure.distribution(), plain.distribution())


def test_the_fall_rate_is_saved_and_a_state_without_it_starts_with_none():
    sampler = _failure_sampler()
    _record(sampler, [35, 100], [True, False])
    state = sampler.state_dict()
    restored = _failure_sampler()
    restored.load_state_dict(state)
    torch.testing.assert_close(restored.fall_rate, sampler.fall_rate)
    del state["fall_rate"]  # a checkpoint written before the failure share existed
    restored.load_state_dict(state)
    assert restored.fall_rate.sum() == 0.0


def test_the_failure_share_without_the_library_is_refused():
    with pytest.raises(
        ValueError, match=r"return_sampler.failure=0.3 places resets before falls within the fall's clip"
    ):
        ReturnSampler(ReturnSamplerOptions(), 12, 120, num_envs=2, max_step_reward=1.0, device="cpu")


def test_shares_that_the_failure_share_leaves_no_focus_for_are_refused():
    with pytest.raises(ValueError, match=r"explore \(0.5\) \+ keep \(0.1\) \+ failure \(0.4\) must leave a share"):
        ReturnSamplerOptions(explore=0.5, keep=0.1, failure=0.4)
