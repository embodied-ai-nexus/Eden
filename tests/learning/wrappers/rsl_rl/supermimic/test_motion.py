"""``SuperMimicMotion``: the library timeline and its reset distribution."""

import pytest
import torch

from eden.tasks import TASK_REGISTRY
from tests.support.supermimic import CLIP_FRAMES

pytestmark = [pytest.mark.requires_genesis]


def _motion(env):
    return env.motion_manager.get_term("default")


def _frames(env):
    return env.motion_manager.get_library_frame("default")


def test_the_library_is_one_timeline_at_the_control_rate(env):
    library = _motion(env).library
    # The clips in the order the dataset lists its files, which the file system decides.
    lengths = (library.ends - library.starts).tolist()
    assert sorted(lengths) == sorted(round((n - 1) / 30.0 * 50.0) + 1 for n in CLIP_FRAMES)
    assert library.starts.tolist() == [0, lengths[0]]
    assert library.num_frames == sum(lengths)
    assert library.fps == pytest.approx(1.0 / env.dt)


def test_the_frame_advances_one_per_step_within_a_clip(env):
    env.reset()
    before = _frames(env)
    env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim))
    stayed = ~env.reset_buf
    assert stayed.any()
    torch.testing.assert_close(_frames(env)[stayed], before[stayed] + 1)


def test_resets_start_in_the_bins_the_distribution_selects(env):
    motion = _motion(env)
    weights = torch.zeros(motion.bin_count)
    weights[3] = 1.0
    motion.set_reset_distribution(weights)
    try:
        env.reset()
        frames_per_bin = (motion.library.num_frames - 2) / motion.bin_count
        frames = _frames(env)
        assert ((frames >= 3 * frames_per_bin - 1) & (frames <= 4 * frames_per_bin)).all()
    finally:
        motion.set_reset_distribution(torch.ones(motion.bin_count))


def test_a_distribution_of_the_wrong_length_names_the_bin_count(env):
    motion = _motion(env)
    with pytest.raises(ValueError, match=rf"shape \({motion.bin_count},\)"):
        motion.set_reset_distribution(torch.ones(motion.bin_count + 1))


def test_an_episode_is_truncated_on_its_clips_last_frame(env):
    motion = _motion(env)
    library = motion.library
    weights = torch.zeros(motion.bin_count)
    weights[(library.ends[0] - 3) * motion.bin_count // library.num_frames] = 1.0
    motion.set_reset_distribution(weights)
    try:
        env.reset()
        actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
        for _ in range(10):
            frames = _frames(env)
            at_end_next = frames + 1 == library.last_frame(frames)
            env.step(actions)
            clip_end = env.termination_manager.get_term_dones("clip_end")
            assert torch.equal(clip_end, at_end_next)
            if clip_end.any():
                assert env.reset_timeouts[clip_end].all()
                return
        pytest.fail("no env reached its clip's last frame")
    finally:
        motion.set_reset_distribution(torch.ones(motion.bin_count))


def test_the_frame_stays_whole_over_a_long_episode(env):
    """A start is stored at the middle of its frame, so the float clock never rounds a frame down."""
    motion = _motion(env)
    selection, start = motion._assigned_env_motion_selection.clone(), motion._motion_buffer_start_time_s.clone()
    try:
        ids = torch.arange(env.num_envs)
        motion.set_start_frames(ids, torch.zeros_like(ids), torch.full_like(ids, 7))
        clock = torch.zeros(env.num_envs)
        for step in range(60):
            assert motion.library_frames(clock).tolist() == [7 + step] * env.num_envs
            clock += env.dt
    finally:
        motion._assigned_env_motion_selection.copy_(selection)
        motion._motion_buffer_start_time_s.copy_(start)


@pytest.mark.parametrize(
    "field, value",
    [("motion_bin_length_s", 1.0), ("env_starting_stub_sampling_strategy", "concat_motion_bins")],
)
def test_a_base_start_sampling_option_raises_and_is_named(lafan1_library, field, value):
    from tests.support.envs import build_env

    cfg = TASK_REGISTRY.build("supermimic", raw_data_path=str(lafan1_library))
    cfg.env_options.num_envs = 1
    setattr(cfg.motion_options.default, field, value)
    with pytest.raises(ValueError, match=rf"would have no effect: \{{'{field}'"):
        build_env(cfg, reset=True)


def test_a_reset_bin_shorter_than_half_a_frame_is_refused_with_the_frame_period(lafan1_library):
    from tests.support.envs import build_env

    cfg = TASK_REGISTRY.build("supermimic", raw_data_path=str(lafan1_library))
    cfg.env_options.num_envs = 1
    cfg.motion_options.default.reset_bin_length_s = 0.005
    with pytest.raises(ValueError, match=r"reset_bin_length_s=0.005 is shorter than half a frame \(0.01 s at 50 Hz\)"):
        build_env(cfg, reset=True)


def test_a_reset_drawn_from_a_bin_starts_on_a_frame_the_sampler_scores_in_that_bin(env):
    from eden.learning.wrappers.rsl_rl.supermimic.reset_sampler import frame_bins

    motion = env.motion_manager.get_term("default")
    kept = motion.reset_distribution.clone()
    try:
        for target in (0, motion.bin_count // 2, motion.bin_count - 2):
            weights = torch.zeros(motion.bin_count)
            weights[target] = 1.0
            motion.set_reset_distribution(weights)
            frames = motion._sample_frames(200)
            bins = frame_bins(frames, motion.bin_count, motion.library.num_frames)
            # A draw on a clip's last frame moves one frame back, which may be the bin before.
            assert ((bins == target) | (bins == target - 1)).all() and (bins == target).any()
    finally:
        motion.set_reset_distribution(kept)


def test_every_reset_bin_holds_a_frame(env):
    from eden.learning.wrappers.rsl_rl.supermimic.reset_sampler import bin_frame_range

    motion = env.motion_manager.get_term("default")
    first, end = bin_frame_range(torch.arange(motion.bin_count), motion.bin_count, motion.library.num_frames)
    assert (end > first).all()
