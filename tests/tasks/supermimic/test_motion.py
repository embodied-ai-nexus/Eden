"""``SuperMimicMotion``: link states by forward kinematics, the library timeline and its reset distribution."""

import pytest
import torch

from eden.tasks.supermimic.config import TRACKED_LINKS
from tests.tasks.supermimic.conftest import CLIP_FRAMES

pytestmark = [pytest.mark.requires_genesis]


def _motion(env):
    return env.motion_manager.get_term("default")


def test_a_noise_free_reset_puts_the_robot_on_the_reference_links(env):
    env.reset()
    robot, ref = env.entities["robot"], env.motion_manager.data["robot"]
    _, links = robot.find_named_links_idx_local(TRACKED_LINKS)
    torch.testing.assert_close(ref.links_pos_w[:, 0], robot.get_links_pos(ls_idx_local=links), atol=1e-4, rtol=0)
    torch.testing.assert_close(ref.links_lin_vel_w[:, 0], robot.get_links_vel(ls_idx_local=links), atol=1e-3, rtol=0)


def test_the_library_is_one_timeline_at_the_control_rate(env):
    library = _motion(env).library
    lengths = [round((n - 1) / 30.0 * 50.0) + 1 for n in CLIP_FRAMES]
    assert library.ends.tolist() == [lengths[0], sum(lengths)]
    assert library.starts.tolist() == [0, lengths[0]]
    assert library.fps == pytest.approx(1.0 / env.dt)


def test_the_frame_advances_one_per_step_within_a_clip(env):
    env.reset()
    motion = _motion(env)
    before = motion.frame_index.clone()
    env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim))
    stayed = ~env.reset_buf
    assert stayed.any()
    torch.testing.assert_close(motion.frame_index[stayed], before[stayed] + 1)


def test_resets_start_in_the_bins_the_distribution_selects(env):
    motion = _motion(env)
    weights = torch.zeros(motion.bin_count)
    weights[3] = 1.0
    motion.set_reset_distribution(weights)
    try:
        env.reset()
        frames_per_bin = (motion.library.num_frames - 2) / motion.bin_count
        assert ((motion.frame_index >= 3 * frames_per_bin - 1) & (motion.frame_index <= 4 * frames_per_bin)).all()
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
            last = library.last_frame(motion.frame_index)
            at_end_next = motion.frame_index + 1 == last
            env.step(actions)
            clip_end = env.termination_manager.get_term_dones("clip_end")
            assert torch.equal(clip_end, at_end_next)
            if clip_end.any():
                assert env.reset_timeouts[clip_end].all()
                return
        pytest.fail("no env reached its clip's last frame")
    finally:
        motion.set_reset_distribution(torch.ones(motion.bin_count))
