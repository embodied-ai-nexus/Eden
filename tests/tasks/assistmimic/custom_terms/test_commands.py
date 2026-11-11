"""``AssistMimicPairMotion`` on a real two-humanoid env: the clip-start reset and the build-time checks."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import eden.utils.geom as geom
from eden.tasks.assistmimic.config import BED_POS, BED_SIZE
from eden.tasks.assistmimic.robots import SMPL_DOF_NAMES
from tests.support.envs import build_env
from tests.tasks.assistmimic.conftest import (
    CLIPS,
    RECIPIENT_SHIFT,
    ROOT_HEIGHT,
    assistmimic_config,
    smpl_zero_pose,
    write_clips,
)


@pytest.mark.requires_genesis
def test_reset_puts_both_humanoids_on_their_clips_first_frame(assistmimic_env):
    env = assistmimic_env
    env.reset()
    motion = env.command_manager.get_term("motion")
    clip_dx = torch.tensor([dx for _, dx in CLIPS], device=env.device)[motion.clip_id]
    for role, shift in (("caregiver", (0.0, 0.0, 0.0)), ("recipient", RECIPIENT_SHIFT)):
        root = env.entities[role].get_links_pos()[:, 0]
        expected = torch.tensor([0.0, 0.0, ROOT_HEIGHT], device=env.device) + torch.tensor(shift, device=env.device)
        offset = torch.stack([clip_dx, torch.zeros_like(clip_dx), torch.zeros_like(clip_dx)], 1)
        torch.testing.assert_close(root, expected + offset, check_dtype=False)
    # The graded reference is the same frame the pair was put on.
    torch.testing.assert_close(
        motion.body_ref("recipient", motion.motion_time()).pos[:, 0],
        env.entities["recipient"].get_links_pos()[:, 0],
        check_dtype=False,
    )


@pytest.mark.requires_genesis
def test_a_clip_whose_bodies_are_in_another_order_is_refused_at_build(tmp_path):
    names, _ = smpl_zero_pose()
    names[1], names[5] = names[5], names[1]  # L_Hip <-> R_Hip
    clips = write_clips(tmp_path / "swapped.npz", body_names=names)
    with pytest.raises(ValueError, match=r"has link 'L_Hip' at index 1 where .* has body 'R_Hip'"):
        build_env(assistmimic_config("assistmimic", clips))


@pytest.mark.requires_genesis
def test_a_recipient_starting_inside_the_bed_is_refused_at_build(tmp_path):
    """The bed's top face is at ~0.44 m; a recipient whose root starts at 0.3 m over it is inside the box."""
    _, pos0 = smpl_zero_pose()
    sunk = (BED_POS[0] - pos0[0, 0], BED_POS[1] - pos0[0, 1], 0.3 - ROOT_HEIGHT)
    clips = write_clips(tmp_path / "sunk.npz", recipient_shift=sunk)
    with pytest.raises(ValueError, match="starts inside the bed 'bed'"):
        build_env(assistmimic_config("assistmimic_bed", clips))


@pytest.mark.requires_genesis
def test_a_recipient_starting_beside_the_bed_is_refused_at_build(tmp_path):
    """Standing 0.5 m past the bed's far edge, no body is over the bed, so none is inside it either."""
    _, pos0 = smpl_zero_pose()
    beside = (BED_POS[0] + BED_SIZE[0] / 2 + 0.5 - pos0[0, 0], BED_POS[1] - pos0[0, 1], 0.0)
    clips = write_clips(tmp_path / "beside.npz", recipient_shift=beside)
    with pytest.raises(ValueError, match=r"reset pose is not over the bed 'bed' .* in 2 of 2 clips"):
        build_env(assistmimic_config("assistmimic_bed", clips))


@pytest.mark.requires_genesis
def test_a_timed_resample_is_refused_at_build(clips_file):
    cfg = assistmimic_config("assistmimic", clips_file)
    cfg.command_options.motion.resampling_time_range = (5.0, 5.0)
    with pytest.raises(ValueError, match=r"resampling_time_range=\(5.0, 5.0\), but the pair's clip may change only"):
        build_env(cfg)


@pytest.mark.requires_genesis
def test_restore_state_reads_the_clip_from_the_frame_and_restarts_the_episode_length(assistmimic_env):
    """The restored pair plays its clip from ``frame`` on, while the episode itself starts over at step 0."""
    env = assistmimic_env
    env.reset()
    motion = env.command_manager.get_term("motion")
    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    for _ in range(2):
        env.step(zero)
    state = motion.capture_state()

    env.reset()
    envs = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    frame = torch.full((env.num_envs,), 3, dtype=torch.long, device=env.device)
    motion.restore_state(envs, state, frame)

    torch.testing.assert_close(motion.capture_state(), state, check_dtype=False)
    assert env.episode_length_buf.tolist() == [0] * env.num_envs
    assert motion.reference_frame.tolist() == [3] * env.num_envs
    obs_ref = motion.body_ref("recipient", motion.motion_time(1)).pos
    torch.testing.assert_close(motion.obs_ref["recipient"].pos, obs_ref, check_dtype=False)

    env.step(zero)
    assert motion.reference_frame.tolist() == [4] * env.num_envs
    # Its first steps are an episode's first steps: PHC's power penalty is still off.
    assert env.metric_manager.get_term("power_caregiver").compute().abs().max() == 0.0
    env.reset()
    assert motion.reference_frame.tolist() == [0] * env.num_envs


@pytest.mark.requires_genesis
def test_a_clip_of_one_frame_is_refused_at_build(tmp_path):
    """Clip 0 cut down to its last frame: no duration to play and no next frame to take velocities from."""
    data = dict(np.load(write_clips(tmp_path / "clips.npz")))
    data["num_frames"] = np.array([1, CLIPS[1][0]])
    for key in [k for k in data if k.startswith(("caregiver_", "recipient_"))]:
        data[key] = data[key][CLIPS[0][0] - 1 :]
    np.savez(tmp_path / "one_frame.npz", **data)
    with pytest.raises(ValueError, match=r"clips \[\('clip0', 1\)\] .* fewer than 2 frames"):
        build_env(assistmimic_config("assistmimic", str(tmp_path / "one_frame.npz")))


@pytest.mark.requires_genesis
def test_a_clip_starting_with_a_knee_bending_through_90_deg_resets_its_hinges_without_spinning(tmp_path):
    """The caregiver's left knee bends about y at 0.9 rad/s through 90 deg with a 0.02 rad twist about z.

    Differencing the hinge angles there would start its x and z hinges at tens of rad/s.
    """
    data = dict(np.load(write_clips(tmp_path / "clips.npz")))
    frames = CLIPS[0][0]
    y = 1.53 + 0.03 * np.arange(frames)
    rotvec = torch.tensor(np.stack([np.zeros(frames), y, np.full(frames, 0.02)], axis=-1))
    knee = [str(b) for b in data["body_names"]].index("L_Knee")
    data["caregiver_lrs"][:frames, knee] = geom.quat_from_axis_angle(rotvec).numpy()
    np.savez(tmp_path / "bending.npz", **data)
    cfg = assistmimic_config("assistmimic", str(tmp_path / "bending.npz"))
    cfg.command_options.motion.fixed_clip = 0
    env = build_env(cfg, reset=True)

    dofs = env.action_manager.get_term("caregiver").dofs_idx_local
    knee_dofs = [SMPL_DOF_NAMES.index(f"L_Knee_{axis}") for axis in "xyz"]
    entity = env.entities["caregiver"]
    vel = entity.get_dofs_vel(dofs_idx_local=dofs)[:, knee_dofs]
    pos = entity.get_dofs_pos(dofs_idx_local=dofs)[:, knee_dofs]
    assert vel[:, [0, 2]].abs().max() < 0.5
    torch.testing.assert_close(vel[:, 1], torch.full_like(vel[:, 1], 0.9), atol=0.05, rtol=0)
    assert pos[:, [0, 2]].abs().max() < 0.05
    torch.testing.assert_close(pos[:, 1], torch.full_like(pos[:, 1], 1.53), atol=0.02, rtol=0)


@pytest.mark.requires_genesis
def test_a_motion_lock_band_of_1_is_refused_at_build(clips_file):
    cfg = assistmimic_config("assistmimic", clips_file)
    cfg.command_options.motion.lock_band = 1.0
    with pytest.raises(ValueError, match=r"AssistMimicPairMotion: lock_band=1.0 bounds \|cos b\|"):
        build_env(cfg)


@pytest.mark.requires_genesis
def test_restore_state_writes_only_the_masked_envs(assistmimic_env):
    """Env 0 is restored to a captured state at frame 3; env 1 keeps its clip-start pose and clock."""
    env = assistmimic_env
    env.reset()
    motion = env.command_manager.get_term("motion")
    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    for _ in range(2):
        env.step(zero)
    captured = motion.capture_state()

    env.reset()
    untouched = motion.capture_state()
    mask = torch.tensor([True, False], device=env.device)
    frame = torch.tensor([3, 7], device=env.device)
    motion.restore_state(mask, captured, frame)

    after = motion.capture_state()
    torch.testing.assert_close(after[0], captured[0], check_dtype=False)
    torch.testing.assert_close(after[1], untouched[1], check_dtype=False)
    assert motion.reference_frame.tolist() == [3, 0]
    env.reset()


@pytest.mark.requires_genesis
def test_restore_state_refuses_env_ids(assistmimic_env):
    motion = assistmimic_env.command_manager.get_term("motion")
    state = motion.capture_state()
    ids = torch.tensor([0], device=assistmimic_env.device)
    with pytest.raises(ValueError, match=r"envs_mask must be an \(2,\) bool mask, got a torch.int64"):
        motion.restore_state(ids, state, torch.zeros(2, dtype=torch.long, device=assistmimic_env.device))


def _psi_env(clips_file, eval_mode: bool = False):
    """Clip 0 only, PSI on every reset, every episode of 2+ frames stored whatever its return."""
    from eden.envs.base import env_class_for_config

    cfg = assistmimic_config("assistmimic", clips_file)
    motion = cfg.command_options.motion
    motion.fixed_clip, motion.psi_ratio, motion.psi_return_threshold, motion.psi_tail_margin = 0, 1.0, -1.0e9, 0
    env = env_class_for_config(cfg).from_config(cfg, eval_mode=eval_mode)
    env.build()
    env.reset()
    return env


@pytest.mark.requires_genesis
def test_psi_restarts_a_reset_pair_from_a_state_its_episode_was_in(clips_file):
    """Five steps recorded at clip steps 0-5; the reset stores frames 0-4 and restarts each env on one of them."""
    env = _psi_env(clips_file)
    motion = env.command_manager.get_term("motion")
    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    visited = [motion.capture_state()]
    for _ in range(5):
        _, _, term, trunc, _ = env.step(zero)
        assert not (term | trunc).any()
        visited.append(motion.capture_state())

    env.reset()

    assert int(motion.psi.stored_total) == env.num_envs
    assert int(motion.psi.restored_total) == env.num_envs
    assert env.episode_length_buf.tolist() == [0] * env.num_envs
    frame = motion.reference_frame
    assert ((frame >= 0) & (frame < 5)).all()
    expected = torch.stack([visited[f][i] for i, f in enumerate(frame.tolist())])
    torch.testing.assert_close(motion.capture_state(), expected, atol=1e-5, rtol=0, check_dtype=False)
    # The observation reference is the restarted frame's next step, not the clip start's.
    obs_ref = motion.body_ref("recipient", motion.motion_time(1)).pos
    torch.testing.assert_close(motion.obs_ref["recipient"].pos, obs_ref, check_dtype=False)


@pytest.mark.requires_genesis
def test_psi_does_not_store_an_episode_below_the_return_threshold(assistmimic_env):
    """The task's threshold is 300 (summed per-step rewards); two steps earn far less."""
    env = assistmimic_env
    env.reset()
    motion = env.command_manager.get_term("motion")
    stored = int(motion.psi.stored_total)
    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    for _ in range(2):
        env.step(zero)
    env.reset()
    assert int(motion.psi.stored_total) == stored
    assert motion.reference_frame.tolist() == [0] * env.num_envs


@pytest.mark.requires_genesis
def test_psi_is_off_in_an_eval_env(clips_file):
    env = _psi_env(clips_file, eval_mode=True)
    assert env.command_manager.get_term("motion").psi is None


@pytest.mark.requires_genesis
def test_a_psi_ratio_above_1_is_refused_at_build(clips_file):
    cfg = assistmimic_config("assistmimic", clips_file)
    cfg.command_options.motion.psi_ratio = 1.5
    with pytest.raises(ValueError, match=r"psi_ratio=1.5 is the probability .* must be in \[0, 1\]"):
        build_env(cfg)
