"""The AMP components (pure torch): expert sampler, discriminator, replay buffer, frame normalizer."""

from __future__ import annotations

import torch

from eden.extensions.policies.husky.constants import NUM_JOINTS
from eden.learning.wrappers.rsl_rl.amp_ppo.amp import (
    AmpDiscriminator,
    AmpMotionSampler,
    AmpReplayBuffer,
    FrameNormalizer,
)


def test_motion_sampler_draws_physical_windows_from_the_clips():
    windows = AmpMotionSampler("objects/skateboard/husky_amp_push_motions.npz", num_frames=5, device="cpu").sample(256)
    assert windows.shape == (256, 5, NUM_JOINTS)
    assert torch.isfinite(windows).all()
    # Frames 0.02 s apart stay below the clips' max per-frame joint speed; values are joint angles.
    assert float((windows[:, 1:] - windows[:, :-1]).abs().max()) < 1.0
    assert float(windows.abs().max()) < 4.0


def test_discriminator_style_reward_is_bounded_and_its_gradient_penalty_finite():
    disc = AmpDiscriminator(window_dim=5 * NUM_JOINTS)
    reward = disc.style_reward(torch.randn(64, 5 * NUM_JOINTS), coef=5.0)
    assert reward.shape == (64,)
    assert ((reward >= 0.0) & (reward <= 5.0)).all()

    penalty = disc.gradient_penalty(torch.randn(32, 5 * NUM_JOINTS), scale=5.0)
    assert torch.isfinite(penalty) and penalty.item() >= 0.0


def test_replay_buffer_overwrites_the_oldest_rows():
    buf = AmpReplayBuffer(window_dim=4, capacity=10, device="cpu")
    buf.insert(torch.arange(28.0).reshape(7, 4))
    assert len(buf) == 7
    buf.insert(torch.arange(28.0, 56.0).reshape(7, 4))
    assert len(buf) == 10
    # The oldest 4 rows were overwritten; every sample comes from the last 10 inserted.
    assert float(buf.sample(32).min()) >= 16.0


def test_frame_normalizer_converges_to_zero_mean_unit_std():
    norm = FrameNormalizer(frame_dim=3)
    data = torch.randn(10000, 3) * torch.tensor([2.0, 0.5, 1.0]) + torch.tensor([1.0, -3.0, 0.0])
    for chunk in data.chunk(10):
        norm.update(chunk)
    out = norm.normalize(data)
    assert float(out.mean(dim=0).abs().max()) < 0.05
    assert float((out.std(dim=0) - 1.0).abs().max()) < 0.05
