"""Tests for generalized ``target_dim`` in ``HIMEstimator``/``HIMActorCritic`` and ``qpos`` clips in ``AmpMotionSampler``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from eden.learning.wrappers.rsl_rl.amp_ppo.amp import AmpMotionSampler
from eden.learning.wrappers.rsl_rl.him.actor_critic import HIMActorCritic
from eden.learning.wrappers.rsl_rl.him.estimator import HIMEstimator


def test_him_estimator_and_actor_critic_custom_target_dim() -> None:
    torch.manual_seed(0)
    est = HIMEstimator(
        temporal_steps=5,
        num_one_step_obs=12,
        latent_dim=16,
        target_skip_dim=2,
        target_dim=10,
        learning_rate=3e-3,
    )
    with pytest.raises(ValueError, match="target_dim >= 1"):
        HIMEstimator(temporal_steps=5, num_one_step_obs=12, target_dim=0)

    hist = torch.randn(32, 60)
    true_target = torch.sigmoid(hist[:, :10] * 1.5)
    next_critic = torch.cat([torch.randn(32, 12), true_target], dim=-1)

    loss0, _ = est.update(hist, next_critic)
    loss_last = loss0
    for _ in range(20):
        loss_last, _ = est.update(hist, next_critic)
    assert loss_last < loss0 * 0.5

    obs = {"policy": torch.zeros(4, 60), "critic": torch.zeros(4, 22)}
    ac = HIMActorCritic.from_groups(
        obs,
        "policy",
        "critic",
        num_actions=8,
        actor_hidden_dims=(64, 64),
        critic_hidden_dims=(64, 64),
        activation="elu",
        init_noise_std=1.0,
        target_skip_dim=2,
        target_dim=10,
    )
    out = ac.act_inference(obs)
    assert out.shape == (4, 8)


def test_amp_motion_sampler_loads_qpos_npz_with_slice(tmp_path: Path) -> None:
    clip = tmp_path / "myomimic_walk.npz"
    qpos = np.linspace(0.0, 1.0, 20 * 25, dtype=np.float32).reshape(20, 25)
    np.savez(clip, qpos=qpos, mocap_framerate=np.float32(50.0))

    sampler = AmpMotionSampler(str(clip), num_frames=4, device="cpu", qpos_slice=(7, None))
    assert sampler.frame_dim == 18
    assert abs(sampler.dt - 0.02) < 1e-6
    win = sampler.sample(batch_size=12)
    assert win.shape == (12, 4, 18)
    assert torch.isfinite(win).all()
