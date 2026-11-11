"""Tests for ``FDRAssistCurriculum``: the object-assist gains decay once the gates hold."""

from __future__ import annotations

import pytest
import torch

from tests.tasks.flashdexretarget.conftest import build_task

pytestmark = [pytest.mark.slow, pytest.mark.requires_genesis]


def test_gains_decay_by_the_upper_ratio_once_every_gate_has_held_for_a_full_window(motion_file):
    """Gates made trivially true (thresholds below any reward, no completion floor): the first epoch past
    ``wait_epochs`` with full windows and no cooldown left decays kp 80 -> 72, the draw lands in [0.8, 1] x 72.
    """
    env = build_task(
        motion_file,
        **{
            "curriculum_options.assist_decay.wait_epochs": 2,
            "curriculum_options.assist_decay.deque_len": 2,
            "curriculum_options.assist_decay.num_steps_per_env": 2,
            "curriculum_options.assist_decay.decay_cooldown_epochs": 0,
            "curriculum_options.assist_decay.completion_thres": 0.0,
            "curriculum_options.assist_decay.rew_thresholds": {"task": -1.0, "imi": -1.0, "bc": -1.0},
        },
    )
    curriculum = env.curriculum_manager._terms["assist_decay"]
    assist = env.action_manager.get_term("assist")
    assert curriculum.upper["kp"] == 80.0
    hand = env.action_manager.get_term("hand")
    steps = 0
    while curriculum.upper["kp"] == 80.0 and steps < 50:
        env.step(torch.zeros(env.num_envs, hand.action_dim))
        steps += 1
    # Epochs are 2 steps; the decay needs epoch >= wait_epochs (2) and two full windows of epoch means.
    assert 4 <= steps <= 8
    assert curriculum.upper["kp"] == pytest.approx(72.0)
    assert curriculum.lower["kp"] == pytest.approx(0.8 * 72.0)
    assert bool(((assist.kp >= 0.8 * 72.0 - 1e-4) & (assist.kp <= 72.0 + 1e-4)).all())


@pytest.mark.parametrize(
    ("kp_zero_thres", "decayed"), [(0.05, 72.0), (100.0, 0.0)], ids=["after-a-decay", "after-a-decay-to-zero"]
)
def test_a_collapsed_completion_rate_dials_the_gains_back_and_holds_them(motion_file, kp_zero_thres, decayed):
    """Every episode fails (a 0 m fingertip threshold terminates every step), so the completion rate is 0. With the
    completion gate at 0 the first window decays kp 80 -> 72 (or to zero, under a ``kp_zero_thres`` above 72); the
    next, under the 0.5 dial-back ceiling, returns to the gains before that decay times 0.98 (78.4), and the later
    ones hold it there rather than compounding the ratio. FlashDexRetarget checks the dial-back only behind the decay
    gates, which a completion rate under its ceiling never passes.
    """
    env = build_task(
        motion_file,
        **{
            "curriculum_options.assist_decay.wait_epochs": 0,
            "curriculum_options.assist_decay.deque_len": 2,
            "curriculum_options.assist_decay.num_steps_per_env": 2,
            "curriculum_options.assist_decay.decay_cooldown_epochs": 0,
            "curriculum_options.assist_decay.dialback_min_epochs": 0,
            "curriculum_options.assist_decay.completion_thres": 0.0,
            "curriculum_options.assist_decay.kp_zero_thres": kp_zero_thres,
            "curriculum_options.assist_decay.rew_thresholds": {"task": -1.0, "imi": -1.0, "bc": -1.0},
            "termination_options.fingertip_diverged.threshold": 0.0,
            "termination_options.fingertip_diverged.grace_steps": 0,
        },
    )
    curriculum = env.curriculum_manager._terms["assist_decay"]
    hand = env.action_manager.get_term("hand")
    seen = [curriculum.upper["kp"]]
    for _ in range(40):  # 10 epochs: one decay, then dial-back windows
        env.step(torch.zeros(env.num_envs, hand.action_dim))
        if curriculum.upper["kp"] != seen[-1]:
            seen.append(curriculum.upper["kp"])
    assert seen == pytest.approx([80.0, decayed, 78.4])
    # The gains before the decay had lower = upper = 80, so the dial-back's lower bound is the upper one.
    assert curriculum.lower["kp"] == pytest.approx(78.4)
    kp = env.action_manager.get_term("assist").kp
    assert bool(((kp >= 78.4 - 1e-4) & (kp <= 78.4 + 1e-4)).all())


def test_an_eval_env_runs_without_the_assist(motion_file):
    """``eden inference`` builds with ``eval_mode=True``, where the curriculum manager freezes every term at the level
    its build set; the assist being a training aid, that level is zero gains, not ``kp_init``.
    """
    env = build_task(motion_file, eval_mode=True)
    curriculum = env.curriculum_manager._terms["assist_decay"]
    assist = env.action_manager.get_term("assist")
    hand = env.action_manager.get_term("hand")
    assert curriculum.upper == {"kp": 0.0, "kv": 0.0, "fr": 0.0}
    for _ in range(4):
        env.step(torch.zeros(env.num_envs, hand.action_dim))
    assert bool((assist.kp == 0.0).all()) and bool((assist.kv == 0.0).all())
