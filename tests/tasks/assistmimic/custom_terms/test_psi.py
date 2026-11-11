"""``PsiTrajectoryBuffer``: which episodes it stores, which frames of them, and which resets it restarts."""

from __future__ import annotations

import torch

from eden.tasks.assistmimic.custom_terms.psi import PsiTrajectoryBuffer

N, CLIPS, DIM, MAX_LEN = 4, 2, 3, 12


def _buffer(**kwargs) -> PsiTrajectoryBuffer:
    kwargs = {"return_threshold": 10.0, "usage_ratio": 1.0, "tail_margin": 2, "buffer_size": 2} | kwargs
    return PsiTrajectoryBuffer(N, CLIPS, DIM, MAX_LEN, "cpu", **kwargs)


def _run(buf: PsiTrajectoryBuffer, frames: int, state_of=lambda f: torch.full((N, DIM), float(f))) -> None:
    """Start every env at frame 0 and record ``frames`` steps, the state at frame ``f`` being ``f`` everywhere."""
    everyone = torch.ones(N, dtype=torch.bool)
    buf.start(everyone, state_of(0), torch.zeros(N, dtype=torch.long))
    for f in range(1, frames + 1):
        buf.record(state_of(f), torch.full((N,), f))


def _end(buf, returns, term=(False,) * N, end_frame=8, clip=(0, 0, 1, 1)) -> None:
    buf.end_episodes(
        torch.tensor(returns, dtype=torch.float).repeat(2, 1).T,
        torch.ones(N, dtype=torch.bool),
        torch.tensor(term),
        torch.full((N,), end_frame),
        torch.tensor(clip),
    )


def test_only_episodes_whose_better_role_reaches_the_threshold_are_stored():
    buf = _buffer()
    _run(buf, 8)
    returns = torch.tensor([[11.0, 0.0], [0.0, 9.0], [0.0, 0.0], [3.0, 10.0]])
    buf.end_episodes(
        returns,
        torch.ones(N, dtype=torch.bool),
        torch.zeros(N, dtype=torch.bool),
        torch.full((N,), 8),
        torch.tensor([0, 0, 1, 1]),
    )
    assert int(buf.stored_total) == 2
    assert (buf.store_hi[0] > 0).sum() == 1 and (buf.store_hi[1] > 0).sum() == 1


def test_the_end_frame_and_an_early_termination_tail_are_left_out():
    buf = _buffer()
    _run(buf, 8)
    _end(buf, [20.0, 0.0, 20.0, 0.0], term=(False, False, True, False))
    assert buf.store_hi[0].max() == 8  # exclusive: frames 0-7, not the end frame
    assert buf.store_hi[1].max() == 8 - 2  # terminated: the 2 frames before the failure go too
    assert buf.store_lo[0].max() == 0


def test_a_recording_with_a_non_finite_state_is_not_stored():
    buf = _buffer()
    _run(buf, 8, state_of=lambda f: torch.full((N, DIM), float("nan") if f == 4 else float(f)))
    _end(buf, [20.0] * N)
    assert int(buf.stored_total) == 0


def test_a_clip_keeps_its_newest_trajectories():
    buf = _buffer(buffer_size=2)
    for episode in range(3):
        _run(buf, 8, state_of=lambda f, e=episode: torch.full((N, DIM), 100.0 * e + f))
        _end(buf, [20.0, 0.0, 0.0, 0.0])
    kept = sorted(buf.store[0, :, 0, 0].tolist())
    assert kept == [100.0, 200.0]


def test_restarts_draw_a_stored_frame_of_the_new_clip_and_skip_clips_with_none():
    buf = _buffer()
    _run(buf, 8)
    _end(buf, [20.0, 0.0, 0.0, 0.0])  # only clip 0 gets a trajectory
    mask, states, frames = buf.sample_restarts(torch.ones(N, dtype=torch.bool), torch.tensor([0, 1, 0, 1]))
    assert mask.tolist() == [True, False, True, False]
    assert ((frames[mask] >= 0) & (frames[mask] < 8)).all()
    torch.testing.assert_close(states[mask][:, 0], frames[mask].float())  # the state recorded at that frame
    assert int(buf.restored_total) == 2


def test_a_restarted_recording_starts_at_its_restart_frame():
    buf = _buffer()
    _run(buf, 8)
    restarted = torch.tensor([True, False, False, False])
    buf.start(restarted, torch.full((N, DIM), 5.0), torch.full((N,), 5))
    for f in range(6, 8):
        buf.record(torch.full((N, DIM), float(f)), torch.full((N,), f))
    buf.end_episodes(
        torch.full((N, 2), 20.0),
        restarted,
        torch.zeros(N, dtype=torch.bool),
        torch.full((N,), 8),
        torch.zeros(N, dtype=torch.long),
    )
    assert int(buf.stored_total) == 1
    slot = int(buf.store_hi[0].argmax())
    assert (int(buf.store_lo[0, slot]), int(buf.store_hi[0, slot])) == (5, 8)


def test_a_clip_with_more_qualifying_episodes_than_slots_writes_them_all():
    """Four episodes of clip 0 end at once into two slots: all four are stored, the FIFO index passes them all.

    Slots written twice in one call hold a blend of the episodes written there (which env's write lands last is
    not fixed), so only each slot's window and its frames' clip-frame range are pinned.
    """
    buf = _buffer(buffer_size=2)
    everyone = torch.ones(N, dtype=torch.bool)
    starts = torch.arange(N)  # env e starts at frame e
    env_offset = 100.0 * torch.arange(N, dtype=torch.float)[:, None]
    buf.start(everyone, env_offset + starts[:, None].float(), starts)
    for f in range(1, 9):
        buf.record(env_offset + float(f), torch.full((N,), f))
    _end(buf, [20.0] * N, clip=(0, 0, 0, 0))
    assert int(buf.stored_total) == 4
    assert int(buf.write_idx[0]) == 0  # advanced by all 4 (mod 2)
    for slot in range(2):
        lo, hi = int(buf.store_lo[0, slot]), int(buf.store_hi[0, slot])
        assert lo in range(N) and hi == 8
        # Each frame f of a slot comes from some env e that recorded 100 e + f there.
        frames = buf.store[0, slot, lo:hi, 0]
        torch.testing.assert_close(torch.remainder(frames, 100.0), torch.arange(lo, hi).float())


def test_an_episode_that_ran_past_the_recording_restarts_only_within_its_recorded_frames():
    """Frames from ``max_len - 1`` on share the last slot, so a restart must land before it."""
    buf = _buffer()
    _run(buf, MAX_LEN + 3)
    _end(buf, [20.0] * N, end_frame=MAX_LEN + 3)
    assert buf.store_hi[0].max() == MAX_LEN - 1
    for _ in range(20):
        mask, states, frames = buf.sample_restarts(torch.ones(N, dtype=torch.bool), torch.tensor([0, 1, 0, 1]))
        assert (frames[mask] < MAX_LEN - 1).all()
        torch.testing.assert_close(states[mask][:, 0], frames[mask].float())
