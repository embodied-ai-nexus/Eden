"""The Extreme Parkour observation terms, on a real teacher env (``parkour_env``, 4 envs, 2x2 terrain grid).

The stateful terms (``ParkourProprio``: contact latch, delta-yaw cache, 10-frame history; ``ParkourScanDots``;
``ParkourDepthImage``) advance in ``post_physics_refresh`` and rebuild a respawned env in ``post_reset``.
Frames are identified by value: a history slot is compared to the proprio block an earlier step observed, the
rate-limited scan to a ``ParkourScanDots`` probe that refreshes every step, and a depth frame to its render count.
"""

from __future__ import annotations

import pytest
import torch

from eden.envs.base import RLEnvBase
from eden.tasks.extreme_parkour.custom_terms.observations import (
    ParkourDeltaYawOk,
    ParkourProprioHistory,
    process_parkour_depth,
)
from eden.utils.geom import quat_from_euler_xyz

pytestmark = pytest.mark.requires_genesis

INTERVAL = 5  # the shipped update_interval of the proprio delta-yaw cache, the scan and the depth frame
HISTORY = 10
# Slots of the 753-d policy group (see the observations module docstring).
PROPRIO, SCAN, HIST = slice(0, 53), slice(53, 185), slice(223, 753)
DELTA_YAW = slice(6, 8)
CONTACT = slice(49, 53)
DOWN, UP = 0.5, -0.5  # the contact slots, latched / clear


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _step(env, done=()):
    """One zero-action step; ``done`` envs time out on it (the real auto-reset path)."""
    for i in done:
        env.episode_length_buf[i] = env.max_episode_length - 1
    obs, *_, extras = env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device))
    return obs, extras


def _reset_on_due_step(env):
    """Full reset at a counter that is a multiple of ``INTERVAL``; later steps are then numbered from 0."""
    while env.common_step_counter % INTERVAL:
        _step(env)
    env.observation_manager.get_term("depth").render_count = 0
    obs, _ = env.reset()
    return obs


def _frame(obs) -> torch.Tensor:
    """The proprio block as the history stores it: with the delta-yaw hints zeroed."""
    frame = obs["policy"][:, PROPRIO].clone()
    frame[:, DELTA_YAW] = 0.0
    return frame


def _history(obs) -> torch.Tensor:
    return obs["policy"][:, HIST].reshape(-1, HISTORY, 53)


def _assert_history(obs, env_i: int, frames: list[torch.Tensor]):
    """Env ``env_i``'s history holds exactly ``frames`` (each a per-env ``_frame``), oldest first."""
    expected = torch.stack([f[env_i] for f in frames])
    torch.testing.assert_close(_history(obs)[env_i], expected, atol=1e-6, rtol=0)


def _scan(obs) -> torch.Tensor:
    return obs["policy"][:, SCAN]


def _probe(obs) -> torch.Tensor:
    return obs["scan_probe"]


def _yaw_robot(env, envs_idx: list[int], offset: float):
    """Turn the given robots to ``target_yaw - offset``, so their delta-yaw becomes ~``offset``."""
    command = env.command_manager.get_term("parkour_command")
    idx = torch.tensor(envs_idx, device=env.device)
    yaw = command.target_yaw[idx] - offset
    zeros = torch.zeros_like(yaw)
    env.entities["robot"].set_quat(quat_from_euler_xyz(zeros, zeros, yaw), envs_idx=idx)


# ---------------------------------------------------------------------------
# ParkourProprio + ParkourProprioHistory
# ---------------------------------------------------------------------------


def test_history_is_the_pre_append_snapshot_seeded_with_the_spawn_frame(parkour_env):
    env = parkour_env
    obs0 = _reset_on_due_step(env)
    assert _history(obs0).eq(0).all()  # post_reset seeded the buffer; nothing is published yet

    obs1, _ = _step(env)  # the episode's first tick publishes the spawn frame x10 ...
    obs2, _ = _step(env)  # ... and backfills the buffer with its own frame
    obs3, _ = _step(env)
    f0, f1, f2 = _frame(obs0), _frame(obs1), _frame(obs2)
    assert not torch.equal(f0, f1), "the robot did not move - the frames would not tell steps apart"
    for i in range(env.num_envs):
        _assert_history(obs1, i, [f0] * HISTORY)
        _assert_history(obs2, i, [f1] * HISTORY)
        _assert_history(obs3, i, [f1] * (HISTORY - 1) + [f2])


def test_randomized_episode_lengths_still_start_from_the_spawn_frame(parkour_env):
    """``learn(init_at_random_ep_len=True)`` overwrites ``episode_length_buf`` between reset and step 1."""
    env = parkour_env
    obs0 = _reset_on_due_step(env)
    env.episode_length_buf[:] = torch.tensor([7, 0, 300, 0], device=env.device)
    obs1, _ = _step(env)
    obs2, _ = _step(env)
    f0, f1 = _frame(obs0), _frame(obs1)
    for i in range(env.num_envs):
        _assert_history(obs1, i, [f0] * HISTORY)
    for i in (0, 2):  # no first-tick backfill: the seeded spawn frame rolls forward
        _assert_history(obs2, i, [f0] * (HISTORY - 1) + [f1])
    for i in (1, 3):
        _assert_history(obs2, i, [f1] * HISTORY)


def test_the_history_carries_no_delta_yaw(parkour_env):
    env = parkour_env
    _reset_on_due_step(env)
    _yaw_robot(env, list(range(env.num_envs)), offset=0.7)
    for _ in range(INTERVAL + 1):  # the cache refreshes at step 5; step 6's history holds frame 5
        obs, _ = _step(env)
    assert (obs["policy"][:, 6].abs() > 0.3).all()
    assert _history(obs)[:, :, DELTA_YAW].eq(0).all()


def test_compute_is_a_read_however_often_it_is_called(parkour_env):
    """What a runner's ``get_observations()`` does between steps must not advance any state."""
    env = parkour_env
    obs0 = _reset_on_due_step(env)
    frames = [_frame(obs0)]
    for _ in range(3):
        obs, _ = _step(env)
        again = env.observation_manager.compute()
        env.observation_manager.compute()
        torch.testing.assert_close(again["policy"], obs["policy"], atol=0, rtol=0)
        frames.append(_frame(obs))
    obs4, _ = _step(env)
    for i in range(env.num_envs):
        _assert_history(obs4, i, [frames[1]] * (HISTORY - 2) + frames[2:4])


def test_contact_latch_holds_a_lost_contact_for_one_step(parkour_env):
    env = parkour_env
    robot = env.entities["robot"]
    _reset_on_due_step(env)
    for _ in range(40):  # settle onto the feet
        obs, _ = _step(env)
    assert torch.equal(obs["policy"][:, CONTACT], torch.full((env.num_envs, 4), DOWN, device=env.device))

    lifted = robot.get_pos()[:1].clone()
    lifted[:, 2] += 1.5
    robot.set_pos(lifted, envs_idx=torch.tensor([0], device=env.device))
    obs, _ = _step(env)  # in the air now, but the previous step's contact is latched
    assert obs["policy"][0, CONTACT].eq(DOWN).all()
    obs, _ = _step(env)
    assert obs["policy"][0, CONTACT].eq(UP).all()
    assert obs["policy"][1, CONTACT].eq(DOWN).all()


@pytest.mark.parametrize("auto_reset", [False, True], ids=["manual_reset", "auto_reset"])
def test_a_reset_does_not_carry_the_ended_episodes_contact_into_the_new_one(parkour_env, auto_reset):
    """Genesis refreshes the contact-force buffer only in a physics step, so after a respawn it still holds the ended
    episode's forces; the latch must not be rebuilt from it.
    """
    env = parkour_env
    _reset_on_due_step(env)
    for _ in range(40):
        _step(env)
    term = env.observation_manager.get_term("proprio")
    assert term._last_contacts[0].any(), "the robot never touched down - the test would prove nothing"

    if auto_reset:
        obs, _ = _step(env, done=[0])
    else:
        obs, _ = env.reset(torch.tensor([0], device=env.device))
    assert obs["policy"][0, CONTACT].eq(UP).all()
    assert obs["policy"][1, CONTACT].eq(DOWN).all()  # an env that did not reset keeps its latch


@pytest.mark.parametrize("record_final_observations", [False, True])
def test_what_a_reset_env_observes_does_not_depend_on_record_final_observations(parkour_env, record_final_observations):
    """Env 1 times out on step 43 (not a due step): with the old compute()-stamp the flag moved the state advance
    across the reset. Now it only adds the pre-reset snapshot.
    """
    env = parkour_env
    previous_flag = env.env_options.record_final_observations
    env.env_options.record_final_observations = record_final_observations
    try:
        obs = _reset_on_due_step(env)
        trace = [obs]
        for _ in range(42):  # settle, so the pre-reset tick measures real contact
            trace.append(_step(env)[0])
        obs43, extras43 = _step(env, done=[1])
        # The env reuses its snapshot buffers across steps: keep this step's rows.
        final = {k: v.clone() for k, v in extras43.get("final_observations", {}).items()}
        obs44, _ = _step(env)
        _yaw_robot(env, [1], offset=0.7)
        obs45, _ = _step(env)
    finally:
        env.env_options.record_final_observations = previous_flag
    frames = [_frame(o) for o in trace]

    # The reset step: a cleared latch, zero history, and zero delta-yaw and scan (their caches wait for step 45) ...
    assert obs43["policy"][1, CONTACT].eq(UP).all()
    assert obs43["policy"][1, DELTA_YAW].eq(0).all()
    assert _history(obs43)[1].eq(0).all()
    assert _scan(obs43)[1].eq(0).all()
    # ... while the other envs keep counting ...
    assert obs43["policy"][0, CONTACT].eq(DOWN).all()
    _assert_history(obs43, 0, frames[33:43])
    torch.testing.assert_close(_scan(obs43)[0], _probe(trace[40])[0], atol=0, rtol=0)
    # ... env 1's history is the spawn frame x10 one step later, its first tick's frame x10 the step after, and
    # delta-yaw and scan return at the next due step.
    _assert_history(obs44, 1, [_frame(obs43)] * HISTORY)
    assert obs44["policy"][1, DELTA_YAW].eq(0).all() and _scan(obs44)[1].eq(0).all()
    _assert_history(obs45, 1, [_frame(obs44)] * HISTORY)
    assert obs45["policy"][1, 6].abs() > 0.3
    torch.testing.assert_close(_scan(obs45)[1], _probe(obs45)[1], atol=0, rtol=0)

    if record_final_observations:
        _assert_history(final, 1, frames[33:43])
        assert final["policy"][1, CONTACT].eq(DOWN).all()
    else:
        assert "final_observations" not in extras43


@pytest.mark.parametrize("sibling", [ParkourProprioHistory, ParkourDeltaYawOk])
def test_a_sibling_reader_without_a_proprio_built_before_it_fails_at_build(small_teacher_config, sibling):
    """Terms build in declaration order and the sibling handshake happens at build, so the owner must come first."""
    unbuilt = RLEnvBase.from_config(small_teacher_config())
    with pytest.raises(RuntimeError, match="requires a ParkourProprio term declared"):
        sibling(unbuilt, sibling.configure()).build()


# ---------------------------------------------------------------------------
# ParkourScanDots
# ---------------------------------------------------------------------------


def test_scan_is_taken_at_the_reset_and_refreshed_at_every_due_tick(parkour_env):
    env = parkour_env
    obs = _reset_on_due_step(env)
    probes = [_probe(obs)]
    torch.testing.assert_close(_scan(obs), probes[0], atol=0, rtol=0)
    for step in range(1, 2 * INTERVAL + 1):
        obs, _ = _step(env)
        probes.append(_probe(obs))
        torch.testing.assert_close(_scan(obs), probes[step - step % INTERVAL], atol=0, rtol=0, msg=f"step {step}")
    assert not torch.equal(probes[1], probes[0]), "the scan never changed - the test would prove nothing"


def test_a_due_step_reset_rescans_the_restarting_env_around_its_spawn(parkour_env):
    env = parkour_env
    _reset_on_due_step(env)
    for _ in range(INTERVAL - 1):
        _step(env)
    obs, _ = _step(env, done=[0])  # step 5: due, and env 0 respawns
    torch.testing.assert_close(_scan(obs), _probe(obs), atol=0, rtol=0)


def test_a_non_due_step_reset_zeroes_the_scan_until_the_next_due_step(parkour_env):
    env = parkour_env
    _reset_on_due_step(env)
    for _ in range(INTERVAL):
        obs5, _ = _step(env)
    _step(env)
    obs, _ = _step(env, done=[0])  # step 7
    assert _scan(obs)[0].eq(0).all()
    torch.testing.assert_close(_scan(obs)[1], _probe(obs5)[1], atol=0, rtol=0)
    for _ in range(2):
        obs, _ = _step(env)
        assert _scan(obs)[0].eq(0).all()
    obs, _ = _step(env)  # step 10
    torch.testing.assert_close(_scan(obs), _probe(obs), atol=0, rtol=0)


# ---------------------------------------------------------------------------
# ParkourDepthImage (render #k is a frame filled with k)
# ---------------------------------------------------------------------------


def _depth(obs) -> list[float]:
    return obs["depth_camera"][:, 0, 0].tolist()


def test_depth_renders_at_the_reset_then_only_at_due_ticks_and_serves_the_older_slot(parkour_env):
    env = parkour_env
    term = env.observation_manager.get_term("depth")
    obs = _reset_on_due_step(env)
    # A reset on a due step renders into both slots, so the first observation is a real frame.
    assert term.render_count == 1
    assert _depth(obs) == [1.0] * 4
    for step in range(1, 16):
        obs, _ = _step(env)
        assert term.render_count == 1 + step // INTERVAL, step
    # Render #3 (step 10) is served from step 15 on: one interval of latency.
    assert _depth(obs) == [3.0] * 4
    obs, _ = _step(env)
    assert _depth(obs) == [3.0] * 4 and term.render_count == 4


def test_depth_non_due_step_reset_never_renders_and_backfills_at_the_next_due_tick(parkour_env):
    env = parkour_env
    term = env.observation_manager.get_term("depth")
    _reset_on_due_step(env)
    for _ in range(2 * INTERVAL + 1):  # step 11: older slot = render 2, newer = render 3
        _step(env)
    obs, _ = _step(env, done=[1])  # step 12
    assert term.render_count == 3
    assert _depth(obs)[1] == 2.0  # the reset env keeps its pre-reset view until the due step
    for _ in range(2):
        obs, _ = _step(env)
    assert term.render_count == 3 and _depth(obs)[1] == 2.0
    obs, _ = _step(env)  # step 15: render 4 rolls the others (older slot = 3) and backfills env 1 (4)
    assert term.render_count == 4
    assert _depth(obs) == [3.0, 4.0, 3.0, 3.0]
    for _ in range(INTERVAL):
        obs, _ = _step(env)
    assert _depth(obs) == [4.0] * 4  # step 20: env 1's newer slot was backfilled too


def test_depth_due_step_reset_rerenders_after_the_respawn_for_the_reset_env_only(parkour_env):
    env = parkour_env
    term = env.observation_manager.get_term("depth")
    _reset_on_due_step(env)
    for _ in range(2 * INTERVAL - 1):
        _step(env)
    obs, _ = _step(env, done=[1])  # step 10: the tick's render 3 rolls everyone; post_reset renders 4
    assert term.render_count == 4
    assert _depth(obs) == [2.0, 4.0, 2.0, 2.0]  # env 1 sees the respawn at once
    for _ in range(INTERVAL):
        obs, _ = _step(env)
    assert term.render_count == 5
    assert _depth(obs) == [3.0, 4.0, 3.0, 3.0]  # the others' pair was not rolled twice
    for _ in range(INTERVAL):
        obs, _ = _step(env)
    assert _depth(obs) == [5.0] * 4


def test_depth_manual_reset_renders_on_a_due_counter_and_marks_pending_otherwise(parkour_env):
    env = parkour_env
    term = env.observation_manager.get_term("depth")
    _reset_on_due_step(env)
    for _ in range(2 * INTERVAL):
        _step(env)
    obs, _ = env.reset(torch.tensor([0], device=env.device))  # step 10: due
    assert term.render_count == 4
    assert _depth(obs) == [4.0, 2.0, 2.0, 2.0]
    _step(env)
    obs, _ = env.reset(torch.tensor([2], device=env.device))  # step 11: not due, pending until 15
    assert term.render_count == 4
    assert _depth(obs) == [4.0, 2.0, 2.0, 2.0]
    for _ in range(INTERVAL - 1):
        obs, _ = _step(env)
    assert term.render_count == 5
    assert _depth(obs) == [4.0, 3.0, 5.0, 3.0]


def test_depth_on_a_reset_step_does_not_depend_on_record_final_observations(parkour_env):
    env = parkour_env
    term = env.observation_manager.get_term("depth")
    previous_flag = env.env_options.record_final_observations
    traces = []
    try:
        for flag in (True, False):
            env.env_options.record_final_observations = flag
            _reset_on_due_step(env)
            trace = []
            for step in range(1, 3 * INTERVAL + 1):
                obs, _ = _step(env, done=[2] if step == 2 * INTERVAL else [])  # a reset on a due step
                trace.append(_depth(obs))
            # The reset + 3 ticks + the due-step reset; the pre-reset snapshot did not render.
            assert term.render_count == 5
            traces.append(trace)
    finally:
        env.env_options.record_final_observations = previous_flag
    assert traces[0] == traces[1]


def test_process_parkour_depth_clips_and_normalizes():
    """Constant 1 m maps to 0 exactly; sky (+inf) clamps to far (+0.5); below-near clamps to near (-0.5)."""
    kwargs = {"near_clip": 0.0, "far_clip": 2.0, "resized_hw": (58, 87)}
    for value, expected in ((1.0, 0.0), (float("inf"), 0.5), (-3.0, -0.5)):
        out = process_parkour_depth(torch.full((2, 60, 106), value), **kwargs)
        torch.testing.assert_close(out, torch.full((2, 58, 87), expected), atol=1e-6, rtol=0)


def test_process_parkour_depth_discards_the_crop_margins():
    """The bottom 2 rows and 4 columns each side must not leak into the resized frame."""
    base = torch.full((1, 60, 106), 1.0)
    marked = base.clone()
    marked[:, -2:, :] = 2.0
    marked[:, :, :4] = 2.0
    marked[:, :, -4:] = 2.0
    kwargs = {"near_clip": 0.0, "far_clip": 2.0, "resized_hw": (58, 87)}
    assert torch.equal(process_parkour_depth(base, **kwargs), process_parkour_depth(marked, **kwargs))


# ---------------------------------------------------------------------------
# ParkourDeltaYawOk
# ---------------------------------------------------------------------------


def test_delta_yaw_gate_reads_the_cache_the_tick_refreshed(parkour_env):
    env = parkour_env
    obs0 = _reset_on_due_step(env)
    gate0 = obs0["delta_yaw_ok"].squeeze(1).tolist()
    _yaw_robot(env, [1, 2], offset=1.0)
    for step in range(1, INTERVAL + 1):
        obs, _ = _step(env)
        gate = obs["delta_yaw_ok"].squeeze(1)
        # The gate and slot 6 read the same cache on every step.
        assert torch.equal(gate, (obs["policy"][:, 6] < 0.6).float()), step
        if step < INTERVAL:
            assert gate.tolist() == gate0, step  # not refreshed until the due step
    assert gate[1] == 0.0 and gate[2] == 0.0
