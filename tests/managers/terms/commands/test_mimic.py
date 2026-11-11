"""Tests for ``eden.managers.terms.commands.mimic``: the BeyondMimic clip loaders and ``MotionCommand``.

Clips are ``.npz`` files under ``tmp_path``; the command term is built against a real G1.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eden.managers.terms.commands.mimic import (
    MotionCommand,
    MotionLoader,
    MultiMotionLoader,
    motion_command_options,
)
from eden.options.robots.unitree_g1 import G1_DOFS29_TRACKED_LINKS_NAME

NUM_DOFS = 29
CPU = torch.device("cpu")
LINKS = ["pelvis", "torso_link", "left_ankle_roll_link"]


def _write_clip(path, n_frames: int, body_names: list[str] | None = None, n_bodies: int = 4, seed: int = 0):
    """A BeyondMimic clip; with ``body_names``, each body column holds its own index as a tag."""
    rng = np.random.RandomState(seed)
    if body_names is not None:
        n_bodies = len(body_names)
        tag = np.broadcast_to(np.arange(n_bodies, dtype=np.float32)[None, :, None], (n_frames, n_bodies, 3)).copy()
        pos = lin = ang = tag
    else:
        pos, lin, ang = (rng.randn(n_frames, n_bodies, 3).astype(np.float32) for _ in range(3))
    quat = rng.randn(n_frames, n_bodies, 4).astype(np.float32)
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    arrays = dict(
        joint_pos=rng.randn(n_frames, NUM_DOFS).astype(np.float32),
        joint_vel=rng.randn(n_frames, NUM_DOFS).astype(np.float32),
        body_pos_w=pos,
        body_quat_w=quat,
        body_lin_vel_w=lin,
        body_ang_vel_w=ang,
    )
    if body_names is not None:
        arrays["body_names"] = np.array(body_names)
    np.savez(path, **arrays)
    return str(path)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def test_named_body_columns_resolve_by_name_not_by_link_index(tmp_path):
    """The invariant that survives Genesis's BFS->DFS link reorder (1.2.0)."""
    clip = _write_clip(tmp_path / "named.npz", 4, body_names=["b3", "b0", "b2", "b1"])

    loader = MotionLoader(clip, torch.tensor([0, 1]), CPU, links_name=["b0", "b1"])  # the index would pick 0, 1

    assert loader.links_pos_w[0, :, 0].tolist() == [1.0, 3.0]
    with pytest.raises(ValueError, match=r"'body_names' is missing"):
        MotionLoader(clip, torch.tensor([0]), CPU, links_name=["b0", "nonexistent_link"])


def test_each_clip_of_a_library_resolves_its_own_body_order(tmp_path):
    a = _write_clip(tmp_path / "a.npz", 5, body_names=["b0", "b1", "b2", "b3"])
    b = _write_clip(tmp_path / "b.npz", 5, body_names=["b3", "b2", "b1", "b0"])

    loader = MultiMotionLoader([a, b], torch.tensor([0, 0]), CPU, links_name=["b0", "b2"])

    assert loader.links_pos_w[0, 0, :, 0].tolist() == [0.0, 2.0]
    assert loader.links_pos_w[1, 0, :, 0].tolist() == [3.0, 1.0]


def test_a_legacy_g1_clip_resolves_through_the_breadth_first_body_order(tmp_path):
    """Clips without ``body_names`` predate 1.2.0 and hold the 30 G1 bodies breadth-first."""
    n = len(G1_DOFS29_TRACKED_LINKS_NAME)
    tag = np.broadcast_to(np.arange(n, dtype=np.float32)[None, :, None], (4, n, 3)).copy()
    quat = np.zeros((4, n, 4), dtype=np.float32)
    quat[..., 0] = 1.0
    path = tmp_path / "legacy_g1.npz"
    zeros = np.zeros((4, NUM_DOFS), dtype=np.float32)
    np.savez(
        path, joint_pos=zeros, joint_vel=zeros, body_pos_w=tag, body_quat_w=quat, body_lin_vel_w=tag, body_ang_vel_w=tag
    )

    loader = MotionLoader(str(path), torch.tensor([0, 1, 2]), CPU, links_name=LINKS)

    assert loader.links_pos_w[0, :, 0].tolist() == [float(G1_DOFS29_TRACKED_LINKS_NAME.index(n)) for n in LINKS]


def test_an_unrecognised_legacy_clip_falls_back_to_the_link_index_and_warns(tmp_path, eden_caplog):
    clip = _write_clip(tmp_path / "legacy.npz", 4, seed=7)  # no body_names, not G1 bodies

    loader = MotionLoader(clip, torch.tensor([0, 2]), CPU, links_name=["b0", "b1"])

    assert any(record.levelname == "WARNING" for record in eden_caplog.records)
    np.testing.assert_allclose(loader.links_pos_w.numpy(), np.load(clip)["body_pos_w"][:, [0, 2]])


def test_a_library_pads_short_clips_with_their_last_frame(tmp_path):
    a = _write_clip(tmp_path / "a.npz", 5, seed=1)
    b = _write_clip(tmp_path / "b.npz", 8, seed=2)

    loader = MultiMotionLoader([a, b], torch.tensor([0, 2]), CPU)

    assert loader.frame_total.tolist() == [5, 8] and loader.time_step_total == 8
    for t in (5, 6, 7):
        assert torch.equal(loader.dofs_pos[0, t], loader.dofs_pos[0, 4])
        assert torch.equal(loader.links_pos_w[0, t], loader.links_pos_w[0, 4])
    np.testing.assert_allclose(loader.links_pos_w[0, :5].numpy(), np.load(a)["body_pos_w"][:, [0, 2]])


def test_a_single_clip_library_is_the_single_clip_loader(tmp_path):
    a = _write_clip(tmp_path / "a.npz", 6, seed=7)
    body_idx = torch.tensor([0, 2, 3])

    single, library = MotionLoader(a, body_idx, CPU), MultiMotionLoader([a], body_idx, CPU)

    for field in ("dofs_pos", "dofs_vel", "links_pos_w", "links_quat_w", "links_lin_vel_w", "links_ang_vel_w"):
        assert torch.equal(getattr(single, field), getattr(library, field)[0]), field


def test_a_library_rejects_mismatched_frame_counts_and_no_clips(tmp_path):
    path = tmp_path / "bad.npz"
    rng = np.random.RandomState(0)
    fields = {k: rng.randn(6, 3, 3).astype(np.float32) for k in ("body_quat_w", "body_lin_vel_w", "body_ang_vel_w")}
    np.savez(
        path,
        joint_pos=rng.randn(6, NUM_DOFS),
        joint_vel=rng.randn(6, NUM_DOFS),
        body_pos_w=rng.randn(4, 3, 3),
        **fields,
    )

    with pytest.raises(ValueError, match="'body_pos_w' has 4 frames but joint_pos has 6"):
        MultiMotionLoader([str(path)], torch.tensor([0, 1, 2]), CPU)
    with pytest.raises(ValueError, match="empty"):
        MultiMotionLoader([], torch.tensor([0]), CPU)


# ---------------------------------------------------------------------------
# MotionCommand against a built G1
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def env():
    import eden as en
    from eden.options.envs import EnvOptions
    from eden.options.robots.unitree_g1 import G1Dofs29
    from eden.utils.configs import EdenRLConfig
    from tests.support.envs import build_env

    from eden import terminations
    from eden.options.managers.terminations import TerminationManagerOptions
    from eden.terms import DoneTerm

    # The G1 terminates below 0.3 m, so a test can decide which episodes failed.
    low = DoneTerm.configure(
        func=terminations.root_height_below_minimum, params={"entity_name": "robot", "minimum_height": 0.3}
    )
    return build_env(
        EdenRLConfig(
            env_options=EnvOptions(num_envs=4),
            scene_options=en.scenes.Empty(robot=G1Dofs29()),
            termination_options=TerminationManagerOptions(low=low),
        )
    )


def _command(env, files, **overrides):
    overrides = {"anchor_link_name": "torso_link"} | overrides
    options = motion_command_options(files, links_name=LINKS, **overrides)["motion"]
    term = MotionCommand(env, options)
    term.build()
    return term


@pytest.fixture
def two_clips(tmp_path):
    return [
        _write_clip(tmp_path / "a.npz", 5, body_names=LINKS, seed=1),
        _write_clip(tmp_path / "b.npz", 8, body_names=LINKS, seed=2),
    ]


def test_a_single_clip_is_read_verbatim_from_the_npz(env, tmp_path):
    clip = _write_clip(tmp_path / "a.npz", 10, body_names=["x", *LINKS], seed=3)
    term = _command(env, clip)
    frames = torch.randint(0, 10, (4, 3))
    raw = np.load(clip)

    for field, key in (("dofs_pos", "joint_pos"), ("dofs_vel", "joint_vel"), ("links_pos_w", "body_pos_w")):
        verbatim = torch.from_numpy(raw[key]).float()
        if field.startswith("links"):
            verbatim = verbatim[:, 1:]  # the file's column 0 is not tracked
        expected = verbatim[frames.reshape(-1)].reshape(4, 3, *verbatim.shape[1:])
        assert torch.equal(term.gather_motion_frames(field, frames).cpu(), expected), field
    assert term.frame_total_per_env().tolist() == [10] * 4


def test_each_env_reads_its_own_clip_and_frame(env, two_clips):
    term = _command(env, two_clips)
    library = MultiMotionLoader(two_clips, torch.tensor([0]), CPU, links_name=LINKS)
    term.motion_id = torch.tensor([0, 1, 0, 1], device=env.device)
    term.time_steps = torch.tensor([2, 3, 4, 7], device=env.device)

    dofs = term.dofs_pos.cpu()

    for e, (m, t) in enumerate([(0, 2), (1, 3), (0, 4), (1, 7)]):
        assert torch.equal(dofs[e], library.dofs_pos[m, t])


def test_frame_total_is_each_envs_own_clip_length(env, two_clips):
    term = _command(env, two_clips)

    assert term.frame_total_per_env().tolist() == [[5, 8][m] for m in term.motion_id.tolist()]


@pytest.mark.parametrize("base_link_name, row", [(None, 0), ("torso_link", 1)])
def test_base_accessors_read_the_base_link_row(env, tmp_path, base_link_name, row):
    term = _command(env, _write_clip(tmp_path / "a.npz", 5, body_names=LINKS), base_link_name=base_link_name)

    for base, links in (
        ("base_pos_w", "links_pos_w"),
        ("base_quat_w", "links_quat_w"),
        ("base_ang_vel_w", "links_ang_vel_w"),
    ):
        assert torch.equal(getattr(term, base), getattr(term, links)[:, row]), base


@pytest.mark.parametrize("field", ["anchor_link_name", "base_link_name"])
def test_special_links_outside_links_name_fail_at_build(env, tmp_path, field):
    with pytest.raises(ValueError, match=f"{field}='nonexistent' must be a member of links_name"):
        _command(env, _write_clip(tmp_path / "a.npz", 5, body_names=LINKS), **{field: "nonexistent"})


def test_one_or_many_clips_configure_the_same_term():
    single = motion_command_options("a.npz", anchor_link_name="pelvis", links_name=["pelvis"])["motion"]
    many = motion_command_options(["a.npz", "b.npz"], anchor_link_name="pelvis", links_name=["pelvis"])["motion"]

    assert single.name == many.name == "motion_command"
    assert single.motion_file == "a.npz" and list(many.motion_files) == ["a.npz", "b.npz"]


# ---------------------------------------------------------------------------
# The two clocks: simulation (wraps and resamples) and deployment (holds the last frame)
# ---------------------------------------------------------------------------


@pytest.fixture
def clips_of_1_3_5(tmp_path):
    return [_write_clip(tmp_path / f"clip_{n}.npz", n, body_names=LINKS, seed=n) for n in (1, 3, 5)]


def test_deployment_playback_holds_each_clip_at_its_own_last_frame(env, clips_of_1_3_5):
    """No wrap and no resample: the clip assignment stands and every clock stops at its clip's end."""
    term = _command(env, clips_of_1_3_5)
    assigned = term.motion_id.clone()

    for _ in range(8):
        term.advance_reference()

    torch.testing.assert_close(term.time_steps, term.frame_total_per_env() - 1)
    torch.testing.assert_close(term.motion_id, assigned)
    assert term.clip_finished().all()


def test_a_clip_is_not_finished_until_its_last_frame(env, clips_of_1_3_5):
    term = _command(env, clips_of_1_3_5[1])  # 3 frames

    finished = []
    for _ in range(3):
        finished.append(bool(term.clip_finished().any()))
        term.advance_reference()

    assert finished == [False, False, True]


def test_simulation_wraps_and_restarts_the_clip_so_it_is_never_finished(env, clips_of_1_3_5):
    """Frames 1, 2, then the wrap to 0 at 3: seven advances land on frame 1, having wrapped twice."""
    term = _command(env, clips_of_1_3_5[1])

    for step in range(7):
        term.pre_physics_advance()
        if step == 2:
            assert not term.clip_finished().any(), "a wrapped clip is not a finished one"

    assert term.time_steps.tolist() == [1] * 4


def test_a_rewind_restarts_the_clock_and_disarms_the_resampling_timer(env, clips_of_1_3_5):
    term = _command(env, clips_of_1_3_5[1])
    term.advance_reference()
    term.time_left.zero_()  # the training timer has run out

    term.rewind_reference()

    assert term.time_steps.tolist() == [0] * 4
    assert torch.isinf(term.time_left).all(), "compute() would otherwise resample and re-place the robot"


# ---------------------------------------------------------------------------
# Resampling and the reset teleport
# ---------------------------------------------------------------------------


@pytest.fixture
def long_clip(tmp_path):
    """4 s at 50 Hz: five one-second adaptive bins."""
    return _write_clip(tmp_path / "long.npz", 200, body_names=LINKS, seed=4)


@pytest.mark.parametrize("sampling_mode", ["adaptive", "uniform", "start"])
def test_a_resample_restarts_and_teleports_only_the_selected_envs(env, long_clip, sampling_mode):
    term = _command(env, long_clip, sampling_mode=sampling_mode)
    robot = env.entities["robot"]
    term.time_steps = torch.arange(4, device=env.device) + 10
    time_steps, start_bin, qpos = term.time_steps.clone(), term.start_bin.clone(), robot.get_qpos().clone()
    mask = torch.tensor([True, False, True, False], device=env.device)

    term._resample_command(mask)

    assert torch.equal(term.time_steps[~mask], time_steps[~mask])
    assert torch.equal(term.start_bin[~mask], start_bin[~mask])
    torch.testing.assert_close(robot.get_qpos()[~mask], qpos[~mask])
    assert ((term.time_steps[mask] >= 0) & (term.time_steps[mask] < 200)).all()
    if sampling_mode == "start":
        assert (term.time_steps[mask] == 0).all()
    # The selected envs are on the new frame, with its velocity: set_qpos after set_dofs_vel must not zero it.
    limits = robot.soft_dofs_pos_limits
    torch.testing.assert_close(
        robot.get_qpos()[mask, 7:], torch.clip(term.dofs_pos, limits[..., 0], limits[..., 1])[mask]
    )
    torch.testing.assert_close(robot.get_dofs_vel()[mask], term.dofs_vel[mask])
    assert term.dofs_vel[mask].abs().min() > 0


def test_adaptive_sampling_counts_the_selected_episodes_that_ran(env, long_clip):
    term = _command(env, long_clip, sampling_mode="adaptive")
    assert term.bin_count == 5
    term.start_bin.copy_(torch.tensor([0, 1, 1, 3], device=env.device))
    env.entities["robot"].set_pos(torch.tensor([[0.0, 0.0, 0.1]] * 3 + [[0.0, 0.0, 0.8]], device=env.device))
    env.termination_manager.compute()  # envs 0-2 fell
    env.episode_length_buf[:] = torch.tensor([5, 0, 5, 5])  # env 1 is a bootstrap reset

    term._adaptive_sampling(torch.tensor([True, True, False, True], device=env.device))  # env 2 is not resetting

    # The start-bin tallies feed the next EMA and have no other reader.
    assert term._current_bin_starts.tolist() == [1.0, 0.0, 0.0, 1.0, 0.0]  # envs 0 and 3
    assert term._current_bin_failed.tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]  # env 0 failed


def test_redrawing_clips_changes_only_the_selected_envs(env, two_clips):
    term = _command(env, two_clips, sampling_mode="uniform")
    mask = torch.tensor([True, False, True, False], device=env.device)
    drawn = set()

    for _ in range(20):
        term.motion_id.zero_()
        term._sample_motions(mask)
        assert (term.motion_id[~mask] == 0).all()
        assert term.frame_total_per_env().tolist() == [[5, 8][m] for m in term.motion_id.tolist()]
        drawn |= set(term.motion_id[mask].tolist())

    assert drawn == {0, 1}


def test_a_resample_and_a_clip_wrap_cost_no_sync_of_their_own(cuda_motion_command, count_cuda_syncs):
    """The wrap's ``.any()`` gate is the one sync; a gather through the mask would cost ~20 apiece."""
    term = cuda_motion_command
    mask = torch.tensor([True, False, True, False], device="cuda")
    # A first call builds the coefficient tables of Genesis' geom helpers, cached per device and dtype
    # (https://github.com/Genesis-Embodied-AI/Genesis/pull/3527): 11 syncs, once per process.
    term._resample_command(mask)
    assert count_cuda_syncs(lambda: term._resample_command(mask)) == 0

    term.time_steps.copy_(torch.tensor([49, 0, 0, 0], device="cuda"))  # env 0's clip ends
    assert count_cuda_syncs(term.pre_physics_advance) == 1
    term.time_steps.zero_()
    assert count_cuda_syncs(term.pre_physics_advance) == 1  # nothing ends: only the gate
