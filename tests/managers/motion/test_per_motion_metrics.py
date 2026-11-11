"""Tests for ``eden.managers.motion.per_motion_metrics``: per-motion tracking-error aggregation.

Per-env errors go into a real term's ``PerMotionMetrics`` (directly, or from the built G1 through
``MotionManager.compute``), are scattered per motion at reset and drained with the curriculum.
"""

from __future__ import annotations

import pytest
import torch

from tests.support.motion import (
    ROBOT_LINKS,
    make_manager,
    pin_clips,
    term_options,
    write_clip,
    write_library,
)
from tests.support.ddp import ddp_spawn


def _manager(env, root, *, terms=("default",), **term_kwargs):
    term_kwargs = {"adaptive_motion_weights": True, "compute_per_motion_metrics": True} | term_kwargs
    return make_manager(env, terms={name: term_options(root, **term_kwargs) for name in terms})


def _metrics(term):
    return term._metrics


def _accumulate(metrics, mpjpe, g_mpjpe=None):
    mpjpe = torch.as_tensor(mpjpe, dtype=torch.float32)
    metrics.accumulate(mpjpe=mpjpe, root_pos_err=0.1 * mpjpe, root_quat_err=0.01 * mpjpe, g_mpjpe=g_mpjpe)


@pytest.fixture
def three_clips(tmp_path):
    return write_library(tmp_path, [50, 50, 50])


def test_metrics_need_the_curriculum_they_drain_with(mimic_env, three_clips):
    with pytest.raises(ValueError, match="compute_per_motion_metrics=True requires adaptive_motion_weights=True"):
        _manager(mimic_env, three_clips, adaptive_motion_weights=False)


def test_a_term_without_metrics_reports_none(mimic_env, three_clips):
    manager = make_manager(mimic_env, terms={"default": term_options(three_clips, adaptive_motion_weights=True)})

    assert manager.get_term("default").summarize_per_motion_metrics() == {}
    assert not any(key.startswith("Mimic/") for key in manager.reset())


def test_episode_means_are_summed_per_motion_at_reset(mimic_env, three_clips):
    term = _manager(mimic_env, three_clips).get_term("default")
    pin_clips(term, [2, 2, 0, 0])
    for _ in range(3):
        _accumulate(_metrics(term), [0.6, 0.6, 0.3, 0.9])

    _metrics(term).scatter_at_reset(torch.arange(4), torch.full((4,), 3))

    per_motion = term.summarize_per_motion_metrics()
    torch.testing.assert_close(per_motion["mpjpe_sum"], torch.tensor([1.2, 0.0, 1.2]))
    torch.testing.assert_close(per_motion["root_pos_err_sum"], torch.tensor([0.12, 0.0, 0.12]))
    assert per_motion["episode_count"].tolist() == [2, 0, 2]
    # The per-env sums were drained: resetting again adds only the new (zero) error.
    _metrics(term).scatter_at_reset(torch.arange(4), torch.full((4,), 3))
    torch.testing.assert_close(term.summarize_per_motion_metrics()["mpjpe_sum"], torch.tensor([1.2, 0.0, 1.2]))


def test_bootstrap_resets_record_nothing(mimic_env, three_clips):
    term = _manager(mimic_env, three_clips).get_term("default")
    _accumulate(_metrics(term), [0.5] * 4)

    _metrics(term).scatter_at_reset(torch.arange(4), torch.zeros(4, dtype=torch.long))

    assert term.summarize_per_motion_metrics()["episode_count"].tolist() == [0, 0, 0]


@pytest.mark.parametrize("as_mask", [False, True], ids=["ids", "mask"])
def test_a_term_after_env_zero_reads_its_own_episode_lengths(mimic_env, tmp_path, as_mask):
    """Local ids index the term's envs; reading the global lengths with them read other envs' (a bug)."""
    root = write_library(tmp_path, [50, 50])
    manager = _manager(mimic_env, root, terms=("first", "second"))
    term = manager.get_term("second")
    assert term.assigned_env_slice == slice(2, 4)
    pin_clips(term, [1, 0])
    _accumulate(_metrics(term), [0.4, 0.8])
    envs = torch.tensor([2, 3])
    if as_mask:
        envs = torch.tensor([False, False, True, True])

    _metrics(term).scatter_at_reset(envs, torch.tensor([0, 0, 2, 1]))

    torch.testing.assert_close(term.summarize_per_motion_metrics()["mpjpe_sum"], torch.tensor([0.8, 0.2]))


def test_a_mask_drains_only_the_envs_it_selects(mimic_env, three_clips):
    term = _manager(mimic_env, three_clips).get_term("default")
    pin_clips(term, [0, 1, 2, 0])
    _accumulate(_metrics(term), [0.2, 0.4, 0.6, 0.8])

    _metrics(term).scatter_at_reset(torch.tensor([False, True, False, True]), torch.ones(4, dtype=torch.long))
    torch.testing.assert_close(term.summarize_per_motion_metrics()["mpjpe_sum"], torch.tensor([0.8, 0.4, 0.0]))

    # The unselected envs kept their sums for their own reset.
    _metrics(term).scatter_at_reset(torch.tensor([True, False, True, False]), torch.ones(4, dtype=torch.long))
    torch.testing.assert_close(term.summarize_per_motion_metrics()["mpjpe_sum"], torch.tensor([1.0, 0.4, 0.6]))


def test_scatter_lands_on_global_motion_ids_after_a_rank_subset(mimic_env, tmp_path):
    manager = _manager(mimic_env, write_library(tmp_path, [50] * 5))
    term = manager.get_term("default")
    term.enable_trajectories(torch.tensor([2, 4]))
    term.set_envs_idx_assignments(slice(0, 4))
    pin_clips(term, [0, 1, 0, 1])
    _accumulate(_metrics(term), [0.5] * 4)

    _metrics(term).scatter_at_reset(torch.arange(4), torch.ones(4, dtype=torch.long))

    per_motion = term.summarize_per_motion_metrics()
    torch.testing.assert_close(per_motion["mpjpe_sum"], torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0]))
    assert per_motion["episode_count"].tolist() == [0, 0, 2, 0, 2]


def test_a_drain_summarises_per_motion_means_and_clears_them(mimic_env, tmp_path):
    manager = _manager(mimic_env, write_library(tmp_path, [50] * 4))
    term = manager.get_term("default")
    pin_clips(term, [0, 0, 1, 2])
    _accumulate(_metrics(term), [0.1, 0.3, 0.3, 0.2])  # per-motion means 0.2, 0.3, 0.2; motion 3 unplayed
    _metrics(term).scatter_at_reset(torch.arange(4), torch.ones(4, dtype=torch.long))

    manager.update_motion_weight_curriculum()

    summary = manager.reset()
    assert summary["Mimic/motions_with_data"] == 3.0
    assert summary["Mimic/mpjpe_mean"] == pytest.approx(0.7 / 3, abs=1e-6)
    assert summary["Mimic/mpjpe_p50"] == pytest.approx(0.2, abs=1e-6)
    assert term.summarize_per_motion_metrics()["episode_count"].tolist() == [0, 0, 0, 0]

    manager.update_motion_weight_curriculum()
    assert {k: v for k, v in manager.reset().items() if k.startswith("Mimic/")} == {"Mimic/motions_with_data": 0.0}


def test_the_metric_drain_waits_for_the_curriculum_drain(mimic_env, three_clips):
    manager = _manager(mimic_env, three_clips, motion_weight_update_every_n_iters=3)
    term = manager.get_term("default")
    pin_clips(term, [0, 0, 1, 1])
    _accumulate(_metrics(term), [0.2, 0.4, 0.3, 0.3])
    _metrics(term).scatter_at_reset(torch.arange(4), torch.ones(4, dtype=torch.long))

    for _ in range(2):
        manager.update_motion_weight_curriculum()
        assert not any(key.startswith("Mimic/") for key in manager.reset())
    manager.update_motion_weight_curriculum()

    summary = manager.reset()
    assert summary["Mimic/motions_with_data"] == 2.0
    assert summary["Mimic/mpjpe_mean"] == pytest.approx(0.3, abs=1e-6)


def test_g_mpjpe_is_tracked_only_with_an_anchor(mimic_env, three_clips):
    plain = _manager(mimic_env, three_clips).get_term("default")
    anchored = _manager(mimic_env, three_clips, metric_anchor_link_name="pelvis").get_term("default")

    assert "g_mpjpe_sum" not in plain.summarize_per_motion_metrics()
    assert "g_mpjpe_sum" in anchored.summarize_per_motion_metrics()
    # Passing g_mpjpe iff an anchor is set; anything else would silently log zeros.
    with pytest.raises(ValueError, match="g_mpjpe / _per_env_g_mpjpe_sum mismatch"):
        _accumulate(_metrics(plain), [0.0] * 4, g_mpjpe=torch.zeros(4))
    with pytest.raises(ValueError, match="g_mpjpe / _per_env_g_mpjpe_sum mismatch"):
        _accumulate(_metrics(anchored), [0.0] * 4)


def test_g_mpjpe_is_scattered_and_summarised_like_mpjpe(mimic_env, three_clips):
    manager = _manager(mimic_env, three_clips, metric_anchor_link_name="pelvis")
    term = manager.get_term("default")
    pin_clips(term, [0, 0, 1, 1])
    _accumulate(_metrics(term), [0.4] * 4, g_mpjpe=torch.tensor([0.2, 0.4, 0.6, 0.8]))
    _metrics(term).scatter_at_reset(torch.arange(4), torch.ones(4, dtype=torch.long))

    torch.testing.assert_close(term.summarize_per_motion_metrics()["g_mpjpe_sum"], torch.tensor([0.6, 1.4, 0.0]))
    manager.update_motion_weight_curriculum()
    assert manager.reset()["Mimic/g_mpjpe_mean"] == pytest.approx(0.5, abs=1e-6)


def test_compute_measures_the_robot_against_the_reference(mimic_env, tmp_path):
    """A reference 0.5 m ahead (x) of the robot's current pose: 0.5 m MPJPE and root error, and
    zero once the anchor is subtracted.
    """
    robot = mimic_env.entities["robot"]
    ls_idx = robot.find_named_links_idx_local(ROBOT_LINKS, preserve_order=True)[1]
    links = robot.get_links_pos(ls_idx_local=ls_idx).cpu()  # (N, L, 3), identical across envs
    ahead = torch.tensor([0.5, 0.0, 0.0])
    frames = 10
    overrides = {
        "robot.links_pos_w": (links[0] + ahead).expand(frames, -1, -1).numpy(),
        "robot.links_quat_w": robot.get_links_quat(ls_idx_local=ls_idx)[0].cpu().expand(frames, -1, -1).numpy(),
        "robot.base_pos_w": (robot.get_pos()[0].cpu() + ahead).expand(frames, -1).numpy(),
        "robot.base_quat_w": robot.get_quat()[0].cpu().expand(frames, -1).numpy(),
    }
    root = tmp_path / "ahead"
    root.mkdir()
    write_clip(root / "clip_00.npz", frames, overrides=overrides)
    manager = make_manager(
        mimic_env,
        terms={
            "default": term_options(
                root,
                entities=("robot",),
                adaptive_motion_weights=True,
                compute_per_motion_metrics=True,
                metric_anchor_link_name="pelvis",
            )
        },
    )
    term = manager.get_term("default")

    manager.compute(mimic_env.dt)
    manager.compute(mimic_env.dt)
    _metrics(term).scatter_at_reset(torch.arange(4), torch.full((4,), 2))

    per_motion = term.summarize_per_motion_metrics()
    torch.testing.assert_close(per_motion["mpjpe_sum"], torch.tensor([4 * 0.5]))
    torch.testing.assert_close(per_motion["root_pos_err_sum"], torch.tensor([4 * 0.5]))
    torch.testing.assert_close(per_motion["root_quat_err_sum"], torch.tensor([0.0]), atol=1e-3, rtol=0)
    torch.testing.assert_close(per_motion["g_mpjpe_sum"], torch.tensor([0.0]), atol=1e-5, rtol=0)


def _ddp_metrics_worker(rank, world_size, out_dir, root):
    import eden as en

    en.init(debug=True)
    import torch

    from eden.options.entities import BoxOptions
    from eden.options.envs import EnvOptions
    from eden.utils.configs import EdenConfig
    from tests.support.motion import make_manager, pin_clips, term_options
    from tests.support.envs import build_env

    env = build_env(EdenConfig(env_options=EnvOptions(num_envs=2), scene_options=en.scenes.Empty(robot=BoxOptions())))
    manager = make_manager(
        env,
        terms={"default": term_options(root, adaptive_motion_weights=True, compute_per_motion_metrics=True)},
    )
    term = manager.get_term("default")
    # Rank 0 plays motions 0 and 1, rank 1 motions 0 and 2, with different errors.
    pin_clips(term, [0, 1] if rank == 0 else [0, 2])
    mpjpe = torch.tensor([0.1, 0.2] if rank == 0 else [0.3, 0.5])
    term._metrics.accumulate(mpjpe=mpjpe, root_pos_err=mpjpe, root_quat_err=mpjpe)
    term._metrics.scatter_at_reset(torch.arange(2), torch.ones(2, dtype=torch.long))
    manager.update_motion_weight_curriculum()
    torch.save({"extras": manager.reset()}, f"{out_dir}/rank_{rank}.pt")


@pytest.mark.slow
def test_ranks_end_with_byte_identical_metric_summaries(tmp_path):
    root = write_library(tmp_path, [50] * 4)
    with ddp_spawn(_ddp_metrics_worker, str(root), world_size=2) as out_dir:
        r0, r1 = (torch.load(out_dir / f"rank_{r}.pt", weights_only=False)["extras"] for r in (0, 1))

    mimic = {k: v for k, v in r0.items() if k.startswith("Mimic/")}
    assert mimic and mimic == {k: v for k, v in r1.items() if k.startswith("Mimic/")}
    # Motions 0 (both ranks), 1 and 2 had episodes; the reduced means are 0.2, 0.2 and 0.5.
    assert mimic["Mimic/motions_with_data"] == 3.0
    assert mimic["Mimic/mpjpe_mean"] == pytest.approx(0.3, abs=1e-6)
