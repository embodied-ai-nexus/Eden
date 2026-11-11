"""Tests for ``eden.managers.motion.outcome_curriculum``: per-motion hard-motion mining.

Outcomes go into a real term's ``MotionOutcomeCurriculum`` through ``record_motion_outcomes`` (or the
manager's reset path), and a drain through ``MotionManager.update_motion_weight_curriculum`` rewrites
the term's per-motion sampling weights.
"""

from __future__ import annotations

import math

import pytest
import torch

from tests.support.motion import build_mimic_env, make_manager, pin_clips, term_options, write_library
from tests.support.ddp import ddp_spawn


class _Wandb:
    """The one ``wandb`` attribute the histogram path touches, recording what it was handed."""

    class Histogram:
        def __init__(self, sequence):
            self.sequence = torch.as_tensor(sequence).tolist()


def _manager(env, tmp_path, num_motions, *, frames=50, **term_kwargs):
    root = write_library(tmp_path / f"lib_{num_motions}", [frames] * num_motions)
    term_kwargs = {"adaptive_motion_weights": True, "log_per_motion_metrics": True} | term_kwargs
    return make_manager(env, terms={"default": term_options(root, **term_kwargs)})


def _term(manager):
    return manager.get_term("default")


def _record(manager, success, failure) -> None:
    """Record ``success[m]`` completed and ``failure[m]`` failed episodes of motion ``m``."""
    term = _term(manager)
    num_motions = len(success)
    pin_clips(term, [m % num_motions for m in range(term.num_assigned_envs)])  # env m plays motion m
    success_ids = torch.repeat_interleave(torch.arange(num_motions), torch.as_tensor(success))
    failure_ids = torch.repeat_interleave(torch.arange(num_motions), torch.as_tensor(failure))
    # Played far past the clip end (a success) / not at all (a failure).
    term._curriculum.record_motion_outcomes(
        success_ids, failure_ids, torch.full((len(success_ids),), 1e6), torch.zeros(len(failure_ids)), 0.02
    )


def _drain(manager, success, failure) -> None:
    _record(manager, success, failure)
    manager.update_motion_weight_curriculum()


def _weights(manager) -> torch.Tensor:
    # The per-motion weight vector has no public reader (get_current_motion_weights is per env).
    return _term(manager)._motion_weights.detach().cpu().clone()


def _histograms(manager) -> dict[str, list[float]]:
    return {key: hist.sequence for key, hist in manager.collect_per_motion_distributions(_Wandb).items()}


def _counts(manager) -> tuple[list[float], list[float]]:
    """Drain and return the per-motion (success, failure) counts it saw."""
    manager.update_motion_weight_curriculum()
    hist = _histograms(manager)
    # A drain with no episodes at all emits no outcome histograms.
    none = [0.0] * len(hist["MotionPerMotion/Weights"])
    return hist.get("MotionPerMotion/Outcome/success_count", none), hist.get(
        "MotionPerMotion/Outcome/failure_count", none
    )


# ---------------------------------------------------------------------------
# Recording outcomes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("as_mask", [False, True], ids=["ids", "mask"])
@pytest.mark.parametrize(
    "selection, success_envs, failure_envs, expected",
    [
        ([0, 1, 2, 1], [0, 2], [1], ([1, 0, 1], [0, 1, 0])),
        ([0, 0, 1, 1], [0, 1], [2, 3], ([2, 0, 0], [0, 2, 0])),  # envs on one motion both count
    ],
)
def test_outcomes_count_per_motion(mimic_env, tmp_path, selection, success_envs, failure_envs, expected, as_mask):
    manager = _manager(mimic_env, tmp_path, 3)
    pin_clips(_term(manager), selection)

    def form(ids):
        if not as_mask:
            return torch.tensor(ids, dtype=torch.long)
        mask = torch.zeros(4, dtype=torch.bool)
        mask[ids] = True
        return mask

    _term(manager)._curriculum.record_motion_outcomes(
        form(success_envs),
        form(failure_envs),
        torch.full((len(success_envs),), 1e6),
        torch.zeros(len(failure_envs)),
        0.02,
    )

    assert _counts(manager) == expected


def test_outcomes_land_on_global_motion_ids_after_a_rank_subset(mimic_env, tmp_path):
    manager = _manager(mimic_env, tmp_path, 8)
    term = _term(manager)
    term.enable_trajectories(torch.tensor([1, 3, 5, 7]))
    term.set_envs_idx_assignments(slice(0, 4))
    pin_clips(term, [0, 1, 2, 3])  # local ids -> globals 1, 3, 5, 7

    term._curriculum.record_motion_outcomes(
        torch.tensor([0, 2]), torch.tensor([1, 3]), torch.full((2,), 1e6), torch.zeros(2), 0.02
    )

    assert _counts(manager) == ([0, 1, 0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0, 1])


@pytest.mark.parametrize(
    "starts, cohort, timestamps, expected",
    [
        # A reset that is neither a termination nor the clip's end says nothing and is dropped.
        ([0.0, 0.0], "success", [0.4, 1.95], ([0, 1], [0, 0])),
        ([0.0, 0.0], "failure", [0.2, 1.5], ([0, 0], [1, 1])),
        # A termination that coincides with the clip's end is a success (ProtoMotions' done_clip).
        ([0.0, 0.0], "failure", [0.95, 0.5], ([1, 0], [0, 1])),
        # Played time is start + episode time.
        ([0.0, 1.5], "success", [0.0, 0.5], ([0, 1], [0, 0])),
    ],
    ids=["timeout-mid-clip", "terminated-mid-clip", "terminated-at-end", "start-offset"],
)
def test_success_means_the_clip_played_to_its_end(mimic_env, tmp_path, starts, cohort, timestamps, expected):
    """``start + played + env_dt >= clip length`` completes a clip; clips are 1.0 s and 2.0 s."""
    root = write_library(tmp_path, [50, 100])
    manager = make_manager(
        mimic_env,
        terms={"default": term_options(root, adaptive_motion_weights=True, log_per_motion_metrics=True)},
    )
    term = _term(manager)
    pin_clips(term, [0, 1, 0, 1], start_s=starts + starts)
    envs, ts, none = torch.tensor([0, 1]), torch.tensor(timestamps), torch.tensor([], dtype=torch.long)

    if cohort == "success":
        term._curriculum.record_motion_outcomes(envs, none, ts, torch.zeros(0), 0.05)
    else:
        term._curriculum.record_motion_outcomes(none, envs, torch.zeros(0), ts, 0.05)

    assert _counts(manager) == expected


# ---------------------------------------------------------------------------
# Drain cadence
# ---------------------------------------------------------------------------


def test_every_drain_empties_the_counts(mimic_env, tmp_path):
    manager = _manager(mimic_env, tmp_path, 2)
    _drain(manager, [3, 0], [0, 2])

    assert _counts(manager) == ([0, 0], [0, 0])


def test_update_every_n_iters_accumulates_until_the_nth_call(mimic_env, tmp_path):
    manager = _manager(mimic_env, tmp_path, 2, motion_weight_update_every_n_iters=3)
    before = _weights(manager)

    _drain(manager, [1, 0], [0, 0])
    _drain(manager, [2, 1], [0, 0])
    assert torch.equal(_weights(manager), before)
    assert _term(manager)._curriculum.last_curriculum_summary == {}

    _drain(manager, [0, 0], [0, 3])
    assert _histograms(manager)["MotionPerMotion/Outcome/success_count"] == [3, 1]
    assert not torch.equal(_weights(manager), before)


def test_the_all_reduce_hook_sees_both_count_tensors_and_may_rewrite_them(mimic_env, tmp_path):
    manager = _manager(mimic_env, tmp_path, 2)
    _record(manager, [1, 0], [0, 1])
    seen = []

    def all_reduce(t):
        seen.append(t.tolist())
        t.mul_(2)  # as if two ranks had summed

    _term(manager)._update_motion_weight_curriculum(all_reduce_fn=all_reduce)

    assert seen == [[1, 0], [0, 1]]
    assert _term(manager)._curriculum.last_curriculum_summary["MotionOutcome/episodes_total"] == 4.0


# ---------------------------------------------------------------------------
# Compound update: log-space multiplicative weights
# ---------------------------------------------------------------------------


def _compound(env, tmp_path, n, **kwargs):
    return _manager(env, tmp_path, n, motion_weight_update_mode="compound", **kwargs)


def test_compound_without_outcomes_only_renormalises(mimic_env, tmp_path):
    manager = _compound(mimic_env, tmp_path, 3, motion_weights=[1.0, 2.0, 1.0])
    _drain(manager, [0, 0, 0], [0, 0, 0])

    torch.testing.assert_close(_weights(manager), torch.tensor([0.25, 0.5, 0.25]))


@pytest.mark.parametrize(
    "discount, amplify, success, failure, rises, falls",
    [(0.5, 1.0, [10, 0, 0], [0, 0, 0], 1, 0), (1.0, 2.0, [0, 0, 0], [0, 0, 4], 2, 0)],
    ids=["success-decays", "failure-amplifies"],
)
def test_compound_moves_weight_from_easy_to_hard_motions(
    mimic_env, tmp_path, discount, amplify, success, failure, rises, falls
):
    manager = _compound(
        mimic_env, tmp_path, 3, motion_weight_success_discount=discount, motion_weight_failure_amplify=amplify
    )
    _drain(manager, success, failure)

    weights = _weights(manager)
    assert weights[rises] > weights[falls]
    assert weights.sum().item() == pytest.approx(1.0)


def test_compound_floor_keeps_every_motion_sampleable(mimic_env, tmp_path):
    decayed = _compound(mimic_env, tmp_path, 2, motion_weight_success_discount=1e-9)
    _drain(decayed, [10, 0], [0, 0])
    assert (_weights(decayed) > 0).all()

    # Every weight flushed to the floor renormalises to uniform, not to an undefined distribution.
    flushed = _compound(mimic_env, tmp_path / "flushed", 4, motion_weight_success_discount=1e-30)
    _drain(flushed, [10, 10, 10, 10], [0, 0, 0, 0])
    torch.testing.assert_close(_weights(flushed), torch.full((4,), 0.25), atol=1e-6, rtol=0)


def test_compound_concentrates_on_hard_motions_over_many_drains(mimic_env, tmp_path):
    manager = _compound(
        mimic_env,
        tmp_path,
        4,
        motion_weight_success_discount=0.9,
        motion_weight_failure_amplify=1.1,
        motion_weight_floor=1e-4,
    )
    for _ in range(100):
        _drain(manager, [1, 1, 0, 0], [0, 0, 1, 1])

    weights = _weights(manager)
    assert weights[2:].sum() > 5 * weights[:2].sum()


def test_compound_stays_finite_past_float32_overflow_of_the_factor(mimic_env, tmp_path):
    """``2.0 ** 256`` is inf in float32; the old linear-space update then produced NaN weights."""
    manager = _compound(mimic_env, tmp_path, 3, motion_weight_failure_amplify=2.0)
    _drain(manager, [0, 0, 0], [0, 0, 256])

    weights = _weights(manager)
    assert torch.isfinite(weights).all() and weights[2] > weights[0]
    assert weights.sum().item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Snapshot update: weights proportional to this drain's failure rates
# ---------------------------------------------------------------------------


def test_snapshot_weights_are_the_clamped_failure_rates(mimic_env, tmp_path):
    # Rates 0, 0.5, 1.0, 0.25; mean 0.4375 * 200 never clamps.
    manager = _manager(mimic_env, tmp_path, 4)
    _drain(manager, [10, 5, 0, 3], [0, 5, 10, 1])

    torch.testing.assert_close(_weights(manager), torch.tensor([0.0, 0.5, 1.0, 0.25]) / 1.75)


def test_snapshot_clamps_an_outlier_rate_at_a_multiple_of_the_mean(mimic_env, tmp_path):
    # Rates 0.1, 0.1, 0.1, 1.0; mean 0.325 * 2 = 0.65 caps the last.
    manager = _manager(mimic_env, tmp_path, 4, motion_weight_failure_rate_max_over_mean=2.0)
    _drain(manager, [9, 9, 9, 0], [1, 1, 1, 10])

    torch.testing.assert_close(_weights(manager), torch.tensor([0.1, 0.1, 0.1, 0.65]) / 0.95)


def test_snapshot_is_stateless(mimic_env, tmp_path):
    manager = _manager(mimic_env, tmp_path, 4)
    _drain(manager, [1, 1, 1, 1], [0, 1, 2, 3])
    first = _weights(manager)
    _drain(manager, [1, 1, 1, 1], [0, 1, 2, 3])

    torch.testing.assert_close(_weights(manager), first)


@pytest.mark.parametrize(
    "success, failure, expected",
    [([0, 0, 0, 0], [0, 0, 0, 0], [0.4, 0.3, 0.2, 0.1]), ([10, 10, 10, 10], [0, 0, 0, 0], [0.25] * 4)],
    ids=["no-episodes-keeps-weights", "no-failures-goes-uniform"],
)
def test_snapshot_without_a_failure_signal(mimic_env, tmp_path, success, failure, expected):
    manager = _manager(mimic_env, tmp_path, 4, motion_weights=[0.4, 0.3, 0.2, 0.1])
    _drain(manager, success, failure)

    torch.testing.assert_close(_weights(manager), torch.tensor(expected))


def test_snapshot_of_a_single_motion_library_is_weight_one(mimic_env, tmp_path):
    manager = _manager(mimic_env, tmp_path, 1, motion_weight_uniform_mix=0.1)
    _drain(manager, [5], [3])

    assert _weights(manager).tolist() == pytest.approx([1.0])


def test_switching_snapshot_to_compound_mid_run_keeps_a_distribution(mimic_env, tmp_path):
    manager = _manager(mimic_env, tmp_path, 3, motion_weight_uniform_mix=0.1, motion_weight_failure_amplify=1.5)
    _drain(manager, [1, 1, 0], [0, 1, 2])
    snapshot = _weights(manager)

    _term(manager).motion_weight_update_mode = "compound"
    _drain(manager, [0, 0, 0], [0, 0, 5])

    weights = _weights(manager)
    assert torch.isfinite(weights).all() and weights.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert weights[2] > snapshot[2]


# ---------------------------------------------------------------------------
# Uniform mix, applied after either update
# ---------------------------------------------------------------------------


def test_uniform_mix_bounds_a_compound_run_away(mimic_env, tmp_path):
    manager = _compound(mimic_env, tmp_path, 4, motion_weight_failure_amplify=2.0, motion_weight_uniform_mix=0.5)
    for _ in range(50):
        _drain(manager, [0, 0, 0, 0], [0, 0, 0, 100])

    weights = _weights(manager)
    assert weights.max() <= 0.5 + 0.5 / 4 + 1e-6 and weights.min() >= 0.5 / 4 - 1e-6


def test_uniform_mix_blends_the_snapshot_with_uniform(mimic_env, tmp_path):
    manager = _manager(mimic_env, tmp_path, 4, motion_weight_uniform_mix=0.1)
    _drain(manager, [10, 10, 10, 0], [0, 0, 0, 10])

    torch.testing.assert_close(_weights(manager), torch.tensor([0.025, 0.025, 0.025, 0.925]))


@pytest.mark.parametrize("mode", ["compound", "snapshot"])
def test_zero_uniform_mix_lets_one_motion_take_the_mass(mimic_env, tmp_path, mode):
    manager = _manager(mimic_env, tmp_path, 3, motion_weight_update_mode=mode, motion_weights=[0.5, 0.3, 0.2])
    _drain(manager, [0, 0, 0], [0, 0, 100])

    assert _weights(manager)[2] > 0.9


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"motion_weight_uniform_mix": 1.5}, "motion_weight_uniform_mix"),
        ({"motion_weight_uniform_mix": -0.1}, "motion_weight_uniform_mix"),
        ({"motion_weight_failure_rate_max_over_mean": 0.0}, "motion_weight_failure_rate_max_over_mean"),
        ({"motion_weight_failure_rate_max_over_mean": -1.0}, "motion_weight_failure_rate_max_over_mean"),
    ],
)
def test_bad_curriculum_knobs_fail_at_construction(mimic_env, tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _manager(mimic_env, tmp_path, 3, **kwargs)


def test_the_max_over_mean_clamp_is_snapshot_only(mimic_env, tmp_path):
    _compound(mimic_env, tmp_path, 3, motion_weight_failure_rate_max_over_mean=0.0)


@pytest.mark.parametrize(
    "attr, value, match",
    [
        ("motion_weight_uniform_mix", -0.5, "motion_weight_uniform_mix"),
        ("motion_weight_update_mode", "totally_made_up", "Unknown motion_weight_update_mode"),
    ],
)
def test_knobs_changed_after_construction_are_still_checked_at_the_drain(mimic_env, tmp_path, attr, value, match):
    manager = _manager(mimic_env, tmp_path, 3, motion_weight_uniform_mix=0.1)
    setattr(_term(manager), attr, value)

    with pytest.raises(ValueError, match=match):
        _drain(manager, [0, 0, 0], [0, 0, 5])


# ---------------------------------------------------------------------------
# Weight and outcome summaries surfaced through MotionManager.reset()
# ---------------------------------------------------------------------------


def test_weights_summary_of_a_uniform_library_is_max_entropy(mimic_env, tmp_path):
    summary = _manager(mimic_env, tmp_path, 8).reset()

    assert summary["MotionWeights/entropy"] == pytest.approx(math.log(8), abs=1e-6)
    assert summary["MotionWeights/entropy_normalized"] == pytest.approx(1.0, abs=1e-6)
    assert summary["MotionWeights/min"] == pytest.approx(1 / 8)
    assert summary["MotionWeights/max"] == pytest.approx(1 / 8)
    # The floor-clip fraction measures raw user weights until the curriculum has run once: noise.
    assert "MotionWeights/floor_clip_ratio" not in summary


def test_weights_summary_after_a_drain_reports_the_floor_clipped_share(mimic_env, tmp_path):
    floor = 1e-3
    manager = _compound(mimic_env, tmp_path, 4, motion_weights=[1 - 3 * floor, floor, floor, floor])
    _drain(manager, [0, 0, 0, 0], [0, 0, 0, 0])

    summary = manager.reset()
    assert 0.0 < summary["MotionWeights/entropy"] < 0.1
    assert summary["MotionWeights/min"] == pytest.approx(floor)
    assert summary["MotionWeights/floor_clip_ratio"] == pytest.approx(0.75)


@pytest.mark.parametrize(
    "raw, normalised", [([100.0] * 4, [0.25] * 4), ([1.0, 2.0, 1.0, 4.0], [1 / 8, 2 / 8, 1 / 8, 4 / 8])]
)
def test_weights_summary_describes_the_sampling_distribution_not_the_raw_scale(mimic_env, tmp_path, raw, normalised):
    summary_raw = _manager(mimic_env, tmp_path / "raw", 4, motion_weights=raw).reset()
    summary_normalised = _manager(mimic_env, tmp_path / "normalised", 4, motion_weights=normalised).reset()

    assert summary_raw == pytest.approx(summary_normalised, abs=1e-6)


def test_normalised_entropy_of_one_motion_is_zero(mimic_env, tmp_path):
    summary = _manager(mimic_env, tmp_path, 1).reset()

    assert summary["MotionWeights/entropy_normalized"] == 0.0


def test_outcome_summary_rates_only_motions_that_had_episodes(mimic_env, tmp_path):
    manager = _manager(mimic_env, tmp_path, 4)
    assert "MotionOutcome/episodes_total" not in manager.reset()  # nothing until the first drain

    _drain(manager, [2, 4, 0, 0], [2, 0, 4, 0])  # rates 0.5, 1.0, 0.0; motion 3 has no episodes

    for _ in range(3):  # the drained summary persists across resets
        summary = manager.reset()
        assert summary["MotionOutcome/episodes_total"] == 12.0
        assert summary["MotionOutcome/motions_with_data"] == 3.0
        assert summary["MotionOutcome/success_rate_mean"] == pytest.approx(0.5)
        assert summary["MotionOutcome/success_rate_p50"] == pytest.approx(0.5)
        assert summary["MotionOutcome/success_rate_p10"] < 0.2
        assert "MotionWeights/entropy" in summary


def test_an_empty_drain_emits_only_the_episode_counts(mimic_env, tmp_path):
    manager = _manager(mimic_env, tmp_path, 3)
    _drain(manager, [5, 0, 2], [0, 3, 8])
    _drain(manager, [0, 0, 0], [0, 0, 0])

    assert _term(manager)._curriculum.last_curriculum_summary == {
        "MotionOutcome/episodes_total": 0.0,
        "MotionOutcome/motions_with_data": 0.0,
    }
    # The histograms follow, rather than re-emitting the previous drain's.
    assert not any(key.startswith("MotionPerMotion/Outcome/") for key in _histograms(manager))


def test_weights_summary_refreshes_when_the_curriculum_rewrites_the_weights(mimic_env, tmp_path):
    manager = _compound(mimic_env, tmp_path, 4, motion_weight_failure_amplify=2.0)
    before = manager.reset()

    _drain(manager, [0, 0, 0, 0], [0, 0, 0, 50])

    after = manager.reset()
    assert after["MotionWeights/entropy_normalized"] < before["MotionWeights/entropy_normalized"]
    assert after["MotionWeights/max"] > before["MotionWeights/max"]


def test_non_adaptive_terms_add_nothing_to_the_log(mimic_env, library):
    manager = make_manager(mimic_env, library)

    assert not manager.has_adaptive_motion_weights
    assert manager.reset() == {}


def test_several_terms_prefix_their_keys_with_the_term_name(mimic_env, tmp_path):
    root = write_library(tmp_path, [50, 50])
    manager = make_manager(
        mimic_env, terms={name: term_options(root, adaptive_motion_weights=True) for name in ("a", "b")}
    )

    extras = manager.reset()
    assert "a/MotionWeights/entropy" in extras and "b/MotionWeights/entropy" in extras
    assert "MotionWeights/entropy" not in extras


# ---------------------------------------------------------------------------
# Per-motion histograms and the argmax diagnostics
# ---------------------------------------------------------------------------


def test_argmax_motion_id_is_always_emitted(mimic_env, tmp_path):
    manager = _manager(mimic_env, tmp_path, 5, log_per_motion_metrics=False, motion_weights=[0.05, 0.05, 0.7, 0.1, 0.1])

    assert manager.reset()["MotionWeights/argmax_motion_id"] == 2.0
    assert manager.collect_per_motion_distributions(_Wandb) == {}


def test_per_motion_histograms_carry_the_weights_and_outcomes(mimic_env, tmp_path):
    manager = _manager(mimic_env, tmp_path, 4, motion_weights=[1.0, 1.0, 2.0, 4.0])
    assert _histograms(manager)["MotionPerMotion/Weights"] == pytest.approx([0.125, 0.125, 0.25, 0.5])

    _drain(manager, [5, 0, 2, 0], [0, 3, 8, 0])

    hist = _histograms(manager)
    assert hist["MotionPerMotion/Outcome/success_count"] == [5, 0, 2, 0]
    assert hist["MotionPerMotion/Outcome/total"] == [5, 3, 10, 0]  # unsampled motions are present, at zero
    assert hist["MotionPerMotion/Outcome/success_rate"] == pytest.approx([1.0, 0.0, 0.2, 0.0])
    assert hist["MotionPerMotion/Weights"] == pytest.approx(_weights(manager).tolist(), abs=1e-6)
    # Scalars stay aggregate: no per-id key leaks into the log.
    assert not any(key.startswith("MotionPerMotion/") for key in manager.reset())
    assert manager.reset()["MotionWeights/argmax_motion_id"] == 1.0  # failure rate 1.0


def test_argmax_filename_is_the_clips_basename_when_enabled(mimic_env, tmp_path):
    weights = [0.1, 0.7, 0.2]
    off = _manager(mimic_env, tmp_path / "off", 3, motion_weights=weights)
    on = _manager(mimic_env, tmp_path / "on", 3, motion_weights=weights, log_motion_id_to_filename=True)

    assert "MotionWeights/argmax_motion_filename" not in off.reset()
    assert on.reset()["MotionWeights/argmax_motion_filename"] == "clip_01.npz"


def test_argmax_filename_held_by_another_rank_is_not_local(mimic_env, tmp_path):
    manager = _manager(mimic_env, tmp_path, 4, motion_weights=[0.7, 0.1, 0.1, 0.1], log_motion_id_to_filename=True)
    term = _term(manager)
    term.enable_trajectories(torch.tensor([1, 3]))
    term.set_envs_idx_assignments(slice(0, 4))

    _drain(manager, [0, 0], [0, 0])  # a drain refreshes the cached weights summary

    assert manager.reset()["MotionWeights/argmax_motion_filename"] == "<not_local>"


# ---------------------------------------------------------------------------
# The manager's reset path routes each resetting env to the right cohort
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def routing_library(tmp_path_factory):
    return write_library(tmp_path_factory.mktemp("routing"), [3, 100])  # 0.06 s and 2 s


@pytest.fixture(scope="module")
def routing_env(routing_library):
    """A mimic env whose robot terminates when its base drops below 0.3 m."""
    from eden import terminations
    from eden.options.managers.terminations import TerminationManagerOptions
    from eden.terms import DoneTerm

    low = DoneTerm.configure(
        func=terminations.root_height_below_minimum, params={"entity_name": "robot", "minimum_height": 0.3}
    )
    return build_mimic_env(routing_library, termination_options=TerminationManagerOptions(low=low))


def _routing_manager(env, root, *, episode_steps=5, low_envs=()):
    """Env e plays clip e % 2 (short, long) from t=0; ``low_envs`` terminate; clocks run 0.06 s."""
    manager = make_manager(
        env, terms={"default": term_options(root, adaptive_motion_weights=True, log_per_motion_metrics=True)}
    )
    pin_clips(_term(manager), [0, 1, 0, 1])
    for _ in range(3):
        manager.compute(env.dt)
    robot = env.entities["robot"]
    pos = torch.tensor([[0.0, 0.0, 0.8]]).repeat(4, 1)
    pos[list(low_envs), 2] = 0.1
    robot.set_pos(pos.to(env.device))
    env.termination_manager.compute()
    env.episode_length_buf[:] = episode_steps
    return manager


@pytest.mark.parametrize("as_mask", [False, True], ids=["ids", "mask"])
def test_terminated_envs_fail_and_finished_clips_succeed(routing_env, routing_library, as_mask):
    manager = _routing_manager(routing_env, routing_library, low_envs=(1, 3))
    envs = torch.ones(4, dtype=torch.bool) if as_mask else torch.arange(4)

    manager.reset(envs)

    # Envs 0 and 2 finished the short clip; 1 and 3 terminated 0.06 s into the long one.
    assert _counts(manager) == ([2, 0], [0, 2])


def test_timeouts_before_the_clip_ends_and_bootstrap_resets_count_nothing(routing_env, routing_library):
    timeout = _routing_manager(routing_env, routing_library)
    timeout.reset(torch.tensor([1, 3]))  # long clip, not terminated
    assert _counts(timeout) == ([0, 0], [0, 0])

    bootstrap = _routing_manager(routing_env, routing_library, episode_steps=0, low_envs=(0, 1, 2, 3))
    bootstrap.reset(torch.arange(4))
    assert _counts(bootstrap) == ([0, 0], [0, 0])


# ---------------------------------------------------------------------------
# DDP: ranks drain different counts and must end byte-identical
# ---------------------------------------------------------------------------


def _ddp_worker(rank, world_size, out_dir, root, update_mode, split, steps):
    import eden as en

    en.init(debug=True)
    import torch

    from eden.options.entities import BoxOptions
    from eden.options.envs import EnvOptions
    from eden.utils.configs import EdenConfig
    from tests.support.motion import make_manager, pin_clips, term_options
    from tests.support.envs import build_env

    # The curriculum reads nothing off the scene: the cheapest real env will do.
    env = build_env(EdenConfig(env_options=EnvOptions(num_envs=2), scene_options=en.scenes.Empty(robot=BoxOptions())))
    manager = make_manager(
        env,
        mp_split_method=split,
        terms={
            "default": term_options(
                root,
                adaptive_motion_weights=True,
                motion_weight_update_mode=update_mode,
                motion_weight_success_discount=0.5,
                motion_weight_failure_amplify=2.0,
                motion_weight_uniform_mix=0.1 if update_mode == "snapshot" else 0.0,
            )
        },
    )
    term = manager.get_term("default")
    # Each rank reports different outcomes for the motions it holds: local id i on env i.
    pin_clips(term, [0, 1])
    for _ in range(steps):
        success = torch.tensor([0, 0] if rank == 0 else [1, 0])
        failure = torch.tensor([rank, 1 + 3 * rank])
        term._curriculum.record_motion_outcomes(
            torch.repeat_interleave(torch.arange(2), success),
            torch.repeat_interleave(torch.arange(2), failure),
            torch.full((int(success.sum()),), 1e6),
            torch.zeros(int(failure.sum())),
            0.02,
        )
        manager.update_motion_weight_curriculum()
    torch.save(
        {
            "weights": term._motion_weights.clone(),
            "local_to_global": term._local_to_global.tolist(),
            "extras": manager.reset(),
        },
        f"{out_dir}/rank_{rank}.pt",
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "update_mode, split, steps, local_to_global",
    [
        ("compound", "None", 1, [[0, 1, 2, 3]] * 2),
        ("compound", "Even", 20, [[0, 2], [1, 3]]),
        ("snapshot", "Even", 20, [[0, 2], [1, 3]]),
    ],
    ids=["full-library", "sharded-compound", "sharded-snapshot"],
)
def test_ranks_end_with_byte_identical_weights_and_summaries(tmp_path, update_mode, split, steps, local_to_global):
    root = write_library(tmp_path, [50] * 4)
    with ddp_spawn(_ddp_worker, str(root), update_mode, split, steps, world_size=2) as out_dir:
        r0, r1 = (torch.load(out_dir / f"rank_{r}.pt", weights_only=False) for r in (0, 1))

    assert [r0["local_to_global"], r1["local_to_global"]] == local_to_global
    assert torch.equal(r0["weights"], r1["weights"])
    assert not torch.allclose(r0["weights"], torch.full((4,), 0.25))
    assert r0["extras"] == r1["extras"]
