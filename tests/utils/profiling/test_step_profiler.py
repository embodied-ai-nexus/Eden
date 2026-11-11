"""``eden.utils.profiling.step_profiler``: ``RLStepProfiler`` attached to a real env's step loop."""

from __future__ import annotations

import pytest
import torch

from eden.utils.profiling import RLStepProfiler


def _run(env, profiler, steps: int) -> None:
    for _ in range(steps):
        with profiler.step():
            env.step(torch.zeros(env.num_envs, 1))


def _calls_per_step(profiler) -> dict[tuple[str, str], float]:
    return {(t["manager"], t["label"]): t["calls_per_step"] for t in profiler.as_dict()["terms"]}


def test_attaches_to_every_manager_shape_and_counts_calls_per_step(profiled_env):
    """Observation terms by group, events by mode, the flat managers by name; the action manager's
    ``apply_actions`` runs once per decimation sub-step, where an action term's real cost lives.
    """
    with RLStepProfiler(profiled_env, sync=False) as profiler:
        _run(profiled_env, profiler, steps=5)

    calls = _calls_per_step(profiler)
    assert calls[("observation", "policy.costly")] == calls[("observation", "critic.dofs_pos")] == 1
    assert calls[("reward", "costly")] == calls[("termination", "costly")] == 1
    assert calls[("action", "pd")] == 1
    assert profiler.as_dict()["managers"]["action"]["<action.apply_actions>"]["calls_per_step"] == 2
    assert ("event", "interval.noop") in calls
    assert profiler.num_steps == 5


def test_reset_clears_the_counts_but_stays_attached(profiled_env):
    with RLStepProfiler(profiled_env, sync=False) as profiler:
        _run(profiled_env, profiler, steps=3)
        profiler.reset()
        assert profiler.num_steps == 0 and set(_calls_per_step(profiler).values()) == {0}
        _run(profiled_env, profiler, steps=2)

    assert profiler.num_steps == 2 and _calls_per_step(profiler)[("reward", "costly")] == 1


def test_detach_restores_every_method(profiled_env):
    profiler = RLStepProfiler(profiled_env, sync=False).attach()
    _run(profiled_env, profiler, steps=2)
    profiler.detach()
    before = _calls_per_step(profiler)

    profiled_env.step(torch.zeros(profiled_env.num_envs, 1))

    assert _calls_per_step(profiler) == before
    assert not {"compute", "reset", "post_reset"} & set(vars(profiled_env.reward_manager))
    assert "compute" not in vars(profiled_env.reward_manager.get_term("costly"))
    assert "step" not in vars(profiled_env.scene)
    assert not {"_reset_idx", "_dispatch_post_reset"} & set(vars(profiled_env))


def test_the_report_rolls_terms_up_per_manager_and_leaves_the_rest_to_physics(profiled_env, tmp_path):
    with RLStepProfiler(profiled_env, sync=False) as profiler:
        _run(profiled_env, profiler, steps=4)

    summary = profiler.as_dict()
    assert (summary["num_steps"], summary["sync"]) == (4, False)
    assert {"reward", "observation", "action"} <= set(summary["managers"])
    report = profiler.report(top=3)
    assert "Per-Term Timing" in report and "Per-Manager Rollup" in report
    assert "physics" in report and "reset" in report and "unattributed" in report


def test_times_the_step_loop_work_no_manager_owns(profiled_env):
    """``scene.step`` once per decimation sub-step, the reset bookkeeping, and the recorder's step hooks."""
    with RLStepProfiler(profiled_env, sync=False) as profiler:
        _run(profiled_env, profiler, steps=3)

    summary = profiler.as_dict()
    assert summary["loop"]["physics"]["scene.step"]["calls_per_step"] == 2
    reset = summary["loop"]["reset"]
    assert reset["_reset_idx"]["calls_per_step"] == reset["_dispatch_post_reset"]["calls_per_step"] == 1
    assert summary["managers"]["observation"]["<observation.post_physics_refresh>"]["calls_per_step"] == 1
    recorder = summary["managers"]["recorder"]
    assert recorder["<recorder._on_step_started>"]["calls_per_step"] == 1
    assert recorder["<recorder._on_step_finished>"]["calls_per_step"] == 1


def _rollup_ms(summary) -> dict[str, float]:
    return {row: r["ms_per_step"] for row, r in summary["rollup"].items()}


def test_the_rollup_partitions_the_step_and_charges_nested_calls_once(profiled_env, clock, monkeypatch):
    """A reset runs the RESET batch and every manager's ``reset`` inside ``_reset_idx``: each is charged to its
    manager, once, and the reset row keeps only its own time, so the rows add up to the step and no more.
    """
    from eden.utils.profiling import step_profiler

    monkeypatch.setattr(step_profiler, "perf_counter", clock.read)
    reward_reset = profiled_env.reward_manager.reset

    def costly_reward_reset(*args, **kwargs):
        clock.advance_us(250)
        return reward_reset(*args, **kwargs)

    monkeypatch.setattr(profiled_env.reward_manager, "reset", costly_reward_reset)
    with RLStepProfiler(profiled_env, sync=False) as profiler:
        with profiler.step():
            profiled_env.step(torch.zeros(profiled_env.num_envs, 1))
            profiled_env.reset()

    summary = profiler.as_dict()
    reset = summary["loop"]["reset"]["_reset_idx"]
    assert reset["total_ms_per_step"] == pytest.approx(0.95) and reset["self_ms_per_step"] == pytest.approx(0.0)
    rollup = summary["rollup"]
    assert (rollup["reward"]["terms_ms_per_step"], rollup["reward"]["overhead_ms_per_step"]) == pytest.approx(
        (2.0, 0.25)
    )
    assert _rollup_ms(summary) == pytest.approx(
        # obs 1.0 (policy, at the step and again at the reset) x2, RESET batch 0.7, reward 2.0 + its reset 0.25
        {"observation": 2.0, "event": 0.7, "reward": 2.25, "termination": 0.5, "action": 0.0, "recorder": 0.0}
        | {"physics": 0.0, "reset": 0.0, "unattributed": 0.0}
    )
    assert sum(_rollup_ms(summary).values()) == pytest.approx(summary["step_ms"]) == pytest.approx(5.45)
    assert summary["outside_step_ms"] == 0.0


def test_calls_outside_a_timed_step_reach_the_term_table_but_not_the_rollup(profiled_env, clock, monkeypatch):
    """An ``env.reset()`` between timed steps would otherwise inflate the rows past the step they partition."""
    from eden.utils.profiling import step_profiler

    monkeypatch.setattr(step_profiler, "perf_counter", clock.read)
    with RLStepProfiler(profiled_env, sync=False) as profiler:
        profiled_env.reset()
        _run(profiled_env, profiler, steps=1)

    summary = profiler.as_dict()
    batch = next(t for t in summary["terms"] if t["label"] == "reset.seed_a+seed_b")
    assert (batch["calls_per_step"], batch["self_ms_per_step"]) == (1, 0.0)
    assert summary["outside_step_ms"] == pytest.approx(1.7)  # RESET batch 0.7 + the reset's observation 1.0
    assert sum(_rollup_ms(summary).values()) == pytest.approx(summary["step_ms"]) == pytest.approx(3.5)
    assert _rollup_ms(summary)["event"] == 0.0 and _rollup_ms(summary)["unattributed"] == pytest.approx(0.0)
    report = profiler.report()
    assert "1.700 ms of instrumented calls ran outside profiler.step()" in report
    assert "incl. calls outside step()" in report  # on the per-term table, whose per-step columns count them


def test_the_motion_window_fill_is_charged_to_motion_not_to_the_reset_row(tmp_path, clock, monkeypatch):
    """``MotionManager.reset`` does not refill the window: ``fill_data`` does, inside ``_reset_idx``."""
    from eden.utils.profiling import step_profiler
    from tests.support.motion import build_mimic_env, write_library

    env = build_mimic_env(write_library(tmp_path / "clips", [20]), num_envs=2)
    monkeypatch.setattr(step_profiler, "perf_counter", clock.read)
    fill_data = env.motion_manager.fill_data

    def costly_fill_data(*args, **kwargs):
        clock.advance_us(400)
        return fill_data(*args, **kwargs)

    monkeypatch.setattr(env.motion_manager, "fill_data", costly_fill_data)
    with RLStepProfiler(env, sync=False) as profiler:
        with profiler.step():
            env.reset()

    summary = profiler.as_dict()
    assert summary["managers"]["motion"]["<motion.fill_data>"]["calls_per_step"] == 1
    assert _rollup_ms(summary)["motion"] == pytest.approx(0.4) and _rollup_ms(summary)["reset"] == 0.0
    # Its terms run inside fill_data, never through a timed compute: no terms/overhead split to show.
    assert summary["rollup"]["motion"]["terms_ms_per_step"] is None
    assert sum(_rollup_ms(summary).values()) == pytest.approx(summary["step_ms"]) == pytest.approx(0.4)


def test_a_batch_of_reset_terms_is_timed_as_one_row(profiled_env, clock):
    """Its terms' own ``compute`` never runs on a reset, so a row per term would read as free."""
    with RLStepProfiler(profiled_env, sync=False) as profiler:
        profiled_env.reset()

    rows = {(t["manager"], t["label"]): t for t in profiler.as_dict()["terms"]}
    batch = rows[("event", "reset.seed_a+seed_b")]
    assert batch["mean_ms"] == pytest.approx(0.7)
    assert ("event", "reset.seed_a") not in rows and ("event", "reset.seed_b") not in rows


def test_step_windows_cannot_nest(profiled_env):
    """An inner window would close the outer one early, and its later calls would read as outside a step."""
    with RLStepProfiler(profiled_env, sync=False) as profiler, profiler.step():
        with pytest.raises(RuntimeError, match="already open: step windows cannot nest"):
            profiler.step().__enter__()


def test_a_missing_step_loop_hook_raises_at_attach_and_patches_nothing(profiled_env, monkeypatch):
    """Skipping it would move its time into ``unattributed`` without a word."""
    monkeypatch.setattr(profiled_env.reward_manager, "post_reset", None)

    with pytest.raises(AttributeError, match=r"RewardManager \(env.reward_manager\) has no method 'post_reset'"):
        RLStepProfiler(profiled_env, sync=False).attach()

    assert "compute" not in vars(profiled_env.reward_manager.get_term("costly"))
    assert "step" not in vars(profiled_env.scene)


def test_a_call_that_raises_is_recorded_and_not_charged_to_its_caller(profiled_env, clock, monkeypatch):
    """Its wall time up to the raise is its own; the reward manager around it keeps only its own time."""
    from eden.utils.profiling import step_profiler

    monkeypatch.setattr(step_profiler, "perf_counter", clock.read)

    def failing_compute():
        clock.advance_us(300)
        raise ValueError("boom")

    monkeypatch.setattr(profiled_env.reward_manager.get_term("costly"), "compute", failing_compute)
    with RLStepProfiler(profiled_env, sync=False) as profiler:
        with pytest.raises(ValueError, match="boom"), profiler.step():
            profiled_env.step(torch.zeros(profiled_env.num_envs, 1))

    summary = profiler.as_dict()
    term = next(t for t in summary["terms"] if (t["manager"], t["label"]) == ("reward", "costly"))
    assert (term["calls_per_step"], term["mean_ms"]) == (1, pytest.approx(0.3))
    assert summary["rollup"]["reward"]["overhead_ms_per_step"] == pytest.approx(0.0)
    # termination 0.5, then the reward raises 0.3 ms in, before the observation compute is reached
    assert sum(_rollup_ms(summary).values()) == pytest.approx(summary["step_ms"]) == pytest.approx(0.8)
