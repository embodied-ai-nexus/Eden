"""``eden.utils.profiling.term_profiler``: the one-shot per-term cost ranking, on a real env and a fake clock."""

from __future__ import annotations

import json
import os

import pytest

from eden.utils.profiling import profile_terms


def test_ranks_the_per_step_terms_by_cost_and_normalizes_by_steps(profiled_env, clock):
    """Action and event terms are left out: their cost is in ``apply_actions`` / mode dispatch."""
    report = profile_terms(profiled_env, warmup=2, measure=10, sync=False)

    costs = {(row.manager, row.term): round(row.us_per_step) for row in report.rows}
    assert costs == {
        ("reward", "costly"): 2000,
        ("observation", "policy.costly"): 1000,
        ("termination", "costly"): 500,
        ("reward", "free"): 0,
        ("observation", "critic.dofs_pos"): 0,
    }
    assert [row.term for row in report.rows[:3]] == ["costly", "policy.costly", "costly"]
    assert report.step_us == pytest.approx(3500.0)  # the fake clock only moves inside the costly terms
    assert report.rows[0].pct_step == pytest.approx(2000 / 35)
    assert "compute" not in vars(profiled_env.reward_manager.get_term("costly"))  # every wrap restored


def test_a_term_computed_twice_a_step_is_charged_both_calls(profiled_env, clock):
    """With final observations on, the observation is computed at phases 12 and 19."""
    profiled_env.env_options.record_final_observations = True
    try:
        report = profile_terms(profiled_env, warmup=1, measure=10, sync=False)
    finally:
        profiled_env.env_options.record_final_observations = False

    (row,) = [row for row in report.rows if row.term == "policy.costly"]
    assert (row.us_per_step, row.calls_per_step) == (pytest.approx(2000.0), pytest.approx(2.0))


def test_warmup_steps_run_but_are_not_attributed(profiled_env, clock):
    before = profiled_env.common_step_counter

    report = profile_terms(profiled_env, warmup=5, measure=20, sync=False)

    assert profiled_env.common_step_counter - before == 25
    assert report.num_steps == 20
    assert all(row.calls_per_step == pytest.approx(1.0) for row in report.rows)


def test_the_report_serializes_and_renders(profiled_env, clock, tmp_path):
    report = profile_terms(profiled_env, warmup=1, measure=5, sync=False)
    path = tmp_path / "out" / "term_cost.json"

    report.to_json(str(path))

    data = json.loads(path.read_text())
    assert data["num_steps"] == 5 and data["terms"][0] == {
        "manager": "reward",
        "term": "costly",
        "us_per_step": 2000.0,
        "pct_step": pytest.approx(57.1),
        "calls_per_step": 1.0,
    }
    assert "hint" in data
    assert "reward:costly" in report.render()


@pytest.mark.requires_genesis
def test_run_smoke_profile_builds_a_task_ranks_its_terms_and_writes_json(tmp_path):
    from eden.utils.profiling import run_smoke_profile

    report, path = run_smoke_profile("reacher", num_envs=2, warmup=2, measure=5, out_dir=str(tmp_path))

    assert path == str(tmp_path / "term_cost.json") and os.path.exists(path)
    assert report.num_steps == 5 and report.rows
    assert all(0.0 <= row.pct_step <= 100.0 for row in report.rows)  # each term is a subset of the step
    assert json.loads(open(path).read())["terms"][0]["term"]
