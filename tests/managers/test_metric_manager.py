"""Tests for ``MetricManager``: ``compute_once`` shares an intermediate within one pass, and the batched
``reset()`` logs the per-term means, the success checks and ``Success``/``Progress``.

Metrics read the box height the test sets; episode lengths are the env's own buffer.
"""

from __future__ import annotations

import pytest
import torch

from eden.constants import MetricDirection, MetricMode
from eden.managers.metric_manager import METRIC_TERM_REGISTRY, MetricManager, MetricTerm
from eden.options.managers.metrics import MetricManagerOptions, MetricTermOptions

NUM_ENVS = 4
HEIGHTS = [0.05, 0.3, 0.55, 0.9]
EPISODE_LENGTHS = [0, 3, 5, 10]  # 0 exercises the clamp to 1


def _shared(env) -> torch.Tensor:
    """A fresh random draw per production, so sharing is visible as equality."""
    return env.metric_manager.compute_once(("metric_test_shared",), lambda: torch.rand(env.num_envs))


@METRIC_TERM_REGISTRY.register()
def shared_first(env) -> torch.Tensor:
    return _shared(env)


@METRIC_TERM_REGISTRY.register()
def shared_doubled(env) -> torch.Tensor:
    return 2.0 * _shared(env)


@METRIC_TERM_REGISTRY.register()
class SharedThenMaybeRaises(MetricTerm):
    """Reads the shared intermediate, then raises while ``fail`` is set."""

    def build(self) -> None:
        self.fail = False

    def compute(self, envs_idx=None) -> torch.Tensor:
        shared = _shared(self._env)
        if self.fail:
            raise RuntimeError("term failed mid-pass")
        return shared


@METRIC_TERM_REGISTRY.register()
def height_metric(env) -> torch.Tensor:
    return env.entities["box"].get_pos()[:, 2]


@METRIC_TERM_REGISTRY.register()
def height_complement(env) -> torch.Tensor:
    return 1.0 - env.entities["box"].get_pos()[:, 2]


@METRIC_TERM_REGISTRY.register()
def height_doubled(env) -> torch.Tensor:
    return 2.0 * env.entities["box"].get_pos()[:, 2]


@METRIC_TERM_REGISTRY.register()
def random_metric(env) -> torch.Tensor:
    return torch.rand(env.num_envs, device=env.device)


@METRIC_TERM_REGISTRY.register()
class ConstantMetric(MetricTerm):
    """``value`` in every env; records the ``envs_idx`` of its last reset."""

    value: float = 0.0

    def build(self) -> None:
        self.last_reset_envs_idx = None

    def compute(self, envs_idx=None) -> torch.Tensor:
        return torch.full((self.num_envs,), float(self.value), device=self.device)

    def reset(self, envs_idx=None) -> None:
        self.last_reset_envs_idx = envs_idx


@METRIC_TERM_REGISTRY.register()
class TenthsParity(MetricTerm):
    """Overrides ``is_success`` with a check the default threshold cannot express: even tenths succeed."""

    metric_mode: MetricMode = MetricMode.RESET

    def compute(self, envs_idx=None) -> torch.Tensor:
        return self._env.entities["box"].get_pos()[:, 2]

    def is_success(self, value, envs_idx=None) -> torch.Tensor:
        return (value * 10).long() % 2 == 0


@pytest.fixture(scope="module")
def sharing_env(make_box_env):
    """An env whose own metric manager holds the three ``compute_once`` terms, found as ``env.metric_manager``."""
    return make_box_env(
        metric_options=MetricManagerOptions(
            first=MetricTermOptions(name="shared_first"),
            doubled=MetricTermOptions(name="shared_doubled"),
            flaky=SharedThenMaybeRaises.configure(),
        )
    )


# ---------------------------------------------------------------------------
# compute_once
# ---------------------------------------------------------------------------


def test_terms_sharing_a_key_get_one_production_per_pass(sharing_env):
    manager = sharing_env.metric_manager

    values = {name: value.clone() for name, value in manager.compute().items()}
    torch.testing.assert_close(values["doubled"], 2.0 * values["first"])

    fresh = manager.compute()  # a new pass starts empty
    assert not torch.equal(fresh["first"], values["first"])
    torch.testing.assert_close(fresh["doubled"], 2.0 * fresh["first"])


def test_outside_a_pass_the_producer_runs_on_every_call(sharing_env):
    """A term's ``compute()`` called on its own must see fresh state, never the last pass's value."""
    manager = sharing_env.metric_manager
    manager.compute()

    assert manager.compute_once(("metric_test_shared",), lambda: "fresh") == "fresh"
    assert not torch.equal(shared_first(sharing_env), shared_first(sharing_env))


def test_a_term_that_raises_mid_pass_does_not_leave_the_memo_behind(sharing_env):
    manager = sharing_env.metric_manager
    flaky = manager.get_term("flaky")
    flaky.fail = True
    try:
        with pytest.raises(RuntimeError, match="term failed mid-pass"):
            manager.compute()
    finally:
        flaky.fail = False

    assert manager.compute_once(("metric_test_shared",), lambda: "fresh") == "fresh"


# ---------------------------------------------------------------------------
# reset(): the batched logs
# ---------------------------------------------------------------------------

#: Per env, what each term logs at reset after two computes: a cumulative term's per-step mean over
#: its episode (lengths clamped to 1), anything else its current value. Hand-computed from HEIGHTS.
LOGGED = {
    "cum_lib": [0.1, 0.2, 0.22, 0.18],  # 2h / L
    "cum_hib": [1.9, 1.4 / 3, 0.18, 0.02],  # 2(1 - h) / L
    "step_lib": [0.1, 0.6, 1.1, 1.8],  # 2h
    "interval": HEIGHTS,
    "shared_only": [0.95, 0.7, 0.45, 0.1],  # 1 - h
    "custom": HEIGHTS,
}
#: Success checks passed per env, of the five that count: cum_lib (<= 0.4) all; cum_hib (>= 0.3) and
#: step_lib (<= 1.0) envs 0-1; interval (>= 0.5, judged at compute) envs 2-3; custom (even tenths) env 0.
PASSED = [4, 3, 2, 2]


@pytest.fixture
def mixed(box_env, set_box_heights):
    set_box_heights(box_env, HEIGHTS)
    box_env.episode_length_buf[:] = torch.tensor(EPISODE_LENGTHS)
    reset, interval = MetricMode.RESET, MetricMode.INTERVAL
    lib, hib = MetricDirection.LIB, MetricDirection.HIB
    manager = MetricManager(
        env=box_env,
        options=MetricManagerOptions(
            cum_lib=MetricTermOptions(
                name="height_metric", is_cumulative=True, metric_mode=reset, direction=lib, success_threshold=0.4
            ),
            cum_hib=MetricTermOptions(
                name="height_complement", is_cumulative=True, metric_mode=reset, direction=hib, success_threshold=0.3
            ),
            step_lib=MetricTermOptions(name="height_doubled", metric_mode=reset, direction=lib, success_threshold=1.0),
            interval=MetricTermOptions(name="height_metric", metric_mode=interval, success_threshold=0.5),
            shared_only=MetricTermOptions(name="height_complement", metric_mode=reset, contributes_to_success=False),
            custom=TenthsParity.configure(),
        ),
    )
    yield manager
    box_env.episode_length_buf[:] = 0


@pytest.mark.parametrize(
    "envs_idx, rows",
    [
        (torch.tensor([True, False, True, True]), [0, 2, 3]),
        (torch.tensor([0, 2, 3]), [0, 2, 3]),
        (slice(1, 3), [1, 2]),
        (None, [0, 1, 2, 3]),
    ],
    ids=["mask", "index", "slice", "all"],
)
def test_reset_logs_each_terms_mean_and_the_success_rate_over_the_reset_envs(mixed, envs_idx, rows):
    mixed.compute()
    mixed.compute()
    before = {name: mixed.get_value(name).clone() for name in LOGGED}

    extras = mixed.reset(envs_idx)

    def mean(values):
        return sum(values[row] for row in rows) / len(rows)

    for name, logged in LOGGED.items():
        assert extras[f"Metrics/{name}"].item() == pytest.approx(mean(logged), rel=1e-5), name
    assert extras["Metrics/Success"].item() == 0.0  # no env passes all five
    assert extras["Metrics/Progress"].item() == pytest.approx(mean([p / 5 for p in PASSED]), rel=1e-5)
    kept = [row for row in range(NUM_ENVS) if row not in rows]
    for name in LOGGED:
        value = mixed.get_value(name)
        assert value[rows].tolist() == [0.0] * len(rows), name
        torch.testing.assert_close(value[kept], before[name][kept], msg=name)  # an env not reset keeps its sums


@pytest.mark.parametrize(
    "envs_idx, rows",
    [
        (torch.tensor([True, False, True, True]), [0, 2, 3]),
        (torch.tensor([0, 2, 3]), [0, 2, 3]),
        (slice(1, 3), [1, 2]),
    ],
    ids=["mask", "index", "slice"],
)
def test_last_episode_holds_the_rows_a_reset_logged_the_mean_of(mixed, envs_idx, rows):
    """An evaluation picking which episodes it counts reads these rows; an env that has not finished one reads zeros
    and unhealthy.
    """
    mixed.compute()
    mixed.compute()
    extras = mixed.reset(envs_idx)

    values, healthy = mixed.last_episode()
    kept = [row for row in range(NUM_ENVS) if row not in rows]
    for name, logged in LOGGED.items():
        value = values[f"Metrics/{name}"]
        assert value[rows].tolist() == pytest.approx([logged[row] for row in rows], rel=1e-5), name
        assert value[rows].mean().item() == pytest.approx(extras[f"Metrics/{name}"].item(), rel=1e-5), name
        assert value[kept].tolist() == [0.0] * len(kept), name
    assert values["Metrics/Progress"][rows].tolist() == pytest.approx([PASSED[row] / 5 for row in rows], rel=1e-5)
    assert values["Metrics/Success"].tolist() == [0.0] * NUM_ENVS
    assert healthy.tolist() == [row in rows for row in range(NUM_ENVS)]


def test_get_value_is_a_live_view_through_compute_and_reset(mixed):
    """``metric_reward`` / ``metric_observation`` hold the buffer once and read it every step."""
    held = mixed.get_value("cum_lib")

    mixed.compute()
    torch.testing.assert_close(held, torch.tensor(HEIGHTS))
    mixed.reset()
    torch.testing.assert_close(held, torch.zeros(NUM_ENVS))
    with pytest.raises(KeyError, match="Metric 'does_not_exist' not found"):
        mixed.get_value("does_not_exist")


def test_an_env_genesis_flags_as_corrupt_adds_nothing_to_its_metrics_and_is_left_out_of_the_logs(
    mixed, flag_solver_error
):
    """It read garbage this step and is reset in it: a cumulative metric must not take that step's value,
    and its zeroed values would pass ``step_lib`` and ``custom`` and log as perfect lower-is-better means.
    """
    mixed.compute()
    flag_solver_error(mixed._env, [3])
    mixed.compute()

    cumulative = mixed.get_value("cum_lib")
    torch.testing.assert_close(cumulative, torch.tensor([2 * h if i != 3 else h for i, h in enumerate(HEIGHTS)]))
    assert mixed.get_value("interval")[3].item() == 0.0
    extras = mixed.reset()

    healthy = [0, 1, 2]
    for name, logged in LOGGED.items():
        expected = sum(logged[row] for row in healthy) / len(healthy)
        assert extras[f"Metrics/{name}"].item() == pytest.approx(expected, rel=1e-5), name
    expected_progress = sum(PASSED[row] / 5 for row in healthy) / len(healthy)
    assert extras["Metrics/Progress"].item() == pytest.approx(expected_progress, rel=1e-5)
    assert mixed.last_episode()[1].tolist() == [True, True, True, False]


@pytest.fixture
def lib_interval(box_env, set_box_heights):
    """One lower-is-better INTERVAL metric that only env 0 (height 0.05) passes."""
    set_box_heights(box_env, HEIGHTS)
    return MetricManager(
        env=box_env,
        options=MetricManagerOptions(
            dist=MetricTermOptions(
                name="height_metric",
                metric_mode=MetricMode.INTERVAL,
                direction=MetricDirection.LIB,
                success_threshold=0.1,
            )
        ),
    )


def test_a_flagged_env_never_passes_a_lower_is_better_check_on_its_zeroed_value(lib_interval, flag_solver_error):
    """0 is the best value of a LIB metric: the step's success and ``on_metric_success`` must not read it."""
    flag_solver_error(lib_interval._env, [3])

    lib_interval.compute()

    assert lib_interval.get_value("dist")[3].item() == 0.0  # no NaN in the buffer...
    assert lib_interval.success_buf.tolist() == [True, False, False, False]  # ...and no success from it
    assert lib_interval.is_current_step_success("dist").tolist() == [True, False, False, False]


def test_a_reset_of_only_flagged_envs_logs_no_success(lib_interval, flag_solver_error):
    flag_solver_error(lib_interval._env, [2, 3])
    lib_interval.compute()

    extras = lib_interval.reset(torch.tensor([False, False, True, True]))

    assert (extras["Metrics/Success"].item(), extras["Metrics/dist"].item()) == (0.0, 0.0)


@pytest.mark.parametrize(
    ("fields", "value", "episode", "success"),
    [
        ({"direction": MetricDirection.HIB}, 1.0, 1, True),  # the threshold is inclusive
        ({"direction": MetricDirection.LIB}, 1.5, 1, False),
        ({"direction": MetricDirection.HIB, "is_cumulative": True}, 2.0, 2, True),  # judged per step: 2.0 / 2
    ],
    ids=["hib", "lib", "cumulative"],
)
def test_is_success_compares_the_threshold_in_the_metrics_direction(box_env, fields, value, episode, success):
    box_env.episode_length_buf[:] = episode
    try:
        term = ConstantMetric(env=box_env, options=ConstantMetric.configure(success_threshold=1.0, **fields))
        assert term.is_success(torch.full((NUM_ENVS,), value)).tolist() == [success] * NUM_ENVS
    finally:
        box_env.episode_length_buf[:] = 0


def test_reset_reaches_every_term_with_the_envs_it_was_given(box_env):
    manager = MetricManager(
        env=box_env, options=MetricManagerOptions(a=ConstantMetric.configure(), b=ConstantMetric.configure())
    )
    envs_idx = torch.tensor([0])

    manager.reset(envs_idx=envs_idx)

    assert all(term.last_reset_envs_idx is envs_idx for term in manager.terms.values())


def test_a_non_contributing_metric_stays_out_of_the_success_accounting(box_env):
    """A term that only publishes a shared value has no notion of succeeding; left in, it would pin the verdict."""
    manager = MetricManager(
        env=box_env,
        options=MetricManagerOptions(
            shared=ConstantMetric.configure(value=1.0, contributes_to_success=False),
            real=ConstantMetric.configure(value=1.0, success_threshold=1.0),
        ),
    )
    manager.compute()

    assert manager.success_buf.all()  # decided by the one contributing term
    torch.testing.assert_close(manager.get_value("shared"), torch.ones(NUM_ENVS))  # still published
    extras = manager.reset()
    torch.testing.assert_close(extras["Metrics/Success"], torch.tensor(1.0))
    assert "Metrics/shared" in extras


def test_with_no_contributing_metric_no_success_is_logged(box_env):
    """Logging 0.0 would read as a real 0% success rate; the task has no success criterion."""
    manager = MetricManager(
        env=box_env, options=MetricManagerOptions(shared=ConstantMetric.configure(contributes_to_success=False))
    )
    manager.compute()

    assert not manager.success_buf.any()
    extras = manager.reset()
    assert "Metrics/Success" not in extras and "Metrics/Progress" not in extras and "Metrics/shared" in extras
    with pytest.raises(ValueError, match="contributes_to_success=False"):
        manager.is_current_step_success("shared")  # gating on it fails loud, not silently never


def test_a_masked_reset_is_sync_free(cuda_stand_in_env, count_cuda_syncs):
    """No sync per term: the auto-reset hands ``reset()`` a bool mask every step."""
    manager = MetricManager(
        env=cuda_stand_in_env,
        options=MetricManagerOptions(
            reached=MetricTermOptions(name="random_metric", success_threshold=0.5),
            held=MetricTermOptions(name="random_metric", success_threshold=0.3, metric_mode=MetricMode.RESET),
            total=MetricTermOptions(name="random_metric", is_cumulative=True),
        ),
    )
    manager.compute()
    mask = torch.tensor([True, False, True, True], device="cuda")

    assert count_cuda_syncs(lambda: manager.reset(mask)) == 0
