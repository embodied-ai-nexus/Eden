"""``SuccessGuidedSampling`` in ``eden/managers/terms/curricula/success_guided.py``.

The term samples a configuration set owned by a reset event. Here that event is ``DrawResetConfigs``,
a minimal consumer that binds the sampler through the real seam and draws from it on every reset.
Episodes end through two real termination terms on the box's height: ``lifted`` (above 1 m, the
success term) and ``dropped`` (below 0.2 m). The box spawns at 0.5 m, so neither fires until a test
places it.
"""

from __future__ import annotations

import pytest
import torch

from eden.constants import EventMode
from eden.envs.base import env_class_for_config
from eden.managers.event_manager import EVENT_TERM_REGISTRY, EventTerm
from eden.managers.resets.sampling import resolve_reset_config_sampler
from eden.managers.termination_manager import TERMINATION_TERM_REGISTRY, TerminationManager
from eden.managers.terms.curricula.success_guided import SuccessGuidedSampling, success_guided_probabilities
from eden.options import BoxOptions, EnvOptions, SceneOptions
from eden.options.managers.curricula import CurriculumManagerOptions
from eden.options.managers.events import EventManagerOptions
from eden.options.managers.terminations import TerminationManagerOptions
from eden.terms import DoneTerm
from eden.utils.configs import EdenRLConfig
from eden.utils.misc import sanitize_envs_idx

pytestmark = pytest.mark.requires_genesis

NUM_ENVS = 4
_STARTUP, _RESET, _INTERVAL = EventMode
LIFTED, RESTING, DROPPED = 2.0, 0.5, 0.05


@TERMINATION_TERM_REGISTRY.register()
def sgs_test_box_lifted(env) -> torch.Tensor:
    return env.entities["box"].get_pos()[:, 2] > 1.0


@TERMINATION_TERM_REGISTRY.register()
def sgs_test_box_dropped(env) -> torch.Tensor:
    return env.entities["box"].get_pos()[:, 2] < 0.2


@EVENT_TERM_REGISTRY.register()
class DrawResetConfigs(EventTerm):
    """The smallest reset consumer: owns ``num_configs`` configurations and draws one per resetting env."""

    mode: EventMode = _RESET
    sampler_name: str | None = "sgs"
    num_configs: int = 1

    def build(self) -> None:
        self.sampler = resolve_reset_config_sampler(self._env, self.sampler_name, consumer="draw")
        self.sampler.bind(self.num_configs, consumer="draw")

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        self.sampler.sample(sanitize_envs_idx(envs_idx, self.num_envs))


def _config(*, num_configs: int = 1, consumer: bool = True, **sgs) -> EdenRLConfig:
    sgs.setdefault("success_term", "lifted")
    return EdenRLConfig(
        env_options=EnvOptions(num_envs=NUM_ENVS),
        scene_options=SceneOptions(box=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.0, 0.0, RESTING))),
        termination_options=TerminationManagerOptions(
            lifted=DoneTerm.configure(func=sgs_test_box_lifted),
            dropped=DoneTerm.configure(func=sgs_test_box_dropped),
        ),
        curriculum_options=CurriculumManagerOptions(sgs=SuccessGuidedSampling.configure(**sgs)),
        event_options=EventManagerOptions(
            **({"draw": DrawResetConfigs.configure(num_configs=num_configs)} if consumer else {})
        ),
    )


def _build(config: EdenRLConfig, *, eval_mode: bool = False):
    env = env_class_for_config(config).from_config(config, show_viewer=False, eval_mode=eval_mode)
    env.build()
    env.reset()
    return env


def _sgs(env) -> SuccessGuidedSampling:
    return env.curriculum_manager.get_term("sgs")


def _step_from(env, set_box_heights, heights) -> None:
    """Place the boxes, then take one step: the terminations grade the placement and the done envs reset."""
    set_box_heights(env, heights)
    env.step(torch.zeros(NUM_ENVS, 0, device=env.device))


@pytest.fixture(scope="module")
def one_config_env():
    """Every env draws the single configuration, so each finished batch lands on one ring of six slots."""
    return _build(_config(num_configs=1, history=6))


@pytest.fixture(scope="module")
def three_config_env():
    return _build(_config(num_configs=3))


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("success_rate", "params", "expected"),
    [
        # Hand-computed from paper Eq. 1-2 + Appendix B.2 with Python floats (math.log / math.exp).
        (
            [0.0, 0.5, 1.0],
            dict(target=0.5, concentration=1.0, temperature=2.0, floor=1e-4),
            [0.110654, 0.778693, 0.110654],
        ),
        (
            [0.0, 0.66, 1.0],
            dict(target=0.66, concentration=5.0, temperature=2.0, floor=1e-8),
            [7.01295e-4, 0.998597, 7.01295e-4],
        ),
        # temperature below 1 runs at 1 (T_eff = max(T, 1)).
        (
            [0.0, 0.66, 1.0],
            dict(target=0.66, concentration=5.0, temperature=0.5, floor=1e-8),
            [4.93197e-7, 0.999999, 4.93197e-7],
        ),
        # concentration 0 is uniform whatever the success rates.
        ([0.2, 0.9], dict(target=0.5, concentration=0.0, temperature=2.0, floor=1e-4), [0.5, 0.5]),
    ],
    ids=["manipulation", "locomotion", "temperature-floor", "flat-kernel"],
)
def test_probabilities_match_the_paper_kernel(success_rate, params, expected):
    probs = success_guided_probabilities(torch.tensor(success_rate), **params)
    torch.testing.assert_close(probs, torch.tensor(expected), rtol=1e-4, atol=1e-9)


# ---------------------------------------------------------------------------
# Grading finished episodes
# ---------------------------------------------------------------------------


def test_envs_finishing_one_config_together_take_consecutive_ring_slots(one_config_env, set_box_heights):
    """Four outcomes on one ring per step: a collision would leave one slot written, not four."""
    sgs = _sgs(one_config_env)
    assert sgs.num_outcomes.tolist() == [0]  # the build-time spawn grades nothing

    _step_from(one_config_env, set_box_heights, [LIFTED] * NUM_ENVS)
    assert sgs.num_outcomes.tolist() == [4]
    torch.testing.assert_close(sgs.success_rate, torch.tensor([1.0]))

    # Four failures wrap the six-slot ring onto two of the first batch's successes: 2 successes in 6.
    _step_from(one_config_env, set_box_heights, [DROPPED] * NUM_ENVS)
    assert sgs.num_outcomes.tolist() == [8]
    torch.testing.assert_close(sgs.success_rate, torch.tensor([2.0 / 6.0]))

    # Published: graded, on the frontier (0 < p < 1), and the whole set is sampled.
    assert {key: round(float(value), 4) for key, value in sgs.compute().items()} == {
        "mean_success": round(2 / 6, 4),
        "visited_frac": 1.0,
        "frontier_frac": 1.0,
        "sampling_ess_frac": 1.0,
    }


def _ring_rate(sgs: SuccessGuidedSampling) -> torch.Tensor:
    """Each configuration's success rate re-reduced from the saved ring, against the running sum."""
    state = sgs.state_dict()
    return state["outcomes"].float().sum(dim=1) / state["count"].clamp(max=sgs.history, min=1)


def test_more_finishes_than_history_record_history_of_them_and_keep_exact_rates(set_box_heights):
    """Four envs end one two-slot ring per step: two are recorded, and the running rate matches the ring."""
    torch.manual_seed(0)
    env = _build(_config(num_configs=1, history=2))
    sgs = _sgs(env)

    _step_from(env, set_box_heights, [LIFTED, DROPPED, LIFTED, LIFTED])
    assert sgs.num_outcomes.tolist() == [2]
    torch.testing.assert_close(sgs.success_rate, _ring_rate(sgs))

    # The next batch overwrites both slots; the running sum must drop whatever each one replaced.
    for heights in ([LIFTED, LIFTED, DROPPED, DROPPED], [DROPPED, LIFTED, DROPPED, LIFTED]):
        _step_from(env, set_box_heights, heights)
        torch.testing.assert_close(sgs.success_rate, _ring_rate(sgs))
    assert sgs.num_outcomes.tolist() == [6]


def test_more_finishes_than_history_are_not_kept_by_env_index(set_box_heights):
    """One slot, four finishers, env 0 the only success: keeping the lowest index would record it every time."""
    torch.manual_seed(0)
    env = _build(_config(num_configs=1, history=1))
    sgs = _sgs(env)

    rates = set()
    for _ in range(40):
        _step_from(env, set_box_heights, [LIFTED, DROPPED, DROPPED, DROPPED])
        rates.add(float(sgs.success_rate))
    assert rates == {0.0, 1.0}


def test_each_episode_is_graded_on_the_config_it_started_from_at_the_step_it_ended(three_config_env, set_box_heights):
    """The success term is read inside the reset event, so it must still hold the finished step's verdict."""
    sgs = _sgs(three_config_env)
    started_from = sgs.assignment.clone()
    counts_before = sgs.num_outcomes.clone()
    successes_before = (sgs.success_rate * counts_before).round()

    _step_from(three_config_env, set_box_heights, [LIFTED, DROPPED, DROPPED, LIFTED])

    succeeded = torch.tensor([1.0, 0.0, 0.0, 1.0])
    expected_counts = counts_before + torch.bincount(started_from, minlength=3)
    expected_successes = successes_before + torch.bincount(started_from, weights=succeeded, minlength=3)
    assert sgs.num_outcomes.tolist() == expected_counts.tolist()
    torch.testing.assert_close(sgs.success_rate, expected_successes / expected_counts.clamp(min=1))


def test_only_episodes_that_ended_on_the_task_are_graded(three_config_env, set_box_heights, flag_solver_error):
    """Env 1 is cut short by ``env.reset()``, env 3 ends on a solver error; only env 2's success counts."""
    env = three_config_env
    sgs = _sgs(env)
    _step_from(env, set_box_heights, [RESTING] * NUM_ENVS)  # one step in, nobody done
    started_from = sgs.assignment.clone()
    counts_before = sgs.num_outcomes.clone()

    set_box_heights(env, [RESTING, RESTING, LIFTED, LIFTED])
    flag_solver_error(env, [3])
    env.termination_manager.compute()
    env.reset(envs_idx=torch.tensor([1, 2, 3]))

    expected = counts_before.clone()
    expected[started_from[2]] += 1
    assert sgs.num_outcomes.tolist() == expected.tolist()


def test_draws_follow_the_sampling_distribution(three_config_env):
    sgs = _sgs(three_config_env)
    draws = torch.stack([sgs.sample(slice(None)).clone() for _ in range(500)]).flatten()
    frequencies = torch.bincount(draws, minlength=3).float() / draws.numel()
    torch.testing.assert_close(frequencies, sgs.probabilities, atol=0.05, rtol=0.0)


def test_at_eval_configs_are_drawn_uniformly_and_nothing_is_recorded(set_box_heights):
    env = _build(_config(num_configs=3, concentration=50.0), eval_mode=True)
    sgs = _sgs(env)
    _step_from(env, set_box_heights, [LIFTED, LIFTED, DROPPED, DROPPED])
    assert sgs.num_outcomes.tolist() == [0, 0, 0]

    draws = torch.stack([sgs.sample(slice(None)).clone() for _ in range(300)]).flatten()
    assert set(draws.tolist()) == {0, 1, 2}


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def test_state_dict_restores_the_outcome_history(three_config_env, set_box_heights):
    sgs = _sgs(three_config_env)
    _step_from(three_config_env, set_box_heights, [LIFTED, DROPPED, LIFTED, DROPPED])
    saved = sgs.state_dict()
    rates, counts, probs = sgs.success_rate.clone(), sgs.num_outcomes.clone(), sgs.probabilities.clone()

    _step_from(three_config_env, set_box_heights, [DROPPED] * NUM_ENVS)
    assert not torch.equal(sgs.num_outcomes, counts)
    sgs.load_state_dict(saved)

    assert torch.equal(sgs.num_outcomes, counts)
    torch.testing.assert_close(sgs.success_rate, rates)
    torch.testing.assert_close(sgs.probabilities, probs)


def test_episodes_in_flight_at_a_load_are_not_graded_into_the_loaded_history(set_box_heights):
    """They started from configurations drawn before the load; the episodes after them are graded as usual."""
    env = _build(_config(num_configs=1, history=8))
    sgs = _sgs(env)
    _step_from(env, set_box_heights, [LIFTED] * NUM_ENVS)
    sgs.load_state_dict(sgs.state_dict())
    assert sgs.first_spawn.all()

    _step_from(env, set_box_heights, [DROPPED] * NUM_ENVS)  # the in-flight episodes end: not graded
    assert sgs.num_outcomes.tolist() == [NUM_ENVS]
    _step_from(env, set_box_heights, [DROPPED] * NUM_ENVS)  # the first post-load episodes end: graded
    assert sgs.num_outcomes.tolist() == [2 * NUM_ENVS]
    torch.testing.assert_close(sgs.success_rate, torch.tensor([0.5]))


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"num_configs": 200}, r"saved for a set of 200 reset configurations, but .* bound this one to 3"),
        ({"history": 7}, r"history=7 outcomes per configuration, this term keeps history=100"),
    ],
    ids=["another-set", "another-history"],
)
def test_loading_another_configuration_set_or_history_raises(three_config_env, override, match):
    sgs = _sgs(three_config_env)
    counts = sgs.num_outcomes.clone()
    with pytest.raises(ValueError, match=match):
        sgs.load_state_dict(sgs.state_dict() | override)
    assert torch.equal(sgs.num_outcomes, counts)


# ---------------------------------------------------------------------------
# Failing fast
# ---------------------------------------------------------------------------


def test_an_unknown_success_term_fails_the_build_listing_the_active_ones():
    with pytest.raises(ValueError, match=r"success_term='lifed'.*Active termination terms: lifted, dropped"):
        _build(_config(success_term="lifed"))


def test_out_of_range_kernel_params_fail_the_build():
    with pytest.raises(ValueError, match=r"target=1.5 \(in \[0, 1\]\); temperature=0.0 \(> 0\)"):
        _build(_config(target=1.5, temperature=0.0))


def test_a_sampler_no_reset_event_binds_fails_the_env_build():
    with pytest.raises(RuntimeError, match="SuccessGuidedSampling reached the end of env construction unbound"):
        _build(_config(consumer=False))


# ---------------------------------------------------------------------------
# GPU-sync budget
# ---------------------------------------------------------------------------


def test_a_training_reset_is_sync_free(cuda_stand_in_env, count_cuda_syncs):
    """``sample`` runs inside every auto-reset; a compacted mask or an ``.item()`` would sync each step."""
    env = cuda_stand_in_env
    env.termination_manager = TerminationManager(
        env=env, options=TerminationManagerOptions(lifted=DoneTerm.configure(func=sgs_test_box_lifted))
    )
    sgs = SuccessGuidedSampling(env, SuccessGuidedSampling.configure(success_term="lifted"))
    sgs.build()
    sgs.bind(8, consumer="draw")
    env.episode_length_buf.fill_(3)
    mask = torch.tensor([True, False, True, True], device="cuda")
    index = torch.tensor([0, 2], device="cuda")

    assert count_cuda_syncs(lambda: sgs.sample(mask)) == 0
    assert count_cuda_syncs(lambda: sgs.sample(index)) == 0
    assert count_cuda_syncs(sgs.compute) == 0
