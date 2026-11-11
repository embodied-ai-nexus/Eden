"""Tests for ``CurriculumManager``: it freezes whole at eval, and ``get_state`` never returns ``None`` —
each way there can be no state is its own raise naming the fix.

Built envs carry a real curriculum (an episode-length tracker, a ``PenaltyCurriculum`` reading it, a
``StageRewardWeightCurriculum``); the lifecycle cases use small inline terms on a standalone manager.
"""

from __future__ import annotations

import pytest
import torch

from eden.managers.curriculum_manager import CURRICULUM_TERM_REGISTRY, CurriculumManager, CurriculumTerm
from eden.managers.reward_manager import REWARD_TERM_REGISTRY
from eden.managers.terms.curricula.locomotion import AverageEpisodeLengthTracker
from eden.managers.terms.curricula.rewards import PenaltyCurriculum, StageRewardWeightCurriculum
from eden.options.managers.curricula import CurriculumManagerOptions
from eden.options.managers.rewards import RewardManagerOptions, RewardTermOptions

NUM_ENVS = 4


@REWARD_TERM_REGISTRY.register()
def curriculum_test_reward(env) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


@CURRICULUM_TERM_REGISTRY.register()
class LevelCurriculum(CurriculumTerm):
    """Publishes ``{"level": level}``; counts the resets that fed it, and raises from ``compute`` on demand."""

    level: int = 3

    def build(self) -> None:
        self.resets_fed = 0
        self.fail = False

    def reset(self, envs_idx=None) -> None:
        self.resets_fed += 1

    def compute(self) -> dict:
        if self.fail:
            raise ValueError("boom")
        return {"level": self.level}


@CURRICULUM_TERM_REGISTRY.register()
class SilentCurriculum(CurriculumTerm):
    def compute(self) -> None:
        return None


def _rewards() -> RewardManagerOptions:
    return RewardManagerOptions(
        torque=RewardTermOptions(name="curriculum_test_reward", weight=-1.0, tags=["penalty"]),
        reaching=RewardTermOptions(name="curriculum_test_reward", weight=1.0),
    )


def _tracker():
    return AverageEpisodeLengthTracker.configure(num_compute_average_epl=10)


def _penalty():
    return PenaltyCurriculum.configure(
        reward_term_tags=["penalty"],
        tracker_name="ep_tracker",
        initial_scale=0.5,
        degree=1.0,
        level_up_threshold=800.0,
        level_down_threshold=100.0,
    )


def _ordered() -> CurriculumManagerOptions:
    return CurriculumManagerOptions(
        ep_tracker=_tracker(),
        penalty=_penalty(),
        reaching_weight=StageRewardWeightCurriculum.configure(reward_term_name="reaching", weight_stages=[(0, -0.01)]),
    )


@pytest.fixture(scope="module")
def env(make_box_env):
    return make_box_env(reward_options=_rewards(), curriculum_options=_ordered())


@pytest.fixture(scope="module")
def eval_env(make_box_env):
    return make_box_env(eval_mode=True, reward_options=_rewards(), curriculum_options=_ordered())


@pytest.fixture(scope="module")
def late_env(make_box_env):
    """The consumer declared before the producer it reads: builds fine, fails on the first compute."""
    return make_box_env(
        reward_options=_rewards(),
        curriculum_options=CurriculumManagerOptions(penalty=_penalty(), ep_tracker=_tracker()),
    )


def _levels(env, **terms) -> CurriculumManager:
    """A standalone manager over ``terms`` (default: one ``LevelCurriculum`` named ``level``)."""
    return CurriculumManager(
        env=env, options=CurriculumManagerOptions(**(terms or {"level": LevelCurriculum.configure()}))
    )


def _step(env) -> None:
    env.step(torch.zeros(NUM_ENVS, 0, device=env.device))


# ---------------------------------------------------------------------------
# Training vs. eval
# ---------------------------------------------------------------------------


def test_while_training_terms_adapt_publish_and_are_fed_the_finished_episodes(env):
    manager = _levels(env)

    manager.compute()
    extras = manager.reset(envs_idx=torch.tensor([0]))

    assert manager.get_state("level") == {"level": 3}
    assert extras == {"Curriculum/level/level": 3}
    assert manager.get_term("level").resets_fed == 1


def test_at_eval_terms_neither_adapt_nor_accumulate_and_get_state_says_why(eval_env):
    """Both halves frozen; running one without the other drifted the internal state from the published one.

    The absence is a raise, not a ``None`` a consumer could read as "not ready yet" and replace with a
    default for the whole rollout.
    """
    manager = _levels(eval_env)

    manager.compute()
    extras = manager.reset(envs_idx=torch.tensor([0]))

    assert extras == {}
    assert manager.get_term("level").resets_fed == 0
    with pytest.raises(RuntimeError) as exc_info:
        manager.get_state("level")
    message = str(exc_info.value)
    assert "eval_mode=True" in message  # the cause, in the words the caller passed
    assert "curriculum_manager.get_term('level')" in message  # the fix, as an expression
    assert "LevelCurriculum" in message  # the class to open, resolved for the reader
    assert "Do NOT catch this and fall back to a default" in message  # the wrong fix, named


def test_an_eval_env_logs_no_curriculum_keys_for_its_whole_rollout(eval_env):
    """The level an eval was graded at is the one its config pins; ``get_term`` works in both modes."""
    manager = eval_env.curriculum_manager
    eval_env.reset()
    for _ in range(2):
        _step(eval_env)

    extras = manager.reset(envs_idx=torch.arange(NUM_ENVS))

    assert not [key for key in extras if key.startswith("Curriculum/")]
    with pytest.raises(RuntimeError, match="eval_mode=True"):
        manager.get_state("reaching_weight")
    assert isinstance(manager.get_term("reaching_weight"), StageRewardWeightCurriculum)


def test_a_training_env_publishes_the_stage_weight_it_is_running(env):
    """The stage at step 0 is in force from the first step, and what reset() logs."""
    env.reset()
    _step(env)

    assert env.curriculum_manager.get_state("reaching_weight") == {"weight": -0.01}
    assert env.curriculum_manager.reset(envs_idx=torch.arange(NUM_ENVS))["Curriculum/reaching_weight/weight"] == -0.01


# ---------------------------------------------------------------------------
# get_state: one raise per way of having no state
# ---------------------------------------------------------------------------


def test_a_term_that_publishes_nothing_is_not_mistaken_for_one_that_never_ran(env):
    manager = _levels(env, silent=SilentCurriculum.configure())
    manager.compute()

    with pytest.raises(RuntimeError, match=r"computed but published nothing"):
        manager.get_state("silent")


def test_a_read_before_the_first_compute_is_not_blamed_on_declaration_order(env):
    """Telling a caller reading from ``build()`` to reorder curriculum_options would send them to a config that is fine."""
    manager = _levels(env)

    with pytest.raises(RuntimeError) as exc_info:
        manager.get_state("level")

    message = str(exc_info.value)
    assert "before the run's first CurriculumManager.compute()" in message
    assert "curriculum_options" not in message.split("Fix:")[0]


def test_a_producer_declared_after_its_consumer_names_the_entry_to_move(late_env):
    """The traceback carries the consumer's class, but a task can hold several entries of it."""
    manager = late_env.curriculum_manager

    with pytest.raises(RuntimeError) as exc_info:
        manager.compute()

    message = str(exc_info.value)
    assert "curriculum_options order: penalty, ep_tracker" in message
    assert "'penalty' is reading 'ep_tracker'" in message
    assert "move 'ep_tracker' above 'penalty' in curriculum_options" in message
    # The loop's bookkeeping does not survive the raise: a later read is not blamed on 'penalty'.
    with pytest.raises(RuntimeError, match=r"before the run's first CurriculumManager.compute\(\)"):
        manager.get_state("ep_tracker")


def test_a_producer_declared_first_serves_its_consumer(env):
    """The same two terms in the right order: the penalty scale moves off the tracker's average."""
    manager = env.curriculum_manager
    env.reset()
    try:
        env.episode_length_buf[:] = 900  # finished episodes well past level_up_threshold
        manager.reset(envs_idx=torch.arange(NUM_ENVS))
        manager.compute()
    finally:
        env.episode_length_buf[:] = 0

    assert manager.get_state("ep_tracker")["average_episode_length"] == pytest.approx(900.0)
    assert manager.get_state("penalty")["scale"] == pytest.approx(1.0)  # 0.5 + degree 1.0, clamped at max_scale


def test_a_value_from_an_earlier_step_is_refused_as_stale(env):
    """The cache survives the step boundary on purpose (reset() at phase 13 logs from it), so it is dated.

    A read earlier in the step than compute() (16) says so rather than serving last step's level;
    refreshing the cache by hand, the attractive wrong fix, double-adapts every self-adapting term.
    """
    manager = _levels(env)
    manager.compute()
    published_at = env.common_step_counter
    assert manager.get_state("level") == {"level": 3}  # a read later in the same step is current

    _step(env)

    with pytest.raises(RuntimeError) as exc_info:
        manager.get_state("level")
    message = str(exc_info.value)
    assert f"last published at step {published_at}, and it is now step {published_at + 1}" in message
    assert "curriculum_manager.get_term('level')" in message
    assert "Do NOT call CurriculumManager.compute() yourself" in message


def test_a_term_whose_compute_raised_keeps_the_date_of_its_last_success(env):
    """Per-term stamps: a caller that catches the failure must be told the value is stale."""
    manager = _levels(env, level=LevelCurriculum.configure(), other=LevelCurriculum.configure())
    manager.compute()
    published_at = env.common_step_counter
    _step(env)
    manager.get_term("level").fail = True

    with pytest.raises(ValueError, match="boom"):
        manager.compute()

    with pytest.raises(
        RuntimeError, match=rf"last published at step {published_at}, and it is now step {published_at + 1}"
    ):
        manager.get_state("level")


def test_an_unknown_term_name_raises_a_key_error_listing_the_declared_terms(env):
    """Without the raise, a typo would read exactly like "not computed yet": ``None``, forever."""
    manager = _levels(env)

    with pytest.raises(KeyError, match=r"No curriculum term named 'levl'.*Declared.*level"):
        manager.get_state("levl")
