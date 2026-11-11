"""The reward curricula in ``eden/managers/terms/curricula/rewards.py``.

``PenaltyCurriculum`` (targets by name or tag, the initial scale, the tracker and legacy paths) and
``StageRewardWeightCurriculum``, built standalone against a real env whose reward manager holds four
tagged terms and whose curriculum manager publishes a state the test scripts.
"""

from __future__ import annotations

import pytest
import torch

from eden.envs.base import env_class_for_config
from eden.managers.curriculum_manager import CURRICULUM_TERM_REGISTRY, CurriculumTerm
from eden.managers.reward_manager import REWARD_TERM_REGISTRY
from eden.managers.terms.curricula.rewards import PenaltyCurriculum, StageRewardWeightCurriculum
from eden.options import BoxOptions, EnvOptions, SceneOptions
from eden.options.managers.curricula import CurriculumManagerOptions
from eden.options.managers.rewards import RewardManagerOptions, RewardTermOptions
from eden.utils.configs import EdenRLConfig

pytestmark = pytest.mark.requires_genesis

NUM_ENVS = 4
WEIGHTS = {"torque_penalty": -1.0, "ang_vel_penalty": -0.5, "feet_smoothness": -2.0, "tracking_reward": 2.0}
TAGS = {
    "torque_penalty": ["penalty", "effort"],
    "ang_vel_penalty": ["penalty", "smooth"],
    "feet_smoothness": ["smooth"],
    "tracking_reward": ["tracking"],
}


@REWARD_TERM_REGISTRY.register()
def curricula_test_reward(env) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


@CURRICULUM_TERM_REGISTRY.register()
class PublishedState(CurriculumTerm):
    """Publishes whatever the test put in ``state``: a tracker the test controls."""

    def build(self) -> None:
        self.state = None

    def compute(self):
        return self.state


def _build(*, eval_mode: bool):
    config = EdenRLConfig(
        env_options=EnvOptions(num_envs=NUM_ENVS),
        scene_options=SceneOptions(box=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.0, 0.0, 0.5))),
        reward_options=RewardManagerOptions(
            **{
                name: RewardTermOptions(name="curricula_test_reward", weight=weight, tags=TAGS[name])
                for name, weight in WEIGHTS.items()
            }
        ),
        curriculum_options=CurriculumManagerOptions(published=PublishedState.configure()),
    )
    env = env_class_for_config(config).from_config(config, show_viewer=False, eval_mode=eval_mode)
    env.build()
    env.reset()
    return env


@pytest.fixture(scope="module")
def env():
    return _build(eval_mode=False)


@pytest.fixture(scope="module")
def eval_env():
    return _build(eval_mode=True)


@pytest.fixture(autouse=True)
def configured_weights(request):
    """A curriculum rescales the env's reward terms in place; put the configured weights back after each test."""
    yield
    for name in ("env", "eval_env"):
        if name in request.fixturenames:
            built = request.getfixturevalue(name)
            for term_name, weight in WEIGHTS.items():
                built.reward_manager.get_term(term_name).weight = weight


def _weights(env) -> dict[str, float]:
    return {name: env.reward_manager.get_term(name).weight for name in WEIGHTS}


def _penalty(env, **kwargs) -> PenaltyCurriculum:
    """Built the way ``CurriculumManager`` builds it: constructed, then ``build()``."""
    kwargs.setdefault("reward_term_tags", ["penalty"])
    term = PenaltyCurriculum(env=env, options=PenaltyCurriculum.configure(**kwargs))
    term.build()
    return term


def _publish(env, state) -> None:
    """Make ``state`` what ``get_state('published')`` serves this step."""
    env.curriculum_manager.get_term("published").state = state
    env.curriculum_manager.compute()


# ---------------------------------------------------------------------------
# PenaltyCurriculum: which terms, at what initial scale
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "targets, initial_scale, expected",
    [
        (dict(reward_term_names=["torque_penalty"]), 0.1, dict(torque_penalty=-0.1)),
        (dict(reward_term_tags=["penalty"]), 0.5, dict(torque_penalty=-0.5, ang_vel_penalty=-0.25)),
        (  # any shared tag selects a term
            dict(reward_term_tags=["smooth", "effort"]),
            0.4,
            dict(torque_penalty=-0.4, ang_vel_penalty=-0.2, feet_smoothness=-0.8),
        ),
    ],
    ids=["by-name", "by-tag", "by-tag-overlap"],
)
def test_the_initial_scale_is_installed_on_exactly_the_targeted_terms(env, targets, initial_scale, expected):
    _penalty(env, initial_scale=initial_scale, **{"reward_term_tags": [], **targets})

    assert _weights(env) == pytest.approx(WEIGHTS | expected)


def test_an_unknown_tag_targets_nothing_and_says_so(env, eden_caplog):
    """A task may legitimately not use a tag yet, so it is a warning, not an error."""
    _penalty(env, reward_term_tags=["nonexistent_tag"], initial_scale=0.5)

    assert _weights(env) == WEIGHTS
    assert any("resolved zero target reward terms" in record.getMessage() for record in eden_caplog.records)


def test_names_and_tags_together_are_refused(env):
    """Renaming a term would otherwise silently leave the tag list active."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        PenaltyCurriculum(
            env=env,
            options=PenaltyCurriculum.configure(reward_term_names=["torque_penalty"], reward_term_tags=["penalty"]),
        )


def test_an_unknown_reward_term_name_raises_at_build(env):
    with pytest.raises(KeyError, match="pne"):
        _penalty(env, reward_term_tags=[], reward_term_names=["pne"])


@pytest.mark.parametrize("which", ["env", "eval_env"], ids=["train", "eval"])
def test_the_initial_scale_is_installed_at_build_in_both_modes(request, which):
    """The eval half is the regression: once both reset() and compute() froze at eval, nothing installed
    it there, and an eval rollout graded the penalty at 10x the intended magnitude.
    """
    built = request.getfixturevalue(which)

    _penalty(built, reward_term_names=["torque_penalty"], reward_term_tags=[], initial_scale=0.1)

    assert built.reward_manager.get_term("torque_penalty").weight == pytest.approx(-0.1)


# ---------------------------------------------------------------------------
# PenaltyCurriculum: the tracker path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "state, scale",
    [
        ({"average_episode_length": 900.0, "is_primed": True}, 1.0),  # above level_up: up by degree, clamped
        ({"average_episode_length": 50.0, "is_primed": True}, 0.0),  # below level_down: down, clamped
        ({"average_episode_length": 0.0, "is_primed": False}, 0.5),  # no reset recorded yet: hold
        ({"average_episode_length": 900.0}, 1.0),  # a tracker without the flag counts as primed
    ],
    ids=["level-up", "level-down", "unprimed", "no-primed-flag"],
)
def test_the_scale_follows_the_trackers_published_average(env, state, scale):
    """``degree=1`` so one breach is visible; three ticks show an unprimed tracker holds the scale for good."""
    curriculum = _penalty(
        env,
        tracker_name="published",
        initial_scale=0.5,
        degree=1.0,
        level_up_threshold=800.0,
        level_down_threshold=100.0,
    )

    for _ in range(3):
        _publish(env, state)
        out = curriculum.compute()

    assert out == {"scale": pytest.approx(scale), "avg_episode_length": state["average_episode_length"]}
    assert env.reward_manager.get_term("torque_penalty").weight == pytest.approx(-1.0 * scale)


def test_the_managers_raise_for_a_missing_tracker_state_reaches_the_caller(env):
    """Swallowing it and no-opping for a step is a silent default that hides the mistake."""
    curriculum = _penalty(env, tracker_name="published")
    _publish(env, None)

    with pytest.raises(RuntimeError, match="computed but published nothing"):
        curriculum.compute()


def test_a_tracker_state_without_an_average_raises_naming_the_key(env):
    curriculum = _penalty(env, tracker_name="published")
    _publish(env, {"some_other_key": 1.0})

    with pytest.raises(KeyError, match="average_episode_length"):
        curriculum.compute()


def test_with_a_tracker_no_legacy_episode_buffer_is_kept(env):
    """The named tracker owns the EMA; the CPU buffer exists only on the legacy path (read directly: it has no output)."""
    curriculum = _penalty(env, tracker_name="published")
    curriculum.reset(envs_idx=torch.arange(NUM_ENVS))

    assert not hasattr(curriculum, "_episode_lengths")


def test_without_a_tracker_the_scale_moves_only_once_an_episode_is_recorded(env):
    curriculum = _penalty(env, initial_scale=0.5, degree=1.0, level_up_threshold=800.0, level_down_threshold=100.0)
    assert curriculum.compute() == {"scale": pytest.approx(0.5), "avg_episode_length": 0.0}

    env.episode_length_buf[:] = 900
    try:
        curriculum.reset(envs_idx=torch.arange(NUM_ENVS))
    finally:
        env.episode_length_buf[:] = 0

    assert curriculum.compute() == {"scale": pytest.approx(1.0), "avg_episode_length": pytest.approx(900.0)}


# ---------------------------------------------------------------------------
# StageRewardWeightCurriculum
# ---------------------------------------------------------------------------


def _stage(env, stages) -> StageRewardWeightCurriculum:
    term = StageRewardWeightCurriculum(
        env=env,
        options=StageRewardWeightCurriculum.configure(reward_term_name="tracking_reward", weight_stages=stages),
    )
    term.build()
    return term


def test_the_stages_are_sorted_on_a_copy_of_the_config(env):
    stages = [(0, 1.0), (100, 0.5), (50, 0.8)]

    term = _stage(env, stages)

    assert stages == [(0, 1.0), (100, 0.5), (50, 0.8)]
    assert term.options.weight_stages == stages
    assert [step for step, _ in term.weight_stages] == [100, 50, 0]


def test_the_published_weight_is_the_one_in_force(env):
    """Before the first stage, the configured weight. Publishing nothing would read back as ``None``."""
    term = _stage(env, [(50, -0.1), (100, -1.0)])
    counter = env.common_step_counter
    try:
        for step, weight in ((0, 2.0), (50, -0.1), (10_000, -1.0)):
            env.common_step_counter = step
            assert term.compute() == {"weight": pytest.approx(weight)}, step
            assert env.reward_manager.get_term("tracking_reward").weight == pytest.approx(weight)
    finally:
        env.common_step_counter = counter


def test_an_unknown_reward_term_raises_at_build(env):
    with pytest.raises(KeyError, match="alve"):
        StageRewardWeightCurriculum(
            env=env, options=StageRewardWeightCurriculum.configure(reward_term_name="alve", weight_stages=[(0, 0.5)])
        ).build()
