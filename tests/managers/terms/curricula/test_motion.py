"""Tests for ``eden.managers.terms.curricula.motion``: ConTrack's success-driven pose-tolerance curriculum.

The curriculum reads episode success off a real motion reference (a clip played to its end) and
writes its tolerances onto a real ``MotionBasePoseTrackingBreak`` term.
"""

from __future__ import annotations

import pytest
import torch

from eden.managers.terms.curricula.motion import MotionBasePoseThresholdCurriculum
from tests.support.motion import build_mimic_env, manager_options, pin_clips, term_options, write_library

ALPHA = MotionBasePoseThresholdCurriculum.success_ema_alpha


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from eden.managers.terms.terminations.tracking import MotionBasePoseTrackingBreak
    from eden.options.managers.terminations import TerminationManagerOptions, TerminationTermOptions

    root = write_library(tmp_path_factory.mktemp("curriculum"), [20])
    return build_mimic_env(
        root,
        termination_options=TerminationManagerOptions(
            object_pose_break=MotionBasePoseTrackingBreak.configure(entity_name="object"),
            robot_pose_break=MotionBasePoseTrackingBreak.configure(entity_name="robot"),
            time_out=TerminationTermOptions(name="time_out", time_out=True),
        ),
    )


def _curriculum(env, **params):
    term = MotionBasePoseThresholdCurriculum(env, MotionBasePoseThresholdCurriculum.configure(**params))
    term.build()
    return term


def _episodes(env, success, stepped=None):
    """End an episode in each env: ``success`` ones with the clip played out, the rest mid-clip."""
    success = list(success) + [False] * (env.num_envs - len(success))
    stepped = [True] * env.num_envs if stepped is None else list(stepped) + [False] * (env.num_envs - len(stepped))
    pin_clips(env.motion_manager.get_term("default"), [0] * env.num_envs, start_s=[1.0 if s else 0.0 for s in success])
    env.motion_manager.fill_data(torch.arange(env.num_envs))
    env.episode_length_buf[:] = torch.tensor([5 if s else 0 for s in stepped])
    # The motion manager takes the ended episodes' validity as it resets, before the window moves on.
    env.motion_manager.reset(torch.arange(env.num_envs))


def _ema(term) -> float:
    return float(term.compute()["success_ema"])


def _thresholds(env) -> tuple[float, float]:
    pos_threshold, rot_threshold = env.termination_manager.get_term("object_pose_break").thresholds
    return float(pos_threshold), float(rot_threshold)


@pytest.mark.parametrize(
    "params, match",
    [
        # high == low once divided by zero: NaN thresholds, and a gate that silently never fires.
        ({"success_rate_low": 0.9, "success_rate_high": 0.9}, "success_rate_high > success_rate_low"),
        ({"success_rate_low": 0.999, "success_rate_high": 0.9}, "success_rate_high > success_rate_low"),
        ({"pos_threshold_min": 0.5, "pos_threshold_max": 0.1}, "pos_threshold_min"),
        ({"success_ema_alpha": 0.0}, "success_ema_alpha"),
        ({"success_ema_alpha": -0.1}, "success_ema_alpha"),
        ({"success_ema_alpha": 1.5}, "success_ema_alpha"),
        ({"termination_term_name": "robot_pose_break"}, "grades success on 'object'"),
    ],
)
def test_misconfigurations_fail_at_build(env, params, match):
    with pytest.raises(ValueError, match=match):
        _curriculum(env, **params)


@pytest.mark.parametrize(
    "params, error, match",
    [
        (
            {"entity_name": "objekt"},
            ValueError,
            r"no motion term tracks 'objekt'; tracked entities: \['object', 'robot'\]",
        ),
        ({"termination_term_name": "object_pose_brake"}, KeyError, "object_pose_brake"),
    ],
)
def test_an_unknown_entity_or_termination_term_fails_at_build(env, params, error, match):
    """The termination manager is built before the curricula, so the driven term resolves in build."""
    with pytest.raises(error, match=match):
        _curriculum(env, **params)


def test_thresholds_start_lenient(env):
    state = _curriculum(env).compute()

    assert _thresholds(env) == pytest.approx((0.1, 1.0))
    assert set(state) == {"success_ema", "pos_threshold", "rot_threshold"}
    assert float(state["success_ema"]) == 0.0


@pytest.mark.parametrize(
    "success, params, expected",
    [
        ([True] * 4, {}, (0.05, 0.5)),
        ([True, False, True, False], {"success_rate_low": 0.0, "success_rate_high": 1.0}, (0.075, 0.75)),
    ],
    ids=["full-success-is-tight", "midpoint-interpolates"],
)
def test_thresholds_follow_the_success_rate(env, success, params, expected):
    term = _curriculum(env, success_ema_alpha=1.0, **params)
    _episodes(env, success)
    term.reset(torch.arange(4))

    term.compute()

    assert _thresholds(env) == pytest.approx(expected)


def test_the_thresholds_are_written_in_place_so_the_gate_reads_one_tensor_all_run(env):
    """A CUDA graph of the graded phase (``EnvOptions.cuda_graph_options``) reads the gate's thresholds where they
    live; a fresh tensor every compute() would ask it to re-capture every step. The gate's configured fields are
    left as configured.
    """
    term = _curriculum(env, success_ema_alpha=1.0)
    gate = env.termination_manager.get_term("object_pose_break")
    bound = gate.thresholds
    configured = gate.pos_threshold, gate.rot_threshold
    _episodes(env, [True] * 4)
    term.reset(torch.arange(4))

    term.compute()

    assert _thresholds(env) == pytest.approx((0.05, 0.5))
    assert all(now is then for now, then in zip(gate.thresholds, bound, strict=True))
    assert (gate.pos_threshold, gate.rot_threshold) == configured


def test_a_curriculum_driving_a_term_that_is_not_a_pose_break_gate_fails_at_build(env):
    """It moves the gate through ``set_thresholds``, which only ``MotionBasePoseTrackingBreak`` has."""
    with pytest.raises(TypeError, match=r"can only move a MotionBasePoseTrackingBreak gate"):
        _curriculum(env, termination_term_name="time_out")


def test_the_ema_rate_is_per_episode_not_per_batch(env):
    """Otherwise its time constant scales with num_envs: at 4096 envs, a per-batch alpha converges
    within a fraction of one episode.
    """
    _episodes(env, [True] * 4)
    batched = _curriculum(env)
    batched.reset(torch.arange(4))
    one_at_a_time = _curriculum(env)
    for e in range(4):
        one_at_a_time.reset(torch.tensor([e]))

    expected = 1.0 - (1.0 - ALPHA) ** 4
    assert _ema(batched) == pytest.approx(expected, rel=1e-6)
    assert _ema(one_at_a_time) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    "success, stepped, expected",
    [
        ([True, False, True, False], None, 0.5 * (1.0 - (1.0 - ALPHA) ** 4)),
        # The bootstrap reset: an all-invalid, never-stepped buffer would read as instant success.
        ([True] * 4, [False] * 4, 0.0),
        ([True, True, False, False], [True, False, False, False], ALPHA),
    ],
    ids=["successful-share", "never-stepped", "partially-stepped"],
)
def test_only_stepped_episodes_move_the_ema_by_their_success(env, success, stepped, expected):
    term = _curriculum(env)
    _episodes(env, success, stepped)

    term.reset(torch.arange(4))

    assert _ema(term) == pytest.approx(expected, rel=1e-6)


def test_an_empty_reset_and_a_state_dict_round_trip_keep_the_ema(env):
    term = _curriculum(env, success_ema_alpha=1.0)
    _episodes(env, [True, False, True, True])
    term.reset(torch.arange(4))

    term.reset(torch.zeros(0, dtype=torch.long))
    assert _ema(term) == pytest.approx(0.75)

    restored = _curriculum(env)
    restored.load_state_dict(term.state_dict())
    assert _ema(restored) == pytest.approx(0.75)


def test_an_env_reset_grades_the_episode_that_ended_even_when_rsi_reseeds_it(tmp_path):
    """Through ``_reset_idx``, with RSI terms seeding the reset envs: the curriculum reads the outgoing clip.

    The curriculum resets after the RESET events, so a refill of the reference during them (an RSI
    term reading its init state off a freshly filled window) would hand it the new clip, which
    is never exhausted at frame 0, and every episode would grade as a failure.
    """
    import eden as en
    from eden.managers.terms.events.motion import ObjectReferenceStateInitialization, ReferenceStateInitialization
    from eden.managers.terms.terminations.tracking import MotionBasePoseTrackingBreak
    from eden.options.managers.curricula import CurriculumManagerOptions
    from eden.options.managers.events import EventManagerOptions
    from eden.options.managers.terminations import TerminationManagerOptions

    reset = en.EventMode.RESET
    root = write_library(tmp_path, [20])
    env = build_mimic_env(
        root,
        # A reset starts the new clip within its first tenth, so frame 0 of the new window is valid.
        motion_options=manager_options(
            terms={"default": term_options(root, motion_start_from_middle_range=(0.0, 0.1))}
        ),
        termination_options=TerminationManagerOptions(
            object_pose_break=MotionBasePoseTrackingBreak.configure(entity_name="object")
        ),
        event_options=EventManagerOptions(
            reset_robot=ReferenceStateInitialization.configure(mode=reset, entity_name="robot"),
            reset_object=ObjectReferenceStateInitialization.configure(mode=reset, entity_name="object"),
        ),
        curriculum_options=CurriculumManagerOptions(
            object_pose=MotionBasePoseThresholdCurriculum.configure(success_ema_alpha=1.0)
        ),
    )
    pin_clips(env.motion_manager.get_term("default"), [0] * env.num_envs, start_s=1.0)  # every clip played out
    env.motion_manager.fill_data(torch.arange(env.num_envs))
    env.episode_length_buf[:] = 5

    env._reset_idx(torch.ones(env.num_envs, dtype=torch.bool, device=env.device))
    env._dispatch_post_reset()

    assert _ema(env.curriculum_manager.get_term("object_pose")) == pytest.approx(1.0)
