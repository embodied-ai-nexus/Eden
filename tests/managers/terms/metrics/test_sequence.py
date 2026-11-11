"""``SequentialMetricTerm`` in ``eden/managers/terms/metrics/sequence.py``: ordered phases and their holds.

A phase whose predicate must change mid-test is "cube X is lifted", toggled with the ``lift`` fixture.
The term's value is the fraction of phases each env has completed.
"""

from __future__ import annotations

import pytest
import torch

from eden.managers.metric_manager import MetricManager
from eden.managers.terms.metrics.proximity import EeNearEntity
from eden.managers.terms.metrics.sequence import SequentialMetricTerm
from eden.options.managers.metrics import MetricManagerOptions, MetricTermOptions, PhaseOptions

pytestmark = pytest.mark.requires_genesis

GRASP, LIFT, PLACE = "cube_a", "cube_b", "cube_c"


def _term(name, **params):
    return MetricTermOptions(name=name, **({"params": params} if params else {}))


def _lifted(cube):
    return _term("object_above_height", entity_name=cube, height=0.5)


TRUE, FALSE = _term("always_true"), _term("always_false")


def _phase(term, **fields):
    return PhaseOptions(term=term, **fields)


@pytest.fixture
def sequence(panda_env, make_term):
    def build(*phases, **params):
        return make_term(SequentialMetricTerm, panda_env, phases=list(phases), **params)

    return build


def _progress(term, *expected):
    """Compute once and compare every env's completed fraction (one value, or one per env)."""
    out = term.compute()
    torch.testing.assert_close(out, torch.tensor(expected if len(expected) > 1 else expected * len(out)))


# ---------------------------------------------------------------------------
# Advancement
# ---------------------------------------------------------------------------


def test_no_phases_is_complete(sequence):
    _progress(sequence(), 1.0)


@pytest.mark.parametrize(("predicate", "expected"), [(TRUE, 1.0), (FALSE, 0.0)])
def test_a_single_phase_passes_with_its_predicate(sequence, predicate, expected):
    _progress(sequence(_phase(predicate)), expected)


def test_phases_advance_one_per_compute_and_stay_complete(sequence):
    term = sequence(_phase(TRUE), _phase(TRUE))
    for expected in (0.5, 1.0, 1.0):  # clamped at the phase count once complete
        _progress(term, expected)


def test_allow_skip_advances_through_every_passing_phase_at_once(sequence):
    _progress(sequence(_phase(TRUE), _phase(TRUE), _phase(TRUE), allow_skip=True), 1.0)


def test_each_env_advances_on_its_own_predicates(sequence):
    term = sequence(_phase(_term("env_specific", pass_envs=[0, 1])), _phase(_term("env_specific", pass_envs=[0])))
    _progress(term, 0.5, 0.5)
    _progress(term, 1.0, 0.5)


def test_phases_can_be_builtin_function_and_class_terms(panda_env, sequence, lift):
    """A lifted cube, then the fingers near it: env 0 brings the cube to the fingers, env 1 does not."""
    reach = EeNearEntity.configure(
        robot_name="robot", ee_link_names=["left_finger", "right_finger"], entity_name=GRASP, threshold=0.05
    )
    term = sequence(_phase(_lifted(GRASP)), _phase(reach))
    lift(GRASP, False)
    _progress(term, 0.0)
    lift(GRASP, True)
    _progress(term, 0.5)
    robot = panda_env.entities["robot"]
    _, fingers = robot.find_named_links_idx_local(["left_finger", "right_finger"])
    at_fingers = robot.get_links_pos(ls_idx_local=fingers).mean(dim=1)
    panda_env.entities[GRASP].set_pos(torch.stack([at_fingers[0], at_fingers[1] + 5.0]))
    _progress(term, 1.0, 0.5)


def test_reset_restarts_the_selected_envs(sequence, lift):
    term = sequence(_phase(_lifted(GRASP)))
    lift(GRASP, True)
    _progress(term, 1.0)
    lift(GRASP, False)  # a reset env cannot re-advance
    term.reset(envs_idx=torch.tensor([0]))
    _progress(term, 0.0, 1.0)
    term.reset()
    _progress(term, 0.0, 0.0)


def test_success_through_the_metric_manager_is_all_phases_complete(panda_env, install_manager):
    options = MetricManagerOptions(
        task=SequentialMetricTerm.configure(phases=[_phase(TRUE), _phase(_term("env_specific", pass_envs=[0]))])
    )
    manager = install_manager(panda_env, "metric_manager", MetricManager(panda_env, options))
    manager.compute()
    assert manager.success_buf.tolist() == [False, False]
    manager.compute()
    assert manager.success_buf.tolist() == [True, False]


def test_a_ref_phase_builds_its_own_copy_of_a_sibling_metric(panda_env, install_manager, restore_state):
    """``ref=`` names a sibling in the manager's options; the phase instantiates it anew."""
    near = EeNearEntity.configure(
        robot_name="robot", ee_link_names=["left_finger", "right_finger"], entity_name=GRASP, threshold=0.05
    )
    options = MetricManagerOptions(
        ee_near=near, task=SequentialMetricTerm.configure(phases=[PhaseOptions(ref="ee_near", name="reach")])
    )
    robot, cube = panda_env.entities["robot"], restore_state(panda_env.entities[GRASP])
    _, fingers = robot.find_named_links_idx_local(["left_finger", "right_finger"])
    cube.set_pos(robot.get_links_pos(ls_idx_local=fingers).mean(dim=1))
    manager = install_manager(panda_env, "metric_manager", MetricManager(panda_env, options))
    manager.compute()
    torch.testing.assert_close(manager.get_value("task"), torch.ones(2))


def test_a_phase_is_addressable_by_name(sequence):
    term = sequence(_phase(TRUE, name="my_phase"))
    torch.testing.assert_close(term.get_phase_term("my_phase").compute(), torch.ones(2))


@pytest.mark.parametrize(
    ("phase", "match"),
    [
        (PhaseOptions(threshold=1.0), "must specify either 'term' or 'ref'"),
        (PhaseOptions(term=TRUE, ref="some_term"), "mutually exclusive"),
    ],
    ids=["neither", "both"],
)
def test_a_phase_names_exactly_one_predicate_source(sequence, phase, match):
    with pytest.raises(ValueError, match=match):
        sequence(phase)


# ---------------------------------------------------------------------------
# Holds
# ---------------------------------------------------------------------------


def test_hold_reverts_when_the_phase_predicate_stops_passing(sequence, lift):
    term = sequence(_phase(TRUE), _phase(_lifted(GRASP), hold=True), _phase(FALSE))
    lift(GRASP, True)
    _progress(term, 1 / 3)
    _progress(term, 2 / 3)
    lift(GRASP, False)
    _progress(term, 1 / 3)  # back to the held phase
    lift(GRASP, True)
    _progress(term, 2 / 3)


def test_without_hold_a_passed_phase_stays_passed(sequence, lift):
    term = sequence(_phase(_lifted(GRASP), hold=False), _phase(FALSE))
    lift(GRASP, True)
    _progress(term, 0.5)
    lift(GRASP, False)
    _progress(term, 0.5)


def test_a_completed_task_never_reverts(sequence, lift):
    term = sequence(_phase(_lifted(GRASP), hold=True), _phase(TRUE))
    lift(GRASP, True)
    _progress(term, 0.5)
    _progress(term, 1.0)
    lift(GRASP, False)
    _progress(term, 1.0)


def test_failing_holds_revert_to_the_earliest(sequence, lift):
    term = sequence(_phase(_lifted(GRASP), hold=True), _phase(_lifted(LIFT), hold=True), _phase(FALSE))
    lift(GRASP, True)
    lift(LIFT, True)
    _progress(term, 1 / 3)
    _progress(term, 2 / 3)
    lift(GRASP, False)
    lift(LIFT, False)
    _progress(term, 0.0)


def test_hold_term_replaces_the_phase_predicate_as_the_hold(sequence, lift):
    """The grasp phase passes on ``cube_a`` lifted; what keeps it passed is ``cube_b`` lifted."""
    term = sequence(_phase(TRUE), _phase(_lifted(GRASP), hold_term=_lifted(LIFT)), _phase(FALSE))
    lift(GRASP, True)
    lift(LIFT, True)
    _progress(term, 1 / 3)
    _progress(term, 2 / 3)
    lift(GRASP, False)
    _progress(term, 2 / 3)  # the phase predicate failing is not the hold failing
    lift(GRASP, True)
    lift(LIFT, False)
    _progress(term, 2 / 3)  # reverted, and re-advanced in the same compute by the passing phase predicate
    lift(GRASP, False)
    _progress(term, 1 / 3)  # reverted, and nothing to re-advance on


def test_holds_revert_per_env(sequence, lift):
    term = sequence(_phase(TRUE), _phase(_lifted(GRASP), hold=True), _phase(FALSE))
    lift(GRASP, True)
    _progress(term, 1 / 3)
    _progress(term, 2 / 3)
    lift(GRASP, [False, True])
    _progress(term, 1 / 3, 2 / 3)


@pytest.mark.parametrize("hold_until", ["place", 3], ids=["by-name", "by-index"])
def test_hold_until_keeps_the_hold_through_the_named_phase(sequence, lift, hold_until):
    term = sequence(
        _phase(TRUE, name="reach"),
        _phase(_lifted(GRASP), name="grasp", hold=True, hold_until=hold_until),
        _phase(TRUE, name="lift"),
        _phase(_lifted(PLACE), name="place"),
    )
    lift(GRASP, True)
    lift(PLACE, False)
    for expected in (0.25, 0.5, 0.75):
        _progress(term, expected)
    lift(GRASP, False)
    _progress(term, 0.25)  # still inside the window at "place": reverted to "grasp"
    lift(GRASP, True)
    _progress(term, 0.5)
    _progress(term, 0.75)
    lift(PLACE, True)
    _progress(term, 1.0)


def test_hold_expires_once_its_until_phase_is_done(sequence, lift):
    term = sequence(
        _phase(_lifted(GRASP), name="a", hold=True, hold_until="c"),
        _phase(TRUE, name="b"),
        _phase(TRUE, name="c"),
        _phase(FALSE, name="d"),
    )
    lift(GRASP, True)
    _progress(term, 0.25)
    lift(GRASP, False)
    _progress(term, 0.0)  # inside the window
    lift(GRASP, True)
    for expected in (0.25, 0.5, 0.75):
        _progress(term, expected)
    lift(GRASP, False)
    _progress(term, 0.75)  # "c" is done: the hold no longer applies


def test_hold_without_until_lasts_to_the_task_end(sequence, lift):
    term = sequence(_phase(_lifted(GRASP), hold=True), _phase(FALSE))
    lift(GRASP, True)
    _progress(term, 0.5)
    lift(GRASP, False)
    _progress(term, 0.0)


@pytest.mark.parametrize(
    ("phases", "match"),
    [
        ([_phase(TRUE, hold=True, hold_until="nonexistent")], "not found in phase names"),
        ([_phase(TRUE, name="a"), _phase(TRUE, name="b", hold=True, hold_until="a")], "must be in"),
        (
            [_phase(TRUE, name="a", hold=True), _phase(TRUE, name="b", hold=True, hold_guard="nonexistent")],
            "hold_guard='nonexistent'",
        ),
        (
            [_phase(TRUE, name="a"), _phase(TRUE, name="b", hold=True, hold_guard="a"), _phase(TRUE, name="c")],
            "has no hold condition",
        ),
    ],
    ids=["until-unknown", "until-backward", "guard-unknown", "guard-without-hold"],
)
def test_a_bad_hold_reference_raises_at_build(sequence, phases, match):
    with pytest.raises(ValueError, match=match):
        sequence(*phases)


# ---------------------------------------------------------------------------
# Hold guards
# ---------------------------------------------------------------------------


@pytest.fixture
def pick_and_place(sequence, lift):
    """Phases reach, grasp (held through place), lift (held through place, guarded by grasp), place."""
    term = sequence(
        _phase(TRUE, name="reach"),
        _phase(_lifted(GRASP), name="grasp", hold=True, hold_until="place"),
        _phase(_lifted(LIFT), name="lift", hold=True, hold_until="place", hold_guard="grasp"),
        _phase(_lifted(PLACE), name="place"),
    )
    lift(GRASP, True)
    lift(LIFT, True)
    lift(PLACE, False)
    for expected in (0.25, 0.5, 0.75):
        _progress(term, expected)
    return term


def test_a_guarded_hold_is_suppressed_while_its_guard_holds(pick_and_place, lift):
    lift(LIFT, False)
    _progress(pick_and_place, 0.75)  # the grasp still holds, so a failing lift hold does not revert
    lift(PLACE, True)
    _progress(pick_and_place, 1.0)
    lift(GRASP, False)
    _progress(pick_and_place, 1.0)  # complete


def test_a_guarded_hold_is_enforced_once_its_guard_fails(pick_and_place, lift):
    lift(GRASP, False)
    lift(LIFT, False)
    _progress(pick_and_place, 0.25)  # both fail: back to the earliest, "grasp"


def test_the_guard_is_evaluated_per_env(pick_and_place, lift):
    """Both envs drop the lifted cube; only env 1 also loses the grasp, so only env 1 reverts."""
    lift(LIFT, False)
    lift(GRASP, [True, False])
    _progress(pick_and_place, 0.75, 0.25)


def test_a_guard_on_a_phase_without_a_hold_changes_nothing(sequence, lift):
    term = sequence(_phase(TRUE, name="a", hold=True), _phase(_lifted(GRASP), name="b", hold_guard="a"), _phase(TRUE))
    lift(GRASP, True)
    for expected in (1 / 3, 2 / 3):
        _progress(term, expected)
    lift(GRASP, False)
    _progress(term, 1.0)  # "b" is not held, so its failing predicate never reverts it
