"""The boolean metric combinators in ``eden/managers/terms/metrics/boolean.py``."""

from __future__ import annotations

import pytest
import torch

from eden.managers.metric_manager import MetricManager
from eden.managers.terms.metrics.boolean import MetricTermAND, MetricTermNOT, MetricTermOR
from eden.managers.terms.metrics.proximity import EeNearEntity
from eden.managers.terms.metrics.sequence import SequentialMetricTerm
from eden.options.managers.metrics import MetricManagerOptions, MetricTermOptions, PhaseOptions

pytestmark = pytest.mark.requires_genesis


def _term(name, **params):
    return MetricTermOptions(name=name, **({"params": params} if params else {}))


def _passes(value):
    return _term("always_true" if value else "always_false")


def _envs(*envs):
    return _term("env_specific", pass_envs=list(envs))


def _children(op_cls, children):
    """AND / OR take ``terms=[...]``; NOT takes one ``term=``."""
    return {"term": children[0]} if op_cls is MetricTermNOT else {"terms": children}


@pytest.mark.parametrize(
    ("op_cls", "inputs", "expected"),
    [
        (MetricTermAND, (True, True), 1.0),
        (MetricTermAND, (True, False), 0.0),
        (MetricTermAND, (False, False), 0.0),
        (MetricTermOR, (True, True), 1.0),
        (MetricTermOR, (True, False), 1.0),
        (MetricTermOR, (False, False), 0.0),
        (MetricTermNOT, (True,), 0.0),
        (MetricTermNOT, (False,), 1.0),
    ],
    ids=lambda v: v.__name__ if isinstance(v, type) else str(v),
)
def test_truth_table(panda_env, make_term, op_cls, inputs, expected):
    term = make_term(op_cls, panda_env, **_children(op_cls, [_passes(v) for v in inputs]))
    torch.testing.assert_close(term.compute(), torch.full((2,), expected))


@pytest.mark.parametrize(
    ("op_cls", "children", "expected"),
    [
        (MetricTermAND, [_envs(0), _envs(0, 1)], [1.0, 0.0]),
        (MetricTermOR, [_envs(0), _envs()], [1.0, 0.0]),
        (MetricTermNOT, [_envs(1)], [1.0, 0.0]),
    ],
    ids=["and", "or", "not"],
)
def test_combinators_reduce_per_env(panda_env, make_term, op_cls, children, expected):
    torch.testing.assert_close(
        make_term(op_cls, panda_env, **_children(op_cls, children)).compute(), torch.tensor(expected)
    )


@pytest.mark.parametrize(
    ("op_cls", "options", "match"),
    [
        (MetricTermAND, {"terms": []}, "requires at least one term"),
        (MetricTermOR, {"terms": []}, "requires at least one term"),
        (MetricTermNOT, {}, "requires a 'term'"),
    ],
    ids=["and", "or", "not"],
)
def test_a_combinator_without_children_raises_at_build(panda_env, make_term, op_cls, options, match):
    """No children would otherwise reduce to a silent constant."""
    with pytest.raises(ValueError, match=match):
        make_term(op_cls, panda_env, **options)


def test_combinators_nest(panda_env, make_term):
    either = [MetricTermOR.configure(terms=[_passes(True), _passes(False)])] * 2
    torch.testing.assert_close(make_term(MetricTermAND, panda_env, terms=either).compute(), torch.ones(2))
    not_both = MetricTermAND.configure(terms=[_passes(True), _passes(False)])
    torch.testing.assert_close(make_term(MetricTermNOT, panda_env, term=not_both).compute(), torch.ones(2))


def test_and_over_a_class_term(panda_env, make_term, restore_state):
    """Env 0's cube sits between the fingers, env 1's is 0.2 m above them."""
    robot, cube = panda_env.entities["robot"], restore_state(panda_env.entities["cube_a"])
    _, fingers = robot.find_named_links_idx_local(["left_finger", "right_finger"])
    cube.set_pos(
        robot.get_links_pos(ls_idx_local=fingers).mean(dim=1) + torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.2]])
    )
    near = EeNearEntity.configure(
        robot_name="robot", ee_link_names=["left_finger", "right_finger"], entity_name="cube_a"
    )
    term = make_term(MetricTermAND, panda_env, terms=[near, _passes(True)])
    torch.testing.assert_close(term.compute(), torch.tensor([1.0, 0.0]))


def test_success_through_the_metric_manager(panda_env, install_manager):
    options = MetricManagerOptions(combined=MetricTermAND.configure(terms=[_envs(0), _passes(True)]))
    manager = install_manager(panda_env, "metric_manager", MetricManager(panda_env, options))
    manager.compute()
    assert manager.success_buf.tolist() == [True, False]


def test_reset_reaches_the_children(panda_env, make_term):
    """A stateful child (a two-phase sequence) restarts in exactly the envs the combinator resets."""
    sequence = SequentialMetricTerm.configure(phases=[PhaseOptions(term=_passes(True))] * 2)
    term = make_term(MetricTermAND, panda_env, terms=[sequence, _passes(True)])
    term.compute()
    torch.testing.assert_close(term.compute(), torch.ones(2))
    term.reset(envs_idx=torch.tensor([0]))
    torch.testing.assert_close(term.compute(), torch.tensor([0.5, 1.0]))
