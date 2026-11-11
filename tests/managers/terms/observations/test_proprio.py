"""The ``Dofs*`` observation terms in ``eden/managers/terms/observations/proprio.py``.

Each term reads one per-DOF quantity of an entity, for every controlled DOF or a ``dofs_name`` /
``terms_idx`` selection resolved at build. The expected values come from the entity's own full-width
getters, indexed by position in its ``dofs_name``.
"""

from __future__ import annotations

import pytest
import torch

from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY
from eden.managers.terms.observations.proprio import DofsControlForce, DofsForce, DofsPos, DofsVel
from eden.options import ObservationTermOptions

pytestmark = pytest.mark.requires_genesis

_ARM = [f"joint{i}" for i in range(1, 8)]
_ARM_Q = [0.3, -0.4, 0.2, -2.0, 0.1, 1.5, 0.6]
_ARM_QD = [0.05, -0.1, 0.15, -0.2, 0.25, -0.3, 0.35]

# Each term and the full-width entity getter it selects from.
_TERMS = [
    pytest.param(DofsPos, "get_dofs_pos", id="DofsPos"),
    pytest.param(DofsVel, "get_dofs_vel", id="DofsVel"),
    pytest.param(DofsForce, "get_dofs_force", id="DofsForce"),
    pytest.param(DofsControlForce, "get_dofs_control_force", id="DofsControlForce"),
]


@pytest.fixture
def posed_panda(panda_env, restore_state):
    """``panda_env`` with the arm at a fixed, non-trivial position and velocity."""
    robot = restore_state(panda_env.entities["robot"])
    pos, vel = robot.get_dofs_pos().clone(), robot.get_dofs_vel().clone()
    for name, q, qd in zip(_ARM, _ARM_Q, _ARM_QD):
        pos[:, robot.dofs_name.index(name)] = q
        vel[:, robot.dofs_name.index(name)] = qd
    robot.set_dofs_pos(pos)
    robot.set_dofs_vel(vel)
    return panda_env


def _columns(robot, names):
    return [robot.dofs_name.index(name) for name in names]


@pytest.mark.parametrize(("cls", "getter"), _TERMS)
def test_with_no_selection_a_term_reads_every_controlled_dof(posed_panda, make_term, cls, getter):
    robot = posed_panda.entities["robot"]
    torch.testing.assert_close(make_term(cls, posed_panda, entity_name="robot").compute(), getattr(robot, getter)())


@pytest.mark.parametrize(("cls", "getter"), _TERMS)
def test_dofs_name_selects_those_dofs_in_the_declared_order(posed_panda, make_term, cls, getter):
    robot = posed_panda.entities["robot"]
    order = list(reversed(_ARM))
    out = make_term(cls, posed_panda, entity_name="robot", dofs_name=order).compute()
    torch.testing.assert_close(out, getattr(robot, getter)()[:, _columns(robot, order)])


def test_dofs_pos_reads_the_set_positions_and_offsets_from_the_same_dofs_defaults(posed_panda, make_term):
    robot = posed_panda.entities["robot"]
    order = list(reversed(_ARM))
    torch.testing.assert_close(
        make_term(DofsPos, posed_panda, entity_name="robot", dofs_name=order).compute(),
        torch.tensor([list(reversed(_ARM_Q))] * posed_panda.num_envs),
    )
    offset = make_term(DofsPos, posed_panda, entity_name="robot", dofs_name=order, offset_from_default=True).compute()
    columns = _columns(robot, order)
    expected = torch.tensor([list(reversed(_ARM_Q))]) - robot.default_dofs_pos[:, columns]
    torch.testing.assert_close(offset, expected.expand_as(offset))


def test_terms_idx_selects_positions_in_dofs_name_including_negative_ones(posed_panda, make_term):
    robot = posed_panda.entities["robot"]
    by_position = make_term(DofsVel, posed_panda, entity_name="robot", terms_idx=[2, 0, -1]).compute()
    by_name = make_term(
        DofsVel,
        posed_panda,
        entity_name="robot",
        dofs_name=[robot.dofs_name[2], robot.dofs_name[0], robot.dofs_name[-1]],
    ).compute()
    torch.testing.assert_close(by_position, by_name)


@pytest.mark.parametrize(
    ("params", "match"),
    [
        pytest.param(dict(dofs_name=["joint99"]), "joint99", id="unknown-name"),
        pytest.param(dict(dofs_name=[]), r"dofs_name=\[\] selects no DOF", id="empty-names"),
        pytest.param(dict(dofs_name=["joint1"], terms_idx=[0]), "Keep dofs_name and drop terms_idx", id="both"),
        pytest.param(dict(terms_idx=[0, 99]), r"positions \[99\] are out of range for its 9 DOFs", id="out-of-range"),
        pytest.param(dict(terms_idx=[]), "it selects no DOF", id="empty-positions"),
    ],
)
def test_a_bad_selection_fails_at_build(panda_env, make_term, params, match):
    with pytest.raises(ValueError, match=match):
        make_term(DofsPos, panda_env, entity_name="robot", **params)


def test_a_config_saved_from_the_function_term_loads_and_reads_the_same(posed_panda):
    """``dofs_pos`` was a function term: a saved config carries ``params``, which load as the class's fields."""
    saved = ObservationTermOptions(
        name="dofs_pos", params={"entity_name": "robot", "offset_from_default": True, "terms_idx": [0, 1]}
    )
    assert (saved.params, saved.entity_name, saved.offset_from_default, saved.terms_idx) == (
        None,
        "robot",
        True,
        [0, 1],
    )
    term = OBSERVATION_TERM_REGISTRY.get("dofs_pos")(env=posed_panda, options=saved)
    term.build()
    robot = posed_panda.entities["robot"]
    torch.testing.assert_close(term.compute(), (robot.get_dofs_pos() - robot.default_dofs_pos)[:, :2])


@pytest.mark.parametrize(
    ("name", "params", "getter"),
    [
        pytest.param("dofs_vel", {"entity_name": "robot", "terms_idx": [1]}, "get_dofs_vel", id="dofs_vel"),
        pytest.param("dofs_force", {"entity_name": "robot"}, "get_dofs_force", id="dofs_force"),
        pytest.param("dofs_control_force", {"entity_name": "robot"}, "get_dofs_control_force", id="dofs_control_force"),
    ],
)
def test_each_terms_function_era_params_load_as_fields(posed_panda, name, params, getter):
    saved = ObservationTermOptions(name=name, params=params)
    assert saved.params is None
    term = OBSERVATION_TERM_REGISTRY.get(name)(env=posed_panda, options=saved)
    term.build()
    expected = getattr(posed_panda.entities["robot"], getter)()
    torch.testing.assert_close(term.compute(), expected[:, params["terms_idx"]] if "terms_idx" in params else expected)


@pytest.mark.parametrize("name", ["dofs_force", "dofs_control_force"])
def test_a_force_term_rejects_terms_idx_in_params_which_its_function_never_took(name):
    with pytest.raises(ValueError, match="'params' is only used by function-based terms"):
        ObservationTermOptions(name=name, params={"entity_name": "robot", "terms_idx": [0]})


def test_dofs_name_listing_every_dof_in_order_reads_the_full_width(posed_panda, make_term):
    robot = posed_panda.entities["robot"]
    every = make_term(DofsVel, posed_panda, entity_name="robot", dofs_name=list(robot.dofs_name)).compute()
    torch.testing.assert_close(every, robot.get_dofs_vel())


def test_a_term_reads_the_entity_swapped_in_after_build(posed_panda, make_term):
    """``compute()`` re-resolves the entity, as a deployment hot-swap of ``env.entities[name]`` needs."""
    term = make_term(DofsPos, posed_panda, entity_name="robot", dofs_name=["joint2", "joint1"])
    robot = posed_panda.entities["robot"]

    class _Hardware:  # the deployed robot: same DOF layout, different readings
        default_dofs_pos = robot.default_dofs_pos

        @staticmethod
        def get_dofs_pos():
            return torch.arange(len(robot.dofs_name), dtype=torch.float32).repeat(posed_panda.num_envs, 1)

    posed_panda.entities["robot"] = _Hardware()
    try:
        out = term.compute()
    finally:
        posed_panda.entities["robot"] = robot
    torch.testing.assert_close(out, torch.tensor([[1.0, 0.0]] * posed_panda.num_envs))


@pytest.fixture(scope="module")
def leap_env():
    """A LEAP hand: its declared ``dofs_name`` order differs from Genesis' kinematic-tree order."""
    import eden as en
    from eden.options import EnvOptions
    from eden.options.robots.leap_hand import LeapHand_R
    from eden.utils.configs import EdenConfig
    from tests.support.envs import build_env

    return build_env(EdenConfig(env_options=EnvOptions(num_envs=1), scene_options=en.scenes.Empty(robot=LeapHand_R())))


def test_dofs_name_reads_in_dofs_name_order_when_genesis_order_differs(leap_env, make_term, restore_state):
    hand = restore_state(leap_env.entities["robot"])
    _, genesis_idx = hand.find_named_dofs_idx_local(list(hand.dofs_name), preserve_order=True)
    assert list(genesis_idx) != list(range(len(genesis_idx))), "the LEAP hand no longer reorders its DOFs"
    values = torch.linspace(0.1, 0.1 * len(hand.dofs_name), len(hand.dofs_name)).unsqueeze(0)
    hand.set_dofs_pos(values)  # dofs_name order
    names = list(hand.dofs_name[:4])
    torch.testing.assert_close(
        make_term(DofsPos, leap_env, entity_name="robot", dofs_name=names).compute(), values[:, :4]
    )
