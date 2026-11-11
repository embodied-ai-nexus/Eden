"""The weld action terms in ``eden/managers/terms/actions/welding_actions.py``.

A fixed cube sits between the Panda's open finger pads. Env 0 closes its fingers into the cube and
steps once, so its pads carry the contact force a grasp does; env 1 stays open and touches nothing.
"""

from __future__ import annotations

import pytest
import torch

from eden.managers.terms.actions.welding_actions import ParallelJawWelding, SuctionCupWelding
from tests.managers.terms.conftest import panda_options
from tests.support.envs import build_env

pytestmark = pytest.mark.requires_genesis

PADS = {"left_gripper_link_name": "left_gripper_pad", "right_gripper_link_name": "right_gripper_pad"}
ON, OFF = torch.ones(2, 1, dtype=torch.bool), torch.zeros(2, 1, dtype=torch.bool)


@pytest.fixture(scope="module")
def env():
    from eden.options import BoxOptions, EnvOptions, SceneOptions
    from eden.utils.configs import EdenConfig

    env = build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=2),
            scene_options=SceneOptions(
                robot=panda_options(),
                # Centred between the pads, which sit at y = ±0.04 when open.
                cube=BoxOptions(size=(0.03, 0.03, 0.03), default_root_pos=(0.5175, 0.0, 0.2551), fixed=True),
            ),
        )
    )
    _close_env_0_into_the_cube(env)
    return env


def _close_env_0_into_the_cube(env):
    robot = env.entities["robot"]
    pos = robot.get_dofs_pos().clone()
    pos[0, 7:] = 0.01  # env 0's fingers 5 mm into the cube
    robot.set_dofs_pos(pos)
    env.scene.step()


def _welded_pairs(env):
    """The weld constraints the solver holds per env, as ``(link_a, link_b)`` sets."""
    welds = env.rigid_solver.get_weld_constraints(as_tensor=True, to_torch=True)
    return [
        {(a, b) for a, b in zip(welds["link_a"][e].tolist(), welds["link_b"][e].tolist()) if a >= 0}
        for e in range(env.num_envs)
    ]


@pytest.fixture
def weld(env):
    """``weld(cls, **params)``: a built weld term; its welds are released after the test."""
    terms = []

    def build(cls, **params):
        term = cls(env, cls.configure(entity_name="robot", **params))
        term.build()
        terms.append(term)
        return term

    yield build
    for term in terms:
        term.reset()
    assert _welded_pairs(env) == [set(), set()]


def test_parallel_jaw_welds_the_object_only_where_both_pads_squeeze_it(env, weld):
    """The grasp check broadcasts over a contact count that is not the env count."""
    hand, cube = env.entities["robot"].get_link("hand").idx, env.entities["cube"].links[0].idx
    term = weld(ParallelJawWelding, ee_link_name="hand", **PADS)
    term.compute(ON)
    term.apply_actions()
    assert _welded_pairs(env) == [{(cube, hand)}, set()]
    term.compute(OFF)  # releasing the grip deletes the weld
    assert _welded_pairs(env) == [set(), set()]


def test_suction_welds_an_object_touching_the_cup(env, weld):
    cube = env.entities["cube"].links[0].idx
    cup = env.entities["robot"].get_link("left_gripper_pad").idx
    term = weld(SuctionCupWelding, ee_link_name="left_gripper_pad")
    term.compute(ON)
    term.apply_actions()
    assert _welded_pairs(env) == [{(cube, cup)}, set()]


def test_suction_never_welds_an_ignored_entity(env, weld):
    """Regression: an ignore check comparing 0-d tensors against a set of ints never matches."""
    term = weld(SuctionCupWelding, ee_link_name="left_gripper_pad", ignore_entities_name=["cube"])
    term.compute(ON)
    term.apply_actions()
    assert _welded_pairs(env) == [set(), set()]


def test_parallel_jaw_never_welds_an_ignored_link(env, weld):
    term = weld(ParallelJawWelding, ee_link_name="hand", ignore_links_idx=[env.entities["cube"].links[0].idx], **PADS)
    term.compute(ON)
    term.apply_actions()
    assert _welded_pairs(env) == [set(), set()]


@pytest.mark.parametrize("threshold", [-1.0, float("nan")], ids=["negative", "nan"])
def test_a_grasp_force_threshold_below_zero_raises_at_build(env, threshold):
    """|projected force| > threshold: a negative threshold would weld on any pad contact, NaN on none."""
    term = ParallelJawWelding(
        env,
        ParallelJawWelding.configure(entity_name="robot", ee_link_name="hand", grasp_force_threshold=threshold, **PADS),
    )
    with pytest.raises(ValueError, match=r"grasp_force_threshold=.* must be >= 0"):
        term.build()


def test_a_welded_env_does_not_weld_again_while_the_action_stays_on(env, weld):
    hand, cube = env.entities["robot"].get_link("hand").idx, env.entities["cube"].links[0].idx
    term = weld(ParallelJawWelding, ee_link_name="hand", **PADS)
    for _ in range(2):
        term.compute(ON)
        term.apply_actions()
    assert _welded_pairs(env) == [{(cube, hand)}, set()]


# ---------------------------------------------------------------------------
# Which link a parallel jaw welds when the pads touch more than one: each scene is its own env, built
# like ``env`` above (env 0's fingers 5 mm closed, env 1 open), so only env 0 has contacts.
# ---------------------------------------------------------------------------


def _squeeze_scene(**objects):
    from eden.options import EnvOptions, SceneOptions
    from eden.utils.configs import EdenConfig

    env = build_env(
        EdenConfig(env_options=EnvOptions(num_envs=2), scene_options=SceneOptions(**objects, robot=panda_options()))
    )
    robot = env.entities["robot"]
    pos = robot.get_dofs_pos().clone()
    pos[0, 7:] = 0.01
    robot.set_dofs_pos(pos)
    env.scene.step()
    return env


def _pad_partners(env) -> dict[str, set[str]]:
    """Env 0: the entities each pad touches (the scene's precondition, checked before the weld is)."""
    robot, contacts = env.entities["robot"], env.entities["robot"].get_contacts(exclude_self_contact=True)
    a, b, valid = contacts["link_a"][0], contacts["link_b"][0], contacts["valid_mask"][0]
    owner = {e.links[0].idx: name for name, e in env.entities.items() if name != "robot"}
    partners = {}
    for pad in PADS.values():
        idx = robot.get_link(pad).idx
        touching = torch.cat([b[valid & (a == idx)], a[valid & (b == idx)]]).unique().tolist()
        partners[pad] = {owner.get(link, f"link {link}") for link in touching}
    return partners


def _apply_parallel_jaw(env):
    term = ParallelJawWelding(env, ParallelJawWelding.configure(entity_name="robot", ee_link_name="hand", **PADS))
    term.build()
    term.compute(ON)
    term.apply_actions()
    return term


@pytest.fixture(scope="module")
def bystander_env():
    """The squeezed cube, plus a small box under the left pad that only that pad touches. The bystander is
    added first, so its link index is lower and the candidate loop reaches it before the cube.
    """
    from eden.options import BoxOptions

    return _squeeze_scene(
        # Centred under the closed left pad (y -0.021..-0.0115, bottom z 0.2464), its top 1.5 mm into it.
        bystander=BoxOptions(size=(0.008, 0.004, 0.008), default_root_pos=(0.5174, -0.01625, 0.2439), fixed=True),
        cube=BoxOptions(size=(0.03, 0.03, 0.03), default_root_pos=(0.5175, 0.0, 0.2551), fixed=True),
    )


def test_parallel_jaw_welds_the_squeezed_link_not_a_bystander_on_one_pad(bystander_env):
    """The pads' forces used to be summed over every link they touched, and the first link touching *a* pad
    was welded: the bystander, with the squeezed cube left free. The grasp is now checked per link.
    """
    env = bystander_env
    assert _pad_partners(env) == {"left_gripper_pad": {"bystander", "cube"}, "right_gripper_pad": {"cube"}}
    term = _apply_parallel_jaw(env)
    hand, cube = env.entities["robot"].get_link("hand").idx, env.entities["cube"].links[0].idx
    assert _welded_pairs(env) == [{(cube, hand)}, set()]
    term.reset()


@pytest.fixture(scope="module")
def split_env():
    """The cube cut into two boxes with a 6 mm gap: each pad squeezes a different one."""
    from eden.options import BoxOptions

    return _squeeze_scene(
        box_a=BoxOptions(size=(0.03, 0.012, 0.03), default_root_pos=(0.5175, 0.009, 0.2551), fixed=True),
        box_b=BoxOptions(size=(0.03, 0.012, 0.03), default_root_pos=(0.5175, -0.009, 0.2551), fixed=True),
    )


def test_parallel_jaw_welds_nothing_when_each_pad_presses_a_different_link(split_env):
    env = split_env
    partners = _pad_partners(env)
    assert sorted(map(len, partners.values())) == [1, 1] and set.union(*partners.values()) == {"box_a", "box_b"}
    term = _apply_parallel_jaw(env)
    assert _welded_pairs(env) == [set(), set()]
    term.reset()


@pytest.fixture(scope="module")
def stacked_env():
    """Two boxes stacked between the pads, each squeezed by both: an env holds one weld, so one is welded."""
    from eden.options import BoxOptions

    return _squeeze_scene(
        upper=BoxOptions(size=(0.03, 0.03, 0.012), default_root_pos=(0.5175, 0.0, 0.2610), fixed=True),
        lower=BoxOptions(size=(0.03, 0.03, 0.012), default_root_pos=(0.5175, 0.0, 0.2490), fixed=True),
    )


def test_parallel_jaw_welds_the_lowest_index_link_when_several_are_squeezed(stacked_env):
    env = stacked_env
    assert _pad_partners(env) == {"left_gripper_pad": {"upper", "lower"}, "right_gripper_pad": {"upper", "lower"}}
    term = _apply_parallel_jaw(env)
    hand, upper = env.entities["robot"].get_link("hand").idx, env.entities["upper"].links[0].idx
    assert upper < env.entities["lower"].links[0].idx
    assert _welded_pairs(env) == [{(upper, hand)}, set()]
    term.reset()


@pytest.mark.parametrize("bad_idx", [-1, 10_000])
def test_an_ignore_link_index_outside_the_solver_raises_at_build(env, bad_idx):
    term = SuctionCupWelding(
        env,
        SuctionCupWelding.configure(entity_name="robot", ee_link_name="left_gripper_pad", ignore_links_idx=[bad_idx]),
    )
    with pytest.raises(ValueError, match=rf"ignore_links_idx \[{bad_idx}\] are not rigid-solver link indices"):
        term.build()


@pytest.mark.parametrize(
    "envs_idx",
    [
        pytest.param(torch.tensor([True, False]), id="bool-mask"),
        pytest.param(torch.tensor([0], dtype=torch.int32), id="index-tensor"),
        pytest.param(slice(0, 1), id="slice"),
    ],
)
def test_reset_releases_the_weld_of_the_selected_envs(env, weld, envs_idx):
    """An index tensor is what ``sanitize_envs_idx`` makes of a manual ``env.reset(envs_idx=[0])``.

    A ``reset`` that ANDs it bitwise into the per-env "welded here" row releases nothing for index 0.
    """
    term = weld(ParallelJawWelding, ee_link_name="hand", **PADS)
    term.compute(ON)
    term.apply_actions()
    term.reset(envs_idx)
    assert _welded_pairs(env) == [set(), set()]


@pytest.mark.parametrize(
    "envs_idx",
    [
        pytest.param(torch.tensor([False, True]), id="bool-mask"),
        pytest.param(torch.tensor([1], dtype=torch.int32), id="index-tensor"),
    ],
)
def test_reset_keeps_the_weld_of_the_envs_it_does_not_select(env, weld, envs_idx):
    """Regression: index 1 ANDed into the welded row gave ``[1, 0]``, read back as envs 0 and 1 -- env 0's grasp."""
    hand, cube = env.entities["robot"].get_link("hand").idx, env.entities["cube"].links[0].idx
    term = weld(ParallelJawWelding, ee_link_name="hand", **PADS)
    term.compute(ON)
    term.apply_actions()
    term.reset(envs_idx)
    assert _welded_pairs(env) == [{(cube, hand)}, set()]


@pytest.mark.parametrize(
    ("cls", "params"),
    [
        pytest.param(ParallelJawWelding, {"ee_link_name": "hand", **PADS}, id="parallel-jaw"),
        pytest.param(SuctionCupWelding, {"ee_link_name": "left_gripper_pad"}, id="suction"),
    ],
)
def test_an_env_reset_while_welded_welds_again_in_the_next_episode(env, weld, cls, params):
    """``env.reset`` resets the scene before the action terms, and the scene reset already drops the weld.

    The term's ``reset`` then found no constraint to delete and left the env marked welded, so it never
    welded again: every episode after one that ended holding an object ran without welds.
    """
    ee, cube = env.entities["robot"].get_link(params["ee_link_name"]).idx, env.entities["cube"].links[0].idx
    term = weld(cls, **params)
    term.compute(ON)
    term.apply_actions()
    assert _welded_pairs(env) == [{(cube, ee)}, set()]

    envs_idx = torch.tensor([0])
    env.reset(envs_idx)  # the scene reset drops env 0's weld (and reopens its fingers) ...
    assert _welded_pairs(env) == [set(), set()]
    term.reset(envs_idx)  # ... before env.reset's manager loop resets the action terms

    _close_env_0_into_the_cube(env)
    term.compute(ON)
    term.apply_actions()
    assert _welded_pairs(env) == [{(cube, ee)}, set()]


@pytest.fixture(scope="module")
def single_env():
    """A Franka hand alone, in one env."""
    from eden.options import EnvOptions, SceneOptions
    from eden.options.robots import FrankaHand
    from eden.utils.configs import EdenConfig

    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=1),
            scene_options=SceneOptions(hand=FrankaHand(default_root_pos=(0.0, 0.0, 0.5))),
        )
    )


@pytest.mark.parametrize("action", [torch.tensor([[True]]), torch.tensor([[0.5]])], ids=["bool", "float"])
def test_a_single_env_action_stays_one_dimensional(single_env, action):
    """``squeeze(-1)`` keeps a ``(1, 1)`` action as shape ``(1,)``, not a 0-d scalar."""
    term = SuctionCupWelding(single_env, SuctionCupWelding.configure(entity_name="hand", ee_link_name="hand"))
    term.build()
    term.compute(action)
    assert term.processed_action.tolist() == [True]
