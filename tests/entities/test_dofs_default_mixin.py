"""``DofDefaultsMixin``: DOFs selected by name into both index spaces, and the property tensors read by name."""

from __future__ import annotations

import pytest
import torch

from eden.entities.rigid import RigidEntity
from eden.options import BoxOptions, EntityOptions, EnvOptions, SceneOptions
from eden.options.entities import RESOLVED_DOF_TENSOR_FIELD_NAMES, RobotOptions
from eden.options.materials import KinematicMaterialOptions
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env

# A floating base and a chain a -> b -> c, declared in that (tree) order. ``dofs_name`` lists the
# joints as c, a, b, so the two index spaces disagree in *both* ways a crossing can go wrong: the
# free base shifts every local index by six, and the declaration order permutes them.
#   column: c=0  a=1  b=2        local: c=8  a=6  b=7
_CHAIN_MJCF = """<mujoco model="chain"><compiler angle="radian" autolimits="true"/><worldbody>
  <body name="torso" pos="0 0 0.6"><freejoint name="root"/>
    <geom type="box" size="0.08 0.08 0.04" mass="1"/>
    <body name="a" pos="0.1 0 0"><joint name="a_joint" axis="0 1 0" range="-1.0 1.0"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.1 0 0" mass="0.1"/>
      <body name="b" pos="0.1 0 0"><joint name="b_joint" axis="0 1 0" range="-0.5 0.5"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0.1 0 0" mass="0.1"/>
        <body name="c" pos="0.1 0 0"><joint name="c_joint" axis="0 1 0" range="-2.0 0.0"/>
          <geom type="capsule" size="0.02" fromto="0 0 0 0.1 0 0" mass="0.1"/></body></body></body></body>
</worldbody></mujoco>"""
_CHAIN_DOFS = ["c_joint", "a_joint", "b_joint"]
_POS = {"a_joint": 0.1, "b_joint": 0.2, "c_joint": -0.3}
_KP = {"a_joint": 11.0, "b_joint": 22.0, "c_joint": 33.0}
_RANGE = {"a_joint": (-1.0, 1.0), "b_joint": (-0.5, 0.5), "c_joint": (-2.0, 0.0)}

# Fixed base, articulated but not actuated (so no gains), with a three-DOF ball joint first.
_BALL_MJCF = """<mujoco model="ball"><compiler angle="radian" autolimits="true"/><worldbody>
  <body name="anchor" pos="1 1 0.4"><geom type="box" size="0.04 0.04 0.02" mass="1"/>
    <body name="socket" pos="0 0 0.05"><joint name="ball_joint" type="ball"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.08 0 0" mass="0.1"/>
      <body name="spindle" pos="0.08 0 0"><joint name="spin_joint" axis="0 0 1" range="-0.4 0.4"/>
        <geom type="cylinder" size="0.02 0.01" mass="0.1"/></body></body></body>
</worldbody></mujoco>"""

# Joint names that read as patterns: ``pin.1`` is also a regex matching ``pin_1``, ``arm(2)`` one that
# does not match itself. Columns follow declaration order: pin.1=0, pin_1=1, arm(2)=2.
_QUIRKY_MJCF = """<mujoco model="quirky"><compiler angle="radian" autolimits="true"/><worldbody>
  <body name="post" pos="-1 1 0.4"><geom type="box" size="0.04 0.04 0.02" mass="1"/>
    <body name="p1" pos="0 0 0.05"><joint name="pin.1" axis="0 0 1" range="-1 1"/>
      <geom type="cylinder" size="0.02 0.01" mass="0.1"/>
      <body name="p2" pos="0 0 0.03"><joint name="pin_1" axis="0 0 1" range="-1 1"/>
        <geom type="cylinder" size="0.02 0.01" mass="0.1"/>
        <body name="p3" pos="0 0 0.03"><joint name="arm(2)" axis="0 0 1" range="-1 1"/>
          <geom type="cylinder" size="0.02 0.01" mass="0.1"/></body></body></body></body>
</worldbody></mujoco>"""

NUM_ENVS = 2


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("dof_defaults")
    (root / "chain.xml").write_text(_CHAIN_MJCF)
    (root / "ball.xml").write_text(_BALL_MJCF)
    (root / "quirky.xml").write_text(_QUIRKY_MJCF)
    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=NUM_ENVS),
            scene_options=SceneOptions(
                chain=RobotOptions(
                    file=str(root / "chain.xml"),
                    dofs_name=_CHAIN_DOFS,
                    default_dofs_pos=_POS,
                    default_dofs_kp=_KP,
                    default_dofs_kd={"*": 1.0},
                    default_root_pos=(0.0, 0.0, 0.6),
                ),
                ball=EntityOptions(
                    file=str(root / "ball.xml"),
                    is_fixed_base=True,
                    is_articulated=True,
                    dofs_name=["ball_joint", "spin_joint"],
                ),
                ghost=EntityOptions(
                    file=str(root / "chain.xml"),
                    is_articulated=True,
                    dofs_name=_CHAIN_DOFS,
                    default_root_pos=(2.0, 0.0, 0.6),
                    material=KinematicMaterialOptions(),
                ),
                quirky=EntityOptions(
                    file=str(root / "quirky.xml"),
                    is_fixed_base=True,
                    is_articulated=True,
                    # Spelled out: read as patterns, ``pin.1`` would also select ``pin_1`` (a multiple match)
                    # and ``arm(2)`` would select nothing, so this config would not build.
                    dofs_name=["pin.1", "pin_1", "arm(2)"],
                ),
                # The same joints with ``pin.1`` passive: as a regex, its exact name would select ``pin_1``.
                passive_pin=EntityOptions(
                    file=str(root / "quirky.xml"),
                    is_fixed_base=True,
                    is_articulated=True,
                    dofs_name=["pin_1", "arm(2)"],
                    default_root_pos=(-1.0, 2.0, 0.4),
                ),
                crate=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(-1.0, 0.0, 0.1)),
            ),
        )
    )


@pytest.fixture
def chain(env):
    return env.entities["chain"]


@pytest.fixture
def drifted(chain):
    """Env 1's ``a_joint`` default moved to 0.5, as ``RandomizeStartupDofsPosBias`` would; restored after."""
    column = chain.find_named_dofs("a_joint").columns.item()
    original = chain.default_dofs_pos[1, column].item()
    chain.default_dofs_pos[1, column] = 0.5
    yield chain
    chain.default_dofs_pos[1, column] = original


def _rows(values: list[float], device) -> torch.Tensor:
    """``values`` as every env's row: what a getter returns for a property every env shares."""
    return torch.tensor(values, device=device).expand(NUM_ENVS, -1)


def _spaces(dofs) -> tuple[tuple[str, ...], list[int], list[int]]:
    """A selection's three fields as plain values: names, columns, local indices."""
    return dofs.names, dofs.columns.tolist(), dofs.dofs_idx_local.tolist()


def test_the_fixture_separates_the_two_index_spaces(chain):
    """Guards every test below: a getter that crossed the spaces could not pass by coincidence."""
    assert chain.find_named_dofs_idx_local(_CHAIN_DOFS)[1] == [8, 6, 7]
    assert _spaces(chain.find_named_dofs(_CHAIN_DOFS)) == (tuple(_CHAIN_DOFS), [0, 1, 2], [8, 6, 7])


def test_a_selection_follows_the_query_in_both_spaces(chain):
    dofs = chain.find_named_dofs(["b_joint", "c_joint"])
    assert _spaces(dofs) == (("b_joint", "c_joint"), [2, 0], [7, 8])
    assert len(dofs) == 2
    expected = torch.tensor([_POS["b_joint"], _POS["c_joint"]], device=chain.device)
    torch.testing.assert_close(chain.default_dofs_pos[0, dofs.columns], expected)


def test_no_names_select_every_dof_in_column_order(chain):
    assert _spaces(chain.find_named_dofs()) == (tuple(_CHAIN_DOFS), [0, 1, 2], [8, 6, 7])


def test_a_selection_pairs_a_full_width_read_with_a_genesis_level_one(chain):
    """Its columns address ``get_dofs_*()`` with no ``dofs_idx_local``; its local indices go to the getter."""
    dofs = chain.find_named_dofs(["b_joint", "a_joint"])
    torch.testing.assert_close(
        chain.get_dofs_limit()[1][:, dofs.columns], chain.get_dofs_limit(dofs_idx_local=dofs.dofs_idx_local)[1]
    )


def test_a_getter_takes_a_selection_in_place_of_names(chain):
    dofs = chain.find_named_dofs(["b_joint", "c_joint"])
    torch.testing.assert_close(chain.get_default_dofs_pos(dofs), chain.get_default_dofs_pos(["b_joint", "c_joint"]))


def test_a_getter_selects_by_name_in_query_order(chain):
    pos = chain.get_default_dofs_pos(["b_joint", "c_joint"])
    torch.testing.assert_close(pos, _rows([_POS["b_joint"], _POS["c_joint"]], chain.device))


def test_no_selection_reads_every_dof_in_dofs_name_order(chain):
    torch.testing.assert_close(chain.get_default_dofs_kp(), _rows([_KP[n] for n in _CHAIN_DOFS], chain.device))


def test_globs_and_regexes_select_like_every_other_name_resolution(chain):
    torch.testing.assert_close(chain.get_default_dofs_kp("?_joint"), _rows([_KP[n] for n in _CHAIN_DOFS], chain.device))
    torch.testing.assert_close(
        chain.get_default_dofs_kp("[ab]_joint"), _rows([_KP["a_joint"], _KP["b_joint"]], chain.device)
    )
    torch.testing.assert_close(
        chain.get_default_dofs_kp("(c|a)_joint"), _rows([_KP["c_joint"], _KP["a_joint"]], chain.device)
    )


def test_a_range_valued_property_keeps_its_trailing_pair(chain):
    limits = chain.get_default_dofs_pos_limits(["a_joint", "c_joint"])
    expected = torch.tensor([_RANGE["a_joint"], _RANGE["c_joint"]], device=chain.device).expand(NUM_ENVS, -1, -1)
    torch.testing.assert_close(limits, expected)
    soft = chain.get_soft_dofs_pos_limits(["a_joint"])
    torch.testing.assert_close(soft, torch.tensor([[[-0.9, 0.9]]] * NUM_ENVS, device=chain.device))


def test_one_shape_whatever_the_stored_layout(chain):
    """The stored tensors disagree (configured ones per env, the rest one row); the getters do not."""
    assert chain.default_dofs_kp.shape[0] == NUM_ENVS and chain.default_dofs_armature.shape[0] == 1
    assert chain.get_default_dofs_kp().shape == chain.get_default_dofs_armature().shape == (NUM_ENVS, 3)


@pytest.mark.parametrize(
    "envs_idx",
    [[1], torch.tensor([1]), slice(1, 2), torch.tensor([False, True])],
    ids=["list", "tensor", "slice", "mask"],
)
def test_envs_idx_selects_those_envs_rows(drifted, envs_idx):
    torch.testing.assert_close(
        drifted.get_default_dofs_pos(["a_joint"], envs_idx=envs_idx), torch.tensor([[0.5]], device=drifted.device)
    )


def test_per_env_rows_are_all_returned(drifted):
    """A row that drifted is not hidden behind row 0."""
    torch.testing.assert_close(
        drifted.get_default_dofs_pos(["a_joint"]), torch.tensor([[_POS["a_joint"]], [0.5]], device=drifted.device)
    )


def test_an_empty_selection_reads_no_columns(chain):
    assert _spaces(chain.find_named_dofs([])) == ((), [], [])
    assert chain.get_default_dofs_pos([]).shape == (NUM_ENVS, 0)


@pytest.mark.parametrize("name, column", [("pin.1", 0), ("pin_1", 1), ("arm(2)", 2)])
def test_an_exact_joint_name_is_taken_literally(env, name, column):
    """Not re-read as a pattern: ``pin.1`` would also match ``pin_1``, ``arm(2)`` would match nothing."""
    quirky = env.entities["quirky"]
    # The local resolver takes the same key the same way, so both spaces name the one joint.
    assert _spaces(quirky.find_named_dofs([name])) == ((name,), [column], quirky.find_named_dofs_idx_local([name])[1])


@pytest.mark.parametrize("dofs_name", [None, ["a_joint"]], ids=["every-dof", "selected"])
def test_every_read_is_a_copy(chain, dofs_name):
    before = chain.default_dofs_kp.clone()
    chain.get_default_dofs_kp(dofs_name).fill_(-1.0)
    torch.testing.assert_close(chain.default_dofs_kp, before)


def test_a_multi_dof_joint_selects_all_of_its_columns(env):
    ball = env.entities["ball"]
    names, columns, local = _spaces(ball.find_named_dofs(["spin_joint", "ball_joint"]))
    assert (names, columns) == (("spin_joint", "ball_joint", "ball_joint", "ball_joint"), [3, 0, 1, 2])
    assert local == ball.find_named_dofs_idx_local(["spin_joint", "ball_joint"])[1]
    assert ball.get_default_dofs_pos("ball_joint").shape == (NUM_ENVS, 3)
    # Unbounded, so the soft range is the hard one rather than NaN.
    assert torch.isinf(ball.get_soft_dofs_pos_limits("ball_joint")).all()


_UNCONTROLLED = "joints this entity has but does not control"
_OVERLAP = "selected by more than one key"


@pytest.mark.parametrize(
    "dofs_name, fragment, why",
    [
        (["knee_joint"], "Not all query patterns are matched", "['knee_joint'] match no joint of this entity"),
        (["root"], "Not all query patterns are matched", _UNCONTROLLED),  # a real joint, but no column
        (["a_joint", "a_joint"], "Multiple matches for 'a_joint'", _OVERLAP),
        (["a_joint", "[ab]_joint"], "Multiple matches for 'a_joint'", _OVERLAP),
    ],
    ids=["unknown", "free-base", "repeated", "overlapping"],
)
def test_a_name_without_a_column_raises_with_the_way_out(chain, dofs_name, fragment, why):
    """Each failure is explained as itself: an overlap is not blamed on a free base, nor the reverse."""
    with pytest.raises(ValueError, match=fragment) as raised:
        chain.get_default_dofs_pos(dofs_name)
    message = str(raised.value)
    assert "Entity 'chain'" in message and why in message and "Fix:" in message
    # One line, and the controlled joints listed once (the resolver's "Available strings"), not twice.
    assert "\n" not in message and message.count("'c_joint'") <= 1
    assert ("find_named_dofs_idx_local" in message) == (why == _UNCONTROLLED)


def test_a_selection_from_another_entity_is_refused(env, chain):
    """Its columns address the other entity's layout: ``ball``'s column 1 is a ball DOF, not ``a_joint``."""
    with pytest.raises(ValueError, match="resolved on entity 'chain' was passed to get_default_dofs_pos"):
        env.entities["ball"].get_default_dofs_pos(chain.find_named_dofs("a_joint"))


def test_a_config_dict_takes_a_passive_joints_exact_name_literally(tmp_path):
    """As ``find_named_dofs`` does: ``pin.1`` is not a pattern that sets the controlled ``pin_1``."""
    (tmp_path / "quirky.xml").write_text(_QUIRKY_MJCF)
    pins = EntityOptions(
        file=str(tmp_path / "quirky.xml"),
        is_fixed_base=True,
        is_articulated=True,
        dofs_name=["pin_1", "arm(2)"],
        default_dofs_pos={"pin.1": 0.5},
    )
    with pytest.raises(ValueError, match="Not all query patterns are matched"):
        build_env(EdenConfig(env_options=EnvOptions(num_envs=1), scene_options=SceneOptions(pins=pins)))


def test_a_passive_joints_exact_name_does_not_fall_back_to_a_pattern(env):
    """``pin.1`` names a joint the entity has but does not control; as a regex it would select ``pin_1``."""
    passive_pin = env.entities["passive_pin"]
    with pytest.raises(ValueError, match=_UNCONTROLLED):
        passive_pin.find_named_dofs(["pin.1"])
    with pytest.raises(ValueError, match="Not all query patterns are matched"):
        passive_pin.find_named_dofs_idx_local(["pin.1"], name_scope=passive_pin.dofs_name)
    # Unscoped, the simulator-side resolver still reaches it.
    assert passive_pin.find_named_dofs_idx_local(["pin.1"])[0] == ["pin.1"]


def test_an_entity_that_controls_no_joint_selects_none(env):
    """Its scope is empty, and an empty scope searches no joint — not every joint the asset has."""
    crate = env.entities["crate"]
    assert _spaces(crate.find_named_dofs()) == _spaces(crate.find_named_dofs([])) == ((), [], [])
    with pytest.raises(ValueError, match="(?s)'crate'.*build the entity articulated or actuated"):
        crate.find_named_dofs(["root"])


@pytest.mark.parametrize(
    "entity_name, getter",
    [
        ("ball", "get_default_dofs_kp"),  # not actuated: no gains
        ("ghost", "get_default_dofs_armature"),  # kinematic: positions and limits only
        ("crate", "get_default_dofs_pos"),  # neither articulated nor actuated: nothing
    ],
)
def test_an_unresolved_property_raises_rather_than_reading_the_config(env, entity_name, getter):
    with pytest.raises(ValueError, match=f"'{entity_name}' has no resolved"):
        getattr(env.entities[entity_name], getter)()


def test_a_kinematic_entity_still_reads_its_positions(env):
    torch.testing.assert_close(
        env.entities["ghost"].get_default_dofs_pos_limits("c_joint")[0], torch.tensor([[-2.0, 0.0]], device=env.device)
    )


def test_every_resolved_property_has_a_getter_named_after_its_field(chain):
    """A field added to ``RESOLVED_DOF_TENSOR_FIELD_NAMES`` without a getter fails here."""
    for field_name in RESOLVED_DOF_TENSOR_FIELD_NAMES:
        getter = getattr(RigidEntity, f"get_{field_name}", None)
        assert getter is not None, f"no get_{field_name}"
        stored = getattr(chain, field_name)
        torch.testing.assert_close(getter(chain), stored.expand(NUM_ENVS, *stored.shape[1:]), msg=field_name)
