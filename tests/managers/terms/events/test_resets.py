"""The reset events in ``eden/managers/terms/events/resets.py``."""

from __future__ import annotations

import math

import pytest
import torch
from genesis.utils.geom import transform_by_quat

from eden.managers.terms.events.resets import (
    reset_base_state_uniform,
    reset_dofs_by_offset,
    reset_dofs_by_scale,
    reset_dofs_by_scale_and_offset,
)
from tests.support.envs import build_env

pytestmark = pytest.mark.requires_genesis

SIDE_LYING = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)  # a 90 deg roll: a term writing a pure yaw is caught
DEFAULT_POS = (0.0, 0.0, 1.0)
PROBE_MJCF = """
<mujoco model="probe">
  <worldbody>
    <body name="base">
      <freejoint/>
      <geom name="g_base" type="box" size="0.05 0.05 0.05" pos="0 0 0.1" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""
#: An MJCF root body with its own pose. The G1 pelvis sits at z = 0.793 in its MJCF, and composed with
#: ``default_root_pos`` it spawned 0.79 m up: the default must replace it.
POSED_ROOT_MJCF = """<mujoco model="posed_root"><worldbody>
  <body name="pelvis" pos="0 0 0.793" quat="0.7071068 0 0 0.7071068">
    <freejoint/><geom type="box" size="0.05 0.05 0.05" mass="1.0"/></body>
</worldbody></mujoco>"""
POSED_ROOT_POS = (0.3, -0.8, 0.78)
#: A free hub carrying a pendulum on a hinge: a floating base under a joint whose velocity moves a link.
SWING_MJCF = """<mujoco model="swing"><compiler angle="radian"/><worldbody>
  <body name="hub"><freejoint/><geom type="box" size="0.05 0.05 0.05" mass="1.0"/>
    <body name="pendulum" pos="0 0 -0.1"><joint name="swing_joint" type="hinge" axis="0 1 0" range="-1 1"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.3" mass="0.2"/></body></body>
</worldbody></mujoco>"""
#: A fixed two-hinge arm, its joints named out of tree order, so its DOFs are no ``range``.
PAIR_MJCF = """<mujoco model="pair"><compiler angle="radian"/><worldbody>
  <body name="link1" pos="0.6 0.6 0.5"><joint name="j1" type="hinge" axis="0 1 0" range="-1 1"/>
    <geom type="capsule" size="0.02" fromto="0 0 0 0.1 0 0" mass="0.2"/>
    <body name="link2" pos="0.1 0 0"><joint name="j2" type="hinge" axis="0 1 0" range="-1 1"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.1 0 0" mass="0.2"/></body></body>
</worldbody></mujoco>"""
#: Every DOF unbounded, a ball joint and a range-less hinge: -/+inf hard limits in Genesis.
FLOPPY_MJCF = """<mujoco model="floppy"><compiler angle="radian" autolimits="true"/><worldbody>
  <body name="anchor" pos="-0.6 -0.6 0.4"><geom type="box" size="0.04 0.04 0.02" density="500"/>
    <body name="socket" pos="0 0 0.05"><joint name="ball_joint" type="ball"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.08 0 0" density="500"/>
      <body name="spindle" pos="0.08 0 0"><joint name="free_spin_joint" axis="0 0 1"/>
        <geom type="cylinder" size="0.02 0.01" density="500"/></body></body></body>
</worldbody></mujoco>"""


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """A box authored lying on its side, a body whose CoM sits 0.1 m above its authored origin, an MJCF
    root body with a pose of its own, a fixed arm whose DOFs are all unbounded, and a free hub swinging a
    pendulum.
    """
    from eden.options import BoxOptions, EntityOptions, EnvOptions, SceneOptions
    from eden.utils.configs import EdenConfig

    assets = tmp_path_factory.mktemp("reset_assets")
    for name, text in (
        ("probe", PROBE_MJCF),
        ("posed_root", POSED_ROOT_MJCF),
        ("floppy", FLOPPY_MJCF),
        ("swing", SWING_MJCF),
        ("pair", PAIR_MJCF),
    ):
        (assets / f"{name}.xml").write_text(text)
    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=4),
            scene_options=SceneOptions(
                box=BoxOptions(size=(0.1, 0.2, 0.3), default_root_pos=DEFAULT_POS, default_root_quat=SIDE_LYING),
                probe=EntityOptions(
                    file=str(assets / "probe.xml"), is_fixed_base=False, default_root_pos=(2.0, 0.0, 1.0)
                ),
                posed=EntityOptions(file=str(assets / "posed_root.xml"), default_root_pos=POSED_ROOT_POS),
                floppy=EntityOptions(
                    file=str(assets / "floppy.xml"),
                    is_fixed_base=True,
                    is_articulated=True,
                    dofs_name=["ball_joint", "free_spin_joint"],
                ),
                swing=EntityOptions(
                    file=str(assets / "swing.xml"),
                    is_fixed_base=False,
                    is_articulated=True,
                    dofs_name=["swing_joint"],
                    default_root_pos=(-2.0, 0.0, 1.0),
                ),
                pair=EntityOptions(
                    file=str(assets / "pair.xml"), is_fixed_base=True, is_articulated=True, dofs_name=["j2", "j1"]
                ),
            ),
        )
    )


def _reset(env, name, envs_idx=None, yaw=0.0, pitch=0.0, x=0.0):
    reset_base_state_uniform(
        env,
        envs_idx,
        entity_name=name,
        pos_x_range=(x, x),
        pos_y_range=(0.0, 0.0),
        pitch_range=(pitch, pitch),
        yaw_range=(yaw, yaw),
    )


def _rotated(quat, vec):
    return transform_by_quat(torch.tensor([vec]).expand(len(quat), 3), quat)


@pytest.mark.parametrize(
    "envs_idx",
    [torch.tensor([False, True, False, True]), torch.tensor([1, 3])],
    ids=["bool-mask", "int-indices"],
)
def test_only_the_selected_envs_are_reset_to_the_default_pose(env, restore_state, envs_idx):
    """The side-lying default survives: the reset box is not stood up."""
    box = restore_state(env.entities["box"])
    box.set_pos(torch.full((4, 3), 5.0), relative=True)
    box.set_quat(torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4), relative=True)

    _reset(env, "box", envs_idx)

    pos, quat = box.get_pos(relative=True), box.get_quat(relative=True)
    torch.testing.assert_close(pos[[1, 3]], torch.tensor([DEFAULT_POS] * 2))
    torch.testing.assert_close(quat[[1, 3]], torch.tensor([SIDE_LYING] * 2))
    torch.testing.assert_close(pos[[0, 2]], torch.full((2, 3), 5.0))
    torch.testing.assert_close(quat[[0, 2]], torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2))


def test_yaw_is_composed_onto_the_default_about_world_z(env, restore_state):
    box = restore_state(env.entities["box"])
    _reset(env, "box", yaw=math.pi / 2)
    quat = box.get_quat(relative=True)
    # The default carries body +y to world +z; a further world yaw leaves that and carries body +x to +y.
    torch.testing.assert_close(_rotated(quat, [0.0, 1.0, 0.0]), torch.tensor([[0.0, 0.0, 1.0]] * 4), atol=1e-6, rtol=0)
    torch.testing.assert_close(_rotated(quat, [1.0, 0.0, 0.0]), torch.tensor([[0.0, 1.0, 0.0]] * 4), atol=1e-6, rtol=0)


def test_pitch_is_composed_in_the_world_frame(env, restore_state):
    """``R_rpy @ R_default`` tips the side-lying box's up axis over; mjlab's body-frame form would not."""
    box = restore_state(env.entities["box"])
    _reset(env, "box", pitch=math.pi / 2)
    quat = box.get_quat(relative=True)
    torch.testing.assert_close(_rotated(quat, [0.0, 1.0, 0.0]), torch.tensor([[1.0, 0.0, 0.0]] * 4), atol=1e-6, rtol=0)
    torch.testing.assert_close(_rotated(quat, [1.0, 0.0, 0.0]), torch.tensor([[0.0, 0.0, -1.0]] * 4), atol=1e-6, rtol=0)


def test_the_offset_is_applied_to_the_authored_origin(env, restore_state):
    """Written with ``relative=True``: the CoM Genesis re-origins the body onto ends up 0.1 m above it."""
    probe = restore_state(env.entities["probe"])
    probe.set_pos(torch.zeros(4, 3))
    _reset(env, "probe", x=0.1)
    torch.testing.assert_close(probe.get_pos(relative=True), torch.tensor([[2.1, 0.0, 1.0]] * 4))
    torch.testing.assert_close(probe.get_pos(), torch.tensor([[2.1, 0.0, 1.1]] * 4))


def test_the_default_replaces_an_mjcf_root_bodys_own_pose(env, restore_state):
    posed = restore_state(env.entities["posed"])
    posed.set_pos(torch.ones(4, 3))
    _reset(env, "posed")
    torch.testing.assert_close(posed.get_pos(relative=True), torch.tensor([POSED_ROOT_POS] * 4), atol=1e-5, rtol=0)
    torch.testing.assert_close(
        posed.get_quat(relative=True).abs(), torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4), atol=1e-5, rtol=0
    )


@pytest.mark.parametrize(
    ("reset_term", "params"),
    [
        (reset_dofs_by_offset, {"dofs_pos_range": (-0.1, 0.1)}),
        (reset_dofs_by_scale, {"dofs_pos_scale_range": (0.95, 1.05)}),
        (reset_dofs_by_scale_and_offset, {"dofs_scale_range": (0.8, 1.2), "dofs_offset_range": (-0.1, 0.1)}),
    ],
    ids=["by-offset", "by-scale", "by-scale-and-offset"],
)
def test_dof_resets_leave_unbounded_dofs_finite(env, restore_state, reset_term, params):
    """The resets clamp to the soft limits: -/+inf there is a no-op, where a NaN limit would write NaN."""
    floppy = restore_state(env.entities["floppy"])
    assert torch.isinf(floppy.soft_dofs_pos_limits).all()
    reset_term(env, slice(None), entity_name="floppy", **params)
    assert torch.isfinite(floppy.get_dofs_pos()).all()


def _dof_and_link_state(entity):
    return {
        "qpos": entity.get_qpos(),
        "dofs_vel": entity.get_dofs_vel(dofs_idx_local=list(range(7))),  # the free base's six, then the hinge
        "links_pos": entity.get_links_pos(),
        "links_quat": entity.get_links_quat(),
        "links_vel": entity.get_links_vel(),
        "links_ang": entity.get_links_ang(),
    }


def test_a_dof_reset_with_velocities_refreshes_link_state_from_what_it_wrote(env, restore_state):
    """One forward-kinematics pass covers the whole reset: velocities are written first.

    That pass must still bring the link poses and velocities up to the written DOF state, the free base's
    velocity is zeroed as a position reset zeroes it, and the envs outside the mask are left alone.
    """
    swing = restore_state(env.entities["swing"])
    swing.set_dofs_vel(torch.tensor([[0.3, -0.2, 0.1, 0.5, -0.4, 0.6]] * 4), dofs_idx_local=list(range(6)))
    before = {name: value.clone() for name, value in _dof_and_link_state(swing).items()}
    mask = torch.tensor([True, False, True, False])

    reset_dofs_by_offset(env, mask, entity_name="swing", dofs_pos_range=(-0.3, 0.3), dofs_vel_range=(0.5, 1.0))
    after = {name: value.clone() for name, value in _dof_and_link_state(swing).items()}

    for name in before:
        torch.testing.assert_close(after[name][~mask], before[name][~mask], rtol=0, atol=0, msg=name)
    hinge_pos, hinge_vel = swing.get_dofs_pos()[mask, 0], after["dofs_vel"][mask, 6]
    assert ((hinge_pos.abs() <= 0.3) & (hinge_vel >= 0.5) & (hinge_vel <= 1.0)).all()
    assert (after["dofs_vel"][mask, :6] == 0).all()
    # The same DOF state written through the plain setters, each running its own forward-kinematics pass.
    swing.set_qpos(after["qpos"])
    swing.set_dofs_vel(after["dofs_vel"], dofs_idx_local=list(range(7)))
    for name in ("links_pos", "links_quat", "links_vel", "links_ang"):
        torch.testing.assert_close(after[name], _dof_and_link_state(swing)[name], msg=name)


def test_the_root_coordinates_are_the_authored_pose_unless_genesis_re_origins_the_body(env):
    """A centred box writes its pose straight into qpos; the probe, whose CoM sits 0.1 m up, cannot."""
    assert env.entities["box"].root_qpos_is_authored_pose
    assert not env.entities["probe"].root_qpos_is_authored_pose
    assert not env.entities["floppy"].root_qpos_is_authored_pose  # no free base at all


def test_a_dof_reset_through_a_mask_writes_dofs_named_out_of_tree_order(env, restore_state):
    """Every env's row with the mask: an entity whose DOFs are no ``range`` gets the selected rows."""
    pair = restore_state(env.entities["pair"])
    assert not isinstance(pair._genesis_dofs_idx, range)
    pair.set_dofs_vel(torch.full((4, 2), 0.7))
    before_pos, before_vel = pair.get_dofs_pos().clone(), pair.get_dofs_vel().clone()
    mask = torch.tensor([True, False, True, False])

    reset_dofs_by_offset(env, mask, entity_name="pair", dofs_pos_range=(0.2, 0.3), dofs_vel_range=(0.5, 0.6))

    pos, vel = pair.get_dofs_pos(), pair.get_dofs_vel()
    assert ((pos[mask] >= 0.2) & (pos[mask] <= 0.3)).all() and ((vel[mask] >= 0.5) & (vel[mask] <= 0.6)).all()
    torch.testing.assert_close(pos[~mask], before_pos[~mask])
    torch.testing.assert_close(vel[~mask], before_vel[~mask])


@pytest.mark.parametrize("name", ["box", "probe"], ids=["root-qpos-write", "pose-setters"])
def test_a_base_reset_zeroes_the_velocities_on_either_path(env, restore_state, name):
    """As Genesis' ``set_pos`` / ``set_quat`` do for a simulated entity, whichever write the term takes."""
    entity = restore_state(env.entities[name])
    entity.set_dofs_vel(torch.full((4, 6), 0.4), dofs_idx_local=list(range(6)))
    mask = torch.tensor([False, True, False, True])

    _reset(env, name, mask, yaw=0.3)

    vel = entity.get_dofs_vel(dofs_idx_local=list(range(6)))
    assert (vel[mask] == 0).all()
    torch.testing.assert_close(vel[~mask], torch.full((2, 6), 0.4))
