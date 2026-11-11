"""``MujocoSim``: model build, DOF mapping, the joint PD, the IMU, and the simulated gantry."""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from eden.extensions.deployment.base import resolve_dofs_values  # noqa: E402
from eden.extensions.deployment.utils.mujoco_sim import MujocoFloor, MujocoSim  # noqa: E402

# Each ``MujocoSim`` below loads the fixed-root two-link arm from ``conftest.two_link_asset``; with
# ``floating_base=True`` (the default) the loader gives it a free joint and a floor.


def _sim(asset, dofs=("j1", "j2"), **kwargs) -> MujocoSim:
    return MujocoSim(str(asset), list(dofs), **kwargs)


def test_build_adds_a_free_joint_a_floor_and_an_imu(two_link_asset):
    sim = _sim(two_link_asset)
    assert sim.model.nq == 9, "7 free-joint qpos + 2 hinges"
    assert int(mujoco.mjtGeom.mjGEOM_PLANE) in sim.model.geom_type.tolist()
    reading = sim.read()
    assert {key: reading[key].shape for key in ("base_quat", "base_ang_vel", "base_lin_acc", "dofs_pos")} == {
        "base_quat": (4,),
        "base_ang_vel": (3,),
        "base_lin_acc": (3,),
        "dofs_pos": (2,),
    }


def test_the_floor_stands_where_it_is_told_with_the_contact_it_states(two_link_asset):
    """A ``MujocoFloor`` places the plane (raised, tilted) and sets what contact it states; MuJoCo keeps the rest."""
    normal = (0.0, -np.sin(0.1), np.cos(0.1))
    sim = _sim(two_link_asset, floor=MujocoFloor(pos=(0.0, 0.0, 0.05), normal=normal, friction=0.6))
    floor = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "__eden_floor__")
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, sim.model.geom_quat[floor])
    np.testing.assert_allclose(sim.model.geom_pos[floor], (0.0, 0.0, 0.05))
    np.testing.assert_allclose(rotation.reshape(3, 3)[:, 2], normal, atol=1e-9)
    assert sim.model.geom_friction[floor, 0] == pytest.approx(0.6)
    assert tuple(sim.model.geom_solref[floor]) == (0.02, 1.0), "no sol_params stated: MuJoCo's default"

    assert _sim(two_link_asset, floor=None).floor is None
    assert int(mujoco.mjtGeom.mjGEOM_PLANE) not in _sim(two_link_asset, floor=None).model.geom_type.tolist()
    with pytest.raises(ValueError, match="7 values"):
        MujocoFloor(sol_params=(0.02, 1.0))


def test_the_robot_is_the_base_bodys_tree_without_the_assets_other_bodies(tmp_path):
    """What a recording frames: the robot, not a stand or a prop the asset also carries."""
    from tests.extensions.deployment.conftest import TWO_LINK_MJCF

    asset = tmp_path / "with_stand.xml"
    stand = '<body name="stand" pos="5 0 0"><geom type="box" size=".2 .2 .5"/></body></worldbody>'
    asset.write_text(TWO_LINK_MJCF.replace("</worldbody>", stand, 1))
    sim = _sim(asset)
    names = [mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_BODY, b) for b in sim.robot_body_ids()]
    assert names == ["base", "link1", "link2"]


def test_state_follows_the_requested_dof_order(two_link_asset):
    sim = _sim(two_link_asset, dofs=("j2", "j1"))
    sim.reset(np.array([0.5, -0.5]))
    np.testing.assert_allclose(sim.read()["dofs_pos"], [0.5, -0.5], atol=1e-6)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"dofs": ("j1", "does_not_exist")}, "missing actuated joints"), ({"integrator": "verlet"}, "Unknown integrator")],
)
def test_a_bad_model_request_raises(two_link_asset, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _sim(two_link_asset, **kwargs)


def test_reset_places_the_base_and_zeroes_every_velocity(two_link_asset):
    sim = _sim(two_link_asset)
    quat = np.array([np.cos(0.2), 0.0, np.sin(0.2), 0.0])
    sim.reset(np.array([0.1, 0.2]), base_pos=[1.0, 2.0, 1.5], base_quat=quat)
    reading = sim.read()
    np.testing.assert_allclose(reading["base_pos"], [1.0, 2.0, 1.5], atol=1e-6)
    np.testing.assert_allclose(reading["base_quat"], quat, atol=1e-6)
    np.testing.assert_allclose(reading["base_lin_vel"], 0.0, atol=1e-9)
    np.testing.assert_allclose(reading["dofs_vel"], 0.0, atol=1e-9)


def test_a_fixed_base_reports_an_identity_base_state(two_link_asset):
    reading = _sim(two_link_asset, floating_base=False).read()
    np.testing.assert_allclose(reading["base_pos"], 0.0, atol=1e-9)
    np.testing.assert_allclose(reading["base_lin_vel"], 0.0, atol=1e-9)


def test_the_pd_settles_on_its_target(two_link_asset):
    """Within gravity droop: a plain PD's steady-state error is gravity torque / kp, a few 0.01 rad here."""
    sim = _sim(two_link_asset, floating_base=False)
    kp, kd = np.full(2, 50.0), np.full(2, 2.0)
    sim.set_implicit_damping(kd)
    sim.reset(np.zeros(2))
    target = np.array([0.3, -0.4])
    for _ in range(500):
        sim.step_pd(target, np.zeros(2), kp, kd, np.zeros(2), max(1, round(0.02 / sim.sim_dt)))
    reading = sim.read()
    np.testing.assert_allclose(reading["dofs_pos"], target, atol=0.1)
    np.testing.assert_allclose(reading["dofs_vel"], 0.0, atol=0.05)


def test_implicit_damping_follows_kd_instead_of_stacking(two_link_asset):
    """A deploy slot switch re-folds kd every ramp step; each call must replace the last one's term."""
    sim = _sim(two_link_asset, floating_base=False)
    armature = sim.model.dof_armature[sim._v_idx].copy()

    for kd in (np.full(2, 2.0), np.full(2, 8.0), np.full(2, 4.0)):
        sim.set_implicit_damping(kd)

    np.testing.assert_allclose(sim.model.dof_armature[sim._v_idx], armature + 4.0 * sim.sim_dt)


def test_dofs_torque_is_the_last_pd_command(two_link_asset):
    sim = _sim(two_link_asset, floating_base=False)
    sim.reset(np.zeros(2))
    sim.step_pd(np.array([0.1, -0.1]), np.zeros(2), np.full(2, 50.0), np.full(2, 2.0), np.zeros(2), n_substeps=1)
    np.testing.assert_allclose(sim.read()["dofs_torque"], [5.0, -5.0], atol=1e-6)


def test_the_accelerometer_reads_one_g_when_the_robot_rests_on_the_floor(two_link_asset):
    """Specific force, as a real IMU reports it. The magnitude: the dangling links tilt the base a little."""
    sim = _sim(two_link_asset)
    sim.reset(np.zeros(2), base_pos=[0.0, 0.0, 0.05])  # the 5 cm root sphere on the floor
    for _ in range(400):
        sim.step_pd(np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), n_substeps=10)
    assert np.linalg.norm(sim.read()["base_lin_acc"]) == pytest.approx(9.81, abs=0.3)


@pytest.mark.parametrize("robot", ["G1Dofs29", "Go2"])
def test_a_shipped_unitree_asset_holds_its_stance_without_diverging(robot):
    from eden.options.robots import unitree_g1, unitree_go2
    from eden.utils.assets import get_asset_path

    options = {"G1Dofs29": unitree_g1.G1Dofs29, "Go2": unitree_go2.Go2}[robot]()
    dofs = list(options.dofs_name)
    sim = MujocoSim(get_asset_path(options.file), dofs)
    q0, kp, kd = (
        resolve_dofs_values(values, dofs)
        for values in (options.default_dofs_pos, options.default_dofs_kp, options.default_dofs_kd)
    )
    sim.set_implicit_damping(kd)
    sim.reset(q0, base_pos=[0.0, 0.0, float(options.default_root_pos[2])])
    for _ in range(200):
        sim.step_pd(q0, np.zeros_like(q0), kp, kd, np.zeros_like(q0), max(1, round(0.02 / sim.sim_dt)))
    reading = sim.read()
    assert np.isfinite(reading["dofs_pos"]).all() and np.isfinite(reading["base_quat"]).all()


# ---------------------------------------------------------------------------
# The gantry: a tension-only rope from a fixed anchor
# ---------------------------------------------------------------------------


def _hung(asset, mounts=((None, (0.0, 0.0, 0.25)),), **kwargs) -> MujocoSim:
    sim = _sim(asset)
    sim.set_gantry(enabled=True, mounts=mounts, **kwargs)
    return sim


def _drop(sim, dz: float = 0.1) -> None:
    sim.data.qpos[sim._free_qpos + 2] -= dz


def _tilt(sim, axis: int, angle: float) -> None:
    quat = np.zeros(4)
    quat[0], quat[1 + axis] = np.cos(angle / 2), np.sin(angle / 2)
    sim.data.qpos[sim._free_qpos + 3 : sim._free_qpos + 7] = quat


def _rope_wrench(sim, body_id=None) -> tuple[np.ndarray, np.ndarray]:
    """The force and torque the rope applies to ``body_id`` (the base by default).

    The tests pose the robot by writing ``qpos``, so the body frames are refreshed first; in the loop
    ``mj_step`` does that every substep.
    """
    mujoco.mj_forward(sim.model, sim.data)
    sim._apply_gantry()
    wrench = sim.data.xfrc_applied[sim._base_body_id if body_id is None else body_id]
    return np.array(wrench[:3]), np.array(wrench[3:])


def test_a_taut_rope_pulls_up_and_a_slack_one_never_pushes(two_link_asset):
    sim = _hung(two_link_asset)
    _drop(sim)
    assert _rope_wrench(sim)[0][2] > 0.0
    _drop(sim, -0.2)  # above the anchor's rest length: slack
    force, torque = _rope_wrench(sim)
    np.testing.assert_allclose(force, 0.0, atol=1e-9)
    np.testing.assert_allclose(torque, 0.0, atol=1e-9)


def test_slack_delays_the_catch(two_link_asset):
    sim = _hung(two_link_asset, slack=0.2)
    _drop(sim)
    np.testing.assert_allclose(_rope_wrench(sim)[0], 0.0, atol=1e-9)
    _drop(sim, 0.2)
    assert _rope_wrench(sim)[0][2] > 0.0


def test_a_rope_above_the_base_rights_a_tilt_but_one_at_the_centre_of_mass_cannot(two_link_asset):
    """The moment arm is measured from the centre of mass, never from the body frame origin."""
    above = _hung(two_link_asset)
    _drop(above)
    _tilt(above, axis=0, angle=np.pi / 4)
    assert np.linalg.norm(_rope_wrench(above)[1]) > 1e-6

    at_com = _sim(two_link_asset)
    mujoco.mj_forward(at_com.model, at_com.data)
    bid = at_com._base_body_id
    com = at_com.data.xmat[bid].reshape(3, 3).T @ (at_com.data.xipos[bid] - at_com.data.xpos[bid])
    at_com.set_gantry(enabled=True, mounts=((None, tuple(com)),))
    _drop(at_com)
    _tilt(at_com, axis=0, angle=np.pi / 4)
    np.testing.assert_allclose(_rope_wrench(at_com)[1], 0.0, atol=1e-9)


def test_a_shoulder_pair_rights_a_roll_one_central_rope_cannot(two_link_asset):
    one = _hung(two_link_asset, mounts=((None, (0.0, 0.0, 0.0)),))
    pair = _hung(two_link_asset, mounts=((None, (0.0, 0.15, 0.0)), (None, (0.0, -0.15, 0.0))))
    for sim in (one, pair):
        _tilt(sim, axis=0, angle=np.pi / 6)
    np.testing.assert_allclose(_rope_wrench(one)[1], 0.0, atol=1e-9)
    assert _rope_wrench(pair)[1][0] < -1e-6, "pushes back against the roll"


def test_two_ropes_on_one_body_sum_rather_than_overwrite(two_link_asset):
    pair = _hung(two_link_asset, mounts=((None, (0.0, 0.15, 0.0)), (None, (0.0, -0.15, 0.0))))
    solo = _hung(two_link_asset, mounts=((None, (0.0, 0.15, 0.0)),))
    for sim in (pair, solo):
        _drop(sim)
    assert _rope_wrench(pair)[0][2] == pytest.approx(2.0 * _rope_wrench(solo)[0][2], rel=1e-6)


def test_a_rope_tied_to_a_named_link_loads_that_link(two_link_asset):
    sim = _hung(two_link_asset, mounts=(("link2", (0.0, 0.0, 0.2)),))
    _drop(sim)
    assert _rope_wrench(sim, sim._body_id("link2"))[0][2] > 0.0
    np.testing.assert_allclose(sim.data.xfrc_applied[sim._base_body_id], 0.0)


@pytest.mark.parametrize(
    ("mounts", "match"), [((("torso_link", (0.0, 0.0, 0.0)),), "Available:"), ((), "at least one mount")]
)
def test_a_bad_rig_is_refused(two_link_asset, mounts, match):
    with pytest.raises(ValueError, match=match):
        _sim(two_link_asset).set_gantry(enabled=True, mounts=mounts)


def test_cranking_the_anchor_takes_up_and_pays_out_rope(two_link_asset):
    """The rest length never changes; the pair's anchors stay level as they move."""
    sim = _hung(two_link_asset, mounts=((None, (0.0, 0.15, 0.0)), (None, (0.0, -0.15, 0.0))))
    _drop(sim)
    taut = _rope_wrench(sim)[0][2]
    heights = [rope["anchor"][2] for rope in sim._gantry["ropes"]]

    assert sim.adjust_anchor(0.05) == pytest.approx(sim.gantry_anchor_height)
    assert _rope_wrench(sim)[0][2] > taut
    moved = [rope["anchor"][2] for rope in sim._gantry["ropes"]]
    assert moved[0] == pytest.approx(moved[1]) and moved[0] == pytest.approx(heights[0] + 0.05)

    sim.adjust_anchor(-0.5)
    np.testing.assert_allclose(_rope_wrench(sim)[0], 0.0, atol=1e-9)


def test_the_anchor_height_can_be_set_at_rig_time_and_is_absent_unrigged(two_link_asset):
    assert _hung(two_link_asset, height=2.5).gantry_anchor_height == pytest.approx(2.5)
    unrigged = _sim(two_link_asset)
    assert unrigged.adjust_anchor(0.1) is None and unrigged.gantry_anchor_height is None


def test_support_scales_the_rope_force(two_link_asset):
    full, half = _hung(two_link_asset), _hung(two_link_asset, support=0.5)
    for sim in (full, half):
        _drop(sim)
    assert 0.0 < _rope_wrench(half)[0][2] < _rope_wrench(full)[0][2]


def test_unrigging_clears_the_rope_force(two_link_asset):
    sim = _hung(two_link_asset)
    _drop(sim)
    _rope_wrench(sim)
    sim.set_gantry(enabled=False)
    np.testing.assert_allclose(sim.data.xfrc_applied, 0.0)
    np.testing.assert_allclose(_rope_wrench(sim)[0], 0.0)


# ---------------------------------------------------------------------------
# The gantry overlay, drawn into a real (GL-free) ``mjvScene``
# ---------------------------------------------------------------------------


def _drawn(sim, maxgeom: int = 32, **kwargs):
    scene = mujoco.MjvScene(sim.model, maxgeom=maxgeom)
    sim.draw_gantry(scene, **kwargs)
    return scene, [int(scene.geoms[i].type) for i in range(scene.ngeom)]


_ARROW = int(mujoco.mjtGeom.mjGEOM_ARROW)


def test_a_taut_rope_draws_the_rope_and_its_force_a_slack_one_only_the_rope(two_link_asset):
    taut, slack = _hung(two_link_asset), _hung(two_link_asset, slack=0.5)
    _drop(taut)
    for sim in (taut, slack):
        _rope_wrench(sim)
    assert sorted(_drawn(taut)[1]).count(_ARROW) == 1 and len(_drawn(taut)[1]) == 2
    assert _drawn(slack)[1] != [] and _ARROW not in _drawn(slack)[1]


def test_a_pair_draws_a_rope_and_arrow_each_and_stops_at_the_scene_limit(two_link_asset):
    """An overrun would be a write past ``maxgeom``, not a missing arrow."""
    sim = _hung(two_link_asset, mounts=((None, (0.0, 0.15, 0.0)), (None, (0.0, -0.15, 0.0))))
    _drop(sim)
    _rope_wrench(sim)
    assert len(_drawn(sim)[1]) == 4
    scene, _ = _drawn(sim, maxgeom=3)
    assert scene.ngeom == 3


def test_the_arrow_scales_with_the_force(two_link_asset):
    sim = _hung(two_link_asset)
    _drop(sim)
    _rope_wrench(sim)
    lengths = []
    for force_scale in (1.0e-3, 2.0e-3):
        scene, types = _drawn(sim, force_scale=force_scale)
        lengths.append(float(scene.geoms[types.index(_ARROW)].size[2]))
    assert lengths[1] == pytest.approx(2.0 * lengths[0], rel=1e-6)


def test_an_unrigged_gantry_draws_nothing(two_link_asset):
    assert _drawn(_sim(two_link_asset))[1] == []
