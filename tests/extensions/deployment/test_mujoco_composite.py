"""``MujocoCompositeDeployment``: a composite rig and its props on one MuJoCo model, through the composite deployment.

The rig is :mod:`test_composite`'s two-link arm with a hand welded to its tip, built into a real env with a
cube to fall and a table to stand still; the MuJoCo model is assembled from that env.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from eden.extensions.deployment.mujoco_composite import MujocoCompositeDeployment, MujocoPartDeployment
from eden.extensions.deployment.utils.mujoco_scene import kinematic_markers, read_floor
from eden.extensions.deployment.utils.mujoco_sim import MujocoFloor
from eden.options.entities import BoxOptions, SphereOptions
from tests.extensions.deployment.conftest import TWO_LINK_MJCF
from tests.extensions.deployment.test_composite import ARM_DOFS, HAND_DEFAULT, HAND_DOFS, _humanoid_config, _rig
from tests.support.deploy import deployments

CUBE_HALF = 0.025
CUBE_SPAWN = (0.0, 0.8, 0.4)  # well clear of the arm, which swings in the x-z plane at y = 0


def _config(assets: Path, **extra_entities):
    from eden import actions, observations
    from eden.options import ObservationGroupOptions
    from eden.utils.configs import (
        ActionManagerOptions,
        EdenRLConfig,
        EnvOptions,
        ObservationManagerOptions,
        SceneOptions,
    )

    return EdenRLConfig(
        env_options=EnvOptions(num_envs=1),
        scene_options=SceneOptions(
            robot=_rig(assets),
            cube=BoxOptions(size=(2 * CUBE_HALF,) * 3, default_root_pos=CUBE_SPAWN),
            table=BoxOptions(size=(0.4, 0.4, 0.1), default_root_pos=(0.0, -0.8, 0.05), fixed=True),
            **extra_entities,
        ),
        action_options=ActionManagerOptions(
            arm=actions.ImplicitPDController.configure(entity_name="robot", dofs_name=ARM_DOFS),
            hand=actions.ImplicitPDController.configure(entity_name="hand", dofs_name=HAND_DOFS),
        ),
        observation_options=ObservationManagerOptions(
            policy=ObservationGroupOptions(
                arm_pos=observations.DofsPos.configure(entity_name="robot"),
            )
        ),
    )


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from tests.support.envs import build_env

    return build_env(_config(tmp_path_factory.mktemp("mujoco_rig")), reset=True)


@pytest.fixture
def deploy(env):
    with deployments(env) as make:
        yield lambda **options: make(MujocoCompositeDeployment, connect=True, **options)


def _step(dep, arm, hand) -> None:
    dep.step({"arm": torch.tensor([arm], dtype=torch.float32), "hand": torch.tensor([hand], dtype=torch.float32)})


def test_every_part_runs_on_one_mujoco_model_of_the_rig_and_its_props(env, deploy):
    dep = deploy()

    assert [type(part) for part in dep.parts.values()] == [MujocoPartDeployment] * 2
    assert dep._prop_names == ["cube", "table"], "every colliding entity besides the rig and the ground"
    np.testing.assert_allclose(dep._rig.prop_pose("cube")[0], CUBE_SPAWN, atol=1e-6)
    np.testing.assert_allclose(
        dep.read_state().dofs_pos, dep.default_dof_pos, atol=1e-9, err_msg="placed at the stance"
    )


def test_the_model_disagreeing_with_the_envs_kinematics_is_refused_naming_the_link(env, deploy, monkeypatch):
    """A weld 1 cm off would put every fingertip the policy observes 1 cm off what it acts on."""
    rig_parts = MujocoCompositeDeployment._rig_parts

    def shifted(self):
        parts = rig_parts(self)
        parts[1].pos = parts[1].pos + np.array([0.01, 0.0, 0.0])
        return parts

    monkeypatch.setattr(MujocoCompositeDeployment, "_rig_parts", shifted)
    with pytest.raises(ValueError, match=r"link 'hand/\w+' is 0\.01\d* \(m \+ rad\) off"):
        deploy()


def test_the_rig_holds_in_stop_and_follows_the_policy_after_start(env, deploy):
    """The test arm carries no gravity compensation, so at its kp of 50 it sags ~0.08 rad wherever it is held."""
    sag = 0.1
    dep = deploy()
    dep.reset()
    for _ in range(30):
        _step(dep, [0.6, -0.4], [0.5, 0.5])
    np.testing.assert_allclose(dep.state.dofs_pos, dep.default_dof_pos, atol=sag, err_msg="STOP holds the boot pose")

    dep.request_start()
    for _ in range(150):
        _step(dep, [0.6, -0.4], [0.5, 0.5])

    # What the action terms made of the policy's action: the PD adds each joint's default pose.
    target = np.concatenate(
        [env.action_manager.terms[term].processed_action[0].cpu().numpy() for term in ("arm", "hand")]
    )
    assert np.abs(target - dep.default_dof_pos).min() > 0.4, "every joint is commanded well away from its hold"
    np.testing.assert_allclose(dep.state.dofs_pos, target, atol=sag, err_msg="every part follows the policy")


def test_a_prop_falls_in_mujoco_and_its_env_entity_follows_while_a_fixed_one_stays(env, deploy):
    dep = deploy()
    dep.reset()
    table = dep._rig.prop_pose("table")[0]
    for _ in range(60):
        _step(dep, [0.0, 0.0], [HAND_DEFAULT, HAND_DEFAULT])

    cube = dep._rig._prop_bodies("cube")
    mass = float(env.entities["cube"].get_mass().reshape(-1)[0])
    assert float(dep._rig.model.body_mass[cube].sum()) == pytest.approx(mass), "what the env built it to weigh"
    assert mass == pytest.approx(200.0 * (2 * CUBE_HALF) ** 3), "at the material's rho, not MuJoCo's 1000 kg/m^3"
    root = dep._rig._prop_body["cube"]
    np.testing.assert_allclose(dep._rig.model.body_inertia[root], mass * 2 * (2 * CUBE_HALF) ** 2 / 12, rtol=1e-4)
    mujoco_pos = dep._rig.prop_pose("cube")[0]
    assert mujoco_pos[2] == pytest.approx(CUBE_HALF, abs=2e-3), "on the floor"
    env_pos = env.entities["cube"].get_pos(relative=True)[0].cpu().numpy()
    np.testing.assert_allclose(env_pos, mujoco_pos, atol=1e-5, err_msg="the policy observes the simulated cube")
    np.testing.assert_allclose(dep._rig.prop_pose("table")[0], table, atol=1e-9)


def test_a_prop_mujoco_cannot_stand_in_for_is_refused_naming_it_and_the_flag_that_leaves_it_out(tmp_path):
    """A sphere has no MuJoCo body here, a mesh file loads as no body at all, and a jointed object's joints
    would move in MuJoCo while the env kept them where it spawned them.
    """
    from eden.options.entities import EntityOptions, RobotOptions
    from eden.options.materials import KinematicMaterialOptions
    from tests.support.envs import build_env

    (tmp_path / "shell.obj").write_text(
        "v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nv 0 0 0.1\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n"
    )
    (tmp_path / "drawer.xml").write_text(TWO_LINK_MJCF)
    env = build_env(
        _config(
            tmp_path,
            ball=SphereOptions(radius=0.05, default_root_pos=(0.0, 1.5, 0.5)),
            shell=EntityOptions(file=str(tmp_path / "shell.obj"), default_root_pos=(0.0, -1.5, 0.5)),
            drawer=RobotOptions(
                file=str(tmp_path / "drawer.xml"),
                dofs_name=["j1", "j2"],
                default_root_pos=(1.5, 1.5, 0.0),
                is_fixed_base=True,
            ),
            marker=SphereOptions(radius=0.02, material=KinematicMaterialOptions(), default_root_pos=(0.5, 0.0, 0.5)),
        ),
        reset=True,
    )
    with deployments(env) as make, pytest.raises(NotImplementedError) as refused:
        make(MujocoCompositeDeployment)
    message = str(refused.value)
    assert "'ball': 'SphereOptions" in message and "'shell': 'a mesh file" in message
    assert "'drawer': \"articulated by joints ['j1', 'j2']" in message
    assert "`--mujoco_props cube table`" in message, "the fix, typeable"
    assert "marker" not in message, "a kinematic goal marker is never a prop: there is nothing to simulate"
    with deployments(env) as make, pytest.raises(NotImplementedError) as refused:
        make(MujocoCompositeDeployment, props=("ball",))
    assert "simulate the rig alone, `--mujoco_props` with no names (props=())" in str(refused.value), (
        "no other prop left to name: the bare flag, not an empty-looking one"
    )
    with deployments(env) as make, pytest.raises(NotImplementedError, match=r"'marker': 'kinematic"):
        make(MujocoCompositeDeployment, props=("marker",))
    with deployments(env) as make:
        dep = make(MujocoCompositeDeployment, props=("cube",))
        assert dep._prop_names == ["cube"]
        assert [name for name, _ in kinematic_markers(dep._env)] == ["marker"], "what a recording draws instead"


def test_a_part_cannot_run_outside_a_rig(env, deploy):
    """Its wire is the rig's model; on its own there is nothing to send to."""
    part = deploy().parts["hand"]
    part.bind(None)
    with pytest.raises(RuntimeError, match="built and connected by MujocoCompositeDeployment"):
        part.read_state()


def test_visual_meshes_stay_for_the_viewer_massless_and_only_an_undecodable_one_is_dropped(tmp_path):
    """Dropping every non-colliding geom left the iiwa as collision primitives in a group the viewer hides.

    So the arm vanished from ``--vis`` while the hand, whose collision geoms are in group 0, floated in the air.
    """
    import mujoco

    from eden.extensions.deployment.utils.mujoco_rig import _load

    (tmp_path / "shell.obj").write_text(
        "v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nv 0 0 0.1\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n"
    )
    (tmp_path / "ascii.stl").write_text("solid s\nendsolid s\n")  # MuJoCo decodes binary STL only
    (tmp_path / "link.xml").write_text(
        """<mujoco><asset><mesh file="shell.obj"/><mesh file="ascii.stl"/></asset><worldbody><body name="link">
             <geom type="box" size="0.05 0.05 0.05" density="1000"/>
             <geom type="mesh" mesh="shell" contype="0" conaffinity="0" group="2" density="1000"/>
             <geom type="mesh" mesh="ascii" contype="0" conaffinity="0" group="2"/>
           </body></worldbody></mujoco>"""
    )
    model = _load(str(tmp_path / "link.xml")).compile()

    assert [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, i) for i in range(model.nmesh)] == ["shell"]
    assert sorted(model.geom_group.tolist()) == [0, 2], "the decodable visual stays for the viewer"
    assert model.body_mass[1] == pytest.approx(1000 * 0.1**3), "and adds no mass: the box's alone"


@pytest.fixture(scope="module")
def floating_env(tmp_path_factory):
    """The humanoid-shaped rig with its root free -- an Oli's base -- and a cube to observe."""
    from eden.utils.configs import SceneOptions
    from tests.support.envs import build_env

    config = _humanoid_config(tmp_path_factory.mktemp("floating_rig"))
    rig = config.scene_options.robot
    # The two-link arm's root body, given the free joint a floating base needs.
    Path(rig.robot.file).write_text(
        TWO_LINK_MJCF.replace('<body name="base" pos="0 0 1.0">', '<body name="base" pos="0 0 1.0">\n<freejoint/>')
    )
    rig.robot = rig.robot.model_copy(update={"is_fixed_base": False, "default_root_pos": (0.0, 0.0, 1.0)})
    scene = SceneOptions(robot=rig, cube=BoxOptions(size=(2 * CUBE_HALF,) * 3, default_root_pos=CUBE_SPAWN))
    return build_env(config.model_copy(update={"scene_options": scene}), reset=True)


def test_a_floating_root_reads_its_base_as_a_robot_would_and_props_turn_with_its_yaw_alignment(floating_env):
    """The root's IMU and ground-truth base reach the observation; a yaw captured at START turns the root's
    links into the trained heading, so a prop must turn with them or its robot-relative pose is off by it.
    """
    import math

    from eden.utils.geom import transform_by_quat

    with deployments(floating_env) as make:
        dep = make(MujocoCompositeDeployment, connect=True)  # the kinematics check passes for a free root too
        root = dep.parts["robot"]
        assert root._publishes_base_lin_vel()
        state = root.read_state()
        np.testing.assert_allclose(state.extra["base_pos"], [0.0, 0.0, 1.0], atol=1e-5)
        # Nothing holds the free root up, so it is in free fall: an accelerometer, not a gravity vector, reads 0.
        assert float(np.linalg.norm(state.base_lin_acc)) == pytest.approx(0.0, abs=1e-6)
        assert state.base_ang_vel.shape == (3,)

        dep.reset()
        dep._state_entity.set_yaw_offset(math.pi / 2)
        dep.step(None)
        cube = torch.as_tensor(dep._rig.prop_pose("cube")[0], dtype=torch.float32)[None]
        remove = torch.tensor([[math.cos(-math.pi / 4), 0.0, 0.0, math.sin(-math.pi / 4)]])
        observed = floating_env.entities["cube"].get_pos(relative=True)[0].cpu()
        torch.testing.assert_close(observed, transform_by_quat(cube, remove)[0], atol=1e-5, rtol=0)


def test_under_the_euler_integrator_the_armature_folds_the_damping_once(tmp_path):
    """Genesis folds ``kd * substep_dt`` into the built armature under euler; the rig folds ``kd * sim_dt`` itself."""
    from tests.support.envs import build_env

    config = _config(tmp_path)
    config = config.model_copy(update={"env_options": config.env_options.model_copy(update={"integrator": "euler"})})
    env = build_env(config, reset=True)
    with deployments(env) as make:
        dep = make(MujocoCompositeDeployment, connect=True)
        arm = dep.parts["robot"]._state_entity._sim_entity
        configured = arm.default_dofs_armature[0].cpu().numpy()
        built = arm.get_dofs_armature().reshape(-1).cpu().numpy()
        assert (built > configured).all(), "Genesis did fold the damping in under euler"
        rig = dep._rig
        expected = configured + dep.default_dof_kd[:2] * rig.sim_dt
        np.testing.assert_allclose(rig.model.dof_armature[rig._v_idx[rig._slices["robot"]]], expected, rtol=1e-6)


NUT_URDF = """<robot name="nut"><link name="nut">
  <collision><geometry><box size="0.04 0.03 0.02"/></geometry></collision>
  <inertial><origin xyz="0.03 0.01 -0.02" rpy="0.3 -0.2 0.5"/><mass value="0.2"/>
    <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.003" iyz="0" izz="0.004"/></inertial>
</link></robot>"""

#: An MJCF object that declares its own free joint, as MuJoCo-authored objects do.
CAN_MJCF = """<mujoco><worldbody><body name="can"><freejoint/>
  <geom type="cylinder" size="0.03 0.06" mass="0.3"/></body></worldbody></mujoco>"""


@pytest.fixture(scope="module")
def props_env(tmp_path_factory):
    from eden.options.entities import EntityOptions
    from tests.support.envs import build_env

    assets = tmp_path_factory.mktemp("props")
    (assets / "nut.urdf").write_text(NUT_URDF)
    (assets / "can.xml").write_text(CAN_MJCF)
    return build_env(
        _config(
            assets,
            nut=EntityOptions(file=str(assets / "nut.urdf"), default_root_pos=(0.5, 0.8, 0.3)),
            can=EntityOptions(file=str(assets / "can.xml"), default_root_pos=(-0.5, 0.8, 0.3)),
        ),
        reset=True,
    )


def _inertia_tensor(model, body) -> np.ndarray:
    import mujoco

    axes = np.zeros(9)
    mujoco.mju_quat2Mat(axes, model.body_iquat[body])
    axes = axes.reshape(3, 3)
    return axes @ np.diag(model.body_inertia[body]) @ axes.T


def test_a_prop_carries_the_inertia_the_env_built_it_with_in_its_assets_own_frame(props_env):
    """Copied link by link from the env, not re-derived by MuJoCo: an explicit, off-centre, rotated URDF inertial
    comes out as MuJoCo reads it from the asset itself -- whatever frame Genesis re-origins a free body onto.
    """
    import mujoco

    from eden.extensions.deployment.utils.mujoco_rig import _load

    with deployments(props_env) as make:
        rig = make(MujocoCompositeDeployment, connect=True)._rig
        nut = rig._prop_body["nut"]

        spec = _load(rig.props["nut"].asset_path)
        spec.worldbody.bodies[0].add_freejoint()
        alone = spec.compile()  # the URDF's inertial as MuJoCo reads it, in the asset's frame
        assert rig.model.body_mass[nut] == pytest.approx(0.2)
        np.testing.assert_allclose(rig.model.body_ipos[nut], alone.body_ipos[1], atol=1e-6)
        np.testing.assert_allclose(_inertia_tensor(rig.model, nut), _inertia_tensor(alone, 1), atol=1e-9)
        assert mujoco.mj_name2id(rig.model, mujoco.mjtObj.mjOBJ_BODY, "nut/nut") == nut


def test_a_prop_is_free_or_fixed_as_the_env_built_it_whatever_free_joint_its_asset_declares(props_env):
    """An MJCF object's own ``<freejoint/>`` was a second free joint on a free prop: MuJoCo refused to compile it."""
    import mujoco

    with deployments(props_env) as make:
        dep = make(MujocoCompositeDeployment, connect=True)
        model, body = dep._rig.model, dep._rig._prop_body
        free = [int(model.jnt_bodyid[j]) for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        assert free.count(body["can"]) == 1, "the asset's free joint, replaced by the one the env built"
        assert body["table"] not in free and not props_env.entities["table"].free_joints, "fixed, as the env built it"
        assert model.body_mass[body["can"]] == pytest.approx(float(props_env.entities["can"].get_mass().reshape(-1)[0]))

        dep.reset()
        for _ in range(30):
            _step(dep, [0.0, 0.0], [HAND_DEFAULT, HAND_DEFAULT])
        assert dep._rig.prop_pose("can")[0][2] < 0.3 - 0.05, "the free one falls"


STIFF = (0.005, 1.0, 0.95, 0.999, 0.0005, 0.5, 2.0)  # DexLift's set_contact_sol_params


def test_contact_solver_parameters_reach_mujoco_as_the_env_built_them(env, eden_caplog):
    """A grasp the task stiffened (a ``set_contact_sol_params`` startup event) must stay stiff in MuJoCo.

    Copying friction alone left every contact at MuJoCo's default impedance, half as stiff as Genesis'
    default and ~50x softer than DexLift's stiffened cube: a policy that holds in Genesis let it slip out.
    Each geom carries its value as stated, unfloored: both engines floor only a pair's mix.
    """
    import mujoco

    from eden.extensions.deployment.utils.mujoco_rig import _solref_solimp

    with deployments(env) as make:
        make(MujocoCompositeDeployment, connect=True)
    assert "floors a contact's" not in eden_caplog.text, "the env's defaults sit at Genesis' floor: nothing to warn"

    cube_geoms = env.entities["cube"]._entity.geoms
    before = [geom.get_sol_params().clone() for geom in cube_geoms]
    for geom in cube_geoms:
        geom.set_sol_params(torch.tensor(STIFF))
    env_options = env.config.env_options
    substep = env_options.sim_dt / env_options.sim_substeps
    assert STIFF[0] < 2.0 * substep, "the case a per-geom floor would distort: below Genesis' floor"
    try:
        with deployments(env) as make:
            dep = make(MujocoCompositeDeployment, connect=True)
            model = dep._rig.model
            cube = int(model.body_geomadr[dep._rig.prop_body_id("cube")])
            np.testing.assert_allclose(model.geom_solref[cube], STIFF[:2], err_msg="as stated, not floored")
            np.testing.assert_allclose(model.geom_solimp[cube], STIFF[2:])

            # The hand keeps the env's default, as stated too.
            hand = dep.parts["hand"]._state_entity._sim_entity._entity  # env.entities["hand"] is the deployed state
            for link in hand.links:
                if not link.geoms:
                    continue
                body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"hand/{link.name}")
                solref, solimp = _solref_solimp(link.geoms[0].get_sol_params().cpu().double().numpy())
                geom = int(model.body_geomadr[body])
                np.testing.assert_allclose(model.geom_solref[geom], solref, rtol=1e-6, err_msg=link.name)
                np.testing.assert_allclose(model.geom_solimp[geom], solimp, rtol=1e-6, err_msg=link.name)
        # Below Genesis' floor, which MuJoCo puts at twice its own (smaller) timestep: said, with the fix.
        assert f"twice its substep ({2.0 * substep:g} s)" in eden_caplog.text
        assert f"sim_dt={substep:g}" in eden_caplog.text, "the fix, typeable"
    finally:
        for geom, params in zip(cube_geoms, before, strict=True):
            geom.set_sol_params(params)


GROUND = (0.03, 1.0, 0.9, 0.95, 0.001, 0.5, 2.0)


def _floor(model) -> tuple[int, np.ndarray]:
    """The MuJoCo floor geom and the normal it faces (its local +z) in the world."""
    import mujoco

    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "__eden_floor__")
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[floor])
    return floor, rotation.reshape(3, 3)[:, 2]


TILT = 0.1  # rad about +x, which turns +z to (0, -sin, cos)


@pytest.mark.parametrize(
    "tilted",
    [{"normal": (0.0, -np.sin(TILT), np.cos(TILT))}, {"default_root_quat": (np.cos(TILT / 2), np.sin(TILT / 2), 0, 0)}],
    ids=["by-its-normal", "by-its-root-quat"],
)
def test_the_floor_stands_where_the_ground_plane_does_with_its_contact_as_the_env_built_it(tmp_path, tilted):
    """The MuJoCo floor takes the ground plane's pose, raised and tilted alike, and its friction and parameters."""
    from eden.extensions.deployment.utils.mujoco_rig import _solref_solimp
    from eden.options.entities import PlaneOptions
    from tests.support.envs import build_env

    normal = (0.0, -np.sin(TILT), np.cos(TILT))
    plane = PlaneOptions(
        default_root_pos=(0.0, 0.0, 0.05),
        default_geoms_friction={"*": 0.6},
        default_geoms_sol_params={"*": GROUND},
        **tilted,
    )
    env = build_env(_config(tmp_path, plane=plane), reset=True)
    with deployments(env) as make:
        model = make(MujocoCompositeDeployment, connect=True)._rig.model
    floor, facing = _floor(model)
    np.testing.assert_allclose(model.geom_pos[floor], (0.0, 0.0, 0.05), atol=1e-6, err_msg="raised as the plane")
    np.testing.assert_allclose(facing, normal, atol=1e-6, err_msg="tilted as the plane")
    assert model.geom_friction[floor, 0] == pytest.approx(0.6)
    solref, solimp = _solref_solimp(np.array(GROUND))
    np.testing.assert_allclose(model.geom_solref[floor], solref, rtol=1e-6)
    np.testing.assert_allclose(model.geom_solimp[floor], solimp, rtol=1e-6)


def test_without_one_ground_the_floor_is_mujocos_plain_one(env, tmp_path, eden_caplog):
    """No ground to stand for, or several the one floor cannot all be: MuJoCo's at z = 0, the latter with a caveat
    the rig says once its floor is in the model.
    """
    from eden.options.entities import PlaneOptions
    from tests.support.envs import build_env

    assert read_floor(env) == MujocoFloor(), "a scene with no ground has nothing to say: no caveat either"

    two = _config(
        tmp_path,
        plane=PlaneOptions(default_geoms_friction={"*": 0.6}),
        basement=PlaneOptions(default_root_pos=(0.0, 0.0, -1.0)),
    )
    two_env = build_env(two, reset=True)
    floor = read_floor(two_env)
    assert dataclasses.replace(floor, caveat=None) == MujocoFloor()
    assert "the scene has 2 grounds" in floor.caveat
    assert "grounds" not in eden_caplog.text, "read_floor says nothing itself"
    with deployments(two_env) as make:
        make(MujocoCompositeDeployment, connect=True)
    assert f"[deploy] MuJoCo: {floor.caveat}" in eden_caplog.text


def test_read_floor_takes_a_terrains_contact_but_not_its_shape(tmp_path):
    """A height field is no plane: the floor read is MuJoCo's flat one at z = 0, with the terrain's contact and a caveat.

    A rig's floor takes all of it; a single robot's keeps only the friction (``test_mujoco.py``).
    """
    from eden.options.entities import TerrainOptions
    from tests.support.envs import build_env

    terrain = TerrainOptions(
        subterrain_size=(4.0, 4.0),
        horizontal_scale=0.25,
        default_geoms_friction={"*": 0.6},
        default_geoms_sol_params={"*": GROUND},
    )
    floor = read_floor(build_env(_config(tmp_path, terrain=terrain), reset=True))
    assert (floor.pos, floor.normal) == (MujocoFloor().pos, MujocoFloor().normal)
    assert floor.friction == pytest.approx(0.6)
    np.testing.assert_allclose(floor.sol_params, GROUND, rtol=1e-6)
    assert "the ground 'terrain' is a terrain" in floor.caveat


_HAND_MJCF = """<mujoco><worldbody><body name="palm"><geom type="box" size=".05 .05 .02" mass="0.2"/>
  <body name="finger" pos="0 0 .05"><joint name="j" type="hinge" axis="1 0 0"/>
    <geom name="phalanx" type="capsule" size=".01 .02" mass=".01" solref="{a} 1" solimp=".9 .95 .001 .5 2"/>
    <geom name="nail" type="box" size=".005 .005 .005" pos="0 0 .02" mass=".001" solref="{b} 1" solimp=".9 .95 .001 .5 2"/>
    <body name="pad" pos="0 0 .03"><geom type="sphere" size=".01" mass=".001" priority="1" solmix="2"/></body>
  </body></body></worldbody></mujoco>"""


def _hand_rig(tmp_path, link_sol_params, *, a=0.02, b=0.02, props=(), **rig):
    from eden.extensions.deployment.utils.mujoco_rig import MujocoRig, RigPart

    tmp_path.mkdir(exist_ok=True)
    asset = tmp_path / "hand.xml"
    asset.write_text(_HAND_MJCF.format(a=a, b=b))
    part = RigPart(
        name="hand",
        asset_path=str(asset),
        dofs_name=["j"],
        pos=np.array([0.0, 0.0, 0.5]),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        link_sol_params=link_sol_params,
    )
    return MujocoRig([part], list(props), sim_dt=0.002, **rig)


def test_a_contact_mixes_in_mujoco_as_in_genesis(tmp_path, eden_caplog):
    """The mean of both geoms' stated values, floored only as a pair; a merged pad takes its link's; no priority."""
    import mujoco

    from eden.extensions.deployment.utils.mujoco_rig import RigProp

    soft_cube = (0.001, 1.0, 0.9, 0.95, 0.001, 0.5, 2.0)  # below MuJoCo's 2 * 0.002 floor on its own
    rig = _hand_rig(
        tmp_path,
        {"finger": np.array([STIFF, STIFF])},
        floor=MujocoFloor(sol_params=GROUND),
        props=[
            RigProp(
                "cube",
                np.array([1.0, 0.0, 0.03]),
                np.array([1.0, 0.0, 0.0, 0.0]),
                True,
                half_size=np.full(3, 0.025),
                sol_params=np.array([soft_cube]),
            )
        ],
    )
    model, data = rig.model, rig.data
    pad = int(model.body_geomadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand/pad")])
    np.testing.assert_allclose(model.geom_solref[pad], STIFF[:2], err_msg="a fixed child carries its link's")
    assert not model.geom_priority.any() and (model.geom_solmix == 1.0).all(), "no geom outranks another"
    assert "hand/pad" in eden_caplog.text, "an asset's priority is reset aloud"

    for _ in range(20):
        mujoco.mj_step(model, data)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "__eden_floor__")
    cube = int(model.body_geomadr[rig.prop_body_id("cube")])
    contacts = [c for c in data.contact[: data.ncon] if {int(c.geom1), int(c.geom2)} == {floor, cube}]
    assert contacts, "the cube rests on the floor"
    # The mean of the stated 0.001 and the ground's 0.03; a per-geom floor would have made it (0.004 + 0.03) / 2.
    assert contacts[0].solref[0] == pytest.approx(0.5 * (soft_cube[0] + GROUND[0]))

    no_floor = _hand_rig(tmp_path, {}, floor=None).model
    assert mujoco.mj_name2id(no_floor, mujoco.mjtObj.mjOBJ_GEOM, "__eden_floor__") == -1, "floor=None adds none"


def test_per_geom_values_an_asset_states_stay_its_own_and_a_mismatch_is_named(tmp_path, eden_caplog):
    """Genesis splits and convexifies meshes, so geoms cannot be paired across engines: the asset's values stay."""
    import mujoco

    solimp = (0.9, 0.95, 0.001, 0.5, 2.0)
    stated = np.array([(0.02, 1.0, *solimp), (0.03, 1.0, *solimp)])
    rig = _hand_rig(tmp_path / "same", {"finger": stated}, a=0.02, b=0.03)
    model = rig.model
    by_name = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"hand/{name}") for name in ("phalanx", "nail")}
    assert [model.geom_solref[g][0] for g in by_name.values()] == pytest.approx([0.02, 0.03])
    assert "contact parameter sets" not in eden_caplog.text, "the same set of values in both engines"

    _hand_rig(tmp_path / "other", {"finger": stated}, a=0.02, b=0.05)
    assert "the geoms of 'hand/finger' carry 2 different contact parameter sets" in eden_caplog.text


def test_a_body_inertia_mujoco_cannot_compile_is_made_isotropic_with_its_mass_and_trace(tmp_path):
    """The XHand1 URDFs give a dozen dummy links every inertia entry 1e-11: a rank-1 tensor.

    Genesis builds it and MuJoCo refuses it ("inertia must have positive eigenvalues"), so no XHand rig could
    run ``--mujoco`` at all. MuJoCo refuses principal moments that break A + B >= C as well, full or diagonal.
    """
    import mujoco

    from eden.extensions.deployment.utils.mujoco_rig import _inertia_refused, _load, _repair_inertias

    (tmp_path / "hand.urdf").write_text(
        """<robot name="hand"><link name="palm"><inertial><mass value="0.3"/>
             <inertia ixx="1e-4" iyy="1e-4" izz="1e-4" ixy="0" ixz="0" iyz="0"/></inertial>
             <collision><geometry><box size="0.05 0.05 0.05"/></geometry></collision></link>
           <link name="ee_link"><inertial><mass value="5.65e-6"/>
             <inertia ixx="1e-11" iyy="1e-11" izz="1e-11" ixy="1e-11" ixz="1e-11" iyz="1e-11"/></inertial></link>
           <joint name="fix" type="fixed"><parent link="tip"/><child link="ee_link"/></joint>
           <joint name="spin" type="revolute"><parent link="palm"/><child link="tip"/><axis xyz="0 0 1"/>
             <limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
           <link name="tip"><inertial><mass value="0.01"/>
             <inertia ixx="1e-6" iyy="1e-6" izz="1e-6" ixy="0" ixz="0" iyz="0"/></inertial></link></robot>"""
    )
    spec = _load(str(tmp_path / "hand.urdf"))
    with pytest.raises(ValueError, match="positive eigenvalues"):
        spec.copy().compile()

    _repair_inertias(spec, str(tmp_path / "hand.urdf"))
    model = spec.compile()

    # The palm, jointless, is the world's in MuJoCo; the dummy link is fused into the moving tip it is fixed to.
    tip = model.nbody - 1
    assert float(model.body_mass[tip]) == pytest.approx(0.01 + 5.65e-6), "its mass kept"
    assert float(model.body_inertia[tip].sum()) == pytest.approx(3e-6 + 3e-11, rel=1e-6), "and its trace"

    # Positive definite but not a rigid body's (a full URDF tensor), and an MJCF's principal moments alike.
    assert _inertia_refused(np.array([1e-10, 1e-12, 1e-12]), full=True) == "A + B < C"
    assert _inertia_refused(np.array([1.0, 1.0, 2.0]), full=False) is None, "MuJoCo's own edge: equality compiles"
    (tmp_path / "rod.xml").write_text(
        """<mujoco><worldbody><body name="base"><geom type="box" size=".05 .05 .05"/>
             <body name="rod"><joint type="hinge"/><inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-6 1e-6"/>
             </body></body></worldbody></mujoco>"""
    )
    spec = _load(str(tmp_path / "rod.xml"))
    with pytest.raises(ValueError, match="A \\+ B >= C"):
        spec.copy().compile()
    _repair_inertias(spec, str(tmp_path / "rod.xml"))
    rod = spec.compile()
    rod_id = mujoco.mj_name2id(rod, mujoco.mjtObj.mjOBJ_BODY, "rod")
    np.testing.assert_allclose(rod.body_inertia[rod_id], [1.02e-4 / 3] * 3, err_msg="isotropic, the trace kept")


def test_a_prop_camera_reports_late_noisy_poses_held_between_frames():
    """What an AprilTag observer gives the policy: a pose some ticks old, at the camera's rate, never exact."""
    from eden.extensions.deployment.mujoco_composite import _PropCamera

    identity = np.array([1.0, 0.0, 0.0, 0.0])
    camera = _PropCamera(pos_noise=0.0, rot_noise=0.0, hz=25.0, latency_ticks=2, control_dt=0.02, seed=0)
    seen = [camera.observe("cube", t, np.array([float(t), 0.0, 0.0]), identity)[0][0] for t in range(10)]
    # A frame every other tick (25 Hz at a 50 Hz loop), each of the pose two ticks earlier.
    assert seen == [0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0]
    # Read again at the same tick (a control tick that sent nothing: the physics stood still), nothing moves on.
    assert camera.observe("cube", 9, np.array([99.0, 0.0, 0.0]), identity)[0][0] == 6.0
    assert camera.observe("cube", 10, np.array([10.0, 0.0, 0.0]), identity)[0][0] == 8.0, "nor did the history"

    def noisy(seed):
        return _PropCamera(pos_noise=0.003, rot_noise=0.035, hz=None, latency_ticks=0, control_dt=0.02, seed=seed)

    camera = noisy(1)
    readings = [camera.observe("cube", t, np.zeros(3), identity) for t in range(4000)]
    positions = np.array([pos for pos, _ in readings])
    quats = np.array([quat for _, quat in readings])
    np.testing.assert_allclose(positions.std(axis=0), 0.003, rtol=0.1)
    angles = 2.0 * np.arccos(np.clip(np.abs(quats[:, 0]), 0.0, 1.0))
    assert np.mean(angles) == pytest.approx(0.035 * np.sqrt(2.0 / np.pi), rel=0.1), "a half-normal angle"
    assert 0.4 < np.mean(quats[:, 0] > 0) < 0.6, "the sign a pose estimator picks is arbitrary"
    np.testing.assert_array_equal(
        noisy(7).observe("cube", 0, np.zeros(3), identity)[0], noisy(7).observe("cube", 0, np.zeros(3), identity)[0]
    )

    exact = _PropCamera(pos_noise=0.0, rot_noise=0.0, hz=None, latency_ticks=0, control_dt=0.02, seed=0)
    pos = np.array([0.1, 0.2, 0.3])
    assert exact.observe("cube", 0, pos, identity)[0] is pos, "off by default: MuJoCo's exact pose, untouched"


def test_the_policy_reads_a_free_props_camera_pose_and_the_exact_one_stays_beside_it(env, deploy):
    dep = deploy(prop_pos_noise=0.01, prop_noise_seed=0)
    dep.reset()
    _step(dep, [0.0, 0.0], [HAND_DEFAULT, HAND_DEFAULT])
    extra = dep.state.extra

    np.testing.assert_allclose(extra["cube.true_pos"], dep._rig.prop_pose("cube")[0], err_msg="MuJoCo's, exactly")
    error = np.abs(extra["cube.pos"] - extra["cube.true_pos"]).max()
    assert 0.0 < error < 0.1, "what the policy reads is the camera's estimate"
    seen_by_env = env.entities["cube"].get_pos(relative=True)[0].cpu().numpy()
    np.testing.assert_allclose(seen_by_env, extra["cube.pos"], atol=1e-5, err_msg="the env's terms read the estimate")
    assert "table.true_pos" not in extra, "a fixed prop is not observed through the camera"
    np.testing.assert_allclose(extra["table.pos"], dep._rig.prop_pose("table")[0])


def test_a_camera_model_run_length_or_timestep_that_cannot_be_is_refused_and_a_latency_is_rounded_aloud(
    env, eden_caplog
):
    for bad in ({"prop_pos_noise": -0.1}, {"prop_latency_s": -0.01}, {"prop_camera_hz": 0.0}, {"policy_seconds": 0.0}):
        with deployments(env) as make, pytest.raises(ValueError, match=next(iter(bad))):
            make(MujocoCompositeDeployment, **bad)

    with deployments(env) as make:
        control_dt = make(MujocoCompositeDeployment).control_dt
    # Rounding 2.5 steps a tick would run the physics ahead of the policy's clock; the message names a fit.
    fit = re.escape(f"--mujoco_sim_dt {control_dt / 3:g}")
    with deployments(env) as make, pytest.raises(ValueError, match=f"does not divide the control period.*{fit}"):
        make(MujocoCompositeDeployment, sim_dt=control_dt / 2.5)
    with deployments(env) as make:
        assert make(MujocoCompositeDeployment, sim_dt=control_dt / 4, connect=True)._n_substeps == 4
    with deployments(env) as make:
        assert make(MujocoCompositeDeployment, prop_latency_s=2 * control_dt)._camera.latency_ticks == 2
    assert "prop_latency_s" not in eden_caplog.text, "a whole number of ticks is taken as given"
    with deployments(env) as make:
        assert make(MujocoCompositeDeployment, prop_latency_s=2.5 * control_dt)._camera.latency_ticks == 3, "half up"
    assert "the camera's frames arrive 3 ticks" in eden_caplog.text


def test_policy_seconds_counts_the_ticks_the_policy_drove_since_its_latest_start(env, deploy):
    """A base option, every simulator's: the run ends at the tick after the last one the policy drove, which is
    observed and recorded whole.
    """
    dep = deploy(policy_seconds=0.2)
    ticks = round(0.2 / dep.control_dt)
    dep.reset()
    for _ in range(3):
        _step(dep, [0.0, 0.0], [HAND_DEFAULT, HAND_DEFAULT])
    assert dep._policy_ticks == 0, "nothing counts before START"
    dep.request_start()
    _step(dep, [0.0, 0.0], [HAND_DEFAULT, HAND_DEFAULT])
    assert dep._policy_ticks == 0, "START's tick holds the rig: the policy has not driven it yet"
    for _ in range(ticks - 1):
        _step(dep, [0.0, 0.0], [HAND_DEFAULT, HAND_DEFAULT])
    dep.request_stop()
    _step(dep, [0.0, 0.0], [HAND_DEFAULT, HAND_DEFAULT])
    assert dep._policy_ticks == 0, "a STOP one tick short: a later START runs policy_seconds afresh"

    dep.request_start()
    for _ in range(ticks + 1):  # START's hold, then every driven tick
        _step(dep, [0.0, 0.0], [HAND_DEFAULT, HAND_DEFAULT])
    assert dep._policy_ticks == ticks
    sim_time = dep._rig.data.time
    with pytest.raises(KeyboardInterrupt):
        _step(dep, [0.0, 0.0], [HAND_DEFAULT, HAND_DEFAULT])
    assert dep._rig.data.time == sim_time, "the run ends before another command is sent"


def test_the_rollout_is_recorded_from_start_on_without_a_gap(env, deploy, tmp_path):
    video = tmp_path / "rollout.mp4"
    try:
        dep = deploy(record_video=str(video))
    except RuntimeError as e:
        if "offscreen renderer" in str(e):
            pytest.skip(f"no OpenGL context for MuJoCo here: {e}")
        raise
    dep.reset()
    for _ in range(3):
        _step(dep, [0.0, 0.0], [HAND_DEFAULT, HAND_DEFAULT])
    assert dep._video.frames == 0, "nothing is recorded before START"
    dep.request_start()
    for _ in range(4):
        _step(dep, [0.0, 0.0], [HAND_DEFAULT, HAND_DEFAULT])
    dep.request_stop()
    for _ in range(2):
        _step(dep, [0.0, 0.0], [HAND_DEFAULT, HAND_DEFAULT])
    assert dep._video.frames == 6, "a STOP's holds are recorded too: the video never skips"
    dep.close()
    assert video.stat().st_size > 0


def test_a_markers_colour_is_its_own_else_its_surfaces():
    from eden.extensions.deployment.utils.mujoco_video import marker_color as _marker_color
    from eden.options.surfaces import SurfaceOptions

    assert _marker_color(SphereOptions(radius=0.02, color=(1.0, 0.0, 0.0))) == (1.0, 0.0, 0.0)
    assert _marker_color(SphereOptions(radius=0.02, surface=SurfaceOptions(color=(0.0, 0.0, 1.0)))) == (0.0, 0.0, 1.0)
    assert _marker_color(SphereOptions(radius=0.02)) == (0.1, 0.9, 0.1, 0.6)


def test_target_lpf_low_passes_each_parts_targets_as_its_hardware_backend_does(env, deploy):
    """``--target_lpf`` was applied by every hardware backend's wire and by no MuJoCo one: a sim run ignored it."""
    from eden.extensions.deployment.utils.state import RobotCommand

    dep = deploy(target_lpf=0.5)
    dep.reset()
    dep.request_start()
    sent = []
    for arm in ([0.0, 0.0], [0.6, -0.4], [0.6, -0.4]):
        _step(dep, arm, [HAND_DEFAULT, HAND_DEFAULT])
        sent.append(dep._rig._target_pos[dep._rig._slices["robot"]].copy())
    commanded = env.action_manager.terms["arm"].processed_action[0].cpu().numpy()
    np.testing.assert_allclose(sent[2], 0.5 * commanded + 0.5 * sent[1], rtol=1e-6, err_msg="a*new + (1-a)*prev")
    assert np.abs(sent[2] - commanded).max() > 0.01, "the filtered target lags the command"

    # What was sent is what the payload holds afterwards, as on hardware: --debug_record records the filtered target.
    target = np.asarray(dep.default_dof_pos, dtype=np.float64) + 0.2
    zeros = np.zeros_like(target)
    payload = RobotCommand(dofs_pos=target.copy(), dofs_vel=zeros, dofs_torque=zeros)
    dep.send_payload(payload)
    rig = dep._rig
    for name, part_slice in dep._part_slices.items():
        np.testing.assert_allclose(payload.dofs_pos[part_slice], rig._target_pos[rig._slices[name]], err_msg=name)
    assert not np.allclose(payload.dofs_pos, target), "filtered, not the raw command"
