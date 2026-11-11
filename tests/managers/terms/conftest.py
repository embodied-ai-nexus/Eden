"""Built scenes and helpers shared by the manager-term tests.

Scenes are module-scoped: a build costs seconds, so each test module builds the ones it asks for
once. A test that moves an entity puts it back (or calls ``env.reset()``) before it returns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from tests.support.envs import build_env

if TYPE_CHECKING:
    from eden.envs.base import EnvBase

NUM_ENVS = 2

#: Low, uniform gains on every Panda DOF. The stock ones (kp 4500 on the shoulder) saturate the
#: ±12..87 N·m force range at a few centiradians of error, so ``get_dofs_control_force`` could not
#: show where a controller put its target; at kp = 10 it reads ``10 * error`` up to ~1 rad.
PANDA_KP, PANDA_KD = 10.0, 1.0
PANDA_ARM = [f"joint{i}" for i in range(1, 8)]

#: A fixed hub on an unlimited hinge (``spin``) carrying a flap on a hinge limited to ±1 rad (``flap``).
WHEEL_MJCF = """
<mujoco model="wheel">
  <compiler angle="radian"/>
  <worldbody>
    <body name="hub">
      <joint name="spin" type="hinge" axis="0 0 1" limited="false"/>
      <geom type="cylinder" size="0.1 0.02" mass="1.0"/>
      <body name="flap_link" pos="0.2 0 0">
        <joint name="flap" type="hinge" axis="0 1 0" range="-1 1" limited="true"/>
        <geom type="box" size="0.05 0.05 0.01" mass="0.2"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

#: A slab carrying a thin mast and a head, jointless so the three links stay welded. Only ``foot``
#: touches the plane, which is what lets a link *subset* be told apart from the whole entity.
STACK_MJCF = """
<mujoco model="stack">
  <worldbody>
    <body name="foot">
      <freejoint/>
      <geom name="g_foot" type="box" size="0.20 0.20 0.04" mass="4.0"/>
      <body name="shin" pos="0 0 0.20">
        <geom name="g_shin" type="box" size="0.03 0.03 0.15" mass="0.3"/>
        <body name="torso" pos="0 0 0.30">
          <geom name="g_torso" type="box" size="0.08 0.08 0.10" mass="0.5"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
#: The stack's weight, 4.8 kg * 9.81 N/kg: what ``foot`` carries once settled.
STACK_WEIGHT_N = 47.088

#: Where ``sloped_env``'s box hangs: 2 m up, over terrain that has dropped well below z = 0.
SLOPE_BOX_XY, SLOPE_BOX_Z = (3.0, 3.0), 2.0

#: A free two-link body whose geoms sit above its base frame, so the solver frame Genesis re-origins
#: the base onto (the CoM) lies above the authored frame ``default_root_pos`` names.
PROBE_MJCF = """
<mujoco model="probe">
  <worldbody>
    <body name="base">
      <freejoint/>
      <geom name="g_base" type="box" size="0.05 0.05 0.05" pos="0 0 0.1" mass="1.0"/>
      <body name="tip" pos="0 0 0.3">
        <geom name="g_tip" type="box" size="0.02 0.02 0.02" mass="0.1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
PROBE_POS = (-3.0, -3.0, 2.0)


def write_stack_urdf(path):
    """A fixed two-link stack, ``bottom`` then ``top`` (0.3 x 0.3 x 0.1 m slabs, welded), written to ``path``."""
    inertia = 'ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"'

    def link(name, z):
        box = f'<origin xyz="0 0 {z}"/><geometry><box size="0.3 0.3 0.1"/></geometry>'
        return (
            f'<link name="{name}"><inertial><origin xyz="0 0 {z}"/><mass value="1"/><inertia {inertia}/></inertial>'
            f"<visual>{box}</visual><collision>{box}</collision></link>"
        )

    path.write_text(
        f'<robot name="stack">{link("bottom", 0.05)}{link("top", 0.15)}'
        '<joint name="weld" type="fixed"><parent link="bottom"/><child link="top"/></joint></robot>'
    )
    return str(path)


def write_mimic_finger_urdf(path, *, mimic_tag: bool = False):
    """A fixed palm carrying a two-joint finger (``a`` then ``b``) and a one-joint thumb (``c``), written to ``path``.

    ``b`` is the natural mimic follower of ``a``. With ``mimic_tag`` the URDF says so — ``<mimic joint="a"
    multiplier="2" offset="0.1"/>`` on ``b`` — which Genesis turns into a joint-equality constraint.
    """
    inertia = 'ixx="1e-4" iyy="1e-4" izz="1e-4" ixy="0" ixz="0" iyz="0"'

    def link(name, mass="0.05"):
        box = '<origin xyz="0 0 0.02"/><geometry><box size="0.01 0.01 0.04"/></geometry>'
        return (
            f'<link name="{name}"><inertial><origin xyz="0 0 0.02"/><mass value="{mass}"/><inertia {inertia}/>'
            f"</inertial><visual>{box}</visual><collision>{box}</collision></link>"
        )

    def joint(name, parent, child, z, extra=""):
        return (
            f'<joint name="{name}" type="revolute"><parent link="{parent}"/><child link="{child}"/>'
            f'<origin xyz="0 0 {z}"/><axis xyz="0 1 0"/><limit lower="-1.5" upper="1.5" effort="10" velocity="10"/>'
            f"{extra}</joint>"
        )

    mimic = '<mimic joint="a" multiplier="2" offset="0.1"/>' if mimic_tag else ""
    path.write_text(
        f'<robot name="finger">{link("palm", "0.5")}{link("proximal")}{link("distal")}{link("thumb")}'
        f"{joint('a', 'palm', 'proximal', 0.04)}{joint('b', 'proximal', 'distal', 0.04, mimic)}"
        f"{joint('c', 'palm', 'thumb', 0.04)}</robot>"
    )
    return str(path)


def panda_options(**overrides):
    """A fixed-base Panda with :data:`PANDA_KP` / :data:`PANDA_KD` on every DOF."""
    from eden.options.robots.franka_emika_panda import FrankaEmikaPanda

    params = {"default_dofs_kp": {".*": PANDA_KP}, "default_dofs_kd": {".*": PANDA_KD}, **overrides}
    return FrankaEmikaPanda(**params)


def wheel_options(path, **overrides):
    """The :data:`WHEEL_MJCF` written to ``path``, articulated and actuated at the Panda's gains."""
    from eden.options import EntityOptions

    path.write_text(WHEEL_MJCF)
    params = {
        "file": str(path),
        "is_fixed_base": True,
        "is_articulated": True,
        "is_actuated": True,
        "default_root_pos": (3.0, 0.0, 0.5),
        "dofs_name": ["spin", "flap"],
        "default_dofs_pos": {"spin": 0.0, "flap": 0.0},
        "default_dofs_kp": {".*": PANDA_KP},
        "default_dofs_kd": {".*": PANDA_KD},
        **overrides,
    }
    return EntityOptions(**params)


@pytest.fixture(scope="module")
def panda_env(tmp_path_factory: pytest.TempPathFactory) -> EnvBase:
    """A Panda (``robot``), the :data:`WHEEL_MJCF` (``wheel``) and three free cubes (``cube_a`` … ``cube_c``)
    in the air, never stepped.

    Nothing touches, so every pose is exactly what a setter wrote.
    """
    from eden.options import BoxOptions, EnvOptions, SceneOptions
    from eden.utils.configs import EdenConfig

    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=NUM_ENVS),
            scene_options=SceneOptions(
                robot=panda_options(),
                wheel=wheel_options(tmp_path_factory.mktemp("wheel") / "wheel.xml"),
                cube_a=BoxOptions(size=(0.05, 0.05, 0.05), default_root_pos=(1.0, 1.0, 1.0)),
                cube_b=BoxOptions(size=(0.05, 0.05, 0.05), default_root_pos=(-1.0, 1.0, 1.0)),
                cube_c=BoxOptions(size=(0.05, 0.05, 0.05), default_root_pos=(0.0, -1.0, 1.0)),
            ),
        )
    )


@pytest.fixture(scope="module")
def box_env() -> EnvBase:
    """One free box (``robot``) hanging upright at the origin in four envs, never stepped."""
    from eden.options import BoxOptions, EnvOptions, SceneOptions
    from eden.utils.configs import EdenConfig

    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=4),
            scene_options=SceneOptions(robot=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.0, 0.0, 1.0))),
        )
    )


@pytest.fixture(scope="module")
def contact_env(tmp_path_factory: pytest.TempPathFactory) -> EnvBase:
    """The three-link :data:`STACK_MJCF` (``stack``) and two cubes come to rest on a plane.

    ``stack`` has one loaded link and two clear ones; ``cube_a`` / ``cube_b`` (0.125 kg each) rest
    apart from it, at ``xy`` (1, 1) and (-1, 1).
    """
    from eden.options import BoxOptions, EntityOptions, EnvOptions, PlaneOptions, SceneOptions
    from eden.utils.configs import EdenConfig

    path = tmp_path_factory.mktemp("stack") / "stack.xml"
    path.write_text(STACK_MJCF)
    env = build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=NUM_ENVS),
            scene_options=SceneOptions(
                plane=PlaneOptions(),
                # Half the slab's height: Eden replaces the MJCF root pose rather than composing with it.
                stack=EntityOptions(file=str(path), is_fixed_base=False, default_root_pos=(0.0, 0.0, 0.04)),
                cube_a=BoxOptions(size=(0.05, 0.05, 0.05), default_root_pos=(1.0, 1.0, 0.025)),
                cube_b=BoxOptions(size=(0.05, 0.05, 0.05), default_root_pos=(-1.0, 1.0, 0.025)),
            ),
        )
    )
    for _ in range(200):
        env.scene.step()
    return env


@pytest.fixture(scope="module")
def sloped_env(tmp_path_factory: pytest.TempPathFactory) -> EnvBase:
    """A box (``box``) held at a known height over a sloped heightfield (``terrain``), never stepped.

    Also the two-link :data:`PROBE_MJCF` (``probe``), upright at :data:`PROBE_POS`.
    """
    from eden.options import BoxOptions, EntityOptions, EnvOptions, SceneOptions, TerrainOptions
    from eden.utils.configs import EdenConfig
    from eden.utils.terrains import SlopedTerrain

    path = tmp_path_factory.mktemp("probe") / "probe.xml"
    path.write_text(PROBE_MJCF)
    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=NUM_ENVS),
            scene_options=SceneOptions(
                terrain=TerrainOptions(
                    subterrain_size=(8.0, 8.0),
                    horizontal_scale=0.25,
                    terrain_generators=[[SlopedTerrain.configure(slope=-0.5)]],
                ),
                box=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(*SLOPE_BOX_XY, SLOPE_BOX_Z)),
                probe=EntityOptions(file=str(path), is_fixed_base=False, default_root_pos=PROBE_POS),
            ),
        )
    )


@pytest.fixture
def make_term():
    """``make_term(cls, env, **params)``: a class term constructed outside its manager, then built."""

    def make(cls, env, **params):
        term = cls(env, cls.configure(**params))
        term.build()
        return term

    return make


@pytest.fixture
def install_manager():
    """``install_manager(env, "action_manager", manager)``: wire a real manager into ``env`` for one test.

    What ``EnvBase._load_managers`` does at build (``setattr(env, name, manager)``), run again with
    other term options, so a term that reaches a sibling manager in ``build()`` (an event term's
    ``env.action_manager.get_term``) sees real terms without a scene rebuild per configuration. The
    env's own manager goes back afterwards.
    """
    saved = []
    absent = object()

    def install(env, name, manager):
        saved.append((env, name, getattr(env, name, absent)))  # a plain EnvBase has no command_manager
        setattr(env, name, manager)
        return manager

    yield install
    for env, name, manager in reversed(saved):
        if manager is absent:
            delattr(env, name)
        else:
            setattr(env, name, manager)


@pytest.fixture
def restore_state():
    """``restore_state(entity)``: put ``entity``'s generalized coordinates back (at rest) after the test."""
    saved = []

    def remember(entity):
        saved.append((entity, entity.get_qpos().clone()))
        return entity

    yield remember
    for entity, qpos in reversed(saved):
        entity.set_qpos(qpos)  # zero_velocity=True: the entity is left at rest, as the scene started


@pytest.fixture
def score_metric(install_manager, restore_state):
    """``drive = score_metric(env, **fields)``: a metric manager holding one metric, ``score``.

    ``score`` is the distance between the scene's ``cube_a`` and ``cube_b`` (higher-is-better unless
    ``fields`` say otherwise); ``drive(values)`` places the cubes ``values[i]`` apart in env ``i`` and
    runs the metric manager once. ``fields`` are the metric term's options (``is_cumulative`` …).
    """
    from eden.constants import MetricDirection
    from eden.managers.metric_manager import MetricManager, MetricTermFuncWrapper
    from eden.managers.terms.metrics.proximity import entity_distance
    from eden.options.managers.metrics import MetricManagerOptions

    def install(env, **fields):
        fields = {"direction": MetricDirection.HIB, **fields}
        score = MetricTermFuncWrapper.configure(
            func=entity_distance, params={"entity_a_name": "cube_a", "entity_b_name": "cube_b"}, **fields
        )
        manager = install_manager(env, "metric_manager", MetricManager(env, MetricManagerOptions(score=score)))
        cube_a, cube_b = restore_state(env.entities["cube_a"]), restore_state(env.entities["cube_b"])

        def drive(values):
            values = torch.as_tensor(values, dtype=torch.float32)
            origin = torch.tensor([0.0, 5.0, 1.0]).expand(len(values), 3)
            cube_a.set_pos(origin.clone())
            cube_b.set_pos(origin + torch.nn.functional.pad(values.unsqueeze(-1), (0, 2)))
            manager.compute()

        return drive

    return install
