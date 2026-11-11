"""Real envs and a loopback wire for constructing deployments.

``eden deploy`` constructs a deployment against a real, built env, and so do these tests. The
one double is the transport (``connect`` / ``read_state`` / ``send_payload``, the hardware
boundary): :class:`LoopbackDeployment` records every payload it is sent and replays scripted
:class:`RobotState` readings. Entities, managers and terms are the real ones.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from eden.managers import OBSERVATION_TERM_REGISTRY
from eden.managers.modifiers.base import OBSERVATION_MODIFIER_REGISTRY, ObservationModifier
from eden.managers.observation_manager import ObservationTerm
from tests.support.deploy import deployments, robot_env

#: A fixed-base arm with two hinges: the cheapest articulated robot, and one MuJoCo loads too.
TWO_LINK_MJCF = """
<mujoco model="two_link">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 1.0">
      <geom name="root" type="sphere" size="0.05" mass="1.0"/>
      <body name="link1" pos="0 0 0">
        <joint name="j1" type="hinge" axis="0 1 0"/>
        <geom name="g1" type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="0.5"/>
        <body name="link2" pos="0.3 0 0">
          <joint name="j2" type="hinge" axis="0 1 0"/>
          <geom name="g2" type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="0.5"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

#: A second, fixed-base entity next to the G1. One of its joints shares a G1 joint's name, which is
#: the case where matching action terms by DOF name alone would misroute them.
HAND_MJCF = """
<mujoco model="hand">
  <worldbody>
    <body name="palm" pos="1.0 0 0.5">
      <geom type="box" size="0.03 0.03 0.03" mass="0.2"/>
      <body name="f0" pos="0 0 0.05">
        <joint name="left_wrist_yaw_joint" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.05" size="0.01" mass="0.05"/>
        <body name="f1" pos="0 0 0.05">
          <joint name="finger_0" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.05" size="0.01" mass="0.05"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

#: Frames in the ``MotionCommand`` clip. Joint ``j`` of frame ``k`` holds ``COMMAND_CLIP_STEP * k``.
COMMAND_CLIP_FRAMES = 3
COMMAND_CLIP_STEP = 0.1

#: Frames (at 30 fps) in the motion-manager clip. Every frame poses joint ``j`` (in LAFAN1 column
#: order, ``G1_DOFS29_NAME``) at ``MANAGER_CLIP_JOINT_STEP * j``, so a mis-mapped DOF reads a wrong value.
MANAGER_CLIP_FRAMES = 4
MANAGER_CLIP_JOINT_STEP = 0.01
MANAGER_CLIP_BASE_POS = (0.2, -0.1, 0.78)


# ---------------------------------------------------------------------------
# Terms a user might write, which cache what the deployment swaps
# ---------------------------------------------------------------------------


@OBSERVATION_TERM_REGISTRY.register()
class CachedEntityQuat(ObservationTerm):
    """The base quaternion, read through an entity handle cached in ``build()``.

    Eden's own terms resolve the entity each ``compute()``; a user term that caches it is what
    the deployment's entity repoint exists for.
    """

    entity_name: str = "robot"

    def build(self) -> None:
        self.robot = self._env.entities[self.entity_name]

    def compute(self) -> torch.Tensor:
        return self.robot.get_quat()


@OBSERVATION_MODIFIER_REGISTRY.register()
class CachedEntityModifier(ObservationModifier):
    """An identity modifier that caches the robot entity, one level below the term."""

    def build(self, frame: torch.Tensor) -> None:
        self._robot = self._env.entities["robot"]

    def compute(self, frame: torch.Tensor) -> torch.Tensor:
        return frame.clone()


# ---------------------------------------------------------------------------
# Envs
# ---------------------------------------------------------------------------


def write_command_clip(path: Path, links_name: list[str], num_dofs: int) -> Path:
    """A BeyondMimic-format ``.npz`` clip for ``MotionCommand``: every link at rest, 0.8 m up."""
    frames = np.arange(COMMAND_CLIP_FRAMES, dtype=np.float32)
    joint_pos = np.repeat(COMMAND_CLIP_STEP * frames[:, None], num_dofs, axis=1)
    body_pos = np.zeros((COMMAND_CLIP_FRAMES, len(links_name), 3), dtype=np.float32)
    body_pos[..., 2] = 0.8
    body_quat = np.zeros((COMMAND_CLIP_FRAMES, len(links_name), 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    np.savez(
        path,
        joint_pos=joint_pos,
        joint_vel=np.zeros_like(joint_pos),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=np.zeros_like(body_pos),
        body_ang_vel_w=np.zeros_like(body_pos),
        body_names=np.array(links_name),
        fps=np.array(50),
    )
    return path


def write_manager_clip(directory: Path) -> Path:
    """A LAFAN1-format CSV clip (base xyz, base quat xyzw, 29 joints) for the motion manager."""
    rows = np.zeros((MANAGER_CLIP_FRAMES, 36))
    rows[:, :3] = MANAGER_CLIP_BASE_POS
    rows[:, 6] = 1.0
    rows[:, 7:] = MANAGER_CLIP_JOINT_STEP * np.arange(29)
    np.savetxt(directory / "clip.csv", rows, delimiter=",")
    return directory


def g1_leg_dofs() -> list[str]:
    from eden.options.robots.unitree_g1 import G1_DOFS29_NAME

    return [name for name in G1_DOFS29_NAME if any(part in name for part in ("hip", "knee", "ankle"))]


def g1_deploy_config(assets: Path):
    """A G1 motion-imitation task with everything a deployment touches.

    Two robots (the G1 and a two-joint ``hand``), three action terms (the legs in reverse order
    behind a ``Compose`` holding an ``ActionDelay``, the arms behind an ``EnvelopeClip``, the hand
    behind an ``ActionDelay``), a velocity command and a ``MotionCommand`` clip, a motion manager
    with its own clip, and policy observations that include terms caching the entity and the
    command, and a stateful one (the velocity-error integral).
    """
    from eden import actions, observations
    from eden.datasets.lafan1_dataset import LAFAN1RetargetedG1Dataset
    from eden.managers.modifiers.actions.actuators import ActionDelay, Compose, EnvelopeClip
    from eden.managers.motion_manager import MotionTerm
    from eden.managers.terms.commands.mimic import MotionCommand
    from eden.managers.terms.commands.velocity import UniformVelocityCommand
    from eden.options import ObservationGroupOptions
    from eden.options.entities import RobotOptions
    from eden.options.managers.motions import MotionManagerOptions
    from eden.options.robots.unitree_g1 import G1Dofs29
    from eden.utils.configs import (
        ActionManagerOptions,
        CommandManagerOptions,
        EdenMimicConfig,
        EnvOptions,
        ObservationManagerOptions,
        SceneOptions,
    )

    g1 = G1Dofs29()
    (assets / "hand.xml").write_text(HAND_MJCF)
    clip_dir = assets / "manager_clip"
    clip_dir.mkdir()
    legs = g1_leg_dofs()
    return EdenMimicConfig(
        env_options=EnvOptions(num_envs=1),
        scene_options=SceneOptions(
            robot=g1,
            hand=RobotOptions(
                file=str(assets / "hand.xml"),
                dofs_name=["left_wrist_yaw_joint", "finger_0"],
                is_fixed_base=True,
                default_dofs_pos={".*": 0.0},
                default_dofs_kp={".*": 5.0},
                default_dofs_kd={".*": 0.1},
            ),
        ),
        action_options=ActionManagerOptions(
            legs=actions.ImplicitPDController.configure(
                entity_name="robot",
                dofs_name=legs[::-1],
                modifier=Compose.configure(
                    modifiers=[EnvelopeClip.configure(), ActionDelay.configure(delay_range_steps=(1, 1))]
                ),
            ),
            arms=actions.ImplicitPDController.configure(
                entity_name="robot",
                dofs_name=[name for name in g1.dofs_name if name not in legs],
                modifier=EnvelopeClip.configure(),
            ),
            hand=actions.ImplicitPDController.configure(
                entity_name="hand", dofs_name=[".*"], modifier=ActionDelay.configure(delay_range_steps=(1, 1))
            ),
        ),
        command_options=CommandManagerOptions(
            velocity_command=UniformVelocityCommand.configure(
                entity_name="robot",
                resampling_time_range=(3.0, 8.0),
                lin_vel_x_range=(-1.0, 1.0),
                lin_vel_y_range=(-1.0, 1.0),
                ang_vel_z_range=(-0.5, 0.5),
            ),
            motion=MotionCommand.configure(
                entity_name="robot",
                motion_file=str(
                    write_command_clip(assets / "command_clip.npz", MotionCommand.links_name, len(g1.dofs_name))
                ),
                sampling_mode="start",
                resampling_time_range=(1.0e9, 1.0e9),
            ),
        ),
        observation_options=ObservationManagerOptions(
            policy=ObservationGroupOptions(
                projected_gravity=observations.ProjectedGravity.configure(entity_name="robot"),
                dofs_pos=observations.DofsPos.configure(
                    entity_name="robot", modifiers=[CachedEntityModifier.configure()]
                ),
                base_quat=CachedEntityQuat.configure(entity_name="robot"),
                command=observations.ScaledCommands.configure(
                    command_name="velocity_command", command_scale=[1.0, 1.0, 1.0]
                ),
                error_integral=observations.VelocityTrackingErrorIntegral.configure(entity_name="robot"),
            )
        ),
        motion_options=MotionManagerOptions(
            mp_split_method="None",
            default=MotionTerm.configure(
                dataset_options=LAFAN1RetargetedG1Dataset.configure(
                    raw_data_path=str(write_manager_clip(clip_dir)),
                    processed_data_path=str(assets / "manager_clip.pt"),
                ),
                entities_map={"robot": "g1"},
                # Wire order. A permuted order would prove the name mapping, but MotionTerm keeps the
                # data in dataset order while reporting the permuted names (see the xfail repro in
                # tests/managers/test_motion_manager.py).
                target_dofs_name={"g1": list(g1.dofs_name)},
                target_links_name={"g1": LAFAN1RetargetedG1Dataset.motion_links_name["g1"]},
            ),
        ),
    )


def g1_command_only_config(assets: Path):
    """A G1 task whose only motion reference is a ``MotionCommand`` (no motion manager), as SONIC deploys."""
    from eden import actions, observations
    from eden.managers.terms.commands.mimic import MotionCommand
    from eden.options import ObservationGroupOptions
    from eden.options.robots.unitree_g1 import G1Dofs29
    from eden.utils.configs import (
        ActionManagerOptions,
        CommandManagerOptions,
        EdenRLConfig,
        EnvOptions,
        ObservationManagerOptions,
        SceneOptions,
    )

    g1 = G1Dofs29()
    clip = write_command_clip(assets / "command_clip.npz", MotionCommand.links_name, len(g1.dofs_name))
    return EdenRLConfig(
        env_options=EnvOptions(num_envs=1),
        scene_options=SceneOptions(robot=g1),
        action_options=ActionManagerOptions(
            joint_pos=actions.ImplicitPDController.configure(entity_name="robot", dofs_name=[".*"])
        ),
        command_options=CommandManagerOptions(
            motion=MotionCommand.configure(
                entity_name="robot",
                motion_file=str(clip),
                sampling_mode="start",
                resampling_time_range=(1.0e9, 1.0e9),
            )
        ),
        observation_options=ObservationManagerOptions(
            policy=ObservationGroupOptions(dofs_pos=observations.DofsPos.configure(entity_name="robot"))
        ),
    )


def two_link_robot(asset: Path):
    from eden.options.entities import RobotOptions

    return RobotOptions(
        file=str(asset),
        dofs_name=["j1", "j2"],
        default_dofs_pos={"j1": 0.0, "j2": 0.0},
        default_dofs_kp={".*": 50.0},
        default_dofs_kd={".*": 2.0},
        default_root_pos=(0.0, 0.0, 1.0),
        is_fixed_base=True,
    )


@pytest.fixture(scope="module")
def two_link_asset(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("two_link") / "two_link.xml"
    path.write_text(TWO_LINK_MJCF)
    return path


@pytest.fixture(scope="module")
def g1_env(tmp_path_factory):
    """The G1 motion-imitation env of :func:`g1_deploy_config`, built and reset once per module."""
    from tests.support.envs import build_env

    return build_env(g1_deploy_config(tmp_path_factory.mktemp("g1_deploy")), reset=True)


@pytest.fixture(scope="module")
def g1_command_env(tmp_path_factory):
    """The G1 env of :func:`g1_command_only_config`, built and reset once per module."""
    from tests.support.envs import build_env

    return build_env(g1_command_only_config(tmp_path_factory.mktemp("g1_command")), reset=True)


@pytest.fixture(scope="module")
def two_link_env(two_link_asset):
    """A plain RL env around the fixed-base two-link arm, built and reset once per module."""
    return robot_env(two_link_robot(two_link_asset))


@pytest.fixture
def deploy(g1_env):
    """``deploy(cls=LoopbackDeployment, **options)`` against :func:`g1_env`, undone after the test."""
    with deployments(g1_env) as make:
        yield make


@pytest.fixture
def deploy_command_only(g1_command_env):
    """``deploy_command_only(cls=LoopbackDeployment, **options)`` against :func:`g1_command_env`."""
    with deployments(g1_command_env) as make:
        yield make


@pytest.fixture
def deploy_two_link(two_link_env):
    """``deploy_two_link(cls=LoopbackDeployment, **options)`` against :func:`two_link_env`, undone after the test."""
    with deployments(two_link_env) as make:
        yield make
