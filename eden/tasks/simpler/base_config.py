"""Shared base config and asset/camera helpers for the SIMPLER Bridge tasks.

Every SIMPLER task drives the same sysid'd WidowX250 at 5 Hz through the same
DifferentialIK + binary-gripper stack, and renders through the same Logitech
C920 intrinsics against a real-inpainting backdrop. Only the scene contents, the
camera extrinsics/backdrop, and the reset placement differ per task.
"""

from __future__ import annotations

import numpy as np

import eden as en
from eden.entities.camera_overlays import BackgroundImageComposite
from eden.managers.terms.actions.binary_actions import BinaryJointController
from eden.managers.terms.actions.task_space_actions import DifferentialIKController
from eden.options.camera import CameraOptions
from eden.options.entities import SceneOptions
from eden.options.envs import EnvOptions
from eden.options.managers.actions import ActionManagerOptions
from eden.options.materials import RigidMaterialOptions
from eden.options.objects.simpler import SimplerObjectOptions
from eden.utils.configs import EdenRLConfig

# System-identified WidowX250: the kp/kd tables come from fitting the real BridgeV2
# arm, so they are not the robot's stock gains. Shared by every SIMPLER task.
# The world frame of every SIMPLER task is the arm's base_link, as in SIMPLER, which mounts its cameras on
# that link: the arm sits at the origin and each camera at SIMPLER's base-relative pose. Do not offset the
# arm by SIMPLER's world y of the base (0.028): that shifts it 2.8 cm against the cameras.
DEFAULT_ROBOT = en.robots.WidowX250(
    default_root_pos=(0.0, 0.0, 0.0),
    default_dofs_pos={
        "waist": -0.02014876,
        "shoulder": 0.04723017,
        "elbow": 0.22625704,
        "forearm_roll": -0.00307271,
        "wrist_angle": 1.365988,
        "wrist_rotate": -0.00168102,
        "left_finger": 0.037,
        "right_finger": -0.037,
    },
    default_dofs_kp={
        "waist": 1169.7891719504198,
        "shoulder": 730.0,
        "elbow": 808.4601346394447,
        "forearm_roll": 1229.1299089624076,
        "wrist_angle": 1272.2760456418862,
        "wrist_rotate": 1056.3326605132252,
        "left_finger": 100,
        "right_finger": 100,
    },
    default_dofs_kd={
        "waist": 330.0,
        "shoulder": 180.0,
        "elbow": 152.12036565582588,
        "forearm_roll": 309.6215302722146,
        "wrist_angle": 201.04998711007383,
        "wrist_rotate": 269.51458932695414,
        "left_finger": 20,
        "right_finger": 20,
    },
    material=RigidMaterialOptions(
        gravity_compensation=1.0,
    ),
    # Contact patch radius [m] resisting an object's twist in the grasp (``enable_torsional_friction``).
    default_geoms_friction_torsional={".*_finger_link": 0.02},
)

# Logitech C920 — the camera the BridgeV2 demonstrations were recorded with.
C920_INTRINSIC = np.array([[623.588, 0, 319.501], [0, 623.588, 239.545], [0, 0, 1]])


def simpler_asset_path(rel_path: str) -> str:
    """Resolve an ``eden_simpler`` dataset asset, downloading it on first use."""
    return SimplerObjectOptions.fetch_family_asset(rel_path)


def simpler_camera(
    *,
    cam_pos: tuple[float, float, float],
    cam_lookat: tuple[float, float, float],
    background_image: str,
    background_entities: list[str],
) -> CameraOptions:
    """A C920 camera looking at the scene over a real-inpainting backdrop.

    ``background_image`` is an ``eden_simpler``-relative path; per-task knobs are the
    extrinsics and which entities are composited into the backdrop rather than rendered.
    """
    backdrop = BackgroundImageComposite.configure(
        image_path=simpler_asset_path(background_image), entities=background_entities
    )
    return CameraOptions(cam_pos=cam_pos, cam_lookat=cam_lookat, cam_intrinsic=C920_INTRINSIC, overlays=[backdrop])


class SimplerBridgeBaseConfig(EdenRLConfig):
    """Robot, physics, and control stack shared by every SIMPLER Bridge task.

    Subclasses supply ``scene_options`` (the objects), ``cameras_options`` (built with
    :func:`simpler_camera`), and ``event_options`` (reset placement).

    A subclass replaces ``scene_options`` wholesale rather than merging into it, so each
    one must repeat ``robot=DEFAULT_ROBOT`` and the default below is never the one that
    gets used. It is declared anyway, matching ``polaris.base_config.DroidBaseConfig``:
    it keeps the base coherent if instantiated and pins the field's type. Do not "clean
    it up" -- dropping it leaves the base resolving to a robot-less ``SceneOptions`` that
    fails later with ``AttributeError`` instead of here.
    """

    env_options: EnvOptions = EnvOptions(
        sim_dt=0.005,  # 200 Hz (SIMPLER simulates at 500 Hz)
        decimation=40,  # control frequency = 200Hz / 40 = 5Hz
        num_envs=1,
        noslip_iterations=5,
        # The WidowX's fingertips close on a thin object (a spoon's handle) at a point; with no torsional
        # friction nothing resists it spinning about the line between them, and a lifted spoon swings to
        # hang below the grasp. Real finger pads resist that twist: see the fingers' patch on DEFAULT_ROBOT.
        enable_torsional_friction=True,
    )

    scene_options: SceneOptions = en.scenes.Empty(is_support_enabled=True, robot=DEFAULT_ROBOT)

    action_options: ActionManagerOptions = ActionManagerOptions(
        ee_controller=DifferentialIKController.configure(
            entity_name="robot",
            ee_link_name="gripper_link",
            dofs_name=[
                "waist",
                "shoulder",
                "elbow",
                "forearm_roll",
                "wrist_angle",
                "wrist_rotate",
            ],
            position_scale=0.5,
            orientation_scale=0.5,
        ),
        gripper_controller=BinaryJointController.configure(
            entity_name="robot",
            dofs_name=[
                "left_finger",
                "right_finger",
            ],
            open_action={"left_finger": 0.037, "right_finger": -0.037},
            close_action=0.0,
        ),
    )
