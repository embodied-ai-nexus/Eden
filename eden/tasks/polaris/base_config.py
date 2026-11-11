"""Shared base config and splat-scene helpers for Polaris tasks."""

from __future__ import annotations

import copy

import numpy as np

import eden as en
from pydantic import Field

from eden.entities.camera_overlays import SplatComposite
from eden.managers.terms.actions.task_space_actions import DifferentialIKController
from eden.managers.terms.actions.binary_actions import BinaryJointController
from eden.options.managers.actions import ActionManagerOptions
from eden.options.envs import EnvOptions
from eden.options.camera import CamerasOptions, CameraOptions
from eden.options.materials import RigidMaterialOptions
from eden.options.renderer import BatchRendererOptions
from eden.options.robots import ManipulatorOptions
from eden.options.robots.composite.droid import _droid_gripper
from eden.options.splat import PCDOptions, SplatOptions
from eden.utils.configs import EdenSplatConfig, EdenSplatRLConfig
from eden.options.entities import PlaneOptions, RobotOptions, SceneOptions


def make_polaris_splat(
    *,
    path: str,
    scale: float = 1.0,
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> SplatOptions:
    """Construct a Polaris-style ``SplatOptions`` with the eden_polaris PCD dataset.

    Per-task knobs (``path`` / ``scale`` / ``pos`` / ``quat``) are forwarded
    to the inner :class:`PCDOptions`.
    """
    return SplatOptions(
        background=PCDOptions(
            path=path,
            dataset="eden_polaris",
            assets_subdir="polaris",
            scale=scale,
            pos=pos,
            quat=quat,
        ),
    )


# Arm home pose calibrated to the Polaris splat scenes/cameras (distinct from the
# manipulation Droid's grasp-ready home in eden.options.robots.composite.droid).
_POLARIS_FRANKA_HOME = {
    "joint1": 0.0,
    "joint2": -1 / 5 * np.pi,
    "joint3": 0.0,
    "joint4": -4 / 5 * np.pi,
    "joint5": 0.0,
    "joint6": 3 / 5 * np.pi,
    "joint7": 0,
}


def _polaris_arm() -> RobotOptions:
    return en.robots.FrankaEmikaPandaNoHand(
        default_dofs_pos=dict(_POLARIS_FRANKA_HOME),
        material=RigidMaterialOptions(gravity_compensation=1.0),
    )


def _polaris_gripper() -> RobotOptions:
    gripper = _droid_gripper()
    # Contact patch radius [m] resisting an object's twist in the grasp (``enable_torsional_friction``): without
    # it a scissor pinched across its middle pivots about the line between the pads and slips out once lifted.
    gripper.default_geoms_friction_torsional = {".*pad": 0.02}
    return gripper


class PolarisDroid(ManipulatorOptions):
    """Franka arm + Robotiq 2F85 for the Polaris splat scenes.

    Differs from :class:`eden.options.robots.Droid` in the arm home pose (calibrated to the
    splat cameras) and the pads' torsional friction; the gripper is otherwise Droid's pad-only
    ``_droid_gripper`` (a full-collision gripper penetrates mesh objects and ejects them on contact).
    """

    arm: RobotOptions = Field(default_factory=_polaris_arm)
    gripper: RobotOptions = Field(default_factory=_polaris_gripper)
    ee_link: str = "attachment"


class DroidSceneOptions(SceneOptions):
    # Declarative scene (no builder __init__) so a saved config round-trips through validated load.
    plane: PlaneOptions = PlaneOptions(is_support_enabled=True)

    # One composite (arm + welded gripper): it expands into the ``robot``/``gripper`` entities
    # welded at ``attachment`` during env build, and round-trips through the config as a PolarisDroid.
    robot: PolarisDroid = Field(default_factory=PolarisDroid)


# EdenSplatConfig (non-RL) routes `env_class_for_config` to SplatEnvBase: these are
# single-env teleop/inference scenes with no reward/termination/command/curriculum surface.
class DroidBaseConfig(EdenSplatConfig):
    env_options: EnvOptions = EnvOptions(
        # Grasp regime (see examples/recipes/collision/robotiq_2f85_grasp.py): substep dt
        # 0.001 keeps the 2F85's tuned 0.002 solref off Genesis's 2*substep_dt clamp.
        # No global constraint_timeconst -- it ejects mesh objects on contact; object-side
        # grip stiffness lives on each object's default_geoms_sol_params instead.
        sim_dt=0.004,
        sim_substeps=4,
        decimation=5,
        num_envs=1,
        episode_length_s=30,
        noslip_iterations=5,
        # The pads' twist resistance (see ``_polaris_gripper``), as on SIMPLER's WidowX.
        enable_torsional_friction=True,
    )

    scene_options: DroidSceneOptions = DroidSceneOptions()

    cameras_options: CamerasOptions = CamerasOptions(
        external_camera=CameraOptions(
            # NOTE camera resolution=(1280, 720)
            cam_intrinsic=np.array(
                [
                    [1280 * 1.0476 / 2.5452, 0.0, 1280 / 2],
                    [0.0, 720 * 1.0476 / 1.4721, 720 / 2],
                    [0.0, 0.0, 1.0],
                ]
            ),
            cam_pos=(-0.01, -0.33, 0.48),
            cam_lookat=(0.716, 0.122, -0.035),  # TODO: check this
            cam_up=(0.432, 0.2774, 0.8552),  # TODO: check this
            overlays=[SplatComposite.configure(entities=["plane"])],
        ),
        wrist_camera=CameraOptions(
            # NOTE camera resolution=(1280, 720)
            cam_intrinsic=np.array(
                [
                    [1280 * 2.8 / 5.376, 0.0, 1280 / 2],
                    [0.0, 720 * 2.8 / 3.024, 720 / 2],
                    [0.0, 0.0, 1.0],
                ]
            ),
            attach_entity_name="robot",
            attach_link_name="attachment",
            # attach_offset=(0.011, -0.031, -0.074),
            attach_offset=(0.07, -0.031, -0.074),
            attach_quat=(0.0, 0.7071067811865475, 0.7071067811865476, 0.0),
            overlays=[SplatComposite.configure(entities=["plane"])],
        ),
    )

    action_options: ActionManagerOptions = ActionManagerOptions(
        ee_controller=DifferentialIKController.configure(
            entity_name="robot",
            ee_link_name="attachment",
            dofs_name=[
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
                "joint7",
            ],
            position_scale=0.5,
            orientation_scale=0.5,
        ),
        gripper_controller=BinaryJointController.configure(
            entity_name="gripper",
            dofs_name=[
                "left_driver_joint",
                "right_driver_joint",
            ],
            open_action={"left_driver_joint": 0.0, "right_driver_joint": 0.0},
            close_action={"left_driver_joint": 0.8, "right_driver_joint": 0.8},
        ),
    )


def droid_field(cls: type, field: str):
    """Deep-copy a field default off a Droid task class (scene/cameras/actions are shared).

    ``copy.deepcopy`` — not pydantic ``model_copy(deep=True)`` — because ``ConfigurableOptions``
    defines ``__deepcopy__(self, memo)`` with no default, which ``model_copy`` calls with no memo.
    """
    return copy.deepcopy(cls.model_fields[field].default)


# Soft key + low-fill rig tuned so the Madrona sim pass sits at the splat scenes' ambient
# level rather than blowing out. No shadows: the splat background is not scene geometry, so
# cast shadows land only on the hidden plane. To carry them onto the splat, set castshadow=True
# here and shadow_strength on the cameras' SplatComposite, and give the plane a plain colour
# (PlaneOptions(color=...)): the default checkerboard would read as shadow, so it is rejected.
_POLARIS_BATCH_LIGHTS = [
    {"pos": (0.6, 0.2, 2.0), "dir": (-0.2, -0.1, -1), "intensity": 1.0, "directional": True, "castshadow": False},
    {"pos": (-0.4, -0.3, 1.5), "dir": (0.2, 0.2, -1), "intensity": 0.4, "directional": True, "castshadow": False},
]


def polaris_batch_splat(base_cls: type, *, per_env_gaussians: bool = False, minibatch: int = 2) -> SplatOptions:
    """Build a batched-tuned :class:`SplatOptions` from a single-env Polaris task's ``SplatOptions``.

    Reuses the single-env task's PCDs verbatim (path / pose / scale already resolved), so a batched
    task never re-specifies the splat placement, and adds the ``per_env_gaussians`` / ``minibatch``
    knobs that bound VRAM at scale (see :class:`SplatOptions`).
    """
    base: SplatOptions = base_cls.model_fields["splat_options"].default
    pcds = {name: copy.deepcopy(pcd) for name, pcd in base.pcds.items()}
    return SplatOptions(per_env_gaussians=per_env_gaussians, minibatch=minibatch, **pcds)


class DroidBatchBaseConfig(EdenSplatRLConfig):
    """Batched (Madrona + gsplat), RL-surfaced counterpart of :class:`DroidBaseConfig`.

    ``EdenSplatRLConfig`` (-> ``SplatRLEnvBase``) so it carries the full RL surface; it reuses
    ``DroidBaseConfig``'s robot, cameras, and action space and adds the gs-madrona batch renderer
    plus a soft two-light rig, so every environment's photorealistic splat is rendered and
    composited under the batched Madrona sim pass. Subclasses set ``scene_options`` and
    ``splat_options`` (and ``language_instructions``), typically copied from the matching single-env
    task via :func:`droid_field` / :func:`polaris_batch_splat`. ``env_options.num_envs`` and
    ``splat_options.minibatch`` trade VRAM for throughput.

    The reward/termination/etc. managers are intentionally empty for now: the batched Madrona +
    num_envs setup is RL-*training* scaffolding, and the reward/termination terms are coming. Until
    then these tasks render batched but train against nothing — RL-typed on purpose.
    """

    env_options: EnvOptions = Field(
        default_factory=lambda: droid_field(DroidBaseConfig, "env_options").model_copy(update={"num_envs": 4})
    )
    cameras_options: CamerasOptions = Field(default_factory=lambda: droid_field(DroidBaseConfig, "cameras_options"))
    action_options: ActionManagerOptions = Field(default_factory=lambda: droid_field(DroidBaseConfig, "action_options"))
    renderer_options: BatchRendererOptions = Field(
        default_factory=lambda: BatchRendererOptions(lights=copy.deepcopy(_POLARIS_BATCH_LIGHTS))
    )
