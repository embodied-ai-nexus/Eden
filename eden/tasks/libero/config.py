"""Eden-native runtime configuration for LIBERO tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
from genesis.options.renderers import Rasterizer
from genesis.options.sensors import RasterizerCameraOptions
from genesis.utils.geom import trans_quat_to_T
from pydantic import Field

from eden import terminations
from eden.managers.terms.actions.task_space_actions import OperationalSpaceController
from eden.managers.terms.observations.camera import rgb_image
from eden.managers.terms.observations.proprio import DofsPos
from eden.options.camera import CameraOptions, CamerasOptions
from eden.options.entities import CylinderOptions
from eden.options.envs import EnvOptions
from eden.options.managers.actions import ActionManagerOptions
from eden.options.managers.events import EventManagerOptions
from eden.options.managers.metrics import MetricManagerOptions
from eden.options.managers.observations import ObservationGroupOptions, ObservationManagerOptions
from eden.options.managers.terminations import TerminationManagerOptions
from eden.options.objects.libero import LiberoFlatStove, LiberoObjectOptions
from eden.options.robots.libero import LIBERO_ROBOT_ENTITY_NAME, LiberoPanda
from eden.options.surfaces import EmissionSurfaceOptions
from eden.tasks import TASK_REGISTRY
from eden.tasks.libero.catalog import get_suite_task_names, get_task_spec
from eden.tasks.libero.custom_terms.actions import LiberoPandaGripperController
from eden.tasks.libero.custom_terms.events import (
    LiberoFixedStateReset,
    LiberoStoveBurnerVisualization,
)
from eden.tasks.libero.custom_terms.metrics import LiberoGoalMetric
from eden.tasks.libero.scenes import (
    _get_libero_family_spec,
    make_libero_scene_options,
)
from eden.tasks.registry import TaskMod
from eden.terms import DoneTerm, ObsTerm
from eden.utils.configs import EdenRLConfig
from eden.utils.misc import get_assets_dir

LIBERO_POLICY_HORIZON_STEPS = 600
LIBERO_RESET_WARMUP_STEPS = 5


def _make_action_options() -> ActionManagerOptions:
    return ActionManagerOptions(
        osc_pose=OperationalSpaceController.configure(
            entity_name=LIBERO_ROBOT_ENTITY_NAME,
            ee_link_name=LiberoPanda.ee_links_name[0],
            ee_local_pos=LiberoPanda.grip_site_pos_local,
            ee_local_quat=LiberoPanda.grip_site_quat_local,
            dofs_name=list(LiberoPanda.arm_dofs_name),
            control_mode="pose",
            position_scale=0.05,
            orientation_scale=0.5,
            action_clip=1.0,
            kp=150.0,
            damping_ratio=1.0,
            inertial_decoupling="position_orientation",
            operational_inertia_damping=1.0e-6,
            nullspace_kp=10.0,
            nullspace_damping_ratio=1.0,
            nullspace_reference="default",
        ),
        gripper=LiberoPandaGripperController.configure(
            entity_name=LIBERO_ROBOT_ENTITY_NAME,
            dofs_name=list(LiberoPanda.gripper_dofs_name),
        ),
    )


def _make_observation_options() -> ObservationManagerOptions:
    return ObservationManagerOptions(
        policy=ObservationGroupOptions(
            concatenate_terms=False,
            enable_corruption=False,
            agentview_rgb=ObsTerm.configure(
                func=rgb_image,
                params={"camera_name": "agentview"},
            ),
            eye_in_hand_rgb=ObsTerm.configure(
                func=rgb_image,
                params={"camera_name": "eye_in_hand"},
            ),
            arm_joint_pos=DofsPos.configure(
                entity_name=LIBERO_ROBOT_ENTITY_NAME, dofs_name=list(LiberoPanda.arm_dofs_name)
            ),
            gripper_joint_pos=DofsPos.configure(
                entity_name=LIBERO_ROBOT_ENTITY_NAME, dofs_name=list(LiberoPanda.gripper_dofs_name)
            ),
        )
    )


def _make_env_options(*, num_envs: int = 20) -> EnvOptions:
    """Return the 20 Hz LIBERO simulation envelope supported by Eden."""
    sim_dt = 0.002
    decimation = 25
    return EnvOptions(
        num_envs=num_envs,
        episode_length_s=(LIBERO_RESET_WARMUP_STEPS + LIBERO_POLICY_HORIZON_STEPS) * sim_dt * decimation,
        sim_dt=sim_dt,
        sim_substeps=1,
        decimation=decimation,
        iterations=100,
        tolerance=1e-8,
        noslip_iterations=5,
        multiplier_collision_broad_phase=16,
        enable_multi_contact=True,
        enable_torsional_friction=True,
        env_separate_rigid=True,
        rasterizer_shadows=False,
    )


def _make_termination_options() -> TerminationManagerOptions:
    """End on goal success or after five warm-up steps and 600 policy actions."""
    return TerminationManagerOptions(
        on_success=DoneTerm.configure(
            func=terminations.on_success,
            time_out=False,
        ),
        time_out=DoneTerm.configure(
            func=terminations.time_out,
            time_out=True,
        ),
    )


class LiberoTaskMod(TaskMod):
    """Select a catalog task and fixed-state initialization for the registered LIBERO task.

    ``init_ids`` and ``num_envs`` are one selection: an explicit ``init_ids`` list assigns exactly
    one fixed state per environment and therefore sets ``env_options.num_envs``. Leave ``init_ids``
    empty to let ``num_envs`` lead and take the first ``num_envs`` states from the bundle.
    """

    def __init__(
        self,
        task_name: Annotated[str, "LIBERO catalog task name."] = get_suite_task_names("libero_object")[0],
        bundle_root: Annotated[
            str | None,
            "Root containing <suite>/<task>/state-bundle directories; by default libero/fixed_states in the asset "
            "directory.",
        ] = None,
        init_ids: Annotated[
            str,
            "Comma-separated fixed-state IDs, one per env; this SETS num_envs, so any later "
            "num_envs override (eden smoke's --num_envs defaults to 2 and always applies) must "
            "match the count. Empty selects the first num_envs states.",
        ] = "",
    ):
        self.task_name = task_name
        self.bundle_root = bundle_root
        self.init_ids = init_ids

    def apply(self, config: LiberoConfig) -> LiberoConfig:
        raw_init_ids = self.init_ids.strip()
        try:
            init_ids = tuple(int(value.strip()) for value in raw_init_ids.split(",")) if raw_init_ids else None
        except ValueError as exc:
            raise ValueError("init_ids must be a comma-separated list of integers") from exc

        task_spec = get_task_spec(self.task_name)
        if init_ids is not None:
            # An explicit selection is one state per env, so it *defines* num_envs rather than
            # being checked against it. Every CLI num_envs override lands after the mods run, so a
            # caller that passes both must keep them consistent; LiberoFixedStateReset.build()
            # raises and names both knobs when they disagree.
            config.env_options.num_envs = len(init_ids)
        indicators = {
            f"{entity_name}_burner_indicator": CylinderOptions(
                radius=0.0625,
                height=0.001,
                default_root_pos=(0.0, 0.0, LiberoStoveBurnerVisualization.hidden_z_offset),
                fixed=True,
                batch_fixed_verts=True,
                collision=False,
                surface=EmissionSurfaceOptions(color=(0.9, 0.05, 0.05, 1.0)),
            )
            for entity_name, category_name in task_spec.entity_categories.items()
            if category_name == LiberoFlatStove.category_name
        }
        config.scene_options = make_libero_scene_options(task_spec, **indicators)
        family_spec = _get_libero_family_spec(task_spec.problem_name)
        lights = [
            {
                "type": "point",
                "pos": (1.0, 1.0, 4.0),
                "color": (0.8, 0.8, 0.8),
                "intensity": 1.0,
            },
            {
                "type": "point",
                "pos": (-3.0, -3.0, 4.0),
                "color": (0.8, 0.8, 0.8),
                "intensity": 1.0,
            },
        ]
        if family_spec.has_directional_light:
            lights.insert(
                0,
                {
                    "type": "directional",
                    "dir": (0.0, 0.0, -1.0),
                    "color": (0.8, 0.8, 0.8),
                    "intensity": 1.0,
                },
            )
        camera_to_world = trans_quat_to_T(
            np.asarray(family_spec.agentview_pos_w),
            np.asarray(family_spec.agentview_quat_w),
        )
        image_resolution = (128, 128)
        config.cameras_options = CamerasOptions(
            agentview=CameraOptions(
                cam_extrinsic=np.linalg.inv(camera_to_world),
                cam_resolution=image_resolution,
                cam_fov=45.0,
                sensor_options=RasterizerCameraOptions(lights=lights),
            ),
            eye_in_hand=CameraOptions(
                cam_resolution=image_resolution,
                cam_fov=75.0,
                attach_entity_name=LIBERO_ROBOT_ENTITY_NAME,
                attach_link_name=LiberoPanda.ee_links_name[0],
                attach_offset=(0.05, 0.0, 0.0),
                attach_quat=(0.0, 0.7071067811865475, 0.7071067811865475, 0.0),
                sensor_options=RasterizerCameraOptions(),
            ),
        )
        bundle_root = self.bundle_root or get_assets_dir(LiberoObjectOptions.assets_subdir, "fixed_states")
        bundle_path = Path(bundle_root).expanduser() / task_spec.suite_name / task_spec.name / "state-bundle"
        event_terms = {
            "libero_fixed_state": LiberoFixedStateReset.configure(
                task_name=task_spec.name,
                bundle_path=str(bundle_path),
                init_ids=[] if init_ids is None else list(init_ids),
            ),
        }
        policy_dt = config.env_options.sim_dt * config.env_options.decimation
        for entity_name, category_name in task_spec.entity_categories.items():
            if category_name != LiberoFlatStove.category_name:
                continue
            indicator_name = f"{entity_name}_burner_indicator"
            event_terms[f"{indicator_name}_reset"] = LiberoStoveBurnerVisualization.configure(
                stove_entity_name=entity_name,
                indicator_entity_name=indicator_name,
            )
            event_terms[f"{indicator_name}_interval"] = LiberoStoveBurnerVisualization.configure(
                mode="interval",
                interval_range_s=(policy_dt, policy_dt),
                stove_entity_name=entity_name,
                indicator_entity_name=indicator_name,
            )
        config.event_options = EventManagerOptions(**event_terms)
        config.metric_options = MetricManagerOptions(goal=LiberoGoalMetric.configure(task_name=task_spec.name))
        config.language_instructions = [task_spec.language_instruction]
        return config


@TASK_REGISTRY.register(modifiers=(LiberoTaskMod(),))
class LiberoConfig(EdenRLConfig):
    """Shared production configuration for data-driven LIBERO tasks."""

    env_options: EnvOptions = Field(default_factory=_make_env_options)
    renderer_options: Rasterizer = Field(default_factory=Rasterizer)
    action_options: ActionManagerOptions = Field(default_factory=_make_action_options)
    observation_options: ObservationManagerOptions = Field(default_factory=_make_observation_options)
    termination_options: TerminationManagerOptions = Field(default_factory=_make_termination_options)
