"""cuRobo-based motion planner."""

from __future__ import annotations
from typing import Any, TYPE_CHECKING
import os
import copy

import torch
import genesis.utils.geom as gu

import eden as en
from eden.extensions.motion_planning import (
    MotionPlannerBase,
    MOTION_PLANNER_REGISTRY,
)
from eden.options.extensions.motion_planning import PlannerOptions


IS_CUROBO_AVAILABLE = True
try:
    from curobo.geom.types import Mesh, WorldConfig
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.robot import RobotConfig
    from curobo.types.state import JointState
    from curobo.wrap.reacher.motion_gen import (  # pyright: ignore[reportMissingImports]
        MotionGen,
        MotionGenConfig,
        MotionGenPlanConfig,
    )
    from curobo.util_file import load_yaml  # pyright: ignore[reportMissingImports]
except ImportError:
    IS_CUROBO_AVAILABLE = False

if TYPE_CHECKING:
    from eden.entities.base import Entity
    from eden.envs.base import EnvBase


@MOTION_PLANNER_REGISTRY.register()
class CuroboPlanner(MotionPlannerBase):
    """Motion planner using cuRobo for collision-aware trajectory generation.

    Options (besides the shared ``entity_name``):

    - ``robot_config_file`` (str): path to the robot configuration YAML file (cuRobo format).
    - ``ee_link_name`` (str): the end-effector link.
    - ``collision_entities_name`` (list[str] | None): entities whose geoms are collision obstacles.
    - ``interpolation_dt`` (float): time step, in seconds, between interpolated waypoints.

    Parameters
    ----------
    env : EnvBase
        The environment instance.
    options : PlannerOptions
        From ``CuroboPlanner.configure()``.
    """

    robot_config_file: str = ""
    ee_link_name: str = ""
    collision_entities_name: list[str] | None = None
    interpolation_dt: float = 0.05

    def __init__(
        self,
        env: EnvBase,
        options: PlannerOptions,
    ) -> None:
        if not IS_CUROBO_AVAILABLE:
            raise ImportError(
                "curobo is not installed. Please see `https://curobo.org/get_started/1_install_instructions.html` for installation instructions."
            )

        super().__init__(env, options)

        if self.collision_entities_name is None:
            self.collision_entities_name = []

        _, ee_link_idx = self.entity.find_named_links_idx_local(self.ee_link_name)
        self.ee_link_idx = ee_link_idx[0]

        self._dtype = torch.float32  # curobo internally only supports fp32
        self._tensor_args = TensorDeviceType(device=env.device, dtype=torch.float32)

        self._collision_entities = [self.env.entities[name] for name in self.collision_entities_name]

        self._init_robot_config()
        self._motion_gen: MotionGen | None = None
        self._world_model_list: list[WorldConfig] | None = None

    @property
    def allow_batch(self) -> bool:
        """Whether the planner allows batch processing."""
        return True

    def _init_robot_config(self) -> None:
        if not os.path.exists(self.robot_config_file):
            raise FileNotFoundError(f"Robot configuration file {self.robot_config_file} not found")
        robot_cfg: dict[str, Any] = load_yaml(self.robot_config_file)["robot_cfg"]
        cspace_joint_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
        lock_joints = set(robot_cfg["kinematics"].get("lock_joints", {}).keys())
        self._joint_names = [j for j in cspace_joint_names if j not in lock_joints]
        self._n_dof = len(self._joint_names)
        self._robot_cfg = RobotConfig.from_dict(robot_cfg)

    def update_world(
        self,
        carried_entities: list[Entity] | None = None,
        num_envs: int = 1,
        envs_idx: int | slice | torch.Tensor | None = None,
    ) -> None:
        """Rebuild the collision world and motion generator.

        Call this before plan_motion when the scene has changed.
        """
        if carried_entities is None:
            carried_entities = []
        self._world_model_list = self._build_world_model(carried_entities, num_envs, envs_idx)
        self._motion_gen = self._build_motion_gen(self._world_model_list)

    def _build_world_model(
        self,
        carried_entities: list[Entity],
        num_envs: int,
        envs_idx: int | slice | torch.Tensor | None = None,
    ) -> list[WorldConfig]:
        collision_entities_not_carried = [
            entity for entity in self._collision_entities if entity not in carried_entities
        ]
        collision_obstacles = self._entities_to_obstacles(collision_entities_not_carried, num_envs, envs_idx)

        world_model_list: list[WorldConfig] = []
        for meshes_per_env in collision_obstacles:
            world_model = WorldConfig(mesh=list(meshes_per_env))
            world_model_list.append(world_model)

        return world_model_list

    def _build_motion_gen(self, world_model_list: list[WorldConfig]) -> MotionGen:
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            copy.deepcopy(self._robot_cfg),
            world_model_list,
            tensor_args=self._tensor_args,
            interpolation_dt=self.interpolation_dt,
        )
        return MotionGen(motion_gen_config)

    def _entities_to_obstacles(
        self,
        collision_entities: list[Entity],
        num_envs: int,
        envs_idx: int | slice | torch.Tensor | None = None,
    ) -> list[list[Mesh]]:
        i_ob = 0

        obstacles: list[list[Mesh]] = [[] for _ in range(num_envs)]
        for entity in collision_entities:
            for geom in entity._entity.geoms:
                geom_trimesh = geom.get_trimesh()
                # relative=False: world frame, consistent with the world-frame robot pose
                # this is later transformed against. Genesis >=1.1.1 geom getters default
                # to the user frame (alignment/inertial offset stripped).
                geom_pos = geom.get_pos(relative=False).to(self._dtype)
                geom_quat = geom.get_quat(relative=False).to(self._dtype)

                collision_mesh_pos, collision_mesh_quat = self._transform_pos_quat_to_robot_frame(
                    geom_pos, geom_quat, envs_idx
                )

                assert collision_mesh_quat is not None
                for i_env in range(num_envs):
                    pose = collision_mesh_pos[i_env].tolist() + collision_mesh_quat[i_env].tolist()

                    obstacle = Mesh(
                        name=f"mesh_{i_ob}",
                        vertices=geom_trimesh.vertices.tolist(),
                        faces=geom_trimesh.faces.tolist(),
                        pose=pose,
                        scale=[1.0, 1.0, 1.0],
                    )
                    obstacles[i_env].append(obstacle)
                i_ob += 1

        return obstacles

    def _transform_pos_quat_to_robot_frame(
        self,
        input_pos: torch.Tensor,
        input_quat: torch.Tensor,
        envs_idx: int | slice | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        robot_pos = self.entity.get_pos(envs_idx).to(self._dtype)
        robot_quat = self.entity.get_quat(envs_idx).to(self._dtype)

        input_pos = input_pos - robot_pos

        input_pos_robot_frame, input_quat_robot_frame = gu.transform_pos_quat_by_trans_quat(
            input_pos, input_quat, robot_pos * 0, gu.inv_quat(robot_quat)
        )

        return input_pos_robot_frame, input_quat_robot_frame

    def _compute_ee_goal_robot_frame(
        self,
        ee_goal_pos: torch.Tensor,
        ee_goal_quat: torch.Tensor,
        num_envs: int,
        envs_idx: int | slice | torch.Tensor | None = None,
    ) -> torch.Tensor:
        ee_goal_pos = ee_goal_pos.to(device=self.env.device, dtype=self._dtype)
        ee_goal_quat = ee_goal_quat.to(device=self.env.device, dtype=self._dtype)

        if ee_goal_pos.shape != (num_envs, 3) or ee_goal_quat.shape != (num_envs, 4):
            raise ValueError(
                f"Expected ee_goal_pos shape {(num_envs, 3)} and ee_goal_quat shape {(num_envs, 4)}, "
                f"but got {ee_goal_pos.shape} and {ee_goal_quat.shape}"
            )

        pos_rf, quat_rf = self._transform_pos_quat_to_robot_frame(ee_goal_pos, ee_goal_quat, envs_idx)
        return torch.cat([pos_rf, quat_rf], dim=-1)

    def _plan_path_batch(
        self,
        qpos_start: torch.Tensor,
        ee_goal_robot_frame: torch.Tensor,
        motion_gen: MotionGen,
        input_dtype: torch.dtype,
        num_envs: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        start_state = JointState.from_position(
            self._tensor_args.to_device(qpos_start).contiguous(),
            joint_names=self._joint_names,
        )

        goal_pose = Pose.from_batch_list(ee_goal_robot_frame.tolist(), self._tensor_args)

        result = motion_gen.plan_batch_env(start_state, goal_pose, MotionGenPlanConfig(max_attempts=200, timeout=10))

        assert result.success is not None

        all_waypoints: list[torch.Tensor] = [qpos_start.new_zeros((0, self._n_dof)) for _ in range(num_envs)]

        if result.success.any():
            result_paths = result.get_paths() if num_envs > 1 else [result.get_interpolated_plan()]
            for i in range(num_envs):
                if result.success[i]:
                    waypoints_at_env = result_paths[i].position
                    assert isinstance(waypoints_at_env, torch.Tensor)
                    all_waypoints[i] = waypoints_at_env.to(input_dtype)
                else:
                    all_waypoints[i] = qpos_start[i].reshape(1, -1)
        else:
            en.logger.warning("Motion planning failed!")

        return result.success, all_waypoints

    def plan_motion(
        self,
        ee_goal_pos: torch.Tensor,
        ee_goal_quat: torch.Tensor,
        qpos_start: torch.Tensor | None = None,
        envs_idx: int | slice | torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Plan a collision-free motion to the target end-effector pose.

        Parameters
        ----------
        ee_goal_pos : torch.Tensor
            Target end-effector position in the world frame, shape (num_envs, 3).
        ee_goal_quat : torch.Tensor
            Target end-effector quaternion (wxyz) in the world frame, shape (num_envs, 4).
        qpos_start : torch.Tensor, optional
            Start joint configuration. Defaults to current entity qpos.
        envs_idx : int | slice | torch.Tensor | None, optional
            Environment indices to plan for. Defaults to every environment.

        Returns
        -------
        list[torch.Tensor]
            Planned waypoints for each environment, each of shape (num_waypoints, n_dof). An
            environment whose plan failed gets its start configuration as a single waypoint, and
            every environment gets an empty ``(0, n_dof)`` tensor when no plan succeeded.
        """
        if envs_idx is None:
            envs_idx = slice(None)
        if isinstance(envs_idx, slice):
            num_envs = len(range(*envs_idx.indices(self.env.num_envs)))
        elif isinstance(envs_idx, torch.Tensor):
            num_envs = envs_idx.shape[0]
        else:
            num_envs = 1

        if qpos_start is None:
            qpos_start = self.entity.get_qpos(envs_idx=envs_idx)

        ee_goal_robot_frame = self._compute_ee_goal_robot_frame(ee_goal_pos, ee_goal_quat, num_envs, envs_idx)

        if self._motion_gen is None:
            self.update_world(num_envs=num_envs, envs_idx=envs_idx)

        solved, all_waypoints = self._plan_path_batch(
            qpos_start[:, : self._n_dof],
            ee_goal_robot_frame,
            self._motion_gen,
            input_dtype=qpos_start.dtype,
            num_envs=num_envs,
        )

        return all_waypoints
