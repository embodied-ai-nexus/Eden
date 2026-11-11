"""Base class and result type for grasp samplers.

A grasp sampler turns one entity's point cloud into 6-DoF grasp poses. :class:`GraspSamplerBase`
owns everything that is the same for every backend — reading the target's points off the camera or
its surface, the obstacle cloud for collision filtering, ranking, the frame the poses place, and
packing the result into Eden's ``pos`` / wxyz ``quat`` tensors — so a backend only implements
:meth:`~GraspSamplerBase._infer` (the model call), declares its pose frame, and, if its grippers
carry collision geometry, implements :meth:`~GraspSamplerBase._collision_free`.

**Frames.** Every backend's model returns *grasp frames*: z is the approach direction and x the
direction the fingers close along, whatever the gripper. A grasp frame is generally no link of the
real gripper. Each backend therefore names the frame its poses place instead (GraspGenX: its gripper
description's base link), and with ``end_effector=`` an Eden robot's link, from the pairings it has
verified (:attr:`GraspSamplerBase.END_EFFECTORS`).

One call samples for one env (``sample_grasps(..., env_idx=)``), or for several with one render and
one batched model call (:meth:`GraspSamplerBase.sample_grasps_batch`). In a batched env the camera
must render each env on its own (``EnvOptions(env_separate_rigid=True)`` or the batch renderer): a
shared view holds every env's copy of the target under one segmentation id.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array

import eden as en
from eden.options.extensions.grasp_sampling import GraspSamplerOptions
from eden.utils.common import ConfigurableMixin
from eden.utils.geom import T_to_trans_quat, trans_quat_to_T
from eden.utils.misc import resolve_named_entity
from eden.utils.pc import get_entity_surface_points, update_entity_surface_points
from eden.utils.registry import Registry

if TYPE_CHECKING:
    from eden.entities.base import Entity
    from eden.entities.camera import Camera
    from eden.envs.base import EnvBase


GRASP_SAMPLER_REGISTRY = Registry(
    "GRASP_SAMPLER",
    populated_by=(
        "eden.extensions.grasp_sampling.graspgen.sampler",
        "eden.extensions.grasp_sampling.graspgenx.sampler",
    ),
)

#: A ``(pos, wxyz quat)`` pose that leaves a frame where it is.
IDENTITY_POSE = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))


@dataclass(frozen=True)
class EndEffectorPairing:
    """How one of a backend's grippers maps onto an Eden robot's link.

    Attributes
    ----------
    gripper : str
        The backend's gripper: the value of its gripper option (GraspGenX's ``gripper_name``,
        GraspGen's ``gripper_config``).
    link_name : str
        The Eden robot's link each pose then places.
    link_in_frame : tuple[tuple[float, float, float], tuple[float, float, float, float]]
        That link's pose, ``(pos, wxyz quat)``, in the frame the backend's poses place without a
        pairing (GraspGenX: the description's base link; GraspGen: its grasp frame).
    """

    gripper: str
    link_name: str
    link_in_frame: tuple[tuple[float, float, float], tuple[float, float, float, float]] = IDENTITY_POSE


@dataclass(frozen=True, eq=False)  # field-wise == on tensors has no truth value
class GraspResult:
    """World-frame grasp poses, best first.

    Each pose places the sampler's :attr:`~GraspSamplerBase.pose_link_name`.

    Attributes
    ----------
    pos : torch.Tensor
        Grasp positions, shape ``(n_grasps, 3)``, on the env's device.
    quat : torch.Tensor
        Grasp orientations as wxyz quaternions, shape ``(n_grasps, 4)``.
    confidences : torch.Tensor
        Discriminator confidences in ``[0, 1]``, shape ``(n_grasps,)``, in descending order.
    """

    pos: torch.Tensor
    quat: torch.Tensor
    confidences: torch.Tensor

    @property
    def success(self) -> bool:
        """Whether any grasp was found."""
        return len(self) > 0

    def __len__(self) -> int:
        return len(self.confidences)

    @classmethod
    def from_transforms(
        cls,
        transforms: np.ndarray,
        confidences: np.ndarray,
        device: torch.device,
        max_grasps: int | None = None,
    ) -> GraspResult:
        """Rank ``(n, 4, 4)`` world-frame ``transforms`` by confidence and convert them to pos/quat."""
        order = np.argsort(-confidences, kind="stable")[:max_grasps]
        transforms = torch.as_tensor(transforms[order], dtype=gs.tc_float, device=device).reshape(-1, 4, 4)
        pos, quat = T_to_trans_quat(transforms)
        return cls(
            pos=pos, quat=quat, confidences=torch.as_tensor(confidences[order], dtype=gs.tc_float, device=device)
        )


class GraspSamplerBase(ConfigurableMixin[GraspSamplerOptions], ABC):
    """Abstract base class for grasp samplers.

    The shared options (``camera_name``, ``object_cloud``, ``grasp_threshold``, ``max_grasps``,
    ``filter_collisions``, ``end_effector``, …) are documented on
    :class:`~eden.options.extensions.grasp_sampling.GraspSamplerOptions`.

    A backend's ``__init__`` calls :meth:`_set_pose_frame` once it knows its gripper, naming the
    frame its poses place without a pairing.

    Parameters
    ----------
    env : EnvBase
        The built environment the grasps are sampled in. With ``num_envs > 1``, a camera the sampler
        reads must render each env on its own.
    options : GraspSamplerOptions
        From the backend's ``configure()``.
    """

    camera_name: str = "camera"
    object_cloud: Literal["camera", "surface"] = "camera"
    surface_points: int = 2048
    grasp_threshold: float = 0.8
    num_grasps: int = 200
    max_grasps: int | None = None
    filter_collisions: bool = False
    collision_threshold: float = 0.02
    collision_ignore_entities: list[str] | None = None
    max_scene_points: int = 8192
    seed: int = 0
    end_effector: str | None = None

    #: The Eden end-effectors this backend pairs a gripper with, by robot options class name. Each one
    #: is verified against the Eden robot's geometry before it is listed here.
    END_EFFECTORS: ClassVar[dict[str, EndEffectorPairing]] = {}
    #: The backend's option that names its gripper, which ``end_effector`` sets.
    _gripper_option: ClassVar[str] = ""
    #: Why a backend that does not override :meth:`_collision_free` cannot ``filter_collisions``.
    _no_collision_filter_reason: ClassVar[str] = "its grippers carry no collision geometry."

    def __init__(self, env: EnvBase, options: GraspSamplerOptions) -> None:
        super().__init__(options)
        self.env = env
        if self.filter_collisions and type(self)._collision_free is GraspSamplerBase._collision_free:
            raise ValueError(
                f"{type(self).__name__} cannot filter_collisions: {self._no_collision_filter_reason} Fix: configure "
                f"filter_collisions=False, then reject grasps by posing the gripper in the scene and calling "
                f"detect_collision — see examples/recipes/planning/grasp_sampling_graspgenx.py."
            )
        self._pairing = self._resolve_end_effector()
        self._rng = np.random.default_rng(self.seed)
        # Set by the backend through _set_pose_frame.
        self._grasp_T_pose: np.ndarray | None = None

        reads_camera = self.object_cloud == "camera" or self.filter_collisions
        self.camera: Camera | None = self._resolve_camera() if reads_camera else None

        # Resolved now, so a typo fails here rather than leaving the arm in the obstacle cloud. Only the
        # names are kept: deployment can swap ``env.entities[name]``, so each call re-reads the entities.
        if self.collision_ignore_entities is None:
            self._ignored_entity_names = [
                name for name, entity in env.entities.items() if getattr(entity, "is_actuated", False)
            ]
        else:
            for name in self.collision_ignore_entities:
                resolve_named_entity(env.entities, name, hint="collision_ignore_entities lists it; list only those.")
            self._ignored_entity_names = list(self.collision_ignore_entities)

    def _resolve_end_effector(self) -> EndEffectorPairing | None:
        """The pairing ``end_effector`` names, with the backend's gripper option set from it."""
        if self.end_effector is None:
            return None
        pairing = self.END_EFFECTORS.get(self.end_effector)
        option = self._gripper_option
        if pairing is None:
            raise ValueError(
                f"end_effector='{self.end_effector}' is not an Eden end-effector {type(self).__name__} pairs a "
                f"gripper with; it pairs {sorted(self.END_EFFECTORS)}. Fix: pass one of those, or leave "
                f"end_effector unset and pass {option}= — the poses then place the backend's own frame, which "
                f"you map onto your robot's link yourself."
            )
        configured = getattr(self, option)
        if configured and configured != pairing.gripper:
            raise ValueError(
                f"end_effector='{self.end_effector}' pairs with {option}='{pairing.gripper}', but "
                f"{option}='{configured}' is configured too. Fix: drop {option}=; end_effector picks it."
            )
        setattr(self, option, pairing.gripper)
        return pairing

    def _set_pose_frame(self, link_name: str | None, grasp_T_frame: np.ndarray, tcp_in_grasp: np.ndarray) -> None:
        """Declare what the poses place: called once by the backend's ``__init__``.

        Parameters
        ----------
        link_name : str | None
            The link the backend's poses place without a pairing; ``None`` when that frame is the
            grasp frame itself, which is no link.
        grasp_T_frame : np.ndarray
            That frame's ``(4, 4)`` transform in the grasp frame.
        tcp_in_grasp : np.ndarray
            The TCP, ``(3,)``, in the grasp frame: the point the grasp puts on the object.
        """
        grasp_T_pose = np.asarray(grasp_T_frame, dtype=np.float64)
        self._pose_link_name = link_name
        if self._pairing is not None:
            pos, quat = (torch.tensor(v, dtype=torch.float64) for v in self._pairing.link_in_frame)
            grasp_T_pose = grasp_T_pose @ trans_quat_to_T(pos, quat).numpy()
            self._pose_link_name = self._pairing.link_name
        self._grasp_T_pose = grasp_T_pose
        self._tcp_in_pose = (np.linalg.inv(grasp_T_pose) @ np.r_[np.asarray(tcp_in_grasp, dtype=np.float64), 1.0])[:3]

    def _pose_frame(self) -> np.ndarray:
        if self._grasp_T_pose is None:
            raise RuntimeError(
                f"{type(self).__name__} never declared the frame its poses place. Fix: call "
                f"self._set_pose_frame(...) at the end of its __init__."
            )
        return self._grasp_T_pose

    @property
    def pose_link_name(self) -> str | None:
        """The link each returned pose places: ``end_effector``'s link when set, else the backend's own.

        ``None`` when the poses are the backend's grasp frame itself, which is no link of the gripper
        (GraspGen without ``end_effector``).
        """
        self._pose_frame()
        return self._pose_link_name

    @property
    def approach_axis(self) -> torch.Tensor:
        """The approach direction, a unit vector of shape ``(3,)`` in the posed link's frame, on the env's device.

        A grasp moves the gripper along this axis onto the object, so a pregrasp backs off along it:
        ``pos - transform_by_quat(approach_axis[None], quat) * distance``.
        """
        # The grasp's z, in the posed link's frame: a row of the rotation, not a column.
        return torch.as_tensor(self._pose_frame()[2, :3], dtype=gs.tc_float, device=self.env.device)

    @property
    def closing_axis(self) -> torch.Tensor:
        """The direction the fingers close along, a unit vector of shape ``(3,)`` in the posed link's frame.

        With :attr:`approach_axis` and :attr:`tcp_offset` it is what a scripted demonstrator's top-down
        ``move_to(pos, yaw)`` needs from a grasp: the heading of
        ``transform_by_quat(closing_axis[None], quat)`` is the one its fingers must close along.
        """
        return torch.as_tensor(self._pose_frame()[0, :3], dtype=gs.tc_float, device=self.env.device)

    @property
    def tcp_offset(self) -> torch.Tensor:
        """The TCP in the posed link's frame, shape ``(3,)``, on the env's device.

        The TCP is the point the grasp puts on the object (for a parallel jaw, the fingertip center),
        so ``pos + transform_by_quat(tcp_offset[None], quat)`` is where a grasp puts it.
        """
        self._pose_frame()
        return torch.as_tensor(self._tcp_in_pose, dtype=gs.tc_float, device=self.env.device)

    def _resolve_camera(self) -> Camera:
        if self.camera_name not in self.env.cameras:
            raise ValueError(
                f"camera_name='{self.camera_name}' is not a camera of this env; it has {sorted(self.env.cameras)}. "
                f"Fix: pass camera_name= one of those, or add a CameraOptions under that key in cameras_options."
            )
        camera = self.env.cameras[self.camera_name]
        if not camera.supports_pointcloud:
            raise ValueError(
                f"Camera '{self.camera_name}' uses the sensor backend, which renders no point cloud or segmentation. "
                f"Fix: remove sensor_options from its CameraOptions so it uses the legacy backend."
            )
        if self.env.num_envs > 1:
            views = tensor_to_array(camera.render_segm())
            num_views = views.shape[0] if views.ndim == 3 else 1
            if num_views != self.env.num_envs:
                raise ValueError(
                    f"Camera '{self.camera_name}' renders {num_views} view(s) for {self.env.num_envs} envs: a shared view "
                    f"holds every env's copy of the target under one segmentation id, so their points would mix into "
                    f"one cloud. Fix: render each env — EnvOptions(env_separate_rigid=True), or the batch renderer — or "
                    f"configure object_cloud='surface' (with filter_collisions=False), which reads no camera."
                )
        return camera

    def sample_grasps(self, entity_name: str, env_idx: int = 0) -> GraspResult:
        """Sample grasps of entity ``entity_name`` in its current pose, in env ``env_idx``.

        Parameters
        ----------
        entity_name : str
            The entity to grasp.
        env_idx : int
            The env to sample in. The returned poses are world-frame poses of that env's copy.

        Returns
        -------
        GraspResult
            World-frame grasps, best first. Empty (``success`` is False) when no grasp clears
            ``grasp_threshold`` or, with ``filter_collisions``, the collision check.

        Raises
        ------
        ValueError
            When there are no points of the entity to sample from: the camera sees none of it
            (``object_cloud="camera"``). That is a different failure from "the model found no grasp",
            so it is not reported as an empty result.
        """
        return self.sample_grasps_batch(entity_name, [env_idx])[0]

    def sample_grasps_batch(self, entity_name: str, envs_idx: Sequence[int] | None = None) -> list[GraspResult]:
        """Sample grasps of entity ``entity_name`` in several envs at once.

        The camera renders once for all of them and the model runs once on all their clouds, where
        :meth:`sample_grasps` per env would render the whole batch and run the model for each. A
        scripted demonstrator that runs one routine per env samples every env here first and hands
        each routine its result.

        Parameters
        ----------
        entity_name : str
            The entity to grasp.
        envs_idx : Sequence[int] | None
            The envs to sample in; ``None`` samples every env.

        Returns
        -------
        list[GraspResult]
            One result per entry of ``envs_idx``, in its order; each as :meth:`sample_grasps` returns it.
        """
        entity = resolve_named_entity(
            self.env.entities, entity_name, hint="sample_grasps(entity_name=...) names the entity to grasp."
        )
        envs_idx = list(range(self.env.num_envs)) if envs_idx is None else [int(i) for i in envs_idx]
        for env_idx in envs_idx:
            if not 0 <= env_idx < self.env.num_envs:
                raise IndexError(f"env_idx={env_idx} is out of range for this env's num_envs={self.env.num_envs}.")
        grasp_T_pose = self._pose_frame()
        clouds = self._read_clouds(entity_name, entity, envs_idx)

        outputs = self._infer_batch([object_points for object_points, _ in clouds])
        if len(outputs) != len(clouds):
            raise RuntimeError(
                f"{type(self).__name__}._infer_batch returned {len(outputs)} results for {len(clouds)} clouds; it "
                f"must return one (transforms, confidences) pair per cloud, in order."
            )
        return [
            self._finish(grasps, confidences, scene_points, grasp_T_pose)
            for (grasps, confidences), (_, scene_points) in zip(outputs, clouds)
        ]

    def _finish(
        self, grasps: np.ndarray, confidences: np.ndarray, scene_points: np.ndarray | None, grasp_T_pose: np.ndarray
    ) -> GraspResult:
        """Check one cloud's grasp frames, filter them, place the posed link, rank them."""
        if grasps.shape != (len(confidences), 4, 4) or confidences.shape != (len(confidences),):
            raise RuntimeError(
                f"{type(self).__name__}._infer returned transforms of shape {grasps.shape} and confidences of shape "
                f"{confidences.shape}; it must return (k, 4, 4) and (k,) for the same k."
            )
        if self.filter_collisions and len(grasps) > 0:
            if len(scene_points) == 0:  # nothing but the target and the ignored entities in view
                keep = np.ones(len(grasps), dtype=bool)
            else:
                keep = np.asarray(self._collision_free(grasps, self._downsample(scene_points)))
            if keep.dtype != bool or keep.shape != (len(grasps),):
                raise RuntimeError(
                    f"{type(self).__name__}._collision_free returned a mask of dtype {keep.dtype} and shape "
                    f"{keep.shape}; it must return a bool mask of shape ({len(grasps)},)."
                )
            en.logger.info(f"{type(self).__name__}: {int(keep.sum())}/{len(grasps)} grasps are collision-free.")
            grasps, confidences = grasps[keep], confidences[keep]
        return GraspResult.from_transforms(
            grasps @ grasp_T_pose, confidences, self.env.device, max_grasps=self.max_grasps
        )

    def _read_clouds(
        self, entity_name: str, entity: Entity, envs_idx: list[int]
    ) -> list[tuple[np.ndarray, np.ndarray | None]]:
        """Per env of ``envs_idx``: target points and obstacle cloud (``None`` when unread), world frame."""
        if self.camera is None:
            return [(points, None) for points in self._surface_points(entity, envs_idx)]

        # The visualizer refreshes once per sim step: a kinematic setter since the last step (set_qpos,
        # set_pos) would otherwise render the poses the entities had before it. update_visual_states(
        # force_render=True) is not enough — it moves the geoms but not what the rasterizer draws.
        self.env.scene.visualizer.update()
        _, _, segmentation, _ = self.camera.render(rgb=False, segmentation=True)
        points, valid = self.camera.render_pointcloud(world_frame=True)
        segmentation, points, valid = (tensor_to_array(x) for x in (segmentation, points, valid))
        if points.ndim == 3:  # one shared view, so one env (checked at construction): give it the env axis
            segmentation, points, valid = segmentation[None], points[None], valid[None]

        target_ids = self._segmentation_ids(entity_name, entity)
        ignored_ids = [
            # An entity with no segmentation id renders nothing, so there is nothing of it to leave out.
            self.env.entity_segmentation_ids.get(self.env.entities[name].idx, [])
            for name in self._ignored_entity_names
        ]
        surface = self._surface_points(entity, envs_idx) if self.object_cloud == "surface" else None
        clouds = []
        for k, env_idx in enumerate(envs_idx):
            env_segmentation, env_points, env_valid = segmentation[env_idx], points[env_idx], valid[env_idx]
            object_mask = env_valid & np.isin(env_segmentation, target_ids)
            scene_mask = env_valid & ~object_mask
            for ids in ignored_ids:
                scene_mask &= ~np.isin(env_segmentation, ids)
            if surface is not None:
                object_points = surface[k]
            else:
                object_points = np.ascontiguousarray(env_points[object_mask], dtype=np.float32)
                if len(object_points) == 0:
                    raise ValueError(
                        f"Camera '{self.camera_name}' sees no point of entity '{entity_name}'"
                        f"{f' in env {env_idx}' if self.env.num_envs > 1 else ''}, so there is nothing to "
                        f"sample grasps from. Fix: aim the camera at it (cam_pos / cam_lookat in its CameraOptions), "
                        f"or configure object_cloud='surface' to sample its whole surface instead."
                    )
            clouds.append((object_points, env_points[scene_mask].astype(np.float32)))
        return clouds

    def _segmentation_ids(self, entity_name: str, entity: Entity) -> list[int]:
        """Every segmentation id of ``entity``: one per link (or geom) at a finer segmentation level."""
        ids = self.env.entity_segmentation_ids.get(entity.idx)
        if not ids:
            raise ValueError(
                f"Entity '{entity_name}' has no segmentation id, so its points cannot be told apart in camera "
                f"'{self.camera_name}'. Fix: configure object_cloud='surface' (and filter_collisions=False), which "
                f"needs no segmentation."
            )
        return ids

    def _surface_points(self, entity: Entity, envs_idx: list[int]) -> list[np.ndarray]:
        """Per env of ``envs_idx``: ``surface_points`` points over ``entity``'s visual surface, world frame.

        The visual mesh, not the collision geometry: a convex decomposition would hand the model hulls,
        and the faces where its pieces meet inside the object.
        """
        canonical = get_entity_surface_points(entity, n_points=self.surface_points, use_visual_mesh=True)
        world = tensor_to_array(update_entity_surface_points(entity, canonical))
        if world.ndim == 2:  # (n, 3): one env
            world = world[None]
        return [np.ascontiguousarray(world[env_idx], dtype=np.float32) for env_idx in envs_idx]

    def _downsample(self, scene_points: np.ndarray) -> np.ndarray:
        if len(scene_points) <= self.max_scene_points:
            return scene_points
        return scene_points[self._rng.choice(len(scene_points), self.max_scene_points, replace=False)]

    @abstractmethod
    def _infer(self, object_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run the model on ``object_points``, shape ``(n, 3)``, world frame.

        Returns the grasps that clear ``grasp_threshold``, as ``(k, 4, 4)`` world-frame *grasp frames*
        (z approach, x closing) and their ``(k,)`` confidences, in any order — the base class filters,
        places the posed link and ranks them.
        """

    def _infer_batch(self, clouds: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        """:meth:`_infer` on each cloud, in order. A backend whose model batches overrides this."""
        return [self._infer(object_points) for object_points in clouds]

    def _collision_free(self, grasps: np.ndarray, scene_points: np.ndarray) -> np.ndarray:
        """Boolean mask, shape ``(k,)``, of the grasp frames ``grasps`` whose gripper clears ``scene_points``.

        Only called with ``filter_collisions=True``. A backend whose grippers carry no collision
        geometry leaves this as is and refuses ``filter_collisions=True`` when constructed.
        """
        raise NotImplementedError(f"{type(self).__name__} does not filter grasps by collision.")
