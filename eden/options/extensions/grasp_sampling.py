"""Configuration options for grasp-sampling extensions.

:class:`GraspSamplerOptions` mirrors
:class:`~eden.extensions.grasp_sampling.base.GraspSamplerBase`'s attributes, so the fields every
backend shares are type-checked at ``configure()`` time. Backend-specific knobs stay as class
attributes on the backend and arrive as extras — e.g. GraspGen's ``gripper_config`` or GraspGenX's
``gripper_name`` / ``planner``.

:class:`GraspMoEOptions` groups GraspGenX's GraspMoE planner knobs so they validate as a unit
(``extra="forbid"`` plus ``Literal`` choices) instead of reaching the planner unchecked.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from eden.options.options import ConfigurableOptions, DynamicConfigurableOptions


class GraspSamplerOptions(DynamicConfigurableOptions):
    """Options shared by every grasp-sampling backend.

    Parameters
    ----------
    camera_name : str
        Camera whose point cloud the sampler reads: the target's visible points when
        ``object_cloud="camera"``, and the obstacle cloud ``filter_collisions`` checks against. It
        must use the legacy camera backend (no ``sensor_options``), the one that renders point
        clouds and segmentation. Not needed when neither of those reads it.
    object_cloud : Literal["camera", "surface"]
        Where the target's points come from. ``"camera"`` takes the pixels the camera sees of it,
        which is what a real robot has. ``"surface"`` samples the entity's whole surface, hidden
        underside included, so the model also proposes grasps from below whatever it rests on.
    surface_points : int
        Points sampled over the entity's surface when ``object_cloud="surface"``.
    grasp_threshold : float
        Discriminator confidence, in ``[0, 1]``, a grasp needs to be returned.
    num_grasps : int
        Grasps the generator proposes per inference pass, before thresholding.
    max_grasps : int | None
        Keep at most this many of the highest-confidence grasps. ``None`` keeps every grasp that
        clears ``grasp_threshold``.
    filter_collisions : bool
        Drop grasps whose gripper collision mesh comes within ``collision_threshold`` of the
        camera's obstacle cloud. Needs a backend whose grippers ship real collision geometry.
    collision_threshold : float
        Distance in meters under which a gripper surface point counts as colliding.
    collision_ignore_entities : list[str] | None
        Entities left out of the obstacle cloud; names are checked whether or not the filter is on.
        ``None`` leaves out every actuated entity: the arm and gripper that move to reach the grasp
        would otherwise reject the grasps they are about to reach. ``[]`` keeps every entity.
    max_scene_points : int
        Randomly downsample the obstacle cloud to at most this many points before the check.
    seed : int
        Seed of the obstacle cloud's downsampling. It does not make grasps reproducible: neither the
        surface sampling nor the model's own sampling reads it.
    end_effector : str | None
        The Eden robot that executes the grasps, by its robot options class name (e.g.
        ``"FrankaEmikaPanda"``, ``"InspireRH56_R"``). The backend picks the gripper it pairs with that
        robot, and each pose then places that robot's link (``pose_link_name``). Each backend lists the
        pairings it has verified in its ``END_EFFECTORS``. ``None`` leaves the gripper to the backend's
        own option and the poses in the backend's own frame.
    """

    camera_name: str = "camera"
    object_cloud: Literal["camera", "surface"] = "camera"
    surface_points: int = Field(default=2048, gt=0)
    grasp_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    num_grasps: int = Field(default=200, gt=0)
    max_grasps: int | None = Field(default=None, gt=0)
    filter_collisions: bool = False
    collision_threshold: float = Field(default=0.02, gt=0.0)
    collision_ignore_entities: list[str] | None = None
    max_scene_points: int = Field(default=8192, gt=0)
    seed: int = 0
    end_effector: str | None = None


class GraspMoEOptions(ConfigurableOptions):
    """GraspGenX's GraspMoE planner knobs (``GraspGenXSampler.planner="graspmoe"``).

    GraspMoE scores the union of the diffusion model's grasps and a set of grasps swept over the
    object's oriented bounding box (OBB). These fields tune only the OBB branch. The defaults are
    those of GraspGenX's own ``scripts/demo_object_pc.py``.

    Parameters
    ----------
    num_yaws : int
        World-Z yaw samples of the OBB sweep.
    z_offsets_cm : tuple[float, ...]
        Offsets in cm, relative to the OBB top, at which candidates are placed.
    outlier_threshold : float
        Distance threshold in meters of the outlier removal run on the object cloud.
    outlier_k : int
        Neighbour count of that outlier removal.
    obb_mode : Literal["advanced", "pca"]
        How the OBB is fitted.
    skip_obb_rule : Literal["auto", "never"]
        ``"auto"`` skips the OBB branch when every OBB extent exceeds the gripper's width (there is
        nothing to grasp across); ``"never"`` always runs it.
    obb_density : Literal["sparse", "dense", "dense-topandside"]
        How densely candidates are placed over the OBB: ``"sparse"`` puts every candidate at its
        centroid, ``"dense"`` sweeps positions along its long axis, ``"dense-topandside"`` adds
        side approaches on its four vertical faces.
    obb_position_spacing_cm : float
        Spacing in cm of candidate positions along the OBB's long axis; read by the ``dense``
        densities only.
    """

    num_yaws: int = Field(default=36, gt=0)
    z_offsets_cm: tuple[float, ...] = (-2.0, 0.0)
    outlier_threshold: float = Field(default=0.014, gt=0.0)
    outlier_k: int = Field(default=20, gt=0)
    obb_mode: Literal["advanced", "pca"] = "advanced"
    skip_obb_rule: Literal["auto", "never"] = "auto"
    obb_density: Literal["sparse", "dense", "dense-topandside"] = "dense-topandside"
    obb_position_spacing_cm: float = Field(default=1.0, gt=0.0)
