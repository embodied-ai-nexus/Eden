"""Camera and multi-camera configuration options."""

from typing import Any, Literal

from genesis.typing import UnitVec4FType, Vec2IType, Vec3FType
from pydantic import model_validator

from eden.options.options import ConfigurableOptions, DynamicConfigurableOptions
from eden.utils.ndarray_field import FloatArray
from eden.utils.params import invalid_parameters_message, unknown_parameter_names, validated_parameter_values


class CameraOverlayOptions(DynamicConfigurableOptions):
    """One overlay a camera paints onto its RGB frames (:mod:`eden.entities.camera_overlays`).

    Built with the overlay class's ``configure()``, e.g.
    ``PlanePaintOverlay.configure(metric_name="drawing", color=(0, 0, 0))``, which fills ``name`` with the overlay's
    registry key and carries its parameters as extras. Those are checked here, when ``name`` resolves: their names
    against the overlay's parameters, their types against the declared ones (converting as a declared field would, so
    a list read back from JSON is the declared tuple again), and their values by the overlay's ``check_parameters()``.
    A typo or a bad value in a config file therefore fails in that file. An overlay whose module is not imported yet is
    checked when the camera is constructed.

    Parameters
    ----------
    name: str | None
        Registry key of the overlay in ``CAMERA_OVERLAY_REGISTRY``.
    """

    name: str | None = None

    def model_post_init(self, context) -> None:
        super().model_post_init(context)
        self._validate_params()

    def _validate_params(self) -> None:
        """Refuse an extra the named overlay has no parameter for (``ManagerTermOptions``' check, for overlays), then a
        value the overlay cannot work with.
        """
        if not self.name:
            return
        # Deferred: the overlays' module imports this one.
        from eden.entities.camera_overlays import CAMERA_OVERLAY_REGISTRY

        if self.name not in CAMERA_OVERLAY_REGISTRY:
            return
        overlay = CAMERA_OVERLAY_REGISTRY.get(self.name)
        invalid = unknown_parameter_names(self.__pydantic_extra__ or {}, overlay)
        if invalid:
            raise ValueError(
                invalid_parameters_message(
                    f"overlay '{self.name}' ({overlay.__name__})",
                    invalid,
                    overlay.get_configure_specs(),
                    overlay._retired_params,
                )
            )
        # Extras are kept as written; give them their declared types, as a declared options field would get.
        extras = validated_parameter_values(overlay, self.__pydantic_extra__ or {}, subject=overlay.__name__)
        if extras:
            self.__pydantic_extra__.update(extras)
        defaults = {name: getattr(overlay, name) for name in overlay.get_parameter_names()}
        overlay.check_parameters({**defaults, **extras})


class CameraOptions(ConfigurableOptions):
    """Configuration for a single camera.

    Parameters
    ----------
    model: Literal["pinhole", "thinlens"]
        Camera model.
    cam_pos: array-like[float, float, float]
        Camera position.
    cam_lookat: array-like[float, float, float]
        Camera lookat.
    cam_resolution: array-like[int, int]
        Camera resolution in (width, height).
    cam_up: array-like[float, float, float]
        Camera up vector.
    cam_intrinsic: array-like, shape (3, 3) | None
        Camera intrinsic matrix, used to set the camera's field of view and resolution.
        If provided, `cam_fov` and `cam_resolution` will be ignored.
    cam_extrinsic: array-like, shape (4, 4) | None
        Camera extrinsic matrix, used to set the camera's position, lookat, and up.
        The extrinsic matrix is the inverse of the transform from the camera frame to the world frame.
        If provided, `cam_pos`, `cam_lookat`, and `cam_up` will be ignored.
    cam_aperture: float
        Camera aperture.
    cam_fov: float
        Vertical field of view, in degrees.
    cam_near: float | None
        Near clipping plane in meters. ``None`` leaves Genesis's default (rasterizer/batch renderer only).
    cam_far: float | None
        Far clipping plane in meters. ``None`` leaves Genesis's default (rasterizer/batch renderer only).
    debug: bool
        Whether to enable debug mode.
    follow_entity_name: str
        Name of the entity to follow.
    attach_entity_name: str
        Name of the entity to attach the camera to.
    attach_link_name: str
        Name of the link to attach the camera to.
    attach_offset: array-like[float, float, float] | None
        Offset of the camera from the link.
    attach_quat: array-like[float, float, float, float] | None
        Quaternion of the camera from the link (w, x, y, z).
    overlays: list[CameraOverlayOptions]
        What :meth:`~eden.entities.camera.Camera.render_rgb` paints onto every frame after the renderer made it, in
        order: a background composite (``BackgroundImageComposite`` for a backdrop image, ``SplatComposite`` for a
        splat env's Gaussian splat), then state the task tracks but the scene does not render, such as the ink on a
        whiteboard. Reaches the recording and anything else built on ``render_rgb``; an overlay leaves the pixels a
        composite before it replaced. An overlay that reads the point cloud costs a second render of the view per
        frame. See :mod:`eden.entities.camera_overlays`.
    sensor_options: Any | None
        Genesis sensor options (e.g. RasterizerCameraOptions, NyxCameraOptions).
        When provided, the camera uses scene.add_sensor() instead of scene.add_camera().
        The sensor backend only supports RGB rendering; depth/segmentation/normal/pointcloud
        are not available. Fields pos/lookat/fov/res/up are auto-populated from
        cam_pos/cam_lookat/cam_fov/cam_resolution/cam_up during build if not explicitly set.
    """

    model: Literal["pinhole", "thinlens"] = "pinhole"
    cam_pos: Vec3FType = (1.5, 0.0, 1)
    cam_lookat: Vec3FType = (0.0, 0.0, 0.5)
    cam_resolution: Vec2IType = (720, 480)
    cam_up: Vec3FType = (0.0, 0.0, 1.0)
    cam_intrinsic: FloatArray[3, 3] | None = None
    cam_extrinsic: FloatArray[4, 4] | None = None
    cam_aperture: float = 2.0
    cam_fov: float = 80.0
    cam_near: float | None = None
    cam_far: float | None = None
    debug: bool = False

    follow_entity_name: str = ""
    attach_entity_name: str = ""
    attach_link_name: str = ""
    attach_offset: Vec3FType | None = None
    attach_quat: UnitVec4FType | None = None

    overlays: list[CameraOverlayOptions] = []

    sensor_options: Any | None = None

    @model_validator(mode="after")
    def _validate_clipping_planes(self):
        """Fail fast if the near/far clipping planes are non-positive or inverted."""
        if self.cam_near is not None and self.cam_near <= 0.0:
            raise ValueError(f"cam_near must be positive, got {self.cam_near}")
        if self.cam_far is not None and self.cam_far <= 0.0:
            raise ValueError(f"cam_far must be positive, got {self.cam_far}")
        if self.cam_near is not None and self.cam_far is not None and self.cam_far <= self.cam_near:
            raise ValueError(
                f"cam_far must be greater than cam_near, got cam_near={self.cam_near}, cam_far={self.cam_far}"
            )
        return self


class CamerasOptions(DynamicConfigurableOptions):
    """Container mapping camera names to their :class:`CameraOptions`.

    Parameters
    ----------
    <camera_name>: CameraOptions
        The camera configuration to be used for the given camera.
    """

    def __init__(self, **data):
        entity_data = {}
        for key, entity in data.items():
            if isinstance(entity, CameraOptions):
                entity_data[key] = entity
            else:
                entity_data[key] = CameraOptions(**entity)

        super().__init__(**entity_data)
