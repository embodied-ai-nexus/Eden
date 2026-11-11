"""Camera entity wrappers, including the gaussian-splat SplatCamera."""

from __future__ import annotations

import contextlib
import dataclasses
import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import genesis as gs
import numpy as np
import torch
from genesis.options.sensors import RasterizerCameraOptions
from genesis.typing import Vec2IType, Vec3FType
from genesis.utils.geom import T_to_pos_lookat_up, trans_quat_to_T
from genesis.utils.misc import tensor_to_array
from genesis.utils.video_encoder import VideoEncoder

import eden as en
from eden.entities.camera_overlays import (
    BackgroundComposite,
    CameraOverlay,
    OverlayBuffers,
    build_camera_overlay,
)
from eden.options.camera import CameraOptions, CameraOverlayOptions
from eden.utils.common import ConfigurableMixin, mirror_options_onto
from eden.utils.misc import get_now, resolve_named_entity, sanitize_envs_idx
from eden.utils.torch import numpy_to_tensor

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


def _apply_rasterizer_sensor_shadow_setting(sensor, *, shadow: bool) -> None:
    """Apply Eden's scene shadow mode to a Genesis Rasterizer camera sensor.

    A sensor-backed rasterizer camera does not render through the scene visualizer:
    ``RasterizerCameraSensor._create_standalone_context`` builds its own ``VisOptions`` from a fixed handful of fields,
    and the scene's is not one of its inputs. So the ``shadow=`` Eden hands ``gs.options.VisOptions`` reaches the viewer
    and never the sensor, and the pinned revision exposes neither a sensor option nor a public setter for it. Keep the
    required private access in this one boundary until Genesis propagates the scene visualization option to that
    standalone context.

    A built sensor always holds a context carrying ``shadow`` (``RasterizerContext.__init__`` assigns it), so raise
    rather than skip when either is missing: Genesis having reworked this is the one thing that reaches here, and
    dropping the setting instead would render shadows back with nothing to say why. The same path is pinned by
    ``tests/entities/test_camera.py``, so a Genesis bump fails there first.
    """
    context = getattr(sensor._shared_metadata, "context", None)
    if context is None or not hasattr(context, "shadow"):
        raise RuntimeError(
            f"Genesis' {type(sensor).__name__} no longer exposes `_shared_metadata.context.shadow`, so "
            "`EnvOptions.rasterizer_shadows` cannot be applied to a camera sensor. Genesis has reworked its "
            "standalone Rasterizer context: re-point this shim, or delete it if the scene's `VisOptions.shadow` "
            "now reaches camera sensors on its own."
        )
    context.shadow = shadow


@contextlib.contextmanager
def _rasterizer_shadows_off(visualizer):
    """Render through the scene rasterizer with shadows off inside the block, then restore both switches.

    Which switch the rasterizer reads depends on whether a viewer is open. Headless, it renders offscreen and
    passes ``shadow=context.shadow`` on every call. With a viewer, it renders through the viewer's own GL context
    (``Viewer.render_offscreen``), which takes no ``shadow`` argument and reads ``render_flags["shadows"]``, set
    once when the viewer is created. Clearing only the context would then render shadows anyway, so every
    background pixel would read as fully lit. Both are cleared, and each is restored to the value it held rather
    than to ``True``.

    A viewer running in its own thread reads the same flag for its on-screen frames, so one of those can be
    drawn without shadows while the block runs. That only affects the window, not what any camera returns.
    Genesis exposes no public setter for either switch; ``tests/entities/test_camera.py`` exercises both (the viewer
    one through a stand-in, as the suite cannot open a viewer).
    """
    context = visualizer.context
    viewer = visualizer.viewer
    flags = None
    if viewer is not None:
        flags = getattr(getattr(viewer, "_pyrender_viewer", None), "render_flags", None)
        if flags is None or "shadows" not in flags:
            raise RuntimeError(
                "Genesis' Viewer no longer exposes `_pyrender_viewer.render_flags['shadows']`, so a background "
                "composite with shadow_strength cannot render its unshadowed pass while the viewer is open. Genesis "
                "has reworked its viewer: re-point `_rasterizer_shadows_off` in eden/entities/camera.py."
            )
    context_shadow = context.shadow
    viewer_shadow = None if flags is None else flags["shadows"]
    context.shadow = False
    if flags is not None:
        flags["shadows"] = False
    try:
        yield
    finally:
        context.shadow = context_shadow
        if flags is not None:
            flags["shadows"] = viewer_shadow


@dataclass
class _Recording:
    """The video recording a camera has open: where it writes, at what rate, with what encoders.

    ``encoders`` stays empty until the first frame, because how many videos a recording
    writes is only known once something has been rendered (see ``Camera._record_frame``).
    A camera holds one of these, or ``None`` when it is not recording.
    """

    path: str
    fps: float
    encoders: list[VideoEncoder] = field(default_factory=list)


def resolved_cam_fov(options: CameraOptions) -> float:
    """The vertical field of view, in degrees, of a camera built from ``options``.

    ``cam_intrinsic``, when set, wins over ``cam_fov``. This is the value :class:`Camera` uses, for a caller that needs
    it before the camera exists (a renderer that bakes the optics into a file).
    """
    if options.cam_intrinsic is None:
        return float(options.cam_fov)
    intrinsic = options.cam_intrinsic
    return float(2 * np.arctan(intrinsic[1, 2] / intrinsic[1, 1]) * 180 / np.pi)


class Camera(ConfigurableMixin[CameraOptions]):
    """Wrapper around a Genesis camera, configured from :class:`CameraOptions`.

    Parameters
    ----------
    model: Literal["pinhole", "thinlens", "fisheye"]
        Camera model.
    cam_pos: array-like[float, float, float]
        Camera position in world frame.
    cam_lookat: array-like[float, float, float]
        Camera lookat in world frame.
    cam_resolution: array-like[int, int]
        Camera resolution in (width, height).
    cam_up: array-like[float, float, float]
        Camera up vector.
    cam_aperture: float
        Camera aperture.
    cam_fov: float
        The vertical field of view of the camera in degrees.
    cam_near: float | None
        Near clipping plane in meters. ``None`` leaves Genesis's default (rasterizer/batch renderer only).
    cam_far: float | None
        Far clipping plane in meters. ``None`` leaves Genesis's default (rasterizer/batch renderer only).
    debug: bool
        Whether this is a debug camera. A debug camera will use non-batch renderer.
    follow_entity_name: str
        Name of the entity to follow.
    attach_entity_name: str
        Name of the entity to attach the camera to.
    attach_link_name: str
        Name of the link to attach the camera to.
    overlays: list[CameraOverlayOptions]
        What :meth:`render_rgb` paints onto every frame, in order (:mod:`eden.entities.camera_overlays`).
    """

    model: Literal["pinhole", "thinlens", "fisheye"] = "pinhole"
    cam_pos: Vec3FType = (5.0, 0.0, 1.8)
    cam_lookat: Vec3FType = (0.0, 0.0, 0.5)
    cam_resolution: Vec2IType = (1280, 720)
    cam_up: Vec3FType = (0.0, 0.0, 1.0)
    cam_aperture: float = 2.0
    cam_fov: float = 55.0
    cam_near: float | None = None
    cam_far: float | None = None
    debug: bool = False
    follow_entity_name: str = ""
    attach_entity_name: str = ""
    attach_link_name: str = ""
    overlays: list[CameraOverlayOptions] = []

    if TYPE_CHECKING:
        # (4, 4) camera-to-link transform from attach_offset/attach_quat; a per-env
        # (num_envs, 4, 4) torch stack after set_attach_transform installed batched mounts.
        attach_T: np.ndarray | torch.Tensor

    def __init__(self, env: EnvBase, options: CameraOptions, key: str = ""):
        self._uid = gs.UID()
        self._options = options
        self._env = env
        #: The camera's key in ``EnvBase.cameras`` ("" when built outside an env). Not ``name``, which a configurable
        #: already has: its class's registry-style name, ``"camera"``.
        self.key = key

        # CameraOptions declares every Camera parameter, so today this only ever takes the
        # options branch -- the class-default fallback is unreachable here. It is still routed
        # through the shared helper so a parameter added to Camera/SplatCamera but not to the
        # options (``overlays`` is a mutable class default) cannot reintroduce the
        # aliasing (see mirror_options_onto).
        mirror_options_onto(self, self._options)

        # Resolve cam_intrinsic → override cam_fov / cam_resolution on self
        if self._options.cam_intrinsic is not None:
            intrinsic = self._options.cam_intrinsic
            assert intrinsic.shape == (3, 3), "`cam_intrinsic` should be a 3x3 matrix"
            en.logger.warning("`cam_intrinsic` is provided, ignoring `cam_fov` and `cam_resolution`")
            self.cam_fov = resolved_cam_fov(self._options)
            self.cam_resolution = (
                int(math.ceil(intrinsic[0, 2] * 2)),
                int(math.ceil(intrinsic[1, 2] * 2)),
            )
            en.logger.info(
                f"Using `cam_intrinsic` to set `cam_fov`: {self.cam_fov} and `cam_resolution`: {self.cam_resolution}"
            )

        # Resolve cam_extrinsic → override cam_pos / cam_lookat / cam_up on self
        if self._options.cam_extrinsic is not None:
            extrinsic = self._options.cam_extrinsic
            assert extrinsic.shape == (4, 4), "`cam_extrinsic` should be a 4x4 matrix"
            en.logger.warning("`cam_extrinsic` is provided, ignoring `cam_pos`, `cam_lookat`, and `cam_up`")
            T = np.linalg.inv(extrinsic)
            pos, lookat, up = T_to_pos_lookat_up(T)
            self.cam_pos = tuple(pos)
            self.cam_lookat = tuple(lookat)
            self.cam_up = tuple(up)

        # Validate follow vs attach
        if self._options.follow_entity_name != "":
            assert self._options.attach_entity_name == "", (
                "`attach_entity_name` should be empty if `follow_entity_name` is provided"
            )
            assert self._options.attach_link_name == "", (
                "`attach_link_name` should be empty if `follow_entity_name` is provided"
            )
            assert self._options.attach_offset is None, (
                "`attach_offset` should be None if `follow_entity_name` is provided"
            )
            assert self._options.attach_quat is None, "`attach_quat` should be None if `follow_entity_name` is provided"
        elif self._options.attach_entity_name != "":
            assert self._options.follow_entity_name == "", (
                "`follow_entity_name` should be empty if `attach_entity_name` is provided"
            )
            assert self._options.attach_link_name != "", (
                "`attach_link_name` should be provided if `attach_entity_name` is provided"
            )
            assert self._options.attach_offset is not None, (
                "`attach_offset` should be provided if `attach_entity_name` is provided"
            )
            assert self._options.attach_quat is not None, (
                "`attach_quat` should be provided if `attach_entity_name` is provided"
            )

        # Compute attach_T from offset/quat (stored on self, not options)
        if self._options.attach_entity_name != "" and self._options.attach_offset is not None:
            self.attach_T = trans_quat_to_T(
                np.array(self._options.attach_offset),
                np.array(self._options.attach_quat),
            )

        # Determine backend: sensor (add_sensor) vs legacy (add_camera)
        self._sensor_options = self._options.sensor_options
        self._cam = None  # Legacy camera object (set in pre_build)
        self._sensor = None  # Sensor object (set in pre_build)
        # Video recording is Eden-owned for both backends: one frame per render call,
        # streamed to file (see start_recording).
        self._recording: _Recording | None = None

        # Constructed now, so an unknown overlay or a misspelled parameter fails at env construction; built in a
        # post-build hook, once the managers they read exist.
        self._overlays: list[CameraOverlay] = [
            build_camera_overlay(self, overlay_options, i) for i, overlay_options in enumerate(self.overlays)
        ]
        composites = [type(overlay).__name__ for overlay in self._overlays if isinstance(overlay, BackgroundComposite)]
        if len(composites) > 1:
            raise ValueError(
                f"Camera {self.display_name}: CameraOptions.overlays declares {len(composites)} background composites "
                f"({composites}); a camera has one background. Keep one, and list in its entities every entity it "
                "should replace."
            )
        self._overlays_built = False
        self._frame_envs_idx_cache: dict[tuple[int, bool], torch.Tensor] = {}

    @property
    def _uses_sensor_backend(self) -> bool:
        """Whether this camera uses the sensor API (add_sensor) instead of legacy add_camera."""
        return self._sensor_options is not None

    @property
    def supports_pointcloud(self) -> bool:
        """Whether :meth:`render_pointcloud` is available (legacy backend only).

        Lets a caller that depends on point clouds reject an incompatible camera up front
        instead of hitting :class:`NotImplementedError` on the first render.
        """
        return not self._uses_sensor_backend

    def pre_build(self):
        if self._uses_sensor_backend:
            self._pre_build_sensor()
        else:
            self._pre_build_legacy()

    def _pre_build_legacy(self):
        """Build camera using the legacy scene.add_camera() API."""
        # Only forward near/far when set, so leaving them None keeps Genesis's own defaults.
        clip_kwargs = {}
        if self.cam_near is not None:
            clip_kwargs["near"] = self.cam_near
        if self.cam_far is not None:
            clip_kwargs["far"] = self.cam_far
        self._cam = self._env.scene.add_camera(
            model=self.model,
            pos=self.cam_pos,
            lookat=self.cam_lookat,
            res=self.cam_resolution,
            fov=self.cam_fov,
            up=self.cam_up,
            aperture=self.cam_aperture,
            **clip_kwargs,
            GUI=False,
            debug=self.debug,
        )

        # Genesis's legacy camera does not validate far > near (it silently builds a degenerate
        # projection). A lone cam_near/cam_far can also cross Genesis's own default on the other
        # side, which the config-time validator cannot see. Check the resolved pair, fail loud.
        if (self.cam_near is not None or self.cam_far is not None) and self._cam.far <= self._cam.near:
            raise ValueError(
                f"Camera {self.display_name} near/far clipping planes are inverted once Genesis defaults "
                f"are resolved: near={self._cam.near}, far={self._cam.far}. A one-sided cam_near/"
                f"cam_far must still sit on the correct side of Genesis's default (near < far)."
            )

        # NOTE: moving camera for moving entity
        if self.follow_entity_name != "":
            follow_entity = resolve_named_entity(
                self._env.entities,
                self.follow_entity_name,
                hint="CameraOptions.follow_entity_name names it; set it to one of those.",
            )
            self._cam.follow_entity(
                follow_entity._entity,
                fix_orientation=True,
            )

        if self.attach_entity_name != "":
            attach_entity = resolve_named_entity(
                self._env.entities,
                self.attach_entity_name,
                hint="CameraOptions.attach_entity_name names it; set it to one of those.",
            )
            rigid_link = attach_entity.get_link(self.attach_link_name)
            self._cam.attach(rigid_link, self.attach_T)

    def _pre_build_sensor(self):
        """Build camera using the sensor API (scene.add_sensor())."""
        # Shallow, not deep: every write below assigns a whole field, so nothing nested is ever
        # mutated in place and the caller's options stay untouched either way. A deep copy would
        # additionally have to *copy* whatever a backend's options carry, and a renderer's own
        # opaque handles cannot be — `NyxCameraOptions.env_maps` / `light_fields` hold native
        # `gs_nyx` assets, which `copy.deepcopy` rejects outright ("cannot pickle ...").
        sensor_opts = self._sensor_options.model_copy()

        # Auto-populate sensor_options fields from CameraOptions for any field
        # the user did not explicitly set when constructing sensor_options.
        _defaults = {
            "pos": list(self.cam_pos),
            "lookat": list(self.cam_lookat),
            "up": list(self.cam_up),
            "fov": self.cam_fov,
            "res": self.cam_resolution,
        }
        for attr, val in _defaults.items():
            if hasattr(sensor_opts, attr) and attr not in sensor_opts.model_fields_set:
                setattr(sensor_opts, attr, val)

        # Propagate near/far only when set (None keeps the sensor's own defaults). Unlike the
        # pose/fov defaults above, a clipping plane is part of the trained depth observation, so it
        # must never be silently dropped. Fail loud on a backend with no such field (e.g. the
        # raytracer sensor), and fail loud on a conflict with an explicitly-set sensor_options plane
        # rather than letting one value quietly win.
        for plane, value in (("near", self.cam_near), ("far", self.cam_far)):
            if value is None:
                continue
            if not hasattr(sensor_opts, plane):
                raise ValueError(
                    f"cam_{plane}={value} was set, but {type(sensor_opts).__name__} has no "
                    f"'{plane}' clipping plane. Remove cam_{plane}, or use a sensor backend that "
                    f"supports it (e.g. RasterizerCameraOptions / BatchRendererCameraOptions)."
                )
            if plane in sensor_opts.model_fields_set and getattr(sensor_opts, plane) != value:
                raise ValueError(
                    f"cam_{plane}={value} conflicts with sensor_options.{plane}="
                    f"{getattr(sensor_opts, plane)}. Set the {plane} clipping plane in exactly one "
                    f"place — either cam_{plane} on CameraOptions or {plane} on "
                    f"{type(sensor_opts).__name__}, not both."
                )
            setattr(sensor_opts, plane, value)

        # Backstop for the one-sided case the config-time validator cannot see: a lone cam_near/
        # cam_far can cross the sensor's own default on the other side. Validate the resolved pair.
        if (
            (self.cam_near is not None or self.cam_far is not None)
            and hasattr(sensor_opts, "near")
            and hasattr(sensor_opts, "far")
            and sensor_opts.far <= sensor_opts.near
        ):
            raise ValueError(
                f"Camera near/far clipping planes are inverted once {type(sensor_opts).__name__} "
                f"defaults are resolved: near={sensor_opts.near}, far={sensor_opts.far}. "
                f"A one-sided cam_near/cam_far must still satisfy near < far."
            )

        # Inject entity attachment if configured via CameraOptions
        if self.attach_entity_name != "":
            entity = resolve_named_entity(
                self._env.entities,
                self.attach_entity_name,
                hint="CameraOptions.attach_entity_name names it; set it to one of those.",
            )
            sensor_opts.entity_idx = entity.idx
            if self.attach_link_name:
                link = entity.get_link(self.attach_link_name)
                sensor_opts.link_idx_local = link.idx_local
            if hasattr(self, "attach_T"):
                sensor_opts.offset_T = self.attach_T
        elif self.attach_link_name:
            raise ValueError("Should provide attach_entity_name if attach_link_name is provided.")

        self._sensor = self._env.scene.add_sensor(sensor_opts)

    # What a camera overlay may ask of its camera, besides render_unshadowed_rgb() and get_transform().

    @property
    def display_name(self) -> str:
        """How error messages name the camera: its key in ``EnvBase.cameras``, quoted, or its uid without one."""
        return f"'{self.key}'" if self.key else f"(uid {self._uid})"

    @property
    def renderer_kind(self) -> Literal["sensor", "raytracer", "batch_renderer", "rasterizer"]:
        """Which renderer makes this camera's frames. A debug camera rasterizes even in a batch-rendered scene."""
        if self._uses_sensor_backend:
            return "sensor"
        if self._cam._raytracer is not None:
            return "raytracer"
        return "batch_renderer" if self._cam._batch_renderer is not None else "rasterizer"

    @property
    def batch_renderer_casts_shadows(self) -> bool:
        """Whether this camera's batch renderer has a light that casts shadows; ``False`` for any other renderer."""
        if self.renderer_kind != "batch_renderer":
            return False
        return any(light.castshadow for light in self._cam._batch_renderer._lights)

    @property
    def splat_single_view(self) -> bool | None:
        """For a splat env's camera, whether it renders the splat once for every env's image (``True``) or once per
        env (``False``); ``None`` for any other camera.
        """
        return None

    def _build_overlays(self) -> None:
        for overlay in self._overlays:
            overlay.build()
        self._overlays_built = True

    def post_build(self):
        if self._overlays:
            self._env.register_post_build_hook(self._build_overlays)
        if self._uses_sensor_backend:
            if isinstance(self._sensor_options, RasterizerCameraOptions):
                _apply_rasterizer_sensor_shadow_setting(
                    self._sensor,
                    shadow=self._env.env_options.rasterizer_shadows,
                )
            return

    def set_attach_transform(self, offset_T) -> None:
        """Replace the camera-to-link attach transform declared via ``attach_offset``/``attach_quat``.

        Accepts a single ``(4, 4)`` transform or a batched ``(n_rendered, 4, 4)`` stack for
        per-env camera mounts (camera-extrinsics domain randomization): Genesis composes the
        attached pose as ``link_T @ offset_T``, which broadcasts over a batched offset. A
        batched offset requires a batched camera (batch renderer or ``env_separate_rigid``)
        and one transform per rendered env — anything else would only crash later, deep
        inside the next render, so it is rejected here.

        Parameters
        ----------
        offset_T : torch.Tensor | np.ndarray
            The new camera pose(s) relative to the attached link, shape ``(4, 4)`` or
            ``(n_rendered, 4, 4)``.
        """
        if self._uses_sensor_backend:
            raise NotImplementedError(
                "set_attach_transform() is not supported with the sensor backend. "
                "Use the legacy camera backend (remove sensor_options from CameraOptions)."
            )
        if self.attach_entity_name == "":
            raise RuntimeError(
                "set_attach_transform() requires an attached camera — declare "
                "attach_entity_name/attach_link_name (+ attach_offset/attach_quat) in CameraOptions."
            )
        shape = tuple(offset_T.shape)
        if shape[-2:] != (4, 4) or len(shape) not in (2, 3):
            raise ValueError(f"offset_T must be (4, 4) or (n_rendered, 4, 4), got {shape}.")
        if len(shape) == 3:
            # Genesis composes batched poses only for a batched camera, over exactly the
            # rendered envs; a mismatch here surfaces as an opaque matmul/set_pose error
            # inside the next render (and a stray (1, 4, 4) would silently broadcast).
            if not self._cam._is_batched:
                raise ValueError(
                    f"offset_T is batched ({shape}) but camera {self.display_name} is not batched — "
                    "batched mounts need the batch renderer or env_separate_rigid."
                )
            n_rendered = len(self._env.scene.visualizer._context.rendered_envs_idx)
            if shape[0] != n_rendered:
                raise ValueError(
                    f"offset_T has batch size {shape[0]} but the camera renders {n_rendered} envs "
                    "(rendered_envs_idx) — provide one transform per rendered env."
                )
        rigid_link = self._env.entities[self.attach_entity_name].get_link(self.attach_link_name)
        self._cam.attach(rigid_link, offset_T)
        self.attach_T = offset_T

    @property
    def segmentation_idx_dict(self):
        return self._env.scene.segmentation_idx_dict

    def get_pos(self, envs_idx=None):
        """Return the current position of the camera."""
        if self._uses_sensor_backend:
            raise NotImplementedError(
                "get_pos() is not supported with the sensor backend. Use sensor.read() to get rendered data."
            )
        return self._cam.get_pos(envs_idx=envs_idx)

    def get_quat(self, envs_idx=None):
        """Return the current quaternion of the camera."""
        if self._uses_sensor_backend:
            raise NotImplementedError(
                "get_quat() is not supported with the sensor backend. Use sensor.read() to get rendered data."
            )
        return self._cam.get_quat(envs_idx=envs_idx)

    def get_lookat(self, envs_idx=None):
        """Return the current lookat point of the camera."""
        if self._uses_sensor_backend:
            raise NotImplementedError("get_lookat() is not supported with the sensor backend.")
        return self._cam.get_lookat(envs_idx=envs_idx)

    def get_up(self, envs_idx=None):
        """Return the current up vector of the camera."""
        if self._uses_sensor_backend:
            raise NotImplementedError("get_up() is not supported with the sensor backend.")
        return self._cam.get_up(envs_idx=envs_idx)

    def get_transform(self, envs_idx=None):
        """Return the current transform matrix of the camera."""
        if self._uses_sensor_backend:
            raise NotImplementedError("get_transform() is not supported with the sensor backend.")
        return self._cam.get_transform(envs_idx=envs_idx)

    def set_pose(self, transform=None, pos=None, lookat=None, up=None, envs_idx=None):
        """Set the camera pose.

        Parameters
        ----------
        transform : np.ndarray, shape (4, 4) or (N, 4, 4), optional
            The transform matrix of the camera.
        pos : array-like, shape (3,) or (N, 3), optional
            The position of the camera.
        lookat : array-like, shape (3,) or (N, 3), optional
            The lookat point of the camera.
        up : array-like, shape (3,) or (N, 3), optional
            The up vector of the camera.
        envs_idx : array of indices in integers, optional
            The environment indices for which to update the pose. If not provided, the camera pose will be set for the
            specific environment bound to the camera if any, all the environments otherwise.
        """
        if self._uses_sensor_backend:
            raise NotImplementedError("set_pose() is not supported with the sensor backend.")
        if self.is_attached:
            raise RuntimeError(
                "Cannot set pose on an attached camera. Detach it first or modify the attachment offset."
            )
        self._cam.set_pose(transform=transform, pos=pos, lookat=lookat, up=up, envs_idx=envs_idx)

    def set_pos(self, pos, envs_idx=None):
        """Set the camera position, keeping current lookat and up.

        Parameters
        ----------
        pos : array-like, shape (3,) or (N, 3)
            The position of the camera.
        envs_idx : array-like, optional
            Environment indices. If None, applies to all environments.
        """
        if self._uses_sensor_backend:
            raise NotImplementedError("set_pos() is not supported with the sensor backend.")
        if self.is_attached:
            raise RuntimeError(
                "Cannot set position on an attached camera. Detach it first or modify the attachment offset."
            )
        self._cam.set_pose(pos=pos, envs_idx=envs_idx)

    def set_quat(self, quat, envs_idx=None):
        """Set the camera orientation from a quaternion (wxyz), keeping current position.

        Parameters
        ----------
        quat : array-like, shape (4,) or (N, 4)
            Quaternion in (w, x, y, z) format.
        envs_idx : array-like, optional
            Environment indices. If None, applies to all environments.
        """
        if self._uses_sensor_backend:
            raise NotImplementedError("set_quat() is not supported with the sensor backend.")
        if self.is_attached:
            raise RuntimeError(
                "Cannot set orientation on an attached camera. Detach it first or modify the attachment offset."
            )
        # Genesis keeps the camera pose on the host in float64 numpy
        # (https://github.com/Genesis-Embodied-AI/Genesis/pull/3571), so the quaternion and the env index join it there.
        quat = tensor_to_array(quat, dtype=np.float64)
        if envs_idx is None:
            idx = ()
        else:
            # Raises on a malformed or out-of-range index, naming it; any accepted form becomes an index tensor, a mask
            # stays a mask.
            envs_idx = sanitize_envs_idx(envs_idx, self._env.num_envs, prefer_slice=False)
            # A device index costs one sync whatever is done here, since Genesis' set_pose brings envs_idx to the host
            # too: this copy waits for the queued work, and the one in set_pose finds none left.
            idx = tensor_to_array(envs_idx)
        # Use internal _pos (without envs_offset) to match set_pose's coordinate space
        pos = self._cam._pos[idx]
        T = trans_quat_to_T(pos, quat)
        self._cam.set_pose(transform=T, envs_idx=envs_idx)

    @property
    def is_attached(self) -> bool:
        """Whether this camera is attached to an entity link."""
        if self._uses_sensor_backend:
            entity_idx = getattr(self._sensor_options, "entity_idx", None)
            return entity_idx is not None and entity_idx != -1
        return bool(self.attach_entity_name)

    def start_recording(self, video_path: str | None = None, fps: float = 60.0):
        """Open a video file and record one frame per ``render()`` / ``render_rgb()`` call.

        Frames are encoded and streamed to the file as they are rendered, so the
        recording length is not bounded by memory. The output path and framerate are
        fixed for the whole video, hence given here rather than at ``stop_recording()``.

        One frame per render call is Eden's recording contract: a kinematic replay that
        never steps the scene records exactly as a stepped rollout does. Genesis' own
        ``camera.start_recording`` instead captures inside ``scene.step()``, which would
        record nothing for a replay script.

        A camera that renders several environments at once (batch renderer, or
        ``env_separate_rigid``) records one video per environment, named
        ``{stem}_{i}{ext}`` after the environment's position in the rendered batch.

        Parameters
        ----------
        video_path : str, optional
            Output video file path, ending in '.mp4'. Defaults to
            ``{log_dir}/video_{timestamp}.mp4``.
        fps : float, optional
            Framerate the video is encoded at. Must be positive. Playback speed is
            ``fps`` render calls per second, whatever the simulation timestep.
        """
        if self._recording is not None:
            raise RuntimeError(f"Already recording to '{self._recording.path}'. Call stop_recording() first.")
        # Checked here rather than left to the encoder: a non-positive framerate otherwise
        # surfaces as a PyAV error on the first frame, deep inside the render loop.
        if fps <= 0.0:
            raise ValueError(f"Framerate of a recording must be positive; got fps={fps}.")

        self._recording = _Recording(path=video_path or f"{en.log_dir}/video_{get_now()}.mp4", fps=fps)

    def stop_recording(self):
        """Finalize the video file(s) named by :meth:`start_recording`.

        Frames are encoded as they are recorded, so this only flushes the ones the
        encoders still hold and closes the files.

        A recording that captured no frame writes no file at all. That is said out loud
        rather than passed over: it means nothing rendered while the recording was open,
        which is a mistake in the calling loop far more often than an intent.
        """
        if self._recording is None:
            raise RuntimeError(
                "Recording not started. Call start_recording(video_path, fps) first — the output path and "
                "framerate belong to start_recording(), not stop_recording()."
            )

        recording, self._recording = self._recording, None

        if not recording.encoders:
            en.logger.warning(
                f"Recording to '{recording.path}' captured no frame, so no video was written. Frames are "
                "captured by render() / render_rgb() — a recording that never renders produces nothing."
            )
            return

        # Every file is finalized before any failure is raised, so one environment failing to
        # encode leaves neither the others open nor the camera unable to record. The first
        # failure is the one raised: a later one is as likely to be its consequence.
        error = None
        for encoder in recording.encoders:
            try:
                encoder.close()
            except gs.GenesisException as e:
                error = error or e
        if error is not None:
            raise error

    def destroy(self) -> None:
        """Finalize a recording still running, as part of tearing the camera down.

        This is what makes stopping a recording explicitly optional: a camera going away
        leaves a playable video behind rather than a file the encoder never flushed.
        """
        if self._recording is None:
            return
        try:
            self.stop_recording()
        except gs.GenesisException as e:
            # Teardown carries on regardless, so a video failing to encode takes neither
            # the other cameras nor the scene down with it. The error was already reported
            # by the encoder when it happened.
            en.logger.warning(f"{e}")

    @property
    def _paints(self) -> bool:
        """Whether :meth:`render_rgb` paints overlays: the camera has some, and they are built (the env is)."""
        return self._overlays_built and bool(self._overlays)

    def _finish_frame(self, rgb, *, may_alias: bool = False, segmentation=None):
        """Paint the overlays onto a frame :meth:`render_rgb` produced, record it, and return it.

        ``may_alias`` is :meth:`_record_frame`'s; it holds for the painted frame too, since an overlay that paints
        nothing returns its input. ``segmentation`` is the sim pass's, rendered with ``rgb`` when an overlay reads it.
        """
        if self._paints:
            rgb = self._paint_overlays(rgb, segmentation)
        self._record_frame(rgb, may_alias=may_alias)
        return rgb

    def _paint_overlays(self, rgb, segmentation=None):
        """``rgb`` with every overlay painted on, in order, in its own array type, device, dtype and shape.

        The overlays work on a batch of images on the env's device, so a single image is painted as a batch of one;
        see :meth:`_frame_envs_idx` for which env each image shows. Pixels an overlay replaced
        (:meth:`~eden.entities.camera_overlays.CameraOverlay.replaced_pixels`, a background composite's) are left as
        they are by the overlays after it: they no longer show the sim surface those locate their marks on.

        Painting by the point cloud costs a second render of the view (its depth), once per frame however many
        overlays read it.
        """
        device = self._env.device
        batched = rgb.ndim == 4

        def as_batch(array, dtype=None):
            tensor = numpy_to_tensor(array, dtype=dtype, device=device)
            return tensor if batched else tensor[None]

        layout = (rgb.shape[0] if batched else 1, batched)
        if layout not in self._frame_envs_idx_cache:  # fixed once built, and a list of num_envs ints to upload
            self._frame_envs_idx_cache[layout] = torch.as_tensor(self._frame_envs_idx(*layout), device=device)
        envs_idx = self._frame_envs_idx_cache[layout]
        frames = as_batch(rgb)
        buffers = OverlayBuffers(segmentation=None if segmentation is None else as_batch(segmentation))
        if any(overlay._needs_pointcloud for overlay in self._overlays):
            pointcloud, valid = self._cam.render_pointcloud(world_frame=True)
            buffers = dataclasses.replace(
                buffers, pointcloud=as_batch(pointcloud, dtype=torch.float32), pointcloud_valid=as_batch(valid)
            )

        painted = frames[..., :3]
        kept = None  # what an earlier overlay replaced, which the later ones leave as it is
        for overlay in self._overlays:
            before, painted = painted, overlay.paint(painted, buffers, envs_idx)
            if (painted.shape, painted.dtype) != (before.shape, before.dtype):
                raise TypeError(
                    f"Camera {self.display_name}: {type(overlay).__name__}.paint() returned {tuple(painted.shape)} "
                    f"{painted.dtype} for a {tuple(before.shape)} {before.dtype} frame. paint() returns the frame in "
                    "its own shape and dtype: blend in float if it must, then round and cast back to rgb.dtype."
                )
            if kept is not None:
                painted = torch.where(kept[..., None], before, painted)
            replaced = overlay.replaced_pixels(buffers)
            if replaced is not None:
                kept = replaced if kept is None else kept | replaced

        if frames.shape[-1] == 3:
            out = painted
        else:  # RGBA: keep the renderer's alpha
            out = frames.clone()
            out[..., :3] = painted
        out = out if batched else out[0]
        if isinstance(rgb, torch.Tensor):
            return out.to(rgb.device)
        return out.cpu().numpy()

    def _frame_envs_idx(self, n_images: int, batched: bool) -> list[int]:
        """The env each of the ``n_images`` images of a rendered frame shows.

        A single image shows the env the rasterizer camera is bound to (Genesis binds it to
        ``rendered_envs_idx[0]``). A batched frame holds one image per env the backend renders: the rasterizer
        renders ``rendered_envs_idx``, while the batch renderer and the sensor backend render every env of the
        scene. Raises rather than guess when the count disagrees, since painting env k's state onto env j's image
        would be silently wrong.
        """
        # A rasterizer (or ray tracer) camera is bound to one env and renders rendered_envs_idx, which has no public
        # accessor in Genesis (it reads it the same way).
        bound = self.renderer_kind in ("rasterizer", "raytracer")
        if not batched:
            envs = [self._cam.env_idx or 0] if bound else [0]
            source = "the env the camera is bound to"
        elif bound:
            envs = [int(i) for i in self._env.scene.visualizer._context.rendered_envs_idx]
            source = "the rasterizer's rendered_envs_idx"
        else:
            envs = list(range(self._env.num_envs))
            source = "every env of the scene, which the batch renderer and the sensor backend render"
        if len(envs) != n_images:
            raise RuntimeError(
                f"Camera {self.display_name} rendered {n_images} image(s), but its overlays expected one per env of "
                f"{source} ({len(envs)}), so which env each image shows is unknown and none is painted. Remove the "
                "camera's overlays, or render every env (leave VisOptions.rendered_envs_idx at its default)."
            )
        return envs

    def _record_frame(self, rgb, *, may_alias: bool = False) -> None:
        """Append one rendered RGB frame to the active recording, if any.

        Called by every public entry point that produces a final RGB frame — exactly once
        per call, so a composited frame (background overlay, splat background) is recorded
        instead of the raw render it was built from.

        Parameters
        ----------
        rgb : array-like
            The frame, either a single image or one per rendered env.
        may_alias : bool, optional
            Whether ``rgb`` may share memory with a buffer its producer reuses. The encoders
            consume frames on their own thread, so such a frame is copied before being handed
            over; a frame the renderer freshly allocated is not, which keeps a batched
            recording from paying a memcpy per env per frame. Only the caller can know:
            the rasterizer and the batch renderer allocate per call, while a sensor read is
            "a pure view into the per-class return cache" (genesis ``Sensor.read``) that the
            next ``step()`` repopulates. Defaults to False.
        """
        recording = self._recording
        if recording is None:
            return

        frame = tensor_to_array(rgb)
        # A batched camera hands back one image per rendered env; an unbatched one a single
        # image. Both are recorded as a stack of one frame per video.
        frames = frame if frame.ndim == 4 else frame[None]

        if not recording.encoders:
            recording.encoders.extend(self._open_encoders(recording, len(frames)))
        elif len(frames) != len(recording.encoders):
            raise RuntimeError(
                f"Recording to '{recording.path}' started on {len(recording.encoders)} env(s) but this frame "
                f"holds {len(frames)}. The number of rendered envs cannot change mid-recording."
            )

        for env_frame, encoder in zip(frames, recording.encoders):
            encoder.write(env_frame.copy() if may_alias else env_frame)

    @staticmethod
    def _open_encoders(recording: _Recording, n_videos: int) -> list[VideoEncoder]:
        """Open one encoder per video of ``recording``, naming and announcing them.

        A batched camera writes one video per rendered env, suffixed with the env's position
        in the batch; an unbatched one writes the single file it was given.
        """
        paths = [recording.path]
        if n_videos > 1:
            stem, ext = os.path.splitext(recording.path)
            paths = [f"{stem}_{env_idx}{ext}" for env_idx in range(n_videos)]
            en.logger.info(f"Camera renders {n_videos} envs; recording one video each: '{paths[0]}' ... '{paths[-1]}'.")
        # Threaded, so a render call only pays for handing the frame over.
        return [VideoEncoder(path, recording.fps, is_threaded=True) for path in paths]

    def read(self, envs_idx=None):
        """Read sensor data. Only available with the sensor backend.

        Returns
        -------
        data : CameraData
            Named tuple with `rgb` field containing the rendered image.
        """
        if not self._uses_sensor_backend:
            raise RuntimeError(
                "read() is only available with the sensor backend. "
                "Use render_rgb()/render_depth()/etc. for legacy cameras."
            )
        return self._sensor.read(envs_idx=envs_idx)

    def pick_pixel(self, camera_index, x, y):
        """Cast a ray through pixel ``(x, y)`` and return what it hits.

        Sensor backend only (e.g. Nyx). Delegates to the underlying sensor's
        ``pick_pixel`` and returns its result — the hit entity / link / world
        position — or ``None`` for background (sky).
        """
        if not self._uses_sensor_backend:
            raise NotImplementedError("pick_pixel() is only available with the sensor backend (e.g. Nyx).")
        return self._sensor.pick_pixel(camera_index, x, y)

    def render(
        self,
        rgb: bool = True,
        depth: bool = False,
        segmentation: bool = False,
        colorize_seg: bool = False,
        normal: bool = False,
    ):
        rgb_out, depth_out, segmentation_out, normal_out = self._render(
            rgb=rgb,
            depth=depth,
            segmentation=segmentation,
            colorize_seg=colorize_seg,
            normal=normal,
        )
        if rgb:
            self._record_frame(rgb_out)
        return rgb_out, depth_out, segmentation_out, normal_out

    def _render(
        self,
        rgb: bool = True,
        depth: bool = False,
        segmentation: bool = False,
        colorize_seg: bool = False,
        normal: bool = False,
    ):
        """Render without recording — the raw pass every public renderer builds on.

        Compositing renderers (background overlay, splat background) go through this one
        and record the frame they composited, so a recording never holds the raw pass.
        """
        if self._uses_sensor_backend:
            raise NotImplementedError(
                "render() with multiple outputs is not supported with the sensor backend. "
                "Use render_rgb() or read() instead."
            )
        return self._cam.render(
            rgb=rgb,
            depth=depth,
            segmentation=segmentation,
            colorize_seg=colorize_seg,
            normal=normal,
        )

    def render_unshadowed_rgb(self) -> torch.Tensor | None:
        """This view rendered again with the scene's shadows off, ``(N, H, W, 3)`` on the env's device, which lets
        :func:`~eden.utils.background_shadows.background_shadow_factor` compare every pixel with its own unshadowed
        brightness; ``None`` when the renderer can't (the batch renderer bakes shadows in at build time, and the
        factor then estimates the unshadowed brightness instead).
        """
        if self.renderer_kind != "rasterizer":
            return None
        with _rasterizer_shadows_off(self._env.scene.visualizer):
            rgb = numpy_to_tensor(self._cam.render(rgb=True)[0], device=self._env.device)
        return (rgb if rgb.ndim == 4 else rgb[None])[..., :3]

    def render_rgb(self):
        """The camera's RGB frame, with its background composite and overlays painted on, recorded if recording."""
        if self._uses_sensor_backend:
            rgb_out = self._sensor.read().rgb
            # A sensor read is a view into a cache the next step() repopulates, so the
            # recording needs its own copy of it (see _record_frame). The view itself is
            # what is returned, as callers of a sensor read expect.
            return self._finish_frame(rgb_out, may_alias=True)

        # The segmentation comes from the same render as the RGB, so an overlay that reads it costs no extra pass.
        segmentation = self._paints and any(overlay._needs_segmentation for overlay in self._overlays)
        rgb_out, _, segm, _ = self._render(rgb=True, segmentation=segmentation)
        return self._finish_frame(rgb_out, segmentation=segm)

    def render_depth(self):
        if self._uses_sensor_backend:
            raise NotImplementedError(
                "Depth rendering is not supported with the sensor backend. "
                "Use the legacy camera backend (remove sensor_options from CameraOptions)."
            )
        _, depth_out, _, _ = self.render(rgb=False, depth=True, segmentation=False, colorize_seg=False, normal=False)
        return depth_out

    def render_pointcloud(self, world_frame=True):
        """Render a point cloud from the camera's current view.

        Parameters
        ----------
        world_frame : bool, optional
            True returns the points in the world frame, False in the camera frame. Defaults to True.

        Returns
        -------
        pc : np.ndarray
            The point of each pixel, shape (H, W, 3) or (N, H, W, 3).
        mask_arr : np.ndarray
            Boolean mask of the pixels with a valid depth, shape (H, W) or (N, H, W).
        """
        if self._uses_sensor_backend:
            raise NotImplementedError(
                "Pointcloud rendering is not supported with the sensor backend. "
                "Use the legacy camera backend (remove sensor_options from CameraOptions)."
            )
        pc, valid_mask = self._cam.render_pointcloud(world_frame=world_frame)
        return pc, valid_mask

    def render_segm(self):
        if self._uses_sensor_backend:
            raise NotImplementedError(
                "Segmentation rendering is not supported with the sensor backend. "
                "Use the legacy camera backend (remove sensor_options from CameraOptions)."
            )
        _, _, segmentation_out, _ = self.render(
            rgb=False, depth=False, segmentation=True, colorize_seg=False, normal=False
        )
        return segmentation_out

    def render_normal(self):
        if self._uses_sensor_backend:
            raise NotImplementedError(
                "Normal rendering is not supported with the sensor backend. "
                "Use the legacy camera backend (remove sensor_options from CameraOptions)."
            )
        _, _, _, normal_out = self.render(rgb=False, depth=False, segmentation=False, colorize_seg=False, normal=True)
        return normal_out

    def snapshot(self, image_path: str | None = None):
        image_path = image_path or f"{en.log_dir}/snapshot_{get_now()}.png"
        rgb = self.render_rgb()
        if rgb.ndim == 4:
            rgb = rgb[0]
        rgb = tensor_to_array(rgb)
        gs.tools.save_img_arr(rgb, image_path)

    @property
    def sensor(self):
        """Access the underlying Genesis sensor object. Only available with sensor backend."""
        if not self._uses_sensor_backend:
            raise RuntimeError("sensor property is only available with the sensor backend.")
        return self._sensor


class _SplatCameraBase(Camera):
    """Shared implementation of the two gsplat splat cameras: a :class:`Camera` that can carry a
    :class:`~eden.entities.camera_overlays.SplatComposite`. The two differ only in how many views of the splat they
    render (:attr:`splat_single_view`). Not instantiated directly; the env picks :class:`SplatCamera` or
    :class:`BatchSplatCamera` via :meth:`~eden.envs.splat.SplatEnvBase._splat_camera_class`.
    """

    def __init__(self, key: str, env, options: CameraOptions):
        super().__init__(env, options, key=key)


class SplatCamera(_SplatCameraBase):
    """Renders the splat **once** from an env-invariant pose and uses it for every env's image.

    The cheap path for a fixed camera over a shared (``per_env_gaussians=False``) background: the splat looks
    identical from every env, so rendering it N times is waste. Only the first env's pose is rendered. The env picks
    this class automatically for non-attached cameras — see :meth:`~eden.envs.splat.SplatEnvBase._splat_camera_class`.
    """

    @property
    def splat_single_view(self) -> bool:
        return True


class BatchSplatCamera(_SplatCameraBase):
    """Renders the splat **per env** — every environment's camera in one gsplat call.

    Needed when the splat differs across envs: an attached/following camera (per-env pose) or
    ``per_env_gaussians`` (per-env geometry). The env picks this class over the single-view
    :class:`SplatCamera` — see :meth:`~eden.envs.splat.SplatEnvBase._splat_camera_class`.
    """

    @property
    def splat_single_view(self) -> bool:
        return False
