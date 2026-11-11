"""Environment that renders gaussian-splat backgrounds via the Nyx renderer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import genesis as gs
from genesis.options.renderers import RendererOptions

import eden as en
from eden.envs.base import EnvBase
from eden.utils.configs import EdenNyxSplatConfig
from eden.options import (
    ActionManagerOptions,
    CameraOptions,
    CamerasOptions,
    EnvOptions,
    EventManagerOptions,
    MetricManagerOptions,
    ObservationManagerOptions,
    RecorderManagerOptions,
    SceneOptions,
    SensorsOptions,
)
from eden.options.splat import NyxSplatOptions

if TYPE_CHECKING:
    from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions

    from eden.options.splat import PCDOptions, ToneMapper


def _patch_nyx_camera_options_deepcopy() -> None:
    """Make :class:`gs_nyx_plugin.NyxCameraOptions` deepcopy-safe.

    Eden calls ``self._sensor_options.model_copy(deep=True)`` in
    :meth:`eden.entities.camera.Camera._pre_build_sensor`, which deep-copies
    every field. The native ``LightFieldAsset`` / ``EnvironmentMapAsset``
    handles held in ``light_fields`` / ``env_maps`` are nanobind objects that
    don't implement ``__reduce__`` and raise ``TypeError: cannot pickle ...``.
    Treat those tuples as shared references and deep-copy the rest.
    """
    try:
        from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions
    except ImportError:
        return

    if getattr(NyxCameraOptions, "_eden_deepcopy_patched", False):
        return

    import copy as _copy

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}
        cls = type(self)
        new_obj = cls.__new__(cls)
        memo[id(self)] = new_obj
        shared = {"env_maps", "light_fields"}
        new_dict = {
            key: value if key in shared else _copy.deepcopy(value, memo) for key, value in self.__dict__.items()
        }
        object.__setattr__(new_obj, "__dict__", new_dict)
        object.__setattr__(new_obj, "__pydantic_extra__", _copy.deepcopy(self.__pydantic_extra__, memo))
        object.__setattr__(new_obj, "__pydantic_fields_set__", _copy.copy(self.__pydantic_fields_set__))
        object.__setattr__(new_obj, "__pydantic_private__", _copy.deepcopy(self.__pydantic_private__, memo))
        return new_obj

    NyxCameraOptions.__deepcopy__ = __deepcopy__
    NyxCameraOptions._eden_deepcopy_patched = True


_patch_nyx_camera_options_deepcopy()


def write_nyx_camera_profile(fov_y_deg: float, tone_mapper: ToneMapper, exposure_compensation: float = 0.0) -> str:
    """Write a gs-nyx camera profile and return its path, for ``NyxCameraOptions(camera_profile=...)``.

    gs-nyx-plugin >= 0.1.7 reads the optics and the colour management of a camera from a profile file; its
    ``NyxCameraOptions`` has no ``tone_mapper`` any more. Left to itself, the plugin writes a profile with Reinhard
    tone mapping and a physical camera (ISO 32000, 1/24 s, f/4) whose exposure scales radiance by 1/1.44.

    This profile carries ``tone_mapper`` and the exposure compensation that brings that camera to unit exposure: a
    pixel shows the radiance it receives, as gs-nyx rendered before cameras had a sensor model, and as every light
    intensity and environment-map brightness in Eden's configs was tuned against. ``exposure_compensation`` is added
    on top, in EV (``+1`` doubles the image).

    Profiles are cached in a per-user folder of the temp directory, one file per (gs-nyx version, settings), so
    repeated calls reuse one file and an SDK upgrade never reads a profile an older SDK wrote. Needs an initialized
    Genesis: the Nyx SDK it calls is started by Genesis.
    """
    import hashlib
    import math
    import os
    import tempfile
    from importlib.metadata import version

    import gs_nyx_plugin.nyx_camera_options  # noqa: F401 -- registers the Nyx SDK with Genesis, which starts it
    from gs_nyx import nyx_py_sdk as nps

    if not gs._initialized:
        raise RuntimeError(
            "write_nyx_camera_profile needs an initialized Genesis: the Nyx SDK it calls is started by `gs.init` "
            "(or `eden.init`). Call one of those first."
        )

    profile = nps.CameraProfileAsset()
    profile.set_fov_y(math.radians(fov_y_deg))
    profile.colorManagement.toneMapper = getattr(nps.EToneMapper, tone_mapper)
    exposure = profile.sensor.exposure
    f_stops = profile.lens.aperture.apertureFStops
    # Saturation-based exposure: H = 1 / (1.2 * 2^EV100), EV100 = log2(N^2 / t * 100 / S). Compensating by
    # log2(1.2) + EV100 makes H = 1 for whatever ISO, shutter and aperture the plugin defaults to.
    ev100 = math.log2(f_stops * f_stops / exposure.shutterSpeed * 100.0 / exposure.iso)
    exposure.exposureCompensation = math.log2(1.2) + ev100 + exposure_compensation

    # The key names the inputs, not the file's bytes, so it carries the SDK version: every profile field this function
    # leaves at the SDK's default (and the file format itself) can change with it.
    key = f"{version('gs-nyx')}|{fov_y_deg!r}|{tone_mapper}|{exposure.exposureCompensation!r}".encode()
    # Per user: on a shared node the first user's folder is not writable by the next. (Windows' temp dir is already
    # per user and has no getuid.)
    owner = f"_{os.getuid()}" if hasattr(os, "getuid") else ""
    folder = os.path.join(tempfile.gettempdir(), f"eden_nyx_camera_profiles{owner}")
    path = os.path.join(folder, f"profile_{hashlib.sha1(key).hexdigest()[:16]}.json")
    if not os.path.exists(path):
        os.makedirs(folder, exist_ok=True)
        partial = f"{path}.{os.getpid()}.tmp"
        nps.export_camera_profile_file(partial, profile)
        # The SDK returns None and raises nothing when the write fails, so check for the file.
        if not os.path.isfile(partial):
            raise RuntimeError(
                f"gs-nyx wrote no camera profile to {partial!r} (export_camera_profile_file reports no error). Check "
                f"that {folder!r} is writable."
            )
        os.replace(partial, path)  # atomic: a concurrent reader never sees half a file
    return path


class NyxSplatEnvBase(EnvBase[EdenNyxSplatConfig]):
    """
    Env (no RL managers) with the Nyx renderer for photorealistic background rendering using gaussian splats.

    Drives the official ``gs_nyx_plugin`` (see ``gs_nyx_plugin.NyxCameraSensor``)
    so gaussian splats render as native ``LightFieldAsset`` instances inside the
    path-traced scene, with proper lighting interaction between splat
    backgrounds and Genesis entities.
    """

    def __init__(
        self,
        env_options: EnvOptions | None = None,
        scene_options: SceneOptions | None = None,
        splat_options: NyxSplatOptions | None = None,
        observation_options: ObservationManagerOptions | None = None,
        event_options: EventManagerOptions | None = None,
        action_options: ActionManagerOptions | None = None,
        metric_options: MetricManagerOptions | None = None,
        recorder_options: RecorderManagerOptions | None = None,
        cameras_options: CamerasOptions | None = None,
        sensors_options: SensorsOptions | None = None,
        renderer_options: RendererOptions | None = None,
        *,
        show_viewer: bool = False,
        eval_mode: bool = False,
        **kwargs,
    ):
        # Checked here rather than in `from_config` so direct construction is covered too, and
        # so `from_config` stays the one generic body every env inherits. `EdenNyxSplatConfig`
        # validates the *type* when the field is set at construction, but not on later
        # assignment (`cfg.splat_options = SplatOptions()` is accepted).
        if splat_options is None:
            raise ValueError(
                "`NyxSplatEnvBase` requires `splat_options` (it owns a Nyx splat renderer). Set "
                "`splat_options=NyxSplatOptions(...)` on the config, or build a task without "
                "splats through `env_class_for_config(cfg).from_config(cfg)`."
            )
        if not isinstance(splat_options, NyxSplatOptions):
            raise TypeError(
                f"`NyxSplatEnvBase` renders through the Nyx path tracer, so `splat_options` must be "
                f"a `NyxSplatOptions`, not a `{type(splat_options).__name__}`. For "
                f"`SplatOptions`, build the config through `SplatEnvBase`."
            )

        # Held publicly so `env.config` can report it live (a declared EdenNyxSplatConfig
        # field); the private alias stays for the render path and entities/base.py's probe.
        self.splat_options = splat_options
        self._nyx_splat_options = splat_options
        self._nyx_light_fields = self._build_light_fields(splat_options.pcds)
        self._nyx_env_maps = self._build_env_maps(splat_options.env_map_paths)

        super().__init__(
            env_options=env_options,
            scene_options=scene_options,
            observation_options=observation_options,
            event_options=event_options,
            action_options=action_options,
            metric_options=metric_options,
            recorder_options=recorder_options,
            cameras_options=cameras_options,
            sensors_options=sensors_options,
            renderer_options=renderer_options,
            show_viewer=show_viewer,
            eval_mode=eval_mode,
            **kwargs,
        )

    @gs.assert_unbuilt
    def _setup_cameras(self):
        from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions

        from eden.entities.camera import Camera, resolved_cam_fov

        self.cameras: dict[str, Camera] = {}
        for camera_name in self.cameras_options.keys():
            camera_options: CameraOptions = getattr(self.cameras_options, camera_name)

            if camera_options.sensor_options is not None:
                sensor_opts = camera_options.sensor_options
                if isinstance(sensor_opts, NyxCameraOptions):
                    merged_lf = list(getattr(sensor_opts, "light_fields", ())) + self._nyx_light_fields
                    merged_em = list(getattr(sensor_opts, "env_maps", ())) + self._nyx_env_maps
                    if merged_lf:
                        sensor_opts.light_fields = tuple(merged_lf)
                    if merged_em:
                        sensor_opts.env_maps = tuple(merged_em)
                    if sensor_opts.camera_profile is None:
                        camera_options = camera_options.model_copy(
                            update={"sensor_options": self._with_camera_profile(sensor_opts, camera_options)},
                        )
            else:
                opts = self._nyx_splat_options
                # The profile owns the optics, so it carries the field of view the camera resolves.
                profile = write_nyx_camera_profile(
                    resolved_cam_fov(camera_options), opts.tone_mapper, opts.exposure_compensation
                )
                sensor_opts = NyxCameraOptions(
                    spp=opts.spp,
                    denoise=opts.denoise,
                    near=opts.near,
                    far=opts.far,
                    lights=opts.lights,
                    camera_profile=profile,
                    light_fields=tuple(self._nyx_light_fields),
                    env_maps=tuple(self._nyx_env_maps),
                )
                camera_options = camera_options.model_copy(
                    update={"sensor_options": sensor_opts},
                )

            camera = Camera(env=self, options=camera_options, key=camera_name)
            self.cameras[camera_name] = camera

    def _with_camera_profile(self, sensor_opts: NyxCameraOptions, camera_options: CameraOptions) -> NyxCameraOptions:
        """A copy of a user-supplied, profile-less ``NyxCameraOptions`` carrying the profile Eden's own cameras get.

        Without one the plugin falls back to Reinhard at its physical exposure (0.53 EV darker), so the camera would
        ignore ``tone_mapper`` / ``exposure_compensation`` and render unlike the env's other cameras. A camera that
        brings its own ``camera_profile`` keeps it: like an explicit ``fov``, a per-camera setting wins over the
        scene's.
        """
        from eden.entities.camera import resolved_cam_fov

        # `Camera._pre_build_sensor` fills `fov` from the camera only when the sensor options leave it unset.
        fov = sensor_opts.fov if "fov" in sensor_opts.model_fields_set else resolved_cam_fov(camera_options)
        opts = self._nyx_splat_options
        profile = write_nyx_camera_profile(fov, opts.tone_mapper, opts.exposure_compensation)
        return sensor_opts.model_copy(update={"camera_profile": profile})

    @staticmethod
    def _default_constant_env_map_uri(brightness: float) -> str:
        """Path to a cached constant-color Radiance ``.hdr`` env map.

        Nyx draws the background through ``EnvironmentMapAsset`` (default
        layout ``LongLat``, which expects a 2:1 equirectangular panorama).
        This writes a 16x8 HDR whose every texel samples to ``(b, b, b)``
        where ``b`` is ``brightness`` clamped to ``[0, 1]``. Use ``0.0`` for
        a pure black background, ``~0.5`` for a neutral gray that doubles as
        soft IBL. The version tag in the filename invalidates older caches
        when the encoder format changes.
        """
        import math
        import os
        import tempfile

        b = max(0.0, min(1.0, float(brightness)))
        # v2 = 16x8 LongLat (2:1 aspect): a 1x1 map is geometrically invalid
        # for LongLat sampling, and Nyx silently renders a dark default.
        cache_path = os.path.join(tempfile.gettempdir(), f"nyx_constant_envmap_v2_16x8_{b:.4f}.hdr")
        if os.path.exists(cache_path):
            return cache_path

        if b <= 0.0:
            pixel = b"\x00\x00\x00\x00"
        else:
            mantissa, exponent = math.frexp(b)
            mantissa_byte = max(0, min(255, int(round(mantissa * 256))))
            exp_byte = exponent + 128
            pixel = bytes([mantissa_byte, mantissa_byte, mantissa_byte, exp_byte])

        width, height = 16, 8
        header = f"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {height} +X {width}\n".encode("ascii")
        # Width < 8 is uncompressed RGBE; width >= 8 may use RLE but plain
        # uncompressed scanlines are also accepted by spec-compliant readers.
        body = pixel * (width * height)
        with open(cache_path, "wb") as f:
            f.write(header + body)
        return cache_path

    @classmethod
    def _build_light_fields(cls, pcds: dict[str, "PCDOptions"]) -> list:
        """Convert :class:`PCDOptions` into Nyx ``LightFieldAsset`` objects.

        Handles coordinate conversion from Genesis Z-up to Nyx Y-up.
        """
        from gs_nyx import nyx_py_sdk as nps

        from eden.utils.splat_ply_converter import convert_to_sh3_ply

        light_fields = []
        for name, pcd in pcds.items():
            light_field = nps.LightFieldAsset()
            light_field.type = nps.ELightFieldType.GaussianField

            asset_path = pcd.path
            if asset_path.lower().endswith(".ply"):
                converted = convert_to_sh3_ply(asset_path)
                if converted != asset_path:
                    en.logger.info(
                        f"Converted PLY '{name}' to SH-degree-3 format (padded SH coefficients): {converted}"
                    )
                asset_path = converted
            light_field.uri = asset_path

            light_field.multiplier = float(pcd.multiplier)
            light_field.tint = nps.float3(pcd.tint[0], pcd.tint[1], pcd.tint[2])

            if pcd.scale != 1.0:
                en.logger.warning(
                    f"Nyx LightFieldAsset does not support per-asset scaling. "
                    f"PLY '{name}' has scale={pcd.scale} which will be ignored. "
                    f"Pre-process the PLY file to the desired scale."
                )

            pos_zup = nps.float3(pcd.pos[0], pcd.pos[1], pcd.pos[2])
            light_field.position = nps.float3_z_up_to_y_up_a(pos_zup)

            quat_wxyz = nps.quaternion(pcd.quat[0], pcd.quat[1], pcd.quat[2], pcd.quat[3])
            quat_xyzw = nps.quaternion_wxyz_to_xyzw(quat_wxyz)
            light_field.rotation = nps.quaternion_z_up_to_y_up_a(quat_xyzw)

            light_fields.append(light_field)

        return light_fields

    @staticmethod
    def _build_env_maps(env_map_paths: list[str]) -> list:
        """Create Nyx ``EnvironmentMapAsset`` objects from HDR texture paths."""
        import os

        from gs_nyx import nyx_py_sdk as nps

        env_maps = []
        for path in env_map_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Nyx env map not found: {path}")
            env_map = nps.EnvironmentMapAsset()
            env_map.texture = path
            env_maps.append(env_map)
            en.logger.debug(f"Nyx env map bound: {path} ({os.path.getsize(path)} bytes)")
        return env_maps
