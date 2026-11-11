"""Gaussian-splat scene options (gsplat and Nyx) and point-cloud options."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal

from pydantic import model_validator

from genesis.typing import UnitVec4FType, Vec3FType
from eden.options.options import ConfigurableOptions, DynamicConfigurableOptions
from eden.utils.assets import get_asset_path
from eden.utils.misc import resolve_local_dir


def _collect_pcds_into_extra(extra: dict | None) -> None:
    """Move all :class:`PCDOptions` entries in ``extra`` into ``extra["pcds"]`` in place.

    ``SplatOptions`` / ``NyxSplatOptions`` accept named PCD entries as
    extras (e.g. ``SplatOptions(bg=PCDOptions(...))``) but expose them as
    a single ``pcds`` dict on the instance.

    A saved config serializes those entries as the already-collected ``pcds`` dict, so on
    reconstruction they arrive under ``extra["pcds"]`` rather than as top-level extras. Seed
    from any existing ``pcds`` dict first so the round-trip preserves them instead of silently
    dropping every PCD on load.
    """
    if not extra:
        return
    existing = extra.get("pcds")
    pcds = dict(existing) if isinstance(existing, dict) else {}
    for key in list(extra):
        value = extra[key]
        if isinstance(value, PCDOptions):
            pcds[key] = value
            del extra[key]
    extra["pcds"] = pcds


class PCDOptions(ConfigurableOptions):
    """
    PCD options for Eden.

    Parameters
    ----------
    path: str
        Path to the ply file, fetched from ``registry``/``dataset`` when it is not a local file.
    local_dir: str | None
        Where the file is cached. Unset, it is ``assets_subdir`` under the asset root, resolved when the
        options are built (:meth:`resolve_local_dir`), so a saved config's ``local_dir`` stays unset.
    assets_subdir: str
        The subdirectory of the asset root that an unset ``local_dir`` means; empty is the root. A field, and so
        saved with the config, unlike ``EntityOptions``' class constant: PCDs have no per-family subclasses to
        carry one, and saving it is harmless here. It is a relative name, never a machine's path, and it is read
        only while the options are built, since ``path`` is resolved to the local file then and saved resolved.
        What the class-constant rule elsewhere keeps out of saved configs is an absolute default path.
    scale: float
        Scale factor.
    pos: array-like[float, float, float]
        Translation of the pcd.
    quat: array-like[float, float, float, float]
        Rotation of the pcd (w, x, y, z).
    multiplier: float
        Per-splat radiance scale. Maps to ``LightFieldAsset.multiplier`` on
        the Nyx backend; ignored by the gsplat backend (``SplatOptions``).
        ``1.0`` leaves the splat unchanged, ``>1`` brightens, ``<1`` darkens.
    tint: array-like[float, float, float]
        Per-channel multiplicative tint applied on top of ``multiplier``.
        Maps to ``LightFieldAsset.tint`` on the Nyx backend; ignored by the
        gsplat backend.
    attach_entity: str | None
        Name of the entity whose pose drives this splat. Requires
        ``SplatOptions(per_env_gaussians=True)``: when set, the splat is treated
        as a *dynamic* rigid body that follows this entity per environment.
        ``None`` marks a static background splat. Ignored by the Nyx backend
        (which renders a static background).
    attach_link: str | None
        Optional link name on ``attach_entity`` to bind to (defaults to the
        entity's base link).
    """

    path: str
    registry: str = "Kashu7100"
    dataset: str = "eden_assets"
    local_dir: str | None = None
    assets_subdir: str = ""
    scale: float = 1.0
    pos: Vec3FType = (0.0, 0.0, 0.0)
    quat: UnitVec4FType = (1.0, 0.0, 0.0, 0.0)
    multiplier: float = 1.0
    tint: Vec3FType = (1.0, 1.0, 1.0)
    attach_entity: str | None = None
    attach_link: str | None = None

    def resolve_local_dir(self) -> str:
        """Where the file is cached: ``local_dir``, else ``assets_subdir`` under the asset root."""
        return resolve_local_dir(self.local_dir, self.assets_subdir)

    def model_post_init(self, context):
        super().model_post_init(context)
        self.path = get_asset_path(
            self.path,
            registry=self.registry,
            dataset=self.dataset,
            local_dir=self.resolve_local_dir(),
        )


# How a splat camera renders the background: infer per camera, or force one class.
SplatCameraMode = Literal["auto", "single", "batched"]


class SplatOptions(DynamicConfigurableOptions):
    """
    Gaussian-splat options for Eden, consumed by :class:`~eden.envs.splat.SplatEnvBase` (non-RL) and
    :class:`~eden.envs.splat.SplatRLEnvBase` (RL).

    The one gsplat backend (:class:`~eden.utils.batch_splat_renderer.BatchSplatRenderer`),
    composited over the sim pass by segmentation mask. There is no ``sh_degree`` field (SH is
    read from each PLY) and no ``bg_color`` field (the background is the composite, not a fill).
    Shadows that sim objects cast on the background entities are carried onto the splat per camera,
    with the ``shadow_strength`` of the camera's ``SplatComposite`` overlay.
    Nyx is the separate path-traced backend (:class:`NyxSplatOptions`).

    Which camera renders each view is selected by ``camera_mode`` — :class:`~eden.entities.camera.SplatCamera`
    renders one env-invariant pose and broadcasts it (cheap; a fixed camera over a shared
    background), :class:`~eden.entities.camera.BatchSplatCamera` renders every env's camera
    (needed for an attached/following camera or ``per_env_gaussians``).

    Parameters
    ----------
    per_env_gaussians: bool
        The shared / per-env Gaussian switch. ``False`` (shared) stores one Gaussian set and
        renders every env's camera from it — memory-light, scales to many envs, but the splat
        scene is identical across envs. ``True`` (per-env) gives each env its own Gaussians so
        splats bound to entities (``PCDOptions.attach_entity``) follow that env's sim poses;
        memory is ``O(num_envs * G)``, fitting tens–hundreds of envs. Forces the batched camera.
    minibatch: int | None
        Chunk size over the env/view axis, bounding peak VRAM (gsplat evaluates view-dependent SH
        per rendered view). ``None`` renders all envs at once.
    camera_mode: "auto" | "single" | "batched"
        How each splat camera renders. ``"auto"`` (default) picks per camera: the batched camera
        when ``per_env_gaussians`` or the camera is attached/following (per-env poses), else the
        single-view camera. ``"single"`` / ``"batched"`` force one class for every camera
        (``"single"`` is rejected with ``per_env_gaussians=True``). This is env-wide: with two or
        more splat cameras the forced modes apply to all of them; per-camera control would need the
        knob on ``CameraOptions``, so rely on ``"auto"`` when different cameras need different modes.
    *
    <pcd_name>: PCDOptions
        Splat PLYs. A PCD with ``attach_entity`` set is a dynamic body (per-env mode); the rest
        form the static background.
    """

    per_env_gaussians: bool = False
    minibatch: int | None = None
    camera_mode: SplatCameraMode = "auto"

    if TYPE_CHECKING:
        pcds: dict[str, PCDOptions]

    @model_validator(mode="after")
    def _validate_camera_mode(self):
        """Fail fast: a single broadcast view cannot represent per-env Gaussians.

        This is a config-level contradiction (both fields live here), so it is rejected at
        construction rather than deep in ``_setup_cameras`` — which keeps its own check only as a
        backstop for a post-construction mutation (assignment does not re-run this validator).
        """
        if self.camera_mode == "single" and self.per_env_gaussians:
            raise ValueError(
                "SplatOptions(camera_mode='single', per_env_gaussians=True) is contradictory: each "
                "env's Gaussians differ, so one broadcast view is wrong. Use camera_mode='auto' or 'batched'."
            )
        return self

    def model_post_init(self, context):
        super().model_post_init(context)
        _collect_pcds_into_extra(self.__pydantic_extra__)


# Names of nps.EToneMapper enum members; the heavy gs_nyx native module is
# only imported when a Nyx camera is built, so we mirror the enum at the type
# level instead of importing it here.
ToneMapper = Literal["ACES_1_3", "Neutral", "Off", "Reinhard"]


class NyxSplatOptions(DynamicConfigurableOptions):
    """
    Nyx-based splat rendering options.

    Uses Nyx's native light field support to render gaussian-splat backgrounds inside the path
    tracer, instead of the gsplat rasterize-and-segmentation-composite approach of
    :class:`SplatOptions`.

    Parameters
    ----------
    spp: int
        Samples per pixel for path tracing.
    denoise: bool
        Whether to apply denoising.
    near: float
        Near clipping plane distance.
    far: float
        Far clipping plane distance.
    max_pt_depth: None
        Must stay ``None``. gs-nyx-plugin exposes no path-depth control, so a depth set here could never take
        effect; it is rejected rather than ignored.
    lights: list[dict[str, Any]]
        Light definitions in Z-up coordinates. Each dict should have a "type" key
        ("point", "directional", or "spot") and type-specific parameters.
    env_map_paths: list[str]
        Paths to HDR environment map textures.
    tone_mapper: ToneMapper
        Name of the ``gs_nyx.nyx_py_sdk.EToneMapper`` member to use. Use
        ``"Off"`` or ``"Reinhard"`` while tuning lighting to see closer-to-
        linear radiance feedback. Written into each camera's profile (see
        :func:`eden.envs.nyx_splat.write_nyx_camera_profile`), including a camera's own
        ``NyxCameraOptions`` that sets no ``camera_profile``; one that does keeps its own.
    exposure_compensation: float
        Exposure, in EV, relative to unit exposure (a pixel shows the radiance it receives; ``+1``
        doubles the image). ``0`` keeps the brightness that light intensities and env-map
        brightnesses were tuned against; Nyx's own default camera renders 0.53 EV darker.
    *
    <pcd_name>: PCDOptions
        PCD options for each gaussian splat PLY file. These will be converted
        to Nyx LightFieldAsset objects.
    """

    spp: int = 32
    denoise: bool = True
    near: float = 0.1
    far: float = 100.0
    max_pt_depth: int | None = None
    lights: list[dict[str, Any]] = []
    env_map_paths: list[str] = []
    tone_mapper: ToneMapper = "ACES_1_3"
    exposure_compensation: float = 0.0

    if TYPE_CHECKING:
        pcds: dict[str, PCDOptions]

    @model_validator(mode="after")
    def _reject_path_depth(self):
        """Fail fast: no gs-nyx-plugin release has a path-depth setting, so a depth here would be silently dropped."""
        if self.max_pt_depth is not None:
            raise ValueError(
                f"NyxSplatOptions(max_pt_depth={self.max_pt_depth}) cannot take effect: gs-nyx-plugin exposes no "
                "path-depth control (NyxCameraOptions has no such field), so the Nyx renderer uses its own. Remove "
                "max_pt_depth from the options. A config.json saved before gs-nyx-plugin 0.1.7 support carries the "
                'old default `"max_pt_depth": 2` under splat_options: delete that key from the file '
                "(the renderer then uses its own depth, as it does for every new config)."
            )
        return self

    def model_post_init(self, context):
        super().model_post_init(context)
        _collect_pcds_into_extra(self.__pydantic_extra__)
