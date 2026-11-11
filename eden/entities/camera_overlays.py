"""Camera overlays: paint what the simulation tracks but does not render onto every RGB frame a camera produces.

Some state a task keeps is visible in the real world but has no geometry in the scene: the ink a marker lays on a
whiteboard, the strokes a brush leaves on paper. The task tracks it as a tensor, so a plain camera shows a clean
surface. An overlay draws that tensor onto the camera's frames after the renderer has made them.

A camera declares its overlays in :attr:`CameraOptions.overlays <eden.options.camera.CameraOptions>`, by registry key,
so they survive a saved ``config.json``. They are applied in :meth:`Camera.render_rgb
<eden.entities.camera.Camera.render_rgb>` and so reach everything built on it: the returned frame, the camera's
recording, :meth:`~eden.entities.camera.Camera.snapshot`, and an observation that reads the camera.
:meth:`~eden.entities.camera.Camera.render` stays the raw multi-output pass.

A background composite is an overlay too: :class:`BackgroundImageComposite` replaces a camera's background with an
image, :class:`SplatComposite` with a Gaussian splat (a splat env's cameras). Both replace the pixels where the sim
shows one of their ``entities`` or nothing at all, and report them through :meth:`CameraOverlay.replaced_pixels`; the
overlays after them leave those pixels as composited, since they no longer show the sim surface an overlay locates its
marks on. A camera takes at most one; declare it first, so the overlays after it paint over the composited frame. A
splat env's camera shows its splat only when it declares a ``SplatComposite``, as a splat config's
:meth:`~eden.utils.configs.EdenSplatConfig.recording_camera` does.

Writing one: subclass :class:`CameraOverlay`, declare its parameters as class attributes, register it with
``@CAMERA_OVERLAY_REGISTRY.register()``, and implement :meth:`CameraOverlay.paint`. Set ``_needs_segmentation = True``
to receive the frame's segmentation, which the sim pass renders with the RGB at no extra render. Set
``_needs_pointcloud = True`` to receive the frame's world-frame point cloud; the camera renders it once per frame for
all its overlays, which costs a second render of the view (its depth). A camera that can render neither (the sensor
backend) refuses such an overlay at env construction.
Resolve whatever the overlay reads (a metric term, an entity) in :meth:`CameraOverlay.build`, which runs after the
managers are built. Frames rendered before then (the observation manager renders once while it loads, to size its
terms) are not painted.

:class:`PlanePaintOverlay` is the general one: it paints the :class:`PaintedPlane` a metric term returns (any
:class:`PaintedPlaneSource`), a weight grid on a plane, where the camera sees that plane. It shades the marks as the
surface they lie on is shaded, so they keep the shadows and the light the renderer gave that surface.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

import genesis as gs
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from eden.options.camera import CameraOverlayOptions
from eden.utils.background_shadows import background_shadow_factor
from eden.utils.common import ConfigurableMixin
from eden.utils.params import validated_parameter_values
from eden.utils.plane_grid import PlaneGrid
from eden.utils.registry import Registry

if TYPE_CHECKING:
    from eden.entities.camera import Camera
    from eden.envs.base import EnvBase

#: Overlays by registry key, ``to_snake_case`` of the class name: what ``CameraOverlayOptions.name`` names.
CAMERA_OVERLAY_REGISTRY = Registry("CAMERA_OVERLAY")


@dataclass(frozen=True)
class OverlayBuffers:
    """What the camera rendered for its overlays alongside one RGB frame, one entry per image ``(N, ...)``.

    ``segmentation`` ``(N, H, W)`` is each pixel's segmentation id, ``None`` unless an overlay of the camera sets
    ``_needs_segmentation``. ``pointcloud`` is each pixel's world position ``(N, H, W, 3)`` and ``pointcloud_valid``
    ``(N, H, W)`` says which pixels hit something; both are ``None`` unless an overlay sets ``_needs_pointcloud``.
    """

    segmentation: torch.Tensor | None = None
    pointcloud: torch.Tensor | None = None
    pointcloud_valid: torch.Tensor | None = None


class CameraOverlay(ConfigurableMixin[CameraOverlayOptions]):
    """Base of the camera overlays: paints onto the RGB frames of the camera it is declared on.

    Constructed with the camera, at env construction; :meth:`build` runs once the env's managers are built.
    """

    #: Whether :meth:`paint` reads :attr:`OverlayBuffers.segmentation`.
    _needs_segmentation: ClassVar[bool] = False
    #: Whether :meth:`paint` reads :attr:`OverlayBuffers.pointcloud`.
    _needs_pointcloud: ClassVar[bool] = False

    def __init__(self, camera: Camera, options: CameraOverlayOptions) -> None:
        super().__init__(options)
        # CameraOverlayOptions converts its extras already, unless they were read before this module was imported.
        params = {name: getattr(self, name) for name in self.get_parameter_names()}
        params = validated_parameter_values(type(self), params, subject=type(self).__name__)
        for name, value in params.items():
            setattr(self, name, value)
        type(self).check_parameters(params)
        self._camera = camera
        self._env = camera._env

    @classmethod
    def check_parameters(cls, params: Mapping[str, Any]) -> None:
        """Refuse parameter values the overlay cannot work with: ranges and rules across parameters. ``params`` holds
        every parameter, defaults filled in and already of its declared type.

        Runs when its :class:`~eden.options.camera.CameraOverlayOptions` are constructed, so a bad value fails in the
        config that wrote it, and again when the overlay is constructed. Checks that need the scene belong in
        :meth:`build`.
        """

    def build(self) -> None:
        """Resolve and check what the overlay reads; runs after the env's managers are built."""

    def paint(self, rgb: torch.Tensor, buffers: OverlayBuffers, envs_idx: torch.Tensor) -> torch.Tensor:
        """Return ``rgb`` with the overlay painted on.

        Parameters
        ----------
        rgb : torch.Tensor
            ``(N, H, W, 3)`` on the env's device, one image per rendered env, in the renderer's dtype (``uint8`` for
            the rasterizer and the batch renderer). Return the same shape and dtype; never write into ``rgb``, which may
            be the renderer's own buffer (returning it as is, when nothing is painted, is fine).
        buffers : OverlayBuffers
            The extra renders of the same frame the overlay asked for.
        envs_idx : torch.Tensor
            ``(N,)`` the env each image shows, on the env's device.
        """
        raise NotImplementedError

    def replaced_pixels(self, buffers: OverlayBuffers) -> torch.Tensor | None:
        """``(N, H, W)`` the pixels :meth:`paint` replaced with something other than the simulated scene (a
        background image), which the overlays after this one leave as they are; ``None`` (the default) when it only
        paints over the scene.
        """
        return None


def build_camera_overlay(camera: Camera, options: CameraOverlayOptions, index: int) -> CameraOverlay:
    """The overlay ``options`` names, constructed for ``camera``: entry ``index`` of its ``CameraOptions.overlays``."""
    where = f"Camera {camera.display_name}: CameraOptions.overlays[{index}]"
    if not options.name:
        raise ValueError(
            f"{where} names no overlay (name={options.name!r}). Build the entry with the overlay class's configure(), "
            "e.g. PlanePaintOverlay.configure(metric_name='drawing', color=(0, 0, 0)), rather than "
            "CameraOverlayOptions(...)."
        )
    try:
        overlay_class = CAMERA_OVERLAY_REGISTRY.get(options.name)
    except KeyError as e:
        raise ValueError(f"{where}: {e.args[0]}") from None
    if (overlay_class._needs_pointcloud or overlay_class._needs_segmentation) and not camera.supports_pointcloud:
        needs = "point cloud" if overlay_class._needs_pointcloud else "segmentation"
        raise ValueError(
            f"{where}: {overlay_class.__name__} paints by the frame's {needs}, which a sensor-backend camera "
            "cannot render. Remove sensor_options from this CameraOptions to use the legacy camera backend."
        )
    return overlay_class(camera, options)


def _unit_rgb(color) -> tuple[float, float, float] | None:
    """``color`` as three floats in ``[0, 1]``, or ``None`` when it is not one (missing, a name, the wrong length)."""
    try:
        rgb = tuple(float(channel) for channel in color)
    except (TypeError, ValueError):
        return None
    if len(rgb) != 3 or not all(0.0 <= channel <= 1.0 for channel in rgb):
        return None
    return rgb


#: The darkest a channel of the surface :class:`PlanePaintOverlay` shades marks against may be: the marks are its
#: rendered pixels divided by its colour, so a channel of 0.05 already multiplies the frame's 8-bit steps by up to 20.
MIN_SURFACE_CHANNEL = 0.05


def _plain_surface_rgb(env: EnvBase, entity_name: str, *, subject: str) -> tuple[float, float, float]:
    """The one plain RGB colour, in ``[0, 1]``, every visual geom of entity ``entity_name`` is rendered in.

    Read off the built Genesis entity's surfaces, so it is the colour the renderer uses however the entity was given it
    (``color=`` or ``surface=``). Refuses an entity it cannot read one colour off (an image texture, a colour that
    varies per env, geoms in different colours, no visual geom), and one whose pixels are not its colour times the
    light: an emissive one (light it gives off) or a translucent one (what is behind it blended in).
    """
    if entity_name not in env.entities:
        raise ValueError(
            f"{subject} '{entity_name}', which is not an entity of the scene (entities: {sorted(env.entities)}). "
            "Name the entity the marks lie on in the PaintedPlane the metric returns (its entity_name)."
        )
    plain_fix = (
        "The marks are shaded as the surface is, by its rendered pixels over its colour, so the surface has to be one "
        "plain colour: give the entity one, e.g. BoxOptions(color=(0.96, 0.94, 0.88, 1.0)) or "
        "PlaneOptions(color=(0.8, 0.8, 0.8)) (Genesis' default plane is a checkerboard texture)."
    )
    colors: list[np.ndarray] = []
    for vgeom in getattr(env.entities[entity_name].entity, "vgeoms", ()):
        emission = vgeom.surface.emission
        if emission is not None and not getattr(emission, "is_black", False):
            raise ValueError(
                f"{subject} '{entity_name}', whose surface emits light ({type(vgeom.surface).__name__}). Its pixels "
                "are then light it gives off, not light it reflects, so the marks cannot be shaded by dividing by its "
                "colour. Give the entity a non-emissive surface (its color= alone makes a plastic one)."
            )
        # Batched: one texture per env (a plain surface gives one), where the unbatched read would keep only the first.
        textures = vgeom.surface.get_rgba(batch=True).textures
        if not all(isinstance(t, gs.textures.ColorTexture) for t in textures):
            kinds = sorted({type(t).__name__ for t in textures})
            raise ValueError(f"{subject} '{entity_name}', which is textured ({', '.join(kinds)}). {plain_fix}")
        opacity = min(float(t.color[3]) if len(t.color) > 3 else 1.0 for t in textures)
        if opacity < 1.0 - 1e-4:
            raise ValueError(
                f"{subject} '{entity_name}', which is translucent (opacity {opacity:g}). Its pixels then blend in "
                "whatever is behind it, so they are not its colour times the light and the marks cannot be shaded by "
                "dividing by its colour. Make the entity opaque (alpha 1 in its color=)."
            )
        per_env = [np.resize(np.asarray(t.color, dtype=float)[:3], 3) for t in textures]
        if any(not np.allclose(c, per_env[0], atol=1e-4) for c in per_env):
            raise ValueError(
                f"{subject} '{entity_name}', whose colour varies per env ({len(per_env)} colours in a batched "
                "texture). The overlay shades the marks against one colour read at build; give the entity one colour "
                "for every env."
            )
        colors.append(per_env[0])
    if not colors:
        raise ValueError(
            f"{subject} '{entity_name}', which has no visual geom to read a colour off. Have the metric name the "
            "rigid entity whose surface the marks lie on."
        )
    if any(not np.allclose(c, colors[0], atol=1e-4) for c in colors):
        distinct = sorted({tuple(round(float(x), 3) for x in c) for c in colors})
        raise ValueError(
            f"{subject} '{entity_name}', whose geoms are in {len(distinct)} colours {distinct}. The marks are shaded "
            "against one plain colour; give the entity a single colour (its color= or surface=), or model the marked "
            "face as an entity of its own (a board's face apart from its frame) and name that one."
        )
    return tuple(float(c) for c in colors[0])


def _srgb_decode(encoded: torch.Tensor) -> torch.Tensor:
    """sRGB-encoded values in ``[0, 1]`` to linear light."""
    return torch.where(encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4)


def _srgb_encode(linear: torch.Tensor) -> torch.Tensor:
    """Linear light in ``[0, 1]`` to sRGB-encoded values."""
    return torch.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear.clamp(min=0.0031308) ** (1 / 2.4) - 0.055)


#: How each renderer :class:`PlanePaintOverlay` runs on turns a surface's colour into its pixels, as measured: the
#: rasterizer's pixel is the colour times the light, as given; the batch renderer (Madrona) multiplies in linear light
#: and encodes its pixels as sRGB, so the mark's tint applies to the decoded pixel (a floor coloured in the ink matches
#: the tinted grey floor to a third of a level, lit or shadowed, in both its modes). A renderer not listed is refused.
_PIXEL_ENCODING = {"rasterizer": "linear", "batch_renderer": "srgb"}


@dataclass(frozen=True)
class PaintedPlane:
    """Marks on a plane, on a grid of square cells: what :class:`PlanePaintOverlay` paints.

    The plane runs through ``origin`` along the orthonormal axes ``u_axis`` and ``v_axis`` (world frame). Cell
    ``(r, c)`` of ``weight`` is cell ``(r, c)`` of ``grid`` in the plane's ``(u, v)``
    (:class:`~eden.utils.plane_grid.PlaneGrid`: row 0 at the top). A curved surface may be described by the plane
    touching it, within :class:`PlanePaintOverlay`'s rule.

    Parameters
    ----------
    origin, u_axis, v_axis : torch.Tensor
        ``(3,)`` each, on the env's device.
    grid : PlaneGrid
        Where the cells lie on the plane: the grid the metric keeps its marks on.
    weight : torch.Tensor
        ``(len(envs_idx), *grid.shape)``, floating in ``[0, 1]`` or bool: how much each cell is marked, for the envs
        ``painted_plane(envs_idx)`` was asked for, in that order.
    entity_name : str
        The entity whose surface the marks lie on (the board, the sheet). Its colour is what the overlay shades the
        marks against, so it has to be one plain, opaque colour: no image texture, no emission, no transparency, the
        same in every env, and the same on every one of its geoms (model a board's marked face apart from its
        frame).
    rgb : torch.Tensor | None
        ``(len(envs_idx), *grid.shape, 3)``, floating in ``[0, 1]``: each cell's colour, for marks of many colours
        (a painting's). ``None`` (marks of one colour, an ink's): the overlay's ``color`` paints them.
    """

    origin: torch.Tensor
    u_axis: torch.Tensor
    v_axis: torch.Tensor
    grid: PlaneGrid
    weight: torch.Tensor
    entity_name: str
    rgb: torch.Tensor | None = None


@runtime_checkable
class PaintedPlaneSource(Protocol):
    """A metric term that keeps marks on a plane, as :class:`PlanePaintOverlay` reads them. Structural: a term qualifies
    by defining the method, without subclassing this.
    """

    def painted_plane(self, envs_idx: torch.Tensor) -> PaintedPlane:
        """The marks of the envs ``envs_idx`` (a ``(K,)`` long tensor), read at every frame for the envs it shows."""


@CAMERA_OVERLAY_REGISTRY.register()
class PlanePaintOverlay(CameraOverlay):
    """Paints a metric's marks, in ``color`` or in their own colours, where the camera sees the plane they lie on.

    The metric term ``metric_name`` is a :class:`PaintedPlaneSource`: its ``painted_plane(envs_idx)`` is read at every
    frame for the envs the frame shows only, so a camera over one env never computes the marks of the other thousand.
    A pixel is on the plane when its world point (from the frame's point cloud, so whatever stands in front hides the
    marks) lies within ``max_gap`` of it. Each pixel is looked up on a ``supersample x supersample`` grid of sub-pixels
    and averaged, since a mark a few millimetres wide can be thinner than a pixel; only the pixels round the grid's
    image are, so the sub-pixel buffers scale with the marks' footprint in the frame, not with the frame.

    **A curved surface qualifies** when, over the grid, it stays within ``max_gap`` of the plane touching it, and its
    distances along the surface differ from the plane's by well under a cell: the marks are then painted where they
    lie. A narrow stripe round a tube's crown does (``mochi_wipe_the_bar``); a mark wrapping round it would not.

    **The marks are shaded as their surface is.** A mark is a change of colour on the surface the metric names
    (:attr:`PaintedPlane.entity_name`), so it takes the light that surface's pixel took: the shadow the brush casts
    over a stroke, the falloff across the sheet, the light's tint. The renderer draws a matte surface as its colour
    times the light reaching it, so a marked pixel is the rendered pixel times ``color / surface colour``, blended by
    how much of the pixel is marked: ``rendered * lerp(1, color / surface colour, weight)``. The surface colour is read
    from the built entity, which therefore has to be one plain colour (an image texture is refused at build), and no
    channel of it may be under :data:`MIN_SURFACE_CHANNEL`, below which the division amplifies the frame's 8-bit steps
    into visible banding. A channel the scaling lifts past 255 is clipped.

    The multiplication is in each renderer's own colour space, as measured: on the rasterizer the pixel itself is the
    colour times the light, while the batch renderer multiplies in linear light and encodes its pixels as sRGB, so
    there the tint applies to the decoded pixel. Other renderers (the ray tracer, whose colour response has not been
    measured) are refused at build. A surface that cannot be shaded against is refused too, with no flat-colour
    fallback: an image-textured or multi-coloured surface (paint marks onto a plain-coloured entity laid over it).

    The one thing the ratio cannot carry is light a surface reflects on top of its colour: a highlight, or the white
    sheen the rasterizer's default plastic surface adds in direct light (about 20 of 255 under the default lights,
    whatever the colour). It is scaled with the mark's colour, so in direct light a dark mark renders without it, up to
    about that much darker in its dark channels than a surface of the mark's colour would. In shadow, where there is
    no sheen, the marks match such a surface to within a level or two.

    Marks of one colour (an ink's) are painted in ``color``, the camera's to choose, not the metric's: grading never
    reads it. A task's :meth:`~eden.utils.configs.EdenConfig.recording_camera` fills it from wherever the task keeps
    it (the whiteboard marker's ink). Marks of many colours (a painting's) come with their colours
    (:attr:`PaintedPlane.rgb`), each shaded as ``color`` is, and ``color`` is then left out.

    Parameters
    ----------
    metric_name : str
        The metric term whose ``painted_plane()`` is painted.
    color : tuple[float, float, float] | None
        RGB in ``[0, 1]`` of the marks, as the surface's colour is given: a mark at full weight renders as the surface
        would in this colour. Required for marks of one colour (no colour is right for every mark); left out for marks
        that come with their own (:attr:`PaintedPlane.rgb`).
    max_gap : float
        How far (m) a pixel's world point may lie off the plane and still be painted. Covers the depth buffer's error
        at the camera's distance, and a curved surface's sag from its plane.
    supersample : int
        Sub-pixels per side each pixel is looked up on.
    """

    _needs_pointcloud = True

    metric_name: str = ""
    color: tuple[float, float, float] | None = None
    max_gap: float = 0.002
    supersample: int = 4

    def __init__(self, camera: Camera, options: CameraOverlayOptions) -> None:
        super().__init__(camera, options)
        self._rgb = None if self.color is None else _unit_rgb(self.color)

    @classmethod
    def check_parameters(cls, params: Mapping[str, Any]) -> None:
        if params["max_gap"] <= 0.0 or params["supersample"] < 1:
            raise ValueError(
                f"PlanePaintOverlay needs max_gap > 0 and supersample >= 1; got max_gap={params['max_gap']}, "
                f"supersample={params['supersample']}."
            )
        # No colour is for a metric that gives its marks' own (checked at build, against the metric).
        if params["color"] is not None and _unit_rgb(params["color"]) is None:
            raise ValueError(
                f"PlanePaintOverlay(metric_name={params['metric_name']!r}) got color={params['color']!r}; it needs the "
                "marks' colour as (r, g, b) with each channel a number in [0, 1], e.g. color=(0.02, 0.03, 0.10), not "
                "in [0, 255]. A task's recording_camera() passes the colour it keeps (the whiteboard marker's ink)."
            )

    def build(self) -> None:
        # Deferred: the managers import the entities, so a module-scope import here would be a cycle.
        from eden.managers.terms.utils import resolve_metric_manager

        manager = resolve_metric_manager(self._env, term="PlanePaintOverlay", metric_name=self.metric_name)
        metric = manager.get_term(self.metric_name)
        if not isinstance(metric, PaintedPlaneSource):
            raise TypeError(
                f"PlanePaintOverlay paints the marks a metric term returns from painted_plane(envs_idx), but metric "
                f"'{self.metric_name}' ({type(metric).__name__}) has no painted_plane(). Set metric_name to a metric "
                f"that keeps marks on a plane (active metrics: {manager.active_terms}); a metric class term becomes "
                "one by defining painted_plane(envs_idx) returning eden.entities.camera_overlays.PaintedPlane."
            )
        self._metric = metric
        # One env's worth, to check the contract here rather than on the first frame.
        plane = metric.painted_plane(torch.zeros(1, dtype=torch.long, device=self._env.device))
        subject = f"PlanePaintOverlay: metric '{self.metric_name}' ({type(metric).__name__}) returned a PaintedPlane"
        if plane.rgb is None and self._rgb is None:
            raise ValueError(
                f"{subject} of marks of one colour (rgb=None), and the overlay has no color to paint them in. "
                "Configure color=(r, g, b) in [0, 1]: a task's recording_camera() passes the colour it keeps (the "
                "whiteboard marker's ink)."
            )
        if plane.rgb is not None and self._rgb is not None:
            raise ValueError(
                f"{subject} with the marks' own colours (rgb), so color={self.color!r} would never be painted. Leave "
                "color out for this metric."
            )
        if plane.rgb is not None and tuple(plane.rgb.shape) != (1, *plane.grid.shape, 3):
            raise ValueError(
                f"{subject} with rgb {tuple(plane.rgb.shape)} for one env; expected (1, *grid.shape, 3) = "
                f"{(1, *plane.grid.shape, 3)}, a colour per cell of the grid it returned."
            )
        if plane.rgb is not None and not plane.rgb.is_floating_point():
            raise ValueError(
                f"{subject} with rgb of dtype {plane.rgb.dtype}; the marks' colours must be floating, in [0, 1]. An "
                "8-bit image would tint by its 0-255 values: divide it by 255 first."
            )
        if tuple(plane.weight.shape) != (1, *plane.grid.shape):
            raise ValueError(
                f"{subject} with weight {tuple(plane.weight.shape)} for one env; expected (1, *grid.shape) = "
                f"{(1, *plane.grid.shape)}, one grid per env asked for, on the PlaneGrid it returned."
            )
        if not (plane.weight.dtype == torch.bool or plane.weight.is_floating_point()):
            raise ValueError(
                f"{subject} weight of dtype {plane.weight.dtype}; it must be floating, in [0, 1], or bool (a cell "
                "marked or not). An integer grid has no scale the overlay could read: divide it by its full value "
                "first."
            )
        axes = torch.stack([plane.u_axis, plane.v_axis])
        if not torch.allclose(axes @ axes.T, torch.eye(2, device=axes.device), atol=1e-5):
            raise ValueError(
                f"{subject} with axes u_axis={plane.u_axis.tolist()}, v_axis={plane.v_axis.tolist()}, which are not "
                "orthonormal; the plane test and the cell lookup measure metres along them. Return unit, "
                f"perpendicular axes from {type(metric).__name__}.painted_plane()."
            )
        renderer = self._camera.renderer_kind
        if renderer not in _PIXEL_ENCODING:
            raise ValueError(
                f"PlanePaintOverlay on camera {self._camera.display_name}, whose frames come from the {renderer}. The "
                "overlay shades marks by how the renderer turns a surface's colour into pixels, which is measured for "
                f"the {' and the '.join(k.replace('_', ' ') for k in _PIXEL_ENCODING)} only. Record this camera with "
                "one of those."
            )
        self._srgb = _PIXEL_ENCODING[renderer] == "srgb"
        surface = _plain_surface_rgb(self._env, plane.entity_name, subject=f"{subject} on entity")
        if min(surface) < MIN_SURFACE_CHANNEL:
            raise ValueError(
                f"{subject} on entity '{plane.entity_name}', whose colour {surface} has a channel under "
                f"{MIN_SURFACE_CHANNEL}. The marks are shaded as the surface is, by dividing its rendered pixels by its "
                "colour, and so dark a channel turns the frame's 8-bit steps into visible bands. Lighten the "
                f"surface's colour to at least {MIN_SURFACE_CHANNEL} per channel."
            )
        # A marked pixel is the surface's rendered pixel times the mark's colour over this, at full weight.
        self._surface = torch.tensor(surface, device=self._env.device)
        self._color = None if self._rgb is None else torch.tensor(self._rgb, device=self._env.device)

    def paint(self, rgb: torch.Tensor, buffers: OverlayBuffers, envs_idx: torch.Tensor) -> torch.Tensor:
        plane: PaintedPlane = self._metric.painted_plane(envs_idx)
        marks = plane.weight.float()  # a bool or half grid, as build() allows
        rows, cols = plane.grid.shape
        offset = buffers.pointcloud - plane.origin
        u, v = (offset * plane.u_axis).sum(-1), (offset * plane.v_axis).sum(-1)
        gap = (offset * torch.linalg.cross(plane.u_axis, plane.v_axis)).sum(-1)
        on_plane = buffers.pointcloud_valid & (gap.abs() < self.max_gap)

        # The pixels whose centre lies on the grid, grown by one: a sub-pixel interpolates between neighbouring
        # centres, so every sub-pixel on the grid belongs to one of them. The crop is their bounding box over the
        # batch, plus a pixel of margin so its sub-pixels interpolate as they would across the whole frame.
        _, on_grid = self._cells(u, v, plane, rows, cols)
        near = F.max_pool2d((on_plane & on_grid)[:, None].float(), 3, stride=1, padding=1)[:, 0].amax(dim=0) > 0
        hit_rows, hit_cols = (
            torch.nonzero(near.any(dim=1)),
            torch.nonzero(near.any(dim=0)),
        )  # syncs: a render, not a step
        if hit_rows.numel() == 0:
            return rgb
        height, width = near.shape
        r0, r1 = max(int(hit_rows[0]) - 1, 0), min(int(hit_rows[-1]) + 2, height)
        c0, c1 = max(int(hit_cols[0]) - 1, 0), min(int(hit_cols[-1]) + 2, width)

        uv = torch.stack([u, v], dim=1)[:, :, r0:r1, c0:c1]
        if self.supersample > 1:
            uv = F.interpolate(uv, scale_factor=self.supersample, mode="bilinear", align_corners=False)
        cell_idx, on_grid = self._cells(uv[:, 0], uv[:, 1], plane, rows, cols)
        weight = marks.flatten(1).gather(1, cell_idx.flatten(1)).view_as(cell_idx) * on_grid
        if plane.rgb is not None:
            # Each sub-pixel's cell's colour, weighted: averaged with the weights, over each pixel's sub-pixels.
            cells = plane.rgb.float().flatten(1, 2)  # (K, cells, 3)
            index = cell_idx.flatten(1)[..., None].expand(-1, -1, 3)
            color = cells.gather(1, index).view(*cell_idx.shape, 3) * weight[..., None]
            if self.supersample > 1:
                color = F.avg_pool2d(color.permute(0, 3, 1, 2), self.supersample).permute(0, 2, 3, 1)
        if self.supersample > 1:
            weight = F.avg_pool2d(weight[:, None], self.supersample)[:, 0]
        on = on_plane[:, r0:r1, c0:c1]
        weight = torch.where(on, weight, 0.0)[..., None]
        if plane.rgb is None:
            color = self._color * weight  # one colour: the weighted colour is it times the weight
        else:
            color = torch.where(on[..., None], color, 0.0)

        painted = rgb.clone()
        # The surface's rendered pixel already holds the light that reached it (its shadows included); the mark keeps
        # that light and changes the colour it falls on. Each sub-pixel's tint is lerp(1, colour / surface, weight),
        # and averaged over a pixel's sub-pixels that is 1 - weight + (colour * weight) / surface.
        tint = 1.0 - weight + color / self._surface
        crop = rgb[:, r0:r1, c0:c1].float()
        if self._srgb:
            # Through linear light, for the marked pixels only: the crop is the marks' bounding box over every image of
            # the frame, most of it unmarked, and an unmarked pixel's tint is 1.
            marked = weight[..., 0] > 0  # syncs, as the crop does: a render, not a step
            blend = crop.clone()
            blend[marked] = _srgb_encode((_srgb_decode(crop[marked] / 255.0) * tint[marked]).clamp_(max=1.0)) * 255.0
        else:
            blend = (crop * tint).clamp_(max=255.0)
        painted[:, r0:r1, c0:c1] = blend if rgb.is_floating_point() else blend.round().to(rgb.dtype)
        return painted

    @staticmethod
    def _cells(u, v, plane: PaintedPlane, rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The flat cell index of each ``(u, v)`` (0 off the grid) and whether it is on the grid."""
        grid = plane.grid
        row = torch.floor((grid.v0 - v) / grid.cell).long()
        col = torch.floor((u - grid.u0) / grid.cell).long()
        on_grid = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
        return torch.where(on_grid, row * cols + col, 0), on_grid


class BackgroundComposite(CameraOverlay):
    """Replaces the pixels where the sim shows a background with another image: one of ``entities``, or nothing at
    all (segmentation id 0, the renderer's own background). The overlays after it leave those pixels as composited
    (:meth:`replaced_pixels`). Subclasses say where the background comes from (:meth:`background`).

    With ``shadow_strength`` the background is darkened where the sim pass shows shadows on ``entities``
    (:mod:`eden.utils.background_shadows`). The rasterizer renders the view a second time without shadows to measure
    them, which needs ``EnvOptions.rasterizer_shadows`` on and a GPU-backed OpenGL context. The batch renderer bakes
    shadows in, so it has them estimated, which is exact for flat, untextured entities under directional lights: an
    image-textured entity is refused, and at least one ``BatchRendererOptions.lights`` entry needs ``castshadow=True``.
    Shading holds about ``20 * N * H * W`` bytes of temporaries per frame (about 1 GB at 1024 envs of 224x224).

    Parameters
    ----------
    entities : list[str]
        Scene entities whose pixels the background replaces, every link of each (``segmentation_level`` may give an
        entity several ids). The renderer's empty background is always replaced.
    shadow_strength : float
        How strongly the sim's shadows on ``entities`` darken the background, in ``[0, 1]``: ``0`` pastes it as is,
        so objects cast no shadow on it; ``1`` turns a background pixel that receives no light black. Needs at least
        one entry in ``entities``, since the shadows are read from them.
    """

    _needs_segmentation = True

    entities: list[str] = []
    shadow_strength: float = 0.0

    @classmethod
    def check_parameters(cls, params: Mapping[str, Any]) -> None:
        strength = params["shadow_strength"]
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"{cls.__name__} needs shadow_strength in [0, 1]; got {strength}.")
        if strength > 0.0 and not params["entities"]:
            raise ValueError(
                f"{cls.__name__}(shadow_strength={strength}) needs at least one entry in entities: the shadows are "
                "read from those entities in the sim pass. Name the surface the background stands in for, e.g. "
                "entities=['plane']."
            )

    def build(self) -> None:
        # Deferred: the env module imports the entities, so a module-scope import here would be a cycle.
        from eden.utils.misc import resolve_named_entity

        entity_seg_ids = self._env.entity_segmentation_ids
        surface_ids = []
        for entity_name in self.entities:
            entity = resolve_named_entity(
                self._env.entities,
                entity_name,
                hint=f"{type(self).__name__}(entities=...) lists it; list only those.",
            )
            # An entity owns one id per link/geom at segmentation_level="link"/"geom", and every one of them has to
            # be listed or the entity is only partly treated as background.
            seg_ids = entity_seg_ids.get(entity.idx)
            if not seg_ids:
                raise ValueError(
                    f"{type(self).__name__}(entities=...) names entity '{entity_name}', but the scene assigns it no "
                    "segmentation id, so it could not be masked as background."
                )
            surface_ids.extend(seg_ids)
        #: The segmentation ids replaced: the renderer's empty background (0) and every id of ``entities``.
        self.background_ids = [0, *surface_ids]
        self._background_ids = torch.tensor(self.background_ids, device=self._env.device)
        # The surfaces that receive shadows: every entity id, not the renderer's empty background.
        self._shadow_surface_ids = torch.tensor(surface_ids, dtype=torch.long, device=self._env.device)
        if self.shadow_strength > 0.0:
            self._check_sim_pass_casts_shadows()

    def background(self, rgb: torch.Tensor, envs_idx: torch.Tensor) -> torch.Tensor:
        """The image the background pixels of ``rgb`` take: ``(1, H, W, 3)`` for every image, or ``(N, H, W, 3)``
        one per image, on the env's device, in any dtype holding ``[0, 255]``.
        """
        raise NotImplementedError

    def replaced_pixels(self, buffers: OverlayBuffers) -> torch.Tensor:
        return torch.isin(buffers.segmentation, self._background_ids)

    def paint(self, rgb: torch.Tensor, buffers: OverlayBuffers, envs_idx: torch.Tensor) -> torch.Tensor:
        background = self.background(rgb, envs_idx).to(self._env.device)
        if self.shadow_strength > 0.0:
            background = self._shade(background, rgb, buffers.segmentation)
        return torch.where(self.replaced_pixels(buffers)[..., None], background.to(rgb.dtype), rgb)

    def _shade(self, background: torch.Tensor, rgb: torch.Tensor, segmentation: torch.Tensor) -> torch.Tensor:
        """``background`` darkened by the sim's shadows on the background entities, one image per image of ``rgb``.

        Shading runs in float32, so at its peak it holds about ``20 * N * H * W`` bytes of temporaries (measured: a
        float copy of the background plus a few single-channel buffers). At 1024 envs of 224x224 that is about 1 GB,
        freed when the call returns.
        """
        unshadowed = self._camera.render_unshadowed_rgb()
        factor = background_shadow_factor(
            rgb,
            segmentation,
            self._shadow_surface_ids,
            self.shadow_strength,
            unshadowed_rgb=unshadowed,
        )
        shaded = background.float() * factor
        del factor
        return shaded if background.is_floating_point() else shaded.round_()

    def _check_sim_pass_casts_shadows(self) -> None:
        """Fail fast if ``shadow_strength`` is set but the sim pass renders no shadows to carry over."""
        camera = self._camera
        name = f"Camera {camera.display_name}'s {type(self).__name__} sets shadow_strength={self.shadow_strength}, but"
        if camera.renderer_kind == "raytracer":
            raise ValueError(
                f"{name} its RGB comes from the ray tracer. Background shadows need the rasterizer or the batch "
                "renderer."
            )
        if camera.renderer_kind == "batch_renderer":
            if not camera.batch_renderer_casts_shadows:
                raise ValueError(
                    f"{name} no batch-renderer light casts shadows. Set castshadow=True on at least one entry of "
                    "BatchRendererOptions.lights."
                )
            self._check_background_untextured(name)
            return
        visualizer = self._env.scene.visualizer
        if not visualizer.context.shadow:
            raise ValueError(f"{name} the rasterizer renders no shadows. Set EnvOptions.rasterizer_shadows=True.")
        if visualizer.is_software:
            # Genesis' pyrender drops shadows on a software GL renderer (llvmpipe, Apple Software Renderer)
            # whatever `context.shadow` says, so both passes would match and nothing would be shaded.
            raise ValueError(
                f"{name} the rasterizer is running on a software OpenGL renderer, which draws no shadows. Run on a "
                "GPU-backed OpenGL context (e.g. PYOPENGL_PLATFORM=egl with a GPU), or set shadow_strength=0."
            )

    def _check_background_untextured(self, name: str) -> None:
        """Fail fast if a background entity is image-textured when the batch renderer has to estimate shadows.

        The batch renderer bakes shadows in, so :func:`~eden.utils.background_shadows.background_shadow_factor`
        takes each surface's brightest pixels as its unshadowed level. A texture's darker texels would then read as
        shadow and be printed onto the background: Genesis' default ``Plane`` checkerboard darkens every other tile
        by a third. The test is the batch renderer's own (an ``ImageTexture`` with image data on a vgeom's surface),
        so it rejects exactly what Madrona draws textured.

        Parameters
        ----------
        name : str
            Opening of the error message, naming the camera and its strength.
        """
        for entity_name in self.entities:
            for vgeom in getattr(self._env.entities[entity_name].entity, "vgeoms", ()):
                textures = vgeom.surface.get_rgba(batch=True).textures
                if any(isinstance(t, gs.textures.ImageTexture) and t.image_array is not None for t in textures):
                    raise ValueError(
                        f"{name} background entity '{entity_name}' is image-textured (Genesis' Plane is, by "
                        "default: a checkerboard). The batch renderer can't render an unshadowed pass, so the "
                        "shadows are estimated from each surface's brightest pixels, and a texture's darker texels "
                        "would be printed onto the background as shadow. Give the entity a plain colour, e.g. "
                        "PlaneOptions(color=(0.5, 0.5, 0.5)), or render with the rasterizer, which measures "
                        "shadows exactly on any texture."
                    )


@CAMERA_OVERLAY_REGISTRY.register()
class BackgroundImageComposite(BackgroundComposite):
    """Replaces the background with the image at ``image_path``, scaled to the camera's resolution: a SimplerEnv-style
    real-image backdrop, or a green screen.

    Parameters
    ----------
    image_path : str
        The background image.
    entities, shadow_strength
        See :class:`BackgroundComposite`.
    """

    image_path: str = ""

    @classmethod
    def check_parameters(cls, params: Mapping[str, Any]) -> None:
        super().check_parameters(params)
        if not params["image_path"]:
            raise ValueError(
                "BackgroundImageComposite needs image_path, the background image: "
                "BackgroundImageComposite.configure(image_path='backdrop.png', entities=['plane'])."
            )

    def __init__(self, camera: Camera, options: CameraOverlayOptions) -> None:
        super().__init__(camera, options)
        image = Image.open(self.image_path).convert("RGB").resize(tuple(camera.cam_resolution))
        self._image = torch.as_tensor(np.array(image), device=self._env.device)[None]

    def background(self, rgb: torch.Tensor, envs_idx: torch.Tensor) -> torch.Tensor:
        return self._image


@CAMERA_OVERLAY_REGISTRY.register()
class SplatComposite(BackgroundComposite):
    """Replaces the background with the env's Gaussian splat, rendered from the camera's pose
    (:class:`~eden.utils.batch_splat_renderer.BatchSplatRenderer`). Only a splat env's camera can carry it.

    Whether the splat is rendered once and used for every env's image, or once per env, is the camera's
    (``Camera.splat_single_view``: :class:`~eden.entities.camera.SplatCamera` or
    :class:`~eden.entities.camera.BatchSplatCamera`, picked by ``SplatOptions.camera_mode``).

    Parameters
    ----------
    entities, shadow_strength
        See :class:`BackgroundComposite`.
    """

    def __init__(self, camera: Camera, options: CameraOverlayOptions) -> None:
        super().__init__(camera, options)
        if camera.splat_single_view is None:
            raise ValueError(
                f"SplatComposite renders the env's Gaussian splat, but camera {camera.display_name} is not a splat "
                "env's camera. Build the task with splat_options (an EdenSplatConfig / EdenSplatRLConfig), or use "
                "BackgroundImageComposite for a static backdrop."
            )

    def background(self, rgb: torch.Tensor, envs_idx: torch.Tensor) -> torch.Tensor:
        env = self._env
        renderer = env._splat_renderer
        # Per-env mode: refresh the animated Gaussians from sim poses (once per render).
        env._refresh_splat_gaussians()

        # Per-env camera poses: a free camera reports one (4,4), an attached one (N,4,4); in per-env Gaussian mode a
        # single shared pose is broadcast to num_envs so every env's (independently animated) Gaussians are rendered.
        cam_T = torch.as_tensor(self._camera.get_transform(), dtype=torch.float32, device=env.device)
        if cam_T.ndim == 2:
            cam_T = cam_T.unsqueeze(0)
        cam_pos, cam_rot = cam_T[:, :3, 3], cam_T[:, :3, :3]
        if renderer.per_env_gaussians and cam_pos.shape[0] == 1:
            cam_pos = cam_pos.expand(env.num_envs, 3)
            cam_rot = cam_rot.expand(env.num_envs, 3, 3)
        if self._camera.splat_single_view:
            cam_pos, cam_rot = cam_pos[:1], cam_rot[:1]  # one view, used for every image

        width, height = self._camera.cam_resolution
        splat, _ = renderer.render(cam_pos, cam_rot, height, width, self._camera.cam_fov)
        return splat.mul(255).byte()  # the sim pass's uint8 RGB
