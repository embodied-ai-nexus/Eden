"""``eden.entities.camera_overlays``: what a camera's overlays paint, where, onto which frames, and what is refused.

The overlays and the metric here are test doubles registered below: ``FloorTintTestOverlay`` paints the floor
magenta by the frame's point cloud, ``TintAllTestOverlay`` paints whole images and needs no point cloud, and
``FloorSquareTestMetric`` keeps a square of marks on the floor for :class:`PlanePaintOverlay` to paint. The splat
renderer is a stand-in, the one Genesis-side boundary here. Magenta and
pure green cannot come from the scene, so a painted pixel cannot be mistaken for a rendered one; nor can a mark, which
:class:`PlanePaintOverlay` shades by the grey floor's own pixel, keeping only its green (:data:`INK_TINT`). The whiteboard
tasks' tests exercise :class:`PlanePaintOverlay` on a real metric (``tests/tasks/whiteboard/test_custom_terms.py``).
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

import av
import numpy as np
import pytest
import torch
import genesis as gs
from genesis.options.sensors import RasterizerCameraOptions
from PIL import Image

from eden import metrics, observations
from eden.entities.camera import SplatCamera
from eden.entities.camera_overlays import (
    CAMERA_OVERLAY_REGISTRY,
    BackgroundImageComposite,
    CameraOverlay,
    PaintedPlane,
    PlanePaintOverlay,
    SplatComposite,
)
from eden.envs.base import EnvBase
from eden.managers.metric_manager import METRIC_TERM_REGISTRY
from eden.managers.metric_manager import MetricTerm as MetricClassTerm
from eden.entities.camera import Camera
from eden.options import BoxOptions, EntityOptions, EnvOptions, ObservationGroupOptions, PlaneOptions, SceneOptions
from eden.options.camera import CameraOptions, CameraOverlayOptions, CamerasOptions
from eden.terms import MetricTerm, ObsTerm
from eden.utils.configs import EdenConfig, MetricManagerOptions, ObservationManagerOptions
from eden.utils.plane_grid import PlaneGrid
from tests.support.envs import build_env

MAGENTA_RGB = (255, 0, 255)
MAGENTA = torch.tensor(MAGENTA_RGB, dtype=torch.float32)
GREEN = (0, 255, 0)
RES = (64, 48)
LOOK = {"cam_pos": (1.5, 0.0, 1.0), "cam_lookat": (0.0, 0.0, 0.3), "cam_resolution": RES}
#: Half the side (m) of the square FloorSquareTestMetric inks, centred on the world origin.
SQUARE = 0.3
#: The floor's plain colour, and the marks' (as the floor's is given): a fully marked pixel is the floor's rendered
#: pixel times INK / FLOOR = INK_TINT, exactly, which no unmarked pixel of the grey floor matches.
FLOOR = (0.8, 0.8, 0.8)
INK = (0.0, 0.4, 0.0)
INK_TINT = np.array((0.0, 0.5, 0.0))
#: A grey ink, half the floor's: every channel is tinted, so the batch renderer's sRGB step shows in all three.
MID_INK = (0.4, 0.4, 0.4)
MID_TINT = np.array((0.5, 0.5, 0.5))
# tests/support/batch_renderer_ink.py builds its scene from this module (_config, _shadowing_cube, LOOK, SQUARE, INK,
# INK_TINT, MID_INK, MID_TINT): keep those names, or update it too.
#: Ink brighter than the floor: its tint lifts the floor's pixel past 255 where the floor is well lit.
BRIGHT_INK = (1.0, 1.0, 1.0)


def _shadowing_cube(light_dir) -> tuple[float, float, float]:
    """Where the cube stands, 0.5 m up, for a directional light along ``light_dir`` to cast its shadow just off the
    square's centre: across part of the square, so the square holds both lit and shadowed floor.
    """
    height, target = 0.5, np.array((0.08, 0.0))
    direction = np.asarray(light_dir, dtype=float)
    x, y = target - height * direction[:2] / -direction[2]
    return (float(x), float(y), height)


@CAMERA_OVERLAY_REGISTRY.register()
class FloorTintTestOverlay(CameraOverlay):
    """Paints the floor (world points below ``height``) magenta in the images of the envs in ``envs``."""

    _needs_pointcloud = True

    height: float = 0.01
    envs: list[int] = [0, 1]

    def paint(self, rgb, buffers, envs_idx):
        floor = buffers.pointcloud_valid & (buffers.pointcloud[..., 2] < self.height)
        chosen = torch.isin(envs_idx, torch.tensor(self.envs, device=envs_idx.device))
        return torch.where((floor & chosen[:, None, None])[..., None], MAGENTA.to(rgb), rgb)


@CAMERA_OVERLAY_REGISTRY.register()
class TintAllTestOverlay(CameraOverlay):
    """Paints the images of the envs in ``envs`` pure green: an overlay that needs no point cloud, so a sensor camera
    may carry it.
    """

    envs: list[int] = [0, 1]

    def paint(self, rgb, buffers, envs_idx):
        chosen = torch.isin(envs_idx, torch.tensor(self.envs, device=envs_idx.device))
        return torch.where(chosen[:, None, None, None], torch.tensor(GREEN, dtype=rgb.dtype, device=rgb.device), rgb)


@CAMERA_OVERLAY_REGISTRY.register()
class FloatTestOverlay(CameraOverlay):
    """Breaks :meth:`CameraOverlay.paint`'s contract on purpose: returns the frame as float."""

    def paint(self, rgb, buffers, envs_idx):
        return rgb.float()


class _FlatSplatRenderer:
    """Stands in for ``BatchSplatRenderer``, whose gsplat CUDA rasterizer this machine may not have: a uniform
    magenta splat, the ``(n_views, H, W, 3)`` float in ``[0, 1]`` the real one returns.
    """

    per_env_gaussians = False

    def render(self, cam_pos, cam_rot, height, width, fov):
        splat = torch.zeros((cam_pos.shape[0], height, width, 3))
        splat[..., 0] = splat[..., 2] = 1.0
        return splat, None


@METRIC_TERM_REGISTRY.register()
class FloorSquareTestMetric(MetricClassTerm):
    """Marks a ``2 SQUARE`` square of the floor, centred on the origin, in the envs ``inked_envs``.

    Records the envs every ``painted_plane()`` call asked for in ``asked``. ``weight_dtype`` picks the grid's
    dtype; ``cell_rgb`` gives every cell that colour (marks of many colours) instead of leaving it to the overlay, in
    ``rgb_dtype``; ``u_axis``, ``flat_weight``, ``grid_extra_rows`` and an integer ``weight_dtype`` or ``rgb_dtype``
    break the :class:`PaintedPlane` contract on purpose.
    """

    inked_envs: list[int] = [0]
    cell_rgb: tuple[float, float, float] | None = None
    u_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    flat_weight: bool = False
    grid_extra_rows: int = 0
    weight_dtype: str = "float32"
    rgb_dtype: str = "float32"
    cell: float = 0.01
    entity_name: str = "floor"

    def build(self) -> None:
        self.asked: list[list[int]] = []
        side = round(2 * SQUARE / self.cell)
        weight = torch.zeros(self.num_envs, side, side, device=self.device)
        weight[self.inked_envs] = 1.0
        self._weight = weight.to(getattr(torch, self.weight_dtype))

    def compute(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def painted_plane(self, envs_idx: torch.Tensor) -> PaintedPlane:
        self.asked.append(envs_idx.tolist())
        weight = self._weight[envs_idx]
        return PaintedPlane(
            origin=torch.zeros(3, device=self.device),
            u_axis=torch.tensor(self.u_axis, device=self.device),
            v_axis=torch.tensor((0.0, 1.0, 0.0), device=self.device),
            grid=PlaneGrid(
                -SQUARE, SQUARE, self.cell, self._weight.shape[1] + self.grid_extra_rows, self._weight.shape[2]
            ),
            weight=weight.flatten(1) if self.flat_weight else weight,
            entity_name=self.entity_name,
            rgb=None
            if self.cell_rgb is None
            else torch.tensor(self.cell_rgb, device=self.device)
            .expand(*weight.shape, 3)
            .to(getattr(torch, self.rgb_dtype)),
        )


def _inked(frame) -> np.ndarray:
    """Where a frame shows a mark: the floor's pixel with its red and blue taken out (:data:`INK_TINT`)."""
    frame = np.asarray(frame)
    return (frame[..., 0] == 0) & (frame[..., 2] == 0) & (frame[..., 1] > 0)


def _green_fraction(frame) -> float:
    """Share of pixels exactly pure green."""
    return float((np.asarray(frame) == GREEN).all(axis=-1).mean())


def _magenta_fraction(frame) -> float:
    """Share of pixels within codec tolerance of magenta."""
    return float((np.abs(np.asarray(frame).astype(int) - MAGENTA.numpy().astype(int)).max(axis=-1) < 60).mean())


def _config(
    num_envs: int = 1,
    metric: dict | None = None,
    floor_color: tuple[float, ...] | None = FLOOR,
    shadows: bool = False,
    cube_pos: tuple[float, float, float] = (0.0, 0.0, 0.5),
    **cameras: CameraOptions,
) -> EdenConfig:
    """Floor (``floor_color``, or Genesis' checkerboard texture for ``None``) and a cube at ``cube_pos``; the
    ``square`` metric configured with ``metric``; one ``CameraOptions`` per camera name.
    """
    floor = PlaneOptions() if floor_color is None else PlaneOptions(color=(*floor_color, 1.0))
    return EdenConfig(
        env_options=EnvOptions(
            num_envs=num_envs, sim_dt=0.005, rasterizer_shadows=shadows, env_separate_rigid=num_envs > 1
        ),
        scene_options=SceneOptions(floor=floor, cube=BoxOptions(size=(0.2, 0.2, 0.2), default_root_pos=cube_pos)),
        metric_options=MetricManagerOptions(
            lifted=MetricTerm.configure(func=metrics.object_lifted, params={"entity_name": "cube", "min_height": 0.3}),
            square=FloorSquareTestMetric.configure(**(metric or {})),
        ),
        cameras_options=CamerasOptions(**cameras),
    )


def _paint_square(metric_name: str = "square", color: tuple[float, float, float] | None = INK) -> CameraOptions:
    overlay = PlanePaintOverlay.configure(metric_name=metric_name, color=color)
    return CameraOptions(**LOOK, debug=True, overlays=[overlay])


def _build_or_skip(config: EdenConfig) -> EnvBase:
    env = build_env(config)
    try:
        next(iter(env.cameras.values())).render_rgb()
    except Exception as e:  # noqa: BLE001 - an offscreen-render failure is a skip, not a test failure
        env.destroy()
        pytest.skip(f"offscreen rendering unavailable: {e}")
    return env


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """One env over a floor, each camera the same view.

    ``plain`` has no overlay, ``tinted`` paints the floor magenta, ``square`` paints the metric's floor square
    (``bright`` in ink brighter than the floor), and
    ``sensor`` is a rasterizer sensor that paints whole images green. ``composited`` and ``splat`` replace the floor
    and the sky with magenta (an image, a stand-in splat) and paint whole images green on top.
    """
    background = tmp_path_factory.mktemp("overlays") / "background.png"
    Image.new("RGB", (32, 24), MAGENTA_RGB).save(background)
    paint_all = {"overlays": [TintAllTestOverlay.configure()]}
    config = _config(
        plain=CameraOptions(**LOOK, debug=True),
        tinted=CameraOptions(**LOOK, debug=True, overlays=[FloorTintTestOverlay.configure()]),
        square=_paint_square(),
        bright=CameraOptions(
            **LOOK, debug=True, overlays=[PlanePaintOverlay.configure(metric_name="square", color=BRIGHT_INK)]
        ),
        sensor=CameraOptions(**LOOK, sensor_options=RasterizerCameraOptions(), **paint_all),
        composited=CameraOptions(
            **LOOK,
            debug=True,
            overlays=[
                BackgroundImageComposite.configure(image_path=str(background), entities=["floor"]),
                TintAllTestOverlay.configure(),
            ],
        ),
    )
    env = EnvBase.from_config(config, show_viewer=False)
    # The splat camera is what a SplatEnvBase would build for this options entry; its renderer is the boundary.
    # Not in the config: a plain env's camera refuses a SplatComposite.
    splat = [SplatComposite.configure(entities=["floor"]), TintAllTestOverlay.configure()]
    env.cameras["splat"] = SplatCamera("splat", env, CameraOptions(**LOOK, debug=True, overlays=splat))
    env.build()
    env._splat_renderer = _FlatSplatRenderer()
    env._refresh_splat_gaussians = lambda: None
    try:
        env.cameras["plain"].render_rgb()
    except Exception as e:  # noqa: BLE001 - an offscreen-render failure is a skip, not a test failure
        env.destroy()
        pytest.skip(f"offscreen rendering unavailable: {e}")
    yield env
    env.destroy()


@pytest.fixture(scope="module")
def batched():
    """Two envs rendered separately (``env_separate_rigid``); the overlays paint env 1 only."""
    env = _build_or_skip(
        _config(
            num_envs=2,
            metric={"inked_envs": [1]},
            cam=CameraOptions(**LOOK, debug=True, overlays=[FloorTintTestOverlay.configure(envs=[1])]),
            square=_paint_square(),
            sensor=CameraOptions(
                **LOOK, sensor_options=RasterizerCameraOptions(), overlays=[TintAllTestOverlay.configure(envs=[1])]
            ),
        )
    )
    yield env
    env.destroy()


def test_an_overlay_paints_the_rgb_frame_and_render_stays_the_raw_pass(env):
    plain = env.cameras["plain"].render_rgb()
    tinted = env.cameras["tinted"].render_rgb()
    raw, _, _, _ = env.cameras["tinted"].render()

    assert _magenta_fraction(plain) == 0.0, "test premise: the scene has no magenta"
    assert _magenta_fraction(tinted) > 0.2
    assert (type(tinted), tinted.shape, tinted.dtype) == (type(plain), plain.shape, plain.dtype)
    # Off the floor the frame is the renderer's, untouched.
    off_floor = np.abs(tinted.astype(int) - MAGENTA.numpy().astype(int)).max(axis=-1) >= 60
    assert np.array_equal(tinted[off_floor], plain[off_floor])
    assert _magenta_fraction(raw) == 0.0


def test_the_recording_holds_the_painted_frame(env, tmp_path):
    out, cam = tmp_path / "tinted.mp4", env.cameras["tinted"]
    cam.start_recording(str(out))
    painted = cam.render_rgb()
    cam.stop_recording()

    with av.open(str(out)) as container:
        (frame,) = [f.to_ndarray(format="rgb24") for f in container.decode(container.streams.video[0])]
    assert np.abs(frame.astype(int) - painted.astype(int)).mean() < 10.0
    assert _magenta_fraction(frame) > 0.2


def test_the_surface_marks_land_where_the_camera_sees_them(env):
    """Every pixel whose floor point is well inside the square is the floor's pixel shaded into the marks' colour; one
    well outside is untouched.
    """
    plain = env.cameras["plain"].render_rgb()
    painted = env.cameras["square"].render_rgb()
    points, valid = env.cameras["square"].render_pointcloud(world_frame=True)
    on_floor = valid & (np.abs(points[..., 2]) < 2e-3)
    reach = np.abs(points[..., :2]).max(axis=-1)

    inside, outside = on_floor & (reach < SQUARE - 0.03), reach > SQUARE + 0.03
    assert inside.sum() > 20, "test premise: the camera sees the square"
    assert np.array_equal(painted[inside], np.round(plain[inside] * INK_TINT).astype(np.uint8))
    assert np.array_equal(painted[outside], plain[outside])


def test_a_mark_keeps_the_shadow_the_scene_casts_on_its_surface():
    """A mark is a change of colour on the floor, so it takes the light the floor took: where the cube's shadow falls
    across the square, the marks darken exactly as the floor does, rather than covering the shadow in flat colour.
    """
    # The rasterizer lights the scene with Genesis' default light (an Eden config does not set it).
    cube = _shadowing_cube(gs.options.VisOptions().lights[0].dir)
    config = _config(shadows=True, cube_pos=cube, plain=CameraOptions(**LOOK, debug=True), square=_paint_square())
    env = _build_or_skip(config)
    try:
        plain = env.cameras["plain"].render_rgb()
        painted = env.cameras["square"].render_rgb()
        points, valid = env.cameras["square"].render_pointcloud(world_frame=True)
        inside = valid & (np.abs(points[..., 2]) < 2e-3) & (np.abs(points[..., :2]).max(axis=-1) < SQUARE - 0.03)
        floor_green = plain[inside][:, 1].astype(float)
        assert floor_green.min() < 0.6 * floor_green.max(), "test premise: the cube shadows part of the square"
        assert np.array_equal(painted[inside], np.round(plain[inside] * INK_TINT).astype(np.uint8))
    finally:
        env.destroy()


def test_ink_brighter_than_its_surface_is_clipped_at_full_brightness(env):
    """A tint above 1 lifts a bright floor pixel past 255 (under light brighter than the default, or in a highlight);
    the frame keeps 255 rather than wrapping round. A frame of near-white floor stands in for that light.
    """
    raw, _, _, _ = env.cameras["bright"].render()
    painted = env.cameras["bright"]._paint_overlays(np.full_like(raw, 250))
    points, valid = env.cameras["bright"].render_pointcloud(world_frame=True)
    inside = valid & (np.abs(points[..., 2]) < 2e-3) & (np.abs(points[..., :2]).max(axis=-1) < SQUARE - 0.03)
    assert inside.sum() > 20, "test premise: the camera sees the square"
    assert 250 * max(BRIGHT_INK) / FLOOR[0] > 255, "test premise: the tint lifts the frame past 255"
    assert (painted[inside] == 255).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the batch renderer needs CUDA")
# Not an Eden dependency, and not built for every platform. Only its absence skips: an
# installed gs_madrona that fails to import fails the test, as it would fail a batch-rendered env.
@pytest.mark.skipif(importlib.util.find_spec("gs_madrona") is None, reason="the batch renderer needs gs-madrona")
def test_on_the_batch_renderer_a_mark_is_shaded_in_linear_light():
    """Madrona multiplies a surface's colour by the light in linear light and encodes its pixels as sRGB, so a mark's
    tint applies to the decoded pixel: the painted square is the floor's pixels decoded, tinted and re-encoded, across
    the cube's shadow (a floor actually coloured in the ink renders the same, to within a level, lit or shadowed).

    In a process of its own (``tests/support/batch_renderer_ink.py``): this one holds Genesis on gs.cpu, where the
    batch renderer does not run. So it runs only on a CUDA machine with gs-madrona installed.
    """
    repo = pathlib.Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "tests.support.batch_renderer_ink"],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=900,
    )
    line = next((line for line in result.stdout.splitlines() if line.startswith("RESULT ")), None)
    assert line is not None, f"batch_renderer_ink printed no RESULT line; stderr tail:\n{result.stderr[-3000:]}"
    measured = json.loads(line.removeprefix("RESULT "))
    assert measured["renderer"] == "batch_renderer" and measured["inside"] > 20, measured
    darkest, brightest = measured["floor_green"]
    assert darkest < 0.6 * brightest, f"test premise: the cube shadows part of the square ({measured})"
    assert measured["max_error"] <= 1.0 and measured["max_error_mid"] <= 1.0, measured


def test_a_renderer_whose_colour_response_is_not_measured_is_refused(monkeypatch):
    """The ray tracer's colour response is not measured, so shading by it could be silently wrong: refused at build.
    The renderer is the boundary (no ray tracer here), so the camera reports one.
    """
    monkeypatch.setattr(Camera, "renderer_kind", property(lambda self: "raytracer"))
    with pytest.raises(
        ValueError,
        match=r"camera 'cam', whose frames come from the raytracer\..*rasterizer and the batch renderer only",
    ):
        build_env(_config(cam=_paint_square()))


def test_painting_a_torch_or_rgba_frame_keeps_its_type_and_alpha(env):
    """The batch renderer hands back torch frames, some RGBA: painted as the rasterizer's, returned in kind."""
    cam = env.cameras["tinted"]
    raw, _, _, _ = cam.render()
    as_numpy = cam._paint_overlays(raw)

    as_torch = cam._paint_overlays(torch.as_tensor(np.ascontiguousarray(raw)))
    assert isinstance(as_torch, torch.Tensor) and as_torch.dtype == torch.uint8
    assert np.array_equal(as_torch.cpu().numpy(), as_numpy)

    rgba = cam._paint_overlays(np.concatenate([raw, np.full_like(raw[..., :1], 7)], axis=-1))
    assert np.array_equal(rgba[..., :3], as_numpy) and (rgba[..., 3] == 7).all()


def test_a_sensor_camera_paints_an_overlay_that_needs_no_point_cloud(env):
    frames = env.cameras["sensor"].render_rgb()
    assert _green_fraction(frames.cpu().numpy()[0]) == 1.0


def test_a_sensor_camera_paints_each_env_s_image_for_that_env(batched):
    """The sensor backend renders every env of the scene, one image each."""
    frames = batched.cameras["sensor"].render_rgb().cpu().numpy()
    assert frames.shape[0] == 2
    assert _green_fraction(frames[0]) == 0.0 and _green_fraction(frames[1]) == 1.0


@pytest.mark.parametrize("name", ["composited", "splat"])
def test_the_overlays_leave_what_a_background_composite_replaced(env, name):
    """The floor and sky are a magenta image (or splat) there, no longer the sim surfaces the overlays locate marks on:
    only the cube, still the sim's, is painted.
    """
    frame = env.cameras[name].render_rgb()
    assert _magenta_fraction(frame) > 0.5, "the composite stays where it replaced the floor and sky"
    assert _green_fraction(frame) * frame.shape[0] * frame.shape[1] >= 10, "the cube, still the sim's, is painted"
    assert _magenta_fraction(frame) + _green_fraction(frame) > 0.99


@pytest.mark.parametrize("weight_dtype", ["bool", "float16"])
def test_a_bool_or_half_grid_paints_like_a_float_one(weight_dtype):
    env = _build_or_skip(_config(metric={"weight_dtype": weight_dtype}, cam=_paint_square()))
    try:
        assert _inked(env.cameras["cam"].render_rgb()).mean() > 0.01
    finally:
        env.destroy()


def test_marks_that_come_with_their_colours_are_shaded_as_one_colour_is():
    """Cells that carry the ink's colour paint exactly as the overlay's ``color`` would: the floor's pixel times INK_TINT."""
    config = _config(
        metric={"cell_rgb": INK}, plain=CameraOptions(**LOOK, debug=True), square=_paint_square(color=None)
    )
    env = _build_or_skip(config)
    try:
        plain = env.cameras["plain"].render_rgb()
        painted = env.cameras["square"].render_rgb()
        points, valid = env.cameras["square"].render_pointcloud(world_frame=True)
        inside = valid & (np.abs(points[..., 2]) < 2e-3) & (np.abs(points[..., :2]).max(axis=-1) < SQUARE - 0.03)
        assert inside.sum() > 20, "test premise: the camera sees the square"
        assert np.array_equal(painted[inside], np.round(plain[inside] * INK_TINT).astype(np.uint8))
    finally:
        env.destroy()


def test_each_image_of_a_batched_frame_is_painted_for_the_env_it_shows(batched):
    tinted = batched.cameras["cam"].render_rgb()
    squared = batched.cameras["square"].render_rgb()

    assert tinted.shape[0] == squared.shape[0] == 2
    assert _magenta_fraction(tinted[0]) == 0.0 and _magenta_fraction(tinted[1]) > 0.2
    assert not _inked(squared[0]).any() and _inked(squared[1]).any()


def test_the_marks_are_read_for_the_painted_envs_only(batched):
    """The build check reads one env, a frame the envs it shows: never the whole batch for a few images."""
    metric = batched.metric_manager.get_term("square")
    assert metric.asked[0] == [0], "build reads one env, not all two"
    batched.cameras["square"].render_rgb()
    assert metric.asked[-1] == [0, 1]


def test_the_batch_renderer_shows_every_env_and_a_count_it_cannot_explain_is_refused(batched, monkeypatch):
    """The batch renderer renders each env of the scene, whatever ``rendered_envs_idx`` says; the rasterizer renders
    ``rendered_envs_idx``. A frame whose image count matches neither cannot be painted, and is refused.
    """
    cam = batched.cameras["cam"]
    context = batched.scene.visualizer._context
    monkeypatch.setattr(context, "rendered_envs_idx", np.array([1]))
    with pytest.raises(RuntimeError, match=r"rendered 2 image\(s\).*rendered_envs_idx.*\(1\)"):
        cam._frame_envs_idx(2, batched=True)

    monkeypatch.setattr(cam._cam, "_batch_renderer", object())
    assert cam._frame_envs_idx(2, batched=True) == [0, 1]


def test_overlays_survive_a_saved_config():
    """``eden inference`` replays the run's ``config.json``: the overlay has to come back by key with its parameters."""
    options = CameraOptions(
        overlays=[PlanePaintOverlay.configure(metric_name="drawing", color=(0.0, 0.0, 0.0), max_gap=0.003)]
    )
    restored = CameraOptions.model_validate_json(options.model_dump_json())

    (overlay,) = restored.overlays
    assert isinstance(overlay, CameraOverlayOptions)
    assert (overlay.name, overlay.metric_name, overlay.max_gap) == ("plane_paint_overlay", "drawing", 0.003)


def test_an_overlay_read_back_from_a_saved_config_equals_the_one_written():
    """JSON has no tuple, but the read-back parameter takes its declared type again, so the overlay is equal."""
    written = PlanePaintOverlay.configure(metric_name="drawing", color=(0.0, 1.0, 0.0))
    (read_back,) = CameraOptions.model_validate_json(CameraOptions(overlays=[written]).model_dump_json()).overlays
    assert read_back.color == (0.0, 1.0, 0.0) and read_back == written
    assert read_back != PlanePaintOverlay.configure(metric_name="drawing", color=(0.0, 0.0, 1.0))


def test_a_misspelled_overlay_parameter_is_refused_at_configure():
    with pytest.raises(ValueError, match="metric_nme"):
        PlanePaintOverlay.configure(metric_nme="drawing")


def test_a_misspelled_overlay_parameter_in_a_config_file_is_refused_in_that_file():
    """An entry written by hand (or loaded from YAML / JSON) is checked against the overlay when it is constructed,
    as a term's options are, not later at env construction.
    """
    with pytest.raises(
        ValueError,
        match=r"\['metric_nme'\] for overlay 'plane_paint_overlay' \(PlanePaintOverlay\), allowed parameters: color",
    ):
        CameraOverlayOptions(name="plane_paint_overlay", metric_nme="drawing")
    with pytest.raises(ValueError, match="metric_nme"):
        CameraOptions.model_validate({"overlays": [{"name": "plane_paint_overlay", "metric_nme": "drawing"}]})
    # An overlay whose module is not imported yet cannot be checked here; the camera checks it.
    CameraOverlayOptions(name="not_imported_overlay", anything=1)


@pytest.mark.parametrize(
    ("overlay", "match"),
    [
        (
            CameraOverlayOptions(name="floor_tint_test_overlai"),
            r"Camera 'cam': CameraOptions\.overlays\[0\].*Did you mean 'floor_tint_test_overlay'",
        ),
        (CameraOverlayOptions(metric_name="drawing"), r"overlays\[0\] names no overlay.*configure\(\)"),
    ],
)
def test_an_overlay_that_cannot_be_constructed_fails_at_env_construction(overlay, match):
    with pytest.raises(ValueError, match=match):
        EnvBase.from_config(_config(cam=CameraOptions(**LOOK, debug=True, overlays=[overlay])))


def test_an_overlay_that_changes_the_frame_s_dtype_is_refused_naming_it(env):
    cam = env.cameras["plain"]
    raw = cam.render()[0]
    cam._overlays.append(FloatTestOverlay(cam, FloatTestOverlay.configure()))
    try:
        with pytest.raises(
            TypeError, match=r"FloatTestOverlay\.paint\(\) returned .* torch\.float32 for a .* torch\.uint8"
        ):
            cam._paint_overlays(raw)
    finally:
        cam._overlays.pop()


def test_a_camera_an_observation_reads_paints_once_the_env_is_built():
    """The observation manager renders each camera term once while it loads, to size it, before the overlays (and the
    metric one reads) are built: that frame is left unpainted, and every frame after the build is painted.
    """
    config = _config(cam=_paint_square())
    config.observation_options = ObservationManagerOptions(
        policy=ObservationGroupOptions(
            rgb=ObsTerm.configure(func=observations.rgb_image, params={"camera_name": "cam"})
        )
    )
    env = _build_or_skip(config)
    try:
        assert _inked(env.cameras["cam"].render_rgb()).mean() > 0.01
    finally:
        env.destroy()


def test_a_point_cloud_overlay_on_a_sensor_camera_fails_at_env_construction():
    sensor = CameraOptions(
        cam_resolution=RES, sensor_options=RasterizerCameraOptions(), overlays=[FloorTintTestOverlay.configure()]
    )
    with pytest.raises(ValueError, match="FloorTintTestOverlay paints by the frame's point cloud.*sensor_options"):
        EnvBase.from_config(_config(cam=sensor))


@pytest.mark.parametrize(
    ("metric_name", "metric", "error", "match"),
    [
        ("lifted", {}, TypeError, r"metric 'lifted' \(MetricTermFuncWrapper\) has no painted_plane\(\)"),
        (
            "square",
            {"flat_weight": True},
            ValueError,
            r"with weight \(1, 3600\) for one env; expected \(1, \*grid.shape\) = \(1, 60, 60\)",
        ),
        (
            "square",
            {"grid_extra_rows": 1},  # a whole grid of weights, but on a grid of another shape
            ValueError,
            r"with weight \(1, 60, 60\) for one env; expected \(1, \*grid.shape\) = \(1, 61, 60\)",
        ),
        ("square", {"u_axis": (2.0, 0.0, 0.0)}, ValueError, r"u_axis=\[2\.0, 0\.0, 0\.0\].*not orthonormal"),
        ("square", {"weight_dtype": "int64"}, ValueError, r"weight of dtype torch\.int64; it must be floating"),
        (
            "square",
            {"entity_name": "flor"},
            ValueError,
            r"on entity 'flor', which is not an entity of the scene \(entities: \['cube', 'floor'\]\)",
        ),
    ],
)
def test_a_metric_that_cannot_be_painted_fails_at_build(metric_name, metric, error, match):
    config = _config(metric=metric, cam=_paint_square(metric_name=metric_name))
    with pytest.raises(error, match=match):
        build_env(config)


@pytest.mark.parametrize(
    ("metric", "color", "match"),
    [
        ({}, None, r"marks of one colour \(rgb=None\), and the overlay has no color.*Configure color=\(r, g, b\)"),
        (
            {"cell_rgb": (1.0, 0.0, 0.0)},
            INK,
            r"with the marks' own colours \(rgb\), so color=\(0\.0, 0\.4, 0\.0\) would never be painted",
        ),
    ],
)
def test_a_colour_is_needed_exactly_when_the_marks_come_without_one(metric, color, match):
    config = _config(metric=metric, cam=_paint_square(color=color))
    with pytest.raises(ValueError, match=match):
        build_env(config)


def test_marks_whose_own_colours_are_not_floats_fail_at_build():
    config = _config(metric={"cell_rgb": (1.0, 0.0, 0.0), "rgb_dtype": "uint8"}, cam=_paint_square(color=None))
    with pytest.raises(
        ValueError, match=r"with rgb of dtype torch\.uint8; the marks' colours must be floating, in \[0, 1\]\. .*by 255"
    ):
        build_env(config)


@pytest.mark.parametrize(
    ("floor_color", "match"),
    [
        (None, r"on entity 'floor', which is textured \(ImageTexture\)\. .*PlaneOptions\(color=\(0\.8, 0\.8, 0\.8\)\)"),
        ((0.9, 0.9, 0.02), r"on entity 'floor', whose colour \(0\.9, 0\.9, 0\.02\) has a channel under 0\.05"),
    ],
)
def test_a_surface_the_marks_cannot_be_shaded_against_fails_at_build(floor_color, match):
    """The marks are the surface's rendered pixels over its colour: that needs one plain colour, not too dark."""
    with pytest.raises(ValueError, match=match):
        build_env(_config(floor_color=floor_color, cam=_paint_square()))


_TWO_TONE = """<mujoco><worldbody><body name="tile" pos="1 1 0.1">
<geom type="box" size="0.1 0.1 0.01" rgba="0.9 0.2 0.2 1"/><geom type="box" pos="0 0 0.05" size="0.05 0.05 0.01" rgba="0.2 0.2 0.9 1"/>
</body></worldbody></mujoco>"""


def _surface(kind: str, tmp_path):
    """An entity the marks cannot be shaded against, for ``kind``."""
    box = {"size": (0.1, 0.1, 0.1), "default_root_pos": (1.0, 1.0, 0.05)}
    if kind == "emissive":
        return BoxOptions(**box, surface=gs.surfaces.Emission(color=(1.0, 1.0, 1.0)))
    if kind == "per_env":
        per_env = [gs.textures.ColorTexture(color=(0.9, 0.2, 0.2)), gs.textures.ColorTexture(color=(0.2, 0.9, 0.2))]
        return BoxOptions(
            **box, surface=gs.surfaces.Default(diffuse_texture=gs.textures.BatchTexture(textures=per_env))
        )
    if kind == "invisible":
        return BoxOptions(**box, visualization=False)
    if kind == "translucent":
        return BoxOptions(**box, color=(0.8, 0.8, 0.8, 0.5))
    path = tmp_path / "two_tone.xml"
    path.write_text(_TWO_TONE)
    return EntityOptions(file=str(path), is_fixed_base=True)


@pytest.mark.parametrize(
    ("kind", "match"),
    [
        ("emissive", r"on entity 'extra', whose surface emits light \(Emission\)"),
        ("per_env", r"on entity 'extra', whose colour varies per env \(2 colours in a batched texture\)"),
        ("invisible", r"on entity 'extra', which has no visual geom to read a colour off"),
        ("translucent", r"on entity 'extra', which is translucent \(opacity 0\.5\)\..*Make the entity opaque"),
        (
            "two_tone",
            r"on entity 'extra', whose geoms are in 2 colours .*model the marked face as an entity of its own",
        ),
    ],
)
def test_a_surface_with_no_one_plain_reflected_colour_fails_at_build(kind, match, tmp_path):
    """Emissive, coloured per env, without visual geoms, translucent, or in several colours: none has one reflected
    colour to divide by.
    """
    config = _config(metric={"entity_name": "extra"}, cam=_paint_square())
    config.scene_options.extra = _surface(kind, tmp_path)
    with pytest.raises(ValueError, match=match):
        build_env(config)
