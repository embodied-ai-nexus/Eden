"""``eden.entities.camera``: pose access, video recording, background replacement and shadows, and the sensor backend.

Every camera here renders through the rasterizer (``debug=True`` or ``RasterizerCameraOptions``), so no batch
renderer is needed; an unrenderable machine skips. The checks that only a batch renderer, a ray tracer, a software
OpenGL context or an open viewer would reach use a stand-in for that one Genesis boundary.
"""

from __future__ import annotations

import math

import av
import genesis as gs
import numpy as np
import pytest
import torch
from genesis.options.sensors import RasterizerCameraOptions, RaytracerCameraOptions
from genesis.utils.geom import trans_quat_to_T, transform_by_quat
from PIL import Image

from eden.entities.camera import (
    Camera,
    SplatCamera,
    _apply_rasterizer_sensor_shadow_setting,
    _rasterizer_shadows_off,
    resolved_cam_fov,
)
from eden.entities.camera_overlays import BackgroundImageComposite, PlanePaintOverlay, SplatComposite
from eden.envs.base import EnvBase
from eden.options import BoxOptions, EnvOptions, PlaneOptions, SceneOptions
from eden.options.camera import CameraOptions, CamerasOptions
from eden.options.robots import FrankaEmikaPanda
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env

MAGENTA = (255, 0, 255)
RES = (64, 48)
LOOK = {"cam_pos": (1.5, 0.0, 1.0), "cam_lookat": (0.0, 0.0, 0.3), "cam_resolution": RES}


def _decode(path):
    """Decoded RGB frames and the encoded framerate of a video."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate)
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)], rate


def _magenta_fraction(frame):
    """Share of pixels within codec tolerance of magenta."""
    return float((np.abs(frame.astype(int) - np.array(MAGENTA)).max(axis=-1) < 60).mean())


def _build_or_skip(config):
    env = build_env(config)
    try:
        next(iter(env.cameras.values())).render_rgb()
    except Exception as e:  # noqa: BLE001 - an offscreen-render failure is a skip, not a test failure
        env.destroy()
        pytest.skip(f"offscreen rendering unavailable: {e}")
    return env


def _backdrop(image_path, entities, **params):
    """A ``BackgroundImageComposite`` entry: ``image_path`` replaces ``entities`` (and the empty background)."""
    return BackgroundImageComposite.configure(image_path=str(image_path), entities=entities, **params)


class _FlatSplatRenderer:
    """Stands in for ``BatchSplatRenderer``, whose gsplat CUDA rasterizer this machine may not have.

    Renders a uniform magenta splat of the requested size, the ``(n_views, H, W, 3)`` float in ``[0, 1]`` the real
    one returns, so what lands in the background is identifiable.
    """

    per_env_gaussians = False

    def render(self, cam_pos, cam_rot, height, width, fov):
        splat = torch.zeros((cam_pos.shape[0], height, width, 3))
        splat[..., 0] = splat[..., 2] = 1.0
        return splat, None


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """One env, a falling cube over a floor, and one camera per concern.

    ``cam`` is a free legacy camera, ``attached`` rides the cube, ``overlay`` replaces the floor with a magenta
    image and ``splat`` with a (stand-in) splat render, ``clipped`` sets near/far, and ``sensor`` / ``agreeing``
    are rasterizer sensors (the first with near/far from ``CameraOptions``, the second agreeing with its own).
    Scene shadows are off, which only a sensor camera needs Eden to forward.
    """
    background = tmp_path_factory.mktemp("camera") / "background.png"
    Image.new("RGB", (32, 24), MAGENTA).save(background)
    config = EdenConfig(
        env_options=EnvOptions(num_envs=1, sim_dt=0.005, decimation=4, rasterizer_shadows=False),
        scene_options=SceneOptions(
            floor=PlaneOptions(), cube=BoxOptions(size=(0.2, 0.2, 0.2), default_root_pos=(0.0, 0.0, 0.5))
        ),
        cameras_options=CamerasOptions(
            cam=CameraOptions(**LOOK, debug=True),
            attached=CameraOptions(
                **LOOK,
                debug=True,
                attach_entity_name="cube",
                attach_link_name="box_baselink",
                attach_offset=(0.5, 0.0, 0.5),
                attach_quat=(1.0, 0.0, 0.0, 0.0),
            ),
            overlay=CameraOptions(**LOOK, debug=True, overlays=[_backdrop(background, ["floor"])]),
            clipped=CameraOptions(**LOOK, debug=True, cam_near=0.1, cam_far=7.0),
            sensor=CameraOptions(**LOOK, cam_near=0.1, cam_far=6.0, sensor_options=RasterizerCameraOptions()),
            agreeing=CameraOptions(**LOOK, cam_near=0.5, sensor_options=RasterizerCameraOptions(near=0.5)),
        ),
    )
    env = EnvBase.from_config(config, show_viewer=False)
    # The splat camera is what a SplatEnvBase would build for this options entry; its renderer is the boundary.
    # Not in the config: a plain env's camera refuses a SplatComposite.
    splat = CameraOptions(**LOOK, debug=True, overlays=[SplatComposite.configure(entities=["floor"])])
    env.cameras["splat"] = SplatCamera("splat", env, splat)
    env.cameras["splat_default"] = SplatCamera("splat_default", env, CameraOptions(**LOOK, debug=True))
    env.build()
    env._splat_renderer = _FlatSplatRenderer()
    env._refresh_splat_gaussians = lambda: None
    try:
        env.cameras["cam"].render_rgb()
    except Exception as e:  # noqa: BLE001 - an offscreen-render failure is a skip, not a test failure
        env.destroy()
        pytest.skip(f"offscreen rendering unavailable: {e}")
    yield env
    env.destroy()


@pytest.fixture(autouse=True)
def _stop_dangling_recordings(request):
    """Close a recording a failing test left open, so the next one does not fail on "Already recording"."""
    yield
    for name in ("env", "batched"):
        if name in request.fixturenames:
            for camera in request.getfixturevalue(name).cameras.values():
                if camera._recording is not None:
                    camera.stop_recording()


# ============================================ pose ============================================


def test_is_attached_reports_the_mount(env):
    assert env.cameras["cam"].is_attached is False
    assert env.cameras["attached"].is_attached is True


def test_free_camera_starts_where_configured_and_moves_when_set(env):
    cam = env.cameras["cam"]
    torch.testing.assert_close(cam.get_pos().reshape(3).cpu(), torch.tensor(LOOK["cam_pos"]), atol=1e-4, rtol=0)
    try:
        cam.set_pos(torch.tensor([5.0, 5.0, 5.0], device=env.device))
        torch.testing.assert_close(cam.get_pos().reshape(3).cpu(), torch.tensor([5.0, 5.0, 5.0]), atol=1e-4, rtol=0)
    finally:
        cam.set_pos(torch.tensor(LOOK["cam_pos"], device=env.device))


def test_set_quat_turns_the_camera_in_place(env):
    cam = env.cameras["cam"]
    pos, quat = cam.get_pos().clone(), cam.get_quat().clone()
    yaw90 = torch.tensor([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)], device=env.device)
    try:
        cam.set_quat(yaw90)
        got = cam.get_quat().reshape(4).float()
        assert float(torch.abs(torch.dot(got, yaw90))) > 0.999  # q and -q are one rotation
        torch.testing.assert_close(cam.get_pos(), pos)
    finally:
        cam.set_quat(quat)


def test_attached_camera_rides_its_link_with_the_offset(env):
    cam, cube = env.cameras["attached"], env.entities["cube"]
    env.step()
    cam.render_rgb()  # moves an attached camera to its link
    link = cube.links[0]
    link_pos, link_quat = link.get_pos(relative=False).reshape(3), link.get_quat(relative=False).reshape(4)
    expected = link_pos + transform_by_quat(torch.tensor([0.5, 0.0, 0.5], device=env.device), link_quat)
    torch.testing.assert_close(cam.get_pos().reshape(3), expected, atol=1e-4, rtol=0)


@pytest.mark.parametrize(
    "setter",
    [
        lambda cam: cam.set_pos(torch.zeros(3)),
        lambda cam: cam.set_quat(torch.tensor([1.0, 0.0, 0.0, 0.0])),
        lambda cam: cam.set_pose(pos=torch.zeros(3)),
    ],
    ids=["set_pos", "set_quat", "set_pose"],
)
def test_an_attached_camera_refuses_to_be_posed(env, setter):
    with pytest.raises(RuntimeError, match="attached"):
        setter(env.cameras["attached"])


# ======================================= clipping planes =======================================


def test_clipping_planes_reach_the_built_camera(env):
    """``cam_near``/``cam_far`` reach the Genesis camera or sensor; left unset, Genesis' defaults stand."""
    # The resolved planes are not mirrored on Eden's Camera; the Genesis objects are where they are read.
    clipped, default = env.cameras["clipped"]._cam, env.cameras["cam"]._cam
    assert (clipped.near, clipped.far) == (0.1, 7.0)
    assert (default.near, default.far) != (0.1, 7.0)
    sensor = env.cameras["sensor"]._sensor._options
    assert (sensor.near, sensor.far) == (0.1, 6.0)
    assert env.cameras["agreeing"]._sensor._options.near == 0.5


@pytest.mark.parametrize(
    ("camera", "match"),
    [
        # A lone cam_near slips past config validation (cam_far is None) yet exceeds Genesis' default far.
        (CameraOptions(**LOOK, cam_near=1000.0, debug=True), "inverted"),
        (CameraOptions(**LOOK, cam_near=0.1, sensor_options=RasterizerCameraOptions(near=0.5)), "conflicts"),
        (CameraOptions(**LOOK, cam_near=0.1, sensor_options=RaytracerCameraOptions()), "has no 'near' clipping plane"),
    ],
    ids=["legacy_inverted", "sensor_conflict", "sensor_without_planes"],
)
def test_clipping_planes_that_cannot_be_honoured_fail_at_build(camera, match):
    """A near plane is part of a trained depth observation, so it is never silently dropped or overridden."""
    env = EnvBase.from_config(
        EdenConfig(
            env_options=EnvOptions(num_envs=1),
            scene_options=SceneOptions(box=BoxOptions(size=(0.2, 0.2, 0.2))),
            cameras_options=CamerasOptions(cam=camera),
        ),
        show_viewer=False,
    )
    try:
        with pytest.raises(ValueError, match=match):
            env.build()
    finally:
        env.destroy()


# ========================================== recording ==========================================


def test_records_one_frame_per_rgb_render(env, tmp_path):
    """Recording follows the rendered RGB image, not the call: a depth render is no frame."""
    out, cam = tmp_path / "stepped.mp4", env.cameras["cam"]
    cam.start_recording(str(out), fps=25)
    for _ in range(6):
        env.step()
        cam.render_rgb()
    cam.render_depth()
    cam.stop_recording()

    frames, rate = _decode(out)
    assert (len(frames), rate) == (6, 25.0)
    assert frames[0].shape == (RES[1], RES[0], 3)


def test_records_a_replay_that_never_steps_the_scene(env, tmp_path):
    """A kinematic replay sets state and renders without ``scene.step()``, and records as a rollout does.

    Genesis' own camera recording captures inside ``scene.step()``, so it would record nothing here.
    """
    out, cam, cube = tmp_path / "replay.mp4", env.cameras["cam"], env.entities["cube"]
    cam.start_recording(str(out), fps=50)
    for i in range(4):
        cube.set_pos([[0.0, 0.0, 0.5 + 0.05 * i]])
        env.scene.visualizer.update()
        cam.render_rgb()
    cam.stop_recording()

    frames, rate = _decode(out)
    assert (len(frames), rate) == (4, 50.0)


def test_restarting_a_recording_writes_a_second_independent_video(env, tmp_path):
    cam = env.cameras["cam"]
    for name, n_renders in (("first", 2), ("second", 3)):
        cam.start_recording(str(tmp_path / f"{name}.mp4"))
        for _ in range(n_renders):
            cam.render_rgb()
        cam.stop_recording()

    assert len(_decode(tmp_path / "first.mp4")[0]) == 2
    assert len(_decode(tmp_path / "second.mp4")[0]) == 3


def test_a_recording_that_renders_nothing_warns_and_writes_no_file(env, tmp_path, eden_caplog):
    """Silence would hide a calling loop that never renders, the failure this API exists to catch."""
    out, cam = tmp_path / "never_rendered.mp4", env.cameras["cam"]
    cam.start_recording(str(out))
    cam.stop_recording()

    assert not out.exists()
    assert "captured no frame" in eden_caplog.text


def test_a_sensor_camera_records_its_frame_not_the_cache_the_next_step_overwrites(env, tmp_path):
    """A sensor read is a view into a cache the next ``step()`` repopulates, and the encoder runs on its own thread.

    ``render_rgb`` must hand the recording a copy. Overwriting the returned view in place is exactly what the next
    step does to it, so the video has to show the frame as rendered, not the overwrite.
    """
    out, cam = tmp_path / "sensor.mp4", env.cameras["sensor"]
    cam.start_recording(str(out))
    view = cam.render_rgb()
    rendered = view[0].cpu().numpy().copy()
    view[:] = 0
    cam.stop_recording()

    (frame,), _ = _decode(out)
    assert rendered.mean() > 20.0, "test premise: the sensor camera sees something"
    assert np.abs(frame.astype(int) - rendered.astype(int)).mean() < 10.0


@pytest.mark.parametrize("name", ["overlay", "splat"])
def test_a_composited_frame_is_what_is_recorded(env, tmp_path, name):
    """The video holds the frame with the background replaced, not the raw pass it was built from.

    Magenta cannot come from the scene itself, so its presence tells the composite from the raw render.
    """
    cam = env.cameras[name]
    raw, _, _, _ = cam.render()
    assert _magenta_fraction(np.asarray(raw)) == 0.0

    out = tmp_path / f"{name}.mp4"
    cam.start_recording(str(out))
    composited = cam.render_rgb()
    cam.stop_recording()

    assert _magenta_fraction(np.asarray(composited)) > 0.2
    (frame,), _ = _decode(out)
    assert np.abs(frame.astype(int) - np.asarray(composited).astype(int)).mean() < 10.0


def test_a_rejected_framerate_leaves_nothing_armed(env, tmp_path):
    """A bad framerate is rejected at the call, not by PyAV on the first frame of the render loop."""
    cam = env.cameras["cam"]
    for fps in (0.0, -30.0):
        with pytest.raises(ValueError, match="must be positive"):
            cam.start_recording(str(tmp_path / "bad_fps.mp4"), fps=fps)
    cam.start_recording(str(tmp_path / "good.mp4"))
    cam.stop_recording()


def test_recording_calls_out_of_order_raise(env, tmp_path):
    cam = env.cameras["cam"]
    with pytest.raises(RuntimeError, match="Recording not started"):
        cam.stop_recording()
    cam.start_recording(str(tmp_path / "busy.mp4"))
    try:
        with pytest.raises(RuntimeError, match="Already recording"):
            cam.start_recording(str(tmp_path / "other.mp4"))
    finally:
        cam.stop_recording()


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")  # the failing encoder thread
def test_teardown_finalizes_recordings_and_a_failed_video_does_not_cost_the_scene(tmp_path, eden_caplog):
    """``env.destroy()`` leaves a playable video for an unstopped recording, and destroys the scene regardless.

    The second camera records into a path whose parent is a regular file, so its encoder fails. The scene still
    goes: it holds SNode trees the process cannot reclaim.
    """
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("")
    env = _build_or_skip(
        EdenConfig(
            env_options=EnvOptions(num_envs=1),
            scene_options=SceneOptions(cube=BoxOptions(size=(0.2, 0.2, 0.2), default_root_pos=(0.0, 0.0, 0.5))),
            cameras_options=CamerasOptions(
                good=CameraOptions(**LOOK, debug=True), bad=CameraOptions(**LOOK, debug=True)
            ),
        )
    )
    good, bad = env.cameras["good"], env.cameras["bad"]
    good.start_recording(str(tmp_path / "torn_down.mp4"), fps=25)
    bad.start_recording(str(blocker / "lost.mp4"))
    bad.render_rgb()  # once: the encoder fails on its thread, and any later write would raise right here
    for _ in range(2):
        good.render_rgb()

    env.destroy()  # no stop_recording()

    frames, rate = _decode(tmp_path / "torn_down.mp4")
    assert (len(frames), rate) == (2, 25.0)
    assert "lost.mp4" in eden_caplog.text
    assert env.scene._sim is None  # Genesis' own marker of a destroyed scene


# ===================================== background replacement =====================================


def test_entity_level_mask_is_the_renderer_background_plus_the_entity(env):
    """At the default ``segmentation_level="entity"`` an entity owns one id; id 0 is the renderer's background."""
    floor_ids = env.entity_segmentation_ids[env.entities["floor"].idx]
    assert len(floor_ids) == 1
    assert env.cameras["overlay"]._overlays[0].background_ids == [0, *floor_ids]


@pytest.mark.parametrize("frame_type", ["torch", "numpy"])
def test_the_composite_replaces_the_background_pixels_only(env, frame_type):
    """A torch frame and segmentation are what the batch renderer returns; this machine's rasterizer returns numpy.

    Either is composited the same and comes back in its own array type.
    """
    cam = env.cameras["overlay"]
    background_id = cam._overlays[0].background_ids[1]
    segm = np.full(RES[::-1], background_id + 1000)
    segm[:, : RES[0] // 2] = background_id  # the left half is background, the right half is not
    rgb = np.zeros((*RES[::-1], 3), dtype=np.uint8)
    if frame_type == "torch":
        segm, rgb = torch.from_numpy(segm), torch.from_numpy(rgb)

    out = cam._paint_overlays(rgb, segm)

    assert isinstance(out, type(rgb))
    out = np.asarray(out)
    assert (out[:, : RES[0] // 2] == MAGENTA).all() and (out[:, RES[0] // 2 :] == 0).all()


def test_a_splat_camera_shows_the_splat_only_when_it_declares_one(env):
    """Nothing is composited that the config does not declare: a splat config's recording camera declares it."""
    cam = env.cameras["splat_default"]
    assert cam.splat_single_view is True and cam._overlays == []
    np.testing.assert_array_equal(np.asarray(cam.render_rgb()), cam.render()[0])


def test_a_camera_built_outside_an_env_is_named_by_its_uid():
    camera = Camera(None, CameraOptions())
    assert camera.display_name == f"(uid {camera._uid})"
    assert Camera(None, CameraOptions(), key="wrist").display_name == "'wrist'"


def test_a_splat_composite_on_a_sensor_camera_is_refused_at_construction(env):
    """The splat is composited by segmentation, which a sensor camera cannot render."""
    sensor = CameraOptions(**LOOK, sensor_options=RasterizerCameraOptions(), overlays=[SplatComposite.configure()])
    with pytest.raises(ValueError, match="SplatComposite paints by the frame's segmentation.*Remove sensor_options"):
        SplatCamera("bad", env, sensor)


def test_background_entity_not_in_the_scene_fails_at_build(tmp_path):
    """A name that cannot be masked is a config error, raised at build naming the entity."""
    Image.new("RGB", (32, 24), MAGENTA).save(tmp_path / "backdrop.png")
    env = EnvBase.from_config(
        EdenConfig(
            env_options=EnvOptions(num_envs=1),
            scene_options=SceneOptions(box=BoxOptions(size=(0.2, 0.2, 0.2))),
            cameras_options=CamerasOptions(
                cam=CameraOptions(**LOOK, debug=True, overlays=[_backdrop(tmp_path / "backdrop.png", ["plane"])])
            ),
        ),
        show_viewer=False,
    )
    try:
        with pytest.raises(
            KeyError, match=r"No scene entity named 'plane'.*BackgroundImageComposite\(entities=\.\.\.\)"
        ):
            env.build()
    finally:
        env.destroy()


# ====================================== background shadows ======================================

GREY = (128, 128, 128)
# Looking down at the cube from above and to the side, so its shadow on the floor is in view.
SHADOW_LOOK = {"cam_pos": (1.2, 0.0, 2.0), "cam_lookat": (0.0, 0.0, 0.0), "cam_resolution": RES}


@pytest.fixture(scope="module")
def shadowed(tmp_path_factory):
    """Two envs rendered separately, a cube floating over a floor, scene shadows on.

    The scene is never stepped, so the cube stays where it is put. Env 0's casts a shadow on the floor; env 1's is
    moved out of view, so its floor is evenly lit. Each camera
    replaces the floor, with a grey image (``image``) or a magenta (stand-in) splat (``splat``), at strength 1, and
    ``image_off`` / ``splat_off`` do the same at the default strength 0. ``cube_image`` replaces the untextured cube
    instead, for the batch renderer's texture check.
    """
    grey = tmp_path_factory.mktemp("shadows") / "grey.png"
    Image.new("RGB", (32, 24), GREY).save(grey)
    floor = {"entities": ["floor"]}
    # Not in the config: a plain env's camera refuses a SplatComposite.
    splats = {
        "splat": CameraOptions(
            **SHADOW_LOOK, debug=True, overlays=[SplatComposite.configure(**floor, shadow_strength=1.0)]
        ),
        "splat_off": CameraOptions(**SHADOW_LOOK, debug=True, overlays=[SplatComposite.configure(**floor)]),
    }
    config = EdenConfig(
        env_options=EnvOptions(num_envs=2, sim_dt=0.005, env_separate_rigid=True),
        scene_options=SceneOptions(
            floor=PlaneOptions(),
            cube=BoxOptions(size=(0.3, 0.3, 0.3), default_root_pos=(0.0, 0.0, 0.4)),
        ),
        cameras_options=CamerasOptions(
            image=CameraOptions(**SHADOW_LOOK, debug=True, overlays=[_backdrop(grey, ["floor"], shadow_strength=1.0)]),
            image_off=CameraOptions(**SHADOW_LOOK, debug=True, overlays=[_backdrop(grey, ["floor"])]),
            cube_image=CameraOptions(
                **SHADOW_LOOK, debug=True, overlays=[_backdrop(grey, ["cube"], shadow_strength=1.0)]
            ),
        ),
    )
    env = EnvBase.from_config(config, show_viewer=False)
    for name in ("splat", "splat_off"):
        env.cameras[name] = SplatCamera(name, env, splats[name])
    try:
        env.build()
    except ValueError as e:
        env.destroy()
        if "software OpenGL renderer" in str(e):  # it draws no shadows, and the camera rightly refuses to build
            pytest.skip(f"background shadows need a GPU-backed OpenGL context: {e}")
        raise
    env._splat_renderer = _FlatSplatRenderer()
    env._refresh_splat_gaussians = lambda: None
    env.entities["cube"].set_pos(torch.tensor([[0.0, 0.0, 0.4], [0.0, 20.0, 0.4]], device=env.device))
    env.scene.visualizer.update()
    try:
        env.cameras["image_off"].render_rgb()
    except Exception as e:  # noqa: BLE001 - an offscreen-render failure is a skip, not a test failure
        env.destroy()
        pytest.skip(f"offscreen rendering unavailable: {e}")
    yield env
    env.destroy()


def _floor(env, segm):
    """Where the floor is in a segmentation."""
    return np.isin(segm, env.entity_segmentation_ids[env.entities["floor"].idx])


@pytest.mark.parametrize(("name", "colour"), [("image", GREY), ("splat", MAGENTA)])
def test_the_sims_shadow_darkens_only_the_background_pixels_it_falls_on(shadowed, name, colour):
    """Every floor pixel the sim shadows is darkened, no other pixel changes, and env 1 (no shadow) is untouched."""
    cam = shadowed.cameras[name]
    out = np.asarray(cam.render_rgb()).astype(int)
    raw, _, segm, _ = cam.render(rgb=True, segmentation=True)
    raw = raw.astype(int)
    # The shadow-free pass has no public form: the camera renders it only to shade the composite.
    unshadowed = cam.render_unshadowed_rgb().cpu().numpy().astype(int)
    floor = _floor(shadowed, segm)
    in_shadow = floor & ((unshadowed - raw).max(-1) > 0)
    clearly_in_shadow = floor & (raw.sum(-1) < 0.98 * unshadowed.sum(-1))  # so rounding cannot hide the darkening
    darkened = floor & (out != colour).any(-1)

    assert clearly_in_shadow[0].any() and not in_shadow[1].any(), "test premise: only env 0's floor is shadowed"
    assert not (darkened & ~in_shadow).any()
    assert (darkened | ~clearly_in_shadow).all()
    assert out[0][clearly_in_shadow[0]].max(-1).min() < max(colour) - 30
    np.testing.assert_array_equal(out[~floor], raw[~floor])


@pytest.mark.parametrize(("name", "colour"), [("image_off", GREY), ("splat_off", MAGENTA)])
def test_strength_zero_pastes_the_background_as_is(shadowed, name, colour):
    """The default strength is the composite from before the option existed, even over env 0's shadow."""
    cam = shadowed.cameras[name]
    out = np.asarray(cam.render_rgb())
    raw, _, segm, _ = cam.render(rgb=True, segmentation=True)
    floor = _floor(shadowed, segm)

    assert (out[floor] == colour).all()
    np.testing.assert_array_equal(out[~floor], raw[~floor])


def test_the_shadow_free_pass_leaves_the_scene_rendering_shadows(shadowed):
    """The composite turns scene shadows off for one render; every render after it must have them again."""
    cam = shadowed.cameras["image"]
    before = cam.render()[0].copy()
    cam.render_rgb()
    np.testing.assert_array_equal(cam.render()[0], before)


def test_torch_frames_are_shaded_like_numpy_frames(shadowed, monkeypatch):
    """The batch renderer's path: torch frames and no shadow-free pass; this machine's rasterizer returns numpy."""
    cam = shadowed.cameras["image"]
    raw, _, segm, _ = cam.render(rgb=True, segmentation=True)
    monkeypatch.setattr(cam, "render_unshadowed_rgb", lambda: None)  # the batch renderer can't render one

    from_numpy = cam._paint_overlays(raw, segm)
    from_torch = cam._paint_overlays(torch.from_numpy(raw.copy()), torch.from_numpy(segm.copy()))

    assert (from_numpy[_floor(shadowed, segm)] != GREY).any(), "test premise: the estimate shaded something"
    assert isinstance(from_torch, torch.Tensor)
    np.testing.assert_array_equal(from_torch.numpy(), from_numpy)


def test_a_shadow_strength_with_no_shadows_to_carry_fails_at_build(tmp_path):
    Image.new("RGB", (32, 24), GREY).save(tmp_path / "grey.png")
    env = EnvBase.from_config(
        EdenConfig(
            env_options=EnvOptions(num_envs=1, rasterizer_shadows=False),
            scene_options=SceneOptions(floor=PlaneOptions(), box=BoxOptions(size=(0.2, 0.2, 0.2))),
            cameras_options=CamerasOptions(
                cam=CameraOptions(
                    **SHADOW_LOOK,
                    debug=True,
                    overlays=[_backdrop(tmp_path / "grey.png", ["floor"], shadow_strength=0.5)],
                )
            ),
        ),
        show_viewer=False,
    )
    try:
        with pytest.raises(ValueError, match="rasterizer_shadows=True"):
            env.build()
    finally:
        env.destroy()


def test_a_camera_has_one_background_and_a_splat_needs_a_splat_env(tmp_path):
    """Each refusal names the camera by its key in the env's cameras."""
    Image.new("RGB", (32, 24), GREY).save(tmp_path / "grey.png")
    backdrop = _backdrop(tmp_path / "grey.png", ["plane"])
    for overlays, match in [
        (
            [backdrop, _backdrop(tmp_path / "grey.png", ["box"])],
            r"Camera 'cam': .*declares 2 background composites \(\['BackgroundImageComposite', "
            r"'BackgroundImageComposite'\]\)",
        ),
        (
            [SplatComposite.configure()],
            "SplatComposite renders the env's Gaussian splat, but camera 'cam' is not a splat env's camera",
        ),
    ]:
        config = EdenConfig(
            env_options=EnvOptions(num_envs=1),
            scene_options=SceneOptions(plane=PlaneOptions(), box=BoxOptions(size=(0.2, 0.2, 0.2))),
            cameras_options=CamerasOptions(cam=CameraOptions(**SHADOW_LOOK, debug=True, overlays=overlays)),
        )
        with pytest.raises(ValueError, match=match):
            EnvBase.from_config(config, show_viewer=False)


class _BatchRendererLights:
    """Stands in for Genesis' ``BatchRenderer``, which needs gs-madrona and a GPU the suite does not build on.

    It carries only the lights, the one thing the camera's build check reads from it.
    """

    def __init__(self, *castshadow):
        self._lights = [type("Light", (), {"castshadow": flag})() for flag in castshadow]


@pytest.mark.parametrize(
    ("boundary", "match"),
    [
        ("software_gl", "software OpenGL renderer, which draws no shadows"),
        ("batch_renderer_without_shadow_light", "no batch-renderer light casts shadows"),
        ("ray_tracer", "comes from the ray tracer"),
    ],
)
def test_a_renderer_that_draws_no_shadows_to_carry_is_refused(shadowed, monkeypatch, boundary, match):
    """None of these can be built on the suite's machine, so the one Genesis boundary each needs is stood in for."""
    cam = shadowed.cameras["image"]
    if boundary == "software_gl":  # pyrender drops shadows on llvmpipe whatever the scene asks for
        monkeypatch.setattr(type(shadowed.scene.visualizer), "is_software", property(lambda self: True))
    elif boundary == "batch_renderer_without_shadow_light":
        monkeypatch.setattr(cam._cam, "_batch_renderer", _BatchRendererLights(False, False))
    else:
        monkeypatch.setattr(cam._cam, "_raytracer", object())
    with pytest.raises(ValueError, match=match):
        cam._overlays[0].build()


@pytest.mark.parametrize(("name", "refused"), [("image", True), ("cube_image", False)])
def test_the_batch_renderer_refuses_an_image_textured_background(shadowed, monkeypatch, name, refused):
    """Its estimate would print a texture's dark texels onto the background: Genesis' default plane is a checkerboard.

    ``cube_image`` replaces the plain-coloured cube instead, which the estimate handles exactly.
    """
    cam = shadowed.cameras[name]
    monkeypatch.setattr(cam._cam, "_batch_renderer", _BatchRendererLights(True))
    if refused:
        with pytest.raises(ValueError, match=r"background entity 'floor' is image-textured.*PlaneOptions\(color="):
            cam._overlays[0].build()
    else:
        cam._overlays[0].build()


class _VisualizerWithViewer:
    """Stands in for a Genesis visualizer with its viewer window open, which the suite cannot open.

    With a viewer, the rasterizer renders through it and reads its ``render_flags["shadows"]``, not
    ``context.shadow``.
    """

    def __init__(self, viewer_shadows=True, *, flags=True):
        self.context = type("Context", (), {"shadow": True})()
        pyrender_viewer = type("PyrenderViewer", (), {})()
        if flags:
            pyrender_viewer.render_flags = {"shadows": viewer_shadows}
        self.viewer = type("Viewer", (), {"_pyrender_viewer": pyrender_viewer})()


@pytest.mark.parametrize("viewer_shadows", [True, False])
def test_the_shadow_free_pass_turns_off_an_open_viewers_shadows_and_restores_both(viewer_shadows):
    """Clearing only ``context.shadow`` shades nothing with the viewer open; each switch is restored as it was."""
    visualizer = _VisualizerWithViewer(viewer_shadows)
    flags = visualizer.viewer._pyrender_viewer.render_flags
    with pytest.raises(RuntimeError, match="render failed"), _rasterizer_shadows_off(visualizer):
        assert (visualizer.context.shadow, flags["shadows"]) == (False, False)
        raise RuntimeError("render failed")
    assert (visualizer.context.shadow, flags["shadows"]) == (True, viewer_shadows)


def test_the_shadow_free_pass_raises_when_genesis_reworks_the_viewer():
    """A flag nothing reads would leave the viewer path silently unshaded, so a reworked Genesis must raise."""
    with (
        pytest.raises(RuntimeError, match="re-point `_rasterizer_shadows_off`"),
        _rasterizer_shadows_off(_VisualizerWithViewer(flags=False)),
    ):
        pass


# ========================================= sensor backend =========================================


def test_scene_shadow_setting_reaches_a_rasterizer_sensor(env):
    """``rasterizer_shadows=False`` lands on the sensor's standalone context, which never sees the scene's options.

    Eden's default matches Genesis' (on), so only ``False`` can tell the forwarding from its absence.
    """
    context = env.cameras["sensor"]._sensor._shared_metadata.context
    assert context is not env.scene.visualizer.context
    assert context.shadow is False


@pytest.mark.parametrize("context", [None, object()], ids=["no_context", "context_without_shadow"])
def test_the_shadow_shim_raises_when_genesis_reworks_the_sensor_context(context):
    """Writing a ``shadow`` nothing reads would silently render shadows again, so a reworked Genesis must raise.

    The stand-in is shaped like the Genesis sensor after such a rework; the real one cannot be made to lack it.
    """

    class _ReworkedSensor:
        class _shared_metadata:  # noqa: N801 - mirrors the Genesis attribute name
            pass

    _ReworkedSensor._shared_metadata.context = context
    with pytest.raises(RuntimeError, match="no longer exposes"):
        _apply_rasterizer_sensor_shadow_setting(_ReworkedSensor(), shadow=False)


# ======================================= batched cameras =======================================


@pytest.fixture(scope="module")
def batched():
    """Two envs rendered separately (``env_separate_rigid``) at link-level segmentation.

    ``cam`` looks at the Franka; ``mount`` rides its hand. The cube fills much of ``cam``'s view in
    env 0 and is out of view in env 1.
    """
    env = _build_or_skip(
        EdenConfig(
            env_options=EnvOptions(num_envs=2, sim_dt=0.005, env_separate_rigid=True, segmentation_level="link"),
            scene_options=SceneOptions(
                robot=FrankaEmikaPanda(), cube=BoxOptions(size=(0.3, 0.3, 0.3), default_root_pos=(0.9, 0.0, 0.3))
            ),
            cameras_options=CamerasOptions(
                cam=CameraOptions(
                    cam_pos=(1.8, 0.0, 0.4),
                    cam_lookat=(0.6, 0.0, 0.1),
                    cam_resolution=RES,
                    debug=True,
                ),
                mount=CameraOptions(
                    attach_entity_name="robot",
                    attach_link_name="hand",
                    attach_offset=(0.5, 0.0, 0.5),
                    attach_quat=(1.0, 0.0, 0.0, 0.0),
                    cam_resolution=RES,
                ),
            ),
        )
    )
    env.entities["cube"].set_pos(torch.tensor([[0.9, 0.0, 0.3], [0.9, 20.0, 0.3]], device=env.device))
    env.scene.visualizer.update()
    yield env
    env.destroy()


def test_link_level_mask_covers_every_link_of_a_background_entity(batched, tmp_path):
    """A mask keeping one id per entity would let most of the arm bleed through a replaced background."""
    Image.new("RGB", (32, 24), GREY).save(tmp_path / "grey.png")
    cam = batched.cameras["cam"]
    composite = BackgroundImageComposite(cam, _backdrop(tmp_path / "grey.png", ["robot"]))
    composite.build()
    robot_ids = batched.entity_segmentation_ids[batched.entities["robot"].idx]
    assert len(robot_ids) > 1
    assert set(composite.background_ids) == {0, *robot_ids}


def test_batched_camera_records_one_video_per_env(batched, tmp_path):
    """A camera rendering several envs writes ``{stem}_{i}.mp4`` for each, env ``i``'s image going to video ``i``."""
    cam, out = batched.cameras["cam"], tmp_path / "batched.mp4"
    cam.start_recording(str(out), fps=30)
    rendered = np.asarray(cam.render_rgb())
    cam.render_rgb()
    cam.stop_recording()

    assert not out.exists()
    videos = [_decode(tmp_path / f"batched_{i}.mp4") for i in range(2)]
    assert [(len(frames), rate) for frames, rate in videos] == [(2, 30.0)] * 2
    assert np.abs(rendered[0].astype(int) - rendered[1].astype(int)).mean() > 5.0, "test premise: the envs differ"
    for i, (frames, _) in enumerate(videos):
        own, other = (np.abs(frames[0].astype(int) - rendered[j].astype(int)).mean() for j in (i, 1 - i))
        assert own < 5.0 < other


def test_batched_attach_offsets_give_per_env_poses(batched):
    """A ``(num_envs, 4, 4)`` mount broadcasts through Genesis' attach, as the camera-mount DR event relies on."""
    cam = batched.cameras["mount"]
    offsets = torch.tensor([[0.5, 0.0, 0.5], [0.5, 0.0, 1.5]], device=batched.device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, device=batched.device)
    cam.set_attach_transform(trans_quat_to_T(offsets, quats))
    batched.step()
    cam.render_rgb()  # moves an attached camera to its link
    pos = cam.get_pos()
    # The two mounts differ by 1 m along the hand's z; the composed poses must too, wherever that axis points.
    assert float((pos[1] - pos[0]).norm()) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("envs_idx_form", ["int", "slice", "list", "index_tensor", "bool_mask"])
def test_set_quat_turns_only_the_selected_env_whatever_the_index_form(batched, envs_idx_form):
    """Genesis keeps the camera pose in float64 numpy; every ``envs_idx`` form must still select env 1 alone."""
    cam = batched.cameras["cam"]
    envs_idx = {
        "int": 1,
        "slice": slice(1, 2),
        "list": [1],
        "index_tensor": torch.tensor([1], device=batched.device),
        "bool_mask": torch.tensor([False, True], device=batched.device),
    }[envs_idx_form]
    pos, quat = cam.get_pos().clone(), cam.get_quat().clone()
    yaw90 = torch.tensor([[math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]], device=batched.device)
    try:
        cam.set_quat(yaw90, envs_idx=envs_idx)
        got = cam.get_quat().float()
        assert float(torch.abs(torch.dot(got[1], yaw90[0]))) > 0.999  # q and -q are one rotation
        torch.testing.assert_close(got[0], quat[0].float())
        torch.testing.assert_close(cam.get_pos(), pos)
    finally:
        cam.set_quat(quat)


@pytest.mark.parametrize(
    ("envs_idx", "match"),
    [(5, "index 5, out of range for num_envs=2"), ("cam", "Expecting integer indices for `envs_idx`")],
    ids=["out_of_range", "not_an_index"],
)
def test_set_quat_names_a_bad_envs_idx_before_touching_the_pose(batched, envs_idx, match):
    cam = batched.cameras["cam"]
    quat = cam.get_quat().clone()
    with pytest.raises((IndexError, gs.GenesisException), match=match):
        cam.set_quat(torch.tensor([1.0, 0.0, 0.0, 0.0]), envs_idx=envs_idx)
    torch.testing.assert_close(cam.get_quat(), quat)


@pytest.mark.parametrize(
    ("offset_T", "match"),
    [
        (trans_quat_to_T(torch.zeros(3, 3), torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3)), "batch size 3"),
        (torch.zeros(3, 3), "offset_T must be"),
    ],
    ids=["wrong_batch_size", "bad_shape"],
)
def test_malformed_attach_transform_fails_fast(batched, offset_T, match):
    with pytest.raises(ValueError, match=match):
        batched.cameras["mount"].set_attach_transform(offset_T.to(batched.device))


def test_a_camera_copies_its_mutable_options():
    """Mutating the camera's list must not reach the options a config holds (no scene needed)."""
    options = CameraOptions(overlays=[PlanePaintOverlay.configure(metric_name="ink", color=(0.0, 0.0, 0.0))])
    camera = Camera(None, options)
    assert camera.overlays == options.overlays and camera.overlays is not options.overlays


def test_resolved_fov_comes_from_the_intrinsic_when_one_is_given():
    """The vertical field of view a renderer bakes into a file before the camera exists must be the camera's own."""
    assert resolved_cam_fov(CameraOptions(cam_fov=42.0)) == 42.0
    intrinsic = np.array([[300.0, 0.0, 160.0], [0.0, 250.0, 120.0], [0.0, 0.0, 1.0]])
    expected = math.degrees(2 * math.atan(120.0 / 250.0))
    assert resolved_cam_fov(CameraOptions(cam_fov=42.0, cam_intrinsic=intrinsic)) == pytest.approx(expected)
