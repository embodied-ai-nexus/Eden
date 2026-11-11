"""``eden.envs.nyx_splat``: Nyx cameras get a camera profile carrying the tone mapper, the field of view and the exposure.

gs-nyx-plugin >= 0.1.7 reads optics and colour management from a profile file instead of ``NyxCameraOptions``. These
tests read back the profiles Eden writes with the Nyx SDK itself. Rendering needs CUDA and Vulkan, which the suite does
not have, so what the renderer does with a profile is outside them.
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pytest

pytest.importorskip("gs_nyx_plugin")

from gs_nyx import nyx_py_sdk as nps  # noqa: E402
from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions  # noqa: E402

from eden.envs.base import env_class_for_config  # noqa: E402
from eden.envs.nyx_splat import write_nyx_camera_profile  # noqa: E402
from eden.options import BoxOptions, EnvOptions, SceneOptions  # noqa: E402
from eden.options.camera import CameraOptions, CamerasOptions  # noqa: E402
from eden.options.splat import NyxSplatOptions  # noqa: E402
from eden.utils.configs import EdenNyxSplatConfig  # noqa: E402


@pytest.fixture(autouse=True)
def _private_temp_dir(tmp_path, monkeypatch):
    """Profiles are cached across processes in the temp dir; a test must never read one an earlier run wrote."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))


def _read(path: str) -> nps.CameraProfileAsset:
    profile = nps.CameraProfileAsset()
    assert nps.import_camera_profile_file(path, profile), path
    return profile


def _exposure(profile: nps.CameraProfileAsset) -> float:
    """The radiance-to-pixel scale of a saturation-based camera: 2^EC / (1.2 * 2^EV100)."""
    exposure = profile.sensor.exposure
    f_stops = profile.lens.aperture.apertureFStops
    ev100 = math.log2(f_stops * f_stops / exposure.shutterSpeed * 100.0 / exposure.iso)
    return 2.0**exposure.exposureCompensation / (1.2 * 2.0**ev100)


def test_a_profile_carries_the_tone_mapper_and_field_of_view():
    profile = _read(write_nyx_camera_profile(47.5, "Neutral"))
    assert profile.colorManagement.toneMapper == nps.EToneMapper.Neutral
    assert math.degrees(profile.fov_y()) == pytest.approx(47.5, abs=1e-4)


def test_a_profile_renders_at_unit_exposure_by_default():
    """The plugin's own default camera (ISO 32000, 1/24 s, f/4) shows radiance at 1/1.44; the profile undoes that."""
    assert _exposure(_read(write_nyx_camera_profile(60.0, "Off"))) == pytest.approx(1.0, rel=1e-5)


def test_exposure_compensation_is_in_ev_on_top_of_unit_exposure():
    assert _exposure(_read(write_nyx_camera_profile(60.0, "Off", exposure_compensation=1.0))) == pytest.approx(
        2.0, rel=1e-5
    )


def test_equal_settings_share_one_profile_file():
    assert write_nyx_camera_profile(60.0, "Reinhard") == write_nyx_camera_profile(60.0, "Reinhard")
    assert write_nyx_camera_profile(60.0, "Reinhard") != write_nyx_camera_profile(60.0, "Neutral")


def test_profiles_are_cached_per_user():
    """A shared node's temp dir is shared by users, and a folder one of them created is not writable by the next."""
    assert os.path.basename(os.path.dirname(write_nyx_camera_profile(60.0, "Off"))).endswith(f"_{os.getuid()}")


def test_a_gs_nyx_upgrade_does_not_reuse_an_older_sdks_profile(monkeypatch):
    """The cache is keyed by the inputs, not the bytes; the SDK's defaults and file format change with its version."""
    import importlib.metadata

    before = write_nyx_camera_profile(60.0, "Off")
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "999.0.0")
    assert write_nyx_camera_profile(60.0, "Off") != before


def test_a_profile_the_sdk_fails_to_write_raises(tmp_path):
    """The SDK's export returns None and raises nothing on failure; a missing profile must not surface later."""
    (tmp_path / f"eden_nyx_camera_profiles_{os.getuid()}").mkdir(mode=0o500)
    if os.access(tmp_path / f"eden_nyx_camera_profiles_{os.getuid()}", os.W_OK):
        pytest.skip("running with permissions that ignore the read-only bit")
    with pytest.raises(RuntimeError, match="wrote no camera profile"):
        write_nyx_camera_profile(60.0, "Off")


def _construct(splat_options: NyxSplatOptions, **cameras: CameraOptions):
    config = EdenNyxSplatConfig(
        env_options=EnvOptions(num_envs=1),
        scene_options=SceneOptions(box=BoxOptions(size=(0.2, 0.2, 0.2))),
        splat_options=splat_options,
        cameras_options=CamerasOptions(**cameras),
    )
    return env_class_for_config(config).from_config(config, show_viewer=False)  # construction only; no build


def test_env_cameras_get_a_profile_with_the_fov_their_intrinsics_resolve_to():
    """The profile owns the optics, so it must carry the camera's own field of view.

    Here that is the one ``cam_intrinsic`` implies: the camera ignores ``cam_fov``, and so must the profile.
    """
    intrinsic = np.array([[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]])
    env = _construct(
        NyxSplatOptions(tone_mapper="Neutral", exposure_compensation=0.5),
        main=CameraOptions(cam_pos=(1.0, 0.0, 0.5), cam_lookat=(0.0, 0.0, 0.0), cam_fov=90.0, cam_intrinsic=intrinsic),
    )
    try:
        camera = env.cameras["main"]
        profile = _read(camera.options.sensor_options.camera_profile)
        assert math.degrees(profile.fov_y()) == pytest.approx(camera.cam_fov, abs=1e-4)
        assert camera.cam_fov == pytest.approx(math.degrees(2 * math.atan(120.0 / 300.0)))
        assert profile.colorManagement.toneMapper == nps.EToneMapper.Neutral
        assert _exposure(profile) == pytest.approx(2.0**0.5, rel=1e-5)
    finally:
        env.destroy()


def test_a_camera_with_its_own_nyx_options_gets_the_env_profile():
    """Without one the plugin falls back to Reinhard 0.53 EV darker, unlike every other camera in the env."""
    camera = CameraOptions(cam_pos=(1.0, 0.0, 0.5), cam_lookat=(0.0, 0.0, 0.0), cam_fov=40.0)
    env = _construct(
        NyxSplatOptions(tone_mapper="Neutral", exposure_compensation=0.5),
        default_fov=camera.model_copy(update={"sensor_options": NyxCameraOptions(spp=4)}),
        own_fov=camera.model_copy(update={"sensor_options": NyxCameraOptions(spp=4, fov=70.0)}),
    )
    try:
        for name, fov in (("default_fov", 40.0), ("own_fov", 70.0)):
            sensor = env.cameras[name].options.sensor_options
            profile = _read(sensor.camera_profile)
            assert profile.colorManagement.toneMapper == nps.EToneMapper.Neutral, name
            assert _exposure(profile) == pytest.approx(2.0**0.5, rel=1e-5), name
            assert math.degrees(profile.fov_y()) == pytest.approx(fov, abs=1e-4), name
            assert sensor.spp == 4, name
    finally:
        env.destroy()


def test_a_camera_that_brings_its_own_profile_keeps_it():
    own = write_nyx_camera_profile(30.0, "Reinhard")
    camera = CameraOptions(
        cam_pos=(1.0, 0.0, 0.5), cam_lookat=(0.0, 0.0, 0.0), sensor_options=NyxCameraOptions(camera_profile=own)
    )
    env = _construct(NyxSplatOptions(tone_mapper="Neutral"), main=camera)
    try:
        assert env.cameras["main"].options.sensor_options.camera_profile == own
    finally:
        env.destroy()
