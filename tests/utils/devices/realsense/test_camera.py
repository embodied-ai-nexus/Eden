"""``RealSenseCamera`` over a fake ``pyrealsense2``, installed as ``sys.modules["pyrealsense2"]`` where the camera imports it.

The fake is the hardware boundary: a set of connected devices, each offering some stream modes and
handing out frames of known pixels. The camera class runs for real on top.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from eden.utils.devices.realsense import MODES, RealSenseCamera, list_devices, parse_mode

W, H = 8, 6
DEPTH_UNITS = 0.001  # the D4xx depth scale: one z16 unit is a millimetre


class _Enum(SimpleNamespace):
    pass


class _Device:
    def __init__(self, serial: str, modes: set) -> None:
        self.serial, self.modes = serial, modes
        self.options: dict = {}

    def get_info(self, field):
        return {"name": "Intel RealSense D435I", "serial_number": self.serial, "usb_type_descriptor": "3.2"}[field]

    def supports(self, field) -> bool:
        return True

    def first_depth_sensor(self):
        return SimpleNamespace(get_depth_scale=lambda: DEPTH_UNITS)

    def first_color_sensor(self):
        return SimpleNamespace(set_option=lambda opt, value: self.options.__setitem__(opt, value))


class _Frame:
    def __init__(self, data: np.ndarray) -> None:
        self._data = data

    def get_data(self) -> np.ndarray:
        return self._data


class _Frameset:
    def __init__(self, depth: bool, aligned: bool = False) -> None:
        self.depth, self.aligned = depth, aligned

    def get_color_frame(self) -> _Frame:
        bgr = np.zeros((H, W, 3), dtype=np.uint8)
        bgr[..., 0] = 255  # pure blue in BGR
        return _Frame(bgr)

    def get_depth_frame(self) -> _Frame:
        assert self.aligned, "depth read before it was aligned to colour"
        return _Frame(np.full((H, W), 500, dtype=np.uint16))


def make_fake_rs(devices: list[_Device]):
    rs = types.ModuleType("pyrealsense2")
    rs.camera_info = _Enum(name="name", serial_number="serial_number", usb_type_descriptor="usb_type_descriptor")
    rs.stream = _Enum(color="color", depth="depth")
    rs.format = _Enum(bgr8="bgr8", z16="z16")
    rs.option = _Enum(enable_auto_exposure="enable_auto_exposure")
    rs.started = []

    class config:
        def __init__(self) -> None:
            self.device, self.streams = None, []

        def enable_device(self, serial: str) -> None:
            self.device = serial

        def enable_stream(self, stream, width, height, fmt, fps) -> None:
            self.streams.append((stream, width, height, fps))

    class pipeline:
        def __init__(self) -> None:
            self.running, self.waits, self.depth = False, 0, False

        def start(self, cfg):
            device = next(d for d in devices if d.serial == cfg.device)
            missing = [s for s in cfg.streams if s not in device.modes]
            if missing:
                raise RuntimeError(f"Couldn't resolve requests {missing}")
            self.running, self.depth = True, any(s[0] == "depth" for s in cfg.streams)
            rs.started.append((self, cfg))
            intr = SimpleNamespace(
                fx=600.0, fy=601.0, ppx=W / 2, ppy=H / 2, coeffs=[0.0] * 5, model="distortion.inverse_brown_conrady"
            )
            stream = SimpleNamespace(as_video_stream_profile=lambda: SimpleNamespace(get_intrinsics=lambda: intr))
            return SimpleNamespace(get_device=lambda: device, get_stream=lambda s: stream)

        def wait_for_frames(self, timeout_ms=5000):
            self.waits += 1
            return _Frameset(self.depth)

        def stop(self) -> None:
            assert self.running, "stopped twice"
            self.running = False

    class align:
        def __init__(self, to) -> None:
            assert to == "color"

        def process(self, frames: _Frameset) -> _Frameset:
            return _Frameset(frames.depth, aligned=True)

    rs.config, rs.pipeline, rs.align = config, pipeline, align
    rs.context = lambda: SimpleNamespace(query_devices=lambda: list(devices))
    return rs


COLOR_720 = ("color", 1280, 720, 30)
COLOR_540 = ("color", 960, 540, 60)
DEPTH_720 = ("depth", 1280, 720, 30)


@pytest.fixture
def fake_rs(monkeypatch):
    """``fake_rs(*devices)`` installs a fake SDK with those devices connected and returns it."""

    def install(*devices: _Device):
        rs = make_fake_rs(list(devices))
        monkeypatch.setitem(sys.modules, "pyrealsense2", rs)
        return rs

    return install


def test_parse_mode_reads_every_listed_mode():
    assert parse_mode("960x540@60") == (960, 540, 60)
    assert [parse_mode(m) for m in MODES]


@pytest.mark.parametrize("mode", ["960x540", "960x540@", "wide@60", ""])
def test_a_malformed_mode_names_the_format(mode):
    with pytest.raises(ValueError, match=r"is not WIDTHxHEIGHT@FPS, e.g. one of \("):
        parse_mode(mode)


def test_without_the_sdk_the_error_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyrealsense2", None)  # import raises ImportError
    with pytest.raises(ImportError, match=r"eden-world\[realsense\]"):
        RealSenseCamera()


def test_no_camera_connected_says_where_to_look(fake_rs):
    fake_rs()
    with pytest.raises(RuntimeError, match="no RealSense camera found: check `lsusb"):
        RealSenseCamera()


def test_two_cameras_without_a_serial_is_ambiguous(fake_rs):
    fake_rs(_Device("A1", {COLOR_720}), _Device("B2", {COLOR_720}))
    with pytest.raises(RuntimeError, match=r"2 RealSense cameras are connected \['A1', 'B2'\]: pass serial="):
        RealSenseCamera()


def test_an_unknown_serial_lists_the_connected_ones(fake_rs):
    fake_rs(_Device("A1", {COLOR_720}))
    with pytest.raises(RuntimeError, match=r"serial='Z9' is connected; connected: \['A1'\]"):
        RealSenseCamera(serial="Z9")


def test_opens_the_requested_camera_with_its_intrinsics_and_auto_exposure(fake_rs):
    a, b = _Device("A1", {COLOR_720}), _Device("B2", {COLOR_540})
    rs = fake_rs(a, b)
    cam = RealSenseCamera(960, 540, 60, serial="B2", warmup_frames=3)
    pipe, cfg = rs.started[-1]
    assert cfg.device == "B2" and cfg.streams == [COLOR_540]
    assert pipe.waits == 3
    assert b.options == {"enable_auto_exposure": 1} and a.options == {}
    np.testing.assert_allclose(cam.K, [[600.0, 0.0, W / 2], [0.0, 601.0, H / 2], [0.0, 0.0, 1.0]])
    assert cam.dist_coeffs.shape == (5,) and cam.distortion_model == "inverse_brown_conrady"


def test_an_unoffered_mode_names_the_modes_to_try(fake_rs):
    fake_rs(_Device("A1", {COLOR_720}))
    with pytest.raises(RuntimeError, match=r"does not offer colour 960x540@60 .*try one of \("):
        RealSenseCamera(960, 540, 60)


def test_grab_returns_bgr_and_an_rgb_view_of_it(fake_rs):
    fake_rs(_Device("A1", {COLOR_720}))
    frame = RealSenseCamera(warmup_frames=0).grab()
    assert frame.depth is None
    assert frame.bgr[0, 0].tolist() == [255, 0, 0]
    assert frame.rgb[0, 0].tolist() == [0, 0, 255]


def test_depth_comes_aligned_to_colour_in_metres(fake_rs):
    rs = fake_rs(_Device("A1", {COLOR_720, DEPTH_720}))
    frame = RealSenseCamera(depth=True, warmup_frames=0).grab()
    assert rs.started[-1][1].streams == [COLOR_720, DEPTH_720]
    assert frame.depth.shape == frame.bgr.shape[:2]
    np.testing.assert_allclose(frame.depth, 0.5)


def test_close_is_idempotent_and_the_context_manager_closes(fake_rs):
    rs = fake_rs(_Device("A1", {COLOR_720}))
    with RealSenseCamera(warmup_frames=0) as cam:
        pass
    pipe = rs.started[-1][0]
    assert not pipe.running
    cam.close()  # the fake's stop() asserts it is not called twice


def test_list_devices_reports_name_serial_and_link(fake_rs):
    fake_rs(_Device("A1", set()))
    assert list_devices() == [{"name": "Intel RealSense D435I", "serial": "A1", "usb": "3.2"}]
