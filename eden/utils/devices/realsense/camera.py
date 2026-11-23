"""An Intel RealSense camera: the colour stream, optionally with depth aligned to it, and its factory intrinsics.

``pyrealsense2`` (Eden's ``realsense`` extra) is imported when a :class:`RealSenseCamera` opens, so
this module, :data:`MODES` and :func:`parse_mode` import without it.

Auto exposure only. On a D435i, a manual colour exposure short enough to freeze fast motion
(3-5 ms) comes back nearly black even under a desk lamp, while auto exposure at 60 fps settles near
8 ms. The sensor keeps a previous process's manual exposure until told otherwise, so opening a
camera switches auto exposure back on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Self

import numpy as np

#: Colour modes worth using: 60 fps at 960x540 for tracking fast motion, 1920x1080 for a static probe.
#: Any mode the camera offers works (``rs-enumerate-devices`` lists them); these are the ones tried.
MODES = ("960x540@60", "1280x720@30", "1920x1080@30")

#: The depth stream opened with ``depth=True`` unless ``depth_mode`` says otherwise. Its rate is capped at
#: the colour rate (``min(fps, 30)``): a D4xx depth stream at 1280x720 tops out at 30 fps.
DEFAULT_DEPTH_SIZE = (1280, 720)


def parse_mode(mode: str) -> tuple[int, int, int]:
    """``"960x540@60"`` -> ``(960, 540, 60)``."""
    try:
        size, fps = mode.split("@")
        width, height = size.split("x")
        return int(width), int(height), int(fps)
    except ValueError:
        raise ValueError(f"camera mode {mode!r} is not WIDTHxHEIGHT@FPS, e.g. one of {MODES}") from None


def _import_rs():
    try:
        import pyrealsense2 as rs
    except ImportError as e:
        raise ImportError(
            "the RealSense camera needs pyrealsense2 (Linux or Windows): pip install 'eden-world[realsense]' "
            "(in a checkout: `uv sync --inexact --extra realsense`)"
        ) from e
    return rs


def list_devices() -> list[dict[str, str]]:
    """Every connected RealSense: ``[{"name", "serial", "usb"}]`` (``usb`` is the link type, ``"3.2"`` or ``"2.1"``)."""
    rs = _import_rs()
    found = []
    for dev in rs.context().query_devices():
        info = {}
        for key, field in (("name", rs.camera_info.name), ("serial", rs.camera_info.serial_number)):
            info[key] = dev.get_info(field)
        usb = rs.camera_info.usb_type_descriptor
        info["usb"] = dev.get_info(usb) if dev.supports(usb) else ""
        found.append(info)
    return found


@dataclass
class RealSenseFrame:
    """One frame from :meth:`RealSenseCamera.grab`.

    Attributes
    ----------
    bgr : np.ndarray
        ``(H, W, 3)`` uint8, BGR (OpenCV's order).
    depth : np.ndarray | None
        ``(H, W)`` float64 depth in metres, aligned to ``bgr`` (0 where the sensor has no reading);
        ``None`` without ``depth=True``.
    stamp : float
        ``time.monotonic()`` when the frame reached this process. That is later than the exposure by
        the transfer time; the device's own timestamp is on a clock this process does not share.
    """

    bgr: np.ndarray
    depth: np.ndarray | None
    stamp: float

    @property
    def rgb(self) -> np.ndarray:
        """``bgr`` in RGB order (a copy)."""
        return np.ascontiguousarray(self.bgr[..., ::-1])


class RealSenseCamera:
    """The colour stream of one RealSense, and with ``depth=True`` its depth aligned to colour.

    Parameters
    ----------
    width, height, fps : int
        A colour mode the camera offers (``rs-enumerate-devices``); see :data:`MODES`.
    depth : bool
        Also stream depth, aligned to colour, in metres.
    depth_mode : str | None
        The depth stream's ``WIDTHxHEIGHT@FPS``; ``None``: :data:`DEFAULT_DEPTH_SIZE` at ``min(fps, 30)``.
    serial : str | None
        The camera to open (:func:`list_devices`); ``None`` opens the only one connected, and refuses
        to guess between several.
    warmup_frames : int
        Frames read and dropped at open, while auto exposure settles.
    timeout_ms : int
        How long :meth:`grab` waits for a frame before raising.

    Attributes
    ----------
    K : np.ndarray
        ``(3, 3)`` colour intrinsics, from the camera's factory calibration.
    dist_coeffs : np.ndarray
        ``(5,)`` colour distortion coefficients in ``distortion_model``. A D4xx colour stream reports
        zeros (its images come out undistorted).
    distortion_model : str
        The SDK's name for the model ``dist_coeffs`` belong to (``"brown_conrady"``, ``"inverse_brown_conrady"``, ...).
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        *,
        depth: bool = False,
        depth_mode: str | None = None,
        serial: str | None = None,
        warmup_frames: int = 15,
        timeout_ms: int = 5000,
    ) -> None:
        rs = _import_rs()
        self._rs = rs
        self._open = False
        devices = list(rs.context().query_devices())
        if not devices:
            raise RuntimeError(
                "no RealSense camera found: check `lsusb | grep -i realsense` and plug it into a USB 3 port "
                "(a USB 2 link does not offer most colour modes)."
            )
        serials = [d.get_info(rs.camera_info.serial_number) for d in devices]
        if serial is None and len(serials) > 1:
            raise RuntimeError(f"{len(serials)} RealSense cameras are connected {serials}: pass serial= to pick one.")
        if serial is not None and serial not in serials:
            raise RuntimeError(f"no RealSense with serial={serial!r} is connected; connected: {serials}")
        self.serial = serial or serials[0]
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        self.timeout_ms = int(timeout_ms)

        self._pipe = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        if depth:
            dw, dh, dfps = parse_mode(depth_mode) if depth_mode else (*DEFAULT_DEPTH_SIZE, min(self.fps, 30))
            config.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, dfps)
        try:
            profile = self._pipe.start(config)
        except RuntimeError as e:
            what = f"colour {self.width}x{self.height}@{self.fps}" + (" with depth" if depth else "")
            raise RuntimeError(
                f"RealSense {self.serial} does not offer {what} ({e}); try one of {MODES}, or list its modes "
                "with rs-enumerate-devices"
            ) from e
        self._open = True
        try:
            device = profile.get_device()
            self._align = rs.align(rs.stream.color) if depth else None
            self.depth_scale = device.first_depth_sensor().get_depth_scale() if depth else 0.0
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self.K = np.array([[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]], dtype=np.float64)
            self.dist_coeffs = np.asarray(intr.coeffs, dtype=np.float64)
            self.distortion_model = str(intr.model).rsplit(".", 1)[-1]
            device.first_color_sensor().set_option(rs.option.enable_auto_exposure, 1)
            for _ in range(warmup_frames):
                self._pipe.wait_for_frames(self.timeout_ms)
        except BaseException:
            self.close()
            raise

    def grab(self) -> RealSenseFrame:
        """The next frame; blocks up to ``timeout_ms``."""
        try:
            frames = self._pipe.wait_for_frames(self.timeout_ms)
        except RuntimeError as e:
            raise RuntimeError(
                f"RealSense {self.serial} delivered no frame within {self.timeout_ms} ms ({e}): the USB link dropped, "
                "or another process holds the camera."
            ) from e
        stamp = time.monotonic()
        if self._align is None:
            return RealSenseFrame(np.asanyarray(frames.get_color_frame().get_data()).copy(), None, stamp)
        frames = self._align.process(frames)
        bgr = np.asanyarray(frames.get_color_frame().get_data()).copy()
        depth = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float64) * self.depth_scale
        return RealSenseFrame(bgr, depth, stamp)

    def close(self) -> None:
        """Stop the stream. Idempotent."""
        if self._open:
            self._open = False
            self._pipe.stop()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
