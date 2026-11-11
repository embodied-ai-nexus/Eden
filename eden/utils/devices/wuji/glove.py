"""Wuji Glove teleoperation device: live hand pose + tactile, in Eden's hand frame.

The glove is a plain SDK device — it streams over USB/Ethernet to ``wuji_sdk`` running
in this process, so unlike the Meta Quest / PICO headsets there is no bridge process to
stand up. :class:`WujiGlove` connects, subscribes to the streams you ask for, and keeps
the newest frame of each so every getter is a cheap read that a sim loop can call once
per step.

What it exposes
---------------
* **hand pose** — the 21 MediaPipe landmarks (:func:`~eden.utils.devices.wuji.protocol.skeleton_to_arrays`),
  the IK solver's 21 joint angles, and the 5 fingertip poses.
* **tactile** — the 24x31 palm-and-finger pressure sheet, its zone-grouped view, the
  calibrated binary/residual contact streams, and the 3D tactile point cloud.

Poses are returned in Eden's **hand frame** (x forward along the fingers, y left, z up
out of the back of the hand) by default, so an axis means the same physical direction on
either glove — the vendor's +y flips between dorsal and ventral by side. See
:mod:`eden.utils.devices.wuji.protocol` for the basis change and the data layouts. Pass
``frame="wuji"`` to any getter for the vendor's raw wrist frame instead.

Install the SDK with the ``wuji`` extra (``uv sync --inexact --extra wuji``); it ships manylinux
wheels only, so this is Linux x86_64 / aarch64.

Example
-------
.. code-block:: python

    from eden.utils.devices.wuji import WujiGlove

    glove = WujiGlove(side="right", streams=("hand_skeleton", "tactile"))
    keypoints = glove.get_command()          # (21, 3) torch tensor, Eden hand frame
    pressure = glove.tactile()               # (24, 31), masked taxels are nan
    glove.close()
"""

from __future__ import annotations

import itertools
import threading
from functools import partial
from typing import Any, Sequence

import torch

# Top-level `import eden` is safe: devices/ is not on eden's package-init import chain,
# and `en.logger` is seeded at package import.
import eden as en
from eden.utils.devices.base import DeviceBase
from eden.utils.distribution import install_hint
from eden.utils.devices.wuji import protocol as P

#: Every glove stream this device can subscribe to. The name is both the ``wuji_sdk``
#: resource accessor and the key the latest frame is stored under.
STREAMS = (
    "hand_skeleton",
    "hand_joint_angles",
    "tip_poses",
    "emf_poses",
    "tactile",
    "tactile_zones",
    "tactile_binary",
    "tactile_residual",
    "tactile_point_cloud",
)

#: Enough for hand teleoperation with contact feedback, and nothing else — every extra
#: stream is another 120 Hz callback into Python.
DEFAULT_STREAMS = ("hand_skeleton", "hand_joint_angles", "tactile")

#: The rate-adjustable source each stream follows, so ``rate_hz`` retunes the source a
#: derived stream rides on rather than silently doing nothing. ``tactile_point_cloud``
#: rides both — its pressures follow ``tactile`` and its positions the skeleton's EMF
#: rate — and is mapped to ``tactile``, the one that sets its payload size.
STREAM_RATE_SOURCE = {
    "emf_poses": "emf_poses",
    "hand_skeleton": "emf_poses",
    "hand_joint_angles": "emf_poses",
    "tip_poses": "emf_poses",
    "tactile": "tactile",
    "tactile_zones": "tactile_zones",
    "tactile_binary": "tactile",
    "tactile_residual": "tactile",
    "tactile_point_cloud": "tactile",
}

#: Streams whose push rate the device can actually change — the sources the table above
#: points at. Everything else is derived and follows its source, raising ``WujiException``
#: on ``set_rate`` (SDK example 6).
RATE_SOURCES = tuple(dict.fromkeys(STREAM_RATE_SOURCE.values()))

#: The streams that arrive as a full 24x31 sheet, readable by :meth:`WujiGlove.tactile_taxels`.
TACTILE_SHEETS = ("tactile", "tactile_binary", "tactile_residual")

#: ``tactile_binary`` / ``tactile_residual`` need a per-user, per-serial contact model.
CALIBRATED_STREAMS = ("tactile_binary", "tactile_residual")

#: The keys :meth:`WujiGlove.state` publishes a decoded stream's tensors under, in the order
#: that stream's decoder returns them — the wire format of the JSON snapshot.
STATE_KEYS = {
    "hand_skeleton": ("keypoints", "keypoint_quat", "keypoint_confidence"),
    "tip_poses": ("fingertip_pos", "fingertip_quat", "fingertip_confidence"),
    "emf_poses": ("emf_pos", "emf_quat", "emf_confidence"),
    "hand_joint_angles": ("joint_angles", "joint_angle_confidence"),
    "tactile_point_cloud": ("tactile_points", "tactile_pressure"),
}

#: ``device_name`` is the handle the SDK registers a connection under — ``connect`` takes it
#: and ``disconnect`` resolves by it — so two :class:`WujiGlove` instances must never share
#: one. A per-instance default keeps bimanual use (one glove per hand, the case ``side=``
#: exists for) from having each ``close()`` tear down the other's link.
DEVICE_NAME_PREFIX = "eden_wuji_glove"
_device_seq = itertools.count()


def _next_device_name() -> str:
    """A connection handle no other live :class:`WujiGlove` in this process is using."""
    return f"{DEVICE_NAME_PREFIX}_{next(_device_seq)}"


_SDK_HINT = (
    "Wuji Glove and Wuji Hand 2 support need the 'wuji_sdk' package. Install it with Eden's 'wuji' extra, "
    f"{install_hint('wuji')}; or alone with `pip install wuji-sdk`. It ships manylinux wheels only, so "
    "it is Linux x86_64 / aarch64."
)


def load_sdk():
    """Import ``wuji_sdk``, raising an install hint instead of a bare ImportError."""
    try:
        import wuji_sdk
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise ImportError(_SDK_HINT) from exc
    return wuji_sdk


def _gloves_among(discovered: Sequence[Any], sdk: Any) -> list[Any]:
    """The gloves in an ``SdkManager.scan()`` result, which lists every Wuji device."""
    return [d for d in discovered if d.device_type == sdk.DeviceType.WujiGlove]


class WujiGlove(DeviceBase):
    """A connected Wuji Glove, streaming hand pose and tactile into Eden.

    Constructing the device scans for gloves, connects to one, and opens a background
    subscription per requested stream; call :meth:`close` when done (or use it as a
    context manager). Every getter returns ``None`` until that stream's first frame has
    arrived, so a teleop loop can hold its last pose instead of blocking.

    Parameters
    ----------
    sn: str | None
        Serial number of the glove to connect to. ``None`` scans and picks the only
        glove on the bus (or the one matching ``side``).
    side: str | None
        ``"left"`` or ``"right"``. Used to pick between two connected gloves, and
        checked against the glove's own ``hand_side`` either way, so a mismatch raises at
        construction rather than silently flipping the radial axis of every pose.
    device_name: str | None
        Handle this connection is registered under inside the SDK. It selects no
        particular glove, but it *does* identify the connection to ``disconnect``, so it
        must be unique per live device — ``None`` (the default) takes a fresh one from
        :func:`_next_device_name`. Pass your own only if something outside Eden has to
        resolve the same connection.
    streams: Sequence[str]
        Which of :data:`STREAMS` to subscribe to. Defaults to :data:`DEFAULT_STREAMS`.
    rate_hz: int | None
        Lower the device push rate of the rate-adjustable sources
        (:data:`RATE_SOURCES`) that ``streams`` needs. ``None`` keeps the device
        default (120 Hz). The device quantizes the request and the applied rate is
        logged.
    hand_model_path: str | None
        Custom hand URDF for the glove's online IK, applied before subscribing. Needs a
        named (non-default) SDK user; without one the SDK's built-in model is used.
    exclusive: bool
        Take the glove for this process alone. By default the SDK bridges the device so
        Wuji Studio / a recorder can read it at the same time.

    Notes
    -----
    ``tactile_binary`` and ``tactile_residual`` are produced by a contact model trained
    per SDK user and glove serial. Subscribing to them before running the vendor's
    tactile calibration yields an all-zero sheet, not an error, so this device logs a
    warning when you ask for one.
    """

    def __init__(
        self,
        *,
        sn: str | None = None,
        side: str | None = None,
        device_name: str | None = None,
        streams: Sequence[str] = DEFAULT_STREAMS,
        rate_hz: int | None = None,
        hand_model_path: str | None = None,
        exclusive: bool = False,
    ):
        super().__init__()

        if not streams:
            raise ValueError(
                f"'streams' is empty — the device would carry no data; pass at least one of {list(STREAMS)}"
            )
        unknown = [s for s in streams if s not in STREAMS]
        if unknown:
            raise ValueError(f"unknown glove stream(s) {unknown}; expected a subset of {list(STREAMS)}")
        if side is not None:
            P.check_side(side)
        if rate_hz is not None and rate_hz <= 0:
            raise ValueError(f"'rate_hz' must be positive (or None for the device default), got {rate_hz}")

        self._sdk = load_sdk()
        self._manager = self._sdk.SdkManager.instance()
        self._device_name = _next_device_name() if device_name is None else device_name
        # Seeded before the connect so teardown and __str__ can read them unconditionally,
        # including on the paths where the connect itself is what failed.
        self._sn = "?"
        self._side = "?"
        self._streams = tuple(dict.fromkeys(streams))  # de-duplicate, keep order
        self._latest: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._subscriptions: list[Any] = []
        self._rate_handles: list[Any] = []
        self._closed = False
        self._connected = False

        options = self._sdk.ConnectOptions(enable_bridge=not exclusive)
        self._glove = self._connect(sn, side, options)
        self._sn = str(self._glove.serial_number)
        self._side = P.check_side(str(self._glove.hand_side().get()).lower())
        if side is not None and self._side != side:
            self.close()  # not `disconnect`: mark the device closed so __del__ doesn't retry
            raise RuntimeError(f"glove {self._sn} reports hand_side {self._side!r}, but {side!r} was requested")

        en.logger.info(f"Wuji Glove {self._sn} connected ({self._side} hand, FW {self._glove.version().get()})")

        if hand_model_path is not None:
            self._glove.hand_model_path().set(str(hand_model_path))
            en.logger.info(f"Wuji Glove {self._sn} online-IK hand model: {self._glove.hand_model_path().get()}")

        for stream in CALIBRATED_STREAMS:
            if stream in self._streams:
                en.logger.warning(
                    f"'{stream}' comes from a contact model trained per SDK user and glove serial. "
                    "Without a tactile calibration for this pair the stream stays at zero — run the "
                    "vendor's tactile calibration (wuji-sdk examples/python/wuji_glove/7.tactile_calibration.py)."
                )

        try:
            if rate_hz is not None:
                self._set_rate(rate_hz)
            for stream in self._streams:
                resource = getattr(self._glove, stream)()
                self._subscriptions.append(resource.subscribe_with_callback(partial(self._on_frame, stream)))
        except BaseException:
            self.close()
            raise

    # ------------------------------------------------------------------
    # connection
    # ------------------------------------------------------------------

    def _open(self, sn: str, options: Any):
        """Connect to ``sn`` under this device's handle, recording that a link is now open.

        Every connect goes through here so :meth:`close` knows there is something to
        release — including when ``_connect`` raises *after* opening one, so a glove that
        drops mid-probe does not leak its connection.
        """
        glove = self._manager.connect(sn=sn, device_name=self._device_name, options=options)
        self._connected = True
        return glove

    def _scanned(self, sn: str) -> Any:
        """The scan entry for ``sn``, or None if it is not enumerated or the scan failed.

        Only ever a diagnostic: an explicit ``sn`` is the path that works when enumeration
        does not, so a scan that raises must not take the connect down with it.
        """
        try:
            return next((d for d in self._manager.scan() if d.sn == sn), None)
        except Exception as exc:  # noqa: BLE001 - any scan failure just means "no opinion"
            en.logger.debug(f"Wuji Glove: could not scan to confirm {sn} is a glove ({exc})")
            return None

    def _connect(self, sn: str | None, side: str | None, options: Any):
        """Connect to the requested glove, or raise naming every candidate on the bus."""
        if sn is not None:
            # A Wuji Hand 2 serial would otherwise connect and then die on `hand_side()`.
            # Only reject what the scan positively identifies as something else, so a glove
            # the scan does not enumerate (reached by address, say) stays connectable.
            seen = self._scanned(sn)
            if seen is not None and seen.device_type != self._sdk.DeviceType.WujiGlove:
                raise RuntimeError(f"device {sn} is a {seen.device_type}, not a Wuji Glove")
            return self._open(sn, options)

        discovered = self._manager.scan()
        found = _gloves_among(discovered, self._sdk)
        if not found:
            others = [f"{d.sn} ({d.device_type})" for d in discovered]
            raise RuntimeError(
                "no Wuji Glove found on the bus"
                + (f"; other Wuji devices seen: {others}" if others else " (no Wuji device seen at all)")
            )
        if len(found) == 1:
            return self._open(found[0].sn, options)

        if side is None:
            listed = ", ".join(f"{d.sn} (likely {P.side_from_sn(d.sn) or 'unknown side'})" for d in found)
            raise RuntimeError(f"{len(found)} Wuji Gloves found: {listed}; pass sn= or side= to pick one")

        # `hand_side` is only readable over a connection, so probe. The serial encodes the
        # side (see `side_from_sn`), so try the likely glove first — a wrong guess just
        # costs the connect it would have cost anyway, and `hand_side` still decides.
        for candidate in sorted(found, key=lambda d: P.side_from_sn(d.sn) != side):
            glove = self._open(candidate.sn, options)
            if str(glove.hand_side().get()).lower() == side:
                return glove
            self._manager.disconnect(self._device_name)
            self._connected = False
        raise RuntimeError(f"no {side} Wuji Glove among {[d.sn for d in found]}")

    def _set_rate(self, rate_hz: int) -> None:
        """Retune the rate-adjustable sources feeding ``self._streams``.

        The rate lives on the stream, not on a handle, but it resets when the last
        subscription to that stream is released — so the plain handles opened here are
        kept alive for the lifetime of the device.

        Not every firmware implements rate control at all (0.11.1 does not), and that is a
        missing optimisation, not a broken glove: refusing to stream because the device
        cannot stream *more slowly* would be the wrong trade. So this warns once, loudly,
        naming the device, and leaves every stream at the device default. Any other
        ``WujiException`` — a stream that turns out to be derived, say — surfaces in the
        same message rather than being swallowed.
        """
        # A derived stream raises on set_rate, so retune the source it rides on — which is
        # usually not itself subscribed (nothing needs raw `emf_poses` to get a skeleton).
        wanted = dict.fromkeys(STREAM_RATE_SOURCE[s] for s in self._streams)
        retuned: list[str] = []
        for stream in wanted:
            handle = getattr(self._glove, stream)().subscribe()
            try:
                applied = handle.set_rate(int(rate_hz))
            except self._sdk.WujiException as exc:
                handle.close()  # no rate to hold open; the stream stays at the device default
                # Name what actually happened: a source retuned before this one keeps its
                # new rate, so "everything is at the default" would be a false log line.
                default = ", ".join(s for s in wanted if s not in retuned)
                kept = f"; {', '.join(retuned)} did take {rate_hz} Hz" if retuned else ""
                en.logger.warning(
                    f"Wuji Glove {self._sn}: could not set '{stream}' to {rate_hz} Hz ({exc}). "
                    f"{default} stay at the device default (120 Hz){kept} — expect more "
                    "callbacks per second, nothing else."
                )
                return
            retuned.append(stream)
            self._rate_handles.append(handle)
            en.logger.info(f"Wuji Glove {self._sn}: {stream} requested {rate_hz} Hz, device applied {applied} Hz")

    def _on_frame(self, stream: str, frame: Any) -> None:
        """SDK callback (background thread): keep only the newest frame of ``stream``."""
        with self._lock:
            self._latest[stream] = frame
        callback = self._callbacks.get(stream)
        if callback is not None:
            callback(frame)

    def _take(self, stream: str) -> Any:
        """Newest frame of ``stream``, or None. Raises if the stream was not subscribed."""
        if stream not in self._streams:
            raise ValueError(
                f"'{stream}' is not subscribed on this device; construct it with "
                f"streams=(..., '{stream}') — currently {list(self._streams)}"
            )
        with self._lock:
            return self._latest.get(stream)

    # ------------------------------------------------------------------
    # identity
    # ------------------------------------------------------------------

    @property
    def serial_number(self) -> str:
        """Serial number of the connected glove."""
        return self._sn

    @property
    def side(self) -> str:
        """``"left"`` or ``"right"``, as the glove reports it."""
        return self._side

    @property
    def streams(self) -> tuple[str, ...]:
        """The streams this device subscribed to."""
        return self._streams

    @property
    def tracking(self) -> bool:
        """True once a hand-pose frame has arrived (the glove is worn and solving).

        ``emf_poses`` is left out on purpose: the raw coil poses stream whether or not the
        IK has solved a hand, so they answer "is the glove powered", not "is it tracking".
        """
        with self._lock:
            return any(s in self._latest for s in ("hand_skeleton", "hand_joint_angles", "tip_poses"))

    # ------------------------------------------------------------------
    # hand pose
    # ------------------------------------------------------------------

    def _decode(self, stream: str, decoder, *args) -> tuple[torch.Tensor, ...] | None:
        """Decode the newest frame of ``stream`` into float tensors, or None if none has landed.

        Every output is cast to float, so a decoder that returns integers — ``tactile_taxels``
        and its grid indices — must convert its own frame rather than come through here.
        """
        frame = self._take(stream)
        if frame is None:
            return None
        return tuple(torch.from_numpy(a).float() for a in decoder(frame, *args))

    def keypoints(self, frame: str = "hand") -> torch.Tensor | None:
        """``(21, 3)`` MediaPipe landmark positions in metres, or None before the first frame.

        Landmarks are in :data:`~eden.utils.devices.wuji.protocol.HAND_LANDMARKS` order,
        wrist-relative, which is exactly what ``DexRetargeter`` and the vendor's
        ``RetargetSession`` consume.
        """
        poses = self.keypoint_poses(frame)
        return None if poses is None else poses[0]

    def keypoint_poses(self, frame: str = "hand") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """``(pos (21, 3), quat (21, 4) wxyz, confidence (21,))``, or None before the first frame."""
        return self._decode("hand_skeleton", P.skeleton_to_arrays, self._side, frame)

    def joint_angle_state(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """``(angles (21,) in radians, confidence (5,))``, or None before the first frame.

        Both halves come out of one decode; :meth:`joint_angles` is the view onto the
        angles alone, which is what a retargeter reads.
        """
        return self._decode("hand_joint_angles", P.pack_joint_angles)

    def joint_angles(self) -> torch.Tensor | None:
        """The IK solver's ``(21,)`` joint angles in radians, or None before the first frame.

        Packed finger-major in :data:`~eden.utils.devices.wuji.protocol.FINGERS` order:
        5 thumb DOFs (CMC x3, MCP, IP) then 4 per finger (MCP x2, PIP, DIP). This is the
        *human* hand's DOFs, not a robot's — retarget from :meth:`keypoints` to drive a
        robot hand.
        """
        state = self.joint_angle_state()
        return None if state is None else state[0]

    def fingertips(self, frame: str = "hand") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """``(pos (5, 3), quat (5, 4) wxyz, confidence (5,))`` fingertip poses, or None.

        Thumb first (:data:`~eden.utils.devices.wuji.protocol.FINGERS` order), from the
        glove's IK->FK chain.
        """
        return self._decode("tip_poses", P.fingertips_to_arrays, self._side, frame)

    def emf_poses(self, frame: str = "hand") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """``(pos (5, 3), quat (5, 4) wxyz, confidence (5,))`` raw EMF coil poses, or None.

        The unfiltered sensor product behind the skeleton. Confidence below 0.9 is the
        vendor's threshold for "unreliable".
        """
        return self._decode("emf_poses", P.emf_poses_to_arrays, self._side, frame)

    # ------------------------------------------------------------------
    # tactile
    # ------------------------------------------------------------------

    def _grid(self, stream: str) -> torch.Tensor | None:
        """Newest frame of a sheet-shaped stream as the ``(24, 31)`` nan-masked grid, or None."""
        frame = self._take(stream)
        return None if frame is None else torch.from_numpy(P.tactile_grid(frame.data)).float()

    def tactile(self) -> torch.Tensor | None:
        """``(24, 31)`` calibrated pressure in ``[0, 1]``, or None before the first frame.

        Row-major over the palm-and-finger sheet. The cells the sheet does not populate
        are ``nan`` (see :func:`~eden.utils.devices.wuji.protocol.tactile_grid`), so reduce
        with ``torch.nanmean`` / ``nan_to_num`` rather than plain ``mean``.
        """
        return self._grid("tactile")

    def tactile_binary(self) -> torch.Tensor | None:
        """``(24, 31)`` contact mask (1 = contact, ``nan`` = masked taxel), or None.

        Needs a tactile calibration for this SDK user + glove serial; without one the
        sheet reads all zero.
        """
        return self._grid("tactile_binary")

    def tactile_residual(self) -> torch.Tensor | None:
        """``(24, 31)`` signed baseline-relative contact signal, or None.

        Positive = pressed harder than the calibrated baseline, ~0 = no contact. The
        continuous signal behind :meth:`tactile_binary`; you own the contact threshold.
        Masked taxels are ``nan``. Needs the same calibration.
        """
        return self._grid("tactile_residual")

    def tactile_taxels(self, stream: str = "tactile", flat: bool = False) -> tuple[torch.Tensor, torch.Tensor] | None:
        """``(index, value)`` for just the taxels ``stream`` actually measured, or None.

        The node-wise view: one entry per live taxel, dead ones dropped rather than
        ``nan``-masked. ``index`` is ``(N, 2)`` ``(row, col)`` into the ``24 x 31`` sheet,
        or ``(N,)`` row-major ids when ``flat=True``; ``value`` is that taxel's reading.
        ``stream`` picks which sheet to read — ``tactile``, ``tactile_binary`` or
        ``tactile_residual`` — and must be one this device subscribed to.

        See :func:`~eden.utils.devices.wuji.protocol.tactile_taxels` for why ``N`` is not
        the point cloud's point count.
        """
        if stream not in TACTILE_SHEETS:
            raise ValueError(f"'{stream}' is not a tactile sheet; expected one of {list(TACTILE_SHEETS)}")
        frame = self._take(stream)
        if frame is None:
            return None
        index, value = P.tactile_taxels(frame.data, flat)
        return torch.from_numpy(index), torch.from_numpy(value).float()

    def tactile_zones(self) -> dict[str, torch.Tensor] | None:
        """Pressure grouped by hand region — ``{"palm", "thumb", ..., "pinky"}`` — or None.

        Each value is a 1-D tensor of that zone's taxels (lengths in
        :data:`~eden.utils.devices.wuji.protocol.TACTILE_ZONE_SIZES`), padding as ``nan``.
        Cheaper to reason about than the full sheet when you only need "is the thumb
        touching something".
        """
        zones = self._take("tactile_zones")
        if zones is None:
            return None
        return {k: torch.from_numpy(v).float() for k, v in P.tactile_zones_to_dict(zones).items()}

    def tactile_points(self, frame: str = "hand") -> tuple[torch.Tensor, torch.Tensor] | None:
        """``(points (N, 3), pressure (N,))`` for the 3D tactile point cloud, or None.

        The 526 real taxels placed on the live hand skeleton, so contact can be drawn in
        the scene or fed to a contact-conditioned policy.
        """
        cloud = self._take("tactile_point_cloud")
        if cloud is None:
            return None
        points, pressure = P.decode_point_cloud(cloud)
        if P.check_frame(frame) == "hand":
            points = P.wuji_points_to_hand(points, self._side)
        return torch.from_numpy(points).float(), torch.from_numpy(pressure).float()

    # ------------------------------------------------------------------
    # DeviceBase surface
    # ------------------------------------------------------------------

    def _keypoints_or_zeros(self, frame: str) -> torch.Tensor:
        """Landmarks in ``frame``, or zeros before the first one, so the shape is always ``(21, 3)``."""
        keypoints = self.keypoints(frame)
        return torch.zeros(P.N_LANDMARKS, 3) if keypoints is None else keypoints

    def get_raw_data(self) -> torch.Tensor:
        """``(21, 3)`` landmarks in the glove's own wrist frame, untransformed.

        Zeros until the first ``hand_skeleton`` frame arrives, so the shape is stable for
        a caller that stacks readings.
        """
        return self._keypoints_or_zeros("wuji")

    def get_command(self) -> torch.Tensor:
        """``(21, 3)`` landmarks in Eden's hand frame — the teleoperation command.

        Feed straight into a retargeter (``DexRetargeter``, ``wuji_sdk.RetargetSession``)
        to drive a robot hand. Zeros before the first frame, matching :meth:`get_raw_data`.
        """
        return self._keypoints_or_zeros("hand")

    def state(self, frame: str = "hand") -> dict:
        """JSON-safe snapshot of every subscribed stream, for logging or a ZMQ bridge.

        Only the streams this device subscribed to appear. Tactile sheets are published
        as the raw flat 744-value payload with the vendor's
        :data:`~eden.utils.devices.wuji.protocol.TACTILE_INVALID` sentinel intact rather
        than the ``nan``-masked grid, because ``nan`` is not valid JSON.
        """
        P.check_frame(frame)
        state: dict[str, Any] = {
            "device": "wuji_glove",
            "sn": self._sn,
            "side": self._side,
            "frame": frame,
            "ready": self.tracking,
        }

        # Every stream that decodes to tensors, as (stream, a thunk reading it). A thunk
        # rather than a getter plus args, so the one stream with no frame of its own —
        # joint angles, already anatomical — sits in the same table as the rest.
        for stream, read in (
            ("hand_skeleton", lambda: self.keypoint_poses(frame)),
            ("tip_poses", lambda: self.fingertips(frame)),
            ("emf_poses", lambda: self.emf_poses(frame)),
            ("hand_joint_angles", self.joint_angle_state),
            ("tactile_point_cloud", lambda: self.tactile_points(frame)),
        ):
            if stream in self._streams and (tensors := read()) is not None:
                # strict: STATE_KEYS is the published wire format, so an arity drift has to
                # raise here rather than ship a snapshot quietly missing a field.
                state.update(zip(STATE_KEYS[stream], (t.tolist() for t in tensors), strict=True))

        for stream in TACTILE_SHEETS:
            if stream in self._streams and (sheet := self._take(stream)) is not None:
                state[stream] = [float(v) for v in sheet.data]

        if "tactile_zones" in self._streams and (zones := self._take("tactile_zones")) is not None:
            state["tactile_zones"] = {zone: [float(v) for v in getattr(zones, zone)] for zone in P.TACTILE_ZONE_SIZES}

        return state

    def reset(self) -> None:
        """Drop every buffered frame, so the getters report None until fresh data lands."""
        with self._lock:
            self._latest.clear()

    def close(self) -> None:
        """Close every subscription and disconnect the glove. Idempotent."""
        if self._closed:
            return
        self._closed = True
        for handle in (*self._subscriptions, *self._rate_handles):
            try:
                handle.close()
            except Exception as exc:  # pragma: no cover - teardown of a dead link
                en.logger.warning(f"Wuji Glove {self._sn}: failed to close a subscription: {exc}")
        self._subscriptions.clear()
        self._rate_handles.clear()
        if not self._connected:
            return  # the connect itself is what failed; there is no link to release
        try:
            self._manager.disconnect(self._device_name)
        except Exception as exc:  # pragma: no cover - teardown of a dead link
            en.logger.warning(f"Wuji Glove {self._sn}: failed to disconnect: {exc}")

    def __enter__(self) -> WujiGlove:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(sn={self._sn}, side={self._side})"


def scan_gloves() -> list[dict]:
    """List every Wuji Glove on the bus as ``{"sn", "address", "transport"}`` dicts.

    A connection-free probe: use it to find a serial to pass to :class:`WujiGlove`. The
    glove's side is not discoverable here — it takes a connection to read.
    """
    sdk = load_sdk()
    manager = sdk.SdkManager.instance()
    return [
        {"sn": str(d.sn), "address": str(d.address), "transport": str(d.transport_type)}
        for d in _gloves_among(manager.scan(), sdk)
    ]
