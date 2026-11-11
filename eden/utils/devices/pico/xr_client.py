"""PICO tracking client: XRoboToolkit PC Service state -> Genesis/robot frame.

The PICO headset does not talk to Eden directly. It streams its tracking state to
the **XRoboToolkit PC Service** running on this host, and the service exposes it
through the ``xrobotoolkit_sdk`` extension module. :class:`PicoXrClient` wraps
that module and hands Eden ready-to-use state:

  * poses as 4x4 matrices in the **Genesis/robot world frame** (z up, x forward,
    y left, metres, origin at the headset's tracking origin),
  * controller buttons / triggers / grips / thumbsticks by name,
  * the PICO-only extras: paired motion trackers and, in Full-body mode, the 24
    SMPL-ordered body joints.

The service reports poses in the OpenXR basis (x right, y up, -z forward), so
every pose is conjugated into the robot basis exactly as Eden's Meta Quest
bridge does — :data:`R_ROBOT_XR` here is the same matrix as televuer's
``R_ROBOT_OPENXR`` and as XRoboToolkit's own ``R_HEADSET_TO_WORLD``.

:func:`xr_pose_to_se3` and friends are module-level and SDK-free so the frame
conversion is testable without a headset; only :class:`PicoXrClient` needs the
extension module, which it imports on construction.

Install: build ``xrobotoolkit_sdk`` from
https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind (it needs the PC
Service's ``libPXREARobotSDK.so``; ``setup_ubuntu.sh`` fetches and builds both).
The service's shared library is preloaded from its install directory when the
dynamic linker can't find it, so no ``LD_LIBRARY_PATH`` export is needed.
"""

from __future__ import annotations

import ctypes
import os
from typing import TYPE_CHECKING

import numpy as np
from genesis.utils.geom import T_to_trans_quat, trans_quat_to_T

if TYPE_CHECKING:
    import numpy.typing as npt

# OpenXR (x right, y up, -z forward) -> Genesis/robot (x forward, y left, z up).
R_ROBOT_XR = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)
T_ROBOT_XR = np.eye(4)
T_ROBOT_XR[:3, :3] = R_ROBOT_XR
T_XR_ROBOT = T_ROBOT_XR.T  # a pure rotation: the transpose is the inverse

UNTRACKED = np.zeros((4, 4))  # what the pose getters return before a device reports

# Body joints in Full-body mode, in the service's (SMPL-like) order.
BODY_JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)
N_BODY_JOINTS = len(BODY_JOINT_NAMES)
N_HAND_JOINTS = 26  # OpenXR XR_HAND_JOINT_COUNT_EXT (WebXR omits the palm, so it has 25)

_SDK_HINT = (
    "PICO teleoperation needs the 'xrobotoolkit_sdk' extension module, which is built from source "
    "against the XRoboToolkit PC Service: clone "
    "https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind and run its setup_ubuntu.sh "
    "(or setup_windows.bat) with Eden's interpreter active."
)

_PXREA_LIB = "libPXREARobotSDK.so"
# The PC Service's install layout; EDEN_PXREA_SDK_DIR prepends a custom location.
_PXREA_SEARCH_DIRS = ("/opt/apps/roboticsservice/SDK/x64",)


def _import_sdk():
    import xrobotoolkit_sdk

    return xrobotoolkit_sdk


def _preload_pxrea() -> bool:
    """``CDLL(RTLD_GLOBAL)`` the service's shared library so the pybind module links."""
    extra = os.environ.get("EDEN_PXREA_SDK_DIR")
    for directory in ((extra,) if extra else ()) + _PXREA_SEARCH_DIRS:
        path = os.path.join(directory, _PXREA_LIB)
        if os.path.exists(path):
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            return True
    return False


def load_sdk():
    """Import ``xrobotoolkit_sdk``, raising a build hint instead of a bare ImportError.

    The pybind module links against the PC Service's ``libPXREARobotSDK.so``,
    which lives outside the default loader path. When the import fails on that
    library, it is preloaded from the service's install directory and the
    import retried — no ``LD_LIBRARY_PATH`` export needed.
    """
    try:
        return _import_sdk()
    except ImportError as exc:
        if _PXREA_LIB in str(exc) and _preload_pxrea():
            try:
                return _import_sdk()
            except ImportError as exc2:  # pragma: no cover - depends on the local build
                raise ImportError(_SDK_HINT) from exc2
        raise ImportError(_SDK_HINT) from exc


def xr_pose_to_se3(pose: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Convert one service pose ``(x, y, z, qx, qy, qz, qw)`` to a robot-frame 4x4.

    Returns :data:`UNTRACKED` (all zeros) when the quaternion is degenerate, which
    is what the service reports for a device it has not seen yet. Eden-side
    consumers test ``np.abs(mat).sum() < 1e-6`` for "tracking dropped".
    """
    values = np.asarray(pose, dtype=float).reshape(-1)
    if values.shape != (7,):
        raise ValueError(f"expected a 7-element (x, y, z, qx, qy, qz, qw) pose, got shape {values.shape}")
    quat = values[[6, 3, 4, 5]]  # the service reports xyzw; Genesis is wxyz
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        return UNTRACKED.copy()
    return T_ROBOT_XR @ trans_quat_to_T(values[:3], quat / norm) @ T_XR_ROBOT


def xr_pose_to_pos_quat(pose: npt.ArrayLike) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
    """Robot-frame ``(pos (3,), wxyz quat (4,))`` for one service pose, or None if untracked."""
    mat = xr_pose_to_se3(pose)
    if np.abs(mat).sum() < 1e-6:
        return None
    pos, quat = T_to_trans_quat(mat)
    return np.asarray(pos, dtype=float), np.asarray(quat, dtype=float)


def xr_points_to_robot(points: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Rotate ``(N, 3)`` XR-frame points into the Genesis/robot frame."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"expected (N, 3) points, got shape {pts.shape}")
    return pts @ R_ROBOT_XR.T  # (R @ pts.T).T


def xr_vel6_to_robot(vel6: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Rotate a 6-vector ``(vx, vy, vz, wx, wy, wz)`` velocity/accel into the robot frame.

    Both the linear and angular halves are free vectors, so each rotates by
    :data:`R_ROBOT_XR` (a proper rotation, so angular pseudovectors transform the
    same way). Used for the motion-tracker and body-joint velocities/accelerations.
    """
    v = np.asarray(vel6, dtype=float).reshape(-1)
    if v.shape != (6,):
        raise ValueError(f"expected a 6-element (linear, angular) vector, got shape {v.shape}")
    return np.concatenate([R_ROBOT_XR @ v[:3], R_ROBOT_XR @ v[3:]])


class PicoXrClient:
    """Live PICO tracking state, converted into Eden's world frame.

    Constructing the client initialises the SDK (which connects to the PC Service
    on this host); call :meth:`close` when done. Every getter is a cheap read of
    the state the service pushes in the background, so calling them once per sim
    step is fine.
    """

    def __init__(self, tracker_roles: dict[str, str] | None = None) -> None:
        """Parameters
        ----------
            tracker_roles: optional ``{serial_number: role}`` map that labels motion
                trackers by body placement (e.g. ``{"PA123...": "waist"}``). The
                service reports trackers only by serial, so this is how a tracker
                becomes ``left_foot`` / ``waist`` / ... in :meth:`trackers`. Unmapped
                trackers fall back to their serial.
        """
        self._sdk = load_sdk()
        self._sdk.init()
        self._tracker_roles = dict(tracker_roles or {})

        sdk = self._sdk
        # Name -> getter tables. The names match XRoboToolkit's own vocabulary
        # (XrClient.get_*_by_name) so recipes written against the upstream
        # samples carry over unchanged.
        self._poses = {
            "head": sdk.get_headset_pose,
            "left_controller": sdk.get_left_controller_pose,
            "right_controller": sdk.get_right_controller_pose,
        }
        self._values = {
            "left_trigger": sdk.get_left_trigger,
            "right_trigger": sdk.get_right_trigger,
            "left_grip": sdk.get_left_grip,
            "right_grip": sdk.get_right_grip,
        }
        self._buttons = {
            "X": sdk.get_X_button,
            "Y": sdk.get_Y_button,
            "A": sdk.get_A_button,
            "B": sdk.get_B_button,
            "left_menu_button": sdk.get_left_menu_button,
            "right_menu_button": sdk.get_right_menu_button,
            "left_axis_click": sdk.get_left_axis_click,
            "right_axis_click": sdk.get_right_axis_click,
        }
        self._axes = {"left": sdk.get_left_axis, "right": sdk.get_right_axis}
        self._hand_active = {"left": sdk.get_left_hand_is_active, "right": sdk.get_right_hand_is_active}
        self._hand_states = {"left": sdk.get_left_hand_tracking_state, "right": sdk.get_right_hand_tracking_state}

    # ------------------------------------------------------------------
    # single readings
    # ------------------------------------------------------------------

    def pose(self, name: str) -> npt.NDArray[np.float64]:
        """Robot-frame 4x4 pose of ``head`` / ``left_controller`` / ``right_controller``."""
        return xr_pose_to_se3(self._get(self._poses, name, "pose")())

    def value(self, name: str) -> float:
        """Analog trigger/grip reading in ``[0, 1]`` (e.g. ``"left_grip"``)."""
        return float(self._get(self._values, name, "analog input")())

    def button(self, name: str) -> bool:
        """Button state (e.g. ``"A"``, ``"left_menu_button"``, ``"right_axis_click"``)."""
        return bool(self._get(self._buttons, name, "button")())

    def axis(self, side: str) -> tuple[float, float]:
        """Thumbstick ``(x, y)`` in ``[-1, 1]`` for ``"left"`` or ``"right"``."""
        x, y = self._get(self._axes, side, "thumbstick")()
        return float(x), float(y)

    def timestamp_ns(self) -> int:
        """Service timestamp of the latest controller/headset sample, in nanoseconds."""
        return int(self._sdk.get_time_stamp_ns())

    @property
    def tracking(self) -> bool:
        """True once the headset reports a pose (i.e. the app is connected and sending)."""
        return bool(np.abs(self.pose("head")).sum() > 1e-6)

    def hand_joints(self, side: str) -> npt.NDArray[np.float64] | None:
        """Robot-frame ``(26, 3)`` hand-joint positions, or None when hand tracking is off.

        This is the OpenXR 26-joint layout (palm included), *not* the 25-joint
        WebXR one the Meta Quest bridge publishes — index it with the OpenXR
        ``XrHandJointEXT`` order.
        """
        if not self._get(self._hand_active, side, "hand")():
            return None
        state = np.asarray(self._hand_states[side](), dtype=float)
        if state.shape != (N_HAND_JOINTS, 7):
            raise ValueError(f"expected ({N_HAND_JOINTS}, 7) hand joint poses from the service, got {state.shape}")
        return xr_points_to_robot(state[:, :3])

    def trackers(self) -> dict[str, dict]:
        """PICO motion trackers (up to 5) in the robot frame, keyed by role or serial.

        Each value is ``{"serial", "role", "pos" (3), "quat" (4), "vel" (6),
        "acc" (6)}``; ``role`` is ``None`` for a tracker with no ``tracker_roles``
        entry, and such a tracker is keyed by its serial instead. ``vel``/``acc`` are
        ``(vx, vy, vz, wx, wy, wz)`` in the robot frame.
        """
        count = int(self._sdk.num_motion_data_available())
        if count == 0:
            return {}
        poses = self._sdk.get_motion_tracker_pose()
        vels = self._sdk.get_motion_tracker_velocity()
        accs = self._sdk.get_motion_tracker_acceleration()
        serials = self._sdk.get_motion_tracker_serial_numbers()
        out: dict[str, dict] = {}
        for i in range(count):
            pq = xr_pose_to_pos_quat(poses[i])
            if pq is None:
                continue
            serial = str(serials[i])
            role = self._tracker_roles.get(serial)
            out[role or serial] = {
                "serial": serial,
                "role": role,
                "pos": pq[0].tolist(),
                "quat": pq[1].tolist(),
                "vel": xr_vel6_to_robot(vels[i]).tolist(),
                "acc": xr_vel6_to_robot(accs[i]).tolist(),
            }
        return out

    def body_joints(self, poses: np.ndarray | None = None) -> dict[str, dict] | None:
        """Full-body joints as ``{name: {pos, quat, vel, acc}}``, or None when unavailable.

        Requires Full-body mode in the PICO app with calibrated motion trackers;
        keys are :data:`BODY_JOINT_NAMES` (``left_wrist``, ``left_ankle``,
        ``left_foot``, ``pelvis``, ...). ``vel``/``acc`` are ``(linear, angular)``
        6-vectors in the robot frame. A joint the service has not solved yet reports
        a degenerate quaternion and is published as identity at the origin.
        ``poses`` reuses an already-read ``(24, 7)`` service-frame array so one
        snapshot needs a single SDK pose read; velocities/accelerations are
        always read live.
        """
        if poses is None:
            if not self._sdk.is_body_data_available():
                return None
            poses = np.asarray(self._sdk.get_body_joints_pose(), dtype=float)
        vels = np.asarray(self._sdk.get_body_joints_velocity(), dtype=float)
        accs = np.asarray(self._sdk.get_body_joints_acceleration(), dtype=float)
        for name, arr, cols in (("pose", poses, 7), ("velocity", vels, 6), ("acceleration", accs, 6)):
            if arr.shape != (N_BODY_JOINTS, cols):
                raise ValueError(f"expected ({N_BODY_JOINTS}, {cols}) body {name} from the service, got {arr.shape}")
        body: dict[str, dict] = {}
        for i, joint in enumerate(BODY_JOINT_NAMES):
            pq = xr_pose_to_pos_quat(poses[i])
            body[joint] = {
                "pos": (pq[0] if pq is not None else np.zeros(3)).tolist(),
                "quat": (pq[1] if pq is not None else np.array([1.0, 0.0, 0.0, 0.0])).tolist(),
                "vel": xr_vel6_to_robot(vels[i]).tolist(),
                "acc": xr_vel6_to_robot(accs[i]).tolist(),
            }
        return body

    def body_raw(self) -> tuple[int, list[list[float]]] | None:
        """Untouched service-frame body poses: ``(timestamp_ns, (24, 7) xyz+xyzw lists)``.

        Deliberately no frame conversion, no untracked-joint substitution, no
        wxyz flip — for consumers that apply their own frame math to exactly
        what the SDK reports (the SONIC teleop pipelines). The timestamp is
        ``get_time_stamp_ns`` (not the body-specific stamp) read immediately
        before the poses, matching the in-process reader's dedupe order
        (``eden.extensions.teleop.sonic.inputs.PicoReader``).
        """
        if not self._sdk.is_body_data_available():
            return None
        stamp_ns = int(self._sdk.get_time_stamp_ns())
        poses = np.asarray(self._sdk.get_body_joints_pose(), dtype=float)
        if poses.shape != (N_BODY_JOINTS, 7):
            raise ValueError(f"expected ({N_BODY_JOINTS}, 7) body poses from the service, got {poses.shape}")
        return stamp_ns, poses.tolist()

    # ------------------------------------------------------------------
    # the published snapshot
    # ------------------------------------------------------------------

    def state(self, derived_body: bool = True) -> dict:
        """Snapshot every input as a JSON-safe dict, in the Genesis/robot frame.

        The controller keys deliberately match what ``eden metaquest bridge
        --tracking controller`` publishes (``head``, ``{side}_wrist``,
        ``{side}_trigger[_value]``, ``{side}_squeeze[_value]``,
        ``{side}_a_button``/``_b_button``, ``{side}_thumbstick[_value]``), so an
        Eden-side teleop loop runs against either headset unchanged. PICO's extras
        are additive: ``{side}_menu_button``, ``trackers`` (see :meth:`trackers`),
        ``body_ready`` and ``body`` (see :meth:`body_joints`), plus the lossless
        ``body_raw`` / ``body_raw_timestamp_ns`` passthrough (see :meth:`body_raw`).

        ``derived_body=False`` skips the per-joint ``body`` dict and ``trackers``
        (a consumer that only reads ``body_raw`` — the SONIC operator — saves
        most of the per-snapshot SDK work; ``body_ready`` then reflects the raw
        read).
        """
        # Read the head pose once so ``ready`` describes the matrix we publish.
        head = self.pose("head")
        state = {
            "device": "pico",
            "mode": "controller",
            "ready": bool(np.abs(head).sum() > 1e-6),
            "head": head.tolist(),
            "left_wrist": self.pose("left_controller").tolist(),
            "right_wrist": self.pose("right_controller").tolist(),
            "timestamp_ns": self.timestamp_ns(),
        }
        # X/Y (left) and A/B (right) are the primary/secondary buttons of their
        # controller, which is what the Meta Quest bridge calls a/b per side.
        primary = {"left": "X", "right": "A"}
        secondary = {"left": "Y", "right": "B"}
        for side in ("left", "right"):
            trigger = self.value(f"{side}_trigger")
            squeeze = self.value(f"{side}_grip")
            state.update(
                {
                    f"{side}_trigger": trigger > 0.5,
                    f"{side}_trigger_value": trigger,
                    f"{side}_squeeze": squeeze > 0.5,
                    f"{side}_squeeze_value": squeeze,
                    f"{side}_a_button": self.button(primary[side]),
                    f"{side}_b_button": self.button(secondary[side]),
                    f"{side}_menu_button": self.button(f"{side}_menu_button"),
                    f"{side}_thumbstick": self.button(f"{side}_axis_click"),
                    f"{side}_thumbstick_value": list(self.axis(side)),
                }
            )

        raw = self.body_raw()
        if raw is not None:
            state["body_raw_timestamp_ns"], state["body_raw"] = raw
        if derived_body:
            state["trackers"] = self.trackers()
            # Reuse the raw read: one SDK pose read per snapshot.
            body = self.body_joints(poses=None if raw is None else np.asarray(raw[1], dtype=float))
            state["body_ready"] = body is not None
            if body is not None:
                state["body"] = body  # {joint_name: {pos, quat, vel, acc}}
        else:
            state["body_ready"] = raw is not None
        return state

    def close(self) -> None:
        """Disconnect from the PC Service."""
        self._sdk.close()

    @staticmethod
    def _get(table: dict, name: str, kind: str):
        """Look ``name`` up in a getter table, naming the valid options on a miss."""
        try:
            return table[name]
        except KeyError:
            raise ValueError(f"unknown PICO {kind} {name!r}; expected one of {sorted(table)}") from None
