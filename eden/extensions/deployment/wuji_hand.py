"""Wuji Hand 1 / Hand 2 real-robot deployment backend.

Both generations are 20-DOF hands (5 fingers x 4 joints) whose SDKs speak joint angles in radians,
finger-major (thumb first), in the MJCF joint directions -- no sign or offset conversion. The SDK
order is the same for a right and a left hand; the side is a property of the joint *names*, which
:data:`WUJI_JOINT_IDX` maps for both sides and both naming schemes in Eden:

- ``{right,left}_finger{1..5}_joint{1..4}`` (:class:`~eden.options.robots.wuji_hand1.WujiHand1_R` and
  the ``wuji-mjlab`` task models), and
- ``{r,l}_{thumb_cmc_flex, ..., pinky_dip}`` (:class:`~eden.options.robots.wuji_hand2beta2.WujiHand2Beta2_R`),
  whose declaration order is the same finger-major order (upstream ``WUJI_HAND2_TASK_NAMES``).

A :class:`HandTransport` is the wire to one hand: :class:`Hand1Transport` (``wujihandpy``) or
:class:`Hand2Transport` (``wuji_sdk``). It streams targets the way upstream ``wuji-mjlab`` does:

- ``"sync"`` (upstream reorient): :meth:`HandTransport.command` blocks for one control period,
  writing ``servo_hz`` linearly interpolated sub-targets from the previous command to the new one.
- ``"thread"`` (upstream pen spin): :meth:`HandTransport.command` starts a new ramp and returns;
  a background thread streams it at ``servo_hz`` and holds the target once the ramp ends.

:class:`WujiHandDeployment` drives one hand through a transport it builds at :meth:`connect`, or
one handed to it with :meth:`WujiHandDeployment.use_transport` (a simulator, for sim-to-sim).
"""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np

import eden as en
from eden.extensions.deployment.base import (
    DEPLOYMENT_REGISTRY,
    DeployStop,
    DeploymentBase,
    bind_deployment,
    resolve_dof_index_map,
    resolve_dofs_values,
)
from eden.extensions.deployment.utils.state import RobotCommand, RobotState
from eden.options.actuators import WujiHand2Actuator
from eden.options.robots.wuji_hand1 import _WujiHand1
from eden.options.robots.wuji_hand2beta2 import _WujiHand2Beta2
from eden.utils.string import first_few, resolve_matching_names_values

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.extensions.deployment import DeploymentOptions

NUM_JOINTS = 20
_SDK_JOINTS = tuple(f"sdk[{i}]" for i in range(NUM_JOINTS))  # joint names when a transport is given none

#: The easing curves :meth:`HandTransport.ease_to` knows: upstream's ``home`` and its start-pose ramp.
EASE_PROFILES = ("cosine", "smoothstep")

#: Settle time after ``wuji_sdk`` ``enable()`` before the first streamed command (upstream's driver).
_HAND2_ARM_DELAY_S = 0.25

# Per finger, thumb first: the Hand2Beta2 anatomical joint names in SDK (finger-major) order.
_HAND2_ANATOMICAL = (
    ("thumb", ("cmc_flex", "cmc_abd", "mcp", "ip")),
    ("index_finger", ("mcp_flex", "mcp_abd", "pip", "dip")),
    ("middle_finger", ("mcp_flex", "mcp_abd", "pip", "dip")),
    ("ring_finger", ("mcp_flex", "mcp_abd", "pip", "dip")),
    ("pinky", ("mcp_flex", "mcp_abd", "pip", "dip")),
)


def _joint_index_table() -> tuple[dict[str, int], dict[str, str]]:
    index: dict[str, int] = {}
    side: dict[str, str] = {}
    for long, short in (("right", "r"), ("left", "l")):
        for f in range(5):
            for j in range(4):
                index[f"{long}_finger{f + 1}_joint{j + 1}"] = 4 * f + j
                side[f"{long}_finger{f + 1}_joint{j + 1}"] = long
            finger, joints = _HAND2_ANATOMICAL[f]
            for j, joint in enumerate(joints):
                index[f"{short}_{finger}_{joint}"] = 4 * f + j
                side[f"{short}_{finger}_{joint}"] = long
    return index, side


WUJI_JOINT_IDX, _JOINT_SIDE = _joint_index_table()
"""Joint name (either side, either naming scheme) -> index in the SDK's finger-major joint array."""

_NAMING_HINT = "Wuji joints are named {right,left}_finger{1-5}_joint{1-4} or {r,l}_thumb_cmc_flex ... {r,l}_pinky_dip."


def sdk_index_map(names: Sequence[str], owner: str) -> np.ndarray:
    """Each of ``names``' index in the SDK's joint array, raising (as ``owner``) unless they cover the 20 once each."""
    index = resolve_dof_index_map(names, WUJI_JOINT_IDX, owner=owner, what="Wuji SDK joint index", hint=_NAMING_HINT)
    if sorted(index) != list(range(NUM_JOINTS)):
        by_index = {i: [n for n, j in zip(names, index) if j == i] for i in sorted(set(index))}
        missing = sorted(set(range(NUM_JOINTS)) - set(index))
        twice = {i: dup for i, dup in by_index.items() if len(dup) > 1}
        raise ValueError(
            f"{owner}: the joints must cover each of the {NUM_JOINTS} Wuji SDK joints once; SDK joints {missing} "
            f"have no joint and {twice} have several (SDK index: joint names). {_NAMING_HINT}"
        )
    return np.asarray(index)


# --------------------------------------------------------------------------------------------- #
# The wire: upstream's drivers (``deploy/reorient/.../drivers``, ``deploy/pen_spin/lib/hand2_adapter.py``)
# --------------------------------------------------------------------------------------------- #


@dataclass
class StreamOptions:
    """How targets reach the hand.

    Parameters
    ----------
    mode : {"sync", "thread"}
        See the module docstring.
    servo_hz : float
        Rate of the interpolated sub-targets.
    interp_s : float
        Duration of one ramp: the control period.
    """

    mode: Literal["sync", "thread"] = "sync"
    servo_hz: float = 200.0
    interp_s: float = 0.02


class HandTransport:
    """The contract every Wuji transport keeps (upstream ``HandDriverBase``): 20 joint angles, SDK order.

    Parameters
    ----------
    stream : StreamOptions
        How :meth:`command` streams a target.
    joint_names : Sequence[str] | None
        The 20 joints' names in SDK order, for error messages (:attr:`WujiHandDeployment.sdk_joint_names`);
        ``None`` names them by SDK index.
    """

    def __init__(self, stream: StreamOptions, joint_names: Sequence[str] | None = None) -> None:
        self.stream = stream
        self.joint_names = _SDK_JOINTS if joint_names is None else tuple(joint_names)
        if len(self.joint_names) != NUM_JOINTS:
            raise ValueError(f"joint_names needs the {NUM_JOINTS} joints in SDK order, got {list(self.joint_names)}")
        self._cmd: np.ndarray | None = None
        self._deadline: float | None = None
        self._lock = threading.Lock()
        self._seg_start: np.ndarray | None = None
        self._seg_target: np.ndarray | None = None
        self._seg_t0 = 0.0
        self._servo_thread: threading.Thread | None = None
        self._servo_stop = threading.Event()
        self._servo_error: BaseException | None = None

    # --- driver primitives -------------------------------------------------

    def connect(self) -> None:
        raise NotImplementedError

    def write(self, qpos: np.ndarray) -> None:
        """Send one joint-position target to the hand now."""
        raise NotImplementedError

    def read(self) -> np.ndarray:
        """The latest measured joint positions."""
        raise NotImplementedError

    def fault(self) -> str | None:
        """A latched hardware fault, described, or ``None``."""
        return None

    def stale(self) -> str | None:
        """Why the joint feedback can no longer be trusted (a lost link), or ``None``."""
        return None

    def clock(self) -> float:
        """The time joint readings are stamped with (wall clock; a simulation's own time in sim-to-sim)."""
        return time.perf_counter()

    def temp_summary(self) -> str | None:
        return None

    def disconnect(self) -> None:
        raise NotImplementedError

    # --- streaming -----------------------------------------------------------

    def command(self, target: np.ndarray) -> None:
        """Stream ``target`` from the previous command, as upstream does for this task."""
        target = _checked(target, self.joint_names)
        if self._cmd is None:
            self._cmd = self.read().copy()
        if self.stream.mode == "sync":
            self._ramp(self._cmd, target, self.stream.interp_s)
            self._cmd = target.copy()
            return
        self._ensure_servo()
        self._set_segment(None, target)

    def hold(self) -> None:
        """Let one control period pass with no new target, holding the last one.

        ``"sync"`` re-streams the last command across the period through :meth:`_ramp`, so a tick
        that sends nothing still lasts one control period (on a simulator's clock too); before any
        command there is nothing to hold, and it only waits. ``"thread"`` returns at once: the servo
        thread is already holding, and the base rate limiter paces the loop.
        """
        if self.stream.mode != "sync":
            self._raise_if_servo_failed()
            return
        if self._cmd is None:
            time.sleep(self.stream.interp_s)
            return
        self._ramp(self._cmd, self._cmd, self.stream.interp_s)

    def ease_to(
        self,
        goal: np.ndarray,
        duration_s: float,
        rate_hz: float,
        profile: str,
        guard: Callable[[], None] | None = None,
    ) -> None:
        """Move from the measured pose to ``goal``: upstream's ``home`` (cosine, 100 Hz) or start-pose smoothstep.

        ``guard`` runs before every waypoint, so a safety check that raises stops a multi-second
        move where it is, holding the last waypoint, instead of only before it starts.
        """
        if not (duration_s >= 0.0 and rate_hz > 0.0):
            raise ValueError(f"ease_to needs duration_s >= 0 and rate_hz > 0; got {duration_s!r}, {rate_hz!r}.")
        goal = _checked(goal, self.joint_names)
        start = self.read().copy()
        n = max(1, round(duration_s * rate_hz))
        path = [start + ease(i / n, profile) * (goal - start) for i in range(1, n + 1)]
        if self.stream.mode == "sync":
            for prev, q in zip([start, *path], path):
                if guard is not None:
                    guard()
                self._ramp(prev, q, 1.0 / rate_hz)
                self._cmd = q.copy()
            return
        if self._servo_thread is None:
            self._cmd = start
        for q in path:
            if guard is not None:
                guard()
            self._ensure_servo()
            self._set_segment(q, q)
            time.sleep(1.0 / rate_hz)

    def _ramp(self, start: np.ndarray, target: np.ndarray, seconds: float) -> None:
        """Write ``servo_hz`` sub-targets from ``start`` to ``target`` over ``seconds`` (every ``"sync"`` motion).

        Paced to the wall clock against a deadline chained from the previous ramp, as upstream anchors
        each period before inference: time spent between ramps (inference, observation) is absorbed
        rather than added, and a gap longer than ``seconds`` restarts the chain. This is the one place
        a ``"sync"`` transport meets time, so a simulator overrides it to step its own clock instead.
        """
        n_sub = max(1, round(seconds * self.stream.servo_hz))
        dt = seconds / n_sub
        now = time.monotonic()
        t0 = self._deadline if self._deadline is not None and now - self._deadline < seconds else now
        for i in range(1, n_sub + 1):
            self.write(start + (i / n_sub) * (target - start))
            if (wait := t0 + dt * i - time.monotonic()) > 0:
                time.sleep(wait)
        self._deadline = t0 + seconds

    def _set_segment(self, start: np.ndarray | None, target: np.ndarray) -> None:
        """Hand the servo thread a ramp from ``start`` (``None``: where it is now) to ``target`` over ``interp_s``."""
        with self._lock:
            start = self._cmd if start is None else start
            self._seg_start, self._seg_target, self._seg_t0 = start.copy(), target.copy(), time.monotonic()

    def _raise_if_servo_failed(self) -> None:
        if self._servo_error is not None:
            raise RuntimeError(f"hand servo stream stopped: {self._servo_error}") from self._servo_error

    def _ensure_servo(self) -> None:
        """Surface a servo thread's error, or start the thread holding :attr:`_cmd` if it is not running."""
        self._raise_if_servo_failed()
        if self._servo_thread is None:
            self._set_segment(self._cmd, self._cmd)
            self._servo_stop.clear()
            self._servo_thread = threading.Thread(target=self._servo_loop, daemon=True, name="wuji-hand-servo")
            self._servo_thread.start()

    def _servo_loop(self) -> None:
        dt = 1.0 / self.stream.servo_hz
        next_t = time.monotonic()
        while not self._servo_stop.is_set():
            now = time.monotonic()
            with self._lock:
                s = min(max((now - self._seg_t0) / self.stream.interp_s, 0.0), 1.0)
                cmd = self._seg_start + s * (self._seg_target - self._seg_start)
                self._cmd = cmd
            try:
                self.write(cmd)
            except BaseException as e:  # surfaced on the next command() / ease_to(), never swallowed
                self._servo_error = e
                en.logger.error(f"[wuji] servo stream error, stopped: {e}")
                return
            next_t += dt
            wait = next_t - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            else:
                next_t = time.monotonic()  # fell behind: resync, don't burst

    def stop_servo(self) -> None:
        self._servo_stop.set()
        if self._servo_thread is not None:
            self._servo_thread.join(timeout=1.0)
            if self._servo_thread.is_alive():
                en.logger.error("[wuji] servo thread did not stop within 1 s; disconnecting anyway")
        self._servo_thread = None

    def close(self) -> None:
        self.stop_servo()
        # A reconnect streams from the pose it then measures, not from the previous connection's last target.
        self._cmd = self._deadline = self._servo_error = None
        self.disconnect()


def ease(x: float, profile: str) -> float:
    """Progress along an :data:`EASE_PROFILES` curve at ``x`` in [0, 1]; 0 and 1 at the ends, zero slope at both."""
    if profile not in EASE_PROFILES:
        raise ValueError(f"ease profile must be one of {EASE_PROFILES}; got {profile!r}.")
    return 0.5 * (1.0 - np.cos(np.pi * x)) if profile == "cosine" else x * x * (3.0 - 2.0 * x)


def _checked(qpos: np.ndarray, joint_names: Sequence[str] = _SDK_JOINTS) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
    if qpos.shape != (NUM_JOINTS,):
        raise ValueError(f"a Wuji hand target has {NUM_JOINTS} joint angles, got shape {qpos.shape}")
    # Never stream NaN/Inf to the motors: an EMA would latch it for good.
    if not np.isfinite(qpos).all():
        bad = [joint_names[i] for i in np.flatnonzero(~np.isfinite(qpos))]
        raise ValueError(f"refusing a non-finite joint target for {bad}: {qpos.tolist()}")
    return qpos


def _nid_index(nid: int) -> int | None:
    """``wuji_sdk`` joint ``nid`` (1-based, stride 5 per finger) -> finger-major index, or ``None`` for a non-joint node."""
    finger, jw = divmod(int(nid) - 1, 5)
    return finger * 4 + jw if jw < 4 and 0 <= finger < 5 else None


class Hand1Transport(HandTransport):
    """Wuji Hand 1 via ``wujihandpy``'s realtime position controller (upstream ``WujiHand1Driver``).

    The hand runs its own position loop behind a low-pass; command gains do not apply. Every
    :meth:`read` is a checked bus round-trip (``read_joint_actual_position``), so the feedback cannot
    freeze silently the way a subscribed feed can: a read that gets no answer within
    ``stale_after_s`` raises :class:`DeployStop`, and :meth:`stale` reports it from then on.

    Parameters
    ----------
    stream : StreamOptions
        How targets are streamed.
    effort_limit_a : float
        Joint current limit, A.
    lowpass_cutoff_hz : float
        Cutoff of the controller's target low-pass.
    joint_names : Sequence[str] | None
        See :class:`HandTransport`.
    stale_after_s : float
        Timeout of each joint-position read.
    """

    def __init__(
        self,
        stream: StreamOptions,
        effort_limit_a: float = 0.5,
        lowpass_cutoff_hz: float = 3.0,
        joint_names: Sequence[str] | None = None,
        stale_after_s: float = 0.5,
    ) -> None:
        super().__init__(stream, joint_names)
        self.effort_limit_a = effort_limit_a
        self.lowpass_cutoff_hz = lowpass_cutoff_hz
        self.stale_after_s = stale_after_s
        self._hand = None
        self._ctrl = None
        self._ctrl_cm = None
        self._stale: str | None = None

    def connect(self) -> None:
        self._stale = None
        try:
            import wujihandpy
        except ImportError as e:
            raise ImportError(
                "Wuji Hand 1 needs wujihandpy (upstream pins 1.5.1): pip install wujihandpy==1.5.1"
            ) from e
        self._hand = wujihandpy.Hand()
        self._hand.write_joint_effort_limit(float(self.effort_limit_a))
        # Energized from here on: any failure must de-energize before propagating.
        try:
            self._hand.write_joint_enabled(True)
            self._ctrl_cm = self._hand.realtime_controller(
                enable_upstream=True, filter=wujihandpy.filter.LowPass(cutoff_freq=float(self.lowpass_cutoff_hz))
            )
            self._ctrl = self._ctrl_cm.__enter__()
        except BaseException:
            self.disconnect()
            raise
        en.logger.info(
            f"[wuji] Hand 1 enabled: current limit {self.effort_limit_a:g} A, target low-pass "
            f"{self.lowpass_cutoff_hz:g} Hz, {self.stream.mode} streaming at {self.stream.servo_hz:g} Hz"
        )

    def write(self, qpos: np.ndarray) -> None:
        self._ctrl.set_joint_target_position(_checked(qpos, self.joint_names).reshape(5, 4))

    def read(self) -> np.ndarray:
        try:
            q = self._hand.read_joint_actual_position(timeout=self.stale_after_s)
        except Exception as e:
            self._stale = (
                f"read_joint_actual_position(timeout={self.stale_after_s:g}) failed with "
                f"{type(e).__name__}: {e} (USB link lost?)"
            )
            raise DeployStop(f"hand feedback lost: {self._stale}") from e
        return np.asarray(q, dtype=np.float64).flatten()

    def stale(self) -> str | None:
        return self._stale

    def disconnect(self) -> None:
        try:
            if self._ctrl_cm is not None:
                self._ctrl_cm.__exit__(None, None, None)
        except BaseException as e:
            en.logger.warning(f"[wuji] Hand 1 realtime_controller close failed: {e}")
        finally:
            self._ctrl = self._ctrl_cm = None
        if self._hand is not None:
            try:
                self._hand.write_joint_enabled(False)
            except BaseException as e:
                en.logger.warning(f"[wuji] Hand 1 disable failed: {e}")


#: Hand 2's impedance modes, by Eden's name: the ``wuji_sdk.ControlMode`` member each selects.
_HAND2_MODES = {"mit": "Mit", "mit_v2": "MitV2"}
#: The first firmware with MIT_V2 (``wuji_sdk`` 2026.11.23 docs).
_MIT_V2_FIRMWARE = (3, 1, 0)


@dataclass
class Hand2Gains:
    """The impedance mode, gains and effort limit programmed on a Wuji Hand 2 at connect.

    Parameters
    ----------
    kp, kd : list[float]
        Per-joint stiffness and damping, SDK (finger-major) order: firmware MIT units in ``"mit"``,
        N·m/rad and N·m·s/rad in ``"mit_v2"``.
    effort_limit : list[float]
        Per-joint effort limit: a current in A in ``"mit"``, a torque in N·m in ``"mit_v2"``.
    control_mode : {"mit", "mit_v2"}
        The mode selected before the gains and limit are written. ``"mit_v2"`` needs firmware 3.1.0.
    """

    kp: list[float]
    kd: list[float]
    effort_limit: list[float]
    control_mode: Literal["mit", "mit_v2"] = "mit"

    def __post_init__(self) -> None:
        for name in ("kp", "kd", "effort_limit"):
            if len(getattr(self, name)) != NUM_JOINTS:
                raise ValueError(f"Hand2Gains.{name} needs {NUM_JOINTS} values, got {len(getattr(self, name))}")
        if self.control_mode not in _HAND2_MODES:
            raise ValueError(f"Hand2Gains.control_mode={self.control_mode!r} is not one of {tuple(_HAND2_MODES)}.")

    @property
    def effort_unit(self) -> str:
        return "A" if self.control_mode == "mit" else "N·m"


def _parse_firmware(version: str) -> tuple[int, ...] | None:
    """``"v2.13.1"`` -> ``(2, 13, 1)``; ``None`` for a string in another format (the mode readback still checks)."""
    parts = version.strip().lstrip("vV").split(".")
    return tuple(int(p) for p in parts) if parts and all(p.isdigit() for p in parts) else None


#: ``device_name`` is the handle ``wuji_sdk`` registers a connection under and ``disconnect``
#: resolves by, so two hands (or a hand and a :class:`~eden.utils.devices.wuji.glove.WujiGlove`) in
#: one process must never share one: each would tear down the other's link on close.
_device_seq = itertools.count()


class Hand2Transport(HandTransport):
    """Wuji Hand 2 via ``wuji_sdk`` (upstream ``WujiHand2Driver`` and pen spin's ``Hand2Compat``).

    Connects to ``serial_number`` (or the single hand on the bus), selects the gains' control mode,
    programs the gains and the effort limit and reads them back, enables the motors, and streams
    ``JointCommand(q, 0, 0)``. Joint positions arrive on the SDK's ``joint_states`` callback;
    ``joint_diagnostics`` feeds :meth:`fault`, and the age of the joint feed feeds :meth:`stale`.

    Parameters
    ----------
    stream : StreamOptions
        How targets are streamed.
    gains : Hand2Gains
        Programmed once at connect.
    serial_number : str
        The hand; empty connects the only hand on the bus.
    require_all_joints : bool
        Refuse a hand with fewer than 20 joints online (upstream reorient). Otherwise (pen spin) the
        joints that report during :meth:`connect` are the ones :meth:`stale` watches.
    feedback_timeout_s : float
        How long :meth:`connect` waits for the first fresh reading of every watched joint.
    stale_after_s : float
        A watched joint (or the whole feed) older than this is stale.
    joint_names : Sequence[str] | None
        See :class:`HandTransport`.
    """

    def __init__(
        self,
        stream: StreamOptions,
        gains: Hand2Gains,
        serial_number: str = "",
        require_all_joints: bool = True,
        feedback_timeout_s: float = 2.0,
        stale_after_s: float = 0.5,
        joint_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__(stream, joint_names)
        self.gains = gains
        self.serial_number = serial_number
        self.require_all_joints = require_all_joints
        self.feedback_timeout_s = feedback_timeout_s
        self.stale_after_s = stale_after_s
        self.device_name = f"eden_wuji_hand_2_{next(_device_seq)}"
        self._sdk = self._mgr = self._hand = self._pub = self._sub = self._diag_sub = None
        self._state_lock = threading.Lock()
        self._latest = np.zeros(NUM_JOINTS)
        self._joint_t = np.full(NUM_JOINTS, -np.inf)
        self._watched = np.ones(NUM_JOINTS, dtype=bool)
        self._last_state_t = 0.0
        # Every (nid, code) ever reported, in arrival order, never replaced: a joint's later warning must
        # not un-latch the error it reported first. (Insertion-ordered dict: a set would lose the order.)
        self._errors: dict[tuple[int, int], None] = {}
        self._temps: dict[int, float] = {}
        self._dismissed: set[tuple[int, int]] = set()

    def connect(self) -> None:
        from eden.utils.devices.wuji.glove import load_sdk

        self._sdk = wuji_sdk = load_sdk()
        self._mgr = wuji_sdk.SdkManager.instance()
        sn = self.serial_number or self._only_hand_on_the_bus()
        self._hand = self._mgr.connect(
            sn=sn, device_name=self.device_name, options=wuji_sdk.ConnectOptions(enable_bridge=False)
        )
        self._firmware = self._firmware_version()
        en.logger.info(f"[wuji] connected Wuji Hand 2 sn={self._hand.serial_number}, firmware {self._firmware}")
        # Connected from here on: any failure must disable and release the link before propagating.
        try:
            n_online = self._hand.online_joints_count().get()
            if (self.require_all_joints and n_online != NUM_JOINTS) or n_online == 0:
                raise RuntimeError(
                    f"Wuji Hand 2 {sn}: {n_online}/{NUM_JOINTS} joints online; refusing to move an incomplete "
                    "hand (a missing joint reads back as its default pose). To run with the joints that are "
                    "online, WujiHandDeployment.configure(require_all_joints=False)."
                )
            g = self.gains
            self._select_control_mode(g.control_mode)
            self._hand.effort_limit().set([float(a) for a in g.effort_limit])
            self._hand.mit_params().set(
                [wuji_sdk.MitParam(float(p), float(d)) for p, d in zip(g.kp, g.kd, strict=True)]
            )
            self._verify_programmed(g)
            self._sub = self._hand.joint_states().subscribe_with_callback(self._on_state)
            try:
                self._diag_sub = self._hand.joint_diagnostics().subscribe_with_callback(self._on_diag)
            except Exception as e:
                en.logger.warning(f"[wuji] joint_diagnostics unavailable ({e}); faults limited to stale feedback")
            self._hand.enable()
            time.sleep(_HAND2_ARM_DELAY_S)
            self._pub = self._hand.joint_command().publish()
            self._await_feedback(n_online)
        except BaseException:
            self.disconnect()
            raise
        partial = "" if self._watched.all() else f", {int(self._watched.sum())}/{NUM_JOINTS} joints online"
        en.logger.info(
            f"[wuji] Hand 2 enabled: kp {min(g.kp):.3g}-{max(g.kp):.3g}, kd {min(g.kd):.3g}-{max(g.kd):.3g}, "
            f"{g.control_mode}, effort limit {min(g.effort_limit):.3g}-{max(g.effort_limit):.3g} {g.effort_unit}, "
            f"{self.stream.mode} streaming at {self.stream.servo_hz:g} Hz{partial}"
        )

    def _firmware_version(self) -> str:
        """The hand's firmware version for the connect log; a failed read is logged, never fatal."""
        try:
            return str(self._hand.firmware_version().get())
        except Exception as e:  # noqa: BLE001 - diagnostic only; the hand is still usable
            return f"unknown ({type(e).__name__}: {e})"

    def _select_control_mode(self, control_mode: str) -> None:
        """Put the hand in ``control_mode``, the mode whose units the gains and effort limit are in.

        The effort limit is in A in MIT mode and in N·m in Position and MIT_V2 (firmware 3.0.0+), the modes
        store their gains separately, and the hand stays in whatever mode the last program set. The SDK
        switches modes only on disabled joints. Its readback covers the whole hand, and when joints disagree
        it logs a warning (listing them) and returns the mode last written; ``wuji_sdk`` has no per-joint
        mode read to check further.
        """
        member = _HAND2_MODES[control_mode]
        if control_mode == "mit_v2" and (version := _parse_firmware(self._firmware)) and version < _MIT_V2_FIRMWARE:
            raise RuntimeError(
                f"Wuji Hand 2 {self._hand.serial_number} runs firmware {self._firmware}, and MIT_V2 needs "
                f"{'.'.join(map(str, _MIT_V2_FIRMWARE))} or later. Deploy it with control_mode='mit', or update its "
                "firmware first."
            )
        if not hasattr(self._sdk.ControlMode, member):
            raise RuntimeError(
                f"wuji_sdk {getattr(self._sdk, '__version__', '(unknown version)')} has no ControlMode.{member}; "
                f"control_mode={control_mode!r} needs wuji-sdk 2026.11.23 or later: pip install -U wuji-sdk."
            )
        want = getattr(self._sdk.ControlMode, member)
        self._hand.disable()
        self._hand.control_mode().set(want)
        if (mode := self._hand.control_mode().get()) != want:
            fix = (
                " MIT_V2 needs firmware 3.1.0 or later; on older firmware deploy with control_mode='mit'."
                if control_mode == "mit_v2"
                else ""
            )
            raise RuntimeError(
                f"Wuji Hand 2 {self._hand.serial_number}: control mode reads {mode} after selecting {want}, so the "
                f"gains and effort limit would be read in another mode's units.{fix} Power-cycle the hand and check "
                "the firmware version; do not enable it in this mode."
            )

    def _verify_programmed(self, g: Hand2Gains) -> None:
        """Read the current limits and MIT gains back; an online joint holding other values refuses the run."""
        limits, params = self._hand.effort_limit().get(), self._hand.mit_params().get()
        wrong = []
        for i, (limit, p) in enumerate(zip(limits, params, strict=True)):
            if limit is None and p is None:  # offline
                continue
            want = (g.effort_limit[i], g.kp[i], g.kd[i])
            got = (limit, getattr(p, "kp", None), getattr(p, "kd", None))
            if None in got or not np.allclose(got, want, rtol=1e-4, atol=1e-6):
                wrong.append(f"{self.joint_names[i]} (limit {g.effort_unit}, kp, kd) wrote {want}, read {got}")
        if wrong:
            raise RuntimeError(
                f"Wuji Hand 2 {self._hand.serial_number}: {len(wrong)} joint(s) read back other values than were "
                f"written: {'; '.join(wrong)}. Power-cycle the hand and reconnect; do not enable it with these "
                "values."
            )

    def _only_hand_on_the_bus(self) -> str:
        """The serial number of the one Wuji hand ``scan()`` finds; a Wuji Glove on the bus is not a hand."""
        devs = [d for d in self._mgr.scan() if d.device_type != self._sdk.DeviceType.WujiGlove]
        if len(devs) != 1:
            listing = ", ".join(f"sn={d.sn} ip={d.ip} type={d.device_type}" for d in devs)
            raise RuntimeError(
                f"found {len(devs)} Wuji hands on the bus ({listing or 'none'}); power on exactly one, "
                "or pin one with its serial number (WujiHandDeployment.configure(serial_number=...))."
            )
        return devs[0].sn

    def _await_feedback(self, n_online: int) -> None:
        """Wait for fresh readings; without ``require_all_joints``, the joints that report become the watched set."""
        t0 = time.monotonic()
        while True:
            fresh = time.monotonic() - self._joint_t <= self.stale_after_s
            if fresh.all() or (not self.require_all_joints and fresh.sum() >= n_online):
                self._watched = fresh.copy()
                return
            if time.monotonic() - t0 > self.feedback_timeout_s:
                silent = [self.joint_names[i] for i in np.flatnonzero(~fresh)]
                fix = "WujiHandDeployment.configure(connect_timeout_s=...)"
                if self.require_all_joints:
                    fix += ", or run on the joints that report with require_all_joints=False"
                raise RuntimeError(
                    f"Wuji Hand 2: no fresh joint_states within {self.feedback_timeout_s:g} s for joints {silent} "
                    f"({int(fresh.sum())} reporting, {n_online} online). Raise the wait with {fix}."
                )
            time.sleep(0.02)

    def _on_state(self, frame) -> None:
        now = time.monotonic()
        with self._state_lock:
            self._last_state_t = now
            for e in frame.joints:
                i = _nid_index(e.nid)
                if i is not None:
                    self._latest[i] = e.position
                    self._joint_t[i] = now

    def _on_diag(self, frame) -> None:
        for e in frame.joints:
            nid = int(e.nid)
            self._temps[nid] = float(e.mcu_temp_c_fb)
            if code := int(e.error_code_current):
                self._errors.setdefault((nid, code))

    def fault(self) -> str | None:
        for nid, code in list(self._errors):
            if (nid, code) in self._dismissed:
                continue
            i = _nid_index(nid)
            joint = f"joint {self.joint_names[i]!r} (nid={nid})" if i is not None else f"node nid={nid}"
            detail, severity = f"0x{code:04X}", ""
            try:
                d = self._hand.describe_error(code)
                if isinstance(d, dict):
                    severity = str(d.get("severity", ""))
                    detail = f"0x{code:04X} {d.get('name')} [{severity}] {d.get('desc')} ({d.get('resolution')})"
            except Exception:
                pass  # the bare code is still reported below
            if "warn" in severity.lower():
                en.logger.warning(f"[wuji] {joint} warning {detail}")
                self._dismissed.add((nid, code))
                continue
            return f"{joint} error {detail}"
        return None

    def stale(self) -> str | None:
        now = time.monotonic()
        with self._state_lock:
            ages, last = now - self._joint_t, self._last_state_t
        if (feed_age := now - last) > self.stale_after_s:
            return f"joint_states feed stale for {feed_age:.1f} s (bus or link lost?)"
        if missing := [self.joint_names[i] for i in np.flatnonzero(self._watched & (ages > self.stale_after_s))]:
            return f"joint feedback stale for {missing}"
        return None

    def temp_summary(self) -> str | None:
        if not self._temps:
            return None
        t = np.array(list(self._temps.values()))
        return f"{t.min():.1f}-{t.max():.1f} C (mean {t.mean():.1f})"

    def write(self, qpos: np.ndarray) -> None:
        jc = self._sdk.JointCommand
        self._pub.send([jc(float(x), 0.0, 0.0) for x in _checked(qpos, self.joint_names)])

    def read(self) -> np.ndarray:
        with self._state_lock:
            return self._latest.copy()

    def disconnect(self) -> None:
        if self._hand is None:
            return
        try:
            self._hand.disable()
        except Exception as e:
            en.logger.warning(f"[wuji] Hand 2 disable failed: {e}")
        for sub in (self._diag_sub, self._sub):
            try:
                if sub is not None:
                    sub.close()
            except Exception as e:
                en.logger.warning(f"[wuji] Hand 2 subscription close failed: {e}")
        # This connection only: ``disconnect_all`` would also drop a glove or the other hand.
        try:
            self._mgr.disconnect(self.device_name)
        except Exception as e:
            en.logger.warning(f"[wuji] Hand 2 disconnect failed: {e}")
        self._hand = self._pub = self._sub = self._diag_sub = None


# --------------------------------------------------------------------------------------------- #
# The backend
# --------------------------------------------------------------------------------------------- #


@DEPLOYMENT_REGISTRY.register()
class WujiHandDeployment(DeploymentBase):
    """A Wuji Hand 1 or Hand 2, right or left, on a fixed mount.

    The side comes from the robot's joint names (see :data:`WUJI_JOINT_IDX`); the deployment DOF
    order may be any permutation of the 20 joints, and is scattered into the SDK's order on the way
    out and gathered back on the way in. The hand has no IMU: :meth:`read_state` reports the mount
    orientation (the robot's ``default_root_quat``) as the base's.

    **Joint velocity is a finite difference** of consecutive reads over the transport's clock, like
    the XHand's (neither SDK reports one), optionally low-passed (``vel_lowpass_hz``).

    **Gains are programmed once at connect, never per command.** Hand 2 takes ``kp`` / ``kd``
    (``mit_kp`` / ``mit_kd``, default the payload's nominal gains: the robot's ``default_dofs_kp`` /
    ``default_dofs_kd`` times ``deploy_kp_kd_scale``) in the units of its ``control_mode``. In
    ``"mit_v2"`` (firmware 3.1.0+) those are the joint-space gains, programmed as they are, with the
    robot's force limits as the torque limit. In legacy ``"mit"`` they are converted to firmware MIT
    units through each joint's :class:`~eden.options.actuators.WujiHand2Actuator` spec. Hand 1 runs its own position
    loop and has no gain to set. So ``disable_joints`` (which works by zeroing command gains) is refused for
    both, and ``deploy_kp_kd_scale`` / ``mit_kp`` / ``mit_kd`` for Hand 1.

    **Loop order.** The base control tick (:meth:`DeploymentBase.step`, the operator state machine
    every backend runs) with this backend's hooks: :meth:`check_safety` first, then the command is
    sent and the period waited out, then the env's episode clock advances (``episode_length_buf``:
    the scene is never stepped here, and action warm-ups and clip clocks read it), then the hand is
    read and :meth:`sync_scene` -- the hook a subclass uses to write anything else it observes (a
    camera-tracked object) into the scene -- runs, and only then do the commands publish
    (``sync_before_commands``), so a command that grades the new state sees it this tick. In
    ``stream_mode="sync"`` sending the command *is* the wait (it streams sub-targets across one
    control period, as upstream does), so the base's rate limiter is not used on top of it: a tick
    takes one control period, and the inference time is absorbed into it (each period's deadline
    is chained from the previous one, as upstream anchors its loop before inference). A tick with
    nothing to send (``action=None`` before a slot has written targets) holds the last target for
    the period instead (:meth:`HandTransport.hold`), so it still lasts one period and the episode
    clock and the reference cannot race ahead at the speed of the loop.

    **Boot, INIT and START.** :meth:`reset` restarts the episode clock and the velocity estimate and
    boots in STOP, holding the measured pose. The operator's INIT ramps from the pose measured at
    the press to ``init_target`` over ``init_steps`` ticks and holds it; START hands the hand to the
    policy. :meth:`check_safety` runs before every one of those ticks, and :meth:`ease_to` runs
    :meth:`check_hand` before every waypoint.

    Parameters
    ----------
    hand : {"hand1", "hand2"} | None
        Hardware generation. ``None`` infers it from the robot options (``WujiHand1*`` -> ``"hand1"``,
        ``WujiHand2Beta2*`` -> ``"hand2"``); any other robot must set it.
    serial_number : str
        Hand 2 to connect; empty connects the only hand on the bus.
    stream_mode : {"sync", "thread"}
        How targets are streamed (see the module docstring).
    servo_hz : float
        Rate of the interpolated sub-targets.
    control_mode : {"mit", "mit_v2"}
        Hand 2's impedance mode. ``"mit"``: legacy MIT, any firmware; gains in firmware MIT units and a
        current limit in A. ``"mit_v2"``: firmware 3.1.0+ and wuji-sdk 2026.11.23+; gains in N·m/rad and
        N·m·s/rad and a torque limit in N·m, and all 20 joints must be online (``require_all_joints``).
    effort_limit_a : float | None
        Joint current limit, A. ``None``: 0.5 on Hand 1 (upstream's default); on Hand 2 in ``"mit"`` the
        robot's ``current_limit_a`` (1.5 for ``WujiHand2Beta2``, what its force limits were trained at),
        else 1.0. In ``"mit_v2"`` a set value becomes each joint's torque limit through its motor's
        ``TORQUE_CONSTANT``; ``None`` programs the robot's ``default_dofs_force_limits``, the ceiling the
        policy trained with.
    lowpass_cutoff_hz : float
        Hand 1's target low-pass cutoff.
    mit_kp, mit_kd : float | dict[str, float] | None
        Hand 2's gains in ``control_mode``'s units (the SDK's ``mit_params`` in either mode): one value
        for every joint, or a pattern-keyed dict covering all 20. ``None``: the payload's nominal gains,
        which in ``"mit"`` needs the robot's ``dofs_spec`` to map every joint to a ``WujiHand2Actuator``.
    require_all_joints : bool
        Hand 2: refuse a hand with fewer than 20 joints online.
    stop_on_fault : bool
        :meth:`check_hand` raises :class:`DeployStop` on a latched hand fault. Stale joint feedback
        stops the run regardless: a policy must never run on a frozen reading.
    stale_after_s : float
        Hand 2: joint feedback older than this is stale. ``connect_timeout_s`` bounds the wait for
        the first fresh reading at connect. Hand 1: the timeout of each joint-position read.
    vel_lowpass_hz : float | None
        First-order low-pass on the finite-difference joint velocity; ``None`` uses it raw.
    clip_targets_to_joint_limits : bool
        Clamp every sent target (and every :meth:`ease_to` goal) to the built robot's joint range.
    auto_yaw_align : bool
        Off: the hand has no IMU, so its reported base orientation is the fixed mount's and there
        is no yaw to align.
    """

    # Gains are programmed once at connect (see the class docstring), never per command.
    sends_command_gains: ClassVar[bool] = False
    # A command grades this tick's hand (and whatever sync_scene writes): see "Loop order".
    sync_before_commands: ClassVar[bool] = True
    # The scene is never stepped here, and action warm-ups and clip clocks read the episode clock.
    advances_episode_clock: ClassVar[bool] = True

    hand: Literal["hand1", "hand2"] | None = None
    serial_number: str = ""
    stream_mode: Literal["sync", "thread"] = "sync"
    servo_hz: float = 200.0
    control_mode: Literal["mit", "mit_v2"] = "mit"
    effort_limit_a: float | None = None
    lowpass_cutoff_hz: float = 3.0
    mit_kp: float | dict[str, float] | None = None
    mit_kd: float | dict[str, float] | None = None
    require_all_joints: bool = True
    stop_on_fault: bool = True
    stale_after_s: float = 0.5
    vel_lowpass_hz: float | None = None
    clip_targets_to_joint_limits: bool = False
    auto_yaw_align: bool = False

    def __init__(self, env: EnvBase, options: DeploymentOptions) -> None:
        self._transport: HandTransport | None = None
        self._connected = False
        super().__init__(env, options)
        self.reset_velocity_filter()

    def _resolve_options(self) -> None:
        """Resolve the side, the SDK index map, the generation, the gains and the target limits."""
        super()._resolve_options()
        owner = type(self).__name__
        robot = type(self.robot_options).__name__
        self._check_option_values()
        if self.num_dofs != NUM_JOINTS:
            raise ValueError(
                f"{owner} drives a {NUM_JOINTS}-joint Wuji hand; robot {robot} deploys {self.num_dofs} DOFs "
                f"{self.dofs_name}. Deploy the hand as its own entity (entity_name=...)."
            )
        self._dof_index = sdk_index_map(self.dofs_name, owner=f"{owner} (robot {robot})")
        #: The deployment's joint names in SDK order, for a transport's messages.
        self.sdk_joint_names = [self.dofs_name[k] for k in np.argsort(self._dof_index)]
        sides = {_JOINT_SIDE[name] for name in self.dofs_name}
        if len(sides) != 1:
            raise ValueError(f"{owner}: robot {robot} mixes right- and left-hand joints: {self.dofs_name}")
        self.side = sides.pop()

        if self.hand is None:
            if isinstance(self.robot_options, _WujiHand1):
                self.hand = "hand1"
            elif isinstance(self.robot_options, _WujiHand2Beta2):
                self.hand = "hand2"
            else:
                raise ValueError(
                    f"{owner} cannot tell which Wuji generation robot {robot} is. Name it: "
                    f"{owner}.configure(hand='hand1') for wujihandpy, or hand='hand2' for wuji_sdk."
                )
        if self.effort_limit_a is None and self.hand == "hand1":
            self.effort_limit_a = 0.5
        self._reject_unforwarded_gain_options()
        if self.hand == "hand2":
            motors = self._resolve_hand2_motors()
            if self.control_mode == "mit_v2":
                per_unit = np.ones(self.num_dofs)
                self._effort_limit = self._resolve_hand2_torque_limit(motors)
            else:
                per_unit = None if motors is None else np.array([motors[n].JOINT_GAIN(1.0) for n in self.dofs_name])
                self._effort_limit = self._resolve_hand2_current_limit()
            #: Per DOF, the joint-space gain of one unit of the programmed gain (``None``: not convertible).
            self._joint_gain_per_unit = per_unit
            self._mit_kp = self._resolve_gain(
                "mit_kp", self.mit_kp, None if per_unit is None else self.default_dof_kp / per_unit
            )
            self._mit_kd = self._resolve_gain(
                "mit_kd", self.mit_kd, None if per_unit is None else self.default_dof_kd / per_unit
            )
            self._warn_on_torque_ceiling(motors)

        self.stream = StreamOptions(mode=self.stream_mode, servo_hz=self.servo_hz, interp_s=self.control_dt)
        if self.stream_mode == "sync":
            self._rate = None  # command() already blocks for the period; see "Loop order"
        self._lower = self._upper = None
        if self.clip_targets_to_joint_limits:
            lower, upper = self._env.entities[self.entity_name].get_dofs_limit()
            self._lower, self._upper = lower[0].cpu().numpy(), upper[0].cpu().numpy()
        self._mount_quat = np.asarray(self.robot_options.default_root_quat, dtype=np.float64)

    def _check_option_values(self) -> None:
        """Refuse an option value the SDKs or the loop cannot act on (the ``Literal`` hints are not enforced)."""
        owner = type(self).__name__
        choices = {
            "hand": (None, "hand1", "hand2"),
            "stream_mode": ("sync", "thread"),
            "control_mode": tuple(_HAND2_MODES),
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{owner}: {name}={getattr(self, name)!r} is not one of {allowed}.")
        positive = {
            "servo_hz": self.servo_hz,
            "lowpass_cutoff_hz": self.lowpass_cutoff_hz,
            "stale_after_s": self.stale_after_s,
            "connect_timeout_s": self.connect_timeout_s,
        }
        for name, optional in (("effort_limit_a", self.effort_limit_a), ("vel_lowpass_hz", self.vel_lowpass_hz)):
            if optional is not None:
                positive[name] = optional
        for name, value in positive.items():
            if not (isinstance(value, (int, float)) and value > 0):
                raise ValueError(f"{owner}: {name} must be a positive number; got {value!r}.")

    def _reject_unforwarded_gain_options(self) -> None:
        """Refuse the options that act through per-command gains, which never reach a Wuji hand."""
        owner = type(self).__name__
        if self.init_dofs_kp_scale != 1.0 or self.init_dofs_kd_scale != 1.0:
            raise ValueError(
                f"{owner}: init_dofs_kp_scale={self.init_dofs_kp_scale:g} / init_dofs_kd_scale="
                f"{self.init_dofs_kd_scale:g} act through per-command gains, and a Wuji hand's gains are programmed "
                "once at connect. Leave both at 1.0; soften the INIT ramp with init_steps instead."
            )
        if self.disabled_dofs:
            raise ValueError(
                f"{owner}: disable_joints matched {list(self.disabled_dofs)}, but a Wuji hand cannot leave a single "
                "joint limp: disable_joints sends kp=kd=0 per command, and the hand's gains are programmed once at "
                "connect. Remove disable_joints."
            )
        if self.hand != "hand1":
            if self.control_mode == "mit_v2" and not self.require_all_joints:
                raise ValueError(
                    f"{owner}: control_mode='mit_v2' needs all 20 joints online (wuji_sdk's MIT_V2 setup), so it "
                    "cannot run with require_all_joints=False. Bring every joint online, or deploy the partial hand "
                    "with control_mode='mit'."
                )
            return
        if self.control_mode != "mit":
            raise ValueError(
                f"{owner}: control_mode={self.control_mode!r} selects a Wuji Hand 2 impedance mode, and Wuji Hand 1 "
                "runs its own position loop. Leave control_mode unset for Hand 1."
            )
        if self.deploy_kp_kd_scale != 1.0 or self.mit_kp is not None or self.mit_kd is not None:
            raise ValueError(
                f"{owner}: Wuji Hand 1 runs its own position loop (wujihandpy realtime controller), so gains cannot "
                f"reach it; got deploy_kp_kd_scale={self.deploy_kp_kd_scale:g}, mit_kp={self.mit_kp!r}, "
                f"mit_kd={self.mit_kd!r}. To soften it, lower the current limit instead: "
                f"{owner}.configure(effort_limit_a=0.3)."
            )

    def programmed_gains(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Hand 2's gains in joint space, a policy's units; ``None`` for Hand 1, or legacy MIT without specs."""
        if self.hand != "hand2" or self._joint_gain_per_unit is None:
            return None
        return self._mit_kp * self._joint_gain_per_unit, self._mit_kd * self._joint_gain_per_unit

    def _resolve_hand2_motors(self) -> dict[str, type[WujiHand2Actuator]] | None:
        """Each deployment DOF's ``WujiHand2Actuator`` spec from the robot's ``dofs_spec``.

        ``None`` when the robot maps no DOF to one; then legacy MIT can program only explicit ``mit_kp`` /
        ``mit_kd``.
        """
        owner, robot = type(self).__name__, type(self.robot_options).__name__
        motors = {}
        if spec := getattr(self.robot_options, "dofs_spec", None):
            _, names, values = resolve_matching_names_values(spec, self.dofs_name)
            motors = {n: m for n, m in zip(names, values) if isinstance(m, type) and issubclass(m, WujiHand2Actuator)}
        if not motors:
            if self.control_mode == "mit" and (self.mit_kp is None or self.mit_kd is None):
                raise ValueError(
                    f"{owner}: robot {robot} maps no joint to a WujiHand2Actuator spec in dofs_spec, so its "
                    "default_dofs_kp / default_dofs_kd cannot be converted to the hand's MIT units. Give both gains "
                    f"in MIT units, e.g. {owner}.configure(mit_kp=3.0, mit_kd=0.05), or map the robot's joints to "
                    "the eden.options.actuators.WujiHand2* specs, or on firmware 3.1.0+ program the joint-space gains "
                    "as they are with control_mode='mit_v2'. Sending the joint-space gains unconverted in MIT mode "
                    "would run the hand at other gains than the policy trained at."
                )
            return None
        if missing := [name for name in self.dofs_name if name not in motors]:
            raise ValueError(
                f"{owner}: robot {robot} maps joints {missing} to no WujiHand2Actuator spec in dofs_spec, while "
                f"{len(motors)} others have one; map every joint so each gain can be converted to MIT units."
            )
        return motors

    def _resolve_hand2_current_limit(self) -> np.ndarray:
        """Per deployment DOF, the current limit [A]: ``effort_limit_a``, else the robot's ``current_limit_a``, else 1.0.

        ``current_limit_a`` is the current the Wuji Hand 2 options derive ``default_dofs_force_limits`` from
        (``TORQUE_CONSTANT * current_limit_a``). :meth:`_warn_on_torque_ceiling` checks the result against the
        force limits the deployed robot actually carries.
        """
        if self.effort_limit_a is None:
            trained = getattr(self.robot_options, "current_limit_a", None)
            return np.full(self.num_dofs, 1.0 if trained is None else float(trained))
        return np.full(self.num_dofs, float(self.effort_limit_a))

    def _warn_on_torque_ceiling(self, motors: dict[str, type[WujiHand2Actuator]] | None) -> None:
        """Warn per joint where the hand's torque ceiling is not the force limit the policy trained with.

        The trained ceiling is the deployed robot's ``default_dofs_force_limits`` as the built entity resolved them:
        for a checkpoint replayed from its ``config.json``, the limits it trained at. The hand's ceiling is the
        programmed effort limit, in N·m under MIT_V2 and under legacy MIT the current times each motor's
        ``TORQUE_CONSTANT``; without motor specs legacy MIT's ceiling in N·m is unknown and nothing is checked.
        """
        if self.control_mode == "mit_v2":
            ceiling = self._effort_limit
        elif motors is not None:
            ceiling = self._effort_limit * np.array([motors[name].TORQUE_CONSTANT for name in self.dofs_name])
        else:
            return
        entity = self._env.entities[self.entity_name]
        trained = entity.get_default_dofs_force_limits(self.dofs_name)[0].cpu().numpy().astype(np.float64)
        differing = [
            f"{name} (trained {t:.3g} N·m, hand {c:.3g} N·m" + (f", {c / t:.0%})" if t > 0 else ")")
            for name, t, c in zip(self.dofs_name, trained, ceiling, strict=True)
            if not np.isclose(c, t, rtol=1e-3)
        ]
        if not differing:
            return
        robot = type(self.robot_options).__name__
        if self.effort_limit_a is not None:
            unset = "force limits" if self.control_mode == "mit_v2" else "current_limit_a"
            advice = (
                f"The hand gets effort_limit_a={self.effort_limit_a:g} A times each motor's TORQUE_CONSTANT. Leave "
                f"effort_limit_a unset to program the robot's {unset}."
            )
        else:  # legacy MIT at the robot's current_limit_a: the robot's force limits are not the ones it derives
            advice = (
                f"The hand gets the robot's current_limit_a ({self._effort_limit.min():g} A) times each motor's "
                "TORQUE_CONSTANT. A checkpoint whose config.json predates the Wuji Hand 2 calibration (5 N·m force "
                "limits, kp 50 / kd 5) trained on another hand model, and its gains convert to MIT gains far above "
                "the firmware's kp 3.0 / kd 0.05: retrain it on WujiHand2Beta2 before running it on the hand."
            )
        en.logger.warning(
            f"[wuji] The hand's torque ceiling is not the force limit {robot} trained with on {len(differing)} of "
            f"{self.num_dofs} joints: {first_few(differing)}. {advice}"
        )

    def _resolve_hand2_torque_limit(self, motors: dict[str, type[WujiHand2Actuator]] | None) -> np.ndarray:
        """MIT_V2's per-DOF torque limit [N·m].

        ``effort_limit_a`` times each joint's ``TORQUE_CONSTANT``; unset, the robot's ``default_dofs_force_limits``
        as the built entity resolved them (spec and asset fallbacks included).
        """
        if self.effort_limit_a is None:
            entity = self._env.entities[self.entity_name]
            return entity.get_default_dofs_force_limits(self.dofs_name)[0].cpu().numpy().astype(np.float64)
        if motors is None:
            raise ValueError(
                f"{type(self).__name__}: effort_limit_a={self.effort_limit_a:g} A cannot become an MIT_V2 torque limit "
                f"(N·m): robot {type(self.robot_options).__name__} maps no joint to a WujiHand2Actuator spec, so no "
                "joint has a torque constant. Leave effort_limit_a unset to program the robot's "
                "default_dofs_force_limits, or set those to the torque you want."
            )
        torque_constant = np.array([motors[n].TORQUE_CONSTANT for n in self.dofs_name])
        return self._resolve_hand2_current_limit() * torque_constant

    def _resolve_gain(
        self, name: str, value: float | dict[str, float] | None, nominal: np.ndarray | None
    ) -> np.ndarray:
        """One MIT gain per deployment DOF: ``None`` -> the nominal gains, a float -> every joint, a dict -> by pattern."""
        if value is None:
            return np.asarray(nominal, dtype=np.float64)
        if self.deploy_kp_kd_scale != 1.0:
            raise ValueError(
                f"{type(self).__name__}: {name} is set explicitly and deploy_kp_kd_scale="
                f"{self.deploy_kp_kd_scale:g} too; the scale applies only to the nominal gains. Scale {name} itself "
                "and leave deploy_kp_kd_scale at 1.0."
            )
        if isinstance(value, (int, float)):
            return np.full(self.num_dofs, float(value))
        matched = set(resolve_matching_names_values(value, self.dofs_name)[0])
        if missing := [n for i, n in enumerate(self.dofs_name) if i not in matched]:
            raise ValueError(
                f"{type(self).__name__}: {name} leaves joints {missing} without a gain; add a pattern for them "
                "(a Wuji hand has no default gain to fall back on)."
            )
        return resolve_dofs_values(value, self.dofs_name)

    # --- the SDK order ------------------------------------------------------------------------

    def to_sdk(self, dofs_values: np.ndarray) -> np.ndarray:
        """Scatter a deployment-order vector into the SDK's finger-major order."""
        out = np.empty(NUM_JOINTS)
        out[self._dof_index] = np.asarray(dofs_values, dtype=np.float64)
        return out

    def from_sdk(self, sdk_values: np.ndarray) -> np.ndarray:
        """Gather an SDK-order vector into the deployment DOF order."""
        return np.asarray(sdk_values, dtype=np.float64)[self._dof_index]

    # --- the transport ------------------------------------------------------------------------

    def use_transport(self, transport: HandTransport) -> None:
        """Drive ``transport`` (e.g. a simulator for sim-to-sim) instead of the hardware one :meth:`connect` builds."""
        owner = type(self).__name__
        if self._connected:
            raise RuntimeError(f"{owner}.use_transport must be called before connect().")
        if transport.stream.mode != self.stream_mode:
            kind = type(transport).__name__
            raise ValueError(
                f"{owner}.use_transport: {kind} streams {transport.stream.mode!r}, but the deployment is configured "
                f"with stream_mode={self.stream_mode!r} (which decides whether the base rate limiter paces the loop). "
                f"Build it with the deployment's StreamOptions: {kind}(dep.stream, ...)."
            )
        self._transport = transport

    @property
    def transport(self) -> HandTransport:
        if self._transport is None:
            raise RuntimeError(f"{type(self).__name__} is not connected: call connect() first.")
        return self._transport

    def _make_transport(self) -> HandTransport:
        if self.hand == "hand1":
            return Hand1Transport(
                self.stream,
                self.effort_limit_a,
                self.lowpass_cutoff_hz,
                self.sdk_joint_names,
                stale_after_s=self.stale_after_s,
            )
        gains = Hand2Gains(
            kp=self.to_sdk(self._mit_kp).tolist(),
            kd=self.to_sdk(self._mit_kd).tolist(),
            effort_limit=self.to_sdk(self._effort_limit).tolist(),
            control_mode=self.control_mode,
        )
        return Hand2Transport(
            self.stream,
            gains,
            self.serial_number,
            self.require_all_joints,
            feedback_timeout_s=self.connect_timeout_s,
            stale_after_s=self.stale_after_s,
            joint_names=self.sdk_joint_names,
        )

    # --- DeploymentBase interface --------------------------------------------------------------

    def connect(self) -> None:
        if self._transport is None:
            self._transport = self._make_transport()
        en.logger.info(f"[wuji] {self.side} Wuji {'Hand 1' if self.hand == 'hand1' else 'Hand 2'}")
        self._transport.connect()
        self._connected = True
        self.reset_velocity_filter()

    def close(self) -> None:
        try:
            self._teardown_common()
        finally:
            # De-energize the hand even if flushing the record or restoring a command raised.
            if self._transport is not None:
                self._transport.close()
            self._connected = False

    @property
    def blocks_for_the_period(self) -> bool:
        return self.stream_mode == "sync"

    def _prepare_reset(self) -> None:
        # A new episode, on the clock this backend advances (``advances_episode_clock``): warm-ups and
        # phase observations start over, and no velocity is differenced across the reset.
        self._env.episode_length_buf.zero_()
        self.reset_velocity_filter()

    def read_state(self) -> RobotState:
        q = self.from_sdk(self.transport.read())
        t = self.transport.clock()
        if self._prev_q is not None and t > self._prev_t:
            dt = t - self._prev_t
            vel = (q - self._prev_q) / dt
            if self.vel_lowpass_hz:
                a = dt / (dt + 1.0 / (2.0 * np.pi * self.vel_lowpass_hz))
                vel = a * vel + (1.0 - a) * self._vel
            self._vel = vel
        self._prev_q, self._prev_t = q.copy(), t
        return RobotState(
            stamp=t,
            base_quat=self._mount_quat,
            base_ang_vel=np.zeros(3),
            base_lin_acc=np.zeros(3),
            dofs_pos=q,
            dofs_vel=self._vel.copy(),
            dofs_torque=np.zeros(self.num_dofs),
        )

    def reset_velocity_filter(self) -> None:
        """Start the finite-difference velocity afresh from the next read (upstream does, entering a new phase)."""
        self._prev_q = self._prev_t = None
        self._vel = np.zeros(self.num_dofs)

    def send_payload(self, payload: RobotCommand) -> None:
        # The payload carries what is sent, so a debug record logs the filtered, clipped target.
        payload.dofs_pos = self._clip(self._apply_target_lpf(np.asarray(payload.dofs_pos, dtype=np.float64)))
        self.transport.command(self.to_sdk(payload.dofs_pos))

    def _clip(self, target: np.ndarray) -> np.ndarray:
        return target if self._lower is None else np.clip(target, self._lower, self._upper)

    # --- the control tick --------------------------------------------------------------------

    def check_hand(self) -> None:
        """Raise :class:`DeployStop` on stale joint feedback, or a latched hand fault (``stop_on_fault``)."""
        if stale := self.transport.stale():
            raise DeployStop(f"hand feedback lost: {stale}")
        if self.stop_on_fault and (fault := self.transport.fault()):
            raise DeployStop(f"hand fault: {fault}")

    def check_safety(self) -> None:
        """Run before every policy tick: :meth:`check_hand`. Subclasses extend it with their own guards."""
        self.check_hand()

    def sync_scene(self) -> None:
        """Write what else the run observes into the scene after the hand is read; a no-op here."""

    def _sync_state(self, state: RobotState) -> None:
        super()._sync_state(state)
        self.sync_scene()

    def observe(self) -> dict[str, Any]:
        """Read the hand and return the observation, advancing nothing."""
        self.state = self.read_state()
        return self.state_to_observation(self.state)

    # --- hooks of the base control tick (see "Loop order") -------------------------------------

    def _before_step(self) -> None:
        self.check_safety()

    def _idle_tick(self) -> None:
        self.transport.hold()  # in "sync" the send is the wait; nothing to send must not skip it

    def ease_to(self, target: np.ndarray, duration_s: float, rate_hz: float, profile: str = "cosine") -> None:
        """Move the hand from its measured pose to ``target`` (deployment DOF order), ``"cosine"`` or ``"smoothstep"``.

        :meth:`check_hand` runs before every waypoint. The eased path is not low-passed (it is smooth
        already), but ``target_lpf`` restarts from ``target``, so the next command filters from where
        the hand was eased to rather than pulling it back toward the target before the ease.
        """
        self.check_hand()
        en.logger.info(f"[wuji] easing the hand to its target over {duration_s:g} s ({profile})")
        target = self._clip(np.asarray(target, dtype=np.float64))
        self.transport.ease_to(self.to_sdk(target), duration_s, rate_hz, profile, guard=self.check_hand)
        if self.target_lpf < 1.0:
            self._lpf_prev = target.copy()


bind_deployment(_WujiHand1, "wuji_hand_deployment")
bind_deployment(_WujiHand2Beta2, "wuji_hand_deployment")
