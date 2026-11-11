"""Clutch-relative pose mapping: XR controller pose -> end-effector target.

Ported from Dream-Machines-Robotics/vr-teleop-kit (Apache-2.0), with the
upstream mujoco quaternion calls swapped for Genesis' numpy-backed wxyz helpers
(``genesis.utils.geom``) — no quaternion math is reimplemented here. The
algorithm and its hardware-tuned defaults are unchanged; see
``tests/utils/devices/metaquest/test_pose_mapping.py`` for the ported sanity checks (T1-T11).

``ClutchPoseMapper`` holds the "engage" state — the controller pose and the EE
pose captured at the moment the operator pressed the clutch (grip) button. While
the clutch is held, the EE target is the engaged EE pose composed with an
effective controller delta, rotated from the XR world frame into the arm's base
frame. On release the mapper disengages and ``target()`` returns ``None`` until
the next engage, so the operator can reposition their hand without moving the
robot (clutch-relative mapping, like lifting a mouse off the desk).

Reach limit (the interesting part). When the caller passes the arm's *current*
EE pose into ``target()``, the position/orientation channels become INCREMENTAL:
per-tick controller deltas accumulate into an effective delta, and anything past
``pos_reach_limit`` / ``rot_reach_limit`` of the current EE pose is absorbed
(a "slipping clutch"). This kills two failure modes the upstream authors hit on
hardware:

  * an absolute orientation map pushed far past a joint stop builds up the ~180°
    error where the shortest-way direction flips sign -> cap-speed shaking;
  * 350° CW and 10° CCW are the same orientation, so an absolute map eventually
    "snaps" the wrist around the other way. The incremental one never can.

The absorbed overshoot does not come back when the operator reverses (mouse at
the edge of the screen: reversing bites immediately). Trade-off: with
``scale_rotation != 1`` the per-increment gain is a rate gain, so curved hand
paths are path-dependent on SO(3); re-clutching realigns. ``0``/``None`` on
either reach limit restores the legacy absolute-delta-since-engage mapping.

Frames: all quaternions are wxyz. ``R`` maps XR-world vectors into the arm base
(``v_armbase = R @ v_xr``); for an Eden bridge that already publishes poses in
the Genesis/robot world frame, ``R`` is identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Quaternion math (numpy-backed, wxyz) comes straight from Genesis; nothing is
# reimplemented or wrapped here. One convention to keep in mind below:
# ``transform_quat_by_quat(v, u)`` returns ``u ⊗ v`` (it reads as "rotate
# orientation v by rotation u"), so a left-to-right Hamilton product ``a ⊗ b`` is
# ``transform_quat_by_quat(b, a)``. A scalar power of a rotation ("scale the
# angle by k") is ``rotvec_to_quat(k * quat_to_rotvec(q))``.
from genesis.utils.geom import (
    R_to_quat,
    inv_quat,
    quat_to_rotvec,
    rotvec_to_quat,
    transform_by_quat,
    transform_quat_by_quat,
)

__all__ = ["ClutchPoseMapper"]


# --------------------------------- the mapper ---------------------------------


@dataclass
class ClutchPoseMapper:
    """One-handed clutch-relative controller -> EE mapping.

    Parameters
    ----------
        R: 3x3 rotation taking XR-world vectors to arm-base vectors
            (``v_armbase = R @ v_xr``). Identity when poses already arrive in the
            robot world frame (the Eden metaquest bridge case).
        scale: linear gain on translation (1.0 = 1:1 motion).
        scale_rotation: gain on rotation delta (1.0 = 1:1; 0.5 halves the EE's
            angular response). Useful near a wrist singularity, where small
            operator twists otherwise drive large elbow corrections.
        rotation_pivot: optional 3-vector in arm-base frame. When set, controller
            rotation swings the EE about this pivot rather than rotating it in
            place. Typically captured per-engage by the caller (e.g. the FK elbow
            position). None = in-place rotation.
        rot_reach_limit: max angle (rad) the orientation target may run ahead of
            the arm's CURRENT orientation; enables the incremental/absorbing
            orientation path. Requires ``ee_quat_armbase`` in ``target()``.
            0/None disables (legacy absolute mapping).
        pos_reach_limit: max distance (m) the position target may run ahead of the
            arm's CURRENT EE position; same absorbing semantics. Requires
            ``ee_pos_armbase`` in ``target()``. 0/None disables.
    """

    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    scale: float = 1.0
    scale_rotation: float = 1.0
    rotation_pivot: np.ndarray | None = None
    rot_reach_limit: float | None = 0.6
    pos_reach_limit: float | None = 0.25

    def __post_init__(self) -> None:
        self._engaged: bool = False
        self._ctrl_engage_pos: np.ndarray | None = None
        self._ctrl_engage_quat: np.ndarray | None = None
        self._ee_engage_pos: np.ndarray | None = None
        self._ee_engage_quat: np.ndarray | None = None
        self._R_quat: np.ndarray = R_to_quat(np.asarray(self.R, float))
        self._R_quat_conj: np.ndarray = inv_quat(self._R_quat)
        # Incremental reach-limit state: previous-tick controller pose (XR frame)
        # and the accumulated effective deltas (arm-base frame). Reset per engage.
        self._ctrl_prev_quat: np.ndarray | None = None
        self._ctrl_prev_pos: np.ndarray | None = None
        self._d_quat_eff: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0])
        self._d_pos_eff: np.ndarray = np.zeros(3)

    @property
    def engaged(self) -> bool:
        return self._engaged

    def set_R(self, R: np.ndarray) -> None:
        """Replace the XR->arm-base rotation. Typically called per engage with a
        yaw-corrected R so the operator can turn their body between engagements and
        still have 'controller forward' mean 'robot forward'.
        """
        self.R = np.asarray(R, dtype=float).copy()
        self._R_quat = R_to_quat(self.R)
        self._R_quat_conj = inv_quat(self._R_quat)

    def engage(
        self,
        controller_pos_xr: np.ndarray,
        controller_quat_xr: np.ndarray,
        ee_pos_armbase: np.ndarray,
        ee_quat_armbase: np.ndarray,
        pivot_armbase: np.ndarray | None = None,
    ) -> None:
        """Capture the engage frame. Call on the rising clutch edge. If
        ``pivot_armbase`` is supplied, rotation deltas during this engagement
        pivot around that point instead of around the EE.
        """
        self._ctrl_engage_pos = np.array(controller_pos_xr, float, copy=True)
        self._ctrl_engage_quat = np.array(controller_quat_xr, float, copy=True)
        self._ee_engage_pos = np.array(ee_pos_armbase, float, copy=True)
        self._ee_engage_quat = np.array(ee_quat_armbase, float, copy=True)
        self.rotation_pivot = None if pivot_armbase is None else np.array(pivot_armbase, float, copy=True)
        self._ctrl_prev_quat = self._ctrl_engage_quat.copy()
        self._ctrl_prev_pos = self._ctrl_engage_pos.copy()
        self._d_quat_eff = np.array([1.0, 0.0, 0.0, 0.0])
        self._d_pos_eff = np.zeros(3)
        self._engaged = True

    def disengage(self) -> None:
        self._engaged = False

    def target(
        self,
        controller_pos_xr: np.ndarray,
        controller_quat_xr: np.ndarray,
        ee_pos_armbase: np.ndarray | None = None,
        ee_quat_armbase: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Compute the current EE target in arm-base frame, or None if disengaged.

        ``ee_pos_armbase`` / ``ee_quat_armbase`` are the arm's CURRENT EE pose;
        passing them enables the pos/rot reach limits (see the class docstring).
        Without them the legacy absolute mapping applies unchanged.
        """
        if not self._engaged:
            return None
        assert self._ctrl_engage_pos is not None
        assert self._ctrl_engage_quat is not None
        assert self._ee_engage_pos is not None
        assert self._ee_engage_quat is not None

        # Position delta. Reach-limited path accumulates per-tick increments
        # (vectors commute, so per-tick-scaled increments compose to exactly the
        # absolute scaled delta until the reach limit absorbs). Legacy path: the
        # absolute delta from the engage frame.
        p_now = np.asarray(controller_pos_xr, float)
        pos_limited = ee_pos_armbase is not None and bool(self.pos_reach_limit)
        if pos_limited:
            assert self._ctrl_prev_pos is not None
            self._d_pos_eff = self._d_pos_eff + self.R @ (self.scale * (p_now - self._ctrl_prev_pos))
            d_pos_arm = self._d_pos_eff
        else:
            d_pos_arm = self.R @ (self.scale * (p_now - self._ctrl_engage_pos))
        self._ctrl_prev_pos = p_now.copy()

        q_now = np.asarray(controller_quat_xr, float)
        rot_limited = ee_quat_armbase is not None and bool(self.rot_reach_limit)
        if rot_limited:
            # Incremental path: accumulate this tick's controller rotation
            # increment (a degree or two — never direction-ambiguous) into the
            # effective delta. At scale_rotation=1 the increments telescope to
            # exactly the absolute delta until the reach limit clamps; at other
            # scales it is a rate gain (path-dependent on curved SO(3) loops).
            assert self._ctrl_prev_quat is not None
            if float(np.dot(q_now, self._ctrl_prev_quat)) < 0.0:
                q_now = -q_now  # hemisphere-align: q and -q are the same rotation
            inc = transform_quat_by_quat(inv_quat(self._ctrl_prev_quat), q_now)  # q_now ⊗ prev⁻¹
            self._ctrl_prev_quat = q_now.copy()
            if self.scale_rotation != 1.0:
                inc = rotvec_to_quat(self.scale_rotation * quat_to_rotvec(inc))
            # express the XR-frame increment in arm frame: R ⊗ inc ⊗ R⁻¹
            inc_arm = transform_quat_by_quat(self._R_quat_conj, transform_quat_by_quat(inc, self._R_quat))
            d_quat_arm = transform_quat_by_quat(self._d_quat_eff, inc_arm)  # inc_arm ⊗ d_quat_eff
            d_quat_arm /= np.linalg.norm(d_quat_arm)
            self._d_quat_eff = d_quat_arm
        else:
            # Legacy absolute path: rotation delta in XR world (now * engage⁻¹),
            # conjugated by R_quat to express in arm base. Keep the incremental
            # state fresh so a live reach-limit toggle doesn't see stale prev.
            if self._ctrl_prev_quat is not None:
                if float(np.dot(q_now, self._ctrl_prev_quat)) < 0.0:
                    q_now = -q_now
                self._ctrl_prev_quat = q_now.copy()
            d_quat_xr = transform_quat_by_quat(inv_quat(self._ctrl_engage_quat), q_now)  # q_now ⊗ engage⁻¹
            if self.scale_rotation != 1.0:
                d_quat_xr = rotvec_to_quat(self.scale_rotation * quat_to_rotvec(d_quat_xr))
            # R ⊗ d_quat_xr ⊗ R⁻¹
            d_quat_arm = transform_quat_by_quat(self._R_quat_conj, transform_quat_by_quat(d_quat_xr, self._R_quat))

        target_quat = transform_quat_by_quat(self._ee_engage_quat, d_quat_arm)  # d_quat_arm ⊗ ee_engage

        if rot_limited:
            # Clamp the target to within rot_reach_limit of the arm's current
            # orientation and ABSORB the excess into the effective delta.
            ee_q = np.asarray(ee_quat_armbase, float)
            e = quat_to_rotvec(transform_quat_by_quat(inv_quat(ee_q), target_quat))  # target ⊗ ee⁻¹
            e_norm = float(np.linalg.norm(e))
            if e_norm > self.rot_reach_limit:
                e *= self.rot_reach_limit / e_norm
                target_quat = transform_quat_by_quat(ee_q, rotvec_to_quat(e))  # rotvec_to_quat(e) ⊗ ee
                self._d_quat_eff = transform_quat_by_quat(inv_quat(self._ee_engage_quat), target_quat)
                d_quat_arm = self._d_quat_eff

        # A pivot turns the rotation into a swing of the EE around it; with
        # rotation_pivot None the offset term vanishes (legacy in-place rotation).
        if self.rotation_pivot is not None:
            offset = self._ee_engage_pos - self.rotation_pivot
            target_pos = self.rotation_pivot + transform_by_quat(offset, d_quat_arm) + d_pos_arm
        else:
            target_pos = self._ee_engage_pos + d_pos_arm

        # Position reach limit: clamp toward the current EE position and ABSORB
        # the excess (mouse-at-screen-edge: reversal moves the target immediately
        # instead of after retracing the overshoot).
        if pos_limited:
            ee_p = np.asarray(ee_pos_armbase, float)
            dp = target_pos - ee_p
            dp_norm = float(np.linalg.norm(dp))
            if dp_norm > self.pos_reach_limit:
                clamped = ee_p + dp * (self.pos_reach_limit / dp_norm)
                self._d_pos_eff = self._d_pos_eff + (clamped - target_pos)
                target_pos = clamped

        return target_pos, target_quat
