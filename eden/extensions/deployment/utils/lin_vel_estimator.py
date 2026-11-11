"""Contact-aided linear Kalman filter estimating base linear velocity at deploy time.

- **State** ``x = [p (3), v (3), p_foot_i (3 each)]`` — base position, base linear
  velocity and per-foot contact-point positions, all in the (yaw-aligned) world frame.
- **Prediction** integrates the IMU specific force rotated to world plus gravity.
- **Correction** uses leg-kinematics odometry: per foot, the FK foot position and
  velocity relative to the base (computed with the base pinned at zero velocity)
  and a "feet are on the ground" height pseudo-measurement.
- **Contact gating**: the measurement/process noise of a foot not in contact is
  inflated by ``high_suspect_number`` (reference value 100), so swing feet barely
  pull the filter while stance feet anchor it.

Contact is detected from the quasi-static contact load each leg carries, solved from the
motors' ``tau_est`` through the foot Jacobian (the lightweight stand-in for the
generalized-momentum observer used by BeyondMimic) and thresholded with hysteresis on its
vertical component. Two terms make that load the *true* one rather than a posture-dependent
apparent one:

- **The foot's moment.** A leg with six or more DOFs ends in a flat foot, which carries a moment
  as well as a force; the solve is for the full 6-D wrench ``w = -(J J^T)^-1 J tau``. Solving for
  a 3-D point force instead attributes the moment the ankle motors hold to a force, and in a
  straight-knee stance, where the ankle-pitch motors carry the moment, that reads a loaded foot
  as *negative* load: the G1 read -190 to -380 N on feet carrying its weight. A leg with fewer
  DOFs (a quadruped's point foot) cannot hold a moment, so its solve stays 3-D.
- **The leg's own weight.** ``tau`` is gravity-compensated with the leg links' gravity torque
  at the measured pose and orientation, so a stance foot reads its share of the robot's weight
  and a swing foot reads about zero.

In a MuJoCo straight-knee stance this reads 112.7 N against 112.1 N of true contact force;
the 3-D solve without compensation read 32 N, below the latch threshold.

The 6-D solve has nowhere to put torque that is not contact load, so the torque accelerating a
swinging leg reads as load too (a Gangnam-style swing foot reads a median 38 N, p90 100 N). A
**height gate** rejects it: a loaded foot more than ``max_contact_height`` above another loaded
foot is swinging, not standing. A raised foot with no loaded foot below it -- the upper foot of a
stair -- is kept, but only while it is nearly stationary in the world (``max_raised_foot_speed``),
which a swinging foot is not. Scored against MuJoCo's own contacts over a closed-loop
BeyondMimic dance, per-foot agreement goes from 0.68 (3-D solve) to 0.90, missing 8% of stance
where the 3-D solve missed 39%.

The thresholds default to a fraction of the robot's weight (``contact_force_threshold`` is 12%
of ``m g``, about 39 N on a G1), since the load they are compared against is the true one.

Two guards bound the case where the load model is still wrong:

- **Support gate.** A robot whose accelerometer reads a sizeable fraction of ``g`` in specific
  force is not in ballistic flight. When for ``support_gate_ticks`` ticks in a row no foot has been
  latched while the IMU read support (``|a| > support_accel_ratio * g``), the most-loaded foot is
  latched anyway -- provided it reads at least ``contact_release_threshold`` *and* is nearly
  still relative to the base (``max_raised_foot_speed``). A robot hanging from a hoist reads 1 g
  on the IMU with both feet in the air; a dangling leg reads no load, and one swung by a policy
  reads its swing dynamics as load but is moving, so neither is latched and the robot falls
  through to the fallback below rather than anchoring the estimate on a free leg. Stillness is
  judged relative to the base, not in the world: the gate only runs when no foot anchors the
  estimate, and judged under the estimate a latched foot would drag it along until the foot
  looked still, locking itself in. The gate is for a robot the load model has misread while it
  stands, whose base is near-still anyway.
- **No-contact fallback.** When no foot has been in contact for ``max_no_contact_time`` the
  estimate stops integrating the IMU open loop: the velocity decays toward zero with time
  constant ``no_contact_velocity_decay``, and leg odometry takes over again the moment a foot
  lands. Its process noise is deliberately *not* inflated: that would hand the odometry of the
  still-unlatched feet the gain, and a leg swinging free under a hoist would drag the estimate
  (0.18 m/s from a leg swung at 0.8 m/s). One warning names the duration, so an IMU-only estimate
  is never silent.

Base position is tracked internally (z is observable through the foot-height
pseudo-measurement, xy slowly drifts) but only the velocity is exported.

Overhead per control tick: two ``get_links_pos`` (link origins and leg centres of mass) +
one ``get_jacobian`` per foot on the deployment's sim entity (FK already synced from real
state) and a 12-state / 14-measurement numpy Kalman update — well under a millisecond.

All math runs in numpy float64 on the CPU; quaternions are wxyz.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from genesis.utils.geom import quat_to_R

import eden as en

if TYPE_CHECKING:
    from eden.entities.rigid import RigidEntity

# Default contact latch threshold as a fraction of the robot's weight: about 39 N on a 33 kg G1, the
# value the G1 was validated at, and in proportion on a lighter or heavier robot.
_CONTACT_WEIGHT_FRACTION = 0.12


class LinVelKalmanEstimator:
    """Estimate base linear velocity from IMU + leg kinematics + joint torques.

    Parameters
    ----------
    entity : RigidEntity
        The deployment's *sim* entity (the kinematics oracle the
        ``RobotStateEntity`` wrapper keeps synced with real robot state).
        Used for FK reads (``get_links_pos`` / ``get_jacobian``) and to
        resolve the leg kinematic chains at construction time.
    feet_links : sequence of str
        Foot link names (e.g. ``("left_ankle_roll_link", "right_ankle_roll_link")``).
    dt : float
        Filter update period in seconds (the deployment control dt).
    contact_force_threshold : float | None
        Vertical contact force (N) above which a foot is latched *in contact*. The force is the
        gravity-compensated load the foot carries, so a stance foot reads its share of the
        robot's weight and a swing foot about zero. Defaults to 12% of the robot's weight
        (about 39 N on a G1).
    contact_release_threshold : float | None
        Vertical contact force (N) below which a latched foot is released
        (hysteresis). Defaults to half of ``contact_force_threshold``.
    foot_sole_offset : float
        Vertical distance (m) from the foot link origin down to the sole /
        contact point. Only biases the internal height states; the exported
        velocity is insensitive to it.
    imu_process_noise_pos, imu_process_noise_vel, foot_process_noise_pos,
    foot_sensor_noise_pos, foot_sensor_noise_vel, foot_height_sensor_noise,
    high_suspect_number :
        Filter noise parameters; defaults are the ``legged_control``
        reference values.
    gravity : float
        Gravitational acceleration magnitude (m/s^2).
    max_contact_height : float | None
        A loaded foot more than this far (m) above another loaded foot is not latched, which
        rejects the swing-leg dynamics the 6-D solve reads as load. ``None`` disables the gate.
    max_raised_foot_speed : float
        Speed (m/s) a foot must stay below to be latched without the load model's full say: a
        raised foot with no loaded foot beneath it (a stance foot on a higher step), judged in the
        world, and the foot the support gate picks, judged relative to the base. A standing foot
        is still; a swinging one is not.
    support_accel_ratio : float
        Fraction of ``gravity`` the accelerometer's specific force must exceed for the robot to
        count as supported by the support gate. ``0`` or less disables the gate.
    support_gate_ticks : int
        Consecutive supported ticks, with no foot latched, before the gate latches the most-loaded
        foot (if it reads at least ``contact_release_threshold`` and is nearly still relative to
        the base).
    max_no_contact_time : float
        Seconds with no foot in contact after which the no-contact fallback engages.
    no_contact_velocity_decay : float
        Time constant (s) of the velocity's decay toward zero while the fallback is engaged.

    Notes
    -----
    - Works in whatever world frame the supplied quaternion lives in. The
      deployment feeds the *yaw-aligned* quaternion the policy observes, so
      the exported velocity is directly in the policy's world frame. When
      ``yaw_tag`` changes between updates (yaw offset re-captured at START),
      the filter re-initializes from FK — the robot is standing at that
      handover, so dropping the old-frame state is safe.
    - The IMU accelerometer is assumed to measure specific force (reads
      ``+g`` on z at rest), the Unitree convention.
    """

    def __init__(
        self,
        entity: RigidEntity,
        feet_links: tuple[str, ...] | list[str],
        dt: float,
        *,
        contact_force_threshold: float | None = None,
        contact_release_threshold: float | None = None,
        foot_sole_offset: float = 0.03,
        imu_process_noise_pos: float = 0.02,
        imu_process_noise_vel: float = 0.02,
        foot_process_noise_pos: float = 0.002,
        foot_sensor_noise_pos: float = 0.005,
        foot_sensor_noise_vel: float = 0.1,
        foot_height_sensor_noise: float = 0.01,
        high_suspect_number: float = 100.0,
        gravity: float = 9.81,
        max_contact_height: float | None = 0.05,
        max_raised_foot_speed: float = 0.5,
        support_accel_ratio: float = 0.5,
        support_gate_ticks: int = 3,
        max_no_contact_time: float = 0.5,
        no_contact_velocity_decay: float = 0.5,
    ) -> None:
        if not feet_links:
            raise ValueError("LinVelKalmanEstimator needs at least one foot link name.")
        self._entity = entity
        self._dt = float(dt)
        self._gravity = float(gravity)
        self._sole_offset = float(foot_sole_offset)
        self._high_suspect = float(high_suspect_number)
        if support_gate_ticks < 1:
            raise ValueError(f"support_gate_ticks must be >= 1, got {support_gate_ticks}.")
        if max_contact_height is not None and max_contact_height <= 0.0:
            raise ValueError(
                f"max_contact_height must be a positive height in metres (or None to disable the gate), got "
                f"{max_contact_height}."
            )
        if max_raised_foot_speed <= 0.0:
            raise ValueError(f"max_raised_foot_speed must be a positive speed in m/s, got {max_raised_foot_speed}.")
        if max_no_contact_time <= 0.0 or no_contact_velocity_decay <= 0.0:
            raise ValueError(
                "max_no_contact_time and no_contact_velocity_decay must be positive seconds, got "
                f"{max_no_contact_time} and {no_contact_velocity_decay}."
            )
        self._max_contact_height = None if max_contact_height is None else float(max_contact_height)
        self._max_raised_speed = float(max_raised_foot_speed)
        self._support_accel = float(support_accel_ratio) * self._gravity
        self._support_gate_ticks = int(support_gate_ticks)
        self._max_no_contact_ticks = max(1, round(float(max_no_contact_time) / self._dt))
        self._no_contact_decay = float(np.exp(-self._dt / float(no_contact_velocity_decay)))

        self._feet_names = list(feet_links)
        self._foot_links = [entity.get_link(name) for name in self._feet_names]
        chains = [self._resolve_leg_chain(entity, name) for name in self._feet_names]
        self._leg_dofs = [dofs for dofs, _ in chains]
        _, ls_idx = entity.find_named_links_idx_local([entity.base_link_name, *self._feet_names], preserve_order=True)
        self._fk_links_idx = ls_idx
        # A leg with six or more DOFs ends in a flat foot that carries a moment as well as a
        # force, so its contact load is solved as a 6-D wrench; fewer DOFs cannot hold one.
        self._wrench_rows = [6 if len(dofs) >= 6 else 3 for dofs in self._leg_dofs]

        # Leg gravity compensation: every link distal to a leg's first joint, its mass, and
        # which of the leg's joints each one hangs from (a joint carries the weight of the
        # whole subtree below it).
        self._grav_links = sorted({li for _, distal in chains for d in distal for li in d})
        col = {li: c for c, li in enumerate(self._grav_links)}
        self._distal = []
        for _, distal in chains:
            mask = np.zeros((len(distal), len(self._grav_links)))
            for j, links in enumerate(distal):
                mask[j, [col[li] for li in links]] = 1.0
            self._distal.append(mask)
        masses = entity.get_links_inertial_mass()[0].double().cpu().numpy()
        self._grav_force = np.outer(masses[self._grav_links], [0.0, 0.0, -self._gravity])  # (n_links, 3)

        if contact_force_threshold is None:
            contact_force_threshold = _CONTACT_WEIGHT_FRACTION * float(masses.sum()) * self._gravity
        self._contact_on = float(contact_force_threshold)
        self._contact_off = (
            float(contact_release_threshold) if contact_release_threshold is not None else 0.5 * self._contact_on
        )

        nf = len(self._feet_names)
        ns = 6 + 3 * nf  # state: base pos, base vel, foot contact points
        no = 6 * nf + nf  # obs: per-foot rel pos, rel vel, foot height
        self._nf, self._ns, self._no = nf, ns, no

        dt2 = self._dt * self._dt
        self._A = np.eye(ns)
        self._A[0:3, 3:6] = self._dt * np.eye(3)
        self._B = np.zeros((ns, 3))
        self._B[0:3, :] = 0.5 * dt2 * np.eye(3)
        self._B[3:6, :] = self._dt * np.eye(3)

        C = np.zeros((no, ns))
        for i in range(nf):
            C[3 * i : 3 * i + 3, 0:3] = np.eye(3)  # p - p_fi
            C[3 * i : 3 * i + 3, 6 + 3 * i : 9 + 3 * i] = -np.eye(3)
            C[3 * nf + 3 * i : 3 * nf + 3 * i + 3, 3:6] = np.eye(3)  # v
            C[6 * nf + i, 6 + 3 * i + 2] = 1.0  # foot height
        self._C = C

        q = np.eye(ns)
        q[0:3, 0:3] *= (self._dt / 20.0) * imu_process_noise_pos
        q[3:6, 3:6] *= (self._dt * self._gravity / 20.0) * imu_process_noise_vel
        q[6:, 6:] *= self._dt * foot_process_noise_pos
        self._Q_template = q

        r = np.eye(no)
        r[0 : 3 * nf] *= foot_sensor_noise_pos
        r[3 * nf : 6 * nf] *= foot_sensor_noise_vel
        r[6 * nf :] *= foot_height_sensor_noise
        self._R_template = r

        # Per-foot exported diagnostics (updated every ``update`` call). ``contacts`` is what the
        # filter used, support gate included; ``forces`` is the solved contact force.
        self.contacts = np.zeros(nf, dtype=bool)
        self.forces = np.zeros((nf, 3))
        self._force_contacts = np.zeros(nf, dtype=bool)  # the hysteresis state, gate excluded
        self._unlatched_supported_ticks = 0
        self._no_contact_ticks = 0
        self._fallback_warned = False

        self._x = np.zeros(ns)
        self._P = 100.0 * np.eye(ns)
        self._initialized = False
        self._yaw_tag: float | None = None

    @staticmethod
    def _resolve_leg_chain(entity: RigidEntity, foot_link_name: str) -> tuple[list[int], list[list[int]]]:
        """Resolve the joints on the foot→base chain, and the links each one carries.

        Returns ``(dofs, distal)``: the chain's DOF indices in dofs_name order, base to foot,
        and for each of those joints the local indices of every link in the subtree it moves
        (its own link and everything below it).
        """
        gs_entity = entity._entity
        link_by_idx = {lk.idx: lk for lk in gs_entity.links}
        base_idx = gs_entity.links[0].idx
        children: dict[int, list[int]] = {}
        for lk in gs_entity.links:
            children.setdefault(lk.parent_idx, []).append(lk.idx)

        def subtree(idx: int) -> list[int]:
            out, stack = [], [idx]
            while stack:
                i = stack.pop()
                out.append(i - base_idx)  # local link index
                stack.extend(children.get(i, ()))
            return out

        lk = gs_entity.get_link(foot_link_name)
        joint_names: list[str] = []
        distal: list[list[int]] = []
        while lk.idx != base_idx:
            for joint in lk.joints:
                if joint.n_dofs == 0:
                    continue
                if joint.n_dofs != 1:
                    raise ValueError(
                        f"Leg chain joint '{joint.name}' has {joint.n_dofs} DOFs; "
                        "the velocity estimator only supports 1-DOF leg joints."
                    )
                joint_names.append(joint.name)
                distal.append(subtree(lk.idx))
            lk = link_by_idx[lk.parent_idx]
        joint_names.reverse()
        distal.reverse()
        missing = [n for n in joint_names if n not in entity.dofs_name]
        if missing:
            raise ValueError(f"Leg chain joints {missing} of '{foot_link_name}' are not in the entity's dofs_name.")
        # Each chain joint has one DOF (checked above), so its column is its one position in ``dofs_name``.
        return [entity.dofs_name.index(n) for n in joint_names], distal

    # ------------------------------------------------------------------
    # FK helpers (read from the sim entity, already synced with real state)
    # ------------------------------------------------------------------

    def _read_kinematics(
        self, ang_vel_world: np.ndarray, dofs_vel: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
        """Per-foot position/velocity relative to the base (world frame, base pinned).

        Returns ``(rel_pos (nf,3), rel_vel (nf,3), leg_jacobians, feet_pos (nf,3))`` where
        ``rel_vel`` is the foot velocity with the base linear velocity zeroed:
        ``J_leg @ dq_leg + w x r`` — exactly the ``eeVel`` term of the
        reference filter. Each leg Jacobian is ``(6, n_leg)``, linear rows first.
        """
        links_pos = self._entity.get_links_pos(ls_idx_local=self._fk_links_idx)
        links_pos = links_pos[0].double().cpu().numpy()  # (1 + nf, 3)
        rel_pos = links_pos[1:] - links_pos[0]

        rel_vel = np.zeros_like(rel_pos)
        jacobians: list[np.ndarray] = []
        for i, link in enumerate(self._foot_links):
            J = self._entity.get_jacobian(link)
            if J.dim() == 3:
                J = J[0]
            J_leg = J[:, 6:][:, self._leg_dofs[i]].double().cpu().numpy()  # (6, n_leg)
            jacobians.append(J_leg)
            rel_vel[i] = J_leg[:3] @ dofs_vel[self._leg_dofs[i]] + np.cross(ang_vel_world, rel_pos[i])
        return rel_pos, rel_vel, jacobians, links_pos[1:]

    def _leg_gravity_torques(self, jacobians: list[np.ndarray], feet_pos: np.ndarray) -> list[np.ndarray]:
        """Generalized gravity force on each leg's joints at the current pose.

        For joint ``j`` with world axis ``a`` through ``o``, the weight ``m g`` of a link it
        carries at centre of mass ``c`` contributes ``m g . (a x (c - o))``. Written about the
        foot origin ``p``, where the Jacobian was taken, that is
        ``m g . (J_lin[:, j] + J_ang[:, j] x (c - p))`` — so the foot Jacobian already holds
        every axis and lever arm needed, prismatic joints (``J_ang = 0``) included.
        """
        com = self._entity.get_links_pos(ls_idx_local=self._grav_links, ref="link_com")
        com = com[0].double().cpu().numpy()  # (n_links, 3)
        torques = []
        for i, J_leg in enumerate(jacobians):
            weight = self._distal[i] @ self._grav_force  # (n_leg, 3): weight each joint carries
            moment = self._distal[i] @ np.cross(com - feet_pos[i], self._grav_force)  # about the foot origin
            torques.append(np.einsum("kj,jk->j", J_leg[:3], weight) + np.einsum("kj,jk->j", J_leg[3:], moment))
        return torques

    def _detect_contacts(
        self,
        jacobians: list[np.ndarray],
        dofs_torque: np.ndarray,
        gravity_torques: list[np.ndarray] | None = None,
        rel_pos: np.ndarray | None = None,
        feet_speed: np.ndarray | None = None,
    ) -> None:
        """Latch per-foot contact from the vertical contact force, with hysteresis.

        With ``rel_pos`` (foot positions relative to the base) and ``feet_speed`` (world-frame
        foot speeds) the height gate applies: see ``max_contact_height``.
        """
        loaded = np.zeros(self._nf, dtype=bool)
        for i, J_leg in enumerate(jacobians):
            tau_leg = dofs_torque[self._leg_dofs[i]]
            if gravity_torques is not None:
                tau_leg = tau_leg + gravity_torques[i]
            # Quasi-static equilibrium of the leg: tau_motor + tau_gravity = -J^T w, solved for
            # the contact wrench w (force, then moment about the foot origin) via the normal
            # equations. A flat foot's moment must be in the solve: leaving it out reads the
            # moment the ankles hold as a force.
            J = J_leg[: self._wrench_rows[i]]
            JJt = J @ J.T + 1e-9 * np.eye(J.shape[0])
            self.forces[i] = -np.linalg.solve(JJt, J @ tau_leg)[:3]
            loaded[i] = self.forces[i][2] > (self._contact_off if self._force_contacts[i] else self._contact_on)
        latched = loaded.copy()
        if rel_pos is not None and self._max_contact_height is not None:
            z = rel_pos[:, 2]
            for i in np.flatnonzero(loaded):
                below = loaded & (z < z[i] - self._max_contact_height)
                if below.any():
                    latched[i] = False  # a loaded foot beneath it is the one standing
                elif (z[i] - z.min() > self._max_contact_height) and feet_speed is not None:
                    latched[i] = feet_speed[i] < self._max_raised_speed  # a higher step, if it is still
        self._force_contacts[:] = latched
        self.contacts[:] = latched

    def _gate_support(self, lin_acc_body: np.ndarray, rel_speed: np.ndarray | None = None) -> None:
        """Latch the most-loaded foot when the IMU has read support with no foot latched.

        The foot must read at least the release threshold and, given ``rel_speed`` (each foot's
        speed relative to the base), be nearly still: a leg swinging free under a hoist reads its
        swing dynamics as load.
        """
        supported = np.linalg.norm(lin_acc_body) > self._support_accel > 0.0
        if self.contacts.any() or not supported:
            self._unlatched_supported_ticks = 0
            return
        self._unlatched_supported_ticks += 1
        if self._unlatched_supported_ticks >= self._support_gate_ticks:
            i = int(np.argmax(self.forces[:, 2]))
            still = rel_speed is None or rel_speed[i] < self._max_raised_speed
            if self.forces[i, 2] >= self._contact_off and still:
                self.contacts[i] = True

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------

    def _initialize(self, rel_pos: np.ndarray) -> None:
        """Anchor the state so the lowest foot contact point sits at z = 0."""
        self._x = np.zeros(self._ns)
        self._x[2] = -np.min(rel_pos[:, 2]) + self._sole_offset
        for i in range(self._nf):
            self._x[6 + 3 * i : 9 + 3 * i] = self._x[0:3] + rel_pos[i]
            self._x[6 + 3 * i + 2] -= self._sole_offset
        self._P = 100.0 * np.eye(self._ns)
        self._initialized = True

    def reset(self) -> None:
        """Drop the filter state; the next ``update`` re-initializes from FK."""
        self._initialized = False
        self._force_contacts[:] = False
        self._unlatched_supported_ticks = 0
        self._no_contact_ticks = 0
        self._fallback_warned = False

    def update(
        self,
        quat_wxyz: np.ndarray | torch.Tensor,
        ang_vel_body: np.ndarray,
        lin_acc_body: np.ndarray,
        dofs_vel: np.ndarray,
        dofs_torque: np.ndarray,
        yaw_tag: float | None = None,
    ) -> np.ndarray:
        """Run one predict/correct cycle; returns the world-frame base linear velocity (3,).

        ``quat_wxyz`` must be the base orientation in the same world frame the
        velocity should be expressed in (the deployment passes the yaw-aligned
        quaternion). ``dofs_vel`` / ``dofs_torque`` are in the entity's
        dofs_name order. ``yaw_tag`` is an opaque marker for that frame: when
        it changes between calls the filter re-initializes.
        """
        if isinstance(quat_wxyz, torch.Tensor):
            quat_wxyz = quat_wxyz.reshape(-1).double().cpu().numpy()
        quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
        ang_vel_body = np.asarray(ang_vel_body, dtype=np.float64).reshape(3)
        lin_acc_body = np.asarray(lin_acc_body, dtype=np.float64).reshape(3)
        dofs_vel = np.asarray(dofs_vel, dtype=np.float64).reshape(-1)
        dofs_torque = np.asarray(dofs_torque, dtype=np.float64).reshape(-1)

        if yaw_tag is not None and yaw_tag != self._yaw_tag:
            self._yaw_tag = yaw_tag
            self._initialized = False

        # genesis `quat_to_R` self-normalises the quaternion and has a numpy path.
        # `quat_wxyz` is already float64 (..., 4) from the coercion above.
        R = quat_to_R(quat_wxyz)
        ang_vel_world = R @ ang_vel_body
        rel_pos, rel_vel, jacobians, feet_pos = self._read_kinematics(ang_vel_world, dofs_vel)
        # World-frame foot speed under the previous estimate of the base velocity: near zero for a
        # standing foot, whatever the terrain. Before the filter (re)initializes -- first tick, after
        # ``reset``, or on a yaw re-capture -- that estimate is absent or in the old frame, so the
        # base is taken as still, which it is at those handovers.
        base_vel = self._x[3:6] if self._initialized else np.zeros(3)
        feet_speed = np.linalg.norm(base_vel + rel_vel, axis=1)
        gravity_torques = self._leg_gravity_torques(jacobians, feet_pos)
        self._detect_contacts(jacobians, dofs_torque, gravity_torques, rel_pos, feet_speed)
        self._gate_support(lin_acc_body, np.linalg.norm(rel_vel, axis=1))
        fallback = self._track_no_contact()

        if not self._initialized:
            self._initialize(rel_pos)

        nf, ns = self._nf, self._ns
        accel_world = R @ lin_acc_body + np.array([0.0, 0.0, -self._gravity])

        Q = self._Q_template.copy()
        Rm = self._R_template.copy()
        for i in range(nf):
            if not self.contacts[i]:
                sl = slice(6 + 3 * i, 9 + 3 * i)
                Q[sl, sl] *= self._high_suspect
                sl_p = slice(3 * i, 3 * i + 3)
                sl_v = slice(3 * nf + 3 * i, 3 * nf + 3 * i + 3)
                Rm[sl_p, sl_p] *= self._high_suspect
                Rm[sl_v, sl_v] *= self._high_suspect
                Rm[6 * nf + i, 6 * nf + i] *= self._high_suspect

        # Measurement vector: ps (base minus foot), vs (base vel from leg odom), foot heights.
        y = np.zeros(self._no)
        for i in range(nf):
            y[3 * i : 3 * i + 3] = -rel_pos[i]
            y[3 * i + 2] += self._sole_offset
            y[3 * nf + 3 * i : 3 * nf + 3 * i + 3] = -rel_vel[i]
            # y[6*nf + i] stays 0: feet-on-ground pseudo-measurement.

        # Predict. With no foot down for too long, stop integrating the IMU open loop and let
        # the velocity decay toward zero instead, so accelerometer noise and bias cannot run
        # it away.
        x = self._A @ self._x + self._B @ accel_world
        if fallback:
            x[3:6] = self._no_contact_decay * self._x[3:6]
        Pm = self._A @ self._P @ self._A.T + Q

        # Correct.
        ey = y - self._C @ x
        S = self._C @ Pm @ self._C.T + Rm
        s_ey = np.linalg.solve(S, ey)
        self._x = x + Pm @ self._C.T @ s_ey
        s_C = np.linalg.solve(S, self._C)
        self._P = (np.eye(ns) - Pm @ self._C.T @ s_C) @ Pm
        self._P = 0.5 * (self._P + self._P.T)

        return self._x[3:6].copy()

    def _track_no_contact(self) -> bool:
        """Count ticks with no foot in contact; ``True`` once the no-contact fallback engages."""
        if self.contacts.any():
            if self._fallback_warned:
                en.logger.info(
                    f"[deploy] estimate_lin_vel: foot contact regained after {self._no_contact_ticks * self._dt:.2f} s "
                    "without; leg odometry is correcting the estimate again."
                )
                self._fallback_warned = False
            self._no_contact_ticks = 0
            return False
        self._no_contact_ticks += 1
        if self._no_contact_ticks < self._max_no_contact_ticks:
            return False
        if not self._fallback_warned:
            self._fallback_warned = True
            en.logger.warning(
                f"[deploy] estimate_lin_vel: no foot in contact for {self._no_contact_ticks * self._dt:.2f} s, and "
                "the support gate has not latched one, so base_lin_vel has no leg odometry behind it. It now decays "
                "toward zero instead of integrating the accelerometer open loop. Expected only while the robot is "
                "airborne or lifted; on the ground it means contact detection is wrong -- record with --debug_record "
                "and check x_forces / x_contacts."
            )
        return True

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def pos(self) -> np.ndarray:
        """Internal base-position state (z observable, xy drifts). Debug only."""
        return self._x[0:3].copy()

    @property
    def vel(self) -> np.ndarray:
        return self._x[3:6].copy()
