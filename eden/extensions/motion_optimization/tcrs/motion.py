"""Motion clip container, footstep resolution, and terrain-adapted reference trajectories.

:class:`MotionClip` wraps a mocap segment (joint positions plus world-frame body
poses) and derives per-foot contact masks — an adaptive percentile z-threshold
combined with a speed gate, stabilized by :func:`stabilize_contact_mask` — and
an optional gait phase clock for gaits with flight phases. :class:`FootstepResolver`
snaps each touchdown to a safe landing pose on the terrain and locks it across
the stance phase. :class:`TerrainReference` builds the terrain-adapted pelvis
and foot reference trajectories consumed by the planner (stance-locked
footholds, smooth swing interpolation, floor clamping, lateral crossover and
step-order enforcement).

Ported from Mondo-Robotics/PMT ``TCRS/stair_mppi/minimal_mppi_demo.py`` @ 88838ca
(viewer demo, TinyMPPI and MuJoCo helpers not ported).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from genesis.utils.geom import transform_by_quat

import eden as en
from eden.extensions.motion_optimization.tcrs.gait_phase import compute_per_frame_phase, fit_phase_clock
from eden.utils.geom import yaw_from_quat

if TYPE_CHECKING:
    from eden.options.extensions.motion_optimization import TCRSRobotOptions
    from eden.extensions.motion_optimization.tcrs.terrain_source import TerrainSource


def stabilize_contact_mask(mask: np.ndarray, min_contact_frames: int = 4, min_swing_frames: int = 2) -> np.ndarray:
    """Remove contact runs shorter than ``min_contact_frames`` and fill swing gaps shorter
    than ``min_swing_frames`` that are surrounded by contact on both sides.
    """
    out = np.asarray(mask, dtype=bool).copy()
    n = out.shape[0]
    if n == 0:
        return out

    i = 0
    while i < n:
        j = i + 1
        while j < n and out[j] == out[i]:
            j += 1
        if out[i] and (j - i) < min_contact_frames:
            out[i:j] = False
        i = j

    i = 0
    while i < n:
        j = i + 1
        while j < n and out[j] == out[i]:
            j += 1
        if (not out[i]) and (j - i) < min_swing_frames:
            left_true = (i > 0) and out[i - 1]
            right_true = (j < n) and out[j]
            if left_true and right_true:
                out[i:j] = True
        i = j

    return out


class MotionClip:
    """A mocap motion segment with derived foot-contact masks and gait phase.

    Parameters
    ----------
    robot_spec : TCRSRobotOptions
        Robot geometry/naming; resolves the pelvis and foot columns of the body
        arrays and provides the nominal ankle-to-sole height.
    fps : float
        Frame rate of the clip (Hz).
    dofs_pos : np.ndarray
        (T, n_dofs) joint positions in ``robot_spec.dofs_name`` order.
    links_pos_w : np.ndarray
        (T, n_links, 3) world body positions in ``robot_spec.links_name`` order.
    links_quat_w : np.ndarray
        (T, n_links, 4) world body quaternions [w, x, y, z].
    links_lin_vel_w : np.ndarray or None
        (T, n_links, 3) world body linear velocities. Finite-differenced from
        ``links_pos_w`` when omitted.
    contact_z_threshold : float
        Minimum foot-z threshold for contact detection (m); the adaptive
        percentile threshold never goes below it.
    contact_speed_threshold : float
        Foot speed threshold for contact detection (m/s).
    """

    def __init__(
        self,
        robot_spec: TCRSRobotOptions,
        fps: float,
        dofs_pos: np.ndarray,
        links_pos_w: np.ndarray,
        links_quat_w: np.ndarray,
        links_lin_vel_w: np.ndarray | None = None,
        contact_z_threshold: float = 0.06,
        contact_speed_threshold: float = 0.5,
    ):
        dofs_pos = np.asarray(dofs_pos)
        links_pos_w = np.asarray(links_pos_w)
        links_quat_w = np.asarray(links_quat_w)
        if fps <= 0.0:
            raise ValueError(f"fps must be > 0, got {fps}")
        if dofs_pos.ndim != 2:
            raise ValueError(f"dofs_pos must be (T, n_dofs), got shape {dofs_pos.shape}")
        n_frames = int(dofs_pos.shape[0])
        if n_frames == 0:
            raise ValueError("Motion clip is empty (0 frames)")
        n_dofs = len(robot_spec.dofs_name)
        n_links = len(robot_spec.links_name)
        if dofs_pos.shape[1] != n_dofs:
            raise ValueError(f"dofs_pos has {dofs_pos.shape[1]} joints but robot_spec.dofs_name has {n_dofs}")
        if links_pos_w.shape != (n_frames, n_links, 3):
            raise ValueError(
                f"links_pos_w must have shape ({n_frames}, {n_links}, 3) to match dofs_pos frames and "
                f"robot_spec.links_name, got {links_pos_w.shape}"
            )
        if links_quat_w.shape != (n_frames, n_links, 4):
            raise ValueError(f"links_quat_w must have shape ({n_frames}, {n_links}, 4), got {links_quat_w.shape}")
        if links_lin_vel_w is not None:
            links_lin_vel_w = np.asarray(links_lin_vel_w)
            if links_lin_vel_w.shape != (n_frames, n_links, 3):
                raise ValueError(
                    f"links_lin_vel_w must have shape ({n_frames}, {n_links}, 3), got {links_lin_vel_w.shape}"
                )

        self.robot_spec = robot_spec
        self.fps = float(fps)
        self.n_frames = n_frames
        self.dofs_pos = dofs_pos.copy()
        self.links_pos = links_pos_w.copy()
        self.links_quat = links_quat_w.copy()
        if links_lin_vel_w is not None:
            self.links_lin_vel = links_lin_vel_w.copy()
        else:
            self.links_lin_vel = np.gradient(self.links_pos, axis=0) * self.fps

        pelvis_idx = robot_spec.pelvis_index
        lfoot_idx = robot_spec.left_foot_index
        rfoot_idx = robot_spec.right_foot_index
        lfoot_speed = np.linalg.norm(self.links_lin_vel[:, lfoot_idx], axis=1)
        rfoot_speed = np.linalg.norm(self.links_lin_vel[:, rfoot_idx], axis=1)
        lfoot_z = self.links_pos[:, lfoot_idx, 2]
        rfoot_z = self.links_pos[:, rfoot_idx, 2]

        # Adaptive contact threshold: percentile-based.
        # Use the 25th percentile of foot z + margin as threshold.
        # This is robust to motions where the ground plane is not at z=0.
        all_foot_z = np.concatenate([lfoot_z, rfoot_z])
        _p25 = float(np.percentile(all_foot_z, 25.0))
        _adaptive_z_thresh = max(float(contact_z_threshold), _p25 + 0.02)
        left_contact_raw = (lfoot_z < _adaptive_z_thresh) & (lfoot_speed < float(contact_speed_threshold))
        right_contact_raw = (rfoot_z < _adaptive_z_thresh) & (rfoot_speed < float(contact_speed_threshold))
        self.left_contact = stabilize_contact_mask(left_contact_raw)
        self.right_contact = stabilize_contact_mask(right_contact_raw)

        # Phase clock fitting — for gaits with flight phases (running/hopping)
        self.phase_params = fit_phase_clock(
            lfoot_z,
            rfoot_z,
            lfoot_speed,
            rfoot_speed,
            self.fps,
            contact_z_threshold,
            contact_speed_threshold,
        )
        if self.phase_params is not None:
            self.per_frame_phase = compute_per_frame_phase(self.phase_params, self.n_frames)
            # Do NOT override contact masks — phase clock can misdetect boundaries.
        else:
            self.per_frame_phase = None

        # The robot's known ankle-to-sole height (upstream: 0.06 m, measured
        # from the G1 ankle_roll_link to the sole).
        self.foot_nominal_z = robot_spec.foot_nominal_z

        # Compute nominal pelvis height above foot contact surface (flat ground).
        # This is the target for pelvis height regulation on terrain.
        pelvis_z = self.links_pos[:, pelvis_idx, 2]
        self.pelvis_height_above_foot = float(np.median(pelvis_z) - self.foot_nominal_z)

    def get_pelvis_pos(self, frame: int) -> np.ndarray:
        return self.links_pos[frame, self.robot_spec.pelvis_index].copy()

    def _future_indices(self, frame: int, horizon: int) -> np.ndarray:
        if horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {horizon}")
        return (int(frame) + np.arange(int(horizon), dtype=np.int64)) % self.n_frames

    def get_future_indices(self, frame: int, horizon: int) -> np.ndarray:
        return self._future_indices(frame, horizon)

    def get_future_segment(self, frame: int, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = self._future_indices(frame, horizon)
        return (
            self.links_pos[idx, self.robot_spec.pelvis_index].copy(),
            self.links_pos[idx, self.robot_spec.left_foot_index].copy(),
            self.links_pos[idx, self.robot_spec.right_foot_index].copy(),
        )

    def get_future_contacts(self, frame: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        idx = self._future_indices(frame, horizon)
        return (
            self.left_contact[idx].copy(),
            self.right_contact[idx].copy(),
        )

    def get_future_support_weights(self, frame: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        """Return continuous support weights for pelvis z computation.

        When phase clock is available, returns smooth cosine-decay weights.
        Otherwise, falls back to binary weights from contact masks (1.0/0.0).
        """
        idx = self._future_indices(frame, horizon)
        if self.per_frame_phase is not None:
            return (
                self.per_frame_phase.support_weight_left[idx].copy(),
                self.per_frame_phase.support_weight_right[idx].copy(),
            )
        return (
            self.left_contact[idx].astype(np.float64),
            self.right_contact[idx].astype(np.float64),
        )

    def get_future_foot_quats(self, frame: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        idx = self._future_indices(frame, horizon)
        return (
            self.links_quat[idx, self.robot_spec.left_foot_index].copy(),
            self.links_quat[idx, self.robot_spec.right_foot_index].copy(),
        )


class FootstepResolver:
    """Footstep resolver: snaps raw mocap ankle positions onto safe terrain support surfaces.

    Core responsibilities:

    1. For each touchdown frame, call :meth:`_resolve_landing_pose` to find the
       best support placement, constraining the ankle so all foot key points
       (heel/midfoot/toe) land on the same support surface.
    2. Broadcast the resolved landing pose across the whole stance phase
       (``contact_support``) so the foot does not slide while in contact.

    Parameters
    ----------
    robot_spec : TCRSRobotOptions
        Robot geometry; provides the default toe probe offset.
    terrain : TerrainSource
        Terrain object; must provide ``height_at(x, y)`` and ``height_batch(x, y)``.
    raw_foot : np.ndarray
        (N, 3) raw mocap ankle positions (world frame).
    contact_mask : np.ndarray
        (N,) boolean contact mask; True = stance.
    foot_nominal_z : float
        Nominal ankle z height when standing on flat ground (z=0).
    foot_quats : np.ndarray or None
        (N, 4) ankle quaternions [w, x, y, z], used to rotate foot offsets into
        the world frame. Defaults to identity (facing forward).
    foot_point_offsets : np.ndarray or None
        (M,) local x offsets of foot key points relative to the ankle. Each
        offset is rotated by the quaternion into world coordinates and checked
        against step edges. Defaults to ``[toe_offset_x]``.
    toe_offset_x : float or None
        Toe x offset relative to the ankle (local frame). Defaults to
        ``robot_spec.toe_probe_offset_x``.
    max_landing_shift : float
        Maximum search displacement along the foot forward direction in
        :meth:`_resolve_landing_pose` (m).
    """

    def __init__(
        self,
        robot_spec: TCRSRobotOptions,
        terrain: TerrainSource,
        raw_foot: np.ndarray,
        contact_mask: np.ndarray,
        foot_nominal_z: float,
        foot_quats: np.ndarray | None = None,
        foot_point_offsets: np.ndarray | None = None,
        toe_offset_x: float | None = None,
        max_landing_shift: float = 0.05,
    ):
        if toe_offset_x is None:
            toe_offset_x = robot_spec.toe_probe_offset_x
        self.robot_spec = robot_spec
        self.terrain = terrain
        self.raw_foot = np.asarray(raw_foot, dtype=np.float64)
        self.contact_mask = np.asarray(contact_mask, dtype=bool)
        self.foot_nominal_z = float(foot_nominal_z)
        self.toe_offset_x = max(float(toe_offset_x), 0.0)
        self.max_landing_shift = max(float(max_landing_shift), 0.005)
        self.n_frames = int(self.contact_mask.shape[0])
        if foot_quats is not None:
            self.foot_quats = np.asarray(foot_quats, dtype=np.float64)
        else:
            # Without quaternions, default to facing forward (identity quaternion).
            self.foot_quats = np.tile(
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                (self.n_frames, 1),
            )
        if foot_point_offsets is not None:
            self.foot_point_offsets = np.asarray(foot_point_offsets, dtype=np.float64)
        else:
            self.foot_point_offsets = np.array([toe_offset_x], dtype=np.float64)

        # Detect touchdown events: rising edges from non-contact to contact.
        # contact_mask[i] = True and contact_mask[i-1] = False → frame i is a touchdown frame.
        self.touchdown = self.contact_mask & ~np.r_[False, self.contact_mask[:-1]]
        # If contact frames exist but no rising edge was found (the whole segment is
        # in contact), mark the first contact frame as a touchdown.
        if np.any(self.contact_mask) and not np.any(self.touchdown):
            first_contact = int(np.flatnonzero(self.contact_mask)[0])
            self.touchdown[first_contact] = True

        # For each touchdown frame, resolve a safe landing position
        # (x constrained to the step's safe interval, z = terrain height + foot_nominal_z).
        self.touchdown_support = np.full((self.n_frames, 3), np.nan, dtype=np.float64)
        for td in np.flatnonzero(self.touchdown):
            self.touchdown_support[td] = self._resolve_landing_pose(self.raw_foot[td], self.foot_quats[td])

        # Build prev/next contact-frame index maps (used during swing to find adjacent stance frames).
        (
            self.prev_contact_idx,  # prev_contact_idx[i] = nearest contact frame index before frame i
            self.next_contact_idx,  # next_contact_idx[i] = nearest contact frame index after frame i
        ) = self._build_prev_next_indices(self.contact_mask)
        # Build per-frame "next touchdown frame" indices (wrap=True: wraps to the start past the end).
        self.next_touchdown_idx = self._build_next_event_indices(self.touchdown, wrap=True)

        # Broadcast touchdown landing poses across the whole stance phase:
        # contact_support[i] = the locked position of that stance segment.
        self.contact_support = np.full((self.n_frames, 3), np.nan, dtype=np.float64)
        self._build_contact_support()

    def _resolve_landing_pose(self, raw_pose: np.ndarray, foot_quat: np.ndarray) -> np.ndarray:
        """Resolve a single frame's safe landing position along the foot's yaw direction.

        Does not assume step edges are aligned with the world x axis: extract
        yaw from foot_quat → the foot's forward direction fwd, then search for
        the smallest ankle offset along fwd that puts all key points
        (heel/midfoot/toe) on the same support surface, so the foot does not
        straddle a step edge.

        Algorithm:
        1. Compute world xy coordinates of the ankle and each key point.
        2. Query the terrain height at each point.
        3. Do a 1D search along fwd (discrete stepping) for the smallest offset
           that puts all key points on the same step surface (height spread < tolerance).
        4. z = max of all key-point terrain heights + foot_nominal_z.

        Parameters
        ----------
        raw_pose : np.ndarray
            (3,) raw mocap ankle position [x, y, z].
        foot_quat : np.ndarray
            (4,) ankle quaternion [w, x, y, z].

        Returns
        -------
        np.ndarray
            (3,) safe landing position [x_safe, y_safe, z_safe].
        """
        ax = float(raw_pose[0])
        ay = float(raw_pose[1])

        # ---- Extract yaw from foot_quat → the foot's forward direction (world xy plane) ----
        yaw = yaw_from_quat(foot_quat)
        fwd_x = np.cos(yaw)  # world x component of the foot forward direction
        fwd_y = np.sin(yaw)  # world y component of the foot forward direction

        # ---- Precompute each key point's world xy offset relative to the ankle ----
        # foot_point_offsets = [heel_x, mid_x, toe_x], all x offsets in the foot local frame.
        point_offsets_world = []  # [(dx, dy), ...] world-frame offsets
        for x_off in self.foot_point_offsets:
            off_world = transform_by_quat(np.array([x_off, 0.0, 0.0]), foot_quat)
            point_offsets_world.append((float(off_world[0]), float(off_world[1])))

        # ---- Helper: evaluate all key points at a given ankle (px, py) ----
        def _eval_at(px, py):
            """Return (h_ankle, h_points[], h_max, all_same_surface)."""
            h_ankle = float(self.terrain.height_at(px, py))
            h_max = h_ankle
            h_pts = []
            for dx, dy in point_offsets_world:
                h = float(self.terrain.height_at(px + dx, py + dy))
                h_pts.append(h)
                if h > h_max:
                    h_max = h
            # Whether all points are on the same step surface (height spread < 1cm).
            all_h = [h_ankle] + h_pts
            all_same = (max(all_h) - min(all_h)) < 0.01
            return h_ankle, h_pts, h_max, all_same

        # ---- Evaluate the original position first ----
        h_ankle_0, h_pts_0, h_max_0, same_0 = _eval_at(ax, ay)
        if same_0:
            # All key points already on the same step surface → no adjustment needed.
            z_safe = h_max_0 + self.foot_nominal_z
            return np.array([ax, ay, z_safe], dtype=np.float64)

        # Terrain height at the original ankle position — reference height of the target step surface.
        # The search prefers staying on a surface at the same height, to avoid
        # being pushed down onto lower flat ground.
        ref_h = h_ankle_0

        # ---- Search for the optimal offset along the foot's forward direction (±fwd) ----
        # Search strategy: step along fwd with step_size until a position is found
        # where all key points are on the same step surface, or the search limit is reached.
        # Negative fwd = whole foot moves backward (toe away from the front edge).
        # Positive fwd = whole foot moves forward (heel away from the rear edge).
        step_size = 0.005  # search step (m)
        max_shift = self.max_landing_shift
        n_steps = int(max_shift / step_size) + 1

        best_shift = 0.0
        best_score = self._landing_score(h_ankle_0, h_pts_0, same_0, 0.0, ref_h)
        best_xy = (ax, ay)
        best_h_max = h_max_0

        for sign in (-1.0, 1.0):
            for i in range(1, n_steps + 1):
                d = sign * i * step_size
                px = ax + d * fwd_x
                py = ay + d * fwd_y
                h_ankle, h_pts, h_max, same = _eval_at(px, py)
                score = self._landing_score(h_ankle, h_pts, same, abs(d), ref_h)
                if score < best_score:
                    best_score = score
                    best_shift = d
                    best_xy = (px, py)
                    best_h_max = h_max
                # If a fully consistent step surface was already found and further
                # shifts can only grow, stop early.
                if same and abs(d) > abs(best_shift) + step_size:
                    break

        x_safe, y_safe = best_xy
        z_safe = best_h_max + self.foot_nominal_z
        return np.array([x_safe, y_safe, z_safe], dtype=np.float64)

    @staticmethod
    def _landing_score(h_ankle, h_pts, same, shift, ref_h):
        """Landing score: lower is better.

        Priority (lexicographic tuple comparison):
        1. same_surface: whether all key points are on the same step surface (0=yes, 1=no).
        2. height_spread: range of key-point heights (smaller is better).
        3. ref_deviation: deviation of the ankle terrain height from the original
           reference height (prefer staying on the original step surface, to avoid
           being pushed onto a lower/higher one).
        4. shift: offset distance along fwd (smaller is better — stay close to the
           original position).

        Returns
        -------
        tuple
            (int, float, float, float) score tuple.
        """
        all_h = [h_ankle] + list(h_pts)
        spread = max(all_h) - min(all_h)
        ref_dev = abs(h_ankle - ref_h)
        return (0 if same else 1, spread, ref_dev, shift)

    def _build_prev_next_indices(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build "previous event frame" and "next event frame" index maps for every frame.

        Uses binary search (searchsorted) over the event-frame list:
        - prev_idx[i]: nearest event frame at or before frame i, -1 if none.
        - next_idx[i]: nearest event frame at or after frame i, wrapping to the
          first event past the end.

        Parameters
        ----------
        mask : np.ndarray
            (N,) boolean array; True positions are event frames.

        Returns
        -------
        tuple
            (prev_idx, next_idx): two (N,) int64 index arrays.
        """
        n = int(mask.shape[0])
        event_idx = np.flatnonzero(mask)  # indices of all event frames (sorted)
        prev_idx = np.full(n, -1, dtype=np.int64)
        next_idx = np.full(n, -1, dtype=np.int64)
        if event_idx.size == 0:
            return prev_idx, next_idx

        for i in range(n):
            # Previous event frame: largest value <= i in event_idx.
            p = np.searchsorted(event_idx, i, side="right") - 1
            prev_idx[i] = int(event_idx[p]) if p >= 0 else -1
            # Next event frame: smallest value >= i in event_idx.
            q = np.searchsorted(event_idx, i, side="left")
            next_idx[i] = int(event_idx[q]) if q < event_idx.size else int(event_idx[0])
        return prev_idx, next_idx

    def _build_next_event_indices(self, mask: np.ndarray, wrap: bool = False) -> np.ndarray:
        """Build a "next event frame" index map for every frame.

        Like :meth:`_build_prev_next_indices` but only in the next direction.
        With wrap=True, frames past the last event wrap around to the first one.

        Parameters
        ----------
        mask : np.ndarray
            (N,) boolean array; True positions are event frames.
        wrap : bool
            Whether to wrap around at the end of the sequence.

        Returns
        -------
        np.ndarray
            (N,) int64 index array; -1 means no subsequent event frame.
        """
        n = int(mask.shape[0])
        event_idx = np.flatnonzero(mask)
        next_idx = np.full(n, -1, dtype=np.int64)
        if event_idx.size == 0:
            return next_idx

        for i in range(n):
            q = np.searchsorted(event_idx, i, side="left")
            if q < event_idx.size:
                next_idx[i] = int(event_idx[q])
            elif wrap:
                next_idx[i] = int(event_idx[0])  # wrap to the start of the sequence
            else:
                next_idx[i] = -1  # no subsequent event
        return next_idx

    def _build_contact_support(self) -> None:
        """Broadcast touchdown landing poses across the whole stance phase.

        Iterate over all frames:
        1. When a "stance start" frame is detected (rising edge from non-contact
           to contact):
           - Prefer touchdown_support (safe landing positions precomputed in __init__).
           - If touchdown_support is invalid (NaN), recompute via _resolve_landing_pose().
        2. Within the whole stance phase (continuous contact), all frames lock to
           the same current position → xy and z are fully fixed while in support,
           so the foot never slides.

        The result is written to self.contact_support: an (N, 3) array with valid
        values on stance frames and NaN on swing frames.
        """
        if not np.any(self.contact_mask):
            return  # no contact frames at all, nothing to do

        current = None  # locked position of the current stance segment
        for i in range(self.n_frames):
            prev_i = i - 1
            if self.contact_mask[i] and (i == 0 or not self.contact_mask[prev_i]):
                # A new stance segment starts: lock its landing position.
                if np.isfinite(self.touchdown_support[i, 2]):
                    current = self.touchdown_support[i].copy()
                else:
                    current = self._resolve_landing_pose(self.raw_foot[i], self.foot_quats[i])
            if self.contact_mask[i]:
                if current is None:
                    # Edge case: the sequence starts in contact but was not flagged as a touchdown.
                    current = self._resolve_landing_pose(self.raw_foot[i], self.foot_quats[i])
                self.contact_support[i] = current.copy()

    def prev_support_pose(self, frame_idx: int) -> np.ndarray | None:
        """Query the nearest support position before frame ``frame_idx``.

        Used during swing to determine the liftoff position: the contact_support
        of the last contact frame before the swing is the liftoff point.

        Returns
        -------
        np.ndarray or None
            (3,) position array [x, y, z], or None (no preceding support frame).
        """
        idx = int(frame_idx) % self.n_frames
        prev_contact = int(self.prev_contact_idx[idx])
        if prev_contact < 0:
            return None
        pose = self.contact_support[prev_contact]
        if not np.isfinite(pose[2]):
            return None
        return pose.copy()

    def next_touchdown_pose(self, frame_idx: int) -> np.ndarray | None:
        """Query the nearest touchdown (landing) position after frame ``frame_idx``.

        Used during swing to determine the landing target: the safe position the
        foot must reach at the end of the swing.

        Returns
        -------
        np.ndarray or None
            (3,) position array [x, y, z], or None (no subsequent touchdown).
        """
        idx = int(frame_idx) % self.n_frames
        td = int(self.next_touchdown_idx[idx])
        if td < 0:
            return None
        pose = self.touchdown_support[td]
        if not np.isfinite(pose[2]):
            return None
        return pose.copy()


class TerrainReference:
    """Terrain reference trajectory builder: adapts raw mocap foot positions to the target terrain.

    Core outputs:
        left_precomputed / right_precomputed: (N, 3) left/right ankle reference positions.
        left_precomputed_quats / right_precomputed_quats: (N, 4) left/right ankle reference quaternions.

    Processing pipeline:
        1. Create :class:`FootstepResolver` → resolve a safe landing position for
           every touchdown frame.
        2. :meth:`_precompute_foot_reference` → stance frames use the locked landing
           position, swing frames use Hermite/quintic interpolation + floor clamping.
        3. :meth:`_flatten_stance_quats` → the sole stays level during stance
           (only yaw is kept).

    Parameters
    ----------
    terrain : TerrainSource
        Terrain object.
    motion : MotionClip
        Mocap motion data.
    lookahead : float
        Terrain lookahead distance (m), lets the pelvis lift ahead of upcoming steps.
    smoothing_alpha : float
        EMA smoothing coefficient for the pelvis z.
    toe_offset_x : float or None
        Toe x offset relative to the ankle (local frame). Defaults to
        ``motion.robot_spec.toe_probe_offset_x``.
    heel_offset_x : float
        Heel x offset relative to the ankle (local frame, negative = behind).
    mid_offset_x : float
        Midfoot x offset relative to the ankle (local frame).
    swing_floor_margin : float
        Extra ground-clearance margin during swing (m).
    soft_reach_limit : float
        Soft pelvis-to-foot reach limit (m).
    max_landing_shift : float
        Maximum landing-position search displacement along fwd (m).
    """

    def __init__(
        self,
        terrain: TerrainSource,
        motion: MotionClip,
        lookahead: float = 0.0,
        smoothing_alpha: float = 0.08,
        toe_offset_x: float | None = None,
        heel_offset_x: float = -0.04,
        mid_offset_x: float = 0.10,
        swing_floor_margin: float = 0.003,
        soft_reach_limit: float = 0.70,
        max_landing_shift: float = 0.05,
    ):
        robot_spec = motion.robot_spec
        if toe_offset_x is None:
            toe_offset_x = robot_spec.toe_probe_offset_x
        lfoot_idx = robot_spec.left_foot_index
        rfoot_idx = robot_spec.right_foot_index
        self.terrain = terrain
        self.motion = motion
        self.lookahead = float(lookahead)
        self.alpha = float(smoothing_alpha)
        self.swing_floor_margin = max(float(swing_floor_margin), 0.0)
        self.soft_reach_limit = max(float(soft_reach_limit), 0.1)
        self._ema_z = None  # exponential moving average of the pelvis z

        # Local x offsets of the foot key points: [heel, midfoot, toe].
        # FootstepResolver uses these three points to check whether the whole foot fits on a step.
        foot_point_offsets = np.array([heel_offset_x, mid_offset_x, toe_offset_x], dtype=np.float64)
        self.left_resolver = FootstepResolver(
            robot_spec=robot_spec,
            terrain=terrain,
            raw_foot=motion.links_pos[:, lfoot_idx],
            contact_mask=motion.left_contact,
            foot_nominal_z=motion.foot_nominal_z,
            foot_quats=motion.links_quat[:, lfoot_idx],
            foot_point_offsets=foot_point_offsets,
            toe_offset_x=toe_offset_x,
            max_landing_shift=max_landing_shift,
        )
        self.right_resolver = FootstepResolver(
            robot_spec=robot_spec,
            terrain=terrain,
            raw_foot=motion.links_pos[:, rfoot_idx],
            contact_mask=motion.right_contact,
            foot_nominal_z=motion.foot_nominal_z,
            foot_quats=motion.links_quat[:, rfoot_idx],
            foot_point_offsets=foot_point_offsets,
            toe_offset_x=toe_offset_x,
            max_landing_shift=max_landing_shift,
        )
        # Precompute the full foot reference trajectories
        # (stance frames = locked footholds, swing frames = interpolation).
        self.left_precomputed = self._precompute_foot_reference(self.left_resolver)
        self.right_precomputed = self._precompute_foot_reference(self.right_resolver)
        # Precompute the ankle reference quaternions (initialized to the raw mocap values).
        self.left_precomputed_quats = motion.links_quat[:, lfoot_idx].copy()
        self.right_precomputed_quats = motion.links_quat[:, rfoot_idx].copy()
        # Keep the sole level during stance: extract the yaw angle, drop pitch/roll.
        # On steps the foot should lie flat, not follow any sole tilt present in the mocap.
        self._flatten_stance_quats(self.left_precomputed_quats, motion.left_contact)
        self._flatten_stance_quats(self.right_precomputed_quats, motion.right_contact)

        # Enforce no lateral crossover between left and right foot references.
        # Check in pelvis local frame: left foot must stay on left side.
        self._enforce_no_crossover(motion, min_sep=0.08)
        # Enforce step-order consistency: the cross product of consecutive
        # left/right landing vectors must match the original motion's sign.
        self._enforce_step_order(motion)

    def _enforce_step_order(self, motion: MotionClip) -> None:
        """Enforce that the left/right order of the foot references matches the original motion.

        Per frame, the sign of (pelvis forward) × (right foot − left foot) in xy must
        match the raw motion's wherever the raw sign is clear (``|cross| > 0.01``).
        Where it does not, both feet are shifted 4 cm along the pelvis lateral axis
        (apart, or toward each other to restore an intentional crossing).
        """
        N = len(self.left_precomputed)
        pelvis_quats = motion.links_quat[:, motion.robot_spec.pelvis_index]
        raw_lfoot = motion.links_pos[:, motion.robot_spec.left_foot_index]
        raw_rfoot = motion.links_pos[:, motion.robot_spec.right_foot_index]

        # This correction only applies when both feet have at least one stance
        # phase; count stance onsets (a rising edge of the contact mask) per foot.
        def _num_stance_onsets(contact_mask: np.ndarray) -> int:
            return int(np.sum(contact_mask & ~np.concatenate(([False], contact_mask[:-1]))))

        if _num_stance_onsets(motion.left_contact) < 1 or _num_stance_onsets(motion.right_contact) < 1:
            return

        n_fixed = 0
        for i in range(N):
            yaw = yaw_from_quat(pelvis_quats[i])
            fwd = np.array([np.cos(yaw), np.sin(yaw)])  # pelvis forward in xy

            # Vector from left to right foot (precomputed)
            lr_vec = self.right_precomputed[i, :2] - self.left_precomputed[i, :2]
            # Cross product z-component: fwd × lr_vec
            cross_precomp = fwd[0] * lr_vec[1] - fwd[1] * lr_vec[0]

            # Same for original motion
            lr_raw = raw_rfoot[i, :2] - raw_lfoot[i, :2]
            cross_raw = fwd[0] * lr_raw[1] - fwd[1] * lr_raw[0]

            # If original has no crossover (cross_raw < 0 means right is to the right
            # of the forward direction, which is normal), enforce same sign
            if cross_raw < -0.01 and cross_precomp > 0:
                # Precomputed crossed over — push apart
                lat = np.array([-np.sin(yaw), np.cos(yaw)])  # left direction
                deficit = 0.08  # push to at least 8cm separation
                self.left_precomputed[i, :2] += 0.5 * deficit * lat
                self.right_precomputed[i, :2] -= 0.5 * deficit * lat
                n_fixed += 1
            elif cross_raw > 0.01 and cross_precomp < 0:
                # Original motion intentionally crossed the feet but the
                # reconstruction uncrossed them — push back *toward* the cross
                # (left foot to the right, right foot to the left) to restore the
                # original step order, i.e. the mirror of the branch above.
                lat = np.array([-np.sin(yaw), np.cos(yaw)])  # left direction
                deficit = 0.08
                self.left_precomputed[i, :2] -= 0.5 * deficit * lat
                self.right_precomputed[i, :2] += 0.5 * deficit * lat
                n_fixed += 1

        if n_fixed > 0:
            en.logger.info(f"  [TerrainRef] Step-order fix: {n_fixed}/{N} frames corrected")

    def _enforce_no_crossover(self, motion: MotionClip, min_sep: float = 0.08) -> None:
        """Enforce that left foot stays left and right foot stays right.

        For each frame, project both feet onto the pelvis lateral axis.
        If the left foot's lateral coordinate is less than the right foot's
        (i.e. they crossed), push them apart to maintain min_sep.

        Only modifies frames where the original motion had no crossover
        (respects the original motion's intent for intentional crossovers).
        """
        N = len(self.left_precomputed)
        pelvis_pos = motion.links_pos[:, motion.robot_spec.pelvis_index]
        pelvis_quats = motion.links_quat[:, motion.robot_spec.pelvis_index]
        raw_lfoot = motion.links_pos[:, motion.robot_spec.left_foot_index]
        raw_rfoot = motion.links_pos[:, motion.robot_spec.right_foot_index]

        n_fixed = 0
        for i in range(N):
            # Pelvis lateral direction
            yaw = yaw_from_quat(pelvis_quats[i])
            lat = np.array([-np.sin(yaw), np.cos(yaw)])
            pel_xy = pelvis_pos[i, :2]

            # Check original motion: did it have crossover at this frame?
            raw_lf_lat = np.dot(raw_lfoot[i, :2] - pel_xy, lat)
            raw_rf_lat = np.dot(raw_rfoot[i, :2] - pel_xy, lat)
            if raw_lf_lat - raw_rf_lat < 0.02:
                # Original motion already has crossover — don't enforce
                continue

            # Check precomputed reference
            lf_lat = np.dot(self.left_precomputed[i, :2] - pel_xy, lat)
            rf_lat = np.dot(self.right_precomputed[i, :2] - pel_xy, lat)
            if lf_lat - rf_lat < min_sep:
                deficit = min_sep - (lf_lat - rf_lat)
                self.left_precomputed[i, :2] += 0.5 * deficit * lat
                self.right_precomputed[i, :2] -= 0.5 * deficit * lat
                n_fixed += 1

        if n_fixed > 0:
            en.logger.info(
                f"  [TerrainRef] Fixed {n_fixed}/{N} frames lateral crossover (min_sep={min_sep * 100:.0f}cm)"
            )

    @staticmethod
    def _flatten_stance_quats(quats: np.ndarray, contact_mask: np.ndarray) -> None:
        """Flatten the ankle quaternions during stance: drop pitch/roll, keep only yaw.

        Rationale: on discrete step terrain the sole should lie level on the step
        surface during stance, rather than follow any sole tilt (pitch/roll) that
        may exist in the mocap data. Only yaw (rotation about z) is kept to
        preserve the foot heading.

        Operation: extract yaw = atan2(2(wz+xy), 1-2(y²+z²)) from the quaternion
        [w, x, y, z], then rebuild a pure-yaw quaternion: [cos(yaw/2), 0, 0, sin(yaw/2)].

        Parameters
        ----------
        quats : np.ndarray
            (N, 4) quaternion array [w, x, y, z], modified in place.
        contact_mask : np.ndarray
            (N,) boolean contact mask; only contact frames are modified.
        """
        for i in range(quats.shape[0]):
            if not contact_mask[i]:
                continue  # swing frames keep the raw mocap quaternion
            # Extract the yaw angle (rotation about z) from the quaternion.
            yaw = yaw_from_quat(quats[i])
            half = yaw * 0.5
            # Rebuild a pure-yaw quaternion: pitch = roll = 0 → the sole is level.
            quats[i] = [np.cos(half), 0.0, 0.0, np.sin(half)]

    def reset(self) -> None:
        """Reset the pelvis-z EMA state."""
        self._ema_z = None

    def _target_height(self, x: float, y: float) -> float:
        """Query the target terrain height = max(height here, height ``lookahead`` m ahead along world +x).

        Lookahead: sense upcoming steps ahead of time so the pelvis lifts early.
        Taking the max is conservative — the terrain height ahead is never underestimated.
        """
        h_here = self.terrain.height_at(x, y)
        h_ahead = self.terrain.height_at(x + self.lookahead, y)
        return max(h_here, h_ahead)

    def update(self, pelvis_x: float, pelvis_y: float) -> float:
        """Single-frame update of the pelvis-z exponential moving average (EMA).

        Formula: ema_z += alpha * (target - ema_z).
        The smaller alpha, the smoother (but slower) the pelvis z response.
        For real-time per-frame use (not precomputation).

        Returns
        -------
        float
            The smoothed pelvis z reference for the current frame.
        """
        target = self._target_height(pelvis_x, pelvis_y)
        if self._ema_z is None:
            self._ema_z = target  # initialize directly on the first frame
        else:
            self._ema_z += self.alpha * (target - self._ema_z)
        return self._ema_z

    def segment_offsets(self, pelvis_positions: np.ndarray) -> np.ndarray:
        """Batch-compute EMA z offsets for a sequence of pelvis positions.

        Parameters
        ----------
        pelvis_positions : np.ndarray
            (N, 3) pelvis position sequence.

        Returns
        -------
        np.ndarray
            (N,) EMA-smoothed z reference per frame.
        """
        offsets = np.zeros(pelvis_positions.shape[0], dtype=np.float64)
        ema = 0.0 if self._ema_z is None else float(self._ema_z)
        for i, pelvis in enumerate(pelvis_positions):
            target = self._target_height(pelvis[0], pelvis[1])
            ema += self.alpha * (target - ema)
            offsets[i] = ema
        return offsets

    def _swing_indices(self, start: int, end: int, n_frames: int) -> np.ndarray:
        """Generate the swing-phase frame index array [start, start+1, ..., end-1].

        Parameters
        ----------
        start : int
            Swing-phase start frame index.
        end : int
            Swing-phase end frame index.
        n_frames : int
            Total frame count (for bounds checking).

        Returns
        -------
        np.ndarray
            (n_swing,) int64 index array; empty if invalid.
        """
        if start < 0 or end < 0:
            return np.zeros(0, dtype=np.int64)
        if start < end:
            return np.arange(start, end, dtype=np.int64)
        return np.zeros(0, dtype=np.int64)

    def _apply_floor_clamp(self, points: np.ndarray, foot_nominal_z: float) -> np.ndarray:
        """Floor clamp: ensure the ankle z is not below terrain height + nominal height + margin.

        Formula: z = max(z, terrain_height(x, y) + foot_nominal_z + swing_floor_margin).

        Note: the terrain height is queried at the ankle's (x, y), not at the toe
        or heel (x, y). If the toe/heel is far from the ankle and the terrain is
        higher there, this clamp will not sense it.

        Parameters
        ----------
        points : np.ndarray
            (M, 3) ankle position array.
        foot_nominal_z : float
            Nominal ankle z on flat ground.

        Returns
        -------
        np.ndarray
            (M, 3) clamped position array (a copy; the input is not modified).
        """
        floor = (
            self.terrain.height_batch(points[:, 0], points[:, 1])  # terrain height at each ankle xy
            + float(foot_nominal_z)  # plus the nominal ankle height
            + self.swing_floor_margin  # plus the extra margin
        )
        out = points.copy()
        out[:, 2] = np.maximum(out[:, 2], floor)
        return out

    def _precompute_foot_reference(self, resolver: FootstepResolver) -> np.ndarray:
        """Precompute the full foot reference trajectory: locked footholds during
        stance, interpolation during swing.

        Processing:
        1. Initialize from the raw mocap foot positions.
        2. Overwrite stance frames with the safe landing positions computed by
           the FootstepResolver (contact_support).
        3. Interpolate swing frames between liftoff and landing:
           - xy: cubic Hermite interpolation (zero endpoint velocity for a smooth transition).
           - z:  quintic baseline interpolation + tapered blend of the raw mocap swing shape.
        4. Floor clamp: ensure no frame's z is below the terrain.

        Parameters
        ----------
        resolver : FootstepResolver
            Resolver instance (holds contact_support, touchdown_support, ...).

        Returns
        -------
        np.ndarray
            (N, 3) ankle reference position array.
        """
        raw = resolver.raw_foot  # (N, 3)
        contact = resolver.contact_mask  # (N,)
        n_frames = int(resolver.n_frames)
        out = raw.copy()

        # === Step 1: overwrite all valid (non-NaN) stance frames with the safe landing positions ===
        valid_contact = np.isfinite(resolver.contact_support[:, 2])
        out[valid_contact] = resolver.contact_support[valid_contact]

        # === Step 2: find the start/end frames of all swing phases ===
        # swing_start: falling edges (contact -> no contact); touchdown: rising edges.
        prev_contact = np.r_[False, contact[:-1]]
        swing_start = np.flatnonzero((~contact) & prev_contact)
        touchdown = np.flatnonzero(contact & (~prev_contact))

        if touchdown.size == 0:
            # No landing events → just floor-clamp and return.
            self._apply_floor_clamp(out, resolver.foot_nominal_z)
            return out

        # === Step 3: interpolate each swing phase ===
        touchdown_sorted = np.sort(touchdown)
        for s in swing_start:
            # The first touchdown after this swing start.
            q = np.searchsorted(touchdown_sorted, s + 1, side="left")
            if q >= touchdown_sorted.size:
                continue
            e = int(touchdown_sorted[q])
            span = self._swing_indices(int(s), e, n_frames)
            if span.size == 0:
                continue

            # liftoff = the last contact frame before the swing.
            liftoff_idx = int(s) - 1
            # landing = the first contact frame after the swing.
            landing_idx = int(e)
            if liftoff_idx < 0:
                continue  # sequence starts mid-swing → no liftoff reference, skip

            liftoff_pose = out[liftoff_idx]  # liftoff position (already overwritten by contact_support)
            # Landing position: prefer touchdown_support (pre-resolved safe position).
            landing_pose = resolver.touchdown_support[landing_idx]
            if not np.isfinite(landing_pose[2]):
                landing_pose = out[landing_idx]  # pre-resolved value invalid → fall back to the current out value

            # Bounds check: with an invalid liftoff or landing, only clamp.
            if not np.isfinite(liftoff_pose[2]) or not np.isfinite(landing_pose[2]):
                out[span] = self._apply_floor_clamp(out[span], resolver.foot_nominal_z)
                continue

            n = int(span.size)
            if n == 1:
                t = np.array([0.5], dtype=np.float64)  # only 1 frame → take the midpoint
            else:
                t = np.linspace(0.0, 1.0, n, dtype=np.float64)  # normalized time 0→1

            # --- xy interpolation: cubic Hermite (zero endpoint velocity for a smooth transition) ---
            # h(t) = 3t² - 2t³, h(0)=0, h(1)=1, h'(0)=h'(1)=0
            xy_blend = (3.0 * t**2) - (2.0 * t**3)
            out[span, 0:2] = (
                (1.0 - xy_blend)[:, np.newaxis] * liftoff_pose[0:2]  # from liftoff xy
                + xy_blend[:, np.newaxis] * landing_pose[0:2]  # to landing xy
            )

            # --- z interpolation: quintic baseline + tapered blend of the raw mocap swing shape ---
            # quintic: q(t) = 6t⁵ - 15t⁴ + 10t³, C2 continuous (position + velocity + acceleration).
            z_blend = (6.0 * t**5) - (15.0 * t**4) + (10.0 * t**3)
            z_base = (1.0 - z_blend) * liftoff_pose[2] + z_blend * landing_pose[2]

            # Raw mocap swing shape: the part of the raw z that exceeds linear interpolation.
            # This carries the natural foot-lift arc present in the mocap data.
            raw_z0 = raw[liftoff_idx, 2]
            raw_z1 = raw[landing_idx, 2]
            raw_linear = (1.0 - t) * raw_z0 + t * raw_z1
            raw_shape = raw[span, 2] - raw_linear

            # Taper window: 16t²(1-t)², zero at t=0 and t=1 (endpoints untouched),
            # maximum = 1 at t=0.5 (largest blend in the middle).
            shape_taper = 16.0 * (t**2) * ((1.0 - t) ** 2)
            out[span, 2] = z_base + shape_taper * raw_shape

            out[span] = self._apply_floor_clamp(out[span], resolver.foot_nominal_z)

        # Global floor clamp: a final pass so no frame penetrates the ground.
        return self._apply_floor_clamp(out, resolver.foot_nominal_z)

    def build_segment(
        self,
        pelvis: np.ndarray,
        lfoot: np.ndarray,
        rfoot: np.ndarray,
        left_contact: np.ndarray,
        right_contact: np.ndarray,
        indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pelvis_z_offsets = self.segment_offsets(pelvis)

        ref_pelvis = pelvis.copy()
        ref_pelvis[:, 2] += pelvis_z_offsets
        idx = np.asarray(indices, dtype=np.int64)
        ref_lfoot = self.left_precomputed[idx].copy()
        ref_rfoot = self.right_precomputed[idx].copy()
        ref_lfoot_quats = self.left_precomputed_quats[idx].copy()
        ref_rfoot_quats = self.right_precomputed_quats[idx].copy()

        # Layer 1: enforce reachability by lowering pelvis z when either foot
        # exceeds the soft reach limit.  Compute pelvis-to-foot distance and
        # lower pelvis z per horizon step so that both legs stay within L_soft.
        L = self.soft_reach_limit
        for i in range(ref_pelvis.shape[0]):
            for foot in (ref_lfoot[i], ref_rfoot[i]):
                dxy = np.sqrt((ref_pelvis[i, 0] - foot[0]) ** 2 + (ref_pelvis[i, 1] - foot[1]) ** 2)
                # max allowed vertical distance given horizontal offset
                max_dz = np.sqrt(max(L**2 - dxy**2, 0.0))
                # pelvis must be above foot, so cap: pelvis_z <= foot_z + max_dz
                ref_pelvis[i, 2] = min(ref_pelvis[i, 2], foot[2] + max_dz)

        return ref_pelvis, ref_lfoot, ref_rfoot, ref_lfoot_quats, ref_rfoot_quats
