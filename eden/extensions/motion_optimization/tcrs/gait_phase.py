"""
Phase-clock gait analysis for bipedal motion capture data.

Ported from Mondo-Robotics/PMT ``TCRS/stair_mppi/gait_phase.py`` @ 88838ca.

Fits a global phase clock φ ∈ [0,1) to periodic locomotion data,
with per-foot phase offsets θ_L, θ_R and swing ratios.

Classifies: walking, running, hopping, galloping, standing.

Key design choices:
  - Period estimation: hybrid of speed autocorrelation + speed minima touchdown
    detection. Autocorrelation works well for walking; speed minima work for
    running (where autocorrelation decays too fast).
  - Phase timeline: local phase resets at detected touchdowns prevent drift
    from stride frequency variation in long sequences.
  - Contact z threshold: 0.06 m by default, terrain-invariant. For data where
    foot z is not ground-referenced, the raw contact mask is empty and the clock
    is fitted from foot speed alone.
  - Clips with no significant flight phase are not fitted (``fit_phase_clock``
    returns None): z-based contact is reliable for them.

Usage:
    from eden.extensions.motion_optimization.tcrs.gait_phase import fit_phase_clock, compute_per_frame_phase
    params = fit_phase_clock(lfoot_z, rfoot_z, lfoot_speed, rfoot_speed, fps)
    if params is not None:
        phase = compute_per_frame_phase(params, n_frames)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GaitPhaseParams:
    """Fitted phase clock parameters for a motion clip."""

    period_frames: float  # stride period in frames
    period_sec: float  # period_frames / fps
    swing_ratio_left: float  # fraction of cycle in swing for left foot [0,1]
    swing_ratio_right: float  # fraction of cycle in swing for right foot [0,1]
    phase_offset_left: float  # θ_L (always 0.0, left foot is reference)
    phase_offset_right: float  # θ_R ∈ [0,1)
    gait_type: str  # "walk" / "run" / "hop" / "gallop" / "stand"
    fit_quality: float  # agreement with raw contact [0,1]
    fps: float  # frames per second
    # Touchdown frames detected during fitting (for phase reset)
    _left_touchdowns: np.ndarray = None  # frame indices of left foot touchdowns
    _right_touchdowns: np.ndarray = None  # frame indices of right foot touchdowns

    @property
    def has_flight_phase(self) -> bool:
        """True if the stance fractions sum to less than one cycle.

        Both feet are then airborne at some point of every cycle, whatever the phase offset.
        """
        stance_L = 1.0 - self.swing_ratio_left
        stance_R = 1.0 - self.swing_ratio_right
        return (stance_L + stance_R) < 1.0


@dataclass
class PerFramePhase:
    """Per-frame phase information for the full motion clip."""

    global_phase: np.ndarray  # (N,) φ ∈ [0,1)
    left_local_phase: np.ndarray  # (N,) (φ + θ_L) mod 1
    right_local_phase: np.ndarray  # (N,) (φ + θ_R) mod 1
    left_contact: np.ndarray  # (N,) bool — phase-derived contact
    right_contact: np.ndarray  # (N,) bool
    flight_mask: np.ndarray  # (N,) bool — both feet in swing
    support_weight_left: np.ndarray  # (N,) float [0,1]
    support_weight_right: np.ndarray  # (N,) float [0,1]


# ---------------------------------------------------------------------------
# Touchdown detection from speed minima
# ---------------------------------------------------------------------------


def _detect_touchdowns(foot_speed: np.ndarray, min_distance: int = 10, prominence: float = 0.1) -> np.ndarray:
    """Detect touchdown frames from foot speed local minima.

    At touchdown, foot speed drops to near zero. We find local minima
    of foot speed with sufficient prominence (speed must rise before/after).

    Parameters
    ----------
    foot_speed : np.ndarray
        (N,) foot speed magnitude
    min_distance : int
        minimum frames between touchdowns
    prominence : float
        minimum speed drop to qualify as touchdown

    Returns
    -------
    np.ndarray
        touchdown frame indices (sorted)
    """
    # Find minima by looking for peaks of negated speed
    neg_speed = -foot_speed
    # Adaptive prominence: use fraction of speed range
    speed_range = foot_speed.max() - foot_speed.min()
    prom = max(prominence, speed_range * 0.1)

    peaks, _ = find_peaks(neg_speed, distance=min_distance, prominence=prom)

    return peaks


def _period_from_touchdowns(
    touchdowns: np.ndarray,
    min_period: int = 10,
    max_period: int = 100,
    max_cv: float = 0.25,
) -> float | None:
    """Estimate stride period from median inter-touchdown interval.

    Filters outlier intervals (>40% from median) and returns robust median.
    Returns None if coefficient of variation exceeds max_cv (not periodic).

    Parameters
    ----------
    touchdowns : np.ndarray
        frame indices of detected touchdowns
    min_period : int
        minimum acceptable period
    max_period : int
        maximum acceptable period
    max_cv : float
        maximum coefficient of variation (std/mean). Above this,
        the motion is considered non-periodic.
    """
    if len(touchdowns) < 3:
        return None

    intervals = np.diff(touchdowns).astype(float)
    if len(intervals) == 0:
        return None

    # First pass median
    med = np.median(intervals)
    if med < min_period or med > max_period:
        return None

    # Filter outliers (>40% from median)
    good = np.abs(intervals - med) / med < 0.4
    if np.sum(good) < 2:
        return None

    good_intervals = intervals[good]
    period = float(np.median(good_intervals))

    # Check periodicity: reject if too variable
    cv = float(np.std(good_intervals) / np.mean(good_intervals))
    if cv > max_cv:
        return None

    return period


# ---------------------------------------------------------------------------
# Autocorrelation-based period estimation (backup method)
# ---------------------------------------------------------------------------


def _autocorrelation(signal: np.ndarray) -> np.ndarray:
    """Normalized autocorrelation via FFT (positive lags only)."""
    n = len(signal)
    s = signal - signal.mean()
    fft_len = 2 * n
    S = np.fft.rfft(s, n=fft_len)
    acf = np.fft.irfft(S * np.conj(S), n=fft_len)[:n]
    if acf[0] > 0:
        acf /= acf[0]
    return acf


def _period_from_autocorrelation(
    foot_speed: np.ndarray,
    min_period: int = 10,
    max_period: int = 100,
    min_peak_height: float = 0.15,
) -> float | None:
    """Estimate stride period from foot speed autocorrelation.

    Works well for walking gaits with clear periodicity.
    May fail for running (autocorrelation decays too fast).
    """
    acf = _autocorrelation(foot_speed)
    search_end = min(max_period, len(acf) - 1)
    if search_end <= min_period:
        return None

    segment = acf[min_period : search_end + 1]
    peaks, props = find_peaks(segment, height=min_peak_height, distance=min_period // 2)

    if len(peaks) == 0:
        return None

    best_idx = peaks[np.argmax(props["peak_heights"])]
    return float(best_idx + min_period)


# ---------------------------------------------------------------------------
# Phase offset estimation
# ---------------------------------------------------------------------------


def _estimate_phase_offset_from_touchdowns(left_tds: np.ndarray, right_tds: np.ndarray, period: float) -> float:
    """Estimate θ_R from touchdown timing difference.

    For each left TD, find the nearest right TD and compute the
    normalized lag. Return median offset.
    """
    if len(left_tds) == 0 or len(right_tds) == 0:
        return 0.5  # default: alternating

    offsets = []
    for ltd in left_tds:
        # Find nearest right TD after this left TD
        candidates = right_tds[right_tds > ltd]
        if len(candidates) == 0:
            continue
        rtd = candidates[0]
        lag = (rtd - ltd) / period
        offsets.append(lag % 1.0)

    if len(offsets) == 0:
        return 0.5

    return float(np.median(offsets))


def _estimate_phase_offset_from_xcorr(lfoot_speed: np.ndarray, rfoot_speed: np.ndarray, period: float) -> float:
    """Estimate phase offset θ_R from cross-correlation of foot speeds."""
    n = len(lfoot_speed)
    ls = lfoot_speed - lfoot_speed.mean()
    rs = rfoot_speed - rfoot_speed.mean()

    fft_len = 2 * n
    Ls = np.fft.rfft(ls, n=fft_len)
    Rs = np.fft.rfft(rs, n=fft_len)
    xcorr = np.fft.irfft(Ls * np.conj(Rs), n=fft_len)

    half_p = int(period / 2)
    lags = np.arange(-half_p, half_p + 1)
    centered = np.array([xcorr[lag % fft_len] for lag in lags])

    best_lag = lags[np.argmax(centered)]
    return (best_lag / period) % 1.0


# ---------------------------------------------------------------------------
# Swing ratio estimation
# ---------------------------------------------------------------------------


def _estimate_swing_ratio(raw_contact: np.ndarray, touchdowns: np.ndarray, period: float) -> float:
    """Estimate swing ratio from raw z-based contact mask.

    For each inter-touchdown cycle, count swing (non-contact) frames.
    Uses the z-threshold contact mask which is reliable on flat ground.

    Parameters
    ----------
    raw_contact : np.ndarray
        (N,) bool from z-threshold detection
    touchdowns : np.ndarray
        frame indices of detected touchdowns
    period : float
        estimated stride period (for filtering bad cycles)
    """
    if len(touchdowns) < 2:
        return float(1.0 - np.mean(raw_contact))

    ratios = []
    for i in range(len(touchdowns) - 1):
        t0 = touchdowns[i]
        t1 = touchdowns[i + 1]
        cycle_len = t1 - t0
        # Only use cycles close to expected period (within 40%)
        if abs(cycle_len - period) / period > 0.4:
            continue
        swing_frames = np.sum(~raw_contact[t0:t1])
        ratios.append(swing_frames / cycle_len)

    if len(ratios) == 0:
        return float(1.0 - np.mean(raw_contact))

    return float(np.median(ratios))


# ---------------------------------------------------------------------------
# Phase timeline with drift reset
# ---------------------------------------------------------------------------


def _build_phase_timeline(
    n_frames: int,
    period: float,
    touchdowns: np.ndarray,
) -> np.ndarray:
    """Build global phase timeline with piecewise linear interpolation between TDs.

    Each detected touchdown anchors phase to 0 (start of new cycle).
    Between consecutive touchdowns, phase increases linearly from 0 to 1.

    For frames before the first TD or after the last TD, extrapolate
    using the local period.
    """
    phi = np.empty(n_frames, dtype=np.float64)

    if len(touchdowns) == 0:
        # No touchdowns detected — uniform phase
        phi[:] = (np.arange(n_frames, dtype=np.float64) / period) % 1.0
        return phi

    if len(touchdowns) == 1:
        # Single touchdown — use constant period
        td = touchdowns[0]
        phi[:] = ((np.arange(n_frames, dtype=np.float64) - td) / period) % 1.0
        return phi

    # Before first touchdown
    first_td = touchdowns[0]
    first_period = float(touchdowns[1] - touchdowns[0])
    for t in range(first_td):
        phi[t] = ((t - first_td) / first_period) % 1.0

    # Between consecutive touchdowns
    for k in range(len(touchdowns) - 1):
        t0 = touchdowns[k]
        t1 = touchdowns[k + 1]
        seg_len = float(t1 - t0)
        for t in range(t0, t1):
            phi[t] = (t - t0) / seg_len

    # After last touchdown
    last_td = touchdowns[-1]
    last_period = float(touchdowns[-1] - touchdowns[-2])
    for t in range(last_td, n_frames):
        phi[t] = ((t - last_td) / last_period) % 1.0

    return phi


# ---------------------------------------------------------------------------
# Main fitting function
# ---------------------------------------------------------------------------


def fit_phase_clock(
    lfoot_z: np.ndarray,
    rfoot_z: np.ndarray,
    lfoot_speed: np.ndarray,
    rfoot_speed: np.ndarray,
    fps: float,
    contact_z_threshold: float = 0.06,
    contact_speed_threshold: float = 0.5,
) -> GaitPhaseParams | None:
    """Fit phase clock parameters from motion capture data.

    Hybrid period estimation:
    1. Try speed minima touchdown detection (robust for running)
    2. Fallback to speed autocorrelation (better for slow walking)

    Parameters
    ----------
    lfoot_z : np.ndarray
        (N,) left foot z positions, m
    rfoot_z : np.ndarray
        (N,) right foot z positions, m
    lfoot_speed : np.ndarray
        (N,) left foot speed magnitude, m/s
    rfoot_speed : np.ndarray
        (N,) right foot speed magnitude, m/s
    fps : float
        frames per second
    contact_z_threshold : float
        foot z (m) below which a foot can be in raw contact
    contact_speed_threshold : float
        foot speed (m/s) below which a foot can be in raw contact

    Returns
    -------
    GaitPhaseParams or None
        None if the clip is shorter than 30 frames, has no significant flight phase (both
        feet off the ground in under 5% of frames by raw contact, where z-based contact is
        reliable), or is not periodic.
    """
    N = len(lfoot_z)
    if N < 30:
        return None

    # Raw contact masks (z-threshold, no stabilization — preserves short contacts)
    raw_left_contact = (lfoot_z < contact_z_threshold) & (lfoot_speed < contact_speed_threshold)
    raw_right_contact = (rfoot_z < contact_z_threshold) & (rfoot_speed < contact_speed_threshold)

    # Quick check: if raw contact never has both feet simultaneously off ground,
    # the gait is walking-like and z-based detection works fine.
    raw_flight = (~raw_left_contact) & (~raw_right_contact)
    flight_ratio = float(np.mean(raw_flight))
    if flight_ratio < 0.05:
        # Walking gait — no significant flight phase, z-based contact is reliable
        return None

    # Step 1: Detect touchdowns from speed minima
    left_tds = _detect_touchdowns(lfoot_speed)
    right_tds = _detect_touchdowns(rfoot_speed)

    # Step 2: Estimate period — primary: touchdown intervals
    period = _period_from_touchdowns(left_tds)
    period_R = _period_from_touchdowns(right_tds)
    if period is not None and period_R is not None:
        period = (period + period_R) / 2.0
    elif period is None:
        period = period_R

    # Fallback: autocorrelation
    if period is None:
        period = _period_from_autocorrelation(lfoot_speed)
    if period is None:
        period = _period_from_autocorrelation(rfoot_speed)
    if period is None:
        return None

    # Step 3: Phase offset
    # Primary: from touchdown timing; fallback: cross-correlation
    if len(left_tds) >= 2 and len(right_tds) >= 2:
        offset = _estimate_phase_offset_from_touchdowns(left_tds, right_tds, period)
    else:
        offset = _estimate_phase_offset_from_xcorr(lfoot_speed, rfoot_speed, period)

    # Step 4: Swing ratios (from z-based contact, reliable on flat ground)
    swing_ratio_L = _estimate_swing_ratio(raw_left_contact, left_tds, period)
    swing_ratio_R = _estimate_swing_ratio(raw_right_contact, right_tds, period)

    # Step 5: Build params and classify
    params = GaitPhaseParams(
        period_frames=period,
        period_sec=period / fps,
        swing_ratio_left=swing_ratio_L,
        swing_ratio_right=swing_ratio_R,
        phase_offset_left=0.0,
        phase_offset_right=offset,
        gait_type="",
        fit_quality=0.0,
        fps=fps,
        _left_touchdowns=left_tds,
        _right_touchdowns=right_tds,
    )

    offset_diff = min(offset, 1.0 - offset)
    avg_swing = (swing_ratio_L + swing_ratio_R) / 2.0

    if avg_swing < 0.05:
        gait_type = "stand"
    elif offset_diff > 0.35:
        gait_type = "run" if params.has_flight_phase else "walk"
    elif offset_diff < 0.1:
        gait_type = "hop"
    else:
        gait_type = "gallop"
    params.gait_type = gait_type

    # Step 6: Compute fit quality
    phase = compute_per_frame_phase(params, N)
    # If raw contact has any frames, compare
    if np.any(raw_left_contact) or np.any(raw_right_contact):
        agree_L = np.mean(phase.left_contact == raw_left_contact) if np.any(raw_left_contact) else 0.5
        agree_R = np.mean(phase.right_contact == raw_right_contact) if np.any(raw_right_contact) else 0.5
        params.fit_quality = (agree_L + agree_R) / 2.0
    else:
        # No raw contact frames (foot z not ground-referenced): nothing to compare
        # against, so a nominal quality.
        params.fit_quality = 0.7

    return params


# ---------------------------------------------------------------------------
# Per-frame phase computation
# ---------------------------------------------------------------------------


def compute_per_frame_phase(
    params: GaitPhaseParams,
    n_frames: int,
) -> PerFramePhase:
    """Compute per-frame phase, contact masks, and support weights.

    Uses detected touchdowns for piecewise linear phase (drift-resistant).
    """
    # Build global phase timeline from left foot touchdowns
    left_tds = params._left_touchdowns if params._left_touchdowns is not None else np.array([], dtype=int)
    global_phase = _build_phase_timeline(n_frames, params.period_frames, left_tds)

    # Local phases
    left_local = (global_phase + params.phase_offset_left) % 1.0
    right_local = (global_phase + params.phase_offset_right) % 1.0

    # Contact from phase: stance = local_phase < stance_ratio
    stance_ratio_L = 1.0 - params.swing_ratio_left
    stance_ratio_R = 1.0 - params.swing_ratio_right
    left_contact = left_local < stance_ratio_L
    right_contact = right_local < stance_ratio_R

    flight_mask = (~left_contact) & (~right_contact)

    # Continuous support weights
    support_weight_left = np.ones(n_frames, dtype=np.float64)
    support_weight_right = np.ones(n_frames, dtype=np.float64)

    # Left foot: cosine decay during swing
    swing_mask_L = ~left_contact
    if params.swing_ratio_left > 1e-6:
        swing_progress_L = (left_local[swing_mask_L] - stance_ratio_L) / params.swing_ratio_left
        swing_progress_L = np.clip(swing_progress_L, 0.0, 1.0)
        support_weight_left[swing_mask_L] = 0.5 * (1.0 + np.cos(np.pi * swing_progress_L))

    # Right foot: cosine decay during swing
    swing_mask_R = ~right_contact
    if params.swing_ratio_right > 1e-6:
        swing_progress_R = (right_local[swing_mask_R] - stance_ratio_R) / params.swing_ratio_right
        swing_progress_R = np.clip(swing_progress_R, 0.0, 1.0)
        support_weight_right[swing_mask_R] = 0.5 * (1.0 + np.cos(np.pi * swing_progress_R))

    return PerFramePhase(
        global_phase=global_phase,
        left_local_phase=left_local,
        right_local_phase=right_local,
        left_contact=left_contact,
        right_contact=right_contact,
        flight_mask=flight_mask,
        support_weight_left=support_weight_left,
        support_weight_right=support_weight_right,
    )
