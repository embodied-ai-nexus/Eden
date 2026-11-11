"""Batched keyframe trajectories with demo-like timing: the motion layer of OCBench's scripted policies.

Port of ``ocbench/oracles/primitives/primitive.py`` (``Primitive._compute_smooth_plan``,
``_smooth_keyframe_times``, ``current_plan_action``), vectorized over environments.

A primitive samples a handful of *keyframes* — tool poses, times and a gripper value — and the
trajectory between them is what makes a scripted rollout read like a human demonstration rather
than a rule-based controller:

* **Position** follows a cubic Hermite spline. The tangent is zero at *stop* keyframes (where a
  person pauses: before closing the gripper, at the place pose, at the end) and passes through the
  others at the neighbours' average velocity, clamped to 1.25x the faster neighbouring segment.
* **Rotation and gripper** ease with the quintic smoothstep ``u^3 (10 - 15u + 6u^2)``.
* **Timing** is stretched so no segment exceeds a peak speed: a minimum-jerk profile peaks at
  ``1.875 d / T``, so a segment of length ``d`` gets at least ``T = 1.875 d / v_peak``.
* **Gates** freeze the plan clock until the tool is within a tolerance of a keyframe, so the plan
  never runs ahead of an arm that lags behind it.
* **Wobble** adds a small sideways bulge ``16 u^2 (1 - u)^2 ((1 - u) c_1 + u c_2)`` to the vertical
  approach and retreat phases, which a human never executes as a straight line.

Every tensor is batched over environments in the first dimension and lives on the env device; a
plan is written for a subset of environments with :meth:`KeyframeTrajectory.assign` and read for
all of them with :meth:`KeyframeTrajectory.evaluate`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from eden.utils.geom import normalize, quat_slerp

#: Ratio of a minimum-jerk profile's peak speed to its mean speed.
MIN_JERK_PEAK_TO_MEAN = 1.875
#: Hermite tangents through a non-stop keyframe are clamped to this multiple of the faster neighbouring segment.
TANGENT_SPEED_LIMIT = 1.25
_EPS = 1e-6


def smoothstep5(u: torch.Tensor) -> torch.Tensor:
    """Quintic smoothstep ``u^3 (10 - 15u + 6u^2)`` on ``u`` clamped to ``[0, 1]`` (zero velocity and acceleration at
    both ends).
    """
    u = u.clamp(0.0, 1.0)
    return u**3 * (10.0 - 15.0 * u + 6.0 * u * u)


@dataclass
class Keyframes:
    """A batch of keyframe plans, one row per environment.

    Optional keyframes (OCBench's ``pick_hold``, or the release half of a pick that only lifts) are
    authored in place and switched off with ``valid``; :meth:`compacted` moves the valid ones to the
    front, so the spline is built over exactly the keyframes each environment uses.

    Attributes
    ----------
    times : torch.Tensor
        ``(B, K)`` absolute keyframe times [s], non-decreasing over the valid keyframes.
    pos : torch.Tensor
        ``(B, K, 3)`` tool positions in the world frame.
    quat : torch.Tensor
        ``(B, K, 4)`` tool orientations (wxyz) in the world frame.
    grasp : torch.Tensor
        ``(B, K)`` gripper command in ``[0, 1]`` (0 open, 1 closed).
    stop : torch.Tensor
        ``(B, K)`` bool, keyframes the tool comes to rest at (zero Hermite tangent).
    valid : torch.Tensor
        ``(B, K)`` bool, keyframes this environment's plan uses. The first must be valid.
    """

    times: torch.Tensor
    pos: torch.Tensor
    quat: torch.Tensor
    grasp: torch.Tensor
    stop: torch.Tensor
    valid: torch.Tensor

    def compacted(self) -> Keyframes:
        """The valid keyframes first, the tail padded with the last valid one (so it adds no motion and no time)."""
        num_kf = self.times.shape[1]
        # A stable sort on ``~valid`` keeps the valid keyframes in their authored order.
        order = torch.argsort((~self.valid).to(torch.int8), dim=1, stable=True)
        num_valid = self.valid.sum(dim=1)
        last = (num_valid - 1).clamp(min=0).unsqueeze(1)
        slots = torch.arange(num_kf, device=self.times.device).unsqueeze(0)
        # Tail slots read the last valid keyframe.
        gather = torch.where(slots < num_valid.unsqueeze(1), order, order.gather(1, last))

        def take(x: torch.Tensor) -> torch.Tensor:
            index = gather.view(*gather.shape, *([1] * (x.dim() - 2))).expand_as(x)
            return x.gather(1, index)

        tail = slots >= num_valid.unsqueeze(1)
        return Keyframes(
            times=take(self.times),
            pos=take(self.pos),
            quat=take(self.quat),
            grasp=take(self.grasp),
            stop=take(self.stop) | tail,
            valid=~tail,
        )

    def filled(self) -> Keyframes:
        """Every invalid keyframe replaced by the last valid one before it (time, pose and grasp).

        Keeps the authored layout, so a primitive can still read a keyframe's time by its slot
        after :meth:`retimed` — an invalid slot then reports its predecessor's time.
        """
        num_kf = self.times.shape[1]
        slots = torch.arange(num_kf, device=self.times.device).expand_as(self.times)
        source = torch.where(self.valid, slots, 0).cummax(dim=1).values

        def take(x: torch.Tensor) -> torch.Tensor:
            return x.gather(1, source.view(*source.shape, *([1] * (x.dim() - 2))).expand_as(x))

        return Keyframes(take(self.times), take(self.pos), take(self.quat), take(self.grasp), self.stop, self.valid)

    def retimed(self, peak_speed: torch.Tensor) -> Keyframes:
        """Stretch every segment that would exceed ``peak_speed`` (``(B,)`` m/s) under a minimum-jerk profile.

        OCBench ``Primitive._smooth_keyframe_times``. Expects a :meth:`filled` or :meth:`compacted`
        plan, where an unused keyframe duplicates a used one: zero length and zero duration, so it
        stays put and the segments around it are timed as if it were absent.
        """
        dt = self.times.diff(dim=1)
        dist = (self.pos[:, 1:] - self.pos[:, :-1]).norm(dim=-1)
        min_dt = MIN_JERK_PEAK_TO_MEAN * dist / peak_speed.unsqueeze(1)
        dt = torch.maximum(dt, min_dt)
        times = torch.cat([self.times[:, :1], self.times[:, :1] + dt.cumsum(dim=1)], dim=1)
        return Keyframes(times, self.pos, self.quat, self.grasp, self.stop, self.valid)


def hermite_tangents(keyframes: Keyframes) -> torch.Tensor:
    """Per-keyframe position tangents ``(B, K, 3)`` for a :meth:`~Keyframes.compacted` plan.

    Zero at the first keyframe, at the last valid one, and at every stop keyframe; elsewhere the
    central difference, clamped to :data:`TANGENT_SPEED_LIMIT` x the faster neighbouring segment.
    """
    times, pos = keyframes.times, keyframes.pos
    tangents = torch.zeros_like(pos)
    if pos.shape[1] < 3:
        return tangents
    t_prev, t_mid, t_next = times[:, :-2], times[:, 1:-1], times[:, 2:]
    p_prev, p_mid, p_next = pos[:, :-2], pos[:, 1:-1], pos[:, 2:]
    central = (p_next - p_prev) / (t_next - t_prev).clamp(min=_EPS).unsqueeze(-1)
    prev_speed = (p_mid - p_prev).norm(dim=-1) / (t_mid - t_prev).clamp(min=_EPS)
    next_speed = (p_next - p_mid).norm(dim=-1) / (t_next - t_mid).clamp(min=_EPS)
    max_speed = TANGENT_SPEED_LIMIT * torch.maximum(prev_speed, next_speed)
    speed = central.norm(dim=-1)
    central = central * torch.where(speed > max_speed, max_speed / speed.clamp(min=_EPS), 1.0).unsqueeze(-1)
    # The last valid keyframe is followed by padding (or nothing), so ``valid[i + 1]`` is False there.
    rests = keyframes.stop[:, 1:-1] | ~keyframes.valid[:, 2:]
    tangents[:, 1:-1] = torch.where(rests.unsqueeze(-1), 0.0, central)
    return tangents


class KeyframeTrajectory:
    """The current keyframe plan of every environment, evaluated at a per-environment plan time.

    Parameters
    ----------
    num_envs : int
        Number of environments.
    num_keyframes : int
        Keyframes per plan (the primitive's layout, before compaction).
    device : torch.device
        Device of every buffer.
    """

    def __init__(self, num_envs: int, num_keyframes: int, device: torch.device):
        if num_keyframes < 2:
            raise ValueError(
                f"A keyframe trajectory needs at least 2 keyframes per plan, got num_keyframes={num_keyframes}."
            )
        self.num_envs = num_envs
        self.num_keyframes = num_keyframes
        self.device = device
        self.times = torch.zeros(num_envs, num_keyframes, device=device)
        self.pos = torch.zeros(num_envs, num_keyframes, 3, device=device)
        self.quat = torch.zeros(num_envs, num_keyframes, 4, device=device)
        self.quat[..., 0] = 1.0
        self.grasp = torch.zeros(num_envs, num_keyframes, device=device)
        self.tangents = torch.zeros(num_envs, num_keyframes, 3, device=device)
        #: ``(N,)`` time of the last valid keyframe: the plan is complete once the plan clock reaches it.
        self.duration = torch.zeros(num_envs, device=device)

    def assign(self, envs_idx: torch.Tensor, keyframes: Keyframes) -> None:
        """Write a new plan for ``envs_idx`` (long indices). ``keyframes`` need not be compacted."""
        if keyframes.times.shape != (len(envs_idx), self.num_keyframes):
            raise ValueError(
                f"Keyframes of shape {tuple(keyframes.times.shape)} do not fit a trajectory of "
                f"{self.num_keyframes} keyframes for {len(envs_idx)} envs; expected ({len(envs_idx)}, {self.num_keyframes})."
            )
        if not bool(keyframes.valid[:, 0].all()):
            raise ValueError("The first keyframe of every plan must be valid: it is where the tool starts.")
        compact = keyframes.compacted()
        self.times[envs_idx] = compact.times
        self.pos[envs_idx] = compact.pos
        self.quat[envs_idx] = normalize(compact.quat)
        self.grasp[envs_idx] = compact.grasp
        self.tangents[envs_idx] = hermite_tangents(compact)
        self.duration[envs_idx] = compact.times[:, -1]

    def evaluate(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tool position ``(N, 3)``, orientation ``(N, 4)`` and gripper command ``(N,)`` at plan time ``t`` ``(N,)``."""
        t = torch.minimum(t, self.duration).unsqueeze(1)
        seg = (torch.searchsorted(self.times, t, right=True) - 1).clamp(0, self.num_keyframes - 2)
        nxt = seg + 1

        def at(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
            return x.gather(1, index.view(*index.shape, *([1] * (x.dim() - 2))).expand(-1, 1, *x.shape[2:])).squeeze(1)

        t0, t1 = at(self.times, seg), at(self.times, nxt)
        dt = (t1 - t0).clamp(min=_EPS)
        u = ((t.squeeze(1) - t0) / dt).clamp(0.0, 1.0)
        u2, u3 = u * u, u * u * u
        h00, h10, h01, h11 = (2 * u3 - 3 * u2 + 1), (u3 - 2 * u2 + u), (-2 * u3 + 3 * u2), (u3 - u2)
        pos = (
            h00.unsqueeze(-1) * at(self.pos, seg)
            + (h10 * dt).unsqueeze(-1) * at(self.tangents, seg)
            + h01.unsqueeze(-1) * at(self.pos, nxt)
            + (h11 * dt).unsqueeze(-1) * at(self.tangents, nxt)
        )
        eased = smoothstep5(u)
        quat = normalize(quat_slerp(at(self.quat, seg), at(self.quat, nxt), eased.unsqueeze(-1)))
        grasp = (1.0 - eased) * at(self.grasp, seg) + eased * at(self.grasp, nxt)
        return pos, quat, grasp


class Wobble:
    """Sideways bulges over a few plan-time windows (OCBench ``CubePrimitive``'s ``_vertical_phases``).

    Inside window ``[t_0, t_1]`` the tool's xy is offset by ``16 u^2 (1 - u)^2 ((1 - u) c_1 + u c_2)``
    with ``u`` the window's progress, so the offset is zero, with zero slope, at both ends.

    Parameters
    ----------
    num_envs : int
        Number of environments.
    num_windows : int
        Windows per plan.
    device : torch.device
        Device of every buffer.
    """

    def __init__(self, num_envs: int, num_windows: int, device: torch.device):
        self.windows = torch.zeros(num_envs, num_windows, 2, device=device)
        self.curves = torch.zeros(num_envs, num_windows, 2, 2, device=device)

    def assign(self, envs_idx: torch.Tensor, windows: torch.Tensor, curves: torch.Tensor) -> None:
        """Set ``(B, W, 2)`` windows ``[t_0, t_1]`` and ``(B, W, 2, 2)`` control offsets ``(c_1, c_2)`` for
        ``envs_idx``.
        """
        self.windows[envs_idx] = windows
        self.curves[envs_idx] = curves

    def offset(self, t: torch.Tensor) -> torch.Tensor:
        """The ``(N, 2)`` xy offset at plan time ``t`` ``(N,)`` (the first window containing ``t`` wins, as in
        OCBench).
        """
        t0, t1 = self.windows[..., 0], self.windows[..., 1]
        t = t.unsqueeze(1)
        inside = (t >= t0) & (t <= t1)
        first = inside & (inside.cumsum(dim=1) == 1)
        u = ((t - t0) / (t1 - t0).clamp(min=_EPS)).clamp(0.0, 1.0).unsqueeze(-1)
        bulge = 16.0 * u**2 * (1.0 - u) ** 2 * ((1.0 - u) * self.curves[..., 0, :] + u * self.curves[..., 1, :])
        return (bulge * first.unsqueeze(-1)).sum(dim=1)
