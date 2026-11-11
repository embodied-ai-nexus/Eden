"""Demo-like pick-and-place keyframes: a batched port of OCBench's ``CubePrimitive``.

``ocbench/oracles/primitives/cube.py`` (``compute_keyframes``). One call samples a whole
pick -> carry -> place -> retreat plan for a batch of environments. Everything that makes it read
like a person rather than a script is sampled per plan:

* a grasp yaw that is usually the face nearest the current wrist yaw but sometimes a quarter turn
  off, a wrist tilt of up to 25 deg that carries (attenuated) into the place,
* approach heights, contact offsets and a slightly off-centre place approach,
* an optional pause before lifting, and *when* the fingers close (on arrival or after a beat),
* a carry midpoint that bows to one side and, a third of the time, overshoots the target,
* +-3 % jitter on every keyframe time.

A failed pick is retried from wherever the hand is (:attr:`PickPlaceRequest.num_retries`), and
every random offset shrinks by ``0.75`` per retry — a person is more careful the second time.

Poses here are *effector rotations* in OCBench's sense — the rotation of a gripper that points
straight down when it is the identity — so yaw, roll and pitch read as the wrist's heading and
tilt. :func:`tool_quat` maps one to the tool link's world orientation through the rig's
``down_quat``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import torch

from eden.extensions.scripted_demos.trajectory import Keyframes
from eden.utils.geom import quat_from_euler_xyz, quat_mul, transform_by_quat, wrap_to_pi, yaw_from_quat


class Slot(IntEnum):
    """Keyframe layout of a pick-and-place plan (OCBench's keyframe names, in time order)."""

    INITIAL = 0
    PICK = 1
    PICK_START = 2
    PICK_END = 3
    PICK_HOLD = 4
    POSTPICK = 5
    CLEARANCE = 6
    PLACE = 7
    PLACE_START = 8
    PLACE_END = 9
    POSTPLACE = 10
    FINAL = 11


NUM_SLOTS = len(Slot)
#: Keyframes the hand comes to rest at (zero spline tangent).
STOP_SLOTS = (Slot.INITIAL, Slot.PICK_HOLD, Slot.PLACE_START, Slot.PLACE_END, Slot.FINAL)
#: The plan-time windows that get a sideways wobble: the two vertical approaches and the two retreats.
WOBBLE_WINDOWS = (
    (Slot.PICK, Slot.PICK_START),
    (Slot.PICK_HOLD, Slot.POSTPICK),
    (Slot.PLACE, Slot.PLACE_START),
    (Slot.PLACE_END, Slot.POSTPLACE),
)
#: Keyframes the plan clock waits at until the hand has caught up, in time order. The two hover
#: poses are OCBench's gates. The grasp and release poses are not in OCBench: its 50 Hz joint-space
#: arm tracks the plan within millimetres, but a laggier one (SIMPLER's WidowX at 5 Hz trails a
#: descent by several cm) would close the fingers above the object, then press it into the table.
GATE_SLOTS = (Slot.PICK, Slot.PICK_START, Slot.PLACE, Slot.PLACE_START)
#: The gates at the object: the hand must be at the grasp / release pose, not merely near it.
CONTACT_GATE_SLOTS = (Slot.PICK_START, Slot.PLACE_START)


@dataclass
class PickPlaceParams:
    """The tunables of :func:`sample_pick_place`, OCBench's values by default.

    Attributes
    ----------
    peak_speed : float
        Peak tool speed [m/s] of a nominal-speed segment; the plan's segments are stretched so a
        minimum-jerk profile does not exceed ``peak_speed / segment_dt``.
    descent_height : tuple[float, float]
        Height [m] of the hover pose above the pick / place pose, along the approach axis.
    descent_xy : tuple[float, float]
        Sideways offset [m] of the place hover pose from straight above the target.
    max_tilt : float
        Largest wrist tilt [rad] at the pick.
    p_pick_hold : float
        Probability of a pause between closing on the object and lifting it.
    p_overshoot : float
        Probability the carry midpoint overshoots the target instead of sitting between the two.
    gate_tolerance : float
        Distance [m] the tool must be within before the plan clock passes a hover gate.
    contact_gate_tolerance : float
        The same at the grasp and release poses (:data:`CONTACT_GATE_SLOTS`).
    retry_noise_decay : float
        Every random offset is scaled by ``retry_noise_decay ** num_retries``.
    """

    peak_speed: float = 0.3
    descent_height: tuple[float, float] = (0.06, 0.22)
    descent_xy: tuple[float, float] = (0.016, 0.06)
    max_tilt: float = math.radians(25.0)
    p_pick_hold: float = 0.4
    p_overshoot: float = 0.35
    gate_tolerance: float = 0.04
    contact_gate_tolerance: float = 0.01
    retry_noise_decay: float = 0.75


@dataclass
class PickPlaceRequest:
    """What to pick and where to put it, for a batch of ``B`` environments.

    Attributes
    ----------
    tool_pos : torch.Tensor
        ``(B, 3)`` current tool-point position.
    tool_rot : torch.Tensor
        ``(B, 4)`` current effector rotation (wxyz, identity = pointing straight down).
    object_pos, object_yaw : torch.Tensor
        ``(B, 3)`` / ``(B,)`` object frame position and heading.
    goal_pos, goal_yaw : torch.Tensor
        ``(B, 3)`` / ``(B,)`` where the object frame should end up.
    final_pos, final_yaw : torch.Tensor
        ``(B, 3)`` / ``(B,)`` where the hand goes once it has let go.
    release : torch.Tensor
        ``(B,)`` bool. False ends the plan holding the object at the goal (a lift), dropping the
        place hover, the release and the retreat.
    segment_dt : torch.Tensor
        ``(B,)`` seconds per nominal segment — the inverse of the demonstrator's speed.
    num_retries : torch.Tensor
        ``(B,)`` long, failed picks so far for this object.
    yaw_window : torch.Tensor
        ``(B, 2)`` effector headings [rad] the wrist can reach from where it is now (low, high).
        Every planned heading is moved into it by a symmetry of what it grasps — see
        :func:`fit_yaw`. ``(-inf, inf)`` when the wrist is unlimited.
    object_symmetry : int
        The object looks the same every ``2 pi / object_symmetry`` of heading (4 for a cube): the
        grasp heading and the place heading are free up to that period.
    """

    tool_pos: torch.Tensor
    tool_rot: torch.Tensor
    object_pos: torch.Tensor
    object_yaw: torch.Tensor
    goal_pos: torch.Tensor
    goal_yaw: torch.Tensor
    final_pos: torch.Tensor
    final_yaw: torch.Tensor
    release: torch.Tensor
    segment_dt: torch.Tensor
    num_retries: torch.Tensor
    yaw_window: torch.Tensor
    object_symmetry: int = 4


@dataclass
class PickPlacePlan:
    """A sampled plan: keyframes in effector rotations, plus its gates, wobble and pick check.

    Attributes
    ----------
    keyframes : Keyframes
        :meth:`~Keyframes.filled` and retimed, in the :class:`Slot` layout; ``quat`` holds
        effector rotations (map through :func:`tool_quat` before driving the arm).
    gate_times, gate_pos, gate_tolerance : torch.Tensor
        ``(B, G)`` / ``(B, G, 3)`` / ``(B, G)`` the :data:`GATE_SLOTS` keyframes and how close the tool
        must come to each.
    gate_valid : torch.Tensor
        ``(B, G)`` bool, gates this plan uses.
    wobble_windows, wobble_curves : torch.Tensor
        ``(B, W, 2)`` / ``(B, W, 2, 2)`` for :class:`~eden.extensions.scripted_demos.trajectory.Wobble`.
    pick_check_time : torch.Tensor
        ``(B,)`` plan time (``postpick``) from which the pick check may run; the controller then also waits
        for the hand to rise before judging it.
    """

    keyframes: Keyframes
    gate_times: torch.Tensor
    gate_pos: torch.Tensor
    gate_tolerance: torch.Tensor
    gate_valid: torch.Tensor
    wobble_windows: torch.Tensor
    wobble_curves: torch.Tensor
    pick_check_time: torch.Tensor


def effector_rot(
    yaw: torch.Tensor, roll: torch.Tensor | float = 0.0, pitch: torch.Tensor | float = 0.0
) -> torch.Tensor:
    """Effector rotation ``(B, 4)`` from heading and tilt: ``Rz(yaw) Ry(pitch) Rx(roll)``, OCBench's ``to_pose``."""
    roll = torch.as_tensor(roll, device=yaw.device).expand_as(yaw)
    pitch = torch.as_tensor(pitch, device=yaw.device).expand_as(yaw)
    return quat_from_euler_xyz(roll, pitch, yaw)


def tool_quat(rot: torch.Tensor, down_quat: torch.Tensor) -> torch.Tensor:
    """World orientation of the tool link for effector rotation ``rot`` (``down_quat`` is the tool's straight-down
    pose).
    """
    return quat_mul(rot, down_quat.expand_as(rot))


#: A two-finger gripper is symmetric under a half turn about its approach axis.
GRIPPER_PERIOD = math.pi


def fit_yaw(yaw: torch.Tensor, window: torch.Tensor, period: float) -> torch.Tensor:
    """Shift each heading by a whole number of ``period`` into its ``(low, high)`` window.

    The shift that lands inside the window and is smallest wins; if none does (a window narrower
    than the period), the one closest to it. This is what keeps a 7-DOF arm's last joint off its
    stop — OCBench's UR5e wrist turns +-2 pi, a Panda's +-2.9 rad: the planner is free to grasp a
    cube by any of its faces and to finish with the fingers either way round, so it picks the
    equivalent heading the wrist can still reach.
    """
    shifts = torch.arange(-4, 5, device=yaw.device, dtype=yaw.dtype) * period
    candidates = yaw.unsqueeze(-1) + shifts  # (B, 9)
    low, high = window[:, :1], window[:, 1:]
    outside = (low - candidates).clamp(min=0.0) + (candidates - high).clamp(min=0.0)
    score = outside + 1e-3 * shifts.abs()
    return candidates.gather(1, score.argmin(dim=1, keepdim=True)).squeeze(1)


class _Sampler:
    """Uniform / Bernoulli / categorical draws from one generator on one device."""

    def __init__(self, generator: torch.Generator, batch: int, device: torch.device):
        self.generator = generator
        self.batch = batch
        self.device = device

    def uniform(self, low, high, *shape: int) -> torch.Tensor:
        u = torch.rand(self.batch, *shape, generator=self.generator, device=self.device)
        return low + (high - low) * u

    def bernoulli(self, p) -> torch.Tensor:
        return self.uniform(0.0, 1.0) < p

    def sign(self) -> torch.Tensor:
        return torch.where(self.bernoulli(0.5), 1.0, -1.0)

    def choice(self, values: tuple[float, ...], probs: tuple[float, ...]) -> torch.Tensor:
        idx = torch.multinomial(
            torch.tensor(probs, device=self.device).expand(self.batch, -1), 1, generator=self.generator
        ).squeeze(1)
        return torch.tensor(values, device=self.device)[idx]


def sample_pick_place(
    request: PickPlaceRequest,
    *,
    generator: torch.Generator,
    workspace: tuple[torch.Tensor, torch.Tensor],
    params: PickPlaceParams | None = None,
) -> PickPlacePlan:
    """Sample a demo-like pick-and-place plan per environment (OCBench ``CubePrimitive.compute_keyframes``).

    Parameters
    ----------
    request : PickPlaceRequest
        The batch to plan for.
    generator : torch.Generator
        Source of every random draw, on the request's device.
    workspace : tuple[torch.Tensor, torch.Tensor]
        ``(3,)`` low / high corners the carry midpoint is clipped to.
    params : PickPlaceParams, optional
        Tunables; OCBench's values by default.

    Returns
    -------
    PickPlacePlan
        The plan, in the :class:`Slot` layout.
    """
    p = params or PickPlaceParams()
    batch, device = request.tool_pos.shape[0], request.tool_pos.device
    rng = _Sampler(generator, batch, device)
    dt = request.segment_dt
    ns = p.retry_noise_decay ** request.num_retries.to(torch.float32)  # OCBench's ``noise_scale``
    ns2 = ns.unsqueeze(-1)
    up = torch.tensor([0.0, 0.0, 1.0], device=device).expand(batch, 3)

    pos = torch.zeros(batch, NUM_SLOTS, 3, device=device)
    rot = torch.zeros(batch, NUM_SLOTS, 4, device=device)
    valid = torch.ones(batch, NUM_SLOTS, dtype=torch.bool, device=device)

    use_hold = rng.bernoulli(p.p_pick_hold * ns)
    release = request.release
    valid[:, Slot.PICK_HOLD] = use_hold
    for slot in (Slot.PLACE, Slot.PLACE_END, Slot.POSTPLACE, Slot.FINAL):
        valid[:, slot] = release

    # --- grasp heading and tilt -----------------------------------------------------------------
    # The object face nearest the current wrist heading (a cube is 4-fold symmetric), usually; a
    # quarter turn off either way 30 % of the time.
    tool_yaw = yaw_from_quat(request.tool_rot)
    face = 2 * math.pi / request.object_symmetry
    quarter = math.pi / 2
    nearest_face = face * torch.round(wrap_to_pi(tool_yaw - request.object_yaw) / face)
    yaw_offset = nearest_face + rng.choice((-quarter, 0.0, quarter), (0.15, 0.7, 0.15)) + rng.uniform(-0.04, 0.04) * ns
    pick_yaw = fit_yaw(request.object_yaw + yaw_offset, request.yaw_window, face)
    # The object turns by the goal heading's nearest equivalent (at most half a face), not by up
    # to a full turn because the two headings were read off different branches of atan2.
    goal_turn = request.goal_yaw - request.object_yaw
    goal_turn = goal_turn - face * torch.round(goal_turn / face)
    place_yaw = fit_yaw(pick_yaw + goal_turn, request.yaw_window, face)
    tilt_angle = rng.uniform(-math.pi, math.pi)
    tilt_mag = rng.uniform(0.0, p.max_tilt) * ns
    pick_tilt = tilt_mag.unsqueeze(-1) * torch.stack([tilt_angle.cos(), tilt_angle.sin()], dim=-1)
    place_tilt = 0.2 * pick_tilt + rng.uniform(-0.06, 0.06, 2) * ns2

    pick_rot = effector_rot(pick_yaw, pick_tilt[:, 0], pick_tilt[:, 1])
    place_rot = effector_rot(place_yaw, place_tilt[:, 0], place_tilt[:, 1])

    # --- pick: hover, descend with a small contact offset, close, (pause), lift -----------------
    pick_axis = transform_by_quat(up, pick_rot)
    contact_angle = rng.uniform(-math.pi, math.pi)
    zeros = torch.zeros_like(contact_angle)
    contact_dir = transform_by_quat(torch.stack([contact_angle.cos(), contact_angle.sin(), zeros], dim=-1), pick_rot)
    contact_shift = (rng.uniform(0.0, 0.012) * ns).unsqueeze(-1)
    start_lift = (rng.uniform(0.0, 0.012) * ns).unsqueeze(-1)
    end_depth = (rng.uniform(-0.004, 0.006) * ns).unsqueeze(-1)
    pick_start_pos = request.object_pos + start_lift * pick_axis - 0.5 * contact_shift * contact_dir
    pick_end_pos = request.object_pos + end_depth * pick_axis + 0.5 * contact_shift * contact_dir
    pick_end_tilt = pick_tilt + rng.uniform(-0.035, 0.035, 2) * ns2
    pick_end_rot = effector_rot(pick_yaw + rng.uniform(-0.06, 0.06) * ns, pick_end_tilt[:, 0], pick_end_tilt[:, 1])
    pick_hover = pick_start_pos + rng.uniform(*p.descent_height).unsqueeze(-1) * pick_axis

    pos[:, Slot.INITIAL], rot[:, Slot.INITIAL] = request.tool_pos, request.tool_rot
    pos[:, Slot.PICK], rot[:, Slot.PICK] = pick_hover, pick_rot
    pos[:, Slot.PICK_START], rot[:, Slot.PICK_START] = pick_start_pos, pick_rot
    pos[:, Slot.PICK_END], rot[:, Slot.PICK_END] = pick_end_pos, pick_end_rot
    pos[:, Slot.PICK_HOLD], rot[:, Slot.PICK_HOLD] = pick_end_pos, pick_end_rot
    pos[:, Slot.POSTPICK], rot[:, Slot.POSTPICK] = pick_hover, pick_rot

    # --- place: hover slightly off to one side, descend, open, retreat --------------------------
    place_forward = torch.stack([place_yaw.cos(), place_yaw.sin()], dim=-1)
    place_side = torch.stack([-place_yaw.sin(), place_yaw.cos()], dim=-1) * rng.sign().unsqueeze(-1)
    place_offset = (rng.uniform(*p.descent_xy) * ns).unsqueeze(-1) * place_side + (
        rng.uniform(-0.02, 0.02) * ns
    ).unsqueeze(-1) * place_forward
    place_z = request.goal_pos[:, 2] + rng.uniform(*p.descent_height)
    place_hover = torch.cat([request.goal_pos[:, :2] + place_offset, place_z.unsqueeze(-1)], dim=-1)

    pos[:, Slot.PLACE], rot[:, Slot.PLACE] = place_hover, place_rot
    pos[:, Slot.PLACE_START], rot[:, Slot.PLACE_START] = request.goal_pos, place_rot
    pos[:, Slot.PLACE_END], rot[:, Slot.PLACE_END] = request.goal_pos, place_rot
    pos[:, Slot.POSTPLACE], rot[:, Slot.POSTPLACE] = place_hover, place_rot
    final_yaw = fit_yaw(request.final_yaw, request.yaw_window, GRIPPER_PERIOD)
    pos[:, Slot.FINAL], rot[:, Slot.FINAL] = request.final_pos, effector_rot(final_yaw)

    # --- carry midpoint: bowed to one side, sometimes overshooting ------------------------------
    # A lift has no place hover: the carry heads for the goal itself.
    carry_end = torch.where(release.unsqueeze(-1), place_hover, request.goal_pos)
    diff = carry_end - pick_hover
    xy_norm = diff[:, :2].norm(dim=-1, keepdim=True)
    random_heading = rng.uniform(-math.pi, math.pi)
    fallback_side = torch.stack([-random_heading.sin(), random_heading.cos()], dim=-1)
    side = torch.where(
        xy_norm > 1e-6, torch.stack([-diff[:, 1], diff[:, 0]], dim=-1) / xy_norm.clamp(min=1e-6), fallback_side
    )
    overshoot = rng.bernoulli(p.p_overshoot)
    alpha = torch.where(
        overshoot,
        rng.uniform(1.15, 1.35) * ns + (1.0 - ns),
        (0.5 + rng.uniform(-0.08, 0.08) * ns).clamp(0.15, 0.85),
    )
    side_offset = (rng.uniform(0.04, 0.14) * ns * rng.sign()).unsqueeze(-1)
    mid_xy = (
        pick_hover[:, :2] + alpha.unsqueeze(-1) * diff[:, :2] + side_offset * side + rng.uniform(-0.02, 0.02, 2) * ns2
    )
    mid_z = torch.maximum(pick_hover[:, 2], carry_end[:, 2]) + rng.uniform(0.03, 0.12)
    low, high = workspace
    mid_pos = torch.maximum(torch.minimum(torch.cat([mid_xy, mid_z.unsqueeze(-1)], dim=-1), high), low)
    mid_yaw = pick_yaw + alpha * (place_yaw - pick_yaw) + rng.uniform(-0.4, 0.4) * ns
    a = alpha.unsqueeze(-1)
    mid_tilt = (1.0 - a) * pick_tilt + a * place_tilt + rng.uniform(-0.05, 0.05, 2) * ns2
    pos[:, Slot.CLEARANCE], rot[:, Slot.CLEARANCE] = mid_pos, effector_rot(mid_yaw, mid_tilt[:, 0], mid_tilt[:, 1])

    # --- timing ---------------------------------------------------------------------------------
    dur = torch.zeros(batch, NUM_SLOTS, device=device)
    dur[:, Slot.PICK] = 1.0
    dur[:, Slot.PICK_START] = 1.8
    dur[:, Slot.PICK_END] = rng.uniform(0.75, 1.25)
    dur[:, Slot.PICK_HOLD] = rng.uniform(0.1, 0.22)
    dur[:, Slot.POSTPICK] = rng.uniform(0.75, 1.05)
    dur[:, Slot.CLEARANCE] = 1.0
    dur[:, Slot.PLACE] = 1.0
    dur[:, Slot.PLACE_START] = 1.5
    dur[:, Slot.PLACE_END] = rng.uniform(0.75, 1.1)
    dur[:, Slot.POSTPLACE] = rng.uniform(0.8, 1.15)
    dur[:, Slot.FINAL] = 1.0
    times = (dur * valid * dt.unsqueeze(-1)).cumsum(dim=1)
    jitter = rng.uniform(-1.0, 1.0, NUM_SLOTS) * 0.03 * dt.unsqueeze(-1)
    jitter[:, Slot.INITIAL] = 0.0
    times = times + jitter * valid

    # --- gripper: closes on arrival or a beat later, opens at the place -------------------------
    # The gripper value ramps over the segment *ending* at its keyframe. OCBench also closes at
    # pick_start (10 % without a pause), ramping over the descent — fine for its continuous gripper,
    # but a binary one fires halfway down, above the object. Dropped here: closing ramps over the
    # contact segment (pick_start -> pick_end) or the pause (pick_end -> pick_hold), with the hand at
    # the object either way.
    close_slot = torch.where(
        use_hold, rng.choice((Slot.PICK_END, Slot.PICK_HOLD), (0.65, 0.35)), int(Slot.PICK_END)
    ).long()
    open_slot = torch.where(release, int(Slot.PLACE_END), NUM_SLOTS).long()
    slots = torch.arange(NUM_SLOTS, device=device)
    grasp = ((slots >= close_slot.unsqueeze(-1)) & (slots < open_slot.unsqueeze(-1))).float()

    stop = torch.zeros(batch, NUM_SLOTS, dtype=torch.bool, device=device)
    stop[:, list(STOP_SLOTS)] = True
    keyframes = Keyframes(times=times, pos=pos, quat=rot, grasp=grasp, stop=stop, valid=valid).filled()
    keyframes = keyframes.retimed(peak_speed=p.peak_speed / dt)

    # --- gates, wobble, pick check (read off the retimed layout by slot) ------------------------
    t = keyframes.times
    gate_slots = list(GATE_SLOTS)
    windows = torch.stack([torch.stack([t[:, a], t[:, b]], dim=-1) for a, b in WOBBLE_WINDOWS], dim=1)
    window_valid = torch.stack([valid[:, b] & (valid[:, a] | (a == Slot.PICK_HOLD)) for a, b in WOBBLE_WINDOWS], dim=1)
    curves = rng.uniform(-0.025, 0.025, len(WOBBLE_WINDOWS), 2, 2) * ns.view(-1, 1, 1, 1)
    curves = curves * window_valid.view(batch, -1, 1, 1)
    return PickPlacePlan(
        keyframes=keyframes,
        gate_times=t[:, gate_slots],
        gate_pos=keyframes.pos[:, gate_slots],
        gate_tolerance=torch.tensor(
            [p.contact_gate_tolerance if slot in CONTACT_GATE_SLOTS else p.gate_tolerance for slot in GATE_SLOTS],
            device=device,
        ).expand(batch, -1),
        gate_valid=valid[:, gate_slots],
        wobble_windows=windows,
        wobble_curves=curves,
        pick_check_time=t[:, Slot.POSTPICK],
    )
