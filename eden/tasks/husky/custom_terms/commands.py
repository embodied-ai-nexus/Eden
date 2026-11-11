"""Skate command + gait-phase state machine for the HUSKY skateboarding task.

Port of ``SkateUniformVelocityCommand`` plus the phase/contact machinery that
`TeleHuman/humanoid_skateboarding <https://github.com/TeleHuman/humanoid_skateboarding>`_
(HUSKY, RSS 2026) implements directly on its custom env
(``G1SkaterManagerBasedRlEnv``). In Eden all of that state lives on this single
command term:

- the command ``[lin_vel_x, relative_heading]`` (heading is relative to the
  skateboard heading captured at resample time),
- the 6 s phase clock, pinned at 0 while the speed command is ~zero,
- the four-way gait phase ``[push, steer, push→steer, steer→push]``,
- latched contact states (feet↔ground, feet↔board, wheels↔ground) and the
  per-foot contact-force vectors the critic observes,
- the bezier/slerp transition targets that blend the tracked body links from
  their pose at transition entry to the recorded push/steer start poses.

Everything is refreshed once per control step in :meth:`post_physics_refresh`
(after physics, before terminations/rewards/observations — the same point in
the step where the source env refreshed it), so every consumer term reads
coherent state via ``env.command_manager.get_term("skate")``.

:class:`ConstantSkateCommand` is the frozen actor's counterpart: a held command plus
the phase clock, and none of the training-only state.

Frame conventions: tracked body poses are expressed in the skateboard root frame
(the MJCF body frames the reference poses were recorded in). The base links — the
pelvis and the board root — are read through ``get_pos/get_quat(relative=True)``,
because Genesis re-origins base links onto their inertial frame.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from genesis.utils.geom import inv_transform_by_quat, transform_by_quat

from eden.managers.command_manager import COMMAND_TERM_REGISTRY, CommandTerm
from eden.managers.terms.utils import resolve_link_idx
from eden.utils.assets import get_asset_path
from eden.utils.geom import heading_from_quat, subtract_frame_transforms, wrap_to_pi
from eden.utils.misc import assign_over_envs

if TYPE_CHECKING:
    from eden.entities.rigid import RigidEntity

# The 14 links whose poses are tracked through the push<->steer transitions
# (mjlab_husky ``beizer_names`` == ``slerp_names``).
TRACKED_LINK_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)

FEET_LINK_NAMES: tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link")
WHEEL_LINK_NAMES: tuple[str, ...] = ("front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel")
DECK_LINK_NAME: str = "board_tilt_body"

# Deck-frame marker sites (``rear_marker`` / ``front_marker`` in skateboard.xml),
# targets for the left / right foot during the steer stance.
MARKER_OFFSETS_DECK: tuple[tuple[float, float, float], ...] = ((-0.2, -0.03, 0.0), (0.2, -0.03, 0.0))
# Foot-plane probe sites on the ankle-roll link (``left_foot_1..4`` in the G1 MJCF).
FOOT_SITE_OFFSETS: tuple[tuple[float, float, float], ...] = (
    (0.035, 0.1, 0.0),
    (0.035, -0.1, 0.0),
    (0.2, 0.0, 0.0),
    (-0.15, 0.0, 0.0),
)


def advance_phase_clock(
    phase_length_buf: torch.Tensor, still: torch.Tensor, half_cycle_steps: int, phase_per_step: float
) -> torch.Tensor:
    """Tick the 6 s phase clock one control step in place and return the phase in ``[0, 1)``.

    The step count advances by one, except that a ``still`` env (speed command below threshold) is
    pinned back to 0 at each half-cycle boundary, so a robot told to stand never leaves the push
    stance. ``phase_per_step`` is ``control_dt / cycle_time_s``.
    """
    phase_length_buf += 1
    pin = still & ((phase_length_buf - 1) % half_cycle_steps == 0)
    phase_length_buf[pin] = 0
    return (phase_length_buf.float() * phase_per_step) % 1.0


@COMMAND_TERM_REGISTRY.register()
class SkateCommand(CommandTerm):
    """``[lin_vel_x, relative_heading]`` command + HUSKY gait-phase state machine."""

    entity_name: str = "robot"
    skateboard_name: str = "skateboard"
    ground_name: str = "plane"

    resampling_time_range: tuple[float, float] = (20.0, 20.0)
    lin_vel_x_range: tuple[float, float] = (0.0, 1.5)
    heading_range: tuple[float, float] = (-math.pi / 4, math.pi / 4)
    rel_heading_envs: float = 1.0
    rel_standing_envs: float = 0.0

    cycle_time_s: float = 6.0
    # Phase boundaries [push_start, push_end, steer_start, steer_end, cycle_end]:
    # push [0, 0.4), push->steer [0.4, 0.5), steer [0.5, 0.95), steer->push [0.95, 1].
    phase_ratios: tuple[float, float, float, float, float] = (0.0, 0.4, 0.5, 0.95, 1.0)
    still_vx_threshold: float = 0.1
    # Transition blends progress over a fixed phase span (source hard-codes /0.1 and /0.05).
    push2steer_blend_span: float = 0.1
    steer2push_blend_span: float = 0.05
    bezier_height_offset: float = 0.2

    # Contact-force thresholds [N] (source sensor pipelines).
    ground_contact_threshold: float = 2.0
    feet_contact_threshold: float = 5.0
    wheel_contact_threshold: float = 1.0

    ref_pose_file: str = "objects/skateboard/husky_skate_ref_poses.npz"

    @property
    def command(self) -> torch.Tensor:
        """The skate command ``[lin_vel_x, relative_heading]``, shape (num_envs, 2)."""
        return self.command_b

    @property
    def target_heading_w(self) -> torch.Tensor:
        """Absolute world heading target: reference heading + relative command."""
        return wrap_to_pi(self.heading_ref + self.command_b[:, 1])

    # ------------------------------------------------------------------
    # Build / reset
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Resolve entities/links/geoms, load the reference poses, allocate state."""
        env = self._env
        num_envs, device = env.num_envs, env.device
        robot: RigidEntity = env.entities[self.entity_name]
        board: RigidEntity = env.entities[self.skateboard_name]
        env.entities[self.ground_name]  # fail fast on a missing ground entity

        self._feet_idx = resolve_link_idx(robot, list(FEET_LINK_NAMES), device)
        self._wheel_idx = resolve_link_idx(board, list(WHEEL_LINK_NAMES), device)
        self._deck_idx = resolve_link_idx(board, [DECK_LINK_NAME], device)
        self._tracked_idx = resolve_link_idx(robot, list(TRACKED_LINK_NAMES), device)
        # Global (solver) link indices, for classifying `get_contacts` rows.
        self._feet_gidx = self._feet_idx + robot._entity.link_start
        self._board_link_range = (board._entity.link_start, board._entity.link_end)
        ground = env.entities[self.ground_name]
        self._ground_link_range = (ground._entity.link_start, ground._entity.link_end)
        self._wheel_gidx = self._wheel_idx + board._entity.link_start

        # The deck link carries two boxes: the full deck and the rear "marker" pad
        # the left foot must stand on. Genesis drops MJCF geom names, so resolve
        # them structurally: the marker is the rear box (init x approx. -0.2).
        deck_link = board._entity.links[int(self._deck_idx[0].item())]
        boxes = list(deck_link.geoms)
        if len(boxes) != 2:
            raise ValueError(
                f"Expected 2 collision geoms on skateboard link '{DECK_LINK_NAME}' (deck + rear marker), "
                f"found {len(boxes)} — the skateboard asset changed."
            )
        marker = min(boxes, key=lambda g: float(g.init_pos[0]))
        if not float(marker.init_pos[0]) < -0.1:
            raise ValueError(
                f"Could not identify the rear marker geom on '{DECK_LINK_NAME}': "
                f"geom init x-positions are {[float(g.init_pos[0]) for g in boxes]}."
            )
        self._marker_geom_idx = int(marker.idx)
        self._deck_geom_idx = int(next(g for g in boxes if g is not marker).idx)

        # Reference start poses (skateboard frame), rows keyed by link name.
        data = np.load(get_asset_path(self.ref_pose_file))
        names = [str(n) for n in data["link_names"]]
        rows = []
        for link_name in TRACKED_LINK_NAMES:
            if link_name not in names:
                raise ValueError(f"Tracked link {link_name!r} missing from {self.ref_pose_file} ({names})")
            rows.append(names.index(link_name))

        def as_t(key: str) -> torch.Tensor:
            return torch.as_tensor(data[key][rows], dtype=torch.float32, device=device)

        self._push_init_pos_b = as_t("push_pos")
        self._push_init_quat_b = as_t("push_quat")
        self._steer_init_pos_b = as_t("steer_pos")
        self._steer_init_quat_b = as_t("steer_quat")

        self._phase_ratios = torch.tensor(self.phase_ratios, dtype=torch.float32, device=device)
        self._marker_offsets = torch.tensor(MARKER_OFFSETS_DECK, dtype=torch.float32, device=device)
        self._foot_site_offsets = torch.tensor(FOOT_SITE_OFFSETS, dtype=torch.float32, device=device)
        self._half_cycle_steps = int(self.cycle_time_s / 2.0 / env.dt)

        n_tracked = len(TRACKED_LINK_NAMES)

        def zeros(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
            return torch.zeros(*shape, dtype=dtype, device=device)

        # Command state.
        self.command_b = zeros(num_envs, 2)
        self.heading_ref = zeros(num_envs)
        self.is_heading_env = zeros(num_envs, dtype=torch.bool)
        self.is_standing_env = zeros(num_envs, dtype=torch.bool)
        # Phase state.
        self.phase_length_buf = zeros(num_envs, dtype=torch.long)
        self.phase = zeros(num_envs)
        self.still = zeros(num_envs, dtype=torch.bool)
        self.contact_phase = zeros(num_envs, 4)
        self.just_exited_push2steer = zeros(num_envs, dtype=torch.bool)
        # Latched contact state (double-buffered like the source sensors).
        self._last_ground_contact = zeros(num_envs, 2, dtype=torch.bool)
        self._last_wheel_contact = zeros(num_envs, 4, dtype=torch.bool)
        self._last_feet_contact_b = zeros(num_envs, 2, dtype=torch.bool)
        self._last_feet_contact_g = zeros(num_envs, 2, dtype=torch.bool)
        self.ground_contact_filt = zeros(num_envs, 2, dtype=torch.bool)
        self.wheel_contact_filt = zeros(num_envs, 4, dtype=torch.bool)
        self.feet_contact_b = zeros(num_envs, 2, dtype=torch.bool)
        self.feet_contact_g = zeros(num_envs, 2, dtype=torch.bool)
        self.left_foot_air_time = zeros(num_envs)
        # Contact-force caches (world frame, on the foot).
        self.foot_ground_force = zeros(num_envs, 2, 3)
        self.foot_board_force = zeros(num_envs, 2, 3)
        self.self_collision = zeros(num_envs, dtype=torch.bool)
        self.illegal_contact = zeros(num_envs, dtype=torch.bool)
        # Transition state.
        self._blend_start_pos_b = zeros(num_envs, n_tracked, 3)
        self._blend_start_quat_b = zeros(num_envs, n_tracked, 4)
        self.transition_target_pos_b = zeros(num_envs, n_tracked, 3)
        self.transition_target_quat_b = zeros(num_envs, n_tracked, 4)
        self.transition_target_quat_b[..., 0] = 1.0
        self.tracked_pos_b = self.transition_target_pos_b.clone()
        self.tracked_quat_b = self.transition_target_quat_b.clone()
        self.in_transition = zeros(num_envs, dtype=torch.bool)
        # Misc caches for reward terms.
        self.feet_marker_dis = zeros(num_envs, 2, 3)
        self.left_foot_site_z = zeros(num_envs, 4)

        self.stats["error_vel"] = zeros(num_envs)
        self.stats["error_yaw"] = zeros(num_envs)
        self._last_refresh_step = -1

        illegal = [
            "pelvis",
            "left_knee_link",
            "right_knee_link",
            "left_shoulder_yaw_link",
            "right_shoulder_yaw_link",
            "left_elbow_link",
            "right_elbow_link",
            "left_wrist_pitch_link",
            "right_wrist_pitch_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ]
        self._illegal_gidx = resolve_link_idx(robot, illegal, device) + robot._entity.link_start
        self._robot_link_range = (robot._entity.link_start, robot._entity.link_end)

        from eden.entities.terrain import Terrain

        self._ground_is_flat = not isinstance(ground, Terrain)

    def reset(self, envs_idx: torch.Tensor | None = None) -> dict[str, float]:
        """Clear phase/contact/transition state for the given envs, then resample."""
        idx = slice(None) if envs_idx is None else envs_idx
        self.phase_length_buf[idx] = 0
        self.phase[idx] = 0.0
        self.contact_phase[idx] = 0.0
        self.just_exited_push2steer[idx] = False
        self._last_ground_contact[idx] = False
        self._last_wheel_contact[idx] = False
        self._last_feet_contact_b[idx] = False
        self._last_feet_contact_g[idx] = False
        self.left_foot_air_time[idx] = 0.0
        self._blend_start_pos_b[idx] = 0.0
        self._blend_start_quat_b[idx] = 0.0
        # The derived caches would otherwise hold the pre-reset robot's state until
        # the next step's refresh (resets run after it), so the freshly-reset envs'
        # observations would mix the dead episode's contact forces and blend targets
        # with post-reset state. Contact forces are honestly unknown until the next
        # physics step — zero them; the kinematic caches are recomputed in
        # :meth:`post_reset`, once every manager has finished resetting.
        self.foot_ground_force[idx] = 0.0
        self.foot_board_force[idx] = 0.0
        self.ground_contact_filt[idx] = False
        self.wheel_contact_filt[idx] = False
        self.feet_contact_b[idx] = False
        self.feet_contact_g[idx] = False
        self.self_collision[idx] = False
        self.illegal_contact[idx] = False
        return super().reset(envs_idx)  # resamples the command from the post-reset board pose

    def post_reset(self, envs_idx: slice | torch.Tensor) -> None:
        """Rebuild the pose-derived caches from the final post-reset scene state."""
        self._refresh_kinematic_caches(envs_idx)

    # ------------------------------------------------------------------
    # Command lifecycle
    # ------------------------------------------------------------------

    def _resample_command(self, envs_idx: torch.Tensor) -> None:
        # Full-batch draws written through the mask with `torch.where`, in the order they were drawn: the
        # mask as an index would be a GPU->CPU sync per gather and per write. Scalar fills keep it.
        r = torch.empty(self._env.num_envs, device=self.device)
        self.command_b[:, 0] = torch.where(envs_idx, r.uniform_(*self.lin_vel_x_range), self.command_b[:, 0])
        assign_over_envs(self.is_heading_env, envs_idx, r.uniform_(0.0, 1.0) <= self.rel_heading_envs)
        self._resample_heading(envs_idx & self.is_heading_env)
        # Non-heading envs hold heading 0 relative to their fresh board pose — without
        # this they would keep a target anchored to an arbitrary earlier episode.
        non_heading = envs_idx & ~self.is_heading_env
        self.command_b[non_heading, 1] = 0.0
        assign_over_envs(self.heading_ref, non_heading, self.board_heading_w())
        assign_over_envs(self.is_standing_env, envs_idx, r.uniform_(0.0, 1.0) <= self.rel_standing_envs)

    def _resample_heading(self, mask: torch.Tensor) -> None:
        """Sample a new relative heading, re-anchored at the board's current heading."""
        r = torch.empty(self._env.num_envs, device=self.device).uniform_(*self.heading_range)
        self.command_b[:, 1] = torch.where(mask, r, self.command_b[:, 1])
        assign_over_envs(self.heading_ref, mask, self.board_heading_w())

    def _update_command(self) -> None:
        # Re-anchor the heading target once per gait cycle, when the push->steer
        # transition completes (the board is about to be steered).
        self._resample_heading(self.is_heading_env & self.just_exited_push2steer)
        self.command_b[self.is_standing_env] = 0.0
        self._update_stats()

    def _update_stats(self) -> None:
        board: RigidEntity = self._env.entities[self.skateboard_name]
        max_command_steps = self.resampling_time_range[1] / self._env.dt
        board_quat = board.get_quat(relative=True)
        vel_b = inv_transform_by_quat(board.get_vel(), board_quat)
        self.stats["error_vel"] += (
            torch.abs(self.command_b[:, 0] - vel_b[:, 0]) / max_command_steps * self.contact_phase[:, 0]
        )
        self.stats["error_yaw"] += (
            torch.abs(wrap_to_pi(self.target_heading_w - self.board_heading_w()))
            / max_command_steps
            * self.contact_phase[:, 1]
        )

    # ------------------------------------------------------------------
    # Per-step refresh (phase, contacts, transition targets)
    # ------------------------------------------------------------------

    def post_physics_refresh(self) -> None:
        """Advance the phase clock and refresh contact/transition state (once per step)."""
        env = self._env
        if env.common_step_counter == self._last_refresh_step:
            return
        self._last_refresh_step = env.common_step_counter

        # Advance the phase clock (this refresh runs only inside step(), so like the
        # source, a freshly-reset env reads phase = 1 tick on its first step).
        self.still.copy_(self.command_b[:, 0] < self.still_vx_threshold)
        self.phase.copy_(
            advance_phase_clock(self.phase_length_buf, self.still, self._half_cycle_steps, env.dt / self.cycle_time_s)
        )

        self._refresh_contacts()
        self._refresh_kinematic_caches()
        self._refresh_contact_phase()
        self._refresh_transition_targets()

    def _refresh_contacts(self) -> None:
        env = self._env
        robot: RigidEntity = env.entities[self.entity_name]
        board: RigidEntity = env.entities[self.skateboard_name]

        contacts = robot.get_contacts()
        valid = contacts.get("valid_mask")
        if valid is None:  # non-parallel scene (n_envs == 0) never happens in RL envs
            valid = torch.ones_like(contacts["link_a"], dtype=torch.bool)
        link_a, link_b = contacts["link_a"], contacts["link_b"]
        geom_a, geom_b = contacts["geom_a"], contacts["geom_b"]
        force_a, force_b = contacts["force_a"], contacts["force_b"]

        def in_board(t: torch.Tensor) -> torch.Tensor:
            return (t >= self._board_link_range[0]) & (t < self._board_link_range[1])

        def in_ground(t: torch.Tensor) -> torch.Tensor:
            return (t >= self._ground_link_range[0]) & (t < self._ground_link_range[1])

        def in_robot(t: torch.Tensor) -> torch.Tensor:
            return (t >= self._robot_link_range[0]) & (t < self._robot_link_range[1])

        for foot, foot_gidx in enumerate(self._feet_gidx):
            foot_a = valid & (link_a == foot_gidx)
            foot_b = valid & (link_b == foot_gidx)
            # Foot <-> ground: net force on the foot.
            ga = foot_a & in_ground(link_b)
            gb = foot_b & in_ground(link_a)
            self.foot_ground_force[:, foot] = (force_a * ga.unsqueeze(-1)).sum(dim=1) + (
                force_b * gb.unsqueeze(-1)
            ).sum(dim=1)
            # Foot <-> board, restricted to the geom the source sensors match:
            # left foot on the rear marker pad, right foot on the deck itself.
            target_geom = self._marker_geom_idx if foot == 0 else self._deck_geom_idx
            ba = foot_a & in_board(link_b) & (geom_b == target_geom)
            bb = foot_b & in_board(link_a) & (geom_a == target_geom)
            self.foot_board_force[:, foot] = (force_a * ba.unsqueeze(-1)).sum(dim=1) + (force_b * bb.unsqueeze(-1)).sum(
                dim=1
            )

        # Self collisions and illegal contacts (any contact involving the listed links).
        both_robot = valid & in_robot(link_a) & in_robot(link_b)
        self.self_collision.copy_(both_robot.any(dim=1))
        illegal_a = torch.isin(link_a, self._illegal_gidx)
        illegal_b = torch.isin(link_b, self._illegal_gidx)
        self.illegal_contact.copy_((valid & (illegal_a | illegal_b)).any(dim=1))

        # Wheels <-> ground.
        board_contacts = board.get_contacts(with_entity=env.entities[self.ground_name])
        bvalid = board_contacts.get("valid_mask")
        if bvalid is None:
            bvalid = torch.ones_like(board_contacts["link_a"], dtype=torch.bool)
        wheel_force = torch.zeros(env.num_envs, 4, device=env.device)
        for wheel, wheel_gidx in enumerate(self._wheel_gidx):
            hit = bvalid & ((board_contacts["link_a"] == wheel_gidx) | (board_contacts["link_b"] == wheel_gidx))
            forces = torch.where(
                (board_contacts["link_a"] == wheel_gidx).unsqueeze(-1),
                board_contacts["force_a"],
                board_contacts["force_b"],
            )
            wheel_force[:, wheel] = (forces * hit.unsqueeze(-1)).sum(dim=1).norm(dim=-1)

        # Latched (double-buffered) contact states, matching the source thresholds.
        # All cache writes are in place: build-time buffers stay normal tensors, so
        # they can be mutated both inside a training rollout's inference_mode region
        # and outside one (reset paths, `eden smoke`).
        ground_contact = self.foot_ground_force.norm(dim=-1) > self.ground_contact_threshold
        self.ground_contact_filt.copy_(ground_contact | self._last_ground_contact)
        self._last_ground_contact.copy_(ground_contact)

        wheel_contact = wheel_force > self.wheel_contact_threshold
        if self._ground_is_flat:
            # The source's fallback assumes wheels rest at axle height on a z=0 plane;
            # it is meaningless on a heightfield, so it only applies on flat ground.
            wheel_z = board.get_links_pos(ls_idx_local=self._wheel_idx)[:, :, 2]
            wheel_contact = wheel_contact | ((wheel_z - 0.03).abs() < 0.005)
        self.wheel_contact_filt.copy_(wheel_contact | self._last_wheel_contact)
        self._last_wheel_contact.copy_(wheel_contact)

        feet_b = self.foot_board_force.norm(dim=-1) > self.feet_contact_threshold
        self.feet_contact_b.copy_(feet_b | self._last_feet_contact_b)
        self._last_feet_contact_b.copy_(feet_b)

        feet_g = self.foot_ground_force.norm(dim=-1) > self.feet_contact_threshold
        self.feet_contact_g.copy_(feet_g | self._last_feet_contact_g)
        self._last_feet_contact_g.copy_(feet_g)

        # Left-foot air time (for the push-phase step reward).
        left_contact = self.ground_contact_filt[:, 0]
        self.left_foot_air_time.add_(env.dt)
        self.left_foot_air_time.mul_((~left_contact).float())

    def _refresh_kinematic_caches(self, idx: slice | torch.Tensor = slice(None)) -> None:
        """Refresh the pose-derived caches (tracked poses, marker distances, probe heights).

        Split from the contact refresh so :meth:`post_reset` can rebuild the reset envs'
        rows from post-reset state — observations computed right after a reset must
        not read the previous episode's poses.
        """
        env = self._env
        robot: RigidEntity = env.entities[self.entity_name]
        board: RigidEntity = env.entities[self.skateboard_name]

        pos_b, quat_b = self._tracked_poses_in_board_frame()
        # `assign_over_envs`: `post_reset` hands the auto-reset's bool mask, and a gather + write through it
        # is two GPU->CPU syncs; the per-step refresh passes `slice(None)`.
        assign_over_envs(self.tracked_pos_b, idx, pos_b)
        assign_over_envs(self.tracked_quat_b, idx, quat_b)
        # Out-of-transition targets are the current poses; reset envs are at phase 0
        # (never in transition), and the per-step refresh overwrites these next step.
        assign_over_envs(self.transition_target_pos_b, idx, pos_b)
        assign_over_envs(self.transition_target_quat_b, idx, quat_b)
        self.in_transition[idx] = False

        # Reward caches: feet -> deck-marker distances, left-foot probe-site heights.
        feet_pos = robot.get_links_pos(ls_idx_local=self._feet_idx)
        deck_pos = board.get_links_pos(ls_idx_local=self._deck_idx)
        deck_quat = board.get_links_quat(ls_idx_local=self._deck_idx)
        marker_w = deck_pos + transform_by_quat(
            self._marker_offsets.unsqueeze(0).expand(env.num_envs, -1, -1), deck_quat.expand(-1, 2, -1)
        )
        assign_over_envs(self.feet_marker_dis, idx, marker_w - feet_pos)
        left_foot_pos = feet_pos[:, 0:1]
        left_foot_quat = robot.get_links_quat(ls_idx_local=self._feet_idx)[:, 0:1]
        sites_w = left_foot_pos + transform_by_quat(
            self._foot_site_offsets.unsqueeze(0).expand(env.num_envs, -1, -1), left_foot_quat.expand(-1, 4, -1)
        )
        assign_over_envs(self.left_foot_site_z, idx, sites_w[:, :, 2])

    def _refresh_contact_phase(self) -> None:
        last = self.contact_phase.clone()
        phase, ratios, active = self.phase, self._phase_ratios, ~self.still
        push = (phase >= ratios[0]) & (phase < ratios[1]) & active
        push2steer = (phase >= ratios[1]) & (phase < ratios[2]) & active
        steer = (phase >= ratios[2]) & (phase < ratios[3]) & active
        steer2push = (phase >= ratios[3]) & (phase <= ratios[4]) & active
        self.contact_phase[:, 0] = push.float()
        self.contact_phase[:, 1] = steer.float()
        self.contact_phase[:, 2] = push2steer.float()
        self.contact_phase[:, 3] = steer2push.float()

        just_entered_push2steer = push2steer & (last[:, 2] < 0.5)
        just_entered_steer2push = steer2push & (last[:, 3] < 0.5)
        self.just_exited_push2steer.copy_((last[:, 2] > 0.5) & ~push2steer)

        # Snapshot current body poses (board frame) as the blend start. A masked
        # `where` instead of boolean indexing: entries are rare, and `.any()` /
        # advanced indexing would sync the GPU on every training step.
        entered = just_entered_push2steer | just_entered_steer2push
        gate = entered.view(-1, 1, 1)
        self._blend_start_pos_b.copy_(torch.where(gate, self.tracked_pos_b, self._blend_start_pos_b))
        self._blend_start_quat_b.copy_(torch.where(gate, self.tracked_quat_b, self._blend_start_quat_b))

    def _refresh_transition_targets(self) -> None:
        phase, ratios, active = self.phase, self._phase_ratios, ~self.still
        push2steer = (phase > ratios[1]) & (phase < ratios[2]) & active
        steer2push = (phase > ratios[3]) & (phase < ratios[4]) & active
        in_transition = push2steer | steer2push
        self.in_transition.copy_(in_transition)

        # Blend for the whole batch and select per env — branch-free (see above).
        num_envs = phase.shape[0]
        t = torch.where(
            push2steer,
            (phase - ratios[1]) / self.push2steer_blend_span,
            (phase - ratios[3]) / self.steer2push_blend_span,
        ).clamp(0.0, 1.0)
        p2s = push2steer.view(-1, 1, 1)
        end_pos = torch.where(
            p2s,
            self._steer_init_pos_b.unsqueeze(0).expand(num_envs, -1, -1),
            self._push_init_pos_b.unsqueeze(0).expand(num_envs, -1, -1),
        )
        end_quat = torch.where(
            p2s,
            self._steer_init_quat_b.unsqueeze(0).expand(num_envs, -1, -1),
            self._push_init_quat_b.unsqueeze(0).expand(num_envs, -1, -1),
        )
        blend_pos = _bezier(self._blend_start_pos_b, end_pos, t, self.bezier_height_offset)
        blend_quat = _slerp(self._blend_start_quat_b, end_quat, t)
        gate = in_transition.view(-1, 1, 1)
        self.transition_target_pos_b.copy_(torch.where(gate, blend_pos, self.tracked_pos_b))
        self.transition_target_quat_b.copy_(torch.where(gate, blend_quat, self.tracked_quat_b))

    # ------------------------------------------------------------------
    # Frame helpers
    # ------------------------------------------------------------------

    def board_heading_w(self) -> torch.Tensor:
        """World heading of the skateboard deck."""
        board: RigidEntity = self._env.entities[self.skateboard_name]
        return heading_from_quat(board.get_quat(relative=True))

    def steer_remaining_time_s(self) -> torch.Tensor:
        """Time until the steer window ends (phase ``ratios[3]``), in seconds."""
        steer_end = self._phase_ratios[3]
        remaining = torch.where(self.phase < steer_end, steer_end - self.phase, 1.0 - self.phase + steer_end)
        return remaining * self.cycle_time_s

    def _tracked_poses_in_board_frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Poses of the tracked links in the skateboard root frame (MJCF body frames)."""
        env = self._env
        robot: RigidEntity = env.entities[self.entity_name]
        board: RigidEntity = env.entities[self.skateboard_name]
        pos_w = robot.get_links_pos(ls_idx_local=self._tracked_idx).clone()
        quat_w = robot.get_links_quat(ls_idx_local=self._tracked_idx).clone()
        # The pelvis is the base link: Genesis re-origins it onto the inertial
        # frame, so read the MJCF pelvis frame through the root-pose getters.
        pos_w[:, 0] = robot.get_pos(relative=True)
        quat_w[:, 0] = robot.get_quat(relative=True)

        board_pos = board.get_pos(relative=True)
        board_quat = board.get_quat(relative=True)
        n, k = pos_w.shape[0], pos_w.shape[1]
        pos_b, quat_b = subtract_frame_transforms(
            board_pos.repeat_interleave(k, dim=0),
            board_quat.repeat_interleave(k, dim=0),
            pos_w.reshape(n * k, 3),
            quat_w.reshape(n * k, 4),
        )
        return pos_b.reshape(n, k, 3), quat_b.reshape(n, k, 4)

    def board_pose_in_robot_frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Skateboard root pose expressed in the robot pelvis frame."""
        env = self._env
        robot: RigidEntity = env.entities[self.entity_name]
        board: RigidEntity = env.entities[self.skateboard_name]
        return subtract_frame_transforms(
            robot.get_pos(relative=True),
            robot.get_quat(relative=True),
            board.get_pos(relative=True),
            board.get_quat(relative=True),
        )


@COMMAND_TERM_REGISTRY.register()
class ConstantSkateCommand(CommandTerm):
    """``[lin_vel_x, relative_heading]`` held at a fixed value, plus the phase clock: the frozen HUSKY actor's command.

    :class:`SkateCommand` without its training machinery, the way
    :class:`~eden.managers.terms.commands.velocity.ConstantVelocityCommand` is a resampled velocity
    command without its random generator. The command is :attr:`command_value`, or whatever
    :meth:`set_command` was last given, and never changes on its own: no resampling and no per-cycle
    heading re-anchor (the simplification the source repo's own MuJoCo deployment makes), so the
    heading is relative to the zero world heading. Of :class:`SkateCommand`'s state only the phase
    clock is kept, because the actor observes it; the contact, board and transition state exist for
    the training rewards and critic.

    The clock ticks in :meth:`pre_physics_advance` rather than :class:`SkateCommand`'s
    :meth:`post_physics_refresh`, because a deployment never runs the latter but drives the former
    (through :meth:`advance_reference`), and :meth:`rewind_reference` restarts it at a deployment's
    reset, its START and every policy switch. Ticking ahead of the step's auto-reset also means a
    reset env reads phase 0 on its first observation, the same as after ``env.reset()``.
    """

    command_value: tuple[float, float] = (0.0, 0.0)
    cycle_time_s: float = 6.0
    still_vx_threshold: float = 0.1
    # Nothing to resample, but ``CommandTerm._resample`` draws from this range, so it cannot be None.
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    def build(self) -> None:
        """Allocate the command and clock, and install :attr:`command_value`."""
        env = self._env
        self._half_cycle_steps = int(self.cycle_time_s / 2.0 / env.dt)
        if self._half_cycle_steps < 1:
            raise ValueError(
                f"{type(self).__name__}: cycle_time_s={self.cycle_time_s} is shorter than two control steps "
                f"(env.dt={env.dt}), so the phase clock has no half cycle to pin at. Pass cycle_time_s >= "
                f"{2.0 * env.dt}."
            )
        self.command_b = torch.zeros(env.num_envs, 2, device=self.device)
        self.phase_length_buf = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        self.phase = torch.zeros(env.num_envs, device=self.device)
        self.set_command(self.command_value)

    @property
    def command(self) -> torch.Tensor:
        """The skate command ``[lin_vel_x, relative_heading]``, shape (num_envs, 2)."""
        return self.command_b

    def set_command(self, command) -> None:
        """Hold ``[lin_vel_x, relative_heading]`` from now on, given as ``(2,)`` or ``(num_envs, 2)``.

        ``lin_vel_x`` is the target board speed in m/s (trained on 0-1.5); below
        :attr:`still_vx_threshold` the phase clock pins at 0 and the robot balances in place.
        ``relative_heading`` (rad, trained on ±pi/4) steers relative to the zero world heading.
        """
        value = torch.as_tensor(command, dtype=torch.float32, device=self.device)
        if value.shape not in ((2,), self.command_b.shape):
            raise ValueError(
                f"A skate command is [lin_vel_x, relative_heading], shaped (2,) or {tuple(self.command_b.shape)}; "
                f"got shape {tuple(value.shape)}: {command}."
            )
        self.command_b[:] = value

    def reset(self, envs_idx: torch.Tensor | None = None) -> dict[str, float]:
        """Restart the phase clock for the given envs; the held command is kept."""
        idx = slice(None) if envs_idx is None else envs_idx
        self.phase_length_buf[idx] = 0
        self.phase[idx] = 0.0
        return super().reset(envs_idx)

    def pre_physics_advance(self) -> None:
        """Tick the phase clock, once per control step."""
        still = self.command_b[:, 0] < self.still_vx_threshold
        phase = advance_phase_clock(
            self.phase_length_buf, still, self._half_cycle_steps, self._env.dt / self.cycle_time_s
        )
        self.phase.copy_(phase)

    def rewind_reference(self) -> None:
        """Restart the phase clock for every env, as a reset does."""
        self.phase_length_buf.zero_()
        self.phase.zero_()

    def _resample_command(self, envs_idx: torch.Tensor) -> None:
        """No-op: the command is held, not sampled."""

    def _update_command(self) -> None:
        """No-op: the held command is written by :meth:`set_command`."""


def _bezier(start: torch.Tensor, end: torch.Tensor, t: torch.Tensor, offset: float) -> torch.Tensor:
    """Quadratic bezier between two link-pose sets, arcing ``offset`` m upward."""
    middle = (start + end) / 2.0
    middle = middle.clone()
    middle[..., 2] += offset
    t = t.view(-1, 1, 1)
    return (1 - t) ** 2 * start + 2 * (1 - t) * t * middle + t**2 * end


def _slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Shortest-path slerp between (N, L, 4) quaternion sets, blend t per row."""
    t = t.view(-1, 1, 1)
    d = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(d < 0, -q1, q1)
    # The cap bounds angle away from 0 (>= ~1.4e-3 rad), so sin(angle) is safely
    # nonzero; for (nearly) aligned quaternions this degrades into a lerp, which
    # the final normalization makes exact enough.
    angle = torch.acos(d.abs().clamp(max=1.0 - 1e-6))
    sin_angle = torch.sin(angle)
    out = (torch.sin((1.0 - t) * angle) * q0 + torch.sin(t * angle) * q1) / sin_angle
    return out / out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
