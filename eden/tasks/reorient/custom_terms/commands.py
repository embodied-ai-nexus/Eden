"""SO(3) goal command for in-hand cube reorientation.

Port of ``wuji-mjlab``'s ``InHandReorientCommand``. The goal is a full orientation drawn
uniformly on SO(3), and the term is a two-phase state machine rather than a timer:

* **APPROACHING** -- the goal is fixed. Once the cube's orientation error stays inside
  ``success_threshold`` for ``success_hold_steps`` consecutive steps, the goal counts as
  reached and the term enters the success window.
* **SUCCESS_WINDOW** -- the goal is still the old one, but a timer runs. Every step spent
  inside the threshold pays an escalating hold reward (see ``rewards.hold_escalation``).
  After ``goal_switch_delay`` steps a fresh goal is drawn *without* moving the cube, so one
  episode becomes a stream of reorientations.

Splitting "reached" from "switched" is what makes the dense hold reward well-defined: the
policy is paid for *staying* on target, not for touching it once.

The machine is split across the two epochs of the step loop, per ``docs/guides/terms.md``
("Which hook: grade the step, or prepare the next one"): it is *measured* in
:meth:`InHandReorientCommand.post_physics_refresh` at phase 5 and the goal is *switched* in
:meth:`InHandReorientCommand._update_command` at phase 17. The reward at phase 9 therefore
reads the counters live, in the step that earned them, and the switch clears them only after
they have been paid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from genesis.utils.geom import transform_by_quat

from eden.managers import COMMAND_TERM_REGISTRY, CommandTerm
from eden.tasks.reorient.constants import (
    PALM_LINK,
    REORIENT_GOAL_IN_PALM_POS,
    TAG_IN_PALM_POS,
    TAG_IN_PALM_QUAT,
)
from eden.utils.geom import quat_error_magnitude, quat_mul, random_quat_uniform

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.commands import CommandTermOptions


@COMMAND_TERM_REGISTRY.register()
class InHandReorientCommand(CommandTerm):
    """Uniform-SO(3) reorientation goal with a hold-then-advance state machine.

    Parameters
    ----------
    hand_entity_name:
        Scene entity name of the (fixed-base) hand.
    object_entity_name:
        Scene entity name of the free object being reoriented.
    palm_link:
        Link pattern selecting the palm on ``hand_entity_name``; the tag frame the goal is
        drawn in is rigidly attached to it.
    success_threshold:
        Orientation error [rad] that counts as on-target.
    success_hold_steps:
        Consecutive on-target steps needed to reach a goal.
    goal_switch_delay:
        Steps spent in the success window before a fresh goal is drawn.
    min_goal_interval:
        Minimum time [s] between goal switches, on top of ``goal_switch_delay``.
    goal_marker_entity_name:
        Scene entity to pose at the goal orientation for rendering, or ``""`` for none.
        Only moved when ``debug_vis`` is set and the env is not training -- see
        :meth:`draw_vis`.
    goal_marker_in_palm:
        Where that marker floats, in the palm frame.
    """

    hand_entity_name: str = "robot"
    object_entity_name: str = "cube"
    palm_link: str = PALM_LINK
    success_threshold: float = 0.2
    success_hold_steps: int = 5
    goal_switch_delay: int = 20
    min_goal_interval: float = 0.0
    goal_marker_entity_name: str = ""
    goal_marker_in_palm: tuple[float, float, float] = REORIENT_GOAL_IN_PALM_POS

    def __init__(self, env: EnvBase, options: CommandTermOptions):
        super().__init__(env, options)

        self._palm_idx: torch.Tensor | None = None
        self._goal_marker_in_palm: torch.Tensor | None = None
        self._tag_in_palm_pos = torch.tensor(TAG_IN_PALM_POS, device=self.device)
        self._tag_in_palm_quat = torch.tensor(TAG_IN_PALM_QUAT, device=self.device)

        self.goal_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.goal_quat_w[:, 0] = 1.0

        # State machine. int32 to match upstream; these are step counts, not physical.
        self.hold_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.goal_timer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.goal_reach_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.in_success_window = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.window_timer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # On-target steps banked inside the current success window -- the value
        # ``rewards.hold_escalation`` escalates with. No shadow buffer beside it: it is measured
        # at phase 5 and read by the reward at phase 9 of the same step, well before the
        # goal switch at 17 can clear it. ``wuji-mjlab`` runs the machine in ``_update_command``
        # and carries three ``reward_*`` snapshots to compensate; the snapshots covered this
        # counter but not the gate beside it, so the highest-paying step of every goal cycle paid
        # zero. That bug is the worked example in ``docs/guides/terms.md``.
        self.reward_hold_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # One-shot flags, overwritten every step: the first two by ``post_physics_refresh``,
        # ``goal_switched`` by ``_update_command``.
        self.within_threshold = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.success_achieved = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_switched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # ``goal_reach_count`` as of the last episode end. The success curriculum grades on
        # it, but ``command_manager`` resets before ``curriculum_manager`` in the reset walk
        # (``RLEnvBase._manager_order``), so by the time the curriculum runs the live counter
        # is already zero. Snapshot it in ``reset`` and let the curriculum read the snapshot.
        self.goal_reach_count_at_reset = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # Pre-allocated so ``CommandTerm.reset``'s per-episode averaging and zeroing land on
        # a stable buffer; each is overwritten wholesale every step.
        for stat in ("ori_error", "goal_reach_count", "hold_counter", "in_success_window", "window_timer"):
            self.stats[stat] = torch.zeros(self.num_envs, device=self.device)

    def build(self) -> None:
        # Touch both entities for fail-fast, but cache only the derived link index.
        hand = self._env.entities[self.hand_entity_name]
        self._env.entities[self.object_entity_name]
        names, palm_idx = hand.find_named_links_idx_local(self.palm_link, preserve_order=True)
        if len(palm_idx) != 1:
            raise ValueError(
                f"InHandReorientCommand: palm_link={self.palm_link!r} matched {len(palm_idx)} links "
                f"({list(names)}) on entity '{self.hand_entity_name}'; expected exactly one. "
                f"Available links: {[link.name for link in hand.links]}"
            )
        self._palm_idx = torch.as_tensor(palm_idx, dtype=torch.long, device=self.device)

        # Resolve the marker here so a typo raises at env construction rather than on the
        # first eval step -- ``draw_vis`` never runs during training, so a lazy lookup would
        # survive a whole run and fail only when someone tried to record it.
        if self.goal_marker_entity_name:
            self._env.entities[self.goal_marker_entity_name]
            self._goal_marker_in_palm = torch.tensor(self.goal_marker_in_palm, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.goal_quat_w

    def _tag_quat_w(self) -> torch.Tensor:
        """World orientation of the wrist tag frame, ``(num_envs, 4)``.

        The palm is the hand's fixed base link, so its link getters report the URDF palm
        origin directly -- Genesis' inertial re-origining (which would shift this by the
        3.8 cm palm COM offset) only applies to a *free* base. Do not "fix" this to
        ``relative=True``: for a fixed base both agree, and the solver frame is the frame
        the cube pose is also read in.
        """
        hand = self.resolve_entity(self.hand_entity_name)
        palm_quat_w = hand.get_links_quat(ls_idx_local=self._palm_idx).squeeze(1)
        return quat_mul(palm_quat_w, self._tag_in_palm_quat.expand_as(palm_quat_w))

    def tag_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """World position and orientation of the wrist tag frame, each ``(num_envs, ...)``.

        Shared with the observation terms, which express the cube pose and the goal in this
        frame so the same observation is reproducible on hardware from one tag detection.
        """
        hand = self.resolve_entity(self.hand_entity_name)
        palm_pos_w = hand.get_links_pos(ls_idx_local=self._palm_idx).squeeze(1)
        palm_quat_w = hand.get_links_quat(ls_idx_local=self._palm_idx).squeeze(1)
        tag_pos_w = palm_pos_w + transform_by_quat(self._tag_in_palm_pos.expand_as(palm_pos_w), palm_quat_w)
        tag_quat_w = quat_mul(palm_quat_w, self._tag_in_palm_quat.expand_as(palm_quat_w))
        return tag_pos_w, tag_quat_w

    def draw_vis(self) -> None:
        """Pose the goal marker at ``goal_quat_w``, floating over the palm.

        The marker is a kinematic twin of the cube wearing the same face texture, so the
        rendered rollout shows the target orientation next to the one the policy has
        actually achieved. ``RLEnvBase.step`` calls this only when the env is not training,
        and ``CommandManager`` only for terms with ``debug_vis`` set, so it costs a training
        run nothing.

        The position is composed through the *live* palm pose rather than written as a world
        constant: that is what keeps every env's marker over its own hand under the per-env
        scene offset. Both writes land in the solver frame -- ``set_pos`` already defaults
        there and ``set_quat`` is pinned there explicitly, against its ``relative=True``
        default -- because ``goal_quat_w`` is a world orientation.
        """
        if self._goal_marker_in_palm is None:
            return
        marker = self.resolve_entity(self.goal_marker_entity_name)
        hand = self.resolve_entity(self.hand_entity_name)
        palm_pos_w = hand.get_links_pos(ls_idx_local=self._palm_idx).squeeze(1)
        palm_quat_w = hand.get_links_quat(ls_idx_local=self._palm_idx).squeeze(1)
        marker.set_pos(palm_pos_w + transform_by_quat(self._goal_marker_in_palm.expand_as(palm_pos_w), palm_quat_w))
        marker.set_quat(self.goal_quat_w, relative=False)

    def _draw_goal(self, mask: torch.Tensor) -> None:
        """Redraw ``goal_quat_w`` where ``mask`` is True, leaving the rest untouched.

        Drawn uniformly in TAG frame and composed to world (``tag_quat_w * tag_rel_goal``),
        matching how the deployed policy receives its goal -- directly in tag frame, with no
        wrist transform applied on the robot side.

        A full-batch draw plus ``torch.where`` rather than an indexed write: it costs a few
        thousand cheap RNG draws and avoids the GPU->CPU sync that turning the mask into
        indices would need on the hot path.
        """
        tag_rel_goal = random_quat_uniform(self.num_envs, device=self.device)
        goal_w = quat_mul(self._tag_quat_w(), tag_rel_goal)
        self.goal_quat_w = torch.where(mask.unsqueeze(-1), goal_w, self.goal_quat_w)

    def _resample_command(self, envs_idx: torch.Tensor) -> None:
        self._draw_goal(envs_idx)

        # Scalar writes through the mask are masked fills, so these need no GPU->CPU sync.
        for buf in (
            self.hold_counter,
            self.goal_timer,
            self.goal_reach_count,
            self.window_timer,
            self.reward_hold_counter,
        ):
            buf[envs_idx] = 0
        for flag in (self.in_success_window, self.within_threshold, self.success_achieved, self.goal_switched):
            flag[envs_idx] = False

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> dict[str, float]:
        # Snapshot before ``super().reset()`` resamples (and so zeroes) the live counter.
        self.goal_reach_count_at_reset.copy_(self.goal_reach_count)
        return super().reset(envs_idx)

    def _min_goal_steps(self) -> int:
        return max(int(self.min_goal_interval / self._env.dt), 0)

    def post_physics_refresh(self) -> None:
        """Measure the step that just happened: orientation error and the success machine.

        Phase 5 -- the *grading* epoch. Everything written here describes the post-physics cube,
        and it is read one and four phases later by the metrics at 6 and the reward at 9, in
        the same step. The goal *switch* is publication and stays in :meth:`_update_command` at
        phase 17, below the reward: that is what lets every consumer read these counters live,
        because the step that earns the peak hold counter is graded before the switch clears it.

        Running the machine in ``_update_command`` instead is the bug this split exists to
        remove, and it is the worked example in ``docs/guides/terms.md`` under "Which hook:
        grade the step, or prepare the next one".

        The advance is not idempotent -- it is a counter, not a cache -- so it runs exactly once
        per control step and cannot be duplicated onto the deployment path, where only
        :meth:`compute` fires. A deployed policy therefore holds whatever goal it was last
        given, which is the arrangement upstream ships (its ``deploy/reorient`` drives the goal
        from an external command term). Nothing the *policy* observes is affected: the two
        cube-tracking observations read ``goal_quat_w`` and the live palm pose, both of which
        survive the crossing.
        """
        obj = self.resolve_entity(self.object_entity_name)
        ori_error = quat_error_magnitude(obj.get_quat(), self.goal_quat_w)
        within_threshold = ori_error < self.success_threshold

        # --- APPROACHING: count consecutive on-target steps, reset the streak on a miss ---
        approaching = ~self.in_success_window
        hold_counter = torch.where(
            approaching & within_threshold,
            self.hold_counter + 1,
            torch.where(approaching, torch.zeros_like(self.hold_counter), self.hold_counter),
        )
        just_succeeded = approaching & (hold_counter >= self.success_hold_steps)

        in_success_window = self.in_success_window | just_succeeded
        window_timer = torch.where(just_succeeded, torch.zeros_like(self.window_timer), self.window_timer)
        hold_counter = torch.where(just_succeeded, torch.zeros_like(hold_counter), hold_counter)
        self.goal_reach_count += just_succeeded.to(self.goal_reach_count.dtype)

        # --- SUCCESS_WINDOW: run the timer, and the hold counter only while on target ---
        window_timer = torch.where(in_success_window, window_timer + 1, window_timer)
        reward_hold_counter = torch.where(
            in_success_window & within_threshold,
            self.reward_hold_counter + 1,
            self.reward_hold_counter,
        )
        reward_hold_counter = torch.where(just_succeeded, torch.zeros_like(reward_hold_counter), reward_hold_counter)

        self.within_threshold = within_threshold
        self.success_achieved = just_succeeded
        self.hold_counter = hold_counter
        self.in_success_window = in_success_window
        self.window_timer = window_timer
        self.reward_hold_counter = reward_hold_counter
        self.goal_timer = self.goal_timer + 1

        # Written here rather than at phase 17 so ``CommandTerm.reset``'s per-episode averaging
        # (phase 13) still sees the episode's final step.
        self.stats["ori_error"].copy_(ori_error)
        self.stats["goal_reach_count"].copy_(self.goal_reach_count)
        self.stats["hold_counter"].copy_(self.hold_counter)
        self.stats["in_success_window"].copy_(self.in_success_window)
        self.stats["window_timer"].copy_(self.window_timer)

    def _update_command(self) -> None:
        """Publish the goal: draw a fresh one once the success window has run its course.

        Phase 17 -- the *preparation* epoch, below the reward and after the auto-reset, so the
        goal left behind here is the one phase 19's observations carry and the next action is
        chosen against. A switch is publication, which is why the decision lives here while the
        counters it reads are measured at phase 5.

        Clearing the machine is part of the switch, and doing it *here* is what makes the
        clearing safe: the step that earned the final hold counter was already paid at phase 9.
        """
        should_switch = (
            self.in_success_window
            & (self.window_timer >= self.goal_switch_delay)
            & (self.goal_timer >= self._min_goal_steps())
        )
        self._draw_goal(should_switch)
        self.goal_switched = should_switch

        self.in_success_window = torch.where(
            should_switch, torch.zeros_like(self.in_success_window), self.in_success_window
        )
        zeros = torch.zeros_like(self.window_timer)
        self.window_timer = torch.where(should_switch, zeros, self.window_timer)
        self.hold_counter = torch.where(should_switch, zeros, self.hold_counter)
        self.goal_timer = torch.where(should_switch, zeros, self.goal_timer)
        self.reward_hold_counter = torch.where(should_switch, zeros, self.reward_hold_counter)


__all__ = ["InHandReorientCommand"]
