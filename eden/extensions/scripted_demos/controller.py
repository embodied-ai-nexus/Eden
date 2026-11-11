"""Scripted pick-and-place demonstrator for an arm + two-finger-gripper rig, batched over environments.

Port of OCBench's controller / primitive split (``ocbench/oracles/controllers/controller.py``,
``ocbench/oracles/primitives/primitive.py``): a task subclass decides *what* to move next
(:meth:`ScriptedPickPlaceController.choose_subgoals`); this base turns each decision into a
demo-like pick-and-place plan (:func:`~eden.extensions.scripted_demos.primitives.sample_pick_place`)
and drives the rig along it:

* **Speed.** Every episode draws one demonstrator speed (``segment_dt`` uniform in
  ``speed_range``), which scales every segment of every plan in it.
* **Plan clock and gates.** Each env has its own plan time, advanced by the control step and
  frozen at each gate (the pick / place hovers and the grasp / release poses,
  :data:`~eden.extensions.scripted_demos.primitives.GATE_SLOTS`) until the tool has caught up.
* **Retries.** If the object does not rise with the hand on the way up from the grasp, the pick
  is re-planned from wherever the hand is, up to ``max_pick_retries`` times.
* **Actions.** The planned tool pose becomes the 6-D delta-pose command of the rig's
  ``DifferentialIKController`` and the plan's gripper value the ``BinaryJointController`` command,
  whatever the task names them (:mod:`~eden.extensions.scripted_demos.rig`) — the task's own action
  space, so a recorded action replays as-is.

Object state is read straight off the scene (OCBench's ``privileged/`` observations), so a
demonstration never depends on the noise the policy observations carry.

The control flow is batched torch on the env device. Only the subgoal decision is a per-env
Python loop on a CPU snapshot: it runs once per finished plan, not per step, and keeps the task
logic as readable as OCBench's.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from eden.extensions.scripted_demos.primitives import (
    CONTACT_GATE_SLOTS,
    GATE_SLOTS,
    NUM_SLOTS,
    WOBBLE_WINDOWS,
    PickPlaceParams,
    PickPlaceRequest,
    Slot,
    sample_pick_place,
)
from eden.extensions.scripted_demos.rig import RigDriver
from eden.extensions.scripted_demos.trajectory import KeyframeTrajectory, Wobble
from eden.utils.geom import yaw_from_quat

if TYPE_CHECKING:
    from eden.envs.base import EnvBase

#: The pick is judged once the hand has risen this fraction of the way from the grasp to its hover,
#: and has failed if the object rose less than :data:`PICK_FOLLOW_FRACTION` of what the hand did.
PICK_CHECK_RISE = 0.75
PICK_FOLLOW_FRACTION = 0.5
#: A hand held at a contact gate passes it once it has stopped against contact — moving slower than
#: this [m/s] for :data:`SETTLE_TIME` [s], within :data:`SETTLE_REACH` [m] of the gate. The way a
#: person lowers a cube onto another: until it stops, then lets go — rather than holding out for a
#: pose the object in the fingers keeps the hand from reaching.
SETTLE_SPEED = 0.01
SETTLE_TIME = 0.6
SETTLE_REACH = 0.04


@dataclass
class Subgoal:
    """One pick-and-place decision for one environment.

    Attributes
    ----------
    object_index : int
        Index into :attr:`ScriptedPickPlaceController.object_names`.
    goal_pos : np.ndarray
        ``(3,)`` world position the object's frame should end at.
    goal_yaw : float
        Heading [rad] the object should end at.
    release : bool
        Open the gripper at the goal and retreat (a place) — or keep holding it there (a lift).
    kind : str
        Free-form label (``goal``, ``clear``, ``mistake``, ...), counted in the generation report.
    """

    object_index: int
    goal_pos: np.ndarray
    goal_yaw: float
    release: bool = True
    kind: str = "goal"


@dataclass
class ObjectSnapshot:
    """CPU copy of the task objects' poses for the environments being decided for.

    Attributes
    ----------
    pos : np.ndarray
        ``(B, M, 3)`` object frame positions (``relative=True``: the authored frame).
    yaw : np.ndarray
        ``(B, M)`` object headings [rad].
    """

    pos: np.ndarray
    yaw: np.ndarray


class ScriptedPickPlaceController(ABC):
    """Base of the scripted demonstrators: plans, clocks and acts; a subclass picks the subgoals.

    Parameters
    ----------
    env : EnvBase
        A built environment whose scene holds an arm + two-finger gripper at ``rig`` (a
        :class:`~eden.options.robots.ManipulatorOptions` composite, or a robot carrying its own
        fingers — see :func:`~eden.extensions.scripted_demos.rig.gripper_rig`), moved by one
        ``DifferentialIKController`` and one ``BinaryJointController``
        (:func:`~eden.extensions.scripted_demos.rig.find_rig_actions`).
    object_names : list[str]
        Scene entities the subgoals refer to, by index.
    rig : str
        Scene entry holding the rig.
    seed : int
        Seed of the controller's own random draws (plans and subgoals).
    speed_range : tuple[float, float]
        Range of ``segment_dt`` [s], one draw per episode (OCBench: 0.7–1.5).
    workspace : tuple[tuple[float, float, float], tuple[float, float, float]]
        Low / high corners every tool target is clipped to. The low ``z`` is the lowest the
        *fingertips* may go: the tool point's floor is raised by :attr:`tip_reach`, measured on the
        built rig.
    effector_bounds : tuple[tuple[float, float, float], tuple[float, float, float]]
        Low / high corners the hand retreats to after a place.
    effector_yaw_range : tuple[float, float]
        Heading range [rad] of that retreat pose, about the rig's straight-down pose.
    max_pick_retries : int
        Failed picks re-planned per subgoal before giving up on it.
    max_overrun_s : float
        A plan still running this long [s] past its own duration (its clock held back at gates) is
        abandoned and a new subgoal chosen. Relative to the plan, not absolute: a slow demonstrator's
        plan legitimately runs long, and cutting it off mid-place ends an episode with the object
        still in the hand.
    gate_timeout_s : float
        A hand held at a gate this long [s] without reaching it is re-planned from where it is.
    max_step_translation, max_step_rotation : float
        Largest end-effector move asked for in one control step, per axis [m] and in angle [rad]
        (OCBench's ``ee_action_delta`` clip). Binding where the task's IK scale is generous
        (SIMPLER, Polaris: 0.5 m per action unit); the robosuite scale (0.05) is already tighter.
    params : PickPlaceParams, optional
        Primitive tunables; OCBench's by default.
    """

    def __init__(
        self,
        env: EnvBase,
        object_names: list[str],
        *,
        rig: str = "robot",
        seed: int = 0,
        speed_range: tuple[float, float] = (0.7, 1.5),
        workspace: tuple[tuple[float, float, float], tuple[float, float, float]],
        effector_bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
        effector_yaw_range: tuple[float, float] = (-math.pi / 2, math.pi / 2),
        max_pick_retries: int = 2,
        max_overrun_s: float = 10.0,
        gate_timeout_s: float = 3.0,
        max_step_translation: float = 0.05,
        max_step_rotation: float = 0.3,
        params: PickPlaceParams | None = None,
    ):
        if speed_range[0] <= 0 or speed_range[0] > speed_range[1]:
            raise ValueError(f"speed_range must be 0 < low <= high seconds per segment, got {speed_range}.")
        self.env = env
        self.object_names = list(object_names)
        self.rig_slot = rig
        self.speed_range = speed_range
        self.max_pick_retries = max_pick_retries
        self.max_overrun_s = max_overrun_s
        self.gate_timeout_steps = math.ceil(gate_timeout_s / env.dt)
        self.params = params or PickPlaceParams()
        self.num_envs = env.num_envs
        self.device = env.device
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.np_rng = np.random.default_rng(seed)
        #: Reads the rig and turns tool poses into the task's actions; resolved here, so a bad rig fails at
        #: construction.
        self.driver = RigDriver(
            env,
            rig,
            workspace=workspace,
            max_step_translation=max_step_translation,
            max_step_rotation=max_step_rotation,
        )
        self.rig = self.driver.rig
        self.down_quat = self.driver.down_quat
        self.tip_reach = self.driver.tip_reach
        self._workspace = self.driver.workspace
        self._effector_bounds = (
            torch.tensor(effector_bounds[0], device=self.device, dtype=torch.float32),
            torch.tensor(effector_bounds[1], device=self.device, dtype=torch.float32),
        )
        self._effector_yaw_range = effector_yaw_range
        #: Subgoal kinds started so far, for the generation report.
        self.subgoal_counts: dict[str, int] = {}
        self._build()

    # --- setup ------------------------------------------------------------------------------------

    def _build(self) -> None:
        """Resolve the objects and allocate the plan state; fail here, not on the first step."""
        env, n, device = self.env, self.num_envs, self.device
        missing = [name for name in self.object_names if name not in env.entities]
        if missing:
            raise KeyError(
                f"{type(self).__name__} moves objects {missing} that this scene does not have; "
                f"scene entities: {sorted(env.entities)}."
            )
        self._objects = [env.entities[name] for name in self.object_names]

        self.trajectory = KeyframeTrajectory(n, NUM_SLOTS, device)
        self.wobble = Wobble(n, len(WOBBLE_WINDOWS), device)
        num_gates = len(GATE_SLOTS)
        self._gate_times = torch.zeros(n, num_gates, device=device)
        self._gate_pos = torch.zeros(n, num_gates, 3, device=device)
        self._gate_tolerance = torch.zeros(n, num_gates, device=device)
        self._gate_valid = torch.zeros(n, num_gates, dtype=torch.bool, device=device)
        self._gate_next = torch.zeros(n, dtype=torch.long, device=device)
        self._gate_wait = torch.zeros(n, dtype=torch.long, device=device)
        self._gate_settle = torch.zeros(n, dtype=torch.long, device=device)
        self._contact_gate = torch.tensor([slot in CONTACT_GATE_SLOTS for slot in GATE_SLOTS], device=device)
        self._settle_steps = math.ceil(SETTLE_TIME / env.dt)
        self._prev_tool_pos = torch.zeros(n, 3, device=device)

        self.segment_dt = torch.ones(n, device=device)
        self.plan_time = torch.zeros(n, device=device)
        self.active = torch.zeros(n, dtype=torch.bool, device=device)
        #: ``(N,)`` bool: the task is complete for this env and its episode should end.
        self.done = torch.zeros(n, dtype=torch.bool, device=device)
        #: Control steps since the current plan started, and how many it may take before it is abandoned.
        self._steps = torch.zeros(n, dtype=torch.long, device=device)
        self._fresh = torch.zeros(n, dtype=torch.bool, device=device)
        self._step_budget = torch.zeros(n, dtype=torch.long, device=device)
        self._object_index = torch.zeros(n, dtype=torch.long, device=device)
        self._goal_pos = torch.zeros(n, 3, device=device)
        self._goal_yaw = torch.zeros(n, device=device)
        self._release = torch.ones(n, dtype=torch.bool, device=device)
        self._num_retries = torch.zeros(n, dtype=torch.long, device=device)
        self._pick_checked = torch.zeros(n, dtype=torch.bool, device=device)
        self._pick_check_time = torch.zeros(n, device=device)
        self._pick_source_z = torch.zeros(n, device=device)

    # --- state readers --------------------------------------------------------------------------

    def tool_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Tool-point position ``(N, 3)`` and effector rotation ``(N, 4)`` (identity = pointing straight down)."""
        return self.driver.tool_pose()

    def object_poses(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Positions ``(N, M, 3)`` and headings ``(N, M)`` of :attr:`object_names`, in their authored frames."""
        pos = torch.stack([obj.get_pos(relative=True) for obj in self._objects], dim=1)
        yaw = torch.stack([yaw_from_quat(obj.get_quat(relative=True)) for obj in self._objects], dim=1)
        return pos, yaw

    def yaw_window(self) -> torch.Tensor:
        """``(N, 2)`` effector headings the arm's last joint can still reach (:meth:`RigDriver.yaw_window`)."""
        return self.driver.yaw_window()

    # --- episode lifecycle ------------------------------------------------------------------------

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        """Start a fresh episode for ``envs_idx`` (bool mask or long indices; all when None).

        Call it for every env after the env itself has been reset — including once for all of
        them before the first :meth:`act`.
        """
        if envs_idx is None:
            envs_idx = torch.arange(self.num_envs, device=self.device)
        elif envs_idx.dtype == torch.bool:
            envs_idx = envs_idx.nonzero(as_tuple=False).flatten()
        if len(envs_idx) == 0:
            return
        low, high = self.speed_range
        u = torch.rand(len(envs_idx), generator=self.generator, device=self.device)
        self.segment_dt[envs_idx] = low + (high - low) * u
        self.active[envs_idx] = False
        self.done[envs_idx] = False
        self.reset_task_state(envs_idx.tolist())

    def reset_task_state(self, env_ids: list[int]) -> None:
        """Clear a subclass's per-episode bookkeeping for ``env_ids``. Default: nothing."""

    @abstractmethod
    def choose_subgoals(self, env_ids: list[int], objects: ObjectSnapshot) -> list[Subgoal | None]:
        """Decide what each of ``env_ids`` moves next; ``None`` when its task is complete.

        Called whenever an env has no plan (episode start) or has finished one.
        """

    # --- stepping ---------------------------------------------------------------------------------

    def update_subgoals(self) -> torch.Tensor:
        """Retire finished plans and start the next subgoals; returns :attr:`done`.

        :meth:`act` calls it first, so calling it is optional — and idempotent within a step. Call
        it before :meth:`act` to learn which envs are complete *before* acting, so their episode can
        end on the step that completed it rather than one hold step later.
        """
        finished = self.active & ((self.plan_time >= self.trajectory.duration) | (self._steps >= self._step_budget))
        needs_subgoal = (~self.active | finished) & ~self.done
        if bool(needs_subgoal.any()):
            self._start_subgoals(needs_subgoal.nonzero(as_tuple=False).flatten())
        return self.done

    def act(self) -> dict[str, torch.Tensor]:
        """The action dict for this control step: the IK term's ``(N, 6)`` command and the gripper term's ``(N, 1)``."""
        self.update_subgoals()
        # A plan installed since the last act() issues its first keyframe before its clock moves.
        running = self.active & ~self._fresh
        self.plan_time += self.env.dt * running
        self._steps += running

        tool_pos, _ = self.tool_pose()
        blocked, gate_pos = self._apply_gates(tool_pos)
        self._gate_wait = torch.where(blocked, self._gate_wait + 1, 0)
        stalled = self._gate_wait >= self.gate_timeout_steps
        if bool(stalled.any()):
            # The hand cannot reach the gate pose it is waiting at: start the pick-and-place over from here.
            self._plan(stalled.nonzero(as_tuple=False).flatten())
            blocked = blocked & ~stalled
        self._check_picks()

        pos, rot, grasp = self.trajectory.evaluate(self.plan_time)
        pos = torch.cat([pos[:, :2] + self.wobble.offset(self.plan_time), pos[:, 2:]], dim=-1)
        pos = torch.where(blocked.unsqueeze(-1), gate_pos, pos)
        self._fresh[:] = False
        # Envs whose task is complete hold still until the generator resets them.
        return self.driver.command(pos, rot, grasp, hold=self.done)

    def _apply_gates(self, tool_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Freeze the plan clock at a due gate the tool has not reached; returns ``(blocked, gate_pos)``."""
        num_gates = self._gate_times.shape[1]
        for _ in range(num_gates):  # step past gates this plan does not use
            g = self._gate_next.clamp(max=num_gates - 1).unsqueeze(1)
            unused = (self._gate_next < num_gates) & ~self._gate_valid.gather(1, g).squeeze(1)
            self._gate_next += unused
        has_gate = self._gate_next < num_gates
        g = self._gate_next.clamp(max=num_gates - 1)
        gate_time = self._gate_times.gather(1, g.unsqueeze(1)).squeeze(1)
        gate_pos = self._gate_pos.gather(1, g.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
        due = self.active & has_gate & (self.plan_time >= gate_time)
        tolerance = self._gate_tolerance.gather(1, g.unsqueeze(1)).squeeze(1)
        distance = (tool_pos - gate_pos).norm(dim=-1)
        far = distance > tolerance
        # A contact gate also passes once the hand has settled against contact near it.
        still = (tool_pos - self._prev_tool_pos).norm(dim=-1) < SETTLE_SPEED * self.env.dt
        pressing = due & far & self._contact_gate[g] & still & (distance < SETTLE_REACH)
        self._gate_settle = torch.where(pressing, self._gate_settle + 1, 0)
        far = far & (self._gate_settle < self._settle_steps)
        self._prev_tool_pos = tool_pos.clone()
        blocked = due & far
        self.plan_time = torch.where(blocked, gate_time, self.plan_time)
        self._gate_next += due & ~far
        return blocked, gate_pos

    def _check_picks(self) -> None:
        """Re-plan the pick of every env whose object did not rise with the hand.

        OCBench checks at plan time ``postpick`` that the object has risen 2.5 cm. That presumes the
        hand is where the plan is; a hand trailing it (SIMPLER's WidowX at 5 Hz, by ~4 cm) has not
        lifted that far yet, and good grasps were judged failed and redone. So the check waits until
        the hand itself has risen :data:`PICK_CHECK_RISE` of the way to its hover, then asks whether
        the object came along: a held object rises with the hand, a missed one stays put.
        """
        due = (
            self.active
            & ~self._pick_checked
            & (self.plan_time >= self._pick_check_time)
            & (self._num_retries < self.max_pick_retries)
        )
        if not bool(due.any()):
            return
        tool_pos, _ = self.tool_pose()
        hover_z = self._gate_pos[:, GATE_SLOTS.index(Slot.PICK), 2]
        grasp_z = self._gate_pos[:, GATE_SLOTS.index(Slot.PICK_START), 2]
        tool_rise = tool_pos[:, 2] - grasp_z
        ready = due & (tool_rise >= PICK_CHECK_RISE * (hover_z - grasp_z))
        if not bool(ready.any()):
            return
        self._pick_checked |= ready
        obj_pos, _ = self.object_poses()
        obj_z = obj_pos.gather(1, self._object_index.view(-1, 1, 1).expand(-1, 1, 3))[:, 0, 2]
        failed = ready & (obj_z - self._pick_source_z < PICK_FOLLOW_FRACTION * tool_rise)
        if bool(failed.any()):
            idx = failed.nonzero(as_tuple=False).flatten()
            self._num_retries[idx] += 1
            self._plan(idx)

    def _start_subgoals(self, envs_idx: torch.Tensor) -> None:
        """Ask the task for the next subgoal of ``envs_idx`` and plan it (or mark the env done)."""
        obj_pos, obj_yaw = self.object_poses()
        snapshot = ObjectSnapshot(pos=obj_pos[envs_idx].cpu().numpy(), yaw=obj_yaw[envs_idx].cpu().numpy())
        env_ids = envs_idx.tolist()
        subgoals = self.choose_subgoals(env_ids, snapshot)
        if len(subgoals) != len(env_ids):
            raise RuntimeError(
                f"{type(self).__name__}.choose_subgoals returned {len(subgoals)} subgoals for {len(env_ids)} envs; "
                "it must return one entry (a Subgoal, or None when the task is complete) per env."
            )
        planned, finished = [], []
        for env_id, subgoal in zip(env_ids, subgoals):
            if subgoal is None:
                finished.append(env_id)
                continue
            planned.append(env_id)
            self._object_index[env_id] = subgoal.object_index
            self._goal_pos[env_id] = torch.as_tensor(subgoal.goal_pos, dtype=torch.float32)
            self._goal_yaw[env_id] = float(subgoal.goal_yaw)
            self._release[env_id] = bool(subgoal.release)
            self.subgoal_counts[subgoal.kind] = self.subgoal_counts.get(subgoal.kind, 0) + 1
        if finished:
            idx = torch.tensor(finished, device=self.device)
            self.done[idx] = True
            self.active[idx] = False
        if planned:
            idx = torch.tensor(planned, device=self.device)
            self._num_retries[idx] = 0
            self._plan(idx)

    def _plan(self, envs_idx: torch.Tensor) -> None:
        """Sample and install a pick-and-place plan for ``envs_idx`` toward their current subgoal."""
        batch = len(envs_idx)
        # Every plan starts from the hand's actual orientation. OCBench starts a fresh plan from the
        # wrist's heading alone, which is the same thing for its UR5e (home is straight down); a
        # robosuite Panda's home is tilted ~7 deg, and the heading-only start would open every plan
        # with a step that snaps the wrist upright.
        tool_pos, tool_rot = self.tool_pose()
        tool_pos, tool_rot = tool_pos[envs_idx], tool_rot[envs_idx]
        obj_pos, obj_yaw = self.object_poses()
        index = self._object_index[envs_idx]
        rows = torch.arange(batch, device=self.device)
        obj_pos, obj_yaw = obj_pos[envs_idx][rows, index], obj_yaw[envs_idx][rows, index]
        # Where the hand grips is where it can reach: an object whose frame is below the tool floor is
        # gripped that much higher, and placed that much higher too, so it still lands on its goal (a
        # place aimed at the object frame would leave the hand above its release pose, never passing it).
        grip_lift = torch.zeros_like(obj_pos)
        grip_lift[:, 2] = (self._workspace[0][2] - obj_pos[:, 2]).clamp(min=0.0)

        low, high = self._effector_bounds
        final_pos = low + (high - low) * torch.rand(batch, 3, generator=self.generator, device=self.device)
        yaw_low, yaw_high = self._effector_yaw_range
        final_yaw = yaw_low + (yaw_high - yaw_low) * torch.rand(batch, generator=self.generator, device=self.device)

        plan = sample_pick_place(
            PickPlaceRequest(
                tool_pos=tool_pos,
                tool_rot=tool_rot,
                object_pos=obj_pos + grip_lift,
                object_yaw=obj_yaw,
                goal_pos=self._goal_pos[envs_idx] + grip_lift,
                goal_yaw=self._goal_yaw[envs_idx],
                final_pos=final_pos,
                final_yaw=final_yaw,
                release=self._release[envs_idx],
                segment_dt=self.segment_dt[envs_idx],
                num_retries=self._num_retries[envs_idx],
                yaw_window=self.yaw_window()[envs_idx],
            ),
            generator=self.generator,
            workspace=self._workspace,
            params=self.params,
        )
        self.trajectory.assign(envs_idx, plan.keyframes)
        self.wobble.assign(envs_idx, plan.wobble_windows, plan.wobble_curves)
        self._gate_times[envs_idx] = plan.gate_times
        # A gate waits for the pose the hand is actually sent to — the workspace-clipped one — or a
        # grasp below the fingertip floor could never be reached, or would pass while still above it.
        low, high = self._workspace
        self._gate_pos[envs_idx] = torch.maximum(torch.minimum(plan.gate_pos, high), low)
        self._gate_tolerance[envs_idx] = plan.gate_tolerance
        self._gate_valid[envs_idx] = plan.gate_valid
        self._gate_next[envs_idx] = 0
        self._gate_wait[envs_idx] = 0
        self._gate_settle[envs_idx] = 0
        self._pick_check_time[envs_idx] = plan.pick_check_time
        self._pick_checked[envs_idx] = False
        self._pick_source_z[envs_idx] = obj_pos[:, 2]
        self.plan_time[envs_idx] = 0.0
        self._fresh[envs_idx] = True
        self._steps[envs_idx] = 0
        self._step_budget[envs_idx] = (
            ((self.trajectory.duration[envs_idx] + self.max_overrun_s) / self.env.dt).ceil().long()
        )
        self.active[envs_idx] = True
