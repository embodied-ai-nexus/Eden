"""SONIC teleoperation policy: runtime encoder-mode switching over the shared motion runtime.

Extends :class:`SonicPolicy` (which drives the g1 encoder from a motion clip)
to the full teleop deployment contract: the reference motion comes from the
:class:`~eden.extensions.policies.sonic.motion.MotionScheduler` (planner
output, streamed SMPL windows, or frozen-upper overrides), and the encoder
slots are filled per mode exactly like the deploy binary:

======  ======================  =============================================
mode    encoder                 slots filled (all others zeroed)
======  ======================  =============================================
0 g1    PLANNER / FROZEN /      encoder_mode_4, motion_joint_{positions,
        streamed v1             velocities}_10frame_step5 (frozen-upper
                                override of the 17 upper-body joints),
                                motion_anchor_orientation_10frame_step5
1 teleop VR_3PT                 encoder_mode_4, motion_joint_{positions,
                                velocities}_lowerbody_10frame_step5,
                                vr_3point_local_{target,orn_target},
                                motion_anchor_orientation
2 smpl  POSE                    encoder_mode_4, smpl_joints_10frame_step1,
                                smpl_anchor_orientation_10frame_step1,
                                motion_joint_positions_wrists_10frame_step1
======  ======================  =============================================

Faithful reference behaviors: future frames hold the current frame while
playback is paused (velocities are zeroed instead), all anchor slots diff the
heading-corrected reference root against the full robot base quaternion, and
a mode whose data is unavailable falls back through the other modes, mutating
the motion's ``encode_mode`` — exactly the deploy binary's silent fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from genesis.utils import geom as gu

from eden import actions
from eden.constants import ReferenceSource
from eden.extensions.policies.base import PolicyOptions
from eden.extensions.policies.sonic.joint_maps import (
    IL_TO_MJ,
    LOWER_BODY_IL_INDICES,
    MODE_G1,
    MODE_SMPL,
    MODE_TELEOP,
    SONIC_DOFS_KD,
    SONIC_DOFS_KP,
    SONIC_DOFS_POS,
    SONIC_FORCE_LIMIT_OVERRIDES_MJ,
    SONIC_HAND_FORCE_LIMIT_OVERRIDES,
    SONIC_MJCF_ARMATURE,
)
from eden.extensions.policies.sonic.motion import MotionScheduler, MotionSnapshot
from eden.extensions.policies.sonic.policy import (
    _ENCODER_INPUT_DIM,
    _ENCODER_OFFSETS,
    RobotStateSnapshot,
    SonicPolicy,
)
from eden.options.robots.unitree_g1 import (
    G1_DEX3_LEFT_DOFS_NAME,
    G1_DEX3_RIGHT_DOFS_NAME,
    G1_DOFS29_NAME,
    G1Dofs29Dex3,
)

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.actions import ActionTermOptions
    from eden.options.managers.commands import CommandTermOptions

# The 17 upper-body joints in ascending IsaacLab index order (waist yaw/roll/
# pitch, L/R shoulder pitch/roll/yaw, L/R elbow, L/R wrist roll/pitch/yaw) —
# `upper_body_joint_isaaclab_order_in_isaaclab_index` in the reference.
UPPER_BODY_IL_INDICES: tuple[int, ...] = (2, 5, 8, 11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28)
# Wrist joints, IsaacLab indices: L/R roll, L/R pitch, L/R yaw.
WRIST_IL_INDICES: tuple[int, ...] = (23, 24, 25, 26, 27, 28)

# Defaults served when no VR 3-point data has ever arrived (input_interface.hpp).
VR_3POINT_DEFAULT_POSITION = np.array([0.0903, 0.1615, -0.2411, 0.1280, -0.1522, -0.2461, 0.0241, -0.0081, 0.4028])
VR_3POINT_DEFAULT_ORIENTATION = np.array(
    [0.7295, 0.3145, 0.5533, -0.2506, 0.7320, -0.2639, 0.5395, 0.3217, 0.9991, 0.011, 0.0402, -0.0002]
)

# Per-mode required encoder slots (the released observation_config.yaml
# `encoder_modes` section). Everything else stays zero for that mode.
_MODE_REQUIRED_SLOTS: dict[int, tuple[str, ...]] = {
    MODE_G1: (
        "encoder_mode_4",
        "motion_joint_positions_10frame_step5",
        "motion_joint_velocities_10frame_step5",
        "motion_anchor_orientation_10frame_step5",
    ),
    MODE_TELEOP: (
        "encoder_mode_4",
        "motion_joint_positions_lowerbody_10frame_step5",
        "motion_joint_velocities_lowerbody_10frame_step5",
        "vr_3point_local_target",
        "vr_3point_local_orn_target",
        "motion_anchor_orientation",
    ),
    MODE_SMPL: (
        "encoder_mode_4",
        "smpl_joints_10frame_step1",
        "smpl_anchor_orientation_10frame_step1",
        "motion_joint_positions_wrists_10frame_step1",
    ),
}

# The union of every mode's required slots, order-preserving and de-duplicated.
# An ``encode_mode`` id that is not one of the configured modes (e.g. the
# documented "all-active" id 3) requires *every* encoder observation, exactly
# like the reference's ``use_mode_filter = false`` branch in
# ``GatherEncoderObservations`` (g1_deploy_onnx_ref.cpp): the reference resolves
# required observations by linear search over the configured modes and, on a
# miss, marks every observation required rather than indexing a per-mode table —
# so an unknown id degrades to gathering all slots, never a lookup error.
_ALL_SLOTS: tuple[str, ...] = tuple(dict.fromkeys(name for slots in _MODE_REQUIRED_SLOTS.values() for name in slots))

_LOWER_BODY_IL = np.asarray(LOWER_BODY_IL_INDICES)
_UPPER_BODY_IL = np.asarray(UPPER_BODY_IL_INDICES)
_WRIST_IL = np.asarray(WRIST_IL_INDICES)


class ReferenceUnavailableError(RuntimeError):
    """No encoder mode has usable reference data (raised by the fallback walk).

    The runtime treats this as the reference's "stop control" condition;
    every other exception propagates with its traceback.
    """


def _target_frames(
    snap: MotionSnapshot, num_frames: int, step_size: int, advance_when_paused: bool = False
) -> np.ndarray:
    """Future frame indices ``frame + i*step`` clamped to the motion end.

    While playback is paused the position/SMPL/anchor gathers hold the current
    frame for every index; the velocity gather advances regardless (it writes
    zeros when paused anyway), matching the reference.
    """
    if snap.play or advance_when_paused:
        frames = snap.frame + np.arange(num_frames) * step_size
        return np.minimum(frames, snap.motion.timesteps - 1)
    return np.full(num_frames, snap.frame)


class SonicTeleopPolicy(SonicPolicy):
    """SONIC whole-body teleoperation controller (all three encoder modes).

    Reads its reference from an attached :class:`MotionScheduler` instead of a
    ``MotionCommand`` clip; the ZMQ manager and the locomotion planner feed
    that scheduler. Single-env only — the scheduler and its heading state are
    per-process, like the deploy binary.

    Deliberately not in ``POLICY_REGISTRY``: the registry is what
    ``eden deploy/inference --policy`` resolve, and neither can attach the
    runtime this policy needs — ``eden teleop sonic robot`` constructs it directly.
    (Registering it would also make the registry's contents depend on whether
    something had imported this module.)
    """

    # The locomotion planner graph (target_vel V2), mirrored to
    # Kashu7100/eden_policies next to the encoder/decoder. A local file can
    # still be substituted via `use_local_checkpoint(path,
    # checkpoint_attr="planner_checkpoint_path")` / the CLI's --planner-path.
    planner_checkpoint_path: str = "sonic/planner_sonic.onnx"
    planner_md5_hash: str | None = "011d9e4f9040e39968032dcacdf02d36"

    def __init__(self, env: EnvBase, options: PolicyOptions) -> None:
        super().__init__(env, options)
        if env.num_envs != 1:
            raise ValueError(f"SonicTeleopPolicy is single-env (the motion runtime is per-process); got {env.num_envs}")
        self._scheduler: MotionScheduler | None = None
        # Numpy staging buffer for the encoder input; copied into the torch
        # scratch buffer once per tick.
        self._encoder_np = np.zeros(_ENCODER_INPUT_DIM)
        self._slot_gathers = self._build_slot_gathers()

    def _build_slot_gathers(self) -> dict:
        # Each gather writes the slot named by its dispatch key; the key is
        # threaded in as ``name`` (single source of truth — no rebuilding it
        # from n/step/indexes inside the gather).
        return {
            "motion_joint_positions_10frame_step5": lambda name, s, q: self._gather_joint_positions(
                name, s, 10, 5, None
            ),
            "motion_joint_velocities_10frame_step5": lambda name, s, q: self._gather_joint_velocities(
                name, s, 10, 5, None
            ),
            "motion_anchor_orientation_10frame_step5": lambda name, s, q: self._gather_anchor_orientation(
                name, s, q, 10, 5
            ),
            "motion_joint_positions_lowerbody_10frame_step5": (
                lambda name, s, q: self._gather_joint_positions(name, s, 10, 5, _LOWER_BODY_IL)
            ),
            "motion_joint_velocities_lowerbody_10frame_step5": (
                lambda name, s, q: self._gather_joint_velocities(name, s, 10, 5, _LOWER_BODY_IL)
            ),
            "vr_3point_local_target": lambda name, s, q: self._gather_vr_position(name, s),
            "vr_3point_local_orn_target": lambda name, s, q: self._gather_vr_orientation(name, s),
            "motion_anchor_orientation": lambda name, s, q: self._gather_anchor_orientation(name, s, q, 1, 1),
            "smpl_joints_10frame_step1": lambda name, s, q: self._gather_smpl_joints(name, s, 10, 1),
            "smpl_anchor_orientation_10frame_step1": lambda name, s, q: self._gather_anchor_orientation(
                name, s, q, 10, 1
            ),
            "motion_joint_positions_wrists_10frame_step1": (
                lambda name, s, q: self._gather_joint_positions(name, s, 10, 1, _WRIST_IL)
            ),
        }

    # ------------------------------------------------------------------
    # Env / action / command specs
    # ------------------------------------------------------------------

    @classmethod
    def robot_options(cls) -> G1Dofs29Dex3:
        """The 29-DOF G1 with Dex3 hands, matched to the reference deploy + sim plant.

        Deltas from the stock spec, each pinned to the reference stack:

        - hip-pitch kp/kd use the 7520-22 motor like the deployed checkpoint
          (``ACTION_SCALE_MJ`` already carries that swap);
        - armature is the MJCF's uniform 0.01 (the value the reference MuJoCo
          sim runs) instead of the actuator-spec reflected inertias — set
          explicitly, since an empty map falls back to the spec, not the asset;
        - torque clamps follow the reference sim's effort-limit list where it
          is tighter than the spec (hip roll 88, non-thumb-base fingers 0.7);
        - the default pose is the SONIC standing pose, so the pre-engage hold
          command and every auto-reset start from the commanded stand instead
          of stepping the elbows by 0.9 rad.
        """
        base = G1Dofs29Dex3()
        return G1Dofs29Dex3(
            # SONIC_DOFS_KP/KD key the body exactly as the stock spec does, so they
            # replace the body entries and leave the hand pattern in place.
            default_dofs_kp={**base.default_dofs_kp, **SONIC_DOFS_KP},
            default_dofs_kd={**base.default_dofs_kd, **SONIC_DOFS_KD},
            default_dofs_armature={"*": SONIC_MJCF_ARMATURE},
            default_dofs_force_limits={
                **base.default_dofs_force_limits,
                **SONIC_FORCE_LIMIT_OVERRIDES_MJ,
                **SONIC_HAND_FORCE_LIMIT_OVERRIDES,
            },
            default_dofs_pos={**SONIC_DOFS_POS, "*_hand_*_joint": 0.0},
        )

    @classmethod
    def env_options(cls) -> dict[str, Any]:
        """SONIC's 50 Hz solver settings, plus contact parity with the reference MuJoCo sim.

        - ``enable_multi_contact`` on, so a box foot gets a multi-point manifold
          instead of one migrating point;
        - ``constraint_timeconst=0.02``, MuJoCo's default ``solref`` (Eden's fallback
          of ``2 * substep_dt = 0.01`` makes every contact twice as stiff).

        Teleop-only on purpose: the base :class:`SonicPolicy` keeps the settings
        ``eden deploy`` / ``inference --policy sonic`` were validated under. The
        timeconst is global, which is safe only because this scene holds the robot
        and the ground plane alone (it *softens* contacts, the opposite of the
        mesh-ejecting over-stiffening the entities guide warns about). A scene that
        adds objects should move it to the robot's ``default_geoms_sol_params``.
        """
        return {**super().env_options(), "enable_multi_contact": True, "constraint_timeconst": 0.02}

    @classmethod
    def command_options(cls) -> dict[str, CommandTermOptions]:
        """No command terms: the reference motion comes from the ZMQ/planner runtime.

        Takes no parameters, so ``--motion_file`` / ``--clip_name`` are not forwarded.
        """
        return {}

    @classmethod
    def action_options(cls, entity_name: str = "robot") -> dict[str, ActionTermOptions]:
        # Body targets come from the policy (absolute joint positions, like the
        # base class); hand targets come from the Dex3 controller in the
        # runtime. Both are absolute positions for Genesis's built-in PD.
        return {
            "dofs_pos_controller": actions.ImplicitPDController.configure(
                entity_name=entity_name,
                dofs_name=G1_DOFS29_NAME,
                scale=1.0,
                reference_source=ReferenceSource.ZERO,
            ),
            "hands_pos_controller": actions.ImplicitPDController.configure(
                entity_name=entity_name,
                dofs_name=G1_DEX3_LEFT_DOFS_NAME + G1_DEX3_RIGHT_DOFS_NAME,
                scale=1.0,
                reference_source=ReferenceSource.ZERO,
            ),
        }

    # ------------------------------------------------------------------
    # Runtime wiring
    # ------------------------------------------------------------------

    def attach_runtime(self, scheduler: MotionScheduler) -> None:
        """Attach the shared motion runtime (must happen before ``reset()`` and the first step)."""
        self._scheduler = scheduler

    def _require_runtime(self, caller: str) -> MotionScheduler:
        if self._scheduler is None:
            raise RuntimeError(
                f"SonicTeleopPolicy.{caller}() needs a motion runtime: it reads its reference from the "
                "`eden teleop sonic robot` ZMQ/planner runtime, which calls attach_runtime(scheduler) before "
                "reset(). Run it with `eden teleop sonic robot`; `eden deploy/inference --policy` cannot drive it "
                "(use `--policy sonic` to track a motion clip)."
            )
        return self._scheduler

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None) -> None:
        """Fail at setup, not on the first tick, when nothing will ever feed the reference."""
        self._require_runtime("reset")
        super().reset(envs_idx)

    def record_state(self, robot_state: RobotStateSnapshot) -> None:
        """Roll the state history for a tick the policy does not act on (the pre-engage hold).

        The action history is left alone: no decoder output exists for that tick.
        """
        self._update_state_history(self.env.entities[self.entity_name], robot_state)

    def resolve_planner_path(self) -> str:
        """Resolve (and verify) the planner ONNX through the policy asset machinery."""
        path = self.fetch_asset(self.planner_checkpoint_path)
        self._verify_md5(path, self.planner_md5_hash)
        return path

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def step(
        self,
        obs: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        robot_state: RobotStateSnapshot | None = None,
    ) -> dict[str, torch.Tensor]:
        # Read off the entity, not an observation group: this policy rides along as an extra deploy
        # slot on another policy's or task's env, whose groups are not its own (see the
        # "Declare the observation" section of docs/guides/policy-porting.md).
        del obs  # raw simulator state comes from the env (or the runtime's per-tick snapshot)
        self._require_runtime("step")
        robot = self.env.entities[self.entity_name]

        pelvis_quat_wxyz = self._update_state_history(robot, robot_state)
        base_quat = pelvis_quat_wxyz[0].detach().to("cpu", torch.float64).numpy()

        # Reset the streamed observation-window hold and refresh heading state
        # before any gather runs (reference control-loop order), then take the
        # per-tick frozen view.
        self._scheduler.begin_control_tick(base_quat)
        snap = self._scheduler.snapshot()

        self._gather_encoder_observations(snap, base_quat)
        self._encoder_buf.copy_(torch.from_numpy(self._encoder_np[None, :].astype(np.float32)))

        action_il = self._decode(self._encode())
        self._push_action_history(action_il)

        # ---- isaaclab→mjcf permutation + action transform ----
        action_mj = action_il[..., self._il_to_mj]
        joint_pos_targets = self._default_angles_mj + self._action_scale_mj * action_mj
        return {"dofs_pos_controller": joint_pos_targets}

    # ------------------------------------------------------------------
    # Encoder-slot gathering (per mode, with the reference's fallback walk)
    # ------------------------------------------------------------------

    def _gather_encoder_observations(self, snap: MotionSnapshot, base_quat: np.ndarray) -> int:
        """Fill the encoder staging buffer for the snapshot's mode.

        Tries the intended mode first, then every other mode; success mutates
        the motion's ``encode_mode`` (the reference's silent fallback). An
        intended id outside the configured modes requires every slot (the
        reference's all-active branch), so it never raises a lookup error — it
        simply falls through to a configured mode when its data is missing.
        Raises when no mode's data is available — the runtime turns that into a
        stop.
        """
        intended = snap.encode_mode
        modes_to_try = ([intended] if intended >= 0 else []) + [m for m in _MODE_REQUIRED_SLOTS if m != intended]

        for mode in modes_to_try:
            self._encoder_np[:] = 0.0
            required = _MODE_REQUIRED_SLOTS.get(mode, _ALL_SLOTS)
            if all(self._gather_slot(name, snap, base_quat, mode) for name in required):
                if mode != intended:
                    print(f"[SonicTeleopPolicy] encoder mode {intended} unavailable; fell back to mode {mode}")
                if snap.motion is not None:
                    snap.motion.encode_mode = mode
                return mode
        raise ReferenceUnavailableError("no encoder mode has usable reference data (no motion attached?)")

    def _gather_slot(self, name: str, snap: MotionSnapshot, base_quat: np.ndarray, mode: int) -> bool:
        if name == "encoder_mode_4":
            start, _ = _ENCODER_OFFSETS[name]
            self._encoder_np[start] = float(mode)  # scalar mode id + 3 zeros, NOT one-hot
            return True
        return self._slot_gathers[name](name, snap, base_quat)

    def _write_np_slot(self, name: str, values: np.ndarray) -> None:
        start, end = _ENCODER_OFFSETS[name]
        self._encoder_np[start:end] = values.reshape(-1)

    def _gather_joint_positions(
        self, name: str, snap: MotionSnapshot, n: int, step: int, indexes: np.ndarray | None
    ) -> bool:
        motion = snap.motion
        width = 0 if motion is None else motion.joint_pos.shape[1]
        if motion is None or motion.timesteps == 0 or width == 0:
            return False
        # A subset gather whose indices exceed the stream's joint count fails,
        # triggering the encoder fallback (reference index validation).
        if indexes is not None and int(indexes.max()) >= width:
            return False
        self._scheduler.note_observation_window(n, step)
        values = motion.joint_pos[_target_frames(snap, n, step)]  # (n, width)
        if indexes is None:
            if width != 29:  # narrow streams fill the leading columns (reference memcpy)
                padded = np.zeros((n, 29))
                padded[:, : min(width, 29)] = values[:, :29]
                values = padded
            if snap.frozen_upper_targets is not None:
                values = values.copy()
                values[:, _UPPER_BODY_IL] = snap.frozen_upper_targets
        else:
            values = values[:, indexes]
        self._write_np_slot(name, values)
        return True

    def _gather_joint_velocities(
        self, name: str, snap: MotionSnapshot, n: int, step: int, indexes: np.ndarray | None
    ) -> bool:
        motion = snap.motion
        stream_width = 0 if motion is None else motion.joint_vel.shape[1]
        if motion is None or motion.timesteps == 0 or stream_width == 0:
            return False
        if indexes is not None and int(indexes.max()) >= stream_width:
            return False
        self._scheduler.note_observation_window(n, step)
        slot_width = 29 if indexes is None else len(indexes)
        if not snap.play:
            values = np.zeros((n, slot_width))  # paused playback zeroes reference velocities
        else:
            values = motion.joint_vel[_target_frames(snap, n, step, advance_when_paused=True)]
            if indexes is None:
                if stream_width != 29:
                    padded = np.zeros((n, 29))
                    padded[:, : min(stream_width, 29)] = values[:, :29]
                    values = padded
                if snap.frozen_upper_targets is not None:
                    # Upper-body override applies to velocities too; the operator station
                    # streams positions only, so the override velocities default to zeros.
                    override = (
                        snap.frozen_upper_velocities if snap.frozen_upper_velocities is not None else np.zeros(17)
                    )
                    values = values.copy()
                    values[:, _UPPER_BODY_IL] = override
            else:
                values = values[:, indexes]
        self._write_np_slot(name, values)
        return True

    def _gather_anchor_orientation(
        self, name: str, snap: MotionSnapshot, base_quat: np.ndarray, n: int, step: int
    ) -> bool:
        motion = snap.motion
        if motion is None or motion.timesteps == 0 or motion.body_quat.shape[1] == 0:
            return False
        self._scheduler.note_observation_window(n, step)
        out = np.zeros((n, 6))
        base_inv = gu.inv_quat(base_quat)
        for i, frame in enumerate(_target_frames(snap, n, step)):
            # 6D = first two columns, row-major, of R(conj(base) ⊗ (apply_delta_heading ⊗ ref_root)).
            # transform_quat_by_quat(v, u) = u ⊗ v.
            ref_root = gu.transform_quat_by_quat(motion.body_quat[frame, 0], snap.apply_delta_heading)
            rot = gu.quat_to_R(gu.transform_quat_by_quat(ref_root, base_inv))
            out[i] = rot[:, :2].reshape(-1)
        self._write_np_slot(name, out)
        return True

    def _gather_smpl_joints(self, name: str, snap: MotionSnapshot, n: int, step: int) -> bool:
        motion = snap.motion
        if motion is None or motion.timesteps == 0 or motion.smpl_joints is None or motion.smpl_joints.shape[1] == 0:
            return False
        self._scheduler.note_observation_window(n, step)
        values = motion.smpl_joints[_target_frames(snap, n, step)]  # (n, 24, 3)
        self._write_np_slot(name, values)
        return True

    def _gather_vr_position(self, name: str, snap: MotionSnapshot) -> bool:
        if snap.motion is None or snap.motion.timesteps == 0:
            return False
        # ZMQ values are used verbatim (already root-local with offsets applied
        # by the sender); before any arrive, the reference defaults hold. The
        # reference's third source — FK from a clip's full body table — never
        # applies here: teleop motions carry only the root body.
        values = snap.vr_position if snap.vr_position is not None else VR_3POINT_DEFAULT_POSITION
        self._write_np_slot(name, np.asarray(values, dtype=np.float64))
        return True

    def _gather_vr_orientation(self, name: str, snap: MotionSnapshot) -> bool:
        if snap.motion is None or snap.motion.timesteps == 0:
            return False
        values = snap.vr_orientation if snap.vr_orientation is not None else VR_3POINT_DEFAULT_ORIENTATION
        self._write_np_slot(name, np.asarray(values, dtype=np.float64))
        return True


# Sanity: the permutation tables this module depends on stay in lockstep.
assert len(UPPER_BODY_IL_INDICES) == 17
assert set(UPPER_BODY_IL_INDICES) | set(LOWER_BODY_IL_INDICES) == set(range(29))
assert set(WRIST_IL_INDICES) < set(UPPER_BODY_IL_INDICES)
assert len(IL_TO_MJ) == 29
