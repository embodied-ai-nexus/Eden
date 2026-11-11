"""GEAR-SONIC whole-body motion-tracking policy port (G1 29-DOF, g1 encoder mode).

Loads NVIDIA's released SONIC ONNX bundle (`model_encoder.onnx` +
`model_decoder.onnx`) and reproduces the deployment-time observation
construction and action transform from the `gear_sonic_deploy` C++ reference
in Python so the policy can be driven inside an Eden Genesis environment.

I/O contract (matches `observation_config.yaml` for the released bundle):

    encoder obs (1762 dim, g1 mode only — other modes' slots zero-filled)
        encoder_mode_4              [mode_id, 0, 0, 0] (g1→all zeros)    (4)
        motion_joint_positions_10frame_step5 future joint pos (raw), IL order (290)
        motion_joint_velocities_10frame_step5 future joint vel (raw), IL order (290)
        motion_root_z_position_10frame_step5  zeros                      (10)
        motion_root_z_position             zeros                         (1)
        motion_anchor_orientation          zeros (6D)                    (6)
        motion_anchor_orientation_10frame_step5 future rot diff, body frame  (60)
        motion_joint_positions_lowerbody_10frame_step5 zeros             (120)
        motion_joint_velocities_lowerbody_10frame_step5 zeros            (120)
        vr_3point_local_target             zeros                         (9)
        vr_3point_local_orn_target         zeros                         (12)
        smpl_joints_10frame_step1          zeros                         (720)
        smpl_anchor_orientation_10frame_step1 zeros                      (60)
        motion_joint_positions_wrists_10frame_step1 zeros                (60)

    encoder out → token_state                                            (64)

    decoder obs (994 dim, history newest-LAST)
        token_state                                                      (64)
        his_base_angular_velocity_10frame_step1  pelvis-frame ω          (30)
        his_body_joint_positions_10frame_step1   q_rel, IL order         (290)
        his_body_joint_velocities_10frame_step1  qdot, IL order          (290)
        his_last_actions_10frame_step1           raw decoder outputs     (290)
        his_gravity_dir_10frame_step1            R(pelvis)ᵀ·[0,0,-1]     (30)

    decoder out → action (29, IsaacLab order)

Action transform (replicates `g1_deploy_onnx_ref.cpp::CreatePolicyCommand`):

    for each MJCF index i (0..28):
        target[i] = DEFAULT_ANGLES_MJ[i] + ACTION_SCALE_MJ[i] * action[IL_TO_MJ[i]]

    `target` (MJCF order) is fed into `ImplicitPDController` with scale=1.0 and
    `reference_source=ZERO` so Genesis's built-in PD treats it as the absolute
    joint-position target (no extra offset / scaling on top).

Status / limitations
--------------------
* Only the **g1** encoder mode is implemented (motion-tracking from a Genesis
  motion file). VR teleop and SMPL paths zero-fill the encoder input — the
  encoder still runs, but those observation slots carry no information.
* num_envs is unrestricted (the ONNX graphs support arbitrary batch sizes);
  per-env state (history ring buffers, heading offset, wrap detection) is
  fully batched and partial resets only touch the affected rows. The
  num_envs=1 configuration matches the upstream deploy contract.
* The kinematic reference is read via Eden's standard
  :class:`~eden.managers.terms.commands.mimic.MotionCommand`
  with ``anchor_link_name="pelvis"``, matching SONIC's
  ``manager_env/commands/terms/motion.yaml``. The motion file is expected to
  carry ``joint_pos`` / ``joint_vel`` in MJCF order (Eden's convention); the
  policy reorders to IsaacLab internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from genesis.utils.geom import inv_quat, inv_transform_by_quat, transform_quat_by_quat

from eden import actions, observations
from eden.constants import ReferenceSource
from eden.extensions.policies.base import POLICY_REGISTRY, PolicyBase, PolicyOptions
from eden.extensions.policies.sonic.joint_maps import (
    ACTION_SCALE_MJ,
    DEFAULT_ANGLES_MJ,
    IL_TO_MJ,
    MJ_TO_IL,
    MODE_G1,
    NUM_MODES,
    SONIC_DOFS_KD,
    SONIC_DOFS_KP,
    SONIC_DOFS_POS,
)
from eden.extensions.policies.sonic.onnx_wrapper import SonicDecoder, SonicEncoder
from eden.managers.terms.commands.mimic import motion_command_options
from eden.options.managers.observations import ObservationGroupOptions
from eden.options.robots.unitree_g1 import G1Dofs29
from eden.utils.geom import quat_to_rot6d_cols, yaw_quat

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.actions import ActionTermOptions
    from eden.options.managers.commands import CommandTermOptions


@dataclass(frozen=True)
class RobotStateSnapshot:
    """One tick's raw robot reads, batched by the caller.

    Lets a single set of device→host reads serve both the policy observation
    and the caller's other consumers (planner context, fall check, g1_debug),
    mirroring the reference deploy, which logs one ``LowState`` per control tick
    and serves every observation from it (g1_deploy_onnx_ref.cpp
    ``GatherRobotStateToLogger``). Joints are in ``dofs_name``/MJCF order and
    ``base_ang_local`` is ``get_ang(frame="body")``; ``pelvis_quat_wxyz`` is in
    whichever frame the caller reads — the teleop runtime reads
    ``get_quat(relative=True)`` (the MJCF pelvis frame its reference sim reports).
    Each tensor keeps its ``(N, ...)`` batch dim.
    """

    q_mj: torch.Tensor  # (N, 29)
    qdot_mj: torch.Tensor  # (N, 29)
    base_ang_local: torch.Tensor  # (N, 3)
    pelvis_quat_wxyz: torch.Tensor  # (N, 4) wxyz


# ---------------------------------------------------------------------------
# Shape / cadence constants — frozen by the release ONNX bundle
# ---------------------------------------------------------------------------

_NUM_DOFS: int = 29
_TOKEN_DIM: int = 64
_HISTORY_LEN: int = 10
_FUTURE_LEN: int = 10
_FUTURE_STEP: int = 5  # every 5th motion frame at 50 fps motion → 0.1 s spacing
_ANCHOR_BODY: str = "pelvis"
# MotionCommand needs to expose pelvis (for floating-base reset) and the bodies
# the policy reads. SONIC reads only the pelvis (anchor) body — joint pos/vel
# come from the command's DOF arrays, not from per-body lookups.
_MOTION_BODY_NAMES: list[str] = [_ANCHOR_BODY]

# Encoder-input slot dims (must sum to the encoder ONNX's static input dim).
_ENCODER_SLOTS: dict[str, int] = {
    "encoder_mode_4": NUM_MODES,  # 4
    "motion_joint_positions_10frame_step5": _FUTURE_LEN * _NUM_DOFS,  # 290
    "motion_joint_velocities_10frame_step5": _FUTURE_LEN * _NUM_DOFS,  # 290
    "motion_root_z_position_10frame_step5": _FUTURE_LEN,  # 10
    "motion_root_z_position": 1,  # 1
    "motion_anchor_orientation": 6,  # 6
    "motion_anchor_orientation_10frame_step5": _FUTURE_LEN * 6,  # 60
    "motion_joint_positions_lowerbody_10frame_step5": _FUTURE_LEN * 12,  # 120
    "motion_joint_velocities_lowerbody_10frame_step5": _FUTURE_LEN * 12,  # 120
    "vr_3point_local_target": 9,  # 9
    "vr_3point_local_orn_target": 12,  # 12
    "smpl_joints_10frame_step1": _HISTORY_LEN * 72,  # 720
    "smpl_anchor_orientation_10frame_step1": _HISTORY_LEN * 6,  # 60
    "motion_joint_positions_wrists_10frame_step1": _HISTORY_LEN * 6,  # 60
}
_ENCODER_INPUT_DIM: int = sum(_ENCODER_SLOTS.values())  # 1762

# Decoder-input slot dims (must sum to the decoder ONNX's static input dim).
_DECODER_SLOTS: dict[str, int] = {
    "token_state": _TOKEN_DIM,  # 64
    "his_base_angular_velocity_10frame_step1": _HISTORY_LEN * 3,  # 30
    "his_body_joint_positions_10frame_step1": _HISTORY_LEN * _NUM_DOFS,  # 290
    "his_body_joint_velocities_10frame_step1": _HISTORY_LEN * _NUM_DOFS,  # 290
    "his_last_actions_10frame_step1": _HISTORY_LEN * _NUM_DOFS,  # 290
    "his_gravity_dir_10frame_step1": _HISTORY_LEN * 3,  # 30
}
_DECODER_INPUT_DIM: int = sum(_DECODER_SLOTS.values())  # 994


def _slot_offsets(slot_dims: dict[str, int]) -> dict[str, tuple[int, int]]:
    """Return ``{name: (start, end)}`` element offsets in a flat encoder/decoder buffer."""
    offsets: dict[str, tuple[int, int]] = {}
    cur = 0
    for name, dim in slot_dims.items():
        offsets[name] = (cur, cur + dim)
        cur += dim
    return offsets


_ENCODER_OFFSETS = _slot_offsets(_ENCODER_SLOTS)
_DECODER_OFFSETS = _slot_offsets(_DECODER_SLOTS)


# ---------------------------------------------------------------------------
# 6D rotation representation: drop the last column of the rotation matrix
# (= what isaaclab's matrix_from_quat(...)[:, :2].reshape(-1, 6) produces). The
# encoder's anchor-orientation slot expects this 6D form, not a quaternion.
# ---------------------------------------------------------------------------


def _quat_to_6d(q_wxyz: torch.Tensor) -> torch.Tensor:
    """Convert wxyz quaternions to the 6D rotation rep used by SONIC.

    Returns the first two columns of the rotation matrix flattened
    **row-by-row**: ``[R00, R01, R10, R11, R20, R21]``. This matches both
    isaaclab's ``matrix_from_quat(q)[..., :2].reshape(-1, 6)`` (numpy/torch
    row-major reshape over the last two dims) and the deploy C++'s
    ``{R[0][0], R[0][1], R[1][0], R[1][1], R[2][0], R[2][1]}`` literal.

    A column-stacked layout ``[R00, R10, R20, R01, R11, R21]`` would *also*
    "round-trip" through a Gram-Schmidt decoder but would feed completely
    different scalars to the encoder than what it was trained against — the
    learned weights interpret position 1 as ``R01`` (row 0 col 1), not
    ``R10`` (row 1 col 0), so the convention matters.

    Parameters
    ----------
    q_wxyz
        Quaternions of shape ``(..., 4)``.

    Returns
    -------
    torch.Tensor
        First two columns of the rotation matrix flattened row-major to
        ``(..., 6)``.
    """
    # Thin named alias for SONIC's frozen input contract; the layout itself is
    # `eden.utils.geom.quat_to_rot6d_cols` (NOT `quat_to_rot6d`, which is the
    # first two *rows* of R — see that function's docstring).
    return quat_to_rot6d_cols(q_wxyz)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@POLICY_REGISTRY.register()
class SonicPolicy(PolicyBase):
    """GEAR-SONIC universal-token whole-body controller (g1 mode, ONNX backend).

    The SONIC release is split into two ONNX graphs (encoder + decoder); both
    are downloaded from ``Kashu7100/eden_policies`` (mirror of NVIDIA's HF repo)
    on first use. To use the official NVIDIA bundle directly, set the
    ``checkpoint_path`` / ``decoder_checkpoint_path`` class attrs to a local
    path before instantiating, or pass them through ``configure(...)``.
    """

    entity_name: str = "robot"

    # Resolved by PolicyBase (``fetch_asset``). Hashes pinned against the
    # nvidia/GEAR-SONIC release mirrored to Kashu7100/eden_policies — any
    # corruption / re-export will fail loudly at construction time.
    checkpoint_path: str = "sonic/model_encoder.onnx"
    md5_hash: str | None = "a21244b1cdceac9dc8625442a55d42e0"
    decoder_checkpoint_path: str = "sonic/model_decoder.onnx"
    decoder_md5_hash: str | None = "1d4391ad404d0ff36281abec349b23fd"

    def __init__(self, env: EnvBase, options: PolicyOptions) -> None:
        super().__init__(env, options)

        # PolicyBase already resolved the encoder path. Resolve the decoder
        # path via the same Hugging Face asset resolver (mirrors omnixtreme).
        self._decoder_path = self.fetch_asset(self.decoder_checkpoint_path)
        self._verify_md5(self._decoder_path, self.decoder_md5_hash)

        device = env.device
        self.device = device
        self._dtype = torch.float32

        # OnnxModule validates the static input dim against these expectations
        # (no-op when the graph has a dynamic axis), raising on a swapped/re-exported file.
        self._encoder = SonicEncoder(self.checkpoint_path, device=str(device), expected_input_dim=_ENCODER_INPUT_DIM)
        self._decoder = SonicDecoder(self._decoder_path, device=str(device), expected_input_dim=_DECODER_INPUT_DIM)

        robot = self.require_entity(self.entity_name)
        # MJCF (Eden) DOF order — matches the motion file and Genesis kinematic state.
        _, dofs_idx = robot.find_named_dofs_idx_local(G1Dofs29().dofs_name, preserve_order=True)
        self._dofs_idx_mj = torch.as_tensor(dofs_idx, dtype=torch.long, device=device)
        # Pelvis (anchor) idx_local in Genesis kinematic-tree order (BFS in genesis 1.1.x,
        # DFS in 1.2.0). Name-resolved, so it indexes the robot's own get_links_* correctly
        # under either ordering (it is NOT used to slice the motion npz).
        _, pelvis_idx = robot.find_named_links_idx_local([_ANCHOR_BODY], preserve_order=True)
        self._pelvis_link_idx = torch.as_tensor(pelvis_idx, dtype=torch.long, device=device)

        # Permutations: MJCF → IsaacLab (gather index that picks each IL slot's
        # value out of an MJCF-ordered tensor) and the inverse.
        self._mj_to_il = torch.as_tensor(MJ_TO_IL, dtype=torch.long, device=device)
        self._il_to_mj = torch.as_tensor(IL_TO_MJ, dtype=torch.long, device=device)

        # Action-transform constants (MJCF / Eden order).
        self._default_angles_mj = torch.tensor(DEFAULT_ANGLES_MJ, device=device, dtype=self._dtype)
        self._action_scale_mj = torch.tensor(ACTION_SCALE_MJ, device=device, dtype=self._dtype)

        # Future motion frame offsets [0, step, 2*step, ..., (N-1)*step] — the
        # first future frame IS the current motion frame, matching both the C++
        # deploy (`target_frame = current_frame + frame_idx * step_size`,
        # frame_idx=0..N-1) and gear_sonic's `future_time_steps_init =
        # arange(num_future_frames) * frame_skips`.
        self._future_offsets = torch.arange(
            0, _FUTURE_LEN * _FUTURE_STEP, _FUTURE_STEP, dtype=torch.long, device=device
        )

        # encoder_mode_4: scalar mode_id at slot 0, zeros at 1..3 — NOT a
        # one-hot vector. The C++ deploy code
        # (`GatherEncoderMode(buf, off, fill_zeros_num=3)`) writes
        # ``buf[off] = static_cast<float>(GetEncodeMode())`` and zero-pads the
        # remaining three slots. For g1 (mode_id=0) this is `[0, 0, 0, 0]`.
        # A one-hot `[1, 0, 0, 0]` would tell the encoder we're in teleop mode
        # (mode_id=1) and feed the g1 motion obs to the wrong branch.
        self._encoder_mode_value = torch.zeros(NUM_MODES, device=device, dtype=self._dtype)
        self._encoder_mode_value[0] = float(MODE_G1)  # = 0.0 — kept explicit for clarity

        # Heading correction (apply_delta_heading in the C++ deploy). Captured
        # per env on the first step() after that env's reset: the yaw-only
        # rotation that aligns the motion's start heading with the robot's
        # start heading. Without this, the robot's world frame drifts away from
        # the motion's world frame and future-anchor obs degrade as the episode
        # progresses. Stored per env so partial resets / per-env motion
        # wrap-around never clobber the offsets of envs mid-episode.
        N = env.num_envs
        self._heading_offset_wxyz = torch.zeros(N, 4, device=device, dtype=self._dtype)
        self._heading_offset_wxyz[:, 0] = 1.0  # identity until first capture
        self._needs_heading_capture = torch.ones(N, dtype=torch.bool, device=device)
        # Previous-step motion frame index. We use this to auto-detect motion
        # wrap-around (`MotionCommand._update_command` resamples when
        # `time_steps >= time_step_total`) per env and re-capture that env's
        # heading offset against the new motion-start pose. Without this, long
        # episodes that outlive the motion silently keep the stale offset
        # across the wrap.
        self._prev_time_steps: torch.Tensor | None = None

        # ----- history ring buffers -----
        # All in IsaacLab order, newest-LAST (so the model gets the most recent
        # frame at index _HISTORY_LEN-1 when the buffer is flattened).
        N = env.num_envs
        self._hist_q = torch.zeros(N, _HISTORY_LEN, _NUM_DOFS, device=device, dtype=self._dtype)
        self._hist_qdot = torch.zeros(N, _HISTORY_LEN, _NUM_DOFS, device=device, dtype=self._dtype)
        self._hist_action = torch.zeros(N, _HISTORY_LEN, _NUM_DOFS, device=device, dtype=self._dtype)
        self._hist_base_ang = torch.zeros(N, _HISTORY_LEN, 3, device=device, dtype=self._dtype)
        self._hist_gravity = torch.zeros(N, _HISTORY_LEN, 3, device=device, dtype=self._dtype)
        self._hist_buffers = (
            self._hist_q,
            self._hist_qdot,
            self._hist_action,
            self._hist_base_ang,
            self._hist_gravity,
        )

        # Scratch buffers, refilled in-place each step.
        self._encoder_buf = torch.zeros(N, _ENCODER_INPUT_DIM, device=device, dtype=self._dtype)
        self._decoder_buf = torch.zeros(N, _DECODER_INPUT_DIM, device=device, dtype=self._dtype)

        # World-frame "down" vector, expanded once to the env-batch shape so
        # the per-step pelvis-frame projection can skip a `zeros_like` + scalar
        # assignment. Read-only on the hot path.
        self._gravity_world = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=self._dtype).expand(N, 3)

    # ------------------------------------------------------------------
    # Env / action / command / observation specs
    # ------------------------------------------------------------------

    @classmethod
    def robot_options(cls) -> G1Dofs29:
        """Return a ``G1Dofs29`` carrying the standing pose and PD gains the checkpoint expects.

        This is the robot a task-less ``eden deploy/inference --policy sonic`` builds its
        scene from, so it decides both where the robot stands before the policy takes over
        and — via :class:`~eden.extensions.deployment.base.DeploymentBase`, which resolves
        ``default_dofs_kp`` / ``default_dofs_kd`` once into the Kp/Kd sent on the wire —
        what PD loop tracks its targets.

        **Pose** is :data:`~eden.extensions.policies.sonic.joint_maps.SONIC_DOFS_POS`, the
        ``default_angles[]`` the action transform is centred on. The stock G1's shallower
        stance would leave the robot 0.37 rad off at the knees and 0.9 rad at the elbows
        when SONIC's first target lands.

        **Gains** differ from the stock G1 in the hip-pitch row alone: SONIC derives them
        from the same motor table by the same Raibert heuristic Eden does, and hip pitch is
        the one row where the tables disagree — SONIC's is a 7520-22, Eden's stock spec a
        7520-14. The stock gains would run the hips at **0.41x** the released checkpoint's
        stiffness (40.18 vs 99.10 N·m/rad) under the 7520-22 action scale, i.e. 2.4x less
        torque per unit action than trained.

        Armature and effort limits stay on the stock spec: those describe the physical
        actuator, not the controller, and which motor a given G1 revision carries is a
        hardware question this port cannot settle. Note the consequence — Genesis builds
        hip pitch with the 7520-14 armature (0.0102 vs SONIC's 0.0251 kg·m²) and clamps its
        torque at 88 rather than 139 N·m, so a Genesis rollout is still not SONIC's
        training dynamics even with the gains fixed.
        """
        return G1Dofs29(default_dofs_pos=SONIC_DOFS_POS, default_dofs_kp=SONIC_DOFS_KP, default_dofs_kd=SONIC_DOFS_KD)

    @classmethod
    def env_options(cls) -> dict[str, Any]:
        # 50 Hz control to match the released SONIC ONNX (motion target_fps=50,
        # decimation=2 → control_dt=0.02 s). The deploy stack also runs at 50 Hz.
        return dict(
            sim_dt=0.01,
            sim_substeps=2,
            decimation=2,
            iterations=20,
            ls_iterations=20,
            max_collision_pairs=20,
            batch_dofs_info=False,
            use_gjk_collision=False,
            enable_self_collision=False,
            enable_multi_contact=False,
        )

    @classmethod
    def action_options(cls, entity_name: str = "robot") -> dict[str, ActionTermOptions]:
        # `SonicPolicy.step` already converts the policy output to absolute
        # joint-position targets (default_angles + scale * action) in MJCF
        # order, so the action term just hands those targets to Genesis's
        # built-in PD as-is.
        return dict(
            dofs_pos_controller=actions.ImplicitPDController.configure(
                entity_name=entity_name,
                dofs_name=G1Dofs29().dofs_name,
                scale=1.0,
                reference_source=ReferenceSource.ZERO,
            ),
        )

    @classmethod
    def command_options(
        cls,
        motion_file: str | list[str] = "motion.npz",
        debug_vis: bool = False,
        entity_name: str = "robot",
        ghost_entity_name: str = "",
        sampling_mode: str = "start",
    ) -> dict[str, CommandTermOptions]:
        # A list of clips -> per-env multi-clip term; a single path -> single-clip.
        # SonicPolicy.step reads motion through the shared accessor interface, so it
        # is unaware which term backs it.
        return motion_command_options(
            motion_file,
            anchor_link_name=_ANCHOR_BODY,
            links_name=_MOTION_BODY_NAMES,
            entity_name=entity_name,
            ghost_entity_name=ghost_entity_name,
            debug_vis=debug_vis,
            sampling_mode=sampling_mode,
        )

    @classmethod
    def observation_options(cls, entity_name: str = "robot") -> dict[str, ObservationGroupOptions]:
        # The policy reads raw simulator state directly in `step()`. Eden still
        # requires at least one observation term to build the env, so we expose
        # a minimal placeholder.
        return dict(
            policy=ObservationGroupOptions(
                concatenate_terms=True,
                enable_corruption=False,
                dofs_pos=observations.DofsPos.configure(entity_name=entity_name),
            ),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None) -> None:
        """Clear per-episode state at episode boundaries.

        Resets:
        * History ring buffers (``q``, ``qdot``, ``action``, ``base_ang``,
          ``gravity``). Without clearing, the first decoder call after a reset
          sees the previous episode's last 10 frames and produces a
          discontinuous initial action.
        * The heading offset — re-captured per env on the next ``step()``
          against whatever pose the motion was just resampled to. Critical for
          ``sampling_mode != "start"`` (where the motion frame index changes
          across episodes) and on motion wrap-around. Only the affected rows
          are marked for re-capture, so partial resets in batched rollouts
          leave the other envs' offsets intact.
        """
        for buf in self._hist_buffers:
            if envs_idx is None:
                buf.zero_()
            else:
                buf[envs_idx] = 0.0
        # Mark the affected envs for heading re-capture on their next step
        # against the fresh post-reset state.
        if envs_idx is None:
            self._needs_heading_capture.fill_(True)
            self._prev_time_steps = None
        else:
            self._needs_heading_capture[envs_idx] = True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def step(self, obs: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        # Read off the entity, not an observation group: this policy rides along as an extra deploy
        # slot on another policy's or task's env, whose groups are not its own (see the
        # "Declare the observation" section of docs/guides/policy-porting.md).
        del obs  # raw simulator state is read directly from the env
        robot = self.env.entities[self.entity_name]
        motion_cmd = self.env.command_manager.get_term("motion")

        pelvis_quat_wxyz = self._update_state_history(robot)

        # ---- motion future references (g1 mode encoder input) ----
        time_steps = motion_cmd.time_steps
        # Per-env motion length (multi-clip envs each track their own clip), so
        # the end-of-motion clamp is per-env. Single-clip: a constant vector.
        last_frame = motion_cmd.frame_total_per_env() - 1  # (N,)
        # NOTE: end-of-motion behavior is `clamp` (hold the last frame), not
        # `wrap` — matches the C++ deploy. If the motion wraps via
        # `_update_command`, the wrap-detection block below re-captures the
        # heading offset against the new motion-start pose.
        # `.minimum` (not `.clamp_max`) because the bound is a per-env tensor.
        future_t = (time_steps[:, None] + self._future_offsets[None, :]).minimum(last_frame[:, None])

        # Motion joint pos / vel are fed RAW (no default-angle subtraction) —
        # this matches both gear_sonic's `command_multi_future` (returns
        # `motion_lib.get_dof_pos(...)` directly) and the C++ deploy's
        # `GatherMotionJointPositionsMultiFrame` (raw memcpy from
        # `motion->JointPositions(frame)`). Only the robot's history obs
        # subtracts the default — see the q_rel computation in _update_state_history.
        # Motion data is stored in MJCF order (Eden's convention); reorder to IL.
        future_q_mj = motion_cmd.gather_motion_frames("dofs_pos", future_t)  # (N, F, 29)
        future_qdot_mj = motion_cmd.gather_motion_frames("dofs_vel", future_t)  # (N, F, 29)
        future_q_il = future_q_mj[..., self._mj_to_il]
        future_qdot_il = future_qdot_mj[..., self._mj_to_il]

        # ---- heading alignment (yaw-only) ----
        # `apply_delta_heading` in the C++ deploy =
        #   R(robot_init_yaw) · R(motion_init_yaw)^{-1}
        # Captured per env on the first step after that env's reset AND on that
        # env's motion wrap-around (when `MotionCommand._update_command`
        # resamples and its `time_steps` regresses). The same yaw offset is
        # applied to all future motion anchor quats until the env's next
        # reset/wrap. Without this, the robot's world heading drifts away from
        # the motion's heading and the anchor-orientation obs degrades.
        motion_pelvis_now_wxyz = motion_cmd.gather_motion_frames("links_quat_w", time_steps[:, None])[:, 0, 0]  # (N, 4)
        capture = self._needs_heading_capture.clone()
        if self._prev_time_steps is not None:
            capture |= time_steps < self._prev_time_steps  # per-env wrap detection
        # Compute the offset full-batch and write only the rows that need (re)capture.
        # No `bool(capture.any())` guard: that scalar read forces a per-step GPU→CPU
        # sync on the inference hot path. The masked `torch.where` is already a no-op
        # for the (usual) all-false case, matching CommandManager's unconditional-
        # masked-op convention.
        # `apply_delta_heading = robot_yaw * inv(motion_yaw)`, left-multiplied onto
        # the raw motion anchor quat.
        robot_yaw = yaw_quat(pelvis_quat_wxyz)
        motion_yaw = yaw_quat(motion_pelvis_now_wxyz)
        offset = transform_quat_by_quat(inv_quat(motion_yaw), robot_yaw)
        self._heading_offset_wxyz = torch.where(capture[:, None], offset, self._heading_offset_wxyz)
        self._needs_heading_capture &= ~capture
        self._prev_time_steps = time_steps.clone()

        # Future anchor (pelvis) orientation as 6D rotation in robot pelvis frame:
        #   corrected = apply_delta_heading · raw_ref
        #   rel       = inv(R(robot_pelvis_now)) · corrected
        # then convert to the 6D rep used by gear_sonic's motion_anchor_ori_b_mf.
        #
        # Both operands are (N, F, 4) — `transform_quat_by_quat` broadcasts on
        # the leading axes, so we skip the explicit `.expand(...).reshape(...)`
        # round-trips and only flatten once before the final `_quat_to_6d`.
        ref_pelvis_quat = motion_cmd.gather_motion_frames("links_quat_w", future_t)[..., 0, :]  # (N, F, 4)
        heading = self._heading_offset_wxyz[:, None, :]  # (N, 1, 4)
        cur_pelvis_inv = inv_quat(pelvis_quat_wxyz)[:, None, :]  # (N, 1, 4)
        corrected_ref = transform_quat_by_quat(ref_pelvis_quat, heading)  # (N, F, 4)
        rel_quat = transform_quat_by_quat(corrected_ref, cur_pelvis_inv)  # (N, F, 4)
        rel_anchor_6d = _quat_to_6d(rel_quat).reshape(-1, _FUTURE_LEN * 6)

        # ---- build encoder input (1762 dim, mostly zero for g1 mode) ----
        N = self._encoder_buf.shape[0]
        self._encoder_buf.zero_()
        self._write_slot(
            self._encoder_buf,
            _ENCODER_OFFSETS,
            "encoder_mode_4",
            self._encoder_mode_value.expand(N, -1),
        )
        self._write_slot(
            self._encoder_buf,
            _ENCODER_OFFSETS,
            "motion_joint_positions_10frame_step5",
            future_q_il.reshape(N, _FUTURE_LEN * _NUM_DOFS),
        )
        self._write_slot(
            self._encoder_buf,
            _ENCODER_OFFSETS,
            "motion_joint_velocities_10frame_step5",
            future_qdot_il.reshape(N, _FUTURE_LEN * _NUM_DOFS),
        )
        self._write_slot(self._encoder_buf, _ENCODER_OFFSETS, "motion_anchor_orientation_10frame_step5", rel_anchor_6d)
        # All other slots stay zero — they're for teleop / SMPL modes.

        # ---- encoder → token_state → decoder → action (IL order) ----
        action_il = self._decode(self._encode())
        self._push_action_history(action_il)

        # ---- isaaclab→mjcf permutation + action transform ----
        action_mj = action_il[..., self._il_to_mj]
        joint_pos_targets = self._default_angles_mj + self._action_scale_mj * action_mj
        return {"dofs_pos_controller": joint_pos_targets}

    # ------------------------------------------------------------------
    # Helpers (shared with the teleop subclass)
    # ------------------------------------------------------------------

    def _update_state_history(self, robot, state: RobotStateSnapshot | None = None) -> torch.Tensor:
        """Read the robot state and roll the four state-history buffers.

        Returns the pelvis quaternion (wxyz) for the caller's frame math. When
        ``state`` is given (the teleop runtime's single per-tick snapshot), its
        pre-read tensors are used instead of re-reading the device.

        Without ``state`` the pelvis quat is the default ``get_quat()``, not
        ``relative=True``: under ``eden deploy`` ``robot`` is a
        :class:`~eden.extensions.deployment.utils.state.RobotStateEntity`, which
        serves the hardware IMU orientation and raises on ``relative=True``.
        """
        # ---- current robot state (MJCF order, then permute to IL) ----
        if state is None:
            q_mj = robot.get_dofs_pos(dofs_idx_local=self._dofs_idx_mj)
            qdot_mj = robot.get_dofs_vel(dofs_idx_local=self._dofs_idx_mj)
            # SONIC's IMU lives on the pelvis: pelvis-frame angular velocity + pelvis quat.
            base_ang_local = robot.get_ang(frame="body")
            pelvis_quat_wxyz = robot.get_quat()
        else:
            q_mj = state.q_mj
            qdot_mj = state.qdot_mj
            base_ang_local = state.base_ang_local
            pelvis_quat_wxyz = state.pelvis_quat_wxyz
        # IsaacLab convention used by SONIC: q_rel = q - default; qdot raw.
        q_rel_mj = q_mj - self._default_angles_mj
        q_rel_il = q_rel_mj[..., self._mj_to_il]
        qdot_il = qdot_mj[..., self._mj_to_il]

        # SONIC's IMU lives on the pelvis, so projected gravity uses the pelvis quat.
        gravity_local = inv_transform_by_quat(self._gravity_world, pelvis_quat_wxyz)

        # Roll history forward by 1 step (drop oldest, write newest at index -1)
        # using an in-place copy-shift so we avoid fresh `(N, _HISTORY_LEN, D)`
        # allocations per step. Cheaper than `torch.roll` and the buffers
        # remain the same tensors across the episode.
        for buf, sample in (
            (self._hist_q, q_rel_il),
            (self._hist_qdot, qdot_il),
            (self._hist_base_ang, base_ang_local),
            (self._hist_gravity, gravity_local),
        ):
            buf[:, :-1].copy_(buf[:, 1:].clone())
            buf[:, -1] = sample
        return pelvis_quat_wxyz

    def _encode(self) -> torch.Tensor:
        """Run the encoder on the (already filled) encoder scratch buffer."""
        return self._encoder(self._encoder_buf)

    def _decode(self, token_state: torch.Tensor) -> torch.Tensor:
        """Fill the decoder buffer from the token + history buffers, run the decoder."""
        N = self._decoder_buf.shape[0]
        self._decoder_buf.zero_()
        self._write_slot(self._decoder_buf, _DECODER_OFFSETS, "token_state", token_state)
        self._write_slot(
            self._decoder_buf,
            _DECODER_OFFSETS,
            "his_base_angular_velocity_10frame_step1",
            self._hist_base_ang.reshape(N, -1),
        )
        self._write_slot(
            self._decoder_buf, _DECODER_OFFSETS, "his_body_joint_positions_10frame_step1", self._hist_q.reshape(N, -1)
        )
        self._write_slot(
            self._decoder_buf,
            _DECODER_OFFSETS,
            "his_body_joint_velocities_10frame_step1",
            self._hist_qdot.reshape(N, -1),
        )
        self._write_slot(
            self._decoder_buf, _DECODER_OFFSETS, "his_last_actions_10frame_step1", self._hist_action.reshape(N, -1)
        )
        self._write_slot(
            self._decoder_buf, _DECODER_OFFSETS, "his_gravity_dir_10frame_step1", self._hist_gravity.reshape(N, -1)
        )
        return self._decoder(self._decoder_buf)

    def _push_action_history(self, action_il: torch.Tensor) -> None:
        """Push the raw decoder output into history *before* the action transform.

        Matches the C++ deploy convention (``last_action[i] = floatarr[i]``:
        unscaled, IsaacLab order). In-place copy-shift mirrors the four
        state-history buffers.
        """
        self._hist_action[:, :-1].copy_(self._hist_action[:, 1:].clone())
        self._hist_action[:, -1] = action_il

    @staticmethod
    def _write_slot(buf: torch.Tensor, offsets: dict[str, tuple[int, int]], name: str, data: torch.Tensor) -> None:
        """In-place write into the encoder/decoder scratch buffer at slot ``name``.

        Pass the offsets table explicitly (``_ENCODER_OFFSETS`` /
        ``_DECODER_OFFSETS``) instead of inferring it from the slot name — the
        encoder and decoder layouts happen to share no names today, but
        relying on that for dispatch makes a future name collision silently
        corrupt the wrong buffer.

        ``data`` must already have shape ``(N, slot_dim)`` and live on the
        same device/dtype as ``buf``.
        """
        start, end = offsets[name]
        buf[:, start:end] = data
