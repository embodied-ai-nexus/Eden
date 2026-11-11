"""Smooth start-transition dataset modifier.

Pre-pends every clip with a synthesized ``stand-still → IK-stepped → SLERP-
merged`` segment so RSI from a default pose to clip-frame-0 is no longer a
teleport. Ports OpenTrack's ``StartEndTransitionHandler`` to Eden's
:class:`DatasetModifier` API. The actual IK math lives in
:mod:`eden.utils.motion_reference.bipedal_ik`; this module is the glue that
maps Eden's :class:`BatchMotionSequence` ↔ a Genesis ``RigidEntity`` and
re-FKs link fields after the transition is inserted.

Why a modifier and not a per-step env hook
------------------------------------------
The IK solve is per-clip and sequential (no benefit from vectorising over
envs), and it runs hundreds of forward-kinematics passes per clip. Doing it
inside the training loop would thrash wall-time. As a modifier it runs once
per dataset load, before training starts. Modifiers are not baked into the
processed cache, so the solve is repeated on every load.

What this modifier mutates
--------------------------
The output ``BatchMotionSequence`` has per-sequence lengths extended by the
synthesized prefix. Fields rebuilt from scratch:
    * ``base_pos_w``, ``base_quat_w``, ``dofs_pos``
    * ``base_lin_vel_w``, ``base_ang_vel_w``, ``dofs_vel`` (FD from positions)
    * ``links_pos_w``, ``links_quat_w`` (re-FK through Genesis)
    * ``links_lin_vel_w``, ``links_ang_vel_w`` (FD from link poses)
    * ``links_pos_b``, ``links_quat_b``, ``links_lin_vel_b``,
      ``links_ang_vel_b`` (recomputed from world-frame + the new base pose)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

import eden as en
from eden.datasets.modifiers.base import DATASET_MODIFIER_REGISTRY, DatasetModifier
from eden.utils.buffers.cbt import ConcatBatchTensor
from eden.utils.genesis_runtime import transient_robot_env
from eden.utils.motion_reference.bipedal_ik import (
    BipedalIKProfile,
    generate_start_transition,
)
from eden.utils.motion_reference.utils import derive_body_frame_link_fields, estimate_motion_velocities
from eden.utils.torch import to_numpy

if TYPE_CHECKING:
    from eden.entities.rigid import RigidEntity
    from eden.utils.motion_reference.motion_data import BatchMotionSequence


def _seq_to_qpos(motion_seq: "BatchMotionSequence", seq_idx: int) -> np.ndarray:
    """Pack one sequence's ``(base_pos, base_quat, dofs_pos)`` into a qpos array.

    Both the dataset and Genesis use ``dofs_name`` ordering for joints, so no
    permutation is applied. Only the width is checked (``generate_start_transition``
    refuses a different DOF count); the dataset's ``motion_dofs_name`` is never
    compared with the robot's ``dofs_name``, so a reordering silently misassigns joints.
    """
    fields = (motion_seq.base_pos_w, motion_seq.base_quat_w, motion_seq.dofs_pos)
    return np.concatenate([to_numpy(field[seq_idx], np.float64) for field in fields], axis=1)


def _refk_links_genesis(
    robot: "RigidEntity",
    qpos: np.ndarray,
    link_idxs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame link world poses via Genesis FK.

    Walks the qpos array sequentially, calling ``set_qpos`` (which triggers
    FK) and reading back ``get_links_pos`` / ``get_links_quat``, then slicing
    to the requested link indices on the device side before a single
    ``.cpu().numpy()`` transfer. ``len(link_idxs) == 0`` short-circuits to
    empty arrays.
    """
    T = qpos.shape[0]
    num_links = link_idxs.shape[0]
    if num_links == 0:
        return np.zeros((T, 0, 3)), np.zeros((T, 0, 4))

    pos = np.zeros((T, num_links, 3), dtype=np.float32)
    quat = np.zeros((T, num_links, 4), dtype=np.float32)
    # Index on-device with a torch long tensor, then sync to host once per
    # field: the dataset usually carries a subset of the robot's links (14 of
    # G1's 30), so syncing the full ``(n_links, 3/4)`` tensor would move more.
    link_idxs_t = torch.as_tensor(link_idxs, dtype=torch.long, device=robot.device)
    for t in range(T):
        qt = torch.as_tensor(qpos[t], dtype=torch.float32, device=robot.device).unsqueeze(0)
        robot.set_qpos(qt, zero_velocity=False, skip_forward=False)
        pos[t] = robot.get_links_pos(envs_idx=0)[0, link_idxs_t].cpu().numpy()
        quat[t] = robot.get_links_quat(envs_idx=0)[0, link_idxs_t].cpu().numpy()
    return pos, quat


def _link_indices(robot: "RigidEntity", links_name: list[str] | None) -> np.ndarray:
    """Resolve dataset link names → Genesis link indices for ``get_links_pos``.

    Returns an empty int64 array when ``links_name`` is empty — the rest of
    the pipeline short-circuits link work in that case.
    """
    if not links_name:
        return np.zeros(0, dtype=np.int64)
    _, idxs = robot.find_named_links_idx_local(links_name, preserve_order=True)
    return np.asarray(idxs, dtype=np.int64)


@DATASET_MODIFIER_REGISTRY.register()
class SmoothStartTransition(DatasetModifier):
    """Pre-pend each clip with a synthesized stand-up → step → merge → clip segment.

    Parameters
    ----------
    ik_profile : BipedalIKProfile | None
        Robot-agnostic IK spec — see :class:`BipedalIKProfile`. The G1 preset
        lives at ``eden.options.robots.unitree_g1.G1_IK_PROFILE`` and is the
        usual value here. Required at ``apply()`` time; defaults to ``None``
        only so the modifier's ``configure()`` works without it.
    motion_links_name_per_key : dict[str, list[str]] | None
        Map ``key → dataset link name order`` used for Genesis link-index
        lookup when re-FK-ing link fields. ``None`` skips link rebuild —
        useful for datasets that don't carry per-link data.
    default_pose_sec : float
        Duration (s) of stand-still padding at the start. OpenTrack default: 1.0 s.
    single_step_threshold_m : float
        If the inter-foot distance differs between stance and clip frame 0 by
        more than this, two steps are synthesized; otherwise one.
    transition_sec : float
        SLERP-merge length between the IK endpoint and clip frame 0.
    foot_height_m : float
        Peak Z bump of the swing foot during the synthesized step.
    com_xy_offset_m : float
        Forward offset of the IK's CoM target relative to the foot midpoint.
    step_sec : float
        Duration of each synthesized step.
    double_stance_frames : int
        Frames of double-stance dwell between successive steps.
    start_foot : Literal["left", "right"]
        Which foot lifts first.
    max_ik_iters : int
        Per-frame IK iteration cap.
    """

    ik_profile: BipedalIKProfile | None = None
    motion_links_name_per_key: dict[str, list[str]] | None = None
    default_pose_sec: float = 1.0
    single_step_threshold_m: float = 0.1
    # ``step_sec`` and ``transition_sec`` are 0.5 s each rather than OpenTrack's
    # 0.3 s: at 0.3 s the ~31-frame moving window reads as "rushed". The ~50-frame
    # window at 0.5 s, with the cosine easing in ``generate_start_transition``,
    # looks naturally paced.
    transition_sec: float = 0.5
    foot_height_m: float = 0.05
    com_xy_offset_m: float = 0.03
    step_sec: float = 0.5
    double_stance_frames: int = 3
    _breaks_body_model: ClassVar[str | None] = "the frames it prepends have no body-model pose"
    start_foot: str = "right"
    max_ik_iters: int = 200

    def apply(self, motion_sequences):
        if self.ik_profile is None:
            raise ValueError(
                "SmoothStartTransition: ik_profile is required — pass a BipedalIKProfile "
                "(e.g. eden.options.robots.unitree_g1.G1_IK_PROFILE)."
            )
        profile = self.ik_profile

        # The modifier owns one bipedal robot — the one in
        # ``profile.robot_options``. Each key in ``motion_sequences`` is
        # treated as that robot's qpos sequence (laid out in the robot's
        # ``dofs_name`` order). Multi-entity datasets (robot + object) need
        # the user to scope the modifier explicitly via
        # ``motion_sequence_keys=(robot_key,)``; otherwise the link-name
        # lookup would silently misindex against the bipedal scene. The
        # ``single=True`` reject catches the "forgot to scope" case the
        # ``motion_sequence_keys`` filter alone can't.
        keys = self._selected_keys(motion_sequences, single=True)

        # Build a single Genesis env + robot, reuse across every sequence in
        # the key. Genesis env build is heavyweight (kernel compile, GPU
        # buffers); one build per ``apply()`` amortises it across the
        # library.
        with transient_robot_env(profile.robot_options) as (_env, robot):
            out = dict(motion_sequences)
            for key in keys:
                out[key] = self._apply_to_key(out[key], profile, robot, key)
            return out

    def _apply_to_key(
        self,
        motion_seq: "BatchMotionSequence",
        profile: BipedalIKProfile,
        robot: "RigidEntity",
        key: str,
    ) -> "BatchMotionSequence":
        dataset_link_names = (self.motion_links_name_per_key or {}).get(key)
        link_idxs = _link_indices(robot, dataset_link_names)
        has_link_data = link_idxs.shape[0] > 0

        # Per-sequence dt. ``framerate`` is per-sequence in BatchMotionSequence
        # but the dataset asserts it's uniform; use the first.
        framerate = float(motion_seq.framerate[0].item())
        dt = 1.0 / framerate

        # Capture original lengths *before* mutating ``buffer_length`` so the
        # log line at the end can report frames added without dividing by zero.
        original_lengths = motion_seq.buffer_length.tolist()

        # One list per output field; populated per-sequence below, then bulk-
        # rebuilt into ConcatBatchTensors at the bottom. The fields with link
        # data are only filled when the dataset carries link names.
        new_fields: dict[str, list[torch.Tensor]] = {
            "base_pos_w": [],
            "base_quat_w": [],
            "dofs_pos": [],
            "base_lin_vel_w": [],
            "base_ang_vel_w": [],
            "dofs_vel": [],
        }
        if has_link_data:
            new_fields.update(
                {
                    "links_pos_w": [],
                    "links_quat_w": [],
                    "links_lin_vel_w": [],
                    "links_ang_vel_w": [],
                    "links_pos_b": [],
                    "links_quat_b": [],
                    "links_lin_vel_b": [],
                    "links_ang_vel_b": [],
                }
            )

        num_seqs = int(motion_seq.buffer_length.shape[0])
        for i in range(num_seqs):
            qpos_in = _seq_to_qpos(motion_seq, i)
            qpos_out = generate_start_transition(
                robot,
                profile,
                qpos_in,
                framerate=framerate,
                default_pose_sec=self.default_pose_sec,
                single_step_threshold_m=self.single_step_threshold_m,
                transition_sec=self.transition_sec,
                foot_height_m=self.foot_height_m,
                com_xy_offset_m=self.com_xy_offset_m,
                step_sec=self.step_sec,
                double_stance_frames=self.double_stance_frames,
                start_foot=self.start_foot,
                max_ik_iters=self.max_ik_iters,
            )

            qpos_out_f32 = qpos_out.astype(np.float32)
            base_pos = torch.from_numpy(qpos_out_f32[:, :3])
            base_quat = torch.from_numpy(qpos_out_f32[:, 3:7])
            dofs_pos = torch.from_numpy(qpos_out_f32[:, 7:])

            # Re-FK link world-frame poses through Genesis when the dataset
            # carries link data. With no link data, the velocity/body-frame
            # pipeline below is a no-op for the link fields.
            if has_link_data:
                lpw_np, lqw_np = _refk_links_genesis(robot, qpos_out, link_idxs)
                links_pos_w = torch.from_numpy(lpw_np)
                links_quat_w = torch.from_numpy(lqw_np)
            else:
                links_pos_w = None
                links_quat_w = None

            # The canonical velocity estimator — same code path
            # ``DatasetBase._pack_motion_sequence`` uses when source data
            # arrives without velocities.
            base_lin_vel, base_ang_vel, dofs_vel, links_lin_vel_w, links_ang_vel_w = estimate_motion_velocities(
                base_pos,
                base_quat,
                dofs_pos,
                links_pos_w,
                links_quat_w,
                dt,
            )

            new_fields["base_pos_w"].append(base_pos)
            new_fields["base_quat_w"].append(base_quat)
            new_fields["dofs_pos"].append(dofs_pos)
            new_fields["base_lin_vel_w"].append(base_lin_vel)
            new_fields["base_ang_vel_w"].append(base_ang_vel)
            new_fields["dofs_vel"].append(dofs_vel)

            if has_link_data:
                # Body-frame fields go through the canonical Eden helper that
                # ``DatasetBase._pack_motion_sequence`` uses, so the modifier's
                # output is bit-identical to what a fresh dataset prepare
                # would produce.
                links_pos_b, links_quat_b, links_lin_vel_b, links_ang_vel_b = derive_body_frame_link_fields(
                    base_pos_w=base_pos,
                    base_quat_w=base_quat,
                    base_lin_vel_w=base_lin_vel,
                    base_ang_vel_w=base_ang_vel,
                    links_pos_w=links_pos_w,
                    links_quat_w=links_quat_w,
                    links_lin_vel_w=links_lin_vel_w,
                    links_ang_vel_w=links_ang_vel_w,
                )
                new_fields["links_pos_w"].append(links_pos_w)
                new_fields["links_quat_w"].append(links_quat_w)
                new_fields["links_lin_vel_w"].append(links_lin_vel_w)
                new_fields["links_ang_vel_w"].append(links_ang_vel_w)
                new_fields["links_pos_b"].append(links_pos_b)
                new_fields["links_quat_b"].append(links_quat_b)
                new_fields["links_lin_vel_b"].append(links_lin_vel_b)
                new_fields["links_ang_vel_b"].append(links_ang_vel_b)

        new_lengths = [t.shape[0] for t in new_fields["base_pos_w"]]
        device = motion_seq.base_pos_w.device

        def _rebuild(tensors: list[torch.Tensor], ref_cbt: ConcatBatchTensor | None) -> ConcatBatchTensor | None:
            if ref_cbt is None or not tensors:
                return ref_cbt
            return ConcatBatchTensor(
                tensors=[t.to(device=device, dtype=ref_cbt.dtype) for t in tensors],
                device=device,
                dtype=ref_cbt.dtype,
            )

        for name, tensors in new_fields.items():
            setattr(motion_seq, name, _rebuild(tensors, getattr(motion_seq, name)))

        motion_seq.buffer_length = torch.tensor(
            new_lengths,
            device=motion_seq.buffer_length.device,
            dtype=motion_seq.buffer_length.dtype,
        )
        added = [nl - bl for nl, bl in zip(new_lengths, original_lengths)]
        # ``np.mean([])`` returns NaN and ``int(nan)`` raises. The upstream
        # ``_apply_file_filter`` already refuses empty libraries, but a
        # configured modifier filter could in principle leave a key with
        # zero sequences — keep the log line robust either way.
        avg_added = int(np.mean(added)) if added else 0
        en.logger.info(f"SmoothStartTransition: extended {num_seqs} clip(s) of '{key}' by avg {avg_added} frames")
        return motion_seq
