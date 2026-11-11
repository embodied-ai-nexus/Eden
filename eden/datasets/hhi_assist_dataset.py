"""HHI-Assist human-human interaction assistance dataset."""

from __future__ import annotations

import os
import re
from typing import ClassVar, Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import eden as en
from eden.datasets.base import DatasetBase, DATASET_REGISTRY
from eden.datasets.utils.bvh import parse_bvh, forward_kinematics
from eden.utils.geom import wxyz_to_xyzw, xyzw_to_wxyz
from eden.utils.motion_reference.motion_data import BatchMotionSequence, MotionSequence
from eden.utils.string import columns_by_name


HHI_JOINT_NAMES = [
    "Hips",
    "Spine",
    "Spine1",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
]


@DATASET_REGISTRY.register()
class HHIAssistDataset(DatasetBase):
    """
    Dataset class for the HHI-Assist dataset.

    The HHI-Assist dataset contains human-human interaction motion capture data
    recorded with an Optitrack system at 120 Hz. Each interaction consists of
    a caregiver (CG) and a care recipient (CR), captured as paired BVH files.

    References
    ----------
    - https://github.com/vita-epfl/HHI-Assist/
    - https://huggingface.co/datasets/jose-barreiros-tri/hhi-assist
    """

    _applies_frame_window: ClassVar[bool] = True
    # Caches without the frame-window marker cut drop_first_n_frames correctly (a long drop_last_n_frames could wrap).
    _drops_cut_before_marker: ClassVar[tuple[str, ...]] = ("drop_first_n_frames",)

    registry: str = "jose-barreiros-tri"
    dataset: str = "hhi-assist"

    unit_scale: float = 1.0

    file_extension: ClassVar[str] = ".bvh"
    motion_sequence_keys: ClassVar[tuple[str, ...]] = ("caregiver", "recipient")
    motion_dofs_name: ClassVar[dict[str, list[str]]] = {
        "caregiver": [],
        "recipient": [],
    }
    motion_links_name: ClassVar[dict[str, list[str]]] = {
        "caregiver": HHI_JOINT_NAMES,
        "recipient": HHI_JOINT_NAMES,
    }

    def prepare(self) -> tuple[dict[str, BatchMotionSequence], dict[str, Any]]:
        pairs = self._pair_files()
        if not pairs:
            raise ValueError(
                f"No valid caregiver/recipient BVH pairs found in {self.raw_data_path}. "
                "Expected paired files with (CG)/(CR) role annotations in filenames."
            )

        en.logger.info(f"Found {len(pairs)} caregiver-recipient pairs")

        # Update all_motion_files to match pair count (one entry per pair)
        self.all_motion_files = [cg for cg, _ in pairs]

        # Parallel paired CPU read; helper takes a single-arg fn so we adapt
        # via tuple. Each worker parses two BVHs (caregiver + recipient).
        results = self._parallel_read_motion_files(
            list(pairs),
            lambda pair: self._read_paired_sequence(*pair),
            desc="HHI-Assist: parsing BVH pairs",
        )
        cg_sequences: list[MotionSequence] = [cg for cg, _ in results]
        cr_sequences: list[MotionSequence] = [cr for _, cr in results]

        data = {
            "caregiver": BatchMotionSequence.from_motion_sequences(
                cg_sequences,
                num_links=len(self.motion_links_name["caregiver"]),
            ).to(self.device),
            "recipient": BatchMotionSequence.from_motion_sequences(
                cr_sequences,
                num_links=len(self.motion_links_name["recipient"]),
            ).to(self.device),
        }

        metadata = {
            "pairs": [(os.path.basename(cg), os.path.basename(cr)) for cg, cr in pairs],
        }
        return data, metadata

    def _pair_files(self) -> list[tuple[str, str]]:
        """
        Pair caregiver and recipient BVH files based on filename conventions.

        HHI-Assist BVH filenames follow the pattern::

            XX(CG)_YY(CR)_<Activity>_<Take N> [<Date>]_[_*]<TailID>_<N>.bvh

        where XX is the caregiver ID, YY is the care recipient ID (both 2-3
        uppercase letters), and TailID identifies which person this file
        captures — either directly (e.g. ``_AA_50.bvh``) or via an annotated
        form (e.g. ``_P5(RM)_50.bvh`` where ``RM`` is the person ID).

        Returns
        -------
        list of (caregiver_path, recipient_path) tuples
        """
        # Regex for the role header: XX(CG)_YY(CR) or XX(CR)_YY(CG)
        # Handles optional spaces (Task3) and either role ordering
        header_re = re.compile(r"([A-Z]{2,3})\s*\((CG|CR)\)_([A-Z]{2,3})\s*\((CG|CR)\)")
        # Regex for the tail person identifier + marker count
        # Matches _AA_50.bvh, __HKG_50.bvh, _P5(RM)_50.bvh, _P1(SR) _50.bvh
        tail_re = re.compile(r"_+([A-Za-z0-9]+(?:\([A-Z]{2,3}\))?)\s*_\d+\.bvh$")

        groups: dict[str, dict[str, str]] = {}  # interaction_key -> {role -> path}
        for filepath in self.all_motion_files:
            filename = os.path.basename(filepath)

            header = header_re.search(filename)
            if not header:
                en.logger.warning(f"Skipping file without CG/CR header: {filename}")
                continue
            # Build person-to-role mapping from header (order-agnostic)
            id_to_role = {
                header.group(1): header.group(2),  # first person -> their role
                header.group(3): header.group(4),  # second person -> their role
            }
            cg_id = next((k for k, v in id_to_role.items() if v == "CG"), None)
            cr_id = next((k for k, v in id_to_role.items() if v == "CR"), None)
            if not cg_id or not cr_id:
                en.logger.warning(f"Skipping file with duplicate roles: {filename}")
                continue

            tail = tail_re.search(filename)
            if not tail:
                en.logger.warning(f"Skipping file with unexpected tail: {filename}")
                continue

            # Extract person ID from tail (handle annotated form like P5(RM) → RM)
            tail_raw = tail.group(1)
            inner = re.match(r"[A-Za-z0-9]+\(([A-Z]{2,3})\)", tail_raw)
            tail_id = inner.group(1) if inner else tail_raw

            # Determine role from tail ID
            if tail_id == cg_id:
                role = "CG"
            elif tail_id == cr_id:
                role = "CR"
            else:
                en.logger.warning(f"Tail ID '{tail_id}' doesn't match CG={cg_id} or CR={cr_id}: {filename}")
                continue

            interaction_key = filename[: tail.start()].rstrip("_")
            if interaction_key not in groups:
                groups[interaction_key] = {}
            groups[interaction_key][role] = filepath

        pairs: list[tuple[str, str]] = []
        for key, files in groups.items():
            if "CG" in files and "CR" in files:
                pairs.append((files["CG"], files["CR"]))
            else:
                present = "CG" if "CG" in files else "CR"
                missing = "CR" if present == "CG" else "CG"
                en.logger.warning(f"Missing {missing} file for interaction '{key}'")

        return sorted(pairs, key=lambda p: os.path.basename(p[0]))

    def _read_paired_sequence(self, cg_path: str, cr_path: str) -> tuple[MotionSequence, MotionSequence]:
        cg_seq = self._read_single_bvh(cg_path)
        cr_seq = self._read_single_bvh(cr_path)

        # Truncate to the shorter sequence so buffer_lengths match across keys
        cg_len = int(cg_seq.buffer_length)
        cr_len = int(cr_seq.buffer_length)
        if cg_len != cr_len:
            min_len = min(cg_len, cr_len)
            en.logger.warning(f"Truncating paired sequences to {min_len} frames (CG={cg_len}, CR={cr_len})")
            cg_seq = self._truncate_sequence(cg_seq, min_len)
            cr_seq = self._truncate_sequence(cr_seq, min_len)

        return cg_seq, cr_seq

    @staticmethod
    def _truncate_sequence(seq: MotionSequence, length: int) -> MotionSequence:
        kwargs: dict[str, torch.Tensor | None] = {}
        for attr in seq.attrs_with_frame_dim:
            val = getattr(seq, attr)
            kwargs[attr] = val[:length] if val is not None else None
        for attr in seq.attrs_only_batch_dim:
            val = getattr(seq, attr)
            if attr == "buffer_length":
                kwargs[attr] = torch.tensor(length, device=val.device)
            else:
                kwargs[attr] = val
        return MotionSequence(**kwargs)

    def _read_single_bvh(self, filepath: str) -> MotionSequence:
        bvh = parse_bvh(filepath)
        positions, quaternions = forward_kinematics(bvh)

        # Keep the frames drop_first_n_frames / drop_last_n_frames leave, and reorder joints to match HHI_JOINT_NAMES
        window = self._frame_window()
        joint_order = columns_by_name(bvh.joint_names, HHI_JOINT_NAMES, what=f"BVH {filepath!r} joints")
        positions = positions[window, joint_order]
        quaternions = quaternions[window, joint_order]

        # Convert from BVH Y-up to Eden Z-up coordinate frame
        # BVH: X=left, Y=up, Z=front  →  Eden: X=front, Y=left, Z=up
        positions = positions[:, :, [2, 0, 1]]

        # Transform quaternions: q_eden = q_T * q_bvh * q_T_inv
        # T = [[0,0,1],[1,0,0],[0,1,0]]  (the axis permutation matrix)
        frame_rot = Rotation.from_matrix(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64))
        n_frames, n_joints = quaternions.shape[:2]
        flat_xyzw = wxyz_to_xyzw(quaternions).reshape(-1, 4)
        transformed = frame_rot * Rotation.from_quat(flat_xyzw) * frame_rot.inv()
        quaternions = xyzw_to_wxyz(transformed.as_quat()).reshape(n_frames, n_joints, 4)

        # Apply unit scale. Packing converts to CPU float32 tensors; prepare() bulk-moves the batch.
        links_pos_w = positions * self.unit_scale  # (num_frames, num_joints, 3)
        links_quat_w = quaternions  # (num_frames, num_joints, 4) wxyz

        # Root (Hips) is the first joint after reordering
        return self._pack_motion_sequence(
            framerate=bvh.framerate,
            base_pos_w=links_pos_w[:, 0],
            base_quat_w=links_quat_w[:, 0],
            links_pos_w=links_pos_w,
            links_quat_w=links_quat_w,
        )
