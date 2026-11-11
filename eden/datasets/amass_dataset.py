"""AMASS retargeted-to-G1 motion dataset."""

from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar, Any

import numpy as np

from eden.datasets.base import DatasetBase, DATASET_REGISTRY
from eden.options.robots.unitree_g1 import G1_DOFS29_NAME, G1_DOFS29_TRACKED_LINKS_NAME
from eden.utils.motion_reference.motion_data import BatchMotionSequence
from eden.utils.string import columns_by_name

if TYPE_CHECKING:
    from eden.utils.motion_reference.motion_data import MotionSequence


@DATASET_REGISTRY.register()
class AMASSRetargetedG1Dataset(DatasetBase):
    """Dataset class for the AMASS retargeted G1 dataset."""

    _applies_frame_window: ClassVar[bool] = True
    # Caches without the frame-window marker cut drop_first_n_frames correctly (a long drop_last_n_frames could wrap).
    _drops_cut_before_marker: ClassVar[tuple[str, ...]] = ("drop_first_n_frames",)

    registry: str = "ember-lab-berkeley"
    dataset: str = "AMASS_Retargeted_for_G1"

    file_extension: ClassVar[str] = ".npz"
    motion_sequence_keys: ClassVar[tuple[str, ...]] = ("g1",)
    motion_dofs_name: ClassVar[dict[str, list[str]]] = {"g1": G1_DOFS29_NAME}
    motion_links_name: ClassVar[dict[str, list[str]]] = {"g1": G1_DOFS29_TRACKED_LINKS_NAME}

    def prepare(self) -> tuple[dict[str, BatchMotionSequence], dict[str, Any]]:
        # No forward kinematics: the AMASS retarget already holds body positions and rotations (IsaacLab AMP
        # format). Velocities and base-frame transforms are computed by _pack_motion_sequence.
        return self._prepare_single_key(self._read_single_sequence, desc="AMASS: reading raw .npz")

    def _read_single_sequence(self, file_path: str) -> MotionSequence:
        # Tensors stay on CPU; prepare() does one bulk move to self.device after
        # packing — avoids 10k+ tiny H2D copies under the parallel read.
        raw = np.load(file_path, allow_pickle=True)
        # Support both pickled dict (np.save as .npy) and standard npz (np.savez)
        if isinstance(raw, np.ndarray) and raw.ndim == 0:
            data = raw.item()
        else:
            data = dict(raw)

        fps = float(data["fps"])
        window = self._frame_window()
        dofs_pos = np.asarray(data["dof_positions"])[window]
        dofs_vel = np.asarray(data["dof_velocities"])[window]
        links_pos_w = np.asarray(data["body_positions"])[window]
        links_quat_w = np.asarray(data["body_rotations"])[window]
        links_lin_vel_w = np.asarray(data["body_linear_velocities"])[window]
        links_ang_vel_w = np.asarray(data["body_angular_velocities"])[window]

        # Reorder DOFs / bodies to match motion_{dofs,links}_name ordering when the npz carries
        # its own name labels — makes the mapping independent of the writer's column order.
        if "dof_names" in data:
            dof_idx = columns_by_name(data["dof_names"], self.motion_dofs_name["g1"], what="motion npz 'dof_names'")
            dofs_pos = dofs_pos[:, dof_idx]
            dofs_vel = dofs_vel[:, dof_idx]

        if "body_names" in data:
            body_idx = columns_by_name(data["body_names"], self.motion_links_name["g1"], what="motion npz 'body_names'")
            links_pos_w = links_pos_w[:, body_idx]
            links_quat_w = links_quat_w[:, body_idx]
            links_lin_vel_w = links_lin_vel_w[:, body_idx]
            links_ang_vel_w = links_ang_vel_w[:, body_idx]

        # Root body (pelvis) is the first link in motion_links_name; its rotation is already wxyz (IsaacLab convention).
        return self._pack_motion_sequence(
            framerate=fps,
            base_pos_w=links_pos_w[:, 0],
            base_quat_w=links_quat_w[:, 0],
            dofs_pos=dofs_pos,
            links_pos_w=links_pos_w,
            links_quat_w=links_quat_w,
            base_lin_vel_w=links_lin_vel_w[:, 0],
            base_ang_vel_w=links_ang_vel_w[:, 0],
            dofs_vel=dofs_vel,
            links_lin_vel_w=links_lin_vel_w,
            links_ang_vel_w=links_ang_vel_w,
        )
