"""LAFAN1 retargeted-to-G1 locomotion/dance dataset."""

from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar, Any

import numpy as np

import eden as en
from eden.datasets.base import DatasetBase, DATASET_REGISTRY
from eden.options.robots.unitree_g1 import G1_DOFS29_NAME, G1_DOFS29_TRACKED_LINKS_NAME
from eden.utils.geom import xyzw_to_wxyz
from eden.utils.motion_reference.motion_data import BatchMotionSequence

if TYPE_CHECKING:
    from eden.utils.motion_reference.motion_data import MotionSequence


# TODO: implement the LAFAN1 original dataset (e.g., a DatasetBase subclass for human motions)
@DATASET_REGISTRY.register()
class LAFAN1RetargetedG1Dataset(DatasetBase):
    """Dataset class for the LAFAN1 retargeted G1 dataset."""

    _applies_frame_window: ClassVar[bool] = True
    _drops_cut_before_marker: ClassVar[tuple[str, ...]] = ("drop_first_n_frames", "drop_last_n_frames")

    registry: str = "lvhaidong"
    dataset: str = "LAFAN1_Retargeting_Dataset"
    # The lvhaidong repo ships g1/, h1/, and h1_2/ retargets side-by-side with
    # different DOF counts (29 / 19 / 27), plus a robot_description/ tree of
    # URDFs and meshes this loader never reads. Fetch only the g1 CSVs (plus
    # the licence and readme that must travel with them) and hand downstream
    # globbing the g1/ directory rather than the snapshot root: mixing row
    # widths makes the ``torch.cat`` over per-motion ``dofs_pos`` tensors in
    # ``BatchMotionSequence.from_motion_sequences`` raise a shape mismatch.
    # A retarget for another robot is a subclass declaring its own two layout
    # attributes plus the DOF/link names that differ with the robot
    # (motion_dofs_name, motion_links_name).
    _hf_allow_patterns: ClassVar[tuple[str, ...]] = ("g1/*", "LICENSE", "README.md")
    _hf_data_subdir: ClassVar[str] = "g1"

    file_extension: ClassVar[str] = ".csv"
    motion_sequence_keys: ClassVar[tuple[str, ...]] = ("g1",)
    motion_dofs_name: ClassVar[dict[str, list[str]]] = {"g1": G1_DOFS29_NAME}
    motion_links_name: ClassVar[dict[str, list[str]]] = {"g1": G1_DOFS29_TRACKED_LINKS_NAME}

    def prepare(self) -> tuple[dict[str, BatchMotionSequence], dict[str, Any]]:
        """Prepare the LAFAN1 retargeting dataset; the CSVs carry base + DOFs only, so links come from FK."""
        return self._prepare_single_key(
            self._read_single_sequence,
            desc="LAFAN1: reading raw csv",
            robot=en.robots.G1Dofs29(dofs_name=self.motion_dofs_name["g1"]),
        )

    def _read_single_sequence(self, file_path: str) -> MotionSequence:
        # Packed into CPU tensors; prepare() bulk-moves the packed batch.
        motion = np.loadtxt(file_path, delimiter=",")[self._frame_window()]
        return self._pack_motion_sequence(
            framerate=30.0,
            base_pos_w=motion[:, :3],
            base_quat_w=xyzw_to_wxyz(motion[:, 3:7]),
            dofs_pos=motion[:, 7:],
        )
