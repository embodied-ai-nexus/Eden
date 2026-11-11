"""SEED retargeted-to-G1 motion dataset."""

from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar, Any

import os

import numpy as np
import torch

import eden as en
from eden.datasets.base import DatasetBase, DATASET_REGISTRY
from eden.options.robots.unitree_g1 import G1_DOFS29_NAME, G1_DOFS29_TRACKED_LINKS_NAME
from eden.utils.motion_reference.motion_data import BatchMotionSequence

if TYPE_CHECKING:
    from eden.utils.motion_reference.motion_data import MotionSequence


@DATASET_REGISTRY.register()
class SEEDRetargetedG1Dataset(DatasetBase):
    """Dataset class for the SEED motion library retargeted to the Unitree G1.

    https://huggingface.co/datasets/bones-studio/seed

    SEED ships three splits (SOMA Uniform BVH, SOMA Proportional BVH, and the
    G1 MuJoCo-compatible CSV split). This loader only consumes the G1 CSV
    split, distributed inside ``g1.tar.gz`` (~23 GB).

    Notes
    -----
    The HuggingFace repo is gated; users must request access on HuggingFace
    and have a token configured (``huggingface-cli login``) before the
    automatic download will succeed.

    CSV layout (one file per motion, header row + one row per frame at 120 fps):
        ``Frame, root_translate{X,Y,Z}, root_rotate{X,Y,Z}, <29 DOFs>``

    where translation is in centimetres and the DOF columns are joint angles
    in degrees in the same order as ``motion_dofs_name['g1']`` below. The
    root rotation columns are XYZ Euler magnitudes in degrees, but the
    rotation is composed in **intrinsic ZYX** order (R = Rz · Ry · Rx) — the
    Maya export convention. Reading them as intrinsic XYZ produces a
    visually plausible standing pose at frame 0 but lifts one foot off the
    ground mid-clip, while ZYX keeps the feet planted throughout.
    """

    _applies_frame_window: ClassVar[bool] = True
    _drops_cut_before_marker: ClassVar[tuple[str, ...]] = ("drop_first_n_frames", "drop_last_n_frames")

    registry: str = "bones-studio"
    dataset: str = "seed"
    # Fetch only the G1 split (~23 GB) plus licence and readme. The repo also
    # ships soma_uniform.tar.gz and soma_proportional.tar.gz (~45 GB each) and
    # the SOMA shape parameters, none of which this loader reads.
    _hf_allow_patterns: ClassVar[tuple[str, ...]] = ("g1.tar.gz", "LICENSE.md", "README.md")
    # g1.tar.gz unpacks to ``g1/csv/{date}/*.csv``.
    _hf_data_subdir: ClassVar[str] = "g1/csv"

    file_extension: ClassVar[str] = ".csv"
    motion_sequence_keys: ClassVar[tuple[str, ...]] = ("g1",)
    motion_dofs_name: ClassVar[dict[str, list[str]]] = {"g1": G1_DOFS29_NAME}
    motion_links_name: ClassVar[dict[str, list[str]]] = {"g1": G1_DOFS29_TRACKED_LINKS_NAME}

    def _post_download_process(self, local_dir: str) -> str:
        # Reruns skip the (slow, 23 GB) untar: ``DatasetBase._extract_once`` marks a finished one inside ``g1/``, so
        # an interrupted untar is redone and deleting ``g1/`` re-arms it. Returns the extraction root; the base class
        # narrows it to ``_hf_data_subdir``.
        self._extract_once(local_dir, "g1.tar.gz", "g1")
        return local_dir

    def _get_all_motion_files(self, data_path: str, extension: str = ".csv") -> list[str]:
        # Allow ``raw_data_path`` to point at any of: the parent directory that
        # contains the raw ``g1.tar.gz`` (e.g. after a manual `huggingface-cli
        # download`), the extracted ``g1`` root, or the ``g1/csv`` directory.
        # Without this the base-class glob would silently return zero files
        # when the user points at the parent directory. This is explicit-path
        # tolerance, which a single ``_hf_data_subdir`` cannot express.
        if os.path.isdir(data_path):
            if os.path.exists(os.path.join(data_path, "g1.tar.gz")):
                data_path = self._resolve_data_subdir(self._post_download_process(data_path))
            elif os.path.isdir(os.path.join(data_path, "csv")):
                data_path = os.path.join(data_path, "csv")
        return super()._get_all_motion_files(data_path, extension)

    def prepare(self) -> tuple[dict[str, BatchMotionSequence], dict[str, Any]]:
        """Prepare the SEED retargeted G1 dataset; the CSVs carry base + DOFs only, so links come from FK."""
        return self._prepare_single_key(
            self._read_single_sequence,
            desc="SEED: reading raw csv",
            robot=en.robots.G1Dofs29(dofs_name=self.motion_dofs_name["g1"]),
        )

    def _read_single_sequence(self, file_path: str) -> MotionSequence:
        from genesis.utils.geom import xyz_to_quat

        # Read into CPU tensor; prepare() bulk-moves the packed batch.
        # skiprows=1 drops the column-name header; usecols drops the leading
        # frame-index column so column 0 of the loaded array is root_translateX.
        raw = np.loadtxt(
            file_path,
            delimiter=",",
            skiprows=1,
            dtype=np.float32,
            usecols=tuple(range(1, 36)),
        )
        motion = torch.from_numpy(raw[self._frame_window()])
        # cm -> m for translation.
        base_pos_w = motion[:, 0:3] * 0.01
        # Euler degrees -> wxyz quaternion. SEED uses Maya's intrinsic ZYX
        # composition (R = Rz · Ry · Rx) for the root_rotate{X,Y,Z} columns,
        # which is `rpy=True` in genesis. Using `rpy=False` (intrinsic XYZ)
        # silently breaks foot contact mid-clip.
        base_quat_w = xyz_to_quat(motion[:, 3:6], rpy=True, degrees=True)
        # 29 DOFs, degrees -> radians.
        dofs_pos = torch.deg2rad(motion[:, 6:35])

        return self._pack_motion_sequence(
            framerate=120.0,
            base_pos_w=base_pos_w,
            base_quat_w=base_quat_w,
            dofs_pos=dofs_pos,
        )
