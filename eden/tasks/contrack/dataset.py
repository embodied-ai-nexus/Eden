"""DexterHand xArm7+XHand manipulation clips (ConTrack reference format).

Each ``.h5`` clip stores one xArm7+XHand hand tracking one rigid object:

- ``qpos``: (T, H, 19) joint positions in ``XARM_JOINT_NAMES + hand joints`` order
- ``object_tracks/<i>/{translations (T,3), orientations_xyzw (T,4), vertices, faces}``
- ``contacts/<i>/{is_contact (T,H,15), points (T,H,15,3)}`` with ``points`` in the
  OBJECT-LOCAL frame and 15 finger segments named in ``contacts/segment_names``
- ``fps (H,)``, ``is_rhand (H,)``, ``base_translation (H,3)``, ``urdf_name (H,)``

The clip's 19 qpos columns span two Eden entities — the composite
:class:`~eden.options.robots.composite.xarm_xhand.XArm7XHand1_R` expands into a ``robot``
(arm) entity and a ``hand`` entity welded to its flange — so the dataset emits three motion
sequences per clip, splitting ``qpos`` at the arm/hand boundary:

- ``"robot"``: the 7 xArm7 arm joints (fixed base).
- ``"hand"``: the 12 XHand finger joints, plus demo contact annotations
  (``contact_points_w`` / ``contact_points_valid``) whose link axis follows
  ``motion_links_name["hand"]`` — the 10 XHand finger links tracked by ConTrack
  (proximal segment alone; middle+distal OR-merged).
- ``"object"``: free-floating object base pose track.

Both robot sequences carry the same ``base_pos_w`` (the clip's ``base_translation``,
i.e. the arm base placement) so a consumer that reads a base pose gets a consistent one;
neither is free-floating, so tracking terms never read it.

Only single-hand, single-object clips are supported for now (matching the
released ConTrack data); the loader fails loudly otherwise.

References
----------
- ConTrack: https://github.com/EmptyBlueBox/ConTrack (``mdp/state.py::load_reference``)
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, ClassVar

import h5py
import numpy as np
import torch
from genesis.utils.geom import transform_by_quat

from eden.datasets.base import DATASET_REGISTRY, DatasetBase
from eden.utils.motion_reference.motion_data import BatchMotionSequence

if TYPE_CHECKING:
    from eden.utils.motion_reference.motion_data import MotionSequence

XARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]

# Actuated XHand joints in the clip's qpos column order (after the 7 arm joints).
_HAND_JOINT_SUFFIXES = [
    "thumb_bend_joint",
    "thumb_rota_joint1",
    "thumb_rota_joint2",
    "index_bend_joint",
    "index_joint1",
    "index_joint2",
    "mid_joint1",
    "mid_joint2",
    "ring_joint1",
    "ring_joint2",
    "pinky_joint1",
    "pinky_joint2",
]

# The 10 contact-tracked finger links, in the row order of the emitted
# ``contact_points_*`` fields. Each entry maps a URDF link-name suffix to the
# dataset's 15 finger segments (proximal alone; middle+distal merged — a
# single URDF link spans both annotation segments).
_CONTACT_LINK_SEGMENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("index_rota_link1", ("index_proximal",)),
    ("index_rota_link2", ("index_middle", "index_distal")),
    ("mid_link1", ("middle_proximal",)),
    ("mid_link2", ("middle_middle", "middle_distal")),
    ("ring_link1", ("ring_proximal",)),
    ("ring_link2", ("ring_middle", "ring_distal")),
    ("pinky_link1", ("pinky_proximal",)),
    ("pinky_link2", ("pinky_middle", "pinky_distal")),
    ("thumb_rota_link1", ("thumb_proximal",)),
    ("thumb_rota_link2", ("thumb_middle", "thumb_distal")),
)


def hand_joint_names(side: str) -> list[str]:
    return [f"{side}_hand_{s}" for s in _HAND_JOINT_SUFFIXES]


def contact_link_names(side: str) -> list[str]:
    return [f"{side}_hand_{suffix}" for suffix, _ in _CONTACT_LINK_SEGMENTS]


def _motion_dofs_name(side: str) -> dict[str, list[str]]:
    """Per-entity DOF axes: the clip's 19 qpos columns split at the arm/hand boundary."""
    return {"robot": XARM_JOINT_NAMES, "hand": hand_joint_names(side), "object": []}


def _motion_links_name(side: str) -> dict[str, list[str]]:
    """Per-entity link axes: only the hand carries the contact-annotated links."""
    return {"robot": [], "hand": contact_link_names(side), "object": []}


def _validate_side(side: str) -> str:
    if side not in ("left", "right"):
        raise ValueError(f"DexterHandDataset.side must be 'left' or 'right', got {side!r}.")
    return side


@DATASET_REGISTRY.register()
class DexterHandDataset(DatasetBase):
    """ConTrack xArm7+XHand reference clips (single hand + single object).

    Parameters
    ----------
    side : str
        ``"left"`` or ``"right"``. Determines the URDF joint/link name prefix
        of the emitted motion axes; every clip's ``is_rhand`` flag must match.
    """

    # No ``registry`` / ``dataset``: a *format* reader, not a corpus, so there is no repo to
    # download whole. The task supplies the clip — see ``eden.utils.assets.sample_data_path``.

    # ``side`` is fixed before any file is opened (it selects the name axes ``configure`` bakes
    # into the options), so it cannot be inferred; ``_read_single_sequence`` rejects a clip that
    # disagrees. The default is therefore a claim about the published data — left, because every
    # released ConTrack clip records ``is_rhand=False``, pinned by
    # ``tests/tasks/contrack/test_dataset.py``. The task never relies on it: ``ContrackDataMod``
    # reads ``is_rhand`` and passes ``side`` explicitly.
    side: str = "left"
    # ``side`` selects the joint/link names the emitted axes carry, so it shapes the prepared
    # bytes. The task bakes it into the cache filename too; this covers a hand-written
    # ``processed_data_path`` as well, rather than relying on the caller's naming.
    _cache_identity_fields: ClassVar[tuple[str, ...]] = ("side",)

    file_extension: ClassVar[str] = ".h5"
    motion_sequence_keys: ClassVar[tuple[str, ...]] = ("robot", "hand", "object")
    # Class-level defaults for the declared ``side`` default (left). They are not the
    # source of truth: both axes are side-dependent, so :meth:`configure` derives them
    # from the configured side and ``__init__`` re-derives them from the loaded options.
    motion_dofs_name: ClassVar[dict[str, list[str]]] = _motion_dofs_name("left")
    motion_links_name: ClassVar[dict[str, list[str]]] = _motion_links_name("left")

    @classmethod
    def configure(cls, **kwargs):
        """Bake the side-derived name axes into the options alongside ``side``.

        ``ConfigurableMixin._build_config_dict`` fills every unspecified parameter from the
        *class* attribute, which for these two would be the declared-side table regardless of
        the requested side — so a right-hand config would serialize left-hand joint/link
        names, and anything reading ``dataset_options.motion_dofs_name`` (rather than the
        constructed dataset) would get the wrong hand. Deriving them here keeps the options
        and the instance in agreement on every path, including save -> load.
        """
        side = _validate_side(kwargs.get("side", cls.side))
        kwargs.setdefault("motion_dofs_name", _motion_dofs_name(side))
        kwargs.setdefault("motion_links_name", _motion_links_name(side))
        return super().configure(**kwargs)

    def __init__(self, options, device: torch.device):
        # Validate before super(): _initialize downloads/prepares/caches, and an
        # invalid side would otherwise surface late as a per-file mismatch error.
        _validate_side(getattr(options, "side", None) or type(self).side)
        super().__init__(options, device)
        # Re-derive rather than trust the mirrored options: a hand-built ``DatasetOptions``
        # that set ``side`` without the axes would otherwise leave the class-level tables in
        # place, which are the *declared* side's. ``side`` is the single source of truth, and
        # every name-based consumer (MotionTerm) reads these instance tables after construction.
        self.motion_dofs_name = _motion_dofs_name(self.side)
        self.motion_links_name = _motion_links_name(self.side)

    def prepare(self) -> tuple[dict[str, BatchMotionSequence], dict[str, Any]]:
        all_motion_sequences = self._parallel_read_motion_files(
            self.all_motion_files,
            self._read_single_sequence,
            desc="DexterHand: reading raw .h5",
        )
        # Transpose once: naming the three columns here beats repeating positional unpacking,
        # where a misplaced ``_`` would silently route one entity's data into another.
        arm_seqs, hand_seqs, object_seqs = (list(col) for col in zip(*all_motion_sequences, strict=True))

        arm_sequences = BatchMotionSequence.from_motion_sequences(arm_seqs, num_links=0).to(self.device)
        arm_sequences.is_free_floating = False
        arm_sequences.entity_kind = "robot"
        hand_sequences = BatchMotionSequence.from_motion_sequences(
            hand_seqs,
            num_links=len(self.motion_links_name["hand"]),
        ).to(self.device)
        hand_sequences.is_free_floating = False
        hand_sequences.entity_kind = "hand"
        object_sequences = BatchMotionSequence.from_motion_sequences(object_seqs, num_dofs=0, num_links=0).to(
            self.device
        )
        object_sequences.entity_kind = "rigid_object"

        return {"robot": arm_sequences, "hand": hand_sequences, "object": object_sequences}, {}

    def _read_single_sequence(self, file_path: str) -> tuple[MotionSequence, MotionSequence, MotionSequence]:
        with h5py.File(file_path, "r") as f:
            qpos = np.asarray(f["qpos"], np.float32)  # (T, H, 19)
            fps = float(np.max(np.asarray(f["fps"], np.float32)))
            is_rhand = np.asarray(f["is_rhand"], np.uint8).astype(bool)
            base_translation = np.asarray(f["base_translation"], np.float32)  # (H, 3)

            if qpos.shape[1] != 1:
                raise ValueError(
                    f"{file_path}: found {qpos.shape[1]} hands; only single-hand clips are supported for now."
                )
            file_side = "right" if bool(is_rhand[0]) else "left"
            if file_side != self.side:
                raise ValueError(
                    f"{file_path}: clip is a {file_side}-hand recording but the dataset is configured "
                    f"with side={self.side!r}. Configure DexterHandDataset with side={file_side!r}."
                )

            obj_keys = sorted((k for k in f["object_tracks"].keys()), key=int)
            if len(obj_keys) != 1:
                raise ValueError(
                    f"{file_path}: found {len(obj_keys)} object tracks; only single-object clips are supported for now."
                )
            obj = f["object_tracks"][obj_keys[0]]
            obj_pos = torch.from_numpy(np.asarray(obj["translations"], np.float32))  # (T, 3)
            obj_quat_xyzw = np.asarray(obj["orientations_xyzw"], np.float32)  # (T, 4)
            obj_quat = torch.from_numpy(obj_quat_xyzw[:, [3, 0, 1, 2]].copy())  # wxyz

            contact_valid, contact_points_local = self._merge_contact_segments(f, obj_keys[0])

        # The clip's qpos concatenates the arm and the hand; the composite rig puts them on
        # two entities, so split at the boundary. Column order within each half is preserved,
        # and each half is packed separately so velocities are finite-differenced per entity.
        num_arm = len(XARM_JOINT_NAMES)

        def fresh_base_pose() -> tuple[torch.Tensor, torch.Tensor]:
            """A new ``(base_pos_w, base_quat_w)`` pair holding the clip's arm mount pose.

            Called once per sequence rather than shared, and that is the point: when no
            retiming is needed (``fps == target_framerate``) ``_pack_motion_sequence`` stores
            the tensors it is handed *by reference*, so passing one pair to both packs would
            leave the arm and hand sequences aliasing a single buffer.
            """
            return (
                torch.from_numpy(np.broadcast_to(base_translation[0], (qpos.shape[0], 3)).copy()),
                torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(qpos.shape[0], 4).clone(),
            )

        arm_pos_w, arm_quat_w = fresh_base_pose()
        arm_seq = self._pack_motion_sequence(
            framerate=fps,
            base_pos_w=arm_pos_w,
            base_quat_w=arm_quat_w,
            dofs_pos=torch.from_numpy(qpos[:, 0, :num_arm]),
        )
        # Pack with OBJECT-LOCAL contact points first: the pack retimes contact
        # fields with nearest-neighbor while poses are lerped/slerped, so
        # composing to world BEFORE retiming would pair NN-sampled points with
        # a blended object pose — a systematic demo-target error at reward
        # time. Composing with the RETIMED object pose (below) keeps the
        # world point exactly consistent with the demo anchor consumers use
        # to re-localize it (MotionContactDistanceReward, chamfer frame="object").
        hand_pos_w, hand_quat_w = fresh_base_pose()
        hand_seq = self._pack_motion_sequence(
            framerate=fps,
            base_pos_w=hand_pos_w,
            base_quat_w=hand_quat_w,
            dofs_pos=torch.from_numpy(qpos[:, 0, num_arm:]),
            contact_points_w=contact_points_local.unsqueeze(2),  # (T, L, 1, 3), object-local until composed below
            contact_points_valid=contact_valid.unsqueeze(2),  # (T, L, 1)
        )
        object_seq = self._pack_motion_sequence(
            framerate=fps,
            base_pos_w=obj_pos,
            base_quat_w=obj_quat,
        )
        num_frames, num_links = hand_seq.contact_points_w.shape[:2]
        assert arm_seq.dofs_pos.shape[0] == num_frames, "arm/hand retiming disagreed on frame count"
        assert object_seq.base_pos_w.shape[0] == num_frames, "hand/object retiming disagreed on frame count"
        hand_seq.contact_points_w = transform_by_quat(
            hand_seq.contact_points_w,
            object_seq.base_quat_w.view(num_frames, 1, 1, 4).expand(-1, num_links, 1, -1),
        ) + object_seq.base_pos_w.view(num_frames, 1, 1, 3)
        return arm_seq, hand_seq, object_seq

    @staticmethod
    def _merge_contact_segments(f: h5py.File, obj_key: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Merge the 15 annotated finger segments into the 10 tracked links.

        Returns ``(contact_valid (T, 10) bool, contact_points (T, 10, 3) object-local)``.
        Merged links OR the segment flags and average the contact points weighted
        by the per-segment contact flag (mirrors ConTrack ``load_reference``).
        """
        seg_names = [s.decode("utf-8") for s in np.asarray(f["contacts/segment_names"])]
        seg_idx = {n: i for i, n in enumerate(seg_names)}
        expected = {segment for _, segments in _CONTACT_LINK_SEGMENTS for segment in segments}
        if missing := sorted(expected - seg_idx.keys()):
            raise ValueError(
                f"{f.filename}: contacts/segment_names is missing {missing}. The loader merges the "
                f"15 annotated finger segments into the 10 tracked XHand links, so every name in "
                f"_CONTACT_LINK_SEGMENTS must be present. Found: {seg_names}"
            )
        is_contact = np.asarray(f["contacts"][obj_key]["is_contact"], np.uint8)[:, 0]  # (T, 15)
        points = np.nan_to_num(np.asarray(f["contacts"][obj_key]["points"], np.float32)[:, 0], nan=0.0)  # (T, 15, 3)

        valid_cols = []
        point_cols = []
        for _, segments in _CONTACT_LINK_SEGMENTS:
            ids = [seg_idx[s] for s in segments]
            flags = is_contact[:, ids].astype(np.float32)  # (T, S)
            valid_cols.append(flags.any(axis=1))
            weighted = (points[:, ids] * flags[..., None]).sum(axis=1)
            count = np.maximum(flags.sum(axis=1), 1.0)
            point_cols.append(weighted / count[:, None])
        contact_valid = torch.from_numpy(np.stack(valid_cols, axis=1))  # (T, 10) bool
        contact_points = torch.from_numpy(np.stack(point_cols, axis=1).astype(np.float32))  # (T, 10, 3)
        return contact_valid, contact_points
