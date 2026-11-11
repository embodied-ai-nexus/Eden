"""HOT3D (Banerjee et al., CVPR 2025): egocentric hand-object recordings, cut into hand-object segments."""

from __future__ import annotations
from typing import ClassVar, Literal, NamedTuple

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from eden.datasets.base import DATASET_REGISTRY
from eden.datasets.hand_object_base import HandObjectClip, HandObjectDatasetBase, MANOHand, rotate_poses
from eden.datasets.utils.body_model import (
    CHORD_TIP_VERTICES,
    mano_pose_from_release,
    mano_root_offset,
    mano_vertex_links,
)


class Recording(NamedTuple):
    """A HOT3D recording's hands, every object's (T, 4, 4) pose, and the object ids, in the world frame; ``gaps`` are
    the frames ``k`` after which frames are missing (a hand or an object lacked a pose), so ``k`` and ``k + 1`` are
    more than one frame apart; ``stamps`` are the frames' release timestamps (ns).
    """

    hands: dict[str, MANOHand]
    poses: dict[str, np.ndarray]
    uids: list[str]
    gaps: np.ndarray
    stamps: list[int]


#: Quest3's y-up world turned z-up.
_QUEST3_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)


@DATASET_REGISTRY.register()
class Hot3dDataset(HandObjectDatasetBase):
    """HOT3D, read as CHORD's ``hot3d_loader.py`` and cut into hand-object segments as its ``segment_to_atomic.py``.

    Layout::

        <root>/dataset/<recording>/{mano_hand_pose_trajectory.jsonl, dynamic_objects.csv,
                                    headset_trajectory.csv, metadata.json}
        <root>/_assets/assets/<uid>.glb

    Recordings with hand and object ground truth; one timestamp per 10 ms, 30 Hz; the frames with both hands and
    every object (a segment across a missing frame is skipped). Quest3's y-up world is turned z-up, then yawed so the
    workspace (the mean of every object position and hand ``transl``) lies along +y from the headset's first position,
    turned 90 degrees further (CHORD's frame). Hands are MANO with 15 PCA components relative to the mean hand.
    ``drop_first_n_frames`` / ``drop_last_n_frames`` trim a segment after its ``pad_frames`` are added, so they take
    the still padding first.

    Segments (clip ``<recording>_segNNN``, CHORD's numbering): a hand is active while a fingertip is within
    ``tip_contact`` of an object (gaps under ``gap_frames`` bridged); windows of activity split where the touched
    object changes (each frame's object: the one most of both hands' 16 MANO links are nearest to, within
    ``link_contact``); segments under ``min_frames`` dropped, ``pad_frames`` still frames added at both ends. Mode A:
    one hand active, B: both on one object, C: on two objects; the segments of ``modes`` are kept. In mode B both hands
    hold the object; in A only the active hand holds one (so ``sides='both'`` skips the segment, its other hand
    holding nothing), and in C each hand holds its own. FlashDexRetarget hands every hand the segment's one object,
    which labels an idle hand (A) or the other hand's object (C) as held. :meth:`recording_object_poses` gives every
    object over the recording. Needs the MANO model files (``body_model_dir``).

    Parameters
    ----------
    tip_contact : float
        Fingertip-object distance (m) under which a hand is active.
    link_contact : float
        Vertex-object distance (m) under which a MANO link touches an object.
    gap_frames : int
        Inactive gaps shorter than this are bridged.
    min_frames : int
        Shorter segments are dropped.
    pad_frames : int
        Still frames repeated at each end of a segment.
    modes : tuple[str, ...]
        Segment modes kept (``A``, ``B``, ``C``).
    """

    tip_contact: float = 0.02
    link_contact: float = 0.01
    gap_frames: int = 15
    min_frames: int = 30
    pad_frames: int = 10
    modes: tuple[Literal["A", "B", "C"], ...] = ("B",)

    source_framerate: ClassVar[float] = 30.0
    has_recording_context: ClassVar[bool] = True
    _cache_identity_fields: ClassVar[tuple[str, ...]] = (
        "tip_contact",
        "link_contact",
        "gap_frames",
        "min_frames",
        "pad_frames",
        "modes",
    )

    def units(self, root: Path) -> list[str]:
        missing = [d for d in ("dataset", "_assets/assets") if not (root / d).is_dir()]
        if missing:
            raise FileNotFoundError(
                f"{root} is not a HOT3D release: {missing} missing. Point raw_data_path at the directory holding "
                "dataset/<recording>/ and _assets/assets/."
            )
        out = []
        for p in sorted((root / "dataset").iterdir()):
            if not (p / "mano_hand_pose_trajectory.jsonl").exists():
                continue
            with open(p / "metadata.json") as f:
                if json.load(f).get("have_hand_object_pose_gt"):
                    out.append(str(p))
        return out

    def read_unit(self, unit: str) -> list[HandObjectClip | tuple[str, str]]:
        rec = Path(unit)
        recording = self._recording(rec)
        if isinstance(recording, str):
            return [(rec.name, recording)]
        hands, poses = recording.hands, recording.poses
        verts = {}
        for side, hand in hands.items():  # the vertices segment the recording, then go: only the keypoints stay
            joints, verts[side] = hand.forward(self.body_model_dir)
            hand.keep_keypoints(joints, verts[side])
        out: list[HandObjectClip | tuple[str, str]] = []
        for i, (a, b, mode, held) in enumerate(self._segments(verts, poses, recording.uids)):
            name = f"{rec.name}_seg{i:03d}"
            if mode not in self.modes:
                out.append((name, f"mode {mode}"))
                continue
            if not held:
                out.append((name, "no object in contact"))
                continue
            missing = recording.gaps[(recording.gaps >= a) & (recording.gaps < b - 1)]
            if len(missing):
                after = recording.stamps[missing[0]]
                out.append((name, f"missing frames ({len(missing)} gap(s), first after timestamp_ns {after})"))
                continue
            pad = self.pad_frames  # still frames at both ends
            idx = np.concatenate([np.full(pad, a), np.arange(a, b), np.full(pad, b - 1)]).astype(int)
            out.append(
                HandObjectClip(
                    name,
                    rec.name,
                    {s: h.frames(idx) for s, h in hands.items()},
                    held,
                    {o: poses[o][idx] for o in set(held.values())},
                )
            )
        return out

    def recording_object_poses(self, recording: str) -> dict[str, np.ndarray]:
        out = self._recording(Path(self.raw_data_path) / "dataset" / recording, hands=False)
        if isinstance(out, str):
            raise ValueError(f"HOT3D recording {recording}: {out}")
        return out.poses

    def _recording(self, rec: Path, hands: bool = True) -> Recording | str:
        """A recording in the world frame, or why it has none; with ``hands=False`` its ``hands`` are left empty (the
        object poses alone need no MANO).
        """
        with open(rec / "metadata.json") as f:
            meta = json.load(f)
        uids = [str(u) for u in meta["object_uids"]]
        hand_rows, obj_rows = _hand_rows(rec), _object_rows(rec)
        stamps, last = [], None
        for t in sorted(hand_rows):  # the camera streams' timestamps of one frame are < 10 ms apart
            if last is None or t - last >= 10_000_000:
                last = t
                if {"0", "1"} <= set(hand_rows[t]) and all(u in obj_rows.get(t, {}) for u in uids):
                    stamps.append(t)
        if not stamps:
            return "no frame with both hands and every object"
        rows = {side: [hand_rows[t][key] for t in stamps] for side, key in (("right", "1"), ("left", "0"))}
        mano = {side: self._hand(r, side) for side, r in rows.items()} if hands else {}
        poses = {u: np.stack([obj_rows[t][u] for t in stamps]) for u in uids}
        # Frames 1.5 periods apart or more: the ones between lacked a hand or an object.
        gaps = np.flatnonzero(np.diff(stamps) > 1.5e9 / self.source_framerate)
        up = _QUEST3_UP if meta["headset"] == "Quest3" else np.eye(3)
        transl = [np.array([h["wrist_xform"]["t_xyz"] for h in r], dtype=np.float64) for r in rows.values()]
        to_world = _yaw(rec, up, poses, transl) @ up
        return Recording(
            {s: h.rotated(to_world) for s, h in mano.items()}, rotate_poses(poses, to_world), uids, gaps, stamps
        )

    def _hand(self, frames: list[dict], side: str) -> MANOHand:
        """MANO as 15 PCA components on the mean hand, the wrist rotation and smplx ``transl``, the first betas."""
        q = np.array([h["wrist_xform"]["q_wxyz"] for h in frames], dtype=np.float64)
        betas = np.array(frames[0]["betas"], dtype=np.float64)[:10]
        transl = np.array([h["wrist_xform"]["t_xyz"] for h in frames], dtype=np.float64)
        fingers = mano_pose_from_release(
            np.array([h["pose"] for h in frames], dtype=np.float64),
            self.body_model_dir,
            side,
            num_pca_comps=15,
            flat_hand_mean=False,
        )
        wrist = transl + mano_root_offset(self.body_model_dir, side, betas)
        return MANOHand(side, wrist, Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_rotvec(), fingers, betas)

    def _segments(self, verts: dict[str, np.ndarray], poses: dict[str, np.ndarray], uids: list[str]) -> list[tuple]:
        """(start, end, mode, {hand side: the object it holds}) per segment, before ``modes`` and padding. An idle hand
        (mode A) holds nothing; in mode C each hand holds its own object.
        """
        active, links = {}, {}
        for side, v in verts.items():
            link_of = mano_vertex_links(self.body_model_dir, side)
            tips = v[:, list(CHORD_TIP_VERTICES.values())]
            tip_dist = np.full(tips.shape[:2], np.inf)
            link_body = np.zeros((len(v), 16), dtype=int)
            link_dist = np.full((len(v), 16), np.inf)
            for k, u in enumerate(uids, 1):
                rot, pos = poses[u][:, :3, :3], poses[u][:, :3, 3]

                def local(x, rot=rot, pos=pos):
                    return np.matmul(x - pos[:, None], rot).reshape(-1, 3)

                tip_dist = np.minimum(tip_dist, self.near(u, local(tips), self.tip_contact).reshape(tip_dist.shape))
                d = self.near(u, local(v), self.link_contact).reshape(v.shape[:2])
                for link in range(16):
                    dl = d[:, link_of == link].min(1)
                    closer = dl < link_dist[:, link]
                    link_dist[closer, link], link_body[closer, link] = dl[closer], k
            active[side] = _fill_gaps(tip_dist.min(1) < self.tip_contact, self.gap_frames)
            links[side] = np.where(np.isfinite(link_dist), link_body, 0)

        both = np.concatenate([links["right"], links["left"]], axis=1)
        per_frame = np.array([Counter(r[r != 0].tolist()).most_common(1)[0][0] if (r != 0).any() else 0 for r in both])
        segments = []
        for start, end in _runs(active["right"] | active["left"]):
            for a, b in _split_by_object(start, end, per_frame, self.gap_frames):
                if b - a < self.min_frames:
                    continue
                body = {s: _majority(links[s][a:b][active[s][a:b]]) for s in verts}
                on = {s: active[s][a:b].any() for s in verts}
                if not (on["right"] and on["left"]):
                    mode = "A"
                elif None in body.values() or body["right"] == body["left"]:
                    mode = "B"
                else:
                    mode = "C"
                if mode == "B":  # one object: the one either hand's links are on
                    k = body["right"] or body["left"]
                    held = {s: uids[k - 1] for s in verts} if k else {}
                else:  # each active hand its own (mode A: the idle hand none)
                    held = {s: uids[body[s] - 1] for s in verts if on[s] and body[s]}
                segments.append((a, b, mode, held))
        return segments

    def _load_object_mesh(self, obj_id: str, materials: bool = False):
        import trimesh

        return trimesh.load(
            Path(self.raw_data_path) / "_assets" / "assets" / f"{obj_id}.glb",
            force="mesh",
            process=False,
            skip_materials=not materials,
        )


def _yaw(rec: Path, up: np.ndarray, poses: dict[str, np.ndarray], transl: list[np.ndarray]) -> np.ndarray:
    """Rotation about z putting the workspace (mean of every object position and hand ``transl``) along +y from the
    headset's first position, then turned 90 deg further (CHORD's convention).
    """
    pts = [p[:, :3, 3] for p in poses.values()] + list(transl)
    centre = (np.concatenate(pts) @ up.T).mean(0)[:2]
    with open(rec / "headset_trajectory.csv") as f:
        row = next(csv.DictReader(f))
    head = up @ np.array([float(row[f"t_wo_{a}[m]"]) for a in "xyz"])
    d = centre - head[:2]
    return Rotation.from_euler("z", np.pi - np.arctan2(d[1], d[0])).as_matrix()


def _hand_rows(rec: Path) -> dict[int, dict]:
    rows = {}
    with open(rec / "mano_hand_pose_trajectory.jsonl") as f:
        for line in f:
            e = json.loads(line)
            rows.setdefault(int(e["timestamp_ns"]), e["hand_poses"])
    return rows


def _object_rows(rec: Path) -> dict[int, dict[str, np.ndarray]]:
    rows: dict[int, dict[str, np.ndarray]] = {}
    with open(rec / "dynamic_objects.csv") as f:
        for r in csv.DictReader(f):
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_quat([float(r[f"q_wo_{a}"]) for a in "xyzw"]).as_matrix()
            pose[:3, 3] = [float(r[f"t_wo_{a}[m]"]) for a in "xyz"]
            rows.setdefault(int(r["timestamp[ns]"]), {})[r["object_uid"]] = pose
    return rows


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]))


def _fill_gaps(mask: np.ndarray, gap: int) -> np.ndarray:
    """Bridge runs of False shorter than ``gap`` between two True frames."""
    out = mask.copy()
    for a, b in _runs(~mask):
        if b - a < gap and a > 0 and b < len(mask):
            out[a:b] = True
    return out


def _split_by_object(start: int, end: int, body: np.ndarray, gap: int) -> list[tuple[int, int]]:
    """``[start, end)`` cut where the frame's touched object changes; no-contact gaps < ``gap`` inside one object
    kept.
    """
    w = body[start:end]
    out, i = [], 0
    while i < len(w):
        if w[i] == 0:
            i += 1
            continue
        cur, j = w[i], i + 1
        while j < len(w):
            if w[j] == cur:
                j += 1
            elif w[j] == 0:
                k = j
                while k < len(w) and w[k] == 0:
                    k += 1
                if k < len(w) and w[k] == cur and k - j < gap:
                    j = k + 1
                else:
                    break
            else:
                break
        out.append((start + i, start + j))
        i = j
    return out or [(start, end)]


def _majority(ids: np.ndarray) -> int | None:
    ids = ids[ids != 0]
    return int(Counter(ids.tolist()).most_common(1)[0][0]) if len(ids) else None
