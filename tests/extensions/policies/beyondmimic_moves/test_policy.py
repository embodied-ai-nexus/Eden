"""``_remap_motion_npz_to_genesis_order``: an mjlab clip permuted into Genesis body order, with ``body_names`` stamped.

With the names stamped, ``MotionCommand`` resolves bodies by name whatever Genesis' internal order. Also: where
construction reads the clip's files from.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np

from eden.extensions.policies.beyondmimic_moves.policy import (
    _BODY_NAMES,
    _MJLAB_TO_GENESIS_BODY_PERM,
    BeyondMimicMovesPolicy,
    _remap_motion_npz_to_genesis_order,
)
from eden.utils import assets
from eden.options.robots.unitree_g1 import G1_DOFS29_TRACKED_LINKS_NAME

NUM_DOFS = 29


def _write_mjlab_npz(path, n_frames: int = 3) -> None:
    """A clip whose every body column holds its own mjlab index, so the permutation is observable."""
    n = len(_MJLAB_TO_GENESIS_BODY_PERM)
    tag = np.broadcast_to(np.arange(n, dtype=np.float32)[None, :, None], (n_frames, n, 3)).copy()
    quat = np.zeros((n_frames, n, 4), dtype=np.float32)
    quat[..., 0] = 1.0
    np.savez(
        path,
        joint_pos=np.zeros((n_frames, NUM_DOFS), dtype=np.float32),
        joint_vel=np.zeros((n_frames, NUM_DOFS), dtype=np.float32),
        body_pos_w=tag,
        body_quat_w=quat,
        body_lin_vel_w=tag.copy(),
        body_ang_vel_w=tag.copy(),
    )


def _eden_path(raw_path) -> str:
    base, ext = os.path.splitext(str(raw_path))
    return f"{base}_eden{ext}"


def test_the_remap_permutes_the_bodies_and_stamps_their_names(tmp_path):
    raw = tmp_path / "clip.npz"
    _write_mjlab_npz(raw)

    remapped = np.load(_remap_motion_npz_to_genesis_order(str(raw)))

    names = [str(x) for x in remapped["body_names"]]
    assert names == list(G1_DOFS29_TRACKED_LINKS_NAME)
    assert remapped["body_pos_w"][0, :, 0].tolist() == [float(src) for src in _MJLAB_TO_GENESIS_BODY_PERM]
    assert remapped["body_pos_w"][0, names.index("torso_link"), 0] == 15.0  # mjlab stores the torso at 15
    assert remapped["joint_pos"].shape == (3, NUM_DOFS)  # non-body fields pass through


def test_the_name_list_and_permutation_stay_aligned():
    """An edit to one that desyncs the other would mis-map bodies silently at runtime."""
    assert sorted(_MJLAB_TO_GENESIS_BODY_PERM) == list(range(len(G1_DOFS29_TRACKED_LINKS_NAME)))
    assert len(set(G1_DOFS29_TRACKED_LINKS_NAME)) == len(G1_DOFS29_TRACKED_LINKS_NAME)
    assert set(_BODY_NAMES) <= set(G1_DOFS29_TRACKED_LINKS_NAME)


def test_a_stale_cache_without_body_names_is_rebuilt(tmp_path):
    raw = tmp_path / "clip.npz"
    _write_mjlab_npz(raw)
    stale = _eden_path(raw)
    np.savez(stale, joint_pos=np.zeros((3, NUM_DOFS), dtype=np.float32))  # a pre-stamp _eden.npz

    assert _remap_motion_npz_to_genesis_order(str(raw)) == stale
    assert "body_names" in np.load(stale).files


def test_a_named_cache_is_reused(tmp_path):
    raw = tmp_path / "clip.npz"
    _write_mjlab_npz(raw)
    _remap_motion_npz_to_genesis_order(str(raw))
    mtime = os.path.getmtime(_eden_path(raw))

    assert _remap_motion_npz_to_genesis_order(str(raw)) == _eden_path(raw)
    assert os.path.getmtime(_eden_path(raw)) == mtime


def test_a_mirror_set_on_the_class_serves_the_manifest_and_motion_as_well_as_the_checkpoint(
    user_assets_dir, tmp_path, monkeypatch
):
    """An offline mirror set on the class: construction reads all three files from it, and never the Hub.

    The actor and the robot-defaults check are stubbed: they need real weights and a built scene, and fetch nothing.
    That pins ``__init__``'s shape: a new step there that needs a real ``env`` breaks this test for a reason that
    has nothing to do with where files come from; stub it here too, or move the fetches ahead of it.
    """

    def hub_unreachable(**kwargs):
        raise AssertionError(f"fetched from the Hub into {kwargs['local_dir']}")

    monkeypatch.setattr(assets, "snapshot_download", hub_unreachable)
    clip, mirror = BeyondMimicMovesPolicy.clip_name, tmp_path / "mirror"
    clip_dir = mirror / "dance" / clip
    (clip_dir / "policy").mkdir(parents=True)
    (clip_dir / "training").mkdir()
    (mirror / "manifest.json").write_text(json.dumps({"clips": {clip: {"category": "dance"}}}))
    (clip_dir / "policy" / f"{clip}_policy.pt").write_text("weights")
    _write_mjlab_npz(clip_dir / "training" / f"{clip}.npz")

    class Mirrored(BeyondMimicMovesPolicy):
        local_dir = str(mirror)
        _build_actor = staticmethod(lambda checkpoint_path, device: None)

        def _verify_robot_defaults(self) -> None:
            pass

    policy = Mirrored(SimpleNamespace(device="cpu"), Mirrored.configure())

    assert policy.checkpoint_path == str(clip_dir / "policy" / f"{clip}_policy.pt")
    assert policy.motion_file == _eden_path(clip_dir / "training" / f"{clip}.npz")
    assert not (user_assets_dir / "policies").exists()
