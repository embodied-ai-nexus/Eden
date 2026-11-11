"""Tests for ``eden.extensions.retargeting.flashdexretarget.pipeline``: imported clips -> ``motion.pt``."""

from __future__ import annotations

import csv
import importlib.util

import mujoco
import numpy as np
import pytest
import torch

from eden.extensions.retargeting.flashdexretarget import clips as C
from eden.extensions.retargeting.flashdexretarget import run_kinematic_retargeting
from eden.extensions.retargeting.flashdexretarget.ik import best_run, shape_fractions
from eden.extensions.retargeting.flashdexretarget.pack import object_meshes
from eden.extensions.retargeting.flashdexretarget.scene import name_id
from eden.options.extensions.flashdexretarget import FDRKinematicRetargetingOptions, FDRRetargetOptions
from tests.extensions.retargeting.flashdexretarget.conftest import self_fk_clip, write_box

T = 24


def _options(dataset, retarget: dict | None = None, **kw) -> FDRKinematicRetargetingOptions:
    """A fast IK: one attempt, a small start search, a short settle; no SDF bake (warp)."""
    return FDRKinematicRetargetingOptions(
        dataset=str(dataset),
        workers=1,
        steps=kw.pop("steps", {"sdf": False}),
        filters=kw.pop("filters", {"too_short": {"min_sec": 0.1}}),
        retarget={"attempts": 1, "n_guess": 4, "start_settle": 10, "approach": {"enable": False}, **(retarget or {})},
        **kw,
    )


def _report(dataset, robot: str = "xhand") -> dict[str, dict]:
    with open(dataset / robot / "report.csv") as f:
        return {r["clip"]: r for r in csv.DictReader(f)}


def _dataset(root, clips: dict[str, dict]):
    (root / "human_demo").mkdir(parents=True)
    for name, clip in clips.items():
        C.save_clip(root / "human_demo" / f"{name}.npz", clip)
    write_box(root / "objects" / "box" / "visual.obj")
    return root


@pytest.fixture(scope="module")
def retargeted(tmp_path_factory):
    """The XHand's own palm and fingertip trajectory imported as a human demo, run through every step."""
    dataset = tmp_path_factory.mktemp("dataset")
    clip, qpos = self_fk_clip(T)
    (dataset / "human_demo").mkdir()
    C.save_clip(dataset / "human_demo" / "self.npz", clip)
    write_box(dataset / "objects" / "box" / "visual.obj")
    motion = run_kinematic_retargeting(_options(dataset))
    with open(dataset / "xhand" / "report.csv") as f:
        report = {r["clip"]: r for r in csv.DictReader(f)}
    return motion, report, qpos


def test_ik_reaches_targets_the_robot_itself_produced(retargeted):
    _, report, _ = retargeted
    row = report["self"]
    assert row["reject"] == "" and row["flags"] == ""
    # Not 0: the fingertip constraints are soft (solimp) and every frame ends with them released while the joints
    # settle under PD, which leaves ~8 mm on this clip with FlashDexRetarget's own defaults too. A wrong frame or
    # joint order lands in the centimetres.
    assert float(row["ik_tip_err_mean_m"]) < 0.015


def test_motion_pt_packs_the_robot_qpos_and_the_object_slot(retargeted):
    motion, _, qpos = retargeted
    packed = torch.load(motion, weights_only=True)  # tensors, lists and strings only: no pickled objects
    frames = T - 3  # the smoothing window crops the first frames
    assert packed["joint_pos"].shape == (frames, qpos.shape[1])
    assert packed["motion_num_frames"].tolist() == [frames]
    assert motion == motion.parent.parent / "xhand" / "motion.pt"
    assert packed["right_object_mesh_dirs"] == ["../objects/box"]  # relative to motion.pt's directory
    assert object_meshes(packed, motion.parent) == [(motion.parent.parent / "objects" / "box" / "visual.obj", 1.0)]
    # The box sits 60 cm away: no fingertip is ever in contact with it.
    assert not packed["contact_contact_right"].any()
    # The wrist slides 5 cm along x over the clip; the IK's forearm slide joint follows it.
    np.testing.assert_allclose(
        packed["joint_pos"][-1, 0] - packed["joint_pos"][0, 0], qpos[-1, 0] - qpos[3, 0], atol=5e-3
    )


def test_the_reported_flags_are_those_of_the_saved_motion(retargeted):
    """The kept attempt's flags are reused rather than replayed again: they must be what a replay of the saved clip
    gives.
    """
    from eden.extensions.retargeting.flashdexretarget.quality import flags

    motion, report, _ = retargeted
    saved = C.load_clip(motion.parent / "retargeted" / "self.npz")
    dataset = motion.parent.parent
    replay = flags(saved, "xhand", dataset, _options(dataset).retarget, float(saved["fps"]))
    row = report["self"]
    assert {k: str(v) for k, v in replay.items()} == {k: row[k] for k in replay}


def test_motion_pt_names_its_robot_joints_and_frame_rate(retargeted):
    motion, _, qpos = retargeted
    packed = torch.load(motion, weights_only=True)
    assert packed["robot"] == "xhand" and packed["fps"] == 60.0
    assert len(packed["joint_names"]) == qpos.shape[1] == packed["joint_pos"].shape[1]
    assert packed["joint_names"][:3] == ["R_forearm_pos_x_joint", "R_forearm_pos_y_joint", "R_forearm_pos_z_joint"]


def test_smoothing_is_centred_so_the_hand_does_not_lag_its_frame():
    """On a ramp a centred average is the ramp itself; a trailing one (FlashDexRetarget's) lags it a frame."""
    T, smooth = 20, 3
    ramp = np.arange(T, dtype=np.float64)[:, None]
    out = best_run([(ramp, [0.0] * T, [])], FDRRetargetOptions(smooth=smooth))
    assert out["crop"] == smooth and out["qpos"].shape == (T - smooth, 1)
    np.testing.assert_allclose(out["qpos"][:-1, 0], ramp[smooth:-1, 0])  # the last frame's window shrinks onto it


def test_a_clip_no_longer_than_the_smoothing_window_is_refused():
    with pytest.raises(ValueError, match=r"3 frames and retarget\.smooth=3 crops the first 3, which leaves none"):
        best_run([(np.zeros((3, 2)), [0.0] * 3, [])], FDRRetargetOptions(smooth=3))


def test_the_approach_ends_on_the_open_hand_whatever_the_preshape():
    """Off the quarter grid it used to stop short: preshape 0.3 tried 0.3 and 0.05, never the open hand."""
    assert shape_fractions(1.0) == [1.0, 0.75, 0.5, 0.25, 0.0]  # FlashDexRetarget's schedule
    assert shape_fractions(0.3) == pytest.approx([0.3, 0.05, 0.0])
    assert shape_fractions(0.0) == [0.0]


def test_a_name_missing_from_the_robot_mjcf_is_named_with_the_ones_it_has():
    m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><site name="right_palm"/></worldbody></mujoco>')
    with pytest.raises(
        ValueError, match=r"No site named 'right_thumb_tip' in the IK model of x\.xml; its sites are \['right_palm'\]"
    ):
        name_id(m, mujoco.mjtObj.mjOBJ_SITE, "right_thumb_tip", "x.xml")


def test_a_saved_clip_is_reused_only_for_the_same_settings_and_each_robot_keeps_its_own(tmp_path):
    clip, _ = self_fk_clip(T)
    dataset = _dataset(tmp_path, {"self": clip})
    run_kinematic_retargeting(_options(dataset))
    run_kinematic_retargeting(_options(dataset))
    assert _report(dataset)["self"]["source"] == "cached"
    run_kinematic_retargeting(_options(dataset, retarget={"quality": {"tip_tracking": {"max_frac": 0.9}}}))
    assert _report(dataset)["self"]["source"] == "cached"  # the flags are scored afresh: no IK to redo
    run_kinematic_retargeting(_options(dataset, retarget={"settle_steps": 40}))
    assert _report(dataset)["self"]["source"] == "ik"  # another IK setting
    sharpa = run_kinematic_retargeting(_options(dataset, retarget={"settle_steps": 40}, robot="sharpa"))
    assert _report(dataset, "sharpa")["self"]["source"] == "ik"  # another robot, into sharpa/
    assert sharpa == dataset / "sharpa" / "motion.pt" and torch.load(sharpa, weights_only=True)["robot"] == "sharpa"
    assert torch.load(dataset / "xhand" / "motion.pt", weights_only=True)["robot"] == "xhand"  # left as it was
    assert _report(dataset, "xhand")["self"]["source"] == "ik"
    run_kinematic_retargeting(_options(dataset, retarget={"settle_steps": 40}))
    assert _report(dataset)["self"]["source"] == "cached"  # xhand/retargeted/ survived the sharpa run


def test_outputs_of_the_layout_before_per_robot_directories_are_pointed_out(tmp_path, eden_caplog):
    """A motion.pt left in the dataset itself would still load, holding whatever robot the old run had."""
    clip, _ = self_fk_clip(T)
    dataset = _dataset(tmp_path, {"self": clip})
    (dataset / "motion.pt").write_bytes(b"")
    run_kinematic_retargeting(_options(dataset, steps={"retarget": False, "pack": False, "sdf": False}))
    assert "still holds ['motion.pt'] of an earlier version" in eden_caplog.text
    assert (dataset / "xhand" / "report.csv").exists()


def test_pack_refuses_clips_retargeted_from_other_inputs(tmp_path):
    clip, _ = self_fk_clip(T)
    dataset = _dataset(tmp_path, {"self": clip})
    run_kinematic_retargeting(_options(dataset))
    no_ik = {"retarget": False, "sdf": False}
    with pytest.raises(ValueError, match=r"made from other inputs than this run's, e\.g\. self \(retarget\)"):
        run_kinematic_retargeting(_options(dataset, retarget={"settle_steps": 40}, steps=no_ik))
    write_box(dataset / "objects" / "box" / "visual.obj", extents=(0.12, 0.12, 0.12))  # new convex parts
    with pytest.raises(ValueError, match=r"e\.g\. self \([a-z0-9_/]*convex_sha1\)"):
        run_kinematic_retargeting(_options(dataset, steps=no_ik))


def test_a_changed_object_mesh_redoes_its_convex_parts_and_the_ik_on_them(tmp_path):
    clip, _ = self_fk_clip(T)
    dataset = _dataset(tmp_path, {"self": clip})
    run_kinematic_retargeting(_options(dataset))
    write_box(dataset / "objects" / "box" / "visual.obj", extents=(0.12, 0.12, 0.12))
    run_kinematic_retargeting(_options(dataset))
    assert _report(dataset)["self"]["source"] == "ik"


def test_a_clip_that_fails_is_left_out_of_motion_pt_with_its_old_result_removed(tmp_path):
    """The short clip retargets with no smoothing, then fails with smooth=3: its first result must not be packed."""
    clip, _ = self_fk_clip(T)
    short = {k: (v[:3] if np.ndim(v) and len(v) == T else v) for k, v in clip.items()}
    dataset = _dataset(tmp_path, {"self": clip, "short": short})
    no_filter = {"enable": False}
    run_kinematic_retargeting(_options(dataset, retarget={"smooth": 1}, filters=no_filter))
    assert (dataset / "xhand" / "retargeted" / "short.npz").exists()
    motion = run_kinematic_retargeting(_options(dataset, filters=no_filter))
    assert "retarget.smooth=3" in _report(dataset)["short"]["error"]
    assert not (dataset / "xhand" / "retargeted" / "short.npz").exists()
    assert torch.load(motion, weights_only=True)["motion_filename"] == ["self"]


def test_a_quality_replay_that_raises_keeps_the_ik_it_scored(tmp_path, monkeypatch):
    """The clip is left out with the error, but its IK is kept: the next run rescores it rather than redoing it. The
    replay runs in a worker process, which a patch here does not reach, so the failing replay runs in this one.
    """
    from eden.extensions.retargeting.flashdexretarget import retarget

    clip, _ = self_fk_clip(T)
    dataset = _dataset(tmp_path, {"self": clip})
    options = _options(dataset)
    run_kinematic_retargeting(options)
    path, out = dataset / "human_demo" / "self.npz", dataset / "xhand" / "retargeted" / "self.npz"

    def raises(*args):
        raise ValueError("no score")

    monkeypatch.setattr(retarget, "flags", raises)
    row = retarget._finish(
        str(path), None, str(dataset), str(out.parent), "xhand", options.retarget, C.clip_fps([path]), None, 0.0, {}
    )
    assert "the quality replay raised ValueError('no score')" in row["error"]
    retarget._failed(out, row, 1, 1)
    assert out.exists()
    monkeypatch.undo()
    motion = run_kinematic_retargeting(options)
    assert _report(dataset)["self"]["source"] == "cached"
    assert torch.load(motion, weights_only=True)["motion_filename"] == ["self"]


def test_ik_refuses_convex_parts_older_than_the_mesh_with_the_convex_step_off(tmp_path):
    """Pack and the SDF read the mesh, so IK on parts of the old one would mix two shapes."""
    clip, _ = self_fk_clip(T)
    dataset = _dataset(tmp_path, {"self": clip})
    no_convex = {"convex": False, "sdf": False}
    with pytest.raises(FileNotFoundError, match=r"box has no convex parts .*steps\.convex: true"):
        run_kinematic_retargeting(_options(dataset, steps=no_convex))
    run_kinematic_retargeting(_options(dataset))
    run_kinematic_retargeting(_options(dataset, steps=no_convex))
    assert _report(dataset)["self"]["source"] == "cached"
    write_box(dataset / "objects" / "box" / "visual.obj", extents=(0.12, 0.12, 0.12))  # a re-import's new mesh
    with pytest.raises(ValueError, match=r"box/convex were not made from the current .*box/visual\.obj"):
        run_kinematic_retargeting(_options(dataset, steps=no_convex))
    with pytest.raises(ValueError, match=r"not made from the current"):
        run_kinematic_retargeting(_options(dataset, steps={"convex": False, "retarget": False, "sdf": False}))


def test_a_preprocess_failure_stops_the_run_naming_the_clip(tmp_path):
    clip, _ = self_fk_clip(T)
    clip["obj_right_id"] = np.array("no_such_object")
    dataset = _dataset(tmp_path, {"self": clip})
    with pytest.raises(RuntimeError, match=r"Preprocessing .*self\.npz failed"):
        run_kinematic_retargeting(_options(dataset))


@pytest.mark.skipif(
    importlib.util.find_spec("warp") is None, reason="the SDF check imports warp (flashdexretarget extra)"
)
def test_a_gpu_index_out_of_range_fails_before_any_step(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    options = _options(tmp_path, steps={}, sdf={"device": "cuda", "gpus": [3]})
    with pytest.raises(ValueError, match=r"sdf\.gpus=\[3\] indexes the visible GPUs, which are 1 \(0\)"):
        run_kinematic_retargeting(options)  # an empty dataset: preprocess would raise FileNotFoundError


def test_every_clip_rejected_points_at_the_report(tmp_path):
    clip, _ = self_fk_clip(T)
    (tmp_path / "human_demo").mkdir()
    C.save_clip(tmp_path / "human_demo" / "self.npz", clip)
    write_box(tmp_path / "objects" / "box" / "visual.obj")
    # 0.4 s < too_short's 2.5 s; the SDF step is off so its warp check does not fire before preprocessing
    options = FDRKinematicRetargetingOptions(dataset=str(tmp_path), workers=1, steps={"sdf": False})
    with pytest.raises(RuntimeError, match=r"Every clip was rejected.*report\.csv"):
        run_kinematic_retargeting(options)


def test_a_dataset_without_clips_or_motion_names_the_import(tmp_path):
    with pytest.raises(FileNotFoundError, match="eden retarget flashdex import"):
        run_kinematic_retargeting(FDRKinematicRetargetingOptions(dataset=str(tmp_path), steps={"sdf": False}))


def test_reviewing_the_report_needs_neither_the_robot_assets_nor_the_convex_parts(tmp_path, monkeypatch):
    """With retarget and pack off, the IK cache keys (robot model files, which may download) are never built."""
    from eden.extensions.retargeting.flashdexretarget import pipeline

    def no_assets(*args):
        raise AssertionError("the robot assets were read")

    monkeypatch.setattr(pipeline, "robot_sha1", no_assets)
    clip, _ = self_fk_clip(T)
    dataset = _dataset(tmp_path, {"self": clip})
    run_kinematic_retargeting(
        _options(dataset, steps={"convex": False, "retarget": False, "pack": False, "sdf": False})
    )
    assert "self" in _report(dataset)
