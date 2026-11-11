"""Tests for ``eden.extensions.retargeting.flashdexretarget.importer``: releases on disk -> imported clips."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import trimesh

from eden.datasets.oakink2_dataset import YUP_TO_ZUP
from eden.datasets.utils.body_model import CHORD_TIP_VERTICES
from eden.extensions.retargeting.flashdexretarget import clips as C
from eden.extensions.retargeting.flashdexretarget import importer
from eden.extensions.retargeting.flashdexretarget.importer import import_demos
from eden.options.extensions.flashdexretarget import FDRImportOptions
from tests.datasets.conftest import write_box
from tests.support.hand_object_releases import (
    HOT3D_RECORDING,
    OAKINK2_SEQ,
    TEXTURE_RGB,
    write_dexycb,
    write_hot3d,
    write_oakink2,
    write_taco,
    write_textured_box,
)

TACO_TRIPLET = "(dust, brush, plate)"
FLIPPED_TRIPLET = "(stir, spoon, bowl)"


def _texture_rgb(obj_dir) -> tuple[int, int, int]:
    """The colour an exported object draws with, read through its visual.obj, material file and texture."""
    material = trimesh.load(obj_dir / "visual.obj", force="mesh", process=False).visual.material
    return material.image.convert("RGB").getpixel((0, 0))


def _import(root, mano_dir, source: dict, **options) -> dict:
    cfg = FDRImportOptions(
        source=source, raw=str(root / "raw"), dataset=str(root / "dataset"), mano_dir=str(mano_dir), **options
    )
    return {r["clip"]: r for r in import_demos(cfg)}


@pytest.fixture(scope="module")
def taco(tmp_path_factory, mano_dir):
    """A two-recording TACO release, imported once: one clean, one whose target flips in a frame."""
    root = tmp_path_factory.mktemp("taco")
    write_taco(root / "raw", TACO_TRIPLET, "20230101_001", tool_pos=(0.3, 0.0, 0.1), target_pos=(-0.3, 0.0, 0.1))
    write_taco(root / "raw", FLIPPED_TRIPLET, "20230101_002", (0.3, 0.0, 0.1), (-0.3, 0.0, 0.1), flip_target_at=10)
    return root / "dataset", _import(root, mano_dir, {"name": "taco"})


def test_taco_clip_lands_in_the_z_up_world_resampled_with_each_hand_on_its_nearest_object(taco, mano_template):
    dataset, _ = taco
    clip = C.load_clip(dataset / "human_demo" / "taco_dust__brush__plate_20230101_001.npz")
    assert len(clip["mano_right_wrist_pos"]) == 61  # 31 frames at 30 Hz -> 60 Hz
    assert (C.obj_id(clip, "right"), C.obj_id(clip, "left")) == ("brush01", "plate02")
    # TACO's world is turned by diag(-1, -1, 1); no table_z keeps the heights.
    np.testing.assert_allclose(clip["obj_right_pose"][0, :3, 3], (-0.3, 0.0, 0.1), atol=1e-6)
    np.testing.assert_allclose(clip["mano_right_wrist_pos"][0], (-0.3, -0.05, 0.2), atol=1e-6)
    # At zero pose and shape the synthetic MANO returns its template, shifted so joint 0 sits on the wrist.
    flip = np.diag([-1.0, -1.0, 1.0])
    expected_tip = (-0.3, -0.05, 0.2) + flip @ (mano_template[CHORD_TIP_VERTICES["index"]] - mano_template[0])
    np.testing.assert_allclose(C.tips(clip, "right")[0, 1], expected_tip, atol=1e-5)


def test_taco_meshes_are_exported_in_metres_from_their_geometry_alone(taco):
    """Without ``textures`` the release's texture is not read: trimesh's placeholder stands in for it."""
    dataset, _ = taco
    mesh = trimesh.load(dataset / "objects" / "brush01" / "visual.obj", force="mesh")
    np.testing.assert_allclose(mesh.extents, (0.1, 0.1, 0.1), atol=1e-6)
    assert _texture_rgb(dataset / "objects" / "brush01") != TEXTURE_RGB


def test_a_clip_whose_object_flips_in_one_frame_is_skipped(taco):
    _, rows = taco
    assert rows["taco_stir__spoon__bowl_20230101_002"]["skip"] == "object flips (bowl02)"


def test_dexycb_moves_poses_into_the_apriltag_frame_and_expands_the_45_component_pca(tmp_path, mano_dir):
    """The loader expands the 45 PCA coefficients itself: smplx turns PCA off when asked for all 45 components."""
    pca = np.linspace(-0.3, 0.3, 45)
    write_dexycb(tmp_path / "raw", mano_dir, pca)
    _import(tmp_path, mano_dir, {"name": "dexycb"}, sides="right")
    clip = C.load_clip(tmp_path / "dataset" / "human_demo" / "dexycb_subject_seq.npz")
    assert len(clip["mano_right_wrist_pos"]) == 59  # 30 frames with a hand at 30 Hz -> 60 Hz
    pose = clip["obj_right_pose"][0]
    np.testing.assert_allclose(pose[:3, 3], (0.1, -0.2, 0.1), atol=1e-6)
    np.testing.assert_allclose(pose[:3, :3], np.diag([1.0, -1.0, -1.0]), atol=1e-6)

    import smplx  # the `dev` group's reference implementation; the mano_dir fixture skips without it

    layer = smplx.create(str(mano_dir / "MANO_RIGHT.pkl"), model_type="mano", use_pca=False, flat_hand_mean=False)
    basis = np.eye(45)[::-1]  # the synthetic model's hands_components
    out = layer(
        global_orient=torch.zeros(1, 3),
        hand_pose=torch.tensor(pca @ basis, dtype=torch.float32)[None],
        betas=torch.zeros(1, 10),
        transl=torch.tensor([[0.1, 0.25, -0.2]]),  # the camera-frame translation, minus the tag's
    )
    world = out.joints[0].detach().numpy() @ np.diag([1.0, -1.0, -1.0])  # R^T x, row vectors
    np.testing.assert_allclose(clip["mano_right_wrist_pos"][0], world[0], atol=1e-5)
    np.testing.assert_allclose(
        C.joints(clip, "right")[0, C.MANO_JOINT_NAMES.index("index_distal")], world[3], atol=1e-5
    )


def test_oakink2_clips_carry_the_whole_recordings_objects_for_the_desk(tmp_path, mano_dir):
    write_oakink2(tmp_path / "raw", mano_dir)
    rows = _import(tmp_path, mano_dir, {"name": "oakink2"}, sides="right", textures=True)
    assert rows[f"{OAKINK2_SEQ}__st01"]["skip"] == "several objects in one hand"
    clip = C.load_clip(tmp_path / "dataset" / "human_demo" / f"{OAKINK2_SEQ}__st00.npz")
    assert C.obj_id(clip, "right") == "obj_b"
    assert clip["desk_context_ids"].tolist() == ["obj_a", "obj_b"]
    assert clip["desk_context_poses"].shape == (2, 20, 4, 4)  # 40 mocap frames at 120 Hz -> 60 Hz
    np.testing.assert_allclose(clip["desk_context_poses"][0, 0, :3, 3], YUP_TO_ZUP @ (-0.6, 0.05, 0.0), atol=1e-6)
    assert _texture_rgb(tmp_path / "dataset" / "objects" / "obj_a") == TEXTURE_RGB  # a context object's mesh too


def test_hot3d_puts_the_surface_the_recording_rests_on_at_table_z(tmp_path, mano_dir):
    """Object 222 (a 10 cm box at z = 0) is the lowest still thing of the recording: its bottom, -0.05, goes to 1.0."""
    write_hot3d(tmp_path / "raw", mano_dir)
    _import(tmp_path, mano_dir, {"name": "hot3d"})
    clip = C.load_clip(tmp_path / "dataset" / "human_demo" / f"{HOT3D_RECORDING}_seg000.npz")
    assert C.obj_id(clip, "right") == C.obj_id(clip, "left") == "111"
    np.testing.assert_allclose(clip["obj_right_pose"][:, 2, 3], 1.05, atol=1e-5)
    # 222 sets the height but no clip keeps it (no desk over the recording): its mesh is not exported.
    assert "desk_context_ids" not in clip
    assert sorted(p.name for p in (tmp_path / "dataset" / "objects").iterdir()) == ["111"]


def test_source_dataset_options_reach_the_dataset_and_cannot_override_the_import(tmp_path, mano_dir):
    write_hot3d(tmp_path / "raw", mano_dir)
    rows = _import(tmp_path, mano_dir, {"name": "hot3d", "dataset": {"modes": ["A", "B"], "pad_frames": 0}})
    assert rows[f"{HOT3D_RECORDING}_seg000"]["frames"] == 99  # 50 frames at 30 Hz, unpadded -> 60 Hz
    with pytest.raises(ValueError, match=r"source.dataset sets \['sides'\], which the import sets itself"):
        _import(tmp_path, mano_dir, {"name": "hot3d", "dataset": {"sides": "right"}})
    with pytest.raises(ValueError, match="pad_frame"):
        _import(tmp_path, mano_dir, {"name": "hot3d", "dataset": {"pad_frame": 0}})


def test_import_names_the_missing_mano_models(tmp_path):
    options = FDRImportOptions(source={"name": "taco"}, raw=str(tmp_path), dataset=str(tmp_path / "d"), mano_dir=".")
    with pytest.raises(FileNotFoundError, match=r"MANO_RIGHT\.pkl missing.*mano\.is\.tue\.mpg\.de"):
        import_demos(options)


def test_a_directory_that_is_not_the_release_names_its_expected_layout(tmp_path, mano_dir):
    options = FDRImportOptions(
        source={"name": "taco"}, raw=str(tmp_path), dataset=str(tmp_path / "d"), mano_dir=str(mano_dir)
    )
    with pytest.raises(FileNotFoundError, match=r"not a TACO release: \['Hand_Poses', 'Object_Poses'"):
        import_demos(options)


def test_a_recording_without_object_meshes_stops_before_writing_a_clip(tmp_path, mano_dir):
    """No object has a mesh, so the only stage's held object has none: the release has no clip, and nothing is
    written that a later step would trip over.
    """
    write_oakink2(tmp_path / "raw", mano_dir)
    for mesh in (tmp_path / "raw" / "object_repair" / "align_ds").glob("*/model.obj"):
        mesh.unlink()
    with pytest.raises(RuntimeError, match=r"held object has no mesh"):
        _import(tmp_path, mano_dir, {"name": "oakink2"}, sides="right")
    assert not list((tmp_path / "dataset" / "human_demo").glob("*.npz"))


def test_a_reimport_removes_the_clips_it_now_skips_and_warns_of_ones_it_did_not_write(tmp_path, mano_dir, eden_caplog):
    """Left in human_demo/, either would be retargeted and packed with the clips of this import."""
    write_taco(tmp_path / "raw", TACO_TRIPLET, "20230101_001", tool_pos=(0.3, 0.0, 0.1), target_pos=(-0.3, 0.0, 0.1))
    write_taco(tmp_path / "raw", FLIPPED_TRIPLET, "20230101_002", (0.3, 0.0, 0.1), (-0.3, 0.0, 0.1), flip_target_at=10)
    demo = tmp_path / "dataset" / "human_demo"
    demo.mkdir(parents=True)
    flipped, foreign = demo / "taco_stir__spoon__bowl_20230101_002.npz", demo / "oakink2_other.npz"
    for path in (flipped, foreign):
        np.savez(path, fps=np.float32(60.0))
    rows = _import(tmp_path, mano_dir, {"name": "taco"})
    assert rows[flipped.stem]["skip"] == "object flips (bowl02)"
    assert not flipped.exists()
    assert foreign.exists() and "also holds 1 clips this import did not write" in eden_caplog.text


def test_an_import_that_keeps_no_clip_raises_and_leaves_the_earlier_clips(tmp_path, mano_dir):
    """Its only recording's object flips, so every clip is skipped: an earlier import's clip must stay."""
    write_taco(tmp_path / "raw", FLIPPED_TRIPLET, "20230101_002", (0.3, 0.0, 0.1), (-0.3, 0.0, 0.1), flip_target_at=10)
    demo = tmp_path / "dataset" / "human_demo"
    demo.mkdir(parents=True)
    earlier = demo / "taco_stir__spoon__bowl_20230101_002.npz"
    np.savez(earlier, fps=np.float32(60.0))
    with pytest.raises(RuntimeError, match=r"every clip of .* was skipped \(1: \{'object flips': 1\}\), so nothing"):
        _import(tmp_path, mano_dir, {"name": "taco"})
    assert earlier.exists()


def test_a_reimport_replaces_an_object_mesh_only_when_the_release_geometry_changed(tmp_path, mano_dir, eden_caplog):
    write_taco(tmp_path / "raw", TACO_TRIPLET, "20230101_001", tool_pos=(0.3, 0.0, 0.1), target_pos=(-0.3, 0.0, 0.1))
    _import(tmp_path, mano_dir, {"name": "taco"}, textures=True)
    mesh = tmp_path / "dataset" / "objects" / "brush01" / "visual.obj"
    stamp = mesh.stat().st_mtime_ns
    _import(tmp_path, mano_dir, {"name": "taco"}, textures=True)
    assert mesh.stat().st_mtime_ns == stamp  # the same geometry: not rewritten, so nothing downstream is redone
    assert "had another mesh" not in eden_caplog.text
    write_box(tmp_path / "raw" / "object_models_released" / "brush01_cm.obj", extents=(0.2, 0.2, 0.2), scale=100.0)
    _import(tmp_path, mano_dir, {"name": "taco"}, textures=True)
    np.testing.assert_allclose(trimesh.load(mesh).extents, (0.2, 0.2, 0.2), atol=1e-6)
    assert "1 objects already in" in eden_caplog.text and "had another mesh" in eden_caplog.text
    # The untextured mesh's export leaves no material or texture of the textured one behind.
    assert sorted(p.name for p in mesh.parent.iterdir()) == ["visual.json", "visual.obj"]


def test_an_object_keeps_its_materials_and_textures(tmp_path, mano_dir):
    """HOT3D's object 222 is a textured glb (exported as context of the desk): visual.obj names a material file,
    which must sit beside it.
    """
    write_hot3d(tmp_path / "raw", mano_dir)
    _import(tmp_path, mano_dir, {"name": "hot3d", "desk_over_recording": True}, textures=True)
    obj = tmp_path / "dataset" / "objects" / "222"
    mtllib = next(
        line.split()[1] for line in (obj / "visual.obj").read_text().splitlines() if line.startswith("mtllib")
    )
    assert (obj / mtllib).exists() and list(obj.glob("*.png"))
    assert not list(obj.parent.glob(".*.tmp"))


def test_a_reimport_gives_an_object_exported_without_its_texture_the_release_one_beside_the_same_visual_obj(
    tmp_path, mano_dir, eden_caplog
):
    """Before the loaders kept materials, an import wrote a grey placeholder texture and recorded only the geometry.
    The next import writes the release's texture under the material name visual.obj already uses, whatever this
    trimesh names the loaded material (the release's own name, or material_0 under Genesis' OBJ loader), so
    visual.obj's bytes, and the convex parts and SDF grids keyed on them, stay as they were.
    """
    write_dexycb(tmp_path / "raw", mano_dir, np.zeros(45))
    release = tmp_path / "raw" / "models" / "002_master_chef_can" / "textured_simple.obj"
    obj = tmp_path / "dataset" / "objects" / "002_master_chef_can"
    obj.mkdir(parents=True)
    earlier = trimesh.load(release, force="mesh", process=False, skip_materials=True)  # what the earlier import read
    earlier.visual.material.name = "earlier_0"  # a name neither loader gives it, so the export has to take it over
    earlier.export(obj / "visual.obj")
    (obj / "visual.json").write_text(json.dumps({"geometry_sha1": importer._geometry_sha1(earlier)}))
    mesh_bytes = (obj / "visual.obj").read_bytes()
    assert _texture_rgb(obj) != TEXTURE_RGB
    _import(tmp_path, mano_dir, {"name": "dexycb"}, sides="right", textures=True)
    assert (obj / "visual.obj").read_bytes() == mesh_bytes
    assert _texture_rgb(obj) == TEXTURE_RGB
    assert [p.name for p in obj.glob("*.png")] == ["earlier_0.png"]
    assert "1 objects already in" in eden_caplog.text and "got this release's materials" in eden_caplog.text
    assert "had another mesh" not in eden_caplog.text and "written again" not in eden_caplog.text
    stamp = (obj / "earlier_0.png").stat().st_mtime_ns
    _import(tmp_path, mano_dir, {"name": "dexycb"}, sides="right", textures=True)
    assert (obj / "earlier_0.png").stat().st_mtime_ns == stamp
    assert eden_caplog.text.count("got this release's materials") == 1


def test_a_reimport_rewrites_only_what_the_release_changed(tmp_path, mano_dir, eden_caplog):
    """Another texture: only the material files. Other UVs: visual.obj too, which is warned of, as its convex parts
    are then redone. (A texture an earlier export left under another name goes: see the TACO test above.)
    """
    write_dexycb(tmp_path / "raw", mano_dir, np.zeros(45))
    release = tmp_path / "raw" / "models" / "002_master_chef_can" / "textured_simple.obj"
    obj = tmp_path / "dataset" / "objects" / "002_master_chef_can"
    _import(tmp_path, mano_dir, {"name": "dexycb"}, sides="right", textures=True)
    assert _texture_rgb(obj) == TEXTURE_RGB
    mesh_bytes = (obj / "visual.obj").read_bytes()

    write_textured_box(release, rgb=(1, 2, 3), material="texture_map")
    _import(tmp_path, mano_dir, {"name": "dexycb"}, sides="right", textures=True)
    assert _texture_rgb(obj) == (1, 2, 3) and (obj / "visual.obj").read_bytes() == mesh_bytes
    assert "got this release's materials" in eden_caplog.text and "written again" not in eden_caplog.text

    write_textured_box(release, rgb=(1, 2, 3), material="texture_map", uv_seed=1)
    _import(tmp_path, mano_dir, {"name": "dexycb"}, sides="right", textures=True)
    assert (obj / "visual.obj").read_bytes() != mesh_bytes
    assert "visual.obj was written again" in eden_caplog.text and "had another mesh" not in eden_caplog.text

    write_textured_box(release, rgb=(4, 5, 6), extents=(0.2, 0.2, 0.2), material="texture_map")
    _import(tmp_path, mano_dir, {"name": "dexycb"}, sides="right", textures=True)
    assert _texture_rgb(obj) == (4, 5, 6) and len(list(obj.glob("*.png"))) == 1
    assert "had another mesh" in eden_caplog.text


def test_an_import_without_textures_leaves_the_ones_an_earlier_import_wrote(tmp_path, mano_dir, eden_caplog):
    """Off, the materials are neither read nor compared: an object of unchanged geometry is kept as it is."""
    write_dexycb(tmp_path / "raw", mano_dir, np.zeros(45))
    _import(tmp_path, mano_dir, {"name": "dexycb"}, sides="right", textures=True)
    obj = tmp_path / "dataset" / "objects" / "002_master_chef_can"
    stamps = {p.name: p.stat().st_mtime_ns for p in obj.iterdir()}
    _import(tmp_path, mano_dir, {"name": "dexycb"}, sides="right")
    assert {p.name: p.stat().st_mtime_ns for p in obj.iterdir()} == stamps
    assert _texture_rgb(obj) == TEXTURE_RGB
    assert "got this release's materials" not in eden_caplog.text
