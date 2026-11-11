"""``WeaveG1InspireHOIDataset``: one Weave npz of concatenated clips, split per clip and per scene entity.

The synthetic tests plant a distinct value in every joint column and body row of a Weave-layout npz and assert it
lands in the named column of the named entity — the failure they guard is a silently transposed joint or body, which
still trains. The real-data test checks the same mapping against the rig's own forward kinematics.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from genesis.utils import geom as gu

from eden.datasets.weave_dataset import (
    WEAVE_BODY_NAMES,
    WEAVE_JOINT_NAMES,
    WEAVE_OBJECTS,
    WeaveG1InspireHOIDataset,
    weave_object_dir,
)

_CPU = torch.device("cpu")
_OBJECTS = ("smallbox", "largebox")
_CLIP_LENGTHS = {"smallbox": (5, 8), "largebox": (6, 4)}
# Weave's fixed hand-mount joints, straight from the merged URDF.
_MOUNT = {"L": ((0.0415, 0.0, 0.0), (0.0, 0.0, np.pi / 2)), "R": ((0.0415, 0.0, 0.0), (np.pi, 0.0, -np.pi / 2))}
#: Weave's name prefix per entity: the hands name their joints and links without it.
_NPZ_PREFIX = {"robot": "", "left_hand": "L_", "right_hand": "R_"}


def _unit_quats(rng, n: int) -> np.ndarray:
    q = rng.normal(size=(n, 4)).astype(np.float32)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _write_object(root, obj: str, clip_lengths, rng) -> None:
    total, nb, nj = sum(clip_lengths), len(WEAVE_BODY_NAMES), len(WEAVE_JOINT_NAMES)
    frame = np.arange(total, dtype=np.float32)[:, None]
    body_pos = np.stack(
        [
            np.broadcast_to(np.arange(nb, dtype=np.float32), (total, nb)),
            np.broadcast_to(frame, (total, nb)),
            rng.normal(size=(total, nb)),
        ],
        axis=-1,
    ).astype(np.float32)  # x = body index, y = frame
    data = dict(
        joint_pos=(np.arange(nj, dtype=np.float32)[None] + 1000.0 * frame).astype(np.float32),
        joint_vel=(-np.arange(nj, dtype=np.float32)[None] - 1000.0 * frame).astype(np.float32),
        body_pos_w=body_pos,
        body_quat_w=_unit_quats(rng, total * nb).reshape(total, nb, 4),
        body_lin_vel_w=rng.normal(size=(total, nb, 3)).astype(np.float32),
        body_ang_vel_w=rng.normal(size=(total, nb, 3)).astype(np.float32),
        object_pos_w=rng.normal(size=(total, 3)).astype(np.float32),
        object_quat_w=_unit_quats(rng, total),
        object_lin_vel_w=rng.normal(size=(total, 3)).astype(np.float32),
        object_ang_vel_w=rng.normal(size=(total, 3)).astype(np.float32),
        contact_label=rng.integers(-1, 2, size=(total, nb)).astype(np.float32),
        motion_lengths=np.asarray(clip_lengths, dtype=np.int64),
        motion_names=np.asarray([f"sub1_{obj}_{i:03d}_v00" for i in range(len(clip_lengths))]),
        object_names=np.asarray([obj] * len(clip_lengths)),
        fps=np.asarray([50]),
    )
    ref = root / "reference" / "test"
    ref.mkdir(parents=True, exist_ok=True)
    np.savez(ref / f"{obj}_{len(clip_lengths)}clip_var5.npz", **data)
    (root / "objects" / obj).mkdir(parents=True, exist_ok=True)
    (root / "objects" / obj / f"{obj}.obj").write_text("v 0 0 0\n")
    return data


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    root = tmp_path_factory.mktemp("weave")
    rng = np.random.default_rng(0)
    raw = {obj: _write_object(root, obj, _CLIP_LENGTHS[obj], rng) for obj in _OBJECTS}
    return root, raw


def _load(root, **kwargs) -> WeaveG1InspireHOIDataset:
    kwargs.setdefault("objects", list(_OBJECTS))
    return WeaveG1InspireHOIDataset(
        WeaveG1InspireHOIDataset.configure(split="test", raw_data_path=str(root), **kwargs), device=_CPU
    )


@pytest.fixture(scope="module")
def dataset(snapshot):
    return _load(snapshot[0])


def _clip(dataset, key: str, field: str, clip: int) -> torch.Tensor:
    return getattr(dataset.motion_sequences[key], field)[clip]


def test_each_clip_of_each_file_becomes_one_trajectory_in_file_order(dataset):
    assert len(dataset) == 4
    assert dataset.buffer_length.tolist() == [5, 8, 6, 4]
    assert dataset.motion_labels("objects") == ["smallbox", "smallbox", "largebox", "largebox"]
    assert dataset.motion_labels("motion_names")[2] == "sub1_largebox_000_v00"
    assert [os.path.basename(f) for f in dataset.all_motion_files] == dataset.metadata["motion_names"]


def test_the_four_entities_carry_their_own_axes(dataset):
    shapes = {
        key: (seq.dofs_pos.concatenated_tensor.shape[-1], seq.links_pos_w.concatenated_tensor.shape[-2])
        for key, seq in dataset.motion_sequences.items()
    }
    assert shapes == {"robot": (29, 30), "left_hand": (12, 13), "right_hand": (12, 13), "object": (0, 0)}
    assert [dataset.motion_sequences[k].is_free_floating for k in ("robot", "left_hand", "object")] == [
        True,
        False,
        True,
    ]
    assert dataset.motion_sequences["object"].entity_kind == "rigid_object"
    # Body-frame link fields are not derived (no tracking term reads them; half the per-link memory).
    assert dataset.motion_sequences["robot"].links_pos_b is None


@pytest.mark.parametrize("key", ["robot", "left_hand", "right_hand"])
def test_every_dof_column_holds_the_npz_column_of_the_same_joint_name(dataset, key):
    """``joint_pos`` was planted as ``joint index + 1000 * frame``, so a column reads its source joint back."""
    names = dataset.motion_dofs_name[key]
    got = _clip(dataset, key, "dofs_pos", 1)  # the second smallbox clip starts at global frame 5
    expected = torch.tensor([WEAVE_JOINT_NAMES.index(_NPZ_PREFIX[key] + n) for n in names], dtype=torch.float32)
    torch.testing.assert_close(got[0], expected + 5000.0)
    torch.testing.assert_close(_clip(dataset, key, "dofs_vel", 1)[0], -expected - 5000.0)


@pytest.mark.parametrize("key", ["robot", "left_hand", "right_hand"])
def test_every_link_row_holds_the_npz_row_of_the_same_body_name(dataset, key):
    """``body_pos_w.x`` was planted as the body index; the hand base link (not an npz body) is checked below."""
    names = [_NPZ_PREFIX[key] + n for n in dataset.motion_links_name[key]]
    rows = [(i, WEAVE_BODY_NAMES.index(n)) for i, n in enumerate(names) if n in WEAVE_BODY_NAMES]
    got = _clip(dataset, key, "links_pos_w", 0)[:, [i for i, _ in rows], 0]
    torch.testing.assert_close(got, torch.tensor([b for _, b in rows], dtype=torch.float32).expand_as(got))
    assert len(rows) == (30 if key == "robot" else 12)


@pytest.mark.parametrize("key", ["left_hand", "right_hand"])
def test_a_hand_names_its_joints_and_links_as_the_hand_entity_does(key):
    """Without Weave's ``L_`` / ``R_`` prefix: the names of ``InspireRH56DFQ_L`` / ``_R``, so the motion maps 1:1."""
    from eden.options.robots import G1Inspire

    hand = getattr(G1Inspire(), key)
    assert sorted(WeaveG1InspireHOIDataset.motion_dofs_name[key]) == sorted(hand.dofs_name)
    links = WeaveG1InspireHOIDataset.motion_links_name[key]
    assert links[0] == "hand_base_link" and not any(n.startswith(("L_", "R_")) for n in links)
    assert set(type(hand).metadata.target_hand_links) <= set(links)


def test_the_robot_base_is_the_pelvis(dataset, snapshot):
    raw = snapshot[1]["smallbox"]
    torch.testing.assert_close(_clip(dataset, "robot", "base_pos_w", 0), torch.from_numpy(raw["body_pos_w"][:5, 0]))
    torch.testing.assert_close(
        _clip(dataset, "robot", "base_lin_vel_w", 0), torch.from_numpy(raw["body_lin_vel_w"][:5, 0])
    )


@pytest.mark.parametrize("key, prefix, side", [("left_hand", "L", "left"), ("right_hand", "R", "right")])
def test_the_hand_base_is_its_wrist_composed_with_the_fixed_mount(dataset, snapshot, key, prefix, side):
    """Independent of the loader: numpy geometry on the merged URDF's joint origin."""
    raw = snapshot[1]["largebox"]
    wrist = WEAVE_BODY_NAMES.index(f"{side}_wrist_yaw_link")
    frames = slice(0, 6)  # trajectory 2 is largebox's first clip: its file's frames 0..5
    pos_w, quat_w = raw["body_pos_w"][frames, wrist], raw["body_quat_w"][frames, wrist]
    ang_w, lin_w = raw["body_ang_vel_w"][frames, wrist], raw["body_lin_vel_w"][frames, wrist]
    xyz, rpy = _MOUNT[prefix]
    mount_quat = gu.xyz_to_quat(np.asarray(rpy), rpy=True)
    lever = gu.transform_by_quat(np.broadcast_to(np.asarray(xyz), pos_w.shape), quat_w)
    expected_pos = pos_w + lever
    expected_quat = np.stack([gu.transform_quat_by_quat(mount_quat, q) for q in quat_w])
    expected_lin = lin_w + np.cross(ang_w, lever)

    base = _clip(dataset, key, "links_pos_w", 2)[:, 0].numpy()
    np.testing.assert_allclose(base, expected_pos, atol=1e-5)
    np.testing.assert_allclose(_clip(dataset, key, "base_pos_w", 2).numpy(), expected_pos, atol=1e-5)
    got_quat = _clip(dataset, key, "links_quat_w", 2)[:, 0].numpy()
    np.testing.assert_allclose(np.abs((got_quat * expected_quat).sum(-1)), 1.0, atol=1e-5)
    np.testing.assert_allclose(_clip(dataset, key, "links_lin_vel_w", 2)[:, 0].numpy(), expected_lin, atol=1e-5)
    np.testing.assert_allclose(_clip(dataset, key, "links_ang_vel_w", 2)[:, 0].numpy(), ang_w, atol=1e-6)


def test_contact_labels_follow_the_links_and_the_hand_base_takes_its_wrists(dataset, snapshot):
    raw = snapshot[1]["smallbox"]["contact_label"][:5]
    robot = _clip(dataset, "robot", "contact_label", 0)
    assert robot.dtype == torch.int8
    np.testing.assert_array_equal(robot.numpy(), raw[:, :30])
    left = _clip(dataset, "left_hand", "contact_label", 0).numpy()
    names = dataset.motion_links_name["left_hand"]
    np.testing.assert_array_equal(left[:, 0], raw[:, WEAVE_BODY_NAMES.index("left_wrist_yaw_link")])
    np.testing.assert_array_equal(left[:, 1:], raw[:, [WEAVE_BODY_NAMES.index(f"L_{n}") for n in names[1:]]])


def test_the_object_sequence_is_its_root_pose_and_velocity(dataset, snapshot):
    raw = snapshot[1]["largebox"]
    torch.testing.assert_close(_clip(dataset, "object", "base_pos_w", 3), torch.from_numpy(raw["object_pos_w"][6:10]))
    torch.testing.assert_close(
        _clip(dataset, "object", "base_ang_vel_w", 3), torch.from_numpy(raw["object_ang_vel_w"][6:10])
    )


def test_filters_select_clips_by_motion_name(snapshot):
    ds = _load(snapshot[0], include_pattern=r"_001_")
    assert ds.metadata["motion_names"] == ["sub1_smallbox_001_v00", "sub1_largebox_001_v00"]
    assert ds.motion_labels("objects") == ["smallbox", "largebox"]
    assert ds.buffer_length.tolist() == [8, 4]


def test_drop_first_and_last_frames_trim_every_clip(snapshot):
    """They are part of the cache identity, so they must reach the clips — Weave once ignored them."""
    ds = _load(snapshot[0], drop_first_n_frames=1, drop_last_n_frames=1)
    assert ds.buffer_length.tolist() == [3, 6, 4, 2]
    first = ds.motion_sequences["robot"].dofs_pos[1][0]  # the second smallbox clip starts at global frame 5
    torch.testing.assert_close(first, torch.arange(29, dtype=torch.float32) + 6000.0)


def test_the_objects_option_loads_only_those_files(snapshot):
    ds = _load(snapshot[0], objects=["largebox"])
    assert ds.motion_labels("objects") == ["largebox", "largebox"]


def test_an_object_weave_does_not_publish_is_refused_naming_the_choices(snapshot):
    with pytest.raises(ValueError, match="unknown object.*'mug'.*smallbox"):
        _load(snapshot[0], objects=["mug"])


def test_a_split_other_than_train_or_test_is_refused():
    with pytest.raises(ValueError, match="split must be 'train' or 'test'"):
        WeaveG1InspireHOIDataset(
            WeaveG1InspireHOIDataset.configure(split="val", raw_data_path="/nonexistent"), device=_CPU
        )


def test_a_snapshot_without_the_objects_file_names_the_directory_it_searched(snapshot):
    with pytest.raises(FileNotFoundError, match="Expected one reference file for object 'tripod'"):
        _load(snapshot[0], objects=["tripod"])


def test_the_download_fetches_only_the_splits_files_of_the_selected_objects(hub):
    """The test split's smallbox reference and the geometry it needs, not the whole multi-GB repo."""
    root = hub.snapshot("appolyn/Weave")
    rng = np.random.default_rng(0)
    for obj in _OBJECTS:
        _write_object(root, obj, _CLIP_LENGTHS[obj], rng)
    dataset = WeaveG1InspireHOIDataset(
        WeaveG1InspireHOIDataset.configure(split="test", objects=["smallbox"]), device=_CPU
    )
    assert [call["allow_patterns"] for call in hub.calls] == [
        ["reference/test/smallbox_*.npz", "objects/smallbox/*", "objects/geometry/*"]
    ]
    assert dataset.motion_labels("objects") == ["smallbox"] * len(_CLIP_LENGTHS["smallbox"])


def test_weave_object_dir_resolves_inside_a_given_snapshot(snapshot):
    root = snapshot[0]
    assert weave_object_dir("smallbox", str(root)) == os.path.join(str(root), "objects", "smallbox")
    with pytest.raises(FileNotFoundError, match="No mesh for Weave object 'tripod'"):
        weave_object_dir("tripod", str(root))
    with pytest.raises(ValueError, match="Unknown Weave object 'mug'"):
        weave_object_dir("mug", str(root))


# ── real data ──────────────────────────────────────────────────────────────────────────────────────


def _cached_snapshot() -> str | None:
    """The locally cached ``appolyn/Weave`` snapshot with the smallbox test file, or None (no network here)."""
    try:
        from huggingface_hub import snapshot_download

        root = snapshot_download(
            repo_type="dataset",
            repo_id="appolyn/Weave",
            allow_patterns=["reference/test/smallbox_*"],
            local_files_only=True,
        )
    except Exception:
        return None
    return root if os.path.isfile(os.path.join(root, "reference/test/smallbox_24clip_var5.npz")) else None


@pytest.mark.slow
@pytest.mark.requires_genesis
def test_the_real_reference_links_are_the_rigs_own_forward_kinematics():
    """Weave's recorded body poses vs Eden FK of ``G1Inspire`` at the recorded joints.

    Checks the name mapping, the mount rebuild and the pelvis base all at once.
    """
    root = _cached_snapshot()
    if root is None:
        pytest.skip("appolyn/Weave smallbox test split is not in the local HuggingFace cache")
    from eden.options.robots import G1Inspire
    from tests.options.robots.conftest import build_scene

    ds = _load(root, objects=["smallbox"])
    env = build_scene(sim_substeps=1, robot=G1Inspire())
    entities = {k: env.entities[k] for k in ("robot", "left_hand", "right_hand")}
    starts = ds.motion_sequences["robot"].dofs_pos.batch_starts
    for clip, frame in ((0, 0), (57, 150), (119, 300)):
        g = int(starts[clip]) + frame
        robot_seq, names = ds.motion_sequences["robot"], ds.motion_dofs_name["robot"]
        q = robot_seq.dofs_pos.concatenated_tensor[g][[names.index(n) for n in entities["robot"].dofs_name]]
        qpos = torch.cat([robot_seq.base_pos_w.concatenated_tensor[g], robot_seq.base_quat_w.concatenated_tensor[g], q])
        entities["robot"].set_qpos(qpos[None].to(env.device))
        for hand in ("left_hand", "right_hand"):
            seq, names = ds.motion_sequences[hand], ds.motion_dofs_name[hand]
            qh = seq.dofs_pos.concatenated_tensor[g][[names.index(n) for n in entities[hand].dofs_name]]
            entities[hand].set_dofs_pos(qh[None].to(env.device))
        for key, entity in entities.items():
            _, idx = entity.find_named_links_idx_local(ds.motion_links_name[key], preserve_order=True)
            sim_pos = entity.get_links_pos(ls_idx_local=idx)[0].cpu()
            sim_quat = entity.get_links_quat(ls_idx_local=idx)[0].cpu()
            ref_pos = ds.motion_sequences[key].links_pos_w.concatenated_tensor[g]
            ref_quat = ds.motion_sequences[key].links_quat_w.concatenated_tensor[g]
            assert float((sim_pos - ref_pos).norm(dim=-1).max()) < 1e-5, key
            torch.testing.assert_close((sim_quat * ref_quat).sum(-1).abs(), torch.ones(len(idx)), atol=1e-5, rtol=0)


def test_weave_publishes_nine_objects():
    assert len(WEAVE_OBJECTS) == 9
