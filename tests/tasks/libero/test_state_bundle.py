"""``load_state_bundle``: the ``task.json`` + ``states.npz`` fixed-state bundle and its validation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from eden.tasks.libero.state_bundle import load_state_bundle

TASK_NAME = "pick_up_the_alphabet_soup_and_place_it_in_the_basket"
QPOS = "entity/robot/joint/joint1/qpos"


def _write_bundle(tmp_path, *, semantic_name="robot", **array_overrides):
    arrays = {
        "entity/robot/root_pos_world": np.zeros((2, 3)),
        "entity/robot/root_quat_wxyz_world": np.asarray([[1.0, 0.0, 0.0, 0.0]] * 2),
        "entity/robot/root_lin_vel_world": np.zeros((2, 3)),
        "entity/robot/root_ang_vel_body": np.zeros((2, 3)),
        QPOS: np.asarray([[0.1], [0.2]]),
        "entity/robot/joint/joint1/qvel": np.zeros((2, 1)),
    }
    arrays.update(array_overrides)
    manifest = {
        "task_name": TASK_NAME,
        "init_ids": [0, 1],
        "entities": [{"semantic_name": semantic_name, "joint_names": ["joint1"]}],
    }
    (tmp_path / "task.json").write_text(json.dumps(manifest), encoding="utf-8")
    np.savez(tmp_path / "states.npz", **arrays)
    return tmp_path / "task.json"


def test_valid_bundle_maps_init_ids_to_rows(tmp_path):
    bundle = load_state_bundle(_write_bundle(tmp_path), expected_task_name=TASK_NAME)

    assert bundle.manifest.init_ids == [0, 1]
    assert bundle.rows_for_init_ids([1, 0, 1]).tolist() == [1, 0, 1]
    assert bundle.arrays[QPOS].dtype == np.float64


@pytest.mark.parametrize(
    ("overrides", "error_type", "message"),
    [
        ({"unexpected": np.zeros((2,))}, ValueError, "unexpected"),
        ({QPOS: np.zeros((2, 1), dtype=np.float32)}, TypeError, "dtype"),
        ({QPOS: np.zeros((2,))}, ValueError, "shape"),
        ({QPOS: np.asarray([[np.nan], [0.0]])}, ValueError, "NaN"),
        ({"entity/robot/root_quat_wxyz_world": np.asarray([[2.0, 0.0, 0.0, 0.0]] * 2)}, ValueError, "non-unit"),
    ],
    ids=["unknown-array", "dtype", "shape", "nan", "non-unit-quat"],
)
def test_invalid_numeric_arrays_are_rejected(tmp_path, overrides, error_type, message):
    with pytest.raises(error_type, match=message):
        load_state_bundle(_write_bundle(tmp_path, **overrides))


def test_arrays_bind_to_the_manifest_semantic_names(tmp_path):
    with pytest.raises(ValueError, match="State array keys mismatch"):
        load_state_bundle(_write_bundle(tmp_path, semantic_name="different_robot"))


def test_unknown_init_id_is_rejected(tmp_path):
    bundle = load_state_bundle(_write_bundle(tmp_path))

    with pytest.raises(KeyError, match="init_id=99"):
        bundle.rows_for_init_ids([99])
