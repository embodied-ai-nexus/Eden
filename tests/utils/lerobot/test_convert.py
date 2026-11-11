"""Tests for ``eden.utils.lerobot.convert``, the HDF5 -> LeRobot v3 converter, on files written to ``tmp_path``.

``convert`` needs the optional ``lerobot`` package; the demo listing, rate and schema steps it runs first are
tested directly so they stay covered without it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


def _write_demo(group, *, length: int, qpos_dim: int, action_dim: int, fps: int = 50, roots: int = 1) -> None:
    dt = 1.0 / fps
    timestamps = np.arange(length, dtype=np.float32) * dt
    group.create_dataset("timestamp", data=timestamps)
    group.create_dataset("action", data=np.random.randn(length, action_dim).astype(np.float32))
    state_group = group.create_group("state")
    state_group.create_dataset("qpos", data=np.random.randn(length, qpos_dim).astype(np.float32))
    state_group.create_dataset("dofs_vel", data=np.random.randn(length, qpos_dim).astype(np.float32))
    state_group.create_dataset("pos", data=np.random.randn(length, 3 * roots).astype(np.float32))
    state_group.create_dataset("quat", data=np.random.randn(length, 4 * roots).astype(np.float32))


def _make_hdf5(path: Path, demo_specs: list[dict]) -> None:
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        for idx, spec in enumerate(demo_specs):
            _write_demo(data.create_group(f"demo_{idx}"), **spec)


def test_list_demos_sorts_numerically(tmp_path):
    from eden.utils.lerobot.convert import _list_demos

    path = tmp_path / "out.hdf5"
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        # Out-of-order so a lexical sort would put demo_10 before demo_2.
        for idx in [10, 2, 0, 1]:
            _write_demo(data.create_group(f"demo_{idx}"), length=2, qpos_dim=2, action_dim=1)

        names = _list_demos(f)

    assert names == ["demo_0", "demo_1", "demo_2", "demo_10"]


def test_list_demos_raises_on_missing_data_group(tmp_path):
    from eden.utils.lerobot.convert import _list_demos

    with h5py.File(tmp_path / "empty.hdf5", "w") as f:
        f.create_group("not_data")
        with pytest.raises(ValueError, match="'data' group"):
            _list_demos(f)


def test_list_demos_skips_groups_without_a_numeric_suffix(tmp_path):
    """``demo_backup`` would slip past a ``startswith("demo_")`` filter."""
    from eden.utils.lerobot.convert import _list_demos

    with h5py.File(tmp_path / "mixed.hdf5", "w") as f:
        data = f.create_group("data")
        _write_demo(data.create_group("demo_0"), length=2, qpos_dim=2, action_dim=1)
        _write_demo(data.create_group("demo_1"), length=2, qpos_dim=2, action_dim=1)
        data.create_group("demo_backup").create_dataset("note", data=[0])

        assert _list_demos(f) == ["demo_0", "demo_1"]


def test_infer_fps_rounds_to_the_nearest_int_and_rejects_constant_timestamps():
    from eden.utils.lerobot.convert import _infer_fps

    assert _infer_fps(np.arange(10) * (1.0 / 30.0)) == 30
    with pytest.raises(ValueError):
        _infer_fps(np.zeros(5, dtype=np.float32))


def test_build_features_rejects_dim_mismatch_across_demos(tmp_path):
    from eden.utils.lerobot.convert import _build_features, _list_demos

    path = tmp_path / "mismatch.hdf5"
    _make_hdf5(
        path,
        [
            dict(length=5, qpos_dim=7, action_dim=4),
            dict(length=5, qpos_dim=8, action_dim=4),  # qpos_dim drift
        ],
    )

    with h5py.File(path, "r") as f:
        demos = _list_demos(f)
        with pytest.raises(ValueError, match="Feature dim mismatch"):
            _build_features(f, demos)


def test_build_features_returns_consistent_schema(tmp_path):
    from eden.utils.lerobot.convert import _build_features, _list_demos

    path = tmp_path / "ok.hdf5"
    _make_hdf5(
        path,
        [
            dict(length=4, qpos_dim=6, action_dim=3),
            dict(length=4, qpos_dim=6, action_dim=3),
        ],
    )

    with h5py.File(path, "r") as f:
        features = _build_features(f, _list_demos(f))

    state_dim = 6 + 6 + 3 + 4
    assert features["observation.state"]["shape"] == (state_dim,)
    assert features["action"]["shape"] == (3,)
    assert features["observation.state"]["names"][:6] == [f"qpos_{i}" for i in range(6)]
    assert features["observation.state"]["names"][-4:] == ["root_quat_w", "root_quat_x", "root_quat_y", "root_quat_z"]


def test_build_features_names_one_root_per_recorded_arm(tmp_path):
    """A two-arm demonstration records both arms' states concatenated; each arm keeps its own root pose."""
    from eden.utils.lerobot.convert import _build_features, _list_demos

    path = tmp_path / "two_arms.hdf5"
    _make_hdf5(path, [dict(length=4, qpos_dim=18, action_dim=14, roots=2)])

    with h5py.File(path, "r") as f:
        names = _build_features(f, _list_demos(f))["observation.state"]["names"]

    assert len(names) == 18 + 18 + 6 + 8
    assert names[36:39] == ["root0_pos_x", "root0_pos_y", "root0_pos_z"]
    assert names[-4:] == ["root1_quat_w", "root1_quat_x", "root1_quat_y", "root1_quat_z"]


def test_build_features_rejects_quaternions_that_do_not_match_the_positions(tmp_path):
    from eden.utils.lerobot.convert import _build_features, _list_demos

    path = tmp_path / "bad_roots.hdf5"
    _make_hdf5(path, [dict(length=4, qpos_dim=6, action_dim=3)])
    with h5py.File(path, "a") as f:
        del f["data/demo_0/state/quat"]
        f["data/demo_0/state"].create_dataset("quat", data=np.zeros((4, 8), dtype=np.float32))
        with pytest.raises(ValueError, match=r"state/quat shape \(T, 4\) for 1 root"):
            _build_features(f, _list_demos(f))


def test_convert_roundtrip(tmp_path):
    pytest.importorskip("lerobot")
    from eden.utils.lerobot.convert import convert

    hdf5_path = tmp_path / "src.hdf5"
    out_dir = tmp_path / "lerobot_out"
    _make_hdf5(
        hdf5_path,
        [
            dict(length=6, qpos_dim=4, action_dim=2),
            dict(length=6, qpos_dim=4, action_dim=2),
        ],
    )

    convert(hdf5_path=hdf5_path, output_dir=out_dir, repo_id="local/test", robot_type="test", task="t")

    assert (out_dir / "meta" / "info.json").is_file()


def _lerobot_modules() -> list[str]:
    return [m for m in sys.modules if m == "lerobot" or m.startswith("lerobot.")]


def _set_aside_lerobot(monkeypatch) -> None:
    """Take every imported lerobot module out of ``sys.modules`` for the test; monkeypatch puts them back. A cached
    ``lerobot.datasets.lerobot_dataset`` would otherwise answer the converter's import whatever ``lerobot`` maps to.
    """
    for name in _lerobot_modules():
        monkeypatch.delitem(sys.modules, name)


@pytest.fixture
def lerobot_stand_in(tmp_path, monkeypatch):
    """``lerobot_stand_in({"datasets/__init__.py": source, ...})`` puts a ``lerobot`` package with those files first
    on ``sys.path``. The real lerobot's modules are set aside for the test and put back after it.
    """
    _set_aside_lerobot(monkeypatch)
    site = tmp_path / "site"

    def install(files: dict[str, str]) -> None:
        (site / "lerobot").mkdir(parents=True)
        (site / "lerobot" / "__init__.py").write_text("")
        for relative, source in files.items():
            (site / "lerobot" / relative).parent.mkdir(parents=True, exist_ok=True)
            (site / "lerobot" / relative).write_text(source)
        monkeypatch.syspath_prepend(str(site))

    yield install
    for name in _lerobot_modules():  # the stand-in's; monkeypatch then restores the real ones
        del sys.modules[name]


def _convert(tmp_path):
    from eden.utils.lerobot.convert import convert

    convert(tmp_path / "missing.hdf5", tmp_path / "out", repo_id="local/x", robot_type="eden", task="t")


def test_convert_without_lerobot_names_the_dataset_extra(tmp_path, monkeypatch):
    """A bare ``pip install lerobot`` leaves out the dataset code, so the message must name the extra."""
    from eden.utils.lerobot.convert import LEROBOT_INSTALL

    _set_aside_lerobot(monkeypatch)
    monkeypatch.setitem(sys.modules, "lerobot", None)  # the package reads as absent
    with pytest.raises(ImportError, match=re.escape(f"lerobot is not installed. Install it with `{LEROBOT_INSTALL}`")):
        _convert(tmp_path)


def test_convert_with_a_lerobot_that_has_no_datasets_package_names_the_extra(tmp_path, lerobot_stand_in):
    from eden.utils.lerobot.convert import LEROBOT_INSTALL

    lerobot_stand_in({})
    with pytest.raises(ImportError, match=re.escape("the installed lerobot has no lerobot.datasets")) as exc:
        _convert(tmp_path)
    assert f"Upgrade it with `{LEROBOT_INSTALL}`" in str(exc.value)


def test_convert_with_a_lerobot_older_than_v3_datasets_names_the_upgrade(tmp_path, lerobot_stand_in):
    """A lerobot 0.3 has ``LeRobotDataset`` but writes v2.1, with no ``finalize()``: refuse before writing anything."""
    from eden.utils.lerobot.convert import LEROBOT_INSTALL

    lerobot_stand_in(
        {
            "__init__.py": '__version__ = "0.3.3"\n',
            "datasets/__init__.py": "",
            "datasets/lerobot_dataset.py": "class LeRobotDataset:\n    pass\n",
        }
    )
    with pytest.raises(ImportError, match=re.escape("the installed lerobot (0.3.3) predates them")) as exc:
        _convert(tmp_path)
    assert f"Upgrade it with `{LEROBOT_INSTALL}`" in str(exc.value)


def test_convert_force_keeps_the_old_dataset_when_lerobot_is_missing(tmp_path, monkeypatch):
    """``--force`` deletes the output only once the converter can actually write the replacement."""
    from eden.utils.lerobot.convert import convert

    _make_hdf5(tmp_path / "demos.hdf5", [{"length": 5, "qpos_dim": 2, "action_dim": 2}])
    old_dataset = tmp_path / "out"
    (old_dataset / "meta").mkdir(parents=True)
    (old_dataset / "meta" / "info.json").write_text("{}")
    _set_aside_lerobot(monkeypatch)
    monkeypatch.setitem(sys.modules, "lerobot", None)

    with pytest.raises(ImportError, match="lerobot is not installed"):
        convert(tmp_path / "demos.hdf5", old_dataset, repo_id="local/x", robot_type="eden", task="t", force=True)
    assert (old_dataset / "meta" / "info.json").read_text() == "{}"


def test_convert_passes_a_module_lerobot_itself_lacks_through_unchanged(tmp_path, lerobot_stand_in):
    """Lerobot's own import error already names what is missing; rewording it would hide which module that is."""
    lerobot_stand_in({"datasets/__init__.py": "", "datasets/lerobot_dataset.py": "import eden_absent_dependency\n"})
    with pytest.raises(ModuleNotFoundError) as exc:
        _convert(tmp_path)
    assert exc.value.name == "eden_absent_dependency"
