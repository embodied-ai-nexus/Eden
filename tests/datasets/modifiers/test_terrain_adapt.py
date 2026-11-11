"""Tests for ``eden.datasets.modifiers.terrain_adapt.TCRSTerrainAdapt``.

Configuration, terrain dispatch and its content-addressed cache are fast; the end-to-end adapts (marked ``slow``)
solve IK against the local G1 MJCF. Several tests call the modifier's private ``_build_*`` / ``_*_hash`` helpers:
without running the full MPPI + IK solve, they are the only observable form of what the modifier resolved.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from eden.datasets.modifiers import DATASET_MODIFIER_REGISTRY, TCRSTerrainAdapt
from eden.extensions.motion_optimization.tcrs.demo import make_demo_clip
from eden.extensions.motion_optimization.tcrs.pipeline import ClipArrays, TCRSMotionAdapter
from eden.extensions.motion_optimization.tcrs.terrain_source import HeightFieldTerrain
from eden.options.datasets import DatasetModifierOptions
from eden.options.entities import EntityOptions, TerrainOptions
from eden.options.extensions.motion_optimization import TCRSPlannerOptions, TCRSRobotOptions
from eden.options.robots.unitree_g1 import G1Dofs29
from eden.utils.distribution import get_eden_version
from eden.utils.motion_reference.motion_data import BatchMotionSequence
from eden.utils.misc import get_assets_dir
from eden.utils.terrains.io import export_height_field

_CELL, _ORIGIN = 0.02, (-1.0, -1.0, 0.0)


def _stair_grid(step_depth: float = 0.3) -> np.ndarray:
    """3 steps of 0.1 m rise from x = 0.4, on a 5 m x 2 m grid at ``_CELL`` whose corner [0, 0] is ``_ORIGIN``."""
    grid = np.zeros((250, 100), dtype=np.float32)
    for k in range(3):
        grid[int((0.4 - _ORIGIN[0] + step_depth * k) / _CELL) :, :] = 0.1 * (k + 1)
    return grid


@pytest.fixture(scope="module")
def g1_mjcf() -> str:
    """The G1 MJCF, probed directly because ``get_asset_path`` would download it on a miss."""
    path = Path(get_assets_dir("humanoids", "unitree_g1", "g1_29dof_rev_1_0.xml"))
    if not path.is_file():
        pytest.skip(f"G1 MJCF asset not downloaded ({path})")
    return str(path)


@pytest.fixture
def stairs_npz(tmp_path) -> str:
    path = str(tmp_path / "stairs.npz")
    export_height_field(path, _stair_grid(), _CELL, origin=_ORIGIN)
    return path


def _modifier(**options) -> TCRSTerrainAdapt:
    return TCRSTerrainAdapt(options=TCRSTerrainAdapt.configure(**options))


def _sequence(lengths: tuple[int, ...]) -> BatchMotionSequence:
    """The demo walk packed as one sequence per length, with link velocities as a real dataset would carry."""
    seq = BatchMotionSequence.make_empty(buffer_lengths=list(lengths), num_dofs=29, num_links=30)
    seq.framerate[:] = 50
    for field in ("links_ang_vel_w", "dofs_vel", "base_lin_vel_w", "base_ang_vel_w"):
        getattr(seq, field).concatenated_tensor.zero_()
    for field in ("links_pos_b", "links_lin_vel_b", "links_ang_vel_b"):
        getattr(seq, field).concatenated_tensor.zero_()
    seq.contact_points_w = seq.contact_points_valid = None
    start = 0
    for n in lengths:
        fps, dofs_pos, links_pos, links_quat = make_demo_clip(num_frames=n)
        rows = slice(start, start + n)
        for field, value in (
            ("dofs_pos", dofs_pos),
            ("links_pos_w", links_pos),
            ("links_quat_w", links_quat),
            ("base_pos_w", links_pos[:, 0]),
            ("base_quat_w", links_quat[:, 0]),
            ("links_lin_vel_w", np.gradient(links_pos, axis=0) * fps),
        ):
            getattr(seq, field).concatenated_tensor[rows] = torch.from_numpy(value).float()
        start += n
    return seq


# ---------------------------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------------------------


def test_configure_resolves_through_the_registry(tmp_path):
    options = TCRSTerrainAdapt.configure(
        terrain=str(tmp_path / "stairs.npz"), robot_options=G1Dofs29(), terrain_offset=(1.0, -0.5, 0.25)
    )
    modifier = DATASET_MODIFIER_REGISTRY.get(options.name)(options=options)
    assert type(modifier) is TCRSTerrainAdapt
    assert modifier.terrain == str(tmp_path / "stairs.npz")
    assert modifier.terrain_offset == (1.0, -0.5, 0.25)


def test_robot_spec_and_planner_accept_live_options_or_their_dict_form(tmp_path):
    spec, planner = TCRSRobotOptions(ik_config={"solver": {"tol": 1e-4}}), TCRSPlannerOptions(mppi_n_samples=64)
    assert _modifier(robot_spec=spec, planner=planner)._build_robot_spec() == spec
    as_dicts = _modifier(robot_spec={"ik_config": {"solver": {"tol": 1e-4}}}, planner={"mppi_n_samples": 64})
    assert as_dicts._build_robot_spec() == spec
    assert as_dicts._build_planner() == planner
    with pytest.raises(TypeError, match="robot_spec"):
        _modifier(robot_spec=3)._build_robot_spec()


def test_nested_options_survive_a_yaml_round_trip(tmp_path):
    """Nested ``extra="forbid"`` options dump without a type stamp, as plain dicts; the modifier re-validates them."""
    yaml = pytest.importorskip("yaml")
    spec = TCRSRobotOptions(ik_config={"solver": {"tol": 1e-4}})
    options = TCRSTerrainAdapt.configure(
        terrain=str(tmp_path / "s.npz"),
        robot_options=G1Dofs29(),
        robot_spec=spec,
        planner=TCRSPlannerOptions(mppi_n_samples=64),
    )
    dumped = yaml.safe_load(yaml.safe_dump(options.model_dump()))
    modifier = DATASET_MODIFIER_REGISTRY.get(dumped["name"])(options=DatasetModifierOptions(**dumped))
    assert modifier._build_robot_spec() == spec
    assert modifier._build_planner().mppi_n_samples == 64


@pytest.mark.parametrize(
    "options, sequences, match",
    [
        ({"robot_options": G1Dofs29()}, {}, "'terrain'"),
        ({"terrain": "stairs.npz"}, {}, "'robot_options'"),
        ({"terrain": "stairs.npz", "robot_options": G1Dofs29()}, {"robot": None, "object": None}, "exactly one"),
    ],
)
def test_apply_rejects_an_incomplete_configuration(options, sequences, match):
    with pytest.raises(ValueError, match=match):
        _modifier(**options).apply(sequences)


def test_apply_refuses_a_sequence_with_body_model_parameters_before_the_solve(make_sequence):
    from tests.datasets.modifiers.conftest import with_body_model

    with pytest.raises(ValueError, match=r"TCRSTerrainAdapt cannot apply to 'robot': it carries body-model"):
        _modifier(terrain="stairs.npz", robot_options=G1Dofs29()).apply({"robot": with_body_model(make_sequence())})


def test_apply_refuses_a_configured_key_the_dataset_lacks_rather_than_skipping_it():
    modifier = _modifier(terrain="stairs.npz", robot_options=G1Dofs29(), motion_sequence_keys=("robot",))
    with pytest.raises(KeyError, match=r"\['robot'\] not in motion_sequences \['g1'\]"):
        modifier.apply({"g1": None})


# ---------------------------------------------------------------------------------------------------------------
# Terrain dispatch
# ---------------------------------------------------------------------------------------------------------------


def test_a_terrain_dict_without_a_file_is_terrain_options_even_with_default_generators():
    terrain = _modifier(
        terrain={"subterrain_size": (2.0, 2.0), "horizontal_scale": 0.1, "vertical_scale": 0.005}
    )._build_terrain()
    assert terrain.height_at(1.0, 1.0) == pytest.approx(0.0)  # the default FlatTerrain generator


def test_a_terrain_dict_with_a_file_is_a_mesh(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    box = trimesh.creation.box(extents=(1.0, 1.0, 0.2))
    box.apply_translation((0.0, 0.0, 0.1))  # top face at z = 0.2
    box.export(str(tmp_path / "box.obj"))
    terrain = _modifier(terrain={"file": str(tmp_path / "box.obj")})._build_terrain()
    assert terrain.height_at(0.0, 0.0) == pytest.approx(0.2, abs=0.02)


@pytest.mark.parametrize(
    "terrain, match", [(TerrainOptions(randomize=True), "randomize"), (EntityOptions(), "file is empty")]
)
def test_an_unreproducible_or_empty_terrain_is_rejected(g1_mjcf, terrain, match):
    with pytest.raises(ValueError, match=match):
        _modifier(terrain=terrain, robot_options=G1Dofs29()).apply({})


# ---------------------------------------------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------------------------------------------


def _clip(num_frames: int = 8, *, shift: float = 0.0, lin_vel: float | None = None) -> ClipArrays:
    fps, dofs_pos, links_pos, links_quat = make_demo_clip(num_frames=num_frames)
    vel = None if lin_vel is None else np.full_like(links_pos, lin_vel)
    return ClipArrays(
        fps=fps, dofs_pos=dofs_pos, links_pos_w=links_pos + shift, links_quat_w=links_quat, links_lin_vel_w=vel
    )


def test_the_context_hash_is_stable_and_changes_with_every_input(g1_mjcf, tmp_path):
    modifier = _modifier()
    terrain = HeightFieldTerrain(_stair_grid(), _CELL, origin=_ORIGIN)
    adapter = TCRSMotionAdapter(model_path=g1_mjcf)
    base = modifier._context_hash(terrain, adapter, g1_mjcf, (0.0, 0.0, 0.0))
    assert modifier._context_hash(terrain, adapter, g1_mjcf, (0.0, 0.0, 0.0)) == base

    other_params = TCRSMotionAdapter(model_path=g1_mjcf, params=TCRSPlannerOptions(mppi_n_samples=64))
    bumped = _stair_grid()
    bumped[10, 10] += 0.01  # one terrain cell
    edited_mjcf = tmp_path / "robot.xml"
    edited_mjcf.write_bytes(Path(g1_mjcf).read_bytes() + b"<!-- edited -->")
    for changed in (
        modifier._context_hash(terrain, other_params, g1_mjcf, (0.0, 0.0, 0.0)),
        modifier._context_hash(HeightFieldTerrain(bumped, _CELL, origin=_ORIGIN), adapter, g1_mjcf, (0.0, 0.0, 0.0)),
        modifier._context_hash(terrain, adapter, g1_mjcf, (0.5, 0.0, 0.0)),
        modifier._context_hash(terrain, adapter, str(edited_mjcf), (0.0, 0.0, 0.0)),
    ):
        assert changed != base


def test_the_clip_hash_is_stable_and_changes_with_the_poses_and_the_velocities():
    """Velocities feed contact detection, so both their presence and their contents key the entry."""
    hashes = [
        TCRSTerrainAdapt._clip_hash("context", clip)
        for clip in (_clip(), _clip(shift=0.01), _clip(lin_vel=0.0), _clip(lin_vel=1.0))
    ]
    assert TCRSTerrainAdapt._clip_hash("context", _clip()) == hashes[0]
    assert TCRSTerrainAdapt._clip_hash("other context", _clip()) != hashes[0]
    assert len(set(hashes)) == len(hashes)


def test_a_cached_entry_carries_the_eden_version_and_an_unstamped_one_is_stale(tmp_path):
    adapted = {
        "dofs_pos": np.zeros((4, 29), dtype=np.float32),
        "links_pos_w": np.zeros((4, 30, 3), dtype=np.float32),
        "links_quat_w": np.zeros((4, 30, 4), dtype=np.float32),
        "ik_pos_errors": np.zeros(4, dtype=np.float32),
        "ik_bad_frame_count": np.int64(0),
    }
    path, clip = str(tmp_path / "entry.npz"), _clip(num_frames=4)
    TCRSTerrainAdapt._save_cached_clip(path, adapted, get_eden_version())
    loaded = TCRSTerrainAdapt._load_cached_clip(path, clip)
    assert str(loaded.pop("eden_version")) == get_eden_version()
    assert set(loaded) == set(adapted)

    np.savez(path, **adapted)  # the pre-v2 layout, without the stamp
    assert TCRSTerrainAdapt._load_cached_clip(path, clip) is None


@pytest.mark.parametrize(
    ("cached", "current", "stale"),
    [
        ("0.1.0", "0.1.0", False),
        ("0.1.0", "0.2.0", True),
        # An environment synced before the rename to eden-world knows no version, so it cannot compare...
        ("0.1.0", "", False),
        # ...and what it wrote is of unknown release to a run that knows its own.
        ("", "0.1.0", True),
    ],
)
def test_only_an_entry_stamped_by_another_known_release_is_stale(cached, current, stale):
    assert TCRSTerrainAdapt._stamped_by_another_version(cached, current) is stale


# ---------------------------------------------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------------------------------------------


def test_a_clip_shorter_than_three_frames_is_refused_before_the_solve(tmp_path):
    """The solver smooths with a 3-tap kernel a 2-frame clip cannot fill; this must fail before the optimization.

    Unlike the dataset's own velocity estimate, which handles 2 frames.
    """
    path = str(tmp_path / "flat.npz")
    export_height_field(path, np.zeros((50, 50), dtype=np.float32), _CELL, origin=_ORIGIN)
    seq = BatchMotionSequence.make_empty(buffer_lengths=[2], num_dofs=29, num_links=30)
    seq.framerate[:] = 50
    with pytest.raises(ValueError, match="at least 3 frames"):
        _modifier(terrain=path, robot_options=G1Dofs29(), cache_dir=None).apply({"g1": seq})


@pytest.mark.slow
def test_adapts_a_clip_up_the_stairs_and_serves_the_rerun_from_its_cache(g1_mjcf, stairs_npz, tmp_path, eden_caplog):
    cache_dir = str(tmp_path / "cache")

    def run():
        modifier = _modifier(
            terrain=stairs_npz, robot_options=G1Dofs29(), cache_dir=cache_dir, planner={"lowpass_cutoff_hz": 0.0}
        )
        return modifier.apply({"g1": _sequence((80,))})["g1"]

    first = run()
    for field in ("dofs_pos", "links_pos_w", "links_quat_w"):
        assert torch.isfinite(getattr(first, field).concatenated_tensor).all(), field
    # Velocities are recomputed from the adapted poses, not left at zero.
    assert first.dofs_vel.concatenated_tensor.abs().max() > 0.0
    assert first.base_lin_vel_w.concatenated_tensor.abs().max() > 0.0
    base_z = first.base_pos_w.concatenated_tensor[:, 2]
    assert base_z[-1] > base_z[0] + 0.05  # the pelvis climbs
    (entry,) = os.listdir(cache_dir)
    with np.load(os.path.join(cache_dir, entry)) as cached:
        assert cached["dofs_pos"].dtype == np.float32
        # The release, not the checkout's commit: a stamp that moved with every commit would call the cache stale
        # after each one.
        assert str(cached["eden_version"]) == get_eden_version()

    second = run()
    assert any("1 cache hit(s)" in record.getMessage() for record in eden_caplog.records)
    assert not any("different eden version" in record.getMessage() for record in eden_caplog.records)
    assert os.listdir(cache_dir) == [entry]
    for field in ("dofs_pos", "links_pos_w", "links_quat_w", "base_pos_w", "dofs_vel"):
        assert torch.equal(getattr(first, field).concatenated_tensor, getattr(second, field).concatenated_tensor)


@pytest.mark.slow
def test_each_clip_of_a_packed_batch_gets_its_own_adaptation(g1_mjcf, stairs_npz, tmp_path):
    """The single-clip reruns hit the entries the batched run wrote, so equality pins the slicing and write-back."""
    modifier = _modifier(
        terrain=stairs_npz,
        robot_options=G1Dofs29(),
        cache_dir=str(tmp_path / "cache"),
        planner={"lowpass_cutoff_hz": 0.0},
    )
    lengths = (80, 50)
    batched = modifier.apply({"g1": _sequence(lengths)})["g1"]
    assert len(os.listdir(tmp_path / "cache")) == 2

    starts = batched.base_pos_w.batch_starts
    for i, n in enumerate(lengths):
        alone = modifier.apply({"g1": _sequence((n,))})["g1"]
        rows = slice(int(starts[i]), int(starts[i]) + n)
        for field in ("dofs_pos", "links_pos_w", "links_quat_w", "base_pos_w", "dofs_vel"):
            assert torch.equal(
                getattr(batched, field).concatenated_tensor[rows], getattr(alone, field).concatenated_tensor
            )
    assert len(os.listdir(tmp_path / "cache")) == 2  # every single-clip run was a cache hit
