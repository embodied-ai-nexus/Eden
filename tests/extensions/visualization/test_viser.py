"""``eden.extensions.visualization.viser.ViserViewer``: recording lifecycle and the exported scene's playback."""

from __future__ import annotations

import base64
import os
import re
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("viser", minversion="1.1.0")

import msgspec
import zstandard

from eden.extensions.visualization.viser import ViserViewer
from eden.options import BoxOptions, EnvOptions, SceneOptions
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env


def _decode_recording(path):
    data = path.read_bytes()
    if path.suffix == ".html":
        encoded = re.search(rb'window.__VISER_EMBED_DATA__="([^"]+)"', data)
        assert encoded is not None
        data = base64.b64decode(encoded[1])
    raw = zstandard.ZstdDecompressor().decompress(data[8:])
    metadata_size = int.from_bytes(raw[:8], "little")
    metadata = msgspec.msgpack.decode(raw[8 : 8 + metadata_size])
    offset = 8 + metadata_size
    buffers = []
    for size in metadata["binaryBufferLengths"]:
        offset += -offset % 8
        buffers.append(raw[offset : offset + size])
        offset += size
    return metadata, buffers


@pytest.fixture(scope="module")
def env():
    """Two envs, 2 m apart, each with a box falling freely: ``env.dt`` is 0.02 s."""
    env = build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=2, sim_dt=0.005, decimation=4, env_spacing=(2.0, 2.0)),
            scene_options=SceneOptions(box=BoxOptions(size=(0.2, 0.2, 0.2), default_root_pos=(0.0, 0.0, 1.0))),
        ),
        reset=True,
    )
    yield env
    env.destroy()


@pytest.fixture
def viewer(env):
    viewer = ViserViewer(env, port=0, enable_gui=False)
    yield viewer
    viewer.close()


@pytest.mark.parametrize(("suffix", "fps"), [(".viser", None), (".html", 25)])
def test_recording_preserves_batched_poses_and_timing(env, viewer, tmp_path, suffix, fps):
    """Every env's box pose is recorded per update, stamped ``env.dt`` (or ``1 / fps``) apart; nothing after stop."""
    box = env.entities["box"].links[0]
    path = tmp_path / "nested" / f"recording{suffix}"
    expected = []
    viewer.start_recording(path, fps=fps)
    for _ in range(3):
        expected.append(box.get_pos(relative=False).cpu().numpy() + env.scene.envs_offset)
        env.step()
        viewer.update()
    expected.append(box.get_pos(relative=False).cpu().numpy() + env.scene.envs_offset)
    assert viewer.is_recording
    assert viewer.stop_recording() == path
    assert not viewer.is_recording
    env.step()
    viewer.update()

    metadata, buffers = _decode_recording(path)
    dt = env.dt if fps is None else 1 / fps
    assert metadata["durationSeconds"] == pytest.approx(3 * dt)
    poses_by_time = {}
    for timestamp, message in metadata["messages"]:
        props = message.get("updates", message.get("props", {}))
        if "batched_positions" in props:
            value = props["batched_positions"]
            poses_by_time[timestamp] = np.frombuffer(buffers[value["__binary_index"]], dtype=value["dtype"]).reshape(
                -1, 3
            )
    np.testing.assert_allclose(list(poses_by_time), np.arange(4) * dt)
    np.testing.assert_allclose(list(poses_by_time.values()), expected, atol=1e-6)
    assert not np.allclose(expected[0], expected[-1])


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_invalid_fps_fails_before_recording(viewer, tmp_path, fps):
    with pytest.raises(ValueError, match="fps"):
        viewer.start_recording(tmp_path / "replay.html", fps=fps)
    assert not viewer.is_recording


def test_recording_lifecycle(viewer, tmp_path):
    with pytest.raises(RuntimeError, match="No Viser recording"):
        viewer.stop_recording()
    with pytest.raises(ValueError, match=".html or .viser"):
        viewer.start_recording(tmp_path / "replay.mp4")
    path = tmp_path / "replay.viser"
    viewer.start_recording(path)
    with pytest.raises(RuntimeError, match="already active"):
        viewer.start_recording(tmp_path / "other.html")
    viewer.update()
    viewer.stop_recording()
    assert _decode_recording(path)[0]["durationSeconds"] == pytest.approx(0.02)
    viewer.start_recording(tmp_path / "second.html")
    viewer.close()
    assert not viewer.is_recording
    assert _decode_recording(tmp_path / "second.html")[0]["durationSeconds"] == 0


def test_output_directory_rejected(viewer, tmp_path):
    path = tmp_path / "directory.html"
    path.mkdir()
    with pytest.raises(IsADirectoryError):
        viewer.start_recording(path)
    assert not viewer.is_recording


def test_start_recording_preserves_a_pending_gui_reset(viewer, tmp_path):
    """Seeding the first frame must not drain an event the simulation loop has not seen.

    ``_gui_event_state["reset_triggered"]`` is exactly what the viewer's "Reset Env"
    button writes, so a reset clicked just before recording starts has to survive into
    the next ``update()`` -- and be reported only once.
    """
    viewer._gui_event_state["reset_triggered"] = True
    viewer.start_recording(tmp_path / "replay.viser")
    assert viewer.update()["reset"] is True
    assert viewer.update()["reset"] is False
    viewer.stop_recording()


def test_html_export_requires_a_viser_that_can_render_it(viewer, tmp_path, monkeypatch):
    """An HTML target must be rejected at start: as_html() consumes the single-use handle."""
    import viser
    from viser.infra import StateSerializer

    monkeypatch.delattr(StateSerializer, "as_html")
    with pytest.raises(RuntimeError, match="standalone HTML"):
        viewer.start_recording(tmp_path / "replay.html")
    assert not viewer.is_recording
    monkeypatch.undo()

    # as_html() renders by reading Viser's bundled client, so that has to be there too.
    monkeypatch.setattr(viser, "__file__", str(tmp_path / "no_client" / "__init__.py"))
    with pytest.raises(RuntimeError, match="no built client"):
        viewer.start_recording(tmp_path / "replay.html")
    assert not viewer.is_recording
    monkeypatch.undo()

    # .viser only needs serialize(), which every supported Viser has.
    viewer.start_recording(tmp_path / "replay.viser")
    assert viewer.stop_recording().exists()


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason="root ignores permissions")
def test_unwritable_output_directory_fails_before_recording(viewer, tmp_path):
    out_dir = tmp_path / "readonly"
    out_dir.mkdir()
    out_dir.chmod(0o500)
    try:
        with pytest.raises(ValueError, match="not writable"):
            viewer.start_recording(out_dir / "replay.viser")
    finally:
        out_dir.chmod(0o700)
    assert not viewer.is_recording


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason="root ignores permissions")
def test_read_only_destination_file_is_replaceable(viewer, tmp_path):
    """Only the directory need be writable: the export renames a sibling into place."""
    path = tmp_path / "replay.viser"
    path.write_bytes(b"stale")
    path.chmod(0o444)
    try:
        viewer.start_recording(path)
        viewer.update()
        assert viewer.stop_recording() == path
        assert _decode_recording(path)[0]["durationSeconds"] == pytest.approx(0.02)
    finally:
        path.chmod(0o644)


@pytest.mark.parametrize("suffix", [".viser", ".html"])
def test_failed_export_leaves_the_destination_untouched(viewer, tmp_path, monkeypatch, suffix):
    """A failed export must not truncate the destination or strand a temporary file."""
    path = tmp_path / f"replay{suffix}"
    path.write_bytes(b"a previous recording")
    viewer.start_recording(path)
    viewer.update()

    targets = []

    def _fail(self, *args, **kwargs):
        targets.append(self)
        raise OSError("No space left on device")

    monkeypatch.setattr(Path, "write_bytes" if suffix == ".viser" else "write_text", _fail)
    with pytest.raises(RuntimeError, match="No space left on device"):
        viewer.stop_recording()

    # The destination is never the write target -- that is what makes the export atomic,
    # and it is why a partial write cannot truncate a recording already saved there.
    assert targets and path not in targets
    assert path.read_bytes() == b"a previous recording"
    assert sorted(q.name for q in tmp_path.iterdir()) == [path.name]
    assert not viewer.is_recording
    with pytest.raises(RuntimeError, match="No Viser recording"):
        viewer.stop_recording()
