"""Tests for ``eden.extensions.pose_estimation.gvhmr.estimator.GvhmrEstimator``.

Recovery needs GVHMR (``hmr4d``), licence-gated checkpoints and a GPU, so ``hmr4d`` is a fake module holding
the two entry points the estimator calls; the videos are real, written with PyAV.
"""

from __future__ import annotations

import os
import sys
import types

import av
import numpy as np
import pytest
import torch

from eden.extensions.pose_estimation import PoseEstimatorBase
from eden.extensions.pose_estimation.gvhmr.estimator import GvhmrEstimator
from eden.options.extensions.pose_estimation import PoseEstimatorOptions

NUM_FRAMES = 7
CHECKPOINT_ROOT_ENV = "GVHMR_CHECKPOINT_ROOT"


def _write_video(path, fps):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for _ in range(4):
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), np.uint8), format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


@pytest.fixture
def video(tmp_path):
    return _write_video(tmp_path / "clip30.mp4", 30)


class _FakeGvhmr:
    """``hmr4d.demo.GVHMR``: records what each ``recover`` saw, and returns zero SMPL-X parameters."""

    def __init__(self, calls, warnings_so_far, ckpt_path=None, checkpoint_root=None):
        self._calls, self._warnings_so_far = calls, warnings_so_far

    def recover(self, video_path, **kwargs):
        self._calls.append(
            {
                "video_path": video_path,
                "kwargs": kwargs,
                "checkpoint_root_env": os.environ.get(CHECKPOINT_ROOT_ENV),
                "warnings_before": self._warnings_so_far(),
            }
        )
        zeros = torch.zeros
        return {
            "smpl_params_global": {
                "global_orient": zeros(NUM_FRAMES, 3),
                "body_pose": zeros(NUM_FRAMES, 63),
                "betas": zeros(NUM_FRAMES, 10),
                "transl": zeros(NUM_FRAMES, 3),
            },
            "smpl_params_incam": {"global_orient": zeros(NUM_FRAMES, 3)},
            "K_fullimg": zeros(NUM_FRAMES, 3, 3),
            "output_dir": "/tmp/gvhmr/out",
        }


class _FakeSkeleton:
    def get_skeleton(self, betas):
        joints = torch.zeros(betas.shape[0], 22, 3)
        joints[:, 0] = torch.tensor([0.0, -0.2, 0.0])  # the rest pelvis
        return joints


@pytest.fixture
def hmr4d(monkeypatch, eden_caplog):
    """Install a fake ``hmr4d``; yields the list of ``recover`` calls it saw."""
    calls: list[dict] = []

    def warnings_so_far():
        return sum(r.levelname == "WARNING" for r in eden_caplog.records)

    demo = types.ModuleType("hmr4d.demo")
    demo.GVHMR = lambda **kwargs: _FakeGvhmr(calls, warnings_so_far, **kwargs)
    smplx_utils = types.ModuleType("hmr4d.utils.smplx_utils")
    smplx_utils.make_smplx = lambda name: _FakeSkeleton()
    for name, module in {
        "hmr4d": types.ModuleType("hmr4d"),
        "hmr4d.demo": demo,
        "hmr4d.utils": types.ModuleType("hmr4d.utils"),
        "hmr4d.utils.smplx_utils": smplx_utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return calls


@pytest.fixture
def no_hmr4d(monkeypatch):
    monkeypatch.setitem(sys.modules, "hmr4d", None)  # an import of it raises ImportError


def test_a_missing_backend_names_the_extra_before_the_video_is_read(no_hmr4d, video):
    estimator = GvhmrEstimator(options=GvhmrEstimator.configure())
    with pytest.raises(ImportError, match=r"eden-world\[pose\]"):
        estimator.estimate(video)


def test_construction_needs_no_backend_and_close_is_idempotent(no_hmr4d):
    """The checkpoint loads on the first ``estimate()``, so building the estimator stays cheap."""
    estimator = GvhmrEstimator(options=GvhmrEstimator.configure())
    estimator.close()
    estimator.close()


def test_the_options_reject_a_typo():
    with pytest.raises(ValueError, match="staic_cam"):
        GvhmrEstimator.configure(staic_cam=True)


def test_configure_carries_both_the_shared_and_the_backend_fields():
    options = GvhmrEstimator.configure(checkpoint_root="/ckpts", motion_fps=30.0, static_cam=True, f_mm=28.0)

    assert (options.checkpoint_root, options.motion_fps) == ("/ckpts", 30.0)  # shared, from PoseEstimatorBase
    assert (options.static_cam, options.f_mm) == (True, 28.0)  # GVHMR's own
    assert options.use_dpvo is False


def test_the_shared_option_defaults_match_the_base_class():
    """The shared fields are declared twice; a default drifting on one side is silent."""
    for name, field in PoseEstimatorOptions.model_fields.items():
        assert getattr(PoseEstimatorBase, name) == field.default, name


def test_a_motion_fps_other_than_gvhmrs_own_rate_is_refused():
    """``--fps 60`` on a 60 fps source relabels a clip whose root motion GVHMR integrated at 30."""
    with pytest.raises(ValueError, match="always expresses its output at 30 fps"):
        GvhmrEstimator(options=GvhmrEstimator.configure(motion_fps=60.0))


def test_estimate_refuses_an_unknown_override_and_a_missing_video(hmr4d, video, tmp_path):
    estimator = GvhmrEstimator(options=GvhmrEstimator.configure())
    with pytest.raises(TypeError, match="statuc_cam"):
        estimator.estimate(video, statuc_cam=True)
    with pytest.raises(FileNotFoundError):
        estimator.estimate(tmp_path / "nope.mp4")
    assert hmr4d == []


def test_a_source_fps_mismatch_warns_before_the_pipeline_runs(hmr4d, tmp_path, eden_caplog):
    """A 60 fps source gives a 2x time-warped clip; saying so after a full run is too late."""
    estimator = GvhmrEstimator(options=GvhmrEstimator.configure())

    result = estimator.estimate(_write_video(tmp_path / "clip60.mp4", 60))

    assert "time-warped by 2.00x" in eden_caplog.text
    assert hmr4d[0]["warnings_before"] == 1
    assert result["video_fps"] == 60.0


def test_a_thirty_fps_source_does_not_warn(hmr4d, video, eden_caplog):
    GvhmrEstimator(options=GvhmrEstimator.configure()).estimate(video)
    assert "time-warped" not in eden_caplog.text


@pytest.mark.parametrize(
    ("before", "root", "during"), [("/pre-existing", "/ckpts", "/ckpts"), (None, "/ckpts", "/ckpts"), (None, "", None)]
)
def test_the_checkpoint_root_is_set_for_the_call_and_restored(hmr4d, video, monkeypatch, before, root, during):
    """GVHMR resolves every weight through a process-wide variable; an empty root keeps its own resolution."""
    if before is None:
        monkeypatch.setenv(CHECKPOINT_ROOT_ENV, "x")  # registers a restore of the unset state
        monkeypatch.delenv(CHECKPOINT_ROOT_ENV)
    else:
        monkeypatch.setenv(CHECKPOINT_ROOT_ENV, before)

    GvhmrEstimator(options=GvhmrEstimator.configure(checkpoint_root=root)).estimate(video)

    assert hmr4d[0]["checkpoint_root_env"] == during
    assert os.environ.get(CHECKPOINT_ROOT_ENV) == before


def test_estimate_clip_returns_the_z_up_amass_clip(hmr4d, video):
    clip = GvhmrEstimator(options=GvhmrEstimator.configure(static_cam=True)).estimate_clip(video)

    assert clip["poses"].shape == (NUM_FRAMES, 165)
    assert hmr4d[0]["kwargs"]["static_cam"] is True
    assert str(clip["source_video"]) == str(video)
    # The rest pelvis J0 sits 0.2 m down Y-up; rotated rigidly it sits 0.2 m down Z-up: J0 + trans == R @ J0.
    pelvis_z_up = np.array([0.0, -0.2, 0.0]) + clip["trans"]
    np.testing.assert_allclose(pelvis_z_up, np.tile([0.0, 0.0, -0.2], (NUM_FRAMES, 1)), atol=1e-5)
