"""Tests for ``eden.utils.cli.retarget``: each backend parses its args, then vets its install, before any work."""

import importlib.util
import os
import subprocess
import sys

import pydantic
import pytest

from eden.utils.cli.retarget import (
    _replace,
    _run_dex,
    _run_flashdex_import,
    _run_flashdex_kinematic,
    _run_omniretarget,
    _run_otretarget,
    _run_soma,
    _with_io_retries,
)
from tests.support.soma_retargeter import NOT_INSTALLED, install_soma_stand_in

_HANDLERS = pytest.mark.parametrize(
    "handler",
    [_run_omniretarget, _run_otretarget, _run_soma, _run_dex],
    ids=["omniretarget", "otretarget", "soma", "dex"],
)


@pytest.mark.parametrize(
    "handler",
    [_run_omniretarget, _run_soma, _run_dex, _run_flashdex_import, _run_flashdex_kinematic],
    ids=["omniretarget", "soma", "dex", "flashdex-import", "flashdex-kinematic"],
)
def test_help_reaches_argparse_without_the_backend_installed(handler):
    """An ImportError here would mean the backend import ran before argument parsing."""
    with pytest.raises(SystemExit) as exc:
        handler(["--help"])
    assert exc.value.code == 0


@_HANDLERS
def test_a_missing_src_is_a_usage_error(handler):
    with pytest.raises(SystemExit) as exc:
        handler([])
    assert exc.value.code == 2


def _soma_args(tmp_path, *extra):
    # --src does not exist: the install check must come first, before any file or scene work.
    return ["--src", str(tmp_path / "missing.npz"), "--out", str(tmp_path), "--body-model-path", "m", *extra]


def test_soma_without_its_backend_names_the_install_command(tmp_path, monkeypatch):
    install_soma_stand_in(monkeypatch, tmp_path, backend=NOT_INSTALLED)

    with pytest.raises(
        ImportError, match=r"backend is not installed\. Install it with `uv sync --inexact --extra soma-retargeter"
    ):
        _run_soma(_soma_args(tmp_path))


def test_soma_vets_the_install_for_the_selected_body_model(tmp_path, monkeypatch):
    """py-soma-x is needed by the fit, so the check must see the body model the user picked."""
    install_soma_stand_in(monkeypatch, tmp_path, py_soma_x=NOT_INSTALLED)  # the backend, but no py-soma-x for the fit

    with pytest.raises(ImportError, match="body_model='smplh' is fitted onto the SOMA rig by py-soma-x, which is not"):
        _run_soma(_soma_args(tmp_path, "--body-model", "smplh"))


def test_otretarget_omomo_needs_the_objects_folder(tmp_path):
    with pytest.raises(SystemExit) as exc:
        _run_otretarget(["--source", "omomo", "--src", str(tmp_path), "--out", str(tmp_path), "--body-model-path", "m"])
    assert exc.value.code == 2


@pytest.mark.parametrize("shard", ["3", "4/4", "5/4", "a/b"])
def test_otretarget_rejects_a_malformed_shard(tmp_path, shard, capsys):
    args = ["--source", "omomo", "--src", str(tmp_path), "--out", str(tmp_path), "--body-model-path", "m"]
    with pytest.raises(SystemExit) as exc:
        _run_otretarget([*args, "--objects-dir", str(tmp_path), "--shard", shard])
    assert exc.value.code == 2
    assert f"--shard {shard!r} must be K/N with integers 0 <= K < N" in capsys.readouterr().err


def test_otretarget_supports_need_an_object_track(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        _run_otretarget(
            ["--source", "seed", "--src", str(tmp_path), "--out", str(tmp_path), "--body-model-path", "m", "--supports"]
        )
    assert exc.value.code == 2
    assert "--supports reads supports off an object track; --source seed has no objects." in capsys.readouterr().err


def test_otretarget_clip_list_names_the_clips_it_cannot_find(tmp_path):
    for clip in ("sub1_box_000", "sub1_box_001"):
        (tmp_path / "seq" / clip).mkdir(parents=True)
        (tmp_path / "seq" / clip / "human.npz").touch()
    (tmp_path / "clips.txt").write_text("sub1_box_001\nsub9_lamp_007\n")
    args = [
        "--source",
        "omomo",
        "--src",
        str(tmp_path / "seq"),
        "--out",
        str(tmp_path / "out"),
        "--body-model-path",
        "m",
    ]
    with pytest.raises(
        SystemExit, match=r"1 clip\(s\) not in --src .*: sub9_lamp_007 \(e\.g\. 'sub1_box_000' is one\)"
    ):
        _run_otretarget([*args, "--objects-dir", str(tmp_path), "--clip-list", str(tmp_path / "clips.txt")])


def test_otretarget_refuses_bvh_clips_whose_outputs_would_collide(tmp_path):
    for folder in ("take_1", "take_2"):
        (tmp_path / "seed" / folder).mkdir(parents=True)
        (tmp_path / "seed" / folder / "walk.bvh").touch()
    (tmp_path / "seed" / "take_1" / "run.bvh").touch()
    args = ["--source", "seed", "--src", str(tmp_path / "seed"), "--out", str(tmp_path / "out")]
    with pytest.raises(SystemExit, match=r"1 clip name\(s\) occur in more than one folder.*: walk \(e\.g\. 'walk' at "):
        _run_otretarget([*args, "--body-model-path", "m"])


def test_a_transient_write_error_is_retried_and_any_other_raised_at_once():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise OSError(22, "Invalid argument")  # what a loaded NFS server returned for a valid open

    _with_io_retries(flaky, wait_s=0.0)
    assert len(calls) == 3

    def flaky_for_good():
        calls.append(1)
        raise OSError(5, "Input/output error")

    calls.clear()
    with pytest.raises(OSError, match="Input/output error"):
        _with_io_retries(flaky_for_good, attempts=2, wait_s=0.0)
    assert len(calls) == 2

    def full():
        calls.append(1)
        raise OSError(28, "No space left on device")

    calls.clear()
    with pytest.raises(OSError, match="No space left"):
        _with_io_retries(full, wait_s=10.0)  # not transient: raised at once, no 10 s waits
    assert len(calls) == 1


def test_a_rename_reported_failed_but_done_counts_as_done(tmp_path, monkeypatch):
    src, dst = tmp_path / "a.partial.npz", tmp_path / "a.npz"
    src.write_text("x")

    def rename_then_complain(a, b):
        os.rename(a, b)
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr("eden.utils.cli.retarget.os.replace", rename_then_complain)
    _replace(src, dst)
    assert dst.read_text() == "x" and not src.exists()
    src.write_text("y")
    monkeypatch.setattr("eden.utils.cli.retarget.os.replace", lambda a, b: (_ for _ in ()).throw(OSError(18, "EXDEV")))
    with pytest.raises(OSError, match="EXDEV"):
        _replace(src, dst)


def test_flashdex_flags_override_the_config_file(tmp_path):
    """The flag's dataset, not the config's, is the one the pipeline looks into."""
    config = tmp_path / "fdr.yaml"
    # The SDF step is off so its warp check (flashdexretarget extra) does not fire before the dataset is looked into
    config.write_text(f"dataset: {tmp_path / 'from_config'}\nrobot: sharpa\nsteps: {{sdf: false}}\n")
    with pytest.raises(FileNotFoundError, match="from_flag has neither human_demo/ nor sharpa/motion.pt"):
        _run_flashdex_kinematic(["--config", str(config), "--dataset", str(tmp_path / "from_flag")])


def test_flashdex_override_with_a_misspelt_field_names_the_right_one(tmp_path):
    with pytest.raises(pydantic.ValidationError, match="Did you mean 'attempts'"):
        _run_flashdex_kinematic(["--dataset", str(tmp_path), "--override", "retarget: {atempts: 3}"])


def test_flashdex_override_switching_the_backend_drops_the_other_backends_fields(tmp_path):
    """The config's CoACD seed is not a CuACD field: a name change replaces the group, so the options validate and the
    run reaches CuACD's own install check.
    """
    config = tmp_path / "fdr.yaml"
    config.write_text("convex: {name: coacd, seed: 2}\n")
    if importlib.util.find_spec("cuacd") is not None:
        pytest.skip("cuacd is installed: the run would decompose")
    with pytest.raises(ImportError, match="convex.name='cuacd' needs the `cuacd` package"):
        _run_flashdex_kinematic(
            ["--dataset", str(tmp_path), "--config", str(config), "--override", "convex: {name: cuacd}"]
        )


def test_flashdex_override_switching_from_an_implicit_backend_drops_its_fields(tmp_path):
    """A config group without a name is the default backend's (CoACD): switching to CuACD replaces it too."""
    config = tmp_path / "fdr.yaml"
    config.write_text("convex: {seed: 2}\n")
    if importlib.util.find_spec("cuacd") is not None:
        pytest.skip("cuacd is installed: the run would decompose")
    with pytest.raises(ImportError, match="convex.name='cuacd' needs the `cuacd` package"):
        _run_flashdex_kinematic(
            ["--dataset", str(tmp_path), "--config", str(config), "--override", "convex: {name: cuacd}"]
        )


def test_flashdex_import_help_imports_neither_torch_nor_the_dataset_loaders():
    """--help must answer before the backend loads (the module docstring's promise and the CLI's startup budget)."""
    code = (
        "import sys\n"
        "from eden.utils.cli.retarget import _run_flashdex_import\n"
        "try:\n"
        "    _run_flashdex_import(['--help'])\n"
        "except SystemExit:\n"
        "    pass\n"
        "print(sorted(m for m in ('torch', 'genesis', 'scipy', 'eden.datasets') if m in sys.modules))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip().splitlines()[-1] == "[]"


def test_flashdex_import_lists_every_source_the_importer_reads():
    from eden.extensions.retargeting.flashdexretarget.importer import SOURCE_DATASETS
    from eden.utils.cli.retarget import _FLASHDEX_SOURCES

    assert sorted(_FLASHDEX_SOURCES) == sorted(SOURCE_DATASETS)
