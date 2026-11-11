"""Tests for ``eden.utils.cli.paths``: the ``--src`` / ``--out`` plumbing ``eden retarget`` and ``eden pose`` share."""

from pathlib import Path

import pytest

from eden.utils.cli.paths import iter_src_files, resolve_out


def test_a_single_file_is_its_own_root(tmp_path):
    clip = tmp_path / "clip.npz"
    clip.write_bytes(b"")
    assert iter_src_files(clip, "*.npz") == (True, tmp_path, [clip])


def test_a_directory_is_searched_recursively(tmp_path):
    (tmp_path / "a.npz").write_bytes(b"")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.npz").write_bytes(b"")

    single, root, files = iter_src_files(tmp_path, "*.npz")

    assert not single and root == tmp_path
    assert sorted(files) == [tmp_path / "a.npz", tmp_path / "sub" / "b.npz"]


@pytest.mark.parametrize(
    ("name", "pattern", "message"),
    [("clip.npz", "*.pkl", "is not a .pkl file"), ("nope.npz", "*.npz", "does not exist")],
    ids=["wrong-suffix", "missing"],
)
def test_a_bad_src_exits_with_the_reason(tmp_path, name, pattern, message):
    (tmp_path / "clip.npz").write_bytes(b"")
    with pytest.raises(SystemExit, match=message):
        iter_src_files(tmp_path / name, pattern)


def test_resolve_out_refuses_to_overwrite_its_input(tmp_path):
    """``eden retarget soma`` maps .npz to .npz, so --src == --out would clobber every input."""
    clip = tmp_path / "clip.npz"
    clip.write_bytes(b"")
    with pytest.raises(SystemExit, match="Refusing to write output onto its input"):
        resolve_out(clip, tmp_path, clip, ".npz", single=True)


def test_resolve_out_refuses_a_file_out_for_a_directory_src(tmp_path):
    with pytest.raises(SystemExit, match="looks like a file, but --src is a directory"):
        resolve_out(Path("out.npz"), tmp_path, tmp_path / "a.npz", ".npz", single=False)


def test_a_single_file_out_carries_the_output_suffix(tmp_path):
    """np.savez / pickle append the suffix themselves, so the returned path must already have it."""
    assert resolve_out(tmp_path / "custom.data", tmp_path, tmp_path / "a.npz", ".npz", single=True).suffix == ".npz"


def test_a_directory_out_mirrors_the_src_tree(tmp_path):
    assert resolve_out(Path("/out"), tmp_path, tmp_path / "sub" / "b.npz", ".pkl", single=False) == Path(
        "/out/sub/b.pkl"
    )


def test_an_existing_dotted_directory_is_treated_as_a_directory(tmp_path):
    """``v1.2`` has the suffix ``.2``, which the file/dir heuristic would read as a filename."""
    out_dir = tmp_path / "runs" / "v1.2"
    out_dir.mkdir(parents=True)
    assert resolve_out(out_dir, tmp_path, tmp_path / "sub" / "b.npz", ".pkl", single=False) == out_dir / "sub" / "b.pkl"
