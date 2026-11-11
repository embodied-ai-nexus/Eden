"""Tests for ``eden.utils.misc``: registry-key casing, env-index sanitising and selection, Genesis index forms, Eden's own provenance, and small conversion helpers."""

import json
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import pytest
import torch

import eden as en
import eden.utils.misc as misc
from eden.utils.distribution import DISTRIBUTION_NAME, get_eden_version
from eden.utils.misc import (
    DofPermutation,
    DofRuns,
    as_genesis_control_index,
    as_genesis_index,
    assign_over_envs,
    contiguous_range,
    env_indices,
    get_assets_dir,
    get_cache_dir,
    describe_provenance_change,
    format_provenance,
    get_eden_commit,
    get_eden_provenance,
    get_eden_working_tree_dirty,
    get_package_provenance,
    invert_segmentation_idx_dict,
    mean_over_envs,
    sanitize_envs_idx,
    sanitize_run_name,
    sharing_env_indices,
    string_to_sha256_hex,
    sum_over_envs,
    to_jsonish,
    to_snake_case,
)


@pytest.mark.parametrize(
    ("text", "digest"),
    [
        ("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        ("hello", "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"),
        # Non-ASCII pins the utf-8 encode; latin-1/utf-16 would give another digest.
        ("你好", "670d9743542cae3ea7ebe36af56bd53648b0a1126162e78d81a32934a711302e"),
    ],
)
def test_string_to_sha256_hex_matches_the_published_utf8_digest(text, digest):
    assert string_to_sha256_hex(text) == digest


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("HelloWorld", "hello_world"),
        ("Hello", "hello"),
        ("hello", "hello"),
        ("hello_world", "hello_world"),
        ("", ""),
        # Consecutive capitals collapse into one word, not one word per letter.
        ("HTTPServer", "http_server"),
        ("XMLParser", "xml_parser"),
        ("HTTP", "http"),
        ("myHTMLParser", "my_html_parser"),
        ("getHTTPResponseCode", "get_http_response_code"),
        ("ABCDef", "abc_def"),
        # Digits stay attached to the word they follow (see docs/guides/terms.md).
        ("Test123Case", "test123_case"),
        ("HTML5Parser", "html5_parser"),
        ("A", "a"),
        ("AB", "ab"),
        ("Ab", "ab"),
        ("testA", "test_a"),
        ("testAB", "test_ab"),
        ("Already_SnakeCase", "already_snake_case"),
    ],
)
def test_to_snake_case(value, expected):
    """Registry keys are derived with it, so a change here silently renames every auto-named term."""
    assert to_snake_case(value) == expected


class TestSanitizeEnvsIdx:
    """Bad env indices raise a readable ``IndexError`` before they can reach a kernel."""

    @pytest.mark.parametrize(
        "envs_idx",
        [
            3,
            np.int64(3),
            [3],
            (0, 3),
            range(4),
            range(0, 5, 2),
            range(3, -1, -1),
            np.array([3]),
            torch.tensor([3]),
            [-3],
        ],
    )
    def test_out_of_range_raises(self, envs_idx):
        with pytest.raises(IndexError, match="out of range for num_envs=2"):
            sanitize_envs_idx(envs_idx, num_envs=2)

    @pytest.mark.parametrize(
        ("envs_idx", "expected"),
        [
            (0, [0]),
            (np.int64(1), [1]),
            (-1, [1]),
            (-2, [0]),
            ([0, 1], [0, 1]),
            ((-1,), [1]),
            (np.array([1]), [1]),
            (torch.tensor([0, 1]), [0, 1]),
            (range(2), [0, 1]),
            (range(1, -1, -1), [1, 0]),
            ([], []),
        ],
    )
    def test_in_range_selectors_become_index_tensors(self, envs_idx, expected):
        assert sanitize_envs_idx(envs_idx, num_envs=2).tolist() == expected

    def test_the_message_names_the_argument_and_the_index(self):
        with pytest.raises(IndexError, match="custom_name contains index 9"):
            sanitize_envs_idx([9], num_envs=4, name="custom_name")

    @pytest.mark.parametrize(("mask", "expected"), [(np.array([True]), [0]), (np.array([True, True, False]), [0, 1])])
    def test_a_numpy_bool_mask_is_not_range_checked_as_zeros_and_ones(self, mask, expected):
        # np.array([True]) with num_envs=1 used to raise, read as the integer index 1.
        assert sanitize_envs_idx(mask, num_envs=int(mask.size)).tolist() == expected

    @pytest.mark.parametrize("envs_idx", [True, False, np.True_, np.False_])
    def test_a_bare_bool_raises_instead_of_selecting_env_one_or_zero(self, envs_idx):
        # bool subclasses int: Genesis read True as the index 1, out of range here and unchecked.
        with pytest.raises(IndexError, match=rf"envs_idx={envs_idx!r} is a bool"):
            sanitize_envs_idx(envs_idx, num_envs=1)

    @pytest.mark.parametrize("envs_idx", [[True, False], (False, True), [np.True_, np.False_], [0, True]])
    def test_a_list_of_bools_raises_instead_of_reading_as_indices_one_and_zero(self, envs_idx):
        with pytest.raises(IndexError, match=r"holds the bool .* at position"):
            sanitize_envs_idx(envs_idx, num_envs=2)

    def test_zero_envs_reject_any_index_but_accept_an_empty_selector(self):
        with pytest.raises(IndexError):
            sanitize_envs_idx([0], num_envs=0)
        assert sanitize_envs_idx([], num_envs=0).tolist() == []

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for the on-device skip path")
    def test_an_on_device_tensor_is_trusted_without_a_host_check(self):
        # Checking it would cost a GPU sync on the hot reset path.
        sanitize_envs_idx(torch.tensor([99], device="cuda"), num_envs=2)

    def test_a_bool_mask_of_the_wrong_length_raises(self):
        with pytest.raises(IndexError, match="boolean mask has length 3, expected num_envs=2"):
            sanitize_envs_idx(torch.tensor([True, False, True]), num_envs=2)

    def test_a_bool_mask_passes_through(self):
        mask = torch.tensor([True, False])
        assert torch.equal(sanitize_envs_idx(mask, num_envs=2), mask)

    def test_none_and_slices_stay_slices_clamped_to_num_envs(self):
        assert sanitize_envs_idx(None, num_envs=4) == slice(None)
        assert sanitize_envs_idx(slice(0, 10), num_envs=4) == slice(0, 4, 1)


class TestInvertSegmentationIdxDict:
    """Genesis maps segmentation id -> scene element; at link/geom level one entity owns several ids."""

    @pytest.mark.parametrize(
        ("mapping", "expected"),
        [
            ({1: 7, 2: 9}, {7: [1], 9: [2]}),
            ({1: (7, 0), 2: (7, 1), 3: (7, 2), 4: (9, 0)}, {7: [1, 2, 3], 9: [4]}),
            ({1: (7, 0, 0), 2: (7, 0, 1), 3: (7, 1, 0)}, {7: [1, 2, 3]}),
            ({1: [7, 0], 2: [7, 1]}, {7: [1, 2]}),
            ({}, {}),
        ],
        ids=["entity-level", "link-level", "geom-level", "list-elements", "empty"],
    )
    def test_keeps_every_id_per_entity(self, mapping, expected):
        assert invert_segmentation_idx_dict(mapping) == expected


class TestGetCacheDir:
    def test_defaults_under_the_home_cache_directory(self, monkeypatch):
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        monkeypatch.setenv("HOME", "/home/somebody")
        assert get_cache_dir() == "/home/somebody/.cache/eden"

    def test_parts_are_joined_onto_the_root(self, monkeypatch):
        monkeypatch.setenv("XDG_CACHE_HOME", "/scratch/cache")
        assert get_cache_dir("beyondmimic", "processed.pt") == "/scratch/cache/eden/beyondmimic/processed.pt"

    def test_xdg_cache_home_is_read_on_every_call(self, monkeypatch):
        monkeypatch.setenv("XDG_CACHE_HOME", "/scratch/a")
        first = get_cache_dir("x")
        monkeypatch.setenv("XDG_CACHE_HOME", "/scratch/b")
        assert get_cache_dir("x") != first

    def test_an_empty_xdg_cache_home_falls_back_to_home(self, monkeypatch):
        monkeypatch.setenv("XDG_CACHE_HOME", "")
        monkeypatch.setenv("HOME", "/home/somebody")
        assert get_cache_dir() == "/home/somebody/.cache/eden"


class TestGetAssetsDir:
    """The download root: ``EDEN_ASSETS_DIR`` when set, else the per-user cache; read on every call."""

    def test_without_the_env_var_it_is_the_user_cache(self, user_assets_dir):
        assert get_assets_dir() == str(user_assets_dir)
        assert get_assets_dir("ycb", "models") == str(user_assets_dir / "ycb" / "models")

    def test_an_empty_part_is_the_root_itself_not_the_root_with_a_trailing_separator(self, user_assets_dir):
        assert get_assets_dir("") == str(user_assets_dir)
        assert get_assets_dir("", "policies") == str(user_assets_dir / "policies")

    def test_a_source_checkout_eden_assets_dir_is_not_read(self, user_assets_dir, monkeypatch, tmp_path):
        package = tmp_path / "checkout" / "eden"
        (package / "assets" / ".cache" / "huggingface").mkdir(parents=True)
        monkeypatch.setattr(misc, "get_src_dir", lambda: str(package))
        assert get_assets_dir() == str(user_assets_dir)

    def test_the_env_var_moves_it(self, user_assets_dir, monkeypatch, tmp_path):
        monkeypatch.setenv("EDEN_ASSETS_DIR", str(tmp_path / "shared"))
        assert get_assets_dir() == str(tmp_path / "shared")
        assert get_assets_dir("policies") == str(tmp_path / "shared" / "policies")

    def test_a_tilde_in_the_env_var_is_expanded(self, user_assets_dir, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setenv("EDEN_ASSETS_DIR", "~/assets")
        assert get_assets_dir() == str(tmp_path / "home" / "assets")

    def test_a_relative_env_var_is_refused_with_the_absolute_one_to_export(
        self, user_assets_dir, monkeypatch, tmp_path
    ):
        """Read on every call, a relative root would move whenever the working directory does."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("EDEN_ASSETS_DIR", "relative")
        with pytest.raises(
            ValueError, match=rf"relative path.*export EDEN_ASSETS_DIR={re.escape(str(tmp_path / 'relative'))}$"
        ):
            get_assets_dir()

    def test_the_relative_path_check_is_only_for_the_env_var(self, user_assets_dir, monkeypatch, tmp_path):
        """Unset, the root is ``get_cache_dir``'s, which a relative ``XDG_CACHE_HOME`` leaves as it was."""
        monkeypatch.setenv("XDG_CACHE_HOME", "relcache")
        assert get_assets_dir("policies") == str(Path("relcache") / "eden" / "assets" / "policies")

    def test_an_empty_env_var_is_unset(self, user_assets_dir, monkeypatch):
        monkeypatch.setenv("EDEN_ASSETS_DIR", "")
        assert get_assets_dir() == str(user_assets_dir)

    def test_the_env_var_is_read_on_every_call(self, user_assets_dir, monkeypatch, tmp_path):
        assert get_assets_dir() == str(user_assets_dir)
        monkeypatch.setenv("EDEN_ASSETS_DIR", str(tmp_path / "shared"))
        assert get_assets_dir() == str(tmp_path / "shared")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (torch.tensor([1, 2]), [1, 2]),
        (np.array([[1, 2], [3, 4]]), [[1, 2], [3, 4]]),
        ((2, 1942), [2, 1942]),  # torch.save round-trips tuples as lists
        ({"a": (1, torch.tensor([2]))}, {"a": [1, [2]]}),
        (None, None),
        (1.5, 1.5),
        ("x", "x"),
    ],
)
def test_to_jsonish_normalises_values_that_must_compare_equal_after_torch_save(value, expected):
    assert to_jsonish(value) == expected


def test_to_jsonish_turns_numpy_scalars_into_python_scalars():
    value = to_jsonish(np.int64(3))
    assert value == 3 and type(value) is int


# ---------------------------------------------------------------------------
# Sync-free env selection
# ---------------------------------------------------------------------------

_NUM_ENVS = 6
_SELECTORS = {
    "mask": torch.tensor([True, False, True, True, False, False]),
    "slice": slice(1, 4),
    "index": torch.tensor([4, 0, 2]),
    "empty mask": torch.zeros(_NUM_ENVS, dtype=torch.bool),
    "0-dim index": torch.tensor(3),
}
# rows 0..5 of an arange(12).view(6, 2); the expected selection for each selector, by hand.
_SELECTED_ROWS = {"mask": [0, 2, 3], "slice": [1, 2, 3], "index": [4, 0, 2], "empty mask": [], "0-dim index": [3]}


@pytest.mark.parametrize("form", list(_SELECTORS))
def test_reductions_over_the_selected_envs(form):
    values = torch.arange(_NUM_ENVS * 2, dtype=torch.float).view(_NUM_ENVS, 2)
    rows = _SELECTED_ROWS[form]
    expected_sum = torch.tensor([sum(2 * r for r in rows), sum(2 * r + 1 for r in rows)], dtype=torch.float)

    torch.testing.assert_close(sum_over_envs(values, _SELECTORS[form]), expected_sum)
    # zero, not NaN, when nothing is selected
    torch.testing.assert_close(mean_over_envs(values, _SELECTORS[form]), expected_sum / max(len(rows), 1))
    dones = values[:, 0] > 3.0  # envs 2..5
    assert int(sum_over_envs(dones, _SELECTORS[form])) == sum(r >= 2 for r in rows)


@pytest.mark.parametrize("form", list(_SELECTORS))
def test_assign_over_envs_writes_only_the_selected_rows(form):
    target = torch.zeros(_NUM_ENVS, 2)
    source = torch.arange(_NUM_ENVS * 2, dtype=torch.float).view(_NUM_ENVS, 2)

    assign_over_envs(target, _SELECTORS[form], source)

    expected = torch.zeros(_NUM_ENVS, 2)
    for row in _SELECTED_ROWS[form]:
        expected[row] = source[row]
    torch.testing.assert_close(target, expected)


def test_the_helpers_are_sync_free_for_a_mask(count_cuda_syncs):
    mask = _SELECTORS["mask"].cuda()
    values = torch.rand(_NUM_ENVS, 3, device="cuda")
    target = torch.zeros(_NUM_ENVS, 3, device="cuda")

    assert count_cuda_syncs(lambda: sum_over_envs(values, mask)) == 0
    assert count_cuda_syncs(lambda: mean_over_envs(values, mask)) == 0
    assert count_cuda_syncs(lambda: assign_over_envs(target, mask, values)) == 0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("plain", "plain"),
        ("a/b:c d", "a-b-c-d"),  # separators and whitespace collapse to one "-"
        ("  spaced  ", "spaced"),
        ("///", ""),  # all unsafe: no middle segment at all
        ("", ""),
        ("x" * 100, "x" * 64),  # length-capped
    ],
)
def test_sanitize_run_name(raw, expected):
    assert sanitize_run_name(raw) == expected


# ---------------------------------------------------------------------------
# Genesis index forms
# ---------------------------------------------------------------------------

MASK = [True, False, True, True, False, False]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def test_contiguous_range():
    """One link or a run of DOFs becomes a range, never a one-element tensor Genesis would upload."""
    assert contiguous_range(torch.arange(6, 35)) == range(6, 35)
    assert as_genesis_index(torch.tensor([7])) == range(7, 8)
    assert contiguous_range([0, 1, 2]) == range(3)
    reordered = torch.tensor([0, 2, 1])
    assert contiguous_range(reordered) is None and as_genesis_index(reordered) is reordered  # a reordered dofs_name
    assert contiguous_range(torch.tensor([], dtype=torch.long)) is None


@pytest.mark.parametrize("dofs", [[-1], [-2, -1], [-1, 0], [-3, -2]])
def test_selections_counted_from_the_end_are_never_ranges(dofs):
    """Genesis makes ``range(-2, 0)`` the empty ``slice(-2, 0)``."""
    idx = torch.tensor(dofs)
    assert contiguous_range(idx) is None
    assert as_genesis_index(idx) is idx and as_genesis_control_index(idx) is idx


@pytest.mark.parametrize("device", DEVICES)
def test_env_indices_turns_a_mask_into_its_ids_and_passes_the_rest(device):
    assert env_indices(torch.tensor(MASK, device=device)).tolist() == [0, 2, 3]
    ids = torch.tensor([4, 1], device=device)
    assert env_indices(ids) is ids
    assert env_indices(slice(1, 3)) == slice(1, 3)


def test_env_indices_is_one_sync_and_the_gathers_through_it_none(count_cuda_syncs):
    mask = torch.tensor(MASK, device="cuda")
    values = torch.rand(len(MASK), 3, device="cuda")
    assert count_cuda_syncs(lambda: env_indices(mask)) == 1
    ids = env_indices(mask)
    assert count_cuda_syncs(lambda: values[ids]) == 0
    assert count_cuda_syncs(lambda: values.__setitem__(ids, values[ids] + 1.0)) == 0


def test_inside_sharing_env_indices_its_mask_costs_no_sync_for_ids_or_count(count_cuda_syncs):
    """The auto-reset's mask: one ``nonzero()`` on entry, then every consumer's ids and the reset gate's count are free."""
    mask = torch.tensor(MASK, device="cuda")
    scope = sharing_env_indices(mask)
    assert count_cuda_syncs(scope.__enter__) == 1
    try:
        assert count_cuda_syncs(lambda: env_indices(mask)) == 0
        assert env_indices(mask).tolist() == [0, 2, 3]
        assert count_cuda_syncs(lambda: sanitize_envs_idx(mask, len(MASK), return_n_envs=True)) == 0
        assert sanitize_envs_idx(mask, len(MASK), return_n_envs=True)[1] == 3
    finally:
        scope.__exit__(None, None, None)


@pytest.mark.parametrize("device", DEVICES)
def test_sharing_env_indices_shares_only_the_mask_it_was_opened_on(device):
    """A mask derived from it, or an equal copy, selects other envs as far as the shared ids know: its own nonzero()."""
    mask = torch.tensor(MASK, device=device)
    with sharing_env_indices(mask):
        assert env_indices(mask & torch.tensor([True] * 3 + [False] * 3, device=device)).tolist() == [0, 2]
        assert env_indices(~mask).tolist() == [1, 4, 5]
        assert sanitize_envs_idx(~mask, len(MASK), return_n_envs=True)[1] == 3
    # Closed: the mask is a mask again, counted and converted on its own.
    assert env_indices(mask).tolist() == [0, 2, 3]


def test_sharing_env_indices_passes_anything_but_a_mask_through():
    ids = torch.tensor([4, 1])
    for envs_idx in (None, slice(1, 3), ids):
        with sharing_env_indices(envs_idx):
            assert env_indices(ids) is ids


def test_sharing_env_indices_nests_and_restores_the_outer_mask():
    outer, inner = torch.tensor(MASK), torch.tensor([not m for m in MASK])
    with sharing_env_indices(outer):
        outer_ids = env_indices(outer)
        with sharing_env_indices(inner):
            assert env_indices(inner).tolist() == [1, 4, 5]
        assert env_indices(outer) is outer_ids


@pytest.mark.parametrize(
    ("dofs", "expected"),
    [
        ([2, 3, 4], range(2, 5)),
        ([0, 3], DofRuns([(range(0, 1), slice(0, 1)), (range(3, 4), slice(1, 2))])),
        ([0, 3, 4, 7], DofRuns([(range(0, 1), slice(0, 1)), (range(3, 5), slice(1, 3)), (range(7, 8), slice(3, 4))])),
    ],
)
def test_control_index_is_a_range_or_the_runs_of_one(dofs, expected):
    resolved = as_genesis_control_index(torch.tensor(dofs))
    assert type(resolved) is type(expected) and resolved == expected


def test_control_index_writes_a_reordered_block_through_its_range():
    """A quadruped's leg-major joints over a joint-major asset: every DOF of 0..11 once, reordered."""
    dofs = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
    resolved = as_genesis_control_index(torch.tensor(dofs))
    assert isinstance(resolved, DofPermutation) and resolved.dofs == range(0, 12)
    written = (torch.arange(12.0).unsqueeze(0) * 10.0).index_select(-1, resolved.order)
    assert written[0].tolist() == [10.0 * dofs.index(d) for d in range(12)]


def test_control_index_keeps_a_tensor_with_many_runs():
    idx = torch.tensor([0, 2, 4, 6, 8, 10])
    assert as_genesis_control_index(idx) is idx


# --- Eden's own provenance -------------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str, cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd).decode("utf-8").strip()


@pytest.mark.skipif(misc._eden_checkout() is None, reason="not run from a git checkout of Eden")
def test_in_a_checkout_the_commit_is_head_even_with_a_stray_egg_info_first_on_the_path(tmp_path, monkeypatch):
    """An in-tree setuptools build leaves an ``eden_world.egg-info`` in the repo root, which ``python -m`` puts first on
    ``sys.path``: looked up by distribution name, the commit read back as the bare version (``0.1.0``).
    """
    egg = tmp_path / "eden_world.egg-info"
    egg.mkdir()
    (egg / "PKG-INFO").write_text(f"Metadata-Version: 2.1\nName: {DISTRIBUTION_NAME}\nVersion: 0.1.0\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert metadata.distribution(DISTRIBUTION_NAME).version == "0.1.0"  # the shadow is in place

    assert get_eden_commit() == _git("rev-parse", "HEAD", cwd=REPO_ROOT)


@pytest.mark.skipif(misc._eden_checkout() is None, reason="not run from a git checkout of Eden")
def test_in_a_checkout_eden_records_the_release_and_the_commit_each_under_its_own_name():
    """``get_eden_version`` stamps what should change only with a release (the TCRS cache), so a commit must not move
    it; the commit has a field of its own.
    """
    assert get_eden_provenance() == {
        "version": metadata.version(DISTRIBUTION_NAME),
        "commit": _git("rev-parse", "HEAD", cwd=REPO_ROOT),
    }


_SHA = "c246ebd90d66c2c6f2e2e95f53ffa322a2176f33"


@pytest.mark.parametrize(
    ("provenance", "shown"),
    [
        ({"version": "0.1.0", "commit": _SHA}, "0.1.0 (c246ebd)"),
        # A version is shown whole: cut to 7 characters, 0.1.0rc1 would read 0.1.0rc and 0.10.0rc1 0.10.0r.
        ({"version": "0.10.0rc1", "commit": None}, "0.10.0rc1"),
        ({"version": None, "commit": _SHA}, "c246ebd"),
        ({"version": None, "commit": None}, "unknown"),
    ],
)
def test_a_log_line_shows_the_version_and_a_short_commit(provenance, shown):
    assert format_provenance(provenance) == shown


@pytest.mark.parametrize(
    ("saved", "current", "change"),
    [
        # Two clones: the commits decide, even at one version.
        (
            {"version": "0.1.0", "commit": "a" * 40},
            {"version": "0.1.0", "commit": "b" * 40},
            "saved 0.1.0 (aaaaaaa), running 0.1.0 (bbbbbbb)",
        ),
        ({"version": "0.1.0", "commit": "a" * 40}, {"version": "0.2.0", "commit": "a" * 40}, None),
        # A clone and an install: only the versions compare.
        ({"version": "0.1.0", "commit": "a" * 40}, {"version": "0.1.0", "commit": None}, None),
        (
            {"version": "0.1.0", "commit": None},
            {"version": "0.2.0", "commit": "b" * 40},
            "saved 0.1.0, running 0.2.0 (bbbbbbb)",
        ),
        # Nothing to compare: an unknown side, or a file written before these records existed.
        ({"version": None, "commit": None}, {"version": "0.2.0", "commit": None}, None),
        (None, {"version": "0.2.0", "commit": None}, None),
        ("c246ebd90d66c2c6f2e2e95f53ffa322a2176f33", {"version": "0.2.0", "commit": None}, None),
    ],
)
def test_a_saved_record_differs_by_commit_when_both_have_one_else_by_version(saved, current, change):
    assert describe_provenance_change(saved, current) == change


def _installed(monkeypatch, site: Path, direct_url: dict | None = None) -> str:
    """Install a distribution ``eden-provenance-probe 1.2.3`` into ``site`` (on ``sys.path``), with ``direct_url``
    as its PEP 610 record.
    """
    dist_info = site / "eden_provenance_probe-1.2.3.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text("Metadata-Version: 2.1\nName: eden-provenance-probe\nVersion: 1.2.3\n")
    if direct_url is not None:
        (dist_info / "direct_url.json").write_text(json.dumps(direct_url))
    monkeypatch.syspath_prepend(str(site))
    return "eden-provenance-probe"


def test_a_package_from_an_index_records_its_version_and_no_commit(tmp_path, monkeypatch):
    name = _installed(monkeypatch, tmp_path)
    assert get_package_provenance(name) == {"version": "1.2.3", "commit": None}


def test_a_git_install_records_the_commit_it_was_installed_from(tmp_path, monkeypatch):
    url = {"url": "https://github.com/x/y", "vcs_info": {"vcs": "git", "commit_id": _SHA, "requested_revision": "main"}}
    name = _installed(monkeypatch, tmp_path, url)
    assert get_package_provenance(name) == {"version": "1.2.3", "commit": _SHA}


def test_an_editable_install_records_its_checkouts_head_or_none_outside_a_repository(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    name = _installed(monkeypatch, tmp_path / "site", {"url": checkout.as_uri(), "dir_info": {"editable": True}})
    assert get_package_provenance(name) == {"version": "1.2.3", "commit": None}

    _repository_with_one_commit(checkout)
    assert get_package_provenance(name) == {"version": "1.2.3", "commit": _git("rev-parse", "HEAD", cwd=checkout)}


def test_a_package_not_installed_records_nothing():
    assert get_package_provenance("eden-no-such-distribution") == {"version": None, "commit": None}


def test_eden_version_is_the_installed_distributions_read_on_access():
    assert en.__version__ == get_eden_version() == metadata.version(DISTRIBUTION_NAME)
    assert "__version__" in dir(en)


def _imported_from(monkeypatch, root: Path) -> Path:
    """Make ``eden.utils.misc`` believe it is imported from ``<root>/eden/utils/misc.py``, a file that exists there."""
    (root / "eden" / "utils").mkdir(parents=True)
    path = root / "eden" / "utils" / "misc.py"
    path.write_text("# eden.utils.misc\n")
    monkeypatch.setattr(misc, "__file__", str(path))
    return path


def _repository_with_one_commit(path: Path, *files: Path) -> list[str]:
    """``git init`` ``path`` and commit ``files`` (none: an empty commit); the ``-c`` overrides, which keep the
    contributor's own signing and hooks out of it, are returned for later commits.
    """
    _git("init", "-q", cwd=path)
    config = ("user.name=t", "user.email=t@t", "commit.gpgsign=false", "core.hooksPath=/dev/null")
    overrides = [arg for setting in config for arg in ("-c", setting)]
    if files:
        _git("add", "--", *map(str, files), cwd=path)
    _git(*overrides, "commit", "-q", "--no-verify", "--allow-empty", "-m", "x", cwd=path)
    return overrides


def test_outside_a_checkout_eden_records_its_version_and_no_commit(tmp_path, monkeypatch):
    _imported_from(monkeypatch, tmp_path / "site-packages")
    assert get_eden_provenance() == {"version": get_eden_version(), "commit": None}
    assert get_eden_working_tree_dirty() is None


def test_a_wheel_in_a_venv_inside_another_repository_does_not_report_that_repositorys_head(tmp_path, monkeypatch):
    """``git rev-parse`` from site-packages would answer for the enclosing project; that HEAD is not Eden's, and the
    wheel's files are not tracked there.
    """
    _repository_with_one_commit(tmp_path)  # HEAD exists, so only the check that it holds Eden can turn the lookup down
    _imported_from(
        monkeypatch,
        tmp_path / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
    )
    assert get_eden_commit() is None
    assert get_eden_working_tree_dirty() is None


def test_eden_vendored_in_a_larger_repository_reports_that_repositorys_head_and_dirty_flag(tmp_path, monkeypatch):
    """Eden copied into a subdirectory of a project's repository is committed there: that HEAD is the code that ran,
    and the dirty flag is that repository's, so an edit outside Eden counts too.
    """
    misc_py = _imported_from(monkeypatch, tmp_path / "third_party")
    app_py = tmp_path / "app.py"
    app_py.write_text("# the enclosing project\n")
    overrides = _repository_with_one_commit(tmp_path, misc_py, app_py)
    assert get_eden_commit() == _git("rev-parse", "HEAD", cwd=tmp_path)
    assert get_eden_working_tree_dirty() is False

    misc_py.write_text("# edited\n")
    assert get_eden_working_tree_dirty() is True
    _git(*overrides, "commit", "-q", "--no-verify", "-am", "y", cwd=tmp_path)
    assert get_eden_commit() == _git("rev-parse", "HEAD", cwd=tmp_path)
    assert get_eden_working_tree_dirty() is False

    app_py.write_text("# edited\n")
    assert get_eden_working_tree_dirty() is True


def test_eden_staged_but_not_yet_committed_in_a_larger_repository_does_not_report_its_head(tmp_path, monkeypatch):
    """``git add`` without a commit: the index knows Eden, but HEAD, the commit that would be reported, does not hold
    it.
    """
    _repository_with_one_commit(tmp_path)
    misc_py = _imported_from(monkeypatch, tmp_path / "third_party")
    _git("add", "--", str(misc_py), cwd=tmp_path)
    assert get_eden_commit() is None
    assert get_eden_working_tree_dirty() is None
