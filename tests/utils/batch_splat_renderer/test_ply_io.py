"""``eden.utils.batch_splat_renderer.ply_io.load_ply``: 3DGS / 2DGS ``.ply`` files into ``GaussianData``.

The package ``__init__`` pulls the gsplat-backed renderer, so these skip without gsplat.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gsplat")
plyfile = pytest.importorskip("plyfile")

from eden.utils.batch_splat_renderer import load_ply  # noqa: E402


def _write_gaussian_ply(path, *, n, n_scales, n_frest, seed=0):
    """Write a synthetic Gaussian ``.ply`` with the fields ``load_ply`` reads, and return the raw record array."""
    rng = np.random.default_rng(seed)
    fields = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", *(f"f_rest_{i}" for i in range(n_frest)), "opacity"]
    fields += [f"scale_{i}" for i in range(n_scales)] + [f"rot_{i}" for i in range(4)]
    arr = np.empty(n, dtype=[(f, "f4") for f in fields])
    for f in fields:
        arr[f] = rng.standard_normal(n).astype(np.float32)
    plyfile.PlyData([plyfile.PlyElement.describe(arr, "vertex")]).write(str(path))
    return arr


@pytest.mark.parametrize(
    ("n_scales", "n_frest", "n_sh"),
    [(3, 45, 16), (2, 45, 16), (3, 0, 1)],
    ids=["3dgs_sh3", "2dgs_padded", "sh0"],  # sh0: e.g. WorldLabs exports
)
def test_load_ply_reads_every_gaussian_field(tmp_path, n_scales, n_frest, n_sh):
    """Shapes follow the file: ``(degree + 1) ** 2`` SH coefficients, DC first.

    A 2DGS file's missing third scale is padded with a near-zero one.
    """
    raw = _write_gaussian_ply(tmp_path / "g.ply", n=5, n_scales=n_scales, n_frest=n_frest)
    g = load_ply(str(tmp_path / "g.ply"), activate=False)

    assert (g.xyz.shape, g.rot.shape, g.scale.shape, g.opacity.shape, g.sh.shape) == (
        (5, 3),
        (5, 4),
        (5, 3),
        (5,),
        (5, n_sh, 3),
    )
    np.testing.assert_allclose(g.sh[:, 0, :], np.stack([raw[f"f_dc_{i}"] for i in range(3)], axis=1), rtol=1e-6)
    if n_scales == 2:
        np.testing.assert_allclose(g.scale[:, 2], 1e-9, atol=1e-12)


@pytest.mark.parametrize("n_frest", [4, 3], ids=["not_a_multiple_of_3", "not_an_sh_degree"])
def test_load_ply_rejects_an_invalid_sh_rest_count(tmp_path, n_frest):
    _write_gaussian_ply(tmp_path / "g.ply", n=2, n_scales=3, n_frest=n_frest)
    with pytest.raises(ValueError, match="f_rest"):
        load_ply(str(tmp_path / "g.ply"))
