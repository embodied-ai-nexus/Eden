"""Load standard 3DGS / 2DGS ``.ply`` files into a :class:`GaussianData`.

Adapted from https://github.com/discoverse-dev/GaussianRenderer (``load_ply_3dgs``),
trimmed to the standard binary 3DGS/2DGS path (the upstream SuperSplat loader is not vendored).
"""

import numpy as np
from plyfile import PlyData

from .gaussiandata import GaussianData


def _sorted_props(props, prefix):
    names = [p.name for p in props if p.name.startswith(prefix)]
    return sorted(names, key=lambda x: int(x.split("_")[-1]))


def load_ply(path: str, *, activate: bool = True) -> GaussianData:
    """Load a 3DGS/2DGS PLY into a numpy :class:`GaussianData` (sh as ``(N, K, 3)``).

    Parameters
    ----------
    path : str
        Path to a standard binary 3DGS/2DGS ``.ply`` (``x,y,z``, ``f_dc_*``,
        ``f_rest_*``, ``opacity``, ``scale_*``, ``rot_*``).
    activate : bool
        Apply the stored-parameter activations (``exp`` on scale, ``sigmoid`` on
        opacity, quaternion normalization) so the returned values are the ones
        gsplat consumes directly. Set ``False`` to inspect raw stored params.

    Returns
    -------
    GaussianData
        ``xyz (N,3)``, ``rot (N,4) wxyz``, ``scale (N,3)``, ``opacity (N,)``,
        ``sh (N, K, 3)`` with ``K = (sh_degree + 1) ** 2``. A 2DGS PLY (only two
        ``scale_*`` fields) gets a near-zero third scale so it reads as a flat
        3DGS ellipsoid.
    """
    plydata = PlyData.read(path)
    el = plydata.elements[0]
    props = el.properties

    xyz = np.stack([np.asarray(el["x"]), np.asarray(el["y"]), np.asarray(el["z"])], axis=1).astype(np.float32)
    opacities = np.asarray(el["opacity"]).astype(np.float32)

    features_dc = np.stack(
        [np.asarray(el["f_dc_0"]), np.asarray(el["f_dc_1"]), np.asarray(el["f_dc_2"])], axis=1
    )  # (N, 3)

    extra_names = _sorted_props(props, "f_rest_")
    n_extra = len(extra_names)
    if n_extra % 3 != 0:
        raise ValueError(f"{path}: expected f_rest_* count divisible by 3, got {n_extra}")
    n_coeffs = n_extra // 3  # rest coefficients per channel
    sh_degree = int(round(np.sqrt(n_coeffs + 1))) - 1
    if (sh_degree + 1) ** 2 != n_coeffs + 1:
        raise ValueError(f"{path}: f_rest count {n_extra} is not a valid SH degree (deg1=9, deg2=24, deg3=45)")

    features_extra = np.zeros((xyz.shape[0], n_extra), dtype=np.float32)
    for idx, name in enumerate(extra_names):
        features_extra[:, idx] = np.asarray(el[name])
    # (N, 3, n_coeffs) -> (N, n_coeffs, 3): channel-major storage -> coeff-major
    features_extra = features_extra.reshape(xyz.shape[0], 3, n_coeffs).transpose(0, 2, 1)

    scale_names = _sorted_props(props, "scale_")
    scales = np.stack([np.asarray(el[n]) for n in scale_names], axis=1).astype(np.float32)

    rot_names = _sorted_props(props, "rot")
    rots = np.stack([np.asarray(el[n]) for n in rot_names], axis=1).astype(np.float32)

    if activate:
        rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)
        scales = np.exp(scales)
        opacities = 1.0 / (1.0 + np.exp(-opacities))

    # 2DGS surfel: only two in-plane scales. Pad a near-flat third axis so the
    # 3DGS rasterizer treats it as a thin disk (see this package's README: 2DGS assets).
    if scales.shape[1] == 2:
        scales = np.hstack([scales, np.full((scales.shape[0], 1), 1e-9, dtype=np.float32)])

    # sh: (N, K, 3) with the DC coeff first, then the rest coeffs.
    sh = np.concatenate([features_dc[:, None, :], features_extra], axis=1).astype(np.float32)  # (N, K, 3)

    return GaussianData(
        xyz=xyz,
        rot=rots.astype(np.float32),
        scale=scales.astype(np.float32),
        opacity=opacities.astype(np.float32),
        sh=sh,
    )
