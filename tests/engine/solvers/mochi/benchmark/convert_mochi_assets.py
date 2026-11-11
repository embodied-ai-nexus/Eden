"""Convert the mochi example assets (Y-up HDF5) into the files of this benchmark (Z-up): the duck tetrahedral mesh to a
tetgen `.node`/`.ele` pair, the t-shirt triangle mesh to `.obj` and the helical spring polyline to `.npy`.

Run once, in an environment that has h5py, then upload ``<out_dir>`` as ``objects/mochi`` of the
``Kashu7100/eden_assets`` Hub dataset, which the benchmark scenes and the Mochi recipes read::

    python tests/engine/solvers/mochi/benchmark/convert_mochi_assets.py /path/to/superdex_physics/assets <out_dir>
"""

import argparse
import os

import h5py
import numpy as np


def y_up_to_z_up(points):
    return np.stack([points[:, 0], -points[:, 2], points[:, 1]], axis=-1)


def main(assets_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(os.path.join(assets_root, "duck", "duck_1899.mochi.h5"), "r") as f:
        verts = y_up_to_z_up(np.asarray(f["mesh/coordinates"], dtype=np.float64))
        tets = np.asarray(f["mesh/connectivity"], dtype=np.int64)
    with open(os.path.join(out_dir, "duck_1899.node"), "w") as fp:
        fp.write(f"{len(verts)} 3 0 0\n")
        fp.writelines(f"{i + 1} {p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n" for i, p in enumerate(verts))
    with open(os.path.join(out_dir, "duck_1899.ele"), "w") as fp:
        fp.write(f"{len(tets)} 4 0\n")
        fp.writelines(f"{i + 1} {t[0] + 1} {t[1] + 1} {t[2] + 1} {t[3] + 1}\n" for i, t in enumerate(tets))

    with h5py.File(os.path.join(assets_root, "garments", "tshirt_visual_subdiv_2.mochi.h5"), "r") as f:
        verts = y_up_to_z_up(np.asarray(f["mesh/coordinates"], dtype=np.float64))
        tris = np.asarray(f["mesh/connectivity"], dtype=np.int64)
    with open(os.path.join(out_dir, "tshirt_3593.obj"), "w") as fp:
        fp.writelines(f"v {p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n" for p in verts)
        fp.writelines(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n" for t in tris)

    with h5py.File(os.path.join(assets_root, "rods", "helix_with_visual.mochi.h5"), "r") as f:
        verts = y_up_to_z_up(np.asarray(f["mesh/coordinates"], dtype=np.float64))
        segs = np.asarray(f["mesh/connectivity"], dtype=np.int64)
    assert (segs[:, 0] == np.arange(len(segs))).all() and (segs[:, 1] == np.arange(1, len(segs) + 1)).all()
    np.save(os.path.join(out_dir, "helix_129.npy"), verts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("assets_root", help="the mochi assets directory (superdex_physics/assets)")
    parser.add_argument("out_dir", help="directory to write the converted files to (upload it as objects/mochi)")
    args = parser.parse_args()
    main(args.assets_root, args.out_dir)
