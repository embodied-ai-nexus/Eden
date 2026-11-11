"""Signed-distance-field helpers the MochiSolver's grid colliders read, beyond the ones ``genesis.utils.sdf`` ships."""

import genesis as gs
import quadrants as qd
from genesis.utils import array_class
from genesis.utils.sdf import sdf_func_ravel_cell_idx


@qd.func
def sdf_func_true_sdf_and_grad(geom_idx, pos_sdf, sdf_info: array_class.SDFInfo):
    """
    Trilinear value interpolant at an in-grid point and its analytic gradient in the mesh frame, from one read of the
    eight cell corners.

    The SDF frame is a scaled translation of the mesh frame, so only the per-axis cell size enters the gradient.
    """
    geom_sdf_res = sdf_info.geoms_info.sdf_res[geom_idx]
    cs = sdf_info.geoms_info.sdf_cell_size[geom_idx]
    base = qd.min(qd.floor(pos_sdf, gs.qd_int), geom_sdf_res - 2)
    signed_dist = gs.qd_float(0.0)
    grad_mesh = qd.Vector.zero(gs.qd_float, 3)
    for offset in qd.grouped(qd.ndrange(2, 2, 2)):
        pos_cell = base + offset
        w_xyz = 1 - qd.abs(pos_sdf - pos_cell)
        val = sdf_info.geoms_sdf_val[sdf_func_ravel_cell_idx(pos_cell, geom_idx, geom_sdf_res, sdf_info)]
        signed_dist += w_xyz[0] * w_xyz[1] * w_xyz[2] * val
        grad_mesh[0] += (2 * offset[0] - 1) * w_xyz[1] * w_xyz[2] * val / cs[0]
        grad_mesh[1] += w_xyz[0] * (2 * offset[1] - 1) * w_xyz[2] * val / cs[1]
        grad_mesh[2] += w_xyz[0] * w_xyz[1] * (2 * offset[2] - 1) * val / cs[2]
    return signed_dist, grad_mesh
