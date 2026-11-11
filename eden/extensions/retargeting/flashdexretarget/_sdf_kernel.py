"""The warp kernel of the SDF bake, in a module of its own so ``warp`` is imported only when a grid is baked."""

import warp as wp


@wp.kernel
def closest_point(
    mesh: wp.uint64,
    pts: wp.array(dtype=wp.vec3),
    max_dist: float,
    closest: wp.array(dtype=wp.vec3),
    sign: wp.array(dtype=float),
):
    i = wp.tid()
    q = wp.mesh_query_point_sign_winding_number(mesh, pts[i], max_dist)
    closest[i] = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
    sign[i] = q.sign
