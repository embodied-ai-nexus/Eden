# Copyright (c) Meta Platforms, Inc. and affiliates.
# Portions of this file are derived from the "mochi" physics library (mochi_core / mochi_physics) of
# https://github.com/facebookresearch/project_superdex, licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Modified by the Eden authors: ported from C++ to Python
# (quadrants) and hosted on Genesis.
# SPDX-License-Identifier: Apache-2.0
"""Forward kinematics of a Newton iterate in one kernel: link poses, joint motion subspaces (the link Jacobians of the
residual), geom poses and geom bounds. The joint-space velocities are left out; the residual never reads them and the
step recomputes them once its final iterate is known.
"""

import genesis as gs
import genesis.utils.geom as gu
import quadrants as qd
from genesis.engine.solvers.rigid.abd.forward_kinematics import (
    func_COM_root,
    func_forward_kinematics_root,
    func_forward_velocity_root,
    func_update_geom_aabbs,
    func_update_geoms_root,
)
from genesis.utils import array_class


@qd.func
def func_joint_q_end(q_start, joint_type):
    """One past the last generalized coordinate of a joint, from its first one and its type.

    The width of a joint's coordinates follows from its type alone: a free joint holds a position and a quaternion, a
    spherical joint a quaternion, a revolute or prismatic joint one coordinate and a fixed joint none. Genesis dropped
    its own `joints.q_end` field, which none of its kernels read, in
    https://github.com/Genesis-Embodied-AI/Genesis/pull/3656.
    """
    n_qs = 0
    if joint_type == gs.JOINT_TYPE.FREE:
        n_qs = 7
    elif joint_type == gs.JOINT_TYPE.SPHERICAL:
        n_qs = 4
    elif joint_type == gs.JOINT_TYPE.REVOLUTE or joint_type == gs.JOINT_TYPE.PRISMATIC:
        n_qs = 1
    return q_start + n_qs


@qd.func
def func_forward_kinematics_env(
    i_b: int,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    with_velocity: qd.template(),
):
    """Compute the link poses, the centers of mass and the geom poses of env i_b, one kinematic root after another.

    The joint-space velocities follow the centers of mass under with_velocity.
    """
    for i_r in range(rigid_info.roots_link_idx.shape[0]):
        i_l_root = rigid_info.roots_link_idx[i_r]
        func_forward_kinematics_root(
            i_l_root, i_b, rigid_info.qpos, dyn_state, dyn_info, rigid_info, rigid_config, is_backward=False
        )
        func_COM_root(i_l_root, i_b, dyn_state, dyn_info, rigid_info, rigid_config)
        if qd.static(with_velocity):
            func_forward_velocity_root(i_l_root, i_b, dyn_state, dyn_info, rigid_info, rigid_config, is_backward=False)
        func_update_geoms_root(
            i_l_root, i_b, dyn_state, dyn_info, rigid_info, rigid_config, force_update_all_geoms=False
        )


@qd.kernel(fastcache=True)
def kernel_update_geom_aabbs(
    geoms_init_AABB: array_class.GeomsInitAABB,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_config: qd.template(),
):
    func_update_geom_aabbs(geoms_init_AABB, dyn_state, dyn_info, rigid_config)


@qd.kernel
def kernel_update_kinematics(
    envs_idx: qd.types.ndarray(),
    geoms_init_AABB: array_class.GeomsInitAABB,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
):
    for i_b_ in range(envs_idx.shape[0]):
        i_b = qd.cast(envs_idx[i_b_], qd.i32)
        func_forward_kinematics_env(i_b, dyn_state, dyn_info, rigid_info, rigid_config, with_velocity=False)
    func_update_geom_aabbs(geoms_init_AABB, dyn_state, dyn_info, rigid_config)


@qd.func
def func_update_kinematics(
    envs: qd.Tensor,
    n_envs: qd.Tensor,
    geoms_init_AABB: array_class.GeomsInitAABB,
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    with_velocity: qd.template(),
):
    """Link poses, motion subspaces, geom poses and bounds of the environments of a list (joint-space velocities on
    request); one thread per environment for the kinematic chain, one per geom for the bounds.
    """
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.ALL))
    for i_slot in range(n_envs[None]):
        i_b = envs[i_slot]
        func_forward_kinematics_env(i_b, dyn_state, dyn_info, rigid_info, rigid_config, with_velocity)
    n_geoms = dyn_state.geoms.pos.shape[0]
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_g, i_slot in qd.ndrange(n_geoms, n_envs[None]):
        i_b = envs[i_slot]
        g_pos = dyn_state.geoms.pos[i_g, i_b]
        g_quat = dyn_state.geoms.quat[i_g, i_b]
        lower = gu.qd_vec3(qd.math.inf)
        upper = gu.qd_vec3(-qd.math.inf)
        for i_corner in qd.static(range(8)):
            corner_pos = gu.qd_transform_by_trans_quat(geoms_init_AABB[i_g, i_corner], g_pos, g_quat)
            lower = qd.min(lower, corner_pos)
            upper = qd.max(upper, corner_pos)
        dyn_state.geoms.aabb_min[i_g, i_b] = lower
        dyn_state.geoms.aabb_max[i_g, i_b] = upper


@qd.kernel(fastcache=True)
def kernel_update_geoms(
    envs_idx: qd.types.ndarray(),
    dyn_state: array_class.DynState,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    rigid_config: qd.template(),
    force_update_all_geoms: qd.template(),
):
    """Place the geoms of every kinematic tree, as `genesis.engine.solvers.rigid.abd.forward_kinematics` does.

    The upstream kernel passes each root's rank among the roots to `func_update_geoms_root`, which takes the root's
    link. The two coincide only while every root before it is a single link, so upstream never places a static
    collider added after an articulated body, and soft bodies pass through it. This copy looks the root's link up.
    """
    qd.loop_config(serialize=qd.static(rigid_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_r, i_b_ in qd.ndrange(rigid_info.roots_link_idx.shape[0], envs_idx.shape[0]):
        i_b = qd.cast(envs_idx[i_b_], qd.i32)
        i_l_root = rigid_info.roots_link_idx[i_r]
        func_update_geoms_root(i_l_root, i_b, dyn_state, dyn_info, rigid_info, rigid_config, force_update_all_geoms)
