"""Shared fixtures of the OTRetarget tests: a Unitree G1 that demonstrates to itself.

A G1 squat is posed through forward kinematics and its link surfaces (convex hulls) become the "human" mesh, each link
its own body part. The robot can then reproduce the demonstration exactly, so the joint angles, root path and
foot-floor contact it must return are known values rather than a recomputation of the optimiser.
"""

from __future__ import annotations

import numpy as np
import pytest

import eden as en
from eden.options import EnvOptions, SceneOptions
from eden.utils.configs import EdenConfig
from eden.utils.mesh import get_link_mesh
from tests.support.envs import build_env

LEG_DOFS = [f"{s}_{j}_joint" for s in ("left", "right") for j in ("hip_pitch", "knee", "ankle_pitch")]
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


@pytest.fixture(scope="module")
def env():
    return build_env(
        EdenConfig(env_options=EnvOptions(num_envs=1), scene_options=SceneOptions(robot=en.robots.G1Dofs29()))
    )


@pytest.fixture(scope="module")
def kin(env):
    from eden.extensions.retargeting.otretarget.kinematics import RigKinematics

    return RigKinematics(env, ["robot"])


@pytest.fixture(scope="module")
def self_retarget(env, kin):
    """``(retargeter, demonstration, demonstrated configurations, result)`` of the G1 squat retargeted to the G1."""
    pytest.importorskip("proxsuite")  # the QP solver; in the ``ot-retargeter`` extra
    pytest.importorskip("threadpoolctl")
    from eden.extensions.retargeting.otretarget import OTRetargeter

    qs = squat(kin)
    human = robot_demonstration(kin, qs)
    ret = OTRetargeter(env, OTRetargeter.configure(**identity_options(kin)))
    return ret, human, qs, ret.retarget_motion(human, progress=False)


def q_index(kin, joint: str) -> int:
    """Configuration index of ``joint``."""
    return kin.n_base_q + kin.var_names.index(joint) - kin.n_base_v


def squat(kin, n_frames=8, depth=0.6):
    """A squat down to ``depth`` with flat feet, standing on z = 0.005: hip -a, knee 2a, ankle -a."""
    links = {link.name: link for link in kin.entities[0].links}
    feet = [kin.link_index(n) for n in FEET]
    foot_verts = {
        li: np.asarray(get_link_mesh(links[kin.link_names[li]], use_visual_mesh=True).vertices) for li in feet
    }
    qs = []
    for t in range(n_frames):
        a = depth * t / (n_frames - 1)
        q = np.zeros(kin.n_q)
        q[3] = 1.0
        q[2] = 1.0
        for s in ("left", "right"):
            q[q_index(kin, f"{s}_hip_pitch_joint")] = -a
            q[q_index(kin, f"{s}_knee_joint")] = 2 * a
            q[q_index(kin, f"{s}_ankle_pitch_joint")] = -a
        q[q_index(kin, "left_shoulder_pitch_joint")] = -0.5 * a
        kin.set_configuration(q)
        pos, rot = kin.link_poses()
        low = min((pos[li] + foot_verts[li] @ rot[li].T)[:, 2].min() for li in feet)
        q[2] += 0.005 - low
        qs.append(q)
    return np.array(qs)


def robot_demonstration(kin, qs, fps=30.0):
    """The robot's own link hulls posed by ``qs``, as a :class:`HumanSurfaceMotion` (one part per link)."""
    from eden.extensions.retargeting.otretarget import HumanSurfaceMotion

    links = {link.name: link for link in kin.entities[0].links}
    local, faces, weights, offset = [], [], [], 0
    for li, name in enumerate(kin.link_names):
        if not links[name].vgeoms:
            continue
        hull = get_link_mesh(links[name], use_visual_mesh=True).convex_hull
        local.append((li, np.asarray(hull.vertices)))
        faces.append(np.asarray(hull.faces) + offset)
        w = np.zeros((len(hull.vertices), len(kin.link_names)))
        w[:, li] = 1.0
        weights.append(w)
        offset += len(hull.vertices)

    def pose(q):
        kin.set_configuration(q)
        pos, rot = kin.link_poses()
        return np.concatenate([pos[li] + v @ rot[li].T for li, v in local]), pos, rot

    frames = [pose(q) for q in qs]
    rest_q = np.zeros(kin.n_q)
    rest_q[3] = 1.0
    rest_v, rest_p, rest_r = pose(rest_q)
    lift = np.array([0.0, 0.0, -rest_v[:, 2].min()])
    return HumanSurfaceMotion(
        vertices=np.stack([f[0] for f in frames]).astype(np.float32),
        faces=np.concatenate(faces),
        skin_weights=np.concatenate(weights),
        joint_names=list(kin.link_names),
        joint_pos=np.stack([f[1] for f in frames]),
        joint_rot=np.stack([f[2] for f in frames]),
        rest_vertices=rest_v + lift,
        rest_joint_pos=rest_p + lift,
        rest_joint_rot=rest_r,
        fps=fps,
    )


def identity_options(kin, stance=None, **overrides):
    """Retargeter options for the self-demonstration: each link its own group and style link, standing on the floor
    (``floor_stance`` with the options ``stance``, None for its defaults).
    """
    links = {link.name: link for link in kin.entities[0].links}
    shaped = [n for n in kin.link_names if links[n].vgeoms]
    opts = dict(
        entity_name="robot",
        correspondences=[[[n], [n]] for n in shaped],
        style_links={n: n for n in shaped},
        human_root_joint="pelvis",
        first_frame_iterations=40,
        iterations=8,
        addons={"floor_stance": dict(stance or {})},
    )
    return opts | overrides


def lowest_foot_point(kin, q) -> float:
    """Height of the lowest vertex of the G1 feet at ``q``."""
    links = {link.name: link for link in kin.entities[0].links}
    kin.set_configuration(q)
    pos, rot = kin.link_poses()
    return min(
        (
            pos[kin.link_index(n)]
            + np.asarray(get_link_mesh(links[n], use_visual_mesh=True).vertices) @ rot[kin.link_index(n)].T
        )[:, 2].min()
        for n in FEET
    )
