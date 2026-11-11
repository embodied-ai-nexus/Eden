"""Interaction-fidelity metrics of a retargeted motion (OTRetarget Sec. IV-A).

Independently of the solver's probes and grids: every body unit (one correspondence group — a set of
human parts and the robot links matched to them) is measured against every channel (the ground, each
object) at every frame with *exact* signed distances from the posed human mesh vertices and the robot's
visual-mesh vertices, each unit's distance being that of its closest vertex. Distances are to the channel's
surface, ``MeshChannel.mesh``: for a scan whose winding sign is unreliable that is its watertight solid, one
``object_solid_voxel`` outside the scan, for the demonstration and the robot alike.

With ``eps_eval = 20 mm``, the demonstrated and retargeted interaction sets are
``H = {(t, u, c) : (d_hat)+ < eps}`` and ``R = {(t, u, c) : (d)+ < eps}``; their agreement is reported as
precision ``|H n R| / |R|``, recall ``|H n R| / |H|`` and Jaccard ``|H n R| / |H u R|``, and the geometric
accuracy over ``H`` as the depth error ``mean |d - (d_hat)+|`` and the placement error ``mean |w - w_hat|``.
Ground and object interactions are reported separately, and the object interactions of the units on the
welded hand entities (palm and phalanges: groups whose links all belong to a hand entity) once more on their own
(``hand/*``), since a whole-body score hides a grasp that misses. A demonstrated witness on a substituted object
(``demo_mesh``) is carried onto the scene mesh as the retargeter carries it before it is compared. Also reported:
stance clearance and foot skate (the median over stance steps of the mean horizontal speed of the foot vertices
within ``stance_height`` of the floor), the 95th-percentile scene penetration (and that of the hand units alone),
the style orientation error, the object drift, and the smoothness of the motion as mean absolute second differences
per frame of the joints, the root position and the human pelvis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from eden.utils.geom import quat_to_R
from eden.utils.mesh import get_link_mesh

if TYPE_CHECKING:
    from eden.extensions.retargeting.otretarget.human import HumanSurfaceMotion
    from eden.extensions.retargeting.otretarget.retargeter import InteractionObject, OTRetargeter, RetargetResult


def _agreement(h: np.ndarray, r: np.ndarray) -> dict[str, float]:
    inter = float((h & r).sum())
    union = float((h | r).sum())
    return {
        "precision": inter / r.sum() if r.sum() else float("nan"),
        "recall": inter / h.sum() if h.sum() else float("nan"),
        "jaccard": inter / union if union else float("nan"),
    }


def evaluate_retarget(
    retargeter: OTRetargeter,
    human: HumanSurfaceMotion,
    objects: list[InteractionObject],
    result: RetargetResult,
    eps_eval: float = 0.02,
    stance_height: float = 0.02,
    max_vertices_per_link: int = 1500,
) -> dict[str, float]:
    """Score a retarget against its demonstration.

    Parameters
    ----------
    retargeter : OTRetargeter
        The retargeter that produced ``result`` (after :meth:`~OTRetargeter.retarget_motion`, so its
        correspondence groups and channels match the demonstration).
    human : HumanSurfaceMotion
        The demonstration.
    objects : list[InteractionObject]
        Its objects (demonstrated poses).
    result : RetargetResult
        The retargeted motion.
    eps_eval : float
        Interaction threshold (20 mm).
    stance_height : float
        A foot is in stance when its lowest point is below this (m).
    max_vertices_per_link : int
        Robot visual-mesh vertices kept per link (evenly subsampled) to bound the cost.

    Returns
    -------
    dict[str, float]
        ``ground/*``, ``object/*`` and ``hand/*`` agreement and errors, ``clearance_mm``, ``skate_cm_s``,
        ``penetration_p95_mm``, ``hand/penetration_p95_mm``, ``rot_deg`` (``rot_body_deg`` over the root entity's
        links when hands are welded on), ``drift_cm``, and
        ``jitter_joint_deg`` (deg/frame^2), ``jitter_root_cm`` and ``jitter_human_pelvis_cm`` (cm/frame^2).
    """
    kin = retargeter.kin
    links = {link.name: link for ent in kin.entities for link in ent.links}
    groups = [retargeter._group(g) for g in retargeter.correspondences]
    # Robot units: visual-mesh vertices of each group's links, in link frames.
    unit_link, unit_local, unit_id = [], [], []
    for ui, (_, robot_links) in enumerate(groups):
        for name in robot_links:
            link = links[name]
            if len(link.vgeoms) == 0:
                continue
            v = np.asarray(get_link_mesh(link, use_visual_mesh=True).vertices)
            if len(v) > max_vertices_per_link:
                v = v[np.linspace(0, len(v) - 1, max_vertices_per_link).astype(int)]
            unit_link.append(np.full(len(v), kin.link_index(name)))
            unit_local.append(v)
            unit_id.append(np.full(len(v), ui))
    hand_links = {link.name for ent in kin.entities[1:] for link in ent.links}
    # All of a group's links, not any: the Wuji presets put the hand's mount link in the elbow group.
    hand_unit = np.array([bool(robot_links) and set(robot_links) <= hand_links for _, robot_links in groups])
    unit_link = np.concatenate(unit_link)
    unit_local = np.concatenate(unit_local)
    unit_id = np.concatenate(unit_id)
    # Human units: mesh vertices of each group's parts.
    vparts = human.vertex_parts
    human_unit = np.full(len(vparts), -1)
    for ui, (human_parts, _) in enumerate(groups):
        human_unit[np.isin(vparts, [human.joint_index(p) for p in human_parts])] = ui

    n_frames, n_units = len(result.qpos), len(groups)
    # Each unit's vertex indices, found once: rebuilding the masks per frame, channel and unit cost O(V * units).
    human_idx = [np.flatnonzero(human_unit == ui) for ui in range(n_units)]
    robot_idx = [np.flatnonzero(unit_id == ui) for ui in range(n_units)]
    channels = retargeter.channels
    d_hat = np.full((n_frames, n_units, len(channels)), np.inf)
    d_rob = np.full_like(d_hat, np.inf)
    w_hat = np.zeros((*d_hat.shape, 3))
    w_rob = np.zeros_like(w_hat)
    pen = np.zeros(n_frames)
    hand_pen = np.zeros(n_frames)
    hand_vertex = hand_unit[unit_id]
    foot_units = [ui for ui, (hp, _) in enumerate(groups) if any("ankle" in p or "Foot" in p for p in hp)]
    foot_low = np.full((n_frames, len(foot_units)), np.inf)
    foot_pts: list[list[np.ndarray]] = [[] for _ in foot_units]
    solved = {o.name: result.object_poses[o.name] for o in objects}
    rot_err = []
    for t in range(n_frames):
        kin.set_configuration(result.qpos[t])
        link_pos, link_rot = kin.link_poses()
        robot_world = link_pos[unit_link] + np.einsum("nab,nb->na", link_rot[unit_link], unit_local)
        human_world = human.vertices[t].astype(np.float64)
        for ci, channel in enumerate(channels):
            oi = retargeter._channel_object[ci]
            if oi < 0:
                R_demo = R_now = np.eye(3)
                tau_demo = tau_now = np.array([0.0, 0.0, retargeter.ground_height])
            else:
                R_demo = quat_to_R(objects[oi].poses[t, 3:])
                tau_demo = objects[oi].poses[t, :3]
                R_now = quat_to_R(solved[objects[oi].name][t, 3:])
                tau_now = solved[objects[oi].name][t, :3]
            demo_channel = retargeter._demo_channels.get(oi, channel)
            hd, hw = demo_channel.exact_distance((human_world - tau_demo) @ R_demo)
            rd, rw = channel.exact_distance((robot_world - tau_now) @ R_now)
            pen[t] = max(pen[t], float(np.clip(-rd, 0.0, None).max()))
            if oi >= 0 and hand_vertex.any():
                hand_pen[t] = max(hand_pen[t], float(np.clip(-rd[hand_vertex], 0.0, None).max()))
            for ui in range(n_units):
                if len(hi := human_idx[ui]):
                    k = hi[np.argmin(hd[hi])]
                    d_hat[t, ui, ci], w_hat[t, ui, ci] = hd[k], hw[k]
                if len(ri := robot_idx[ui]):
                    k = ri[np.argmin(rd[ri])]
                    d_rob[t, ui, ci], w_rob[t, ui, ci] = rd[k], rw[k]
            if oi in retargeter._witness_map:
                # The demonstrated witness lies on the demo mesh; the robot's on the scene mesh. Mapped after the
                # closest vertex is chosen, so only one witness per unit goes through the tree.
                tree, s_pts, _ = retargeter._witness_map[oi]
                w_hat[t, :, ci] = s_pts[tree.query(w_hat[t, :, ci])[1]]
        for k, ui in enumerate(foot_units):
            pts = robot_world[robot_idx[ui]]
            foot_low[t, k] = pts[:, 2].min() - retargeter.ground_height
            foot_pts[k].append(pts)
        # Style orientation error against the retargeter's own targets.
        _, style_rot = retargeter._style_targets(human, t)
        rel = np.einsum("lab,lcb->lac", link_rot[retargeter._style_link_idx], style_rot)
        cos = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
        rot_err.append(np.degrees(np.arccos(cos)))

    out: dict[str, float] = {}
    is_ground = np.array([retargeter._channel_object[ci] < 0 for ci in range(len(channels))])
    every_unit = np.ones(n_units, dtype=bool)
    for name, mask, units in (
        ("ground", is_ground, every_unit),
        ("object", ~is_ground, every_unit),
        ("hand", ~is_ground, hand_unit),
    ):
        if not mask.any() or not units.any():
            continue
        dh = d_hat[:, units][:, :, mask]
        dr = d_rob[:, units][:, :, mask]
        H = np.maximum(dh, 0.0) < eps_eval
        R = np.maximum(dr, 0.0) < eps_eval
        out |= {f"{name}/{k}": v for k, v in _agreement(H, R).items()}
        if H.any():
            out[f"{name}/depth_err_mm"] = 1e3 * float(np.abs(dr[H] - np.maximum(dh[H], 0.0)).mean())
            out[f"{name}/place_err_mm"] = 1e3 * float(
                np.linalg.norm(w_rob[:, units][:, :, mask][H] - w_hat[:, units][:, :, mask][H], axis=1).mean()
            )
    stance = foot_low < stance_height
    out["clearance_mm"] = 1e3 * float(np.median(foot_low[stance])) if stance.any() else float("nan")
    skate = []
    dt = 1.0 / result.fps
    for k in range(len(foot_units)):
        for t in range(n_frames - 1):
            if stance[t, k] and stance[t + 1, k]:
                # The vertices in contact at t: the least moving vertex of the whole foot read a foot yawing or
                # rolling heel-to-toe in place as no skate (any rigid motion has a point that barely moves).
                pts = foot_pts[k][t]
                down = pts[:, 2] - retargeter.ground_height < stance_height
                disp = np.linalg.norm((foot_pts[k][t + 1][down] - pts[down])[:, :2], axis=1)
                skate.append(disp.mean() / dt)
    out["skate_cm_s"] = 100.0 * float(np.median(skate)) if skate else float("nan")
    out["penetration_p95_mm"] = 1e3 * float(np.percentile(pen, 95))
    if hand_unit.any() and (~is_ground).any():
        out["hand/penetration_p95_mm"] = 1e3 * float(np.percentile(hand_pen, 95))
    if n_frames > 2:
        q = result.qpos
        out["jitter_joint_deg"] = float(np.degrees(np.abs(np.diff(q[:, kin.n_base_q :], 2, axis=0))).mean())
        if kin.floating:
            out["jitter_root_cm"] = 100.0 * float(np.linalg.norm(np.diff(q[:, :3], 2, axis=0), axis=1).mean())
            pelvis = human.joint_pos[:n_frames, retargeter._pelvis_joint]
            out["jitter_human_pelvis_cm"] = 100.0 * float(np.linalg.norm(np.diff(pelvis, 2, axis=0), axis=1).mean())
    rot_err = np.array(rot_err)  # (T, tracked links)
    out["rot_deg"] = float(rot_err.mean())
    # The same over the links of the root entity alone: the hand presets weigh finger style down on purpose.
    body = np.array([kin.link_names[li] not in hand_links for li in retargeter._style_link_idx])
    if body.any() and not body.all():
        out["rot_body_deg"] = float(rot_err[:, body].mean())
    if objects:
        out["drift_cm"] = 100.0 * float(
            np.mean([np.linalg.norm(solved[o.name][:, :3] - o.poses[:n_frames, :3], axis=1).mean() for o in objects])
        )
    return out
