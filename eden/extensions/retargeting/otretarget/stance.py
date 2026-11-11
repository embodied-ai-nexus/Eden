"""Legged stance on the floor: an add-on of :class:`~.retargeter.OTRetargeter` (``addons={"floor_stance": {...}}``).

The paper keeps foot-floor contact in the cost only, as the floor channel's interaction terms, normalised per link and
at the demonstrator's own footprints. On a robot of other proportions that floats the feet, slides them and pulls them
onto the human's stance width. Three additions, each an option below (measurements in
``examples/recipes/retargeting/otretarget/README.md``):

* **Footprint offset** — the floor is translation invariant, so where on it a foot lands is posture: each robot probe's
  floor witness is shifted sideways by how much wider than its human source it sits from the root in the T-pose.
* **Support** — a body unit (correspondence group) whose demonstrated sole is on the floor keeps its lowest point
  there, as a constraint.
* **No-slip stance** — a foot planted on two frames follows the human foot's planar rigid motion about its own contact
  (anchor probes as constraints), and its floor witnesses move with it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from eden.extensions.retargeting.otretarget.retargeter import Addon, Frame, Linearization, ProbeSet
from eden.utils.geom import R_to_quat, planar_rigid_transform, quat_from_axis_angle, quat_to_R, yaw_from_quat

if TYPE_CHECKING:
    from eden.extensions.retargeting.otretarget.human import HumanSurfaceMotion


class FloorStance(Addon):
    """Footprint offset, floor support and no-slip stance on the floor channel (needs ``ground``).

    Parameters
    ----------
    footprint_offset : bool
        Shift each robot probe's demonstrated floor witness sideways by how far that probe sits from the root in the
        T-pose beyond where its human source sits from the pelvis (lateral, in the root frame, turned with the
        demonstrated heading). A probe whose source stays in floor contact keeps the offset it landed with.
    footprint_contact : float
        (m) A probe whose human source is within this of the floor is in contact.
    support_height, support_slack : float
        (m) On every body unit with a downward-facing probe whose human source is within ``support_height`` of the
        floor, the unit's lowest downward-facing probe may rise at most ``support_slack`` above the demonstrated
        height (0 height disables).
    stick : bool
        No-slip stance: while probes of a unit have their sources in contact on two consecutive frames, a few of its
        probes on the floor (anchors) follow the human foot's planar rigid motion about the robot foot's own contact,
        within ``stick_slack``. After lift-off its floor witnesses return to the footprint targets over
        ``stick_release`` of the human foot's travel.
    stick_slack : float
        (m) Horizontal displacement allowed per frame beyond the demonstrated one.
    stick_height : float
        (m) Anchors are probes in contact within this of the robot unit's lowest point on the previous solution.
    stick_release : float
        (m) Human foot travel after lift-off over which the floor witnesses return to the footprint targets.
    stick_anchors : int
        Anchors per unit in contact: the lowest one, then each farthest from those already chosen.
    """

    footprint_offset: bool = True
    footprint_contact: float = 0.02
    support_height: float = 0.02
    support_slack: float = 0.005
    stick: bool = True
    stick_slack: float = 2e-4
    stick_height: float = 0.005
    stick_release: float = 0.1
    stick_anchors: int = 3

    def __init__(self, rt, options) -> None:
        super().__init__(rt, options)
        if not rt.ground:
            raise ValueError(
                "FloorStance: the retargeter has no floor channel (ground=False); set ground=True or drop it."
            )
        checks = {
            "footprint_contact": self.footprint_contact >= 0,
            "support_height": self.support_height >= 0,
            "support_slack": self.support_slack >= 0,
            "stick_slack": self.stick_slack >= 0,
            "stick_height": self.stick_height > 0,
            "stick_release": self.stick_release > 0,
            "stick_anchors": self.stick_anchors >= 1,
        }
        bad = [k for k, ok in checks.items() if not ok]
        if bad:
            raise ValueError(
                f"FloorStance: {', '.join(f'{k}={getattr(self, k)!r}' for k in bad)} out of range: heights, slacks and "
                f"contact must be >= 0, stick_height and stick_release > 0, stick_anchors >= 1."
            )

    def setup(self, human: HumanSurfaceMotion) -> None:
        rt = self.rt
        self._ground = rt._channel_object.index(-1)
        # Footprint offsets: where each robot probe sits relative to the root in the T-pose, minus where its
        # human source sits relative to the pelvis, in the root's T-pose frame.
        root = rt._style_link_idx[0]
        R_root_t, x_root_t = rt._tpose_link_rot[root], rt._tpose_link_pos[root]
        R_hp, x_hp = human.rest_joint_rot[rt._pelvis_joint], human.rest_joint_pos[rt._pelvis_joint]
        r_tpose = rt.robot_probes.world(rt._tpose_link_pos, rt._tpose_link_rot)
        h_rest = rt.human_probes.positions(human.rest_vertices)
        src = np.maximum(rt.src, 0)
        robot_rel = (r_tpose - x_root_t) @ R_root_t
        human_rel = ((h_rest[src] - x_hp) @ R_hp) @ rt._style_offset[0]
        offset = robot_rel - human_rel
        # Lateral only: the offset is there for stance width. The fore-aft part measures where a skeleton's
        # ankle sits under its pelvis in the reference pose, which is no reason to plant a foot elsewhere.
        offset[:, [0, 2]] = 0.0
        offset[rt.src < 0] = 0.0
        self.offset = offset
        self._footprint = None  # (held floor-witness offsets, contact) of the previous frame
        self._stance = None  # (demonstrated floor distance, human source positions) of the previous frame
        self._carry = None  # (per-probe floor-witness offset carried off the floor, human travel since, planted)

    def frame(self, human: HumanSurfaceMotion, frame: Frame, forward: bool) -> None:
        rt = self.rt
        targets = frame.robot_t[self._ground]
        if self.footprint_offset:
            # A planted foot does not turn with the pelvis: a probe whose source is in floor contact keeps the offset
            # it landed with, and only a lifted one re-aims it at the current heading (turned every frame, a 10 cm
            # offset swept the stance foot of a spin round a 10 cm circle).
            R_root = human.joint_rot[frame.t, rt._pelvis_joint] @ rt._style_offset[0]
            heading = quat_from_axis_angle(np.array([0.0, 0.0, yaw_from_quat(R_to_quat(R_root))]))
            turned = self.offset @ quat_to_R(heading).T  # the offsets are horizontal: z stays 0
            h = frame.human_t[self._ground]
            contact = (rt.src >= 0) & (h.distance[np.maximum(rt.src, 0)] < self.footprint_contact)
            if self._footprint is not None:
                held, was_contact = self._footprint
                keep = contact & was_contact
                turned[keep] = held[keep]
            if forward:
                # Only the forward pass advances: a frame read again later (a stage re-solving earlier frames) would
                # leave the next frame comparing its contact against a frame long past.
                self._footprint = (turned, contact)
            targets.witness = targets.witness + turned
        if forward and self.stick:
            frame.extra["stick"] = self._anchors(human, frame.t, targets, frame.q_prev)

    def _anchors(self, human, t, targets, q_prev):
        """No-slip stance at frame ``t``; updates ``targets``' floor witnesses in place.

        Returns ``(anchor indices, (k, 2) anchor targets, (k,) tolerances)`` or None. Anchor targets and the floor
        witnesses of the probes in contact are positions on the previous solution carried by the human unit's planar
        rigid motion since that frame; after lift-off, the offset from the footprint targets fades over the human
        foot's travel.
        """
        rt = self.rt
        has_src = rt.src >= 0
        d_hat = np.where(has_src, targets.distance, np.inf)
        h_src = rt.human_probes.positions(human.vertices[t].astype(np.float64))[np.maximum(rt.src, 0)]
        prev, self._stance = self._stance, (d_hat, h_src)
        if prev is None or q_prev is None:
            return None
        d_prev, h_prev = prev
        if self._carry is None:
            self._carry = (np.zeros((len(d_hat), 2)), np.zeros(len(d_hat)), np.zeros(len(d_hat), dtype=bool))
        carry, travel, was_planted = self._carry
        footprint = targets.witness[:, :2].copy()
        contact = has_src & (d_hat < self.footprint_contact) & (d_prev < self.footprint_contact)
        # Off the floor: the carried offset fades over the human foot's travel.
        travel = travel + np.linalg.norm((h_src - h_prev)[:, :2], axis=1)
        fade = np.clip(1.0 - travel / self.stick_release, 0.0, 1.0)
        witness = footprint + carry * fade[:, None]
        anchors = []
        if contact.any():
            rt.kin.set_configuration(q_prev)
            r_prev = rt.robot_probes.world(*rt.kin.link_poses())
            planted = np.zeros(len(d_hat), dtype=bool)
            anchor_xy = np.zeros((len(d_hat), 2))
            for unit in np.unique(rt.probe_group[contact]):
                on_unit = np.flatnonzero(contact & (rt.probe_group == unit))
                rot, _ = planar_rigid_transform(h_prev[on_unit, :2], h_src[on_unit, :2])
                # The robot foot turns about its own counterpart of the human contact (the centroid of the same probes)
                # and moves as that contact's centroid did: about the human's centroid, a pivot on the ball of the
                # human foot swung a robot foot standing a few centimetres away round it.
                c_r = r_prev[on_unit, :2].mean(axis=0)
                move = h_src[on_unit, :2].mean(axis=0) - h_prev[on_unit, :2].mean(axis=0)
                anchor_xy[on_unit] = (r_prev[on_unit, :2] - c_r) @ rot.T + c_r + move
                # The floor witnesses of the whole (rigid) unit are its footprint targets shifted by one offset,
                # latched when the stance begins (where the robot foot landed against where the human's did) and held:
                # carried with the foot every frame, nothing resisted a slow drift; left on the bare footprints, they
                # pulled the foot away at lift-off; frozen per probe, they stopped a heel rising.
                whole = np.flatnonzero(has_src & (rt.probe_group == unit))
                if not was_planted[whole].any():
                    carry[whole] = (anchor_xy[on_unit] - footprint[on_unit]).mean(axis=0)
                witness[whole] = footprint[whole] + carry[whole]
                planted[whole] = True
                # Anchors only where the robot foot is down too: the unit's probes within ``stick_height`` of its
                # lowest one, if that is within the contact band of the demonstrated height.
                z = r_prev[on_unit, 2]
                demo_low = max(float(d_hat[on_unit].min()), 0.0)
                if z.min() - rt.ground_height > demo_low + self.footprint_contact:
                    continue
                cand = on_unit[z - z.min() < self.stick_height]
                dense = cand[rt._collision_probe[cand]]
                cand = dense if len(dense) else cand
                chosen = [cand[np.argmin(r_prev[cand, 2])]]
                gap = np.linalg.norm(r_prev[cand] - r_prev[chosen[0]], axis=1)
                while len(chosen) < min(self.stick_anchors, len(cand)):
                    k = int(np.argmax(gap))
                    if gap[k] <= 0.0:
                        break
                    chosen.append(cand[k])
                    gap = np.minimum(gap, np.linalg.norm(r_prev[cand] - r_prev[cand[k]], axis=1))
                anchors += chosen
            # In contact: the offset is refreshed and its travel restarts.
            travel = np.where(planted, 0.0, travel)
        else:
            planted = np.zeros(len(d_hat), dtype=bool)
        # The latched offset is kept as it was: ``fade`` is already a function of the travel since lift-off, so
        # storing the faded offset would fade it again every frame (0.8, 0.48, 0.19 of it instead of 0.8, 0.6, 0.4).
        self._carry = (carry, travel, planted)
        targets.witness = targets.witness.copy()
        targets.witness[:, :2] = witness
        if not anchors:
            return None
        idx = np.array(anchors, dtype=np.int64)
        return idx, anchor_xy[idx], np.full(len(idx), self.stick_slack)

    def set_terms(self, lin: Linearization, s: ProbeSet) -> None:
        if s.kind != "robot" or s.ci != self._ground or self.support_height <= 0:
            return
        # Support: a body unit (e.g. ankle + foot links) whose demonstrated sole is on the floor keeps its lowest point
        # there, d <= (d_hat)+ + slack. One row per unit, on its lowest downward-facing probe whether that probe is the
        # supported one or not: a rigid foot cannot put every probe paired with a floor-touching human point down.
        rt = self.rt
        probes = rt.robot_probes
        down = np.einsum("nab,nb->na", lin.link_rot[probes.carrier], probes.normals)[:, 2] < -0.7
        candidates = down & (rt.src >= 0)
        supported = np.flatnonzero((s.targets.distance < self.support_height) & candidates)
        units = rt.probe_group[supported]
        for unit in np.unique(units):
            on_unit = np.flatnonzero(candidates & (rt.probe_group == unit))
            k = on_unit[np.argmin(s.d[on_unit])]
            target = max(float(s.targets.distance[supported[units == unit]].min()), 0.0)
            target += self.support_slack
            rows = rt._rows(
                "robot", s.ci, np.array([k]), s.world[[k]], s.R, s.tau, lin.obj_pos, lin.link_pos, s.grad[[k]]
            )
            lin.support_C.append(-lin.gn.dense(rows))
            lin.support_l.append(np.array([s.d[k] - target]))

    def terms(self, lin: Linearization) -> None:
        stick = lin.frame.extra.get("stick")
        if stick is None:
            return
        # No-slip stance: each anchor's horizontal position stays within ``tol`` of its target.
        rt = self.rt
        idx, target_xy, tol = stick
        R, tau = rt._channel_pose(self._ground, lin.obj_pos, lin.obj_rot)
        slide = lin.r_world[idx, :2] - target_xy
        for axis in range(2):
            gamma = np.zeros((len(idx), 3))
            gamma[:, axis] = 1.0
            rows = rt._rows("robot", self._ground, idx, lin.r_world[idx], R, tau, lin.obj_pos, lin.link_pos, gamma)
            G = lin.gn.dense(rows)
            lin.support_C += [G, -G]
            lin.support_l += [-tol - slide[:, axis], -tol + slide[:, axis]]
