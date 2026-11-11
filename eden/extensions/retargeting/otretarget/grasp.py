"""Dexterous-hand grasps: an add-on of :class:`~.retargeter.OTRetargeter` (``addons={"hand_grasp": {...}}``).

The interaction terms copy the demonstrated hand-object distances. Where the demonstration is loose (fingers hovering
2-4 cm off a pole, the pole sunk into the palm) or the robot hand is of other proportions, copied distances do not make
a grasp: the robot fingers hover, and a hand that starts or ends up on the wrong side of a thin part stays there. Three
stages, each an option below (measurements in the recipe README):

* **Grasp closure** — after each frame's solve, the fingers of a grasping hand close on the object in a solve of their
  own (hand joints only).
* **Palm cold start** — frame 0 is also started with the touching palms pulled to the human palms' places; the solve
  that holds more objects on the human palms' side wins.
* **Side re-approach** — a palm that holds a thin part on the other side from the human palm's is brought round: the
  frame is solved again with that arm only and a palm goal, and the earlier frames are re-solved back to where the hand
  is clear.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from eden.extensions.retargeting.otretarget.retargeter import Addon, Frame, Linearization, ProbeSet, Run, Targets
from eden.extensions.retargeting.otretarget.solver import pad, solve_qp
from eden.utils.geom import R_to_quat, axis_angle_from_quat, quat_to_R, skew

if TYPE_CHECKING:
    from eden.extensions.retargeting.otretarget.human import HumanSurfaceMotion

#: Weight (m^-2) of the palm goal of ``side_reapproach``.
_PALM_GOAL_WEIGHT = 1000.0
#: Frames over which that goal fades out after an accepted re-approach.
_REAPPROACH_FADE = 15
#: Deepest a frame rewritten by ``side_reapproach`` may hold its arm and hand in an object (m), exact distance: correct
#: OMOMO grasps hold 0.5-0.8 mm (median) and up to 2.2 mm.
_REAPPROACH_DEPTH = 0.0025
#: A palm orientation error of one radian weighs as much as this palm position error (m).
_PALM_ROT_SCALE = 0.05


class HandGrasp(Addon):
    """Grasp closure, palm cold start and side re-approach of the rig's hands.

    Parameters
    ----------
    palms : dict[str, str | list[str]]
        ``{robot palm link: human wrist joint}``, one entry per hand; or ``{link: [frame joint, joints...]}`` when the
        human palm's skin belongs to several joints (the SOMA rig's metacarpals).
    fingertips : dict[str, str]
        ``{robot fingertip link: human distal-phalanx joint}``; each fingertip belongs to the palm it hangs from. The
        finger's middle phalanx is the joint one less in the name's trailing digit (``left_index3`` -> ``left_index2``);
        a name without one closes its distal part alone.
    contact : float
        (m) Human palm/finger distance that counts as touching.
    closure : float
        (m) While a hand grasps (its human palm, or two of its finger pads, within ``contact`` of an object), a finger
        whose human distal phalanx is within this of the object closes on it: its middle and distal phalanx probes whose
        human sources face the object are pulled onto the object, and so are the palm probes whose sources touch it.
        The closure is its own solve of the hand joints alone (arm, root and objects held), from the frame's solution;
        the next frame starts from the open-handed solution. 0 disables.
    closure_weight : float
        (m^-2 per link) Weight of the closure residual, normalised over the probes of each link.
    cold_start_palm_weight : float
        (m^-2) A second cold start: the posture fit with each palm whose human palm touches a free object at frame 0
        also pulled to the human palm's place. Frame 0 is solved from both starts; the solve with more palms facing
        the object as the human palms do wins, then the one with fewer relaxed QPs, then the lower cost. 0 disables.
    rest_palm_normal : tuple[float, float, float]
        The direction the human palms face in the body model's rest pose (SMPL-X: down, ``(0, 0, -1)``); the robot's
        T-pose matches the rest pose, so its palm links face the same way there.
    side_reapproach : bool
        After each frame's solve, a palm whose human palm holds a thin part of a free object on the other side of the
        palm from the robot palm comes round (see ``_reapproach``); ``info["side_reapproach"]`` is 1 at a corrected
        frame and a negative code at a dropped attempt (see ``_reapproach_palm``).
    """

    _terms = ("grasp_closure", "palm_goal")
    _info = ("side_reapproach",)

    palms: dict[str, str | list[str]] = {}
    fingertips: dict[str, str] = {}
    contact: float = 0.01
    closure: float = 0.0
    closure_weight: float = 2000.0
    cold_start_palm_weight: float = 0.0
    rest_palm_normal: tuple[float, float, float] = (0.0, 0.0, -1.0)
    side_reapproach: bool = False

    def __init__(self, rt, options) -> None:
        super().__init__(rt, options)
        kin = rt.kin
        if not self.palms:
            raise ValueError(
                "HandGrasp: palms is empty; give {robot palm link: human wrist joint} per hand (the Wuji presets do)."
            )
        rt._refuse_unknown("hand_grasp palms keys", self.palms, kin.link_names, "the rig's links")
        rt._refuse_unknown("hand_grasp fingertips keys", self.fingertips, kin.link_names, "the rig's links")
        bad = [
            k for k in ("contact", "closure", "closure_weight", "cold_start_palm_weight") if not getattr(self, k) >= 0
        ]
        if bad:
            raise ValueError(f"HandGrasp: {', '.join(f'{k}={getattr(self, k)!r}' for k in bad)} must be >= 0.")
        if self.closure > 0 and not self.fingertips:
            raise ValueError("HandGrasp: closure needs `fingertips`; set them (the Wuji presets do) or leave it at 0.")
        if abs(float(np.linalg.norm(self.rest_palm_normal)) - 1.0) > 1e-6:
            raise ValueError(
                f"HandGrasp: rest_palm_normal {tuple(self.rest_palm_normal)!r} must be a unit vector (the direction the "
                "human palms face in the rest pose; SMPL-X: (0, 0, -1))."
            )
        palm_links = {kin.link_index(n) for n in self.palms}
        link_palm = np.full(len(kin.link_names), -1, dtype=np.int64)
        for li in range(len(kin.link_names)):
            cur = li
            while cur >= 0 and cur not in palm_links:
                cur = kin.link_parent[cur]
            link_palm[li] = cur
        self._link_palm = link_palm  # the palm link each link hangs from (-1: none)
        for tip in self.fingertips:
            if link_palm[kin.link_index(tip)] < 0:
                raise ValueError(
                    f"HandGrasp: fingertip {tip!r} hangs from none of the palms {list(self.palms)}; add its hand's palm "
                    f"link to `palms`."
                )
        # The hand joints (closure): every joint below a palm link, not the palm's own.
        hand = (link_palm[kin._joint_link] >= 0) & (link_palm[kin._joint_link] != kin._joint_link)
        self._hand_cols = np.zeros(kin.n_v, dtype=bool)
        self._hand_cols[kin._joint_var[hand]] = True
        self._normal = np.asarray(self.rest_palm_normal, dtype=np.float64)

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def setup(self, human: HumanSurfaceMotion) -> None:
        rt, kin = self.rt, self.rt.kin
        # Per palm link: the human joint whose frame is the palm's, and the joints whose skin is the palm.
        joints = {kin.link_index(n): [j] if isinstance(j, str) else list(j) for n, j in self.palms.items()}
        self._palm_joint = {li: human.joint_index(j[0]) for li, j in joints.items()}
        self._palm_parts = {li: [human.joint_index(n) for n in j] for li, j in joints.items()}
        # Fingers: (palm link, human distal joint) per fingertip; and per robot probe, the finger whose middle or
        # distal phalanx its human source lies on (closure).
        self._fingers = []
        self._closure_finger = np.full(len(rt.robot_probes), -1, dtype=np.int64)
        src_part = rt.human_probes.part[np.maximum(rt.src, 0)]
        for f, (tip, joint) in enumerate(self.fingertips.items()):
            j = human.joint_index(joint)
            if not (human.vertex_parts == j).any():
                raise ValueError(f"HandGrasp: fingertips[{tip!r}] = {joint!r} skins no vertex of this body model.")
            self._fingers.append((int(self._link_palm[kin.link_index(tip)]), j))
            if self.closure > 0:
                parts = [j]
                if joint[-1].isdigit():
                    middle = joint[:-1] + str(int(joint[-1]) - 1)
                    if middle not in human.joint_names:
                        raise ValueError(
                            f"HandGrasp: fingertips[{tip!r}] = {joint!r}: its middle phalanx {middle!r} (the name with "
                            f"its trailing digit less one) is not a joint of this body model."
                        )
                    parts.append(human.joint_index(middle))
                self._closure_finger[(rt.src >= 0) & np.isin(src_part, parts)] = f
        self._side_state: dict = {}  # per palm link: the grasp being watched by ``side_reapproach``
        self._open_q = np.zeros((human.num_frames, kin.n_q))  # the open-hand solutions
        self._q_closed = None  # the previous frame's closed-hand configuration

    def _palm_probes(self, li: int) -> np.ndarray:
        """Centroid of palm link ``li``'s probes, in the link frame."""
        probes = self.rt.robot_probes
        return probes.local[probes.carrier == li].mean(axis=0)

    def _touching_palms(self, human: HumanSurfaceMotion, obj_pos: np.ndarray, obj_rot: np.ndarray, t: int = 0) -> list:
        """``[(palm link, probe centroid in the link frame, human palm centroid (world))]`` of the palms whose human palm
        is within ``contact`` of a free object at frame ``t`` (objects at ``obj_pos``, ``obj_rot``).
        """
        rt = self.rt
        verts = np.asarray(human.vertices[t], dtype=np.float64)
        out = []
        for li, parts in self._palm_parts.items():
            on_link = rt.robot_probes.carrier == li
            part = verts[np.isin(human.vertex_parts, parts)]
            if not on_link.any() or not len(part):
                continue
            touching = False
            for _, _, d, _ in rt._free_distance(part, obj_pos, obj_rot):
                touching |= bool(d.min() <= self.contact)
            if touching:
                out.append((li, rt.robot_probes.local[on_link].mean(axis=0), part.mean(axis=0)))
        return out

    # ------------------------------------------------------------------
    # palm cold start
    # ------------------------------------------------------------------

    def cold_starts(self, human, obj_pos, obj_rot) -> list[np.ndarray]:
        if self.cold_start_palm_weight <= 0:
            return []
        palms = self._touching_palms(human, obj_pos, obj_rot)
        if not palms:
            return []
        return [
            self.rt._initial_configuration(
                human, obj_pos, obj_rot, pulls=palms, pull_weight=self.cold_start_palm_weight
            )
        ]

    def rank_start(self, human, frame: Frame, q, obj_pos, obj_rot, n_starts: int) -> float:
        """Minus the number of touching palms that hold the object on the human palm's side (with two starts or more).

        Either start can be the wrong one: without the pull a hand of other reach started on the far side of a lamp
        pole, with it a palm pulled onto a chair's back rail put the arm through the chair. The cost alone does not
        see the side: the OMOMO palms sink into poles to their middle.
        """
        if n_starts < 2:
            return 0.0
        palms = self._touching_palms(human, frame.demo_pos, frame.demo_rot)
        return -float(self._palms_facing(palms, human, frame.demo_pos, frame.demo_rot, q, obj_pos, obj_rot))

    def _palms_facing(self, palms, human, demo_pos, demo_rot, q, obj_pos, obj_rot, t: int = 0) -> int:
        """How many of ``palms`` (``_touching_palms``) hold the object on the side of the palm the human palm does at
        frame ``t`` (``_palm_offsets``).
        """
        kin = self.rt.kin
        kin.set_configuration(q)
        link_pos, link_rot = kin.link_poses()
        n = 0
        for li, local, c_h in palms:
            h_off, _, r_off = self._palm_offsets(
                li, local, c_h, human, t, demo_pos, demo_rot, link_pos, link_rot, obj_pos, obj_rot
            )
            n += int((h_off > 0) == (r_off > 0))
        return n

    def _palm_offsets(self, li, local, c_h, human, t, demo_pos, demo_rot, link_pos, link_rot, obj_pos, obj_rot):
        """``(human offset, human depth, robot offset)`` (m) of palm link ``li`` at frame ``t``.

        An offset is where the nearest free object's material is near the palm centre (robot: its probes' centroid
        ``local``; human: its part's centroid ``c_h``), along the palm normal from that centre (+: in front of the
        palm): the centroid of that object's surface samples within 3 cm of its point nearest the centre (on a pole,
        about its axis), or that point when fewer than 5 are. The nearest point alone read a pole held against a sunk
        human palm as behind it. The human depth is the centre's signed distance to that object (negative: inside).
        """
        rt = self.rt
        jw = self._palm_joint[li]
        n_h = human.joint_rot[t, jw] @ human.rest_joint_rot[jw].T @ self._normal
        n_r = link_rot[li] @ rt._tpose_link_rot[li].T @ self._normal
        c_r = link_pos[li] + link_rot[li] @ local
        out = []
        for c, nrm, pos, rot in ((c_h, n_h, demo_pos, demo_rot), (c_r, n_r, obj_pos, obj_rot)):
            best = None
            for ci, oi, d, w in rt._free_distance(c, pos, rot):
                if best is None or abs(d[0]) < abs(best[0]):
                    g = rt.channels[ci].geodesics
                    near = g.tree.query_ball_point(w[0], 0.03)
                    m = g.points[near].mean(axis=0) if len(near) >= 5 else w[0]
                    best = (float(d[0]), float((rot[oi] @ m + pos[oi] - c) @ nrm))
            out.append(best if best is not None else (np.inf, 0.0))
        return out[0][1], out[0][0], out[1][1]

    # ------------------------------------------------------------------
    # per frame
    # ------------------------------------------------------------------

    def frame(self, human: HumanSurfaceMotion, frame: Frame, forward: bool) -> None:
        rt = self.rt
        if self.closure > 0:
            for ci, oi in enumerate(rt._channel_object):
                if oi >= 0:
                    self._grasp_closure(human, frame.t, frame.human_t[ci], frame.robot_t[ci], frame.demo_rot[oi])
        if forward and self._side_state:
            frame.extra["palm_goals"] = self._fading_goals(human, frame)

    def _grasp_closure(self, human: HumanSurfaceMotion, t: int, h: Targets, robot_t: Targets, R) -> None:
        """Mark the probes of a grasping hand that close on the object (see ``closure``) in ``robot_t.closed``."""
        rt = self.rt
        part = rt.human_probes.part
        dist = h.distance
        skin = rt.human_probes.normals(human.vertices[t].astype(np.float64)) @ R  # channel frame
        facing = (skin * h.normal).sum(axis=1) < -0.3  # the skin looks at the object
        src = np.maximum(rt.src, 0)
        hands: dict[int, list[int]] = {}
        for f, (li, _) in enumerate(self._fingers):
            hands.setdefault(li, []).append(f)
        close = np.zeros(len(rt.robot_probes), dtype=bool)
        src_part = part[src]
        for li, fingers in hands.items():
            palm = self._palm_parts[li]
            palm_d = dist[np.isin(part, palm)].min(initial=np.inf)
            tip_d = np.array([dist[part == self._fingers[f][1]].min(initial=np.inf) for f in fingers])
            if not (palm_d <= self.contact or (tip_d <= self.contact).sum() >= 2):
                continue
            for f, d in zip(fingers, tip_d):
                if d <= self.closure:
                    close |= self._closure_finger == f
            if palm_d <= self.contact:
                # The palm keeps its contact as strongly, or the closing fingers drag the hand along the object.
                close |= (rt.src >= 0) & np.isin(src_part, palm) & (dist[src] <= self.contact)
        robot_t.closed = close & facing[src] & np.isfinite(robot_t.distance)

    def set_terms(self, lin: Linearization, s: ProbeSet) -> None:
        targets = s.targets
        if s.kind != "robot" or targets.closed is None or lin.frame.stage != "closure":
            return
        rt = self.rt
        n_v = rt.kin.n_v
        d = s.d
        closing = targets.closed & np.isfinite(d)
        probe_palm = self._link_palm[rt.robot_probes.carrier]
        for pl in np.unique(probe_palm[closing]):
            # A hand closes once its palm faces the object: the palm probes whose human sources touch it are within
            # 1.5 cm of its nearest (its back is a palm's thickness, 3.3 cm, farther) and within 5 cm. Closed earlier,
            # the fist met the object first and kept it out of the palm.
            on_palm = rt.robot_probes.carrier == pl
            touch = closing & on_palm
            faces = touch.any() and d[touch].min() <= d[on_palm].min() + 0.015 and d[touch].min() < 0.05
            if pl < 0 or (touch.any() and not faces):
                closing &= probe_palm != pl
        kc = np.flatnonzero(closing)
        if not len(kc):
            return
        # Each closing probe onto the point of the object its human source is nearest (the side of a pole the human
        # thumb is on), through the hand's own joints only.
        _, inv, cnt = np.unique(rt.robot_probes.carrier[kc], return_inverse=True, return_counts=True)
        sw = np.sqrt(self.closure_weight / cnt[inv])
        miss = s.x[kc] - targets.witness[kc]
        for axis in range(3):
            gamma = np.zeros((len(kc), 3))
            gamma[:, axis] = 1.0
            rows = rt._rows("robot", s.ci, kc, s.world[kc], s.R, s.tau, lin.obj_pos, lin.link_pos, gamma)
            J = lin.gn.dense(rows)
            J[:, :n_v][:, ~self._hand_cols] = 0.0
            J[:, n_v:] = 0.0
            lin.gn.add_dense("grasp_closure", sw * miss[:, axis], sw[:, None] * J, 1.0)

    def terms(self, lin: Linearization) -> None:
        rt = self.rt
        n_x = rt.n_x
        link_pos, link_rot, link_jac = lin.link_pos, lin.link_rot, lin.link_jac
        for li, local, p_goal, R_goal, w in lin.frame.extra.get("palm_goals", ()):
            # The palm's probe centroid at the human palm's place relative to the object across the palm (the component
            # along the palm normal is left to the depth terms: the human palm sinks into what it holds, and a goal
            # there pressed the robot palm into a pole), and the palm at the human palm's orientation. The
            # non-penetration rows keep the hand on the side it starts on.
            n = R_goal @ rt._tpose_link_rot[li].T @ self._normal
            across = np.eye(3) - np.outer(n, n)
            p = link_pos[li] + link_rot[li] @ local
            J = np.vstack(
                [
                    across @ (link_jac[li, 0:3] - skew(p - link_pos[li]) @ link_jac[li, 3:6]),
                    _PALM_ROT_SCALE * link_jac[li, 3:6],
                ]
            )
            r = np.concatenate(
                [across @ (p - p_goal), _PALM_ROT_SCALE * axis_angle_from_quat(R_to_quat(link_rot[li] @ R_goal.T))]
            )
            lin.gn.add_dense("palm_goal", r, pad(J, n_x), w)

    def after_frame(self, run: Run, frame: Frame, q, obj_pos, obj_rot, penetration: float):
        t = frame.t
        if self.side_reapproach:
            q, penetration = self._reapproach(run, frame, q, obj_pos, obj_rot, penetration)
        self._open_q[t] = q
        q_out = q
        if self.closure > 0 and self._fingers:
            q_out, closed_pen = self._close_hands(q, obj_pos, obj_rot, frame, self._q_closed)
            self._q_closed = q_out.copy()
            penetration = max(penetration, closed_pen)
        return q, q_out, penetration

    def _close_hands(self, q, obj_pos, obj_rot, frame: Frame, q_closed):
        """The frame's solution with the fingers closed on what the hands hold (``closure``).

        A trust-region solve of the hand joints alone, from the frame's solution and smoothed toward the previous
        frame's closed hands, with the closure residual on. Closed in the main solve instead, the fingers' contact rows
        pushed the hand and the object about and the next frames inherited it. Returns the configuration and its
        deepest probe penetration (m).
        """
        extra = {k: v for k, v in frame.extra.items() if k != "palm_goals"}
        stage = replace(
            frame,
            q_prev=frame.q_prev if q_closed is None else q_closed,
            free_cols=self._hand_cols,
            stage="closure",
            extra=extra,
        )
        q_c, _, _, local, _ = self.rt.solve(q.copy(), obj_pos, obj_rot, stage, self.rt.iterations)
        return q_c, local.penetration

    # ------------------------------------------------------------------
    # side re-approach
    # ------------------------------------------------------------------

    def _reapproach(self, run: Run, frame: Frame, q, obj_pos, obj_rot, penetration):
        """The side check at frame ``t``, and the re-approach of each palm on the wrong side.

        The human side counts when the human palm centre is outside the object, what is within 4 cm of it is one thin
        part (a pole, leg, slat or rail), and that part is at least 5 mm in front of (or behind) the palm, on the same
        side as at the frame before. The robot palm is wrong when the part is more than 1 cm on its other side. A grasp
        ends when the human palm has been off the objects for 5 frames; each grasp gets at most two attempts. Returns
        the frame's open-hand configuration and its deepest probe penetration (m), corrected when accepted.
        """
        kin = self.rt.kin
        t, human, state, info = frame.t, run.human, self._side_state, run.info
        demo_pos, demo_rot = frame.demo_pos, frame.demo_rot
        palms = self._touching_palms(human, demo_pos, demo_rot, t)
        touching = {li for li, _, _ in palms}
        for li in [k for k, v in state.items() if k not in touching and t - v["last"] > 5]:
            del state[li]
        for li, local, c_h in palms:
            st = state.setdefault(li, {"last": t, "tries": 0, "floor": -1})
            st["last"] = t
            if st["tries"] >= 2:
                continue
            kin.set_configuration(q)
            link_pos, link_rot = kin.link_poses()
            h_off, h_depth, r_off = self._palm_offsets(
                li, local, c_h, human, t, demo_pos, demo_rot, link_pos, link_rot, obj_pos, obj_rot
            )
            prev, st["side"] = st.get("side", 0.0), h_off
            if h_depth < 0.002 or abs(h_off) < 0.005 or prev * h_off <= 0.0 or r_off * np.sign(h_off) > -0.01:
                continue
            if not self._thin_part_near(c_h, demo_pos, demo_rot):
                continue
            st["tries"] += 1
            fixed = self._reapproach_palm(
                run, frame, li, local, c_h, np.sign(h_off), q, obj_pos, obj_rot, penetration, st["floor"]
            )
            if isinstance(fixed, tuple):
                q, penetration = fixed
                st["floor"] = st["goal_from"] = t
                info["side_reapproach"][t] = 1.0
            else:
                info["side_reapproach"][t] = fixed
        return q, penetration

    def _palm_goal(self, li, human, f, demo_pos, demo_rot, obj_pos, obj_rot) -> tuple[np.ndarray, np.ndarray]:
        """The human palm's place (its part's centroid) and orientation at frame ``f`` relative to the free object it is
        nearest (at its demonstrated pose), carried onto that object's robot pose: the target position of palm link
        ``li``'s probe centroid and the target orientation of the link.
        """
        rt = self.rt
        jw = self._palm_joint[li]
        c = np.asarray(human.vertices[f][np.isin(human.vertex_parts, self._palm_parts[li])], dtype=np.float64)
        c = c.mean(axis=0)
        oi, best = None, np.inf
        for _, o, d, _ in rt._free_distance(c, demo_pos, demo_rot):
            if abs(d[0]) < best:
                oi, best = o, abs(d[0])
        to_robot = obj_rot[oi] @ demo_rot[oi].T
        R = to_robot @ human.joint_rot[f, jw] @ human.rest_joint_rot[jw].T @ rt._tpose_link_rot[li]
        return obj_pos[oi] + to_robot @ (c - demo_pos[oi]), R

    def _fading_goals(self, human, frame: Frame) -> list:
        """Palm goals for this frame's solve after an accepted re-approach: weight falling linearly from
        ``_PALM_GOAL_WEIGHT`` to 0 over ``_REAPPROACH_FADE`` frames, so the arm settles from the goal pose gradually
        instead of in one frame.
        """
        out = []
        for li, st in self._side_state.items():
            k = frame.t - st.get("goal_from", -(10**9))
            if 0 < k < _REAPPROACH_FADE:
                p, R = self._palm_goal(
                    li, human, frame.t, frame.demo_pos, frame.demo_rot, frame.obj_prev_pos, frame.obj_prev_rot
                )
                out.append((li, self._palm_probes(li), p, R, _PALM_GOAL_WEIGHT * (1.0 - k / _REAPPROACH_FADE)))
        return out

    def _thin_part_near(self, c, pos, rot, radius=0.04, width=0.05) -> bool:
        """Whether the surface samples of the free object nearest ``c`` (world), within ``radius`` of its point nearest
        ``c``, form one thin part: four standard deviations of their second principal spread under ``width``.
        """
        best = None
        for ci, _, d, w in self.rt._free_distance(c, pos, rot):
            if best is None or abs(d[0]) < best[0]:
                best = (abs(d[0]), ci, w[0])
        if best is None:
            return False
        g = self.rt.channels[best[1]].geodesics
        near = g.points[g.tree.query_ball_point(best[2], radius)]
        if len(near) < 10:
            return False
        spread = np.linalg.eigvalsh(np.cov((near - near.mean(axis=0)).T))
        return bool(4.0 * np.sqrt(max(spread[1], 0.0)) < width)

    def _side_dofs(self, li) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(velocity columns, finger configuration indices, links)`` of the arm that carries palm link ``li`` (its
        limb) and of the fingers that hang from it.
        """
        rt, kin = self.rt, self.rt.kin
        arm = rt._limb_of_link[li]
        finger = (self._link_palm[kin._joint_link] == li) & (kin._joint_link != li)  # not the palm's own joint
        side = (rt._joint_limb == arm) | finger
        cols = np.zeros(kin.n_v, dtype=bool)
        cols[kin._joint_var[side]] = True
        links = np.flatnonzero((rt._limb_of_link == arm) | (self._link_palm == li))
        return cols, rt._joint_q[finger], links

    def _clearance(self, q, links, obj_pos, obj_rot) -> float:
        """Smallest signed distance (m) from the probes on ``links`` to the free objects, at ``q``."""
        rt = self.rt
        rt.kin.set_configuration(q)
        on = np.isin(rt.robot_probes.carrier, links)
        world = rt.robot_probes.world(*rt.kin.link_poses())[on]
        out = np.inf
        if len(world):
            for _, _, d, _ in rt._free_distance(world, obj_pos, obj_rot):
                out = min(out, float(d.min()))
        return out

    def _reapproach_start(self, q, li, local, cols, finger_q, links, p_tgt, R_tgt, back, obj_pos, obj_rot):
        """``q`` with the hand of palm ``li`` open (T-pose fingers) and its arm (``cols``) fitted so that the palm's
        probe centroid ``local`` is at ``p_tgt`` and the palm link at orientation ``R_tgt``. No contact rows: the
        target is moved along ``back`` by 5 mm until the arm and hand are clear of the objects (up to 6 cm).
        """
        rt, kin = self.rt, self.rt.kin
        out = q.copy()
        out[finger_q] = np.clip(
            rt._q_tpose[finger_q], kin.joint_lower[finger_q - kin.n_base_q], kin.joint_upper[finger_q - kin.n_base_q]
        )
        arm = cols.copy()
        arm[kin._joint_var[(self._link_palm[kin._joint_link] == li) & (kin._joint_link != li)]] = False
        tgt = np.asarray(p_tgt, dtype=np.float64).copy()
        for _ in range(13):
            for _ in range(60):
                kin.set_configuration(out)
                link_pos, link_rot = kin.link_poses()
                jac = kin.link_jacobians(link_pos, link_rot)
                p = link_pos[li] + link_rot[li] @ local
                r = np.concatenate([p - tgt, _PALM_ROT_SCALE * axis_angle_from_quat(R_to_quat(link_rot[li] @ R_tgt.T))])
                J = np.vstack([jac[li, 0:3] - skew(p - link_pos[li]) @ jac[li, 3:6], _PALM_ROT_SCALE * jac[li, 3:6]])
                J[:, ~arm] = 0.0
                lo = np.where(arm, -rt.trust_region[1], 0.0)
                lo, hi = rt._joint_bounds(out, lo, -lo)
                step, _ = solve_qp(
                    J.T @ J + 1e-4 * np.eye(kin.n_v), J.T @ r, np.zeros((0, kin.n_v)), np.zeros(0), lo, hi
                )
                out = kin.integrate(out, step)
                if np.abs(step).max() < rt.step_tolerance:
                    break
            if self._clearance(out, links, obj_pos, obj_rot) > -0.001:
                break
            tgt = tgt + 0.005 * back
        return out

    def _reapproach_palm(self, run: Run, frame: Frame, li, local, c_h, side, q, obj_pos, obj_rot, penetration, floor):
        """One re-approach of palm link ``li`` at frame ``t``; ``side`` is +1 when the human palm holds the object in
        front of it.

        That arm alone is fitted to the human palm's place and orientation relative to the object, with the hand open,
        backed off along the palm normal until clear (``_reapproach_start``); the frame is solved again with only that
        arm and hand free and a palm goal. If the palm then faces the object as the human palm does, with no failed QP
        and no deeper penetration, the earlier frames are solved again backwards from it in the same way until the hand
        is 5 cm clear of the objects, and joined to the first solution with no arm joint moving more than 0.08 rad a
        frame (8 to 30 frames). Returns ``(configuration, penetration)`` of frame ``t`` when accepted (earlier frames
        rewritten in ``run``); else nothing changes and a negative code says why: -1 the re-solved frame fails (its
        palm is not on the human's side, a QP failed, or it penetrates deeper), -2 the hand does not come clear within
        45 frames, -3 the join would pass through the object (or runs out of frames), -4 a rewritten frame holds the
        hand deeper than ``_REAPPROACH_DEPTH``, and 0.5 mm deeper than it was.
        """
        rt, kin = self.rt, self.rt.kin
        t, human = frame.t, run.human
        demo_pos, demo_rot = frame.demo_pos, frame.demo_rot
        cols, finger_q, links = self._side_dofs(li)

        def goal(f, d_pos, d_rot, o_pos, o_rot):
            return self._palm_goal(li, human, f, d_pos, d_rot, o_pos, o_rot)

        def solve(q0, fr, o_pos, o_rot, g):
            stage = replace(
                fr,
                free_cols=cols,
                stage="side_reapproach",
                extra={**fr.extra, "palm_goals": [(li, local, *g, _PALM_GOAL_WEIGHT)]},
            )
            out = rt.solve(q0, o_pos, o_rot, stage, rt.first_frame_iterations)
            return out[0], out[3], out[4]

        p_tgt, R_tgt = goal(t, demo_pos, demo_rot, obj_pos, obj_rot)
        back = -side * (R_tgt @ rt._tpose_link_rot[li].T @ self._normal)
        start = self._reapproach_start(q, li, local, cols, finger_q, links, p_tgt, R_tgt, back, obj_pos, obj_rot)
        # Not smoothed toward the previous frame, which holds the object on the wrong side.
        q_t, local_t, failed = solve(start, replace(frame, q_prev=None), obj_pos, obj_rot, (p_tgt, R_tgt))
        kin.set_configuration(q_t)
        link_pos, link_rot = kin.link_poses()
        _, _, r_off = self._palm_offsets(
            li, local, c_h, human, t, demo_pos, demo_rot, link_pos, link_rot, obj_pos, obj_rot
        )
        if failed or r_off * side <= 0.0 or local_t.penetration > max(penetration, 0.003):
            return -1

        # Backwards from the corrected frame, this arm and hand only, until the hand is clear of the objects in both
        # solutions; then joined to the first solution over ``blend`` frames.
        clear, reach = 0.05, 45
        q_idx = kin.n_base_q + np.flatnonzero(cols) - kin.n_base_v
        corrected = {t: q_t}
        joint_at = None
        nxt = q_t
        lowest = max(floor + 1, t - reach, 0)  # never into an earlier re-approach of this palm
        for s in range(t - 1, lowest - 1, -1):
            pos_s, rot_s = self._solved_objects(run, s)
            fr = rt._frame_inputs(human, s, nxt, pos_s, rot_s)
            q0 = self._open_q[s].copy()
            q0[q_idx] = nxt[q_idx]
            q_s, _, _ = solve(q0, fr, pos_s, rot_s, goal(s, fr.demo_pos, fr.demo_rot, pos_s, rot_s))
            corrected[s] = q_s
            nxt = q_s
            if (
                self._clearance(q_s, links, pos_s, rot_s) >= clear
                and self._clearance(self._open_q[s], links, pos_s, rot_s) >= clear
            ):
                joint_at = s
                break
        if joint_at is None:
            return -2  # the hand never came clear in the window: the join would pass through the object
        # Join: over the frames before ``joint_at``, the arm moves from the first solution to the corrected pose at
        # ``joint_at`` (the hand is clear there) by at most 0.08 rad per frame and joint; every joined frame must stay
        # clear.
        joined = {}
        gap = np.abs(corrected[joint_at][q_idx] - self._open_q[joint_at][q_idx]).max()
        blend = int(np.clip(np.ceil(gap / 0.08), 8, 30))
        for k in range(1, blend + 1):
            s = joint_at - k
            if s < max(floor + 1, 0):
                return -3
            w = 1.0 - k / (blend + 1)
            q_b = self._open_q[s].copy()
            q_b[q_idx] = (1.0 - w) * self._open_q[s][q_idx] + w * corrected[joint_at][q_idx]
            pos_s, rot_s = self._solved_objects(run, s)
            if self._clearance(q_b, links, pos_s, rot_s) < -0.002:
                return -3
            joined[s] = q_b
        # Close the hands of the earlier frames as the forward pass does; no rewritten frame may hold this arm and hand
        # (exact distance; the solve's own grid measure reads shallower) deeper than ``_REAPPROACH_DEPTH`` and 0.5 mm
        # deeper than it was.
        new_q, new_pen, q_closed = {}, {}, None
        for s in sorted({**corrected, **joined}):
            if s == t:
                continue
            q_s = corrected.get(s, joined.get(s))
            pos_s, rot_s = self._solved_objects(run, s)
            fr = rt._frame_inputs(human, s, q_s if q_closed is None else q_closed, pos_s, rot_s)
            q_out = q_s
            if self.closure > 0 and self._fingers:
                q_out, _ = self._close_hands(q_s, pos_s, rot_s, fr, q_closed)
                q_closed = q_out.copy()
            pen = max(-self._clearance(q_out, links, pos_s, rot_s), 0.0)
            was = max(-self._clearance(run.out_q[s], links, pos_s, rot_s), 0.0)
            if pen > max(was + 0.0005, _REAPPROACH_DEPTH):
                return -4
            new_q[s], new_pen[s] = (q_out, q_s), pen
        for s, (q_out, q_s) in new_q.items():
            run.out_q[s], self._open_q[s] = q_out, q_s
            run.info["penetration_mm"][s] = max(run.info["penetration_mm"][s], 1e3 * new_pen[s])
        return q_t, local_t.penetration

    @staticmethod
    def _solved_objects(run: Run, s: int) -> tuple[np.ndarray, np.ndarray]:
        """The object poses solved at frame ``s``: ``(positions, rotations)``."""
        return run.out_obj[:, s, :3].copy(), quat_to_R(run.out_obj[:, s, 3:].reshape(-1, 4)).reshape(-1, 3, 3)
