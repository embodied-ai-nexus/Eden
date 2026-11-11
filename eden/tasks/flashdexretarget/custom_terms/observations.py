"""FlashDexRetarget's observations, in each hand's palm frame."""

from __future__ import annotations

from typing import ClassVar

import torch

from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY, ObservationTerm
from eden.tasks.flashdexretarget.custom_terms.common import HandTermMixin
from eden.utils.geom import inv_quat, inv_transform_by_quat, quat_mul, quat_to_R, transform_by_quat
from eden.utils.string import term_subject


#: A contact force below this (N) is no contact.
FORCE_EPS = 1e-6


def _to_wrist(vec_w: torch.Tensor, palm_quat: torch.Tensor) -> torch.Tensor:
    """World vectors ``(N, S, ..., 3)`` into each side's palm frame (``palm_quat`` ``(N, S, 4)``)."""
    q = palm_quat
    while q.dim() < vec_w.dim():
        q = q.unsqueeze(2)
    return inv_transform_by_quat(vec_w, q.expand(*vec_w.shape[:-1], 4))


def _flat(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


def _log_norm(v: torch.Tensor) -> torch.Tensor:
    """``(..., 3) -> (..., 4)``: ``[unit * log(|v| + 1), log(|v| + 1)]``."""
    norm = v.norm(dim=-1, keepdim=True)
    mag = torch.log(norm + 1.0)
    return torch.cat([v / (norm + 1e-6) * mag, mag], dim=-1)


@OBSERVATION_TERM_REGISTRY.register()
class FDRHandObservation(HandTermMixin, ObservationTerm):
    """One of FlashDexRetarget's per-hand observations, stacked over the hands: ``(N, S * width)``.

    ``kind`` (FlashDexRetarget's term name, all in the palm ("wrist") frame unless ``_w``):

    ============================================  ==========================================================
    ``robot_wrist_pose_w``                        palm position + orientation (7)
    ``robot_wrist_lin_vel_w`` / ``_ang_vel_w``    palm velocities (3 each)
    ``robot_tip_trans_wrist``                     fingertips relative to the palm (5 x 3)
    ``robot_tip_lin_vel_wrist``                   fingertip velocities, palm axes (5 x 3)
    ``ref_future_mano_body_delta_w``              next-frame MANO bodies + tips minus the robot's, positions
                                                  then velocities (2 x (M + 5) x 3)
    ``obj_trans_wrist`` / ``obj_quat_wrist``      object pose (3 + 4)
    ``obj_lin_vel_wrist`` / ``obj_ang_vel_wrist`` object velocities (3 each)
    ``obj_point_cloud_wrist``                     object surface keypoints (P x 3)
    ``ref_future_obj_trans_delta_wrist``          next reference object position minus the object's (3)
    ``ref_future_obj_quat_delta_wrist``           next reference object orientation relative to the object's (4)
    ``ref_future_obj_point_cloud_delta_wrist``    keypoints at the next reference pose minus at the object's (P x 3)
    ``robot_body_contact_wrench_wrist``           per contact link, log-norm force + torque (C x 8)
    ``robot_body_contact_trans_wrist``            per contact link, contact point relative to the palm (C x 3)
    ``robot_links_obj_sdf``                       object signed distance at the palm + fingertips (6)
    ``robot_links_obj_normal_wrist``              its outward gradient (6 x 3)
    ``robot_body_contact_wrist``                  the two contact kinds in one term, wrench then point
    ``robot_links_obj_sdf_normal_wrist``          the two palm/fingertip SDF kinds in one term, distance then
                                                  gradient
    ============================================  ==========================================================

    The last two produce exactly the concatenation of the pair they fuse, from one SDF query instead of two: the
    task uses them.

    Parameters
    ----------
    kind : str
        The observation, from the table.
    n_points : int
        Object surface keypoints of the point-cloud observations.
    """

    kind: str = "robot_wrist_pose_w"
    n_points: int = 128

    _KINDS: ClassVar[tuple[str, ...]] = (
        "robot_wrist_pose_w",
        "robot_wrist_lin_vel_w",
        "robot_wrist_ang_vel_w",
        "robot_tip_trans_wrist",
        "robot_tip_lin_vel_wrist",
        "ref_future_mano_body_delta_w",
        "obj_trans_wrist",
        "obj_quat_wrist",
        "obj_lin_vel_wrist",
        "obj_ang_vel_wrist",
        "obj_point_cloud_wrist",
        "ref_future_obj_trans_delta_wrist",
        "ref_future_obj_quat_delta_wrist",
        "ref_future_obj_point_cloud_delta_wrist",
        "robot_body_contact_wrench_wrist",
        "robot_body_contact_trans_wrist",
        "robot_links_obj_sdf",
        "robot_links_obj_normal_wrist",
        "robot_body_contact_wrist",
        "robot_links_obj_sdf_normal_wrist",
    )
    _GEOMETRY_KINDS: ClassVar[frozenset[str]] = frozenset(
        {
            "obj_point_cloud_wrist",
            "ref_future_obj_point_cloud_delta_wrist",
            "robot_body_contact_wrench_wrist",
            "robot_body_contact_trans_wrist",
            "robot_links_obj_sdf",
            "robot_links_obj_normal_wrist",
            "robot_body_contact_wrist",
            "robot_links_obj_sdf_normal_wrist",
        }
    )

    def build(self) -> None:
        if self.kind not in self._KINDS:
            raise ValueError(f"{term_subject(self)}: unknown kind {self.kind!r}; choose from {list(self._KINDS)}.")
        self._geometry_ = self.kind in self._GEOMETRY_KINDS
        self._build_state(self.n_points)

    def compute(self) -> torch.Tensor:
        return _flat(getattr(self, f"_{self.kind}")())

    # Each returns (N, S, ...).
    def _robot_wrist_pose_w(self):
        return torch.cat(self.state.palm_pose(), dim=-1)

    def _robot_wrist_lin_vel_w(self):
        return self.state.palm_vel()[0]

    def _robot_wrist_ang_vel_w(self):
        return self.state.palm_vel()[1]

    def _robot_tip_trans_wrist(self):
        pos, quat = self.state.palm_pose()
        return _to_wrist(self.state.tips_pos() - pos[:, :, None], quat)

    def _robot_tip_lin_vel_wrist(self):
        return _to_wrist(self.state.tips_vel(), self.state.palm_pose()[1])

    def _ref_future_mano_body_delta_w(self):
        st = self.state
        trans = torch.cat([st.mano_bodies(1) - st.bodies_pos(), st.mano_tips(1) - st.tips_pos()], dim=2)
        vel = torch.cat(
            [
                st.mano_bodies(1, "links_lin_vel_w") - st.bodies_vel(),
                st.mano_tips(1, "links_lin_vel_w") - st.tips_vel(),
            ],
            dim=2,
        )
        return torch.cat([_flat_side(trans), _flat_side(vel)], dim=-1)

    def _obj_trans_wrist(self):
        pos, quat = self.state.palm_pose()
        return _to_wrist(self.state.object_pose()[0] - pos, quat)

    def _obj_quat_wrist(self):
        return quat_mul(inv_quat(self.state.palm_pose()[1]), self.state.object_pose()[1])

    def _obj_lin_vel_wrist(self):
        return _to_wrist(self.state.object_vel()[0], self.state.palm_pose()[1])

    def _obj_ang_vel_wrist(self):
        return _to_wrist(self.state.object_vel()[1], self.state.palm_pose()[1])

    def _keypoints(self, pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        """Each side's surface keypoints placed at ``pos`` / ``quat`` ``(N, S, ·)``: ``(N, S, P, 3)``."""
        out = []
        for i in range(len(self.sides)):
            pts = self.state.surface_points(i)
            out.append(pos[:, i, None] + transform_by_quat(pts, quat[:, i, None].expand(*pts.shape[:-1], 4)))
        return torch.stack(out, 1)

    def _obj_point_cloud_wrist(self):
        pos, quat = self.state.palm_pose()
        return _to_wrist(self._keypoints(*self.state.object_pose()) - pos[:, :, None], quat)

    def _ref_future_obj_trans_delta_wrist(self):
        return _to_wrist(self.state.ref_object(1)["pos"] - self.state.object_pose()[0], self.state.palm_pose()[1])

    def _ref_future_obj_quat_delta_wrist(self):
        palm_inv = inv_quat(self.state.palm_pose()[1])
        sim = quat_mul(palm_inv, self.state.object_pose()[1])
        nxt = quat_mul(palm_inv, self.state.ref_object(1)["quat"])
        return quat_mul(nxt, inv_quat(sim))

    def _ref_future_obj_point_cloud_delta_wrist(self):
        ref = self.state.ref_object(1)
        delta = self._keypoints(ref["pos"], ref["quat"]) - self._keypoints(*self.state.object_pose())
        return _to_wrist(delta, self.state.palm_pose()[1])

    def _contact(self):
        """Per contact link: SDF-gated force ``(N, S, C, 3)``, contact point ``(N, S, C, 3)``, found ``(N, S, C)``."""
        obj_pos, obj_quat = self.state.object_pose()
        links = self.state.contact_links_pos()
        force = self.state.contact_links_force()
        sdf, grad = [], []
        for i in range(len(self.sides)):
            d, g = self.state.sdf(i, links[:, i], obj_pos[:, i], obj_quat[:, i])
            sdf.append(d)
            grad.append(g)
        sdf, grad = torch.stack(sdf, 1), torch.stack(grad, 1)
        found = (force.norm(dim=-1) > FORCE_EPS) & (sdf < self.state.contact_margins[None])
        point = links - sdf[..., None] * grad
        return force * found[..., None], point, found, links

    def _contact_wrench_and_point(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``robot_body_contact_wrench_wrist`` and ``robot_body_contact_trans_wrist`` from one contact query."""
        force, point, found, links = self._contact()
        torque = torch.cross(point - links, force, dim=-1)
        pos, quat = self.state.palm_pose()
        wrench = torch.cat([_log_norm(_to_wrist(force, quat)), _log_norm(_to_wrist(torque, quat))], dim=-1)
        return wrench, _to_wrist((point - pos[:, :, None]) * found[..., None], quat)

    def _robot_body_contact_wrench_wrist(self):
        return self._contact_wrench_and_point()[0]

    def _robot_body_contact_trans_wrist(self):
        return self._contact_wrench_and_point()[1]

    def _robot_body_contact_wrist(self):
        return torch.cat([_flat(x) for x in self._contact_wrench_and_point()], dim=-1)

    def _palm_tips_sdf(self):
        pos, quat = self.state.palm_pose()
        pts = torch.cat([pos[:, :, None], self.state.tips_pos()], dim=2)  # (N, S, 6, 3)
        obj_pos, obj_quat = self.state.object_pose()
        out = [self.state.sdf(i, pts[:, i], obj_pos[:, i], obj_quat[:, i]) for i in range(len(self.sides))]
        return torch.stack([o[0] for o in out], 1), _to_wrist(torch.stack([o[1] for o in out], 1), quat)

    def _robot_links_obj_sdf(self):
        return self._palm_tips_sdf()[0]

    def _robot_links_obj_normal_wrist(self):
        return self._palm_tips_sdf()[1]

    def _robot_links_obj_sdf_normal_wrist(self):
        return torch.cat([_flat(x) for x in self._palm_tips_sdf()], dim=-1)


def _flat_side(x: torch.Tensor) -> torch.Tensor:
    """``(N, S, ...) -> (N, S, -1)``."""
    return x.reshape(x.shape[0], x.shape[1], -1)


@OBSERVATION_TERM_REGISTRY.register()
class FDRFutureTrajectory(HandTermMixin, ObservationTerm):
    """FlashDexRetarget's ``ref_future_traj_wrist``: the next ``n_future`` reference frames relative to the current
    state, in the current palm frame, ``(N, n_future * S * 39)``.

    Per future frame and hand: object position (3), orientation (6D, 6), linear and angular velocity (3 + 3), all
    minus the simulated object's; MANO wrist position (3) and orientation (6D, 6) minus the palm's; MANO fingertips
    minus the robot's (15). The motion manager must serve ``n_future + 1`` frames, one control step apart.

    Parameters
    ----------
    n_future : int
        Future frames (FlashDexRetarget: 10).
    """

    n_future: int = 10

    def build(self) -> None:
        self._build_state()
        frames = self._env.motion_manager.num_frames
        if frames < self.n_future + 1:
            raise ValueError(
                f"{term_subject(self)}: n_future={self.n_future} reads {self.n_future + 1} reference frames but the "
                f"motion manager serves {frames}; set MotionManagerOptions.num_frames >= {self.n_future + 1}."
            )

    def compute(self) -> torch.Tensor:
        st = self.state
        palm_pos, palm_quat = st.palm_pose()
        obj_pos, obj_quat = st.object_pose()
        obj_lin, obj_ang = st.object_vel()
        tips = st.tips_pos()
        R_palm, R_obj = quat_to_R(palm_quat), quat_to_R(obj_quat)
        steps = []
        for k in range(1, self.n_future + 1):
            ref = st.ref_object(k)
            R_fut = quat_to_R(ref["quat"])
            d_rot = R_palm.transpose(-1, -2) @ R_fut @ R_obj.transpose(-1, -2) @ R_palm
            wpos, wquat = st.mano_wrist(k)
            m_rot = R_palm.transpose(-1, -2) @ quat_to_R(wquat)
            steps.append(
                torch.cat(
                    [
                        _to_wrist(ref["pos"] - obj_pos, palm_quat),
                        d_rot[..., :2].flatten(-2),
                        _to_wrist(ref["lin_vel"] - obj_lin, palm_quat),
                        _to_wrist(ref["ang_vel"] - obj_ang, palm_quat),
                        _to_wrist(wpos - palm_pos, palm_quat),
                        m_rot[..., :2].flatten(-2),
                        _flat_side(_to_wrist(st.mano_tips(k) - tips, palm_quat)),
                    ],
                    dim=-1,
                )
            )
        return _flat(torch.stack(steps, 1))


@OBSERVATION_TERM_REGISTRY.register()
class FDRTargetError(ObservationTerm):
    """FlashDexRetarget's ``dof_target_pos``: the action term's PD target minus the joint positions, ``(N, D)``.

    Parameters
    ----------
    action_term : str
        The action term whose position target is read.
    """

    action_term: str = "hand"

    def build(self) -> None:
        self._term = self._env.action_manager.get_term(self.action_term)

    def compute(self) -> torch.Tensor:
        term = self._term
        return term.processed_action - term.entity.get_dofs_pos(dofs_idx_local=term.genesis_dofs_idx)
