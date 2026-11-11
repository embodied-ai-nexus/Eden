"""The retargeted robot as one kinematic tree over several Eden entities, with analytic Jacobians.

A dexterous humanoid is usually a composite (``G1Wuji``, ``OliWuji``): a floating-base body entity
plus hand entities welded to its wrists, each with its own ``qpos``. The solver needs one
configuration and one tangent space across all of them, and the Jacobian of any point on any link
with respect to every degree of freedom that moves it — including the body's joints for a point on a
fingertip, which no single entity's ``get_jacobian`` provides.

:class:`RigKinematics` builds that from the entities' joint anchors and axes. Forward kinematics
stays Genesis' (``set_qpos`` + link poses), so the link frames, joint placements and the meshes
probes are sampled from all come from one source. The tangent space is::

    v = [base translation (3, world), base rotation (3, world rotation vector), joints...]

for a floating root (the base block is absent for a fixed one), with the retraction
``p <- p + dp`` and ``R <- exp(dtheta) R`` — the same convention the paper uses for object poses, and
independent of how Genesis parameterises its free joint internally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import genesis as gs
import numpy as np
import torch

from eden.utils.geom import R_to_quat, axis_angle_from_quat, quat_from_axis_angle, quat_to_R, skew

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


class RigKinematics:
    """Kinematics of a robot spread over a root entity and welded child entities.

    Parameters
    ----------
    env : EnvBase
        A built env holding the entities (``num_envs`` of 1 is enough; env 0 is used).
    entity_names : list[str]
        The root entity first (floating or fixed base), then every entity welded below it. Each child
        must be attached (no free root of its own) and every joint must be fixed, revolute or
        prismatic, apart from the root's floating joint.

    Attributes
    ----------
    link_names : list[str]
        Every link of the rig, root entity first; names must be unique across the entities.
    n_v : int
        Tangent dimension (6 for a floating base, plus one per joint).
    n_q : int
        Configuration dimension (7 for a floating base, plus one per joint).
    var_names : list[str]
        One name per tangent entry (``base_x`` ... ``base_rz``, then the joint names).
    """

    def __init__(self, env: EnvBase, entity_names: list[str]) -> None:
        missing = [n for n in entity_names if n not in env.entities]
        if missing:
            raise ValueError(
                f"RigKinematics: entities {missing} are not in this env. Entities: {sorted(env.entities)}."
            )
        self.entities = [env.entities[n] for n in entity_names]
        self.entity_names = list(entity_names)
        root = self.entities[0]
        self.floating = not root.has_immobile_base
        for name, ent in zip(entity_names[1:], self.entities[1:]):
            if not ent.has_immobile_base:
                raise ValueError(
                    f"RigKinematics: entity {name!r} has a free root, so it is not welded to {entity_names[0]!r}. "
                    f"List only the root and the entities attached below it."
                )
        self.device = root.device

        # -- links ---------------------------------------------------------
        self.link_names: list[str] = []
        self._link_global: list[int] = []
        self._entity_link_slices: list[slice] = []
        for ent in self.entities:
            start = len(self.link_names)
            for link in ent.links:
                self.link_names.append(link.name)
                self._link_global.append(link.idx)
            self._entity_link_slices.append(slice(start, len(self.link_names)))
        dupes = sorted({n for n in self.link_names if self.link_names.count(n) > 1})
        if dupes:
            raise ValueError(f"RigKinematics: link names {dupes} appear in more than one entity of {entity_names}.")
        global_to_local = {g: i for i, g in enumerate(self._link_global)}
        all_links = {link.idx: link for ent in self.entities for link in ent.links}
        self.link_parent = np.array(
            [global_to_local.get(all_links[g].parent_idx, -1) for g in self._link_global], dtype=np.int64
        )
        roots = [self.link_names[i] for i in np.flatnonzero(self.link_parent < 0)]
        if len(roots) != 1:
            raise ValueError(
                f"RigKinematics: entities {entity_names} form {len(roots)} trees (roots {roots}); weld every child "
                f"entity below the root entity so the rig is a single tree."
            )

        # -- variables -----------------------------------------------------
        self.var_names: list[str] = (
            ["base_x", "base_y", "base_z", "base_rx", "base_ry", "base_rz"] if self.floating else []
        )
        self._entity_var_slices: list[slice] = []
        self._entity_q_slices: list[slice] = []
        q_cursor = 7 if self.floating else 0
        for ent in self.entities:
            start = len(self.var_names)
            self.var_names.extend(ent.dofs_name)
            self._entity_var_slices.append(slice(start, len(self.var_names)))
            n_joint_q = len(ent.dofs_name)
            own_q = slice(q_cursor - (7 if (ent is root and self.floating) else 0), q_cursor + n_joint_q)
            self._entity_q_slices.append(own_q)
            q_cursor += n_joint_q
            expect_q = n_joint_q + (7 if (ent is root and self.floating) else 0)
            if ent.n_qs != expect_q or ent.n_dofs != n_joint_q + (6 if (ent is root and self.floating) else 0):
                raise ValueError(
                    f"RigKinematics: entity has n_qs={ent.n_qs}, n_dofs={ent.n_dofs} for {n_joint_q} named DOFs; only "
                    f"one-DOF (revolute/prismatic) joints are supported besides the root's floating joint."
                )
        self.n_v = len(self.var_names)
        self.n_q = q_cursor
        self.n_base_v = 6 if self.floating else 0
        self.n_base_q = 7 if self.floating else 0

        # -- joints: child link, type, anchor/axis in the child link frame ----
        self._set_neutral()
        link_pos, link_rot = self.link_poses()
        joint_link, joint_var, joint_type, joint_anchor, joint_axis = [], [], [], [], []
        for e_idx, ent in enumerate(self.entities):
            names = list(ent.dofs_name)
            for joint in ent._entity.joints:
                jtype = int(joint.type)
                if jtype in (int(gs.JOINT_TYPE.FIXED), int(gs.JOINT_TYPE.FREE)):
                    continue
                if jtype not in (int(gs.JOINT_TYPE.REVOLUTE), int(gs.JOINT_TYPE.PRISMATIC)):
                    raise ValueError(
                        f"RigKinematics: joint {joint.name!r} of entity {self.entity_names[e_idx]!r} has type "
                        f"{gs.JOINT_TYPE(jtype).name}; only revolute and prismatic joints can be retargeted."
                    )
                if joint.name not in names:
                    raise ValueError(
                        f"RigKinematics: joint {joint.name!r} is not in entity {self.entity_names[e_idx]!r}'s dofs_name "
                        f"{names}; every actuated joint must be a named DOF."
                    )
                li = global_to_local[joint.link.idx]
                anchor = joint.get_anchor_pos().reshape(-1, 3)[0].cpu().numpy().astype(np.float64)
                axis = joint.get_anchor_axis().reshape(-1, 3)[0].cpu().numpy().astype(np.float64)
                joint_link.append(li)
                joint_var.append(self._entity_var_slices[e_idx].start + names.index(joint.name))
                joint_type.append(jtype)
                joint_anchor.append(link_rot[li].T @ (anchor - link_pos[li]))
                joint_axis.append(link_rot[li].T @ axis)
        self._joint_link = np.array(joint_link, dtype=np.int64)
        self._joint_var = np.array(joint_var, dtype=np.int64)
        self._joint_revolute = np.array(joint_type) == int(gs.JOINT_TYPE.REVOLUTE)
        self._joint_anchor_local = np.array(joint_anchor).reshape(-1, 3)
        self._joint_axis_local = np.array(joint_axis).reshape(-1, 3)

        # joint_moves_link[l, k]: joint k moves link l (k's child link is l or one of l's ancestors).
        n_links = len(self.link_names)
        chain = [set() for _ in range(n_links)]
        for li in range(n_links):
            cur = li
            while cur >= 0:
                chain[li].add(cur)
                cur = self.link_parent[cur]
        self.joint_moves_link = np.array(
            [[self._joint_link[k] in chain[li] for k in range(len(self._joint_link))] for li in range(n_links)],
            dtype=bool,
        ).reshape(n_links, len(self._joint_link))

        # -- limits (joint DOFs only; dofs_name order) ------------------------
        lower, upper = [], []
        for ent in self.entities:
            lo, hi = (
                b.reshape(-1, len(ent.dofs_name))[0].cpu().numpy().astype(np.float64) for b in ent.get_dofs_limit()
            )
            lower.append(lo)
            upper.append(hi)
        self.joint_lower = np.concatenate(lower)
        self.joint_upper = np.concatenate(upper)

    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------

    def _set_neutral(self) -> None:
        q = np.zeros(self.n_q)
        if self.floating:
            q[3] = 1.0
        self.set_configuration(q)

    def entity_qpos(self, q: np.ndarray) -> list[np.ndarray]:
        """Split a rig configuration into each entity's ``qpos`` (dofs_name order)."""
        return [np.asarray(q)[..., s] for s in self._entity_q_slices]

    def set_configuration(self, q: np.ndarray) -> None:
        """Write configuration ``q`` to the entities (runs Genesis forward kinematics)."""
        # The floating joint rotates the rig about its qpos position, whatever point of the root link
        # Genesis anchors it at, so that point is the lever origin of the base rotation columns.
        self._base_pivot = np.array(q[0:3], dtype=np.float64) if self.floating else None
        for ent, qe in zip(self.entities, self.entity_qpos(q)):
            ent.set_qpos(torch.as_tensor(qe, dtype=gs.tc_float, device=self.device).unsqueeze(0))

    def link_poses(self) -> tuple[np.ndarray, np.ndarray]:
        """World ``(L, 3)`` positions and ``(L, 3, 3)`` orientations of every rig link at the set configuration."""
        pos, quat = [], []
        for ent in self.entities:
            pos.append(ent.get_links_pos(relative=False).reshape(-1, len(ent.links), 3)[0])
            quat.append(ent.get_links_quat(relative=False).reshape(-1, len(ent.links), 4)[0])
        pos = torch.cat(pos).cpu().numpy().astype(np.float64)
        quat = torch.cat(quat).cpu().numpy().astype(np.float64)
        return pos, quat_to_R(quat)

    def integrate(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Retract tangent step ``v`` onto configuration ``q`` (``q (+) v``)."""
        q = np.array(q, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        if self.floating:
            q[0:3] += v[0:3]
            quat = R_to_quat(quat_to_R(quat_from_axis_angle(v[3:6])) @ quat_to_R(q[3:7]))
            # The hemisphere of q, not whichever the matrix conversion lands in: the root quaternion of a retargeted
            # clip then never jumps q -> -q between frames (finite-difference angular velocity and slerp read that).
            q[3:7] = quat if quat @ q[3:7] >= 0.0 else -quat
        q[self.n_base_q :] += v[self.n_base_v :]
        return q

    def difference(self, q1: np.ndarray, q0: np.ndarray) -> np.ndarray:
        """Tangent ``v`` with ``q0 (+) v = q1`` (``q1 (-) q0``)."""
        v = np.zeros(self.n_v)
        if self.floating:
            v[0:3] = q1[0:3] - q0[0:3]
            R1, R0 = quat_to_R(np.asarray(q1, dtype=np.float64)[3:7]), quat_to_R(np.asarray(q0, dtype=np.float64)[3:7])
            v[3:6] = axis_angle_from_quat(R_to_quat(R1 @ R0.T))
        v[self.n_base_v :] = np.asarray(q1)[self.n_base_q :] - np.asarray(q0)[self.n_base_q :]
        return v

    def link_index(self, name: str) -> int:
        """Rig index of link ``name``, raising with the available names if absent."""
        if name not in self.link_names:
            raise KeyError(f"Link {name!r} is not in rig {self.entity_names}. Links: {self.link_names}.")
        return self.link_names.index(name)

    # ------------------------------------------------------------------
    # Jacobians
    # ------------------------------------------------------------------

    def _joint_world(self, link_pos: np.ndarray, link_rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rot = link_rot[self._joint_link]
        anchor = link_pos[self._joint_link] + np.einsum("kab,kb->ka", rot, self._joint_anchor_local)
        axis = np.einsum("kab,kb->ka", rot, self._joint_axis_local)
        return anchor, axis

    def link_jacobians(self, link_pos: np.ndarray, link_rot: np.ndarray) -> np.ndarray:
        """Spatial Jacobians of every link: ``(L, 6, n_v)``, rows ``[v of the link origin; omega]`` (world).

        A point ``p`` on link ``l`` moves with ``J_v - [p - o_l]x J_omega``, so a scalar residual with
        world gradient ``c`` at ``p`` has the row ``[c; (p - o_l) x c]^T J6_l``: residuals on one link can
        be accumulated as a 6x6 block before touching the ``n_v`` columns.
        """
        n_links = len(self.link_names)
        jac = np.zeros((n_links, 6, self.n_v))
        if self.floating:
            jac[:, 0:3, 0:3] = np.eye(3)
            jac[:, 0:3, 3:6] = -skew(link_pos - self._base_pivot)
            jac[:, 3:6, 3:6] = np.eye(3)
        anchor, axis = self._joint_world(link_pos, link_rot)
        lever = link_pos[:, None, :] - anchor[None, :, :]  # (L, K, 3)
        lin = np.where(self._joint_revolute[None, :, None], np.cross(axis[None], lever), axis[None])
        ang = np.where(self._joint_revolute[None, :, None], axis[None], 0.0) * np.ones((n_links, 1, 1))
        moves = self.joint_moves_link[:, :, None]
        jac[:, 0:3, self._joint_var] = np.transpose(lin * moves, (0, 2, 1))
        jac[:, 3:6, self._joint_var] = np.transpose(ang * moves, (0, 2, 1))
        return jac
