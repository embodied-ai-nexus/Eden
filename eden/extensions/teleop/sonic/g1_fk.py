"""G1 key-frame forward kinematics for 3-point calibration (Genesis-backed).

Computes the poses of the two wrist frames and the torso frame — with the
training-time local offsets applied — for a given 29-DOF joint configuration,
root fixed at the origin with identity orientation. This is the reference
frame the teleop encoder's ``vr_3point_local_*`` observations were trained
against, so the offsets must match
``gear_sonic/config/manager_env/commands/terms/motion.yaml`` exactly.

The kinematics run on the same Genesis model the robot process simulates
(:class:`~eden.options.robots.unitree_g1.G1Dofs29Dex3`), via a single-env
kinematics-only scene (``transient_robot_env``) held open for the process
lifetime. One subtlety: Genesis re-origins the base link onto its inertial
frame, so with the FK root at identity the world origin is the *solver* base
frame, not the MJCF pelvis body frame the reference (Pinocchio) uses. The key
frames are therefore expressed relative to the pelvis body frame recovered via
``get_pos``/``get_quat(relative=True)``, which cancels the re-origin exactly —
pinned against a MuJoCo reference in ``tests/extensions/teleop/sonic/test_g1_fk.py``.
"""

from __future__ import annotations

from contextlib import ExitStack

import numpy as np

from eden.options.robots.unitree_g1 import G1_DOFS29_NAME, G1Dofs29Dex3

# Key frames + local offsets (identical to the reference and the training config).
G1_KEY_FRAMES = {
    "left_wrist": ("left_wrist_yaw_link", np.array([0.18, -0.025, 0.0])),
    "right_wrist": ("right_wrist_yaw_link", np.array([0.18, 0.025, 0.0])),
    "torso": ("torso_link", np.array([0.0, 0.0, 0.35])),
}


class G1KeyFrameFk:
    """FK for the calibration key frames, root pinned at the origin.

    Builds the kinematics scene once (the first construction per process pays
    the kernel compile, up to ~30 s) and reuses it for every query; requires
    ``eden.init`` beforehand and a ``close()`` when done.
    """

    def __init__(self, robot_options: G1Dofs29Dex3 | None = None) -> None:
        import torch

        from eden.utils.genesis_runtime import transient_robot_env

        self._torch = torch
        self._stack = ExitStack()
        self._env, self._robot = self._stack.enter_context(transient_robot_env(robot_options or G1Dofs29Dex3()))
        robot = self._robot
        if list(robot.dofs_name[:29]) != list(G1_DOFS29_NAME):
            self._stack.close()
            raise ValueError("the FK robot's dofs_name must start with the 29 body joints in MJCF order")
        # Raises (naming the available links) on a miss — same fail-fast as a
        # bad body name in the reference FK.
        _, self._links_idx = self._robot.find_named_links_idx_local([body for body, _ in G1_KEY_FRAMES.values()])
        self._offsets = torch.tensor(
            np.stack([offset for _, offset in G1_KEY_FRAMES.values()]), dtype=torch.float32, device=self._env.device
        )
        self._n_qs = 7 + len(robot.dofs_name)

    def key_frame_poses(self, q_mj29: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
        """Key-frame poses for a 29-joint configuration (MJCF order, radians).

        Returns ``{key: {"position" (3,), "orientation_wxyz" (4,)}}`` with the
        local offsets folded into the positions.
        """
        from genesis.utils.geom import transform_by_quat

        from eden.utils.geom import subtract_frame_transforms

        torch = self._torch
        q_mj29 = np.asarray(q_mj29, dtype=np.float64).reshape(29)
        qpos = torch.zeros((1, self._n_qs), dtype=torch.float32, device=self._env.device)
        qpos[0, 3] = 1.0  # free-base identity quaternion (wxyz), root at the origin
        qpos[0, 7:36] = torch.as_tensor(q_mj29, dtype=torch.float32, device=self._env.device)
        self._robot.set_qpos(qpos)  # hands stay at zero; refreshes link poses

        n = len(self._links_idx)
        # relative=True: the MJCF pelvis body frame (the reference's FK root),
        # not the re-origined solver frame the qpos root addresses.
        root_pos = self._robot.get_pos(relative=True)
        root_quat = self._robot.get_quat(relative=True)
        link_pos = self._robot.get_links_pos(ls_idx_local=self._links_idx)[0]
        link_quat = self._robot.get_links_quat(ls_idx_local=self._links_idx)[0]
        rel_pos, rel_quat = subtract_frame_transforms(
            root_pos.expand(n, -1), root_quat.expand(n, -1), link_pos, link_quat
        )
        rel_pos = rel_pos + transform_by_quat(self._offsets, rel_quat)

        return {
            key: {
                "position": rel_pos[i].detach().cpu().numpy().astype(np.float64),
                "orientation_wxyz": rel_quat[i].detach().cpu().numpy().astype(np.float64),
            }
            for i, key in enumerate(G1_KEY_FRAMES)
        }

    def close(self) -> None:
        """Destroy the kinematics scene."""
        self._stack.close()
