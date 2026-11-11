"""Elastic suspension band for the SONIC teleop sim (port of the reference sim's ElasticBand).

The reference MuJoCo sim hangs the robot from a virtual spring-damper anchored
above the origin, **enabled at spawn** — the operator engages teleop first and
then drops the robot (key ``9``) so the policy catches it standing. The wrench
has a positional spring toward ``point + [0, 0, length]`` and an orientation
spring toward identity:

    f   = kp_pos * (point - pos + [0, 0, length]) - kd_pos * lin_vel
    tau = -kp_ang * rotvec(quat) - kd_ang * ang_vel

Keys: ``9`` toggles the band, ``7``/``8`` lower/raise ``length`` by 0.1 m.
"""

from __future__ import annotations

import genesis as gs
import numpy as np
import torch
from genesis.utils import geom as gu


class ElasticBand:
    """The virtual band's state and wrench law (world frame, wxyz quaternion)."""

    #: The keys :meth:`handle_key` takes, for the runtime's key help.
    key_help = "9 drop/suspend band, 7/8 band length"

    def __init__(self) -> None:
        self.kp_pos = 10000.0
        self.kd_pos = 1000.0
        self.kp_ang = 1000.0
        self.kd_ang = 10.0
        self.point = np.array([0.0, 0.0, 1.0])
        self.length = 0.0
        self.enable = True  # starts suspended; press 9 to drop

    def wrench(
        self, pos: np.ndarray, quat_wxyz: np.ndarray, lin_vel: np.ndarray, ang_vel: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """World-frame force and torque on the attachment link."""
        force = self.kp_pos * (self.point - pos + np.array([0.0, 0.0, self.length])) - self.kd_pos * lin_vel
        # Orientation spring toward identity: torque along the rotation vector.
        # Canonicalize the sign so the torque takes the short way (scipy's
        # as_rotvec semantics, which the reference relies on).
        quat = gu.normalize(quat_wxyz)
        if quat[0] < 0.0:
            quat = -quat
        angle = 2.0 * np.arccos(np.clip(quat[0], -1.0, 1.0))
        norm = float(np.linalg.norm(quat[1:]))
        rotvec = (quat[1:] / norm) * angle if norm > 1e-9 else np.zeros(3)
        torque = -self.kp_ang * rotvec - self.kd_ang * ang_vel
        return force, torque

    def handle_key(self, key: str) -> bool:
        """Apply a band key (``9`` toggle, ``7``/``8`` length); returns True if handled."""
        if key == "9":
            self.enable = not self.enable
            print(f"[ElasticBand] {'suspended' if self.enable else 'released'}")
            return True
        if key == "7":
            self.length -= 0.1
            print(f"[ElasticBand] length {self.length:+.1f} m")
            return True
        if key == "8":
            self.length += 0.1
            print(f"[ElasticBand] length {self.length:+.1f} m")
            return True
        return False


class ElasticBandApplier:
    """Applies the band wrench to a Genesis entity's link.

    Genesis clears applied external forces after every ``scene.step()``, so
    ``apply()`` must run once per scene step (each decimation substep), not
    once per control tick — the teleop runtime's env subclass does this.
    """

    def __init__(self, env, entity_name: str = "robot", link_name: str = "pelvis") -> None:
        self.band = ElasticBand()
        self._env = env
        entity = env.entities[entity_name]
        self._entity = entity
        resolved, _ = entity.find_named_links_idx_local([link_name], preserve_order=True)
        # The solver takes scene-global link indices, not the entity-local ones.
        self._links_idx = torch.tensor(
            [entity._entity.get_link(name).idx for name in resolved], dtype=torch.long, device=env.device
        )

    def apply(self) -> None:
        """Compute and apply the wrench for env 0 (teleop is single-env)."""
        if not self.band.enable:
            return
        # Base pose in the frame default_root_pos was specified in (the MJCF
        # root), matching the reference's free-joint qpos — not the re-origined
        # solver frame (see docs/guides/entities.md).
        pos = self._entity.get_pos(relative=True)[0]
        quat = self._entity.get_quat(relative=True)[0]
        lin_vel = self._entity.get_vel()[0]
        ang_vel = self._entity.get_ang()[0]
        force, torque = self.band.wrench(
            pos.detach().cpu().numpy().astype(np.float64),
            quat.detach().cpu().numpy().astype(np.float64),
            lin_vel.detach().cpu().numpy().astype(np.float64),
            ang_vel.detach().cpu().numpy().astype(np.float64),
        )
        device = self._env.device
        self._env.rigid_solver.apply_links_external_wrench(
            force=torch.as_tensor(force, dtype=torch.float32, device=device).reshape(1, 1, 3),
            torque=torch.as_tensor(torque, dtype=torch.float32, device=device).reshape(1, 1, 3),
            links_idx=self._links_idx,
            envs_idx=torch.zeros(1, dtype=torch.long, device=device),
            ref=gs.link_ref_frame.link_COM,
            local=False,
        )
