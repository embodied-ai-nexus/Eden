"""FlashDexRetarget's reset: hands and objects onto the reference."""

from __future__ import annotations

import math

import torch

from eden.managers.event_manager import EVENT_TERM_REGISTRY
from eden.managers.terms.events.motion import ReferenceStateInitialization
from eden.tasks.flashdexretarget.assets import robot_spec


@EVENT_TERM_REGISTRY.register()
class FDRHandReset(ReferenceStateInitialization):
    """Reset the hands onto the reference with FlashDexRetarget's Gaussian noise: per joint group a standard deviation
    (``wrist_trans`` m, ``wrist_rot_deg`` degrees, ``finger_range_frac`` of the joint range), all times
    ``noise_level``; then clipped to the joint limits.

    Parameters
    ----------
    hand_model : str
        The FlashDexRetarget robot, for its joint groups.
    sides : list[str]
        Its hands.
    noise_level : float
        Factor on every standard deviation (FlashDexRetarget: 0.1).
    wrist_trans : float
        Forearm-slide noise (m) at ``noise_level`` 1.
    wrist_rot_deg : float
        Forearm-hinge noise (deg) at ``noise_level`` 1.
    finger_range_frac : float
        Finger noise as a fraction of the joint range at ``noise_level`` 1.
    """

    hand_model: str = "xhand"
    sides: list[str] = ["right"]
    noise_level: float = 0.1
    wrist_trans: float = 0.01
    wrist_rot_deg: float = 10.0
    finger_range_frac: float = 0.125
    dofs_position_range: tuple[float, float] = (0.0, 0.0)

    def build(self):
        super().build()
        spec = robot_spec(self.hand_model, tuple(self.sides))
        names = self._env.motion_manager.get_target_dofs_name(self.entity_name)
        trans = {n for s in self.sides for n in spec.wrist_trans[s]}
        rot = {n for s in self.sides for n in spec.wrist_rot[s]}
        lim = self.robot.soft_dofs_pos_limits[0]
        if self._soft_limits_cols is not None:
            lim = lim[self._soft_limits_cols]
        frac = self.finger_range_frac * (lim[:, 1] - lim[:, 0])
        sigma = [
            self.wrist_trans if n in trans else math.radians(self.wrist_rot_deg) if n in rot else float(frac[i])
            for i, n in enumerate(names)
        ]
        self._sigma = torch.tensor(sigma, device=self.device) * self.noise_level

    def _dofs_pos_with_noise(self, dofs_pos: torch.Tensor, envs_idx: torch.Tensor) -> torch.Tensor:
        dofs_pos = dofs_pos + torch.randn_like(dofs_pos) * self._sigma
        return super()._dofs_pos_with_noise(dofs_pos, envs_idx)
