"""Left-right (sagittal-plane) mirror maps for the extreme-parkour teacher.

Used by the symmetry (mirror-equivariance) loss of
:class:`~eden.learning.wrappers.rsl_rl.rma.ppo.RmaPPO` (the teacher config names this
class in ``algorithm.mirror_class_name``) to push the policy toward
``pi(mirror(obs)) == mirror(pi(obs))`` and remove the asymmetric "limping" gaits an
unconstrained RMA teacher falls into.

The mirror is a reflection across the x-z plane (y -> -y): polar vectors negate y,
axial quantities (angular velocity, roll/yaw angles) negate x and z, and the legs
swap FL<->FR / RL<->RR (hip abduction also negates). The whole 753-d obs mirror is a
single (permutation, sign) pair; see the per-chunk map below (verified against the
observation term code). The 12-d action uses the leg-swap + hip-negate map.
"""

from __future__ import annotations

import torch

from eden.learning.wrappers.rsl_rl.rma.dims import RmaObsDims

# 12-DOF block, order [FL_hip, RL_hip, FR_hip, RR_hip, FL_thigh, RL_thigh, FR_thigh,
# RR_thigh, FL_calf, RL_calf, FR_calf, RR_calf]. Swap L<->R within each joint-type group of four and
# negate the four hip-abduction joints (their default pose is +-0.1, antisymmetric).
_DOF_PERM = [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9]  # out[i] = src[_DOF_PERM[i]]
_DOF_SIGN = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# Scandots grid geometry (measured_points_x x measured_points_y), y symmetric about 0.
_SCAN_NX = 12
_SCAN_NY = 11

# Gain-block (kp then kd) L/R swap pairs within a 12-joint block (no sign; gains > 0).
_GAIN_SWAP = [(0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11)]


class ParkourMirror:
    """Precomputed sagittal mirror for the flat 753-d obs and 12-d action."""

    def __init__(self, dims: RmaObsDims, device):
        self.dims = dims
        perm = list(range(dims.total))
        sign = [1.0] * dims.total

        def apply_proprio(base: int) -> None:
            # angular velocity (wx,wy,wz)*0.25 -> negate wx, wz; roll (3) negate; pitch (4) keep
            for k in (0, 2, 3):
                sign[base + k] = -1.0
            # delta_yaw (6), delta_next_yaw (7) negate
            sign[base + 6] = -1.0
            sign[base + 7] = -1.0
            # dof position-delta (13), velocity (25), last action (37): leg-swap + hip-negate
            for off in (13, 25, 37):
                for i in range(12):
                    perm[base + off + i] = base + off + _DOF_PERM[i]
                    sign[base + off + i] = _DOF_SIGN[i]
            # foot contacts (49:53) order [FL,FR,RL,RR]: swap 0<->1, 2<->3
            c = base + 49
            perm[c + 0], perm[c + 1] = c + 1, c + 0
            perm[c + 2], perm[c + 3] = c + 3, c + 2

        # proprio [0:53]
        apply_proprio(0)

        # scandots [scan]: reverse the lateral (y) axis; flat index = x*NY + y
        ss = dims.scan_slice.start
        for x in range(_SCAN_NX):
            for y in range(_SCAN_NY):
                perm[ss + x * _SCAN_NY + y] = ss + x * _SCAN_NY + (_SCAN_NY - 1 - y)

        # priv_explicit [pe]: body lin vel (vx,vy,vz)*2 then zeros -> negate vy
        pe = dims.priv_explicit_slice.start
        sign[pe + 1] = -1.0

        # priv_latent [pl]: mass(1) | com xyz(3) | friction_mean(1) | kp(12) | kd(12)
        pl = dims.priv_latent_slice.start
        sign[pl + 2] = -1.0  # com_y
        for blk in (pl + 5, pl + 17):  # kp block, kd block
            for a, b in _GAIN_SWAP:
                perm[blk + a], perm[blk + b] = blk + b, blk + a

        # history [hist]: num_hist frames of the 53-d proprio (slots [6:8] pre-zeroed)
        hs = dims.hist_slice.start
        for f in range(dims.num_hist):
            apply_proprio(hs + dims.num_prop * f)

        self.obs_perm = torch.as_tensor(perm, dtype=torch.long, device=device)
        self.obs_sign = torch.as_tensor(sign, dtype=torch.float32, device=device)
        self.act_perm = torch.as_tensor(_DOF_PERM, dtype=torch.long, device=device)
        self.act_sign = torch.as_tensor(_DOF_SIGN, dtype=torch.float32, device=device)
        self._validate()

    def _validate(self) -> None:
        """Fail fast if the map is not a valid sign-flipped involution."""
        for name, perm, sign in (("obs", self.obs_perm, self.obs_sign), ("act", self.act_perm, self.act_sign)):
            n = perm.numel()
            if torch.sort(perm).values.ne(torch.arange(n, device=perm.device)).any():
                raise ValueError(f"ParkourMirror {name} permutation is not a bijection.")
            if perm[perm].ne(torch.arange(n, device=perm.device)).any():
                raise ValueError(f"ParkourMirror {name} permutation is not an involution.")
            if sign.mul(sign[perm]).sub(1.0).abs().max() > 1e-6:
                raise ValueError(f"ParkourMirror {name} sign map is not an involution.")

    def mirror_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return self.obs_sign * obs[:, self.obs_perm]

    def mirror_action(self, action: torch.Tensor) -> torch.Tensor:
        return self.act_sign * action[:, self.act_perm]
