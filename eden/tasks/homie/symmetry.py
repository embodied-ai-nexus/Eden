"""The homie task's sagittal mirror, as an rsl_rl data-augmentation function.

:func:`mirror_observations_and_actions` has rsl-rl-lib's ``data_augmentation_func`` signature, so the
one function serves HIMPPO's ``data_augmentation_func`` and stock PPO's symmetry extension alike.
Mirroring across the robot's sagittal plane swaps left/right joints, negates roll/yaw joints, and
negates the lateral velocity, its command and the yaw rate (OpenHomie's ``flip_g1_actor_obs`` /
``flip_g1_critic_obs`` / ``flip_g1_actions``).

The maps index the observations by position, so they hold for exactly the layout of ``_OBSERVATIONS``
in :mod:`eden.tasks.homie.config` — a frame of
``[command(4), base_ang_vel(3), projected_gravity(3), dofs_pos(29), dofs_vel(29), last_action(12)]``,
the ``policy`` group 6 stacked frames, the ``critic`` group one frame + the base linear velocity.
``tests/tasks/homie/test_config.py`` pins the config to these widths.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from eden.options.robots.unitree_g1 import G1_DOFS29_NAME

_N_DOF = 29
ACTION_DIM = 12
FRAME_DIM = 80  # command(4) + ang_vel(3) + gravity(3) + dofs_pos(29) + dofs_vel(29) + action(12)
HISTORY_LENGTH = 6
POLICY_DIM = FRAME_DIM * HISTORY_LENGTH  # 480
CRITIC_DIM = FRAME_DIM + 3  # + the privileged base linear velocity


def _sign_for(name: str) -> float:
    """Sagittal-mirror sign for a joint: roll/yaw negate, pitch/knee/elbow keep."""
    return -1.0 if ("roll" in name or "yaw" in name) else 1.0


def _build_dof_maps() -> tuple[list[int], list[float]]:
    """Left/right permutation and sign for the 29 DOFs (declaration order)."""
    perm = list(range(_N_DOF))
    leg_pairs = [(0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11)]
    arm_pairs = [(15, 22), (16, 23), (17, 24), (18, 25), (19, 26), (20, 27), (21, 28)]
    for a, b in leg_pairs + arm_pairs:
        perm[a], perm[b] = b, a  # swap L<->R (waist 12-14 stay)
    sign = [_sign_for(n) for n in G1_DOFS29_NAME]
    return perm, sign


_DOF_PERM, _DOF_SIGN = _build_dof_maps()
_ACTION_PERM = _DOF_PERM[:ACTION_DIM]  # leg subset (values all in 0..11)
_ACTION_SIGN = _DOF_SIGN[:ACTION_DIM]


def _build_frame_maps() -> tuple[list[int], list[float]]:
    """Permutation + sign for one 80-dim observation frame."""
    perm: list[int] = []
    sign: list[float] = []
    # command [vx, vy, wz, height]
    perm += [0, 1, 2, 3]
    sign += [1.0, -1.0, -1.0, 1.0]
    # base_ang_vel [wx, wy, wz]  (roll & yaw negate)
    perm += [4, 5, 6]
    sign += [-1.0, 1.0, -1.0]
    # projected_gravity [gx, gy, gz]  (y negates)
    perm += [7, 8, 9]
    sign += [1.0, -1.0, 1.0]
    # dofs_pos (29)
    perm += [10 + p for p in _DOF_PERM]
    sign += _DOF_SIGN
    # dofs_vel (29)
    perm += [39 + p for p in _DOF_PERM]
    sign += _DOF_SIGN
    # last_action (12 legs)
    perm += [68 + p for p in _ACTION_PERM]
    sign += _ACTION_SIGN
    return perm, sign


_FRAME_PERM, _FRAME_SIGN = _build_frame_maps()

# Per-dim mirror maps, keyed by feature dimension.
_POLICY_PERM = [_FRAME_PERM[j] + FRAME_DIM * k for k in range(HISTORY_LENGTH) for j in range(FRAME_DIM)]
_POLICY_SIGN = _FRAME_SIGN * HISTORY_LENGTH
_CRITIC_PERM = _FRAME_PERM + [FRAME_DIM, FRAME_DIM + 1, FRAME_DIM + 2]  # + base_lin_vel [vx, vy, vz]
_CRITIC_SIGN = _FRAME_SIGN + [1.0, -1.0, 1.0]

_MIRROR_BY_DIM: dict[int, tuple[list[int], list[float]]] = {
    POLICY_DIM: (_POLICY_PERM, _POLICY_SIGN),
    CRITIC_DIM: (_CRITIC_PERM, _CRITIC_SIGN),
    ACTION_DIM: (_ACTION_PERM, _ACTION_SIGN),
}

# Observation group -> its width in the config.
GROUP_DIMS: dict[str, int] = {"policy": POLICY_DIM, "critic": CRITIC_DIM}

_TENSOR_CACHE: dict[tuple[int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}


def _maps_for(dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (dim, device)
    if key not in _TENSOR_CACHE:
        perm, sign = _MIRROR_BY_DIM[dim]
        _TENSOR_CACHE[key] = (
            torch.as_tensor(perm, dtype=torch.long, device=device),
            torch.as_tensor(sign, dtype=torch.float32, device=device),
        )
    return _TENSOR_CACHE[key]


def mirror(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Mirror the last axis of ``x``, laid out as the ``dim``-wide table.

    ``out[..., i] = sign[i] * x[..., perm[i]]``.
    """
    if x.shape[-1] != dim or dim not in _MIRROR_BY_DIM:
        raise ValueError(
            f"homie symmetry: a {x.shape[-1]}-wide tensor has no mirror map (expected {dim}; the maps cover "
            f"{sorted(_MIRROR_BY_DIM)}). The maps are laid out for eden/tasks/homie/config.py's observation and "
            f"action terms; changing those means rebuilding them."
        )
    perm, sign = _maps_for(dim, x.device)
    return x[..., perm] * sign


def mirror_observations_and_actions(env=None, obs: TensorDict | None = None, actions: torch.Tensor | None = None):
    """rsl-rl-lib's ``data_augmentation_func``: each input stacked with its mirror along the batch axis.

    Parameters
    ----------
    env : VecEnv, optional
        Unused; part of the signature.
    obs : TensorDict | None
        Observation groups to mirror, any of ``policy`` / ``critic``.
    actions : torch.Tensor | None
        The 12 leg actions.

    Returns
    -------
    tuple[TensorDict | None, torch.Tensor | None]
        ``[original; mirrored]`` for each input given, ``None`` for each left out.
    """
    obs_aug = None
    if obs is not None:
        unknown = [k for k in obs.keys() if k not in GROUP_DIMS]
        if unknown:
            raise KeyError(
                f"homie symmetry has no mirror map for observation groups {unknown}; it covers {sorted(GROUP_DIMS)}."
            )
        obs_aug = TensorDict(
            {k: torch.cat([obs[k], mirror(obs[k], GROUP_DIMS[k])], dim=0) for k in obs.keys()},
            batch_size=[2 * obs.batch_size[0]],
            device=obs.device,
        )
    actions_aug = None if actions is None else torch.cat([actions, mirror(actions, ACTION_DIM)], dim=0)
    return obs_aug, actions_aug
