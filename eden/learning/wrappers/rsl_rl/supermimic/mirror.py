"""The left-right mirror image of a state, an action and a simulation context, for augmenting world-model batches.

The mirror reflects the world y axis (:mod:`eden.utils.symmetry`), and a joint takes the value of its left/right
partner, negated for a roll or yaw joint. That mapping follows from the joint names alone, so
:meth:`Mirror.check_kinematics` compares it with the robot's forward kinematics at construction. After BlinkMimic's
``mirror.py`` (https://github.com/fuyichen0725/blinkmimic, MIT).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from eden.utils.symmetry import mirror_xz_ang_vel_, mirror_xz_joint, mirror_xz_quat_, mirror_xz_y_negate_

if TYPE_CHECKING:
    from eden.learning.wrappers.rsl_rl.supermimic.models import Dynamics
    from eden.learning.wrappers.rsl_rl.supermimic.sim_context import SimContext

_SIDE = re.compile(r"(?<![A-Za-z])(left|right)(?=_)")
_FLIPPED_AXIS = re.compile(r"(?<![A-Za-z])(roll|yaw)(?![A-Za-z])")
#: Largest distance (m) :meth:`Mirror.check_kinematics` accepts between a link and its partner's mirror image.
_KINEMATICS_TOLERANCE_M = 1e-3


def mirror_name(name: str) -> str:
    """``name`` with ``left`` and ``right`` swapped; a name with neither is returned as it is."""
    return _SIDE.sub(lambda side: "right" if side.group(1) == "left" else "left", name)


def mirror_permutation(names: Sequence[str], what: str) -> list[int]:
    """The index of each name's mirror partner in ``names``."""
    index = {name: i for i, name in enumerate(names)}
    missing = [name for name in names if mirror_name(name) not in index]
    if missing:
        raise ValueError(
            f"The left-right mirror pairs each {what} with the one named with the other side, but {missing} have no "
            f"partner ({[mirror_name(name) for name in missing]}) among the {what}s {list(names)}. Mirror "
            "augmentation needs a left/right symmetric set; set runner_options.world_model.mirror_share=0 otherwise."
        )
    return [index[mirror_name(name)] for name in names]


def joint_signs(joint_names: Sequence[str]) -> list[float]:
    """-1 for a roll or yaw joint, whose angle changes sign under the mirror, and 1 for every other joint."""
    return [-1.0 if _FLIPPED_AXIS.search(name) else 1.0 for name in joint_names]


def context_permutation(
    columns: Sequence[tuple[str, str]], joint_names: Sequence[str]
) -> tuple[list[int], list[float]]:
    """The partner column and the sign of each simulation context column (``SimContext.columns``)."""
    signs = {"friction": 1.0, "com_x": 1.0, "com_y": -1.0, "com_z": 1.0}
    joint_sign = dict(zip(joint_names, joint_signs(joint_names), strict=True))
    unknown = sorted({kind for kind, _ in columns} - signs.keys() - {"encoder_bias"})
    if unknown:
        raise ValueError(
            f"The left-right mirror has no rule for simulation context columns of kind {unknown}; it mirrors "
            f"{[*signs, 'encoder_bias']}. Add the rule to context_permutation, or set "
            "runner_options.world_model.mirror_share=0."
        )
    index = {column: i for i, column in enumerate(columns)}
    missing = [column for column in columns if (column[0], mirror_name(column[1])) not in index]
    if missing:
        raise ValueError(
            f"The left-right mirror swaps each simulation context column with the one of the other side, but "
            f"{missing} have no partner among the columns {list(columns)}. Randomize both sides in the task's "
            "event, or set runner_options.world_model.mirror_share=0."
        )
    perm = [index[(kind, mirror_name(name))] for kind, name in columns]
    sign = [joint_sign[name] if kind == "encoder_bias" else signs[kind] for kind, name in columns]
    return perm, sign


class Mirror:
    """Mirrors full states, raw actions and normalized simulation contexts (module docstring).

    Parameters
    ----------
    joint_names : Sequence[str]
        The robot's joints, in the order of the state and the action.
    dynamics : Dynamics
        The world model, whose action scale and offset turn an action into the joint target that is mirrored.
    sim_context : SimContext
        The simulation context, whose columns are mirrored in their physical units.
    """

    def __init__(self, joint_names: Sequence[str], dynamics: Dynamics, sim_context: SimContext):
        self.num_joints = len(joint_names)
        device = dynamics.action_scale.device
        # On the device from the start: a captured CUDA graph refuses a host-to-device copy.
        self.joint_perm = torch.tensor(mirror_permutation(joint_names, "joint"), device=device)
        self.joint_sign = torch.tensor(joint_signs(joint_names), device=device)
        context_perm, context_sign = context_permutation(sim_context.columns, joint_names)
        self.context_perm = torch.tensor(context_perm, device=device)
        self.context_sign = torch.tensor(context_sign, device=device)
        self._dynamics = dynamics
        self._context = sim_context
        centre, half = sim_context.centre, sim_context.scale
        uneven = ((centre[self.context_perm] * self.context_sign - centre).abs() > 1e-6) | (
            (half[self.context_perm] - half).abs() > 1e-6
        )
        if uneven.any():
            names = [column for column, bad in zip(sim_context.columns, uneven.tolist(), strict=True) if bad]
            raise ValueError(
                f"Mirror augmentation (runner_options.world_model.mirror_share) replaces a simulation context by "
                f"its left-right mirror image, but the randomization ranges of columns {names} are not the mirror "
                "image of their partners', so the simulator never draws that context. Give both sides one range, "
                "centred on 0 for a lateral offset or a roll or yaw joint, or set mirror_share to 0."
            )

    def joints(self, x: torch.Tensor) -> torch.Tensor:
        """Joint positions, velocities or position targets ``(..., J)``."""
        return mirror_xz_joint(x, self.joint_perm, self.joint_sign)

    def state(self, full_state: torch.Tensor) -> torch.Tensor:
        """Full states ``(..., 13 + 2 J)``: ``[root_pos, root_quat, q, root_lin_vel_w, root_ang_vel_w, qd]``."""
        j = self.num_joints
        root_pos, root_quat, q, lin, ang, qd = full_state.split((3, 4, j, 3, 3, j), dim=-1)
        return torch.cat(
            [
                mirror_xz_y_negate_(root_pos.clone()),
                mirror_xz_quat_(root_quat.clone()),
                self.joints(q),
                mirror_xz_y_negate_(lin.clone()),
                mirror_xz_ang_vel_(ang.clone()),
                self.joints(qd),
            ],
            dim=-1,
        )

    def action(self, action: torch.Tensor) -> torch.Tensor:
        """Raw actions ``(..., J)``: the action whose joint target is the mirror image of ``action``'s."""
        return self._dynamics.action(self.joints(self._dynamics.target(action)))

    def sim_context(self, sim_ctx: torch.Tensor) -> torch.Tensor:
        """Normalized simulation contexts ``(..., dim)``, mirrored in their physical units."""
        centre, scale = self._context.centre, self._context.scale
        physical = sim_ctx * scale + centre
        return (physical[..., self.context_perm] * self.context_sign - centre) / scale

    def bodies(self, bodies: tuple[torch.Tensor, ...], link_perm: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """World ``(pos, quat, lin_vel, ang_vel)`` of links ``(..., L, ·)``, each link taking its partner's."""
        maps = (mirror_xz_y_negate_, mirror_xz_quat_, mirror_xz_y_negate_, mirror_xz_ang_vel_)
        return tuple(reflect(x[..., link_perm, :].clone()) for x, reflect in zip(bodies, maps, strict=True))

    @torch.no_grad()
    def check_kinematics(self, fk: torch.nn.Module, joint_limits: torch.Tensor, samples: int = 256) -> None:
        """Raise unless the tracked links (``fk.link_names``) of a mirrored state lie on the mirror image of their
        partner links, over ``samples`` random poses within ``joint_limits`` ``(J, 2)``.
        """
        device = joint_limits.device
        link_perm = torch.tensor(mirror_permutation(fk.link_names, "tracked link"), device=device)
        generator = torch.Generator().manual_seed(0)
        joints = self.num_joints
        lo, hi = joint_limits.cpu().unbind(-1)
        state = torch.cat(
            [
                torch.randn(samples, 3, generator=generator),
                torch.nn.functional.normalize(torch.randn(samples, 4, generator=generator), dim=-1),
                lo + torch.rand(samples, joints, generator=generator) * (hi - lo),
                torch.randn(samples, 6 + joints, generator=generator),
            ],
            dim=-1,
        ).to(device)

        def links(full_state: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return fk(*full_state.split((3, 4, joints, 3, 3, joints), dim=-1))

        expected = self.bodies(links(state), link_perm)[0]
        error = (links(self.state(state))[0] - expected).norm(dim=-1).amax(dim=0)
        worst = int(error.argmax())
        if error[worst] > _KINEMATICS_TOLERANCE_M:
            raise ValueError(
                f"The left-right mirror derived from the joint names does not match the robot's kinematics: over "
                f"{samples} random poses, link {fk.link_names[worst]!r} of a mirrored state lies up to "
                f"{float(error[worst]):.4f} m from the mirror image of its partner "
                f"{fk.link_names[int(link_perm[worst])]!r} (tolerance {_KINEMATICS_TOLERANCE_M} m). The mirror "
                "swaps left_/right_ joints and negates joints named roll or yaw, which fits a robot whose roll "
                "joints turn about x and yaw joints about z. Set runner_options.world_model.mirror_share=0 for "
                "this robot."
            )
