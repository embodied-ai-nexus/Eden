"""UWLab's OmniReset episode checks: the robot running away, and whether the pair is assembled."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.managers.termination_manager import TERMINATION_TERM_REGISTRY, TerminationTerm
from eden.utils.geom import euler_xyz_from_quat, subtract_frame_transforms, transform_by_quat, wrap_to_pi

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.tasks.omnireset.constants import Pair


@TERMINATION_TERM_REGISTRY.register()
class ArmOverspeed(TerminationTerm):
    """End an episode once any listed joint moves faster than twice its velocity limit (``abnormal_robot_state``)."""

    entity_name: str = "robot"
    velocity_limits: dict[str, float] = {}

    def build(self) -> None:
        entity = self.resolve_entity(self.entity_name)
        names, self._dofs_idx = entity.find_named_dofs_idx_local(list(self.velocity_limits), preserve_order=True)
        self._limit = 2.0 * torch.tensor([self.velocity_limits[name] for name in names], device=self.device)

    def compute(self, *args, **kwargs) -> torch.Tensor:
        dofs_vel = self.resolve_entity(self.entity_name).get_dofs_vel(dofs_idx_local=self._dofs_idx)
        return (dofs_vel.abs() > self._limit).any(dim=-1)


def assembly_errors(env: EnvBase, pair: Pair, insertive: str, receptive: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Position error (m) and tilt error (rad) of the insertive object from its assembled pose; yaw is free.

    UWLab's ``ProgressContext``: the insertive object's assembly frame expressed in the receptive object's,
    the receptive frame shifted by ``pair.assembled_pos``. Tilt is ``|roll| + |pitch|`` of the relative rotation.
    """
    receptive_entity, insertive_entity = env.get_entity(receptive), env.get_entity(insertive)
    receptive_pos, receptive_quat = receptive_entity.get_pos(relative=True), receptive_entity.get_quat(relative=True)
    offset = torch.tensor(pair.assembled_pos, device=receptive_pos.device, dtype=receptive_pos.dtype)
    target_pos = receptive_pos + transform_by_quat(offset.expand_as(receptive_pos), receptive_quat)
    pos, quat = subtract_frame_transforms(
        target_pos, receptive_quat, insertive_entity.get_pos(relative=True), insertive_entity.get_quat(relative=True)
    )
    roll, pitch, _ = euler_xyz_from_quat(quat)
    return pos.norm(dim=-1), wrap_to_pi(roll).abs() + wrap_to_pi(pitch).abs()
