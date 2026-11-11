"""Reset event terms for base-state and DOF initialization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from genesis.typing import Vec2FType
from genesis.utils.geom import transform_quat_by_quat, xyz_to_quat

from eden.managers.event_manager import EVENT_TERM_REGISTRY
from eden.utils.misc import env_indices, is_env_mask, sanitize_envs_idx
from eden.utils.sample import sample_uniform, sample_uniform_ranges

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


def _rows_to_write(values: torch.Tensor, envs_idx: slice | torch.Tensor):
    """``values`` (one row per env) as the rows to hand a state setter with ``envs_idx``.

    Every row with the auto-reset's bool mask: :class:`~eden.entities.rigid.RigidEntity`'s
    ``set_qpos`` / ``set_dofs_vel`` / ``set_dofs_pos`` take them on the device, or gather the selected
    ones through the reset's shared ids where Genesis cannot. The selected rows otherwise.
    """
    return values if is_env_mask(envs_idx) else values[envs_idx]


#: A free base's root coordinates: position then wxyz quaternion.
_ROOT_QS = range(7)


@EVENT_TERM_REGISTRY.register()
def reset_base_state_uniform(
    env: EnvBase,
    envs_idx: slice | torch.Tensor | None,
    *,
    entity_name: str,
    pos_x_range: Vec2FType = (-0.5, 0.5),
    pos_y_range: Vec2FType = (-0.5, 0.5),
    pos_z_range: Vec2FType = (0.0, 0.0),
    roll_range: Vec2FType = (0.0, 0.0),
    pitch_range: Vec2FType = (0.0, 0.0),
    yaw_range: Vec2FType = (-3.14, 3.14),
):
    """Reset the base to its default pose, offset and rotated at random.

    The offsets are added to the entity's default root position, and the sampled roll / pitch / yaw (world
    x-y-z, ``R_rpy = Rz @ Ry @ Rx``) rotate the default orientation — ``R_rpy @ R_default`` — so an object
    authored lying on its side stays on its side. Both writes use the authored frame (``relative=True``),
    the frame ``default_root_pos`` is specified in: the asset's declared root origin lands on the target,
    not the centre of mass Genesis re-origins a free body onto. A heterogeneous entity's per-variant default
    poses are honored (``Entity.get_default_root_pose``). Similar to mjlab's ``reset_root_state_uniform``,
    which composes its rotation in the body frame instead (``R_default @ R_rpy``); the two agree for an
    identity default orientation or a pure yaw about an upright default.

    Rigid-only: relies on ``set_pos``/``set_quat``, which raise on
    :class:`ParticleEntity` (MPM/SPH). For deformable entities, bake pose into
    the morph at creation time.
    """
    from eden.entities.rigid import RigidEntity

    envs_idx = sanitize_envs_idx(envs_idx, env.num_envs)

    entity = env.get_entity(entity_name)
    if not isinstance(entity, RigidEntity):
        raise TypeError(
            f"reset_base_state_uniform requires a RigidEntity for '{entity_name}', got {type(entity).__name__}."
        )

    default_pos, default_quat = entity.get_default_root_pose()  # fresh (num_envs, 3) / (num_envs, 4)

    ranges = (pos_x_range, pos_y_range, pos_z_range, roll_range, pitch_range, yaw_range)
    sampled = sample_uniform_ranges(ranges, env.num_envs, env.device)
    new_pos = default_pos + sampled[:, :3]
    new_quat = transform_quat_by_quat(default_quat, xyz_to_quat(sampled[:, 3:], rpy=True))

    if entity.root_qpos_is_authored_pose:
        # One write of the 7 root coordinates, which are the authored pose here. Velocities are zeroed as
        # Genesis' `set_pos` / `set_quat` zero them: a simulated entity's, not a kinematic one's.
        root_qpos = _rows_to_write(torch.cat([new_pos, new_quat], dim=-1), envs_idx)
        entity.set_qpos(root_qpos, qs_idx_local=_ROOT_QS, envs_idx=envs_idx, zero_velocity=not entity.is_kinematic)
        return

    # ``relative=True`` takes Genesis' general path, which wants the selected rows under every
    # ``envs_idx`` form (slice, index tensor or bool mask). Orientation first, as the build write does:
    # the solver position of a re-origined body depends on its orientation. One `nonzero()` for the
    # auto-reset's mask covers both gathers and both setters (`env_indices`).
    envs_idx = env_indices(envs_idx)
    entity.set_quat(new_quat[envs_idx], envs_idx=envs_idx, relative=True)
    entity.set_pos(new_pos[envs_idx], envs_idx=envs_idx, relative=True)


@EVENT_TERM_REGISTRY.register()
def reset_dofs_by_offset(
    env: EnvBase,
    envs_idx: slice | torch.Tensor | None,
    *,
    entity_name: str,
    dofs_pos_range: Vec2FType = (-0.1, 0.1),
    dofs_vel_range: Vec2FType | None = None,
):
    """Reset DOFs to ``default + U(offset)``, clamped to soft limits.

    The offset is drawn per env and DOF. Velocities are zeroed (the base's too), then set to
    ``U(dofs_vel_range)`` when that is given.
    Rigid-only: requires a :class:`RigidEntity` (DOFs are not defined on
    :class:`ParticleEntity`).
    """
    from eden.entities.rigid import RigidEntity

    envs_idx = sanitize_envs_idx(envs_idx, env.num_envs)

    entity = env.get_entity(entity_name)
    if not isinstance(entity, RigidEntity):
        raise TypeError(
            f"reset_dofs_by_offset requires a RigidEntity for '{entity_name}', got {type(entity).__name__}."
        )
    dofs_pos = entity.default_dofs_pos.clone() + sample_uniform(
        dofs_pos_range[0],
        dofs_pos_range[1],
        (env.num_envs, entity.num_dofs),
        device=env.device,
    )
    dofs_pos = dofs_pos.clamp(entity.soft_dofs_pos_limits[:, :, 0], entity.soft_dofs_pos_limits[:, :, 1])
    dofs_pos = _rows_to_write(dofs_pos, envs_idx)
    if dofs_vel_range is None:
        # `set_dofs_pos` zeroes every dof's velocity (the base's too) and runs forward kinematics once.
        entity.set_dofs_pos(dofs_pos, envs_idx=envs_idx)
        return

    # Velocities first, without forward kinematics: the position write below runs the one pass over these envs,
    # and that pass refreshes link velocities from the dof velocities written here; writing velocities after
    # the positions would run a second pass. Final state: every dof's velocity zero (the base's too, as
    # `set_dofs_pos`' default zeroing would leave it), then the sampled joint velocities.
    dofs_vel = _rows_to_write(
        sample_uniform(dofs_vel_range[0], dofs_vel_range[1], (env.num_envs, entity.num_dofs), device=env.device),
        envs_idx,
    )
    entity.zero_all_dofs_velocity(envs_idx=envs_idx, skip_forward=True)
    entity.set_dofs_vel(dofs_vel, envs_idx=envs_idx, skip_forward=True)
    entity.set_dofs_pos(dofs_pos, envs_idx=envs_idx, zero_velocity=False)


@EVENT_TERM_REGISTRY.register()
def reset_dofs_by_scale(
    env: EnvBase,
    envs_idx: slice | torch.Tensor | None,
    *,
    entity_name: str,
    dofs_pos_scale_range: Vec2FType = (0.95, 1.05),
):
    """Reset DOFs to ``default * U(scale)`` per env/DOF; velocities to 0.

    Multiplicative variant of :func:`reset_dofs_by_offset` (legged_gym-style init used by
    e.g. extreme-parkour). Rigid-only: requires a :class:`RigidEntity`.
    """
    from eden.entities.rigid import RigidEntity

    envs_idx = sanitize_envs_idx(envs_idx, env.num_envs)

    entity = env.get_entity(entity_name)
    if not isinstance(entity, RigidEntity):
        raise TypeError(f"reset_dofs_by_scale requires a RigidEntity for '{entity_name}', got {type(entity).__name__}.")
    dofs_pos = entity.default_dofs_pos * sample_uniform(
        dofs_pos_scale_range[0],
        dofs_pos_scale_range[1],
        (env.num_envs, entity.num_dofs),
        device=env.device,
    )
    dofs_pos = dofs_pos.clamp(entity.soft_dofs_pos_limits[:, :, 0], entity.soft_dofs_pos_limits[:, :, 1])
    # `set_dofs_pos` zeroes every dof's velocity, the base's too.
    entity.set_dofs_pos(_rows_to_write(dofs_pos, envs_idx), envs_idx=envs_idx)


@EVENT_TERM_REGISTRY.register()
def reset_dofs_by_scale_and_offset(
    env: EnvBase,
    envs_idx: slice | torch.Tensor | None,
    *,
    entity_name: str,
    dofs_scale_range: Vec2FType = (0.8, 1.2),
    dofs_offset_range: Vec2FType = (-0.1, 0.1),
):
    """Reset DOFs to ``default * U(scale) + U(offset)``; velocities to 0.

    Sampled per env and DOF, then clamped to the soft limits. OpenHomie's ``_reset_dofs``:
    :func:`reset_dofs_by_scale` and :func:`reset_dofs_by_offset` composed.
    Rigid-only: requires a :class:`RigidEntity`.
    """
    from eden.entities.rigid import RigidEntity

    envs_idx = sanitize_envs_idx(envs_idx, env.num_envs)

    entity = env.get_entity(entity_name)
    if not isinstance(entity, RigidEntity):
        raise TypeError(
            f"reset_dofs_by_scale_and_offset requires a RigidEntity for '{entity_name}', got {type(entity).__name__}."
        )
    shape = (env.num_envs, entity.num_dofs)
    scale = sample_uniform(dofs_scale_range[0], dofs_scale_range[1], shape, device=env.device)
    offset = sample_uniform(dofs_offset_range[0], dofs_offset_range[1], shape, device=env.device)
    limits = entity.soft_dofs_pos_limits
    dofs_pos = (entity.default_dofs_pos * scale + offset).clamp(limits[..., 0], limits[..., 1])
    # `set_dofs_pos` zeroes every dof's velocity, the base's too.
    entity.set_dofs_pos(_rows_to_write(dofs_pos, envs_idx), envs_idx=envs_idx)
