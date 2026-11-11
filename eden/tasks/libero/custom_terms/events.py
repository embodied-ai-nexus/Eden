"""LIBERO event terms for fixed-state reset and state-dependent visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch
from genesis.utils.geom import inv_quat, inv_transform_by_quat, transform_by_quat, transform_quat_by_quat

from eden.managers.event_manager import EVENT_TERM_REGISTRY, EventTerm
from eden.options.objects.libero import LiberoFlatStove
from eden.options.robots.libero import LIBERO_ROBOT_ENTITY_NAME, LiberoPanda
from eden.tasks.libero.catalog import get_task_spec
from eden.tasks.libero.scenes import _get_libero_family_spec
from eden.tasks.libero.state_bundle import _EntityDescriptor, load_state_bundle
from eden.utils.misc import sanitize_envs_idx

if TYPE_CHECKING:
    from eden.entities.base import Entity


@dataclass(frozen=True)
class _FixedEntityRuntime:
    descriptor: _EntityDescriptor
    is_root_pose_settable: bool
    validation_root_link_idx_local: torch.Tensor | None
    root_dofs_idx_local: torch.Tensor | None
    solver_root_pos_in_source_frame: torch.Tensor | None
    solver_root_quat_in_source_frame: torch.Tensor | None
    joint_dofs_idx_local: torch.Tensor | None


@EVENT_TERM_REGISTRY.register()
class LiberoFixedStateReset(EventTerm):
    """Restore a validated official-state bundle through Eden's public entity API.

    Position state is restored before velocity state so root pose setters cannot erase nonzero source velocities.
    Callers must take ``LIBERO_RESET_WARMUP_STEPS`` (5) zero-action env steps before the first policy observation.
    """

    task_name: str = ""
    bundle_path: str = ""
    init_ids: list[int] = []  # noqa: RUF012 - empty selects the first num_envs bundle states
    priority: int = 100

    def build(self) -> None:
        if not self.task_name:
            raise ValueError("official fixed-state reset requires task_name")
        if not self.bundle_path:
            raise ValueError("official fixed-state reset requires bundle_path")
        self._bundle = load_state_bundle(
            self.bundle_path,
            expected_task_name=self.task_name,
        )
        task_spec = get_task_spec(self.task_name)
        self._workspace_name = _get_libero_family_spec(task_spec.problem_name).workspace_entity_name
        self._fixture_names = task_spec.fixture_names
        self._validate_descriptor_coverage(task_spec)
        self._entity_runtimes = self._prepare_entity_runtimes()
        self._prepare_state_rows()

    def _validate_descriptor_coverage(self, task_spec) -> None:
        descriptors = {entity.semantic_name: entity for entity in self._bundle.manifest.entities}
        expected_names = set(task_spec.entity_categories) | {LIBERO_ROBOT_ENTITY_NAME}
        if set(descriptors) != expected_names:
            raise ValueError(
                "Fixed-state semantic entity coverage mismatch: "
                f"missing={sorted(expected_names - set(descriptors))}, "
                f"unexpected={sorted(set(descriptors) - expected_names)}"
            )

        robot_descriptor = descriptors[LIBERO_ROBOT_ENTITY_NAME]
        robot_joint_names = tuple(robot_descriptor.joint_names)
        expected_robot_joint_names = (*LiberoPanda.arm_dofs_name, *LiberoPanda.gripper_dofs_name)
        if robot_joint_names != expected_robot_joint_names:
            raise ValueError(
                f"LIBERO Panda semantic joints must be {expected_robot_joint_names}, got {robot_joint_names}"
            )

    def _prepare_entity_runtimes(self) -> tuple[_FixedEntityRuntime, ...]:
        runtimes: list[_FixedEntityRuntime] = []
        for descriptor in self._bundle.manifest.entities:
            entity = self.resolve_entity(descriptor.semantic_name)
            is_robot = descriptor.semantic_name == LIBERO_ROBOT_ENTITY_NAME
            is_fixture = descriptor.semantic_name in self._fixture_names
            is_root_pose_settable = not is_robot and descriptor.semantic_name != self._workspace_name
            validation_root_link_indices = None
            if not is_robot:
                if descriptor.root_link_name is None:
                    raise ValueError(f"Entity {descriptor.semantic_name!r} must declare its source root link")
                names, indices = entity.find_named_links_idx_local(
                    [descriptor.root_link_name],
                    preserve_order=True,
                )
                if names != [descriptor.root_link_name] or len(indices) != 1:
                    raise ValueError(
                        f"Entity {descriptor.semantic_name!r} must resolve source root link "
                        f"{descriptor.root_link_name!r}; got names={names}, indices={indices}"
                    )
                if is_root_pose_settable and indices[0] != 0:
                    raise ValueError(
                        f"Settable entity {descriptor.semantic_name!r} source root link "
                        f"{descriptor.root_link_name!r} must be Eden's base link"
                    )
                validation_root_link_indices = torch.tensor(
                    indices,
                    dtype=torch.int32,
                    device=self.device,
                ).contiguous()

            if is_fixture:
                self._validate_fixture_joint_coverage(entity, descriptor)

            root_indices = None
            solver_root_pos_in_source_frame = None
            solver_root_quat_in_source_frame = None
            if is_root_pose_settable and is_fixture:
                if not entity.fixed:
                    raise ValueError(f"Settable fixture {descriptor.semantic_name!r} must be fixed-base in Eden")
                if not entity.morph.batch_fixed_verts:
                    raise ValueError(f"Settable fixture {descriptor.semantic_name!r} requires batch_fixed_verts=True")
            elif is_root_pose_settable:
                _, indices = entity.find_named_dofs_idx_local("root_joint", preserve_order=True)
                if len(indices) != 6:
                    raise ValueError(
                        f"Settable entity {descriptor.semantic_name!r} must resolve root_joint to 6 DOFs; "
                        f"got {len(indices)}"
                    )
                root_indices = torch.as_tensor(indices, dtype=torch.int32, device=self.device).contiguous()
                assert validation_root_link_indices is not None
                solver_root_pos_in_source_frame, solver_root_quat_in_source_frame = self._prepare_root_frame_offsets(
                    entity,
                    validation_root_link_indices,
                )

            joint_indices: list[int] = []
            for joint_name in descriptor.joint_names:
                names, indices = entity.find_named_dofs_idx_local([joint_name], preserve_order=True)
                if names != [joint_name] or len(indices) != 1:
                    raise ValueError(
                        f"Joint {descriptor.semantic_name}.{joint_name} must resolve to one DOF; "
                        f"got names={names}, indices={indices}"
                    )
                joint_indices.append(int(indices[0]))
            joint_indices_tensor = (
                torch.tensor(joint_indices, dtype=torch.int32, device=self.device).contiguous()
                if joint_indices
                else None
            )
            runtimes.append(
                _FixedEntityRuntime(
                    descriptor=descriptor,
                    is_root_pose_settable=is_root_pose_settable,
                    validation_root_link_idx_local=validation_root_link_indices,
                    root_dofs_idx_local=root_indices,
                    solver_root_pos_in_source_frame=solver_root_pos_in_source_frame,
                    solver_root_quat_in_source_frame=solver_root_quat_in_source_frame,
                    joint_dofs_idx_local=joint_indices_tensor,
                )
            )
        return tuple(runtimes)

    @staticmethod
    def _validate_fixture_joint_coverage(entity: Entity, descriptor: _EntityDescriptor) -> None:
        actual_joint_names: list[str] = []
        for joint_name in entity.joints_name:
            names, indices = entity.find_named_dofs_idx_local([joint_name], preserve_order=True)
            if not indices:
                continue
            if names != [joint_name] * len(indices) or len(indices) != 1:
                raise ValueError(
                    f"Fixture joint {descriptor.semantic_name}.{joint_name} must resolve to one DOF; "
                    f"got names={names}, indices={indices}"
                )
            actual_joint_names.append(joint_name)

        descriptor_joint_names = set(descriptor.joint_names)
        actual_joint_name_set = set(actual_joint_names)
        if descriptor_joint_names != actual_joint_name_set:
            raise ValueError(
                f"Fixed-state fixture joint coverage mismatch for {descriptor.semantic_name!r}: "
                f"missing={sorted(actual_joint_name_set - descriptor_joint_names)}, "
                f"unexpected={sorted(descriptor_joint_names - actual_joint_name_set)}"
            )

    @staticmethod
    def _prepare_root_frame_offsets(
        entity: Entity,
        source_root_link_idx_local: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_pos = entity.get_links_pos(
            ls_idx_local=source_root_link_idx_local,
            relative=True,
        )[:, 0]
        source_quat = entity.get_links_quat(
            ls_idx_local=source_root_link_idx_local,
            relative=True,
        )[:, 0]
        solver_pos = entity.get_links_pos(
            ls_idx_local=source_root_link_idx_local,
            relative=False,
        )[:, 0]
        solver_quat = entity.get_links_quat(
            ls_idx_local=source_root_link_idx_local,
            relative=False,
        )[:, 0]

        solver_pos_in_source_frame = inv_transform_by_quat(solver_pos - source_pos, source_quat)
        solver_quat_in_source_frame = transform_quat_by_quat(solver_quat, inv_quat(source_quat))
        same_quat_sign = (solver_quat_in_source_frame * solver_quat_in_source_frame[:1]).sum(dim=1, keepdim=True) >= 0
        solver_quat_in_source_frame = torch.where(
            same_quat_sign,
            solver_quat_in_source_frame,
            -solver_quat_in_source_frame,
        )
        if not torch.allclose(
            solver_pos_in_source_frame,
            solver_pos_in_source_frame[:1].expand_as(solver_pos_in_source_frame),
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError(f"Entity {entity.name!r} has environment-dependent source-to-solver root position offsets")
        if not torch.allclose(
            solver_quat_in_source_frame,
            solver_quat_in_source_frame[:1].expand_as(solver_quat_in_source_frame),
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError(
                f"Entity {entity.name!r} has environment-dependent source-to-solver root orientation offsets"
            )
        return solver_pos_in_source_frame[:1].clone(), solver_quat_in_source_frame[:1].clone()

    def _prepare_state_rows(self) -> None:
        if self.init_ids:
            requested = list(self.init_ids)
        else:
            available = self._bundle.manifest.init_ids
            if self.num_envs > len(available):
                raise ValueError(f"LIBERO bundle has {len(available)} init states, fewer than num_envs={self.num_envs}")
            requested = available[: self.num_envs]
        if self.init_ids and len(requested) != self.num_envs:
            # `init_ids` and `num_envs` are one selection: `LiberoTaskMod` derives `num_envs` from
            # the requested IDs while the task is built, so reaching here means a `num_envs` override
            # was applied *after* the mod ran. The message names that, not just the width mismatch,
            # which would point the reader at the one value they set correctly.
            raise ValueError(
                f"LIBERO init_ids selects {len(requested)} fixed state(s) but the environment has "
                f"num_envs={self.num_envs}. Selecting init_ids sets num_envs when the task is built "
                f"(init_ids={list(self.init_ids)} -> num_envs={len(requested)}); a num_envs override "
                f"applied after that (eden smoke's --num_envs, which defaults to 2 and is always "
                f"applied, eden train's --num_envs, or build_and_step(num_envs=...)) must match the "
                f"selection. Either pass num_envs={len(requested)} alongside it, or drop init_ids to "
                f"take the first num_envs bundle states."
            )
        rows = self._bundle.rows_for_init_ids(requested)
        self._state_row_by_env = torch.as_tensor(rows, dtype=torch.long, device=self.device)
        self._arrays = {
            key: torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for key, value in self._bundle.arrays.items()
        }

    def _joint_values(self, runtime: _FixedEntityRuntime, suffix: str, rows: torch.Tensor) -> torch.Tensor | None:
        values = [
            self._arrays[f"entity/{runtime.descriptor.semantic_name}/joint/{joint_name}/{suffix}"][rows]
            for joint_name in runtime.descriptor.joint_names
        ]
        return torch.cat(values, dim=1) if values else None

    def _restore_joint_positions(self, rows: torch.Tensor, envs_idx: torch.Tensor) -> None:
        for runtime in self._entity_runtimes:
            joint_qpos = self._joint_values(runtime, "qpos", rows)
            if joint_qpos is None:
                continue
            assert runtime.joint_dofs_idx_local is not None
            self.resolve_entity(runtime.descriptor.semantic_name).set_dofs_pos(
                joint_qpos,
                dofs_idx_local=runtime.joint_dofs_idx_local,
                envs_idx=envs_idx,
                zero_velocity=False,
            )

    def _restore_root_poses(self, rows: torch.Tensor, envs_idx: torch.Tensor) -> None:
        for runtime in self._entity_runtimes:
            if not runtime.is_root_pose_settable:
                continue
            prefix = f"entity/{runtime.descriptor.semantic_name}"
            entity = self.resolve_entity(runtime.descriptor.semantic_name)
            entity.set_quat(
                self._arrays[f"{prefix}/root_quat_wxyz_world"][rows],
                envs_idx=envs_idx,
                relative=True,
            )
            entity.set_pos(
                self._arrays[f"{prefix}/root_pos_world"][rows],
                envs_idx=envs_idx,
                relative=True,
            )

    def _restore_velocities(self, rows: torch.Tensor, envs_idx: torch.Tensor) -> None:
        for runtime in self._entity_runtimes:
            prefix = f"entity/{runtime.descriptor.semantic_name}"
            values: list[torch.Tensor] = []
            indices: list[torch.Tensor] = []
            if runtime.root_dofs_idx_local is not None:
                source_quat = self._arrays[f"{prefix}/root_quat_wxyz_world"][rows]
                source_lin_vel = self._arrays[f"{prefix}/root_lin_vel_world"][rows]
                source_ang_vel = self._arrays[f"{prefix}/root_ang_vel_body"][rows]
                assert runtime.solver_root_pos_in_source_frame is not None
                assert runtime.solver_root_quat_in_source_frame is not None
                solver_root_offset_world = transform_by_quat(
                    runtime.solver_root_pos_in_source_frame.expand_as(source_lin_vel),
                    source_quat,
                )
                source_ang_vel_world = transform_by_quat(source_ang_vel, source_quat)
                solver_lin_vel = source_lin_vel + torch.cross(
                    source_ang_vel_world,
                    solver_root_offset_world,
                    dim=-1,
                )
                solver_ang_vel = inv_transform_by_quat(
                    source_ang_vel,
                    runtime.solver_root_quat_in_source_frame.expand(source_ang_vel.shape[0], -1),
                )
                values.append(torch.cat((solver_lin_vel, solver_ang_vel), dim=1))
                indices.append(runtime.root_dofs_idx_local)
            joint_qvel = self._joint_values(runtime, "qvel", rows)
            if joint_qvel is not None:
                values.append(joint_qvel)
                assert runtime.joint_dofs_idx_local is not None
                indices.append(runtime.joint_dofs_idx_local)
            if values:
                self.resolve_entity(runtime.descriptor.semantic_name).set_dofs_vel(
                    torch.cat(values, dim=1),
                    dofs_idx_local=torch.cat(indices),
                    envs_idx=envs_idx,
                )

    def _validate_root_state(
        self,
        entity: Entity,
        prefix: str,
        rows: torch.Tensor,
        envs_idx: torch.Tensor,
        *,
        validation_root_link_idx_local: torch.Tensor | None,
    ) -> None:
        expected_pos = self._arrays[f"{prefix}/root_pos_world"][rows]
        expected_quat = self._arrays[f"{prefix}/root_quat_wxyz_world"][rows]
        expected_lin_vel = self._arrays[f"{prefix}/root_lin_vel_world"][rows]
        expected_ang_vel = self._arrays[f"{prefix}/root_ang_vel_body"][rows]
        if validation_root_link_idx_local is None:
            actual_pos = entity.get_pos(envs_idx=envs_idx, relative=True)
            actual_quat = entity.get_quat(envs_idx=envs_idx, relative=True)
            actual_lin_vel = entity.get_vel(envs_idx=envs_idx, frame="world")
            actual_ang_vel = entity.get_ang(envs_idx=envs_idx, frame="body")
        else:
            actual_pos = entity.get_links_pos(
                ls_idx_local=validation_root_link_idx_local,
                envs_idx=envs_idx,
                relative=True,
            )[:, 0]
            actual_quat = entity.get_links_quat(
                ls_idx_local=validation_root_link_idx_local,
                envs_idx=envs_idx,
                relative=True,
            )[:, 0]
            solver_pos = entity.get_links_pos(
                ls_idx_local=validation_root_link_idx_local,
                envs_idx=envs_idx,
                relative=False,
            )[:, 0]
            solver_lin_vel = entity.get_links_vel(
                ls_idx_local=validation_root_link_idx_local,
                envs_idx=envs_idx,
            )[:, 0]
            actual_ang_vel_world = entity.get_links_ang(
                ls_idx_local=validation_root_link_idx_local,
                envs_idx=envs_idx,
            )[:, 0]
            actual_lin_vel = solver_lin_vel - torch.cross(
                actual_ang_vel_world,
                solver_pos - actual_pos,
                dim=-1,
            )
            actual_ang_vel = inv_transform_by_quat(actual_ang_vel_world, actual_quat)

        tolerance = 1e-5
        if not torch.allclose(actual_pos, expected_pos, atol=tolerance, rtol=0.0):
            raise ValueError(f"fixed-state root position mismatch for {entity.name!r}")
        quat_error = torch.minimum(
            torch.linalg.vector_norm(actual_quat - expected_quat, dim=1),
            torch.linalg.vector_norm(actual_quat + expected_quat, dim=1),
        )
        if bool((quat_error > tolerance).any().item()):
            raise ValueError(f"fixed-state root quaternion mismatch for {entity.name!r}")
        if not torch.allclose(actual_lin_vel, expected_lin_vel, atol=tolerance, rtol=0.0):
            raise ValueError(f"fixed-state root linear velocity mismatch for {entity.name!r}")
        if not torch.allclose(actual_ang_vel, expected_ang_vel, atol=tolerance, rtol=0.0):
            raise ValueError(f"fixed-state root angular velocity mismatch for {entity.name!r}")

    def _validate_joint_state(
        self,
        entity: Entity,
        runtime: _FixedEntityRuntime,
        rows: torch.Tensor,
        envs_idx: torch.Tensor,
    ) -> None:
        if runtime.joint_dofs_idx_local is None:
            return
        expected_qpos = self._joint_values(runtime, "qpos", rows)
        expected_qvel = self._joint_values(runtime, "qvel", rows)
        assert expected_qpos is not None and expected_qvel is not None
        actual_qpos = entity.get_dofs_pos(
            dofs_idx_local=runtime.joint_dofs_idx_local,
            envs_idx=envs_idx,
        )
        actual_qvel = entity.get_dofs_vel(
            dofs_idx_local=runtime.joint_dofs_idx_local,
            envs_idx=envs_idx,
        )
        tolerance = 1e-5
        if not torch.allclose(actual_qpos, expected_qpos, atol=tolerance, rtol=0.0):
            raise ValueError(f"restored joint position mismatch for {entity.name!r}")
        if not torch.allclose(actual_qvel, expected_qvel, atol=tolerance, rtol=0.0):
            raise ValueError(f"restored joint velocity mismatch for {entity.name!r}")

    def _validate_restored_state(self, rows: torch.Tensor, envs_idx: torch.Tensor) -> None:
        for runtime in self._entity_runtimes:
            entity = self.resolve_entity(runtime.descriptor.semantic_name)
            self._validate_root_state(
                entity,
                f"entity/{runtime.descriptor.semantic_name}",
                rows,
                envs_idx,
                validation_root_link_idx_local=runtime.validation_root_link_idx_local,
            )
            self._validate_joint_state(entity, runtime, rows, envs_idx)

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        selected = sanitize_envs_idx(envs_idx, self.num_envs, prefer_slice=False)
        assert isinstance(selected, torch.Tensor)
        rows = self._state_row_by_env[selected]
        self._restore_joint_positions(rows, selected)
        self._restore_root_poses(rows, selected)
        self._restore_velocities(rows, selected)
        self._validate_restored_state(rows, selected)


@EVENT_TERM_REGISTRY.register()
class LiberoStoveBurnerVisualization(EventTerm):
    """Mirror the flat-stove button state into its red burner indicator.

    Upstream LIBERO represents the indicator as a MuJoCo site whose alpha is
    toggled from the ``button`` joint position. Genesis does not import that
    site as visual geometry, so Eden composes an equivalent collision-free
    scene entity and moves it into or out of view through public entity APIs.

    A pose setter drops every contact of the environments it writes (Genesis clears the contacts found at the old
    pose), so the indicator is written only where it has to change: where the button crossed the threshold, and
    after a reset, which may have moved the fixed-base stove.
    """

    stove_entity_name: str = ""
    indicator_entity_name: str = ""
    hidden_z_offset: ClassVar[float] = -10.0
    priority: int = 300

    def build(self) -> None:
        if not self.stove_entity_name or not self.indicator_entity_name:
            raise ValueError("LIBERO stove visualization requires stove_entity_name and indicator_entity_name")

        stove = self.resolve_entity(self.stove_entity_name)
        self.resolve_entity(self.indicator_entity_name)
        dof_names, button_dofs_idx_local = stove.find_named_dofs_idx_local(
            LiberoFlatStove.button_joint_name,
            preserve_order=True,
        )
        if dof_names != [LiberoFlatStove.button_joint_name] or len(button_dofs_idx_local) != 1:
            raise ValueError(
                f"LIBERO stove {self.stove_entity_name!r} must have exactly one "
                f"{LiberoFlatStove.button_joint_name!r} DOF"
            )
        link_names, burner_links_idx_local = stove.find_named_links_idx_local(
            LiberoFlatStove.burner_link_name,
            preserve_order=True,
        )
        if link_names != [LiberoFlatStove.burner_link_name] or len(burner_links_idx_local) != 1:
            raise ValueError(
                f"LIBERO stove {self.stove_entity_name!r} must have exactly one "
                f"{LiberoFlatStove.burner_link_name!r} link"
            )

        self._button_dofs_idx_local = torch.tensor(button_dofs_idx_local, dtype=torch.int32, device=self.device)
        self._burner_links_idx_local = torch.tensor(burner_links_idx_local, dtype=torch.int32, device=self.device)
        self._burner_local_pos = torch.tensor(
            LiberoFlatStove.burner_local_pos,
            dtype=torch.float32,
            device=self.device,
        )
        self._hidden_offset = torch.tensor(
            (0.0, 0.0, self.hidden_z_offset),
            dtype=torch.float32,
            device=self.device,
        )
        # What the indicator of each environment shows: 1 lit, 0 hidden, -1 not written since the last reset
        self._shown = torch.full((self.num_envs,), -1, dtype=torch.int8, device=self.device)

    def post_reset(self, envs_idx: slice | torch.Tensor) -> None:
        self._shown[envs_idx] = -1

    def compute(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        stove = self.resolve_entity(self.stove_entity_name)
        button_pos = stove.get_dofs_pos(dofs_idx_local=self._button_dofs_idx_local).squeeze(1)
        burner_pos = stove.get_links_pos(ls_idx_local=self._burner_links_idx_local, relative=False).squeeze(1)
        burner_quat = stove.get_links_quat(ls_idx_local=self._burner_links_idx_local, relative=False).squeeze(1)
        local_pos = self._burner_local_pos.unsqueeze(0).expand_as(burner_pos)
        visible_pos = burner_pos + transform_by_quat(local_pos, burner_quat)
        is_lit = (button_pos >= LiberoFlatStove.on_threshold).to(torch.int8)
        target_pos = torch.where(is_lit.bool().unsqueeze(1), visible_pos, visible_pos + self._hidden_offset)

        # A mask over every environment rather than the indices of those to write, which would need a device sync
        written = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        written[sanitize_envs_idx(envs_idx, self.num_envs)] = True
        written &= self._shown != is_lit
        self._shown.copy_(torch.where(written, is_lit, self._shown))

        indicator = self.resolve_entity(self.indicator_entity_name)
        indicator.set_pos(target_pos, envs_idx=written, relative=False)
        indicator.set_quat(burner_quat, envs_idx=written, relative=False)


__all__ = [
    "LiberoFixedStateReset",
    "LiberoStoveBurnerVisualization",
]
