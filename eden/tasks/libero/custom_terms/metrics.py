"""LIBERO metric terms and official-compatible predicate evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from genesis.utils.geom import quat_to_R, transform_by_quat

from eden.managers.metric_manager import METRIC_TERM_REGISTRY, MetricTerm
from eden.options.objects.libero import LiberoFlatStove
from eden.tasks.libero.catalog import LiberoPredicate, get_task_spec
from eden.tasks.libero.custom_terms.predicate_assets import _load_asset_predicate_metadata
from eden.tasks.libero.scenes import _get_libero_family_spec
from eden.utils.geom import quat_mul

_Comparison = Literal["lt", "gt", "ge"]
_Reduction = Literal["any", "all"]


@dataclass(frozen=True)
class _PreparedSite:
    """Build-time resolved site metadata stored on the evaluation device."""

    parent_entity_name: str
    parent_link_idx_local: torch.Tensor | None
    pos_local: torch.Tensor
    quat_local: torch.Tensor
    size: torch.Tensor
    is_workspace: bool


@dataclass(frozen=True)
class _PreparedObjectPredicate:
    source_entity_name: str
    target_entity_name: str


@dataclass(frozen=True)
class _PreparedSitePredicate:
    source_entity_name: str
    relation: Literal["In", "On"]
    site: _PreparedSite


@dataclass(frozen=True)
class _PreparedArticulationPredicate:
    """Build-time resolved joint indices and exact category threshold rule."""

    source_entity_name: str
    dofs_idx_local: torch.Tensor
    comparison: _Comparison
    threshold: float
    reduction: _Reduction


_PreparedPredicate = _PreparedObjectPredicate | _PreparedSitePredicate | _PreparedArticulationPredicate


def _site_contains_object(
    object_position: torch.Tensor,
    site_position: torch.Tensor,
    site_rotation_matrix: torch.Tensor,
    site_size: torch.Tensor,
) -> torch.Tensor:
    """Evaluate official ``SiteObject.in_box`` point containment."""
    lower_z_extension = 0.01
    extent = torch.bmm(site_rotation_matrix, site_size.unsqueeze(-1)).squeeze(-1).abs()
    lower = site_position - extent
    upper = site_position + extent
    is_inside_xy = ((object_position[..., :2] > lower[..., :2]) & (object_position[..., :2] < upper[..., :2])).all(
        dim=-1
    )
    is_inside_z = (object_position[..., 2] > lower[..., 2] - lower_z_extension) & (
        object_position[..., 2] < upper[..., 2]
    )
    return is_inside_xy & is_inside_z


def _site_under_object(
    object_position: torch.Tensor,
    site_position: torch.Tensor,
    site_rotation_matrix: torch.Tensor,
    site_size: torch.Tensor,
) -> torch.Tensor:
    """Evaluate official ``SiteObject.under`` support geometry."""
    lower_z_tolerance = 0.005
    object_height = 0.10
    delta = torch.bmm(site_rotation_matrix, (object_position - site_position).unsqueeze(-1)).squeeze(-1)
    inside_xy = (delta[..., :2].abs() < site_size[..., :2]).all(dim=-1)
    above_lower = delta[..., 2] > site_size[..., 2] - lower_z_tolerance
    below_upper = delta[..., 2] < site_size[..., 2] + object_height
    return inside_xy & above_lower & below_upper


def _object_on_object(
    object_position: torch.Tensor,
    target_position: torch.Tensor,
    contact: torch.Tensor,
) -> torch.Tensor:
    """Evaluate official object-on-object support semantics."""
    xy_distance_threshold = 0.03
    target_is_below = target_position[..., 2] <= object_position[..., 2]
    xy_distance = torch.linalg.vector_norm(target_position[..., :2] - object_position[..., :2], dim=-1)
    return target_is_below & contact.bool() & (xy_distance < xy_distance_threshold)


def _articulation_predicate(
    qpos: torch.Tensor,
    *,
    comparison: _Comparison,
    threshold: float,
    reduction: _Reduction,
) -> torch.Tensor:
    """Apply an official per-joint threshold and reduction."""
    if qpos.ndim != 2 or qpos.shape[1] == 0:
        raise ValueError(f"LIBERO articulation qpos must have shape [N, J] with J > 0, got {tuple(qpos.shape)}")
    if comparison == "lt":
        per_joint = qpos < threshold
    elif comparison == "gt":
        per_joint = qpos > threshold
    elif comparison == "ge":
        per_joint = qpos >= threshold
    else:
        raise ValueError(f"Unsupported articulation comparison {comparison!r}")

    if reduction == "any":
        return per_joint.any(dim=-1)
    if reduction == "all":
        return per_joint.all(dim=-1)
    raise ValueError(f"Unsupported articulation reduction {reduction!r}")


def _device_row(values: Sequence[float], device: torch.device) -> torch.Tensor:
    return torch.tensor(tuple(values), dtype=torch.float32, device=device).unsqueeze(0)


@METRIC_TERM_REGISTRY.register()
class LiberoGoalMetric(MetricTerm):
    """Compile and evaluate one task's complete BDDL goal."""

    task_name: str = ""

    def build(self) -> None:
        task_spec = get_task_spec(self.task_name)
        self._task_spec = task_spec
        self._family_spec = _get_libero_family_spec(task_spec.problem_name)
        self._prepared = tuple(self._prepare_predicate(predicate) for predicate in task_spec.goal_state)

    def _entity_category(self, name: str) -> str:
        try:
            return self._task_spec.entity_categories[name]
        except KeyError as exc:
            raise KeyError(f"Task {self._task_spec.name!r} has no BDDL entity {name!r}") from exc

    def _prepare_site(self, site_name: str) -> _PreparedSite:
        try:
            region = self._task_spec.regions[site_name]
        except KeyError as exc:
            raise KeyError(f"Task {self._task_spec.name!r} has no BDDL region {site_name!r}") from exc

        if region.target == self._family_spec.workspace_entity_name:
            if len(region.ranges) != 1:
                raise ValueError(
                    f"Workspace region {site_name!r} must contain exactly one rectangle; got {len(region.ranges)}"
                )
            x0, y0, x1, y1 = region.ranges[0]
            if x1 < x0 or y1 < y0:
                raise ValueError(f"Workspace region {site_name!r} has inverted ranges {region.ranges[0]!r}")
            self.resolve_entity(region.target)
            origin = self._family_spec.workspace_predicate_origin_w
            target_zone_height = 0.007
            return _PreparedSite(
                parent_entity_name=region.target,
                parent_link_idx_local=None,
                pos_local=_device_row(
                    (origin[0] + (x0 + x1) / 2.0, origin[1] + (y0 + y1) / 2.0, origin[2]),
                    self.device,
                ),
                quat_local=_device_row((1.0, 0.0, 0.0, 0.0), self.device),
                size=_device_row(((x1 - x0) / 2.0, (y1 - y0) / 2.0, target_zone_height), self.device),
                is_workspace=True,
            )

        try:
            parent_category = self._entity_category(region.target)
        except KeyError as exc:
            raise KeyError(f"Region {site_name!r} refers to unknown entity {region.target!r}") from exc
        parent = self.resolve_entity(region.target)
        metadata = _load_asset_predicate_metadata(parent_category)
        try:
            site = metadata.sites[region.local_name]
        except KeyError as exc:
            raise KeyError(
                f"LIBERO category {parent_category!r} has no site {region.local_name!r} "
                f"required by region {site_name!r}"
            ) from exc
        if site.site_type != "box" or len(site.size) != 3:
            raise ValueError(
                f"Region {site_name!r} requires a 3D box site; got type={site.site_type!r}, size={site.size}"
            )

        link_idx = None
        if site.anchor_body_name is not None:
            names, indices = parent.find_named_links_idx_local([site.anchor_body_name], preserve_order=True)
            if names != [site.anchor_body_name] or len(indices) != 1:
                raise ValueError(
                    f"Expected one dynamic link {site.anchor_body_name!r} on {region.target!r}; "
                    f"got names={names}, indices={indices}"
                )
            link_idx = torch.as_tensor(indices, dtype=torch.int32, device=self.device).contiguous()

        return _PreparedSite(
            parent_entity_name=region.target,
            parent_link_idx_local=link_idx,
            pos_local=_device_row(site.pos_local, self.device),
            quat_local=_device_row(site.quat_local, self.device),
            size=_device_row(site.size, self.device),
            is_workspace=False,
        )

    def _prepare_articulation(self, predicate: LiberoPredicate) -> _PreparedArticulationPredicate:
        region = self._task_spec.regions.get(predicate.entity_name)
        if region is not None:
            if region.target == self._family_spec.workspace_entity_name:
                raise ValueError(f"Workspace region {predicate.entity_name!r} cannot be an articulation operand")
            entity_name = region.target
        else:
            entity_name = predicate.entity_name

        try:
            category_name = self._entity_category(entity_name)
        except KeyError as exc:
            raise KeyError(
                f"Task {self._task_spec.name!r} has no articulation operand {predicate.entity_name!r}"
            ) from exc

        rule = {
            ("microwave", "Open"): ("lt", -1.3, "any"),
            ("microwave", "Close"): ("gt", -0.005, "all"),
            ("wooden_cabinet", "Open"): ("lt", -0.14, "any"),
            ("wooden_cabinet", "Close"): ("gt", 0.0, "all"),
            ("white_cabinet", "Open"): ("lt", -0.14, "any"),
            ("white_cabinet", "Close"): ("gt", 0.0, "all"),
            (LiberoFlatStove.category_name, "Turnon"): ("ge", LiberoFlatStove.on_threshold, "any"),
            (LiberoFlatStove.category_name, "Turnoff"): ("lt", 0.0, "all"),
        }.get((category_name, predicate.relation))
        if rule is None:
            raise ValueError(f"Official LIBERO has no {predicate.relation!r} semantics for category {category_name!r}")
        comparison, threshold, reduction = rule

        metadata = _load_asset_predicate_metadata(category_name)
        if region is not None:
            try:
                joint_names = metadata.sites[region.local_name].joint_names
            except KeyError as exc:
                raise KeyError(
                    f"LIBERO category {category_name!r} has no site {region.local_name!r} "
                    f"required by articulation operand {predicate.entity_name!r}"
                ) from exc
        else:
            joint_names = metadata.joint_names
        if not joint_names:
            raise ValueError(
                f"Articulation operand {predicate.entity_name!r} resolves to no source joints "
                f"for {predicate.relation!r}"
            )

        entity = self.resolve_entity(entity_name)
        try:
            names, indices = entity.find_named_dofs_idx_local(list(joint_names), preserve_order=True)
        except KeyError as exc:
            raise KeyError(
                f"Entity {entity_name!r} is missing articulation joints {joint_names!r} "
                f"for operand {predicate.entity_name!r}"
            ) from exc
        if names != list(joint_names) or len(indices) != len(joint_names):
            raise ValueError(
                f"Expected one DOF for each source joint {joint_names!r} on {entity_name!r}; "
                f"got names={names}, indices={indices}"
            )
        return _PreparedArticulationPredicate(
            source_entity_name=entity_name,
            dofs_idx_local=torch.as_tensor(indices, dtype=torch.int32, device=self.device).contiguous(),
            comparison=comparison,
            threshold=threshold,
            reduction=reduction,
        )

    def _prepare_predicate(self, predicate: LiberoPredicate) -> _PreparedPredicate:
        if predicate.relation in {"Open", "Close", "Turnon", "Turnoff"}:
            if predicate.target_name is not None:
                raise ValueError(f"Unary predicate must not have a target: {predicate.key}")
            return self._prepare_articulation(predicate)

        if predicate.relation not in {"On", "In"}:
            raise ValueError(f"Unsupported official LIBERO predicate relation {predicate.relation!r}")
        if predicate.target_name is None:
            raise ValueError(f"Binary predicate requires a target: {predicate.key}")

        self._entity_category(predicate.entity_name)
        self.resolve_entity(predicate.entity_name)
        if predicate.target_name in self._task_spec.regions:
            return _PreparedSitePredicate(
                source_entity_name=predicate.entity_name,
                relation=predicate.relation,
                site=self._prepare_site(predicate.target_name),
            )

        try:
            self._entity_category(predicate.target_name)
        except KeyError as exc:
            raise KeyError(f"Task {self._task_spec.name!r} has no predicate target {predicate.target_name!r}") from exc
        if predicate.relation == "In":
            raise ValueError(
                "Direct-object In semantics require the target category's source in_box implementation; "
                f"unsupported operand: {predicate.key}"
            )
        self.resolve_entity(predicate.target_name)
        return _PreparedObjectPredicate(
            source_entity_name=predicate.entity_name,
            target_entity_name=predicate.target_name,
        )

    def _site_world_state(self, site: _PreparedSite) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if site.is_workspace:
            return (
                site.pos_local.expand(self.num_envs, -1),
                quat_to_R(site.quat_local.expand(self.num_envs, -1)),
                site.size.expand(self.num_envs, -1),
            )

        parent = self.resolve_entity(site.parent_entity_name)
        if site.parent_link_idx_local is None:
            parent_pos = parent.get_pos(relative=True)
            parent_quat = parent.get_quat(relative=True)
        else:
            parent_pos = parent.get_links_pos(ls_idx_local=site.parent_link_idx_local, relative=True).squeeze(1)
            parent_quat = parent.get_links_quat(ls_idx_local=site.parent_link_idx_local, relative=True).squeeze(1)
        pos_local = site.pos_local.expand(self.num_envs, -1)
        quat_local = site.quat_local.expand(self.num_envs, -1)
        world_position = parent_pos + transform_by_quat(pos_local, parent_quat)
        world_quat = quat_mul(parent_quat, quat_local)
        return world_position, quat_to_R(world_quat), site.size.expand(self.num_envs, -1)

    def _entity_contact(self, entity_name: str, other_entity_name: str) -> torch.Tensor:
        contacts = self.resolve_entity(entity_name).get_contacts(with_entity=self.resolve_entity(other_entity_name))
        try:
            valid_mask = contacts["valid_mask"]
        except KeyError as exc:
            raise NotImplementedError(
                "LIBERO contact-dependent predicates require Genesis raw contacts with valid_mask"
            ) from exc
        if valid_mask.ndim != 2:
            raise ValueError(f"LIBERO contact valid_mask must have shape [N, C], got {tuple(valid_mask.shape)}")
        return valid_mask.bool().any(dim=-1)

    def _evaluate_predicate(self, prepared: _PreparedPredicate) -> torch.Tensor:
        if isinstance(prepared, _PreparedArticulationPredicate):
            qpos = self.resolve_entity(prepared.source_entity_name).get_dofs_pos(dofs_idx_local=prepared.dofs_idx_local)
            return _articulation_predicate(
                qpos,
                comparison=prepared.comparison,
                threshold=prepared.threshold,
                reduction=prepared.reduction,
            )

        object_position = self.resolve_entity(prepared.source_entity_name).get_pos(relative=True)
        if isinstance(prepared, _PreparedObjectPredicate):
            target_position = self.resolve_entity(prepared.target_entity_name).get_pos(relative=True)
            contact = self._entity_contact(prepared.source_entity_name, prepared.target_entity_name)
            return _object_on_object(object_position, target_position, contact)

        site_position, site_rotation_matrix, site_size = self._site_world_state(prepared.site)
        if prepared.relation == "In":
            return _site_contains_object(object_position, site_position, site_rotation_matrix, site_size)

        result = _site_under_object(object_position, site_position, site_rotation_matrix, site_size)
        if not prepared.site.is_workspace:
            result = result & self._entity_contact(
                prepared.site.parent_entity_name,
                prepared.source_entity_name,
            )
        return result

    def compute(self) -> torch.Tensor:
        return (
            torch.stack([self._evaluate_predicate(predicate) for predicate in self._prepared], dim=1).all(dim=1).float()
        )


__all__ = ["LiberoGoalMetric"]
