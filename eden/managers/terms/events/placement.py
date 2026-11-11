"""Object-placement event terms and support-surface samplers."""

from __future__ import annotations

import math
from functools import partial
from typing import TYPE_CHECKING, Callable, Literal

import genesis as gs
import torch
from genesis.utils.geom import transform_quat_by_quat, xyz_to_quat

from eden.constants import EventMode
from eden.envs.base import EnvBase
from eden.managers.event_manager import EVENT_TERM_REGISTRY, EventTerm
from eden.utils.geom import transform_by_T
from eden.utils.mesh import ExtendedSupportData
from eden.utils.misc import sanitize_envs_idx
from eden.utils.sample import sample_uniform

if TYPE_CHECKING:
    from eden.entities.base import Entity
    from eden.options.managers.events import EventTermOptions


@torch.jit.script
def _fused_gaussian_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    mu_x: float,
    mu_y: float,
    inv_var_x_2: float,
    inv_var_y_2: float,
    norm_const: float,
) -> torch.Tensor:
    # Math: exp( - (x-mu)^2 / 2var )
    # Optimized: exp( - (x-mu)^2 * (0.5 * inv_var) )
    term_x = (x - mu_x).square() * inv_var_x_2
    term_y = (y - mu_y).square() * inv_var_y_2
    return torch.exp(-(term_x + term_y)) * norm_const


class Gaussian2D:
    def __init__(self, mu_x: float, mu_y: float, var_x: float, var_y: float):
        self.mu_x = float(mu_x)
        self.mu_y = float(mu_y)
        # Pre-calculate the normalization constant: 1 / (2 * pi * sigma_x * sigma_y)
        sigma_x = math.sqrt(var_x)
        sigma_y = math.sqrt(var_y)
        self.norm_const = 1.0 / (2 * math.pi * sigma_x * sigma_y)

        # Pre-calculate the inverse denominator for the exponent: 1 / (2 * var)
        # We multiply by this later instead of dividing for speed
        self.inv_var_x_2 = 0.5 / var_x
        self.inv_var_y_2 = 0.5 / var_y

    @torch.inference_mode()
    def get_probs(self, points):
        """Compute the Gaussian weights for the given points.

        Parameters
        ----------
        points: torch.Tensor
            The points to compute the weights for, of shape (N, 2)

        Returns
        -------
        torch.Tensor
            The weights for the points, of shape (N,)
        """
        x, y = points.unbind(dim=-1)

        return _fused_gaussian_kernel(
            x,
            y,
            self.mu_x,
            self.mu_y,
            self.inv_var_x_2,
            self.inv_var_y_2,
            self.norm_const,
        )


# ------------------------------------------------------------------------------------
# ---------------------------- Placement Samplers ------------------------------------
# ------------------------------------------------------------------------------------


def _z_on_support(pos_xy: torch.Tensor, d: ExtendedSupportData) -> torch.Tensor:
    """World z of the support plane directly above/below the given *world* xy.

    The support plane passes through ``d.transform``'s translation ``p0`` with
    normal ``d.normal``; solve ``(P - p0) . n = 0`` for ``P_z``. For a level
    surface (``n = +z``) this is simply the plane height ``p0_z``.
    """
    transform = d.transform
    p0 = transform[:, :3, 3] if transform.ndim == 3 else transform[:3, 3]  # (B, 3) / (3,)
    n = d.normal  # (B, 3)
    dx = pos_xy[..., 0] - p0[..., 0]
    dy = pos_xy[..., 1] - p0[..., 1]
    return p0[..., 2] - (n[..., 0] * dx + n[..., 1] * dy) / n[..., 2]


def _describe_surfaces(support_data: list[ExtendedSupportData]) -> str:
    """One line per support surface, for an error message: index, height, centre, area, whether it holds the object.

    Read off env 0. Never raises: a surface it cannot describe is listed as such.
    """
    lines = []
    for i, d in enumerate(support_data):
        try:
            transform = d.transform[0] if d.transform.ndim == 3 else d.transform
            centroid = d.polygon.centroid
            local = torch.tensor([centroid.x, centroid.y, 0.0, 1.0], dtype=transform.dtype, device=transform.device)
            x, y, z = (transform @ local)[:3].tolist()
            holds = "can hold the object" if bool(d.valid_mask.any()) else "cannot hold the object"
            lines.append(f"{i}: z={z:.3f} m, centre ({x:.3f}, {y:.3f}), area {d.polygon.area * 1e4:.0f} cm2, {holds}")
        except Exception:  # noqa: BLE001 - message building must not raise over the error it describes
            lines.append(f"{i}: (could not be described)")
    return "; ".join(lines) if lines else "none"


def _support_surface(support_data: list[ExtendedSupportData], support_idx: int | None) -> ExtendedSupportData:
    """The support surface to place onto.

    The support's surfaces come link by link, each link's largest first. ``support_idx=None`` takes the first
    one that can hold the object (a valid placement point in some env). An integer picks from *all* of them,
    like a Python list index, so its meaning does not depend on the object being placed; it raises when out
    of range or when that surface cannot hold the object, rather than placing it somewhere else.
    """
    if support_idx is None:
        for d in support_data:
            if d.valid_mask.any():
                return d
        raise IndexError(
            "The support entity has no surface that can hold this object in any env, so there is nothing to "
            f"place onto. Its surfaces: {_describe_surfaces(support_data)}. Check support_entity_name names the "
            "surface the object should stand on, and that the object fits on it."
        )
    if not -len(support_data) <= support_idx < len(support_data):
        raise IndexError(
            f"support_idx={support_idx} is out of range: the support entity has {len(support_data)} surface(s): "
            f"{_describe_surfaces(support_data)}. Pass one of those indices (negative ones count from the end), "
            "or leave support_idx=None for the first surface that can hold the object."
        )
    d = support_data[support_idx]
    if not d.valid_mask.any():
        raise ValueError(
            f"support_idx={support_idx} names a surface that cannot hold this object in any env (too small for "
            f"it, or not facing up). The support's surfaces: {_describe_surfaces(support_data)}. Pick one that can, "
            "or leave support_idx=None for the first that can."
        )
    return d


def constant_xy_sampler(
    support_data: list[ExtendedSupportData],
    asset_AABB: torch.Tensor,
    batch_size: int = 1,
    *,
    pos_xy: tuple[float, float],
    support_idx: int | None = None,
    frame: Literal["world", "local"] = "world",
    **kwargs,
):
    """Sample a constant xy placement position on the support surface.

    Parameters
    ----------
    support_data: list[ExtendedSupportData]
        the support data of the asset to place
    asset_AABB: torch.Tensor
        the AABB of the asset to place
    batch_size: int
        the number of placement positions to sample (one per environment).
    pos_xy: tuple[float, float]
        the xy position of the object to place, interpreted in ``frame`` (see below).
    support_idx: int | None
        which of the support entity's surfaces to place onto. ``None`` (the default) is the first that can hold
        the object. An integer indexes all of the support's surfaces, link by link and each link's largest
        first, like a Python list; out of range raises ``IndexError``, and a surface that cannot hold the
        object raises ``ValueError``. Both errors list the surfaces with their heights, centres and areas.
    frame: Literal["world", "local"]
        coordinate frame for ``pos_xy``. ``"world"`` (default) treats it as world-frame
        xy and reads z off the support plane at that xy. ``"local"`` treats it in the
        support facet's local frame (origin at an arbitrary mesh corner, axes possibly
        rotated) and maps it through ``d.transform``.
    **kwargs
        additional keyword arguments, ignored by this sampler.

    NOTE: This sampler aligns z-level to the support surface but *no guarantee* on the given pos_xy is on the surface.
    NOTE: This sampler does *not* check collisions.
          It is a user's responsibility to ensure the specified pos_xy is free from collision.
    """
    del asset_AABB  # unused
    d = _support_surface(support_data, support_idx)
    res_dir = d.normal

    if frame == "local":
        vertices_3d_on_plane = torch.zeros((batch_size, 1, 3), device=gs.device, dtype=gs.tc_float)  # (B, N, 3)
        if isinstance(pos_xy, tuple):
            vertices_3d_on_plane[:, :, 0] = pos_xy[0]
            vertices_3d_on_plane[:, :, 1] = pos_xy[1]
        else:
            vertices_3d_on_plane[:, :, :2] = pos_xy
        return transform_by_T(vertices_3d_on_plane, d.transform).squeeze(1), res_dir, None

    if frame != "world":
        raise ValueError(f"Unknown frame {frame!r}; expected 'world' or 'local'.")
    # NOTE: pos_xy is in *world* coordinates; the support plane only sets the z level
    # (the facet-local frame's origin is an arbitrary mesh corner, not the world origin).
    res_pos = torch.zeros((batch_size, 3), device=gs.device, dtype=gs.tc_float)  # (B, 3)
    if isinstance(pos_xy, tuple):
        res_pos[:, 0] = pos_xy[0]
        res_pos[:, 1] = pos_xy[1]
    else:
        res_pos[:, :2] = pos_xy
    res_pos[:, 2] = _z_on_support(res_pos, d)
    return res_pos, res_dir, None


def range_xy_sampler(
    support_data: list[ExtendedSupportData],
    asset_AABB: torch.Tensor,
    batch_size: int = 1,
    *,
    range_x: tuple[float, float] | torch.Tensor,
    range_y: tuple[float, float] | torch.Tensor,
    support_idx: int | None = None,
    frame: Literal["world", "local"] = "world",
    **kwargs,
):
    """Sample an xy placement position uniformly from a rectangular range on the support surface.

    Parameters
    ----------
    support_data: list[ExtendedSupportData]
        the support data of the asset to place
    asset_AABB: torch.Tensor
        the AABB of the asset to place
    batch_size: int
        the number of placement positions to sample (one per environment).
    range_x: tuple[float, float] | torch.Tensor
        the range of the x-coordinate of the object to place, interpreted in ``frame``.
    range_y: tuple[float, float] | torch.Tensor
        the range of the y-coordinate of the object to place, interpreted in ``frame``.
    support_idx: int | None
        which of the support entity's surfaces to place onto. ``None`` (the default) is the first that can hold
        the object. An integer indexes all of the support's surfaces, link by link and each link's largest
        first, like a Python list; out of range raises ``IndexError``, and a surface that cannot hold the
        object raises ``ValueError``. Both errors list the surfaces with their heights, centres and areas.
    frame: Literal["world", "local"]
        coordinate frame for the ranges. ``"world"`` (default) samples world-frame xy
        and reads z off the support plane; ``"local"`` samples in the support facet's
        local frame and maps through ``d.transform``.
    **kwargs
        additional keyword arguments, ignored by this sampler.

    NOTE: This sampler aligns z-level to the support surface but *no guarantee* on the given pos_xy is on the surface.
    NOTE: This sampler does *not* check collisions.
          It is a user's responsibility to ensure the specified pos_xy is free from collision.
    """
    del asset_AABB  # unused
    d = _support_surface(support_data, support_idx)
    res_dir = d.normal

    if frame == "local":
        lower = torch.zeros(batch_size, 3, device=gs.device, dtype=gs.tc_float)
        upper = torch.zeros(batch_size, 3, device=gs.device, dtype=gs.tc_float)
        lower[:, 0], lower[:, 1] = range_x[0], range_y[0]
        upper[:, 0], upper[:, 1] = range_x[1], range_y[1]
        vertices_3d_on_plane = sample_uniform(lower, upper, (batch_size, 3), device=gs.device)  # (B, 3)
        return transform_by_T(vertices_3d_on_plane, d.transform).squeeze(1), res_dir, None

    if frame != "world":
        raise ValueError(f"Unknown frame {frame!r}; expected 'world' or 'local'.")
    # NOTE: range_x/range_y are in *world* coordinates; the support plane only sets the z level.
    lower = torch.zeros(batch_size, 2, device=gs.device, dtype=gs.tc_float)
    upper = torch.zeros(batch_size, 2, device=gs.device, dtype=gs.tc_float)
    lower[:, 0], lower[:, 1] = range_x[0], range_y[0]
    upper[:, 0], upper[:, 1] = range_x[1], range_y[1]
    res_pos = torch.zeros((batch_size, 3), device=gs.device, dtype=gs.tc_float)  # (B, 3)
    res_pos[:, :2] = sample_uniform(lower, upper, (batch_size, 2), device=gs.device)
    res_pos[:, 2] = _z_on_support(res_pos, d)
    return res_pos, res_dir, None


def uniform_sampler(
    support_data: list[ExtendedSupportData],
    asset_AABB: torch.Tensor,
    holes: torch.Tensor | None = None,
    **kwargs,
):
    """Sample a placement position uniformly from the pre-sampled support points.

    Parameters
    ----------
    support_data: list[ExtendedSupportData]
        the support data of the asset to place
    asset_AABB: torch.Tensor
        the AABB of the asset to place (B, 2, 3)
    holes: torch.Tensor | None
        the AABBs of existing objects concatenated in dim=1, e.g., (B, L, 2, 3) for L holes
    **kwargs
        additional keyword arguments, ignored by this sampler.
    """
    offset = (asset_AABB[:, 1] - asset_AABB[:, 0]) / 2  # (B, 3)
    # consider only supports that are valid for current gravity
    support_data = [d for d in support_data if d.valid_mask.any()]

    res_pos = []
    res_dir = []
    res_valid_mask = []

    for d in support_data:
        if d.sample_points is not None:
            vertices_3d_on_plane = torch.zeros((1, d.num_sample_points, 3), device=gs.device, dtype=gs.tc_float)
            vertices_3d_on_plane[0, :, :2] = d.sample_points
            res_pos.append(transform_by_T(vertices_3d_on_plane, d.transform))  # B, N, 3
            res_dir.append(d.normal.unsqueeze(1).expand(-1, d.num_sample_points, 3))
            res_valid_mask.append(d.valid_mask.unsqueeze(1).expand(-1, d.num_sample_points))  # B, N

    if len(res_pos) > 0:
        res_pos = torch.cat(res_pos, dim=1)  # (B, sum(N), 3)
        res_dir = torch.cat(res_dir, dim=1)  # (B, sum(N), 3)
        res_valid_mask = torch.cat(res_valid_mask, dim=1)  # (B, sum(N),)

        # NOTE: filter out points inside holes
        if holes is not None:
            assert holes.ndim == 4, f"Unexpected holes shape {holes.shape}"
            if holes.shape[1] > 0:
                min_occ = holes[..., 0, :2] - offset[:, None, :2]
                max_occ = holes[..., 1, :2] + offset[:, None, :2]
                is_inside_tmp = torch.all(
                    (min_occ[:, None] <= res_pos[..., None, :2]) & (res_pos[..., None, :2] < max_occ[:, None]),
                    dim=-1,
                ).any(dim=-1)  # (B, sum(N))
                res_valid_mask = res_valid_mask & torch.logical_not(is_inside_tmp)

        valid_mask = res_valid_mask.any(dim=1)  # (B, )
        # multinomial rejects all-zero rows; make them uniform so sampling never
        # raises — the returned valid_mask flags those envs as failures downstream.
        probs = res_valid_mask.float()
        probs[probs.sum(dim=1) == 0] = 1.0
        random_true_indices = torch.multinomial(probs, 1).flatten()
        res_pos = res_pos[torch.arange(res_pos.shape[0], device=res_pos.device), random_true_indices]  # (B', 3)
        res_dir = res_dir[torch.arange(res_dir.shape[0], device=res_dir.device), random_true_indices]  # (B', 3)
        return res_pos, res_dir, valid_mask
    else:
        raise ValueError("No pre-sampled points found for the support data")


def gaussian_sampler(
    support_data: list[ExtendedSupportData],
    asset_AABB: torch.Tensor,
    holes: torch.Tensor | None = None,
    gaussian_weights: Callable | None = None,
    **kwargs,
):
    """Sample a placement position from the support points weighted by a Gaussian.

    Parameters
    ----------
    support_data: list[ExtendedSupportData]
        the support data of the asset to place
    asset_AABB: torch.Tensor
        the AABB of the asset to place
    holes: torch.Tensor | None
        the AABBs of existing objects concatenated in dim=1, e.g., (B, L, 2, 3) for L holes
    gaussian_weights: Callable | None
        callable mapping candidate points to per-point sampling weights. If None, a default is used.
    **kwargs
        additional keyword arguments, ignored by this sampler.
    """
    offset = (asset_AABB[:, 1] - asset_AABB[:, 0]) / 2  # (B, 3)
    # consider only supports that are valid for current gravity
    support_data = [d for d in support_data if d.valid_mask.any()]

    res_pos = []
    res_dir = []
    res_valid_mask = []
    res_weights = []

    for d in support_data:
        if d.sample_points is not None:
            vertices_3d_on_plane = torch.zeros((1, d.num_sample_points, 3), device=gs.device, dtype=gs.tc_float)
            vertices_3d_on_plane[0, :, :2] = d.sample_points
            pos = transform_by_T(vertices_3d_on_plane, d.transform)
            res_pos.append(pos)  # B, N, 3
            res_dir.append(d.normal.unsqueeze(1).expand(-1, d.num_sample_points, 3))

            res_weights.append(gaussian_weights(pos[..., :2]))  # B, N
            res_valid_mask.append(d.valid_mask.unsqueeze(1).expand(-1, d.num_sample_points))  # B, N

    if len(res_pos) > 0:
        res_pos = torch.cat(res_pos, dim=1)  # (B, sum(N), 3)
        res_dir = torch.cat(res_dir, dim=1)  # (B, sum(N), 3)
        res_valid_mask = torch.cat(res_valid_mask, dim=1)  # (B, sum(N),)
        res_weights = torch.cat(res_weights, dim=1)  # (B, sum(N),)

        # NOTE: filter out points inside holes
        if holes is not None:
            assert holes.ndim == 4, f"Unexpected holes shape {holes.shape}"
            if holes.shape[1] > 0:
                min_occ = holes[..., 0, :2] - offset[:, None, :2]
                max_occ = holes[..., 1, :2] + offset[:, None, :2]
                is_inside_tmp = torch.all(
                    (min_occ[:, None] <= res_pos[..., None, :2]) & (res_pos[..., None, :2] < max_occ[:, None]),
                    dim=-1,
                ).any(dim=-1)  # (B, sum(N))
                res_valid_mask = res_valid_mask & torch.logical_not(is_inside_tmp)

        valid_mask = res_valid_mask.any(dim=1)  # (B, )
        # multinomial rejects all-zero rows, which happen two ways: (1) the row has valid points
        # but their Gaussian weights underflowed to 0 — fall back to a uniform draw over just those
        # valid points (never invalid ones, e.g. inside a hole); (2) the row has no valid point at
        # all — give it all-ones so sampling doesn't raise (valid_mask flags it as a failure).
        probs = res_weights * res_valid_mask.float()
        zero_rows = probs.sum(dim=1) == 0
        probs[zero_rows] = res_valid_mask.float()[zero_rows]
        probs[~valid_mask] = 1.0
        random_true_indices = torch.multinomial(probs, 1).flatten()
        res_pos = res_pos[torch.arange(res_pos.shape[0], device=res_pos.device), random_true_indices]  # (B', 3)
        res_dir = res_dir[torch.arange(res_dir.shape[0], device=res_dir.device), random_true_indices]  # (B', 3)
        return res_pos, res_dir, valid_mask
    else:
        raise ValueError("No pre-sampled points found for the support data")


@EVENT_TERM_REGISTRY.register()
def place_constant_xy(
    env: EnvBase,
    envs_idx: slice | torch.Tensor | None,
    *,
    entity_name: str,
    support_entity_name: str,
    pos_xy: tuple[float, float],
    frame: Literal["world", "local"] = "world",
    support_idx: int | None = None,
):
    envs_idx = sanitize_envs_idx(envs_idx, env.num_envs)
    entity: Entity = env.get_entity(entity_name)
    support_entity: Entity = env.get_entity(support_entity_name)
    # NOTE: reset the orientation to the default root quat
    entity.set_quat(entity.get_default_root_pose()[1][envs_idx], envs_idx=envs_idx)
    sampler = partial(constant_xy_sampler, pos_xy=pos_xy, frame=frame, support_idx=support_idx)
    entity.place_on_to(support_entity, sampler=sampler, envs_idx=envs_idx)


@EVENT_TERM_REGISTRY.register()
def place_range_xy(
    env: EnvBase,
    envs_idx: slice | torch.Tensor | None,
    *,
    entity_name: str,
    support_entity_name: str,
    range_x: tuple[float, float],
    range_y: tuple[float, float],
    range_roll: tuple[float, float] = (0, 0),
    range_pitch: tuple[float, float] = (0, 0),
    range_yaw: tuple[float, float] = (0, 0),
    frame: Literal["world", "local"] = "world",
    support_idx: int | None = None,
):
    envs_idx = sanitize_envs_idx(envs_idx, env.num_envs)

    entity: Entity = env.get_entity(entity_name)
    support_entity: Entity = env.get_entity(support_entity_name)

    lower = torch.tensor(
        [range_roll[0], range_pitch[0], range_yaw[0]],
        device=env.device,
        dtype=torch.float32,
    )
    upper = torch.tensor(
        [range_roll[1], range_pitch[1], range_yaw[1]],
        device=env.device,
        dtype=torch.float32,
    )
    sampled = sample_uniform(lower, upper, (env.num_envs, 3), device=env.device)[envs_idx]
    quat = transform_quat_by_quat(entity.get_default_root_pose()[1][envs_idx], xyz_to_quat(sampled, rpy=True))
    entity.set_quat(quat, envs_idx=envs_idx)
    sampler = partial(range_xy_sampler, range_x=range_x, range_y=range_y, frame=frame, support_idx=support_idx)
    entity.place_on_to(support_entity, sampler=sampler, envs_idx=envs_idx)


@EVENT_TERM_REGISTRY.register()
def place_uniform(
    env: EnvBase,
    envs_idx: slice | torch.Tensor | None,
    *,
    entity_name: str,
    support_entity_name: str,
    samples_per_area: int = 100,
):
    envs_idx = sanitize_envs_idx(envs_idx, env.num_envs)
    entity: Entity = env.get_entity(entity_name)
    support_entity: Entity = env.get_entity(support_entity_name)
    # NOTE: reset the orientation to the default root quat
    entity.set_quat(entity.get_default_root_pose()[1][envs_idx], envs_idx=envs_idx)
    sampler = partial(uniform_sampler, samples_per_area=samples_per_area)
    entity.place_on_to(support_entity, sampler=sampler, envs_idx=envs_idx)


@EVENT_TERM_REGISTRY.register()
class PlaceGaussian(EventTerm):
    """Event term that places an entity on a support surface via Gaussian-weighted sampling.

    Parameters
    ----------
    entity_name: str
        The name of the entity to place
    support_entity_name: str
        The name of the support entity
    mu_x: float
        The mean of the x-coordinate in world frame
    mu_y: float
        The mean of the y-coordinate in world frame
    var_x: float
        The variance of the x-coordinate (m^2)
    var_y: float
        The variance of the y-coordinate (m^2)
    """

    entity_name: str = ""
    support_entity_name: str = ""
    mu_x: float = 0.0
    mu_y: float = 0.0
    var_x: float = 0.1
    var_y: float = 0.1

    def __init__(
        self,
        env: EnvBase,
        options: EventTermOptions,
    ):
        super().__init__(env=env, options=options)
        self.entity: Entity | None = None
        self.support_entity: Entity | None = None
        self.sampler = None

    def build(self) -> None:
        self.entity = self.resolve_entity(self.entity_name)
        self.support_entity = self.resolve_entity(self.support_entity_name)
        self.sampler = partial(
            gaussian_sampler,
            gaussian_weights=Gaussian2D(
                mu_x=self.mu_x,
                mu_y=self.mu_y,
                var_x=self.var_x,
                var_y=self.var_y,
            ).get_probs,
        )

    def compute(self, envs_idx: slice | torch.Tensor | None):
        envs_idx = sanitize_envs_idx(envs_idx, self._env.num_envs)

        # NOTE: reset the orientation to the default root quat
        self.entity.set_quat(self.entity.get_default_root_pose()[1][envs_idx], envs_idx=envs_idx)
        self.entity.place_on_to(self.support_entity, sampler=self.sampler, envs_idx=envs_idx)


@EVENT_TERM_REGISTRY.register()
class PlaceFromLayouts(EventTerm):
    """Each reset env draws one of a fixed list of layouts and places every entity the layout names.

    A layout maps entity names to ``(x, y, yaw_deg)``: the world-frame xy on the support surface (z comes
    off the surface) and a yaw in degrees on top of the entity's default root orientation. Drawing all the
    entities' poses as one layout is what keeps them apart: two entities placed by independent ranges over
    the same area can start overlapping, and physics then throws them apart on the first step.

    Parameters
    ----------
    layouts: list[dict[str, tuple[float, float, float]]]
        The layouts, each naming the same entities. An env's layout is drawn uniformly at every reset.
    support_entity_name: str
        The entity whose surface the entities stand on.
    """

    mode: EventMode = EventMode.RESET
    layouts: list[dict[str, tuple[float, float, float]]] = []
    support_entity_name: str = "plane"

    def __init__(self, env: EnvBase, options: EventTermOptions):
        super().__init__(env=env, options=options)
        if not self.layouts:
            raise ValueError(
                "PlaceFromLayouts needs at least one layout: pass layouts=[{'<entity>': (x, y, yaw_deg), ...}, ...]."
            )
        self.entity_names = list(self.layouts[0])
        for i, layout in enumerate(self.layouts):
            if sorted(layout) != sorted(self.entity_names):
                raise ValueError(
                    f"PlaceFromLayouts layout {i} names {sorted(layout)}, but layout 0 names "
                    f"{sorted(self.entity_names)}; every layout must place the same entities."
                )
        poses = torch.tensor(
            [[layout[name] for name in self.entity_names] for layout in self.layouts],
            dtype=gs.tc_float,
            device=env.device,
        )  # (layouts, entities, 3)
        self._xy = poses[..., :2]
        rpy = torch.zeros_like(poses)
        rpy[..., 2] = torch.deg2rad(poses[..., 2])
        self._yaw_quat = xyz_to_quat(rpy.reshape(-1, 3), rpy=True).reshape(*poses.shape[:2], 4)

    def build(self) -> None:
        # Touch every entity once so a misspelled name raises at env construction, not at the first reset.
        for name in (*self.entity_names, self.support_entity_name):
            self.resolve_entity(name)

    def compute(self, envs_idx: slice | torch.Tensor | None):
        envs_idx, n_envs = sanitize_envs_idx(envs_idx, self._env.num_envs, return_n_envs=True)
        if n_envs == 0:
            return
        layout = torch.randint(len(self.layouts), (n_envs,), device=self._env.device)
        support = self.resolve_entity(self.support_entity_name)
        for k, name in enumerate(self.entity_names):
            entity = self.resolve_entity(name)
            quat = transform_quat_by_quat(entity.get_default_root_pose()[1][envs_idx], self._yaw_quat[layout, k])
            entity.set_quat(quat, envs_idx=envs_idx)
            entity.place_on_to(
                support, sampler=partial(constant_xy_sampler, pos_xy=self._xy[layout, k]), envs_idx=envs_idx
            )
