"""Reset events for Mochi soft bodies."""

from __future__ import annotations

import torch
from genesis.utils.geom import transform_by_quat

from eden.managers.event_manager import EVENT_TERM_REGISTRY, EventTerm
from eden.utils.geom import quat_from_euler_xyz
from eden.utils.misc import sanitize_envs_idx
from eden.utils.string import term_subject


@EVENT_TERM_REGISTRY.register()
class PinSoftBodyVertices(EventTerm):
    """Pin vertices of a Mochi soft body where they stand (a Dirichlet condition).

    Fire it at RESET. Mochi keeps a pin, and the position it holds the vertex to, outside the state a reset
    restores, and writing a pinned vertex's position moves that hold too — so a pin made once would keep
    holding wherever the vertex was last written, not where the reset puts it. Re-pinning on every reset holds
    the vertices where the restored state has them. A rod is clamped by pinning its first two nodes (the second
    fixes the twist).

    Parameters
    ----------
    entity_name : str
        The Mochi soft body.
    vertices : list[int]
        Local indices of the vertices to pin.
    """

    entity_name: str = ""
    vertices: list[int] = []

    def build(self) -> None:
        body = self.resolve_entity(self.entity_name)
        if not hasattr(body, "set_vertices_fixed"):
            raise ValueError(
                f"{term_subject(self)} pins a Mochi soft body's vertices, but the entity is a {type(body).__name__}."
            )
        bad = [v for v in self.vertices if not 0 <= v < body.n_vertices]
        if not self.vertices or bad:
            raise ValueError(
                f"{term_subject(self)} needs vertices of the entity (0..{body.n_vertices - 1}); got "
                f"{self.vertices}{f', of which {bad} are out of range' if bad else ''}."
            )

    def compute(self, envs_idx: slice | torch.Tensor | None) -> None:
        self.resolve_entity(self.entity_name).set_vertices_fixed(self.vertices, is_fixed=True, envs_idx=envs_idx)


@EVENT_TERM_REGISTRY.register()
class ResetSoftBodyPlanarPose(EventTerm):
    """Place a Mochi soft body at its rest shape, shifted in x / y and turned about z by uniform draws.

    The rest geometry (as the scene was built) is turned by the drawn yaw about its own centroid and shifted by
    the drawn offset, and every vertex velocity is zeroed — a rigid placement of the undeformed body, which
    ``set_pos`` / ``set_quat`` cannot express for a vertex-based body.

    Parameters
    ----------
    entity_name : str
        The Mochi soft body.
    pos_x_range, pos_y_range : tuple[float, float]
        Offset ranges (m) from the rest position.
    yaw_range : tuple[float, float]
        Rotation range (rad) about the vertical axis through the rest centroid.
    """

    entity_name: str = ""
    pos_x_range: tuple[float, float] = (0.0, 0.0)
    pos_y_range: tuple[float, float] = (0.0, 0.0)
    yaw_range: tuple[float, float] = (0.0, 0.0)

    def build(self) -> None:
        body = self.resolve_entity(self.entity_name)
        if not hasattr(body, "set_vertex_positions"):
            raise ValueError(
                f"{term_subject(self)} places a Mochi soft body, but the entity is a "
                f"{type(body).__name__}; place rigid bodies with events.reset_base_state_uniform."
            )
        rest = torch.as_tensor(body.init_positions, dtype=torch.float, device=self.device)
        self._centroid = rest.mean(dim=0)
        self._rest_centred = rest - self._centroid

    def compute(self, envs_idx: slice | torch.Tensor | None) -> None:
        envs_idx, n_envs = sanitize_envs_idx(envs_idx, self.num_envs, return_n_envs=True)
        if n_envs == 0:
            return
        draws = torch.rand(n_envs, 3, device=self.device)
        low = torch.tensor([self.pos_x_range[0], self.pos_y_range[0], self.yaw_range[0]], device=self.device)
        high = torch.tensor([self.pos_x_range[1], self.pos_y_range[1], self.yaw_range[1]], device=self.device)
        offset_x, offset_y, yaw = (low + (high - low) * draws).unbind(dim=-1)
        zero = torch.zeros_like(yaw)
        quat = quat_from_euler_xyz(zero, zero, yaw)[:, None, :].expand(-1, len(self._rest_centred), -1)
        verts = transform_by_quat(self._rest_centred.expand(n_envs, -1, -1), quat) + self._centroid
        verts[..., 0] += offset_x[:, None]
        verts[..., 1] += offset_y[:, None]
        body = self.resolve_entity(self.entity_name)
        body.set_vertex_positions(verts, envs_idx=envs_idx)
        body.set_vertex_velocities(torch.zeros_like(verts), envs_idx=envs_idx)


@EVENT_TERM_REGISTRY.register()
class DisableMochiContact(EventTerm):
    """Switch off contact between two Mochi bodies, in every env. Fire it at STARTUP.

    Both bodies keep every other contact, so this lets one pass through a surface that still stops the rest:
    ``mochi_route_the_cable`` takes its cable off the bench plane the arms stand on, and lays it on a bed of its own.

    Parameters
    ----------
    entity_name, other_name : str
        The two Mochi bodies (rigid or soft).
    """

    entity_name: str = ""
    other_name: str = ""

    def build(self) -> None:
        for name in (self.entity_name, self.other_name):
            if not hasattr(self.resolve_entity(name), "set_contact_enabled"):
                raise ValueError(
                    f"{term_subject(self)} switches contact between Mochi bodies, but '{name}' is a "
                    f"{type(self.resolve_entity(name)).__name__}."
                )

    def compute(self, envs_idx: slice | torch.Tensor | None) -> None:
        self.resolve_entity(self.entity_name).set_contact_enabled(self.resolve_entity(self.other_name), False)
