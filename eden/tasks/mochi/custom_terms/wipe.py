"""Terms of ``mochi_wipe_the_bar``: a marker stripe on a grab bar's crown, worn off by a Mochi sponge.

RoboSimStudio's ``MarkerStripe`` (``examples/08_demos/wipe_the_bar/bar.py``), ported: the stripe is a grid of
ink cells on the tube's surface, ``pitch`` apart along the bar and round it, covering ``half_length`` either side
of the bar's middle and ``half_angle`` either side of the crown. Each control step a cell wears under its
nearest pressing contact within ``reach`` by Preston's law, ``ink' = -k_p * pressure * sliding speed``, with no
wear under ``min_pressure``; a cell with ``cleaned_below`` of its ink or less is clean, and the share of clean
cells is the job's measure. The constants are RoboSimStudio's.

RoboSimStudio feeds the law SuperDex's contact samples against the bar (force over quadrature area). Eden's
contact samples sit on the sponge's surface vertices: each vertex within ``band`` of the tube's surface presses
with its contact force along the tube's outward normal, over the share of the sponge's surface it stands for,
and slides with its velocity across that normal. A sample stands for a patch as wide as the vertex spacing, so
``reach`` is the half-diagonal of that spacing rather than RoboSimStudio's 4 mm (its samples are 2.5 mm apart).

:class:`BarStripeWipe` owns the ink and is the success metric; the observation and reward terms read it through
the metric manager, so the wear integral runs once per step. The stripe is not geometry in the scene: a camera
shows it through a :class:`~eden.entities.camera_overlays.PlanePaintOverlay`, which paints
:meth:`BarStripeWipe.painted_plane` onto its frames (the camera the task gives ``eden inference --rec`` has one).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from eden.entities.camera_overlays import PaintedPlane
from eden.managers.metric_manager import METRIC_TERM_REGISTRY, MetricTerm
from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY, ObservationTerm
from eden.managers.reward_manager import REWARD_TERM_REGISTRY, RewardTerm
from eden.managers.terms.utils import resolve_metric_manager, resolve_metric_term
from eden.utils.plane_grid import PlaneGrid
from eden.utils.string import term_subject

# Faces of a tetrahedron (vertex positions within it), wound outward for a positively oriented tet.
_TET_FACES = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))


def surface_lumped_areas(verts: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Surface vertices of a tetrahedral mesh and the share of its surface area each stands for.

    Each boundary triangle hands a third of its area to each corner: the area over which a contact force
    on that vertex is spread.
    """
    faces = tets[:, _TET_FACES].reshape(-1, 3)
    _, index, count = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
    surface = faces[index[count == 1]]
    a, b, c = verts[surface[:, 0]], verts[surface[:, 1]], verts[surface[:, 2]]
    third = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1) / 3.0
    area = np.zeros(len(verts))
    np.add.at(area, surface.reshape(-1), np.repeat(third, 3))
    surface_verts = np.flatnonzero(area > 0.0)
    return surface_verts, area[surface_verts]


@METRIC_TERM_REGISTRY.register()
class BarStripeWipe(MetricTerm):
    """Share of a marker stripe on a horizontal bar's crown that a soft body has wiped clean (the success metric).

    The bar's straight run is a cylinder whose axis runs along world y through ``bar_axis`` (its middle); its
    crown is the top, world +z. A cell sits at ``(u, angle)``: ``u`` along the bar from its middle, ``angle``
    round it from the crown toward +x.

    Parameters
    ----------
    entity_name : str
        The Mochi soft body doing the wiping (a tetrahedral solid).
    bar_name : str
        The bar entity the stripe lies on: a recording camera shades the stripe as it shades the bar.
    bar_axis : tuple[float, float, float]
        The middle of the bar's axis, world frame (m).
    radius : float
        The tube's radius (m).
    half_length : float
        The stripe covers ``u`` in ``[-half_length, half_length]`` (m).
    half_angle_deg : float
        The stripe covers ``angle`` within this of the crown (degrees).
    pitch : float
        Cell size on the surface (m); RoboSimStudio's 2.5 mm makes 81 x 3 cells.
    band : float
        A surface vertex within this over the tube's surface, or inside it, presses on it (m): RoboSimStudio's 4 mm.
    reach : float
        A cell wears under its nearest pressing vertex only within this (m), measured on the surface.
    wear_coefficient : float
        Preston coefficient ``k_p`` (1 / (Pa m)).
    min_pressure : float
        Contact pressure (Pa) below which a vertex wears nothing.
    cleaned_below : float
        Ink level (starting from 1) at or under which a cell counts as clean.
    """

    entity_name: str = "sponge"
    bar_name: str = "bar"
    bar_axis: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 0.0125
    half_length: float = 0.100
    half_angle_deg: float = 9.0
    pitch: float = 0.0025
    band: float = 0.004
    reach: float = 0.006
    wear_coefficient: float = 1.2e-2
    min_pressure: float = 300.0
    cleaned_below: float = 0.1
    success_threshold: float = 0.96

    def build(self) -> None:
        sponge = self.resolve_entity(self.entity_name)
        self.resolve_entity(self.bar_name)
        if sponge.elems.shape[1] != 4:
            raise ValueError(
                f"{term_subject(self)} needs a tetrahedral solid to wipe with, but the entity has "
                f"{sponge.elems.shape[1]}-vertex elements (a shell or rod). Give it MochiElasticMaterialOptions."
            )
        # RoboSimStudio's grid: round(half / pitch) cells either side of the middle, both ways.
        n_u = round(self.half_length / self.pitch)
        n_a = round(math.radians(self.half_angle_deg) * self.radius / self.pitch)
        self._n_u, self._n_a = n_u, n_a
        u = torch.arange(-n_u, n_u + 1, device=self.device, dtype=torch.float) * self.pitch
        angle = torch.arange(-n_a, n_a + 1, device=self.device, dtype=torch.float) * (self.pitch / self.radius)
        grid_u, grid_a = torch.meshgrid(u, angle, indexing="ij")
        self.cells = torch.stack([grid_u.reshape(-1), grid_a.reshape(-1)], dim=-1)  # (n_cells, 2): u [m], angle [rad]
        self._axis = torch.tensor(self.bar_axis, device=self.device, dtype=torch.float)
        surface, area = surface_lumped_areas(np.asarray(sponge.init_positions), np.asarray(sponge.elems))
        self._surface = torch.as_tensor(surface, dtype=torch.long, device=self.device)
        self._area = torch.as_tensor(area, dtype=torch.float, device=self.device)
        self.ink = torch.ones(self.num_envs, len(self.cells), device=self.device)
        # Ink removed this step, as a share of the whole stripe (the dense reward reads it).
        self.wiped_this_step = torch.zeros(self.num_envs, device=self.device)
        # The sponge's press on the bar (N): the outward contact force of every vertex in the band.
        self.press = torch.zeros(self.num_envs, device=self.device)

    def compute(self) -> torch.Tensor:
        sponge = self.resolve_entity(self.entity_name)
        rel = sponge.get_vertex_positions()[:, self._surface] - self._axis  # (B, n_surface, 3)
        radial = rel[..., (0, 2)]  # across the axis: x and z
        distance = radial.norm(dim=-1).clamp_min(1e-9)
        normal = torch.zeros_like(rel)
        normal[..., 0], normal[..., 2] = radial[..., 0] / distance, radial[..., 1] / distance
        # Within the band over the surface, or pressed into it: a crown narrower than the vertex spacing sinks between
        # vertices, which then sit inside the tube's radius.
        on_tube = (distance - self.radius <= self.band) & (rel[..., 1].abs() <= self.half_length + self.reach)
        force = (sponge.get_vertex_contact_forces()[:, self._surface] * normal).sum(dim=-1).clamp_min(0.0)
        force = torch.where(on_tube, force, 0.0)
        self.press.copy_(force.sum(dim=-1))
        pressure = force / self._area
        live = pressure >= self.min_pressure
        velocity = sponge.get_vertex_velocities()[:, self._surface]
        slide = (velocity - normal * (velocity * normal).sum(dim=-1, keepdim=True)).norm(dim=-1)
        rate = torch.where(live, self.wear_coefficient * pressure * slide, 0.0)  # (B, n_surface)
        # Each cell wears under its nearest live vertex, measured on the surface: along the bar and round it.
        angle = torch.atan2(radial[..., 0], radial[..., 1])  # from the crown (+z) toward +x
        du = self.cells[None, :, None, 0] - rel[:, None, :, 1]
        da = torch.remainder(self.cells[None, :, None, 1] - angle[:, None, :] + math.pi, 2 * math.pi) - math.pi
        d2 = (du.square() + (da * self.radius).square()).masked_fill(~live[:, None, :], torch.inf)
        nearest_d2, nearest = d2.min(dim=-1)  # (B, n_cells)
        wear = torch.where(nearest_d2 <= self.reach**2, rate.gather(1, nearest), 0.0) * self._env.dt
        before = self.ink.clone()
        self.ink.sub_(wear).clamp_(min=0.0)
        self.wiped_this_step.copy_((before - self.ink).mean(dim=-1))
        return self.cleaned

    @property
    def cleaned(self) -> torch.Tensor:
        """Share of the stripe's cells with ``cleaned_below`` of their ink or less, ``(num_envs,)``."""
        return (self.ink <= self.cleaned_below).float().mean(dim=-1)

    def painted_plane(self, envs_idx: torch.Tensor) -> PaintedPlane:
        """The ink left on the stripe of the envs ``envs_idx``, cell by cell, for a camera's
        :class:`~eden.entities.camera_overlays.PlanePaintOverlay`, on the plane touching the crown: ``u`` along the bar,
        ``v`` round it (toward +x), the arc length standing for the plane's distance.

        It meets the overlay's rule for a curved surface: over the cells (the three round the bar reach 17 degrees at
        the default pitch, wider than the nominal 9) the tube sags at most 0.6 mm from the plane, and arc and plane
        distances differ by under 0.1 mm. Each cell is painted as wide as it is graded, as dark as the ink it holds.
        """
        pitch = self.pitch
        ink = self.ink[envs_idx].view(-1, 2 * self._n_u + 1, 2 * self._n_a + 1)  # (K, along, round), cells' order
        return PaintedPlane(
            origin=self._axis + torch.tensor((0.0, 0.0, self.radius), device=self.device),
            u_axis=torch.tensor((0.0, 1.0, 0.0), device=self.device),
            v_axis=torch.tensor((1.0, 0.0, 0.0), device=self.device),
            grid=PlaneGrid(
                -(self._n_u + 0.5) * pitch, (self._n_a + 0.5) * pitch, pitch, 2 * self._n_a + 1, 2 * self._n_u + 1
            ),
            # Rows run round the bar from the largest angle (row 0 is the grid's top, its largest v), columns along it.
            weight=ink.transpose(1, 2).flip(1),
            entity_name=self.bar_name,
        )

    def cell_positions(self) -> torch.Tensor:
        """World positions of the stripe's cells on the tube's surface, ``(n_cells, 3)``."""
        u, angle = self.cells[:, 0], self.cells[:, 1]
        offset = torch.stack([self.radius * torch.sin(angle), u, self.radius * torch.cos(angle)], dim=-1)
        return self._axis + offset

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        self.ink[envs_idx] = 1.0
        self.wiped_this_step[envs_idx] = 0.0
        self.press[envs_idx] = 0.0


@METRIC_TERM_REGISTRY.register()
class BarPress(MetricTerm):
    """The sponge's press on the bar (N), from the :class:`BarStripeWipe` metric — a shared quantity, not a gate.

    Configure ``contributes_to_success=False``, and declare it after its ``BarStripeWipe`` metric: the metric manager
    computes its terms in declaration order, so declared first it would read the previous step's press. RoboSimStudio's
    routine presses to 4 N.
    """

    metric_name: str = "cleaned"

    def build(self) -> None:
        # Sibling metrics are built in this same pass, so the wipe term is checked in `resolve_refs` and fetched on
        # the first `compute`, once the metric manager exists.
        self._wipe: BarStripeWipe | None = None

    def resolve_refs(self, manager_options) -> None:
        """Check that ``metric_name`` is a ``BarStripeWipe`` metric declared before this one (every metric is built)."""
        names = list(manager_options.term_keys())
        own = next((name for name in names if getattr(manager_options, name) is self._options), "<this term>")
        source = getattr(manager_options, self.metric_name) if self.metric_name in names else None
        if source is None or source.name != "bar_stripe_wipe":
            wipes = [name for name in names if getattr(manager_options, name).name == "bar_stripe_wipe"]
            raise ValueError(
                f"BarPress metric '{own}' reads metric_name='{self.metric_name}', which is not a BarStripeWipe metric "
                f"(BarStripeWipe metrics: {wipes}; all metrics: {names}). Set metric_name to one of the first list."
            )
        if own in names and names.index(self.metric_name) > names.index(own):
            raise ValueError(
                f"BarPress metric '{own}' is declared before its BarStripeWipe metric '{self.metric_name}', so it would "
                f"read the previous step's press (metrics compute in declaration order: {names}). Declare '{own}' "
                f"after '{self.metric_name}'."
            )

    def compute(self) -> torch.Tensor:
        if self._wipe is None:
            self._wipe = resolve_metric_term(
                self._env, term="BarPress", metric_name=self.metric_name, expected=BarStripeWipe
            )
        return self._wipe.press


@OBSERVATION_TERM_REGISTRY.register()
class StripeInk(ObservationTerm):
    """Ink left in every cell of the stripe, ``(num_envs, n_cells)``, from the :class:`BarStripeWipe` metric."""

    metric_name: str = "cleaned"

    def build(self) -> None:
        self._wipe = resolve_metric_term(
            self._env, term="StripeInk", metric_name=self.metric_name, expected=BarStripeWipe
        )

    def compute(self) -> torch.Tensor:
        return self._wipe.ink


@REWARD_TERM_REGISTRY.register()
class WipeProgress(RewardTerm):
    """Share of the stripe's ink removed this step (dense; sums to the ink removed over an episode)."""

    metric_name: str = "cleaned"

    def build(self) -> None:
        self._wipe = resolve_metric_term(
            self._env, term="WipeProgress", metric_name=self.metric_name, expected=BarStripeWipe
        )

    def compute(self) -> torch.Tensor:
        return self._wipe.wiped_this_step


@REWARD_TERM_REGISTRY.register()
class ReachDirtyCell(RewardTerm):
    """``exp(-d^2 / std^2)`` while the sponge is held, ``d`` from its underside to the stroke's next dirty cell.

    ``d`` runs from the centre of the sponge's lowest vertices (its wiping face) to the crown at the dirty cell
    nearest the stripe's near end (0 once the stripe is clean): RoboSimStudio's strokes run one way, from the near end
    to the far one, and the target walks along the bar ahead of the sponge as the cells under it come clean. Pulling
    toward the *nearest* dirty cell instead was farmed: the policy landed mid-stripe, wiped the far part and hovered,
    since the near part lay behind it. Reaching the stripe's centre or an ink-weighted centroid pays for pressing in
    one place (both were farmed). Nothing is paid while the grip metric
    ``grip_metric_name`` is under ``grip_threshold``, so carrying the sponge is the only way to it.
    """

    entity_name: str = "sponge"
    metric_name: str = "cleaned"
    grip_metric_name: str = "grip"
    grip_threshold: float = 2.0
    std: float = 0.1
    face_depth: float = 0.006

    def build(self) -> None:
        self.resolve_entity(self.entity_name)
        self._wipe = resolve_metric_term(
            self._env, term="ReachDirtyCell", metric_name=self.metric_name, expected=BarStripeWipe
        )
        resolve_metric_manager(self._env, term="reach_dirty_cell", metric_name=self.grip_metric_name)
        # The crown over the bar's middle; the target slides along it.
        self._crown = self._wipe._axis + torch.tensor([0.0, 0.0, self._wipe.radius], device=self.device)

    def compute(self) -> torch.Tensor:
        verts = self.resolve_entity(self.entity_name).get_vertex_positions()
        z = verts[..., 2]
        face = z <= z.amin(dim=-1, keepdim=True) + self.face_depth
        face_centre = (verts * face[..., None]).sum(dim=1) / face.sum(dim=-1, keepdim=True)
        # The first dirty cell along the bar (smallest u), on the crown.
        along = self._wipe.cells[:, 0].expand_as(self._wipe.ink)
        dirty = self._wipe.ink > self._wipe.cleaned_below
        first = along.masked_fill(~dirty, torch.inf).amin(dim=-1)
        target = self._crown + torch.zeros_like(face_centre)
        target[:, 1] = target[:, 1] + torch.where(torch.isinf(first), 0.0, first)
        d2 = torch.where(dirty.any(dim=-1), (target - face_centre).square().sum(dim=-1), 0.0)
        held = self._env.metric_manager.get_value(self.grip_metric_name) >= self.grip_threshold
        return torch.where(held, torch.exp(-d2 / self.std**2), 0.0)
