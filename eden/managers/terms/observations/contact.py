"""Per-link contact-force-norm observation.

Reads :meth:`RigidEntity.get_links_net_contact_force` (Genesis-aggregated
per-link net force; ``(num_envs, n_links, 3)`` in world frame), slices to
the configured link set, and returns the per-link force magnitude.

Trade-off vs. ``gs.sensors.ContactForce``: the Genesis sensor binds to
**one** link per ``SensorOptions`` declaration, so a 14-link hand needs
14 sensor entries plus 14 obs-term entries to do what
:class:`ContactForceNorm` does in a single declaration. The cost is no
``NoisySensorMixin`` noise / delay / decimation modeling; that path is
reachable via the :class:`~eden.managers.terms.observations.sensors.SensorRead` term plus a
per-link sensor declaration if a config wants noise on a small subset
of links.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY, ObservationTerm
from eden.managers.terms.utils import resolve_link_idx
from eden.utils.motion_reference.lookup import (
    assert_motion_field_present,
    get_motion_link_idx,
    resolve_frame_offsets,
    select_motion_frames,
)
from eden.utils.string import term_subject

if TYPE_CHECKING:
    from eden.envs.base import MimicRLEnvBase


@OBSERVATION_TERM_REGISTRY.register()
class ContactForceNorm(ObservationTerm):
    """Per-link contact-force magnitude (Euclidean, world frame).

    Returns ``(num_envs, K)`` where ``K = len(matched links)``, each
    entry being ``‖f_link‖`` at the most recent ``scene.step()``.

    Parameters
    ----------
    entity_name:
        Entity to read forces from (must be a ``RigidEntity``).
    links_name:
        :func:`resolve_matching_names`-compatible patterns selecting which
        links contribute. Empty list raises in :meth:`build` — this term
        is per-link by design; use :func:`self_collision_cost` for a
        global "any contact" scalar.

    Notes
    -----
    Output scaling, noise, clipping, and history are controlled via
    the inherited :class:`ObservationTermOptions` fields (``scale``,
    ``noise``, ``clip``, ``history_length``) — pass e.g. ``scale=0.01``
    on the term options when forces are in newtons and the policy
    expects roughly-unit observations.

    The underlying force is in **world frame** — Genesis's
    ``get_links_net_contact_force`` returns the world-frame net force per
    link. ``gs.sensors.ContactForce`` (the noise-aware path reachable via
    the generic :class:`~eden.managers.terms.observations.sensors.SensorRead` obs term) returns the same vector
    in the **link's local frame**. Magnitudes are equal under any choice
    of frame, so this term's per-link norm is unaffected by the choice;
    a config swapping between this term and a sensor-reading-plus-norm
    composition gets identical scalar values but watch the frame if a
    downstream consumer uses the raw force vector.
    """

    entity_name: str = "robot"
    links_name: list[str] = []

    def build(self) -> None:
        # Touch the entity here for fail-fast, but cache only the derived link idx —
        # resolve the entity lazily in compute() so deployment hot-swaps of
        # env.entities[name] are honoured (see ManagerTermBase.resolve_entity).
        entity = self.resolve_entity(self.entity_name)
        self._link_idx = resolve_link_idx(entity, self.links_name, self.device, term=type(self).__name__)

    def compute(self) -> torch.Tensor:
        forces = self.resolve_entity(self.entity_name).get_links_net_contact_force()  # (B, n_links, 3)
        return torch.norm(forces[:, self._link_idx], dim=-1)


@OBSERVATION_TERM_REGISTRY.register()
class GroupedContactFlags(ObservationTerm):
    """Per-(link, partner-link) boolean contact flags with a specific entity.

    Wraps :meth:`RigidEntity.get_grouped_contacts` and returns its ``valid``
    mask as floats in ``{0, 1}``, flattened to ``(num_envs, n_a * n_b)`` —
    the simulated counterpart of :func:`motion_contact_flags` for
    contact-annotated mimic tasks (ConTrack ``xhand_contact``).

    Parameters
    ----------
    entity_name:
        Active entity (usually the hand); its links index the first axis.
    object_entity_name:
        Contact partner (usually the manipulated object).
    links_name:
        Link patterns on ``entity_name`` selecting the rows. Required —
        the row order defines the observation layout.
    object_links_name:
        Link patterns on the partner (columns). Empty selects all links.
    force_threshold:
        ``None`` (default) flags a pair whose aggregate contact force is strictly positive (the ``valid`` mask).
        A number flags a pair only when that aggregate force (``force_norm_sum``, N) exceeds it — a contact
        *sensor* threshold, like Weave's 1 N, that ignores grazing contacts carrying almost no force.
    """

    entity_name: str = "robot"
    object_entity_name: str = "object"
    links_name: list[str] = []
    object_links_name: list[str] = []
    force_threshold: float | None = None

    def build(self) -> None:
        # Touch both entities here for fail-fast, but cache only the derived link
        # idx — resolve the entities lazily in compute() so deployment hot-swaps of
        # env.entities[name] are honoured (see ManagerTermBase.resolve_entity).
        entity = self.resolve_entity(self.entity_name)
        obj = self.resolve_entity(self.object_entity_name)
        if not self.links_name:
            raise ValueError(f"{term_subject(self)}: links_name is required (defines the observation layout).")
        self._links_idx = resolve_link_idx(entity, self.links_name, self.device)
        self._object_links_idx = (
            resolve_link_idx(obj, self.object_links_name, self.device) if self.object_links_name else None
        )

    def compute(self) -> torch.Tensor:
        contacts = self.resolve_entity(self.entity_name).get_grouped_contacts(
            self.resolve_entity(self.object_entity_name),
            links_a_idx_local=self._links_idx,
            links_b_idx_local=self._object_links_idx,
        )
        flags = contacts["valid"] if self.force_threshold is None else contacts["force_norm_sum"] > self.force_threshold
        return flags.reshape(flags.shape[0], -1).to(torch.float32)


@OBSERVATION_TERM_REGISTRY.register()
def motion_contact_flags(env: MimicRLEnvBase, *, entity_name: str, links_name: list[str]) -> torch.Tensor:
    """Demo per-(link, partner) contact flags at the current motion frame.

    Reads ``motion_manager.data[entity_name].contact_points_valid`` (frame 0)
    for the named motion links; returns floats in ``{0, 1}`` flattened to
    ``(num_envs, L * num_partners)`` — the reference counterpart of
    :class:`GroupedContactFlags` (ConTrack ``ref_xhand_contact``).
    """
    assert_motion_field_present(env, entity_name, "contact_points_valid")
    motion_idx = get_motion_link_idx(env, entity_name, links_name)
    valid = env.motion_manager.data[entity_name].contact_points_valid[:, 0].index_select(1, motion_idx)
    return valid.reshape(valid.shape[0], -1).to(torch.float32)


@OBSERVATION_TERM_REGISTRY.register()
def motion_future_contact_label(
    env: MimicRLEnvBase,
    *,
    entity_name: str,
    frame_offsets: Sequence[int],
    links_name: list[str] | None = None,
    flatten: bool = True,
) -> torch.Tensor:
    """Demo tri-state per-link contact labels (+1 contact / -1 no contact / 0 unlabelled) at future frames.

    Shape ``(N, K*L)`` (or ``(N, K, L)`` if ``flatten=False``), as floats. Reads
    ``motion_manager.data[entity_name].contact_label``; ``links_name=None`` returns every motion link of the entity.
    The reference counterpart of :class:`GroupedContactFlags` for a dataset whose contact annotation distinguishes
    "must not touch" from "don't care" (e.g. Weave's G1-Inspire-HOI labels).
    """
    assert_motion_field_present(env, entity_name, "contact_label")
    ref = env.motion_manager.data[entity_name]
    offsets = resolve_frame_offsets(env, "contact_label", frame_offsets, ref.contact_label.shape[1])
    sel = None if links_name is None else get_motion_link_idx(env, entity_name, links_name)
    labels = select_motion_frames(ref.contact_label, offsets, sel).to(torch.float32)  # (N, K, L)
    return labels.reshape(labels.shape[0], -1) if flatten else labels
