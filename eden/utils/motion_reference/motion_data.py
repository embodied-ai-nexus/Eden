"""Motion-reference data containers (MotionSequence, buffers, state)."""

from __future__ import annotations
from typing import ClassVar, Literal

from dataclasses import dataclass, replace
import torch

from eden.utils.buffers.cbt import ConcatBatchTensor


# Advisory `entity_kind` values. Used for logging / UI only; never branched on
# by tracking terms.
EntityKind = Literal[
    "robot",
    "articulated_object",
    "rigid_object",
    "human",
    "humanoid",
    "hand",
    "other",
]


def _check_body_model_fields(seq: MotionSequence | BatchMotionSequence) -> None:
    """Raise unless ``body_pose_quat``, ``body_shape`` and ``body_model`` are all set or all None."""
    body = [attr for attr in ("body_pose_quat", "body_shape", "body_model") if getattr(seq, attr) is not None]
    if body and len(body) != 3:
        raise ValueError(
            f"Body-model fields come together: body_pose_quat, body_shape and body_model are all set or all None; "
            f"got only {', '.join(body)}."
        )


@dataclass
class MotionSequence:
    """A standard motion reference buffer for a single motion sequence.

    Additional link poses are precomputed for faster access during simulation.
    Each motion has leading dims as [num_frames, ...] with no batch dimension.

    Parameters
    ----------
    framerate : torch.Tensor, optional
        The framerate of the motion data.
        Shape: [], a scalar tensor.
    buffer_length : torch.Tensor, optional
        The number of frames of the motion data.
        Shape: [], a scalar tensor.
    dofs_pos : torch.Tensor, optional
        Positions of the joints of the given robot model.
        Shape: [num_frames, num_dofs].
    dofs_vel : torch.Tensor, optional
        Velocities of the joints of the given robot model.
        Shape: [num_frames, num_dofs].
    base_pos_w : torch.Tensor, optional
        Positions of the base of the robot model.
        Shape: [num_frames, 3].
    base_lin_vel_w : torch.Tensor, optional
        Linear velocities of the base of the robot model.
        Shape: [num_frames, 3].
    base_quat_w : torch.Tensor, optional
        Quaternions of the base of the robot model.
        Shape: [num_frames, 4].
    base_ang_vel_w : torch.Tensor, optional
        Angular velocities of the base of the robot model.
        Shape: [num_frames, 3].
    links_pos_b : torch.Tensor, optional
        The positions of the interested links (w.r.t the base frame of the motion reference).
        NOTE: Derived from the world-frame base and link fields (see
        :func:`~eden.utils.motion_reference.utils.derive_body_frame_link_fields`), as are the other ``links_*_b``.
        Shape: [num_frames, num_links, 3].
    links_quat_b : torch.Tensor, optional
        The quaternions of the interested links (w.r.t the base).
        Shape: [num_frames, num_links, 4], in (w, x, y, z).
    links_pos_w : torch.Tensor, optional
        The positions of the interested links (w.r.t the world).
        NOTE: This is a passive parameter computed from base_pos, base_quat, and dofs_pos.
        Shape: [num_frames, num_links, 3].
    links_quat_w : torch.Tensor, optional
        The quaternions of the interested links (w.r.t the world).
        Shape: [num_frames, num_links, 4], in (w, x, y, z).
    links_lin_vel_b : torch.Tensor, optional
        The linear velocities of the interested links relative to the base, in the base frame.
        Shape: [num_frames, num_links, 3].
    links_ang_vel_b : torch.Tensor, optional
        The angular velocities of the interested links relative to the base, in the base frame.
        Shape: [num_frames, num_links, 3].
    links_lin_vel_w : torch.Tensor, optional
        The linear velocities of the interested links (w.r.t the world).
        NOTE: This is a passive parameter computed from base_lin_vel, base_quat, and dofs_vel.
        Shape: [num_frames, num_links, 3].
    links_ang_vel_w : torch.Tensor, optional
        The angular velocities of the interested links (w.r.t the world).
        NOTE: This is a passive parameter computed from base_ang_vel, base_quat, and dofs_vel.
        Shape: [num_frames, num_links, 3].
    contact_points_w : torch.Tensor, optional
        Demo contact points between this entity's links and a contact partner
        (w.r.t the world). Rows follow the dataset's ``motion_links_name`` order
        (same axis as the ``links_*`` fields). The partner axis
        (``num_contact_partners``, 1 by default) carries one column per object
        link for multi-part contact partners — e.g. an ARCTIC two-part box:
        [bottom, top], mirroring upstream DexMachina's per-part contact tensors.
        The column↔object-link correspondence is NOT tied to Genesis's internal
        link order: consumer terms declare it by name (the reward terms'
        ``object_links_name``, which reorders the policy contact axis to match),
        so a dataset packs the columns in whatever order it names them and the
        config pins that same order. A multi-partner reference consumed without
        such a declaration is a config error, not a silent misalignment.
        Shape: [num_frames, num_links, num_contact_partners, 3].
    contact_points_valid : torch.Tensor, optional
        Whether the (link, partner) pair is in contact at the frame; gates
        ``contact_points_w`` reads. dtype bool.
        Shape: [num_frames, num_links, num_contact_partners].

        Standalone — populated with ``contact_points_w=None`` and the partner
        axis collapsed to ``num_contact_partners=1`` — this is the supported
        *partner-less binary contact signal* ("is link L in contact at frame
        t", e.g. a foot touching the ground). All the reference-side plumbing
        (all-or-none packing, nearest-neighbor retiming, ``select_links``,
        buffer allocation, the ``motion_contact_flags`` observation) is
        agnostic to whether contact points or a tracked partner exist, so a
        locomotion dataset can emit foot-contact flags through this field with
        no contact-point data. Despite the name, ``valid`` IS the whole signal
        in this mode (there is no point whose validity it gates).

        TODO(loose-contact): the sim-side consumers (the contact rewards and
        ``GroupedContactFlags``) are all keyed to a *tracked partner entity*
        via ``get_grouped_contacts`` and the ``obj.num_links ==
        num_demo_partners`` pairing, which foot-ground contact does not have.
        Serving the binary case end-to-end needs one additive term (e.g. a
        reward comparing this field against a ``get_links_net_contact_force``
        norm threshold) that stays out of the partner-object code path.
    contact_label : torch.Tensor, optional
        Tri-state per-link contact annotation against the entity's interaction partner, dtype int8:
        ``+1`` the link should be in contact, ``-1`` it should not, ``0`` no label (ignored). Rows follow the
        dataset's ``motion_links_name`` order. Unlike ``contact_points_valid`` it has no partner axis, and it keeps
        the "must not touch" / "don't care" distinction a boolean flag collapses — what a contact-matching reward
        and a lost-contact termination need (e.g. Weave's G1-Inspire-HOI labels).
        Shape: [num_frames, num_links].
    body_pose_quat : torch.Tensor, optional
        Parametric body-model pose (MANO / SMPL family): the local rotation of every joint below the root, wxyz, in
        the model's joint order, relative to the model's rest pose (for MANO the flat hand, ``flat_hand_mean=True``;
        a release that stores poses relative to the mean hand has the mean added). The root is the sequence's base:
        ``base_pos_w`` is the root joint's position (not the model's ``transl``, which sits a shape-dependent offset
        away) and ``base_quat_w`` its global orientation. Opt in through ``DatasetBase.body_model_params``;
        :mod:`eden.datasets.utils.body_model` turns it back into joints and vertices.
        Shape: [num_frames, num_body_joints, 4].
    body_shape : torch.Tensor, optional
        The body model's shape coefficients (``betas``), per frame so a release with per-frame shapes keeps them.
        Present exactly when ``body_pose_quat`` is. Shape: [num_frames, num_betas].
    body_model : str, optional
        Which body model ``body_pose_quat`` / ``body_shape`` parameterise, e.g. ``"mano_right"``; see
        :data:`eden.datasets.utils.body_model.BODY_MODELS`. Set exactly when they are.
    """

    framerate: torch.Tensor | None = None
    buffer_length: torch.Tensor | None = None
    dofs_pos: torch.Tensor | None = None
    dofs_vel: torch.Tensor | None = None
    base_pos_w: torch.Tensor | None = None
    base_lin_vel_w: torch.Tensor | None = None
    base_quat_w: torch.Tensor | None = None
    base_ang_vel_w: torch.Tensor | None = None
    links_pos_b: torch.Tensor | None = None
    links_quat_b: torch.Tensor | None = None
    links_pos_w: torch.Tensor | None = None
    links_quat_w: torch.Tensor | None = None
    links_lin_vel_b: torch.Tensor | None = None
    links_ang_vel_b: torch.Tensor | None = None
    links_lin_vel_w: torch.Tensor | None = None
    links_ang_vel_w: torch.Tensor | None = None
    contact_points_w: torch.Tensor | None = None
    contact_points_valid: torch.Tensor | None = None
    contact_label: torch.Tensor | None = None
    body_pose_quat: torch.Tensor | None = None
    body_shape: torch.Tensor | None = None
    body_model: str | None = None

    attrs_with_frame_dim: ClassVar[tuple] = (
        "dofs_pos",
        "dofs_vel",
        "base_pos_w",
        "base_lin_vel_w",
        "base_quat_w",
        "base_ang_vel_w",
        "links_pos_b",
        "links_quat_b",
        "links_pos_w",
        "links_quat_w",
        "links_lin_vel_b",
        "links_ang_vel_b",
        "links_lin_vel_w",
        "links_ang_vel_w",
        "contact_points_w",
        "contact_points_valid",
        "contact_label",
        "body_pose_quat",
        "body_shape",
    )
    attrs_only_batch_dim: ClassVar[tuple] = ("framerate", "buffer_length")
    #: The opt-in parametric body-model fields: allocated by no ``make_empty``, all-or-none across a dataset.
    body_model_attrs: ClassVar[tuple] = ("body_pose_quat", "body_shape")

    @property
    def num_contact_partners(self) -> int:
        """Size of the contact-partner axis (1 when the sequence has no contact annotations)."""
        return self.contact_points_valid.shape[-1] if self.contact_points_valid is not None else 1

    @classmethod
    def make_empty(
        cls,
        num_frames: int,
        num_dofs: int,
        num_links: int,
        device=torch.device("cpu"),
        num_contact_partners: int = 1,
    ) -> "MotionSequence":
        seq = cls(
            framerate=torch.zeros((), dtype=torch.int, device=device),
            buffer_length=torch.tensor(num_frames, device=device),
            dofs_pos=torch.zeros(num_frames, num_dofs, device=device),
            dofs_vel=torch.zeros(num_frames, num_dofs, device=device),
            base_pos_w=torch.zeros(num_frames, 3, device=device),
            base_lin_vel_w=torch.zeros(num_frames, 3, device=device),
            base_quat_w=torch.zeros(num_frames, 4, device=device),
            base_ang_vel_w=torch.zeros(num_frames, 3, device=device),
            links_pos_b=torch.zeros(num_frames, num_links, 3, device=device),
            links_quat_b=torch.zeros(num_frames, num_links, 4, device=device),
            links_pos_w=torch.zeros(num_frames, num_links, 3, device=device),
            links_quat_w=torch.zeros(num_frames, num_links, 4, device=device),
            links_lin_vel_b=torch.zeros(num_frames, num_links, 3, device=device),
            links_ang_vel_b=torch.zeros(num_frames, num_links, 3, device=device),
            links_lin_vel_w=torch.zeros(num_frames, num_links, 3, device=device),
            links_ang_vel_w=torch.zeros(num_frames, num_links, 3, device=device),
            contact_points_w=torch.zeros(num_frames, num_links, num_contact_partners, 3, device=device),
            contact_points_valid=torch.zeros(
                num_frames, num_links, num_contact_partners, dtype=torch.bool, device=device
            ),
            contact_label=torch.zeros(num_frames, num_links, dtype=torch.int8, device=device),
        )
        seq.base_quat_w[:, 0] = 1.0
        seq.links_quat_b[:, :, 0] = 1.0
        seq.links_quat_w[:, :, 0] = 1.0
        return seq

    quat_attrs: ClassVar[tuple] = ("base_quat_w", "links_quat_b", "links_quat_w", "body_pose_quat")

    def __post_init__(self):
        # Ensure that all tensors with a frame dimension have the same leading size.
        # Use the first non-None attribute in `attrs_with_frame_dim` as the reference.
        reference_tensor = None
        reference_name = None
        for attr in self.attrs_with_frame_dim:
            value = getattr(self, attr)
            if value is not None:
                reference_tensor = value
                reference_name = attr
                break
        if reference_tensor is not None:
            ref_len = reference_tensor.shape[0]
            assert all(
                getattr(self, attr).shape[0] == ref_len
                for attr in self.attrs_with_frame_dim
                if getattr(self, attr) is not None
            ), (
                f"All attributes with a frame dimension should have the same leading "
                f"dimension as '{reference_name}' (size {ref_len})."
            )
        _check_body_model_fields(self)
        # Guarantee unit-norm quaternions so downstream consumers (tracking
        # rewards, ref-state observations, body-frame transforms) don't have
        # to defensively renormalize on every read. Loaders that already
        # ship unit quats pay one O(n) divide here; loaders that don't
        # produce a silent training bug if we skip this.
        for attr in self.quat_attrs:
            quat = getattr(self, attr)
            if quat is None:
                continue
            norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
            # Zero-rows are sentinels (e.g. unfilled make_empty defaults
            # before the w=1 patch); leave them as-is to avoid div-by-zero.
            safe = torch.where(norm > 0, norm, torch.ones_like(norm))
            setattr(self, attr, quat / safe)


#: The ``links_*_b`` fields :func:`~eden.utils.motion_reference.utils.derive_body_frame_link_fields` derives.
_BODY_FRAME_LINK_FIELDS = frozenset({"links_pos_b", "links_quat_b", "links_lin_vel_b", "links_ang_vel_b"})


@dataclass
class BatchMotionSequence:
    """A batched version of MotionSequence used for DatasetBase.

    Parameters
    ----------
    framerate : torch.Tensor, optional
        The framerate of the motion data.
        Shape: [N], N is the batch size.
    buffer_length : torch.Tensor, optional
        The number of frames of each motion data across batch.
        Shape: [N], N is the batch size.
    dofs_pos : ConcatBatchTensor, optional
        Positions of the joints of the given robot model.
    dofs_vel : ConcatBatchTensor, optional
        Velocities of the joints of the given robot model.
    base_pos_w : ConcatBatchTensor, optional
        Positions of the base of the robot model.
    base_lin_vel_w : ConcatBatchTensor, optional
        Linear velocities of the base of the robot model.
    base_quat_w : ConcatBatchTensor, optional
        Quaternions of the base of the robot model.
    base_ang_vel_w : ConcatBatchTensor, optional
        Angular velocities of the base of the robot model.
    links_pos_b, links_quat_b, links_lin_vel_b, links_ang_vel_b : ConcatBatchTensor, optional
        The body-frame link fields; see :class:`MotionSequence`.
    links_pos_w : ConcatBatchTensor, optional
        The positions of the interested links (w.r.t the world).
        NOTE: This is a passive parameter computed from base_pos, base_quat, and dofs_pos.
    links_quat_w : ConcatBatchTensor, optional
        The quaternions of the interested links (w.r.t the world).
        in (w, x, y, z).
    links_lin_vel_w : ConcatBatchTensor, optional
        The linear velocities of the interested links (w.r.t the world).
        NOTE: This is a passive parameter computed from base_lin_vel, base_quat, and dofs_vel.
    links_ang_vel_w : ConcatBatchTensor, optional
        The angular velocities of the interested links (w.r.t the world).
        NOTE: This is a passive parameter computed from base_ang_vel, base_quat, and dofs_vel.
    contact_points_w, contact_points_valid, contact_label : ConcatBatchTensor, optional
        The opt-in contact annotations; see :class:`MotionSequence`.
    body_pose_quat, body_shape : ConcatBatchTensor, optional
        The opt-in parametric body-model fields; see :class:`MotionSequence`.
    body_model : tuple[str, ...], optional
        The body model of each clip (``"mano_left"`` / ``"mano_right"`` can mix within one key), set exactly when the
        body-model fields are. Reindexed with the clips by :meth:`select`.
    """

    framerate: torch.Tensor | None = None
    buffer_length: torch.Tensor | None = None
    dofs_pos: ConcatBatchTensor | None = None
    dofs_vel: ConcatBatchTensor | None = None
    base_pos_w: ConcatBatchTensor | None = None
    base_lin_vel_w: ConcatBatchTensor | None = None
    base_quat_w: ConcatBatchTensor | None = None
    base_ang_vel_w: ConcatBatchTensor | None = None
    links_pos_b: ConcatBatchTensor | None = None
    links_quat_b: ConcatBatchTensor | None = None
    links_pos_w: ConcatBatchTensor | None = None
    links_quat_w: ConcatBatchTensor | None = None
    links_lin_vel_b: ConcatBatchTensor | None = None
    links_ang_vel_b: ConcatBatchTensor | None = None
    links_lin_vel_w: ConcatBatchTensor | None = None
    links_ang_vel_w: ConcatBatchTensor | None = None
    contact_points_w: ConcatBatchTensor | None = None
    contact_points_valid: ConcatBatchTensor | None = None
    contact_label: ConcatBatchTensor | None = None
    body_pose_quat: ConcatBatchTensor | None = None
    body_shape: ConcatBatchTensor | None = None
    body_model: tuple[str, ...] | None = None

    # Advisory metadata. Branched on by ``assert_motion_field_present`` (the
    # ``is_free_floating`` flag gates ``base_*`` field reads); otherwise used
    # for logging / UI / docs-driven consistency checks. Preserved across
    # ``select`` / ``select_dofs`` / ``select_links`` / ``to`` and round-tripped
    # through ``to_state_dict`` / ``from_state_dict`` (the mmap cache); a state
    # dict without them restores the defaults.
    is_free_floating: bool = True
    entity_kind: EntityKind | None = None

    attrs_with_frame_dim: ClassVar[tuple] = MotionSequence.attrs_with_frame_dim
    attrs_only_batch_dim: ClassVar[tuple] = MotionSequence.attrs_only_batch_dim
    body_model_attrs: ClassVar[tuple] = MotionSequence.body_model_attrs
    metadata_fields: ClassVar[tuple] = ("is_free_floating", "entity_kind")

    def __post_init__(self):
        _check_body_model_fields(self)
        if self.body_model is not None:
            if self.buffer_length is not None and len(self.body_model) != self.buffer_length.shape[0]:
                raise ValueError(
                    f"body_model names {len(self.body_model)} clip(s) but the sequence holds "
                    f"{self.buffer_length.shape[0]}; it is per clip."
                )

    @staticmethod
    def _infer_device(motion: MotionSequence) -> torch.device:
        for attr in MotionSequence.attrs_with_frame_dim:
            value = getattr(motion, attr)
            if isinstance(value, torch.Tensor):
                return value.device
        for attr in MotionSequence.attrs_only_batch_dim:
            value = getattr(motion, attr)
            if isinstance(value, torch.Tensor):
                return value.device
        return torch.device("cpu")

    @staticmethod
    def _infer_num_dofs(motion: MotionSequence) -> int:
        for attr in ("dofs_pos", "dofs_vel"):
            value = getattr(motion, attr)
            if isinstance(value, torch.Tensor):
                return int(value.shape[-1])
        return 0

    @staticmethod
    def _infer_num_links(motion: MotionSequence) -> int:
        for attr in (
            "links_pos_w",
            "links_quat_w",
            "links_pos_b",
            "links_quat_b",
            "links_lin_vel_w",
            "links_ang_vel_w",
            "links_lin_vel_b",
            "links_ang_vel_b",
        ):
            value = getattr(motion, attr)
            if isinstance(value, torch.Tensor):
                return int(value.shape[-2])
        return 0

    @property
    def num_contact_partners(self) -> int:
        """Size of the contact-partner axis (1 when the pack has no contact annotations)."""
        return self.contact_points_valid._data_shape[-1] if self.contact_points_valid is not None else 1

    @classmethod
    def _data_shape_for(cls, attr: str, num_dofs: int, num_links: int, num_contact_partners: int = 1) -> tuple:
        """Per-frame shape for an ``attrs_with_frame_dim`` field.

        Single source of truth used by both ``make_empty`` (allocator) and
        ``from_motion_sequences`` (absent-field fallback). Adding a new field
        means updating this table and ``attrs_with_frame_dim`` only.
        """
        if attr in ("dofs_pos", "dofs_vel"):
            return (num_dofs,)
        if attr in ("base_pos_w", "base_lin_vel_w", "base_ang_vel_w"):
            return (3,)
        if attr == "base_quat_w":
            return (4,)
        if attr in (
            "links_pos_b",
            "links_pos_w",
            "links_lin_vel_b",
            "links_lin_vel_w",
            "links_ang_vel_b",
            "links_ang_vel_w",
        ):
            return (num_links, 3)
        if attr in ("links_quat_b", "links_quat_w"):
            return (num_links, 4)
        if attr == "contact_points_w":
            return (num_links, num_contact_partners, 3)
        if attr == "contact_points_valid":
            return (num_links, num_contact_partners)
        if attr == "contact_label":
            return (num_links,)
        if attr in cls.body_model_attrs:
            raise ValueError(
                f"'{attr}' is an opt-in body-model field: its shape comes from the body model, never filled."
            )
        raise ValueError(f"Unknown attr: {attr}")

    @classmethod
    def _dtype_for(cls, attr: str) -> torch.dtype:
        """Per-field dtype for ``attrs_with_frame_dim`` (allocator counterpart of ``_data_shape_for``)."""
        if attr == "contact_points_valid":
            return torch.bool
        if attr == "contact_label":
            return torch.int8
        return torch.float32

    @classmethod
    def make_empty(
        cls,
        buffer_lengths: list[int],
        num_dofs: int,
        num_links: int,
        device=torch.device("cpu"),
        num_contact_partners: int = 1,
    ) -> "BatchMotionSequence":
        cbts: dict[str, ConcatBatchTensor] = {
            attr: ConcatBatchTensor(  # type: ignore[arg-type]
                batch_sizes=buffer_lengths,
                data_shape=cls._data_shape_for(attr, num_dofs, num_links, num_contact_partners),
                dtype=cls._dtype_for(attr),
                device=device,
            )
            for attr in cls.attrs_with_frame_dim
            if attr not in cls.body_model_attrs
        }
        seq = cls(
            framerate=torch.zeros((len(buffer_lengths),), device=device),
            buffer_length=torch.tensor(buffer_lengths, device=device),
            **cbts,
        )
        # Initialise quaternion fields to identity (w=1) so unfilled rows
        # represent valid no-rotation rather than zero-quaternions.
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        seq.base_quat_w.fill_data(identity_quat)  # type: ignore[union-attr]
        if num_links > 0:
            link_identity = identity_quat.unsqueeze(0).expand(num_links, -1)
            seq.links_quat_b.fill_data(link_identity)  # type: ignore[union-attr]
            seq.links_quat_w.fill_data(link_identity)  # type: ignore[union-attr]
        return seq

    def _metadata_kwargs(self) -> dict:
        """Advisory metadata kwargs to pass through ``select`` / ``to`` derivatives."""
        return {f: getattr(self, f) for f in self.metadata_fields}

    def select(self, batch_indices: torch.Tensor | list[int] | slice) -> "BatchMotionSequence":
        """Select a subset of motion sequences by batch index.

        Parameters
        ----------
        batch_indices : torch.Tensor | list[int] | slice
            The indices of the motion sequences to select.

        Returns
        -------
        BatchMotionSequence
            A new BatchMotionSequence containing only the selected sequences.
        """
        kwargs = {}
        for attr in self.attrs_only_batch_dim:
            val = getattr(self, attr)
            if val is not None:
                kwargs[attr] = val[batch_indices]
        for attr in self.attrs_with_frame_dim:
            val = getattr(self, attr)
            if val is not None:
                kwargs[attr] = val.select_batches(batch_indices)
        if self.body_model is not None:
            if isinstance(batch_indices, torch.Tensor):
                batch_indices = torch.arange(len(self.body_model))[batch_indices.cpu()].tolist()
            elif isinstance(batch_indices, slice):
                batch_indices = range(len(self.body_model))[batch_indices]
            kwargs["body_model"] = tuple(self.body_model[i] for i in batch_indices)
        return BatchMotionSequence(**kwargs, **self._metadata_kwargs())

    def select_dofs(self, dofs_idx: slice | list[int] | torch.Tensor) -> "BatchMotionSequence":
        """Select a subset of dofs for dofs_* fields, discarding the rest.

        Parameters
        ----------
        dofs_idx : slice | list[int] | torch.Tensor
            The dofs indices to keep.
        """
        return self._select_columns(dofs_idx, ("dofs_",))

    def select_links(self, links_idx: slice | list[int] | torch.Tensor) -> "BatchMotionSequence":
        """Select a subset of links for links_* fields, discarding the rest.

        Parameters
        ----------
        links_idx : slice | list[int] | torch.Tensor
            The links indices to keep.
        """
        # Contact rows share the links axis (dataset ``motion_links_name`` order), so they must be sliced together
        # with the ``links_*`` fields.
        return self._select_columns(links_idx, ("links_", "contact_"))

    def _select_columns(
        self, idx: slice | list[int] | torch.Tensor, prefixes: tuple[str, ...]
    ) -> "BatchMotionSequence":
        """A copy keeping only columns ``idx`` (the dim after the frame dim) of the fields named ``<prefix>*``.

        Every other field is shared with ``self``.
        """
        selected = {}
        for attr in self.attrs_with_frame_dim:
            val = getattr(self, attr)
            if val is not None and attr.startswith(prefixes):
                cols = idx.to(val.device) if isinstance(idx, torch.Tensor) else idx
                selected[attr] = ConcatBatchTensor.from_concatenated(val.concatenated_tensor[:, cols], val.batch_sizes)
        return replace(self, **selected)

    @classmethod
    def from_motion_sequences(
        cls,
        motion_list: list[MotionSequence],
        num_dofs: int | None = None,
        num_links: int | None = None,
    ) -> "BatchMotionSequence":
        # Vectorised pack: one torch.cat per field over the list (each motion stores its frames contiguously),
        # wrapped with ConcatBatchTensor.from_concatenated. A per-motion scatter loop is O(N_motions * N_fields)
        # Python work, which dominates the prepare step at ~73k motions.
        num_dofs = num_dofs or cls._infer_num_dofs(motion_list[0])
        num_links = num_links or cls._infer_num_links(motion_list[0])
        device = cls._infer_device(motion_list[0])

        buffer_lengths = [int(m.buffer_length) for m in motion_list]
        framerates = torch.stack(
            [
                m.framerate
                if isinstance(m.framerate, torch.Tensor)
                else torch.tensor(float(m.framerate), device=device)
                for m in motion_list
            ]
        )

        kwargs: dict = {
            "framerate": framerates,
            "buffer_length": torch.tensor(buffer_lengths, device=device),
        }
        for attr in cls.attrs_with_frame_dim:
            present = [getattr(m, attr) for m in motion_list]
            # Field is "present" only when every motion provides it; partial
            # presence is ambiguous (would need padding semantics).
            if any(p is None for p in present):
                # Opt-in fields stay None when absent: consumers gate on presence via assert_motion_field_present,
                # and zeros would both read as data (a zero quaternion included) and cost a library's worth of
                # memory. Contact annotations are opt-in, and so are the body-frame link fields beside world-frame
                # links (``_pack_motion_sequence(derive_body_frame=False)``). Partial presence is a dataset bug.
                if attr.startswith("contact_"):
                    opt_in, hint = "contact annotations", ""
                elif attr in cls.body_model_attrs:
                    opt_in, hint = "body-model parameters", ""
                elif attr in _BODY_FRAME_LINK_FIELDS and all(m.links_pos_w is not None for m in motion_list):
                    opt_in, hint = (
                        "body-frame link fields",
                        ": pack every clip with the same _pack_motion_sequence(derive_body_frame=...) setting",
                    )
                else:
                    opt_in = None
                if opt_in is not None:
                    if any(p is not None for p in present):
                        raise ValueError(
                            f"Field '{attr}' is provided by some motions but not all; {opt_in} must be all-or-none "
                            f"across a dataset{hint}."
                        )
                    continue
                # Allocate zeros for fields the source dataset doesn't populate
                # so downstream code that always reads (e.g. links_lin_vel_b
                # via _forward_velocity_kinematics) sees uniform shapes.
                empty_shape = cls._data_shape_for(attr, num_dofs, num_links)
                cat = torch.zeros((sum(buffer_lengths),) + empty_shape, device=device, dtype=cls._dtype_for(attr))
            else:
                cat = torch.cat(present, dim=0)
            kwargs[attr] = ConcatBatchTensor.from_concatenated(cat, batch_sizes=buffer_lengths, device=device)
        if kwargs.get("body_pose_quat") is not None:
            kwargs["body_model"] = tuple(m.body_model for m in motion_list)

        return cls(**kwargs)

    def to_state_dict(self) -> dict:
        """Serialize to a plain dict of tensors + scalar metadata for ``torch.save``.

        The output contains *no* custom objects, so the saved file can be
        ``torch.load(mmap=True)``-ed and read across multiple ranks without
        deserializing pickled Python objects. Restore via ``from_state_dict``.

        Compatibility is gated by the cache-level ``_format_version`` marker
        in ``DatasetBase``; this dict carries no separate version field.
        """
        sd: dict = {
            "_field_names": [],
            "framerate": self.framerate,
            "buffer_length": self.buffer_length,
        }
        for attr in self.attrs_with_frame_dim:
            cbt = getattr(self, attr)
            if cbt is None:
                continue
            sd["_field_names"].append(attr)
            sd[f"{attr}/concat"] = cbt.concatenated_tensor
            sd[f"{attr}/batch_sizes"] = cbt.batch_sizes
        for f in self.metadata_fields:
            sd[f"_metadata/{f}"] = getattr(self, f)
        if self.body_model is not None:
            sd["body_model"] = list(self.body_model)
        return sd

    @classmethod
    def from_state_dict(cls, sd: dict) -> "BatchMotionSequence":
        """Inverse of ``to_state_dict``. Tensors may be mmap'd CPU tensors."""
        device = sd["framerate"].device
        kwargs: dict = {
            "framerate": sd["framerate"],
            "buffer_length": sd["buffer_length"],
        }
        for attr in sd["_field_names"]:
            kwargs[attr] = ConcatBatchTensor.from_concatenated(
                concatenated_tensor=sd[f"{attr}/concat"],
                batch_sizes=sd[f"{attr}/batch_sizes"],
                device=device,
            )
        for f in cls.metadata_fields:
            key = f"_metadata/{f}"
            if key in sd:
                kwargs[f] = sd[key]
        if "body_model" in sd:
            kwargs["body_model"] = tuple(sd["body_model"])
        return cls(**kwargs)

    def to(self, device: torch.device | str) -> "BatchMotionSequence":
        """Move all tensors to the specified device."""
        device = torch.device(device)
        kwargs = {}
        for attr in self.attrs_with_frame_dim + self.attrs_only_batch_dim:
            val = getattr(self, attr)
            if val is not None:
                kwargs[attr] = val.to(device)
        return BatchMotionSequence(**kwargs, body_model=self.body_model, **self._metadata_kwargs())


@dataclass
class MotionReferenceData:
    """
    Each motion can have leading dims as [N, ...] or [...] with no batch dimension.

    Parameters
    ----------
    dofs_pos : torch.Tensor, optional
        Positions of the joints in urdf coordinate of the given robot model.
        Shape: [N, num_frames, num_dofs], N is the batch size.
    dofs_vel : torch.Tensor, optional
        Velocities of the joints of the given robot model.
        Shape: [N, num_frames, num_dofs], N is the batch size.
    dofs_pos_mask : torch.Tensor, optional
        Mask for the joint positions.
        Shape: [N, num_frames, num_dofs], N is the batch size.
    dofs_vel_mask : torch.Tensor, optional
        Mask for the joint velocities.
        Shape: [N, num_frames, num_dofs], N is the batch size.
    base_pos_w : torch.Tensor, optional
        Positions of the base of the robot model.
        Shape: [N, num_frames, 3], N is the batch size.
    base_lin_vel_w : torch.Tensor, optional
        Linear velocities of the base of the robot model.
        Shape: [N, num_frames, 3], N is the batch size.
    base_quat_w : torch.Tensor, optional
        Quaternions of the base of the robot model.
        Shape: [N, num_frames, 4], in (w, x, y, z), N is the batch size.
    base_ang_vel_w : torch.Tensor, optional
        Angular velocities of the base of the robot model.
        Shape: [N, num_frames, 3], N is the batch size.
    base_pos_plane_mask : torch.Tensor, optional
        Mask for the base position on the plane.
        Shape: [N, num_frames], N is the batch size for each term.
    base_pos_height_mask : torch.Tensor, optional
        Mask for the base position height.
        Shape: [N, num_frames], N is the batch size for each term.
    base_orientation_mask : torch.Tensor, optional
        Mask for the base orientation.
        Shape: [N, num_frames], N is the batch size for each term.
    base_heading_mask : torch.Tensor, optional
        Mask for the base heading.
        Shape: [N, num_frames], N is the batch size for each term.
    links_pos_b : torch.Tensor, optional
        The positions of the interested links (w.r.t the base when sensor refreshes).
        Shape: [N, num_frames, num_links, 3], N is the batch size.
    links_quat_b : torch.Tensor, optional
        The quaternions of the interested links (w.r.t the base when sensor refreshes).
        Shape: [N, num_frames, num_links, 4], in (w, x, y, z), N is the batch size.
    links_pos_w : torch.Tensor, optional
        The positions of the interested links (w.r.t the world).
        Shape: [N, num_frames, num_links, 3], N is the batch size.
    links_quat_w : torch.Tensor, optional
        The quaternions of the interested links (w.r.t the world).
        Shape: [N, num_frames, num_links, 4], in (w, x, y, z), N is the batch size.
    links_lin_vel_b : torch.Tensor, optional
        The linear velocities of the interested links (w.r.t the base when sensor refreshes).
        Shape: [N, num_frames, num_links, 3], N is the batch size.
    links_ang_vel_b : torch.Tensor, optional
        The angular velocities of the interested links (w.r.t the base when sensor refreshes).
        Shape: [N, num_frames, num_links, 3], N is the batch size.
    links_lin_vel_w : torch.Tensor, optional
        The linear velocities of the interested links (w.r.t the world).
        Shape: [N, num_frames, num_links, 3], N is the batch size.
    links_ang_vel_w : torch.Tensor, optional
        The angular velocities of the interested links (w.r.t the world).
        Shape: [N, num_frames, num_links, 3], N is the batch size.
    links_pos_mask : torch.Tensor, optional
        The mask for the link positions.
        Shape: [N, num_frames, num_links], N is the batch size.
    links_rot_mask : torch.Tensor, optional
        The mask for the link rotations.
        Shape: [N, num_frames, num_links], N is the batch size.
    contact_points_w : torch.Tensor, optional
        Demo contact points between the entity's links and its contact partner
        (w.r.t the world; env origins applied). Rows follow the motion links axis;
        the partner axis follows the dataset (one column per object link for
        multi-part partners, 1 otherwise).
        Shape: [N, num_frames, num_links, num_contact_partners, 3], N is the batch size.
    contact_points_valid : torch.Tensor, optional
        Whether the (link, partner) pair is in contact; gates ``contact_points_w``.
        Shape: [N, num_frames, num_links, num_contact_partners], N is the batch size.
    contact_label : torch.Tensor, optional
        Tri-state per-link contact label (int8: +1 contact, -1 no contact, 0 unlabelled); stays all-zero when the
        dataset carries none. Shape: [N, num_frames, num_links], N is the batch size.
    time_to_target_frame : torch.Tensor, optional
        The time expected to reach the specific frame.
        Shape: [N, num_frames], N is the batch size.
    validity : torch.Tensor, optional
        A boolean tensor indicating whether the current data is valid or not.
        Usually indicating the data buffer is exhausted or not.
        Shape: [N, num_frames], N is the batch size.
    """

    dofs_pos: torch.Tensor | None = None
    dofs_vel: torch.Tensor | None = None
    dofs_pos_mask: torch.Tensor | None = None
    dofs_vel_mask: torch.Tensor | None = None
    base_pos_w: torch.Tensor | None = None
    base_lin_vel_w: torch.Tensor | None = None
    base_quat_w: torch.Tensor | None = None
    base_ang_vel_w: torch.Tensor | None = None
    base_pos_plane_mask: torch.Tensor | None = None
    base_pos_height_mask: torch.Tensor | None = None
    base_orientation_mask: torch.Tensor | None = None
    base_heading_mask: torch.Tensor | None = None
    links_pos_b: torch.Tensor | None = None
    links_quat_b: torch.Tensor | None = None
    links_pos_w: torch.Tensor | None = None
    links_quat_w: torch.Tensor | None = None
    links_lin_vel_b: torch.Tensor | None = None
    links_ang_vel_b: torch.Tensor | None = None
    links_lin_vel_w: torch.Tensor | None = None
    links_ang_vel_w: torch.Tensor | None = None
    links_pos_mask: torch.Tensor | None = None
    links_rot_mask: torch.Tensor | None = None
    contact_points_w: torch.Tensor | None = None
    contact_points_valid: torch.Tensor | None = None
    contact_label: torch.Tensor | None = None
    time_to_target_frame: torch.Tensor | None = None
    validity: torch.Tensor | None = None

    @classmethod
    def make_empty(
        cls,
        num_envs: int,
        num_frames: int,
        num_dofs: int,
        num_links: int,
        device=torch.device("cpu"),
        num_contact_partners: int = 1,
    ) -> "MotionReferenceData":
        data = cls(
            dofs_pos=torch.zeros(num_envs, num_frames, num_dofs, device=device),
            dofs_vel=torch.zeros(num_envs, num_frames, num_dofs, device=device),
            dofs_pos_mask=torch.ones(num_envs, num_frames, num_dofs, device=device, dtype=torch.bool),
            dofs_vel_mask=torch.ones(num_envs, num_frames, num_dofs, device=device, dtype=torch.bool),
            base_pos_w=torch.zeros(num_envs, num_frames, 3, device=device),
            base_lin_vel_w=torch.zeros(num_envs, num_frames, 3, device=device),
            base_quat_w=torch.zeros(num_envs, num_frames, 4, device=device),
            base_ang_vel_w=torch.zeros(num_envs, num_frames, 3, device=device),
            base_pos_plane_mask=torch.ones(num_envs, num_frames, device=device, dtype=torch.bool),
            base_pos_height_mask=torch.ones(num_envs, num_frames, device=device, dtype=torch.bool),
            base_orientation_mask=torch.ones(num_envs, num_frames, device=device, dtype=torch.bool),
            base_heading_mask=torch.ones(num_envs, num_frames, device=device, dtype=torch.bool),
            links_pos_b=torch.zeros(num_envs, num_frames, num_links, 3, device=device),
            links_quat_b=torch.zeros(num_envs, num_frames, num_links, 4, device=device),
            links_pos_w=torch.zeros(num_envs, num_frames, num_links, 3, device=device),
            links_quat_w=torch.zeros(num_envs, num_frames, num_links, 4, device=device),
            links_lin_vel_b=torch.zeros(num_envs, num_frames, num_links, 3, device=device),
            links_ang_vel_b=torch.zeros(num_envs, num_frames, num_links, 3, device=device),
            links_lin_vel_w=torch.zeros(num_envs, num_frames, num_links, 3, device=device),
            links_ang_vel_w=torch.zeros(num_envs, num_frames, num_links, 3, device=device),
            links_pos_mask=torch.ones(num_envs, num_frames, num_links, device=device, dtype=torch.bool),
            links_rot_mask=torch.ones(num_envs, num_frames, num_links, device=device, dtype=torch.bool),
            contact_points_w=torch.zeros(num_envs, num_frames, num_links, num_contact_partners, 3, device=device),
            contact_points_valid=torch.zeros(
                num_envs, num_frames, num_links, num_contact_partners, dtype=torch.bool, device=device
            ),
            contact_label=torch.zeros(num_envs, num_frames, num_links, dtype=torch.int8, device=device),
            time_to_target_frame=torch.zeros(num_envs, num_frames, device=device),
            validity=torch.zeros(num_envs, num_frames, dtype=torch.bool, device=device),
        )
        data.base_quat_w[:, :, 0] = 1.0
        data.links_quat_b[:, :, :, 0] = 1.0
        data.links_quat_w[:, :, :, 0] = 1.0
        return data


@dataclass
class MotionReferenceState:
    """
    The robot state at the beginning of the motion reference. For the purpose of reset.

    Parameters
    ----------
    dofs_pos : torch.Tensor, optional
        Joint positions.
    dofs_vel : torch.Tensor, optional
        Joint velocities.
    base_pos_w : torch.Tensor, optional
        Base position in world frame.
    base_quat_w : torch.Tensor, optional
        Base quaternion in world frame.
    base_lin_vel_w : torch.Tensor, optional
        Base linear velocity in world frame.
    base_ang_vel_w : torch.Tensor, optional
        Base angular velocity in world frame.
    """

    dofs_pos: torch.Tensor | None = None
    dofs_vel: torch.Tensor | None = None
    base_pos_w: torch.Tensor | None = None
    base_quat_w: torch.Tensor | None = None
    base_lin_vel_w: torch.Tensor | None = None
    base_ang_vel_w: torch.Tensor | None = None

    @classmethod
    def make_empty(
        cls,
        num_envs: int,
        num_dofs: int,
        device=torch.device("cpu"),
    ) -> "MotionReferenceState":
        return_ = cls(
            dofs_pos=torch.zeros(num_envs, num_dofs, device=device),
            dofs_vel=torch.zeros(num_envs, num_dofs, device=device),
            base_pos_w=torch.zeros(num_envs, 3, device=device),
            base_quat_w=torch.zeros(num_envs, 4, device=device),
            base_lin_vel_w=torch.zeros(num_envs, 3, device=device),
            base_ang_vel_w=torch.zeros(num_envs, 3, device=device),
        )
        return_.base_quat_w[:, 0] = 1.0
        return return_

    def __post_init__(self):
        if self.dofs_pos is not None:
            assert self.dofs_pos.shape == self.dofs_vel.shape, "The shape of dofs_pos and dofs_vel should be the same."
            assert self.dofs_pos.shape[0] == self.base_pos_w.shape[0] == self.base_quat_w.shape[0], (
                "The first dimension of dofs_pos, base_pos, base_quat, and base_lin_vel should be the same."
            )
        else:
            assert (
                self.dofs_pos is None
                and self.dofs_vel is None
                and self.base_pos_w is None
                and self.base_quat_w is None
                and self.base_lin_vel_w is None
                and self.base_ang_vel_w is None
            ), "All attributes should be None or not None."

    def __getitem__(self, idx: slice | torch.Tensor | None):
        if idx is None:
            return self

        return MotionReferenceState(
            dofs_pos=self.dofs_pos[idx],
            dofs_vel=self.dofs_vel[idx],
            base_pos_w=self.base_pos_w[idx],
            base_quat_w=self.base_quat_w[idx],
            base_lin_vel_w=self.base_lin_vel_w[idx],
            base_ang_vel_w=self.base_ang_vel_w[idx],
        )
