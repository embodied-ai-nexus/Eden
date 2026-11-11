"""Ground-the-first-frame dataset modifier.

For each motion sequence, set the robot to frame-0's qpos in a transient
Genesis env, read the world-frame AABB's bottom-z, and (if below the
configured margin) shift the entire clip uniformly along +Z so the first
frame sits at-or-above the floor. The shift is a pure translation, so all
velocity and body-frame fields stay invariant — same idiom as
:class:`~eden.datasets.modifiers.transforms.Translate`, only the offset
is auto-computed per clip from FK instead of being configured up front.

Use this when retargeting can leave clips penetrating the ground at clip
start — e.g. AMASS clips retargeted to a robot whose feet sit lower than
the source actor, or any pipeline where the source capture's floor plane
isn't perfectly aligned with z=0.

It shares the transient-env plumbing (:func:`eden.utils.genesis_runtime.transient_robot_env`)
with :class:`~eden.datasets.modifiers.smooth_start_end.SmoothStartTransition`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import eden as en
from eden.datasets.modifiers.base import DATASET_MODIFIER_REGISTRY, DatasetModifier
from eden.options.entities import EntityOptions
from eden.utils.genesis_runtime import transient_robot_env

if TYPE_CHECKING:
    from eden.utils.motion_reference.motion_data import BatchMotionSequence


@DATASET_MODIFIER_REGISTRY.register()
class GroundFirstFrame(DatasetModifier):
    """Shift each clip along +Z so frame-0's world AABB sits at-or-above z=0.

    Pipeline per clip:

    1. Build the robot from ``robot_options`` in a transient kinematics
       env (one env per clip — frame-0 FK runs in a single batched pass).
    2. ``set_qpos`` to ``(base_pos_w[0], base_quat_w[0], dofs_pos[0])``.
    3. Read ``robot.get_AABB()[:, 0, 2]`` per env → per-clip floor min-z.
    4. ``dz = max(0, safety_margin_m - min_z)``. Clips already above the
       floor get ``dz=0`` and pass through unchanged.
    5. Add ``dz`` to every frame of ``base_pos_w`` (and ``links_pos_w``
       when populated). Body-frame link fields and all velocity fields
       are invariant under pure translation and are left untouched.

    Parameters
    ----------
    robot_options : EntityOptions
        Eden robot options used to build the transient kinematics env.
        Must match the robot the dataset is retargeted to (same DOF order,
        same morph). Required.
    safety_margin_m : float
        Extra clearance above z=0 to enforce. ``dz`` is computed as
        ``max(0, safety_margin_m - aabb_min_z)``, so a positive value
        nudges the foot off the floor; ``0.0`` (default) lets the foot
        sit exactly on the plane.
    chunk_size : int | None
        Maximum number of clips processed per Genesis env. ``None``
        (default) allocates one env per clip in a single batched FK pass
        — fastest for LAFAN1-class libraries (~75 clips). For
        SEED-class libraries (100k+ clips) one-env-per-clip would OOM;
        set this to a few hundred / few thousand to bound the env count.
        The transient Genesis env is built **once** at ``chunk_size``
        and re-used across chunks via repeated ``set_qpos`` — only the
        last chunk's qpos is padded with the previous chunk's last
        entry to fill the env count (cheap extra FK on a few unused
        envs).

    Notes
    -----
    Multi-entity datasets: this modifier shifts a single robot key in
    place. If the dataset carries paired entities (robot + object) that
    must move together, scope the modifier to the robot key with
    ``motion_sequence_keys=(robot_key,)`` and add a matching shift on
    the object key — translating the robot alone would desync them.
    """

    # ``= None`` is a framework requirement: ``ConfigurableMixin.get_parameter_names``
    # discovers fields via class ``__dict__``, so a value-less typed-only field
    # vanishes from the modifier's parameter list. Misuse surfaces inside
    # ``transient_robot_env`` rather than at configure time.
    robot_options: EntityOptions = None
    safety_margin_m: float = 0.0
    chunk_size: int | None = None

    def apply(self, motion_sequences):
        if self.safety_margin_m < 0:
            raise ValueError(
                f"GroundFirstFrame: safety_margin_m must be >= 0 (clearance above the floor); "
                f"got {self.safety_margin_m}."
            )
        if self.chunk_size is not None and self.chunk_size < 1:
            raise ValueError(
                f"GroundFirstFrame: chunk_size must be >= 1 (or None for no chunking); got {self.chunk_size}."
            )

        out = dict(motion_sequences)
        for key in self._selected_keys(motion_sequences, single=True):
            out[key] = self._apply_to_key(out[key], key)
        return out

    def _apply_to_key(self, motion_seq: "BatchMotionSequence", key: str) -> "BatchMotionSequence":
        num_seqs = motion_seq.buffer_length.shape[0]
        # Chunk size: cap at num_seqs so we never allocate more envs than
        # we have clips. ``None`` means "one chunk, all clips" — the
        # default fast path for LAFAN1-class libraries.
        chunk_size = min(self.chunk_size, num_seqs) if self.chunk_size is not None else num_seqs
        starts = motion_seq.base_pos_w.batch_starts
        dz_chunks: list[torch.Tensor] = []

        # Build the Genesis env **once** at ``chunk_size`` and re-use it
        # across chunks — cheaper than rebuilding (kernel compile cost).
        with transient_robot_env(self.robot_options, num_envs=chunk_size) as (_env, robot):
            device = robot.device

            for chunk_start in range(0, num_seqs, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_seqs)
                chunk_len = chunk_end - chunk_start

                # Each clip's frame 0 lives at ``batch_starts[i]`` in the
                # flat buffer — gather this chunk in one fancy-index op.
                chunk_starts = starts[chunk_start:chunk_end]
                base_pos_0 = motion_seq.base_pos_w.concatenated_tensor[chunk_starts].to(
                    device=device, dtype=torch.float32
                )
                base_quat_0 = motion_seq.base_quat_w.concatenated_tensor[chunk_starts].to(
                    device=device, dtype=torch.float32
                )
                dofs_pos_0 = motion_seq.dofs_pos.concatenated_tensor[chunk_starts].to(
                    device=device, dtype=torch.float32
                )
                qpos = torch.cat([base_pos_0, base_quat_0, dofs_pos_0], dim=1)

                # Last chunk may be shorter than ``chunk_size``; pad qpos
                # with a duplicate of its last row so we always populate
                # every env (cheap extra FK on the padding envs, simpler
                # than ``envs_idx``-juggling).
                if chunk_len < chunk_size:
                    pad = chunk_size - chunk_len
                    qpos = torch.cat([qpos, qpos[-1:].expand(pad, -1)], dim=0)

                robot.set_qpos(qpos, zero_velocity=False, skip_forward=False)
                aabb = robot.get_AABB()  # (chunk_size, 2, 3) — corners (min, max)
                min_z = aabb[:chunk_len, 0, 2]  # drop padding entries
                dz_chunks.append(torch.clamp(self.safety_margin_m - min_z, min=0.0))

        dz = torch.cat(dz_chunks, dim=0)  # (num_seqs,)

        cbt_device = motion_seq.base_pos_w.device
        cbt_dtype = motion_seq.base_pos_w.dtype
        dz_dev = dz.to(device=cbt_device, dtype=cbt_dtype)
        sizes = motion_seq.base_pos_w.batch_sizes
        # Expand the per-clip dz to a flat (total_frames,) buffer matching
        # the ConcatBatchTensor layout, then in-place add to the z column.
        expanded = dz_dev.repeat_interleave(sizes)

        motion_seq.base_pos_w.concatenated_tensor[:, 2].add_(expanded)
        if motion_seq.links_pos_w is not None:
            # (total_frames, num_links, 3) — broadcast across the links dim.
            motion_seq.links_pos_w.concatenated_tensor[:, :, 2].add_(expanded.unsqueeze(-1))

        num_shifted = int((dz > 0).sum().item())
        max_dz = float(dz.max().item())
        en.logger.info(
            f"GroundFirstFrame: shifted {num_shifted}/{num_seqs} clip(s) of '{key}' "
            f"(max dz = {max_dz:.4f} m, safety_margin = {self.safety_margin_m:+.4f} m)"
        )
        return motion_seq
