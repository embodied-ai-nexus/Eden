"""Per-motion tracking-error metrics (MPJPE / G-MPJPE / root error).

Split out of ``MotionTerm``: a self-contained collaborator owning the
metric buffers + link-index cache and the record → scatter → drain → summary
pipeline. One instance per opted-in term, created in
``MotionTerm.set_envs_idx_assignments`` when ``compute_per_motion_metrics`` is
set; it reads the term's env-assignment / DDP id state through a back-reference
(the term stays the owner of that state).

Pipeline (and why each stage runs when it does):

* ``record_step`` — every ``compute()`` step: per-env error vs the reference,
  added into the per-env running sums. No CPU↔GPU sync.
* ``scatter_at_reset`` — at episode end: mean each finished env's sums over its
  episode and scatter into the per-motion accumulators, sized to the GLOBAL
  motion-id space so DDP all_reduce stays shape-symmetric (mirrors the
  curriculum counts).
* ``drain`` — once per curriculum drain: reduce across DDP and cache the
  aggregate summary surfaced via ``MotionManager._collect_log_extras``. Shares
  the curriculum's ``all_reduce_fn`` so it pays one extra collective.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.utils.misc import env_indices

if TYPE_CHECKING:
    from eden.entities.base import Entity
    from eden.managers.motion_manager import MotionTerm
    from eden.utils.motion_reference.motion_data import MotionReferenceData


class PerMotionMetrics:
    """Owns the per-motion tracking-error buffers and their accumulation pipeline."""

    def __init__(self, term: MotionTerm) -> None:
        self._term = term
        n_envs = term.num_assigned_envs
        n_motions = term._global_num_motions
        device = term.device
        # Lazily-resolved link-index caches; populated on the first ``record_step``.
        self._metric_robot_link_idx: torch.Tensor | None = None
        self._metric_motion_link_idx: torch.Tensor | None = None
        self._metric_robot_anchor_idx: torch.Tensor | None = None
        self._metric_motion_anchor_idx: torch.Tensor | None = None
        # Aggregate summary cache; refreshed only on drain.
        self._last_metrics_summary: dict[str, float] = {}
        # Per-env buffers are sized to the rank's local assigned slice; per-motion
        # buffers to the GLOBAL motion-id space (DDP all_reduce shape-symmetry).
        self._per_env_mpjpe_sum = torch.zeros(n_envs, dtype=torch.float, device=device)
        self._per_env_root_pos_err_sum = torch.zeros(n_envs, dtype=torch.float, device=device)
        self._per_env_root_quat_err_sum = torch.zeros(n_envs, dtype=torch.float, device=device)
        self._motion_mpjpe_sum = torch.zeros(n_motions, dtype=torch.float, device=device)
        self._motion_root_pos_err_sum = torch.zeros(n_motions, dtype=torch.float, device=device)
        self._motion_root_quat_err_sum = torch.zeros(n_motions, dtype=torch.float, device=device)
        self._motion_metric_episode_count = torch.zeros(n_motions, dtype=torch.long, device=device)
        # G-MPJPE buffers exist iff an anchor is configured; a single ``is None``
        # check on ``_per_env_g_mpjpe_sum`` gates the path.
        self._per_env_g_mpjpe_sum: torch.Tensor | None = None
        self._motion_g_mpjpe_sum: torch.Tensor | None = None
        if term.metric_anchor_link_name is not None:
            self._per_env_g_mpjpe_sum = torch.zeros(n_envs, dtype=torch.float, device=device)
            self._motion_g_mpjpe_sum = torch.zeros(n_motions, dtype=torch.float, device=device)

    def record_step(self, entity: Entity, ref: MotionReferenceData, env_name: str) -> None:
        """Compute one step of per-env tracking error for ``entity`` and add it
        into the running sums.

        Per-entity no-op when the reference lacks world-frame link/root fields
        (e.g. dof-only motion data) or the term tracks no links for this entity.
        A ``get_links_pos`` (two with an anchor) + one ``get_pos``/``get_quat``
        and per-env reductions, all on the GPU with no sync.
        """
        term = self._term
        if ref.links_pos_w is None or ref.base_pos_w is None or ref.base_quat_w is None:
            return
        motion_key = term.entities_map.get(env_name, env_name)
        links_name = term.target_links_name.get(motion_key, [])
        if not links_name:
            return

        # Lazy-cache link indices on first call. Motion-side uses a contiguous
        # arange because ``data_source.select_links`` already reordered the ref
        # tensor to match ``target_links_name``.
        if self._metric_robot_link_idx is None:
            _, robot_idx = entity.find_named_links_idx_local(links_name, preserve_order=True)
            self._metric_robot_link_idx = torch.as_tensor(robot_idx, device=term.device, dtype=torch.long)
            self._metric_motion_link_idx = torch.arange(len(links_name), device=term.device, dtype=torch.long)
            if term.metric_anchor_link_name is not None:
                anchor_name = term.metric_anchor_link_name
                if anchor_name not in links_name:
                    raise ValueError(
                        f"metric_anchor_link_name={anchor_name!r} must be present in "
                        f"target_links_name for entity {env_name!r}; got {links_name}"
                    )
                _, robot_anchor = entity.find_named_links_idx_local([anchor_name], preserve_order=True)
                self._metric_robot_anchor_idx = torch.as_tensor(robot_anchor, device=term.device, dtype=torch.long)
                self._metric_motion_anchor_idx = torch.tensor(
                    [links_name.index(anchor_name)], device=term.device, dtype=torch.long
                )

        # World-frame link MPJPE.
        robot_links = entity.get_links_pos(ls_idx_local=self._metric_robot_link_idx)
        motion_links = ref.links_pos_w[:, 0].index_select(1, self._metric_motion_link_idx)
        mpjpe = torch.linalg.vector_norm(robot_links - motion_links, dim=-1).mean(dim=-1)

        # G-MPJPE: link MPJPE relative to the anchor link's position — removes
        # the anchor's translation error (heading error still counts). Only
        # computed when an anchor is configured.
        g_mpjpe = None
        if self._metric_robot_anchor_idx is not None:
            robot_anchor_pos = entity.get_links_pos(ls_idx_local=self._metric_robot_anchor_idx)
            motion_anchor_pos = ref.links_pos_w[:, 0].index_select(1, self._metric_motion_anchor_idx)
            g_mpjpe = torch.linalg.vector_norm(
                (robot_links - robot_anchor_pos) - (motion_links - motion_anchor_pos),
                dim=-1,
            ).mean(dim=-1)

        # Root error: L2 distance + quat geodesic angle 2*acos(|q1·q2|).
        # ``clamp`` guards the |dot|>1 overshoot from float roundoff.
        root_pos_err = torch.linalg.vector_norm(entity.get_pos() - ref.base_pos_w[:, 0], dim=-1)
        q_dot = (entity.get_quat() * ref.base_quat_w[:, 0]).sum(dim=-1).abs().clamp(max=1.0)
        root_quat_err = 2.0 * torch.acos(q_dot)

        sl = term.assigned_env_slice
        self.accumulate(
            mpjpe=mpjpe[sl],
            root_pos_err=root_pos_err[sl],
            root_quat_err=root_quat_err[sl],
            g_mpjpe=g_mpjpe[sl] if g_mpjpe is not None else None,
        )

    def accumulate(
        self,
        mpjpe: torch.Tensor,
        root_pos_err: torch.Tensor,
        root_quat_err: torch.Tensor,
        g_mpjpe: torch.Tensor | None = None,
    ) -> None:
        """Add one step of per-env error into the running sums.

        Inputs are ``[num_assigned_envs]`` float tensors on the term's device.
        ``g_mpjpe`` must be passed iff ``metric_anchor_link_name`` is set on the
        term — a mismatch is a config bug and raises.
        """
        self._per_env_mpjpe_sum += mpjpe
        self._per_env_root_pos_err_sum += root_pos_err
        self._per_env_root_quat_err_sum += root_quat_err
        if (g_mpjpe is None) != (self._per_env_g_mpjpe_sum is None):
            raise ValueError(
                "g_mpjpe / _per_env_g_mpjpe_sum mismatch: pass g_mpjpe iff metric_anchor_link_name is set on the term."
            )
        if g_mpjpe is not None:
            self._per_env_g_mpjpe_sum += g_mpjpe

    def scatter_at_reset(self, envs_idx: torch.Tensor, episode_lengths: torch.Tensor) -> None:
        """Drain per-env running sums into the per-motion accumulators at episode end.

        ``episode_lengths`` is the full-size per-env ``episode_length_buf``
        (``[num_envs]``); we slice it to this term's ``assigned_env_slice`` FIRST
        so all downstream indexing operates in the LOCAL assigned-env space —
        ``envs_idx_to_assigned_ids`` returns local-space indices, so indexing the
        global ``episode_lengths`` with them silently reads the wrong env's
        length when ``assigned_env_slice.start > 0``.

        Supports both int-index and bool-mask ``envs_idx`` (the contract of
        ``record_motion_outcomes``). Envs at their bootstrap reset (length 0) are
        filtered out — they have no real outcome. Maps motion ids through
        ``_local_to_global`` so per-rank contributions land in shape-compatible
        tensors for the downstream DDP ``all_reduce``.
        """
        term = self._term
        episode_lengths_local = episode_lengths[term.assigned_env_slice.start : term.assigned_env_slice.stop]

        assigned = term.envs_idx_to_assigned_ids(envs_idx)
        # For bool input ``assigned`` is a fixed-length mask (len always
        # num_assigned_envs), so the empty-tensor shortcut only applies to int input.
        if not assigned.dtype == torch.bool and len(assigned) == 0:
            return

        if assigned.dtype == torch.bool:
            assigned = env_indices(assigned)

        # Only envs that actually ran. ``episode_length_buf == 0`` is a bootstrap
        # reset — no real data to record. Selected with `torch.where` / 0-1 weights over every resetting
        # env rather than gathered: an `.any()` gate and a bool-mask gather are a GPU->CPU sync each, and
        # the counts and the zero contributions it scatters are exact.
        active_lengths = episode_lengths_local[assigned]
        ran_mask = active_lengths > 0
        lengths_f = active_lengths.clamp(min=1).to(self._per_env_mpjpe_sum.dtype)

        def episode_mean(per_env_sum: torch.Tensor) -> torch.Tensor:
            return torch.where(ran_mask, per_env_sum[assigned] / lengths_f, 0.0)

        def drain_ran(per_env_sum: torch.Tensor) -> None:
            per_env_sum[assigned] = torch.where(ran_mask, 0.0, per_env_sum[assigned])

        # Episode means (per active assigned env).
        mpjpe_mean = episode_mean(self._per_env_mpjpe_sum)
        root_pos_mean = episode_mean(self._per_env_root_pos_err_sum)
        root_quat_mean = episode_mean(self._per_env_root_quat_err_sum)

        motion_ids_local = term._assigned_env_motion_selection[assigned]
        motion_ids_global = term._local_to_global[motion_ids_local]

        self._motion_mpjpe_sum.scatter_add_(0, motion_ids_global, mpjpe_mean)
        self._motion_root_pos_err_sum.scatter_add_(0, motion_ids_global, root_pos_mean)
        self._motion_root_quat_err_sum.scatter_add_(0, motion_ids_global, root_quat_mean)
        self._motion_metric_episode_count.scatter_add_(
            0, motion_ids_global, ran_mask.to(self._motion_metric_episode_count.dtype)
        )

        # Zero per-env running sums for envs we just drained.
        drain_ran(self._per_env_mpjpe_sum)
        drain_ran(self._per_env_root_pos_err_sum)
        drain_ran(self._per_env_root_quat_err_sum)

        # G-MPJPE scatter is conditional on the anchor being configured.
        if self._per_env_g_mpjpe_sum is not None:
            self._motion_g_mpjpe_sum.scatter_add_(0, motion_ids_global, episode_mean(self._per_env_g_mpjpe_sum))
            drain_ran(self._per_env_g_mpjpe_sum)

    def drain(self, all_reduce_fn=None) -> None:
        """Drain per-motion accumulators, reduce across DDP, refresh summary cache.

        Called from ``MotionTerm._update_motion_weight_curriculum`` right after
        the success/failure drain so both share one DDP cadence. After this
        returns, ``summary()`` holds the aggregate scalars surfaced via
        ``MotionManager._collect_log_extras``.
        """
        sums = {
            "mpjpe": self._motion_mpjpe_sum.clone(),
            "root_pos_err": self._motion_root_pos_err_sum.clone(),
            "root_quat_err": self._motion_root_quat_err_sum.clone(),
        }
        if self._motion_g_mpjpe_sum is not None:
            sums["g_mpjpe"] = self._motion_g_mpjpe_sum.clone()
            self._motion_g_mpjpe_sum.zero_()
        counts = self._motion_metric_episode_count.clone()
        self._motion_mpjpe_sum.zero_()
        self._motion_root_pos_err_sum.zero_()
        self._motion_root_quat_err_sum.zero_()
        self._motion_metric_episode_count.zero_()

        if all_reduce_fn is not None:
            for t in sums.values():
                all_reduce_fn(t)
            all_reduce_fn(counts)

        self._record_summary(sums, counts)

    def _record_summary(self, sums: dict[str, torch.Tensor], counts: torch.Tensor) -> None:
        """Stash per-motion aggregate quantiles for ``_collect_log_extras``.

        Motions with no episodes in the drain window are excluded from the
        quantiles. All four scalars per metric are pulled with a single
        ``.tolist()`` to keep the GPU→CPU sync count at one per metric.
        """
        has_data = counts > 0
        n_with_data = int(has_data.sum().item())
        if n_with_data == 0:
            self._last_metrics_summary = {"Mimic/motions_with_data": 0.0}
            return

        safe_counts = counts.clamp_min(1).to(sums["mpjpe"].dtype)
        summary: dict[str, float] = {"Mimic/motions_with_data": float(n_with_data)}
        for name, total in sums.items():
            per_motion = (total / safe_counts)[has_data]
            stats = torch.stack(
                [
                    per_motion.mean(),
                    torch.quantile(per_motion, 0.10),
                    torch.quantile(per_motion, 0.50),
                    torch.quantile(per_motion, 0.95),
                ]
            ).tolist()
            mean, p10, p50, p95 = stats
            summary[f"Mimic/{name}_mean"] = mean
            summary[f"Mimic/{name}_p10"] = p10
            summary[f"Mimic/{name}_p50"] = p50
            summary[f"Mimic/{name}_p95"] = p95
        self._last_metrics_summary = summary

    def summary(self) -> dict[str, float]:
        """Aggregate scalar summary cached at the last ``drain`` (empty before then)."""
        return self._last_metrics_summary

    def per_motion_tensors(self) -> dict[str, torch.Tensor]:
        """Per-motion-id tensors (CPU dict views) suitable for CSV dump.

        Returns the raw per-motion sum tensors *and* the episode count; callers
        compute the per-motion mean as ``sum / count.clamp_min(1)`` once they
        hold the tensors locally. Tensors are copied to CPU so the caller can
        iterate without holding GPU memory.
        """
        out = {
            "mpjpe_sum": self._motion_mpjpe_sum.detach().cpu().clone(),
            "root_pos_err_sum": self._motion_root_pos_err_sum.detach().cpu().clone(),
            "root_quat_err_sum": self._motion_root_quat_err_sum.detach().cpu().clone(),
            "episode_count": self._motion_metric_episode_count.detach().cpu().clone(),
        }
        if self._motion_g_mpjpe_sum is not None:
            out["g_mpjpe_sum"] = self._motion_g_mpjpe_sum.detach().cpu().clone()
        return out
