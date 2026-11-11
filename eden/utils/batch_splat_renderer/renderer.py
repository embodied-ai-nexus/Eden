"""Batched Gaussian-splat renderer for Eden, backed by gsplat.

Two rendering regimes, selected by ``per_env_gaussians``:

* **shared** (``per_env_gaussians=False``) — one Gaussian set is shared by every
  environment and rasterized from the per-env camera poses (the env axis is used
  as gsplat's *camera* batch). The Gaussian **geometry** (means/quats/scales) is
  stored once — never replicated or transformed per env — so this is the mode
  for a scene that is static across envs (a photorealistic background composited
  under Madrona-rendered robots/objects), and it is the memory-light choice.
  Given a single camera pose it renders exactly one view, which the caller uses
  for every env (:class:`~eden.entities.camera.SplatCamera`).

* **per-env** (``per_env_gaussians=True``) — each env carries its own Gaussians,
  whose dynamic points are rigidly driven by that env's per-body sim poses
  (:meth:`update`), then rasterized with the env axis as gsplat's *batch*
  dimension. Means and quats are materialized ``num_envs`` times, so this fits
  tens–hundreds of envs. Use when splat objects move independently per env, or
  the scene is domain-randomized per env.

Memory note: gsplat evaluates **view-dependent** spherical harmonics per rendered
view, so peak VRAM scales with ``(views_per_chunk * G * K)`` in *both* modes
(``K = (sh_degree+1)**2``). ``minibatch`` chunks the env/view axis to bound that
peak — set it whenever ``num_envs`` is large; shared mode's win is avoiding the
per-env geometry copy, not escaping the per-view SH cost.

The renderer is deliberately sim-agnostic: it consumes camera extrinsics and
per-body poses as plain torch tensors. Eden's camera/env layer is responsible
for reading those out of Genesis (see ``SplatComposite`` in
:mod:`eden.entities.camera_overlays`).
"""

from __future__ import annotations

from dataclasses import dataclass

import eden as en
import numpy as np
import torch
from genesis.utils.geom import transform_by_quat
from torch import Tensor

from eden.utils.geom import quat_mul

from .batch_rasterization import batch_env_render, batch_render, batch_update_gaussians
from .gaussiandata import GaussianBatchData, GaussianData
from .ply_io import load_ply


@dataclass
class SplatEntry:
    """One splat PLY plus its placement and (optional) rigid-body binding.

    Parameters
    ----------
    ply_path : str
        Path to a standard 3DGS/2DGS ``.ply``.
    pos, quat, scale : placement
        Baked into the loaded Gaussians. For a static entry this is the world
        placement; for a dynamic entry (``body_row`` set) it is an extra local
        offset applied *before* the per-env body pose.
    body_row : int | None
        ``None`` marks a static entry (rendered identically in every env). An
        integer binds the entry to a rigid body: in per-env mode its Gaussians
        follow ``body_pos[:, body_row]`` / ``body_quat[:, body_row]`` supplied
        to :meth:`update`.
    """

    ply_path: str
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # wxyz
    scale: float = 1.0
    body_row: int | None = None


@dataclass
class _Loaded:
    template: GaussianData  # concatenated, baked, on-device
    dynamic_mask: Tensor  # (G,) bool
    point_to_body_idx: Tensor  # (G,) long
    n_bodies: int


class BatchSplatRenderer:
    """Batched gsplat renderer with a shared / per-env Gaussian switch."""

    def __init__(
        self,
        entries: list[SplatEntry],
        num_envs: int,
        *,
        per_env_gaussians: bool = False,
        device: torch.device | str = "cuda",
        minibatch: int | None = None,
    ) -> None:
        if not entries:
            raise ValueError("BatchSplatRenderer requires at least one SplatEntry.")
        self.num_envs = num_envs
        self.per_env_gaussians = per_env_gaussians
        self.device = torch.device(device)
        self.minibatch = minibatch

        self._loaded = self._load_and_bake(entries)

        n_dyn = int(self._loaded.dynamic_mask.sum())
        if not per_env_gaussians and n_dyn:
            en.logger.warning(
                f"BatchSplatRenderer: shared mode ignores per-env motion, but "
                f"{n_dyn} Gaussian(s) are bound to rigid bodies; they render frozen at "
                f"their loaded pose. Set per_env_gaussians=True to animate them."
            )
        # Cached per-env Gaussians for per-env mode (refreshed by update()).
        self._gsb: GaussianBatchData | None = None

    # ------------------------------------------------------------------ load

    def _load_and_bake(self, entries: list[SplatEntry]) -> _Loaded:
        xyz, rot, scale, opacity, sh = [], [], [], [], []
        dyn_mask, body_idx = [], []
        loaded = [load_ply(e.ply_path) for e in entries]
        max_k = max(g.sh.shape[1] for g in loaded)
        n_bodies = 0

        for e, g in zip(entries, loaded):
            # to device
            gx = torch.as_tensor(g.xyz, dtype=torch.float32, device=self.device)
            gr = torch.as_tensor(g.rot, dtype=torch.float32, device=self.device)
            gs_ = torch.as_tensor(g.scale, dtype=torch.float32, device=self.device)
            go = torch.as_tensor(g.opacity, dtype=torch.float32, device=self.device)
            gsh = torch.as_tensor(g.sh, dtype=torch.float32, device=self.device)  # (N, K, 3)

            # bake placement (scale, then rotate+translate by quat/pos)
            pos = torch.tensor(e.pos, dtype=torch.float32, device=self.device)
            quat = torch.tensor(e.quat, dtype=torch.float32, device=self.device)
            if e.scale != 1.0:
                gx = gx * e.scale
                gs_ = gs_ * e.scale
            gx = transform_by_quat(gx, quat.expand(gx.shape[0], 4)) + pos
            gr = quat_mul(quat.expand(gr.shape[0], 4), gr)

            # pad SH up to the widest degree present so all entries share K
            if gsh.shape[1] < max_k:
                pad = torch.zeros(gsh.shape[0], max_k - gsh.shape[1], 3, device=self.device)
                gsh = torch.cat([gsh, pad], dim=1)

            n = gx.shape[0]
            xyz.append(gx)
            rot.append(gr)
            scale.append(gs_)
            opacity.append(go)
            sh.append(gsh)
            if e.body_row is None:
                dyn_mask.append(torch.zeros(n, dtype=torch.bool, device=self.device))
                body_idx.append(torch.zeros(n, dtype=torch.long, device=self.device))
            else:
                dyn_mask.append(torch.ones(n, dtype=torch.bool, device=self.device))
                body_idx.append(torch.full((n,), e.body_row, dtype=torch.long, device=self.device))
                n_bodies = max(n_bodies, e.body_row + 1)

        template = GaussianData(
            xyz=torch.cat(xyz),
            rot=torch.cat(rot),
            scale=torch.cat(scale),
            opacity=torch.cat(opacity),
            sh=torch.cat(sh),  # (G, K, 3)
        )
        return _Loaded(
            template=template,
            dynamic_mask=torch.cat(dyn_mask),
            point_to_body_idx=torch.cat(body_idx),
            n_bodies=n_bodies,
        )

    @property
    def world_xyz(self) -> np.ndarray:
        """World-frame xyz of every loaded Gaussian (baked placement), for debug-point drawing."""
        return self._loaded.template.xyz.detach().cpu().numpy()

    @property
    def n_bodies(self) -> int:
        return self._loaded.n_bodies

    # ---------------------------------------------------------------- update

    def update(self, body_pos: Tensor, body_quat: Tensor, *, scalar_first: bool = True) -> None:
        """Refresh per-env Gaussians from rigid-body poses (per-env mode).

        Parameters
        ----------
        body_pos : Tensor
            ``(num_envs, n_bodies, 3)`` world positions.
        body_quat : Tensor
            ``(num_envs, n_bodies, 4)`` orientations. ``scalar_first`` selects
            wxyz (default, Genesis) vs xyzw.
        scalar_first : bool
            Whether ``body_quat`` is wxyz (True) or xyzw (False).
        """
        if not self.per_env_gaussians:
            return  # shared mode: nothing to animate
        body_pos = torch.as_tensor(body_pos, dtype=torch.float32, device=self.device)
        body_quat = torch.as_tensor(body_quat, dtype=torch.float32, device=self.device)
        self._gsb = batch_update_gaussians(
            self._loaded.template,
            body_pos,
            body_quat,
            point_to_body_idx=self._loaded.point_to_body_idx,
            dynamic_mask=self._loaded.dynamic_mask,
            scalar_first=scalar_first,
        )

    # ---------------------------------------------------------------- render

    def render(
        self,
        cam_pos: Tensor,  # (num_envs, 3) world
        cam_rot: Tensor,  # (num_envs, 3, 3) world camera rotation (Genesis/OpenGL frame)
        height: int,
        width: int,
        fovy: float | np.ndarray,  # vertical FOV, degrees
        *,
        bg_imgs: Tensor | None = None,  # (num_envs, H, W, 3)
        return_alpha: bool = False,
        clamp: bool = True,
    ) -> tuple[Tensor, ...]:
        """Render every env's view.

        Returns ``(rgb, depth)`` or ``(rgb, depth, alpha)`` if ``return_alpha``.
        ``rgb`` is ``(num_envs, H, W, 3)`` float in ``[0, 1]`` (if ``clamp``),
        ``depth`` and ``alpha`` are ``(num_envs, H, W, 1)``.
        """
        cam_pos = torch.as_tensor(cam_pos, dtype=torch.float32, device=self.device).reshape(-1, 3)
        cam_rot = torch.as_tensor(cam_rot, dtype=torch.float32, device=self.device).reshape(-1, 3, 3)
        n = cam_pos.shape[0]
        fovy_arr = np.broadcast_to(np.asarray(fovy, dtype=np.float32), (n,)).copy()

        if self.per_env_gaussians:
            if self._gsb is None:
                # No update() yet (e.g. no dynamic bodies): expand the template once.
                self.update(
                    torch.zeros(self.num_envs, max(self.n_bodies, 1), 3, device=self.device),
                    torch.tensor([1.0, 0, 0, 0], device=self.device).expand(self.num_envs, max(self.n_bodies, 1), 4),
                )
            if n != self._gsb.batch_size:
                # gsplat batches means and viewmats on the same leading axis; a mismatch would
                # otherwise surface as a cryptic gsplat shape error. Broadcast a shared camera
                # to num_envs (SplatComposite does this) before calling render() in per-env mode.
                raise ValueError(
                    f"per-env render expects one camera pose per env: got {n} camera pose(s) for "
                    f"{self._gsb.batch_size} envs."
                )
            out = batch_env_render(
                self._gsb,
                cam_pos.reshape(n, 1, 3),
                cam_rot.reshape(n, 1, 9),
                height,
                width,
                fovy_arr.reshape(n, 1),
                bg_imgs=bg_imgs.reshape(n, 1, height, width, 3) if bg_imgs is not None else None,
                minibatch=self.minibatch,
                return_alpha=return_alpha,
            )
            # drop the singleton camera axis
            out = tuple(t.reshape(n, height, width, -1) for t in out)
        else:
            # Shared geometry: env axis is gsplat's camera batch. Chunk it so the
            # per-view SH evaluation stays within VRAM (see module memory note).
            out = self._render_shared_chunked(cam_pos, cam_rot, height, width, fovy_arr, bg_imgs, return_alpha)

        if clamp:
            out = (out[0].clamp(0.0, 1.0),) + tuple(out[1:])
        return out

    def _render_shared_chunked(self, cam_pos, cam_rot, height, width, fovy_arr, bg_imgs, return_alpha):
        n = cam_pos.shape[0]
        mb = self.minibatch if (self.minibatch and self.minibatch < n) else n
        chunks = []
        for i in range(0, n, mb):
            j = min(i + mb, n)
            chunks.append(
                batch_render(
                    self._loaded.template,
                    cam_pos[i:j],
                    cam_rot[i:j].reshape(j - i, 9),
                    height,
                    width,
                    fovy_arr[i:j],
                    bg_imgs=bg_imgs[i:j] if bg_imgs is not None else None,
                    return_alpha=return_alpha,
                )
            )
        if len(chunks) == 1:
            return chunks[0]
        return tuple(torch.cat([c[k] for c in chunks], dim=0) for k in range(len(chunks[0])))
