# Gaussian-splat rendering (gsplat)

Eden's one Gaussian-splat renderer, backing both splat env classes —
:class:`~eden.envs.splat.SplatEnvBase` (non-RL) and :class:`~eden.envs.splat.SplatRLEnvBase`
(RL), which split on the RL surface exactly as ``EnvBase`` / ``RLEnvBase`` do. It is the
default GS backend after the standalone 2DGS-surfel `SplatRenderer` was retired; Nyx
(:class:`~eden.envs.nyx_splat.NyxSplatEnvBase`) remains a separate, higher-fidelity
path-traced option.

`BatchSplatRenderer` renders the splat, and each camera that declares a `SplatComposite`
overlay composites it over the sim pass wherever the segmentation is background (a camera
that declares none shows the plain sim pass). **Which camera** drives it is chosen per camera by
`SplatOptions.camera_mode` (default `"auto"`):

| camera | renders | when (auto) |
|---|---|---|
| `SplatCamera` | one env-invariant view, broadcast to all envs | fixed camera over a shared (`per_env_gaussians=False`) background — cheap |
| `BatchSplatCamera` | every env's camera in one gsplat call | attached/following camera, or `per_env_gaussians=True` |

The gsplat rasterization core (`batch_rasterization.py`, `gaussiandata.py`,
`ply_io.py`) is vendored from the MIT-licensed
[GaussianRenderer](https://github.com/discoverse-dev/GaussianRenderer); the
Genesis glue (`renderer.py`, and `SplatEnvBase` / `SplatCamera` / `BatchSplatCamera`
in the env/entities trees) is Eden's.

## The shared / per-env switch

`SplatOptions.per_env_gaussians` picks the geometry regime:

- **shared** (`False`, default) — one Gaussian set, rendered from each env's
  camera. The geometry is stored once (never replicated or transformed per env),
  so this is the memory-light choice for a **static** photoreal background. A fixed
  camera over it renders once and broadcasts (`SplatCamera`).
- **per-env** (`True`) — each env carries its own Gaussians; splats bound to an
  entity (`PCDOptions.attach_entity`) follow that env's sim poses via rigid-link
  kinematics. Geometry is materialized `num_envs`×, so this fits tens–hundreds of
  envs. Required whenever any splat is entity-attached, and forces `BatchSplatCamera`.

**Memory:** gsplat evaluates view-dependent SH *per rendered view*, so peak VRAM
scales with `minibatch × G × K` (`K = (sh_degree+1)²`). Set `SplatOptions.minibatch`
to bound it (a 500k-Gaussian SH-3 scene renders 256 envs in ~1.2 GB at `minibatch=8`).

## Usage (task config)

```python
# EdenSplatConfig -> SplatEnvBase (non-RL); EdenSplatRLConfig -> SplatRLEnvBase (adds the RL surface).
from eden.utils.configs import EdenSplatRLConfig
from eden.options import SplatOptions
from eden.options.splat import PCDOptions

cfg = EdenSplatRLConfig(
    env_options=...,          # num_envs, decimation, ...
    scene_options=...,        # robot + colliders
    renderer_options=BatchRendererOptions(...),   # Madrona: batched RGB + segmentation
    cameras_options=...,      # a camera with segmentation
    splat_options=SplatOptions(
        per_env_gaussians=False,     # geometry switch
        minibatch=16,
        camera_mode="auto",          # per-camera single-view vs batched (default)
        background=PCDOptions(path="scenes/kitchen/background.ply", pos=(0, 0, 1)),
        # a dynamic object splat that follows an entity (needs per_env_gaussians=True):
        # cube=PCDOptions(path="....ply", attach_entity="cube"),
    ),
    reward_options=..., termination_options=...,   # RL surface
)
env = env_class_for_config(cfg).from_config(cfg)   # -> SplatRLEnvBase
```

A splat camera that declares a `SplatComposite` overlay composites the splat wherever
the sim segmentation is background (the composite's `entities`, plus empty space),
keeping robot/object geometry from the sim pass. `EdenSplatConfig` /
`EdenSplatRLConfig.recording_camera()` declare one, so `eden inference --rec` records the
splat; a task's own cameras declare theirs (Polaris: `entities=["plane"]`).

## Status / caveats

Validated end-to-end on `polaris_block_stack_kitchen` (FR3 + 2F85 + 7 objects, Madrona,
521k-Gaussian kitchen splat, `num_envs=2`):

- **Camera-frame convention vs Madrona** — the splat background aligns with the
  Madrona-rendered robot/objects from the same camera (objects rest on the splat
  table exactly where they sit on the sim plane). The external camera returns
  per-env transforms `(N,4,4)` even when "fixed"; the per-env path handles it.
- **Body-attached (per-env) splats** — `get_links_pos` **default (`relative=False`)**
  is the frame to place splats in: it matches the declared/placed pose exactly
  (`|Δ|=0` vs `default_root_pos`), whereas `relative=True` re-origins to the CoM
  (~3 cm off). This is the opposite of the `get_pos()` gotcha in the root AGENTS.md, which
  is about the free-body *root* getter — `get_links_pos` differs. Shifting an
  attached entity per env visibly moves that env's splat (`mean|Δ| = 44/255`).
- 2DGS assets (only two `scale_*` fields) load via a near-zero third-scale pad and
  render through the 3DGS rasterizer — an approximation, but validated negligible:
  the 549k-Gaussian 2DGS `polaris_pan_clean` scene rendered here vs the retired diff-surfel
  renderer from the same camera differs by `mean|Δ| ≈ 5.8/255` (~2%) with matching
  coverage, the residual being the surfel-vs-flat-ellipsoid edge falloff.
