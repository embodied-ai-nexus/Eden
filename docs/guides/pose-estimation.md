# Pose Estimation from Video

`eden pose` recovers human motion from ordinary monocular video, so a phone clip can become a robot trajectory without
a motion-capture studio. It writes an AMASS-style SMPL-X clip, which is exactly what
[`eden retarget soma`](retargeting.md#soma-smpl-family-motion-a-humanoid) reads. This page covers installing the
estimator, running it, the frame and timing conventions of its output, and the Python API.

```bash
eden pose gvhmr --src clip.mp4 --out motions/clip.npz --static-cam
eden retarget soma --src motions/clip.npz --out qpos/ --body-model smplx \
    --body-model-path /data/body_models/smplx/SMPLX_NEUTRAL.npz --snap-to-ground
```

The backend is [GVHMR](https://github.com/zju3dv/GVHMR) (world-grounded human motion recovery, SIGGRAPH Asia 2024),
installed from the `eden-gvhmr` package (import name `hmr4d`). It is the only `eden pose` subcommand.

## Install

GVHMR installs into the same environment as Eden, through the `pose` extra:

=== "pip"

    ```bash
    pip install 'eden-world[pose]'
    ```

=== "uv (checkout)"

    ```bash
    uv sync --inexact --extra pose
    ```

Then download the model weights and provide the body models:

```bash
gvhmr-download-checkpoints          # GVHMR, HMR2, ViTPose and YOLO weights, from eden-gvhmr
export GVHMR_CHECKPOINT_ROOT=/data/gvhmr/checkpoints   # or pass --checkpoint-root on every run
```

The checkpoint root holds `gvhmr/`, `hmr2/`, `vitpose/`, `yolo/` and `body_models/`. The SMPL and SMPL-X body models
in `body_models/` are licence-gated and are not downloaded for you: register at
[smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de/) and [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/) and
place them there as GVHMR's own install instructions describe. Without `--checkpoint-root` or the environment
variable, GVHMR looks in `./inputs/checkpoints`.

GVHMR runs on CUDA only; pick a GPU with `CUDA_VISIBLE_DEVICES`. `hmr4d` is imported lazily, so `import eden` never
needs it, and a missing install reports the extra to install instead of a bare `ModuleNotFoundError`.

## Running `eden pose gvhmr`

```bash
eden pose gvhmr --src videos/ --out motions/ --pattern '*.mp4' --static-cam --skip-existing
```

| flag | default | effect |
|---|---|---|
| `--src` | required | one video, or a directory searched recursively |
| `--out` | required | output directory (the source tree is mirrored), or a `.npz` file for a single video |
| `--pattern` | `*.mp4` | glob for a directory `--src` |
| `--checkpoint-root` | GVHMR's own resolution | weights directory (see [Install](#install)) |
| `--ckpt` | `<root>/gvhmr/gvhmr_siga24_release.ckpt` | GVHMR checkpoint file |
| `--static-cam` | off | assume the camera does not move, skipping visual odometry |
| `--use-dpvo` | off | use DPVO instead of SimpleVO for camera motion (needs GVHMR's DPVO install and checkpoint) |
| `--f-mm` | estimated | full-frame-equivalent focal length in mm, when you know the lens |
| `--output-root` | GVHMR's `outputs/demo` | where GVHMR writes, and reuses, its per-video intermediates |
| `--skip-existing` | off | skip a video whose output exists |
| `--verbose` | off | also save GVHMR's bounding-box and keypoint preprocessing overlays |

One model load serves the whole batch. GVHMR keeps each video's intermediates (tracking, ViTPose features, the raw
`hmr4d_results.pt`) under `--output-root` and reuses them, so a re-run of the same video is nearly free.

!!! tip "`--static-cam` only for a fixed camera"
    For tripod footage `--static-cam` is correct and much faster. For handheld video it is wrong: the camera's own
    motion would be absorbed into the body motion.

## The output

Each video becomes one `.npz`:

| key | shape | content |
|---|---|---|
| `poses` | `(T, 165)` | SMPL-X axis-angle pose, AMASS layout: root (3), body (63), then jaw, eyes and hands, which are zero because GVHMR does not estimate them |
| `trans` | `(T, 3)` | root translation, **Z-up** |
| `betas` | `(10,)` | body shape, averaged over the clip |
| `gender` | | `"neutral"` (GVHMR predicts with the neutral SMPL-X model) |
| `mocap_frame_rate` | | always `30` |
| `surface_model_type` | | `"smplx"` |
| `gvhmr_*` | | the raw Y-up estimate (`gvhmr_global_orient`, `gvhmr_transl`, `gvhmr_betas`, `gvhmr_pelvis_j0`, `gvhmr_K_fullimg`, `gvhmr_video_fps`), for provenance |
| `source_video` | | the input path |

Three properties of the output matter downstream.

**Up axis.** GVHMR predicts in a gravity-aligned frame whose +Y is up; Eden, Genesis and AMASS are Z-up. The clip is
rotated into Z-up before it is written. Only the root changes (SMPL joint rotations are parent-relative), and its
translation is corrected for SMPL-X's rest-pose pelvis offset `J0`, because the body rotates about `J0` rather than
the world origin:

```
trans' = R @ (trans + J0) - J0
```

Rotating `trans` alone, as GVHMR's own `get_tgtcoord_rootparam` does, offsets the whole clip. `J0` comes from GVHMR's
own SMPL-X model.

**Frame rate.** GVHMR re-encodes every input to 30 fps without resampling frames and integrates root motion at that
rate, so the clip is always stamped 30 fps. There is deliberately no `--fps` flag: any other stamp would relabel a
clip whose motion was already integrated at 30 fps. A 60 fps video therefore gives a time-warped trajectory; the
command warns before running the pipeline, and the fix is to re-encode the video to 30 fps first.

**Height.** GVHMR's world translation is only defined up to a vertical offset, so the clip may float or sink by a few
centimetres. Retarget with `eden retarget soma --snap-to-ground`, which applies one constant vertical shift so the
robot's lowest collision point touches `z = 0`.

### No overlay videos

Eden does not expose GVHMR's in-camera and world overlay rendering. That path needs pytorch3d (built against your
torch and CUDA) and chumpy from git, neither of which can be declared as a dependency, so the `pose` extra does not
install them; a flag that failed after the whole pipeline had run would be worse than none. To look at an estimate,
install those two and use GVHMR's own CLI, `python -m hmr4d.demo --video clip.mp4 -s`. `--verbose` works from Eden:
its overlays are drawn with OpenCV.

### Reproducibility

In Eden's environment GVHMR runs on Eden's video decoder (`av`) and person detector (`ultralytics`), not the versions
its checkpoint was tuned with. The estimator's module docstring records the effect, measured one change at a time on
GVHMR's bundled tennis clip: `av` 13 → 17 moves body joints by up to 3.6°, and `ultralytics` 8.2.42 → 8.4.113 moves
the detected boxes by 8.5 px and body joints by 3.4°; numpy (including the 1.x → 2.x jump), timm and imageio are
bit-identical. The estimate stays valid, but it is not bit-comparable with the published reference stack. To
reproduce published numbers, install GVHMR in its own environment with its pinned versions and run it there.

## Python API

No Eden scene or entity is involved: estimation is offline preprocessing, and the robot only enters at retargeting.

```python
from eden.extensions.pose_estimation.gvhmr import GvhmrEstimator, save_amass_npz

estimator = GvhmrEstimator(
    options=GvhmrEstimator.configure(
        # checkpoint_root is optional: $GVHMR_CHECKPOINT_ROOT is used when unset.
        static_cam=True,
    )
)
try:
    clip = estimator.estimate_clip("tennis.mp4")   # AMASS-style, Z-up
    save_amass_npz(clip, "motions/tennis.npz")

    raw = estimator.estimate("tennis.mp4")         # GVHMR's own output: Y-up world and camera frame
finally:
    estimator.close()
```

- The shared options (`PoseEstimatorOptions`): `checkpoint_root`, `ckpt_path`, `output_root`, `motion_fps` (must stay
  30; any other value raises), `verbose`. GVHMR's own: `static_cam`, `use_dpvo`, `f_mm`.
- The model loads on the first `estimate` call and is reused; `close()` frees it and its GPU memory.
- `estimate(video, **overrides)` returns `global` and `incam` SMPL-X parameters, `pelvis_j0`, `K_fullimg`,
  `output_dir` and `video_fps`. It accepts per-call overrides of `static_cam`, `use_dpvo`, `f_mm`, `output_root` and
  `verbose`; anything else raises.
- `estimate_clip` converts that result with `to_amass_smplx_clip` (in `eden.extensions.pose_estimation.gvhmr`).

A runnable script: `examples/recipes/pose_estimation/gvhmr/video_to_motion.py`.

## Licensing

GVHMR is released for non-commercial research use (see the licence in its repository), and the SMPL and SMPL-X
body models carry their own licences. Eden ships no GVHMR code or weights, only this wrapper; check those terms
before using recovered motion in a product.

## See also

- [Retargeting](retargeting.md): `eden retarget soma` turns the clip into robot qpos.
- [Motion datasets](motion-datasets.md): loading motion for training.
- API: [pose estimation](../api/extensions/pose-estimation.md).
