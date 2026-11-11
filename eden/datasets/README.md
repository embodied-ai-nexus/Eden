# Motion Datasets

Motion-capture dataset loaders for motion-imitation and manipulation tasks. Each
dataset is a [`DatasetBase`](base.py) subclass that downloads (usually from
HuggingFace), parses, retargets, and caches motion sequences into the
`BatchMotionSequence` layout consumed by the motion manager. Offline
[`DatasetModifier`](modifiers/)s apply augmentations (mirroring, jitter, time-warping,
grounding, …) after load.

```python
from eden.datasets import DATASET_REGISTRY, LAFAN1RetargetedG1Dataset
from eden.options.datasets import DatasetOptions
```

Each loader self-registers under a snake-case key derived from its class name, so it
can also be looked up via `DATASET_REGISTRY`. Runnable examples live under
[`examples/recipes/datasets/`](../../examples/recipes/datasets/).

| Dataset | Class | Module |
|---|---|---|
| [AMASS (retargeted → G1)](#amass) | `AMASSRetargetedG1Dataset` | `amass_dataset.py` |
| [ARCTIC](#arctic) | `ArcticDataset` | `arctic_dataset.py` |
| [DexYCB](#dexycb) | `DexYCBDataset` | `dex_ycb_dataset.py` |
| [HHI-Assist](#hhi-assist) | `HHIAssistDataset` | `hhi_assist_dataset.py` |
| [HOT3D](#mano-hand-object-releases-taco-oakink2-hot3d) | `Hot3dDataset` | `hot3d_dataset.py` |
| [LAFAN1](#lafan1) | `LAFAN1RetargetedG1Dataset` | `lafan1_dataset.py` |
| [OakInk2](#mano-hand-object-releases-taco-oakink2-hot3d) | `OakInk2Dataset` | `oakink2_dataset.py` |
| [OMOMO (Holosoma)](#omomo) | `OMOMOHolosomaDataset` | `omomo_dataset.py` |
| [PHUMA](#phuma) | `PHUMADataset` | `phuma_dataset.py` |
| [SEED](#seed) | `SEEDRetargetedG1Dataset` | `seed_dataset.py` |
| [TACO](#mano-hand-object-releases-taco-oakink2-hot3d) | `TacoDataset` | `taco_dataset.py` |

### Downloading only what a loader reads

Leaving `raw_data_path` unset downloads `registry/dataset` from HuggingFace. Several
repos hold more than one loader needs — other robots' retargets, other splits, other
humanoid configurations — so the download runs three stages, each opt-in:

| Stage | Declared as | Job |
|---|---|---|
| fetch | `_hf_allow_patterns: ClassVar[tuple[str, ...] \| None]` | Glob patterns handed to `snapshot_download`. `None` fetches the whole repo. |
| unpack | `_post_download_process(local_dir) -> str` | Extract any archive **in place** and return the root it produced. |
| narrow | `_hf_data_subdir: ClassVar[str \| None]` | Directory under that root holding the motion files, as a relative `/`-separated path. Missing ⇒ `FileNotFoundError` naming the repo. |

The stages are independent — SEED's subdirectory only exists after its tarball is
unpacked, so it cannot be an allow-pattern, and PHUMA's single `data.zip` cannot be
narrowed at fetch time at all. Each dataset opts into the stages it needs; the five
that own their whole repo declare neither attribute.

Both are class attributes, not options: a repo's layout belongs to the loader that reads
it, so a retarget for another robot or split is a subclass declaring its own. A subclass
**replaces** the pattern tuple rather than extending it (unlike `_cache_identity_fields`,
which is unioned over the MRO) — an h1 retarget wants `("h1/*", ...)`, never
`("g1/*", "h1/*")`. Both are checked when the class is defined: an absolute or
`..`-bearing subdirectory (which would silently escape the snapshot —
`os.path.join(root, "/etc")` is `"/etc"`) and an empty pattern tuple (which matches
nothing rather than everything) raise there rather than at download time.

**Where the files land.** The fetch goes through
`eden.utils.assets._snapshot_download_locked`, as `get_asset_path` does, with no
`local_dir`, so the snapshot lives in the HuggingFace cache under a revision-keyed
`snapshots/<sha>/` directory — and "unpack **in place**" means the unpack stage
extracts into that directory. The cost is that an upstream revision bump re-downloads
*and* re-extracts into a fresh snapshot directory, leaving the old one behind; for
SEED that is a 23 GB untar. Point `raw_data_path` at a directory you manage if you
need a stable location. SEED marks a finished untar with `g1/.eden_extracted` (PHUMA a
finished unzip with `data/.eden_extracted`), so an interrupted one is redone on the next
run and a finished one is not; deleting `g1/` (`data/`) forces it.

**What `raw_data_path` may point at.** Only the download path runs the three stages —
an explicit `raw_data_path` is used exactly as given, so as a rule it must already be
the equivalent of the narrowed directory (for LAFAN1, `.../LAFAN1_Retargeting_Dataset/g1/`).
SEED is the one exception, because its gated 114 GB repo makes a manual
`huggingface-cli download` the common route: it also accepts the directory holding the
raw `g1.tar.gz` (which it extracts) or the extracted `g1/` root. That tolerance is
per-dataset by design — it lives in `SEEDRetargetedG1Dataset._get_all_motion_files`,
not in the stages, because "accept any of these three shapes" is not something a single
subdirectory can express.
Pointing at the archive's directory makes SEED extract there, so that directory must be
writable unless `g1/.eden_extracted` is already in it. For a read-only copy — a shared
mirror, or a tree extracted by hand — point `raw_data_path` at its `g1/csv` instead,
which is read as is; pointing at the archive's directory raises a `PermissionError`
that names that path.

---

## AMASS

Retargeted-to-G1 split of [AMASS](https://amass.is.tue.mpg.de/), hosted on HuggingFace
as [`ember-lab-berkeley/AMASS_Retargeted_for_G1`](https://huggingface.co/datasets/ember-lab-berkeley/AMASS_Retargeted_for_G1).
Clips are already mapped to the Unitree G1 humanoid (29 DOFs) and stored in the
IsaacLab AMP `.npz` format (body positions and rotations pre-computed, so no FK is
needed at load time).

### Example

```bash
python examples/recipes/datasets/amass.py
```

---

## ARCTIC

**ARCTIC 🥶: A Dataset for Dexterous Bimanual Hand-Object Manipulation.**

### Example

```bash
python examples/recipes/datasets/arctic.py
```

### License

See [LICENSE](https://github.com/zc-alexfan/arctic/blob/master/LICENSE).

### Citation

```bibtex
@inproceedings{fan2023arctic,
  title = {{ARCTIC}: A Dataset for Dexterous Bimanual Hand-Object Manipulation},
  author = {Fan, Zicong and Taheri, Omid and Tzionas, Dimitrios and Kocabas, Muhammed and Kaufmann, Manuel and Black, Michael J. and Hilliges, Otmar},
  booktitle = {Proceedings IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2023}
}
```

---

## DexYCB

**DexYCB: A Benchmark for Capturing Hand Grasping of Objects.**
[Project page](https://dex-ycb.github.io/) · [Toolkit](https://github.com/NVlabs/dex-ycb-toolkit)

DexYCB contains 1,000 sequences of 10 subjects performing single-hand grasps
of 21 YCB objects, captured by an 8-camera rig at 30 fps with ground-truth
MANO and 6-DoF object poses.

### Download

The upstream is hosted on Google Drive (gated). Download the per-subject
split plus `calibration.tar.gz` and extract under one root. The loader
reads `<subject>/<sequence>/pose.npz`, `meta.yml`, the master camera's
`labels_*.npz` (the MANO keypoints) and the `calibration/` directory.
`models.tar.gz` is needed only with `apply_mesh_correction=False` (the
poses then place the release's meshes; Eden otherwise ships its own YCB
meshes via HuggingFace), and `bop.tar.gz` never.

```bash
for f in 2020*-subject-*.tar.gz calibration.tar.gz; do
    tar zxvf "$f" -C $DEX_YCB_DIR
done
```

**Why calibration matters.** Raw `pose.npz` is in the master camera's
frame (x-right, y-down, z-forward). With `calibration_dir` set the
loader applies each clip's master-camera extrinsic to land in DexYCB's
AprilTag world frame (z-up, origin on the table tag); without it the
loader keeps the master-cam frame and emits a one-time warning, and a
Genesis viewer will render the scene tilted ~90°.

Expected layout after extraction:

```
$DEX_YCB_DIR/
├── 20200709-subject-01/
│   └── 20200709_141754/
│       ├── meta.yml
│       ├── pose.npz
│       └── <camera_serial>/...
├── ...
└── calibration/
```

### Example

```bash
python examples/recipes/datasets/dex_ycb.py --data-path /path/to/DEX_YCB_DIR --calibration-dir /path/to/calibratoion --traj-idx 0
```

The example auto-detects `calibration/` next to `--data-path`. Pass
`--calibration-dir /path/to/calibration` to point elsewhere.

Or directly in Python:

```python
from eden.datasets import DexYCBDataset
from eden.options.datasets import DatasetOptions

ds = DexYCBDataset(
    options=DatasetOptions(
        raw_data_path="/path/to/DEX_YCB_DIR",
        calibration_dir="/path/to/DEX_YCB_DIR/calibration",
        target_framerate=30.0,
    ),
    device="cuda",
)
hand = ds.motion_sequences["hand"]  # wrist pos + quat per frame
obj = ds.motion_sequences["object"]  # grasped-object pos + quat per frame
print(ds.metadata["objects"][:3])  # YCB folder name per clip
print(ds.metadata["sides"][:3])  # "left" or "right" per clip
```

`hand.base_pos_w` / `base_quat_w` give the MANO wrist trajectory (the
labelled wrist keypoint: `pose_m`'s translation is MANO's `transl`, a
shape-dependent ~9 cm away from it) and `hand.links_pos_w` the 21 labelled
keypoints. With `body_model_params=True` and `body_model_dir` (the MANO
model files) the hand also carries its finger pose and shape as
`body_pose_quat` / `body_shape` — see [Body-model parameters](#body-model-parameters).
`object.base_pos_w` / `base_quat_w` give the trajectory of the grasped object
indicated by `meta.yml::ycb_grasp_ind` (the other YCB instances in the scene
are ignored). Both are expressed in the master-camera frame defined by
`meta.yml::extrinsics`.

### License

DexYCB is released under the [DexYCB Dataset Software License](https://dex-ycb.github.io/assets/license.txt).

### Citation

```bibtex
@inproceedings{chao2021dexycb,
  title = {{DexYCB}: A Benchmark for Capturing Hand Grasping of Objects},
  author = {Chao, Yu-Wei and Yang, Wei and Xiang, Yu and Molchanov, Pavlo and Handa, Ankur and Tremblay, Jonathan and Narang, Yashraj S. and Van Wyk, Karl and Iqbal, Umar and Birchfield, Stan and Kautz, Jan and Fox, Dieter},
  booktitle = {Proceedings IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2021},
}
```

---

## MANO hand-object releases (TACO, OakInk2, HOT3D)

Bimanual hand-object releases that ship MANO parameters, read by
`TacoDataset`, `OakInk2Dataset` and `Hot3dDataset` (shared base:
`hand_object_base.HandObjectDatasetBase`). They need the MANO model files
(`body_model_dir`, from [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/));
Eden reads their `.pkl` files itself, with no extra. `eden retarget flashdex import` reads them.

| Release | `raw_data_path` holds | Clips |
|---|---|---|
| [TACO](https://github.com/Satorinal/TACO) | `Hand_Poses/`, `Object_Poses/`, `object_models_released/` | one per recording; each hand holds the object it stays nearest to |
| [OakInk2](https://oakink.net/v2/) | `anno_preview/`, `program/program_info/`, `object_repair/align_ds/` | one per program stage; each hand holds the object its MANO links touch most (`contact`) |
| [HOT3D](https://facebookresearch.github.io/hand_tracking_toolkit/dataset) | `dataset/<recording>/`, `_assets/assets/` | hand-object segments by fingertip contact (`tip_contact`, `link_contact`, `gap_frames`, `min_frames`, `pad_frames`, `modes`) |

Each clip holds `<side>_hand` (the wrist as base, the 21 MANO keypoints as
links) and `object_<side>` (the object that hand holds) for the hands in
`sides` (`both`, `right` or `left`); a clip where one of them holds nothing,
or (OakInk2, HOT3D) whose span misses a frame, is left out and named, with
every other rejected clip, in `metadata["skipped"]`; so is an OakInk2
recording whose mocap frame ids step by more than 1. `drop_first_n_frames` /
`drop_last_n_frames` trim every clip as its reader cut it (a HOT3D segment
with its `pad_frames` padding). Per-clip metadata: `clip_names`, `recordings`,
`objects_<side>`. Clips cut from a longer recording (OakInk2, HOT3D) get
every object of the recording from `recording_object_poses(recording)`, and
every object's mesh comes from `object_mesh(obj_id)`. The world frames
follow CHORD: z-up, TACO turned 180 degrees about z, HOT3D yawed to face its
workspace. Pass the release's rate as `target_framerate` (the class's
`source_framerate`) to keep the clips unretimed.

```python
from eden.datasets import TacoDataset

ds = TacoDataset(
    options=TacoDataset.configure(
        raw_data_path="/data/taco", body_model_dir="/data/mano_v1_2/models", target_framerate=30.0
    ),
    device="cpu",
)
print(ds.metadata["clip_names"][:3], ds.metadata["objects_right"][:3])
```

---

## Body-model parameters

`body_model_params=True` (off by default) also loads a parametric body
model's pose and shape, for the datasets that have one (DexYCB, TACO,
OakInk2, HOT3D): `body_pose_quat` (each joint below the root, wxyz,
relative to the model's rest pose — the flat hand for MANO), `body_shape`
(`betas`) and the per-clip `body_model` name (`mano_right` / `mano_left`).
The root is the sequence's base: `base_pos_w` is the root joint, not the
model's `transl`. `eden.datasets.utils.body_model.body_model_forward` turns
them back into joints and vertices. With the flag off nothing extra is
read, stored or cached, and no simulation-side buffer ever holds them;
`MirrorLR`, `SmoothStartTransition` and `TCRSTerrainAdapt` refuse sequences
that carry them.

---

## HHI-Assist

Human-human interaction dataset for physical-assistance scenarios. The loader parses
paired BVH files (caregiver + care recipient) via the shared BVH parser in
[`utils/bvh.py`](utils/bvh.py).

### Example

```bash
python examples/recipes/datasets/hhi_assist.py
```

### Citation

```bibtex
@article{saadatnejad2025hhiassist,
  title={HHI-Assist: A Dataset and Benchmark of Human-Human Interaction in Physical Assistance Scenario},
  author={Saadatnejad, Saeed and Hosseininejad, Reyhaneh and Barreiros, Jose and Tsui, Katherine and Alahi, Alexandre},
  journal={IEEE Robotics and Automation Letters (RA-L)},
  year={2025},
  publisher={IEEE}
}
```

---

## LAFAN1

**Ubisoft La Forge Animation Dataset** —
[LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset).

The repo ships side-by-side retargets for three robots — `g1/` (29 DOFs), `h1/`
(19 DOFs), `h1_2/` (27 DOFs) — plus a `robot_description/` tree of URDFs and
meshes. This loader fetches and reads the `g1/` CSVs only (87 MB of the repo's
314 MB); a retarget for another robot is a subclass declaring its own
`_hf_allow_patterns` / `_hf_data_subdir`. An explicit `raw_data_path` should
point at the pre-split `.../LAFAN1_Retargeting_Dataset/g1/` directory — mixing
row widths makes dataset preparation fail with a tensor-shape mismatch.

### Example

```bash
python examples/recipes/datasets/lafan1.py
```

### License

See [LICENSE](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/license.txt).

### Citation

```bibtex
@article{harvey2020robust,
    author    = {Félix G. Harvey and Mike Yurick and Derek Nowrouzezahrai and Christopher Pal},
    title     = {Robust Motion In-Betweening},
    booktitle = {ACM Transactions on Graphics (Proceedings of ACM SIGGRAPH)},
    publisher = {ACM},
    volume    = {39},
    number    = {4},
    year      = {2020}
}
```

---

## OMOMO

[OMOMO](https://github.com/lijiaman/omomo_release) object-manipulation dataset,
processed by the Holosoma pipeline and hosted on HuggingFace as
[`Kashu7100/eden_omomo_holosoma`](https://huggingface.co/datasets/Kashu7100/eden_omomo_holosoma).
Each sequence is an InterMimic-format `.pt` tensor of shape `(T, 591)` encoding root
pose, 52 pre-computed SMPL-H joint world positions, and the manipulated object's 6-DoF
pose (30 fps). No FK is required at load time.

### Example

```bash
python examples/recipes/datasets/omomo.py
```

---

## PHUMA

**PHUMA: Physically-Grounded Humanoid Locomotion Dataset.**

The repo is a single 3.4 GB `data.zip` holding both `g1/` and `h1_2/`
configurations; the loader unzips it once and reads `data/g1`.

### Example

```bash
python examples/recipes/datasets/phuma.py
```

### License

See [LICENSE](https://github.com/DAVIAN-Robotics/PHUMA/blob/main/LICENSE).

### Citation

```bibtex
@article{lee2025phuma,
    title={PHUMA: Physically-Grounded Humanoid Locomotion Dataset},
    author={Kyungmin Lee and Sibeen Kim and Minho Park and Hyunseung Kim and Dongyoon Hwang and Hojoon Lee and Jaegul Choo},
    journal={arXiv preprint arXiv:2510.26236},
    year={2025}
}
```

---

## SEED

**SEED: BONES Studio Embodied-AI Dataset** —
[bones-studio/seed](https://huggingface.co/datasets/bones-studio/seed).

SEED is a large-scale motion-capture library released by BONES Studio. The
loader consumes the **G1 retargeted CSV split** (`g1.tar.gz`, ~23 GB), which
contains ~142k motion clips at 120 fps already mapped to the Unitree G1
humanoid (29 DOFs).

The HuggingFace repo is **gated**: request access on HuggingFace and run
`huggingface-cli login` before triggering the automatic download. Other splits
(SOMA Uniform / Proportional BVH, shape parameters) are skipped — only
`g1.tar.gz` is fetched.

### Example

```bash
python examples/recipes/datasets/seed.py
```

### License

See [LICENSE](https://huggingface.co/datasets/bones-studio/seed/blob/main/LICENSE.md).
