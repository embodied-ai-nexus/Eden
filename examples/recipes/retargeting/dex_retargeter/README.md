# dex_retargeter examples

Eden port of [`dex-retargeting`](https://github.com/dexsuite/dex-retargeting),
backed by **Genesis kinematics** (no Pinocchio, no SAPIEN). Three optimizers
are available, mirroring the upstream API:

- `vector` — match relative wrist-to-fingertip vectors. Good for live teleop
  from monocular video (frame-invariant, handles scale via `scaling_factor`).
- `dexpilot` — vector retargeting with a thumb-to-fingertip projection prior
  (encourages stable pinch grasps); see DexPilot ([arXiv 1910.03135](https://arxiv.org/abs/1910.03135)).
- `position` — match absolute keypoint positions. Good for offline retargeting
  of hand-object pose datasets.

See the brief comparison in `DexRetargeter`'s class docstring or the
upstream README for when to pick each.

## Install

```bash
uv pip install -e '.[dex-retargeter-example]'
```

This installs `nlopt`, `mediapipe`, `opencv-python`, and `tyro` alongside the
core Eden dependencies. The base library (`.[dex-retargeter]`) only needs
`nlopt`.

## Realtime teleoperation

```bash
uv run python examples/recipes/retargeting/dex_retargeter/show_realtime_retargeting.py \
    --preset xhand1_right_dexpilot \
    --hand-type right
```

A webcam stream feeds MediaPipe; the resulting 21 keypoints drive `XHand1_R`
inside the Genesis viewer. Press `q` in the OpenCV window to exit.

## Offline video → robot trajectory

```bash
uv run python examples/recipes/retargeting/dex_retargeter/detect_from_video.py \
    --video-path data/human_hand.mp4 \
    --output-path data/qpos.pkl \
    --preset xhand1_right_dexpilot \
    --hand-type right
```

Writes a pickle with `{"data": [qpos_t, ...], "meta_data": {...}}`.

### Replay a saved trajectory

```bash
uv run python examples/recipes/retargeting/dex_retargeter/render_robot_hand.py \
    --pickle-path data/qpos.pkl \
    --fps 30
```

## Offline position retargeting (hand-object datasets)

Two variants ship — one generic, one wired to Eden's `DexYCBDataset` loader.

### Generic NPZ input

```bash
uv run python examples/recipes/retargeting/dex_retargeter/position_retargeting.py \
    --keypoints-npz data/mano_traj.npz \
    --output-path data/qpos_position.pkl \
    --preset xhand1_right_position
```

`mano_traj.npz` should contain a `keypoints` array of shape `(T, 21, 3)` in
MANO ordering plus a scalar `fps`. Use any source you like (ARCTIC, your own
collection, or DexYCB exported to npz).

### DexYCB direct

Uses [`eden.datasets.dex_ycb_dataset.DexYCBDataset`](../../../../eden/datasets/dex_ycb_dataset.py)
to load AprilTag-world-frame MANO keypoints; each frame is re-anchored to
the robot's wrist mount before retargeting. The robot's fingertip link
names and MANO keypoint indices come from `XHand1_R.metadata.target_hand_links`
and `XHand1_R.metadata.mano_hand_idx` — no hand-coded lists.

Each retargeted clip is written as `retargeted_<label>.npz` **next to**
the source `pose.npz`, with a schema (`dofs_pos`, `dofs_name`,
`base_pos_w`, `base_quat_w`, `framerate`) that `DexYCBDataset` can reload
as a third motion key.

Browse the available clips first:

```bash
uv run python examples/recipes/retargeting/dex_retargeter/dex_ycb_position_retargeting.py \
    --data-path /path/to/dex-ycb/extracted/root \
    --calibration-dir /path/to/dex-ycb/calibration \
    --list
```

Then narrow down with any combination of filters (all substring matches,
case-insensitive, AND-combined):

```bash
uv run python examples/recipes/retargeting/dex_retargeter/dex_ycb_position_retargeting.py \
    --data-path /path/to/dex-ycb/extracted/root \
    --calibration-dir /path/to/dex-ycb/calibration \
    --side right \
    --objects mustard tomato_soup \
    --subjects subject-01 \
    --max-clips 5
```

| Flag | Matches against | Example |
| --- | --- | --- |
| `--side {right,left}` | hand side (exact) | `--side right` |
| `--objects PAT [PAT ...]` | YCB folder name | `--objects mustard sugar_box` |
| `--subjects PAT [PAT ...]` | sequence-id prefix | `--subjects subject-01` |
| `--max-clips N` | post-filter cap | `--max-clips 5` |
| `--retargeted-label LABEL` | filename suffix (default: preset without optimizer suffix) | `--retargeted-label xhand1_right` |
| `--target-framerate HZ` | dataset resample rate, also stamped on the output | `--target-framerate 30` |
| `--list` | print the filtered catalog and exit | |
| `--view` | open the Genesis viewer instead of retargeting; replays YCB object + robot hand for each matched clip (requires the NPZs to exist) | |
| `--record` | with `--view`: also write an MP4 per clip as `retargeted_<label>.mp4` | |
| `--debug-vis` | with `--view`: overlay the 21 MANO keypoints + 20 skeleton edges on top of the robot (blue/green spheres + pink bones) | |

### Inspect the retargeting

After retargeting once, replay each clip in the Genesis viewer alongside
the YCB object it interacted with. The robot is rebuilt floating-base for
viz so its full pose `(base_pos_w, base_quat_w, dofs_pos)` can be driven
directly from the saved NPZ:

```bash
uv run python examples/recipes/retargeting/dex_retargeter/dex_ycb_position_retargeting.py \
    --data-path /path/to/dex-ycb/extracted/root \
    --calibration-dir /path/to/dex-ycb/calibration \
    --side right --objects mustard --max-clips 3 \
    --view
```

Add `--record` to write `retargeted_<label>.mp4` next to each clip's
`pose.npz`. Add `--debug-vis` to overlay the source MANO hand
(21 keypoint spheres + 20 skeleton edges) on top of the robot — handy
for eyeballing retargeting fidelity at a glance:

```bash
uv run python examples/recipes/retargeting/dex_retargeter/dex_ycb_position_retargeting.py \
    --data-path /path/to/dex-ycb/extracted/root \
    --calibration-dir /path/to/dex-ycb/calibration \
    --side right --objects mustard --max-clips 3 \
    --view --debug-vis
```

The viewer function (`view_clip(dataset, clip_idx, spec, ..., debug_vis=True)`)
is also importable for use in your own scripts.

### Round-trip: reload the retargeted robot motion

Pass the same `--retargeted-label` to `DexYCBDataset` to get a third
motion key (`"robot"`) alongside `"hand"` and `"object"`:

```python
from eden.datasets.dex_ycb_dataset import DexYCBDataset

dataset = DexYCBDataset(options=DexYCBDataset.configure(
    raw_data_path="/path/to/dex-ycb/extracted/root",
    calibration_dir="/path/to/dex-ycb/calibration",
    target_framerate=30.0,
    retargeted_label="xhand1_right",
))

robot = dataset.motion_sequences["robot"]
robot.dofs_pos        # (n_clips, T_max, n_dofs)
robot.base_pos_w      # (n_clips, T_max, 3)   — mirrors hand.base_pos_w
robot.base_quat_w     # (n_clips, T_max, 4)   wxyz
dataset.motion_dofs_name["robot"]   # ['right_hand_thumb_bend_joint', ...]
```

Clips that don't have a `retargeted_<label>.npz` next to their `pose.npz`
are dropped (with a warning) so the hand / object / robot sequences stay
aligned. If you only retarget a subset, expect the loaded dataset to be
restricted to that subset.

> Heads up: the upstream position-retargeting config injects a 6-DOF dummy
> free joint so the hand can fly. The Eden port targets a **fixed-base**
> entity for now — keypoints are matched relative to the entity's mounted
> wrist pose. Floating-base position retargeting is planned as a follow-up.

## Adding a new robot

Append an entry to `examples/recipes/retargeting/dex_retargeter/hand_presets.py`. A preset is
a `(robot_options_factory, retargeter_options)` pair. Any Eden
`RobotOptions` works — `XHand1_R` / `XHand1_L` / `InspireRH56_R` /
`InspireRH56_L` / `TesolloDG5F_R` / `TesolloDG5F_L` / `ManoHand_R` /
`ManoHand_L` / `FrankaHand` / `Robotiq2f85` are all registered in
`eden.robots`. You need to know the **link names** for the fingertip/wrist
targets; they come from the robot's URDF.

## Differences from upstream

| Aspect | Upstream `dex-retargeting` | Eden `DexRetargeter` |
| --- | --- | --- |
| Kinematics | Pinocchio (`pin.forwardKinematics`, body Jacobian) | Genesis (`entity.set_qpos`, spatial Jacobian) |
| Joint ordering | Pinocchio's | `entity.dofs_name` (MuJoCo XML order) |
| Viewer | SAPIEN | Genesis viewer (`env.scene.visualizer`) |
| YAML configs | Yes (`RetargetingConfig.load_from_file`) | Replaced by Python presets / `DexRetargeter.configure(...)` |
| Mimic joints | Parsed via yourdfpy + `MimicJointKinematicAdaptor` | Expected to be pre-baked in the URDF |
| Floating-base position retargeting | Via `add_dummy_free_joint` URDF surgery | Not yet supported |
