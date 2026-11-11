# Motion Reference

Motion tracking tasks score a robot against reference motion. Eden serves that motion along two paths, and this page
documents the classes of both. The [motion tracking guide](../guides/motion-tracking.md) explains the data schema,
the multi-actor pattern and which tracking term reads which field.

- **The manager path** — `MimicRLEnvBase` owns a `MotionManager` whose `MotionTerm` loads a
  [dataset](datasets.md), reads the frame at `floor((start + t) * fps)` (no interpolation at lookup), serves a
  window of future frames and runs the
  motion-id and start-time curricula. Tracking terms read it as `env.motion_manager.data[<entity_name>]`. This is
  the training path.
- **The command path** — `MotionCommand` (a command term) plays `.npz` clips verbatim on a plain `RLEnvBase`, for
  frozen-policy inference and single-clip tracking. It is documented with the
  [command terms](terms/commands.md#motion-clips).

Both expose the same world-frame vocabulary: `links_*_w` for the tracked links and `base_*_w` for the floating
base. Quaternions are **wxyz**.

## Motion manager and MotionTerm

::: eden.managers.motion_manager
    options:
      show_root_heading: false

## MotionTerm collaborators

The pieces `MotionTerm` delegates to: adaptive start-time bins, the per-motion outcome curriculum, per-motion
tracking-error metrics, and the opt-in resync that keeps an episode running past a clip's end.

::: eden.managers.motion.bin_sampler
    options:
      show_root_heading: false

::: eden.managers.motion.outcome_curriculum
    options:
      show_root_heading: false

::: eden.managers.motion.per_motion_metrics
    options:
      show_root_heading: false

::: eden.managers.motion.resync
    options:
      show_root_heading: false

## Options

`MotionManagerOptions`, `MotionTermOptions` and `MotionGroupOptions` are on the
[manager options page](options/managers.md#motions).

## Data containers and lookup helpers

The motion data containers live in `eden/utils/motion_reference/`, which has no `__init__.py`, so the API generator
cannot render them here. Read their docstrings in the source:

| Module | Contents |
|---|---|
| `motion_data.py` | `MotionSequence` (one clip, `[num_frames, ...]` arrays), `BatchMotionSequence`, `MotionReferenceData`, `MotionReferenceState` |
| `lookup.py` | index resolution for tracking terms: `get_motion_link_idx`, `get_motion_dofs_idx`, `current_motion_frame`, `assert_motion_field_present`, `get_anchor_state` |
| `utils.py` | robot-vs-reference differences (`get_link_position_distance`, `get_dofs_position_difference`, ...) and interpolation (`interpolate_motion`, `slerp_rows`) |
| `transforms.py` | anchor-relative link transforms |
| `augmentation_mixin.py`, `filter_mixin.py` | `SymmetricAugmentationMixin`, `FilterMixin` |
| `bipedal_ik.py` | start-transition IK used in preprocessing |

## See also

- [Motion tracking guide](../guides/motion-tracking.md)
- [Datasets](datasets.md)
- [Tracking observations](terms/observations.md#motion-tracking), [rewards](terms/rewards.md#motion-tracking) and
  [terminations](terms/terminations.md#motion-tracking)
