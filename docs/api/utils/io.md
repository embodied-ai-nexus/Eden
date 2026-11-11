# I/O Utilities

Reading and writing episode data, converting recordings to LeRobot datasets, image conversion, and reading body-model
pickles without `chumpy`. The [datasets and recording guide](../../guides/datasets-and-recording.md) explains the
recording workflow.

## File handlers

The recorder manager writes episodes through a file handler, chosen by name from `FILE_HANDLER_REGISTRY` (HDF5 or
NPZ). `EpisodeData` is the in-memory container of one recorded trajectory.

::: eden.utils.file_handler.base
    options:
      show_root_heading: false

::: eden.utils.file_handler.episode_data
    options:
      show_root_heading: false

::: eden.utils.file_handler.hdf5_file_handler
    options:
      show_root_heading: false

::: eden.utils.file_handler.npz_file_handler
    options:
      show_root_heading: false

::: eden.options.file_handler
    options:
      show_root_heading: false

## LeRobot conversion

Convert an Eden HDF5 recording into a LeRobot v3 dataset (`eden lerobot convert`).

::: eden.utils.lerobot.convert
    options:
      show_root_heading: false

## Images

::: eden.utils.image
    options:
      show_root_heading: false

## Body-model pickles

MANO, SMPL and SMPL-H `.pkl` files pickle `chumpy` arrays; these helpers read them without installing `chumpy`.

::: eden.utils.chumpy_pickle
    options:
      show_root_heading: false

## Buffers

The buffers in `eden/utils/buffers/` (`CircularBuffer` for observation history, `ConcatBatchTensor`, and the
`ResetStateBuffer` that the `ResetFromStateBuffer` event restores from) are not rendered here: the directory has no
`__init__.py`. Read their docstrings in the source.

## See also

- [Recorder terms](../terms/recorders.md)
- [Datasets and recording guide](../../guides/datasets-and-recording.md)
