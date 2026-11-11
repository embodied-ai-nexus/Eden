# Environment variables

This page lists every environment variable Eden's code reads or sets, and the ones from the libraries underneath it
that you are most likely to need. Variables that hold a path expect an absolute one unless the entry says
otherwise.

## Eden's own

| Variable | Effect | Default |
|---|---|---|
| `EDEN_ASSETS_DIR` | The root that robot, object, scene and policy assets are downloaded into. Must be absolute: a relative path raises a `ValueError`. Read every time a path is resolved. See [Assets](../getting-started/assets.md). | `$XDG_CACHE_HOME/eden/assets`, i.e. `~/.cache/eden/assets` |
| `EDEN_LOG_THEME` | The console theme: `agent` (plain text, no colors, no greeting box), `human` (colors) or `auto` (colors on a terminal). An empty value counts as unset; any other value raises. `eden --log-theme` and `eden.init(logger_theme=...)` set it for the process and its children. | `agent` |
| `EDEN_SMOKE` | Smoke mode, in which examples run a few steps headless (`en.is_smoke()`). Read as a boolean: `0`, `false` and `no` turn it off, any other non-empty value on. `eden examples smoke` sets it to `1`. | unset (then `PYTEST_VERSION` decides) |
| `EDEN_RESYNC_DEBUG` | `1` turns on debug output from the motion manager's motion-end resync. | unset |
| `EDEN_GO1_SDK_PATH` | A directory holding an uninstalled build of the Unitree Go1 SDK (`robot_interface.*.so`), added to `sys.path` for the Go1 deployment backend. | unset |
| `EDEN_PXREA_SDK_DIR` | An extra directory searched for the XRoboToolkit PC Service's shared library, which the PICO SDK module links against. Searched before `/opt/apps/roboticsservice/SDK/x64`. | unset |

## Caches and downloads

| Variable | Effect | Default |
|---|---|---|
| `XDG_CACHE_HOME` | The root of Eden's per-user cache (`$XDG_CACHE_HOME/eden/`): the default asset root, sample clips, processed motion caches and generated meshes. | `~/.cache` |
| `HF_HOME` | Read by `huggingface_hub`. Moves the Hugging Face cache, where motion datasets are downloaded (`$HF_HOME/hub`). | `~/.cache/huggingface` |
| `HF_HUB_OFFLINE` | Read by `huggingface_hub`. `1` makes a missing file fail at once instead of after a network timeout; files already on disk are unaffected. | unset |
| `HF_TOKEN` | Read by `huggingface_hub`. A Hub access token, the same as `hf auth login`. | unset |
| `QD_OFFLINE_CACHE_FILE_PATH` | Read by Quadrants, Genesis' kernel compiler: where compiled kernels are cached. In a multi-GPU run, `import eden` appends `rank_<LOCAL_RANK>` to it so that ranks do not contend for one cache lock. | `~/.cache/quadrants/qdcache/kernel_compilation_manager` |

## Multi-GPU runs

`torchrun` sets these; Eden reads them. See [Multi-GPU training](../guides/ddp.md).

| Variable | Effect |
|---|---|
| `WORLD_SIZE` | The number of ranks. Above 1, `import eden` pins each rank to one GPU, and `eden.init()` sets up NCCL. Default 1. |
| `LOCAL_RANK` | The rank's index on its node. `import eden` uses it to pick the rank's GPU from `CUDA_VISIBLE_DEVICES`. Default 0. |
| `RANK` | The global rank. Default 0. |
| `CUDA_VISIBLE_DEVICES` | Read and **rewritten** by `import eden` in a multi-GPU run: each rank keeps only its own GPU. A list shorter than the number of ranks on the node raises. |

In a multi-GPU run `import eden` also sets `QD_VISIBLE_DEVICE` and `TI_VISIBLE_DEVICE` to the rank's GPU. All of
this happens when `eden` is imported, which is why `eden` must be imported before `genesis` and `torch`.

## Console output

| Variable | Effect |
|---|---|
| `NO_COLOR` | Any non-empty value disables colors under the `auto` theme. |
| `FORCE_COLOR` | Any non-empty value enables colors under the `auto` theme, even when not writing to a terminal. |

## Rendering

These are read by Genesis' renderer.

| Variable | Effect |
|---|---|
| `PYOPENGL_PLATFORM` | The OpenGL platform for offscreen rendering: `egl` (the default on Linux), `osmesa` (software rendering), `cgl` (macOS) or `pyglet`. Must be set before `genesis` is imported. When it is unset on a Linux machine with no display and no EGL library, Genesis selects `osmesa` if it is installed. |
| `EGL_DEVICE_ID` | The GPU that EGL renders on. |

## Training and logging

| Variable | Effect |
|---|---|
| `WANDB_RUN_GROUP` | Read by Weights & Biases. `eden train --logger wandb` sets it to `--wandb_group` (default: the task name) unless it is already set. |

Weights & Biases reads its other variables too (`WANDB_API_KEY`, `WANDB_PROJECT`, ...); the project defaults to
`eden` through `RslRlOnPolicyRunnerOptions.wandb_project`.

## Policies and ONNX Runtime

Read when a policy runs through ONNX Runtime (`eden/extensions/policies/onnx.py`). Boolean variables treat `0`,
`false`, `no` and the empty string as off.

| Variable | Effect | Default |
|---|---|---|
| `ONNX_CPU` | On: use only the CPU execution provider. | off |
| `ONNX_INTRA_OP_THREADS` | Intra-op threads per ONNX session; a positive integer, anything else raises. | 1 |
| `TENSORRT_EP`, `ONNX_TRT`, `ORT_TRT` | Any of them on: use the TensorRT execution provider when it is available, ahead of CUDA. | off |
| `ORT_TRT_FP16` | TensorRT FP16. | on |
| `ORT_TRT_INT8` | TensorRT INT8. | off |
| `ORT_TRT_CACHE` | Cache TensorRT engines. | on |
| `ORT_TRT_CACHE_PATH` | Where TensorRT engines are cached (relative to the working directory). | `.ort_trt_cache` |
| `ORT_TRT_WORKSPACE` | TensorRT's maximum workspace size, in bytes. | TensorRT's own |
| `RESIDUAL_GAIN` | The OmniXtreme policy only: the gain on its residual correction. | 1.0 |

## Hardware and devices

| Variable | Effect |
|---|---|
| `GS_UNITREE_NETWORK_INTERFACE`, `UNITREE_NETWORK_INTERFACE`, `CYCLONEDDS_NETWORK_INTERFACE` | The network interface for Unitree DDS traffic, checked in that order. Unset, Eden picks an interface that is up. |
| `ROBOT_IP` | The LimX Oli's IP address, when the backend's `robot_ip` option is not set. Default `10.192.1.2`. |
| `WUJI_TELEMETRY_DISABLE` | Read by `wuji-sdk`. `1` turns off the background telemetry the SDK starts on import. Eden does not set it. |

## Extensions

| Variable | Effect |
|---|---|
| `GVHMR_CHECKPOINT_ROOT` | GVHMR's checkpoint directory for `eden pose gvhmr`, when `--checkpoint-root` is not given. Default `./inputs/checkpoints`. |
| `GRASPGEN_CHECKPOINT_DIR` | The directory GraspGen downloaded its models to (`grasp_gen download`); a relative `gripper_config` is resolved against it. |

## See also

- [Assets and the download cache](../getting-started/assets.md)
- [Command reference](cli.md)
- [Troubleshooting](../getting-started/troubleshooting.md)
