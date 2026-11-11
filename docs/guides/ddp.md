# Multi-GPU Training (DDP)

Eden trains on several GPUs, and several nodes, with PyTorch Distributed Data Parallel: one process (rank) per GPU,
each simulating its own batch of envs and averaging gradients with the others. This page covers launching a run with
`torchrun`, what Eden does for each rank, the import-order rule DDP depends on, and a job script for SLURM clusters.

## Quick start

```bash
# 4 GPUs on one machine; --num_envs is per GPU, so this run simulates 4 x 4096 envs
torchrun --standalone --nproc_per_node=4 -m eden train --task velocity_tracking_flat --num_envs 4096
```

In a checkout, prefix the command with `uv run` (`uv run torchrun ...`). A training script of your own launches the
same way: `torchrun --standalone --nproc_per_node=4 my_train.py`. With `--nproc_per_node=1` (or plain `eden train`)
the run is an ordinary single-GPU run; an algorithm that cannot train distributed refuses only when `WORLD_SIZE > 1`.

## What Eden does for each rank

`torchrun` sets `WORLD_SIZE`, `RANK` and `LOCAL_RANK` for every process. When `WORLD_SIZE > 1`, Eden:

1. **Isolates one GPU per rank, at `import eden`.** Before anything initializes CUDA, it sets `CUDA_VISIBLE_DEVICES`
   (and Quadrants' `QD_VISIBLE_DEVICE` / `TI_VISIBLE_DEVICE`) to the rank's GPU: entry `LOCAL_RANK` of an existing
   `CUDA_VISIBLE_DEVICES`, or GPU `LOCAL_RANK` when none is set. Every rank then sees its GPU as `cuda:0`, and the
   runner's `device="auto"` resolves to it.
2. **Gives each rank its own kernel cache.** Each rank compiles into `<cache>/rank_<LOCAL_RANK>` (under
   `QD_OFFLINE_CACHE_FILE_PATH`, by default `~/.cache/quadrants/qdcache/kernel_compilation_manager`), because ranks
   sharing one cache contend for its lock file and crash.
3. **Creates the NCCL process group in `en.init()`**, right after Genesis initializes and before any scene is built.
4. **Shares one run directory.** Rank 0 names it and broadcasts the name, so every rank logs under the same
   `logs/<task>/<run>/`.
5. **Quiets the other ranks.** Genesis logs at warning level on every rank but 0.

`eden train` builds the DDP variant of the runner (`DDPOnPolicyRunner`, `DDPDistillationRunner`, or FlashSAC's), which
broadcasts the parameters at the start and averages gradients every update.

!!! danger "Import Eden before Genesis and torch"
    GPU isolation runs when `eden` is imported, so `import eden` must come before anything that can initialize CUDA.
    Importing `torch` or `genesis` first lets CUDA see every GPU, and the ranks end up on the same device.

    ```python
    # Correct
    import eden as en
    import genesis as gs
    import torch

    # Wrong: CUDA may already have initialized before Eden can isolate the GPU
    import torch
    import eden as en
    ```

    `import eden` itself imports neither Genesis nor torch (both load in `en.init()`), so any order that starts with
    `import eden` is correct. A module that needs `genesis` must import it itself rather than rely on Eden having
    pulled it in.

## What changes in a distributed run

- **`num_envs` is per rank.** The total is `num_envs × WORLD_SIZE`.
- **Seeds differ per rank.** Rank `r` seeds with `runner_options.seed + r`, so ranks sample different randomization.
  `config.json` and `run.json` record the base seed and `world_size`, which together reproduce every rank. See
  [Seeding](training-runs.md#seeding).
- **Rank 0 writes the run files.** `config.json`, `run.json` and `progress.json` are written by rank 0 only. In a
  script of your own, guard file writes the same way (`en.distributed.is_main_process()`, or the
  `en.distributed.main_only` decorator).
- **Dataset caches are prepared once.** A motion dataset with no cache yet is downloaded and processed by rank 0
  while the other ranks wait at a barrier, then every rank loads the shared cache.

Some learners cannot train distributed, and `eden train` refuses them before the build rather than letting each
rank train its own copy:

- an algorithm class with a `distributed_unsupported` message (the RMA teacher's `RmaPPO`);
- the RMA depth-distillation runner (`extreme_parkour_student`), which is single-GPU;
- a custom runner (`runner_options.class_name` as an import path) that does not compose `DDPRunnerMixin`.

Launch those without `torchrun`.

## Distributed helpers

`en.distributed` (`eden.utils.distributed`) holds the rank-aware helpers Eden uses itself:

| Function | Returns / does |
|----------|----------------|
| `is_distributed()` | `True` when `WORLD_SIZE > 1` |
| `is_main_process()` | `True` on global rank 0 |
| `get_global_rank()` / `get_local_rank()` / `get_world_size()` | `RANK`, `LOCAL_RANK`, `WORLD_SIZE` |
| `barrier()` | Synchronize all ranks; a no-op without a process group |
| `main_only` | Decorator: run the function on rank 0 only (others return `None`) |
| `setup(backend="nccl")` / `cleanup()` | Create / destroy the process group; `en.init()` already calls `setup` |

## Running on a SLURM cluster

One job script covers one or many GPUs and nodes: `srun` starts one `torchrun` per node, and `torchrun` starts one
rank per GPU. Fill in the lines marked *site-specific*, create the `slurm/` directory (SLURM will not create it), and
`sbatch` the script from the repository root.

```bash
#!/bin/bash
#SBATCH --job-name=eden
#SBATCH --nodes=1                  # >1 for multi-node; nothing else changes
#SBATCH --ntasks-per-node=1        # one torchrun per node; it spawns the per-GPU ranks
#SBATCH --gres=gpu:4               # site-specific GPU type, if your cluster needs one; the count must equal NGPU
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=<partition>    # site-specific
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
set -euo pipefail
NGPU=4

cd "$SLURM_SUBMIT_DIR"
# module load cuda               # site-specific; drop it if the cluster has no modules
# `module load` puts the system CUDA ahead of torch's bundled libraries, and a mixed cuBLAS / cuBLASLt pair fails
# batched-gemm IK with CUBLAS_STATUS_INVALID_VALUE: put torch's own NVIDIA libraries first.
export LD_LIBRARY_PATH="$(ls -d .venv/lib/python*/site-packages/nvidia/*/lib | paste -sd:):${LD_LIBRARY_PATH:-}"
# Assets download on first use into ~/.cache/eden/assets, which on a cluster is often a small home quota or a
# node-local disk (every node then fetches its own copy). Point it at a shared filesystem with tens of GB free.
export EDEN_ASSETS_DIR=<shared filesystem>/eden_assets   # site-specific

# Several nodes: node 0 hosts the rendezvous store, on a port derived from the job id.
export RDZV_ENDPOINT="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1):$((20000 + SLURM_JOB_ID % 10000))"

srun --kill-on-bad-exit=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=none bash -c '
    if [ "$SLURM_NNODES" -eq 1 ]; then
        rdzv=(--standalone)
    else
        rdzv=(--nnodes="$SLURM_NNODES" --rdzv_id="$SLURM_JOB_ID" --rdzv_backend=c10d
              --rdzv_endpoint="$RDZV_ENDPOINT" --rdzv_conf=is_host=$((SLURM_NODEID == 0)))
    fi
    exec uv run --no-sync torchrun "${rdzv[@]}" "$@"
' torchrun --nproc_per_node="$NGPU" \
    -m eden train --task <task> --num_envs 4096   # or a script: drop -m and give its path
```

Why the script looks the way it does:

- **The rendezvous must belong to this job alone.** When `torchrun` has to guess whether it hosts the rendezvous
  store and finds the port taken, it does not fail: it silently joins whatever store holds the port (another job's)
  and dies mid-run when that job ends. So one node uses `--standalone` (the OS picks a free port), and several nodes
  name the host explicitly (`is_host`), which turns a taken port into an `EADDRINUSE` at launch.
- **`--kill-on-bad-exit=1`** stops the other nodes when one fails, instead of leaving them waiting out the join
  timeout.
- **`--cpus-per-task` is repeated on `srun`**, because Slurm 22.05 to 23.02 do not pass `sbatch`'s CPU count on to
  `srun`, which would confine the step, and every rank in it, to one CPU.
- **`uv run --no-sync`**, so the ranks do not race to rewrite one virtual environment.

Track the job with `squeue -j <id>` / `sacct -j <id>` and read `slurm/eden_<id>.out`. The run's logs and checkpoints
land under `logs/` as usual, and `progress.json` there tells you how far it got (see
[The run directory](training-runs.md#the-run-directory)).

## See also

- [Training Runs](training-runs.md): the run directory, checkpoints, seeding
- [Getting assets](../getting-started/assets.md): `EDEN_ASSETS_DIR` and the asset cache
