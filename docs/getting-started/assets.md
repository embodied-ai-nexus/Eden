# Assets and the download cache

Eden ships code, not meshes. Robot models, objects, scenes, published policy checkpoints and motion datasets are
downloaded from the [Hugging Face Hub](https://huggingface.co/) the first time something needs them, and read from
disk after that. This page says where those files go, how to move them, how to run without network access, and how
to fetch everything in advance for a cluster.

## Where files go

| What | Where, by default | Moved by |
|---|---|---|
| Robots, objects, scenes, splat scans, benchmark-suite assets, published policy checkpoints, MuJoCo models for `eden deploy --mujoco` | `~/.cache/eden/assets/` | `EDEN_ASSETS_DIR` |
| Motion datasets (`eden.datasets`: AMASS, LAFAN1, PHUMA, Weave, ...) | the Hugging Face cache, `~/.cache/huggingface/hub/` | `HF_HOME`, or the dataset's `raw_data_path` |
| Per-task sample clips, processed motion caches, generated meshes and other files Eden writes | `~/.cache/eden/` | `XDG_CACHE_HOME` |
| Genesis' compiled kernels | `~/.cache/quadrants/` | (Genesis) |

`~/.cache` is `$XDG_CACHE_HOME` when that variable is set. The asset root is shared by every virtual environment and
reinstall of Eden on the machine, and needs no root access to write.

To print the asset root of the Eden you are running:

```bash
python -c "from eden.utils.misc import get_assets_dir; print(get_assets_dir())"
```

Assets are fetched **file by file**, not as whole repositories. A robot's URDF or MJCF file comes first, then the
meshes, materials and textures it references, so the cache holds only what the tasks you built actually used.
Families of objects keep their files in a subdirectory of the asset root: YCB objects in `ycb/`, LIBERO in `libero/`,
robosuite in `robosuite/`, published policies in `policies/`, and so on.

Deleting `~/.cache/quadrants/` is safe: nothing in it is downloaded, but the next run compiles its kernels again.

## Moving the asset root

Set `EDEN_ASSETS_DIR` to an **absolute** path to put every asset somewhere else, for example on a larger disk:

```bash
export EDEN_ASSETS_DIR=/data/eden_assets
```

It is read every time a path is resolved, so it applies to everything the process downloads after you set it, and
to every entity whose `local_dir` is unset. A relative path raises a `ValueError` that prints the absolute path to
export instead, because a relative one would point somewhere different whenever the working directory changed.

An entity's `local_dir` option, when set, overrides the root for that entity alone. `EDEN_ASSETS_DIR` does not move
an explicit `local_dir`, and a config saved on another machine carries that machine's explicit paths; leave
`local_dir` unset unless you mean to pin a location.

If the asset root cannot be written, the first download fails with a `PermissionError` that names the directory and
says whether `EDEN_ASSETS_DIR` chose it. A read-only cache that already holds every file keeps working.

## Which Hub repositories Eden reads

All of them are dataset repositories. The repository an entity reads from is its `registry` and `dataset` options
(`<registry>/<dataset>`), so you can point any entity at a mirror. The defaults:

| Repository | Holds | Read by |
|---|---|---|
| `Kashu7100/eden_assets` | robots, grippers, hands, scene assets, splat scans | robot and most object options (`eden/options/entities.py`) |
| `Kashu7100/eden_policies` | published policy checkpoints | `eden inference --policy` and `eden deploy --policy` (`eden/extensions/policies/`) |
| `Kashu7100/eden_example` | small per-task sample files | tasks with a default sample clip |
| `Kashu7100/mujoco_menagerie` | MuJoCo Menagerie and vendor MJCF models | `eden deploy --mujoco --menagerie` |
| `Kashu7100/eden_libero`, `eden_robosuite`, `eden_simpler`, `eden_polaris`, `eden_robocasa`, `eden_robocasa365`, `eden_lbm_eval`, `eden_robolab`, `eden_calvin` | benchmark-suite scenes and objects | `eden/options/objects/` and the matching tasks |
| `Kashu7100/eden_ycb`, `eden_objaverse`, `eden_dextoolbench`, `eden_arctic`, `eden_omomo_holosoma` | object meshes | `eden/options/objects/` |
| `Kashu7100/eden_art` | reference pictures for the painting tasks | `eden/tasks/art/painting/` |
| `haosulab/ReplicaCAD` | ReplicaCAD scenes | `eden/options/objects/replica_cad.py` |
| `ember-lab-berkeley/AMASS_Retargeted_for_G1`, `lvhaidong/LAFAN1_Retargeting_Dataset`, `DAVIAN-Robotics/PHUMA`, `bones-studio/seed`, `appolyn/Weave`, `jose-barreiros-tri/hhi-assist`, `exptech/g1-moves` | retargeted motion data | `eden/datasets/`, the `beyondmimic_moves` policy |
| `UW-Lab/uwlab-assets` (pinned revision) | OmniReset reset states and checkpoints | the `omnireset` tasks and policy |

A failed fetch names the repository and the file in its error, so you can look the file up on the Hub directly.

## Running offline

Once a machine holds everything a task needs, it runs with no network access. A file that is already on disk is
read without contacting the Hub. Set `HF_HUB_OFFLINE=1` so that a missing file fails at once, with a message naming
it, instead of waiting for a network timeout:

```bash
export HF_HUB_OFFLINE=1
eden train --task reacher
```

The error for a missing file reads `... could not be fetched: the Hub repo ... cannot be reached ... Offline
(HF_HUB_OFFLINE=1) or no network?`. Fetch that file once on a machine with network access, then copy or share the
cache as described below.

## Pre-fetching for a cluster

Compute nodes often have no internet access, or a home directory with a small quota that each node sees
separately. Fetch on a machine with network access (a login node) into storage every node mounts:

1. **Put the caches on shared storage.** Point the asset root and the Hugging Face cache at a shared path, and the
   Eden cache too if you use motion datasets or sample clips:

    ```bash
    export EDEN_ASSETS_DIR=/shared/$USER/eden_assets
    export HF_HOME=/shared/$USER/huggingface
    export XDG_CACHE_HOME=/shared/$USER/cache
    ```

2. **Build every task you will run, once.** Building a task fetches every file it uses, and a CPU build with one
   environment is enough:

    ```bash
    eden smoke --task velocity_tracking_flat --cpu --num_envs 1
    eden smoke --task robosuite_lift --cpu --num_envs 1
    ```

    Pass the same variant flags the job will use (`--robot`, `--task_name`, ...), because they can change which
    assets load. For a published policy, roll it out for one step the same way:
    `eden inference --policy <name> --cpu --steps 1`.

3. **Run the job offline.** Export the same variables in the job script, plus `HF_HUB_OFFLINE=1`.

Several processes, such as the ranks of a multi-GPU run, can share one cache safely: `huggingface_hub` locks each
file while it downloads, so one process fetches it and the others wait for the finished file.

To mirror a whole repository instead, use the `hf` command that comes with `huggingface_hub`. Asset repositories
are large, so prefer step 2 unless you need everything:

```bash
hf download Kashu7100/eden_assets --repo-type dataset --local-dir "$EDEN_ASSETS_DIR"
```

<!-- TODO(material): approximate disk size of the asset cache after building the common tasks. -->

## When a download fails

The error names the file and the repository. The common cases:

- **`... the Hub repo ... cannot be reached ...`**: the machine cannot reach the Hub, or `HF_HUB_OFFLINE=1` is set.
  Behind a proxy, export it (`export HTTPS_PROXY=http://proxy.example.com:3128`). On a node without internet,
  pre-fetch as above. If the Hub rate-limits you (HTTP 429), log in so requests are authenticated:
  `hf auth login`.
- **`Asset ... is not a file in the Hub repo ...`**: the path does not exist in that repository. Asset paths are
  full repository paths (`models/eggplant/object.urdf`), never a bare file name, and the message lists the
  repository paths that end in the name you gave.
- **A warning that a dependency was not delivered**: a model references a file its repository lacks, and loads
  without it (a missing texture renders untextured). Report it with the warning's text. Copying the file into your
  cache fixes only your machine.
- **An interrupted download left a broken file**: delete the file named in the error and run again, so that it is
  fetched again.

[Troubleshooting](troubleshooting.md) covers the rest.

## See also

- [Environment variables](../reference/environment-variables.md)
- [Multi-GPU training](../guides/ddp.md): the SLURM job template sets `EDEN_ASSETS_DIR`.
- [Motion datasets](../guides/motion-datasets.md): `raw_data_path` and the processed caches.
