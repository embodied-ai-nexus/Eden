# Troubleshooting

Each entry gives the symptom, its cause, and a fix you can type. Most of Eden's own errors already name the fix in
their message, so read the last line of the traceback first; this page covers the problems whose cause lies
outside the message, in the machine, the environment or the install.

Run the commands from the directory that holds your virtual environment: the repository root for a clone, or your
project for a PyPI install.

## Check the install first

```bash
python -c "import eden, torch; print('eden', eden.__version__, eden.__file__); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
eden list
```

The first command should print an `eden` path inside the clone you are working in (for a source install) and
`available True` on a GPU machine. `eden list` should print a table of tasks. Include the output of both when you
ask for help.

## GPU, driver and Genesis

### `torch.cuda.is_available()` is `False`

The NVIDIA driver is not visible, or the torch wheel was built for a newer CUDA than the driver supports:

```bash
nvidia-smi                                                 # the driver works and lists your GPUs
python -c "import torch; print(torch.version.cuda)"        # the CUDA version torch was built for
```

`nvidia-smi` prints the newest CUDA version the driver supports in its top-right corner, and that must be at
least the version torch was built for. If it is lower, upgrade the driver. If `nvidia-smi` itself fails, the driver
is not installed or not loaded, and no change on the Python side fixes that.

To keep working without a GPU, use the CPU backend with a few environments:

```bash
eden train --task reacher --cpu --num_envs 64
```

### `Backend gpu not available on this machine. Falling back to CPU.`

Genesis tried every GPU backend (CUDA, then AMD, then Metal) and found none, so the run continues on the CPU, much
more slowly. On a machine with an NVIDIA GPU this is the same problem as the previous entry: torch cannot see the
GPU.

### `CUBLAS_STATUS_INVALID_VALUE` after `module load cuda`

On a cluster, `module load cuda-...` puts the system's CUDA libraries ahead of the ones torch ships with. A mixed
pair of cuBLAS and cuBLASLt then fails, typically in batched matrix products such as the IK solvers. Put torch's
own NVIDIA libraries first:

```bash
export LD_LIBRARY_PATH="$(ls -d .venv/lib/python*/site-packages/nvidia/*/lib | paste -sd:):${LD_LIBRARY_PATH:-}"
```

The SLURM template in [Multi-GPU training](../guides/ddp.md) already does this.

### The first run spends a long time before the first step

Genesis compiles its kernels the first time a scene is built and caches them under `~/.cache/quadrants/`, so later
runs start much faster. Upgrading Genesis invalidates the cache and the first run compiles again. Both are
expected.

If a run fails inside kernel compilation after an earlier run was interrupted, clear the cache and retry:

```bash
rm -rf ~/.cache/quadrants/qdcache
```

### Every multi-GPU rank lands on GPU 0, or `CUDA_VISIBLE_DEVICES has N entries but LOCAL_RANK=...`

`import eden` gives each rank its own GPU, by rewriting `CUDA_VISIBLE_DEVICES` from `LOCAL_RANK`, and it has to do
that before anything initializes CUDA. **Import `eden` before `genesis` and `torch`** in every script:

```python
import eden as en      # first: it pins this rank's GPU
import genesis as gs
import torch

en.init(backend=gs.gpu)   # en.init, not gs.init: it also sets up logging and the run directory
```

The `LOCAL_RANK` error means `CUDA_VISIBLE_DEVICES` lists fewer GPUs than the launch starts ranks. Make
`--nproc_per_node` match the number of visible GPUs. See [Multi-GPU training](../guides/ddp.md).

### An opaque CUDA `unknown error`, after `eden.init() called again`

Calling `eden.init()` twice in one process is a common cause of an unexplained CUDA `unknown error`, and the
second call also starts a new run directory. Eden warns when it happens:
`eden.init() called again in a process where Genesis is already initialized`. Call `eden.init()` exactly once per
process, and start a new process for each run.

## Installing and environments

### `pip install eden` installed something else

The PyPI name `eden` belongs to an unrelated project. Uninstall it and install `eden-world`:

```bash
pip uninstall eden
pip install eden-world
```

### `ModuleNotFoundError` for an optional package

The feature needs an extra you have not installed. Eden's message for a missing extra names both install commands,
for example `pip install "eden-world[vis]"`, or `uv sync --inexact --extra vis` in a checkout. The extras are
listed in [Installation](installation.md#optional-extras).

### A `uv sync` removed my extras

`uv sync` is exact: it installs the extras and groups you name and **uninstalls every other one**. After
`uv sync --all-extras --all-groups`, a later plain `uv sync`, or a sync that names a single extra without
`--inexact`, removes everything else, and the next `import gsplat` or `eden pico bridge` fails with
`ModuleNotFoundError`.

Repeat the flags you installed with, every time:

```bash
uv sync --all-extras --all-groups
```

Or keep what is installed and only add to it:

```bash
uv sync --inexact --extra docs
```

#### ...and now `import torch` fails with `libcudnn.so.9: cannot open shared object file`

Some extras put libraries in the same `site-packages/nvidia/` directory as torch's CUDA packages. Uninstalling them
can delete files that belong to `nvidia-cudnn-cu12`, `nvidia-nccl-cu12` and similar packages while their metadata
stays behind, so uv considers them installed and a new sync changes nothing. Find every package with missing files
and reinstall those:

```bash
broken=$(.venv/bin/python -c 'import importlib.metadata as m; print(" ".join("--reinstall-package " + d.metadata["Name"] for d in m.distributions() if any(not f.locate().exists() for f in (d.files or []) if f.suffix != ".pyc")))')
echo "$broken"                                  # e.g. --reinstall-package nvidia-cudnn-cu12 ...
uv sync --all-extras --all-groups $broken       # use the flags you normally sync with
```

If the list is empty and torch still fails to load, reinstall torch itself:

```bash
uv sync --all-extras --all-groups --reinstall-package torch
```

### `onnxruntime ... exposes no InferenceSession`

Both ONNX extras are installed and one of their wheels was uninstalled, which deletes files the other one shares.
Reinstall the wheel you want, last:

```bash
uv sync --inexact --extra onnx --reinstall-package onnxruntime                # CPU runtime
uv sync --inexact --extra onnx-gpu --reinstall-package onnxruntime-gpu        # CUDA runtime
```

With pip, `pip install --force-reinstall --no-deps "onnxruntime==<the installed version>"`, or the same for
`onnxruntime-gpu`. [Installation](installation.md#extras-with-special-install-rules) explains why the two clash.

### `pip install` of the `soma-retargeter` extra fails to resolve

That extra installs only with uv in a clone: `uv sync --inexact --extra soma-retargeter`. Its dependency pins a
`trimesh` version that conflicts with Genesis, and only the clone's `[tool.uv]` override reconciles the two.

### `import eden` resolves to another checkout

A clone installs Eden in editable mode, so `import eden` loads the source tree that the environment was last synced
from. Two clones or git worktrees sharing one virtual environment, `uv run --active`, or a stale `VIRTUAL_ENV`
variable all send the import to the wrong tree. Give each clone its own environment, and do not use `--active`:

```bash
env -u VIRTUAL_ENV uv sync --all-extras --all-groups
.venv/bin/python -c "import eden; print(eden.__file__)"
```

## Tasks and configs

### `No task named '...' found in task registry`

The message suggests the closest name and lists every task. For a task in your own package, the package must be
installed and named under the `eden.tasks` entry-point group of its `pyproject.toml`; see
[Your first task](first-task.md#where-the-task-lives).

### `Extra inputs are not permitted`

A config or option class received a field it does not have, usually a misspelling. Options refuse unknown fields on
purpose, so that a typo cannot be silently ignored. The validation error names the field and the class; fix the
spelling. To see the fields a term accepts, run `eden terms --grep <term name>`.

### A task builds from `TASK_REGISTRY.get(name)()` and then fails in `env.build()`

`get()` returns the bare config class, and instantiating it runs none of the task's mods. For a task whose robot or
data is chosen by a mod, the config constructs fine and then fails during the build, often on an empty name.
Construct registered tasks with `TASK_REGISTRY.build(name)`.

### `cannot register ... as '...': that key is taken by ...`

Your term's name gives the same registry key as another term, possibly one of Eden's built-ins. Term keys are the
term's name in snake case, and `register()` does not take a `name=`. Rename your function or class.

### `Task mod flag(s) ... cannot apply to a replay`

You passed a variant flag to `eden inference`, `eden export` or `eden deploy` together with a checkpoint that has a
`config.json` beside it. The saved config is what the policy was trained with, so it decides the variant. Drop the
flag, or pass a saved config of your own with `--config`.

### `--log-theme came after the subcommand`

`--log-theme` belongs to `eden` itself, so it goes before the subcommand: `eden --log-theme human train ...`. Or
set `EDEN_LOG_THEME=human` for the whole shell.

### The console output has no color

That is the default. Eden's console theme defaults to `agent`, plain text without ANSI colors or the greeting box,
because most of its output is read from log files. Use `eden --log-theme human <command>` or
`export EDEN_LOG_THEME=human` for colored output, or `auto` to color only when writing to a terminal.

## Assets

See [Assets and the download cache](assets.md#when-a-download-fails) for download errors. Two more errors come from
the asset root itself:

- **`EDEN_ASSETS_DIR=... is a relative path`**: export an absolute path. The message prints the one it would have
  meant.
- **`Cannot download assets into ...`** (a `PermissionError`): the directory is not writable. The message says
  whether `EDEN_ASSETS_DIR` chose it; point `EDEN_ASSETS_DIR` at a directory you can write.

## Headless machines: servers, SSH, containers

Training never opens a window; only `--vis` (or `--show_viewer`) does. On a machine without a display, leave
`--vis` off and record a video instead:

```bash
eden inference --task reacher --checkpoint <path/to/model.pt> --rec    # writes an .mp4 into the run directory
```

A viewer opened without a display fails with `No display detected`.

Cameras render offscreen through EGL on Linux. When the EGL library is missing and OSMesa is installed, Genesis
falls back to OSMesa, a software rasterizer that is much slower. Install EGL (Debian and Ubuntu shown) and select it
explicitly:

```bash
sudo apt-get install -y libegl1
export PYOPENGL_PLATFORM=egl
```

Over SSH with X forwarding, a `DISPLAY` that points at a forwarded or dead X server makes the viewer and offscreen
rendering fail or crawl. Unset it for headless runs:

```bash
unset DISPLAY
export PYOPENGL_PLATFORM=egl
```

`PYOPENGL_PLATFORM` must be set before `genesis` is imported, so export it in the shell rather than setting it
inside a script after its imports. On a machine with several GPUs, `EGL_DEVICE_ID` picks the one EGL renders on.

## Still stuck?

Open an issue on [GitHub](https://github.com/embodied-ai-nexus/Eden/issues) with the exact command, the full
traceback, and the output of the [install check](#check-the-install-first).

## See also

- [Installation](installation.md)
- [Environment variables](../reference/environment-variables.md)
- [Error messages](../guides/error-messages.md): how Eden's errors are written, for contributors.
