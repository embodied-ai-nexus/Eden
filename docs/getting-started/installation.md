# Installation

This page installs Eden, either from PyPI into your own project or from a clone of the repository, and explains
the optional extras that add rendering, teleoperation, retargeting and other features. It ends with a check that
the install works.

## Requirements

- **Python 3.12 or 3.13.** `pyproject.toml` declares `requires-python = ">=3.12,<3.14"`; Eden has not been run on
  Python 3.14 yet.
- **Linux x86_64 with an NVIDIA GPU** is the primary platform, and the one GPU-parallel training is built for.
- **macOS on Apple silicon and Windows AMD64** are supported on a best-effort basis: the dependency resolution is
  required to stay installable on both, but a few extras need more than they have. Nyx path tracing has no macOS
  build, `onnx-gpu` has no macOS wheels, Gaussian-splat rendering needs CUDA, and the Wuji Glove SDK is
  Linux-only.

Eden runs on the CPU too (every command takes `--cpu`), which is enough to build tasks and try things with a few
environments, but training at scale needs a GPU.

## Install from PyPI

Eden is published on PyPI as **`eden-world`**. The import name is `eden`, and the command it installs is `eden`:

=== "pip"

    ```bash
    pip install eden-world
    pip install "eden-world[vis]"        # with an optional extra
    ```

=== "uv project"

    ```bash
    uv add eden-world
    uv add "eden-world[vis,onnx]"        # with optional extras
    ```

!!! warning "`pip install eden` installs a different package"
    The PyPI name `eden` belongs to an unrelated project. Always install `eden-world`, and write extras as
    `eden-world[<extra>]`. Only the distribution name differs: you still `import eden` and run `eden`.

A PyPI install carries the `eden` package only. The `examples/` directory, the tests and these docs live in the
repository, so `eden examples list` and the example scripts need a clone.

## Install from a clone

A clone is the way to run the examples, work on Eden itself, or follow `main`. [uv](https://docs.astral.sh/uv/)
creates the virtual environment and installs Eden into it in editable mode:

```bash
git clone https://github.com/embodied-ai-nexus/Eden.git
cd Eden
uv sync                              # the core, plus the dev tools (ruff, pre-commit, pytest)
source .venv/bin/activate            # Windows: .venv\Scripts\activate
```

To install every extra instead:

```bash
uv sync --all-extras --all-groups
```

`--all-groups` matters: two extras need a package that is not on PyPI, and those packages come from dependency
groups (see [Extras that need a git package](#extras-that-need-a-git-package)).

To add one extra to what is already installed:

```bash
uv sync --inexact --extra vis
```

!!! warning "`uv sync` removes what you do not name"
    Without `--inexact`, `uv sync` is exact: it uninstalls every extra and group the command does not name. After
    `uv sync --all-extras --all-groups`, a later sync that names only `vis` (and not `--inexact`) removes every
    other extra. Use `--inexact` to add an extra, or repeat the full set of flags every time you sync.
    [Troubleshooting](troubleshooting.md#a-uv-sync-removed-my-extras) covers repairing an environment this has
    already happened to.

With plain pip in a clone, `pip install -e ".[vis]"` works for most extras, but pip ignores the
`[tool.uv]` settings in `pyproject.toml`, so the `soma-retargeter` extra cannot be installed that way.

### Running `eden`

Inside an activated virtual environment, `eden` is on your `PATH`. Without activating it, prefix every command with
`uv run` (in a clone) or call the module directly:

```bash
uv run eden list          # in a clone, without activating .venv
python -m eden list       # any environment where Eden is installed
```

The rest of these docs write commands as `eden <command>`.

## Optional extras

The core install covers building tasks, training with rsl_rl, inference, export to TorchScript and the asset
cache. Everything else is an extra. Each row gives the packages it adds, read from
`[project.optional-dependencies]` in `pyproject.toml`.

| Extra | Adds | Use it for |
|---|---|---|
| `vis` | plyfile, gsplat, viser, imgui-bundle | Viser visualization, `eden arrange`, Gaussian-splat rendering |
| `batch-splat` | plyfile, gsplat | Batched Gaussian-splat rendering in `SplatEnvBase` / `SplatRLEnvBase` |
| `nyx` | gs-nyx, gs-nyx-plugin | The Nyx path tracer (`NyxSplatEnvBase`); not on macOS |
| `onnx` | onnx, onnxruntime | Running the published ONNX policies on the CPU runtime |
| `onnx-gpu` | onnx, onnxruntime-gpu | The same on the CUDA runtime; not on macOS |
| `rl-games` | rl-games | Training tasks whose `runner_options` select rl_games |
| `sb3` | stable-baselines3 | Stable-Baselines3 through the gymnasium vector-env wrapper |
| `skrl` | skrl | skrl through the gymnasium vector-env wrapper |
| `flashsac` | rsl-rl-lib >= 5.4.2, rsl-rl-flashsac (git) | The FlashSAC off-policy learner |
| `policy-client` | pyzmq, msgpack, msgpack-numpy, websockets, json-numpy, pillow | Clients for server-backed policies (VLAs) |
| `pose` | eden-gvhmr | Motion from video: `eden pose gvhmr` |
| `soma-retargeter` | eden-soma-retargeter, py-soma-x | `eden retarget soma`; **uv in a clone only** |
| `ot-retargeter` | libigl, proxsuite, scikit-image, threadpoolctl | `eden retarget otretarget` |
| `dex-retargeter` | nlopt | `eden retarget dex` |
| `dex-retargeter-example` | nlopt, mediapipe, opencv-python, tyro | The video-based hand-retargeting examples |
| `flashdexretarget` | warp-lang | `eden retarget flashdex` |
| `graspgen` | eden-grasp-gen | GraspGen grasp sampling |
| `graspgenx` | eden-graspgenx | GraspGenX grasp sampling |
| `gello` | pyzmq | GELLO leader-arm teleoperation (`eden gello`) |
| `metaquest` | params_proto, pyzmq, televuer (git) | Meta Quest WebXR teleoperation (`eden metaquest bridge`) |
| `pico` | av, pyzmq | PICO teleoperation (`eden pico bridge`); see the note below |
| `sonic` | msgpack, pynput, pyzmq | SONIC whole-body teleoperation (`eden teleop sonic`) |
| `sonic-viz` | pyvista | The SONIC operator's `--vis-vr3pt` view |
| `wuji` | wuji-sdk | Wuji Glove teleoperation (`eden wuji`); Linux only |
| `franka` | websockets | The Franka Panda hardware stack; see the note below |
| `docs` | mkdocs-material, mkdocstrings, mkdocs-section-index | Building these docs |
| `all` | plyfile, gsplat, viser, pynput, inputs, cvxpy, pyzmq | A visualization and input-device bundle |

Despite its name, `all` is not every extra. To install genuinely everything in a clone, use
`uv sync --all-extras --all-groups`.

### Extras with special install rules

**`onnx` and `onnx-gpu` exclude each other.** Their runtime wheels, `onnxruntime` and `onnxruntime-gpu`, both own
the `onnxruntime/` directory, so whichever is installed last provides it. Pick one. `uv sync --all-extras --all-groups`
installs both, and the CPU wheel usually ends up on top; to use the GPU wheel, reinstall it last:

```bash
uv sync --inexact --extra onnx-gpu --reinstall-package onnxruntime-gpu
```

Do not uninstall either wheel to fix this: removing one deletes files the other needs. If the GPU runtime falls
back to the CPU, a policy still runs, only slower, and Eden warns once.

**`soma-retargeter` installs only with uv, in a clone:** `uv sync --inexact --extra soma-retargeter`.
`eden-soma-retargeter` 0.4.0 pins `trimesh==4.11.3` while Genesis needs `trimesh>=5`, and only the clone's
`[tool.uv] override-dependencies` reconciles the two. pip ignores that table, so a pip install of this extra fails
to resolve.

**`batch-splat` compiles CUDA code on first use.** gsplat JIT-compiles a CUDA extension the first time it is
imported, so the CUDA toolkit (`nvcc`) has to be installed.

**`graspgen` is under a non-commercial license.** GraspGen ships under NVIDIA's non-commercial license; GraspGenX
(`graspgenx`) is Apache-2.0. Both download their models on first use (`grasp_gen download` for GraspGen).

**`pose` needs separate checkpoints.** GVHMR's checkpoints are a separate download (`gvhmr-download-checkpoints`),
and the SMPL and SMPL-X body models are licence-gated, so you obtain them yourself.

### Extras that need a git package

Two extras depend on a package that is not on PyPI. PyPI refuses to publish a package whose requirements point at
a git URL, so the pinned git requirement lives in a dependency group named after the extra:

| Extra | Git package | In a clone |
|---|---|---|
| `metaquest` | televuer | `uv sync --inexact --extra metaquest --group metaquest-git` |
| `flashsac` | rsl-rl-flashsac | `uv sync --inexact --extra flashsac --group flashsac-git` |

With pip, install the extra and the pinned git requirement in one command. When the package is missing, the error
Eden raises prints that command in full; the pins themselves are
`eden.utils.distribution.UNPUBLISHED_REQUIREMENTS`.

### Hardware SDKs Eden cannot declare

Some device SDKs cannot be a dependency at all:

- **PICO**: the `pico` extra covers Eden's side of the link only. Tracking also needs the `xrobotoolkit_sdk`
  extension module, which you build from source against the XRoboToolkit PC Service. See
  `examples/recipes/teleop/pico/README.md` in the repository.
- **Franka Panda**: `panda-python` wheels ship inside a zip per libfranka version on the panda-py GitHub releases,
  so you install the one that matches your robot's firmware by hand. The comment above the `franka` extra in
  `pyproject.toml` gives the commands for libfranka 0.8.0.
- **Wuji Glove**: `wuji-sdk` publishes Linux wheels only (x86_64 and aarch64). Off Linux the `wuji` extra installs
  nothing.

!!! note "The Wuji SDK sends telemetry"
    Importing `wuji_sdk` starts a background worker that batches anonymous usage events to a third-party analytics
    host. Set `WUJI_TELEMETRY_DISABLE=1` in your environment to turn it off. Eden does not set it for you.

## GPU and CUDA

Eden installs PyTorch from PyPI, whose Linux wheels bundle their own CUDA libraries. What your machine needs is an
NVIDIA driver new enough for the CUDA version that torch was built for. Compare the two:

```bash
nvidia-smi                                               # top right: the newest CUDA the driver supports
python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
```

The driver's version must be at least torch's, and the second command should print `True`.

When a script or command asks for the GPU (`gs.gpu`, the default of every `eden` command), Genesis tries CUDA,
then AMD GPUs, then Metal on macOS, and falls back to the CPU with the warning
`Backend gpu not available on this machine. Falling back to CPU.` If you see that warning on a machine with an
NVIDIA GPU, torch cannot see the GPU; see [Troubleshooting](troubleshooting.md#torchcudais_available-is-false).

The first run of a task compiles Genesis' kernels, which takes a while; the compiled kernels are cached under
`~/.cache/quadrants/` and later runs start much faster.

## Check the install

```bash
python -c "import eden; print(eden.__version__, eden.__file__)"
eden --help
eden list
```

The first line prints the installed version (for example `0.1.0`) and where `eden` is imported from: a path
inside your clone for a source install. `eden list` prints a table of every registered task. Neither command
builds a scene, so both finish in a few seconds.

To check that simulation works too, build a task and step it twice:

```bash
eden smoke --task reacher             # GPU
eden smoke --task reacher --cpu       # no GPU
```

It ends with `✅ smoke OK: built 'reacher' with 2 env(s) and ran 2 step(s).` The first time, it also downloads the
robot's assets ([Assets](assets.md) explains where they go).

## Next steps

- [Quick start](quickstart.md): train and watch your first policy.
- [Troubleshooting](troubleshooting.md): if any of the checks above failed.
- [Contributing](../contributing/index.md): the dev setup, pre-commit hooks and tests, if you will change Eden
  itself.

## See also

- [Assets and the download cache](assets.md)
- [Environment variables](../reference/environment-variables.md)
- [Command reference](../reference/cli.md)
