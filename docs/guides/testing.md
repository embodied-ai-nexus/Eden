# Testing

This page is for anyone writing or reviewing a test in Eden: where a test goes, what it may build and what it may
fake, what it should assert, how to keep the cost of Genesis scene builds down, and how to run the suite. Eden's
suite is written **Detroit style** (classicist): a test drives real Eden objects and asserts on what they produce.
It does not replace Eden's own collaborators with stand-ins, and it does not check which calls were made.

## Layout

`tests/` mirrors `eden/`. The tests of `eden/managers/terms/rewards/locomotion.py` live in
`tests/managers/terms/rewards/test_locomotion.py`; a task's end-to-end tests live under `tests/tasks/<task>/`. Every
directory is a package (an `__init__.py` with a one-line docstring), so two `test_base.py` files in different
directories do not collide, and `tests` itself is a regular package so that a dependency shipping its own top-level
`tests` package cannot shadow it. `tests/test_suite_layout.py` enforces both: a test file directly under `tests/`
must mirror a top-level `eden` module (or be one of the few root files it lists).

| Where | What goes there |
|---|---|
| `tests/<mirror of the eden path>/test_<module>.py` | tests of that module's public behaviour |
| `tests/<dir>/conftest.py` | fixtures and helpers shared by that directory, e.g. its module-scoped scenes |
| `tests/conftest.py` | suite-wide fixtures (below) and the marker declarations |
| `tests/support/` | helpers several directories share: `asserts` (tensor comparisons), `envs.build_env`, `ddp.ddp_spawn`, `motion` (synthetic motion clips and mimic envs), `deploy` (`LoopbackDeployment` and an in-memory robot wire), `policy_servers` (local fake policy servers), `hand_object_releases` (synthetic TACO / OakInk2 / HOT3D releases), `soma_retargeter` (a stand-in `eden-soma-retargeter` + py-soma-x install that shadows a real one), and the CUDA probes described under [Cost](#cost) |
| `tests/data/` | small golden files a test compares against: `.npz` arrays, `mochi/` solver goldens, and the server-backed policies' wire records in `policy_client_goldens/*.json` |

Name a file after the module it tests, never after an issue or a pull request. Name a test after the behaviour it
pins (`test_fell_over_fires_past_limit_angle`). If a regression test needs to say which bug it guards, put that in
one docstring line; the history belongs in git.

Never import from another test module (`from tests.managers.test_x import _helper`). A helper two files share goes
in their common `conftest.py`, or in `tests/support/` if it serves the whole suite. Fixtures in a `conftest.py` reach
the tests below it automatically. Import plain functions and classes from it explicitly
(`from tests.extensions.deployment.conftest import robot_env`): every test directory is a package, so this resolves
to the same module pytest loaded.

### Suite-wide fixtures

`tests/conftest.py` calls `eden.init(debug=True)` once per session, which initializes Genesis on the CPU backend, and
provides fixtures for the things a test would otherwise get subtly wrong by hand:

| Fixture | Use it for |
|---|---|
| `eden_caplog` | asserting on a log line from `en.logger`. Plain `caplog` sees nothing from it: the `eden` logger does not propagate to the root logger, so a plain-`caplog` assertion passes vacuously or fails while the message is in the captured output. |
| `user_assets_dir` | a test that downloads through a default `local_dir`: it unsets `EDEN_ASSETS_DIR` and points `XDG_CACHE_HOME` into `tmp_path`. |
| `task_registry_alias` | registering an extra name on `TASK_REGISTRY` for one test and removing it afterwards. |
| `task_package` | writing an importable package of task files for a registry-discovery test. |
| `isolated_task_registry` | a test that changes what the task registry's scan sees, restored afterwards. |
| `count_cuda_syncs` | pinning a hot path's GPU→CPU sync budget (skips without CUDA). |

## Real collaborators

Build the real thing: options, managers, terms, entities, and envs from built Genesis scenes. Adapted from
`tests/managers/terms/conftest.py` and `tests/managers/terms/rewards/test_locomotion.py`:

```python
import pytest
import torch

from eden.managers.terms.rewards.locomotion import base_height_l2
from eden.options import BoxOptions, EnvOptions, SceneOptions
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env


@pytest.fixture(scope="module")
def env():
    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=2),
            scene_options=SceneOptions(box=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(3.0, 3.0, 2.0))),
        )
    )


def test_base_height_l2_squares_the_error_of_the_world_height(env):
    torch.testing.assert_close(
        base_height_l2(env, entity_name="box", target_height=0.5),
        torch.full((2,), (2.0 - 0.5) ** 2),
    )
```

`build_env(config, reset=False, eval_mode=False)` picks the env class the config pairs with, builds it with the
viewer off, and optionally resets it (an RL env needs `reset=True` before `step()`).

These are off limits for Eden's own types:

- a `SimpleNamespace` or hand-written class standing in for an env, entity, manager, term or config;
- `MagicMock` / `Mock` / `patch` on anything under `eden.`;
- `monkeypatch.setattr` on an Eden module to steer control flow.

A stand-in only has to agree with the real object on what one test reads, and nothing keeps that agreement when the
real object changes. Tests written that way kept passing after the behaviour they claimed to pin had broken.

**Test doubles are allowed only at a real boundary**, for things a test cannot or should not run for real:

| Boundary | What to use |
|---|---|
| robot hardware and its SDK (`unitree_sdk2py`, LimX, Dynamixel, glove and XR devices) | a hand-written **fake** of the SDK's interface that holds state (the last command sent, a scripted sensor stream); `tests/support/deploy.py` drives a deployment through an in-memory wire |
| the network (the Hugging Face Hub, HTTP, a remote policy server) | a local fake server (`tests/support/policy_servers.py`) or a pre-seeded cache in `tmp_path` |
| wall-clock time and sleeps | an injected clock, or `monkeypatch` on `time.monotonic` / `time.sleep` |
| model weights you don't have (ONNX graphs, checkpoints) | a tiny **real** artifact built in the test (a two-node ONNX graph, a saved `state_dict`) rather than a mocked session |
| environment variables, `sys.argv` | `monkeypatch.setenv` / `monkeypatch.setattr(sys, "argv", ...)` |
| the filesystem | real files under `tmp_path`, never a mocked `open` |
| `torch.distributed` | real subprocess ranks through `tests.support.ddp.ddp_spawn` |

Even at a boundary, assert on **outcomes**: what the fake received, the file written, the value returned. Don't
assert on `assert_called_once_with`.

**One exception: GPU-sync budgets.** The suite runs Genesis on the CPU, so no built env lives on CUDA. A test that
pins a hot path's sync count (the `count_cuda_syncs` fixture) may build the unit under test against a minimal
CUDA-resident env stand-in, kept in the directory's `conftest.py` (`tests/managers/conftest.py`,
`tests/managers/terms/commands/conftest.py`). It must count syncs and nothing else. What the unit computes on the
same inputs belongs to a real-env test next to it. To pin the sync budget of a whole built env, use the CUDA probe
`tests.support.step_syncs` instead (see [Cost](#cost)).

## What a test asserts

Assert on public outputs: returned tensors, the state of a built entity, a raised error and its message
(`pytest.raises(..., match=...)`), a file on disk. Reach into a private attribute only when it is the only
observable form of the contract, and say why in a comment. When a test pins an error, match the parts of the message
the [error-message rules](error-messages.md) require: the offending value, and the fix.

Drop checks that cannot fail, or that fail only when something else's tests already fail:

- **self-referential oracles.** The test recomputes the production formula and compares against its own copy. Pin
  known values instead (hand-computed, or a boundary case), or compare against an independent implementation.
- **tautologies and restatements.** A Pydantic default read back unchanged, `isinstance` after a constructor, a
  shape assert followed by a value assert that implies it, a class attribute re-read.
- **tests of Python, Pydantic, torch or Genesis** rather than of Eden.
- **source-text tests** (`inspect.getsource` plus a regex) when a behavioural test can see the same thing.
- **import-only smoke tests** that a behavioural test in the same area already covers.

One behaviour gets one test. Families that differ only in data become one `@pytest.mark.parametrize`.

## Cost

A Genesis scene build takes seconds, so:

- build **once per module** (`@pytest.fixture(scope="module")`) and share it. Several tests that need slightly
  different scenes usually fit in one scene with a few more entities.
- keep `num_envs` small (1–4) unless the test is about batching.
- if a test mutates a shared env, it restores what it changed or calls `env.reset()`.
- mark genuinely slow tests (training iterations, long rollouts) `@pytest.mark.slow`.

**Behaviour that exists only on CUDA** (a sync count, a CUDA graph, the batch renderer, a timing) cannot be tested
in the suite's own process: Genesis is initialized there on the CPU and cannot be re-initialized in-process. Such a
test runs a probe in a subprocess. The probes are modules under `tests/support/` (`step_syncs`, `graded_graph`,
`reset_bench`, `batch_renderer_ink`), each run as `python -m tests.support.<probe> ...` from the repository root,
which prints one line, `RESULT {...}`, that the test parses and asserts on.

## Running

Run the files your change touches, one file at a time, rather than the whole directory in one process:

```bash
uv run pytest tests/managers/terms/rewards/test_locomotion.py
for f in $(find tests -name 'test_*.py' | sort); do uv run pytest "$f" || break; done
```

Whole-suite runs in one process used to crash partway through, because every `scene.build()` used up kernel-tree
slots that a process has a fixed number of. That leak is fixed in the Genesis version Eden pins, but a single-process
run of the whole suite has not been re-verified since, so running per file remains the supported way.

The markers are declared in `tests/conftest.py`: `slow`, `requires_genesis`, and `precision(bits)` (the Genesis
precision a MochiSolver test runs at, in `tests/engine/solvers/mochi`). `pyproject.toml` passes `--strict-markers`,
so an undeclared marker is a collection error rather than a decorator that silently does nothing. Nothing is marked
automatically; apply a marker per test or per module (`pytestmark = pytest.mark.requires_genesis`).

## See also

- [Error Messages](error-messages.md)
- [Contributing](../contributing/index.md)
- [Terms guide](terms.md) — the term contract the tests pin
