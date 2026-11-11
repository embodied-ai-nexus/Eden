"""``eden.utils.example_smoke``: why an example is skipped, and the recipe examples smoke-run end to end.

The smoke run is GPU- and dependency-gated by design; every non-run is a visible skip with its reason.
"""

import dataclasses
import sys

import pytest

from eden.utils import example_smoke
from eden.utils.example_smoke import has_gpu, skip_reason, smoke_run
from eden.utils.examples_catalog import CATALOG, OPTIONAL_DEPS, ExampleSpec

_RECIPES = [s for s in CATALOG.specs() if s.tier == "recipe"]
_HAS_GPU = has_gpu()


@pytest.fixture(scope="module")
def runnable_cpu_spec():
    """A real CPU recipe with no optional deps that passes every gate."""
    for spec in _RECIPES:
        if spec.hardware == "cpu" and not spec.optional_deps and skip_reason(spec, has_gpu=False) is None:
            return spec
    pytest.fail("no smoke-runnable CPU recipe in the catalog")


def test_missing_optional_dep_names_its_install_path(runnable_cpu_spec, monkeypatch):
    """The harness skips before the example's own ImportError hint can surface, so the skip message
    itself must carry the install path.
    """
    monkeypatch.setitem(sys.modules, OPTIONAL_DEPS["metamotivo"].module, None)  # the package reads as absent
    reason = skip_reason(dataclasses.replace(runnable_cpu_spec, optional_deps=("metamotivo",)), has_gpu=True)
    assert reason == f"optional dependency 'metamotivo' not installed ({OPTIONAL_DEPS['metamotivo'].install})"


def test_gpu_example_skips_on_cpu_only_machine(runnable_cpu_spec):
    """``hardware: gpu`` gates on the machine even when every other gate passes."""
    gpu_spec = dataclasses.replace(runnable_cpu_spec, hardware="gpu")
    assert skip_reason(gpu_spec, has_gpu=False) == "needs a CUDA GPU"
    assert skip_reason(gpu_spec, has_gpu=True) is None


def test_smoke_run_runs_the_example_in_the_given_directory(tmp_path, monkeypatch):
    """Examples write their outputs under the working directory, so ``cwd`` decides where a run leaves them."""
    repo = tmp_path / "repo"  # a stand-in checkout holding one example, so the spec's path stays repo-relative
    repo.mkdir()
    (repo / "writes_to_cwd.py").write_text(
        "import os, pathlib\npathlib.Path('output.txt').write_text(os.environ['EDEN_SMOKE'])\n"
    )
    monkeypatch.setattr(example_smoke, "REPO_ROOT", repo)
    spec = ExampleSpec(
        key="writes_to_cwd",
        path="writes_to_cwd.py",
        tier="recipe",
        purpose="writes one file into its working directory",
        capabilities=(),
        hardware="cpu",
        run="python writes_to_cwd.py",
    )
    caller, scratch = tmp_path / "caller", tmp_path / "scratch"
    caller.mkdir()
    scratch.mkdir()
    monkeypatch.chdir(caller)

    result = smoke_run(spec, timeout=60, cwd=scratch)

    assert result.returncode == 0, result.stderr
    assert not any(caller.iterdir()), "the run wrote into the caller's directory, not the given cwd"
    assert (scratch / "output.txt").read_text() == "1"


@pytest.mark.slow
@pytest.mark.parametrize("spec", _RECIPES, ids=lambda s: s.key)
def test_recipe_smoke(spec, tmp_path):
    reason = skip_reason(spec, has_gpu=_HAS_GPU)
    if reason:
        pytest.skip(reason)
    result = smoke_run(spec, timeout=900, cwd=tmp_path)  # the example's outputs go to the scratch dir, not the checkout
    assert result.returncode == 0, (
        f"{spec.key} exited {result.returncode}:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
    )
