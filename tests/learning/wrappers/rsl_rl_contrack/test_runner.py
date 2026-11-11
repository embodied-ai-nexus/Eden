"""``ContrackOnPolicyRunner``: what the contrack task's ``runner_options.class_name`` names for ``eden train``."""

import pytest

from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper, make_rsl_rl_runner, rsl_rl_runner_class
from eden.learning.wrappers.rsl_rl_contrack import RslRlContrackRunnerOptions
from eden.learning.wrappers.rsl_rl_contrack.env import ContrackVecEnvWrapper
from eden.tasks import TASK_REGISTRY
from tests.support.envs import build_env

pytestmark = pytest.mark.requires_genesis


def test_the_task_selects_the_contrack_runner_by_import_path():
    assert isinstance(TASK_REGISTRY.build("contrack").runner_options, RslRlContrackRunnerOptions)
    pytest.importorskip("rsl_rl_contrack")
    from eden.learning.wrappers.rsl_rl_contrack.runner import CONTRACK_RUNNER_CLASS_NAME, ContrackOnPolicyRunner

    assert RslRlContrackRunnerOptions.model_fields["class_name"].default == CONTRACK_RUNNER_CLASS_NAME
    assert rsl_rl_runner_class(CONTRACK_RUNNER_CLASS_NAME) is ContrackOnPolicyRunner


def test_without_the_backend_the_runner_names_the_install_command(monkeypatch):
    import sys

    for name in ("rsl_rl_contrack", "rsl_rl_contrack.runners"):  # a None entry makes the import fail
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.delitem(sys.modules, "eden.learning.wrappers.rsl_rl_contrack.runner", raising=False)
    with pytest.raises(ImportError, match=r"pip install 'git\+https://github.com/Kashu7100/rsl_rl_contrack@"):
        rsl_rl_runner_class(RslRlContrackRunnerOptions().class_name)


@pytest.mark.slow
def test_the_contrack_runner_keeps_only_its_newest_checkpoint_like_edens_runners(tmp_path, monkeypatch):
    """The fork's runner saves model_<it>.pt as rsl_rl does; without Eden's retention it ignored the option."""
    pytest.importorskip("rsl_rl_contrack")
    import eden

    monkeypatch.setattr(eden, "_log_dir", str(tmp_path))
    cfg = TASK_REGISTRY.build("contrack")
    cfg.env_options.num_envs = 2
    options = RslRlContrackRunnerOptions(logger="tensorboard", device="cpu")
    # What `eden train` hands it: Eden's plain wrapper, which the runner re-wraps for the CMDP extras.
    runner = make_rsl_rl_runner(RslRlVecEnvWrapper(build_env(cfg), options))
    assert isinstance(runner.env, ContrackVecEnvWrapper)

    # The fork's own save, as its learn() calls it every save_interval (a short learn() cannot run here: its CMDP
    # update averages over finished episodes, and a few steps finish none). learn() names the logger before saving.
    runner.logger_type = options.logger
    for it in (100, 200, 300):
        runner.save(str(tmp_path / f"model_{it}.pt"))

    assert [p.name for p in tmp_path.glob("model_*.pt")] == ["model_300.pt"]
