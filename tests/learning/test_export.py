"""Tests for ``eden.learning.export``: the ONNX / TorchScript graph and ``policy_spec.json`` of ``eden export``.

Built on the ``tests/envs`` pendulum, rewired so every spec field has something to say: a scaled and clipped
position, a velocity with a term history, and a scaled PD action. One runner trained for two iterations (so the
observation normalizer holds real statistics) backs the round-trip tests; a recurrent actor and an actor whose export
hook exports something other than what it runs get envs of their own.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import pathlib
import sys
from importlib import metadata
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from rsl_rl.models.mlp_model import MLPModel

from eden import observations
from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper, make_rsl_rl_runner
from eden.learning.export import (
    POLICY_SPEC_FILENAME,
    check_export_dependencies,
    disable_contract_randomization,
    export_policy,
)
from eden.managers.terms.actions.joint_actions import ImplicitPDController, VelocityFeedforwardPDController
from eden.managers.terms.events.domain_rand import RandomizeDofsPosOffset, RandomizeStartupDofsPosBias
from eden.options import (
    ActionManagerOptions,
    EventManagerOptions,
    ObservationGroupOptions,
    ObservationManagerOptions,
)
from eden.options.learning.rsl_rl import RslRlActorOptions, RslRlActorRNNOptions
from tests.envs.conftest import PENDULUM_MJCF
from tests.envs.wrappers.conftest import gym_pendulum_config, rsl_rl_runner_options


def _installed_onnx_module(model: MLPModel) -> torch.nn.Module:
    """The installed rsl-rl-lib's own ``MLPModel`` ONNX module, whichever ``as_onnx`` signature that release has."""
    if "verbose" in inspect.signature(MLPModel.as_onnx).parameters:
        return MLPModel.as_onnx(model, verbose=False)
    return MLPModel.as_onnx(model)


class VerboseOnnxModel(MLPModel):
    """An actor written against rsl-rl-lib 4.x to 5.5.1, whose ``MLPModel.as_onnx(verbose)`` requires the argument."""

    def as_onnx(self, verbose: bool) -> torch.nn.Module:
        return _installed_onnx_module(self)


class PlainOnnxModel(MLPModel):
    """An actor written against rsl-rl-lib 5.5.2, which dropped ``verbose``: ``as_onnx()``."""

    def as_onnx(self) -> torch.nn.Module:
        return _installed_onnx_module(self)


class DoubledOutputModel(MLPModel):
    """An actor that doubles its action but inherits ``MLPModel``'s export hooks, which export the undoubled one."""

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        return 2.0 * super().forward(obs, masks, hidden_state, stochastic_output)


@pytest.fixture(scope="module")
def pendulum_file(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("assets") / "pendulum.xml"
    path.write_text(PENDULUM_MJCF)
    return str(path)


def _pendulum_config(pendulum_file, *, actor=None, action=None, events=None, num_envs=2):
    config = gym_pendulum_config(pendulum_file, num_envs=num_envs)
    config.observation_options = ObservationManagerOptions(
        policy=ObservationGroupOptions(
            dofs_pos=observations.DofsPos.configure(entity_name="pendulum", scale=2.0, clip=(-3.0, 3.0)),
            dofs_vel=observations.DofsVel.configure(entity_name="pendulum", history_length=3),
        )
    )
    config.action_options = ActionManagerOptions(
        hinge=action or ImplicitPDController.configure(entity_name="pendulum", dofs_name=["hinge"], scale=0.5)
    )
    if events is not None:
        config.event_options = EventManagerOptions(**events)
    config.runner_options = rsl_rl_runner_options(**({"actor": actor} if actor is not None else {}))
    return config


def _pendulum_runner(pendulum_file, log_dir, **config_kwargs):
    env = RslRlVecEnvWrapper.from_config(_pendulum_config(pendulum_file, **config_kwargs))
    return env, make_rsl_rl_runner(env, log_dir=str(log_dir))


@pytest.fixture(scope="module")
def exported(pendulum_file, tmp_path_factory):
    """``(env, runner, files, spec, checkpoint)``: a two-iteration pendulum policy exported in both formats."""
    pytest.importorskip("onnxruntime", exc_type=ImportError)
    root = tmp_path_factory.mktemp("export")
    env, runner = _pendulum_runner(pendulum_file, root / "run")
    runner.learn(num_learning_iterations=2)
    checkpoint = str(root / "model_2.pt")
    runner.save(checkpoint)
    files = export_policy(runner, env, str(root / "out"), task="pendulum", checkpoint=checkpoint)
    spec = json.loads(pathlib.Path(files["spec"]).read_text())
    yield env, runner, files, spec, checkpoint
    env.close()


def _onnx_actions(path: str, obs: torch.Tensor, **state) -> list[np.ndarray]:
    import onnxruntime

    session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
    return session.run(None, {"obs": obs.cpu().numpy(), **state})


def _flat_obs(env, spec) -> tuple[dict, torch.Tensor]:
    obs = env.get_observations()
    return obs, torch.cat([obs[group["name"]] for group in spec["observations"]["groups"]], dim=-1)


def test_the_onnx_graph_reproduces_the_policy_normalizer_included(exported):
    env, runner, files, spec, _ = exported
    policy = runner.get_inference_policy(device=env.device)
    # Two learning iterations have moved the normalizer off identity, so matching the policy means it is baked in.
    assert not torch.allclose(policy.obs_normalizer._mean, torch.zeros_like(policy.obs_normalizer._mean))
    obs, flat = _flat_obs(env, spec)

    (actions,) = _onnx_actions(files["onnx"], flat)

    assert flat.shape[0] == 2  # the dynamic batch axis takes every env at once
    torch.testing.assert_close(torch.from_numpy(actions), policy(obs).cpu(), atol=1e-6, rtol=1e-6)


def test_the_onnx_graph_loads_back_through_edens_onnx_module(exported):
    from eden.extensions.policies.onnx import OnnxModule

    env, runner, files, spec, _ = exported
    obs, flat = _flat_obs(env, spec)
    module = OnnxModule(
        files["onnx"],
        "obs",
        "actions",
        device="cpu",
        expected_input_dim=spec["observations"]["dim"],
        expected_output_dim=spec["actions"]["dim"],
    )

    torch.testing.assert_close(module(flat), runner.get_inference_policy(device=env.device)(obs).cpu())


def test_the_torchscript_module_reproduces_the_policy(exported):
    env, runner, files, spec, _ = exported
    obs, flat = _flat_obs(env, spec)

    module = torch.jit.load(files["jit"])

    torch.testing.assert_close(module(flat), runner.get_inference_policy(device=env.device)(obs).cpu())


def test_the_spec_lays_out_the_observation_term_by_term(exported):
    _, _, files, spec, _ = exported

    (group,) = spec["observations"]["groups"]
    pos, vel = group["terms"]
    assert (group["name"], group["slice"], spec["observations"]["dim"]) == ("policy", [0, 4], 4)
    assert (group["history_length"], group["backfill"], group["nan_policy"]) == (0, None, "disabled")
    assert (pos["name"], pos["slice"], pos["scale"], pos["clip"], pos["history_length"]) == (
        "dofs_pos",
        [0, 1],
        2.0,
        [-3.0, 3.0],
        0,
    )
    assert (vel["slice"], vel["frame_dim"], vel["history_length"], vel["scale"]) == ([1, 4], 1, 3, 1.0)
    assert (pos["backfill"], vel["backfill"]) == (None, True)
    assert vel["config"]["entity_name"] == "pendulum" and "history_length" not in vel["config"]
    assert spec["model"]["onnx"]["inputs"] == [{"name": "obs", "shape": ["batch", 4]}]
    assert spec["model"]["onnx"]["outputs"] == [{"name": "actions", "shape": ["batch", 1]}]
    assert spec["model"]["jit"] == {"file": "policy.pt"}
    assert files["spec"].endswith(POLICY_SPEC_FILENAME)


def test_the_spec_says_how_an_action_becomes_a_joint_target(exported):
    env, _, _, spec, _ = exported

    (term,) = spec["actions"]["terms"]
    assert (spec["actions"]["dim"], spec["actions"]["clip"]) == (1, None)
    assert (term["name"], term["class"], term["slice"], term["entity_name"]) == (
        "hinge",
        "ImplicitPDController",
        [0, 1],
        "pendulum",
    )
    assert (term["target"], term["reference_source"], term["dofs_name"], term["scale"], term["offset"]) == (
        "position",
        "default",
        ["hinge"],
        0.5,
        [0.0],
    )
    assert spec["entities"] == {
        "pendulum": {"dofs_name": ["hinge"], "default_dofs_pos": [0.0], "kp": [10.0], "kd": [1.0]}
    }
    assert spec["rate"] == {"control_dt": env.unwrapped.dt, "decimation": 2, "sim_dt": env.unwrapped.dt / 2}


def test_the_spec_records_where_the_graph_came_from(exported):
    env, _, _, spec, checkpoint = exported
    with open(checkpoint, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()

    provenance = spec["provenance"]

    assert (provenance["task"], provenance["checkpoint"]["sha256"]) == ("pendulum", digest)
    assert provenance["config_class"] == f"{type(env.unwrapped.config).__module__}.EdenRLConfig"
    assert provenance["rl_library"]["version"] == metadata.version("rsl-rl-lib")
    assert provenance["torch"] == torch.__version__
    assert provenance["eden"]["version"] == metadata.version("eden-world")
    assert set(provenance["eden"]) == {"version", "commit", "dirty"}
    assert provenance["genesis"]["version"] == metadata.version("genesis-world")


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
def test_a_recurrent_policy_exports_its_state_as_explicit_inputs_and_outputs(pendulum_file, tmp_path, rnn_type):
    pytest.importorskip("onnxruntime", exc_type=ImportError)
    actor = RslRlActorRNNOptions(rnn_type=rnn_type, rnn_hidden_dim=8, hidden_dims=[8])
    env, runner = _pendulum_runner(pendulum_file, tmp_path / "run", actor=actor)
    try:
        # The export checks itself over several steps, the state threaded through; this repeats that by hand.
        files = export_policy(runner, env, str(tmp_path / "out"), task="pendulum", checkpoint=_save(runner, tmp_path))
        spec = json.loads(pathlib.Path(files["spec"]).read_text())
        names = [tensor["name"] for tensor in spec["model"]["onnx"]["inputs"]]
        policy = runner.get_inference_policy(device=env.device)
        policy.reset()
        state = {name: np.zeros((1, 2, 8), dtype=np.float32) for name in names[1:]}
        with torch.inference_mode():  # as rsl_rl's rollout and the export's own check step the env
            for _ in range(3):
                obs, flat = _flat_obs(env, spec)
                expected = policy(obs)
                actions, *new_state = _onnx_actions(files["onnx"], flat, **state)
                torch.testing.assert_close(torch.from_numpy(actions), expected.cpu(), atol=1e-6, rtol=1e-6)
                state = dict(zip(names[1:], new_state, strict=True))
                env.step(expected)
    finally:
        env.close()

    assert names == (["obs", "h_in", "c_in"] if rnn_type == "lstm" else ["obs", "h_in"])


def test_an_export_hook_that_exports_something_else_is_refused_and_leaves_the_out_dir_as_it_was(
    pendulum_file, tmp_path
):
    actor = RslRlActorOptions(class_name="tests.learning.test_export:DoubledOutputModel", hidden_dims=[8])
    env, runner = _pendulum_runner(pendulum_file, tmp_path / "run", actor=actor)
    earlier = tmp_path / "out" / POLICY_SPEC_FILENAME
    earlier.parent.mkdir()
    earlier.write_text("an earlier export's spec")
    try:
        with pytest.raises(RuntimeError, match=r"exported jit graph \(policy.pt\) does not reproduce the policy"):
            export_policy(
                runner, env, str(tmp_path / "out"), formats=("jit",), task="p", checkpoint=_save(runner, tmp_path)
            )
    finally:
        env.close()

    # Neither the failed graph nor its staging directory is left behind, and the earlier export is untouched.
    assert [path.name for path in (tmp_path / "out").iterdir()] == [POLICY_SPEC_FILENAME]
    assert earlier.read_text() == "an earlier export's spec"


@pytest.mark.parametrize("model", ["VerboseOnnxModel", "PlainOnnxModel"])
def test_an_actor_written_against_either_as_onnx_signature_exports(pendulum_file, tmp_path, model):
    """rsl-rl-lib 5.5.2 dropped ``as_onnx``'s ``verbose``, and the pin spans releases on both sides of it, so the
    installed models and a learner's own model may take either form; the export passes ``verbose`` only to a hook
    that declares it. The export's own parity check confirms the graph runs the policy.
    """
    pytest.importorskip("onnxruntime", exc_type=ImportError)
    actor = RslRlActorOptions(class_name=f"tests.learning.test_export:{model}", hidden_dims=[8])
    env, runner = _pendulum_runner(pendulum_file, tmp_path / "run", actor=actor)
    try:
        files = export_policy(
            runner, env, str(tmp_path / "out"), formats=("onnx",), task="p", checkpoint=_save(runner, tmp_path)
        )
    finally:
        env.close()
    assert pathlib.Path(files["onnx"]).is_file()


def test_a_re_export_with_fewer_formats_removes_the_graph_its_spec_does_not_describe(exported, tmp_path):
    env, runner, _, _, checkpoint = exported
    out = tmp_path / "out"
    export_policy(runner, env, str(out), task="p", checkpoint=checkpoint)

    files = export_policy(runner, env, str(out), formats=("jit",), task="p", checkpoint=checkpoint)

    assert sorted(path.name for path in out.iterdir()) == ["policy.pt", POLICY_SPEC_FILENAME]
    assert list(json.loads(pathlib.Path(files["spec"]).read_text())["model"]) == ["jit"]


def test_a_randomized_joint_zero_offset_stays_out_of_the_spec(pendulum_file, tmp_path):
    jitter = RandomizeDofsPosOffset.configure(
        entity_name="pendulum", action_term_name="hinge", dofs_pos_range=(0.1, 0.2)
    )
    env, runner = _pendulum_runner(pendulum_file, tmp_path / "run", events={"encoder_bias": jitter})
    try:
        # The reset has drawn an offset for every env, which the controller applies on top of the nominal one.
        assert (env.unwrapped.action_manager.get_term("hinge").effective_offset >= 0.1).all()
        files = export_policy(
            runner, env, str(tmp_path / "out"), formats=("jit",), task="p", checkpoint=_save(runner, tmp_path)
        )
    finally:
        env.close()

    (term,) = json.loads(pathlib.Path(files["spec"]).read_text())["actions"]["terms"]
    assert term["offset"] == [0.0]


def test_a_startup_default_pose_bias_is_refused_until_the_config_drops_it(pendulum_file, tmp_path):
    bias = RandomizeStartupDofsPosBias.configure(entity_name="pendulum", action_terms_name=["hinge"])
    env, runner = _pendulum_runner(pendulum_file, tmp_path / "run", events={"calibration_drift": bias})
    try:
        with pytest.raises(
            ValueError, match=r"'calibration_drift' \(RandomizeStartupDofsPosBias\).*env 0's random draw"
        ):
            export_policy(runner, env, str(tmp_path / "out"), formats=("jit",), task="p", checkpoint="unused.pt")
    finally:
        env.close()
    assert not (tmp_path / "out").exists()

    config = _pendulum_config(
        pendulum_file,
        events={
            "calibration_drift": bias,
            "encoder_bias": RandomizeDofsPosOffset.configure(entity_name="pendulum", action_term_name="hinge"),
        },
    )
    assert disable_contract_randomization(config) == ["calibration_drift"]
    assert config.event_options.term_keys() == ("encoder_bias",)


def test_a_velocity_feedforward_controller_describes_its_velocity_columns(pendulum_file, tmp_path):
    action = VelocityFeedforwardPDController.configure(
        entity_name="pendulum", dofs_name=["hinge"], scale=0.5, velocity_scale=0.2
    )
    env, runner = _pendulum_runner(pendulum_file, tmp_path / "run", action=action)
    try:
        files = export_policy(
            runner, env, str(tmp_path / "out"), formats=("jit",), task="p", checkpoint=_save(runner, tmp_path)
        )
    finally:
        env.close()

    spec = json.loads(pathlib.Path(files["spec"]).read_text())
    (term,) = spec["actions"]["terms"]
    assert (spec["actions"]["dim"], term["slice"], term["dofs_name"], term["scale"]) == (2, [0, 2], ["hinge"], 0.5)
    assert term["velocity_feedforward"] == {"slice": [1, 2], "scale": 0.2, "feed_forward_ratio": 1.0}


def test_a_missing_onnx_dependency_is_named_with_the_way_around_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)

    with pytest.raises(ImportError, match=r"needs onnxruntime .*eden-world\[onnx\].*--format jit.*--no_verify"):
        check_export_dependencies(("onnx", "jit"), verify=True)
    check_export_dependencies(("onnx",), verify=False)  # only the parity check runs onnxruntime
    monkeypatch.setitem(sys.modules, "onnx", None)
    check_export_dependencies(("jit",), verify=True)  # TorchScript needs neither


def test_a_model_without_export_hooks_is_named_with_what_to_define(exported, tmp_path):
    env = exported[0]
    model = torch.nn.Linear(4, 1)
    model.obs_groups = ["policy"]
    runner = SimpleNamespace(alg=SimpleNamespace(get_policy=lambda: model))

    with pytest.raises(TypeError, match=r"torch.nn.modules.linear.Linear has no `as_jit\(\)`.*RmaActorModel"):
        export_policy(runner, env, str(tmp_path), formats=("jit",), task="p", checkpoint="unused.pt")


def test_an_unknown_format_is_refused_before_anything_is_written(exported, tmp_path):
    env, runner = exported[:2]

    with pytest.raises(ValueError, match=r"\['tflite'\] are not supported; choose one or more of \('onnx', 'jit'\)"):
        export_policy(runner, env, str(tmp_path / "out"), formats=("tflite",), task="p", checkpoint="unused.pt")
    assert not (tmp_path / "out").exists()


def _save(runner, tmp_path) -> str:
    path = str(tmp_path / "model_0.pt")
    runner.save(path)
    return path
