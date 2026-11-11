"""Export a trained policy for a runtime outside Eden: an ONNX / TorchScript graph plus ``policy_spec.json``.

``eden export`` (:func:`eden.utils.cli.tasks.export_task`) builds the checkpoint's own env, loads the runner, and
hands both to :func:`export_policy`, which writes

- ``policy.onnx`` — input ``obs`` (the actor's observation groups, concatenated, *un-normalized*), output
  ``actions`` (the raw policy output, before the action terms scale it), with a dynamic batch axis. A recurrent
  policy also takes ``h_in`` (and ``c_in`` for an LSTM), shaped ``(num_layers, batch, hidden)``, and returns
  ``h_out`` (``c_out``): the caller threads the state, starting from zeros;
- ``policy.pt`` — the same composition as TorchScript, batch 1 when recurrent: its hidden state lives in the
  module, and ``reset()`` zeroes it;
- ``policy_spec.json`` — the contract around the graph (:func:`build_policy_spec`): what fills ``obs``, how
  ``actions`` becomes joint targets, the control rate, and where the graph came from.

The observation normalizer is part of both graphs. The graph comes from the policy model's own ``as_onnx()`` /
``as_jit()`` (rsl-rl-lib >= 4's export hooks), which is also how a learner overrides the default: RMA's
:class:`~eden.learning.wrappers.rsl_rl.rma.model.RmaActorModel` defines both to export its deployment composition
(estimator injected). An inherited hook can export something other than what the model runs, so every export is
checked against the runner's inference policy on the env's own observations before anything is written
(:func:`verify_export`). The files are written into a staging directory and moved into place only once checked, so a
failed export leaves the output directory as it was.

The spec records the robot's *nominal* command: the joint-zero offset domain randomization draws per env is left out
of the action ``offset``, and an env whose default pose itself was randomized at startup is refused
(:func:`disable_contract_randomization` strips those terms from a config before it is built).
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import shutil
import tempfile
from typing import TYPE_CHECKING, Any

import torch

from eden.utils.distribution import install_hint

if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase
    from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper
    from eden.utils.configs import EdenRLConfig

#: The formats ``eden export --format`` accepts, in the order they are written.
EXPORT_FORMATS = ("onnx", "jit")
#: The file each format is written to, inside the output directory.
EXPORT_FILENAMES = {"onnx": "policy.onnx", "jit": "policy.pt"}
POLICY_SPEC_FILENAME = "policy_spec.json"
#: Bumped on any change a consumer of ``policy_spec.json`` would have to handle. v2: ``provenance`` records Eden and
#: Genesis each as ``{"version", "commit"}`` (Eden adds ``dirty``).
POLICY_SPEC_SCHEMA_VERSION = 2
#: Control steps the export is checked over: enough to carry a recurrent state through a few updates.
VERIFY_STEPS = 4
#: The exported graph runs on CPU and the reference on the env's device, so a GPU env needs some slack.
VERIFY_ATOL = 1e-4
VERIFY_RTOL = 1e-4


def export_policy(
    runner: Any,
    env: RslRlVecEnvWrapper,
    out_dir: str,
    *,
    formats: tuple[str, ...] = EXPORT_FORMATS,
    task: str,
    checkpoint: str,
    verify: bool = True,
) -> dict[str, str]:
    """Export ``runner``'s policy to ``out_dir`` and write ``policy_spec.json`` beside it.

    Parameters
    ----------
    runner:
        An rsl_rl runner with ``checkpoint`` loaded.
    env:
        The env the runner was built on: the observations the export is checked on, and what the spec describes.
    out_dir:
        Directory to write into (created if missing). An earlier export there is replaced, and a graph of a format
        not exported this time is removed with it; on a failure the directory is left as it was.
    formats:
        Any of :data:`EXPORT_FORMATS`.
    task:
        The registered task name, recorded in the spec.
    checkpoint:
        The checkpoint path, hashed into the spec.
    verify:
        Check every exported graph against the runner's inference policy (:func:`verify_export`) before writing the
        spec. The ONNX check needs ``onnxruntime``.

    Returns
    -------
    dict[str, str]
        ``{format: path}`` for every file written, the spec under ``"spec"``.
    """
    unknown = sorted(set(formats) - set(EXPORT_FORMATS))
    if unknown or not formats:
        raise ValueError(
            f"Export formats {unknown or list(formats)} are not supported; choose one or more of {EXPORT_FORMATS}."
        )
    check_export_dependencies(formats, verify=verify)
    get_policy = getattr(getattr(runner, "alg", None), "get_policy", None)
    if get_policy is None:
        raise TypeError(
            f"{type(runner).__qualname__} has no `alg.get_policy()`, so there is no policy model to export. Export "
            f"reads the model through it, as rsl-rl-lib >= 4's algorithms provide it; under rsl-rl-lib 3, upgrade "
            f"(a 3.x checkpoint loads into a 4.x / 5.x runner)."
        )
    policy = get_policy()
    obs_groups = policy_obs_groups(policy)
    refuse_randomized_contract(env.unwrapped)
    os.makedirs(out_dir, exist_ok=True)
    # Staged, then moved in only once checked and described: a graph that failed its check, or a spec cut short, must
    # not land where a deployment could pick it up, nor beside the files of an earlier export.
    staging = tempfile.mkdtemp(prefix=".eden-export-", dir=out_dir)
    try:
        staged = {}
        for fmt in EXPORT_FORMATS:
            if fmt in formats:
                staged[fmt] = os.path.join(staging, EXPORT_FILENAMES[fmt])
                (_export_onnx if fmt == "onnx" else _export_jit)(policy, staged[fmt])
        if verify:
            verify_export(staged, runner.get_inference_policy(device=env.device), env, obs_groups)
        spec = build_policy_spec(env.unwrapped, policy, staged, task=task, checkpoint=checkpoint, runner_env=env)
        staged["spec"] = os.path.join(staging, POLICY_SPEC_FILENAME)
        with open(staged["spec"], "w") as f:
            json.dump(spec, f, indent=2)
            f.write("\n")
        files = {key: os.path.join(out_dir, os.path.basename(path)) for key, path in staged.items()}
        for key, path in staged.items():  # the spec last, so it never names a graph not yet in place
            os.replace(path, files[key])
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    for fmt in EXPORT_FORMATS:
        stale = os.path.join(out_dir, EXPORT_FILENAMES[fmt])
        if fmt not in files and os.path.exists(stale):
            os.remove(stale)  # an earlier export's graph, which the new spec does not describe
    return files


def check_export_dependencies(formats: tuple[str, ...], *, verify: bool) -> None:
    """Raise ``ImportError`` if a package ``formats`` needs is missing: ``onnx`` for the ONNX graph's spec, and
    ``onnxruntime`` to check it.

    ``eden export`` calls this before it builds anything, so a missing extra costs no env build.
    """
    if "onnx" not in formats:
        return
    missing = []
    for package in ("onnx", "onnxruntime") if verify else ("onnx",):
        try:
            importlib.import_module(package)
        except (ImportError, OSError) as exc:  # a broken onnxruntime-gpu wheel fails on its CUDA libraries
            missing.append(f"{package} ({' '.join(str(exc).split())})")
    if missing:
        skip = " onnxruntime only runs the parity check, which --no_verify (verify=False) skips." if verify else ""
        raise ImportError(
            f"Exporting the ONNX graph needs {' and '.join(missing)}. Install the extra with {install_hint('onnx')} "
            f"(for the GPU wheel: {install_hint('onnx-gpu')}). Or export TorchScript only with --format jit.{skip}"
        )


def _contract_randomization_terms() -> tuple[type, ...]:
    """Event terms that randomize what ``policy_spec.json`` records as nominal: the default pose (and with it every
    ``reference_source="default"`` controller's offset), baked in per env at startup.

    The per-reset joint-zero jitter (``RandomizeDofsPosOffset``) is not among them: it lives in a controller's
    ``set_dofs_pos_offset``, which the spec leaves out of ``offset`` (:func:`_joint_target_spec`).
    """
    from eden.managers.terms.events.domain_rand import RandomizeStartupDofsPosBias

    return (RandomizeStartupDofsPosBias,)


def disable_contract_randomization(config: EdenRLConfig) -> list[str]:
    """Disable ``config``'s event terms that would randomize what the spec records as nominal; return their names.

    Call it before the env is built: ``eden export`` does. :func:`export_policy` refuses an env built with them.
    """
    from eden.managers.event_manager import EVENT_TERM_REGISTRY

    if config.event_options is None:
        return []
    disabled = []
    for name in config.event_options.term_keys():
        term_cls = EVENT_TERM_REGISTRY.get(getattr(config.event_options, name).name)
        if isinstance(term_cls, type) and issubclass(term_cls, _contract_randomization_terms()):
            config.event_options.disable_term(name)
            disabled.append(name)
    return disabled


def refuse_randomized_contract(env: RLEnvBase) -> None:
    """Raise ``ValueError`` if ``env`` was built with an event term that randomizes what the spec records as nominal."""
    event_manager = getattr(env, "event_manager", None)
    if event_manager is None:
        return
    randomizing = {
        name: type(term).__name__
        for _, name, term in event_manager.iter_terms()
        if isinstance(term, _contract_randomization_terms())
    }
    if randomizing:
        listed = ", ".join(f"'{name}' ({cls})" for name, cls in randomizing.items())
        raise ValueError(
            f"The env was built with event term(s) {listed}, which bias each env's default joint pose at startup, so "
            f"policy_spec.json would record env 0's random draw as the robot's default pose and action offset. Build "
            f"the env from a config passed through eden.learning.export.disable_contract_randomization(config) "
            f"first, as eden export does; reading another env's row would only record another draw."
        )


def policy_obs_groups(policy: Any) -> list[str]:
    """The observation groups ``policy`` concatenates into its input, in order — the layout of the graph's ``obs``."""
    groups = getattr(policy, "obs_groups", None)
    if not isinstance(groups, (list, tuple)) or not groups:
        raise TypeError(
            f"{type(policy).__qualname__} has no `obs_groups` list, so the observation layout of an export cannot be "
            f"known. Export reads the groups the model concatenates from that attribute, as rsl-rl-lib >= 4's "
            f"MLPModel sets it; a custom model must set it too."
        )
    return list(groups)


def _export_hook(policy: Any, hook: str) -> Any:
    """``policy.<hook>()``'s exportable module, or a ``TypeError`` naming what is missing."""
    method = getattr(policy, hook, None)
    if method is None:
        raise TypeError(
            f"{type(policy).__module__}.{type(policy).__qualname__} has no `{hook}()`, so it cannot be exported. "
            f"rsl-rl-lib >= 4 models define `as_onnx()` and `as_jit()` (`as_onnx(verbose)` before rsl-rl-lib 5.5.2; "
            f"either form exports); a learner's own model exports by defining them, returning a module that runs the "
            f"deployed composition on the concatenated observation (see RmaActorModel in "
            f"eden/learning/wrappers/rsl_rl/rma/model.py)."
        )
    module = method(verbose=False) if hook == "as_onnx" and _takes_verbose(method) else method()
    return module.cpu().eval()


def _takes_verbose(as_onnx: Any) -> bool:
    """Whether an ``as_onnx`` hook takes ``verbose``: rsl-rl-lib 4.x to 5.5.1 declare ``as_onnx(verbose)`` (required on
    ``MLPModel``), 5.5.2 dropped the argument. Read from the bound method itself, so a model written against either
    contract exports under either rsl-rl-lib.
    """
    return "verbose" in inspect.signature(as_onnx).parameters


def _export_onnx(policy: Any, path: str) -> None:
    """Write ``policy.as_onnx()`` to ``path`` with a dynamic batch axis."""
    module = _export_hook(policy, "as_onnx")
    inputs, outputs = list(module.input_names), list(module.output_names)
    # The observation and action carry the batch first; a recurrent state is (num_layers, batch, hidden).
    dynamic_axes = {name: {0: "batch"} for name in (inputs[0], outputs[0])}
    dynamic_axes.update({name: {1: "batch"} for name in inputs[1:] + outputs[1:]})
    # The TorchScript-based exporter rather than the dynamo one: it needs no onnxscript (not in the `onnx` extra),
    # and rsl_rl's GRU export returns a `None` third output, which it drops.
    torch.onnx.export(
        module,
        module.get_dummy_inputs(),
        path,
        export_params=True,
        opset_version=18,
        input_names=inputs,
        output_names=outputs,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )


def _export_jit(policy: Any, path: str) -> None:
    """Write ``policy.as_jit()``, scripted, to ``path``."""
    torch.jit.script(_export_hook(policy, "as_jit")).save(path)


def verify_export(files: dict[str, str], inference_policy: Any, env: RslRlVecEnvWrapper, obs_groups: list[str]) -> None:
    """Check every exported graph reproduces ``inference_policy`` over :data:`VERIFY_STEPS` real control steps.

    The env is stepped by ``inference_policy`` from its current state, under ``torch.inference_mode`` as rsl_rl's
    rollout steps it; each graph then replays the same observations from a zero recurrent state.
    ``inference_policy`` is what ``eden inference`` runs, so a learner whose export hook exports anything else — an
    inherited ``as_onnx()`` that skips the model's own forward — fails here, before a spec is written next to it.

    Raises
    ------
    RuntimeError
        Naming the format, the step and the largest deviation, when a graph disagrees.
    """
    reset = getattr(inference_policy, "reset", None)
    if reset is not None:
        reset()
    observations, expected = [], []
    obs = env.get_observations()
    # inference_mode, as rsl_rl's own rollout steps the env: after a `learn()` its buffers are inference tensors,
    # which only an inference-mode step may update in place.
    with torch.inference_mode():
        for _ in range(VERIFY_STEPS):
            actions = inference_policy(obs)
            observations.append(torch.cat([obs[group] for group in obs_groups], dim=-1).cpu())
            expected.append(actions.cpu())
            obs, *_ = env.step(actions)
    for fmt, path in files.items():
        produced = _run_onnx(path, observations) if fmt == "onnx" else _run_jit(path, observations)
        name = os.path.basename(path)
        for step, (got, want) in enumerate(zip(produced, expected, strict=True)):
            if got.shape != want.shape:
                mismatch = f"actions of shape {tuple(got.shape)} where the policy returned {tuple(want.shape)}"
            elif not torch.allclose(got, want, atol=VERIFY_ATOL, rtol=VERIFY_RTOL):
                mismatch = (
                    f"actions up to {(got - want).abs().max().item():.3e} away from the policy's "
                    f"(tolerance atol={VERIFY_ATOL}, rtol={VERIFY_RTOL})"
                )
            else:
                continue
            raise RuntimeError(
                f"The exported {fmt} graph ({name}) does not reproduce the policy: at step {step} it returned {mismatch}. "
                f"The model's export hook exports something other than what it runs; fix the hook rather than the "
                f"tolerance."
            )


def _run_onnx(path: str, observations: list[torch.Tensor]) -> list[torch.Tensor]:
    """Run the ONNX graph at ``path`` over ``observations`` in order, threading any recurrent state."""
    import numpy as np

    from eden.extensions.policies.onnx import _rebuild_session_cpu

    session = _rebuild_session_cpu(path)
    inputs = {i.name: i for i in session.get_inputs()}
    state_names = [name for name in inputs if name != "obs"]
    unexpected = [name for name in state_names if name not in ("h_in", "c_in")]
    if "obs" not in inputs or unexpected:
        raise RuntimeError(
            f"The exported ONNX graph ({os.path.basename(path)}) takes inputs {sorted(inputs)}; the check feeds 'obs' and a recurrent "
            f"'h_in' / 'c_in' only."
        )
    batch = observations[0].shape[0]
    # (num_layers, batch, hidden); the export's dummy state is batch 1, so the layer and hidden dims are static.
    state = {
        name: np.zeros((inputs[name].shape[0], batch, inputs[name].shape[2]), dtype=np.float32) for name in state_names
    }
    if not state:
        # Stateless: all steps in one batch, which also exercises the dynamic batch axis.
        (actions, *_) = session.run(None, {"obs": torch.cat(observations).numpy()})
        return list(torch.from_numpy(actions).split(batch))
    produced = []
    for obs in observations:
        actions, *new_state = session.run(None, {"obs": obs.numpy(), **state})
        state = {name: value for name, value in zip(state_names, new_state, strict=True)}
        produced.append(torch.from_numpy(actions))
    return produced


def _run_jit(path: str, observations: list[torch.Tensor]) -> list[torch.Tensor]:
    """Run the TorchScript module at ``path`` over ``observations`` in order, one env at a time from a reset state.

    One at a time because a recurrent module keeps a batch-1 state of its own.
    """
    module = torch.jit.load(path)
    rows = []
    with torch.inference_mode():
        for row in range(observations[0].shape[0]):
            module.reset()
            rows.append([module(obs[row : row + 1]) for obs in observations])
    return [torch.cat(step) for step in zip(*rows, strict=True)]


def build_policy_spec(
    env: RLEnvBase,
    policy: Any,
    files: dict[str, str],
    *,
    task: str,
    checkpoint: str,
    runner_env: RslRlVecEnvWrapper | None = None,
) -> dict[str, Any]:
    """The ``policy_spec.json`` contract for ``policy`` as built on ``env``.

    Sections: ``model`` (the files and their tensors), ``observations`` (what fills ``obs``, slice by slice),
    ``actions`` (how ``actions`` becomes a command), ``entities`` (each commanded entity's DOFs in column order,
    with default positions and PD gains), ``rate`` and ``provenance``. ``runner_env`` is the rsl_rl wrapper, for
    its action clip. ``env`` must pass :func:`refuse_randomized_contract`, as :func:`export_policy` checks first.
    """
    obs_groups = policy_obs_groups(policy)
    clip_actions = getattr(runner_env, "clip_actions", None)
    action_terms = _action_spec(env)
    return {
        "schema_version": POLICY_SPEC_SCHEMA_VERSION,
        "model": _model_spec(files),
        "observations": _observation_spec(env, obs_groups),
        "actions": {
            "dim": env.action_manager.total_action_dim,
            "clip": None if clip_actions is None else [-float(clip_actions), float(clip_actions)],
            "terms": action_terms,
        },
        "entities": _entity_spec(env, {term["entity_name"] for term in action_terms if term.get("dofs_name")}),
        "rate": {
            "control_dt": float(env.dt),
            "decimation": int(env.env_options.decimation),
            "sim_dt": float(env.env_options.sim_dt),
        },
        "provenance": _provenance(env, task, checkpoint),
    }


def _model_spec(files: dict[str, str]) -> dict[str, Any]:
    """Each exported file with its tensors; ONNX tensor shapes as the graph declares them (``"batch"`` is dynamic)."""
    spec: dict[str, Any] = {}
    if "onnx" in files:
        import onnx

        graph = onnx.load(files["onnx"]).graph

        def tensors(values) -> list[dict[str, Any]]:
            return [
                {
                    "name": value.name,
                    "shape": [dim.dim_param or dim.dim_value for dim in value.type.tensor_type.shape.dim],
                }
                for value in values
            ]

        spec["onnx"] = {
            "file": os.path.basename(files["onnx"]),
            "inputs": tensors(graph.input),
            "outputs": tensors(graph.output),
        }
    if "jit" in files:
        spec["jit"] = {"file": os.path.basename(files["jit"])}
    return spec


def _observation_spec(env: RLEnvBase, obs_groups: list[str]) -> dict[str, Any]:
    """The groups concatenated into ``obs``, each with its terms and their slices.

    A term's ``slice`` is within one frame of its group. Without a group history a frame is the group's whole
    block; with one (``history_length``) the block is that many frames, oldest first.
    """
    from eden.options.managers.observations import ObservationTermOptions

    manager = env.observation_manager
    group_options = manager._group_obs_options
    term_dims = manager.group_obs_term_dim
    groups, start = [], 0
    for group in obs_groups:
        names = manager.group_term_names[group]
        terms, offset = [], 0
        group_history = group_options[group].history_length
        for name, dims, term in zip(names, term_dims[group], manager._group_obs_terms[group], strict=True):
            options = term._options
            # A group history buffers the raw frames, so a term's own history applies only without one.
            history = 0 if group_history is not None else options.history_length
            width = int(torch.tensor(dims).prod())
            scale = term.effective_scale
            terms.append(
                {
                    "name": name,
                    "term": options.name,
                    "slice": [offset, offset + width],
                    "frame_dim": width // history if history else width,
                    "history_length": history,
                    "backfill": bool(options.backfill) if history else None,
                    "clip": None if options.clip is None else [float(v) for v in options.clip],
                    "scale": 1.0 if scale is None else (scale.flatten().tolist() if torch.is_tensor(scale) else scale),
                    "modifiers": [modifier.dict() for modifier in options.modifiers],
                    "config": {
                        key: value
                        for key, value in options.dict().items()
                        if key not in ObservationTermOptions.POST_COMPUTE_FIELDS
                    },
                }
            )
            offset += width
        width = int(torch.tensor(manager.group_obs_dim[group]).prod())
        groups.append(
            {
                "name": group,
                "slice": [start, start + width],
                "history_length": group_history or 0,
                "backfill": bool(group_options[group].backfill) if group_history else None,
                "nan_policy": group_options[group].nan_policy,
                "terms": terms,
            }
        )
        start += width
    return {"dim": start, "groups": groups}


def _action_spec(env: RLEnvBase) -> list[dict[str, Any]]:
    """The action terms in action-column order, with how each joint controller turns its slice into targets."""
    from eden.managers.terms.actions.joint_actions import _JointPDControllerBase

    manager = env.action_manager
    terms, start = [], 0
    for name, dim in zip(manager.active_terms, manager.action_term_dim, strict=True):
        term = manager.get_term(name)
        spec: dict[str, Any] = {
            "name": name,
            "term": term._options.name,
            "class": type(term).__name__,
            "slice": [start, start + dim],
            "entity_name": getattr(term, "entity_name", None),
            "config": term._options.dict(),
        }
        if isinstance(term, _JointPDControllerBase):
            spec.update(_joint_target_spec(term, start, dim))
        terms.append(spec)
        start += dim
    return terms


def _joint_target_spec(term: Any, start: int, dim: int) -> dict[str, Any]:
    """How a joint controller turns its action columns into targets; ``{}`` for a column layout the spec doesn't
    describe, which then carries its class and config only, as a non-joint term does.
    """
    from eden.managers.terms.actions.joint_actions import VelocityFeedforwardPDController

    # One column per commanded DOF: under mimic_dofs the drivers lead `dofs_name` and the followers take none.
    n_columns = len(term.dofs_name) - len(term.mimic_follower_dofs_name)
    feedforward = isinstance(term, VelocityFeedforwardPDController)
    if dim != (2 * n_columns if feedforward else n_columns):
        return {}
    scale = term.effective_scale
    spec = {
        "target": "position" if term._targets_position_ else "velocity",
        "reference_source": str(term.reference_source),
        "dofs_name": list(term.dofs_name[:n_columns]),
        "scale": scale.flatten().tolist() if torch.is_tensor(scale) else float(scale),
        # The offset build resolved, not `effective_offset`: that adds the per-env joint-zero offset of
        # `set_dofs_pos_offset`, a domain-randomization draw (RandomizeDofsPosOffset) rather than the robot's command.
        "offset": term._offset[0, :n_columns].tolist(),
        "mimic_dofs": {k: list(v) for k, v in term.mimic_dofs.items()},
    }
    if feedforward:
        # The second half of the columns is a target velocity for the same DOFs.
        spec["velocity_feedforward"] = {
            "slice": [start + n_columns, start + dim],
            "scale": float(term.velocity_scale),
            "feed_forward_ratio": float(term.feed_forward_ratio),
        }
    return spec


def _entity_spec(env: RLEnvBase, entity_names: set[str]) -> dict[str, Any]:
    """Each named entity's DOFs in column order — the order every per-DOF list in the spec uses — with its defaults."""
    entities = {}
    for name in sorted(entity_names):
        entity = env.entities[name]
        entities[name] = {
            "dofs_name": list(entity.dofs_name),
            "default_dofs_pos": entity.get_default_dofs_pos()[0].tolist(),
            "kp": entity.get_default_dofs_kp()[0].tolist(),
            "kd": entity.get_default_dofs_kd()[0].tolist(),
        }
    return entities


def _provenance(env: RLEnvBase, task: str, checkpoint: str) -> dict[str, Any]:
    """Where the graph came from: the task, its config class, the checkpoint's hash and the library versions."""
    from eden.utils.misc import get_eden_run_provenance, get_package_provenance
    from eden.utils.run_manifest import get_rl_library

    digest = hashlib.sha256()
    with open(checkpoint, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    config_cls = type(env.config)
    return {
        "task": task,
        "config_class": f"{config_cls.__module__}.{config_cls.__qualname__}",
        "checkpoint": {"file": os.path.abspath(checkpoint), "sha256": digest.hexdigest()},
        "eden": get_eden_run_provenance(),
        "genesis": get_package_provenance("genesis-world"),
        "rl_library": get_rl_library("rsl_rl"),
        "torch": torch.__version__,
    }
