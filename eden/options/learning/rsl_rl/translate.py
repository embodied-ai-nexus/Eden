"""Translation functions between rsl_rl config versions.

These functions convert the canonical v5-style runner dict (separate
``actor`` / ``critic`` with ``distribution_cfg``) into the formats
expected by older rsl_rl versions.

Typical usage::

    from eden.options.learning.rsl_rl.translate import translate_runner_dict

    runner_dict = translate_runner_dict(config.runner_options.dict())  # auto-detect
    runner = OnPolicyRunner(env, runner_dict, ...)
"""

from __future__ import annotations

import copy
import functools
from typing import Any

import eden as en


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def detect_rsl_rl_major_version() -> int:
    """Return the major version of the installed ``rsl-rl-lib`` package.

    Falls back to ``3`` if the package metadata is unavailable (e.g. editable
    install from source without version).
    """
    try:
        from importlib.metadata import version

        return int(version("rsl-rl-lib").split(".")[0])
    except Exception:
        return 3


@functools.lru_cache(maxsize=1)
def detect_rsl_rl_version() -> tuple[int, ...]:
    """Return the installed ``rsl-rl-lib`` release as a tuple (``(5, 5, 1)``), for checks finer than the major.

    Falls back to ``(3,)``, as :func:`detect_rsl_rl_major_version` does, if the package metadata is unavailable.
    """
    try:
        from importlib.metadata import version

        from packaging.version import Version

        return Version(version("rsl-rl-lib")).release
    except Exception:
        return (3,)


def require_rsl_rl_version(minimum: tuple[int, ...], module: str, needs: str) -> None:
    """Raise an ``ImportError`` that names the upgrade when the installed ``rsl-rl-lib`` is older than ``minimum``.

    Called at the top of a module before its ``rsl_rl`` imports, so an older install reads "upgrade to X" rather than
    ``cannot import name ...`` from inside rsl_rl.

    Parameters
    ----------
    minimum : tuple[int, ...]
        The oldest release the module runs on.
    module : str
        The module's dotted name, for the message.
    needs : str
        What it uses that older releases lack, for the message.
    """
    installed = detect_rsl_rl_version()
    if installed < minimum:
        wanted = ".".join(map(str, minimum))
        raise ImportError(
            f"{module} needs rsl-rl-lib >= {wanted} ({needs}); the installed version is "
            f"{'.'.join(map(str, installed))}. Upgrade with `uv pip install 'rsl-rl-lib>={wanted}'`."
        )


def translate_runner_dict(runner_dict: dict, algorithm: Any = None) -> dict:
    """Translate *runner_dict* to whatever the installed rsl_rl version expects.

    Auto-detects the installed rsl_rl version and translates from the canonical
    v5 format to whatever that version expects.

    Parameters
    ----------
    runner_dict : dict
        A runner configuration dict (v5-style with ``actor`` / ``critic`` keys).
        **Not modified in-place** — a deep copy is returned.
    algorithm : ConfigurableOptions, optional
        The runner's algorithm options. When they define ``adapt_runner_dict(runner_dict, major)``,
        it runs on the translated dict, so an algorithm outside rsl_rl (an
        ``eden.learning.wrappers.rsl_rl`` one) adapts its own entries for the installed major
        — the generic translation only knows rsl_rl's own models.

    Returns
    -------
    dict
        A runner dict compatible with the installed rsl_rl version.
    """
    major = detect_rsl_rl_major_version()
    if major < 3:
        raise ValueError(
            f"Detected rsl_rl version {major}.X is not supported. Only versions 3, 4, and 5 are supported."
        )
    elif major == 3:
        en.logger.info("Detected rsl_rl version 3.X. Translating runner dict to v3 format.")
        out = translate_runner_dict_to_v3(runner_dict)
    elif major == 4:
        en.logger.info("Detected rsl_rl version 4.X. Translating runner dict to v4 format.")
        out = translate_runner_dict_to_v4(runner_dict)
    else:  # v5+
        out = copy.deepcopy(runner_dict)
    adapt = getattr(algorithm, "adapt_runner_dict", None)  # an optional hook, not a required field
    return adapt(out, major) if adapt is not None else out


# ---------------------------------------------------------------------------
# v5 → v3.2  (separated actor/critic with distribution_cfg → combined policy)
# ---------------------------------------------------------------------------


def translate_runner_dict_to_v3(runner_dict: dict) -> dict:
    """Convert a v5-style runner dict to the v3.2 format.

    v3.2 expects:
    - A single ``"policy"`` dict with combined actor/critic fields
      (``actor_hidden_dims``, ``critic_hidden_dims``, ``init_noise_std``, …).
    - The actor observation set keyed as ``"policy"`` in ``obs_groups``.
    - No ``share_cnn_encoders`` or ``check_for_nan`` in the algorithm.

    Parameters
    ----------
    runner_dict : dict
        A runner configuration dict produced by ``RslRlOnPolicyRunnerOptions.dict()``.
        **Not modified in-place** — a deep copy is returned.

    Returns
    -------
    dict
        A v3.2-compatible runner dict.
    """
    out = copy.deepcopy(runner_dict)

    # If already in v3.2 format, return as-is
    if "policy" in out and "actor" not in out:
        return out

    actor = out.pop("actor", {})
    critic = out.pop("critic", {})

    # --- build combined policy dict ------------------------------------------
    dist_cfg = actor.pop("distribution_cfg", None)
    class_name = _model_class_to_v3(actor.get("class_name", "MLPModel"))

    policy: dict[str, Any] = {
        "class_name": class_name,
        "actor_hidden_dims": actor.get("hidden_dims", [256, 256, 256]),
        "critic_hidden_dims": critic.get("hidden_dims", [256, 256, 256]),
        "activation": actor.get("activation", "elu"),
        "actor_obs_normalization": actor.get("obs_normalization", False),
        "critic_obs_normalization": critic.get("obs_normalization", False),
    }

    # Distribution → noise params
    if dist_cfg is not None:
        policy["init_noise_std"] = dist_cfg.get("init_std", 1.0)
        policy["noise_std_type"] = dist_cfg.get("std_type", "scalar")

    # Forward RNN fields
    for key in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
        if key in actor:
            policy[key] = actor[key]

    # CNN: v3's ``ActorCriticCNN`` takes separate ``actor_cnn_cfg`` / ``critic_cnn_cfg``
    # (one conv stack per 2D obs group per side), not the v5 single ``cnn_cfg`` — a
    # bare ``cnn_cfg`` lands in its ``**kwargs`` and is silently ignored, leaving
    # the model with no CNN for its image obs.
    if class_name == "ActorCriticCNN":
        if actor.get("cnn_cfg") is not None:
            policy["actor_cnn_cfg"] = actor["cnn_cfg"]
        if critic.get("cnn_cfg") is not None:
            policy["critic_cnn_cfg"] = critic["cnn_cfg"]
    elif actor.get("cnn_cfg") is not None:
        policy["cnn_cfg"] = actor["cnn_cfg"]

    # --- obs_groups: rename "actor" → "policy" --------------------------------
    obs_groups = out.get("obs_groups", {})
    if "actor" in obs_groups and "policy" not in obs_groups:
        obs_groups["policy"] = obs_groups.pop("actor")
    out["obs_groups"] = obs_groups

    # --- algorithm: strip v5-only fields -------------------------------------
    alg = out.get("algorithm", {})
    alg.pop("share_cnn_encoders", None)
    alg.pop("check_for_nan", None)

    # --- runner: strip v5-only fields ----------------------------------------
    out.pop("check_for_nan", None)

    # --- runner: add deprecated field for compat -----------------------------
    out.setdefault("empirical_normalization", None)

    out["policy"] = policy
    return out


# ---------------------------------------------------------------------------
# v5 → v4  (distribution_cfg → stochastic / init_noise_std / …)
# ---------------------------------------------------------------------------


def translate_runner_dict_to_v4(runner_dict: dict) -> dict:
    """Convert a v5-style runner dict to the v4 format.

    v4 uses separated model dicts (like v5) — ``actor`` / ``critic`` for PPO,
    ``student`` / ``teacher`` for distillation — but a model takes
    ``stochastic``, ``init_noise_std``, ``noise_std_type``,
    ``state_dependent_std`` instead of ``distribution_cfg``. Both v4 runners
    splat each model dict straight into ``MLPModel.__init__``, so an
    untranslated ``distribution_cfg`` is a ``TypeError`` there.

    Parameters
    ----------
    runner_dict : dict
        A runner configuration dict produced by ``RslRlOnPolicyRunnerOptions.dict()``
        or ``RslRlDistillationRunnerOptions.dict()``.
        **Not modified in-place** — a deep copy is returned.

    Returns
    -------
    dict
        A v4-compatible runner dict.
    """
    out = copy.deepcopy(runner_dict)

    # --- convert distribution_cfg to v4 noise params -------------------------
    # Only the heads the dict has: a distillation dict gains no ``actor``.
    for head in ("actor", "student", "teacher"):
        if isinstance(out.get(head), dict):
            _model_to_v4(out[head])

    # --- critic: strip distribution_cfg if present (shouldn't be, but safe) --
    if isinstance(out.get("critic"), dict):
        out["critic"].pop("distribution_cfg", None)

    # --- algorithm: strip v5-only fields -------------------------------------
    alg = out.get("algorithm", {})
    alg.pop("check_for_nan", None)

    # --- runner: strip v5-only fields ----------------------------------------
    out.pop("check_for_nan", None)

    return out


def _model_to_v4(model: dict) -> None:
    """Rewrite one v5 model dict's ``distribution_cfg`` into v4's flat noise params, in place."""
    # Already in v4 format (has stochastic but no distribution_cfg): leave as-is.
    if "stochastic" in model and "distribution_cfg" not in model:
        return
    dist_cfg = model.pop("distribution_cfg", None)
    if dist_cfg is not None:
        dist_class = dist_cfg.get("class_name", "GaussianDistribution")
        model["stochastic"] = True
        model["init_noise_std"] = dist_cfg.get("init_std", 1.0)
        model["noise_std_type"] = dist_cfg.get("std_type", "scalar")
        model["state_dependent_std"] = dist_class == "HeteroscedasticGaussianDistribution"
    else:
        model["stochastic"] = False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


_V5_TO_V3_CLASS_MAP = {
    "MLPModel": "ActorCritic",
    "RNNModel": "ActorCriticRecurrent",
    "CNNModel": "ActorCriticCNN",
}


def _model_class_to_v3(class_name: str) -> str:
    """Map a v5 model class name to its v3.2 ``ActorCritic`` equivalent."""
    return _V5_TO_V3_CLASS_MAP.get(class_name, class_name)
