"""Translation from Eden rl_games options to the nested dict rl_games expects.

Typical usage::

    from eden.options.learning.rl_games.translate import translate_runner_dict

    runner_dict = translate_runner_dict(config.runner_options.dict(), num_envs=env.num_envs)

    from rl_games.torch_runner import Runner
    runner = Runner()
    runner.load(runner_dict)
"""

from __future__ import annotations

from typing import Any


def translate_runner_dict(runner_dict: dict[str, Any], *, num_envs: int) -> dict[str, Any]:
    """Convert a flat Eden options dict into the nested ``params`` dict rl_games expects.

    The output matches what ``rl_games.torch_runner.Runner.load()`` expects.

    Parameters
    ----------
    runner_dict:
        A dict produced by ``RlGamesPpoRunnerOptions.dict()``.
    num_envs:
        Number of parallel environments (injected by the wrapper).

    Returns
    -------
    dict
        The full ``{"params": {...}}`` dict ready for ``Runner.load()``.
    """
    from eden.utils.torch import resolve_device

    d = runner_dict
    net = d.get("network", {})

    minibatch_size = d.get("minibatch_size")
    if minibatch_size is None:
        # Full batch. A multi-agent env (`agents`) gives rl_games `num_envs * len(agents)` rows a step;
        # `agents` itself is a wrapper concern rl_games never reads, so it is not forwarded.
        minibatch_size = d.get("horizon_length", 24) * num_envs * max(len(d.get("agents") or ()), 1)

    lr_schedule = d.get("lr_schedule", "adaptive")
    if lr_schedule == "None":
        lr_schedule = None

    return {
        "params": {
            "seed": d.get("seed", 7100),
            "algo": {"name": "a2c_continuous"},
            "model": {"name": "continuous_a2c_logstd"},
            "network": {
                "name": net.get("name", "actor_critic"),
                "separate": net.get("separate", True),
                "space": {
                    "continuous": {
                        "mu_activation": net.get("mu_activation", "None"),
                        "sigma_activation": net.get("sigma_activation", "None"),
                        "mu_init": {"name": net.get("mu_init_name", "default")},
                        "sigma_init": {
                            "name": "const_initializer",
                            "val": net.get("init_sigma_val", 0.0),
                        },
                        "fixed_sigma": net.get("fixed_sigma", True),
                    },
                },
                "mlp": {
                    "units": list(net.get("units", [512, 256, 128])),
                    "activation": net.get("activation", "elu"),
                    "initializer": {"name": "default"},
                },
            },
            "config": {
                "name": d.get("experiment_name", ""),
                "env_name": "eden",
                "device": resolve_device(d.get("device", "auto")),
                "num_actors": num_envs,
                "horizon_length": d.get("horizon_length", 24),
                "minibatch_size": minibatch_size,
                "mini_epochs": d.get("mini_epochs", 5),
                "learning_rate": d.get("learning_rate", 1e-3),
                "lr_schedule": lr_schedule,
                "kl_threshold": d.get("kl_threshold", 0.01),
                "gamma": d.get("gamma", 0.99),
                "tau": d.get("tau", 0.95),
                "e_clip": d.get("e_clip", 0.2),
                "clip_value": d.get("clip_value", True),
                "entropy_coef": d.get("entropy_coef", 0.01),
                "critic_coef": d.get("critic_coef", 1.0),
                "grad_norm": d.get("grad_norm", 1.0),
                "truncate_grads": d.get("truncate_grads", True),
                "normalize_input": d.get("normalize_input", False),
                "normalize_value": d.get("normalize_value", False),
                "normalize_advantage": d.get("normalize_advantage", True),
                "value_bootstrap": d.get("value_bootstrap", True),
                "reward_shaper": {"scale_value": d.get("reward_scale", 1.0)},
                "bounds_loss_coef": d.get("bounds_loss_coef", 0.0),
                # Disable rl_games' built-in action rescaling — Eden
                # handles action scaling internally via controllers.
                "clip_actions": False,
                "max_epochs": d.get("max_epochs", 30000),
                "save_frequency": d.get("save_frequency", 500),
                "torch_compile": d.get("torch_compile", True),
                "print_stats": True,
            },
        },
    }
