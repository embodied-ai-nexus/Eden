"""``RslRlContrackRunnerOptions`` refuses the inherited fields the rsl_rl_contrack fork would silently ignore."""

import pytest

from eden.learning.wrappers.rsl_rl_contrack import RslRlContrackAlgorithmOptions, RslRlContrackRunnerOptions
from eden.options.learning.rsl_rl import (
    RslRlCriticOptions,
    RslRlCriticRNNOptions,
    RslRlGaussianDistributionOptions,
    RslRlModelWithDistributionOptions,
    RslRlRndOptions,
)


def test_the_defaults_and_the_contrack_task_build():
    from eden.tasks import TASK_REGISTRY

    RslRlContrackRunnerOptions()
    assert isinstance(TASK_REGISTRY.build("contrack").runner_options, RslRlContrackRunnerOptions)


@pytest.mark.parametrize(
    "fields, refused",
    [
        ({"critic": RslRlCriticRNNOptions(hidden_dims=[64])}, "critic.class_name"),
        ({"critic": RslRlCriticOptions(hidden_dims=[64], activation="tanh")}, "critic.activation"),
        (
            {
                "actor": RslRlModelWithDistributionOptions(
                    distribution_cfg=RslRlGaussianDistributionOptions(std_type="scalar", std_range=(0.1, 2.0))
                )
            },
            "actor.distribution_cfg",
        ),
        ({"algorithm": RslRlContrackAlgorithmOptions(rnd_cfg=RslRlRndOptions())}, "algorithm.rnd_cfg"),
        ({"algorithm": RslRlContrackAlgorithmOptions(share_cnn_encoders=True)}, "algorithm.share_cnn_encoders"),
    ],
    ids=["recurrent_critic", "critic_activation", "std_range", "rnd", "shared_cnn"],
)
def test_a_field_the_fork_cannot_honour_is_refused_by_name(fields, refused):
    with pytest.raises(ValueError, match=rf"the rsl_rl_contrack fork cannot honour \{{'{refused}'"):
        RslRlContrackRunnerOptions(**fields)
