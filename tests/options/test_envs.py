"""``EnvOptions`` reaching the built rigid solver: the convergence tolerance and the contact-friction switches."""

from __future__ import annotations

import genesis as gs
import pytest
import torch

from eden.options import BoxOptions, CudaGraphOptions, EnvOptions, PlaneOptions, SceneOptions
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env

pytestmark = pytest.mark.requires_genesis


def _build(**env_options):
    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=1, **env_options),
            scene_options=SceneOptions(plane=PlaneOptions(), box=BoxOptions(size=(0.2, 0.2, 0.2))),
        )
    )


@pytest.fixture(scope="module")
def default_env():
    return _build()


@pytest.fixture(scope="module")
def tuned_env():
    return _build(tolerance=1e-4, enable_torsional_friction=True, enable_rolling_friction=True)


def test_the_default_tolerance_is_delegated_to_genesis(default_env):
    """The tolerance is a ratio Genesis quotes against its own reference (re-scaled in genesis-world#3194).

    A literal Eden default would have silently loosened every env tenfold at that bump.
    """
    if gs.tc_float != torch.float32:  # Genesis resolves a different value in double precision
        pytest.skip("the resolved value is precision-dependent")
    assert default_env.rigid_solver._options.tolerance == pytest.approx(1e-6, rel=1e-6)


def test_the_default_friction_setup_is_a_pyramidal_convex_program(default_env):
    solver = default_env.rigid_solver
    assert (solver._options.enable_torsional_friction, solver._options.enable_rolling_friction) == (False, False)
    assert solver._options.friction_cone == gs.friction_cone.pyramidal
    assert solver._contact_resolution == gs.contact_resolution.convex


def test_explicit_options_reach_the_solver(tuned_env):
    solver = tuned_env.rigid_solver
    assert solver._options.tolerance == pytest.approx(1e-4)
    assert (solver._options.enable_torsional_friction, solver._options.enable_rolling_friction) == (True, True)


def test_the_elliptic_cone_resolves_contacts_by_signorini():
    solver = _build(friction_cone="elliptic", solver="newton").rigid_solver
    assert solver._options.friction_cone == gs.friction_cone.elliptic
    assert solver._contact_resolution == gs.contact_resolution.signorini


@pytest.mark.parametrize(
    "env_options, match",
    [
        # Bounding friction against the developed normal force needs a cone that can split.
        (dict(contact_resolution="signorini"), "friction_cone"),
        # The rolling rows sit after the spin row in the contact row layout.
        (dict(enable_rolling_friction=True), "enable_torsional_friction"),
    ],
)
def test_an_impossible_friction_setup_raises(env_options, match):
    with pytest.raises(gs.GenesisException, match=match):
        _build(**env_options)


def test_resetting_the_training_only_fields_resets_exactly_those_and_names_the_changed_ones():
    options = EnvOptions(num_envs=3, cuda_graph_options=CudaGraphOptions(validation_steps=5))

    assert options.reset_training_only_fields() == ["cuda_graph_options"]
    assert options.cuda_graph_options is None
    assert options.num_envs == 3  # not a training-only field
    assert options.reset_training_only_fields() == []  # already at the defaults


def test_every_training_only_field_is_a_declared_field():
    """A misspelt entry would make the replay reset silently skip the field it meant."""
    assert set(EnvOptions.training_only_fields) <= set(EnvOptions.model_fields)


def test_a_graph_needs_at_least_one_validated_replay():
    """A validated replay is what catches work a capture could not record; without one, nothing does."""
    with pytest.raises(ValueError, match="validation_steps"):
        CudaGraphOptions(validation_steps=0)
