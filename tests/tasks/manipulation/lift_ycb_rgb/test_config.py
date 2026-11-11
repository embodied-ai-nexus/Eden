"""Tests for ``eden.tasks.manipulation.lift_ycb_rgb.config``: per-env resting heights for the YCB object bank."""

from __future__ import annotations

import pytest
import torch

from eden.options import EnvOptions, SceneOptions
from eden.options.entities import GroupedEntityOptions
from eden.options.objects.ycb import YCB003CrackerBox, YCB013Apple, YCB061FoamBrick
from eden.tasks import TASK_REGISTRY
from eden.tasks.manipulation.lift_ycb_rgb.config import _YCB_PHYSICAL, LiftYCBObjectPropertiesMod
from eden.utils.configs import EdenConfig


def test_ycb_gates_read_the_per_env_rest_height_metric():
    cfg = TASK_REGISTRY.build("manipulation_lift_ycb_rgb")
    assert isinstance(cfg.scene_options.cube, GroupedEntityOptions)
    metric_names = set(cfg.metric_options.term_keys())
    for name in ("lift_height", "lifting_object", "object_goal_tracking", "object_goal_tracking_fine"):
        params = getattr(cfg.reward_options, name).params
        assert params.get("rest_z") is None, f"{name} still carries a scalar rest_z"
        assert params["rest_z_metric"] in metric_names, f"{name} names an unconfigured metric"


def test_object_rest_height_stays_out_of_success_accounting():
    """A resting height trivially clears any threshold, so it would mark every episode successful.

    Checked on the serialized options too, or a reconstructed config re-enters accounting.
    """
    metric_options = TASK_REGISTRY.build("manipulation_lift_ycb_rgb").with_overrides_from_dict({}).metric_options
    assert metric_options.object_rest_height.contributes_to_success is False


def test_every_ycb_variant_keeps_its_own_friction():
    """Each variant sets its own friction; a regression dropped it and ran the bank at the default."""
    bank = TASK_REGISTRY.build("manipulation_lift_ycb_rgb").scene_options.cube

    assert {material.friction for material in bank.variant_materials} == {2.0}


def test_ycb_object_properties_give_each_object_its_own_density_and_friction():
    bank = TASK_REGISTRY.build("manipulation_lift_ycb_rgb", object_properties="ycb").scene_options.cube

    for variant, material in zip(bank.grouped_entities, bank.variant_materials):
        mass_g, volume_cm3, friction = _YCB_PHYSICAL[type(variant)]
        assert material.rho * volume_cm3 / 1000.0 == pytest.approx(mass_g * variant.scale**3)
        assert material.friction == friction
    assert len({m.rho for m in bank.variant_materials}) == len(bank.grouped_entities)


def _bank_config(*variants, **group_fields):
    """``manipulation_lift_ycb_rgb``'s config with its bank replaced by ``variants``, as a subclass or earlier mod would."""
    config = TASK_REGISTRY.build("manipulation_lift_ycb_rgb")
    config.scene_options.cube = GroupedEntityOptions(grouped_entities=list(variants), **group_fields)
    return config


def test_ycb_object_properties_edit_the_bank_in_the_config_and_keep_what_set_it():
    """The mod is inherited by a subclass with its own object set, which a rebuild from this module would replace."""
    config = _bank_config(YCB013Apple(), YCB061FoamBrick(), default_root_pos=(0.1, 0.2, 0.9))

    bank = LiftYCBObjectPropertiesMod("ycb").apply(config).scene_options.cube

    assert [type(v) for v in bank.grouped_entities] == [YCB013Apple, YCB061FoamBrick]
    assert bank.default_root_pos == (0.1, 0.2, 0.9)
    assert [m.friction for m in bank.variant_materials] == [1.2, 2.0]


def test_ycb_object_properties_refuse_an_object_they_hold_no_properties_for():
    config = _bank_config(YCB013Apple(), YCB003CrackerBox())

    with pytest.raises(
        ValueError,
        match=r"no physical properties for scene_options.cube's grouped_entities\[1\] "
        r"\(YCB003CrackerBox\).*--object_properties uniform",
    ):
        LiftYCBObjectPropertiesMod("ycb").apply(config)


@pytest.mark.requires_genesis
def test_each_ycb_object_builds_at_its_published_mass_scaled_with_its_mesh():
    """Pins the measured volumes in ``_YCB_PHYSICAL``: they are Genesis' convexified meshes, which a Genesis bump or
    a re-exported asset could change without any error.
    """
    from tests.support.envs import build_env

    bank = TASK_REGISTRY.build("manipulation_lift_ycb_rgb", object_properties="ycb").scene_options.cube
    try:
        for variant in bank.grouped_entities:
            variant.fetch_asset()
    except Exception as error:  # the YCB meshes are a Hugging Face download
        pytest.skip(f"YCB assets unavailable: {error}")
    num_envs = len(bank.grouped_entities)
    cube = build_env(EdenConfig(env_options=EnvOptions(num_envs=num_envs), scene_options=SceneOptions(cube=bank)))
    built = cube.entities["cube"].get_mass().cpu()

    expected = [_YCB_PHYSICAL[type(v)][0] * v.scale**3 / 1000.0 for v in bank.grouped_entities]
    torch.testing.assert_close(built, torch.tensor(expected, dtype=built.dtype), rtol=0.01, atol=0.0)


def test_uniform_object_properties_are_the_default():
    default = TASK_REGISTRY.build("manipulation_lift_ycb_rgb").scene_options.cube
    uniform = TASK_REGISTRY.build("manipulation_lift_ycb_rgb", object_properties="uniform").scene_options.cube

    assert default == uniform
    assert {(m.rho, m.friction) for m in default.variant_materials} == {(200.0, 2.0)}
