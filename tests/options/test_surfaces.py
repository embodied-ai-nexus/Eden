"""``eden.options.surfaces``: surfaces declared before ``gs.init()`` and resolved into Genesis ones at build."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import genesis as gs
import pytest

from eden.options import BoxOptions, EnvOptions, PlaneOptions, SceneOptions
from eden.options.surfaces import (
    BSDFSurfaceOptions,
    EmissionSurfaceOptions,
    PlasticSurfaceOptions,
    SurfaceOptions,
    resolve_surface,
)
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env


def test_a_config_holding_surfaces_constructs_before_gs_init():
    """A raw ``gs.surfaces.*`` needs an initialized Genesis; a config built at import time must not.

    Run in a fresh interpreter so ``gs.init()`` has definitely not been called.
    """
    script = textwrap.dedent(
        """
        import genesis as gs
        assert not gs._initialized, "precondition: genesis must be uninitialized"

        from eden.options.entities import BoxOptions
        from eden.options.surfaces import PlasticSurfaceOptions

        BoxOptions(size=(0.1, 0.1, 0.1), surface=PlasticSurfaceOptions(color=(1.0, 0.0, 0.0)))
        BoxOptions(size=(0.1, 0.1, 0.1), color=(0.0, 0.0, 1.0))
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("options", "genesis_cls", "fields", "color"),
    [
        (
            PlasticSurfaceOptions(color=(1.0, 0.0, 0.0), roughness=0.5),
            gs.surfaces.Plastic,
            {"roughness": 0.5},
            (1.0, 0.0, 0.0),
        ),
        (EmissionSurfaceOptions(color=(0.0, 1.0, 0.0)), gs.surfaces.Emission, {}, None),
        (SurfaceOptions(), gs.surfaces.Default, {}, None),
        (
            BSDFSurfaceOptions(color=(0.2, 0.4, 0.85), metallic=0.0, roughness=0.25),
            gs.surfaces.BSDF,
            {"metallic": 0.0, "roughness": 0.25},
            (0.2, 0.4, 0.85),
        ),
    ],
    ids=["plastic", "emission", "default", "bsdf"],
)
def test_to_genesis_surface_builds_the_matching_surface(options, genesis_cls, fields, color):
    surface = options.to_genesis_surface()
    assert isinstance(surface, genesis_cls)
    assert {name: getattr(surface, name) for name in fields} == fields
    if color is not None:
        assert tuple(surface.diffuse_texture.color) == color


def test_resolve_surface_passes_a_built_surface_and_none_through():
    built = gs.surfaces.Plastic(color=(0.0, 0.0, 1.0))
    assert resolve_surface(built) is built
    assert resolve_surface(None) is None


def test_surface_options_serialize_as_plain_data():
    box = BoxOptions(size=(0.2, 0.2, 0.2), surface=PlasticSurfaceOptions(color=(1.0, 0.0, 0.0)))
    assert box.dict()["surface"]["color"] == (1.0, 0.0, 0.0)


def test_surface_and_color_shortcut_reach_the_built_entity():
    """``surface=`` resolves at build; ``color=`` is shorthand for a plastic surface of that colour."""
    env = build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=1),
            scene_options=SceneOptions(
                plane=PlaneOptions(),
                red=BoxOptions(size=(0.2, 0.2, 0.2), surface=PlasticSurfaceOptions(color=(1.0, 0.0, 0.0))),
                blue=BoxOptions(size=(0.2, 0.2, 0.2), color=(0.0, 0.0, 1.0)),
            ),
        )
    )
    try:
        for name, color in (("red", (1.0, 0.0, 0.0)), ("blue", (0.0, 0.0, 1.0))):
            surface = env.entities[name].entity.surface
            assert isinstance(surface, gs.surfaces.Plastic)
            assert tuple(surface.diffuse_texture.color) == color
    finally:
        env.destroy()
