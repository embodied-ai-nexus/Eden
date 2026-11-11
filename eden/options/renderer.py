"""Renderer configuration options (ray tracer, madrona batch renderer)."""

import functools
from typing import Any

import genesis as gs
from genesis.typing import Vec3FType
from genesis.options.surfaces import Surface
from genesis.options.renderers import BatchRenderer, RayTracer
from pydantic import Field


@functools.cache
def _env_texture() -> gs.textures.ImageTexture:
    """The ray tracer's default environment map, decoded on first use and shared by every instance.

    ``indoor_bright.png`` is an 11584x5792 panorama, so constructing the ``ImageTexture`` decodes
    ~770 MB of peak RSS and holds 192 MB of pixels. A plain class-body default would pay that on
    ``import eden`` -- in every test, CLI call and training run, whether or not anything renders
    with the ray tracer -- and pydantic would deep-copy the pixels again per instance.
    """
    return gs.textures.ImageTexture(image_path="textures/indoor_bright.png")


def _default_env_surface() -> Surface:
    """A fresh default environment surface wrapping the shared (lazily decoded) texture."""
    return gs.surfaces.Emission(emissive_texture=_env_texture())


class RayTracerOptions(RayTracer):
    """Options for the ray-tracing renderer.

    Parameters
    ----------
    env_surface : Surface, optional
        Environment surface.
    env_radius : float, optional
        Environment radius.
    env_pos : array-like[float, float, float], optional
        Environment position.
    env_euler : array-like[float, float, float], optional
        The euler angles of the environment, in degrees.
    lights : list of dict, optional
        List of lights. Each light is a dictionary with keys: ['pos', 'color', 'intensity', 'radius'].
        - 'pos': array-like[float, float, float]
        - 'color': array-like[float, float, float]
        - 'intensity': float
        - 'radius': float
    """

    state_limit: int = 2**25
    tracing_depth: int = 32
    rr_depth: int = 0
    rr_threshold: float = 0.95

    env_surface: Surface = Field(default_factory=_default_env_surface)
    env_radius: float = 15.0
    env_pos: Vec3FType = (0.0, 0.0, 0.0)
    env_euler: Vec3FType = (0, 0, 180)
    lights: list[dict[str, Any]] = [
        {
            "pos": (0.0, 0.0, 10.0),
            "radius": 2.0,
            "color": (15.0, 15.0, 15.0),
            "intensity": 10.0,
        },
    ]


class BatchRendererOptions(BatchRenderer):
    """Options for the gs-madrona batch renderer, with declarative scene lights.

    Madrona renders **ambient-only until lights are added** (cameras come out
    near-black), and it bakes the light count at ``scene.build()`` time — so
    lights must be registered via ``gs.Scene.add_light`` *before* the build.
    The raw genesis ``gs.renderers.BatchRenderer`` has no ``lights`` field, so
    this eden subclass carries them declaratively (mirroring
    :class:`RayTracerOptions.lights`); :class:`eden.envs.base.EnvBase` applies
    them right after scene creation.

    Parameters
    ----------
    lights : list of dict, optional
        One kwargs dict per ``gs.Scene.add_light`` call, e.g.
        ``{"pos": (0.5, 0.5, 2.0), "dir": (0, 0, -1), "intensity": 3.0,
        "directional": True, "castshadow": False}``.
    """

    lights: list[dict[str, Any]] = []
