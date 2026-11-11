"""``eden.options.renderer``: the ray tracer's default environment map is decoded lazily and shared.

``RayTracerOptions.env_surface`` defaults to an ``ImageTexture`` over an 11584x5792 panorama that decodes eagerly,
so as a plain class-body default it ran on every ``import eden``.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

from eden.options import RayTracerOptions
from eden.options.renderer import _env_texture


def test_importing_eden_does_not_decode_the_env_map():
    """Run fresh: an earlier test in this process may already have built one."""
    code = textwrap.dedent(
        """
        import eden  # noqa: F401
        from eden.options.renderer import _env_texture

        assert _env_texture.cache_info().currsize == 0, "importing eden decoded the ray tracer's environment map"
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_instances_share_the_default_pixels_but_not_the_surface():
    """Pydantic used to deep-copy the 192 MB array per instance; only the surface needs to be per-instance."""
    a, b = RayTracerOptions(), RayTracerOptions()
    assert a.env_surface is not b.env_surface
    assert a.env_surface.emissive_texture is _env_texture()
    assert a.env_surface.emissive_texture.image_array is b.env_surface.emissive_texture.image_array
    assert a.env_surface.emissive_texture.image_path.endswith("textures/indoor_bright.png")
