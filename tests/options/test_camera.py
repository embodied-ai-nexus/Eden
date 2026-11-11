"""``eden.options.camera``: config-time validation of the clipping planes and of the overlays' parameter values."""

from __future__ import annotations

import pytest

from eden.entities.camera_overlays import CAMERA_OVERLAY_REGISTRY
from eden.options.camera import CameraOptions, CameraOverlayOptions


@pytest.mark.parametrize("kwargs", [{"cam_near": 0.1}, {"cam_near": 0.1, "cam_far": 7.0}, {"cam_far": 7.0}])
def test_one_sided_or_ordered_planes_are_accepted(kwargs):
    """Either plane may be set alone; the other keeps the renderer's default until build checks the pair."""
    options = CameraOptions(**kwargs)
    assert (options.cam_near, options.cam_far) == (kwargs.get("cam_near"), kwargs.get("cam_far"))


@pytest.mark.parametrize("kwargs", [{"cam_near": -1.0}, {"cam_far": 0.0}, {"cam_near": 2.0, "cam_far": 1.0}])
def test_non_positive_or_inverted_planes_are_rejected(kwargs):
    with pytest.raises(ValueError):
        CameraOptions(**kwargs)


@pytest.mark.parametrize(
    ("key", "params", "match"),
    [
        ("plane_paint_overlay", {"color": (0, 0, 0), "max_gap": 0.0}, "max_gap > 0"),
        (
            "plane_paint_overlay",
            {"color": (0, 255, 0)},
            r"color=\(0\.0, 255\.0, 0\.0\).*each channel a number in \[0, 1\]",
        ),
        ("plane_paint_overlay", {"color": "red"}, r"needs color as tuple\[float, float, float\] \| None; got 'red'"),
        ("background_image_composite", {"image_path": "a.png", "shadow_strength": 0.5}, "needs at least one entry"),
        (
            "background_image_composite",
            {"image_path": "a.png", "entities": ["plane"], "shadow_strength": 1.1},
            r"needs shadow_strength in \[0, 1\]; got 1\.1",
        ),
        ("background_image_composite", {"entities": ["plane"]}, "BackgroundImageComposite needs image_path"),
        ("splat_composite", {"entities": ["plane"], "shadow_strength": -0.1}, r"SplatComposite needs shadow_strength"),
        ("splat_composite", {"shadow_strength": 0.5}, r"SplatComposite\(shadow_strength=0\.5\) needs at least one"),
        # A type the declared one does not admit, named with the parameter's type and default.
        (
            "splat_composite",
            {"entities": "plane"},
            r"SplatComposite needs entities as list\[str\]; got 'plane' \(str\): Input should be a valid list\. "
            r"E\.g\. entities=\[\] \(the default\)",
        ),
        (
            "plane_paint_overlay",
            {"color": (0, 0, 0), "max_gap": None},
            r"needs max_gap as float; got None \(NoneType\)",
        ),
        ("plane_paint_overlay", {"color": (0, 0, 0), "supersample": 2.5}, r"needs supersample as int; got 2\.5"),
        ("background_image_composite", {"image_path": 3}, r"needs image_path as str; got 3 \(int\)"),
        # A bool is not a number, though pydantic's lax mode would take True as 1.0 (`shadow_strength: yes` in YAML).
        (
            "splat_composite",
            {"entities": ["plane"], "shadow_strength": True},
            r"needs shadow_strength as float; got True \(bool\)",
        ),
        ("plane_paint_overlay", {"color": (0, 0, 0), "supersample": False}, r"needs supersample as int; got False"),
    ],
)
def test_an_overlay_value_it_cannot_work_with_fails_where_the_config_writes_it(key, params, match):
    """In configure(), and in an entry read back from a config file, before any camera exists."""
    with pytest.raises(ValueError, match=match):
        CAMERA_OVERLAY_REGISTRY.get(key).configure(**params)
    with pytest.raises(ValueError, match=match):
        CameraOverlayOptions(name=key, **params)


def test_overlay_parameters_take_their_declared_types_as_a_declared_field_would():
    """A number written as a string, or a list read back from JSON, becomes the declared type."""
    composite = CAMERA_OVERLAY_REGISTRY.get("splat_composite").configure(entities=("plane",), shadow_strength="0.5")
    assert (composite.entities, composite.shadow_strength) == (["plane"], 0.5)
    ink = CameraOverlayOptions(name="plane_paint_overlay", metric_name="ink", color=[0, 1, 0])
    assert ink.color == (0.0, 1.0, 0.0)


def test_a_paint_overlay_with_no_colour_is_left_to_its_metric():
    """No colour is for marks that come with their own (a painting's): whether the metric gives them is checked at
    build, against it (``tests/entities/test_camera_overlays.py``).
    """
    options = CAMERA_OVERLAY_REGISTRY.get("plane_paint_overlay").configure(metric_name="painting")
    assert options.color is None
