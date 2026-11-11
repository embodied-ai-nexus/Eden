"""Predicate metadata parsed from LIBERO MJCF descriptors: where each site sits and which joints move it."""

from __future__ import annotations

from eden.tasks.libero.custom_terms.predicate_assets import (
    _load_asset_predicate_metadata,
    _load_asset_predicate_metadata_cached,
)


def _parse(path):
    stat = path.stat()
    return _load_asset_predicate_metadata_cached(str(path), stat.st_mtime_ns, stat.st_size)


def test_site_on_a_fixed_child_is_composed_into_the_root_frame(tmp_path):
    descriptor = tmp_path / "fixture.xml"
    descriptor.write_text(
        """
<mujoco>
  <worldbody>
    <body name="root">
      <freejoint name="root_joint"/>
      <body name="fixed_child" pos="1 0 0">
        <site name="contain_region" type="box" pos="0 2 0" size="0.1 0.2 0.3"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )

    site = _parse(descriptor).sites["contain_region"]

    assert site.anchor_body_name is None
    assert site.pos_local == (1.0, 2.0, 0.0)
    assert site.quat_local == (1.0, 0.0, 0.0, 0.0)


def test_site_on_an_articulated_body_anchors_to_it_and_keeps_its_joints(tmp_path):
    descriptor = tmp_path / "cabinet.xml"
    descriptor.write_text(
        """
<mujoco>
  <worldbody>
    <body name="root">
      <body name="drawer">
        <joint name="drawer_slide" type="slide"/>
        <site name="drawer_region" type="box" size="0.1 0.2 0.3"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )

    metadata = _parse(descriptor)

    assert metadata.joint_names == ("drawer_slide",)
    assert metadata.sites["drawer_region"].anchor_body_name == "drawer"
    assert metadata.sites["drawer_region"].joint_names == ("drawer_slide",)


def test_registered_cabinet_regions_follow_their_drawers():
    metadata = _load_asset_predicate_metadata("wooden_cabinet")

    assert metadata.joint_names == ("top_level", "middle_level", "bottom_level")
    for level in ("top", "middle", "bottom"):
        site = metadata.sites[f"{level}_region"]
        assert (site.anchor_body_name, site.joint_names) == (f"cabinet_{level}", (f"{level}_level",))
    assert metadata.sites["top_side"].anchor_body_name is None  # the cabinet's fixed top
