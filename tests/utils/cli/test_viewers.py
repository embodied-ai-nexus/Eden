"""Tests for ``eden.utils.cli.viewers``: name resolution and scene prep behind ``eden view`` / ``snap`` / ``arrange``.

Composite rigs are not ``BaseEntityOptions``, so they resolve through ``COMPOSITE_ROBOT_REGISTRY``; one
must go into the scene whole under its ``primary_name`` and expands into several actuated bodies.
"""

import pytest
import torch

from eden.options import ActionManagerOptions, EnvOptions, SceneOptions
from eden.options.entities import CompositeEntityOptions
from eden.options.robots import COMPOSITE_ROBOT_REGISTRY, Droid, KukaSharpa, OliWuji
from eden.utils.configs import EdenConfig
from eden.utils.cli.viewers import (
    _composite_robots,
    _declaration_reaches_base,
    _find_entity_options,
    _viewer_scene_entities,
)
from tests.support.envs import build_env


def test_every_registered_composite_resolves_by_class_name_and_registry_key():
    """The registry is the source of truth, so no rig can be added and left unviewable."""
    assert list(COMPOSITE_ROBOT_REGISTRY.keys())
    for key, cls in COMPOSITE_ROBOT_REGISTRY:
        assert _find_entity_options(key) is cls
        assert _find_entity_options(cls.__name__) is cls


def test_single_entity_lookup_is_untouched():
    # The catalog path still answers class names, bare keys and namespaced keys.
    assert _find_entity_options("Oli").__name__ == "Oli"
    assert _find_entity_options("g1_dofs29").__name__ == "G1Dofs29"
    assert _find_entity_options("robot/franka_emika_panda").__name__ == "FrankaEmikaPanda"


def test_no_composite_name_shadows_a_catalog_entry():
    """``ur5e`` / ``ur10e`` were keys for both the bare arms and the Robotiq rigs, so ``eden view ur5e`` changed meaning."""
    from eden.options.catalog import entity_options_index

    shadowed = sorted(set(_composite_robots()) & set(entity_options_index()))
    assert shadowed == [], f"composite names shadowing catalog entries: {shadowed}"


@pytest.mark.parametrize(
    "name,expected",
    [("ur5e", "UR5e"), ("ur10e", "UR10e"), ("ur5e_2f85", "UR5eRobotiq"), ("ur10e_2f85", "UR10eRobotiq")],
)
def test_ur_names_split_the_bare_arm_from_the_rig(name, expected):
    assert _find_entity_options(name).__name__ == expected


@pytest.fixture
def droid_also_registered_as_oli():
    """Register the Droid rig under the humanoid's name, and remove the alias afterwards."""
    COMPOSITE_ROBOT_REGISTRY.register(Droid, name="Oli")
    yield
    COMPOSITE_ROBOT_REGISTRY._obj_map.pop("Oli", None)


@pytest.mark.usefixtures("droid_also_registered_as_oli")
def test_a_single_entity_wins_a_name_collision():
    """Precedence, pinned whether or not a collision exists today; the rig stays reachable by class name."""
    assert _find_entity_options("Oli").__name__ == "Oli"  # the humanoid, not the rig
    assert _find_entity_options("Droid") is Droid


def test_unknown_name_suggests_composites_too():
    """``OliWuj`` used to answer only "Did you mean: Oli?"."""
    with pytest.raises(ValueError, match="Composite rigs matching: OliWuji"):
        _find_entity_options("OliWuj")

    # A name close to nothing at all still raises the catalog's own error, unembellished.
    with pytest.raises(ValueError, match="No entity options found") as err:
        _find_entity_options("zzzzNotAThing")
    assert "Composite rigs" not in str(err.value)


def test_composite_goes_into_the_scene_whole_under_its_primary_name():
    """Unexpanded, so the env build creates the welds; pre-expanding would drop the attachments."""
    scene, _ = _viewer_scene_entities(OliWuji, show_collision=False, fixed_base=True, hold_pose=True)
    assert list(scene) == ["robot"]
    assert isinstance(scene["robot"], CompositeEntityOptions)
    SceneOptions(**scene)  # the placement check accepts it under exactly this key


def test_composite_pins_only_the_root_body():
    """A welded child is posed by its parent link; pinning it to the world too would fight the weld."""
    scene, _ = _viewer_scene_entities(OliWuji, show_collision=False, fixed_base=True, hold_pose=True)
    subs = scene["robot"].entity_options()
    assert subs["robot"].is_fixed_base
    assert not subs["left_hand"].is_fixed_base
    assert not subs["right_hand"].is_fixed_base


def test_composite_reports_every_actuated_body():
    """One slider group and one PD controller per body — 31 + 20 + 20 for OliWuji."""
    _, dofs = _viewer_scene_entities(OliWuji, show_collision=False, fixed_base=True, hold_pose=True)
    assert {name: len(d) for name, d in dofs.items()} == {"robot": 31, "left_hand": 20, "right_hand": 20}

    _, droid_dofs = _viewer_scene_entities(Droid, show_collision=False, fixed_base=True, hold_pose=True)
    assert set(droid_dofs) == {"robot", "gripper"}


@pytest.mark.parametrize("show_collision", [False, True])
def test_viewer_knobs_reach_composite_sub_entities(show_collision):
    scene, _ = _viewer_scene_entities(OliWuji, show_collision=show_collision, fixed_base=True, hold_pose=True)
    expected = "collision" if show_collision else "visual"
    for sub in scene["robot"].entity_options().values():
        assert sub.vis_mode == expected


def test_hold_pose_adds_gravity_compensation_without_dropping_the_rigs_own_material():
    """KukaSharpa declares ``friction=0.5`` on both halves; compensation is set *on* that material."""
    scene, _ = _viewer_scene_entities(KukaSharpa, show_collision=False, fixed_base=True, hold_pose=True)
    for sub in scene["robot"].entity_options().values():
        assert sub.material.gravity_compensation == 1.0
        assert sub.material.friction == 0.5

    # ...and a rig whose sub-entities declare no material gets one.
    scene, _ = _viewer_scene_entities(OliWuji, show_collision=False, fixed_base=True, hold_pose=True)
    assert scene["robot"].entity_options()["left_hand"].material.gravity_compensation == 1.0


def test_hold_pose_off_leaves_materials_alone():
    # ``eden snap`` only takes a picture, so it must not quietly restate the rig's dynamics.
    scene, _ = _viewer_scene_entities(OliWuji, show_collision=False, fixed_base=False, hold_pose=False)
    assert scene["robot"].entity_options()["left_hand"].material is None


def test_single_entity_prep_keeps_the_obj_key_and_flat_shape():
    scene, dofs = _viewer_scene_entities(
        _find_entity_options("Oli"), show_collision=False, fixed_base=True, hold_pose=True
    )
    assert list(scene) == ["obj"]
    assert scene["obj"].is_fixed_base
    assert scene["obj"].material.gravity_compensation == 1.0
    assert list(dofs) == ["obj"]
    assert len(dofs["obj"]) == 31


def test_unarticulated_entity_reports_no_dofs():
    # A rigid mesh object gets no controller and no sliders — the passive viewer loop, and no
    # gravity compensation restated onto an object whose pose the user cannot drive anyway.
    scene, dofs = _viewer_scene_entities(
        _find_entity_options("CalvinRedBlock"), show_collision=False, fixed_base=True, hold_pose=True
    )
    assert list(scene) == ["obj"]
    assert dofs == {}
    assert scene["obj"].material is None


def test_pinning_leaves_a_free_mjcf_on_its_own_declaration():
    """Genesis cannot pin an MJCF and the entity raises on a declaration its file contradicts, so a G1 stays free."""
    scene, _ = _viewer_scene_entities(
        _find_entity_options("G1Dofs29"), show_collision=False, fixed_base=True, hold_pose=True
    )
    assert scene["obj"].is_fixed_base is False


def test_pinning_reaches_an_mjcf_its_family_rewrites_from_the_flag():
    # robosuite strips the can's root free joint for a fixture, so pinning works as it does on a URDF.
    scene, _ = _viewer_scene_entities(
        _find_entity_options("RobosuiteCan"), show_collision=False, fixed_base=True, hold_pose=True
    )
    assert scene["obj"].is_fixed_base is True


@pytest.mark.parametrize(
    "name, reaches",
    [
        ("Oli", True),  # URDF: Genesis builds `fixed=` as declared
        ("CalvinRedBlock", True),  # mesh
        ("FrankaEmikaPanda", False),  # plain jointless MJCF: `eden arrange` cannot free it
        ("G1Dofs29", False),  # plain free MJCF: `eden view` cannot pin it
        ("RobosuitePickPlaceBin1", True),  # robosuite fixture: adds the free joint when freed
        ("LiberoTabletopWorkspace", False),  # LIBERO workspace: derived without reading the flag
    ],
)
def test_the_viewers_declare_a_base_only_where_the_flag_reaches_the_asset(name, reaches):
    assert _declaration_reaches_base(_find_entity_options(name)()) is reaches


def test_arrange_rejects_composites_with_an_actionable_message():
    """``eden arrange`` owns one free-base body per scene key, so a welded rig has no meaning there."""
    from eden.utils.cli.viewers import arrange

    with pytest.raises(ValueError, match="does not support composite rigs"):
        arrange(["OliWuji"])


def test_composite_action_terms_drive_every_body():
    """The slider loop splits one flat vector per entity by span; each body must track its own target."""
    from eden.constants import ReferenceSource
    from eden.managers.terms.actions.joint_actions import ImplicitPDController

    scene, dofs_by_entity = _viewer_scene_entities(OliWuji, show_collision=False, fixed_base=True, hold_pose=True)
    env = build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=1),
            scene_options=SceneOptions(**scene),
            action_options=ActionManagerOptions(
                **{
                    f"{name}_joints": ImplicitPDController.configure(
                        entity_name=name, dofs_name=dofs, reference_source=ReferenceSource.ZERO, scale=1.0
                    )
                    for name, dofs in dofs_by_entity.items()
                }
            ),
        ),
        reset=True,
    )

    # One flat slider vector, split per entity by span, exactly as show_entity does.
    spans, total = {}, 0
    for name, dofs in dofs_by_entity.items():
        spans[name] = slice(total, total + len(dofs))
        total += len(dofs)
    reading = torch.zeros((1, total), device=env.device)
    for i, name in enumerate(dofs_by_entity):
        reading[:, spans[name]] = 0.2 * (i + 1)  # a different target per body

    action = {f"{name}_joints": reading[:, span] for name, span in spans.items()}
    for _ in range(200):
        env.step(action)

    for i, (name, dofs) in enumerate(dofs_by_entity.items()):
        entity = env.entities[name]
        _, idx = entity.find_named_dofs_idx_local(dofs, preserve_order=True)
        # Not every joint can hold its target against its limits, but each body moves toward its own.
        assert entity.get_dofs_pos(idx)[0].mean().item() == pytest.approx(0.2 * (i + 1), abs=0.1)
