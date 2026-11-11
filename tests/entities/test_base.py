"""``Entity``: mirrored options share nothing mutable, grouped variants get their own poses and joints, and base-DOF reads ask the right flag."""

from __future__ import annotations

import ast
import pathlib

import pytest
from pydantic import BaseModel

from eden.entities.rigid import RigidEntity
from eden.entities.terrain import Terrain
import torch

from eden.options import BoxOptions, EntityOptions, EnvOptions, GroupedEntityOptions, PlaneOptions, SceneOptions
from eden.options.entities import BaseEntityOptions, TerrainOptions
from eden.options.materials import KinematicMaterialOptions, RigidMaterialOptions
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env

_EDEN = pathlib.Path(__file__).resolve().parents[2] / "eden"


_POS_1, _POS_2 = (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)
_QUAT_1, _QUAT_2 = (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)
_BOX = '<geom type="box" size="{h} {h} {h}" pos="{x} 0 0" mass="0.1"/>'
# Three free-body variants; ``wide`` carries two geoms, so its geom index is not its variant index.
_VARIANT_BODIES = {
    "wide": _BOX.format(h=0.02, x=-0.03) + _BOX.format(h=0.02, x=0.03),  # x extent 0.10
    "small": _BOX.format(h=0.02, x=0.0),  # 0.04
    "large": _BOX.format(h=0.04, x=0.0),  # 0.08
}
# Two variants of one topology (a free root, a lid on a hinge) whose lid frame and joint params differ.
_HINGED_BODY = (
    '<body name="root"><joint name="root" type="free" damping="{free_damping}"/>'
    '<geom type="box" size="0.05 0.05 0.02" mass="0.2"/>'
    '<body name="lid" pos="0 0 {lid_z}"><joint name="hinge" type="hinge" axis="0 1 0" range="0 {upper}" '
    'damping="{damping}" frictionloss="{frictionloss}" actuatorfrcrange="-{effort} {effort}"/>'
    '<geom type="box" size="0.05 0.05 0.005" mass="0.05"/></body></body>'
)
_HINGED_VARIANTS = (
    {
        "lid_z": 0.03,
        "upper": 1.5,
        "damping": 0.01,
        "frictionloss": 0.0,
        "effort": 50.0,
        "free_damping": 0.0,
        "kp": 10.0,
    },
    {
        "lid_z": 0.06,
        "upper": 2.5,
        "damping": 0.2,
        "frictionloss": 0.02,
        "effort": 1000.0,
        "free_damping": 0.5,
        "kp": 40.0,
    },
)
NUM_ENVS = 6


def _two_box_group(**group_fields) -> GroupedEntityOptions:
    return GroupedEntityOptions(
        grouped_entities=[
            BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=_POS_1, default_root_quat=_QUAT_1),
            BoxOptions(size=(0.2, 0.2, 0.2), default_root_pos=_POS_2, default_root_quat=_QUAT_2),
        ],
        **group_fields,
    )


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """The *same* options object placed twice (where sharing bites), and three heterogeneous groups.

    Genesis assigns a group's variants to contiguous env blocks: two variants over six envs are 0,0,0,1,1,1.
    """
    assets = tmp_path_factory.mktemp("variants")
    variants = []
    for (name, geoms), z in zip(_VARIANT_BODIES.items(), (0.0, 0.3, 0.6)):
        path = assets / f"{name}.xml"
        path.write_text(
            f'<mujoco model="{name}"><worldbody><body name="root"><freejoint/>{geoms}</body></worldbody></mujoco>'
        )
        variants.append(EntityOptions(file=str(path), default_root_pos=(0.0, 0.0, z)))
    hinged, driven = [], []
    for i, params in enumerate(_HINGED_VARIANTS):
        body = _HINGED_BODY.format(**params)
        path = assets / f"hinged_{i}.xml"
        path.write_text(f'<mujoco model="hinged_{i}"><compiler angle="radian"/><worldbody>{body}</worldbody></mujoco>')
        hinged.append(EntityOptions(file=str(path), is_articulated=True, default_root_pos=(3.0, 0.0, 0.3)))
        # The same body, driven by a position actuator whose gain differs between the variants.
        path = assets / f"driven_{i}.xml"
        path.write_text(
            f'<mujoco model="driven_{i}"><compiler angle="radian"/><worldbody>{body}</worldbody>'
            f'<actuator><position joint="hinge" kp="{params["kp"]}"/></actuator></mujoco>'
        )
        driven.append(
            EntityOptions(file=str(path), is_articulated=True, is_actuated=True, default_root_pos=(5.0, 0.0, 0.3))
        )
    shared = BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.0, 0.0, 0.3))
    built = build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=NUM_ENVS),
            scene_options=SceneOptions(
                plane=PlaneOptions(),
                box_a=shared,
                box_b=shared,
                group=_two_box_group(),
                offset_group=_two_box_group(default_root_pos=(10.0, 0.0, 0.0)),
                mixed=GroupedEntityOptions(grouped_entities=variants, default_root_pos=(0.0, 0.0, 0.5)),
                hinged=GroupedEntityOptions(grouped_entities=hinged, dofs_name=["hinge"]),
                driven=GroupedEntityOptions(grouped_entities=driven, dofs_name=["hinge"]),
                # In the kinematic solver, which batches its DOF info by itself once it holds a group.
                ghost=GroupedEntityOptions(
                    grouped_entities=[variant.model_copy() for variant in hinged],
                    dofs_name=["hinge"],
                    material=KinematicMaterialOptions(),
                ),
            ),
        )
    )
    return built, shared


@pytest.mark.requires_genesis
def test_entities_built_from_one_options_instance_share_nothing_mutable(env):
    built, shared = env
    a, b = built.entities["box_a"], built.entities["box_b"]
    # ``material`` and ``surface`` are mutable pydantic models: aliasing one rewrites the config too.
    for name in type(a).get_parameter_names():
        value = getattr(a, name)
        if isinstance(value, (list, dict, set, BaseModel)):
            assert value is not getattr(type(a), name, None), f"{name} aliases the class default"
            assert value is not getattr(a.options, name, None), f"{name} aliases the options"

    a.support_links_name.append("leaked")
    assert b.support_links_name == [], "two entities built from one options instance share state"
    assert shared.support_links_name == [], "mutating an entity rewrote its options"


@pytest.mark.requires_genesis
@pytest.mark.parametrize("name, offset", [("group", (0.0, 0.0, 0.0)), ("offset_group", (10.0, 0.0, 0.0))])
def test_each_env_gets_the_default_pose_of_the_variant_it_holds(env, name, offset):
    """A group-level pose composes onto each variant's own."""
    pos, quat = env[0].entities[name].get_default_root_pose()
    torch.testing.assert_close(pos, (torch.tensor([_POS_1] * 3 + [_POS_2] * 3) + torch.tensor(offset)).to(pos.device))
    torch.testing.assert_close(quat, torch.tensor([_QUAT_1] * 3 + [_QUAT_2] * 3, device=quat.device))


@pytest.mark.requires_genesis
def test_a_multi_geom_variant_does_not_shift_the_other_variants_poses(env):
    """The pose was once read off ``links[0].geoms[i]`` as if geom ``i`` were variant ``i``.

    Each env's geometry, placed by Genesis, is checked against the pose computed for it.
    """
    mixed = env[0].entities["mixed"]
    expected_z = torch.tensor([0.5, 0.5, 0.8, 0.8, 1.1, 1.1])
    torch.testing.assert_close(mixed.get_default_root_pose()[0][:, 2].cpu(), expected_z)
    aabb = mixed.get_AABB().cpu()
    torch.testing.assert_close(aabb[:, 1, 0] - aabb[:, 0, 0], torch.tensor([0.10, 0.10, 0.04, 0.04, 0.08, 0.08]))
    torch.testing.assert_close(aabb[:, :, 2].mean(dim=-1), expected_z, atol=1e-5, rtol=0)


@pytest.mark.requires_genesis
def test_env_variant_idx_names_the_variant_genesis_placed_in_each_env(env):
    """Read against the built geometry: each env's AABB width identifies the variant that is really there."""
    mixed = env[0].entities["mixed"]
    widths = {0.10: 0, 0.04: 1, 0.08: 2}
    aabb = mixed.get_AABB().cpu()
    placed = [widths[round(float(w), 2)] for w in aabb[:, 1, 0] - aabb[:, 0, 0]]
    assert mixed.num_variants == 3
    assert mixed.env_variant_idx.tolist() == placed == [0, 0, 1, 1, 2, 2]


@pytest.mark.requires_genesis
def test_env_variant_idx_is_undefined_for_an_entity_every_env_shares(env):
    box = env[0].entities["box_a"]
    for attr in ("env_variant_idx", "num_variants"):
        with pytest.raises(TypeError, match=rf"`{attr}` is only defined for a heterogeneous entity.*BoxOptions"):
            getattr(box, attr)


@pytest.mark.requires_genesis
def test_env_variant_idx_follows_genesis_when_the_envs_do_not_divide_evenly():
    """Five envs over two variants: the blocks are uneven, and the index still matches the placed geometry."""
    built = build_env(
        EdenConfig(env_options=EnvOptions(num_envs=5), scene_options=SceneOptions(group=_two_box_group()))
    )
    group = built.entities["group"]
    aabb = group.get_AABB().cpu()
    placed = [0 if float(w) < 0.15 else 1 for w in aabb[:, 1, 0] - aabb[:, 0, 0]]
    assert group.env_variant_idx.tolist() == placed
    assert sorted(set(placed)) == [0, 1]


def _per_env(values: torch.Tensor) -> list[float]:
    """One value per env, from a single-DOF ``(num_envs, 1)`` tensor."""
    return values.squeeze(1).tolist()


def _per_variant_env(key: str) -> list[float]:
    """``_HINGED_VARIANTS``' ``key`` for each env: Genesis gives two variants over six envs as 0,0,0,1,1,1."""
    return [params[key] for params in _HINGED_VARIANTS for _ in range(NUM_ENVS // 2)]


@pytest.mark.requires_genesis
def test_each_env_simulates_the_link_frames_and_joint_params_of_the_variant_it_holds(env):
    """Genesis simulated the primary's joints everywhere, then refused differing variants.

    Fixed upstream in https://github.com/Genesis-Embodied-AI/Genesis/pull/3497.
    """
    hinged = env[0].entities["hinged"]
    assert _per_env(hinged.get_dofs_damping()) == pytest.approx(_per_variant_env("damping"))
    assert _per_env(hinged.get_dofs_frictionloss()) == pytest.approx(_per_variant_env("frictionloss"))
    assert _per_env(hinged.get_dofs_limit()[1]) == pytest.approx(_per_variant_env("upper"))
    # post_build used to write variant 0's force range over every env.
    assert _per_env(hinged.get_dofs_force_range()[1]) == pytest.approx(_per_variant_env("effort"))
    lid = hinged.get_link("lid").idx_local
    lid_height = hinged.get_links_pos()[:, lid, 2] - hinged.get_links_pos()[:, hinged.get_link("root").idx_local, 2]
    assert lid_height.tolist() == pytest.approx(_per_variant_env("lid_z"), abs=1e-5)


@pytest.mark.requires_genesis
def test_the_dof_defaults_of_a_heterogeneous_entity_are_each_variants_own(env):
    """What the reward, termination and reset terms read; it held variant 0's values for every env."""
    hinged = env[0].entities["hinged"]
    assert _per_env(hinged.default_dofs_damping) == pytest.approx(_per_variant_env("damping"))
    assert _per_env(hinged.default_dofs_frictionloss) == pytest.approx(_per_variant_env("frictionloss"))
    assert _per_env(hinged.default_dofs_force_limits) == pytest.approx(_per_variant_env("effort"))
    assert _per_env(hinged.default_dofs_pos_limits[..., 1]) == pytest.approx(_per_variant_env("upper"))
    soft_upper = [0.5 * upper + 0.9 * 0.5 * upper for upper in _per_variant_env("upper")]  # 0.9 about the midpoint
    assert _per_env(hinged.soft_dofs_pos_limits[..., 1]) == pytest.approx(soft_upper)


@pytest.mark.requires_genesis
def test_the_free_joint_damping_bump_keeps_each_variants_own_damping(env):
    """post_build adds 2e-4 to a non-actuated floating base's free-joint damping; it wrote variant 0's everywhere."""
    hinged = env[0].entities["hinged"]
    free_dofs = [idx for joint in hinged.free_joints for idx in joint.dofs_idx_local]
    damping = hinged.get_dofs_damping(dofs_idx_local=free_dofs)
    expected = [[free_damping + 0.0002] * 6 for free_damping in _per_variant_env("free_damping")]
    assert damping.tolist() == [pytest.approx(row) for row in expected]


@pytest.mark.requires_genesis
def test_the_pd_gains_of_a_heterogeneous_entity_are_each_variants_own(env):
    driven = env[0].entities["driven"]
    assert driven.get_dofs_kp().squeeze(-1).tolist() == pytest.approx(_per_variant_env("kp"))
    assert driven.default_dofs_kp.squeeze(-1).tolist() == pytest.approx(_per_variant_env("kp"))


@pytest.mark.requires_genesis
def test_a_kinematic_heterogeneous_entity_reads_its_dof_info_per_env(env):
    """Eden took the kinematic solver's DOF info to be unbatched, and its post_build failed on the shape."""
    ghost = env[0].entities["ghost"]
    assert ghost.has_batched_dofs_info
    assert ghost.get_dofs_limit()[1].squeeze(-1).tolist() == pytest.approx(_per_variant_env("upper"))
    assert ghost.default_dofs_pos_limits[..., 1].squeeze(-1).tolist() == pytest.approx(_per_variant_env("upper"))


@pytest.mark.requires_genesis
def test_qpos_limit_follows_the_qpos_layout_of_a_floating_base(env):
    """7 root qs (an unlimited position, a quaternion bounded by the unit) lead each variant's own hinge limit."""
    hinged = env[0].entities["hinged"]
    lower, upper = hinged.get_qpos_limit()
    assert lower.shape == upper.shape == hinged.get_qpos().shape
    inf = float("inf")
    for i_env, upper_hinge in enumerate(_per_variant_env("upper")):
        assert lower[i_env].tolist() == pytest.approx([-inf] * 3 + [-1.0] * 4 + [0.0])
        assert upper[i_env].tolist() == pytest.approx([inf] * 3 + [1.0] * 4 + [upper_hinge])
    torch.testing.assert_close(hinged.get_qpos_limit(envs_idx=4)[1], upper[4:5])


def test_a_fixed_group_fixes_every_variant_morph():
    from eden.envs.base import EnvBase

    config = EdenConfig(
        env_options=EnvOptions(num_envs=2), scene_options=SceneOptions(obj=_two_box_group(is_fixed_base=True))
    )
    morphs = EnvBase.from_config(config).entities["obj"].morph  # constructed, not built
    assert len(morphs) == 2 and all(morph.fixed is True for morph in morphs)


class _HeavyBox(BoxOptions):
    material: RigidMaterialOptions = RigidMaterialOptions(rho=120.0)


def test_a_group_edited_after_construction_is_checked_before_the_build():
    """A TaskMod's assignment never re-runs the options' ``model_post_init``, so the entity checks again."""
    from eden.envs.base import EnvBase

    bank = GroupedEntityOptions(grouped_entities=[_HeavyBox(), BoxOptions()])
    bank.material = RigidMaterialOptions(rho=500.0)  # as a mod would, after the group validated
    config = EdenConfig(env_options=EnvOptions(num_envs=2), scene_options=SceneOptions(obj=bank))

    with pytest.raises(ValueError, match=r"sets material.rho=500.0, but grouped_entities\[0\] \(_HeavyBox\)"):
        EnvBase.from_config(config)  # constructed, not built


@pytest.mark.parametrize("runtime_cls, options_cls", [(RigidEntity, BaseEntityOptions), (Terrain, TerrainOptions)])
def test_class_defaults_match_the_options_defaults_that_shadow_them(runtime_cls, options_cls):
    """``get_parameter_names()`` needs the class attributes, but the options value always wins, so a drift is silent."""
    drifted = []
    for name in sorted(runtime_cls.get_parameter_names()):
        field = options_cls.model_fields.get(name)
        if field is None or field.is_required():
            continue
        options_default = field.get_default(call_default_factory=True)
        if getattr(runtime_cls, name, "<missing>") != options_default:
            drifted.append(f"{name}: class={getattr(runtime_cls, name, '<missing>')!r} options={options_default!r}")
    assert not drifted, f"{runtime_cls.__name__} defaults drifted from {options_cls.__name__}: {drifted}"


# ── is_fixed_base vs has_immobile_base ────────────────────────────────────────────────────────────
#
# A welded child reports ``is_fixed_base=False`` while owning no base DOFs, so "does this entity own
# base DOFs" must be asked with ``has_immobile_base``. The sites below ask about the declared option
# itself, deliberately; any other entity-level read is a new, unreviewed one.

REVIEWED_IS_FIXED_BASE_READERS: dict[tuple[str, str], str] = {
    ("eden/entities/base.py", "Entity.has_immobile_base"): "defines the predicate",
    ("eden/entities/base.py", "Entity.fixed"): "public alias of the declared option",
    ("eden/entities/rigid.py", "RigidEntity.fixed"): "public alias of the declared option",
    ("eden/entities/rigid.py", "RigidEntity._check_mjcf_base_matches_is_fixed_base"): (
        "compares the declared option against the built MJCF base link; returns first for a welded child"
    ),
    ("eden/entities/rigid.py", "RigidEntity._set_initial_root_pose"): (
        "unreachable for a welded child: post_build guards the whole call on `not _is_attached`"
    ),
    ("eden/entities/rigid.py", "RigidEntity.place_on_to"): (
        "reached only after the `is_attached` guard raises; the rest is about `batch_fixed_verts`"
    ),
    ("eden/entities/rigid.py", "RigidEntity.offset"): "same `batch_fixed_verts` question as place_on_to",
    ("eden/entities/support_mixin.py", "SupportSurfaceMixin._prepare_support"): (
        "asks whether the entity is anchored in the WORLD so world -Z is 'down'; a welded child moves"
    ),
    ("eden/extensions/deployment/utils/state.py", "RobotStateEntity.__init__"): (
        "snapshots the raw flag; the derived `has_immobile_base` is what consumers read"
    ),
}


def _entity_level_is_fixed_base_reads() -> set[tuple[str, str]]:
    """Every ``<expr>.is_fixed_base`` load under ``eden/`` that is not a read off an Options object.

    ``eden/options/`` is skipped wholesale, as is any receiver named ``options`` / ``*_options``:
    there the flag is the declaration morph creation and asset loading want, and ``has_immobile_base``
    does not exist on an Options object anyway.
    """
    found: set[tuple[str, str]] = set()
    for path in sorted(_EDEN.rglob("*.py")):
        rel = path.relative_to(_EDEN.parent).as_posix()
        if rel.startswith("eden/options/"):
            continue
        tree = ast.parse(path.read_text())
        scopes: dict[int, str] = {}

        def walk(node, prefix):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    name = f"{prefix}.{child.name}" if prefix else child.name
                    for sub in ast.walk(child):
                        scopes[id(sub)] = name  # deeper scopes are visited later and win
                    walk(child, name)

        walk(tree, "")
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Attribute) and node.attr == "is_fixed_base" and isinstance(node.ctx, ast.Load)
            ):
                continue
            receiver = node.value
            name = receiver.attr if isinstance(receiver, ast.Attribute) else getattr(receiver, "id", None)
            if name is not None and (name == "options" or name.endswith("_options")):
                continue
            found.add((rel, scopes.get(id(node), "<module>")))
    return found


def test_no_unreviewed_is_fixed_base_reads():
    """A lint: the failure (a welded hand taking a floating-base branch) spans subsystems too costly to build."""
    found, reviewed = _entity_level_is_fixed_base_reads(), set(REVIEWED_IS_FIXED_BASE_READERS)
    unreviewed = sorted(found - reviewed)
    assert not unreviewed, (
        f"New entity-level `is_fixed_base` read(s): {[f'{f}::{s}' for f, s in unreviewed]}. If the question is "
        "'does this entity own base DOFs', use `has_immobile_base` (a welded child reports is_fixed_base=False "
        "but has no base); if the declared option really is the question, add the site to "
        "REVIEWED_IS_FIXED_BASE_READERS with the reason."
    )
    stale = sorted(reviewed - found)
    assert not stale, f"REVIEWED_IS_FIXED_BASE_READERS lists sites that no longer read it: {stale}"
