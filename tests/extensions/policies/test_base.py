"""``PolicyBase``'s contract, and the sweeps that hold every in-tree policy to it."""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import sys
import textwrap

import pytest

import eden.extensions.policies as policies_pkg
from eden.extensions.policies.base import POLICY_REGISTRY, PolicyBase
from eden.options.entities import RobotOptions


def _import_policy_packages() -> list[str]:
    """Import every ``eden.extensions.policies.<pkg>.policy`` the way ``--policy`` does; return the registry keys.

    A package whose optional dependency is missing is tolerated, as ``--policy`` tolerates it.
    """
    for found in pkgutil.iter_modules(policies_pkg.__path__):
        if found.ispkg:
            try:
                importlib.import_module(f"eden.extensions.policies.{found.name}.policy")
            except Exception:  # noqa: BLE001, S110 — an optional dep missing is not this test's subject
                pass
    return sorted(POLICY_REGISTRY.keys())


POLICY_KEYS = _import_policy_packages()
#: Every ``PolicyBase`` subclass defined under ``eden/extensions/policies``, registered or not
#: (``OctoPolicy`` is not, and was the one whose ``reads_env_observations`` was wrong).
POLICY_CLASSES = sorted(
    {
        attr
        for name, module in list(sys.modules.items())
        if name.startswith("eden.extensions.policies.") and name.endswith(".policy")
        for attr in vars(module).values()
        if isinstance(attr, type) and issubclass(attr, PolicyBase) and attr.__module__ == name
    },
    key=lambda cls: f"{cls.__module__}.{cls.__name__}",
)


def test_the_sweeps_found_the_policies():
    """An import that silently came up empty would make every sweep below vacuous."""
    assert {"holosoma_locomotion_policy", "beyond_mimic_moves_policy"} <= set(POLICY_KEYS)
    assert {"OctoPolicy", "HolosomaLocomotionPolicy", "BeyondMimicMovesPolicy"} <= {c.__name__ for c in POLICY_CLASSES}
    assert {cls.reads_env_observations for cls in POLICY_CLASSES} == {True, False}


def test_checkpoints_cache_under_policies_in_the_current_asset_root(user_assets_dir, monkeypatch, tmp_path):
    assert PolicyBase.resolve_local_dir() == str(user_assets_dir / "policies")
    monkeypatch.setenv("EDEN_ASSETS_DIR", str(tmp_path / "shared"))
    assert PolicyBase.resolve_local_dir() == str(tmp_path / "shared" / "policies")


def test_a_policy_fetches_from_its_assets_subdir(user_assets_dir):
    """Pre-seeded files: a cached file is returned without a Hub request (a .txt has no dependencies)."""

    class OwnFolderPolicy(PolicyBase):
        assets_subdir = "policies/own"

    for policy, folder in ((PolicyBase, "policies"), (OwnFolderPolicy, "policies/own")):
        cached = user_assets_dir / folder / "head.txt"
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text("head")
        assert policy.fetch_asset("head.txt") == str(cached)


def test_construction_resolves_the_checkpoint_in_the_policy_assets_subdir(user_assets_dir):
    """Pre-seeded: PolicyBase.__init__ returns the cached checkpoint without a Hub request."""
    from eden.options.extensions.policies import PolicyOptions

    class CachedPolicy(PolicyBase):
        assets_subdir = "policies/cached"

    cached = user_assets_dir / "policies" / "cached" / "cached" / "model.txt"
    cached.parent.mkdir(parents=True)
    cached.write_text("weights")

    assert CachedPolicy(None, PolicyOptions(checkpoint_path="cached/model.txt")).checkpoint_path == str(cached)


@pytest.mark.parametrize("name", ["registry", "dataset", "local_dir"])
def test_where_files_come_from_is_not_configured_through_the_options(name):
    """A lookup made before the policy exists reads the class, so an instance-only value would split the two."""
    from eden.options.extensions.policies import PolicyOptions

    with pytest.raises(ValueError, match=rf"invalid: \['{name}'\].*fixed by its class.*EDEN_ASSETS_DIR"):
        PolicyBase(None, PolicyOptions(**{name: "/data/mirror"}))


@pytest.mark.parametrize("cls", POLICY_CLASSES, ids=lambda cls: cls.__name__)
def test_where_files_come_from_stays_out_of_every_policy_parameters(cls):
    """A saved config carrying them would pin one machine's cache folder, which the class could no longer move."""
    assert not {"registry", "dataset", "local_dir", "assets_subdir"} & set(cls.get_parameter_names())


def test_a_local_dir_set_on_the_class_reaches_the_class_lookups_and_the_instance(
    user_assets_dir, tmp_path, monkeypatch
):
    """The documented way to point a policy at a mirror: one setting that every fetch reads."""

    class MirroredPolicy(PolicyBase):
        pass

    mirror = tmp_path / "mirror"
    (mirror / "cached").mkdir(parents=True)
    (mirror / "cached" / "model.txt").write_text("weights")
    monkeypatch.setattr(MirroredPolicy, "local_dir", str(mirror))

    assert MirroredPolicy.fetch_asset("cached/model.txt") == str(mirror / "cached" / "model.txt")
    policy = MirroredPolicy(None, MirroredPolicy.configure(checkpoint_path="cached/model.txt"))
    assert policy.checkpoint_path == str(mirror / "cached" / "model.txt")
    assert PolicyBase.resolve_local_dir() == str(user_assets_dir / "policies")  # a subclass's setting stays its own


# ---------------------------------------------------------------------------
# use_local_checkpoint
# ---------------------------------------------------------------------------


@pytest.fixture
def sonic_cls():
    """A fresh subclass, so a redirect cannot leak into the registered ``SonicPolicy``."""
    from eden.extensions.policies.sonic.policy import SonicPolicy

    return type("LocalSonic", (SonicPolicy,), {})


def test_redirecting_a_checkpoint_clears_its_own_pin_and_nothing_else(sonic_cls):
    """``--ckpt-path`` failed the released hash at eight example sites that set the path without the clear."""
    parent = sonic_cls.__mro__[1]
    decoder, decoder_md5 = parent.decoder_checkpoint_path, parent.decoder_md5_hash

    sonic_cls.use_local_checkpoint("/tmp/mine.onnx")
    assert (sonic_cls.checkpoint_path, sonic_cls.md5_hash) == ("/tmp/mine.onnx", None)
    assert (sonic_cls.decoder_checkpoint_path, sonic_cls.decoder_md5_hash) == (decoder, decoder_md5)

    sonic_cls.use_local_checkpoint("/tmp/dec.onnx", checkpoint_attr="decoder_checkpoint_path")
    assert (sonic_cls.decoder_checkpoint_path, sonic_cls.decoder_md5_hash) == ("/tmp/dec.onnx", None)
    # Set on the subclass: the registered policy keeps its released weights and pins.
    assert parent.md5_hash is not None and parent.decoder_md5_hash == decoder_md5


def test_an_attribute_off_the_naming_convention_is_rejected():
    """HOMIE's ``walk_model_path`` would derive ``walk_model_pathmd5_hash`` and leave the real pin armed."""
    from eden.extensions.policies.homie.policy import HomiePolicy

    homie_cls = type("LocalHomie", (HomiePolicy,), {})
    with pytest.raises(ValueError, match="must end with 'checkpoint_path'"):
        homie_cls.use_local_checkpoint("/tmp/x.pt", checkpoint_attr="walk_model_path")


def test_a_checkpoint_without_a_matching_pin_is_rejected(sonic_cls):
    with pytest.raises(AttributeError, match="has no `unknown_checkpoint_path`"):
        sonic_cls.use_local_checkpoint("/tmp/x.pt", checkpoint_attr="unknown_checkpoint_path")


# ---------------------------------------------------------------------------
# robot_options
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "modpath, clsname, base",
    [
        ("eden.extensions.policies.holosoma.policy", "HolosomaLocomotionPolicy", "G1Dofs29"),
        ("eden.extensions.policies.homie.policy", "HomiePolicy", "G1Dofs29"),
        ("eden.extensions.policies.beyondmimic_moves.policy", "BeyondMimicMovesPolicy", "G1Dofs29"),
        ("eden.extensions.policies.amo.policy", "AMOPolicy", "G1Dofs23"),
        ("eden.extensions.policies.gmt.policy", "GMTPolicy", "G1Dofs23"),
        ("eden.extensions.policies.omnixtreme.policy", "OmniXtremePolicy", "G1Dofs29NoHands"),
        ("eden.extensions.policies.metamotivo.policy", "MetaMotivoPolicy", "SmplHumanoid"),
        ("eden.extensions.policies.sonic.policy", "SonicPolicy", "G1Dofs29"),
        ("eden.extensions.policies.protomotionv3.policy", "ProtoMotionV3Policy", "G1Dofs29"),
        ("eden.extensions.policies.husky.policy", "HuskyPolicy", "HuskyG1"),
    ],
)
def test_robot_options_pins_the_training_robot(modpath, clsname, base):
    """A task-less ``--policy`` run builds its scene from this; ``None`` would spawn the generic robot mis-posed."""
    robot = getattr(pytest.importorskip(modpath), clsname).robot_options()
    assert isinstance(robot, RobotOptions)
    assert base in {cls.__name__ for cls in type(robot).__mro__}


# ---------------------------------------------------------------------------
# reads_env_observations
# ---------------------------------------------------------------------------


def _discards_its_obs_argument(cls) -> bool:
    """Whether ``cls.step`` opens with ``del obs`` -- the in-tree idiom for reading raw state instead.

    Read off the AST because the policies cannot be constructed without their released weights,
    and a name cannot be both deleted and read in one body; an AST (unlike the source text)
    cannot be fooled by a comment mentioning ``del obs``.
    """
    fn = ast.parse(textwrap.dedent(inspect.getsource(cls.step))).body[0]
    args = [a.arg for a in fn.args.args]
    assert len(args) >= 2, f"{cls.__name__}.step takes no observation argument: {args}"
    return any(
        isinstance(node, ast.Delete) and any(isinstance(t, ast.Name) and t.id == args[1] for t in node.targets)
        for node in ast.walk(fn)
    )


@pytest.mark.parametrize("cls", POLICY_CLASSES, ids=lambda cls: cls.__name__)
def test_reads_env_observations_says_what_step_does(cls):
    """The flag picks whose observation layout the deploy env publishes; wrong, a policy reads another's."""
    discards = _discards_its_obs_argument(cls)
    assert cls.reads_env_observations is not discards, (
        f"{cls.__name__}.reads_env_observations is {cls.reads_env_observations} but step() "
        f"{'discards' if discards else 'reads'} its observation argument"
    )


# ---------------------------------------------------------------------------
# every registered policy reaches a deploy slot whose controllers can follow the entity swap
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def robots_env():
    """A G1, an Oli, MetaMotivo's SMPL humanoid and OmniReset's UR5e: every robot a registered policy pins its DOFs on."""
    from eden.options import EnvOptions, SceneOptions
    from eden.extensions.policies.metamotivo.policy import MetaMotivoPolicy
    from eden.extensions.policies.omnireset.policy import OmniResetPolicy
    from eden.options.robots import Oli
    from eden.options.robots.unitree_g1 import G1Dofs29
    from eden.utils.configs import EdenConfig
    from tests.support.envs import build_env

    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=1),
            scene_options=SceneOptions(
                g1=G1Dofs29(),
                oli=Oli(is_fixed_base=True, default_root_pos=(2.0, 0.0, 1.0)),
                smpl=MetaMotivoPolicy.robot_options().model_copy(update={"default_root_pos": (-2.0, 0.0, 1.0)}),
                ur5e=OmniResetPolicy.robot_options().model_copy(update={"default_root_pos": (0.0, 2.0, 0.0)}),
            ),
        )
    )


@pytest.mark.parametrize("key", POLICY_KEYS)
def test_a_policy_builds_repointable_controllers_or_is_refused(robots_env, key):
    """EnvelopeClip (OmniXtreme's) re-reads the entity every step one level below the term: it must be repointed.

    Controllers are built against the simulated robot (they resolve DOF names), then repointed at
    real robot state; anything the identity scan cannot reach would read the simulation on hardware.
    """
    from eden.extensions.deployment.base import cached_entity_refs, with_owned_modifiers
    from eden.extensions.deployment.utils.state import RobotStateEntity
    from eden.utils.cli.tasks import _repoint_slot_controllers, _slot_controllers

    cls = POLICY_REGISTRY.get(key)
    robot = cls.robot_options()
    if robot is None or not getattr(robot, "dofs_name", None):
        pytest.skip(f"{key} pins no single-entity robot (it rides on --robot, or is a composite)")
    entity_name = next(
        (
            name
            for name in ("g1", "oli", "smpl", "ur5e")
            if set(robot.dofs_name) <= set(robots_env.entities[name].dofs_name)
        ),
        None,
    )
    assert entity_name is not None, f"{type(robot).__name__}'s DOFs are on none of the scene's robots; add it"
    sim_entity = robots_env.entities[entity_name]

    try:
        controllers = _slot_controllers(robots_env, entity_name, cls)
    except SystemExit as exc:
        assert "joint-position field" in str(exc), f"{key} was refused without saying why"
        return
    assert controllers, f"{key} declares no action terms"

    live_entity = RobotStateEntity(sim_entity, device=robots_env.device)
    _repoint_slot_controllers([(key, None, controllers)], sim_entity, live_entity)
    scanned = with_owned_modifiers(controllers.values())
    for term in scanned:
        held = [attr for attr, value in vars(term).items() if value is sim_entity]
        assert not held, f"{type(term).__name__}.{held} still reads the simulated robot"
    # The scan accepts the built tree: a handle in a form it cannot repoint raises instead.
    cached_entity_refs(scanned, live_entity)


# ---------------------------------------------------------------------------
# require_entity
# ---------------------------------------------------------------------------


class _EntityReader(PolicyBase):
    """The smallest concrete policy: no checkpoint, no specs; only ``require_entity`` is exercised."""

    @classmethod
    def action_options(cls):
        return {}

    @classmethod
    def observation_options(cls):
        return {}

    def step(self, obs):
        return {}


def test_require_entity_names_the_missing_entity_what_the_scene_has_and_where_to_declare_it(robots_env):
    """A missing entity is named at the call, not by a policy that builds fine and dies on its first step."""
    policy = _EntityReader(env=robots_env, options=_EntityReader.configure())
    assert policy.require_entity("g1") is robots_env.entities["g1"]
    with pytest.raises(KeyError) as excinfo:
        policy.require_entity("object")
    message = str(excinfo.value)
    assert "'object'" in message and "scene_entity_options()" in message
    assert all(f"'{name}'" in message for name in robots_env.entities)
