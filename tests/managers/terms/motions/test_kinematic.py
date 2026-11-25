"""``KinematicLinksMotion``: link fields by forward kinematics over a root-and-joints dataset."""

import pytest
import torch

from eden.datasets import LAFAN1RetargetedG1Dataset, LAFAN1RootJointsG1Dataset
from eden.entities.rigid import StagedStateWrite
from eden.managers.terms.motions import KinematicLinksMotion
from eden.managers.terms.motions.kinematic import link_fields
from eden.options.entities import BoxOptions
from eden.options.managers.motions import MotionManagerOptions
from eden.options.robots.unitree_g1 import G1Dofs29
from eden.utils.geom import inv_transform_by_quat, normalize
from eden.utils.motion_reference.kinematics import ForwardKinematics
from tests.support.motion import FRAMERATE, ROBOT_DOFS, build_mimic_env, make_manager, write_lafan1_csv_library

pytestmark = [pytest.mark.requires_genesis]

LINKS = ["pelvis", "left_knee_link", "right_ankle_roll_link", "torso_link", "left_wrist_yaw_link"]


def _term(library, dataset=LAFAN1RootJointsG1Dataset, **term_kwargs):
    kwargs = {"entities_map": {"robot": "g1"}, "target_links_name": {"g1": LINKS}} | term_kwargs
    return KinematicLinksMotion.configure(
        dataset_options=dataset.configure(raw_data_path=str(library), target_framerate=FRAMERATE),
        target_dofs_name={"g1": ROBOT_DOFS},
        **kwargs,
    )


def _manager_options(term):
    return MotionManagerOptions(
        num_frames=2,
        data_start_from="current_time",
        frame_interval_s_range=(1 / FRAMERATE, 1 / FRAMERATE),
        default=term,
    )


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    return write_lafan1_csv_library(tmp_path_factory.mktemp("lafan1"), [90, 45])


@pytest.fixture(scope="module")
def env(library):
    """A G1 and a box, the G1 tracked by the term alone with the base term's start sampling."""
    scene = {"robot": G1Dofs29(), "box": BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(2.0, 0.0, 0.05))}
    motion = _manager_options(_term(library, motion_start_from_middle_range=(0.2, 0.8)))
    return build_mimic_env(library, scene=scene, motion_options=motion)


def _motion(env):
    return env.motion_manager.get_term("default")


def _write_robot_state(env, root_pos, root_quat, lin, ang, q, qd):
    """Put the robot at a root pose, world-frame root twist and joint state."""
    robot = env.entities["robot"]
    write = StagedStateWrite()
    base = robot.find_free_base_dofs_idx_local()
    write.set_dofs_vel(
        robot, torch.cat([lin, inv_transform_by_quat(ang, root_quat), qd], 1), [*base, *robot.dofs_idx_local.tolist()]
    )
    write.set_qpos(robot, torch.cat([root_pos, root_quat, q], 1), robot.qs_idx_local.tolist())
    write.apply(torch.arange(env.num_envs))


def test_the_term_keeps_the_base_terms_start_sampling(env):
    motion = _motion(env)
    assert type(motion) is KinematicLinksMotion
    assert (motion._motion_buffer_start_time_s > 0).all()


def test_the_links_are_the_configured_ones_and_the_dataset_stores_none(env):
    motion = _motion(env)
    sequence = motion.data_source.motion_sequences["g1"]
    assert sequence.links_pos_w is None and sequence.links_quat_b is None
    assert motion.sequence_key == "g1"
    assert motion.target_links_name["g1"] == LINKS
    assert env.motion_manager.data["robot"].links_pos_w.shape[-2] == len(LINKS)
    assert motion.provides_field("g1", "links_quat_b")


def test_the_reference_links_are_those_of_the_robot_placed_on_the_reference(env):
    robot, ref = env.entities["robot"], env.motion_manager.data["robot"]
    env.motion_manager.fill_data()
    _write_robot_state(
        env,
        ref.base_pos_w[:, 0],
        ref.base_quat_w[:, 0],
        ref.base_lin_vel_w[:, 0],
        ref.base_ang_vel_w[:, 0],
        ref.dofs_pos[:, 0],
        ref.dofs_vel[:, 0],
    )
    try:
        _, links = robot.find_named_links_idx_local(LINKS)
        torch.testing.assert_close(ref.links_pos_w[:, 0], robot.get_links_pos(ls_idx_local=links), atol=1e-4, rtol=0)
        torch.testing.assert_close(
            ref.links_lin_vel_w[:, 0], robot.get_links_vel(ls_idx_local=links), atol=1e-3, rtol=0
        )
    finally:
        env.reset()


def test_the_terms_kinematics_match_genesis_at_random_states(env):
    robot, fk = env.entities["robot"], _motion(env).fk
    n, j = env.num_envs, len(robot.dofs_name)
    gen = torch.Generator().manual_seed(0)
    root_pos = torch.rand(n, 3, generator=gen) + torch.tensor([0.0, 0.0, 1.0])
    root_quat = normalize(torch.randn(n, 4, generator=gen))
    q = 0.5 * torch.randn(n, j, generator=gen)
    lin, ang, qd = (torch.randn(n, k, generator=gen) for k in (3, 3, j))
    _write_robot_state(env, root_pos, root_quat, lin, ang, q, qd)
    try:
        _, links = robot.find_named_links_idx_local(LINKS)
        pos, quat, lin_vel, ang_vel = fk(root_pos, root_quat, q, lin, ang, qd)
        torch.testing.assert_close(pos, robot.get_links_pos(ls_idx_local=links), atol=1e-5, rtol=0)
        # q and -q are one rotation.
        torch.testing.assert_close(
            (quat * robot.get_links_quat(ls_idx_local=links)).sum(-1).abs(),
            torch.ones(n, len(links)),
            atol=1e-5,
            rtol=0,
        )
        torch.testing.assert_close(lin_vel, robot.get_links_vel(ls_idx_local=links), atol=1e-4, rtol=0)
        torch.testing.assert_close(ang_vel, robot.get_links_ang(ls_idx_local=links), atol=1e-4, rtol=0)
    finally:
        env.reset()


def test_a_cpu_device_runs_the_kinematics_eagerly(env):
    motion = _motion(env)
    assert motion.compile_kinematics and motion._link_fields is link_fields


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the term compiles its kinematics on a CUDA device only")
def test_the_compiled_kinematics_match_the_eager_ones_to_rounding():
    fk = ForwardKinematics(G1Dofs29().fetch_asset(), ROBOT_DOFS, LINKS).cuda()
    compiled = torch.compile(link_fields)
    # A full batch, then a subset of it, which the compiled pass takes with a dynamic row count.
    for rows in (64, 6):
        gen = torch.Generator().manual_seed(rows)
        root_pos, root_quat = torch.rand(rows, 3, generator=gen), normalize(torch.randn(rows, 4, generator=gen))
        lin, ang = torch.randn(rows, 3, generator=gen), torch.randn(rows, 3, generator=gen)
        q, qd = (
            0.5 * torch.randn(rows, len(ROBOT_DOFS), generator=gen),
            torch.randn(rows, len(ROBOT_DOFS), generator=gen),
        )
        state = [x.cuda() for x in (root_pos, root_quat, lin, ang, q, qd)]
        for got, want in zip(compiled(fk, *state), link_fields(fk, *state), strict=True):
            torch.testing.assert_close(got, want, atol=2e-5, rtol=0)


def test_a_dataset_that_stores_links_is_named(env, library):
    with pytest.raises(ValueError, match=r"dataset LAFAN1RetargetedG1Dataset stores \['links_pos_w'"):
        make_manager(env, terms={"default": _term(library, dataset=LAFAN1RetargetedG1Dataset)})


@pytest.mark.parametrize("links", [[], ".*_link"])
def test_the_links_must_be_listed_by_name(env, library, links):
    with pytest.raises(ValueError, match=r"target_links_name\['g1'\] must list the links to fill by name, got"):
        make_manager(env, terms={"default": _term(library, target_links_name={"g1": links})})


def test_a_link_the_robot_does_not_have_is_named_with_the_robots_links(env, library):
    with pytest.raises(ValueError, match=r"has no link\(s\) \['tail'\].*Links: .*'pelvis'"):
        make_manager(env, terms={"default": _term(library, target_links_name={"g1": ["pelvis", "tail"]})})


def test_an_entity_without_an_asset_file_is_refused(env, library):
    with pytest.raises(ValueError, match=r"entity 'box' \(BoxOptions\) has none"):
        make_manager(env, terms={"default": _term(library, entities_map={"box": "g1"})})


def test_an_entity_the_scene_does_not_have_is_named(env, library):
    with pytest.raises(ValueError, match=r"entities_map names entity 'ghost', which the scene options do not have"):
        make_manager(env, terms={"default": _term(library, entities_map={"ghost": "g1"})})


def test_the_root_joints_cache_and_the_full_datasets_cache_refuse_each_other(library, tmp_path):
    import torch as _torch

    def load(dataset, path):
        options = dataset.configure(
            raw_data_path=str(library), processed_data_path=str(path), target_framerate=FRAMERATE
        )
        return dataset(options, device=_torch.device("cpu"))

    root_joints, full = tmp_path / "root_joints.pt", tmp_path / "full.pt"
    load(LAFAN1RootJointsG1Dataset, root_joints)
    load(LAFAN1RetargetedG1Dataset, full)
    with pytest.raises(
        ValueError,
        match=r"earlier LAFAN1RetargetedG1Dataset or by another .*\(cache_semantics 'root and joint state only",
    ):
        load(LAFAN1RetargetedG1Dataset, root_joints)
    with pytest.raises(ValueError, match=r"earlier LAFAN1RootJointsG1Dataset or by another .*\(cache_semantics None\)"):
        load(LAFAN1RootJointsG1Dataset, full)
    assert load(LAFAN1RootJointsG1Dataset, root_joints).motion_sequences["g1"].links_pos_w is None
