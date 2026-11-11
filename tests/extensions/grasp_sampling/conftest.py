"""The scene the grasp-sampling tests sample in, and the backends' model packages as fakes.

The scene is a box on a table, a Franka hand beside it, one camera. Inference needs the backends'
packages, their downloaded models and a CUDA device, so ``graspgenx`` and ``grasp_gen`` are fake
packages holding the entry points the samplers call; they record what they were handed. Gripper
descriptions, configs and checkpoint directories are real files under ``tmp_path``.
"""

from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest
import torch

from eden.options import CameraOptions, CamerasOptions, EnvOptions, SceneOptions
from eden.options.entities import BoxOptions
from eden.options.robots import FrankaHand
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env

TABLE_TOP = 0.2
#: The target: a 6 cm cube resting on the table.
BOX_CENTER = (0.5, 0.0, TABLE_TOP + 0.03)
BOX_HALF = 0.03


def grasp_scene_config(num_envs: int = 1, env_separate_rigid: bool = False) -> EdenConfig:
    return EdenConfig(
        env_options=EnvOptions(num_envs=num_envs, env_separate_rigid=env_separate_rigid),
        scene_options=SceneOptions(
            table=BoxOptions(
                size=(0.6, 0.6, TABLE_TOP), default_root_pos=(0.5, 0.0, TABLE_TOP / 2), is_fixed_base=True
            ),
            box=BoxOptions(size=(2 * BOX_HALF,) * 3, default_root_pos=BOX_CENTER, is_fixed_base=True),
            # Out of the camera's view: there is nothing of it to sample from.
            hidden=BoxOptions(
                size=(0.05,) * 3, default_root_pos=(-2.0, 0.0, 0.025), is_fixed_base=True, batch_fixed_verts=True
            ),
            # An actuated entity in view, which the default collision_ignore_entities leaves out.
            hand=FrankaHand(default_root_pos=(0.5, 0.2, TABLE_TOP + 0.15)),
        ),
        cameras_options=CamerasOptions(
            camera=CameraOptions(
                cam_pos=(1.0, 0.0, 0.8), cam_lookat=(0.5, 0.0, TABLE_TOP), cam_resolution=(160, 120), cam_fov=60.0
            )
        ),
    )


@pytest.fixture(scope="module")
def grasp_env():
    return build_env(grasp_scene_config())


# --- GraspGenX -------------------------------------------------------------------------------------

#: Not a GraspGenXGripper: every test also shows a custom description on disk is accepted.
GRIPPER = "toy_gripper"
OPEN = {"finger": 0.04}
CLOSE = {"finger": 0.0}

URDF = """<?xml version="1.0"?>
<robot name="{name}">
  <link name="world"/>
  <link name="toy_base"/>
  <joint name="world_joint" type="fixed">
    <parent link="world"/>
    <child link="toy_base"/>
    <origin xyz="{xyz}" rpy="{rpy}"/>
  </joint>
</robot>
"""

#: The revision of GraspGenX's gripper descriptions (Hugging Face dataset) GraspGenXGripper was taken from.
DESCRIPTIONS_REVISION = "ee8b7db55d3a0671901a8bffd37fdcfdc9f74046"


def write_description(root, *, name=GRIPPER, xyz="0 0 0.01", rpy="0 0 1.5708"):
    gripper = root / "x_grippers" / name
    gripper.mkdir(parents=True, exist_ok=True)
    (gripper / "gripper.urdf").write_text(URDF.format(name=name, xyz=xyz, rpy=rpy))
    (gripper / "config.json").write_text(json.dumps({"open": OPEN, "close": CLOSE, "fingertip": [0.0, 0.0, 0.1]}))
    return root / "x_grippers"


def write_checkpoints(root, parts=("gen", "dis")):
    for part in parts:
        (root / part).mkdir(parents=True)
        (root / part / "config.yaml").write_text("{}")
    return root


def cached_descriptions():
    """GraspGenX's real gripper descriptions from the local Hugging Face cache; skips the test without them."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        root = snapshot_download(
            "Kashu7100/graspgenx_gripper_descriptions",
            repo_type="dataset",
            revision=DESCRIPTIONS_REVISION,
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        pytest.skip("GraspGenX's gripper descriptions are not in the local Hugging Face cache")
    return f"{root}/gripper_descriptions/assets/x_grippers"


class FakeGraspGenX:
    """The ``graspgenx`` package: description and checkpoint resolvers, the model and the planners."""

    def __init__(self, descriptions, checkpoints):
        self.descriptions, self.checkpoints = descriptions, checkpoints
        self.loaded = []  # (gen_dir, dis_dir, gripper_name) per model load
        self.planner_calls = []
        self.batch_calls = []
        self.grasp_frames = np.eye(4)[None]

    def install(self, monkeypatch):
        package = types.ModuleType("graspgenx")
        package.get_gripper_descriptions_assets = lambda: self.descriptions
        package.get_checkpoints_version_dir = lambda: self.checkpoints

        fake = self

        class Model:
            def __init__(self, cfg, gripper_name):
                fake.loaded.append((*cfg, gripper_name))

        def output():
            return fake.grasp_frames, np.full(len(fake.grasp_frames), 0.9, dtype=np.float32), [], None

        def run_planner_on_object(points, model, **kwargs):
            fake.planner_calls.append({"points": points, **kwargs})
            return output()

        def run_planner_on_batch(clouds, model, **kwargs):
            fake.batch_calls.append({"clouds": clouds, **kwargs})
            return [output() for _ in clouds]

        modules = {
            "graspgenx": package,
            "graspgenx.grasp_server": types.SimpleNamespace(GraspGenXSampler=Model),
            "graspgenx.samplers": types.SimpleNamespace(
                run_planner_on_object=run_planner_on_object, run_planner_on_batch=run_planner_on_batch
            ),
            "graspgenx.utils": types.ModuleType("graspgenx.utils"),
            "graspgenx.utils.checkpoint_io": types.SimpleNamespace(load_model_cfg=lambda gen, dis: (gen, dis)),
        }
        for name, module in modules.items():
            monkeypatch.setitem(sys.modules, name, module)
        return self


@pytest.fixture
def graspgenx(tmp_path, monkeypatch):
    # GraspGenX runs on CUDA only, and the suite's env is on gs.cpu with no device required.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    return FakeGraspGenX(write_description(tmp_path / "descriptions"), write_checkpoints(tmp_path / "release")).install(
        monkeypatch
    )


# --- GraspGen --------------------------------------------------------------------------------------

GRASPGEN_CONFIG = "graspgen_toy.yml"
#: GraspGen's franka_panda model config, which end_effector="FrankaHand" picks.
GRASPGEN_FRANKA_CONFIG = "graspgen_franka_panda.yml"
#: GraspGen's tool TCP for franka_panda: its config's ``depth``, along the grasp frame's approach axis.
GRASPGEN_FRANKA_DEPTH = 0.10527314


class FakeGraspGen:
    """The ``grasp_gen`` package: config loading, the model, the gripper's mesh and TCP, the collision filter."""

    def __init__(self):
        self.filter_calls = []
        self.keep = None
        # GraspGen leaves each grasp's homogeneous corner unset.
        grasps = torch.eye(4).repeat(3, 1, 1)
        grasps[:, :3, 3] = torch.tensor([[0.1, 0.0, 0.3], [0.2, 0.0, 0.3], [0.3, 0.0, 0.3]])
        grasps[:, 3, 3] = 0.0
        self.output = (grasps, torch.tensor([0.81, 0.95, 0.88]))

    def install(self, monkeypatch):
        fake = self

        class Model:
            def __init__(self, cfg):
                self.cfg = cfg

            @staticmethod
            def run_inference(points, model, **kwargs):
                return fake.output

        def filter_colliding_grasps(**kwargs):
            fake.filter_calls.append(kwargs)
            return fake.keep

        def get_gripper_info(name):
            tcp = np.eye(4)
            tcp[2, 3] = GRASPGEN_FRANKA_DEPTH
            return types.SimpleNamespace(collision_mesh=f"{name} mesh", transform_from_base_link_to_tool_tcp=tcp)

        def load_grasp_cfg(path):
            name = "franka_panda" if path.endswith(GRASPGEN_FRANKA_CONFIG) else "toy"
            return types.SimpleNamespace(data=types.SimpleNamespace(gripper_name=name))

        modules = {
            "grasp_gen": types.ModuleType("grasp_gen"),
            "grasp_gen.grasp_server": types.SimpleNamespace(GraspGenSampler=Model, load_grasp_cfg=load_grasp_cfg),
            "grasp_gen.robot": types.SimpleNamespace(get_gripper_info=get_gripper_info),
            "grasp_gen.utils": types.ModuleType("grasp_gen.utils"),
            "grasp_gen.utils.point_cloud_utils": types.SimpleNamespace(filter_colliding_grasps=filter_colliding_grasps),
        }
        for name, module in modules.items():
            monkeypatch.setitem(sys.modules, name, module)
        return self


@pytest.fixture
def grasp_gen(tmp_path, monkeypatch):
    for config in (GRASPGEN_CONFIG, GRASPGEN_FRANKA_CONFIG):
        (tmp_path / config).write_text("{}")
    monkeypatch.setenv("GRASPGEN_CHECKPOINT_DIR", str(tmp_path))
    return FakeGraspGen().install(monkeypatch)
