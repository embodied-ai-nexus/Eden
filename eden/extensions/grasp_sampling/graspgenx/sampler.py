"""Eden wrapper around GraspGenX: cross-embodiment 6-DoF grasp generation.

GraspGenX (`NVlabs/GraspGenX <https://github.com/NVlabs/GraspGenX>`_) conditions one diffusion
model, and its discriminator, on a *gripper description*: the gripper's swept volume, its open and
closed point clouds, and its URDF. A gripper is therefore picked by name, with no per-gripper model.
Eden consumes the inference-only fork `Kashu7100/GraspGenX <https://github.com/Kashu7100/GraspGenX>`_,
published as ``eden-graspgenx`` (import name ``graspgenx``) and installed by the ``graspgenx``
extra. The checkpoints and the gripper descriptions download from Hugging Face on first use. See
``docs/guides/grasp-sampling.md``.

**Frames.** GraspGenX poses a grasp at the ``world`` link of the description's URDF: its z axis is
the approach direction and its x axis the closing direction, whatever the gripper. That frame is no
link of the real gripper; the description welds the gripper's base link to it through a fixed
``world_joint`` (rotated 90° about z for the Franka hand and the Robotiq 2F-85, offset by several
centimeters for the dexterous hands). The sampler applies that joint, so each pose it returns is
the pose of the description's base link (``pose_link_name``) — or, with ``end_effector=``, of the
Eden robot's link it pairs that base link with (:attr:`GraspGenXSampler.END_EFFECTORS`).
"""

from __future__ import annotations

import contextlib
import io
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, get_args

import numpy as np
import torch
import yourdfpy

import eden as en
from eden.extensions.grasp_sampling.base import GRASP_SAMPLER_REGISTRY, EndEffectorPairing, GraspSamplerBase
from eden.options.extensions.grasp_sampling import GraspMoEOptions
from eden.utils.distribution import install_hint

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.extensions.grasp_sampling import GraspSamplerOptions

Planner = Literal["graspmoe", "diffusion"]

#: Every gripper description GraspGenX ships, so an editor or type checker can offer them for
#: ``gripper_name``: the Hugging Face dataset Kashu7100/graspgenx_gripper_descriptions at revision
#: ee8b7db5. The descriptions found on disk at construction stay the authority, so a custom one under
#: ``$GRASPGENX_GRIPPER_CFG_DIR`` is accepted too (a type checker will flag its name). Refresh this when
#: the dataset adds a description; ``tests/extensions/grasp_sampling/graspgenx/test_sampler.py`` checks
#: it against a cached copy.
GraspGenXGripper = Literal[
    # grouped by the "type" each description's config.json declares
    # parallel_2f: two fingers that slide
    "abb_yumi",
    "arx_x5",
    "fetch_robot",
    "franka_panda",
    "franka_umi",
    "galaxea_g1",
    "piper_hand",
    "robotiq_hande",
    # revolute_2f: two fingers on linkages
    "bd_spot",
    "dh_ag95",
    "ezgripper",
    "onrobot_RG2",
    "onrobot_RG6",
    "robotiq_2f_140",
    "robotiq_2f_85",
    "sawyer_hand",
    "schunk_wsg50",
    "tesollo_delto2f",
    "xarm_hand",
    # revolute_3f: three or more fingers, dexterous hands included
    "barrett_hand",
    "inspire_hand",
    "robotiq_3f",
    "sharpa_wave",
    "surge_hand",
    "unitree_g1",
    "wuji_hand",
]

#: The Eden robots :attr:`GraspGenXSampler.END_EFFECTORS` pairs a description with (``end_effector=``).
GraspGenXEndEffector = Literal[
    "FrankaEmikaPanda",
    "FrankaHand",
    "InspireRH56_R",
    "Robotiq2f85",
    "SharpaWave_R",
    "WujiHand1_R",
]

#: A half turn about (0, 1, -1)/sqrt(2), as a wxyz quaternion.
_HALF_TURN_Y_MINUS_Z = (0.0, 0.0, math.sqrt(0.5), -math.sqrt(0.5))

_INSTALL_HINT = (
    "GraspGenX (`graspgenx`, shipped by `eden-graspgenx`) is not installed. Install it with "
    f"{install_hint('graspgenx')}. Its checkpoints and gripper descriptions download from Hugging Face on first use. "
    "See docs/guides/grasp-sampling.md."
)


def _require_graspgenx():
    """Import GraspGenX's entry points, turning a missing install into an actionable error."""
    try:
        import graspgenx
        from graspgenx.grasp_server import GraspGenXSampler as Model
        from graspgenx.samplers import run_planner_on_batch, run_planner_on_object
        from graspgenx.utils.checkpoint_io import load_model_cfg
    except ImportError as exc:
        raise ImportError(f"{_INSTALL_HINT}\nImport failed with: {exc}") from exc
    return graspgenx, Model, run_planner_on_object, run_planner_on_batch, load_model_cfg


@contextlib.contextmanager
def _stdout_to_debug_log():
    """Send what GraspGenX ``print``s while loading (checkpoint paths, missing-asset notes) to the debug log.

    If loading fails, the lines go to the warning log instead: they are what explains the failure.
    """
    buffer = io.StringIO()
    log = en.logger.debug
    try:
        with contextlib.redirect_stdout(buffer):
            yield
    except BaseException:
        log = en.logger.warning
        raise
    finally:
        for line in buffer.getvalue().splitlines():
            log(f"[graspgenx] {line}")


def _root_joint(urdf_path: Path) -> tuple[str, np.ndarray]:
    """The base link welded to the URDF's root and the root-to-base transform, from ``urdf_path``."""
    urdf = yourdfpy.URDF.load(
        str(urdf_path),
        build_scene_graph=False,
        load_meshes=False,
        build_collision_scene_graph=False,
        load_collision_meshes=False,
    )
    children = {joint.child for joint in urdf.robot.joints}
    roots = [link.name for link in urdf.robot.links if link.name not in children]
    root_joints = [joint for joint in urdf.robot.joints if roots and joint.parent == roots[0]]
    if len(roots) != 1 or len(root_joints) != 1 or root_joints[0].type != "fixed":
        raise ValueError(
            f"Gripper description '{urdf_path}' does not weld a single base link to a single root link "
            f"(root links {roots}, joints out of the root {[(j.name, j.type) for j in root_joints]}), so the "
            f"pose of its base link cannot be derived from a GraspGenX grasp. GraspGenX's descriptions all have "
            f"one fixed 'world_joint' from 'world'; check $GRASPGENX_GRIPPER_CFG_DIR if it points at a custom one."
        )
    joint = root_joints[0]
    origin = np.eye(4) if joint.origin is None else np.asarray(joint.origin, dtype=np.float64)
    return joint.child, origin


@GRASP_SAMPLER_REGISTRY.register()
class GraspGenXSampler(GraspSamplerBase):
    """Sample grasps with GraspGenX for any gripper it has a description of.

    The shared options (``camera_name``, ``object_cloud``, ``grasp_threshold``, ``max_grasps``, …)
    are documented on :class:`~eden.options.extensions.grasp_sampling.GraspSamplerOptions`; the
    fields below are GraspGenX's own and reach the options as extras. ``grasp_threshold`` defaults
    to 0.7 here, GraspGenX's own default: its confidences run lower than GraspGen's.

    Each returned pose places the description's base link, or with ``end_effector`` that Eden
    robot's link (:attr:`pose_link_name` either way); see the module docstring. The model runs on
    CUDA only, and is loaded at construction.

    ``filter_collisions`` is refused: no GraspGenX description ships a collision mesh
    (``coll_mesh.obj``), and GraspGenX substitutes a 1 cm box that every grasp clears. Check a
    grasp against the scene in simulation instead, by posing the gripper there and calling
    ``detect_collision`` (as ``examples/recipes/planning/grasp_sampling_graspgenx.py`` does).

    Parameters
    ----------
    gripper_name : GraspGenXGripper | None
        The GraspGenX gripper description to sample for, one of :data:`GraspGenXGripper` (e.g.
        ``"franka_panda"``, ``"robotiq_2f_85"``, ``"sharpa_wave"``). It names GraspGenX's description,
        not an Eden robot class. Required unless ``end_effector`` picks it; a name with no description
        on disk raises with the ones that are listed.
    checkpoint_dir : str
        Directory holding GraspGenX's ``gen/`` and ``dis/`` checkpoint directories. Empty uses its
        release checkpoints: ``$GRASPGENX_CHECKPOINT_DIR`` when set, else a Hugging Face download.
    end_effector : GraspGenXEndEffector | None
        The Eden robot that executes the grasps (see
        :class:`~eden.options.extensions.grasp_sampling.GraspSamplerOptions`). It picks
        ``gripper_name``, and each pose then places that robot's link: one of :attr:`END_EFFECTORS`.
    planner : Literal["graspmoe", "diffusion"]
        ``"graspmoe"`` scores the diffusion model's grasps together with grasps swept over the
        object's bounding box, which helps on boxy objects; ``"diffusion"`` returns the diffusion
        model's grasps alone.
    graspmoe : GraspMoEOptions
        The GraspMoE planner's knobs. Read only when ``planner="graspmoe"``.
    """

    gripper_name: GraspGenXGripper | None = None
    end_effector: GraspGenXEndEffector | None = None
    checkpoint_dir: str = ""
    planner: Planner = "graspmoe"
    graspmoe: GraspMoEOptions = GraspMoEOptions()
    grasp_threshold: float = 0.7

    #: Each pairing was checked against the Eden robot, by
    #: ``tests/extensions/grasp_sampling/test_end_effectors.py``: a parallel jaw's pads straddle the
    #: description's TCP along its closing axis; a dexterous hand's links match the description's at a
    #: grasp. The Franka's ``hand`` is the description's ``panda_hand`` and the 2F-85's root ``world`` its
    #: ``robotiq_arg2f_base_link``; the dexterous hands' roots are their descriptions' base links, the
    #: Inspire's after a half turn (its grasps land upside down without it).
    END_EFFECTORS: ClassVar[dict[str, EndEffectorPairing]] = {
        "FrankaEmikaPanda": EndEffectorPairing("franka_panda", "hand"),
        "FrankaHand": EndEffectorPairing("franka_panda", "hand"),
        "InspireRH56_R": EndEffectorPairing("inspire_hand", "base", ((0.0, 0.0, 0.0), _HALF_TURN_Y_MINUS_Z)),
        "Robotiq2f85": EndEffectorPairing("robotiq_2f_85", "world"),
        "SharpaWave_R": EndEffectorPairing("sharpa_wave", "right_hand_C_MC"),
        "WujiHand1_R": EndEffectorPairing("wuji_hand", "right_palm_link"),
    }
    _gripper_option: ClassVar[str] = "gripper_name"
    _no_collision_filter_reason: ClassVar[str] = (
        "no GraspGenX gripper description ships a collision mesh (coll_mesh.obj), and GraspGenX substitutes a "
        "1 cm box that every grasp clears, so the filter would pass every grasp."
    )

    def __init__(self, env: EnvBase, options: GraspSamplerOptions) -> None:
        super().__init__(env, options)
        if not self.gripper_name:
            raise ValueError(
                "GraspGenXSampler needs gripper_name=, the GraspGenX gripper description to sample for, e.g. "
                "gripper_name='franka_panda' or 'robotiq_2f_85' — or end_effector=, the Eden robot that executes the "
                f"grasps, which picks it: one of {sorted(self.END_EFFECTORS)}."
            )
        if self.planner not in get_args(Planner):
            raise ValueError(f"planner='{self.planner}' is not a GraspGenX planner; choose one of {get_args(Planner)}.")
        # A saved config loads this back as a plain dict; validating either form also rejects a typo'd knob.
        self.graspmoe = GraspMoEOptions.model_validate(self.graspmoe)
        graspgenx, Model, self._run_planner, self._run_planner_on_batch, load_model_cfg = _require_graspgenx()
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GraspGenX runs on CUDA only, and torch sees no CUDA device. Fix: run on a machine with an NVIDIA "
                "GPU (select one with CUDA_VISIBLE_DEVICES); the Eden env itself may stay on gs.cpu."
            )

        self._description_dir = self._resolve_description(graspgenx)
        base_link_name, grasp_T_base = _root_joint(self._description_dir / "gripper.urdf")
        with open(self._description_dir / "config.json") as f:
            config = json.load(f)
        missing = [key for key in ("open", "close", "fingertip") if key not in config]
        if missing:
            raise ValueError(
                f"Gripper description '{self._description_dir}' has no {missing} configuration in its config.json, "
                f"which every GraspGenX description declares; check $GRASPGENX_GRIPPER_CFG_DIR if it points at a "
                f"custom one."
            )
        # The TCP — the fingertip center GraspGenX grasps at — sits along the grasp frame's approach axis.
        self._set_pose_frame(base_link_name, grasp_T_base, np.asarray(config["fingertip"], dtype=np.float64))
        self._open_dofs_pos = {name: float(value) for name, value in config["open"].items()}
        self._close_dofs_pos = {name: float(value) for name, value in config["close"].items()}

        checkpoint_dir = self._resolve_checkpoint_dir(graspgenx)
        en.logger.info(f"GraspGenX: loading '{self.gripper_name}' with the checkpoints under {checkpoint_dir}")
        with _stdout_to_debug_log():
            model_cfg = load_model_cfg(str(checkpoint_dir / "gen"), str(checkpoint_dir / "dis"))
            self._model = Model(model_cfg, self.gripper_name)

    def _resolve_description(self, graspgenx) -> Path:
        assets = graspgenx.get_gripper_descriptions_assets()
        if assets is None:
            raise RuntimeError(
                "GraspGenX could not fetch its gripper descriptions from Hugging Face "
                "(dataset Kashu7100/graspgenx_gripper_descriptions). Fix: check the network, or set "
                "$GRASPGENX_GRIPPER_CFG_DIR to a local copy of that dataset."
            )
        assets = Path(assets)
        available = sorted(path.name for path in assets.iterdir() if (path / "config.json").is_file())
        if self.gripper_name not in available:
            raise ValueError(
                f"gripper_name='{self.gripper_name}' is not a GraspGenX gripper description; the descriptions "
                f"under {assets} are {available}. It names GraspGenX's description, not an Eden robot class "
                f"(e.g. 'franka_panda', not 'FrankaEmikaPanda')."
            )
        return assets / self.gripper_name

    def _resolve_checkpoint_dir(self, graspgenx) -> Path:
        if self.checkpoint_dir:
            checkpoint_dir = Path(self.checkpoint_dir).expanduser()
        else:
            checkpoint_dir = graspgenx.get_checkpoints_version_dir()
            if checkpoint_dir is None:
                raise RuntimeError(
                    "GraspGenX could not fetch its checkpoints from Hugging Face (model adithyamurali/GraspGenXModel). "
                    "Fix: check the network, or set checkpoint_dir= (or $GRASPGENX_CHECKPOINT_DIR) to a local copy."
                )
            checkpoint_dir = Path(checkpoint_dir)
        missing = [part for part in ("gen", "dis") if not (checkpoint_dir / part / "config.yaml").is_file()]
        if missing:
            raise FileNotFoundError(
                f"checkpoint_dir='{checkpoint_dir}' has no {' or '.join(f'{m}/config.yaml' for m in missing)}; it "
                f"must hold GraspGenX's 'gen' and 'dis' checkpoint directories (the 'release' directory of "
                f"adithyamurali/GraspGenXModel). Fix: point checkpoint_dir at that directory, or leave it empty."
            )
        return checkpoint_dir

    @property
    def open_dofs_pos(self) -> dict[str, float]:
        """The description's open configuration, by joint name: the one GraspGenX's grasps assume.

        Joint names follow the description's URDF. Eden's dexterous hands share them (``sharpa_wave``
        with ``SharpaWave_R``, ``inspire_hand`` with ``InspireRH56_R``); its parallel jaws carry their own
        open/close targets (``ManipulatorOptions.gripper_open``).
        """
        return dict(self._open_dofs_pos)

    @property
    def close_dofs_pos(self) -> dict[str, float]:
        """The description's closed configuration, by joint name (see :attr:`open_dofs_pos`)."""
        return dict(self._close_dofs_pos)

    def _infer(self, object_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        grasps, confidences, _, _ = self._run_planner(object_points, self._model, **self._planner_kwargs())
        return self._as_grasp_frames(grasps, confidences)

    def _infer_batch(self, clouds: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        if len(clouds) == 1:
            return [self._infer(clouds[0])]
        # One diffusion pass for every cloud; GraspMoE's box sweep still scores each object on its own.
        outputs = self._run_planner_on_batch(clouds, self._model, **self._planner_kwargs())
        return [self._as_grasp_frames(grasps, confidences) for grasps, confidences, _, _ in outputs]

    def _planner_kwargs(self) -> dict:
        moe = self.graspmoe
        return dict(
            planner=self.planner,
            grasp_threshold=self.grasp_threshold,
            num_grasps=self.num_grasps,
            topk_num_grasps=-1,  # the base class applies max_grasps, after any filtering
            moe_num_yaws=moe.num_yaws,
            moe_z_offsets_cm=moe.z_offsets_cm,
            moe_outlier_threshold=moe.outlier_threshold,
            moe_outlier_k=moe.outlier_k,
            moe_obb_mode=moe.obb_mode,
            moe_skip_obb_rule=moe.skip_obb_rule,
            moe_obb_density=moe.obb_density,
            moe_obb_position_spacing_cm=moe.obb_position_spacing_cm,
        )

    @staticmethod
    def _as_grasp_frames(grasps, confidences) -> tuple[np.ndarray, np.ndarray]:
        frames = np.asarray(grasps, dtype=np.float64).reshape(-1, 4, 4)
        return frames, np.asarray(confidences, dtype=np.float32).reshape(-1)
