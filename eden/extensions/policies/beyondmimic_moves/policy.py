"""BeyondMimic-Moves motion-tracking policy."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, ClassVar

import json
import os

import numpy as np
import torch
import torch.nn as nn

from eden.extensions.policies.base import PolicyBase, PolicyOptions, POLICY_REGISTRY
from eden.options.managers.observations import ObservationGroupOptions
from eden.options.robots.unitree_g1 import G1Dofs29, G1_ACTION_SCALE, G1_DOFS29_TRACKED_LINKS_NAME
from eden.terms import ObsTerm
from eden import actions, observations
from eden.utils.string import resolve_matching_names_values

from eden.managers.terms.commands.mimic import MotionCommand
from eden.extensions.policies.beyondmimic_moves.custom_terms.observations import (
    g1_moves_imu_lin_vel_b,
    g1_moves_motion_anchor_pos_b,
    g1_moves_motion_anchor_ori_b,
)
from eden.constants import ReferenceSource

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.actions import ActionTermOptions
    from eden.options.managers.commands import CommandTermOptions


_BODY_NAMES: list[str] = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]
_ANCHOR_BODY: str = "torso_link"
_OBS_DIM: int = 160
_ACT_DIM: int = 29
_HIDDEN_DIMS: tuple[int, ...] = (512, 256, 128)

# Permutation from mjlab's MuJoCo XML body order (used by every g1-moves NPZ) to Genesis's
# BFS link order. `_remap_motion_npz_to_genesis_order` reorders each clip's columns by this
# perm AND stamps `body_names` (== G1_DOFS29_TRACKED_LINKS_NAME), so MotionCommand resolves
# columns BY NAME — robust to genesis 1.2.0's BFS->DFS reorder, which would mis-map a file
# sliced by runtime idx_local. The perm is load-bearing: it physically puts the columns in
# the order the stamped names describe. Dropping it silently mis-maps bodies (locked by
# tests/extensions/policies/beyondmimic_moves/test_policy.py).
#
# The mapping is fixed for the 30-link 29-DOF G1 (`g1_29dof_rev_1_0.xml`).
# `genesis_npz[t, i] = mjlab_npz[t, _MJLAB_TO_GENESIS_BODY_PERM[i]]`.
_MJLAB_TO_GENESIS_BODY_PERM: tuple[int, ...] = (
    0,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    6,
    12,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
    22,
    29,
)
_BODY_AXIS_KEYS: tuple[str, ...] = (
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def _remap_motion_npz_to_genesis_order(raw_npz_path: str) -> str:
    """Cache-or-build a Genesis-BFS-ordered, name-stamped copy of an mjlab tracking NPZ.

    The output filename is ``<basename>_eden.npz`` next to the original. A cached copy is
    reused only if it already carries ``body_names`` (older caches predate the name stamp
    and would mis-slice under genesis 1.2.0's depth-first link ordering, so they are rebuilt).
    """
    base, ext = os.path.splitext(raw_npz_path)
    out_path = f"{base}_eden{ext}"
    if os.path.exists(out_path):
        with np.load(out_path) as cached:
            if "body_names" in cached.files:
                return out_path

    data = np.load(raw_npz_path)
    perm = np.asarray(_MJLAB_TO_GENESIS_BODY_PERM)
    # body_names == the canonical G1 link list (perm/name alignment guarded by
    # tests/extensions/policies/beyondmimic_moves/test_policy.py).
    assert len(G1_DOFS29_TRACKED_LINKS_NAME) == perm.size, "G1 link list must match the permutation size"
    new_data: dict[str, np.ndarray] = {}
    for k in data.files:
        arr = data[k]
        if k in _BODY_AXIS_KEYS and arr.ndim >= 2 and arr.shape[1] == perm.size:
            new_data[k] = arr[:, perm]
        else:
            new_data[k] = arr
    new_data["body_names"] = np.array(G1_DOFS29_TRACKED_LINKS_NAME)
    np.savez(out_path, **new_data)
    return out_path


# Default joint positions used by mjlab during g1-moves training (the env.yaml
# `entities.robot.init_state.joint_pos` block). Both the action offset
# (`reference_source=ReferenceSource.DEFAULT`) AND the observation `joint_pos_rel` term subtract
# this same vector — the policy operates relative to it. The robot entity
# passed to the policy MUST have `default_dofs_pos` matching these values, or
# the policy will be controlling in a shifted frame and won't track the motion.
# Build the robot with `g1_moves_robot()` (= `BeyondMimicMovesPolicy.robot_options()`).
G1_MOVES_DEFAULT_DOFS_POS: dict[str, float] = {
    "*_hip_pitch_joint": -0.312,
    "*_hip_roll_joint": 0.0,
    "*_hip_yaw_joint": 0.0,
    "*_knee_joint": 0.669,
    "*_ankle_pitch_joint": -0.363,
    "*_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "*_shoulder_yaw_joint": 0.0,
    "*_elbow_joint": 0.6,
    "*_wrist_roll_joint": 0.0,
    "*_wrist_pitch_joint": 0.0,
    "*_wrist_yaw_joint": 0.0,
}


def g1_moves_robot(**overrides: Any) -> G1Dofs29:
    """Build a ``G1Dofs29`` with ``default_dofs_pos`` pinned to :data:`G1_MOVES_DEFAULT_DOFS_POS`.

    The policy operates relative to this frame (see that constant's note), so the
    robot must spawn in it.
    """
    return G1Dofs29(default_dofs_pos=G1_MOVES_DEFAULT_DOFS_POS, **overrides)


class _NormalizedActor(nn.Module):
    """Frozen MLP actor with running-mean/std obs normalization baked in."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, mlp: nn.Sequential) -> None:
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.mlp = mlp

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp((x - self.mean) / self.std)


@POLICY_REGISTRY.register()
class BeyondMimicMovesPolicy(PolicyBase):
    """BeyondMimic G1-Moves: 60 motion-tracking policies for the 29-DOF Unitree G1.

    Source dataset: https://huggingface.co/datasets/exptech/g1-moves

    The clip is selected at configure time via the ``clip_name`` option (e.g. ``"B_DadDance"``).
    The policy resolves the per-clip checkpoint and motion file from the dataset's
    ``manifest.json`` and downloads them on first use via ``huggingface_hub``.

    Notes
    -----
    The expected env / observation / action setup mirrors ``examples/projects/mimic/beyondmimic`` (and the
    ``BeyondMimicConfig`` task), since the upstream g1-moves env.yaml is functionally identical:

    * ``decimation=4``, ``sim_dt=0.005``, 29-DOF G1
    * Observation (160 dims): ``[motion(58), motion_anchor_pos_b(3), motion_anchor_ori_b(6),
      base_lin_vel(3), base_ang_vel(3), dofs_pos_rel(29), dofs_vel(29), last_action(29)]``
    * Joint order is the MuJoCo XML declaration order (no permutation needed).

    The robot entity ``default_dofs_pos`` MUST equal :data:`G1_MOVES_DEFAULT_DOFS_POS`,
    otherwise both ``dofs_pos_rel`` (obs) and the action target end up biased relative
    to the policy's training frame and the robot won't track. ``eden.robots.G1Dofs29``'s
    out-of-the-box defaults differ (e.g. ``hip_pitch=-0.1`` vs. mjlab's ``-0.312``,
    ``elbow=1.5`` vs. ``0.6``), so build the robot with :func:`g1_moves_robot` (what
    :meth:`robot_options` returns).
    """

    # step() reads the observation dict the env publishes, so this policy owns the deploy env.
    reads_env_observations = True

    registry: ClassVar[str] = "exptech"
    dataset: ClassVar[str] = "g1-moves"
    # Resolved per-clip in __init__.
    checkpoint_path: str | None = None
    clip_name: str = "B_DadDance"

    def __init__(self, env: EnvBase, options: PolicyOptions) -> None:
        # NOTE: PolicyBase.__init__ skips asset resolution while checkpoint_path is None.
        super().__init__(env, options)

        clip_dir = f"{self._clip_category(self.clip_name)}/{self.clip_name}"
        self.checkpoint_path = self.fetch_asset(f"{clip_dir}/policy/{self.clip_name}_policy.pt")
        self.motion_file = self.prepare_motion_file(self.clip_name)

        self._policy = self._build_actor(self.checkpoint_path, env.device)

        self._verify_robot_defaults()

    @classmethod
    def _clip_category(cls, clip_name: str) -> str:
        with open(cls.fetch_asset("manifest.json")) as f:
            clips: dict[str, Any] = json.load(f)["clips"]
        if clip_name not in clips:
            raise ValueError(f"Unknown g1-moves clip '{clip_name}'. Available clips: {sorted(clips.keys())}")
        return clips[clip_name]["category"]

    @classmethod
    def prepare_motion_file(cls, clip_name: str) -> str:
        """Resolve a clip's motion file, downloading and remapping body axes as needed.

        Returns the path to a Genesis-BFS-ordered NPZ next to the original mjlab
        download. Safe to call before constructing the env (so the resolved path
        can be passed to :meth:`command_options`); subsequent calls hit the cached
        ``<clip>_eden.npz`` without re-reading the source.
        """
        category = cls._clip_category(clip_name)
        raw_npz = cls.fetch_asset(f"{category}/{clip_name}/training/{clip_name}.npz")
        return _remap_motion_npz_to_genesis_order(raw_npz)

    def _verify_robot_defaults(self) -> None:
        """Fail loudly if the robot's default_dofs_pos drifts from the mjlab training frame."""
        robot = self.require_entity("robot")
        defaults = robot.get_default_dofs_pos()  # (num_envs, num_dofs): a per-env drift counts too
        # Matched as the robot's own config dict is (exact names literal, every key must match), into its columns.
        columns, _, values = resolve_matching_names_values(G1_MOVES_DEFAULT_DOFS_POS, robot.dofs_name)
        expected = torch.zeros(defaults.shape[1], device=defaults.device)
        expected[columns] = torch.tensor(values, dtype=expected.dtype, device=expected.device)
        drift = (defaults - expected).abs().amax(dim=1)
        if drift.max().item() <= 1e-6:
            return
        env_idx = int(drift.argmax().item())
        actual_cpu = defaults[env_idx].cpu().tolist()
        expected_cpu = expected.cpu().tolist()
        mismatches = [
            f"{name}: got {a:+.4f}, expected {e:+.4f}"
            for name, a, e in zip(robot.dofs_name, actual_cpu, expected_cpu)
            if abs(a - e) > 1e-6
        ]
        raise ValueError(
            "BeyondMimicMovesPolicy requires the robot's default_dofs_pos to match the "
            f"mjlab g1-moves training defaults (env.yaml init_state.joint_pos). Mismatches in env {env_idx}:\n  "
            + "\n  ".join(mismatches)
            + "\nFix: build the robot with BeyondMimicMovesPolicy.robot_options() (g1_moves_robot())."
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    @staticmethod
    def _build_actor(checkpoint_path: str, device: torch.device | str) -> _NormalizedActor:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"]

        layers: list[nn.Module] = []
        in_dim = _OBS_DIM
        for i, h in enumerate(_HIDDEN_DIMS):
            lin = nn.Linear(in_dim, h)
            lin.weight.data.copy_(sd[f"actor.{2 * i}.weight"])
            lin.bias.data.copy_(sd[f"actor.{2 * i}.bias"])
            layers += [lin, nn.ELU()]
            in_dim = h
        head = nn.Linear(in_dim, _ACT_DIM)
        head.weight.data.copy_(sd[f"actor.{2 * len(_HIDDEN_DIMS)}.weight"])
        head.bias.data.copy_(sd[f"actor.{2 * len(_HIDDEN_DIMS)}.bias"])
        layers.append(head)
        mlp = nn.Sequential(*layers)

        mean = sd["actor_obs_normalizer._mean"].view(1, -1).float()
        std = sd["actor_obs_normalizer._std"].view(1, -1).float().clamp_min(1e-8)

        return _NormalizedActor(mean, std, mlp).to(device).eval()

    # ------------------------------------------------------------------
    # Env / action / command / observation specs
    # ------------------------------------------------------------------

    @classmethod
    def robot_options(cls) -> G1Dofs29:
        """Return a ``G1Dofs29`` pinned to the g1-moves training default pose."""
        return g1_moves_robot()

    @classmethod
    def env_options(cls) -> dict[str, Any]:
        # Mirror mjlab's SimulationCfg + the eden BeyondMimicConfig settings used
        # to validate sim2sim transfer. Eden's EnvOptions defaults
        # (iterations=50, ls_iterations=50, enable_self_collision=True,
        # enable_multi_contact=True, use_gjk_collision=True, max_collision_pairs=50)
        # diverge enough from MuJoCo's solver behaviour that physics drift makes
        # the policy fall.
        return dict(
            sim_dt=0.005,
            sim_substeps=1,
            decimation=4,
            iterations=10,
            ls_iterations=20,
            max_collision_pairs=20,
            batch_dofs_info=False,
            use_gjk_collision=False,
            enable_self_collision=False,
            enable_multi_contact=False,
        )

    @classmethod
    def action_options(cls, entity_name: str = "robot") -> dict[str, ActionTermOptions]:
        return dict(
            dofs_pos_controller=actions.ExplicitPDController.configure(
                entity_name=entity_name,
                dofs_name=G1Dofs29().dofs_name,
                scale=G1_ACTION_SCALE,
                reference_source=ReferenceSource.DEFAULT,
            ),
        )

    @classmethod
    def command_options(
        cls,
        clip_name: str | None = None,
        debug_vis: bool = False,
        entity_name: str = "robot",
        ghost_entity_name: str = "",
        sampling_mode: str = "start",
    ) -> dict[str, CommandTermOptions]:
        # This is a per-clip tracker, not a general one: the motion file is derived from the
        # clip, never passed in. command_options runs during env.build() (before the policy
        # instance exists, so self.motion_file is not available yet), so resolve it here via the
        # classmethod resolver — the same path __init__ uses. Defaults to the configured clip.
        return dict(
            motion=MotionCommand.configure(
                resampling_time_range=(1.0e9, 1.0e9),
                debug_vis=debug_vis,
                entity_name=entity_name,
                ghost_entity_name=ghost_entity_name,
                motion_file=cls.prepare_motion_file(clip_name or cls.clip_name),
                anchor_link_name=_ANCHOR_BODY,
                links_name=_BODY_NAMES,
                pose_range={},
                velocity_range={},
                dofs_position_range=(0.0, 0.0),
                sampling_mode=sampling_mode,
            ),
        )

    @classmethod
    def observation_options(cls, entity_name: str = "robot") -> dict[str, ObservationGroupOptions]:
        return dict(
            policy=ObservationGroupOptions(
                concatenate_terms=True,
                enable_corruption=False,
                motion=ObsTerm.configure(
                    func=observations.generated_commands,
                    params={"command_name": "motion"},
                ),
                motion_anchor_pos_b=ObsTerm.configure(
                    func=g1_moves_motion_anchor_pos_b,
                    params={"command_name": "motion"},
                ),
                motion_anchor_ori_b=ObsTerm.configure(
                    func=g1_moves_motion_anchor_ori_b,
                    params={"command_name": "motion"},
                ),
                base_lin_vel=ObsTerm.configure(
                    func=g1_moves_imu_lin_vel_b,
                    params={"entity_name": entity_name},
                ),
                base_ang_vel=ObsTerm.configure(
                    func=observations.base_ang_vel,
                    params={"entity_name": entity_name, "frame": "body"},
                ),
                dofs_pos=observations.DofsPos.configure(entity_name=entity_name, offset_from_default=True),
                dofs_vel=observations.DofsVel.configure(entity_name=entity_name),
                last_action=ObsTerm.configure(
                    func=observations.last_action,
                ),
                terms_order=[
                    "motion",
                    "motion_anchor_pos_b",
                    "motion_anchor_ori_b",
                    "base_lin_vel",
                    "base_ang_vel",
                    "dofs_pos",
                    "dofs_vel",
                    "last_action",
                ],
            ),
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def step(self, obs: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            raw_action = self._policy(obs["policy"])
        return dict(dofs_pos_controller=raw_action)
