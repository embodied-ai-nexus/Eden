r"""ProtoMotions v3 motion-tracking policy (g1-bones-deploy) — torch checkpoint loader.

Loads the actor straight from an AMP training ``last.ckpt`` and reproduces the
obs-construction + action-transform pipeline that the ProtoMotions ONNX
exporter (`deployment/export_bm_tracker_onnx.py`) bakes into its unified ONNX.

Pipeline reproduced (matches `unified_pipeline.yaml` / training config):

    [raw context]                                                obs (349 dim)
      current.dof_pos                                       \\
      current.dof_vel                                        \\
      current.anchor_rot           --reduced_coords-->  noisy_reduced_coords_obs (64)
      current.root_local_ang_vel                             /
                                                            /
      mimic.future_anchor_rot                              \\
      mimic.future_dof_pos      --reduced_coords_target--> noisy_mimic_reduced_coords_target_poses (256)
      mimic.future_dof_vel                                 /
                                                            \\
      historical.processed_actions  --identity-->            historical_previous_processed_actions (29)

    obs --normalize(mean, var, eps, clamp ±5)--> 6×Linear(1024)+ReLU --Linear(29)--> mean_action

    joint_pos_targets = pd_action_offset + action_scale * mean_action  (bm_pd_action)

The env's ``policy`` observation group declares the 349-dim input
(:meth:`ProtoMotionV3Policy.observation_options`, terms in
:mod:`eden.extensions.policies.protomotionv3.custom_terms.observations`), and the
action term (``ImplicitPDController``, per-joint ``scale`` and ``offset``,
``reference_source="zero"``) applies ``bm_pd_action``, so ``step`` hands it the
raw ``mean_action``; stiffness/damping on ``G1Dofs29`` already match the
training values.

Limitations
-----------
* The action transform constants ``_PD_ACTION_OFFSET`` and ``_ACTION_SCALE``
  are baked for the ``g1-bones-deploy`` bundle — they come straight from the
  exported ``unified_pipeline.yaml``. A different checkpoint would need its
  own constants (or the YAML loaded alongside).
* Restricted to ``num_envs=1`` to match the bundle's deployment contract.
"""

from __future__ import annotations

from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from eden import actions, observations
from eden.constants import ReferenceSource
from eden.extensions.policies.base import POLICY_REGISTRY, PolicyBase, PolicyOptions
from eden.extensions.policies.protomotionv3.custom_terms.observations import (
    ProtomotionAnchorGravity,
    ProtomotionPrevTargets,
    ProtomotionTargetPoses,
)
from eden.managers.terms.commands.mimic import MotionCommand
from eden.options.managers.observations import ObservationGroupOptions
from eden.options.robots.unitree_g1 import G1_ACTION_SCALE, G1_DOFS29_NAME, G1Dofs29
from eden.terms import ObsTerm

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.actions import ActionTermOptions
    from eden.options.managers.commands import CommandTermOptions


# ---------------------------------------------------------------------------
# Body / DOF / motion constants
# ---------------------------------------------------------------------------

_ANCHOR_BODY: str = "torso_link"
_ROOT_BODY: str = "pelvis"
# The motion only needs to expose pelvis (for floating-base reset) and torso
# (for the policy's anchor rotation). MotionCommand reads these via
# find_named_links_idx_local against the robot's MJCF; index 0 must be the
# floating-base body so MotionCommand can set qpos.
_MOTION_BODY_NAMES: list[str] = [_ROOT_BODY, _ANCHOR_BODY]
_FUTURE_STEP_OFFSETS: tuple[int, ...] = (1, 2, 4, 8)
_NUM_DOFS: int = 29

#: The env observation group the actor reads.
OBS_GROUP = "policy"

# ---------------------------------------------------------------------------
# Actor architecture (set by the trained checkpoint)
# ---------------------------------------------------------------------------

_OBS_DIM: int = 349  # 64 (reduced_coords) + 256 (target_poses) + 29 (previous processed actions)
_HIDDEN_DIM: int = 1024
_NUM_HIDDEN: int = 6
_NORM_EPS: float = 1e-5
_NORM_CLAMP: float = 5.0

# ---------------------------------------------------------------------------
# Action-transform constants. Order matches G1Dofs29().dofs_name (MJCF order).
# ---------------------------------------------------------------------------
#
# `_PD_ACTION_OFFSET` is the protomotions training default_dof_pos, lifted
# from the exported `unified_pipeline.yaml`.
#
# `_ACTION_SCALE` is `bm_pd_action`'s per-joint scale = `effort_limit / kp`.
# eden's `G1_ACTION_SCALE` already encodes the Raibert heuristic
# `0.25 * effort_limit / kp` (see `RAIBERT_HEURISTIC_ACTION_SCALE`), so
# multiplying by 4 recovers the protomotions training scale.

_PD_ACTION_OFFSET: tuple[float, ...] = (
    -0.31200000643730164,
    0.0,
    0.0,
    0.6690000295639038,
    -0.3630000054836273,
    0.0,
    -0.31200000643730164,
    0.0,
    0.0,
    0.6690000295639038,
    -0.3630000054836273,
    0.0,
    0.0,
    0.0,
    0.0,
    0.20000000298023224,
    0.20000000298023224,
    0.0,
    0.6000000238418579,
    0.0,
    0.0,
    0.0,
    0.20000000298023224,
    -0.20000000298023224,
    0.0,
    0.6000000238418579,
    0.0,
    0.0,
    0.0,
)
_ACTION_SCALE: tuple[float, ...] = tuple(
    4.0 * next(v for p, v in G1_ACTION_SCALE.items() if fnmatchcase(name, p)) for name in G1Dofs29().dofs_name
)


# ---------------------------------------------------------------------------
# Actor module
# ---------------------------------------------------------------------------


class _CkptActor(nn.Module):
    """Frozen MLP actor with running-mean/std obs normalization, loaded from .ckpt."""

    def __init__(self, ckpt_path: str, device: torch.device | str) -> None:
        super().__init__()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["model"]

        layers: list[nn.Module] = []
        in_dim = _OBS_DIM
        for i in range(_NUM_HIDDEN):
            lin = nn.Linear(in_dim, _HIDDEN_DIM)
            lin.weight.data.copy_(sd[f"_actor.mu.mlp.{2 * i}.weight"].float())
            lin.bias.data.copy_(sd[f"_actor.mu.mlp.{2 * i}.bias"].float())
            layers += [lin, nn.ReLU()]
            in_dim = _HIDDEN_DIM
        head = nn.Linear(in_dim, _NUM_DOFS)
        head.weight.data.copy_(sd[f"_actor.mu.mlp.{2 * _NUM_HIDDEN}.weight"].float())
        head.bias.data.copy_(sd[f"_actor.mu.mlp.{2 * _NUM_HIDDEN}.bias"].float())
        layers.append(head)
        self.mlp = nn.Sequential(*layers)

        mean = sd["_actor.mu.norm.running_obs_norm.mean"].view(1, -1).float()
        var = sd["_actor.mu.norm.running_obs_norm.var"].view(1, -1).float()
        std = torch.sqrt(var + _NORM_EPS)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        self.to(device).eval()

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        norm_obs = ((obs - self.mean) / self.std).clamp(-_NORM_CLAMP, _NORM_CLAMP)
        return self.mlp(norm_obs)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@POLICY_REGISTRY.register()
class ProtoMotionV3Policy(PolicyBase):
    """ProtoMotions v3 (`g1-bones-deploy`) motion-tracking policy on Unitree G1 (29 DOF).

    Loads the AMP actor directly from ``last.ckpt`` and reproduces the obs +
    action transform pipeline in PyTorch, so checkpoints can be tested without
    re-exporting an ONNX bundle.
    """

    # step() reads the observation dict the env publishes, so this policy owns the deploy env.
    reads_env_observations = True

    # Resolved by PolicyBase via the Hugging Face asset resolver
    # (Kashu7100/eden_policies). Override on a subclass / via configure() to
    # point at a different ckpt — must come from the same g1-bones-deploy
    # export config (see _PD_ACTION_OFFSET / _ACTION_SCALE above).
    checkpoint_path: str = "protomotion_v3/g1_bones_deploy.ckpt"
    md5_hash: str | None = "205a22b9becc4c8b589da7509d940b5c"

    def __init__(self, env: EnvBase, options: PolicyOptions) -> None:
        super().__init__(env, options)

        if env.num_envs != 1:
            raise ValueError(
                f"{type(self).__name__} only supports num_envs=1 (got {env.num_envs}); "
                "matches the g1-bones-deploy deployment contract."
            )
        self.device = env.device
        robot = self.require_entity("robot")
        # The joint observations and the action term read the robot's DOFs in its own order.
        if list(robot.dofs_name) != G1_DOFS29_NAME:
            raise ValueError(
                f"ProtoMotionV3Policy drives the G1's 29 DOFs in G1Dofs29 order; this robot's dofs_name is "
                f"{robot.dofs_name}. Build the robot from ProtoMotionV3Policy.robot_options()."
            )

        # The env has to be the one this policy declares: a task env publishes its own
        # ``policy`` group, which the actor would read as garbage.
        group_dims = env.observation_manager.group_obs_dim
        if group_dims.get(OBS_GROUP) != (_OBS_DIM,):
            raise ValueError(
                f"ProtoMotionV3Policy reads a {_OBS_DIM}-dim '{OBS_GROUP}' observation group, but this env "
                f"publishes {group_dims}. Build the env from ProtoMotionV3Policy.observation_options() (a "
                f"task-less `eden inference --policy protomotionv3` does), not from another task's config."
            )

        self._actor = _CkptActor(self.checkpoint_path, env.device)

    # ------------------------------------------------------------------
    # Env / action / command / observation specs
    # ------------------------------------------------------------------

    @classmethod
    def robot_options(cls) -> G1Dofs29:
        """Return the stock ``G1Dofs29``, whose gains match the training values.

        Its default pose does not: the training pose is the action term's ``offset``.
        """
        return G1Dofs29()

    @classmethod
    def env_options(cls) -> dict[str, Any]:
        # Matches `simulation_params.genesis` from the training resolved config:
        #   fps=100, decimation=2, substeps=2 → control_dt=0.02s (50 Hz).
        return dict(
            sim_dt=0.01,
            sim_substeps=2,
            decimation=2,
            iterations=20,
            ls_iterations=20,
            max_collision_pairs=20,
            batch_dofs_info=False,
            use_gjk_collision=False,
            enable_self_collision=False,
            enable_multi_contact=False,
        )

    @classmethod
    def action_options(cls, entity_name: str = "robot") -> dict[str, ActionTermOptions]:
        """``bm_pd_action``: the actor's mean action, scaled per joint about the training default pose."""
        dofs_name = G1Dofs29().dofs_name
        return dict(
            dofs_pos_controller=actions.ImplicitPDController.configure(
                entity_name=entity_name,
                dofs_name=dofs_name,
                scale=dict(zip(dofs_name, _ACTION_SCALE)),
                # ProtoMotions' default pose, not the robot's (the stock G1Dofs29 differs on 12 joints).
                offset=dict(zip(dofs_name, _PD_ACTION_OFFSET)),
                reference_source=ReferenceSource.ZERO,
            ),
        )

    @classmethod
    def command_options(
        cls,
        motion_file: str = "motion.npz",
        debug_vis: bool = False,
        entity_name: str = "robot",
        ghost_entity_name: str = "",
        sampling_mode: str = "start",
    ) -> dict[str, CommandTermOptions]:
        return dict(
            motion=MotionCommand.configure(
                resampling_time_range=(1.0e9, 1.0e9),
                debug_vis=debug_vis,
                entity_name=entity_name,
                ghost_entity_name=ghost_entity_name,
                motion_file=motion_file,
                anchor_link_name=_ANCHOR_BODY,
                links_name=_MOTION_BODY_NAMES,
                pose_range={},
                velocity_range={},
                dofs_position_range=(0.0, 0.0),
                sampling_mode=sampling_mode,
            ),
        )

    @classmethod
    def observation_options(cls, entity_name: str = "robot") -> dict[str, ObservationGroupOptions]:
        """The 349-dim input: reduced coords (64), future target poses (256), previous joint targets (29)."""
        entity = {"entity_name": entity_name}
        return {
            OBS_GROUP: ObservationGroupOptions(
                enable_corruption=False,
                # noisy_reduced_coords_obs (64)
                dofs_pos=observations.DofsPos.configure(entity_name=entity_name),
                dofs_vel=observations.DofsVel.configure(entity_name=entity_name),
                root_ang_vel=ObsTerm.configure(func=observations.base_ang_vel, params={**entity, "frame": "body"}),
                anchor_gravity=ProtomotionAnchorGravity.configure(
                    entity_name=entity_name, anchor_link_name=_ANCHOR_BODY
                ),
                # noisy_mimic_reduced_coords_target_poses (256)
                target_poses=ProtomotionTargetPoses.configure(
                    entity_name=entity_name, anchor_link_name=_ANCHOR_BODY, future_offsets=_FUTURE_STEP_OFFSETS
                ),
                # historical_previous_processed_actions (29)
                prev_targets=ProtomotionPrevTargets.configure(),
            )
        }

    # ------------------------------------------------------------------
    # Lifecycle / inference
    # ------------------------------------------------------------------

    def step(self, obs: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {"dofs_pos_controller": self._actor(obs[OBS_GROUP])}
