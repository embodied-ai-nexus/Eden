r"""Humanoid-GPT whole-body motion-tracking policy on Unitree G1 (29 DOF).

Reproduces Humanoid-GPT's 136-dim obs + action transform from eden state + a
:class:`MotionCommand`, running the released Transformer ``pns_wo_priv216_trans.onnx``
(default) or MLP ``pns_wo_priv216.onnx`` (``policy_type="mlp"``) ONNX checkpoint.
Both share the per-frame ``obs`` -> ``continuous_actions`` contract; the Transformer
also takes a K-frame obs history ``(B, K, 136)``. Reference (obs = ``get_nn_state``,
action = ``nn2motor_action``): https://github.com/GalaxyGeneralRobotics/Humanoid-GPT

Observation (136 dim), with ``t`` = ``MotionCommand.time_steps``, ``t_next =
clamp(t+1)``, pelvis = motion body 0, ``default`` = :data:`HGPT_DEFAULT_DOFS_POS`,
``proj_grav(q) = R(q)^T @ [0,0,-1]``:

    gyro (3)            robot pelvis angular velocity, body frame
    gvec (3)            proj_grav(robot pelvis quat)
    q - default (29)    robot joint pos delta
    qd (29)             robot joint vel
    last_action (29)    previous raw network output (not the scaled target)
    ref_q_next-default  (29) reference joint pos delta at t_next
    ref_height (1)      reference pelvis z at t_next
    ref_gvec (3)        proj_grav(reference pelvis quat at t_next)
    ref_cvel_gv (6)     [R_yaw^T @ ang, R_yaw^T @ lin] of ref pelvis at t_next
    yaw_cmd (2)         [cos, sin] of wrap(yaw(ref@t) - yaw(robot))
    xy_cmd (2)          R(-yaw_robot) @ (ref_xy@t - robot_xy)

The env's ``policy`` observation group declares it (:meth:`HumanoidGptPolicy.observation_options`):
core proprioception terms plus the terms in
:mod:`eden.extensions.policies.humanoid_gpt.custom_terms.observations`. For the Transformer the
group stacks a K-frame history, oldest first, un-flattened to ``(B, K, 136)``.

Action: ``joint_pos_targets = default + nn_action * action_scale``, where
``action_scale == G1_ACTION_SCALE == 0.25 * torque_limit / kp``, applied by the
action term (``ImplicitPDController``, per-joint ``scale`` and ``offset``,
``reference_source="zero"``), so the env's ``last_action`` is the raw network
output; ``G1Dofs29`` gains already match the reference.

Notes
-----
* Batched (num_envs>=1); the static batch axis is dynamic-batch patched only for
  num_envs>1 (num_envs=1, the deploy path, runs unpatched).
* The reference stream is EMA-smoothed + velocities recomputed (``ref_ema_alpha``),
  faithfully porting ``apply_ema_qpos`` + ``recompute_qvel`` (see :class:`HgptReference`);
  ``ref_ema_alpha=None`` uses the raw buffers.
* ``ref_cvel_gv`` matches the reference's MuJoCo ``cvel`` (velocity at the body CoM);
  ``ref_cvel_at_com=True`` (default) applies the exact CoM transfer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from eden import actions, observations
from eden.constants import ReferenceSource
from eden.extensions.policies.base import POLICY_REGISTRY, PolicyBase, PolicyOptions
from eden.extensions.policies.humanoid_gpt.constants import HGPT_DEFAULT_DOFS_POS, OBS_DIM, resolve_action_scale
from eden.extensions.policies.humanoid_gpt.custom_terms.observations import HgptReference, hgpt_dofs_pos_rel
from eden.extensions.policies.humanoid_gpt.onnx_wrapper import HumanoidGptOnnx
from eden.managers.terms.commands.mimic import motion_command_options
from eden.options.managers.observations import ObservationGroupOptions
from eden.options.robots.unitree_g1 import G1_DOFS29_NAME
from eden.terms import ObsTerm

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.actions import ActionTermOptions
    from eden.options.managers.commands import CommandTermOptions


_ROOT_BODY: str = "pelvis"
_ANCHOR_BODY: str = "pelvis"
# MotionCommand body set. Index 0 must be the floating-base body (pelvis) so the
# reset can set qpos from the reference. torso_link is kept only so the example
# can report a second tracking-error link; the policy obs only reads pelvis.
_MOTION_BODY_NAMES: list[str] = [_ROOT_BODY, "torso_link"]

#: The env observation group the network reads.
OBS_GROUP = "policy"

#: ``observation_options`` default for a knob the caller leaves out: the class's own setting.
_CLASS_SETTING = object()

# Released checkpoint paths (relative to HF ``Kashu7100/eden_policies``).
_MLP_CHECKPOINT: str = "humanoid_gpt/pns_wo_priv216.onnx"
_TRANSFORMER_CHECKPOINT: str = "humanoid_gpt/pns_wo_priv216_trans.onnx"


@POLICY_REGISTRY.register()
class HumanoidGptPolicy(PolicyBase):
    """Humanoid-GPT whole-body motion-tracking policy for the Unitree G1 (29 DOF).

    Runs the released Transformer (default) or MLP (``policy_type="mlp"``)
    ONNX checkpoint; both share the 136-dim observation / action contract, the
    transformer additionally consuming a K-frame obs history.

    ``policy_type``, ``transformer_history_len``, ``ref_ema_alpha`` and ``ref_cvel_at_com`` shape the
    observation as well as the network, so pass the same values to :meth:`observation_options` and
    :meth:`configure`; an instance configured differently from the env it is handed is refused.
    """

    # step() reads the observation dict the env publishes, so this policy owns the deploy env.
    reads_env_observations = True

    # Active checkpoint PolicyBase resolves + md5-verifies. Default None selects the
    # mlp_/transformer_ pair below by ``policy_type`` (see ``_select_checkpoint``); the
    # example's ``--ckpt-path`` sets a local path here (md5 cleared).
    checkpoint_path: str | None = None
    md5_hash: str | None = None

    # The two released checkpoint pairs, selected by ``policy_type``. Both share the
    # ``obs`` -> ``continuous_actions`` contract (transformer obs is (B, K, 136));
    # both are hosted on HF ``Kashu7100/eden_policies``.
    mlp_checkpoint_path: str = _MLP_CHECKPOINT
    mlp_md5_hash: str | None = "7147c555a68432c76723c52468bbdd26"
    transformer_checkpoint_path: str = _TRANSFORMER_CHECKPOINT
    transformer_md5_hash: str | None = "c2a884643074cee94ab57722bea116b4"

    # "transformer" (default) | "mlp": which released checkpoint pair runs (and
    # whether the K-frame history is fed).
    policy_type: str = "transformer"

    # Report the reference pelvis velocity at the CoM (MuJoCo ``cvel``), not the
    # link origin: v_com = v_origin + w x R(q) @ PELVIS_COM_OFFSET. False = link origin.
    ref_cvel_at_com: bool = True

    # Reference smoothing (``apply_ema_qpos`` + ``recompute_qvel``): EMA-filter the
    # root pos/quat + joints and recompute root velocity by finite difference. Exact
    # at 50 Hz (one frame per control step). ``None`` = raw buffers.
    ref_ema_alpha: float | None = 0.8

    # K-frame obs history for the transformer, (B, K, 136), newest last. K=16 is
    # baked into the released graph; ``_validate_transformer_graph`` checks it.
    transformer_history_len: int = 16

    def __init__(self, env: EnvBase, options: PolicyOptions) -> None:
        # Select the checkpoint pair on ``options`` (not ``self``) BEFORE super():
        # ConfigurableMixin mirrors options -> self, clobbering a pre-super attr.
        if options.policy_type not in ("mlp", "transformer"):
            raise ValueError(f"Unknown policy_type={options.policy_type!r} (expected 'mlp' or 'transformer').")
        options.checkpoint_path, options.md5_hash = self._select_checkpoint(
            options.policy_type,
            options.checkpoint_path,
            options.md5_hash,
            options.mlp_checkpoint_path,
            options.mlp_md5_hash,
            options.transformer_checkpoint_path,
            options.transformer_md5_hash,
        )
        super().__init__(env, options)

        self.device = env.device
        robot = self.require_entity("robot")
        if list(robot.dofs_name) != G1_DOFS29_NAME:
            raise ValueError(
                f"HumanoidGptPolicy drives the G1's 29 DOFs in G1Dofs29 order; this robot's dofs_name is "
                f"{robot.dofs_name}. Run it with --robot G1Dofs29."
            )
        self._check_env_matches_configuration(env)

        # Patch the batch axis only for num_envs>1; num_envs=1 runs unpatched.
        self._model = HumanoidGptOnnx(self.checkpoint_path, device=str(env.device), dynamic_batch=(env.num_envs > 1))
        if self.policy_type == "transformer":
            self._validate_transformer_graph()

    def _check_env_matches_configuration(self, env: EnvBase) -> None:
        """Refuse an env whose observation group was declared for another configuration of this policy.

        The env is declared by ``observation_options``, which takes these knobs separately from
        ``configure``, so an instance configured with, say, ``policy_type="mlp"`` on an env built for
        the Transformer would feed the MLP a frame history; one with another ``ref_ema_alpha`` would
        read differently smoothed references. Neither raises on its own.
        """
        expected_shape = (self.transformer_history_len, OBS_DIM) if self.policy_type == "transformer" else (OBS_DIM,)
        group_dims = env.observation_manager.group_obs_dim
        if group_dims.get(OBS_GROUP) != expected_shape:
            raise ValueError(
                f"HumanoidGptPolicy(policy_type={self.policy_type!r}) reads a {expected_shape} '{OBS_GROUP}' "
                f"observation group, but this env publishes {group_dims}. Build the env from "
                f"HumanoidGptPolicy.observation_options(policy_type={self.policy_type!r}), not from another "
                f"task's config."
            )
        terms = {(group, name): term for group, name, term in env.observation_manager.iter_terms()}
        reference = terms.get((OBS_GROUP, "reference"))
        configured = {"ema_alpha": self.ref_ema_alpha, "cvel_at_com": self.ref_cvel_at_com}
        built = {key: getattr(reference, key, None) for key in configured}
        if built != configured:
            raise ValueError(
                f"The env's '{OBS_GROUP}/reference' term was built with {built}, but this policy is configured "
                f"with {configured} (as ref_ema_alpha / ref_cvel_at_com). Pass the same ref_ema_alpha / "
                f"ref_cvel_at_com to HumanoidGptPolicy.observation_options() when building the env."
            )

    @staticmethod
    def _select_checkpoint(
        policy_type: str,
        checkpoint_path: str | None,
        md5_hash: str | None,
        mlp_checkpoint_path: str,
        mlp_md5_hash: str | None,
        transformer_checkpoint_path: str,
        transformer_md5_hash: str | None,
    ) -> tuple[str, str | None]:
        """Pick the ``(checkpoint_path, md5_hash)`` pair to load.

        An explicit ``checkpoint_path`` (set by ``--ckpt-path`` via
        ``use_local_checkpoint``, which also clears the md5) wins; otherwise select the
        ``mlp_*`` or ``transformer_*`` pair by ``policy_type``.
        """
        if checkpoint_path is not None:
            return checkpoint_path, md5_hash
        if policy_type == "transformer":
            return transformer_checkpoint_path, transformer_md5_hash
        return mlp_checkpoint_path, mlp_md5_hash

    def _validate_transformer_graph(self) -> None:
        """Fail fast if the transformer ONNX input isn't ``(B, transformer_history_len,
        OBS_DIM)`` — a static-shaped mismatch would otherwise raise cryptically at the
        first step. Symbolic dims are left unchecked.
        """
        shape = self._model.session.get_inputs()[0].shape  # e.g. [batch, K, 136]
        if len(shape) != 3:
            raise ValueError(
                f"Transformer ONNX {self.checkpoint_path!r} expects a rank-3 input (B, K, {OBS_DIM}); "
                f"got shape {shape}. An MLP export is rank-2 — use policy_type='mlp' for it."
            )
        k_dim, feat_dim = shape[1], shape[2]
        if isinstance(feat_dim, int) and feat_dim != OBS_DIM:
            raise ValueError(f"Transformer ONNX {self.checkpoint_path!r} feature dim {feat_dim} != OBS_DIM {OBS_DIM}.")
        if isinstance(k_dim, int) and k_dim != self.transformer_history_len:
            raise ValueError(
                f"Transformer ONNX {self.checkpoint_path!r} history length {k_dim} != "
                f"transformer_history_len {self.transformer_history_len}; set "
                f"transformer_history_len={k_dim}."
            )

    # ------------------------------------------------------------------
    # Env / action / command / observation specs
    # ------------------------------------------------------------------

    @classmethod
    def env_options(cls) -> dict[str, Any]:
        # control_dt = sim_dt * decimation = 0.01 * 2 = 0.02s (50 Hz), matching
        # Humanoid-GPT's ctrl_dt and the 50 Hz reference motion.
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
        """The network's raw action, scaled per joint about Humanoid-GPT's nominal stance."""
        return dict(
            dofs_pos_controller=actions.ImplicitPDController.configure(
                entity_name=entity_name,
                dofs_name=list(G1_DOFS29_NAME),
                scale=dict(zip(G1_DOFS29_NAME, resolve_action_scale())),
                # The nominal stance, not the robot's default pose (the stock G1Dofs29 differs on six joints).
                offset=dict(zip(G1_DOFS29_NAME, HGPT_DEFAULT_DOFS_POS)),
                reference_source=ReferenceSource.ZERO,
            ),
        )

    @classmethod
    def command_options(
        cls,
        motion_file: str | list[str] = "motion.npz",
        debug_vis: bool = False,
        entity_name: str = "robot",
        ghost_entity_name: str = "",
        sampling_mode: str = "start",
    ) -> dict[str, CommandTermOptions]:
        # list of clips -> multi-clip term; single path -> single-clip; step() is
        # agnostic to which backs it.
        return motion_command_options(
            motion_file,
            anchor_link_name=_ANCHOR_BODY,
            links_name=_MOTION_BODY_NAMES,
            entity_name=entity_name,
            ghost_entity_name=ghost_entity_name,
            debug_vis=debug_vis,
            sampling_mode=sampling_mode,
        )

    @classmethod
    def observation_options(
        cls,
        entity_name: str = "robot",
        policy_type: str = _CLASS_SETTING,
        transformer_history_len: int = _CLASS_SETTING,
        ref_ema_alpha: float | None = _CLASS_SETTING,
        ref_cvel_at_com: bool = _CLASS_SETTING,
    ) -> dict[str, ObservationGroupOptions]:
        """The 136-dim frame; for the Transformer, stacked into a ``(K, 136)`` history backfilled at reset.

        Each knob not passed is the class's own setting, which is what a task-less
        ``eden inference|deploy --policy humanoid_gpt`` builds with. Pass the values the policy is
        configured with otherwise, e.g. ``observation_options(policy_type="mlp")`` alongside
        ``configure(policy_type="mlp")``.
        """
        knobs = {
            "policy_type": policy_type,
            "transformer_history_len": transformer_history_len,
            "ref_ema_alpha": ref_ema_alpha,
            "ref_cvel_at_com": ref_cvel_at_com,
        }
        knobs = {name: getattr(cls, name) if value is _CLASS_SETTING else value for name, value in knobs.items()}
        if knobs["policy_type"] not in ("mlp", "transformer"):
            raise ValueError(f"Unknown policy_type={knobs['policy_type']!r} (expected 'mlp' or 'transformer').")
        entity = {"entity_name": entity_name}
        history = (
            {"history_length": knobs["transformer_history_len"], "flatten_history_dim": False}
            if knobs["policy_type"] == "transformer"
            else {}
        )
        return {
            OBS_GROUP: ObservationGroupOptions(
                enable_corruption=False,
                **history,
                gyro=ObsTerm.configure(func=observations.base_ang_vel, params={**entity, "frame": "body"}),
                gvec=observations.ProjectedGravity.configure(entity_name=entity_name),
                dofs_pos=ObsTerm.configure(func=hgpt_dofs_pos_rel, params=entity),
                dofs_vel=observations.DofsVel.configure(entity_name=entity_name),
                last_action=ObsTerm.configure(func=observations.last_action),
                reference=HgptReference.configure(
                    root_link_name=_ROOT_BODY,
                    entity_name=entity_name,
                    ema_alpha=knobs["ref_ema_alpha"],
                    cvel_at_com=knobs["ref_cvel_at_com"],
                ),
            )
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def step(self, obs: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        # clone: infer() aliases the ONNX output buffer, which the next call overwrites.
        return {"dofs_pos_controller": self._model.infer(obs[OBS_GROUP]).clone()}
