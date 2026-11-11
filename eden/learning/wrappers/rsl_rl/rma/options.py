"""Options for the RMA teacher (actor model + PPO) and its depth-student distillation.

Declarative only — importing this module pulls in neither rsl_rl's model/algorithm API nor torch modules, so a task
config that uses it still imports under any rsl_rl Eden supports. The classes their ``class_name`` fields name need
rsl-rl-lib >= 5.5.1 and say so when imported without it. Defaults are the extreme-parkour teacher's (Cheng et al. 2023).
"""

from __future__ import annotations

from typing import Any

from eden.options.learning.rsl_rl.rl import (
    RslRlBaseRunnerOptions,
    RslRlGaussianDistributionOptions,
    RslRlModelWithDistributionOptions,
    RslRlPpoAlgorithmOptions,
)
from eden.options.options import ConfigurableOptions

# Literals (not imported from the implementation modules) so this module stays importable without rsl-rl-lib >= 5.5.1;
# ``tests/learning/wrappers/rsl_rl/rma/`` pins each against its implementation.
_RMA_ACTOR_MODEL_CLASS_NAME = "eden.learning.wrappers.rsl_rl.rma.model:RmaActorModel"
_RMA_PPO_CLASS_NAME = "eden.learning.wrappers.rsl_rl.rma.ppo:RmaPPO"
_RMA_DEPTH_DISTILLATION_CLASS_NAME = "eden.learning.wrappers.rsl_rl.rma.distillation:RmaDepthDistillation"
_RMA_DEPTH_DISTILLATION_RUNNER_CLASS_NAME = "eden.learning.wrappers.rsl_rl.rma.distillation:RmaDepthDistillationRunner"


class RslRlRmaActorOptions(RslRlModelWithDistributionOptions):
    """The RMA actor: estimator + scan / privileged / history encoders + backbone over one flat observation.

    The actor observation set must concatenate to ``[prop | scan | priv_explicit | priv_latent | num_hist x prop]``
    with the widths below (checked when the model is built).

    Parameters
    ----------
    class_name : str
        The model class rsl_rl resolves; leave at the default.
    num_prop, num_scan, num_priv_explicit, num_priv_latent, num_hist : int
        Chunk widths of the flat observation (``num_hist`` frames of the ``num_prop``-d proprioception).
    scan_encoder_dims : list[int]
        Scan encoder widths; the last is the scan latent width (Tanh-bounded).
    priv_encoder_dims : list[int]
        Privileged encoder widths; the last is the adaptation latent width, which the history encoder matches.
    hidden_dims : list[int]
        Actor backbone hidden widths.
    activation : str
        Activation of every encoder and the backbone.
    estimator_hidden_dims : list[int]
        Hidden widths of the base-velocity estimator (proprioception -> privileged-explicit chunk).
    tanh_backbone_output : bool
        Squash the action mean with Tanh.
    obs_normalization : bool
        Must stay ``False``: the chunks are sliced by position and carry the task's fixed scales.
    distribution_cfg : RslRlGaussianDistributionOptions
        A state-independent Gaussian (``std_type="scalar"`` is the teacher's parameterization).
    """

    class_name: str = _RMA_ACTOR_MODEL_CLASS_NAME
    num_prop: int = 53
    num_scan: int = 132
    num_priv_explicit: int = 9
    num_priv_latent: int = 29
    num_hist: int = 10
    scan_encoder_dims: list[int] = [128, 64, 32]
    priv_encoder_dims: list[int] = [64, 20]
    hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu"
    estimator_hidden_dims: list[int] = [128, 64]
    tanh_backbone_output: bool = False
    obs_normalization: bool = False
    distribution_cfg: RslRlGaussianDistributionOptions = RslRlGaussianDistributionOptions(
        init_std=1.0, std_type="scalar"
    )

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        name = type(self).__name__
        if self.num_hist != 10:
            raise ValueError(
                f"{name}(num_hist={self.num_hist}): the history encoder's conv stack is sized for exactly 10 frames "
                "of proprioception. Set num_hist=10 and make the history observation term emit 10 frames."
            )
        empty = [field for field in ("scan_encoder_dims", "priv_encoder_dims") if not getattr(self, field)]
        if empty:
            raise ValueError(
                f"{name}: {', '.join(f'{field}=[]' for field in empty)} — the last width of each is a latent width "
                "the actor needs (the scan latent, the adaptation latent); give each at least one layer."
            )
        if self.obs_normalization:
            raise ValueError(
                f"{name}(obs_normalization=True): the actor slices its flat observation by position and the task "
                "already scales each chunk, so a running normalizer would shift the chunks it slices. Leave it False."
            )


class RslRlRmaPpoAlgorithmOptions(RslRlPpoAlgorithmOptions):
    """PPO for an RMA actor: concurrent estimator, privileged-latent regularization, DAgger, optional mirror loss.

    Needs ``actor=RslRlRmaActorOptions(...)``. The PPO defaults are the extreme-parkour teacher's.

    Parameters
    ----------
    class_name : str
        The algorithm class rsl_rl resolves; leave at the default.
    estimator_learning_rate : float
        Learning rate of the estimator's own Adam.
    train_with_estimated_states : bool
        Act on the estimator's prediction of the privileged-explicit chunk during rollouts (storage keeps the truth).
    dagger_update_freq : int
        Every this-many learning iterations, roll out with the history latent and regress the history encoder onto
        the privileged latent afterwards.
    priv_reg_coef_schedule : list[float]
        ``(start, end, ramp_start, ramp_length)`` of the privileged-regularization coefficient, ramped linearly over
        update counts (PPO and DAgger updates both count).
    symmetry_coef : float
        Weight of the mirror-equivariance loss (0 disables it).
    mirror_class_name : str | None
        ``"module.path:Class"`` of the task's left-right mirror, built as ``Class(dims, device)`` with
        ``mirror_obs(flat)`` / ``mirror_action(actions)``. Required when ``symmetry_coef > 0``.
    """

    class_name: str = _RMA_PPO_CLASS_NAME
    learning_rate: float = 2.0e-4
    entropy_coef: float = 0.01
    estimator_learning_rate: float = 1.0e-4
    train_with_estimated_states: bool = True
    dagger_update_freq: int = 20
    priv_reg_coef_schedule: list[float] = [0.0, 0.1, 2000, 3000]
    symmetry_coef: float = 0.0
    mirror_class_name: str | None = None

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        name = type(self).__name__
        if len(self.priv_reg_coef_schedule) != 4:
            raise ValueError(
                f"{name}(priv_reg_coef_schedule={self.priv_reg_coef_schedule}) must be 4 numbers: "
                "[start, end, ramp_start, ramp_length]."
            )
        if self.priv_reg_coef_schedule[3] <= 0 or self.priv_reg_coef_schedule[2] < 0:
            raise ValueError(
                f"{name}(priv_reg_coef_schedule={self.priv_reg_coef_schedule}): ramp_start (3rd) must be >= 0 and "
                "ramp_length (4th) > 0 updates; for a constant coefficient use start == end."
            )
        if self.dagger_update_freq < 1:
            raise ValueError(f"{name}(dagger_update_freq={self.dagger_update_freq}) must be >= 1 iteration.")
        if self.symmetry_coef < 0.0:
            raise ValueError(f"{name}(symmetry_coef={self.symmetry_coef}) must be >= 0 (0 disables the mirror loss).")
        if self.symmetry_coef > 0.0 and not self.mirror_class_name:
            raise ValueError(
                f"{name}(symmetry_coef={self.symmetry_coef}) needs mirror_class_name='module.path:Class' — the task's "
                "left-right mirror (e.g. 'eden.tasks.extreme_parkour.symmetry:ParkourMirror'); or set symmetry_coef=0."
            )
        unsupported = {
            field: getattr(self, field)
            for field, off in (("rnd_cfg", None), ("symmetry_cfg", None), ("share_cnn_encoders", False))
            if getattr(self, field) != off
        }
        if unsupported:
            raise ValueError(
                f"{name} does not support {unsupported}: RmaPPO has no RND or CNN encoders, and its mirror loss is "
                "symmetry_coef + mirror_class_name (not rsl_rl's symmetry_cfg). Leave them at their defaults."
            )


class RslRlRmaDepthDistillationAlgorithmOptions(ConfigurableOptions):
    """DAgger distillation of an RMA teacher into a depth student (recurrent depth encoder + copied actor net).

    Parameters
    ----------
    class_name : str
        The algorithm class the runner resolves; leave at the default.
    learning_rate : float
        Adam learning rate of the student (depth actor + depth encoder).
    max_grad_norm : float
        Gradient-norm clip on the depth actor.
    gru_hidden_dim : int
        Hidden width of the depth encoder's GRU.
    update_interval : int
        Control steps between depth-encoder passes; must equal the env's depth refresh cadence.
    yaw_scale : float
        Stretch of the Tanh-bounded yaw head.
    yaw_slots : list[int]
        ``(start, stop)`` of the two delta-yaw slots inside the proprioception chunk, which the student predicts.
    num_pretrain_iters : int
        Iterations driven by the teacher's actions before the student takes over.
    depth_obs_group : str
        Observation group carrying the processed depth frame.
    yaw_gate_obs_group : str
        Observation group whose flag (> 0.5) lets the student see its own yaw prediction.
    """

    class_name: str = _RMA_DEPTH_DISTILLATION_CLASS_NAME
    learning_rate: float = 1.0e-3
    max_grad_norm: float = 1.0
    gru_hidden_dim: int = 512
    update_interval: int = 5
    yaw_scale: float = 1.5
    yaw_slots: list[int] = [6, 8]
    num_pretrain_iters: int = 0
    depth_obs_group: str = "depth_camera"
    yaw_gate_obs_group: str = "delta_yaw_ok"

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        name = type(self).__name__
        bad = {
            field: getattr(self, field)
            for field, ok in (
                ("update_interval", self.update_interval >= 1),
                ("num_pretrain_iters", self.num_pretrain_iters >= 0),
                ("gru_hidden_dim", self.gru_hidden_dim >= 1),
                ("learning_rate", self.learning_rate > 0.0),
                ("max_grad_norm", self.max_grad_norm > 0.0),
                ("yaw_scale", self.yaw_scale > 0.0),
            )
            if not ok
        }
        if bad:
            raise ValueError(
                f"{name}: {bad} out of range — update_interval and gru_hidden_dim need >= 1, num_pretrain_iters "
                ">= 0, learning_rate / max_grad_norm / yaw_scale > 0."
            )
        if len(self.yaw_slots) != 2 or self.yaw_slots[0] < 0 or self.yaw_slots[1] - self.yaw_slots[0] != 2:
            raise ValueError(
                f"{name}(yaw_slots={self.yaw_slots}) must be [start, start + 2]: the (delta_yaw, delta_next_yaw) "
                "slots of the proprioception chunk, which the yaw head predicts exactly 2 of."
            )


class RslRlRmaDepthDistillationRunnerOptions(RslRlBaseRunnerOptions):
    """Runner options selecting the depth-student DAgger loop; train with ``--checkpoint <teacher checkpoint>``.

    Parameters
    ----------
    class_name : str
        The runner ``eden train`` builds; leave at the default.
    actor : RslRlRmaActorOptions
        The teacher's architecture — it must match the teacher checkpoint being distilled.
    algorithm : RslRlRmaDepthDistillationAlgorithmOptions
        The distillation's knobs.
    """

    class_name: str = _RMA_DEPTH_DISTILLATION_RUNNER_CLASS_NAME
    actor: RslRlRmaActorOptions = RslRlRmaActorOptions()
    algorithm: RslRlRmaDepthDistillationAlgorithmOptions = RslRlRmaDepthDistillationAlgorithmOptions()

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        name = type(self).__name__
        interval = self.algorithm.update_interval
        if self.num_steps_per_env <= 0 or self.num_steps_per_env % interval != 0:
            raise ValueError(
                f"{name}(num_steps_per_env={self.num_steps_per_env}) must be a positive multiple of "
                f"algorithm.update_interval ({interval}), so an iteration boundary never splits a depth-refresh window."
            )
        if self.algorithm.yaw_slots[1] > self.actor.num_prop:
            raise ValueError(
                f"{name}: algorithm.yaw_slots={self.algorithm.yaw_slots} runs past the {self.actor.num_prop}-d "
                "proprioception chunk (actor.num_prop)."
            )
