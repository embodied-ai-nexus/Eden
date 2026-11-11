"""AssistMimic: a PHC-initialized caregiver lifts a weakened recipient, both tracking a paired clip.

Simplified port of `AssistMimic <https://github.com/YutoShibata07/AssistMimic>`_ (Shibata et al., CVPR 2026)
on PHC's SMPL humanoid. Two tasks, one per dataset:

``assistmimic`` (:class:`AssistMimicConfig`, ``inter-x``)
    ``env_im_interx_helpup.yaml`` + ``im_simpleliftup_mlp.yaml`` — the recipient sits on the floor and is
    helped to their feet.
``assistmimic_bed`` (:class:`AssistMimicBedConfig`, ``hhi-assist``)
    ``env_im_hhi-assist.yaml`` + ``im_hhi-assist_mlp.yaml`` — the recipient lies on a bed and is helped out
    of it, with the hips and trunk weakened further and the caregiver paid for unloading them.

Both mirror AssistMimic: 30 Hz control over a 1/60 s, 2-substep simulation, one env holding the
caregiver/recipient pair. The two observation groups and the two reward groups are the two agents, which no
single-agent runner can train, so the configs declare no ``runner_options`` and ``eden train`` refuses them.
The trainer — role-separated actors initialized from PHC and a shared critic — is example-local:
``examples/projects/mimic/assistmimic/main.py``. AssistMimic's PSI (restarting episodes from states of good past
rollouts) is part of the env, in the ``motion`` command term (``psi_ratio=0.5``, training envs only).
"""

import math

from eden import observations, terminations
from eden.managers.terms.rewards.common import MetricReward
from eden.options import ObservationGroupOptions
from eden.options.entities import BoxOptions, PlaneOptions, SceneOptions
from eden.options.managers.rewards import RewardGroupOptions
from eden.options.robots.smpl_humanoid import PhcSmplHumanoid
from eden.tasks import TASK_REGISTRY
from eden.tasks.assistmimic.custom_terms.actions import PhcSphericalPDAction
from eden.tasks.assistmimic.custom_terms.commands import PARTNER, ROLES, AssistMimicPairMotion
from eden.tasks.assistmimic.custom_terms.metrics import (
    CaregiverHandReward,
    JointPower,
    PhcImitationReward,
    RecipientHeadHeight,
    TorqueReduction,
)
from eden.tasks.assistmimic.custom_terms.observations import (
    PartnerObservation,
    PartnerRelativeToWrists,
    PhcImitationObservation,
    PhcSelfObservation,
    RoleFlag,
    WristContact,
    WristForce,
)
from eden.tasks.assistmimic.custom_terms.terminations import PairTrackingFailure, ReferenceMotionEnd
from eden.tasks.assistmimic.robots import SMPL_DOF_NAMES, AssistMimicBedRecipient, AssistMimicRecipient
from eden.terms import DoneTerm, ObsTerm
from eden.utils.configs import (
    ActionManagerOptions,
    CommandManagerOptions,
    EdenRLConfig,
    EnvOptions,
    MetricManagerOptions,
    ObservationManagerOptions,
    RewardManagerOptions,
    TerminationManagerOptions,
)

#: The task's sample clips, under ``assistmimic/`` of the shared sample-data repo (``sample_data_path``).
INTERX_SAMPLE_CLIP = "interx_cluster1.npz"
HHI_ASSIST_SAMPLE_CLIP = "hhi_assist_aa_rm.npz"
#: Caregiver / recipient colors (orange / blue).
CAREGIVER_COLOR = (0.9, 0.55, 0.2)
RECIPIENT_COLOR = (0.25, 0.45, 0.85)
#: Leading part of every agent observation: PHC's self (358) + imitation (576) observation.
PHC_OBS_DIM = 358 + 576


def agent_obs_dim(env) -> int:
    """Size of one agent's observation vector (the two groups are identical in layout)."""
    return math.prod(env.observation_manager.group_obs_dim["caregiver"])


def _agent_observations(role: str, retarget_hands: bool = False) -> ObservationGroupOptions:
    """One agent's observation group. ``retarget_hands``: AssistMimic's Dynamic Reference Retargeting."""
    partner = PARTNER[role]
    return ObservationGroupOptions(
        concatenate_terms=True,
        enable_corruption=False,
        self_obs=PhcSelfObservation.configure(entity_name=role),
        imitation=PhcImitationObservation.configure(
            entity_name=role,
            role=role,
            command_name="motion",
            partner_entity_name=partner if retarget_hands else "",
        ),
        wrist_contact=WristContact.configure(entity_name=role, partner_entity_name=partner),
        wrist_force=WristForce.configure(entity_name=role),
        partner=PartnerObservation.configure(entity_name=role, partner_entity_name=partner),
        partner_to_wrists=PartnerRelativeToWrists.configure(entity_name=role, partner_entity_name=partner),
        partner_wrist_contact=WristContact.configure(entity_name=partner, partner_entity_name=role),
        prev_action=ObsTerm.configure(func=observations.last_action, params={"action_name": role}),
        role=RoleFlag.configure(value=float(ROLES.index(role))),
    )


def _metrics(torque_reduction: bool = False) -> MetricManagerOptions:
    """The reward components, computed once per step and weighted by both agents' reward groups.

    ``torque_reduction`` adds the recipient's total-torque metric that only the HHI-Assist task pays for.
    """
    extra = {}
    if torque_reduction:
        extra["torque_recipient"] = TorqueReduction.configure(entity_name="recipient", dofs_name=SMPL_DOF_NAMES)
    return MetricManagerOptions(
        imitation_caregiver=PhcImitationReward.configure(entity_name="caregiver", role="caregiver"),
        imitation_recipient=PhcImitationReward.configure(entity_name="recipient", role="recipient"),
        hand_caregiver=CaregiverHandReward.configure(entity_name="caregiver", partner_entity_name="recipient"),
        head_height=RecipientHeadHeight.configure(entity_name="recipient"),
        power_caregiver=JointPower.configure(entity_name="caregiver", dofs_name=SMPL_DOF_NAMES),
        power_recipient=JointPower.configure(entity_name="recipient", dofs_name=SMPL_DOF_NAMES),
        **extra,
    )


def _rewards(torque_reduction_weight: float | None = None) -> RewardManagerOptions:
    """Both agents' rewards (AssistMimic's ``_compute_reward`` + ``partner_reward_integration``).

    Per agent ``r_cg = im_cg + hand + head - (0.001 + 0.0005) P_cg`` and ``r_rc = im_rc + head - (0.001 + 0.001)
    P_rc`` (PHC's power term plus AssistMimic's role-specific one); then ``caregiver = 0.5 r_cg + 0.5 r_rc`` and
    ``recipient = r_rc``. ``torque_reduction_weight`` adds HHI-Assist's ``caregiver_torque_reduction_reward`` to
    ``r_cg``, so it is halved by the partner mix like the rest of ``r_cg``.
    """
    extra = {}
    if torque_reduction_weight is not None:
        extra["cg_torque_reduction"] = MetricReward.configure(
            metric_name="torque_recipient", weight=0.5 * torque_reduction_weight
        )
    return RewardManagerOptions(
        caregiver=RewardGroupOptions(
            cg_imitation=MetricReward.configure(metric_name="imitation_caregiver", weight=0.5),
            cg_hand=MetricReward.configure(metric_name="hand_caregiver", weight=0.5),
            cg_head_height=MetricReward.configure(metric_name="head_height", weight=1.0),
            cg_power=MetricReward.configure(metric_name="power_caregiver", weight=-0.5 * 0.0015),
            **extra,
            cg_partner_imitation=MetricReward.configure(metric_name="imitation_recipient", weight=0.5),
            cg_partner_power=MetricReward.configure(metric_name="power_recipient", weight=-0.5 * 0.002),
        ),
        recipient=RewardGroupOptions(
            rc_imitation=MetricReward.configure(metric_name="imitation_recipient", weight=1.0),
            rc_head_height=MetricReward.configure(metric_name="head_height", weight=1.0),
            rc_power=MetricReward.configure(metric_name="power_recipient", weight=-0.002),
        ),
    )


@TASK_REGISTRY.register(name="assistmimic")
class AssistMimicConfig(EdenRLConfig):
    """AssistMimic help-up on Inter-X with PHC's SMPL humanoid (no fingers)."""

    env_options: EnvOptions = EnvOptions(
        num_envs=1024,
        episode_length_s=10.0,  # episode_length 300 control steps at 30 Hz
        sim_dt=1.0 / 60.0,
        sim_substeps=2,
        decimation=2,
        # CG, not Newton: Newton's dense Cholesky is most of a step with two 24-body humanoids in contact. CG
        # halves the step (907 -> 442 ms at 8192 envs; Inter-X training 11.5 h against 21.3 h) and trains as well
        # (the example's README has the runs); a policy from either replays the same under the other.
        solver="cg",
        iterations=50,
        ls_iterations=20,
        max_collision_pairs=100,
        enable_self_collision=True,
        enable_multi_contact=True,
        env_spacing=(5.0, 5.0),
    )

    scene_options: SceneOptions = SceneOptions(
        plane=PlaneOptions(),
        caregiver=PhcSmplHumanoid(color=CAREGIVER_COLOR),
        recipient=AssistMimicRecipient(default_root_pos=(1.0, 0.0, 0.95), color=RECIPIENT_COLOR),
    )

    command_options: CommandManagerOptions = CommandManagerOptions(
        motion=AssistMimicPairMotion.configure(sample_clip=INTERX_SAMPLE_CLIP, psi_ratio=0.5),
    )

    action_options: ActionManagerOptions = ActionManagerOptions(
        caregiver=PhcSphericalPDAction.configure(entity_name="caregiver", dofs_name=SMPL_DOF_NAMES),
        recipient=PhcSphericalPDAction.configure(entity_name="recipient", dofs_name=SMPL_DOF_NAMES),
    )

    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        caregiver=_agent_observations("caregiver"),
        recipient=_agent_observations("recipient"),
    )

    metric_options: MetricManagerOptions = _metrics()
    reward_options: RewardManagerOptions = _rewards()

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        tracking_failure=PairTrackingFailure.configure(command_name="motion"),
        motion_end=ReferenceMotionEnd.configure(command_name="motion"),
        time_out=DoneTerm.configure(func=terminations.time_out, time_out=True),
    )


BED_SIZE = (1.905, 0.9906, 0.42)
BED_POS = (0.3, 0.19, 0.2286)


@TASK_REGISTRY.register(name="assistmimic_bed")
class AssistMimicBedConfig(AssistMimicConfig):
    """AssistMimic's HHI-Assist task: helping a recipient out of bed (``env_im_hhi-assist.yaml``)."""

    scene_options: SceneOptions = SceneOptions(
        plane=PlaneOptions(),
        caregiver=PhcSmplHumanoid(color=CAREGIVER_COLOR),
        # Only the pose before the first reset; every reset writes the reference pose over it.
        recipient=AssistMimicBedRecipient(default_root_pos=BED_POS[:2] + (0.65,), color=RECIPIENT_COLOR),
        bed=BoxOptions(fixed=True, size=BED_SIZE, default_root_pos=BED_POS, color=(0.8, 0.6, 0.4)),
    )

    command_options: CommandManagerOptions = CommandManagerOptions(
        motion=AssistMimicPairMotion.configure(
            sample_clip=HHI_ASSIST_SAMPLE_CLIP, bed_entity_name="bed", psi_ratio=0.5
        ),
    )

    # Only the caregiver retargets, and only here: ``enable_adjust_caregiver_hand_reference`` is on in
    # ``env_im_hhi-assist.yaml`` and off in ``env_im_interx_helpup.yaml``.
    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        caregiver=_agent_observations("caregiver", retarget_hands=True),
        recipient=_agent_observations("recipient"),
    )

    # The Inter-X metrics and rewards (actions and terminations are inherited unchanged) plus
    # ``caregiver_torque_reduction_reward`` (coefficient 0.5), which only this task turns on.
    metric_options: MetricManagerOptions = _metrics(torque_reduction=True)
    reward_options: RewardManagerOptions = _rewards(torque_reduction_weight=0.5)


#: ``--dataset`` of ``main.py`` and of ``scripts/convert_pair_motion.py``: one registered task each.
TASK_FOR_DATASET = {"inter-x": "assistmimic", "hhi-assist": "assistmimic_bed"}


def separate_pair(cfg: AssistMimicConfig, offset_y: float = 3.0) -> AssistMimicConfig:
    """Move the recipient — and whatever it is lying on — ``offset_y`` metres aside, out of contact range.

    Used by the PHC adaptation stage and by ``main.py play --apart``, which answers "how much of this
    behaviour is the caregiver's doing?" by making contact impossible.
    """
    cfg.command_options.motion.recipient_offset = (0.0, offset_y, 0.0)
    pos = cfg.scene_options.recipient.default_root_pos
    cfg.scene_options.recipient.default_root_pos = (pos[0], pos[1] + offset_y, pos[2])
    bed = getattr(cfg.scene_options, "bed", None)
    if bed is not None:
        x, y, z = bed.default_root_pos
        bed.default_root_pos = (x, y + offset_y, z)
    return cfg


def remove_caregiver(cfg: AssistMimicConfig, offset_y: float = 10.0) -> AssistMimicConfig:
    """Leave the recipient and its bed exactly where the clip puts them, and take the caregiver away.

    The ablation behind the whole task: if the recipient gets out of bed by itself, the caregiver is not
    assisting and every success number above means nothing. The caregiver entity stays in the scene — the
    policy's partner observations are part of its input and dropping it would change the observation size —
    but both it and its reference track move ``offset_y`` metres aside, so it can neither touch the recipient
    nor be touched. It is also excused from the pair's tracking-failure termination (its own clip has it
    walking to a bed that is no longer there), so the episode ends on the *recipient's* tracking alone and
    the recipient gets the full clip to try.
    """
    cfg.command_options.motion.caregiver_offset = (0.0, offset_y, 0.0)
    pos = cfg.scene_options.caregiver.default_root_pos
    cfg.scene_options.caregiver.default_root_pos = (pos[0], pos[1] + offset_y, pos[2])
    cfg.termination_options.tracking_failure.caregiver_threshold = 1.0e9
    return cfg


def strong_recipient(cfg: AssistMimicConfig) -> PhcSmplHumanoid:
    """A full-strength recipient at the weakened one's place: the diagnostic for "is it too weak to help?"."""
    weak = cfg.scene_options.recipient
    return PhcSmplHumanoid(default_root_pos=weak.default_root_pos, color=weak.color)


def phc_adaptation_config(cfg: AssistMimicConfig) -> AssistMimicConfig:
    """Stage 1: adapt PHC to Genesis on the same tracks, one humanoid per track, no interaction.

    AssistMimic does not start from the raw PHC either but from a PHC fine-tuned on Inter-X
    (``phc_interx_recipient_finetune2``). Here that fine-tune also absorbs the Isaac Gym -> Genesis gap: the
    humanoids are kept 3 m apart (``recipient_offset``), both at full strength, rewarded by PHC's imitation
    reward and power penalty only, and terminated on PHC's tracking distance, always from the clip start (no
    PSI). A bed moves with the recipient it supports — left behind, it would hold nothing up and drop the
    recipient 44 cm onto the floor.
    """
    separate_pair(cfg)
    cfg.command_options.motion.psi_ratio = 0.0
    cfg.scene_options.recipient = strong_recipient(cfg)
    power = 0.0005  # PHC ``power_coefficient``
    cfg.reward_options = RewardManagerOptions(
        caregiver=RewardGroupOptions(
            cg_imitation=MetricReward.configure(metric_name="imitation_caregiver", weight=1.0),
            cg_power=MetricReward.configure(metric_name="power_caregiver", weight=-power),
        ),
        recipient=RewardGroupOptions(
            rc_imitation=MetricReward.configure(metric_name="imitation_recipient", weight=1.0),
            rc_power=MetricReward.configure(metric_name="power_recipient", weight=-power),
        ),
    )
    return cfg
