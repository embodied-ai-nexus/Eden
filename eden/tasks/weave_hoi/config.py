"""Weave G1-Inspire-HOI: a G1 with two Inspire hands tracks human-object interaction clips, object in the loop.

Port of Weave's ``G1-Inspire-HOI-v0`` (github.com/xiaohu-art/Weave, Isaac Lab): one policy tracks OMOMO human-object
clips retargeted onto the G1 + Inspire RH56DFQ rig (:class:`~eden.options.robots.G1Inspire`), reproducing both the
whole-body motion and the object's trajectory. The reference and the objects come from the ``appolyn/Weave``
HuggingFace dataset through :class:`~eden.datasets.weave_dataset.WeaveG1InspireHOIDataset`.

What makes it more than a whole-body tracker, and where each piece lives:

* **The object is simulated and tracked.** It is its own motion-reference entity, reset from the reference with the
  robot (:class:`~eden.managers.terms.events.motion.ObjectReferenceStateInitialization`), rewarded and terminated on
  its world pose, and observed in the robot's pelvis frame
  (:func:`~eden.managers.terms.observations.tracking.motion_future_base_pos_b` /
  :func:`~eden.managers.terms.observations.tracking.entity_base_pos_b`).
* **Nine objects in one policy.** The object is a heterogeneous entity — one OMOMO mesh per env, in balanced env
  blocks — and ``MotionTermOptions.env_motion_groups`` restricts each env to its own object's clips.
* **Contact.** The reference carries a tri-state per-link contact label (``contact_label``), observed at five frames
  (now to 0.4 s ahead) and matched by :class:`~eden.tasks.weave_hoi.custom_terms.ObjectContactLabelReward`; losing a
  grasp the reference expects terminates (:class:`~eden.tasks.weave_hoi.custom_terms.LostObjectContact`).
* **Object geometry.** A basis-point SDF descriptor and fingertip-to-surface vectors, from Weave's precomputed
  per-object files (:mod:`eden.tasks.weave_hoi.custom_terms`).
* **Coupled fingers.** Each hand's six followers are driven in software from its six drivers (``mimic_dofs``), so the
  action is 41-D: 29 body joints + 2 x 6 finger drivers.
* **Learning.** SimBaV2 actor-critic trained by PPO with Muon (:mod:`eden.tasks.weave_hoi.runner`).

The MDP follows Weave's ``configs/track/train.yaml`` (which overrides its Python defaults); deviations forced by the
engine are noted where they are made. Build and train with ``eden train --task weave_hoi`` (all nine objects of the
test split, as Weave's training config does) or narrow it: ``--objects smallbox --split train``.
"""

from __future__ import annotations

from typing import Annotated

import eden as en
from eden import events, observations, rewards, terminations
from eden.constants import ReferenceSource
from eden.datasets.weave_dataset import WEAVE_OBJECTS, WeaveG1InspireHOIDataset, weave_object_dir
from eden.managers.modifiers.observations.noise import UniformNoise
from eden.managers.motion_manager import MotionTerm
from eden.managers.terms.actions.joint_actions import ImplicitPDController
from eden.managers.terms.events.motion import ObjectReferenceStateInitialization, ReferenceStateInitialization
from eden.managers.terms.observations.contact import GroupedContactFlags, motion_future_contact_label
from eden.managers.terms.observations.locomotion import ProjectedGravity
from eden.managers.terms.observations.tracking import (
    entity_base_ori_b,
    entity_base_pos_b,
    motion_future_base_ori_b,
    motion_future_base_pos_b,
    motion_future_dofs_pos,
    motion_future_links_ori_b,
    motion_future_links_pos_b,
    robot_body_ori_b,
    robot_body_pos_b,
)
from eden.managers.terms.rewards.tracking import (
    motion_base_orientation_tracking_exp,
    motion_base_position_tracking_exp,
    motion_link_orientation_tracking_exp,
    motion_link_position_tracking_exp,
)
from eden.managers.terms.terminations.tracking import (
    BadMotionLinkPosZOnly,
    bad_anchor_ori,
    bad_base_ori,
    motion_base_position_tracking_error_above,
    motion_link_position_tracking_error_above,
)
from eden.options import ObservationGroupOptions
from eden.options.entities import EntityOptions, GroupedEntityOptions, PlaneOptions
from eden.options.managers.motions import MotionGroupOptions, MotionManagerOptions
from eden.options.materials import RigidMaterialOptions
from eden.options.robots import G1Dofs29NoHands, G1Inspire
from eden.options.robots.composite.g1_inspire import G1_BARE_WRIST_FILE
from eden.options.robots.inspire_rh56 import INSPIRE_FINGERTIP_LINKS_NAME
from eden.tasks import TASK_REGISTRY
from eden.tasks.beyondmimic.custom_terms.metrics import (
    MotionErrorBodyPos,
    MotionErrorBodyRot,
    motion_error_anchor_pos,
    motion_error_anchor_rot,
)
from eden.tasks.registry import TaskMod
from eden.tasks.weave_hoi.custom_terms import (
    HandOppositionReward,
    LostObjectContact,
    ObjectBpsSdf,
    ObjectContactLabelReward,
    ObjectNearestSurfaceVector,
    WeaveFeetAirTime,
    WeaveFeetSlide,
    weave_error_object_pos,
    weave_error_object_rot,
)
from eden.tasks.weave_hoi.runner import RslRlWeaveRunnerOptions, weave_runner_options
from eden.terms import DoneTerm, MetricTerm, ObsTerm, RewardTerm
from eden.utils.configs import (
    ActionManagerOptions,
    EdenMimicConfig,
    EnvOptions,
    EventManagerOptions,
    MetricManagerOptions,
    ObservationManagerOptions,
    RewardManagerOptions,
    SceneOptions,
    TerminationManagerOptions,
)

#: Weave's ``G1_INSPIRE_CFG.init_state``: the default posture (also the action offset) and pelvis height. Every other
#: joint (waist, hip roll/yaw, ankle roll, shoulder yaw, wrists) rests at 0.
_WEAVE_DEFAULT_DOFS_POS = {
    "*_hip_pitch_joint": -0.312,
    "*_knee_joint": 0.669,
    "*_ankle_pitch_joint": -0.363,
    "*_elbow_joint": 0.6,
    "*_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
}


def _weave_rig() -> G1Inspire:
    """Weave's rig: the G1 + Inspire RH56DFQ composite standing in Weave's default posture.

    Built fresh for every scene: the posture is also the joint action's offset, so a rig taken from
    ``G1Inspire()`` instead would shift every joint target the policy commands.
    """
    return G1Inspire(
        robot=G1Dofs29NoHands(
            file=G1_BARE_WRIST_FILE, default_root_pos=(0.0, 0.0, 0.76), default_dofs_pos=_WEAVE_DEFAULT_DOFS_POS
        )
    )


_RIG = _weave_rig()
_ANCHOR = "pelvis"
_HANDS = ("left_hand", "right_hand")

#: Weave's tracked bodies (``MotionCommandCfg.body_names``).
_TRACKED_LINKS = [
    "left_hip_pitch_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
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
_FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]
#: Future reference frames, in 50 Hz control steps (0 to 0.4 s ahead).
_FUTURE = [0, 5, 10, 15, 20]


#: Weave's contact bodies: every hand link past the proximals (thumb intermediate + distal, finger intermediates).
_CONTACT_LINKS = {
    hand: [
        "thumb_intermediate",
        "thumb_distal",
        "index_intermediate",
        "middle_intermediate",
        "ring_intermediate",
        "pinky_intermediate",
    ]
    for hand in _HANDS
}
#: Digit tips, thumb first: the nearest-surface queries and the opposition reward.
_TIP_LINKS = {hand: list(INSPIRE_FINGERTIP_LINKS_NAME) for hand in _HANDS}
#: Every motion link per entity — the contact-flag rows (Weave observes all 54 bodies).
_ALL_LINKS = {key: list(WeaveG1InspireHOIDataset.motion_links_name[key]) for key in ("robot", *_HANDS)}

#: Weave's ``max_depenetration_velocity`` on both the robot and the object rigid bodies, m/s.
_MAX_DEPENETRATION_VELOCITY = 1.0

_PUSH_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


def _density(obj: str) -> float:
    """Weave's object density, kg/m^3 (``largebox`` is the light one, "~1 kg")."""
    return 25.0 if obj == "largebox" else 200.0


def _ground() -> PlaneOptions:
    """The ground, authored below every randomized friction so each contact with it takes the entity's own value.

    Genesis resolves a contact pair's friction as ``max(a, b)``; at Weave's 1.0 the plane would win against every
    sampled value below it (most of the robot range, all of the object's).
    """
    return PlaneOptions(default_geoms_friction={"*": 0.01})


def _object_options(obj: str, geometry_root: str | None) -> EntityOptions:
    """One OMOMO object, at its mesh origin (``align=False``), with Weave's density.

    Named by its path inside the ``appolyn/Weave`` repo rather than a resolved local path, so a saved config replays
    on another machine (the asset resolver fetches it there); an explicit ``geometry_root`` is used as given.
    """
    location = (
        dict(file=f"{weave_object_dir(obj, geometry_root)}/{obj}.obj")
        if geometry_root is not None
        else dict(
            file=f"objects/{obj}/{obj}.obj",
            registry=WeaveG1InspireHOIDataset.registry,
            dataset=WeaveG1InspireHOIDataset.dataset,
        )
    )
    return EntityOptions(**location, align=False, material=RigidMaterialOptions(rho=_density(obj)))


class WeaveHOIDataMod(TaskMod):
    """Select the Weave split and objects, and build everything that depends on them.

    Object-dependent pieces are the scene's object bank, the motion term (dataset, per-object clip groups) and the
    object-geometry terms, so they are all set here rather than as class defaults. Override via CLI::

        eden train --task weave_hoi --objects smallbox --split train
    """

    def __init__(
        self,
        objects: Annotated[
            str,
            "Comma-separated Weave objects to train on, or 'all'. One object gives a single-object specialist; "
            f"several share one policy, each env holding one object. Choices: {', '.join(WEAVE_OBJECTS)}.",
        ] = "all",
        split: Annotated[str, "Weave reference split: 'test' (1,605 clips, Weave's train.yaml) or 'train' (7,869)."] = (
            "test"
        ),
        raw_data_path: Annotated[
            str | None,
            "Root of an appolyn/Weave snapshot (holding reference/ and objects/). Omit to download the selected "
            "split and objects into the HuggingFace cache.",
        ] = None,
        processed_data_path: Annotated[
            str | None, "Processed-motion cache .pt. Omit to read the npz files directly on every build."
        ] = None,
    ):
        self.objects = objects
        self.split = split
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path

    def _object_list(self) -> list[str]:
        if self.objects.strip() == "all":
            return list(WEAVE_OBJECTS)
        names = [o.strip() for o in self.objects.split(",") if o.strip()]
        unknown = [o for o in names if o not in WEAVE_OBJECTS]
        if unknown or not names:
            raise ValueError(
                f"--objects {self.objects!r} names unknown object(s) {unknown}; choose from {list(WEAVE_OBJECTS)} "
                f"(comma-separated) or 'all'."
            )
        return list(dict.fromkeys(names))

    def apply(self, config: WeaveHOIConfig) -> WeaveHOIConfig:
        if self.split not in ("train", "test"):
            raise ValueError(f"--split must be 'train' or 'test', got {self.split!r}.")
        objects = self._object_list()
        root = self.raw_data_path

        if len(objects) == 1:
            obj_opts = _object_options(objects[0], root)
            obj_opts.is_fixed_base = False
        else:
            # Each variant keeps its own density (largebox is the light one), applied per env after the build.
            obj_opts = GroupedEntityOptions(
                grouped_entities=[_object_options(o, root) for o in objects],
                is_fixed_base=False,
            )
        config.scene_options = en.scenes.Empty(robot=_weave_rig(), plane=_ground(), object=obj_opts)

        config.motion_options = MotionManagerOptions(
            num_frames=max(_FUTURE) + 1,
            data_start_from="current_time",
            frame_interval_s_range=(0.02, 0.02),
            # Weave rolls a finished clip over to a fresh one inside the episode, without a done signal.
            resync_on_motion_end=True,
            default=MotionTerm.configure(
                dataset_options=WeaveG1InspireHOIDataset.configure(
                    split=self.split,
                    objects=objects,
                    raw_data_path=self.raw_data_path,
                    processed_data_path=self.processed_data_path,
                ),
                entities_map={
                    "robot": "robot",
                    "left_hand": "left_hand",
                    "right_hand": "right_hand",
                    "object": "object",
                },
                target_dofs_name={
                    "robot": _RIG.robot_dofs,
                    "left_hand": _RIG.left_hand_dofs,
                    "right_hand": _RIG.right_hand_dofs,
                    "object": [],
                },
                target_links_name={**_ALL_LINKS, "object": []},
                motion_start_from_middle_range=(0.0, 1.0),
                env_motion_groups=(
                    MotionGroupOptions(entity_name="object", metadata_key="objects", variant_labels=objects)
                    if len(objects) > 1
                    else None
                ),
            ),
        )

        geometry = dict(objects=objects, geometry_root=root)
        config.observation_options = _observation_options(geometry)
        config.reward_options = _reward_options(geometry)

        # A run on other objects logs apart, so `--resume latest` never continues one trained on a different set.
        if objects != list(WEAVE_OBJECTS):
            config.runner_options.experiment_name = f"weave_hoi_{'-'.join(objects)}"
        return config


def _future_joint_pos(entity: str) -> ObsTerm:
    return ObsTerm.configure(func=motion_future_dofs_pos, params={"entity_name": entity, "frame_offsets": _FUTURE})


def _future_contact_label(entity: str) -> ObsTerm:
    return ObsTerm.configure(func=motion_future_contact_label, params={"entity_name": entity, "frame_offsets": _FUTURE})


def _contact_flags(entity: str) -> ObsTerm:
    return GroupedContactFlags.configure(
        entity_name=entity, object_entity_name="object", links_name=_ALL_LINKS[entity], force_threshold=1.0
    )


def _proprio(noisy: bool) -> dict:
    """Weave's ``robot_proprio`` (noisy) / the noise-free core of ``robot_privileged``."""

    def noise(n: float):
        return {"noise": UniformNoise.configure(n_min=-n, n_max=n)} if noisy else {}

    terms = dict(
        base_ang_vel=ObsTerm.configure(
            func=observations.base_ang_vel, params={"entity_name": "robot", "frame": "body"}, **noise(0.2)
        ),
    )
    for entity in ("robot", *_HANDS):
        terms[f"{entity}_dofs_pos"] = observations.DofsPos.configure(
            entity_name=entity, offset_from_default=True, **noise(0.01)
        )
    for entity in ("robot", *_HANDS):
        terms[f"{entity}_dofs_vel"] = observations.DofsVel.configure(entity_name=entity, **noise(0.5))
    terms["projected_gravity"] = ProjectedGravity.configure(entity_name="robot", **noise(0.05))
    terms["last_action"] = ObsTerm.configure(func=observations.last_action)
    return terms


_REF_MOTION_BODY = dict(
    robot_future_dofs_pos=_future_joint_pos("robot"),
    left_hand_future_dofs_pos=_future_joint_pos("left_hand"),
    right_hand_future_dofs_pos=_future_joint_pos("right_hand"),
    future_anchor_pos_b=ObsTerm.configure(
        func=motion_future_links_pos_b,
        params={"entity_name": "robot", "links_name": [_ANCHOR], "anchor_link_name": _ANCHOR, "frame_offsets": _FUTURE},
    ),
    future_anchor_ori_b=ObsTerm.configure(
        func=motion_future_links_ori_b,
        params={"entity_name": "robot", "links_name": [_ANCHOR], "anchor_link_name": _ANCHOR, "frame_offsets": _FUTURE},
    ),
)
_REF_MOTION_OBJECT = dict(
    future_object_pos_b=ObsTerm.configure(
        func=motion_future_base_pos_b,
        params={
            "entity_name": "object",
            "anchor_entity_name": "robot",
            "anchor_link_name": _ANCHOR,
            "frame_offsets": _FUTURE,
        },
    ),
    future_object_ori_b=ObsTerm.configure(
        func=motion_future_base_ori_b,
        params={
            "entity_name": "object",
            "anchor_entity_name": "robot",
            "anchor_link_name": _ANCHOR,
            "frame_offsets": _FUTURE,
        },
    ),
    robot_future_contact_label=_future_contact_label("robot"),
    left_hand_future_contact_label=_future_contact_label("left_hand"),
    right_hand_future_contact_label=_future_contact_label("right_hand"),
)


def _object_state(geometry: dict | None) -> dict:
    """Weave's ``object_state`` group; the two geometry terms need the object list, so only a built task has them."""
    terms = dict(
        object_pos_b=ObsTerm.configure(
            func=entity_base_pos_b,
            params={"entity_name": "object", "anchor_entity_name": "robot", "anchor_link_name": _ANCHOR},
        ),
        object_ori_b=ObsTerm.configure(
            func=entity_base_ori_b,
            params={"entity_name": "object", "anchor_entity_name": "robot", "anchor_link_name": _ANCHOR},
        ),
    )
    if geometry is not None:
        terms["object_nearest_point"] = ObjectNearestSurfaceVector.configure(
            object_entity_name="object",
            anchor_entity_name="robot",
            anchor_link_name=_ANCHOR,
            links=_TIP_LINKS,
            **geometry,
        )
        terms["object_bps_sdf"] = ObjectBpsSdf.configure(
            object_entity_name="object", anchor_entity_name="robot", anchor_link_name=_ANCHOR, **geometry
        )
    for entity in ("robot", *_HANDS):
        terms[f"{entity}_object_contact"] = _contact_flags(entity)
    return terms


def _observation_options(geometry: dict | None) -> ObservationManagerOptions:
    """The five Weave observation groups (policy = the first four, critic = the first three + privileged)."""

    def group(corrupt: bool, **terms) -> ObservationGroupOptions:
        return ObservationGroupOptions(concatenate_terms=True, enable_corruption=corrupt, **terms)

    return ObservationManagerOptions(
        ref_motion_body=group(False, **_REF_MOTION_BODY),
        ref_motion_object=group(False, **_REF_MOTION_OBJECT),
        object_state=group(False, **_object_state(geometry)),
        robot_proprio=group(True, **_proprio(True)),
        robot_privileged=group(
            False,
            **_proprio(False),
            base_lin_vel=ObsTerm.configure(
                func=observations.base_lin_vel, params={"entity_name": "robot", "frame": "body"}
            ),
            body_pos_b=ObsTerm.configure(
                func=robot_body_pos_b,
                params={"entity_name": "robot", "anchor_link_name": _ANCHOR, "tracked_link_names": _TRACKED_LINKS},
            ),
            body_ori_b=ObsTerm.configure(
                func=robot_body_ori_b,
                params={"entity_name": "robot", "anchor_link_name": _ANCHOR, "tracked_link_names": _TRACKED_LINKS},
            ),
        ),
    )


def _reward_options(geometry: dict | None) -> RewardManagerOptions:
    """Weave's ``train.yaml`` rewards; ``hand_opposition`` needs the object list, so only a built task has it."""
    extra = {}
    if geometry is not None:
        extra["hand_opposition"] = HandOppositionReward.configure(
            weight=2.0, object_entity_name="object", hands=_TIP_LINKS, **geometry
        )
    return RewardManagerOptions(
        motion_anchor_pos=RewardTerm.configure(
            func=motion_link_position_tracking_exp,
            weight=1.0,
            params={"entity_name": "robot", "links_name": [_ANCHOR], "std": 0.3},
        ),
        motion_anchor_ori=RewardTerm.configure(
            func=motion_link_orientation_tracking_exp,
            weight=1.0,
            params={"entity_name": "robot", "links_name": [_ANCHOR], "std": 0.4},
        ),
        # World frame, one exp over the mean squared error of the 13 bodies (Weave's motion_body_pos/ori).
        motion_body_pos=RewardTerm.configure(
            func=motion_link_position_tracking_exp,
            weight=1.0,
            params={"entity_name": "robot", "links_name": _TRACKED_LINKS, "std": 0.3, "exp_first": False},
        ),
        motion_body_ori=RewardTerm.configure(
            func=motion_link_orientation_tracking_exp,
            weight=1.0,
            params={"entity_name": "robot", "links_name": _TRACKED_LINKS, "std": 0.4, "exp_first": False},
        ),
        object_pos=RewardTerm.configure(
            func=motion_base_position_tracking_exp, weight=2.0, params={"entity_name": "object", "std": 0.2}
        ),
        object_ori=RewardTerm.configure(
            func=motion_base_orientation_tracking_exp, weight=2.0, params={"entity_name": "object", "std": 0.3}
        ),
        **extra,
        contact=ObjectContactLabelReward.configure(
            weight=2.0, object_entity_name="object", links=_CONTACT_LINKS, saturate_force=5.0
        ),
        feet_air_time=WeaveFeetAirTime.configure(weight=0.5, entity_name="robot", feet_link_names=_FEET, threshold=0.3),
        feet_slide=WeaveFeetSlide.configure(
            weight=-1.0, entity_name="robot", feet_link_names=_FEET, force_threshold=0.5
        ),
        action_rate_l2=RewardTerm.configure(func=rewards.action_rate_l2, weight=-0.1),
        # The body entity only: Weave excludes the fingers from its joint-limit penalty.
        joint_limit=RewardTerm.configure(func=rewards.dofs_pos_limits, weight=-10.0, params={"entity_name": "robot"}),
    )


@TASK_REGISTRY.register(modifiers=(WeaveHOIDataMod(),))
class WeaveHOIConfig(EdenMimicConfig):
    """Weave G1-Inspire-HOI: G1 + two Inspire RH56DFQ hands tracking OMOMO human-object clips, object in the loop.

    ``WeaveHOIDataMod`` (a default modifier) sets the split, the objects and everything that depends on them;
    construct via ``TASK_REGISTRY.build("weave_hoi", objects="smallbox")`` or ``eden train --task weave_hoi``.
    """

    scene_options: SceneOptions = en.scenes.Empty(robot=_weave_rig(), plane=_ground())

    env_options: EnvOptions = EnvOptions(
        num_envs=4096,
        episode_length_s=20.0,
        sim_dt=0.005,
        sim_substeps=1,
        decimation=4,
        max_collision_pairs=60,
        batch_dofs_info=True,  # finger gain randomization writes per-env kp/kd
        batch_links_info=True,  # torso COM randomization writes per-env link COMs
        enable_self_collision=False,
        enable_multi_contact=True,
    )

    # Both are rebuilt by WeaveHOIDataMod with the object-geometry terms, which need the object list.
    observation_options: ObservationManagerOptions = _observation_options(None)
    reward_options: RewardManagerOptions = _reward_options(None)

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        time_out=DoneTerm.configure(func=terminations.time_out, time_out=True),
        anchor_pos=DoneTerm.configure(
            func=motion_link_position_tracking_error_above,
            params={"entity_name": "robot", "links_name": [_ANCHOR], "threshold": 0.25},
        ),
        anchor_ori=DoneTerm.configure(
            func=bad_anchor_ori, params={"entity_name": "robot", "anchor_link_name": _ANCHOR, "threshold": 0.8}
        ),
        object_pos=DoneTerm.configure(
            func=motion_base_position_tracking_error_above, params={"entity_name": "object", "threshold": 0.3}
        ),
        object_ori=DoneTerm.configure(func=bad_base_ori, params={"entity_name": "object", "threshold": 0.3}),
        ee_body_pos=BadMotionLinkPosZOnly.configure(
            entity_name="robot",
            anchor_link_name=_ANCHOR,
            threshold=0.25,
            links_name=[*_FEET, "left_wrist_yaw_link", "right_wrist_yaw_link"],
        ),
        lost_contact=LostObjectContact.configure(
            object_entity_name="object", links=_CONTACT_LINKS, max_lost_steps=10, force_threshold=1.0
        ),
    )

    event_options: EventManagerOptions = EventManagerOptions(
        # Weave resets straight onto the reference: no pose, velocity or joint noise.
        # The reference overlaps the simulated geometry (feet in the floor, fingers in the object); Weave's PhysX
        # bodies separate at max_depenetration_velocity=1.0, so cap the seeded bases the same way.
        reset_robot_from_motion=ReferenceStateInitialization.configure(
            mode=en.EventMode.RESET,
            entity_name="robot",
            dofs_position_range=(0.0, 0.0),
            max_depenetration_velocity=_MAX_DEPENETRATION_VELOCITY,
            priority=101,
        ),
        reset_left_hand_from_motion=ReferenceStateInitialization.configure(
            mode=en.EventMode.RESET, entity_name="left_hand", dofs_position_range=(0.0, 0.0), priority=100
        ),
        reset_right_hand_from_motion=ReferenceStateInitialization.configure(
            mode=en.EventMode.RESET, entity_name="right_hand", dofs_position_range=(0.0, 0.0), priority=100
        ),
        reset_object_from_motion=ObjectReferenceStateInitialization.configure(
            mode=en.EventMode.RESET,
            entity_name="object",
            xy_range=0.0,
            max_depenetration_velocity=_MAX_DEPENETRATION_VELOCITY,
            priority=99,
        ),
        # Friction: PhysX averages a contact pair's coefficients and Genesis takes max(a, b). So the ranges below are
        # Weave's static ranges averaged against its 1.0 ground (robot (0.3, 1.6) -> (0.65, 1.3), object (0.4, 1.2) ->
        # (0.7, 1.1)), and the ground itself is authored at 0.01 (`_ground`) so these values, not the ground's, decide
        # every ground contact. Hand-object contacts take the larger of the two here, where Weave averages them.
        robot_friction=events.RandomizeFrictionRatio.configure(
            mode=en.EventMode.STARTUP, entity_name="robot", links_name=["*"], friction_range=(0.65, 1.3)
        ),
        left_hand_friction=events.RandomizeFrictionRatio.configure(
            mode=en.EventMode.STARTUP, entity_name="left_hand", links_name=["*"], friction_range=(0.65, 1.3)
        ),
        right_hand_friction=events.RandomizeFrictionRatio.configure(
            mode=en.EventMode.STARTUP, entity_name="right_hand", links_name=["*"], friction_range=(0.65, 1.3)
        ),
        object_friction=events.RandomizeFrictionRatio.configure(
            mode=en.EventMode.STARTUP, entity_name="object", links_name=["*"], friction_range=(0.7, 1.1)
        ),
        base_com=events.RandomizeComShift.configure(
            mode=en.EventMode.STARTUP,
            entity_name="robot",
            links_name=["torso_link"],
            com_shift_x_range=(-0.025, 0.025),
            com_shift_y_range=(-0.05, 0.05),
            com_shift_z_range=(-0.05, 0.05),
        ),
        # Weave scales finger stiffness/damping log-uniformly in (0.5, 2.0); this draws uniformly on that range.
        left_finger_gains=events.RandomizeKpKdGains.configure(
            mode=en.EventMode.STARTUP, entity_name="left_hand", kp_range=(0.5, 2.0), kd_range=(0.5, 2.0)
        ),
        right_finger_gains=events.RandomizeKpKdGains.configure(
            mode=en.EventMode.STARTUP, entity_name="right_hand", kp_range=(0.5, 2.0), kd_range=(0.5, 2.0)
        ),
        push_robot=events.PushByVelocity.configure(
            mode=en.EventMode.INTERVAL, interval_range_s=(2.0, 4.0), entity_name="robot", velocity_range=_PUSH_RANGE
        ),
    )

    action_options: ActionManagerOptions = ActionManagerOptions(
        body_dofs_pos=ImplicitPDController.configure(
            entity_name="robot", dofs_name=_RIG.robot_dofs, scale=1.0, reference_source=ReferenceSource.DEFAULT
        ),
        left_hand_dofs_pos=ImplicitPDController.configure(
            entity_name="left_hand",
            dofs_name=_RIG.left_hand_actuated_dofs,
            mimic_dofs=_RIG.left_hand_mimic_dofs,
            scale=1.0,
            reference_source=ReferenceSource.DEFAULT,
        ),
        right_hand_dofs_pos=ImplicitPDController.configure(
            entity_name="right_hand",
            dofs_name=_RIG.right_hand_actuated_dofs,
            mimic_dofs=_RIG.right_hand_mimic_dofs,
            scale=1.0,
            reference_source=ReferenceSource.DEFAULT,
        ),
    )

    metric_options: MetricManagerOptions = MetricManagerOptions(
        error_anchor_pos=MetricTerm.configure(
            func=motion_error_anchor_pos, params={"entity_name": "robot", "anchor_link_name": _ANCHOR}
        ),
        error_anchor_rot=MetricTerm.configure(
            func=motion_error_anchor_rot, params={"entity_name": "robot", "anchor_link_name": _ANCHOR}
        ),
        error_body_pos=MotionErrorBodyPos.configure(
            entity_name="robot", anchor_link_name=_ANCHOR, links_name=_TRACKED_LINKS
        ),
        error_body_rot=MotionErrorBodyRot.configure(
            entity_name="robot", anchor_link_name=_ANCHOR, links_name=_TRACKED_LINKS
        ),
        error_object_pos=MetricTerm.configure(func=weave_error_object_pos, params={"entity_name": "object"}),
        error_object_rot=MetricTerm.configure(func=weave_error_object_rot, params={"entity_name": "object"}),
    )

    runner_options: RslRlWeaveRunnerOptions = weave_runner_options(experiment_name="weave_hoi")
