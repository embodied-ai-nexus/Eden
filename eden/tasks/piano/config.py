"""Piano: two Wuji Hand2 hands play a song on a simulated 88-key keyboard.

Eden's take on `RoboPianist <https://github.com/google-research/robopianist>`_'s ``PianoWithShadowHands``
(Apache-2.0), with the Wuji Hand2 (beta2) in place of the Shadow Hand. The score is given to the policy as the key
states of the current control step and the next ten; the reward pays for pressing the keys that are due, and
nothing else, at the moment they are due.

    eden train --task piano --song twinkle_twinkle
    eden train --task piano --song chpn-p7.mid      # any MIDI file in `songs.SCORES_DIR`
    eden inference --task piano --song twinkle_twinkle

What is the same as RoboPianist: the keyboard (dimensions, key springs, and the "within half a degree of the joint
limit" test for a pressed key), the 20 Hz control rate over 200 Hz physics, the ten-step half-second lookahead it
trains with, hands whose weight is compensated, the form of the key-press and energy rewards and the equal weight
on the key-press and shaping terms, and precision / recall / F1 over the keys — scored per step, counting a step
with nothing due and nothing down as perfect, then averaged over the song.

What differs, and why. Most of it follows from the learner: RoboPianist trains one environment off-policy (DroQ,
5M steps), this trains 1024 of them on-policy (PPO, ~100M), which needs a slope to climb and cannot afford a
penalty that outprices a note.

* **The hand flies.** The Shadow Hand rides a forearm with two actuated slides, along the keyboard and up, and so
  cannot leave the keys; constraining it that way is one of RoboPianist's stated human priors. The Wuji Hand2 ends
  at the wrist, so each hand here is a free body whose pose is driven in all six axes by
  :class:`~eden.managers.terms.actions.FreeBasePoseController`, its weight carried by the controller the way an
  arm's would carry it. That is 26 axes a hand — 20 finger joints and the wrist's six — against RoboPianist's 19:
  17 hand actuators, it having disabled three and clamped the thumb as its other human prior, and the two slides.
* **Actions are offsets from a rest pose**, scaled per joint, not RoboPianist's absolute targets over each joint's
  whole range. The rest pose is a hand curled over the keys, so a zero action is already a reasonable place to be
  and PPO's initial exploration noise moves fingers rather than throwing the hand across the keyboard.
* **The key-press reward falls off half as fast** (Gaussian margin 1.0 of the stroke, not 0.5), so that a partly
  pressed key is worth something. Its other half is RoboPianist's unchanged: all of it, lost the moment any one
  key that should be up is down.
* **The energy penalty is ten times lighter** (5e-4 per watt). RoboPianist's 5e-3 is sized for the Shadow Hand's
  actuators; with this hand's stiff blanket gains it outweighs everything a note can earn.
* **Two of RoboPianist's five reward terms are gone.** One pays for matching the score's pedal marks: the sustain
  pedal is an action there, a software latch rather than a mechanism, and no score here carries a pedal mark at
  all. The other is a bonus for the two forearms not colliding, and these hands have no forearm.
* **Two are new**, both about how a hand carries itself, which the key-press reward is wholly indifferent to. A
  rest-pose penalty: without it a policy that played a scale perfectly did so with its idle hand splayed and its
  fingers bent back. And the usual action-rate penalty.
* **No fingering annotations**, which come from the PIG dataset RoboPianist trains on and cannot be
  redistributed. Its shaping reward pulls the labelled finger toward each due key; this is the annotation-free
  fallback RoboPianist itself now carries (from RP1M) — pull whichever fingertip is nearest — in a batched
  nearest-tip form rather than a Hungarian assignment, and it looks a few steps ahead so that a finger is over a
  key before it is due.
* **The built-in songs are written out by hand** (:mod:`eden.tasks.piano.songs`) for the same reason, and an
  episode opens with a second of silence for the hands to settle over the first notes (RoboPianist starts on
  them). Real repertoire is a MIDI file away: ``--song <file>.mid`` reads a score
  (:mod:`eden.tasks.piano.midi`) — by filename alone if it sits in :data:`~eden.tasks.piano.songs.SCORES_DIR`,
  which is the form that keeps the saved config portable — and where the file says which hand plays what, each
  hand is placed over its own part.

The keyboard makes no sound here. To hear a policy, add a ``PianoAudio`` sensor and a microphone to the scene at
evaluation time: the voice reads only the key joints, so it needs nothing from this task.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated

import numpy as np
from genesis.utils.geom import axis_angle_to_quat, transform_quat_by_quat

from eden import observations, rewards
from eden.managers.terms.actions import FreeBasePoseController, ImplicitPDController
from eden.options import ObservationGroupOptions
from eden.options.entities import PlaneOptions, SceneOptions
from eden.options.learning.rsl_rl import (
    RslRlActorOptions,
    RslRlCriticOptions,
    RslRlGaussianDistributionOptions,
    RslRlOnPolicyRunnerOptions,
    RslRlPpoAlgorithmOptions,
)
from eden.options.materials import RigidMaterialOptions
from eden.options.objects.piano import Piano, key_of_midi_note, key_y
from eden.options.robots import WujiHand2Beta2_L, WujiHand2Beta2_R
from eden.tasks.piano import custom_terms as ct
from eden.tasks.piano.songs import SCORES_DIR, SONGS, Song, get_song, song_steps
from eden.tasks.registry import TASK_REGISTRY, TaskMod
from eden.terms import DoneTerm, MetricTerm, ObsTerm, RewardTerm
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

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

# RoboPianist's rates: 200 Hz physics, 20 Hz control.
_SIM_DT = 0.005
_DECIMATION = 10
_STEP_DT = _SIM_DT * _DECIMATION
_LEAD_IN_S = 1.0
_DEFAULT_SONG = "twinkle_twinkle"

# ---------------------------------------------------------------------------
# The rig. The keyboard sits at the origin with its keys running along y and the player on the +x side, low notes
# to the player's left (-y).
# ---------------------------------------------------------------------------

HANDS = ("right", "left")

# Keyboard height: about where a piano's keys are. It must clear the floor in any case, since a pressed key
# dips below the keyboard's underside.
KEYBOARD_Z = 0.7
# Wrist position over the keyboard, from the keys' centreline and from the keyboard's underside.
_WRIST_X, _WRIST_Z = 0.145, KEYBOARD_Z + 0.10
# Where each wrist rests along the keyboard: right hand on C4 to G4 with the thumb on middle C, left hand on C3
# to G3 with the little finger on C3. A hand-written tune starts there; a score that says which hand plays what
# is placed over its own notes instead (`hand_placement`).
_WRIST_Y = {"right": -0.027, "left": -0.160}
# The five keys the rest pose was fitted over, as MIDI notes, thumb and little finger at the ends.
_REST_NOTES = {"right": (60, 67), "left": (48, 55)}

# The hand loads fingers-up, palm-forward (see `WujiHand2Beta2_R`). Half a turn about (1, 0, -1) lays it fingers
# toward the keyboard (-x) and palm down. It is then rolled about the forearm's axis, thumb side down, the way a
# pianist's forearm sits short of fully pronated: this thumb swings under the palm rather than down beside it, and
# with the palm dead level its tip cannot get below the key tops anywhere beside the index finger.
_HAND_ROLL = math.radians(15.0)


def hand_rest_quat(side: str) -> tuple[float, float, float, float]:
    """Orientation a hand rests over the keyboard in, wxyz.

    Parameters
    ----------
    side : str
        ``"right"`` or ``"left"``.

    Returns
    -------
    tuple[float, float, float, float]
        Palm down and fingers toward the keyboard, then rolled thumb-side down.
    """
    # The right thumb is on the -y side, the left on +y, so the two roll opposite ways.
    roll = axis_angle_to_quat(np.array(_HAND_ROLL if side == "right" else -_HAND_ROLL), np.array([1.0, 0.0, 0.0]))
    palm_down = axis_angle_to_quat(np.array(math.pi), np.array([1.0, 0.0, -1.0]))
    # transform_quat_by_quat(v, u) is u * v: the roll about x applied after the palm-down half turn.
    return tuple(float(c) for c in transform_quat_by_quat(palm_down, roll))


# Rest pose: each fingertip a centimetre above the front half of its own white key, thumb to little finger on five
# neighbouring keys. Fitted by sampling the mounted hand's forward kinematics; both hands fit the same angles to
# within 0.1 rad, so they share them. The outer fingers flex more at the knuckle because the roll lifts them.
_REST = {
    "thumb": {"cmc_flex": 0.35, "cmc_abd": -0.17, "mcp": 0.80, "ip": 0.0},
    "index_finger": {"mcp_flex": 0.38, "mcp_abd": -0.08, "pip": 1.15, "dip": 0.80},
    "middle_finger": {"mcp_flex": 0.56, "mcp_abd": 0.02, "pip": 0.95, "dip": 0.70},
    "ring_finger": {"mcp_flex": 0.56, "mcp_abd": 0.0, "pip": 0.90, "dip": 0.55},
    "pinky": {"mcp_flex": 0.72, "mcp_abd": 0.12, "pip": 0.45, "dip": 0.26},
}

# Action scales: how far a unit action moves each joint's target from rest.
_HAND_ACTION_SCALE = 0.5  # rad
# And how far it moves the wrist itself, in world axes: along the keys, toward the keyboard and above it, then the
# three turns about the same axes — pronation, wrist flexion, and the deviation that swings the hand along the keys.
_WRIST_POSITION_SCALE = (0.04, 0.10, 0.03)  # m
_WRIST_ORIENTATION_SCALE = (0.15, 0.15, 0.30)  # rad
# Stiff enough to hold a hand against a key's return force with a millimetre or two of give, and to settle a hand's
# width along the keyboard inside three control steps.
_WRIST_GAINS = {
    "position_kp": 2000.0,
    "position_kd": 80.0,
    "orientation_kp": 40.0,
    "orientation_kd": 2.0,
    "force_limit": 100.0,
    "torque_limit": 10.0,
}

_NUM_ENVS = 1024
_MAX_ITERATIONS = 3000
_NUM_STEPS_PER_ENV = 32


def _hand_options(side: str):
    return WujiHand2Beta2_R if side == "right" else WujiHand2Beta2_L


def _keys_centre(low_note: int, high_note: int) -> float:
    """Halfway along the keyboard between two notes' keys.

    Parameters
    ----------
    low_note : int
        MIDI note at one end.
    high_note : int
        MIDI note at the other.

    Returns
    -------
    float
        Offset along ``y`` from the keyboard's origin, in metres.
    """
    return 0.5 * (key_y(key_of_midi_note(low_note)) + key_y(key_of_midi_note(high_note)))


# How far the wrist sits from the middle of the five keys its fingers rest over. It leans toward the thumb, and
# the thumbs face each other, so the two hands lean opposite ways. Taken from the rests above rather than
# measured again, so that a score asking for exactly those keys reproduces them.
_WRIST_LEAN = {side: _WRIST_Y[side] - _keys_centre(*_REST_NOTES[side]) for side in HANDS}


def hand_placement(song: Song, side: str) -> tuple[float, float]:
    """Where a hand rests for a score, and how far one unit of action slides it along the keys.

    A score that says which hand plays each note (a MIDI file whose two staves are named) places the hand in the
    middle of its own part and gives it enough travel to reach either end of it. A hand the score says nothing
    about — every hand-written tune here — keeps the rest it was fitted at, and the default travel.

    Parameters
    ----------
    song : Song
        The score.
    side : str
        ``"right"`` or ``"left"``.

    Returns
    -------
    tuple[float, float]
        The wrist's ``y`` and the metres per unit of its ``y`` action.
    """
    reach = song.reach(side)
    if reach is None:
        return _WRIST_Y[side], _WRIST_POSITION_SCALE[1]
    low, high = key_y(reach[0]), key_y(reach[1])
    # A unit action has to carry the hand from the middle of the part to either end of it; the fingers span five
    # keys from there, which is what covers the last few centimetres. Never less travel than a built-in tune gets.
    return 0.5 * (low + high) + _WRIST_LEAN[side], max(_WRIST_POSITION_SCALE[1], 0.5 * (high - low))


def hand_reach(rest_y: float, travel: float, side: str) -> tuple[float, float]:
    """The lowest and highest key a hand can get *a* finger to, wherever its action takes it.

    Wider than :func:`hand_placement`'s window, and answering a different question. Placement asks
    "is this hand set up to play this part", which wants the wrist over the middle of it; this asks
    "could any finger physically touch this key", which is the wrist's travel plus the span from the
    middle of the keys it rests over out to its thumb and little finger. That span is what lets the
    left hand of ``ode_to_joy`` take a bass G its wrist alone could not carry it to.

    Parameters
    ----------
    rest_y : float
        Where the wrist rests along the keyboard, in metres.
    travel : float
        Metres one unit of the wrist's ``y`` action moves it.
    side : str
        ``"right"`` or ``"left"``.

    Returns
    -------
    tuple[float, float]
        Offsets along ``y`` from the keyboard's origin, low then high.
    """
    low_note, high_note = _REST_NOTES[side]
    half_span = 0.5 * abs(key_y(key_of_midi_note(high_note)) - key_y(key_of_midi_note(low_note)))
    # The wrist leans toward the thumb, so the fingers are centred on `_WRIST_LEAN` off the wrist.
    centre = rest_y - _WRIST_LEAN[side]
    return centre - half_span - travel, centre + half_span + travel


def hand_rest_pose(side: str) -> dict[str, float]:
    """Rest joint angles of one hand, by DOF name.

    Parameters
    ----------
    side : str
        ``"right"`` or ``"left"``.

    Returns
    -------
    dict[str, float]
        One entry per DOF of that hand.
    """
    return {f"{side[0]}_{finger}_{joint}": angle for finger, joints in _REST.items() for joint, angle in joints.items()}


def fingertip_links(side: str) -> list[str]:
    """Fingertip link names of one hand, thumb first.

    Parameters
    ----------
    side : str
        ``"right"`` or ``"left"``.

    Returns
    -------
    list[str]
        Five link names.
    """
    return list(_hand_options(side)().metadata.target_hand_links)


def _piano_scene() -> SceneOptions:
    """The keyboard and, per side, a hand resting over the keys on nothing at all."""
    entities = {}
    for side in HANDS:
        entities[f"{side}_hand"] = _hand_options(side)(
            # A free body: `FreeBasePoseController` is what holds it up and moves it along the keys.
            is_fixed_base=False,
            default_root_pos=(_WRIST_X, _WRIST_Y[side], _WRIST_Z),
            default_root_quat=hand_rest_quat(side),
            default_dofs_pos=hand_rest_pose(side),
            # A real forearm carries the hand's weight; here the material does, so the wrist controller only has to
            # move the hand. Without it the hand hangs a few millimetres below every target it is given.
            material=RigidMaterialOptions(gravity_compensation=1.0),
        )
    return SceneOptions(
        floor=PlaneOptions(),
        piano=Piano(default_root_pos=(0.0, 0.0, KEYBOARD_Z)),
        **entities,
    )


def _actions() -> ActionManagerOptions:
    terms = {}
    for side in HANDS:
        terms[f"{side}_wrist"] = FreeBasePoseController.configure(
            entity_name=f"{side}_hand",
            position_scale=_WRIST_POSITION_SCALE,
            orientation_scale=_WRIST_ORIENTATION_SCALE,
            **_WRIST_GAINS,
        )
        terms[f"{side}_hand"] = ImplicitPDController.configure(
            entity_name=f"{side}_hand",
            dofs_name=list(_hand_options(side)().dofs_name),
            scale=_HAND_ACTION_SCALE,
        )
    return ActionManagerOptions(**terms)


_HANDS_ENTITIES = [f"{side}_hand" for side in HANDS]


def _observations() -> dict:
    terms = {
        "song": ObsTerm.configure(func=observations.generated_commands, params={"command_name": "song"}),
        # Not "keys": that is ConfigurableOptions.keys, which a term under the name would hide on the group.
        "key_positions": ObsTerm.configure(func=ct.piano_key_positions, params={"command_name": "song"}),
        "last_action": ObsTerm.configure(func=observations.last_action),
    }
    for name in _HANDS_ENTITIES:
        terms[f"{name}_dofs_pos"] = observations.DofsPos.configure(entity_name=name, offset_from_default=True)
        terms[f"{name}_dofs_vel"] = observations.DofsVel.configure(entity_name=name, scale=0.1)
        # The wrist's own six axes, in the units and the frame its action is in: where it is relative to where it
        # rests, and how fast it is going.
        terms[f"{name}_wrist_pose"] = ObsTerm.configure(func=ct.base_pose_from_default, params={"entity_name": name})
        terms[f"{name}_wrist_lin_vel"] = ObsTerm.configure(
            func=observations.base_lin_vel, params={"entity_name": name}, scale=0.1
        )
        terms[f"{name}_wrist_ang_vel"] = ObsTerm.configure(
            func=observations.base_ang_vel, params={"entity_name": name}, scale=0.1
        )
    return terms


def _episode_seconds(song: str) -> float:
    """Length of an episode: the lead-in plus the score, rounded up to whole control steps."""
    return song_steps(song, _STEP_DT, _LEAD_IN_S) * _STEP_DT


class SongMod(TaskMod):
    """Pick the song via ``--song``.

    The episode is as long as the score, so its length changes with it, and a score that says which hand plays
    each note also moves the hands onto their own parts and gives them the travel to cover them
    (:func:`hand_placement`). Because placement comes from the score, this is the only supported way to choose
    one: a score set straight onto a built config keeps the previous hands, which
    :class:`~eden.tasks.piano.custom_terms.commands.SongCommand` refuses at build rather than train.
    """

    def __init__(
        self,
        song: Annotated[
            str,
            f"Which song to play: one of {sorted(SONGS)}, or a .mid file — its path, or its filename alone if it "
            f"is in {SCORES_DIR}, which is the form that keeps the saved config portable.",
        ] = _DEFAULT_SONG,
    ):
        self.song = song

    def apply(self, config: PianoConfig) -> PianoConfig:
        score = get_song(self.song)  # an unknown name raises here, listing the songs
        config.command_options.song.song = self.song
        config.env_options.episode_length_s = _episode_seconds(self.song)
        config.runner_options.run_name = Path(self.song).stem if self.song not in SONGS else self.song
        for side in HANDS:
            wrist_y, travel = hand_placement(score, side)
            hand = getattr(config.scene_options, f"{side}_hand")
            hand.default_root_pos = (_WRIST_X, wrist_y, _WRIST_Z)
            wrist = getattr(config.action_options, f"{side}_wrist")
            wrist.position_scale = (_WRIST_POSITION_SCALE[0], travel, _WRIST_POSITION_SCALE[2])
        return config


@TASK_REGISTRY.register(modifiers=(SongMod(),))
class PianoConfig(EdenRLConfig):
    env_options: EnvOptions = EnvOptions(
        num_envs=_NUM_ENVS,
        sim_dt=_SIM_DT,
        decimation=_DECIMATION,
        episode_length_s=_episode_seconds(_DEFAULT_SONG),
        # Fingers on neighbouring keys touch each other constantly; the solver should resolve it, not ignore it.
        enable_self_collision=True,
        # A few dozen contacts at most (ten fingertips on keys, neighbouring fingers on each other), and the contact
        # buffers are what fills the GPU: 128 pairs cost 2 GiB more than this per 1024 environments.
        max_collision_pairs=48,
    )

    scene_options: SceneOptions = _piano_scene()

    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        policy=ObservationGroupOptions(**_observations()),
    )

    action_options: ActionManagerOptions = _actions()

    command_options: CommandManagerOptions = CommandManagerOptions(
        song=ct.SongCommand.configure(
            piano_entity_name="piano",
            song=_DEFAULT_SONG,
            # So that a score reaching past where the hands are placed is refused at build, instead of training
            # a policy that is commanded toward keys it cannot get to. See `hand_placement`.
            wrist_action_names={side: f"{side}_wrist" for side in HANDS},
            lookahead_steps=10,
            lead_in_s=_LEAD_IN_S,
            shaping_horizon_steps=5,
        ),
    )

    # Cumulative, so that what is logged at the end of an episode is its mean over the song, which is how
    # RoboPianist reports a performance. (Non-cumulative metrics log the value of the episode's last step.) None of
    # them defines task success: Eden thresholds a cumulative metric's *sum*, which would tie the threshold to the
    # length of the song.
    metric_options: MetricManagerOptions = MetricManagerOptions(
        **{
            name: MetricTerm.configure(
                func=func, params={"command_name": "song"}, is_cumulative=True, contributes_to_success=False
            )
            for name, func in (
                ("key_press_f1", ct.key_press_f1),
                ("key_press_precision", ct.key_press_precision),
                ("key_press_recall", ct.key_press_recall),
            )
        }
    )

    reward_options: RewardManagerOptions = RewardManagerOptions(
        # RoboPianist's kernel (margin 0.5) pays 2% for a key a third of the way down, which is as far as one
        # finger joint moves it from this rest pose: a full press takes the knuckle flexing while the middle joint
        # extends. Twice the margin pays 38% there, so a half-found press is already worth finding.
        key_press=RewardTerm.configure(func=ct.key_press, weight=1.0, params={"command_name": "song", "margin": 1.0}),
        fingertip_to_key=ct.FingertipToKey.configure(
            weight=1.0,
            command_name="song",
            piano_entity_name="piano",
            fingertip_links={f"{side}_hand": fingertip_links(side) for side in HANDS},
        ),
        # RoboPianist's quantity at a tenth of its coefficient (5e-3). That was tuned for the Shadow Hand's small
        # actuators; here a hand draws about 56 W under PPO's exploration noise (32 of them the wrist flying the
        # whole hand around, 24 the fingers), so at 5e-3 moving at all would cost more than a correctly played note
        # could earn and the policy goes still. At 5e-4 the noise costs 0.06 a step against a note's 1.0.
        energy=RewardTerm.configure(func=ct.actuation_energy, weight=-5.0e-4, params={"entity_names": _HANDS_ENTITIES}),
        # Not in RoboPianist. A played note costs about 0.01 of this, a splayed idle hand 0.4 a step, and a hand
        # that has wandered a keyboard-width away and twisted to reach back 0.01.
        rest_pose=RewardTerm.configure(
            func=ct.rest_pose_deviation, weight=-0.02, params={"entity_names": _HANDS_ENTITIES}
        ),
        action_rate=RewardTerm.configure(func=rewards.action_rate_l2, weight=-1.0e-3),
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        song_over=DoneTerm.configure(func=ct.song_over, params={"command_name": "song"}, time_out=True),
    )

    runner_options: RslRlOnPolicyRunnerOptions = RslRlOnPolicyRunnerOptions(
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        algorithm=RslRlPpoAlgorithmOptions(
            entropy_coef=0.002,
            num_learning_epochs=5,
            num_mini_batches=8,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.97,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        actor=RslRlActorOptions(
            activation="elu",
            hidden_dims=[512, 256, 128],
            distribution_cfg=RslRlGaussianDistributionOptions(init_std=0.5),
        ),
        critic=RslRlCriticOptions(activation="elu", hidden_dims=[512, 256, 128]),
        num_steps_per_env=_NUM_STEPS_PER_ENV,
        max_iterations=_MAX_ITERATIONS,
        save_interval=250,
        experiment_name="piano",
        run_name=_DEFAULT_SONG,
    )
