"""The score as a command: which keys should be down now, and which are coming.

:class:`SongCommand` holds a piano roll sampled at the control rate and reads it off the episode clock, so there is
nothing to resample and no state to carry: control step ``t`` of an episode is graded against ``roll[t]``.

It also measures the keyboard once per step, for every term that grades it. The reward, the F1 metrics and the
fingertip shaping all need "which keys are down" and "which keys should be"; computing that here, in
:meth:`SongCommand.post_physics_refresh`, gives them one answer.
"""

from __future__ import annotations

import torch

from eden.managers import COMMAND_TERM_REGISTRY, CommandTerm
from eden.tasks.piano.songs import Song, get_song, piano_roll

# RoboPianist counts a key as pressed within half a degree of its joint limit: 87% of a white key's stroke, and 90%
# of a black key's, which turns further on its shorter lever. This is the white-key figure applied to every key, so
# a black key counts as down a few percent early. It is also where `PianoAudio` throws the hammer by default, so
# "down" here is "sounding" there.
KEY_DOWN_FRACTION = 0.87
# Slack on the reach check, to absorb a placement that went through float32 on its way into the scene. Rounding a
# quarter-metre offset costs ~1e-8 m; anything this check is meant to catch is centimetres.
_REACH_TOLERANCE = 1.0e-6


@COMMAND_TERM_REGISTRY.register()
class SongCommand(CommandTerm):
    """A song to play, published as the key states of this control step and the next few.

    The command is ``(num_envs, (lookahead_steps + 1) * n_keys)``: row ``i`` of the unflattened
    ``(lookahead_steps + 1, n_keys)`` block is 1 where a key should be down ``i`` control steps from now. Row 0 is
    what the step the policy is about to take will be graded against.

    Parameters
    ----------
    piano_entity_name : str
        The keyboard entity. Its DOFs are its keys, lowest note first.
    song : str
        One of :data:`eden.tasks.piano.songs.SONGS`, or a MIDI file (see
        :func:`~eden.tasks.piano.songs.get_song`).
    wrist_action_names : dict[str, str]
        Which action term flies each hand, ``{"right": "right_wrist", ...}``. Given these, :meth:`build`
        refuses a score whose notes the hand is not placed to reach. An empty mapping says the scene has no
        hand to place, and nothing is checked.
    lookahead_steps : int
        Control steps of the score the policy sees beyond the current one. RoboPianist uses 10 (0.5 s at 20 Hz).
    lead_in_s : float
        Seconds of silence before the first note, for the hands to settle over the keys.
    shaping_horizon_steps : int
        :attr:`goal_soon` marks the keys due within this many control steps, which is what the fingertip shaping
        reward pulls the fingers toward: a finger has to be over a key before the key is due.
    key_down_fraction : float
        Fraction of its stroke past which a key counts as down.
    """

    piano_entity_name: str = "piano"
    song: str = "twinkle_twinkle"
    wrist_action_names: dict[str, str] = {}
    lookahead_steps: int = 10
    lead_in_s: float = 1.0
    shaping_horizon_steps: int = 5
    key_down_fraction: float = KEY_DOWN_FRACTION
    # The score advances with the episode clock; nothing is ever resampled.
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    def build(self) -> None:
        if self.lookahead_steps < 0 or self.shaping_horizon_steps < 0:
            raise ValueError(
                f"SongCommand: lookahead_steps={self.lookahead_steps} and shaping_horizon_steps="
                f"{self.shaping_horizon_steps} must both be >= 0."
            )
        piano = self.resolve_entity(self.piano_entity_name)
        score = get_song(self.song)
        self._check_hands_can_reach(score)
        roll = torch.as_tensor(piano_roll(score, self._env.dt, self.lead_in_s), device=self.device)
        self.n_steps, self.n_keys = roll.shape
        self._check_the_episode_outlasts_the_score()
        n_dofs = len(piano.dofs_name)
        if n_dofs != self.n_keys:
            raise ValueError(
                f"SongCommand: entity '{self.piano_entity_name}' has {n_dofs} DOFs, but a piano roll has "
                f"{self.n_keys} keys. The keyboard's DOFs must be its keys, lowest note first."
            )
        # Padded with silence so a window that runs past the end of the score needs no special case.
        padding = torch.zeros((max(self.lookahead_steps, self.shaping_horizon_steps), self.n_keys), device=self.device)
        self._roll = torch.cat((roll.float(), padding))
        self._window = torch.arange(self.lookahead_steps + 1, device=self.device)
        self._shaping_window = torch.arange(self.shaping_horizon_steps + 1, device=self.device)

        lower, upper = piano.get_dofs_limit()
        lower, upper = lower.reshape(-1, self.n_keys)[0], upper.reshape(-1, self.n_keys)[0]
        if not bool(torch.all(torch.isfinite(upper - lower) & (upper > lower))):
            raise ValueError(f"SongCommand: every key of '{self.piano_entity_name}' needs a finite joint range.")
        self._key_lower, self._key_travel = lower.clone(), (upper - lower).clone()

        self._command = torch.zeros((self.num_envs, (self.lookahead_steps + 1) * self.n_keys), device=self.device)
        self.goal = torch.zeros((self.num_envs, self.n_keys), dtype=torch.bool, device=self.device)
        self.goal_soon = torch.zeros((self.num_envs, self.n_keys), dtype=torch.bool, device=self.device)
        self.key_pos = torch.zeros((self.num_envs, self.n_keys), device=self.device)
        self.key_down = torch.zeros((self.num_envs, self.n_keys), dtype=torch.bool, device=self.device)

    def _check_hands_can_reach(self, score: Song) -> None:
        """Refuse a score whose notes the hands cannot get to.

        Where a wrist rests, and how far one unit of its action slides it, are derived from the score by the
        task's ``--song`` modifier. A score set any other way — a saved ``config.json``, an ``--override``, a
        config built without the modifier — therefore keeps whatever hands the previous score left behind, and
        nothing about that looks wrong: the policy is commanded toward keys it cannot physically get to, and the
        run simply never scores.

        The two standards below are applied per *note*, not per score, because a file can name a hand for some of
        its notes and not others — a two-staff export where only one track is named, or a third staff for a part
        no hand here plays. A note whose hand this scene places is held to the stricter of the two: that hand has
        to be *placed for* its part, wrist over the middle of it, because a hand covering only half its part
        trains to a capped F1 that reads as a tuning problem. Every other note — one naming no hand at all, or
        naming one that is not in :attr:`wrist_action_names` — has nothing to place a hand by, so it is held to
        bare reachability instead: its key has to fall inside *some* hand's span (:func:`hand_reach`). Splitting
        it any coarser leaves the notes in between checked by neither.

        Parameters
        ----------
        score : Song
            The score about to be played.
        """
        # Imported here: the task config reads this module.
        from eden.tasks.piano.config import hand_placement, hand_reach

        spans: dict[str, tuple[float, float]] = {}
        for side, term_name in self.wrist_action_names.items():
            wrist = self._env.action_manager.get_term(term_name)
            scale = wrist.position_scale
            travel = float(scale[1] if isinstance(scale, tuple | list) else scale)
            rest_y = float(self.resolve_entity(wrist.entity_name).default_root_pos[0, 1])
            spans[side] = hand_reach(rest_y, travel, side)
            reach = score.reach(side)
            if reach is None:  # the score never says what this hand plays, so there is nothing to place it by
                continue
            needed_y, needed_travel = hand_placement(score, side)
            low, high = needed_y - needed_travel, needed_y + needed_travel
            if rest_y - travel > low + _REACH_TOLERANCE or rest_y + travel < high - _REACH_TOLERANCE:
                raise ValueError(
                    f"SongCommand: the {side} hand is not placed to play {self.song!r}. That part runs from key "
                    f"{reach[0]} to key {reach[1]}, so the wrist has to cover y {low:.3f} to {high:.3f}; "
                    f"'{wrist.entity_name}' rests at y {rest_y:.3f} with {travel:.3f} m of travel per unit of "
                    f"'{term_name}', which covers {rest_y - travel:.3f} to {rest_y + travel:.3f}. Placement is "
                    f"derived from the score, so name the song when the config is built rather than setting it "
                    f"afterwards: TASK_REGISTRY.build('piano', song={self.song!r}), or --song {self.song} on the "
                    f"command line. Widening '{term_name}'.position_scale on its own would buy the reach without "
                    f"moving the hand over its part, which coarsens every note it already played."
                )
        if spans:
            self._check_unplaced_notes_are_within_reach(score, spans)

    def _check_unplaced_notes_are_within_reach(self, score: Song, spans: dict[str, tuple[float, float]]) -> None:
        """Refuse notes no hand was placed for that lie outside every hand's span.

        The weaker half of :meth:`_check_hands_can_reach`, and the only one available for a note that does not
        say who plays it — or says a hand this scene does not have. No wrist moved for such a note, so all that
        can be checked is that it is not asking for a key none of them could touch.

        Parameters
        ----------
        score : Song
            The score about to be played.
        spans : dict[str, tuple[float, float]]
            What each hand can reach along ``y``, by side.
        """
        from eden.options.objects.piano import key_of_midi_note, key_y

        def within_reach(key: int) -> bool:
            return any(low - _REACH_TOLERANCE <= key_y(key) <= high + _REACH_TOLERANCE for low, high in spans.values())

        # Exactly the notes the placement check above did not cover: its loop runs over `wrist_action_names`, so
        # a note naming anything else — including nothing — was never anyone's part to be placed over.
        unplaced = [note for note in score.notes if note.hand not in self.wrist_action_names]
        if not unplaced:
            return
        keys = sorted({key_of_midi_note(note.midi) for note in unplaced})
        out_of_reach = [key for key in keys if not within_reach(key)]
        if not out_of_reach:
            return
        covers = "; ".join(f"the {side} hand reaches y {low:.3f} to {high:.3f}" for side, (low, high) in spans.items())
        if len(out_of_reach) == 1:
            which = f"key {out_of_reach[0]} (y {key_y(out_of_reach[0]):.3f}) is outside"
        else:
            which = (
                f"{len(out_of_reach)} of its keys, {out_of_reach[0]} (y {key_y(out_of_reach[0]):.3f}) to "
                f"{out_of_reach[-1]} (y {key_y(out_of_reach[-1]):.3f}), are outside"
            )
        # `spans` is non-empty here, so there is always at least one hand to name.
        placed = " or ".join(repr(side) for side in self.wrist_action_names)
        if len(unplaced) == len(score.notes):
            subject = "Its notes name"
        else:
            subject = f"{len(unplaced)} of its notes name{'' if len(unplaced) > 1 else 's'}"
        extent = f" Those notes run from key {keys[0]} to key {keys[-1]}." if len(keys) > 1 else ""
        raise ValueError(
            f"SongCommand: {self.song!r} asks for keys no hand can reach. {subject} no hand this scene "
            f"places (it has {placed}), so no wrist was moved onto them — and of the keys they ask for, {which} "
            f"any hand's span.{extent} {covers[0].upper()}{covers[1:]}. Only a note that says which "
            f"hand plays it can have a hand placed over it: a MIDI file exported with two staves says so "
            f"(piano-midi.de names its tracks 'Piano right' / 'Piano left'), a single staff does not, and a "
            f"staff this scene has no hand for never will. Export it with two staves, drop the part nothing "
            f"plays, or transpose it into reach. Widening a wrist's position_scale by hand would buy the reach "
            f"without centring the hand on anything, which coarsens every note it already played."
        )

    def _check_the_episode_outlasts_the_score(self) -> None:
        """Refuse an episode that ends before the score does.

        The episode's length is derived from the score by the task's ``--song`` modifier, exactly as the hands'
        placement is, and it goes the same way when the score is set any other way: the score is simply cut off
        wherever the episode ends. The policy trains on a prefix and the metrics — means over the steps that did
        run — report a perfectly healthy number for it.
        """
        episode_steps = int(self._env.max_episode_length)
        if episode_steps >= self.n_steps:
            return
        raise ValueError(
            f"SongCommand: the episode ends before {self.song!r} does. The score is {self.n_steps} control steps "
            f"long with its lead-in, and the episode is {episode_steps} "
            f"(env_options.episode_length_s = {self._env.max_episode_length_s:.3f} at dt {self._env.dt:.3f}), so "
            f"the last {self.n_steps - episode_steps} steps of it would never be played and the metrics would "
            f"average over the prefix that was. The episode length is derived from the score, so name the song "
            f"when the config is built rather than setting it afterwards: "
            f"TASK_REGISTRY.build('piano', song={self.song!r}), or --song {self.song} on the command line."
        )

    @property
    def command(self) -> torch.Tensor:
        return self._command

    @property
    def song_over(self) -> torch.Tensor:
        """``(num_envs,)`` bool: the episode clock has reached the end of the score."""
        return self._env.episode_length_buf >= self.n_steps

    def key_positions(self) -> torch.Tensor:
        """``(num_envs, n_keys)`` key travel right now, 0 at rest and 1 at the bottom of the stroke."""
        q = self.resolve_entity(self.piano_entity_name).get_dofs_pos()
        return ((q - self._key_lower) / self._key_travel).clamp(0.0, 1.0)

    def post_physics_refresh(self) -> None:
        """Measure the keyboard against the score for the step that just ran (phase 5).

        The env has already advanced ``episode_length_buf``, so the step being graded is one behind it: the policy
        that saw ``roll[t]`` as row 0 of its command acted, physics ran, and the clock now reads ``t + 1``.
        """
        graded = (self._env.episode_length_buf - 1).clamp(0, self.n_steps - 1)
        self.goal.copy_(self._roll[graded] > 0.5)
        self.goal_soon.copy_(self._roll[graded[:, None] + self._shaping_window].amax(dim=1) > 0.5)
        self.key_pos.copy_(self.key_positions())
        self.key_down.copy_(self.key_pos >= self.key_down_fraction)

    def reset(self, envs_idx=None) -> dict[str, float]:
        extras = super().reset(envs_idx)
        idx = slice(None) if envs_idx is None else envs_idx
        # Measurements of the episode that just ended; the next physics step refills them.
        self.goal[idx] = False
        self.goal_soon[idx] = False
        self.key_pos[idx] = 0.0
        self.key_down[idx] = False
        return extras

    def _resample_command(self, envs_idx) -> None:
        """Nothing to draw: the score is fixed and :meth:`_update_command` reads it off the episode clock."""

    def _update_command(self) -> None:
        """Publish the score from the step about to be taken to ``lookahead_steps`` beyond it (phase 17)."""
        now = self._env.episode_length_buf.clamp(max=self.n_steps - 1)
        self._command.copy_(self._roll[now[:, None] + self._window].flatten(1))


__all__ = ["KEY_DOWN_FRACTION", "SongCommand"]
