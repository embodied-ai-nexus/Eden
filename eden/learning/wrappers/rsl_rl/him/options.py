"""The :class:`~eden.learning.wrappers.rsl_rl.him.ppo.HIMPPO` algorithm options."""

from __future__ import annotations

from pydantic import model_validator

from eden.options.learning.rsl_rl import RslRlPpoAlgorithmOptions

# Fields of the stock PPO options HIMPPO has no counterpart for, with the value it implies.
_UNSUPPORTED: dict[str, object] = {
    "rnd_cfg": None,
    "symmetry_cfg": None,
    "normalize_advantage_per_mini_batch": False,
    "share_cnn_encoders": False,
}


class HIMPPOAlgorithmOptions(RslRlPpoAlgorithmOptions):
    """PPO options for OpenHomie's HIMLoco learner.

    Built by rsl-rl-lib's own ``OnPolicyRunner`` on any major Eden supports (3, 4, 5), so ``eden train``
    and ``eden inference`` need nothing algorithm-specific; :meth:`adapt_runner_dict` routes the
    construction for the installed major. The task supplies the two things that tie
    HIMPPO to an observation layout:

    - the observation groups: ``obs_groups["actor"]`` one group of stacked frames,
      ``obs_groups["critic"]`` one group of a single frame + the true base linear velocity (3, last);
    - the mirror, ``data_augmentation_func``.

    Parameters
    ----------
    data_augmentation_func: str
        ``"module:function"`` mirroring the task's observations and actions, with rsl-rl-lib's symmetry
        signature ``f(env=, obs=TensorDict | None, actions=Tensor | None) -> (obs_aug, actions_aug)``,
        each the input stacked with its mirror along the batch axis — the same function stock PPO's
        symmetry extension takes. Required: every transition is stored with its mirror.
    use_flip: bool
        Also train the estimator on mirrored observations and add the actor and critic mirror losses
        (OpenHomie ``use_flip``). Symmetry lives here, so ``symmetry_cfg`` must stay ``None``.
    symmetry_scale: float
        Weight of the two mirror losses.
    target_skip_dim: int
        Leading entries of the next critic observation the SwAV target encoder leaves out (OpenHomie
        drops the three velocity commands a frame starts with).
    """

    class_name: str = "eden.learning.wrappers.rsl_rl.him.ppo:HIMPPO"
    entropy_coef: float = 0.01
    data_augmentation_func: str
    use_flip: bool = True
    symmetry_scale: float = 1.0
    target_skip_dim: int = 3

    @model_validator(mode="after")
    def _check_supported(self) -> HIMPPOAlgorithmOptions:
        set_unsupported = {k: getattr(self, k) for k, v in _UNSUPPORTED.items() if getattr(self, k) != v}
        if set_unsupported:
            raise ValueError(
                f"HIMPPO does not implement {sorted(set_unsupported)} (got {set_unsupported!r}); leave them at "
                f"{ {k: _UNSUPPORTED[k] for k in set_unsupported} }. Its symmetry is `use_flip` / `symmetry_scale`."
            )
        if ":" not in self.data_augmentation_func:
            raise ValueError(
                f"data_augmentation_func must be a 'module:function' import path, got {self.data_augmentation_func!r} "
                f"(e.g. 'eden.tasks.homie.symmetry:mirror_observations_and_actions')."
            )
        return self

    def adapt_runner_dict(self, runner_dict: dict, major: int) -> dict:
        """Point the version-translated runner dict at HIM's constructors for rsl-rl-lib ``major``.

        4 and 5 build the algorithm through ``class_name`` → ``construct_algorithm``, which reads the
        translated dict as it is. 3 builds the model from ``policy.class_name`` and calls the algorithm
        class with it, so both go to HIM's v3 factories, and the model gets ``target_skip_dim``.

        Parameters
        ----------
        runner_dict: dict
            The dict :func:`~eden.options.learning.rsl_rl.translate.translate_runner_dict` produced for ``major``.
        major: int
            The installed rsl-rl-lib major version.

        Returns
        -------
        dict
            ``runner_dict``, adapted in place.
        """
        if major == 3:
            module = "eden.learning.wrappers.rsl_rl.him"
            runner_dict["policy"]["class_name"] = f"{module}.actor_critic:HIMActorCritic.from_rsl_rl_v3"
            runner_dict["policy"]["target_skip_dim"] = self.target_skip_dim
            runner_dict["algorithm"]["class_name"] = f"{module}.ppo:HIMPPO.from_rsl_rl_v3"
        return runner_dict
