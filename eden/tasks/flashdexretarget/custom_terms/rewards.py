"""FlashDexRetarget's rewards: object, wrist and fingertip tracking, joint BC, action penalty."""

from __future__ import annotations

from typing import ClassVar

import torch

from eden.managers.reward_manager import REWARD_TERM_REGISTRY, RewardTerm
from eden.tasks.flashdexretarget.custom_terms.actions import FDRHandAction
from eden.tasks.flashdexretarget.custom_terms.common import HandTermMixin
from eden.utils.geom import quat_error_magnitude, quat_to_R


class _HandReward(HandTermMixin, RewardTerm):
    """A per-hand ``exp`` reward: ``side=None`` averages over the hands (the value stays in ``[0, 1]``)."""

    side: str | None = None

    def build(self) -> None:
        self._build_state(getattr(self, "n_points", 128))
        self._idx = self._side_indices(self.side)

    def compute(self) -> torch.Tensor:
        return self._per_side_value()[:, self._idx].mean(dim=1)


@REWARD_TERM_REGISTRY.register()
class FDRWristTransReward(_HandReward):
    """``exp(-scale * |MANO wrist - palm|)``.

    Parameters
    ----------
    scale : float
        Sharpness (FlashDexRetarget: 10).
    side : str | None
        One hand, or the mean over the hands.
    """

    scale: float = 10.0

    def _per_side_value(self):
        return torch.exp(-self.scale * (self.state.mano_wrist(0)[0] - self.state.palm_pose()[0]).norm(dim=-1))


@REWARD_TERM_REGISTRY.register()
class FDRWristRotReward(_HandReward):
    """``exp(-scale * angle(MANO wrist, palm))``.

    Parameters
    ----------
    scale : float
        Sharpness (FlashDexRetarget: 3).
    side : str | None
        One hand, or the mean over the hands.
    """

    scale: float = 3.0

    def _per_side_value(self):
        return torch.exp(-self.scale * quat_error_magnitude(self.state.mano_wrist(0)[1], self.state.palm_pose()[1]))


@REWARD_TERM_REGISTRY.register()
class FDRTipTransReward(_HandReward):
    """Mean over the fingertips of ``exp(-scale * |MANO tip - robot tip|)``.

    Parameters
    ----------
    scale : float
        Sharpness (FlashDexRetarget: 20).
    side : str | None
        One hand, or the mean over the hands.
    """

    scale: float = 20.0

    def _per_side_value(self):
        return torch.exp(-self.scale * (self.state.mano_tips(0) - self.state.tips_pos()).norm(dim=-1)).mean(-1)


@REWARD_TERM_REGISTRY.register()
class FDRTipDistanceBoundReward(_HandReward):
    """Mean over the fingertips of ``exp(-max(0, sdf(robot tip) - d_ref) / sigma)``: each robot fingertip at most as
    far from the object surface as the MANO fingertip is from the reference object.

    ``d_ref`` is ``|sdf|`` of the MANO tip in the reference object pose (FlashDexRetarget precomputes the distance to
    the nearest mesh vertex; the SDF is the surface distance, a little smaller).

    Parameters
    ----------
    sigma : float
        Distance scale (m) of the decay (FlashDexRetarget: 0.01).
    side : str | None
        One hand, or the mean over the hands.
    """

    sigma: float = 0.01
    _geometry_: ClassVar[bool] = True

    def _per_side_value(self):
        st = self.state
        obj_pos, obj_quat = st.object_pose()
        ref = st.ref_object(0)
        tips, mano = st.tips_pos(), st.mano_tips(0)
        out = []
        for i in range(len(self.sides)):
            d_sim, _ = st.sdf(i, tips[:, i], obj_pos[:, i], obj_quat[:, i])
            d_ref, _ = st.sdf(i, mano[:, i], ref["pos"][:, i], ref["quat"][:, i])
            out.append(torch.exp(-(d_sim - d_ref.abs()).clamp(min=0.0) / self.sigma).mean(-1))
        return torch.stack(out, 1)


@REWARD_TERM_REGISTRY.register()
class FDRObjectPointCloudReward(_HandReward):
    """``exp(-scale * e)``, ``e`` the mean of the ``topk`` largest per-keypoint displacements between the object's
    surface keypoints at its reference pose and at its simulated pose.

    With ``norm_radius > 0`` each object's keypoints are first rescaled to that largest radius, so a rotation error
    costs the same whatever the object's size (translation stays 1:1).

    Parameters
    ----------
    scale : float
        Sharpness (FlashDexRetarget: 10).
    n_points : int
        Surface keypoints (FlashDexRetarget: 128).
    topk : int
        Worst keypoints averaged; 0 averages them all (FlashDexRetarget: 3).
    norm_radius : float
        Keypoint radius normalisation (m); 0 keeps the object's (FlashDexRetarget: 0.058).
    side : str | None
        One hand's object, or the mean over the hands.
    """

    scale: float = 10.0
    n_points: int = 128
    topk: int = 3
    norm_radius: float = 0.058
    _geometry_: ClassVar[bool] = True

    def _per_side_value(self):
        st = self.state
        pos, quat = st.object_pose()
        ref = st.ref_object(0)
        out = []
        for i in range(len(self.sides)):
            sp = st.surface_points(i)
            if self.norm_radius > 0.0:
                sp = sp * (self.norm_radius / sp.norm(dim=-1).amax(dim=1, keepdim=True).clamp(min=1e-6))[..., None]
            dR = quat_to_R(ref["quat"][:, i]) - quat_to_R(quat[:, i])
            disp = (ref["pos"][:, i] - pos[:, i])[:, None] + torch.einsum("nij,npj->npi", dR, sp)
            d = disp.norm(dim=-1)
            err = d.topk(self.topk, dim=-1).values.mean(-1) if 0 < self.topk < d.shape[-1] else d.mean(-1)
            out.append(torch.exp(-self.scale * err))
        return torch.stack(out, 1)


@REWARD_TERM_REGISTRY.register()
class FDRJointBCReward(RewardTerm):
    """Behaviour cloning of the retargeted joints: mean over the joints of ``exp(-scale * 0.5 (target - ref)^2 /
    range)``, ``target`` the action's PD target and ``ref`` the reference it was built on.

    Parameters
    ----------
    action_term : str
        The :class:`FDRHandAction`.
    scale : float
        Sharpness (FlashDexRetarget: 500).
    dofs_name : list[str] | None
        Only these DOFs (e.g. one hand's); ``None`` all of the action's.
    """

    action_term: str = "hand"
    scale: float = 500.0
    dofs_name: list[str] | None = None

    def build(self) -> None:
        term = self._env.action_manager.get_term(self.action_term)
        if not isinstance(term, FDRHandAction):
            raise TypeError(
                f"FDRJointBCReward reads an FDRHandAction; action term {self.action_term!r} is {type(term).__name__}."
            )
        self._term = term
        names = term.dofs_name if self.dofs_name is None else self.dofs_name
        missing = [n for n in names if n not in term.dofs_name]
        if missing:
            raise ValueError(f"FDRJointBCReward: dofs {missing} are not the action's; it controls {term.dofs_name}.")
        self._cols = torch.tensor([term.dofs_name.index(n) for n in names], dtype=torch.long, device=self.device)

    def compute(self) -> torch.Tensor:
        t = self._term
        rng = (t._upper - t._lower)[self._cols]
        err = 0.5 * (t.processed_action[:, self._cols] - t.reference_used[:, self._cols]).pow(2) / rng
        return torch.exp(-self.scale * err).mean(-1)


@REWARD_TERM_REGISTRY.register()
class FDRActionPenalty(RewardTerm):
    """Mean squared action (use a negative weight).

    Parameters
    ----------
    action_term : str
        The action term whose raw action is penalised.
    dofs_name : list[str] | None
        Only these DOFs' columns; ``None`` every column.
    """

    action_term: str = "hand"
    dofs_name: list[str] | None = None

    def build(self) -> None:
        term = self._env.action_manager.get_term(self.action_term)
        names = term.dofs_name if self.dofs_name is None else self.dofs_name
        self._term = term
        self._cols = torch.tensor([term.dofs_name.index(n) for n in names], dtype=torch.long, device=self.device)

    def compute(self) -> torch.Tensor:
        return self._term.action[:, self._cols].pow(2).mean(-1)
