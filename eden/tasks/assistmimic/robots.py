"""AssistMimic's weakened recipients: PHC's SMPL humanoid at 0.7x mass with a weakened subset of bodies.

The caregiver is :class:`~eden.options.robots.smpl_humanoid.PhcSmplHumanoid` itself. The recipients use the same
skeleton with every geom density scaled by 0.7 (``smpl_humanoid_phc_70pct_mass.xml``, AssistMimic's
``recipient_mass_scale``) and, per dataset, a set of bodies whose DOFs keep half their gains and take a lower
effort limit (``_apply_recipient_weakness``).

Isaac Gym parameterises the three hinges of one body as a single spherical joint in exponential-map
coordinates, while Genesis keeps three hinges (intrinsic X-Y-Z Euler angles). The conversion lives in the
action term (``custom_terms/actions.py``) and in the reset (``custom_terms/commands.py``).
"""

from eden.options.robots.smpl_humanoid import PHC_EFFORT_LIMIT, SMPL_BODY_NAMES, SMPL_DOF_NAMES, PhcSmplHumanoid

#: AssistMimic's weakened recipient (``_apply_recipient_weakness``): every DOF of a listed body keeps this
#: fraction of its ``kp`` / ``kd`` and takes the effort limit it is listed under.
RECIPIENT_WEAKNESS_SCALE = 0.5
#: ``env_im_interx_helpup.yaml``: one ``recipient_weakness_effort`` over the whole lower body.
INTERX_WEAKNESS = {80.0: ("L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe")}
#: ``env_im_hhi-assist.yaml``: getting out of bed needs the hips and the trunk weakened further than the legs
#: (``recipient_hip_effort`` 20, ``recipient_upper_body_effort`` 40), and the toes/ankles/knees left at 80.
HHI_ASSIST_WEAKNESS = {
    80.0: ("L_Knee", "L_Ankle", "L_Toe", "R_Knee", "R_Ankle", "R_Toe"),
    20.0: ("L_Hip", "R_Hip"),
    40.0: ("Torso", "Chest", "Spine"),
}

_PHC = PhcSmplHumanoid()


def _weakened(weakness: dict[float, tuple[str, ...]]) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """``{dof: kp}``, ``{dof: kd}``, ``{dof: effort limit}`` of a recipient weakened per ``{effort: bodies}``."""
    unknown = sorted({b for bodies in weakness.values() for b in bodies} - set(SMPL_BODY_NAMES))
    if unknown:
        raise ValueError(
            f"Recipient weakness profile names {unknown}, which are not bodies of PHC's SMPL humanoid. "
            f"Pick from {SMPL_BODY_NAMES[1:]} (the root 'Pelvis' carries no actuated DOF)."
        )
    limit_of_body = {body: limit for limit, bodies in weakness.items() for body in bodies}
    limit = {dof: limit_of_body.get(dof[:-2]) for dof in SMPL_DOF_NAMES}  # None: not weakened
    gain_scale = {dof: 1.0 if lim is None else RECIPIENT_WEAKNESS_SCALE for dof, lim in limit.items()}
    kp = {dof: _PHC.default_dofs_kp[dof] * gain_scale[dof] for dof in SMPL_DOF_NAMES}
    kd = {dof: _PHC.default_dofs_kd[dof] * gain_scale[dof] for dof in SMPL_DOF_NAMES}
    effort = {dof: PHC_EFFORT_LIMIT if lim is None else lim for dof, lim in limit.items()}
    return kp, kd, effort


_INTERX_KP, _INTERX_KD, _INTERX_EFFORT = _weakened(INTERX_WEAKNESS)
_HHI_KP, _HHI_KD, _HHI_EFFORT = _weakened(HHI_ASSIST_WEAKNESS)


class AssistMimicRecipient(PhcSmplHumanoid):
    """AssistMimic's Inter-X recipient: 0.7x mass (baked into the MJCF densities) and a weakened lower body."""

    file: str = "humanoids/smpl_humanoid/smpl_humanoid_phc_70pct_mass.xml"
    default_dofs_kp: dict[str, float] = _INTERX_KP
    default_dofs_kd: dict[str, float] = _INTERX_KD
    default_dofs_force_limits: dict[str, float] = _INTERX_EFFORT


class AssistMimicBedRecipient(AssistMimicRecipient):
    """AssistMimic's HHI-Assist recipient: the same 0.7x mass, weakened hips and trunk as well as legs."""

    default_dofs_kp: dict[str, float] = _HHI_KP
    default_dofs_kd: dict[str, float] = _HHI_KD
    default_dofs_force_limits: dict[str, float] = _HHI_EFFORT
