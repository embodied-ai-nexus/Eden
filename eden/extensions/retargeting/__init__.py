"""Retargeting extensions (dex-retargeting, SOMA, OmniRetarget, OTRetarget, FlashDexRetarget) with lazy imports."""

from eden.extensions.retargeting.base import RetargetingBase

__all__ = ["RetargetingBase"]


def __getattr__(name: str):
    # Lazy re-exports so submodule-only deps (nlopt, mediapipe, ...) aren't
    # required to import the base `RetargetingBase`.
    if name == "DexRetargeter":
        from eden.extensions.retargeting.dex_retargeter import DexRetargeter

        return DexRetargeter
    if name == "OTRetargeter":
        from eden.extensions.retargeting.otretarget import OTRetargeter

        return OTRetargeter
    raise AttributeError(name)
