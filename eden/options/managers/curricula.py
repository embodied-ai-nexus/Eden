"""Curriculum manager and curriculum-term configuration options."""

from eden.options.managers.base import FunctionTermOptions, ManagerOptions


class CurriculumTermOptions(FunctionTermOptions):
    """Curriculum term specification."""


class CurriculumManagerOptions(ManagerOptions[CurriculumTermOptions]):
    """
    Curriculum manager options.

    Parameters
    ----------
    <curriculum_term_name>: CurriculumTermOptions
        The curriculum terms configuration to be used.
    """
