"""Recorder manager and recorder-term configuration options."""

import os
import tempfile

from pydantic import Field

from eden.options.managers.base import ManagerOptions, ManagerTermOptions
from eden.constants import DatasetExportMode
from eden.options.file_handler import FileHandlerOptions


def _default_file_handler_options() -> FileHandlerOptions:
    """``NPZFileHandler.configure()``, resolved on first construction rather than at import.

    ``NPZFileHandler`` lives in ``eden.utils.file_handler``, whose ``base`` module needs
    ``FileHandlerOptions`` as a generic parameter (``ConfigurableMixin[FileHandlerOptions]``) and
    so must import ``eden.options`` at module scope — which runs ``eden/options/__init__.py``,
    which imports *this* module. Importing the handler here at module scope would close that loop,
    so ``import eden.utils.file_handler`` would raise ``ImportError`` whenever it was the first of
    the two to be imported.

    A ``default_factory`` breaks the loop at the only edge that can move: a *concrete handler*
    imported by an options module is the layering inversion, while ``base``'s need for the
    options class is structural. Deferred to construction, both modules are loaded by the time it
    runs. Still spelled ``.configure()`` rather than ``FileHandlerOptions(name="npz_file_handler")``
    so the key stays derived from the class name, which is what the registry resolves it back out
    by — a literal here would be a second place to edit when the class is renamed.

    Returns
    -------
    FileHandlerOptions
        Options naming the NPZ handler, the default recorder backend.
    """
    from eden.utils.file_handler.npz_file_handler import NPZFileHandler

    return NPZFileHandler.configure()


class RecorderTermOptions(ManagerTermOptions):
    """Recorder term specification."""

    ...


class RecorderManagerOptions(ManagerOptions[RecorderTermOptions]):
    """
    Recorder manager options.

    Parameters
    ----------
    file_handler_options: FileHandlerOptions
        The options for the file handler.
    dataset_export_dir_path: str
        The directory path where the recorded datasets are exported.
    dataset_filename: str
        Dataset file name without file extension.
    dataset_export_mode: DatasetExportMode
        The mode to handle episode exports.
    override: bool
        Whether to overwrite an existing dataset file. ``False`` (default) appends to it.
    <recorder_term_name>: RecorderTermOptions
        The recorder terms configuration to be used.
    """

    file_handler_options: FileHandlerOptions = Field(default_factory=_default_file_handler_options)
    # Use the platform tempdir so the default is valid on Windows (no ``/tmp``).
    # Resolves to ``/tmp/eden/logs`` on Linux/Mac and ``%TEMP%\eden\logs`` on Windows.
    dataset_export_dir_path: str = os.path.join(tempfile.gettempdir(), "eden", "logs")
    dataset_filename: str = "dataset"
    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_ALL
    override: bool = False
