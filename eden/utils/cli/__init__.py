"""Eden command-line interface.

Command bodies are grouped by area (``tasks``, ``viewers``, ``devices``,
``teleop``, ``dataset``, ``maintenance``, ``examples``, ``pose``, ``retarget``,
``terms``);
:mod:`eden.utils.cli.registry` declares the ``Command`` table and
dispatch that ``eden.__main__`` wires into an argparse entry point, and
:mod:`eden.utils.cli.paths` holds the ``--src`` / ``--out`` file plumbing the
batch commands share.
"""
