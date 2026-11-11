"""Algorithms built on an RL library, one subpackage per library.

- :mod:`.rsl_rl` — models, algorithms and runners for rsl-rl-lib 5.
- :mod:`.rsl_rl_contrack` — the seam to ConTrack's CMDP-PPO fork of rsl_rl (an external backend).

Each algorithm is task-agnostic: whatever ties it to a task (a mirror function for one observation
layout, which observation group carries a privileged target) comes in through its options, so a
task's config selects the algorithm and hands it the task's own pieces. Keep task code out of here,
and keep algorithm code out of ``examples/``: a saved ``config.json`` names the algorithm by import
path, and only a path inside the package resolves for ``eden train --resume`` / ``eden inference``.
"""
