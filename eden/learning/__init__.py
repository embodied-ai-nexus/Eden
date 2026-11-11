"""Learning-side code that is not a task: algorithms that extend an RL library.

A task (:mod:`eden.tasks`) defines the problem — observations, actions, rewards, terminations,
randomization. How it is solved is a separate choice, and the algorithms here are reusable
answers: :mod:`eden.learning.wrappers` holds one subpackage per RL library they extend. A task selects
one in its ``runner_options``, so ``eden train --task <name>`` trains it.
"""
