"""robosuite manipulation benchmark, ported to Eden/Genesis.

Mirrors the single-arm and two-arm tasks under
https://github.com/ARISE-Initiative/robosuite/tree/master/robosuite/environments/manipulation
(MIT licensed): ``Lift``, ``Stack``, ``PickPlace``, ``NutAssembly``, ``Door``, ``Wipe``, ``ToolHang``,
``TwoArmLift``, ``TwoArmPegInHole``, ``TwoArmHandover`` and ``TwoArmTransport``. Each task lives in
its own ``<task>/config.py`` as a registered ``robosuite_<task>`` config; the arena geometry,
object placement ranges, dense-reward shaping and success predicates follow robosuite's numbers,
with the deviations forced by Genesis (fixed object sizes instead of per-reset resampling, an
episode length sized for RL rather than robosuite's 50 s horizon) called out in each module.

Layout — one rule: a module used by one task lives in that task's directory, a module more than one
task uses lives here at the top (``tests/tasks/robosuite/test_init.py`` checks it).

* ``<task>/config.py`` — the registered ``robosuite_<task>`` config, and the TaskMods only that task takes.
* ``<task>/custom_terms.py`` — the terms only that task uses (not every task has one).
* ``<task>/routine.py`` — the task's scripted tool routine for ``eden datagen`` (Door, PickPlace, Wipe,
  ToolHang, TwoArmTransport); Lift and Stack use the keyframe controllers instead.
* :mod:`~eden.tasks.robosuite.common` — the table arenas, robot placement, sim/control rates, the
  camera every task renders from, and the proprioceptive observation / action factories.
* :mod:`~eden.tasks.robosuite.rigs` — the robot embodiment. Every robot slot holds a
  :class:`~eden.options.robots.ManipulatorOptions` composite from
  :data:`~eden.options.robots.COMPOSITE_ROBOT_REGISTRY`; robosuite's default, the Franka Panda with
  its own gripper, is :class:`~eden.options.robots.PandaFrankaHand`. Every task takes ``--robot``
  (:class:`~eden.tasks.robosuite.rigs.RobosuiteRobotMod`) to swap in any other arm + two-finger
  rig; the terms are keyed on a small, fixed projection of the composite so the swap is a
  mechanical rewrite.
* :mod:`~eden.tasks.robosuite.custom_terms` — the terms more than one task uses: grasp-frame
  observations, tanh reaching rewards and the contact-based grasp / success metrics robosuite's
  ``reward()`` and ``_check_success()`` are built from.
* :mod:`~eden.tasks.robosuite.staged_objects` / :mod:`~eden.tasks.robosuite.variants` — the staged
  per-object MDP and the single-object variants PickPlace and NutAssembly share.
* :mod:`~eden.tasks.robosuite.scripted_demos` — the ``eden datagen`` demonstrator of every task,
  registered under its task name.

Both levels name their terms module ``custom_terms``, so a call site says which one a term comes from:
a task imports its own terms by name (``from eden.tasks.robosuite.door.custom_terms import DofAbove``)
and reaches the shared ones through the package (``from eden.tasks.robosuite import custom_terms``, then
``custom_terms.ReachTanh``) — a bare term name is the task's own, ``custom_terms.X`` is shared.

Objects are the option classes of :mod:`eden.options.objects.robosuite` (robosuite's own MJCF files plus
re-authored composites, loaded through a Genesis-ready runtime transform): a local copy under
``robosuite/`` in the asset directory (:func:`eden.utils.misc.get_assets_dir`) wins, otherwise they
download from the ``Kashu7100/eden_robosuite`` dataset.
"""
