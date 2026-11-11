"""Manipulation tasks ported from ``mjlab_manipulation`` to Eden/Genesis.

This package mirrors the structure of
https://github.com/omarrayyann/mjlab_manipulation, which defines four task
families parameterized by robot:

* ``Reach-{Robot}``         — move the end-effector to a sampled 3D target.
* ``Lift-{Robot}``          — grasp a cube and lift it to a sampled 3D target.
* ``Lift-{Robot}-RGB``      — same, with a wrist RGB camera observation.
* ``Lift-Any-{Robot}-RGB``  — RGB lift with a randomized graspable object.

In mjlab the ``{Robot}`` is one of ``Droid`` (Franka + Robotiq 2F85), ``xArm``,
``Kuka``, ``Gen3`` or ``UR5e`` — an arm with a *separately attached* gripper.
Eden already ships ``FrankaEmikaPandaNoHand`` (arm) and ``Robotiq2f85`` (gripper),
attached via ``scene.attachments_dict`` exactly like mjlab's ``with_gripper``,
so the ``Droid`` robot needs no asset porting.

Registered tasks, all on the ``Droid`` robot unless noted:

* ``manipulation_reach_droid`` (``reach_droid/config.py``) — close to the canonical ``benchmark/reacher`` task.
* ``manipulation_lift_box`` (``lift_box/config.py``) — the full reach/lift/goal-tracking reward structure of mjlab's
  ``lift_env_cfg``.
* ``manipulation_lift_box_rgb`` (``lift_box_rgb/config.py``) — ``manipulation_lift_box`` with the actor seeing a wrist
  RGB camera instead of the object pose.
* ``manipulation_lift_ycb_rgb`` (``lift_ycb_rgb/config.py``) — ``manipulation_lift_box_rgb`` over a heterogeneous bank
  of YCB objects.
* ``manipulation_dex_lift`` (``dex_lift/config.py``) — ``manipulation_lift_box`` with the 12-DOF ``XHand1_L``
  dexterous hand instead of the 2F85.

Other robots reuse the MDP terms in :mod:`eden.tasks.manipulation.custom_terms` and the composite manipulator
specs in :mod:`eden.options.robots.composite` (``Droid`` / ``COMPOSITE_ROBOT_REGISTRY``).

Layout: ``custom_terms/`` splits the MDP terms by manager
(``observations`` / ``rewards`` / ``terminations`` / ``curricula``,
with ``common`` for shared helpers); each task's ``<task>/config.py`` holds its config, which self-registers via
``@TASK_REGISTRY.register()`` (auto-discovered by the registry's AST scan).
"""
