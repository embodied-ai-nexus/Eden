"""Deformable-manipulation tasks on the Mochi solver (the Genesis port of Meta's SuperDex).

Eden's versions of RoboSimStudio's three demos (``examples/08_demos``), at their scenes' numbers, on a bimanual
YAM workcell, each a batched RL task with the demo's own goal as its success:

* ``mochi_wipe_the_bar`` — a YAM picks a foam sponge off the mat and wipes a marker stripe off a grab bar
  (Preston wear of the stripe's 243 cells; 90 % clean).
* ``mochi_fold_the_towel`` — two YAMs fold a terry towel in half, one pinching its far edge, the other pressing
  the near half (fold score, fold extent, at rest, let go).
* ``mochi_route_the_cable`` — two YAMs route a cable from its anchor past two spools and into a port (the
  board's ``routed`` goal).

``common`` holds the workcell (arm mounts, gains, the bench) and the contact stiffness they share;
``custom_terms`` their terms, including RoboSimStudio's two grasp latches.
"""
