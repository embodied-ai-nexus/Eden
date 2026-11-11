"""Art: a Panda copies a work of calligraphy or of painting on a sheet of paper.

The suite's tasks are ``calligraphy`` (a Chinese brush and a pointed pen) and ``painting`` (acrylic paint with a
round brush): tool models and works of Eden's own, graded by the picture and hand rules the RoboPaint benchmark
publishes. What the two share is here: the workcell, a Panda on a desk with a tool welded in its Franka Hand and a
sheet of paper in front of it (``workcell``); works written as plans of the tool's tip (``plans``); the base of
their success metric, which samples the tip and counts the hand, and the terms that read only it (``terms``); the
scripted demonstrators' plan follower (``scripted``); and the config pieces both build their tasks from
(``common``).
"""
