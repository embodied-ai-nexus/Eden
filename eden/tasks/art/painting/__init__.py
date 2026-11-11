"""Painting: a Panda paints in acrylic with a round brush, dipping it in a palette.

A paint model and paintings planned for Eden, graded by the picture and hand rules the RoboPaint benchmark
(``robopaint.embodied-first-exam.ai``) publishes for acrylic. The workcell, the plan format and what the metric
shares with calligraphy are the art suite's (``eden.tasks.art``). ``paint`` holds the brush and the paint model,
``palette`` the tray of paint wells, ``references`` what a painting is and how its target is painted, ``planner`` how a
work is planned from a picture (and its plan cached), ``works`` the paintings themselves, ``grading`` the picture
rules, ``custom_terms`` the Eden terms, ``scripted_demos`` a privileged painter that shows each painting can be
painted.
"""
