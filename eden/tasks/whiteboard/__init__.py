"""Whiteboard drawing: RoboWorld ``whiteboard``'s drawing tasks, on a Panda holding a whiteboard marker.

Port of ``gym-anything/robotics``'s ``whiteboard`` environment (MIT): its board, marker, ink law
and verifiers, for its three drawing tasks — ``whiteboard_write_text`` (HELLO), ``whiteboard_write_in_colour``
(WELCOME) and ``whiteboard_draw_table``. Ink is laid only where the solver reports the marker's nib pressing on
the board, and the marker is held by the Franka Hand's friction alone.

``common`` is the workcell, ``strokes`` the references, ``grading`` the ink and the verifiers, ``custom_terms``
the Eden terms over them, ``scripted_demos`` a privileged scripted drawer that shows each task can be done.
"""
