"""Objects whose pose a deployment reads from a camera, written into the scene each control tick.

A deployment backend drives one robot; the objects the policy observes (a cube, a pen, a tool) have
no driver of their own. :class:`~eden.extensions.deployment.object_pose.tracked_object.TrackedObjectMixin`
gives any backend a set of tracked objects: after each read of the robot it takes each object's
latest pose from a :class:`~eden.extensions.deployment.object_pose.pose_source.PoseSource`, carries it
into the world frame through the frame the source reports in, and poses the object's scene entity
there, so every term that reads the object reads the camera's estimate.

Where the pose comes from is the source's business, and the mixin knows none of it:

- :class:`~eden.extensions.deployment.object_pose.apriltag.AprilTagEstimator`: fiducials on the
  object, optionally relative to an anchor tag on the robot or the table (no camera extrinsic then);
- any other :class:`~eden.extensions.deployment.object_pose.pose_source.PoseEstimator`, such as a
  model-based tracker working from RGB-D and the object's mesh;
- :class:`~eden.extensions.deployment.object_pose.zmq_stream.ZmqPoseSource`: any estimator running in
  another process (a GPU-heavy estimator should not share the control loop's GIL).

An estimator turns one camera frame into a pose; :class:`~eden.extensions.deployment.object_pose.pose_source.EstimatorPoseSource`
runs one on its own thread over a :class:`~eden.extensions.deployment.object_pose.pose_source.FrameSource`
and keeps the latest result; :func:`~eden.extensions.deployment.object_pose.zmq_stream.serve_pose_stream`
runs the same pair as a ZMQ publisher for :class:`ZmqPoseSource` to read.
"""

from eden.extensions.deployment.object_pose.pose_source import (
    CameraFrame,
    EstimatorPoseSource,
    FrameSource,
    PoseEstimator,
    PoseSample,
    PoseSource,
    wait_for_pose,
)
from eden.extensions.deployment.object_pose.tracked_object import TrackedObjectMixin, TrackedObjectSpec
from eden.extensions.deployment.object_pose.zmq_stream import ZmqPoseSource, serve_pose_stream

__all__ = [
    "CameraFrame",
    "EstimatorPoseSource",
    "FrameSource",
    "PoseEstimator",
    "PoseSample",
    "PoseSource",
    "TrackedObjectMixin",
    "TrackedObjectSpec",
    "ZmqPoseSource",
    "serve_pose_stream",
    "wait_for_pose",
]
