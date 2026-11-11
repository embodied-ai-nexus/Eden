"""MediaPipe wrapper that detects a single hand and returns MANO keypoints.

Copied verbatim (minus the SAPIEN imports) from ``dex-retargeting``'s
``example/vector_retargeting/single_hand_detector.py`` so the Eden examples
can be run without cloning the upstream repository. See the AnyTeleop paper
for the orientation-estimation derivation.
"""

from __future__ import annotations

import mediapipe as mp
import mediapipe.framework as framework
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import hands_connections
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.hands import HandLandmark


OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ]
)


class SingleHandDetector:
    def __init__(
        self,
        hand_type: str = "Right",
        min_detection_confidence: float = 0.8,
        min_tracking_confidence: float = 0.8,
        selfie: bool = False,
    ) -> None:
        self.hand_detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.selfie = selfie
        self.operator2mano = OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
        inverse_hand_dict = {"Right": "Left", "Left": "Right"}
        self.detected_hand_type = hand_type if selfie else inverse_hand_dict[hand_type]

    @staticmethod
    def draw_skeleton_on_image(image, keypoint_2d, style: str = "default"):
        if style == "default":
            mp.solutions.drawing_utils.draw_landmarks(
                image,
                keypoint_2d,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style(),
            )
            return image

        landmark_style = {
            landmark: DrawingSpec(color=(255, 48, 48), circle_radius=4, thickness=-1) for landmark in HandLandmark
        }
        connection_style = {pair: DrawingSpec(thickness=2) for pair in hands_connections.HAND_CONNECTIONS}
        mp.solutions.drawing_utils.draw_landmarks(
            image, keypoint_2d, mp.solutions.hands.HAND_CONNECTIONS, landmark_style, connection_style
        )
        return image

    def detect(self, rgb):
        results = self.hand_detector.process(rgb)
        if not results.multi_hand_landmarks:
            return 0, None, None, None

        desired = -1
        for i in range(len(results.multi_hand_landmarks)):
            label = results.multi_handedness[i].ListFields()[0][1][0].label
            if label == self.detected_hand_type:
                desired = i
                break
        if desired < 0:
            return 0, None, None, None

        keypoint_3d = results.multi_hand_world_landmarks[desired]
        keypoint_2d = results.multi_hand_landmarks[desired]
        num_box = len(results.multi_hand_landmarks)

        keypoint_3d_array = self._parse_keypoint_3d(keypoint_3d)
        keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]
        mediapipe_wrist_rot = self._estimate_frame_from_hand_points(keypoint_3d_array)
        joint_pos = keypoint_3d_array @ mediapipe_wrist_rot @ self.operator2mano

        return num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot

    @staticmethod
    def _parse_keypoint_3d(keypoint_3d: framework.formats.landmark_pb2.LandmarkList) -> np.ndarray:
        keypoint = np.empty([21, 3])
        for i in range(21):
            keypoint[i][0] = keypoint_3d.landmark[i].x
            keypoint[i][1] = keypoint_3d.landmark[i].y
            keypoint[i][2] = keypoint_3d.landmark[i].z
        return keypoint

    @staticmethod
    def parse_keypoint_2d(keypoint_2d: landmark_pb2.NormalizedLandmarkList, img_size) -> np.ndarray:
        keypoint = np.empty([21, 2])
        for i in range(21):
            keypoint[i][0] = keypoint_2d.landmark[i].x
            keypoint[i][1] = keypoint_2d.landmark[i].y
        return keypoint * np.array([img_size[1], img_size[0]])[None, :]

    @staticmethod
    def _estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        assert keypoint_3d_array.shape == (21, 3)
        points = keypoint_3d_array[[0, 5, 9], :]
        x_vector = points[0] - points[2]

        points = points - np.mean(points, axis=0, keepdims=True)
        _, _, v = np.linalg.svd(points)
        normal = v[2, :]

        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)
        z = np.cross(x, normal)

        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1
        return np.stack([x, normal, z], axis=1)
