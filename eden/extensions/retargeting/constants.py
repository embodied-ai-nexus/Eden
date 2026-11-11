"""Shared constants for retargeting."""

from __future__ import annotations


JOINTS_MAP_UNITREE_G1 = {
    "smplh": {
        "Pelvis": "pelvis",
        "L_Hip": "left_hip_pitch_link",
        "R_Hip": "right_hip_pitch_link",
        "L_Knee": "left_knee_link",
        "R_Knee": "right_knee_link",
        "L_Shoulder": "left_shoulder_roll_link",
        "R_Shoulder": "right_shoulder_roll_link",
        "L_Elbow": "left_elbow_link",
        "R_Elbow": "right_elbow_link",
        "L_Ankle": "left_ankle_pitch_link",
        "R_Ankle": "right_ankle_pitch_link",
        "L_Toe": "left_ankle_roll_link",
        "R_Toe": "right_ankle_roll_link",
        "L_Wrist": "left_wrist_yaw_link",
        "R_Wrist": "right_wrist_yaw_link",
    },
    "smplx": {
        "Pelvis": "pelvis",
        "L_Hip": "left_hip_pitch_link",
        "R_Hip": "right_hip_pitch_link",
        "L_Knee": "left_knee_link",
        "R_Knee": "right_knee_link",
        "L_Shoulder": "left_shoulder_roll_link",
        "R_Shoulder": "right_shoulder_roll_link",
        "L_Elbow": "left_elbow_link",
        "R_Elbow": "right_elbow_link",
        "L_Ankle": "left_ankle_pitch_link",
        "R_Ankle": "right_ankle_pitch_link",
        "L_Foot": "left_ankle_roll_link",
        "R_Foot": "right_ankle_roll_link",
        "L_Wrist": "left_wrist_yaw_link",
        "R_Wrist": "right_wrist_yaw_link",
    },
    "mocap": {
        "Spine1": "pelvis",
        "LeftUpLeg": "left_hip_pitch_link",
        "LeftLeg": "left_knee_link",
        "LeftToeBase": "left_ankle_roll_link",
        "RightUpLeg": "right_hip_pitch_link",
        "RightLeg": "right_knee_link",
        "RightToeBase": "right_ankle_roll_link",
        "LeftArm": "left_shoulder_roll_link",
        "LeftForeArm": "left_elbow_link",
        "LeftHandMiddle3": "left_sphere_hand_link",
        "RightArm": "right_shoulder_roll_link",
        "RightForeArm": "right_elbow_link",
        "RightHandMiddle3": "right_sphere_hand_link",
        "LeftFoot": "left_ankle_pitch_link",
        "RightFoot": "right_ankle_pitch_link",
    },
    "lafan1": {
        "Spine1": "pelvis",
        "LeftUpLeg": "left_hip_pitch_link",
        "RightUpLeg": "right_hip_pitch_link",
        "LeftLeg": "left_knee_link",
        "RightLeg": "right_knee_link",
        "LeftArm": "left_shoulder_roll_link",
        "RightArm": "right_shoulder_roll_link",
        "LeftForeArm": "left_elbow_link",
        "RightForeArm": "right_elbow_link",
        "LeftFoot": "left_ankle_pitch_link",
        "RightFoot": "right_ankle_pitch_link",
        "LeftToeBase": "left_ankle_roll_link",
        "RightToeBase": "right_ankle_roll_link",
        "LeftHand": "left_wrist_yaw_link",
        "RightHand": "right_wrist_yaw_link",
    },
}


JOINTS_MAP_BOOSTER_T1 = {
    "smplh": {
        "Pelvis": "Trunk",
        "L_Hip": "Hip_Pitch_Left",
        "R_Hip": "Hip_Pitch_Right",
        "L_Knee": "Shank_Left",
        "R_Knee": "Shank_Right",
        "L_Shoulder": "AL1",
        "R_Shoulder": "AR1",
        "L_Elbow": "left_hand_link",
        "R_Elbow": "right_hand_link",
        "L_Ankle": "Ankle_Cross_Left",
        "R_Ankle": "Ankle_Cross_Right",
        "L_Toe": "left_foot_sphere_5_link",
        "R_Toe": "right_foot_sphere_5_link",
        "L_Wrist": "left_wrist_yaw_link",
        "R_Wrist": "right_wrist_yaw_link",
    },
    "mocap": {
        "Spine1": "Trunk",
        "LeftUpLeg": "Hip_Pitch_Left",
        "LeftLeg": "Shank_Left",
        "LeftToeBase": "left_foot_sphere_5_link",
        "RightUpLeg": "Hip_Pitch_Right",
        "RightLeg": "Shank_Right",
        "RightToeBase": "right_foot_sphere_5_link",
        "LeftArm": "AL1",
        "LeftForeArm": "left_hand_link",
        "LeftHandMiddle3": "left_hand_sphere_link",
        "RightArm": "AR1",
        "RightForeArm": "right_hand_link",
        "RightHandMiddle3": "right_hand_sphere_link",
        "LeftFoot": "Ankle_Cross_Left",
        "RightFoot": "Ankle_Cross_Right",
    },
    "lafan1": {
        "Spine1": "Trunk",
        "LeftUpLeg": "Hip_Pitch_Left",
        "RightUpLeg": "Hip_Pitch_Right",
        "LeftLeg": "Shank_Left",
        "RightLeg": "Shank_Right",
        "LeftArm": "AL1",
        "RightArm": "AR1",
        "LeftForeArm": "left_hand_link",
        "RightForeArm": "right_hand_link",
        "LeftFoot": "Ankle_Cross_Left",
        "RightFoot": "Ankle_Cross_Right",
        "LeftToeBase": "left_foot_sphere_5_link",
        "RightToeBase": "right_foot_sphere_5_link",
        "LeftHand": "left_hand_sphere_link",
        "RightHand": "right_hand_sphere_link",
    },
}


# Data format specific constants
TOE_NAMES_BY_FORMAT = {
    "lafan1": ["LeftToeBase", "RightToeBase"],
    "smplh": ["L_Toe", "R_Toe"],
    "mocap": ["LeftToeBase", "RightToeBase"],
    "smplx": ["L_Foot", "R_Foot"],
}


DATA_FORMAT_CONSTANTS: dict[str, dict[str, float]] = {
    "lafan1": {
        "default_scale_factor": 1.27 / 1.7,
    },
    "mocap": {
        "default_human_height": 1.78,
    },
}
