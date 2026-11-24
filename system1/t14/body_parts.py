"""T14 round 4 / T13 section 2.3: robot link <-> human body part, following the retargeting keypoint correspondence
(config_types/data_type.py JOINTS_MAPPING: Pelvis, hips, knees, ankles, shoulders, elbows, wrists): every simulator link
between two corresponding keypoints maps to the human segment between the same keypoints; a human segment = the SMPL-X
vertices whose largest skinning weight is one of the listed SMPL-X joints. The two rubber hands are left out (hand contact is
wanted).
  SMPL-X joints: 0 pelvis, 1/2 L/R hip (thigh), 3/6/9 spine1-3, 4/5 L/R knee (shank), 7/8 L/R ankle, 10/11 L/R foot,
  12 neck, 13/14 L/R collar, 15 head, 16/17 L/R shoulder (upper arm), 18/19 L/R elbow (forearm), 22-24 jaw / eyes."""
LINK_TO_SMPLX = {
    "pelvis": ("pelvis", (0,)),
    "left_hip_roll_link": ("left thigh", (1,)), "left_hip_yaw_link": ("left thigh", (1,)),
    "left_knee_link": ("left shank", (4,)), "left_ankle_roll_link": ("left foot", (7, 10)),
    "right_hip_roll_link": ("right thigh", (2,)), "right_hip_yaw_link": ("right thigh", (2,)),
    "right_knee_link": ("right shank", (5,)), "right_ankle_roll_link": ("right foot", (8, 11)),
    "torso_link": ("trunk", (3, 6, 9, 13, 14)), "head_link": ("head and neck", (12, 15, 22, 23, 24)),
    "left_shoulder_yaw_link": ("left upper arm", (16,)), "left_elbow_link": ("left forearm", (18,)),
    "left_wrist_roll_link": ("left forearm", (18,)), "left_wrist_pitch_link": ("left forearm", (18,)), "left_wrist_yaw_link": ("left forearm", (18,)),
    "right_shoulder_yaw_link": ("right upper arm", (17,)), "right_elbow_link": ("right forearm", (19,)),
    "right_wrist_roll_link": ("right forearm", (19,)), "right_wrist_pitch_link": ("right forearm", (19,)), "right_wrist_yaw_link": ("right forearm", (19,)),
}
HANDS = ("left_rubber_hand_link", "right_rubber_hand_link")


def smplx_part_vertices():
    """{part name: vertex ids} from the SMPL-X neutral skinning weights."""
    import numpy as np
    from hand_landmarks import SMPLX_NEUTRAL
    lab = np.load(SMPLX_NEUTRAL)["weights"].argmax(1)
    return {name: np.flatnonzero(np.isin(lab, joints)) for name, joints in set(LINK_TO_SMPLX.values())}
