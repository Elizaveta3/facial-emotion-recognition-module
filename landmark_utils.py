import math

import numpy as np

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_UPPER_INNER_LEFT = 82
MOUTH_LOWER_INNER_LEFT = 87
MOUTH_UPPER_INNER_RIGHT = 312
MOUTH_LOWER_INNER_RIGHT = 317
MOUTH_LEFT_CORNER = 61
MOUTH_RIGHT_CORNER = 291

RIGHT_BROW_INNER = 65
RIGHT_BROW_MID = 105
LEFT_BROW_INNER = 295
LEFT_BROW_MID = 334

RIGHT_UPPER_EYE = 159
LEFT_UPPER_EYE = 386

FACE_LEFT = 234
FACE_RIGHT = 454
FACE_TOP = 10
FACE_BOTTOM = 152

UPPER_LIP_TOP = 0
NOSE_BASE = 2

MAX_FACE_YAW_RATIO = 0.18
MAX_FACE_ROLL_DEGREES = 15.0


def _xy(landmarks):
    lm = np.asarray(landmarks)
    return lm[:, :2]


def euclidean_distance(p1, p2):
    a = np.asarray(p1).ravel()[:2]
    b = np.asarray(p2).ravel()[:2]
    return np.linalg.norm(a - b)


def compute_ear(landmarks, eye_indices):
    lm = _xy(landmarks)
    p1, p2, p3, p4, p5, p6 = [lm[i] for i in eye_indices]
    vertical_a = euclidean_distance(p2, p6)
    vertical_b = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical_a + vertical_b) / (2.0 * horizontal)


def compute_mar(landmarks):
    lm = _xy(landmarks)
    vertical_a = euclidean_distance(lm[MOUTH_TOP], lm[MOUTH_BOTTOM])
    vertical_b = euclidean_distance(lm[MOUTH_UPPER_INNER_LEFT], lm[MOUTH_LOWER_INNER_LEFT])
    vertical_c = euclidean_distance(lm[MOUTH_UPPER_INNER_RIGHT], lm[MOUTH_LOWER_INNER_RIGHT])
    horizontal = euclidean_distance(lm[MOUTH_LEFT_CORNER], lm[MOUTH_RIGHT_CORNER])
    if horizontal == 0:
        return 0.0
    return (vertical_a + vertical_b + vertical_c) / (3.0 * horizontal)


def compute_mouth_width(landmarks):
    lm = _xy(landmarks)
    mouth_w = euclidean_distance(lm[MOUTH_LEFT_CORNER], lm[MOUTH_RIGHT_CORNER])
    face_w = euclidean_distance(lm[FACE_LEFT], lm[FACE_RIGHT])
    if face_w == 0:
        return 0.0
    return mouth_w / face_w


def compute_smile_coefficient(landmarks):
    lm = _xy(landmarks)
    corner_avg_y = (lm[MOUTH_LEFT_CORNER][1] + lm[MOUTH_RIGHT_CORNER][1]) / 2.0
    center_y = lm[MOUTH_TOP][1]
    face_h = euclidean_distance(lm[FACE_TOP], lm[FACE_BOTTOM])
    if face_h == 0:
        return 0.0
    return (center_y - corner_avg_y) / face_h


def compute_brow_distance(landmarks):
    lm = _xy(landmarks)
    right_inner = euclidean_distance(lm[RIGHT_BROW_INNER], lm[RIGHT_UPPER_EYE])
    right_mid = euclidean_distance(lm[RIGHT_BROW_MID], lm[RIGHT_UPPER_EYE])
    left_inner = euclidean_distance(lm[LEFT_BROW_INNER], lm[LEFT_UPPER_EYE])
    left_mid = euclidean_distance(lm[LEFT_BROW_MID], lm[LEFT_UPPER_EYE])
    face_h = euclidean_distance(lm[FACE_TOP], lm[FACE_BOTTOM])
    if face_h == 0:
        return 0.0
    return (right_inner + right_mid + left_inner + left_mid) / (4.0 * face_h)


def compute_mouth_asymmetry(landmarks):
    lm = _xy(landmarks)
    face_h = euclidean_distance(lm[FACE_TOP], lm[FACE_BOTTOM])
    if face_h == 0:
        return 0.0
    return abs(lm[MOUTH_LEFT_CORNER][1] - lm[MOUTH_RIGHT_CORNER][1]) / face_h


def compute_upper_lip_raise(landmarks):
    lm = _xy(landmarks)
    face_h = euclidean_distance(lm[FACE_TOP], lm[FACE_BOTTOM])
    if face_h == 0:
        return 0.0
    return abs(lm[UPPER_LIP_TOP][1] - lm[NOSE_BASE][1]) / face_h


def compute_eye_position(landmarks, eye_indices):
    lm = _xy(landmarks)
    pts = np.array([lm[i] for i in eye_indices])
    min_x, min_y = pts[:, 0].min(), pts[:, 1].min()
    max_x, max_y = pts[:, 0].max(), pts[:, 1].max()
    center = pts.mean(axis=0)
    w = max_x - min_x
    h = max_y - min_y
    rel_x = (center[0] - min_x) / w if w > 0 else 0.5
    rel_y = (center[1] - min_y) / h if h > 0 else 0.5
    return rel_x, rel_y


def estimate_face_orientation(landmarks):
    lm = _xy(landmarks)
    face_w = euclidean_distance(lm[FACE_LEFT], lm[FACE_RIGHT])
    if face_w == 0:
        return {
            "yaw_ratio": 0.0,
            "roll_degrees": 0.0,
            "is_frontal": False,
        }

    face_center_x = (lm[FACE_LEFT][0] + lm[FACE_RIGHT][0]) / 2.0
    yaw_ratio = (lm[NOSE_BASE][0] - face_center_x) / face_w

    right_eye_center = np.array([lm[i] for i in RIGHT_EYE]).mean(axis=0)
    left_eye_center = np.array([lm[i] for i in LEFT_EYE]).mean(axis=0)
    eye_delta = left_eye_center - right_eye_center
    roll_degrees = math.degrees(math.atan2(eye_delta[1], eye_delta[0]))

    return {
        "yaw_ratio": yaw_ratio,
        "roll_degrees": roll_degrees,
        "is_frontal": (
            abs(yaw_ratio) <= MAX_FACE_YAW_RATIO
            and abs(roll_degrees) <= MAX_FACE_ROLL_DEGREES
        ),
    }


def is_face_frontal(landmarks):
    return estimate_face_orientation(landmarks)["is_frontal"]


def extract_all_parameters(landmarks):
    ear_right = compute_ear(landmarks, RIGHT_EYE)
    ear_left = compute_ear(landmarks, LEFT_EYE)
    ear_avg = (ear_right + ear_left) / 2.0

    mar = compute_mar(landmarks)
    mouth_width = compute_mouth_width(landmarks)
    smile_coeff = compute_smile_coefficient(landmarks)
    brow_dist = compute_brow_distance(landmarks)
    mouth_asymmetry = compute_mouth_asymmetry(landmarks)
    upper_lip_raise = compute_upper_lip_raise(landmarks)

    eye_pos_right = compute_eye_position(landmarks, RIGHT_EYE)
    eye_pos_left = compute_eye_position(landmarks, LEFT_EYE)
    orientation = estimate_face_orientation(landmarks)

    return {
        "ear_right": ear_right,
        "ear_left": ear_left,
        "ear_avg": ear_avg,
        "mar": mar,
        "mouth_width": mouth_width,
        "smile_coeff": smile_coeff,
        "brow_dist": brow_dist,
        "mouth_asymmetry": mouth_asymmetry,
        "upper_lip_raise": upper_lip_raise,
        "eye_pos_right": eye_pos_right,
        "eye_pos_left": eye_pos_left,
        "face_yaw_ratio": orientation["yaw_ratio"],
        "face_roll_degrees": orientation["roll_degrees"],
        "is_face_frontal": orientation["is_frontal"],
    }


def landmarks_to_list(landmarks_3d):
    return [[round(float(x), 5), round(float(y), 5), round(float(z), 5)]
            for x, y, z in landmarks_3d]
