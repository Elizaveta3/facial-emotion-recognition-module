import numpy as np

# MediaPipe Face Mesh landmark indices

# Right eye: p1(outer), p2(upper-outer), p3(upper-inner), p4(inner), p5(lower-inner), p6(lower-outer)
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
# Left eye: same ordering mirrored
LEFT_EYE = [362, 385, 387, 263, 373, 380]

# Mouth landmarks
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_UPPER_INNER_LEFT = 82
MOUTH_LOWER_INNER_LEFT = 87
MOUTH_UPPER_INNER_RIGHT = 312
MOUTH_LOWER_INNER_RIGHT = 317
MOUTH_LEFT_CORNER = 61
MOUTH_RIGHT_CORNER = 291

# Face width / height reference points
FACE_LEFT = 234
FACE_RIGHT = 454
FACE_TOP = 10
FACE_BOTTOM = 152


def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)


def compute_ear(landmarks, eye_indices):
    """Eye Aspect Ratio = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)"""
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
    vertical_a = euclidean_distance(p2, p6)
    vertical_b = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical_a + vertical_b) / (2.0 * horizontal)


def compute_mar(landmarks):
    """Mouth Aspect Ratio using three vertical distances divided by horizontal width."""
    vertical_a = euclidean_distance(landmarks[MOUTH_TOP], landmarks[MOUTH_BOTTOM])
    vertical_b = euclidean_distance(landmarks[MOUTH_UPPER_INNER_LEFT], landmarks[MOUTH_LOWER_INNER_LEFT])
    vertical_c = euclidean_distance(landmarks[MOUTH_UPPER_INNER_RIGHT], landmarks[MOUTH_LOWER_INNER_RIGHT])
    horizontal = euclidean_distance(landmarks[MOUTH_LEFT_CORNER], landmarks[MOUTH_RIGHT_CORNER])
    if horizontal == 0:
        return 0.0
    return (vertical_a + vertical_b + vertical_c) / (3.0 * horizontal)


def compute_mouth_width(landmarks):
    """Mouth corner distance normalized by face width."""
    mouth_w = euclidean_distance(landmarks[MOUTH_LEFT_CORNER], landmarks[MOUTH_RIGHT_CORNER])
    face_w = euclidean_distance(landmarks[FACE_LEFT], landmarks[FACE_RIGHT])
    if face_w == 0:
        return 0.0
    return mouth_w / face_w


def compute_smile_coefficient(landmarks):
    """Vertical elevation of mouth corners relative to mouth center, normalized by face height."""
    corner_avg_y = (landmarks[MOUTH_LEFT_CORNER][1] + landmarks[MOUTH_RIGHT_CORNER][1]) / 2.0
    center_y = landmarks[MOUTH_TOP][1]
    face_h = euclidean_distance(landmarks[FACE_TOP], landmarks[FACE_BOTTOM])
    if face_h == 0:
        return 0.0
    # Positive when corners are above center (pixel y increases downward, so subtract)
    return (center_y - corner_avg_y) / face_h


def compute_eye_position(landmarks, eye_indices):
    """Iris center position relative to eye bounding box (0-1 range for x and y)."""
    pts = np.array([landmarks[i] for i in eye_indices])
    min_x, min_y = pts[:, 0].min(), pts[:, 1].min()
    max_x, max_y = pts[:, 0].max(), pts[:, 1].max()
    center = pts.mean(axis=0)
    w = max_x - min_x
    h = max_y - min_y
    rel_x = (center[0] - min_x) / w if w > 0 else 0.5
    rel_y = (center[1] - min_y) / h if h > 0 else 0.5
    return rel_x, rel_y


def extract_all_parameters(landmarks):
    """Compute all facial parameters and return as a dict."""
    ear_right = compute_ear(landmarks, RIGHT_EYE)
    ear_left = compute_ear(landmarks, LEFT_EYE)
    ear_avg = (ear_right + ear_left) / 2.0

    mar = compute_mar(landmarks)
    mouth_width = compute_mouth_width(landmarks)
    smile_coeff = compute_smile_coefficient(landmarks)

    eye_pos_right = compute_eye_position(landmarks, RIGHT_EYE)
    eye_pos_left = compute_eye_position(landmarks, LEFT_EYE)

    return {
        "ear_right": ear_right,
        "ear_left": ear_left,
        "ear_avg": ear_avg,
        "mar": mar,
        "mouth_width": mouth_width,
        "smile_coeff": smile_coeff,
        "eye_pos_right": eye_pos_right,
        "eye_pos_left": eye_pos_left,
    }
