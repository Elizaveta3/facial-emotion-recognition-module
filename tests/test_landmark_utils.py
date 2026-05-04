import unittest

import numpy as np

from landmark_utils import (
    FACE_BOTTOM,
    FACE_LEFT,
    FACE_RIGHT,
    FACE_TOP,
    LEFT_EYE,
    MOUTH_BOTTOM,
    MOUTH_LEFT_CORNER,
    MOUTH_LOWER_INNER_LEFT,
    MOUTH_LOWER_INNER_RIGHT,
    MOUTH_RIGHT_CORNER,
    MOUTH_TOP,
    MOUTH_UPPER_INNER_LEFT,
    MOUTH_UPPER_INNER_RIGHT,
    NOSE_BASE,
    RIGHT_BROW_INNER,
    RIGHT_BROW_MID,
    RIGHT_EYE,
    RIGHT_UPPER_EYE,
    UPPER_LIP_TOP,
    compute_brow_distance,
    compute_ear,
    compute_eye_position,
    compute_mar,
    compute_mouth_asymmetry,
    compute_mouth_width,
    compute_smile_coefficient,
    compute_upper_lip_raise,
    euclidean_distance,
    extract_all_parameters,
    landmarks_to_list,
)


def blank_landmarks():
    return np.zeros((478, 3), dtype=float)


class LandmarkUtilsTest(unittest.TestCase):
    def test_euclidean_distance_uses_first_two_coordinates(self):
        self.assertAlmostEqual(euclidean_distance([0, 0, 100], [3, 4, -100]), 5.0)

    def test_compute_ear_returns_zero_when_eye_width_is_zero(self):
        landmarks = blank_landmarks()

        self.assertEqual(compute_ear(landmarks, RIGHT_EYE), 0.0)

    def test_compute_ear_uses_standard_eye_aspect_ratio_formula(self):
        landmarks = blank_landmarks()
        for index, point in zip(
            RIGHT_EYE,
            [(0, 0), (1, 1), (3, 1), (4, 0), (3, -1), (1, -1)],
        ):
            landmarks[index, :2] = point

        self.assertAlmostEqual(compute_ear(landmarks, RIGHT_EYE), 0.5)

    def test_mouth_metrics_are_normalized_by_face_size(self):
        landmarks = blank_landmarks()
        landmarks[FACE_LEFT, :2] = [0, 0]
        landmarks[FACE_RIGHT, :2] = [100, 0]
        landmarks[FACE_TOP, :2] = [50, 0]
        landmarks[FACE_BOTTOM, :2] = [50, 200]
        landmarks[MOUTH_LEFT_CORNER, :2] = [30, 100]
        landmarks[MOUTH_RIGHT_CORNER, :2] = [70, 110]
        landmarks[MOUTH_TOP, :2] = [50, 90]
        landmarks[MOUTH_BOTTOM, :2] = [50, 130]
        landmarks[MOUTH_UPPER_INNER_LEFT, :2] = [40, 95]
        landmarks[MOUTH_LOWER_INNER_LEFT, :2] = [40, 115]
        landmarks[MOUTH_UPPER_INNER_RIGHT, :2] = [60, 95]
        landmarks[MOUTH_LOWER_INNER_RIGHT, :2] = [60, 115]
        landmarks[UPPER_LIP_TOP, :2] = [50, 95]
        landmarks[NOSE_BASE, :2] = [50, 75]

        mouth_width = np.sqrt(40 ** 2 + 10 ** 2)
        self.assertAlmostEqual(compute_mouth_width(landmarks), mouth_width / 100)
        self.assertAlmostEqual(compute_mar(landmarks), (40 + 20 + 20) / (3 * mouth_width))
        self.assertAlmostEqual(compute_smile_coefficient(landmarks), (90 - 105) / 200)
        self.assertAlmostEqual(compute_mouth_asymmetry(landmarks), 10 / 200)
        self.assertAlmostEqual(compute_upper_lip_raise(landmarks), 20 / 200)

    def test_normalized_metrics_return_zero_when_face_size_is_zero(self):
        landmarks = blank_landmarks()

        self.assertEqual(compute_mouth_width(landmarks), 0.0)
        self.assertEqual(compute_smile_coefficient(landmarks), 0.0)
        self.assertEqual(compute_brow_distance(landmarks), 0.0)
        self.assertEqual(compute_mouth_asymmetry(landmarks), 0.0)
        self.assertEqual(compute_upper_lip_raise(landmarks), 0.0)

    def test_compute_eye_position_handles_degenerate_box(self):
        landmarks = blank_landmarks()
        for index in LEFT_EYE:
            landmarks[index, :2] = [2, 3]

        self.assertEqual(compute_eye_position(landmarks, LEFT_EYE), (0.5, 0.5))

    def test_extract_all_parameters_returns_expected_keys(self):
        landmarks = blank_landmarks()
        landmarks[FACE_LEFT, :2] = [0, 0]
        landmarks[FACE_RIGHT, :2] = [100, 0]
        landmarks[FACE_TOP, :2] = [50, 0]
        landmarks[FACE_BOTTOM, :2] = [50, 200]
        landmarks[MOUTH_LEFT_CORNER, :2] = [30, 100]
        landmarks[MOUTH_RIGHT_CORNER, :2] = [70, 100]
        landmarks[MOUTH_TOP, :2] = [50, 90]
        landmarks[MOUTH_BOTTOM, :2] = [50, 130]
        landmarks[MOUTH_UPPER_INNER_LEFT, :2] = [40, 95]
        landmarks[MOUTH_LOWER_INNER_LEFT, :2] = [40, 115]
        landmarks[MOUTH_UPPER_INNER_RIGHT, :2] = [60, 95]
        landmarks[MOUTH_LOWER_INNER_RIGHT, :2] = [60, 115]
        landmarks[RIGHT_BROW_INNER, :2] = [35, 40]
        landmarks[RIGHT_BROW_MID, :2] = [40, 35]
        landmarks[RIGHT_UPPER_EYE, :2] = [35, 60]
        landmarks[UPPER_LIP_TOP, :2] = [50, 95]
        landmarks[NOSE_BASE, :2] = [50, 75]
        for indices in (RIGHT_EYE, LEFT_EYE):
            for index, point in zip(
                indices,
                [(0, 0), (1, 1), (3, 1), (4, 0), (3, -1), (1, -1)],
            ):
                landmarks[index, :2] = point

        params = extract_all_parameters(landmarks)

        self.assertEqual(
            set(params),
            {
                "ear_right",
                "ear_left",
                "ear_avg",
                "mar",
                "mouth_width",
                "smile_coeff",
                "brow_dist",
                "mouth_asymmetry",
                "upper_lip_raise",
                "eye_pos_right",
                "eye_pos_left",
            },
        )
        self.assertAlmostEqual(params["ear_avg"], 0.5)

    def test_landmarks_to_list_rounds_coordinates_to_five_digits(self):
        result = landmarks_to_list([[1.123456, 2.987654, -0.000004]])

        self.assertEqual(result, [[1.12346, 2.98765, -0.0]])


if __name__ == "__main__":
    unittest.main()
