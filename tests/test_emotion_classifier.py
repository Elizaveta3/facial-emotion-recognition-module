import unittest

from emotion_classifier import classify_emotion


class EmotionClassifierTest(unittest.TestCase):
    def test_absolute_fear_uses_moderately_open_mouth_not_surprise(self):
        params = {
            "ear_avg": 0.335,
            "mar": 0.34,
            "smile_coeff": -0.004,
            "mouth_width": 0.36,
            "brow_dist": 0.126,
            "mouth_asymmetry": 0.004,
            "upper_lip_raise": 0.072,
        }
        self.assertEqual(classify_emotion(params), "Fear")

    def test_absolute_disgust_uses_eyes_brows_and_slightly_open_mouth(self):
        params = {
            "ear_avg": 0.235,
            "mar": 0.24,
            "smile_coeff": -0.004,
            "mouth_width": 0.35,
            "brow_dist": 0.112,
            "mouth_asymmetry": 0.004,
            "upper_lip_raise": 0.080,
        }
        self.assertEqual(classify_emotion(params), "Disgusted")

    def test_absolute_disgust_requires_narrow_eyes(self):
        params = {
            "ear_avg": 0.285,
            "mar": 0.24,
            "smile_coeff": -0.004,
            "mouth_width": 0.35,
            "brow_dist": 0.112,
            "mouth_asymmetry": 0.004,
            "upper_lip_raise": 0.032,
        }
        self.assertEqual(classify_emotion(params), "Neutral")

    def test_absolute_disgust_requires_low_brows(self):
        params = {
            "ear_avg": 0.235,
            "mar": 0.24,
            "smile_coeff": -0.004,
            "mouth_width": 0.35,
            "brow_dist": 0.123,
            "mouth_asymmetry": 0.004,
            "upper_lip_raise": 0.032,
        }
        self.assertEqual(classify_emotion(params), "Neutral")

    def test_absolute_disgust_allows_mouth_asymmetry(self):
        params = {
            "ear_avg": 0.235,
            "mar": 0.24,
            "smile_coeff": -0.004,
            "mouth_width": 0.35,
            "brow_dist": 0.112,
            "mouth_asymmetry": 0.012,
            "upper_lip_raise": 0.032,
        }
        self.assertEqual(classify_emotion(params), "Disgusted")

    def test_absolute_angry_requires_closed_mouth_and_neutral_eye_width(self):
        params = {
            "ear_avg": 0.28,
            "mar": 0.05,
            "smile_coeff": -0.010,
            "mouth_width": 0.34,
            "brow_dist": 0.110,
            "mouth_asymmetry": 0.004,
            "upper_lip_raise": 0.067,
        }
        self.assertEqual(classify_emotion(params), "Angry")

    def test_absolute_angry_is_blocked_when_eyes_are_too_narrow(self):
        params = {
            "ear_avg": 0.24,
            "mar": 0.05,
            "smile_coeff": -0.010,
            "mouth_width": 0.34,
            "brow_dist": 0.110,
            "mouth_asymmetry": 0.004,
            "upper_lip_raise": 0.067,
        }
        self.assertEqual(classify_emotion(params), "Neutral")

    def test_absolute_angry_accepts_strong_brow_furrow_with_narrower_eyes(self):
        params = {
            "ear_avg": 0.248,
            "mar": 0.05,
            "smile_coeff": -0.010,
            "mouth_width": 0.34,
            "brow_dist": 0.108,
            "mouth_asymmetry": 0.004,
            "upper_lip_raise": 0.040,
        }
        self.assertEqual(classify_emotion(params), "Angry")

    def test_absolute_angry_is_blocked_when_eyes_are_too_wide(self):
        params = {
            "ear_avg": 0.33,
            "mar": 0.05,
            "smile_coeff": -0.010,
            "mouth_width": 0.34,
            "brow_dist": 0.110,
            "mouth_asymmetry": 0.004,
            "upper_lip_raise": 0.067,
        }
        self.assertEqual(classify_emotion(params), "Neutral")

    def test_absolute_angry_is_blocked_when_mouth_is_open(self):
        params = {
            "ear_avg": 0.28,
            "mar": 0.11,
            "smile_coeff": -0.010,
            "mouth_width": 0.34,
            "brow_dist": 0.110,
            "mouth_asymmetry": 0.004,
            "upper_lip_raise": 0.067,
        }
        self.assertEqual(classify_emotion(params), "Neutral")

    def test_absolute_contempt_is_driven_by_mouth_asymmetry(self):
        params = {
            "ear_avg": 0.27,
            "mar": 0.04,
            "smile_coeff": -0.001,
            "mouth_width": 0.35,
            "brow_dist": 0.129,
            "mouth_asymmetry": 0.017,
            "upper_lip_raise": 0.080,
        }
        self.assertEqual(classify_emotion(params), "Contempt")

    def test_delta_disgust_uses_eyes_brows_and_more_open_mouth(self):
        baseline = {
            "ear_avg": 0.29,
            "mar": 0.02,
            "smile_coeff": -0.002,
            "mouth_width": 0.34,
            "brow_dist": 0.13,
            "mouth_asymmetry": 0.003,
            "upper_lip_raise": 0.078,
        }
        params = {
            "ear_avg": 0.245,
            "mar": 0.14,
            "smile_coeff": -0.006,
            "mouth_width": 0.34,
            "brow_dist": 0.112,
            "mouth_asymmetry": 0.005,
            "upper_lip_raise": 0.077,
        }
        self.assertEqual(classify_emotion(params, baseline), "Disgusted")

    def test_delta_disgust_requires_narrower_eyes_than_neutral(self):
        baseline = {
            "ear_avg": 0.29,
            "mar": 0.02,
            "smile_coeff": -0.002,
            "mouth_width": 0.34,
            "brow_dist": 0.13,
            "mouth_asymmetry": 0.003,
            "upper_lip_raise": 0.078,
        }
        params = {
            "ear_avg": 0.27,
            "mar": 0.14,
            "smile_coeff": -0.006,
            "mouth_width": 0.34,
            "brow_dist": 0.112,
            "mouth_asymmetry": 0.005,
            "upper_lip_raise": 0.048,
        }
        self.assertEqual(classify_emotion(params, baseline), "Neutral")

    def test_delta_disgust_requires_lower_brows_than_neutral(self):
        baseline = {
            "ear_avg": 0.29,
            "mar": 0.02,
            "smile_coeff": -0.002,
            "mouth_width": 0.34,
            "brow_dist": 0.13,
            "mouth_asymmetry": 0.003,
            "upper_lip_raise": 0.078,
        }
        params = {
            "ear_avg": 0.245,
            "mar": 0.14,
            "smile_coeff": -0.006,
            "mouth_width": 0.34,
            "brow_dist": 0.121,
            "mouth_asymmetry": 0.005,
            "upper_lip_raise": 0.048,
        }
        self.assertEqual(classify_emotion(params, baseline), "Neutral")

    def test_delta_disgust_allows_mouth_asymmetry(self):
        baseline = {
            "ear_avg": 0.29,
            "mar": 0.02,
            "smile_coeff": -0.002,
            "mouth_width": 0.34,
            "brow_dist": 0.13,
            "mouth_asymmetry": 0.003,
            "upper_lip_raise": 0.078,
        }
        params = {
            "ear_avg": 0.245,
            "mar": 0.14,
            "smile_coeff": -0.006,
            "mouth_width": 0.34,
            "brow_dist": 0.112,
            "mouth_asymmetry": 0.014,
            "upper_lip_raise": 0.048,
        }
        self.assertEqual(classify_emotion(params, baseline), "Disgusted")

    def test_delta_fear_uses_less_mouth_opening_than_surprise(self):
        baseline = {
            "ear_avg": 0.28,
            "mar": 0.05,
            "smile_coeff": -0.002,
            "mouth_width": 0.34,
            "brow_dist": 0.125,
            "mouth_asymmetry": 0.003,
            "upper_lip_raise": 0.078,
        }
        params = {
            "ear_avg": 0.335,
            "mar": 0.24,
            "smile_coeff": -0.004,
            "mouth_width": 0.35,
            "brow_dist": 0.128,
            "mouth_asymmetry": 0.004,
            "upper_lip_raise": 0.077,
        }
        self.assertEqual(classify_emotion(params, baseline), "Fear")

    def test_surprise_still_requires_wide_mouth_opening(self):
        params = {
            "ear_avg": 0.335,
            "mar": 0.58,
            "smile_coeff": -0.004,
            "mouth_width": 0.36,
            "brow_dist": 0.126,
            "mouth_asymmetry": 0.004,
            "upper_lip_raise": 0.072,
        }
        self.assertEqual(classify_emotion(params), "Surprised")

    def test_delta_angry_requires_brows_and_neutral_eye_width(self):
        baseline = {
            "ear_avg": 0.29,
            "mar": 0.02,
            "smile_coeff": -0.002,
            "mouth_width": 0.34,
            "brow_dist": 0.13,
            "mouth_asymmetry": 0.003,
            "upper_lip_raise": 0.078,
        }
        params = {
            "ear_avg": 0.30,
            "mar": 0.04,
            "smile_coeff": -0.009,
            "mouth_width": 0.34,
            "brow_dist": 0.114,
            "mouth_asymmetry": 0.005,
            "upper_lip_raise": 0.070,
        }
        self.assertEqual(classify_emotion(params, baseline), "Angry")

    def test_delta_angry_is_blocked_when_eyes_get_too_narrow(self):
        baseline = {
            "ear_avg": 0.29,
            "mar": 0.02,
            "smile_coeff": -0.002,
            "mouth_width": 0.34,
            "brow_dist": 0.13,
            "mouth_asymmetry": 0.003,
            "upper_lip_raise": 0.078,
        }
        params = {
            "ear_avg": 0.24,
            "mar": 0.04,
            "smile_coeff": -0.009,
            "mouth_width": 0.34,
            "brow_dist": 0.114,
            "mouth_asymmetry": 0.005,
            "upper_lip_raise": 0.070,
        }
        self.assertEqual(classify_emotion(params, baseline), "Neutral")

    def test_delta_angry_accepts_strong_brow_drop_with_narrower_eyes(self):
        baseline = {
            "ear_avg": 0.29,
            "mar": 0.02,
            "smile_coeff": -0.002,
            "mouth_width": 0.34,
            "brow_dist": 0.13,
            "mouth_asymmetry": 0.003,
            "upper_lip_raise": 0.078,
        }
        params = {
            "ear_avg": 0.242,
            "mar": 0.04,
            "smile_coeff": -0.009,
            "mouth_width": 0.34,
            "brow_dist": 0.108,
            "mouth_asymmetry": 0.005,
            "upper_lip_raise": 0.055,
        }
        self.assertEqual(classify_emotion(params, baseline), "Angry")

    def test_delta_angry_is_blocked_when_eyes_open_too_wide(self):
        baseline = {
            "ear_avg": 0.29,
            "mar": 0.02,
            "smile_coeff": -0.002,
            "mouth_width": 0.34,
            "brow_dist": 0.13,
            "mouth_asymmetry": 0.003,
            "upper_lip_raise": 0.078,
        }
        params = {
            "ear_avg": 0.35,
            "mar": 0.04,
            "smile_coeff": -0.009,
            "mouth_width": 0.34,
            "brow_dist": 0.114,
            "mouth_asymmetry": 0.005,
            "upper_lip_raise": 0.070,
        }
        self.assertEqual(classify_emotion(params, baseline), "Neutral")

    def test_delta_angry_is_blocked_when_mouth_opens(self):
        baseline = {
            "ear_avg": 0.29,
            "mar": 0.02,
            "smile_coeff": -0.002,
            "mouth_width": 0.34,
            "brow_dist": 0.13,
            "mouth_asymmetry": 0.003,
            "upper_lip_raise": 0.078,
        }
        params = {
            "ear_avg": 0.28,
            "mar": 0.07,
            "smile_coeff": -0.009,
            "mouth_width": 0.34,
            "brow_dist": 0.114,
            "mouth_asymmetry": 0.005,
            "upper_lip_raise": 0.070,
        }
        self.assertEqual(classify_emotion(params, baseline), "Neutral")

    def test_delta_sad_can_be_detected_with_mild_frown_from_neutral(self):
        baseline = {
            "ear_avg": 0.31,
            "mar": 0.03,
            "smile_coeff": -0.014,
            "mouth_width": 0.34,
            "brow_dist": 0.124,
            "mouth_asymmetry": 0.003,
            "upper_lip_raise": 0.078,
        }
        params = {
            "ear_avg": 0.312,
            "mar": 0.03,
            "smile_coeff": -0.0185,
            "mouth_width": 0.34,
            "brow_dist": 0.126,
            "mouth_asymmetry": 0.0035,
            "upper_lip_raise": 0.077,
        }
        self.assertEqual(classify_emotion(params, baseline), "Sad")

    def test_delta_sad_with_relaxed_brows_is_not_angry(self):
        baseline = {
            "ear_avg": 0.30,
            "mar": 0.03,
            "smile_coeff": -0.013,
            "mouth_width": 0.34,
            "brow_dist": 0.124,
            "mouth_asymmetry": 0.003,
            "upper_lip_raise": 0.078,
        }
        params = {
            "ear_avg": 0.272,
            "mar": 0.04,
            "smile_coeff": -0.019,
            "mouth_width": 0.34,
            "brow_dist": 0.126,
            "mouth_asymmetry": 0.0035,
            "upper_lip_raise": 0.079,
        }
        self.assertEqual(classify_emotion(params, baseline), "Sad")

    def test_delta_contempt_is_mouth_asymmetry(self):
        baseline = {
            "ear_avg": 0.28,
            "mar": 0.02,
            "smile_coeff": -0.004,
            "mouth_width": 0.34,
            "brow_dist": 0.13,
            "mouth_asymmetry": 0.003,
            "upper_lip_raise": 0.079,
        }
        params = {
            "ear_avg": 0.27,
            "mar": 0.05,
            "smile_coeff": -0.003,
            "mouth_width": 0.34,
            "brow_dist": 0.127,
            "mouth_asymmetry": 0.014,
            "upper_lip_raise": 0.076,
        }
        self.assertEqual(classify_emotion(params, baseline), "Contempt")


if __name__ == "__main__":
    unittest.main()
