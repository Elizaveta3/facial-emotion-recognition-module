import unittest

from calibration import BaselineCalibrator, SCALAR_KEYS


def sample_params(offset=0.0):
    return {
        "ear_avg": 0.30 + offset,
        "mar": 0.10 + offset,
        "mouth_width": 0.40 + offset,
        "smile_coeff": -0.01 + offset,
        "brow_dist": 0.12 + offset,
        "mouth_asymmetry": 0.03 + offset,
        "upper_lip_raise": 0.07 + offset,
    }


class BaselineCalibratorTest(unittest.TestCase):
    def test_progress_is_capped_at_100_percent(self):
        calibrator = BaselineCalibrator(num_frames=2)

        calibrator.add_frame(sample_params())
        calibrator.add_frame(sample_params(0.1))
        calibrator.add_frame(sample_params(0.2))

        self.assertTrue(calibrator.is_complete())
        self.assertEqual(calibrator.get_progress(), 100)

    def test_compute_baseline_averages_only_baseline_scalar_keys(self):
        calibrator = BaselineCalibrator(num_frames=2)

        calibrator.add_frame(sample_params(0.0))
        calibrator.add_frame(sample_params(0.2))

        baseline = calibrator.compute_baseline()

        self.assertEqual(set(baseline), set(SCALAR_KEYS))
        self.assertAlmostEqual(baseline["ear_avg"], 0.40)
        self.assertAlmostEqual(baseline["mar"], 0.20)
        self.assertNotIn("mouth_asymmetry", baseline)
        self.assertNotIn("upper_lip_raise", baseline)

    def test_compute_baseline_returns_none_without_frames(self):
        calibrator = BaselineCalibrator(num_frames=3)

        self.assertIsNone(calibrator.compute_baseline())
        self.assertIsNone(calibrator.baseline)


if __name__ == "__main__":
    unittest.main()
