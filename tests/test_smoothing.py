import unittest

from smoothing import ParameterSmoother, SCALAR_KEYS


class ParameterSmootherTest(unittest.TestCase):
    def test_first_update_uses_raw_values_and_defaults_missing_scalars_to_zero(self):
        smoother = ParameterSmoother(alpha=0.25)

        result = smoother.update({"ear_avg": 0.4, "mar": 0.2})

        self.assertEqual(set(result), set(SCALAR_KEYS))
        self.assertEqual(result["ear_avg"], 0.4)
        self.assertEqual(result["mar"], 0.2)
        self.assertEqual(result["mouth_width"], 0.0)

    def test_next_update_applies_exponential_moving_average(self):
        smoother = ParameterSmoother(alpha=0.25)
        first = {key: 0.0 for key in SCALAR_KEYS}
        second = {key: 1.0 for key in SCALAR_KEYS}

        smoother.update(first)
        result = smoother.update(second)

        for key in SCALAR_KEYS:
            self.assertAlmostEqual(result[key], 0.25)

    def test_returned_values_are_copies_not_internal_state(self):
        smoother = ParameterSmoother(alpha=0.5)

        result = smoother.update({key: 0.1 for key in SCALAR_KEYS})
        result["ear_avg"] = 99

        self.assertNotEqual(smoother.smoothed["ear_avg"], 99)


if __name__ == "__main__":
    unittest.main()
