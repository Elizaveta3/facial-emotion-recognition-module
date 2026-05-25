SCALAR_KEYS = [
    "ear_avg",
    "mar",
    "mouth_width",
    "smile_coeff",
    "brow_dist",
    "mouth_asymmetry",
    "upper_lip_raise",
]


class ParameterSmoother:

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.smoothed = {}

    def update(self, params):
        for key in SCALAR_KEYS:
            raw = params.get(key, 0.0)

            if key in self.smoothed:
                self.smoothed[key] = (
                    self.alpha * raw
                    + (1 - self.alpha) * self.smoothed[key]
                )
            else:
                self.smoothed[key] = raw

        return dict(self.smoothed)
