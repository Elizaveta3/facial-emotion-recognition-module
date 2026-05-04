SCALAR_KEYS = ["ear_avg", "mar", "mouth_width", "smile_coeff", "brow_dist"]


class BaselineCalibrator:
    """Builds a neutral baseline."""

    def __init__(self, num_frames=90):
        self.num_frames = num_frames
        self.frames = []
        self.baseline = None

    def add_frame(self, params):
        """Stores one frame."""
        self.frames.append({k: params[k] for k in SCALAR_KEYS})

    def is_complete(self):
        return len(self.frames) >= self.num_frames

    def get_progress(self):
        """Returns progress in percent."""
        return min(100, int(len(self.frames) / self.num_frames * 100))

    def compute_baseline(self):
        """Averages collected values."""
        if not self.frames:
            return None
        self.baseline = {}
        for key in SCALAR_KEYS:
            self.baseline[key] = sum(f[key] for f in self.frames) / len(self.frames)
        return dict(self.baseline)
