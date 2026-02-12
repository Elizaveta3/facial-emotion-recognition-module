# Threshold constants — tune these to adjust sensitivity

# ── Absolute thresholds (original) ──────────────────────────────────

# Smile coefficient (normalized by face height, so values are small)
SMILE_COEFF_HIGH = 0.005
SMILE_COEFF_NEGATIVE = -0.005

# Eye Aspect Ratio
EAR_HIGH = 0.30
EAR_LOW = 0.26                 # raised from 0.24 — subtle squinting counts

# Mouth Aspect Ratio
MAR_HIGH = 0.5
MAR_MODERATE = 0.15
MAR_LOW = 0.1

# Mouth width (normalized by face width) — widens during smiles
MOUTH_WIDTH_SMILE = 0.43

# Brow-to-eye distance (normalized by face height) — drops when brows furrow
BROW_DIST_LOW = 0.055          # raised from 0.045 — catches moderate furrow

# ── Delta thresholds (relative to calibrated neutral) ───────────────

DELTA_SMILE_HIGH = 0.006       # smile increase from neutral
DELTA_SMILE_LOW = -0.006       # frown from neutral
DELTA_EAR_HIGH = 0.04          # eyes widen
DELTA_EAR_LOW = -0.03          # eyes narrow
DELTA_MAR_HIGH = 0.35          # mouth opens wide
DELTA_MAR_MOD = 0.08           # mouth opens moderately
DELTA_MAR_LOW = -0.01          # mouth not significantly tighter than neutral
DELTA_MW_SMILE = 0.015         # mouth widens (subtle smile threshold)
DELTA_BROW_LOW = -0.02         # brows furrow


def classify_emotion(params, baseline=None):
    """Rule-based emotion classification from facial parameters.

    If baseline is provided, uses delta-from-neutral thresholds.
    Otherwise falls back to absolute thresholds.

    Returns one of: Happy, Surprised, Angry, Sad, Neutral
    """
    if baseline is None:
        return _classify_absolute(params)
    return _classify_delta(params, baseline)


def _classify_absolute(params):
    """Original absolute-threshold classification."""
    ear = params["ear_avg"]
    mar = params["mar"]
    smile = params["smile_coeff"]
    mouth_width = params["mouth_width"]
    brow_dist = params["brow_dist"]

    # Surprised: wide eyes + open mouth
    if ear > EAR_HIGH and mar > MAR_HIGH:
        return "Surprised"

    # Happy: elevated mouth corners, or wide mouth from smiling
    if smile > SMILE_COEFF_HIGH and (mar >= MAR_LOW or mouth_width > MOUTH_WIDTH_SMILE):
        return "Happy"

    # Angry: no smile + (furrowed brows OR narrowed eyes with tight mouth)
    # Two paths so a strong signal in one dimension isn't gated by a borderline other
    if smile < SMILE_COEFF_HIGH:
        if brow_dist < BROW_DIST_LOW:
            return "Angry"
        if ear < EAR_LOW and mar < MAR_MODERATE:
            return "Angry"

    # Sad: mouth corners pulled down + normal eyes + normal brows
    # Brow guard prevents angry faces from leaking into Sad
    if smile < SMILE_COEFF_NEGATIVE and ear >= EAR_LOW and brow_dist >= BROW_DIST_LOW:
        return "Sad"

    return "Neutral"


def _classify_delta(params, baseline):
    """Delta-from-neutral classification using calibrated baseline."""
    d_ear = params["ear_avg"] - baseline["ear_avg"]
    d_mar = params["mar"] - baseline["mar"]
    d_smile = params["smile_coeff"] - baseline["smile_coeff"]
    d_mw = params["mouth_width"] - baseline["mouth_width"]
    d_brow = params["brow_dist"] - baseline["brow_dist"]

    # Surprised: eyes wider than neutral + mouth opened wide
    if d_ear > DELTA_EAR_HIGH and d_mar > DELTA_MAR_HIGH:
        return "Surprised"

    # Happy: smile increase + (mouth opens a bit OR mouth widens)
    if d_smile > DELTA_SMILE_HIGH and (d_mar > DELTA_MAR_LOW or d_mw > DELTA_MW_SMILE):
        return "Happy"

    # Angry: smile drops below neutral + (brows furrowed OR eyes narrowed with tight mouth)
    if d_smile < -DELTA_SMILE_HIGH:
        if d_brow < DELTA_BROW_LOW:
            return "Angry"
        if d_ear < DELTA_EAR_LOW and d_mar < DELTA_MAR_MOD:
            return "Angry"

    # Sad: mouth corners drop + eyes not narrowed + brows not furrowed
    if d_smile < DELTA_SMILE_LOW and d_ear >= DELTA_EAR_LOW and d_brow >= DELTA_BROW_LOW:
        return "Sad"

    return "Neutral"
