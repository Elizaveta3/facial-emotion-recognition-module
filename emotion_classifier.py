# Threshold constants — tune these to adjust sensitivity

# ── Absolute thresholds ─────────────────────────────────────────────
SMILE_COEFF_HIGH = 0.005
SMILE_COEFF_NEGATIVE = -0.005
SMILE_COEFF_SAD = -0.012

EAR_HIGH = 0.30
EAR_LOW = 0.26
EAR_TENSE = 0.245

MAR_HIGH = 0.50
MAR_MODERATE = 0.15
MAR_LOW = 0.10

MOUTH_WIDTH_SMILE = 0.43

BROW_DIST_FURROWED = 0.118
BROW_DIST_RELAXED = 0.128

# Emotion-specific metrics
MOUTH_ASYM_HIGH = 0.013         # contempt should be visibly asymmetric
MOUTH_ASYM_LOW = 0.009          # disgust / anger should stay comparatively symmetric
UPPER_LIP_RAISE_DISGUST = 0.040  # smaller = upper lip pulled toward nose
UPPER_LIP_RAISE_BLOCK_ANGER = 0.050

# ── Delta thresholds (relative to calibrated neutral) ──────────────
DELTA_SMILE_HIGH = 0.006
DELTA_SMILE_LOW = -0.006
DELTA_SMILE_SAD = -0.003
DELTA_EAR_HIGH = 0.04
DELTA_EAR_LOW = -0.03
DELTA_EAR_TENSE = -0.02
DELTA_MAR_HIGH = 0.35
DELTA_MAR_MOD = 0.08
DELTA_MAR_LOW = -0.01
DELTA_MW_SMILE = 0.015
DELTA_BROW_LOW = -0.02
DELTA_BROW_FURROWED = -0.012
DELTA_BROW_SAD = -0.004
DELTA_ASYM_HIGH = 0.008
DELTA_ASYM_LOW = 0.005
DELTA_LIP_RAISE = 0.015
DELTA_LIP_RAISE_STRONG = 0.020


EMOTIONS = (
    "Happy",
    "Surprised",
    "Angry",
    "Sad",
    "Disgusted",
    "Contempt",
    "Neutral",
)


def classify_emotion(params, baseline=None):
    """Rule-based emotion classification from facial parameters.

    If baseline is provided, uses delta-from-neutral thresholds.
    Otherwise falls back to absolute thresholds.
    """
    if baseline is None:
        return _classify_absolute(params)
    return _classify_delta(params, baseline)


def _absolute_disgust_signal(upper_lip_raise, mar, mouth_asym, smile):
    return (
        upper_lip_raise < UPPER_LIP_RAISE_DISGUST
        and mar < MAR_MODERATE
        and mouth_asym < MOUTH_ASYM_LOW
        and smile < SMILE_COEFF_HIGH
    )


def _absolute_angry_signal(ear, brow_dist, mar, mouth_asym, smile, upper_lip_raise):
    if smile >= SMILE_COEFF_HIGH or mouth_asym >= MOUTH_ASYM_HIGH:
        return False
    if upper_lip_raise < UPPER_LIP_RAISE_BLOCK_ANGER:
        return False
    brow_path = brow_dist < BROW_DIST_FURROWED and ear < EAR_HIGH
    eyes_path = ear < EAR_TENSE and mar < MAR_MODERATE and brow_dist < BROW_DIST_RELAXED
    return brow_path or eyes_path


def _delta_disgust_signal(d_lip_raise, d_mar, d_asym, d_smile):
    return (
        d_lip_raise < -DELTA_LIP_RAISE_STRONG
        and d_mar < DELTA_MAR_MOD
        and d_asym < DELTA_ASYM_LOW
        and d_smile < DELTA_SMILE_HIGH
    )


def _delta_angry_signal(d_ear, d_brow, d_mar, d_asym, d_smile, d_lip_raise):
    if d_smile >= -0.002 or d_asym >= DELTA_ASYM_HIGH:
        return False
    if d_lip_raise < -DELTA_LIP_RAISE:
        return False
    brow_path = d_brow < DELTA_BROW_FURROWED and d_ear < DELTA_EAR_HIGH
    eyes_path = d_ear < DELTA_EAR_TENSE and d_mar < DELTA_MAR_MOD and d_brow < DELTA_BROW_SAD
    return brow_path or eyes_path


def _absolute_sad_signal(ear, smile, brow_dist, mouth_asym, upper_lip_raise):
    return (
        smile < SMILE_COEFF_SAD
        and ear >= EAR_TENSE
        and brow_dist >= BROW_DIST_RELAXED
        and mouth_asym < MOUTH_ASYM_LOW
        and upper_lip_raise >= UPPER_LIP_RAISE_BLOCK_ANGER
    )


def _delta_sad_signal(d_ear, d_smile, d_brow, d_asym, d_lip_raise):
    return (
        d_smile < DELTA_SMILE_SAD
        and d_ear >= DELTA_EAR_LOW
        and d_brow >= DELTA_BROW_SAD
        and d_asym < DELTA_ASYM_HIGH
        and d_lip_raise >= -DELTA_LIP_RAISE
    )


def _classify_absolute(params):
    ear = params["ear_avg"]
    mar = params["mar"]
    smile = params["smile_coeff"]
    mouth_width = params["mouth_width"]
    brow_dist = params["brow_dist"]
    mouth_asym = params.get("mouth_asymmetry", 0.0)
    upper_lip_raise = params.get("upper_lip_raise", 0.0)

    # Surprised: wide eyes + open mouth
    if ear > EAR_HIGH and mar > MAR_HIGH:
        return "Surprised"

    # Happy: elevated mouth corners, or wide mouth from smiling
    if smile > SMILE_COEFF_HIGH and (mar >= MAR_LOW or mouth_width > MOUTH_WIDTH_SMILE):
        return "Happy"

    # Contempt: mouth asymmetry should dominate the decision.
    if (
        mouth_asym > MOUTH_ASYM_HIGH
        and mar < MAR_MODERATE
        and smile > -0.002
        and upper_lip_raise >= UPPER_LIP_RAISE_DISGUST
    ):
        return "Contempt"

    # Disgust: primarily driven by upper-lip raise. Eye/brow tension can co-exist,
    # but lip raise must be the lead signal.
    if _absolute_disgust_signal(upper_lip_raise, mar, mouth_asym, smile):
        return "Disgusted"

    # Angry: furrowed brows and/or tense eyes, but without the upper-lip raise
    # signature of disgust.
    if _absolute_angry_signal(ear, brow_dist, mar, mouth_asym, smile, upper_lip_raise):
        return "Angry"

    # Sad: downturned mouth with relaxed or slightly raised brows. Use a more
    # conservative absolute smile threshold so neutral faces are not swallowed.
    if _absolute_sad_signal(ear, smile, brow_dist, mouth_asym, upper_lip_raise):
        return "Sad"

    return "Neutral"


def _classify_delta(params, baseline):
    d_ear = params["ear_avg"] - baseline["ear_avg"]
    d_mar = params["mar"] - baseline["mar"]
    d_smile = params["smile_coeff"] - baseline["smile_coeff"]
    d_mw = params["mouth_width"] - baseline["mouth_width"]
    d_brow = params["brow_dist"] - baseline["brow_dist"]
    d_asym = params.get("mouth_asymmetry", 0.0) - baseline.get("mouth_asymmetry", 0.0)
    d_lip_raise = params.get("upper_lip_raise", 0.0) - baseline.get("upper_lip_raise", 0.0)

    # Surprised: eyes wider than neutral + mouth opened wide
    if d_ear > DELTA_EAR_HIGH and d_mar > DELTA_MAR_HIGH:
        return "Surprised"

    # Happy: smile increase + (mouth opens a bit OR mouth widens)
    if d_smile > DELTA_SMILE_HIGH and (d_mar > DELTA_MAR_LOW or d_mw > DELTA_MW_SMILE):
        return "Happy"

    # Contempt: asymmetric mouth movement should dominate while the mouth stays
    # fairly closed and the upper lip does not strongly lift.
    if (
        d_asym > DELTA_ASYM_HIGH
        and d_mar < DELTA_MAR_MOD
        and d_smile > -0.003
        and d_lip_raise > -DELTA_LIP_RAISE
    ):
        return "Contempt"

    # Disgust: strong upper-lip raise is the primary cue.
    if _delta_disgust_signal(d_lip_raise, d_mar, d_asym, d_smile):
        return "Disgusted"

    # Angry: brows/eyes lead the decision, and strong lip raise blocks it so
    # disgust wins when both share some facial tension.
    if _delta_angry_signal(d_ear, d_brow, d_mar, d_asym, d_smile, d_lip_raise):
        return "Angry"

    # Sad: mouth corners drop below the user's neutral baseline while brows stay
    # relaxed. This is intentionally softer than the absolute rule because
    # downturned neutral mouths are common.
    if _delta_sad_signal(d_ear, d_smile, d_brow, d_asym, d_lip_raise):
        return "Sad"

    return "Neutral"
