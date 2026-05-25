SMILE_COEFF_HIGH = 0.005
SMILE_COEFF_NEGATIVE = -0.005
SMILE_COEFF_SAD = -0.012

EAR_HIGH = 0.30
EAR_LOW = 0.26
EAR_TENSE = 0.245

MAR_HIGH = 0.50
MAR_MODERATE = 0.15
MAR_LOW = 0.10
MAR_CLOSED = 0.08

MOUTH_WIDTH_SMILE = 0.43

BROW_DIST_FURROWED = 0.118
BROW_DIST_RELAXED = 0.128

MOUTH_ASYM_HIGH = 0.013
MOUTH_ASYM_LOW = 0.009
UPPER_LIP_RAISE_DISGUST = 0.040
UPPER_LIP_RAISE_BLOCK_ANGER = 0.050

DELTA_SMILE_HIGH = 0.006
DELTA_SMILE_LOW = -0.006
DELTA_SMILE_SAD = -0.003
DELTA_EAR_HIGH = 0.04
DELTA_EAR_LOW = -0.03
DELTA_EAR_TENSE = -0.02
DELTA_MAR_HIGH = 0.35
DELTA_MAR_MOD = 0.08
DELTA_MAR_LOW = -0.01
DELTA_MAR_CLOSED = 0.03
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
    "Fear",
    "Surprised",
    "Angry",
    "Sad",
    "Disgusted",
    "Contempt",
    "Neutral",
)


def classify_emotion(params, baseline=None):
    if baseline is None:
        return _classify_absolute(params)
    return _classify_delta(params, baseline)


def _absolute_disgust_signal(ear, brow_dist, mar, smile):
    return (
        ear < EAR_LOW
        and brow_dist < BROW_DIST_FURROWED
        and MAR_LOW < mar < MAR_HIGH
        and smile < SMILE_COEFF_HIGH
    )


def _absolute_angry_signal(ear, brow_dist, mar, mouth_asym, smile, upper_lip_raise):
    if smile >= SMILE_COEFF_HIGH or mouth_asym >= MOUTH_ASYM_HIGH:
        return False
    if upper_lip_raise < UPPER_LIP_RAISE_BLOCK_ANGER:
        return False
    if mar >= MAR_CLOSED:
        return False
    return brow_dist < BROW_DIST_FURROWED and EAR_LOW <= ear <= EAR_HIGH


def _delta_disgust_signal(d_ear, d_brow, d_mar, d_smile):
    return (
        d_ear < DELTA_EAR_LOW
        and d_brow < DELTA_BROW_FURROWED
        and DELTA_MAR_MOD < d_mar < DELTA_MAR_HIGH
        and d_smile < DELTA_SMILE_HIGH
    )


def _delta_angry_signal(d_ear, d_brow, d_mar, d_asym, d_smile, d_lip_raise):
    if d_smile >= -0.002 or d_asym >= DELTA_ASYM_HIGH:
        return False
    if d_lip_raise < -DELTA_LIP_RAISE:
        return False
    if d_mar >= DELTA_MAR_CLOSED:
        return False
    return d_brow < DELTA_BROW_FURROWED and DELTA_EAR_LOW <= d_ear <= DELTA_EAR_HIGH


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


def _absolute_fear_signal(ear, mar, smile, mouth_asym):
    return (
        ear > EAR_HIGH
        and MAR_MODERATE < mar <= MAR_HIGH
        and smile < SMILE_COEFF_HIGH
        and mouth_asym < MOUTH_ASYM_HIGH
    )


def _delta_fear_signal(d_ear, d_mar, d_smile, d_asym):
    return (
        d_ear > DELTA_EAR_HIGH
        and DELTA_MAR_MOD < d_mar <= DELTA_MAR_HIGH
        and d_smile < DELTA_SMILE_HIGH
        and d_asym < DELTA_ASYM_HIGH
    )


def _classify_absolute(params):
    ear = params["ear_avg"]
    mar = params["mar"]
    smile = params["smile_coeff"]
    mouth_width = params["mouth_width"]
    brow_dist = params["brow_dist"]
    mouth_asym = params.get("mouth_asymmetry", 0.0)
    upper_lip_raise = params.get("upper_lip_raise", 0.0)

    if ear > EAR_HIGH and mar > MAR_HIGH:
        return "Surprised"

    if _absolute_fear_signal(ear, mar, smile, mouth_asym):
        return "Fear"

    if smile > SMILE_COEFF_HIGH and (mar >= MAR_LOW or mouth_width > MOUTH_WIDTH_SMILE):
        return "Happy"

    if (
        mouth_asym > MOUTH_ASYM_HIGH
        and mar < MAR_MODERATE
        and smile > -0.002
        and upper_lip_raise >= UPPER_LIP_RAISE_DISGUST
    ):
        return "Contempt"

    if _absolute_disgust_signal(ear, brow_dist, mar, smile):
        return "Disgusted"

    if _absolute_angry_signal(ear, brow_dist, mar, mouth_asym, smile, upper_lip_raise):
        return "Angry"

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

    if d_ear > DELTA_EAR_HIGH and d_mar > DELTA_MAR_HIGH:
        return "Surprised"

    if _delta_fear_signal(d_ear, d_mar, d_smile, d_asym):
        return "Fear"

    if d_smile > DELTA_SMILE_HIGH and (d_mar > DELTA_MAR_LOW or d_mw > DELTA_MW_SMILE):
        return "Happy"

    if (
        d_asym > DELTA_ASYM_HIGH
        and d_mar < DELTA_MAR_MOD
        and d_smile > -0.003
        and d_lip_raise > -DELTA_LIP_RAISE
    ):
        return "Contempt"

    if _delta_disgust_signal(d_ear, d_brow, d_mar, d_smile):
        return "Disgusted"

    if _delta_angry_signal(d_ear, d_brow, d_mar, d_asym, d_smile, d_lip_raise):
        return "Angry"

    if _delta_sad_signal(d_ear, d_smile, d_brow, d_asym, d_lip_raise):
        return "Sad"

    return "Neutral"
