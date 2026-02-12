# Threshold constants — tune these to adjust sensitivity

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


def classify_emotion(params):
    """Rule-based emotion classification from facial parameters.

    Returns one of: Happy, Surprised, Angry, Sad, Neutral
    """
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
