# Threshold constants — tune these to adjust sensitivity

# Smile coefficient
SMILE_COEFF_HIGH = 0.02
SMILE_COEFF_NEGATIVE = -0.005

# Eye Aspect Ratio
EAR_HIGH = 0.30
EAR_LOW = 0.20

# Mouth Aspect Ratio
MAR_HIGH = 0.5
MAR_MODERATE = 0.15
MAR_LOW = 0.1


def classify_emotion(params):
    """Rule-based emotion classification from facial parameters.

    Returns one of: Happy, Surprised, Angry, Sad, Neutral
    """
    ear = params["ear_avg"]
    mar = params["mar"]
    smile = params["smile_coeff"]

    # Surprised: wide eyes + open mouth
    if ear > EAR_HIGH and mar > MAR_HIGH:
        return "Surprised"

    # Happy: elevated mouth corners + some mouth opening
    if smile > SMILE_COEFF_HIGH and mar >= MAR_LOW:
        return "Happy"

    # Sad: mouth corners pulled down + moderate eyes
    if smile < SMILE_COEFF_NEGATIVE and ear >= EAR_LOW:
        return "Sad"

    # Angry: narrowed eyes + flat mouth + no smile
    if ear < EAR_LOW and smile < SMILE_COEFF_HIGH and mar < MAR_LOW:
        return "Angry"

    return "Neutral"
