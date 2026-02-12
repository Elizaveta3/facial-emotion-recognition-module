# Facial Emotion Recognition

A real-time facial emotion recognition system that uses MediaPipe Face Mesh landmarks and rule-based classification to detect emotions from a webcam feed. The system computes geometric facial parameters (eye openness, mouth shape, brow position, smile curvature) and classifies them into one of five emotions: **Happy**, **Surprised**, **Angry**, **Sad**, or **Neutral**.

It includes per-person baseline calibration so thresholds adapt to individual facial proportions, and exponential moving average (EMA) temporal smoothing to eliminate frame-to-frame label flickering.

## Setup

### Requirements

- Python 3.8+
- A webcam

### Dependencies

Install the Python packages:

```bash
pip install -r requirements.txt
```

The dependencies are:

| Package | Purpose |
|---|---|
| `opencv-python` | Webcam capture, image display, drawing overlays |
| `mediapipe` | Face landmark detection (468-point Face Mesh) |
| `numpy` | Euclidean distance and array operations |

### Face Landmarker Model

The system requires the MediaPipe Face Landmarker model file `face_landmarker.task` in the project root. Download it from the [MediaPipe documentation](https://developers.google.com/mediapipe/solutions/vision/face_landmarker#models) if not already present.

### Running

```bash
python main.py
```

On launch the system opens your webcam, runs a ~3-second calibration phase, then enters the main detection loop. Press **q** at any time to quit.

## System Overview

```
main.py                  Entry point — calibration, main loop, display, export
landmark_utils.py        Extracts geometric parameters from 468 face landmarks
emotion_classifier.py    Rule-based classifier (absolute + delta modes)
calibration.py           Collects neutral-face frames and computes per-person baseline
smoothing.py             Exponential moving average filter for parameter smoothing
```

### `main.py`

Orchestrates the full pipeline:
1. Opens the webcam and initializes the MediaPipe FaceLandmarker in video mode.
2. Runs a calibration phase (`run_calibration`) that collects 90 neutral-face frames (~3 seconds at 30 fps) and computes a per-person baseline. If the user presses `q` during calibration or no face is detected, the system falls back to absolute-threshold mode.
3. Enters the main loop: each frame is captured, landmarks are extracted, raw parameters are smoothed via EMA, and the smoothed values are passed to the classifier along with the baseline (if available).
4. Draws landmark points, the current emotion label, the classification mode (`[CALIBRATED]` or `[ABSOLUTE]`), and live parameter values on the webcam feed.
5. Logs debug output to the console every 30 frames.
6. On exit, writes per-frame data to CSV and JSON files in the `output/` directory.

### `landmark_utils.py`

Computes all geometric facial parameters from the 468-point MediaPipe Face Mesh. Each parameter is normalized by face width or face height so values are scale-invariant. See [Parameters](#parameters) below for details on each one.

### `emotion_classifier.py`

Contains the rule-based classification logic. Operates in two modes:
- **Absolute mode** (`_classify_absolute`) — uses fixed thresholds; works without calibration.
- **Delta mode** (`_classify_delta`) — computes the difference between the current value and the calibrated neutral baseline, then applies delta thresholds. Activated automatically when calibration succeeds.

### `calibration.py`

`BaselineCalibrator` collects facial parameter snapshots over 90 frames while the user holds a neutral expression. It averages each parameter across all collected frames to produce a stable baseline dictionary.

### `smoothing.py`

`ParameterSmoother` applies an exponential moving average (EMA) to each scalar parameter. The formula is:

```
smoothed = alpha * new_value + (1 - alpha) * previous_smoothed
```

The default `alpha=0.3` means each frame contributes 30% of the new value and retains 70% of the previous smoothed value, producing stable output while still responding to real expression changes.

## Parameters

Five scalar parameters are computed from face landmarks each frame:

### Eye Aspect Ratio (EAR)

Measures how open the eyes are.

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```

Where p1-p6 are six points around each eye (outer corner, upper-outer, upper-inner, inner corner, lower-inner, lower-outer). Computed separately for each eye, then averaged. Higher values indicate wider-open eyes; lower values indicate squinting or closed eyes.

### Mouth Aspect Ratio (MAR)

Measures how open the mouth is.

```
MAR = (v_a + v_b + v_c) / (3 * horizontal)
```

Three vertical mouth distances (top-bottom center, upper-inner-left to lower-inner-left, upper-inner-right to lower-inner-right) divided by the horizontal mouth corner distance. Higher values mean a more open mouth.

### Smile Coefficient

Measures mouth corner elevation relative to the mouth center, normalized by face height.

```
smile_coeff = (center_y - corner_avg_y) / face_height
```

Since pixel y-coordinates increase downward, a positive value means the mouth corners are higher than the center (smiling). A negative value means corners are pulled down (frowning). Values are small (typically -0.01 to +0.01) because they are normalized by face height.

### Mouth Width

Horizontal distance between the left and right mouth corners, normalized by face width.

```
mouth_width = ||left_corner - right_corner|| / face_width
```

Increases during smiles as the mouth stretches horizontally. Typical neutral values are around 0.38-0.42.

### Brow Distance

Average distance from the eyebrow to the upper eyelid, normalized by face height.

```
brow_dist = (right_inner + right_mid + left_inner + left_mid) / (4 * face_height)
```

Uses both the inner and mid brow points on each side for robustness. Lower values indicate furrowed or lowered brows (associated with anger or concentration).

## Emotions Recognized

The classifier evaluates rules in priority order and returns the first match:

### 1. Surprised

Wide eyes **and** wide-open mouth.

| Mode | Condition |
|---|---|
| Absolute | `EAR > 0.30` and `MAR > 0.5` |
| Delta | `d_ear > +0.04` and `d_mar > +0.35` |

### 2. Happy

Mouth corners raised **and** (mouth slightly open **or** mouth widened).

| Mode | Condition |
|---|---|
| Absolute | `smile > 0.005` and (`MAR >= 0.1` or `mouth_width > 0.43`) |
| Delta | `d_smile > +0.006` and (`d_mar > -0.01` or `d_mw > +0.015`) |

### 3. Angry

Smile drops below neutral **and** (brows furrowed **or** eyes narrowed with tight mouth). Two detection paths so a strong signal in one dimension is not gated by a borderline other.

| Mode | Condition |
|---|---|
| Absolute | `smile < 0.005` and (`brow_dist < 0.055` or (`EAR < 0.26` and `MAR < 0.15`)) |
| Delta | `d_smile < -0.006` and (`d_brow < -0.02` or (`d_ear < -0.03` and `d_mar < 0.08`)) |

### 4. Sad

Mouth corners pulled down with normal eyes and normal brows. The brow guard prevents angry faces from leaking into Sad.

| Mode | Condition |
|---|---|
| Absolute | `smile < -0.005` and `EAR >= 0.26` and `brow_dist >= 0.055` |
| Delta | `d_smile < -0.006` and `d_ear >= -0.03` and `d_brow >= -0.02` |

### 5. Neutral

Returned when no other rule matches.

## Calibration and Smoothing

### Per-Person Baseline Calibration

**Problem:** Fixed absolute thresholds (e.g., `EAR_LOW = 0.26`) assume everyone has the same resting facial proportions. Someone with naturally narrow eyes may permanently read as "Angry" because their resting EAR is below the threshold.

**Solution:** On startup, the system collects 90 frames (~3 seconds) of the user's neutral face and averages each parameter to build a personal baseline. During classification, the system computes deltas (current value minus baseline) and applies delta thresholds instead. This means "Happy" is defined as *your smile coefficient increasing from your neutral*, not exceeding a fixed number.

**Fallback:** If calibration is skipped (press `q`) or fails (no face detected), the system falls back to absolute-threshold mode and still works.

### Temporal Smoothing (EMA)

**Problem:** MediaPipe landmark positions jitter slightly between frames even when the face is still. This causes parameters to fluctuate, making the emotion label flicker rapidly.

**Solution:** An exponential moving average (EMA) with `alpha = 0.3` smooths each parameter before classification. The smoothed value is a weighted blend of the new reading (30%) and the previous smoothed value (70%). This eliminates jitter while still tracking real expression changes within a few frames.

Both raw and smoothed values are exported so the smoothing effect can be verified.

## Output

### Webcam Overlay

The live video feed displays:
- Green dots on all 468 face landmarks
- The detected emotion label and classification mode (e.g., `Emotion: Happy  [CALIBRATED]`) in yellow
- Live parameter values (EAR, MAR, Smile, Mouth W, Brow D) in white

During calibration, a progress bar and instruction text are shown instead.

### Console Logs

Every 30 frames, a debug line is printed:

```
[CALIBRATED] [     Happy]  EAR=0.287  MAR=0.152  Smile=0.0081  MouthW=0.441  BrowD=0.0612
```

This includes the classification mode, the detected emotion, and all smoothed parameter values.

### CSV Export

Saved to `output/session_<timestamp>.csv`. Each row is one frame:

| Column | Description |
|---|---|
| `frame` | Frame number (1-indexed) |
| `timestamp` | Unix timestamp in milliseconds |
| `emotion` | Detected emotion label |
| `ear_avg_raw` | Raw (unsmoothed) eye aspect ratio |
| `mar_raw` | Raw mouth aspect ratio |
| `smile_coeff_raw` | Raw smile coefficient |
| `mouth_width_raw` | Raw mouth width |
| `brow_dist_raw` | Raw brow distance |
| `ear_avg_smooth` | Smoothed eye aspect ratio |
| `mar_smooth` | Smoothed mouth aspect ratio |
| `smile_coeff_smooth` | Smoothed smile coefficient |
| `mouth_width_smooth` | Smoothed mouth width |
| `brow_dist_smooth` | Smoothed brow distance |

### JSON Export

Saved to `output/session_<timestamp>.json`. Contains metadata and all frame records:

```json
{
  "session": "20260212_143052",
  "calibration": {
    "enabled": true,
    "baseline": {
      "ear_avg": 0.285,
      "mar": 0.098,
      "mouth_width": 0.401,
      "smile_coeff": 0.00012,
      "brow_dist": 0.0589
    }
  },
  "smoothing": {
    "enabled": true,
    "alpha": 0.3
  },
  "frames": [
    {
      "frame": 1,
      "timestamp": 1739367052000,
      "emotion": "Neutral",
      "ear_avg_raw": 0.2853,
      "mar_raw": 0.0981,
      "smile_coeff_raw": 0.00015,
      "mouth_width_raw": 0.4012,
      "brow_dist_raw": 0.05891,
      "ear_avg_smooth": 0.2853,
      "mar_smooth": 0.0981,
      "smile_coeff_smooth": 0.00015,
      "mouth_width_smooth": 0.4012,
      "brow_dist_smooth": 0.05891
    }
  ]
}
```

When calibration is skipped, `calibration.enabled` is `false` and `calibration.baseline` is `null`.
