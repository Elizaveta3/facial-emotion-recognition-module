import csv
import json
import os
import time

import cv2
import mediapipe as mp
import numpy as np

from landmark_utils import extract_all_parameters
from emotion_classifier import classify_emotion
from calibration import BaselineCalibrator
from smoothing import ParameterSmoother

MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

CSV_FIELDS = [
    "frame", "timestamp", "emotion",
    "ear_avg_raw", "mar_raw", "smile_coeff_raw", "mouth_width_raw", "brow_dist_raw",
    "ear_avg_smooth", "mar_smooth", "smile_coeff_smooth", "mouth_width_smooth", "brow_dist_smooth",
]

SCALAR_KEYS = ["ear_avg", "mar", "smile_coeff", "mouth_width", "brow_dist"]


def run_calibration(cap, landmarker, calibrator):
    """Run calibration phase, collecting neutral-face frames.

    Returns the baseline dict, or None if the user quits or no face was detected.
    """
    print("CALIBRATION: Keep a neutral face for ~3 seconds...")

    while not calibrator.is_complete():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        progress = calibrator.get_progress()

        # Draw calibration overlay
        cv2.putText(frame, "CALIBRATION — keep a neutral face",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Progress bar
        bar_x, bar_y, bar_w, bar_h = 30, 70, w - 60, 25
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (255, 255, 255), 2)
        fill_w = int(bar_w * progress / 100)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                      (0, 255, 0), -1)
        cv2.putText(frame, f"{progress}%", (bar_x + bar_w // 2 - 20, bar_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        if not results.face_landmarks:
            cv2.putText(frame, "No face detected — move into frame",
                        (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            face_lms = results.face_landmarks[0]
            landmarks = np.array([[lm.x * w, lm.y * h] for lm in face_lms])
            params = extract_all_parameters(landmarks)
            calibrator.add_frame(params)

        cv2.imshow("Facial Emotion Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Calibration cancelled by user.")
            return None

    if not calibrator.frames:
        print("Calibration failed — no face was detected.")
        return None

    baseline = calibrator.compute_baseline()
    print("CALIBRATION COMPLETE — baseline values:")
    for k, v in baseline.items():
        print(f"  {k}: {v:.5f}")
    return baseline


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session_id = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f"session_{session_id}.csv")
    json_path = os.path.join(OUTPUT_DIR, f"session_{session_id}.json")

    json_records = []
    frame_count = 0

    smoother = ParameterSmoother(alpha=0.3)

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    csv_writer.writeheader()

    with FaceLandmarker.create_from_options(options) as landmarker:
        # ── Calibration phase ───────────────────────────────────
        calibrator = BaselineCalibrator(num_frames=90)
        baseline = run_calibration(cap, landmarker, calibrator)

        if baseline is not None:
            mode = "CALIBRATED"
        else:
            mode = "ABSOLUTE"
            baseline = None  # explicit for clarity
        print(f"Running in [{mode}] mode.\n")

        # ── Main loop ───────────────────────────────────────────
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)

            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            if not results.face_landmarks:
                cv2.putText(frame, "No face detected", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                face_lms = results.face_landmarks[0]

                # Convert normalised landmarks to pixel coordinates
                landmarks = np.array([
                    [lm.x * w, lm.y * h] for lm in face_lms
                ])

                # Draw landmark points
                for x, y in landmarks.astype(int):
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                # Compute raw parameters, smooth, then classify
                raw_params = extract_all_parameters(landmarks)
                smoothed = smoother.update(raw_params)
                emotion = classify_emotion(smoothed, baseline)

                # Debug: print raw values every 30 frames
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"[{mode}] [{emotion:>10s}]  EAR={smoothed['ear_avg']:.3f}  "
                          f"MAR={smoothed['mar']:.3f}  Smile={smoothed['smile_coeff']:.4f}  "
                          f"MouthW={smoothed['mouth_width']:.3f}  BrowD={smoothed['brow_dist']:.4f}")

                # Save per-frame record with both raw and smoothed values
                record = {
                    "frame": frame_count,
                    "timestamp": timestamp_ms,
                    "emotion": emotion,
                }
                for k in SCALAR_KEYS:
                    record[f"{k}_raw"] = round(raw_params[k], 5)
                    record[f"{k}_smooth"] = round(smoothed[k], 5)
                csv_writer.writerow(record)
                json_records.append(record)

                # Overlay metrics
                y_offset = 30
                lines = [
                    f"Emotion: {emotion}  [{mode}]",
                    f"EAR: {smoothed['ear_avg']:.3f}",
                    f"MAR: {smoothed['mar']:.3f}",
                    f"Smile: {smoothed['smile_coeff']:.4f}",
                    f"Mouth W: {smoothed['mouth_width']:.3f}",
                    f"Brow D: {smoothed['brow_dist']:.4f}",
                ]
                for i, line in enumerate(lines):
                    color = (0, 255, 255) if i == 0 else (255, 255, 255)
                    cv2.putText(frame, line, (10, y_offset + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Facial Emotion Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    csv_file.close()

    # Build JSON with metadata
    calibration_enabled = mode == "CALIBRATED"
    json_output = {
        "session": session_id,
        "calibration": {
            "enabled": calibration_enabled,
            "baseline": baseline,
        },
        "smoothing": {
            "enabled": True,
            "alpha": smoother.alpha,
        },
        "frames": json_records,
    }
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nResults saved to:\n  CSV:  {csv_path}\n  JSON: {json_path}")


if __name__ == "__main__":
    main()
