import csv
import json
import os
import re
import time

import cv2
import mediapipe as mp
import numpy as np

from calibration import BaselineCalibrator
from emotion_classifier import classify_emotion
from landmark_utils import extract_all_parameters, landmarks_to_list
from smoothing import ParameterSmoother


MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

CSV_FIELDS = [
    "frame", "timestamp", "emotion",
    "ear_avg_raw", "mar_raw", "smile_coeff_raw", "mouth_width_raw", "brow_dist_raw", "mouth_asymmetry_raw", "upper_lip_raise_raw",
    "ear_avg_smooth", "mar_smooth", "smile_coeff_smooth", "mouth_width_smooth", "brow_dist_smooth", "mouth_asymmetry_smooth", "upper_lip_raise_smooth",
]

SCALAR_KEYS = ["ear_avg", "mar", "smile_coeff", "mouth_width", "brow_dist", "mouth_asymmetry", "upper_lip_raise"]


class CameraUnavailableError(Exception):
    pass


class FaceNotFoundError(Exception):
    pass


def _create_landmarker_options():
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    return FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def run_calibration(cap, landmarker, calibrator, return_face_coordinates=False):
    print("CALIBRATION: Keep a neutral face for ~3 seconds...")
    last_landmarks_3d = None

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

        cv2.putText(frame, "CALIBRATION - keep a neutral face",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        bar_x, bar_y, bar_w, bar_h = 30, 70, w - 60, 25
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (255, 255, 255), 2)
        fill_w = int(bar_w * progress / 100)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                      (0, 255, 0), -1)
        cv2.putText(frame, f"{progress}%", (bar_x + bar_w // 2 - 20, bar_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        if not results.face_landmarks:
            cv2.putText(frame, "No face detected - move into frame",
                        (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            face_lms = results.face_landmarks[0]
            landmarks_3d = np.array([
                [lm.x * w, lm.y * h, lm.z * w]
                for lm in face_lms
            ])
            params = extract_all_parameters(landmarks_3d)
            calibrator.add_frame(params)
            last_landmarks_3d = landmarks_3d

        cv2.imshow("Facial Emotion Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Calibration cancelled by user.")
            return (None, None) if return_face_coordinates else None

    if not calibrator.frames:
        print("Calibration failed - no face was detected.")
        return (None, None) if return_face_coordinates else None

    baseline = calibrator.compute_baseline()
    print("CALIBRATION COMPLETE - baseline values:")
    for k, v in baseline.items():
        print(f"  {k}: {v:.5f}")

    if return_face_coordinates:
        return baseline, last_landmarks_3d
    return baseline


def capture_face_profile(num_frames=90):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise CameraUnavailableError("Camera is unavailable.")

    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    try:
        with FaceLandmarker.create_from_options(_create_landmarker_options()) as landmarker:
            calibrator = BaselineCalibrator(num_frames=num_frames)
            baseline, landmarks_3d = run_calibration(
                cap,
                landmarker,
                calibrator,
                return_face_coordinates=True,
            )
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if baseline is None or landmarks_3d is None:
        raise FaceNotFoundError("Face not found.")

    return {
        "baseline": baseline,
        "landmarks_3d": landmarks_to_list(landmarks_3d),
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def parse_face_profile(face_coordinates):
    if not face_coordinates:
        return None
    try:
        profile = json.loads(face_coordinates)
    except json.JSONDecodeError:
        return None
    if not isinstance(profile, dict):
        return None
    return profile


def serialize_face_profile(profile):
    return json.dumps(profile, ensure_ascii=False)


def _safe_session_owner(value):
    if not value:
        return None
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "user"


def run_emotion_recognition(face_profile=None, save_outputs=True, session_owner=None):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise CameraUnavailableError("Camera is unavailable.")

    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    session_id = time.strftime("%Y%m%d_%H%M%S")
    json_records = []
    frame_count = 0
    smoother = ParameterSmoother(alpha=0.3)
    csv_file = None
    csv_writer = None
    csv_path = None
    json_path = None
    baseline = face_profile.get("baseline") if face_profile else None

    if save_outputs:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        safe_owner = _safe_session_owner(session_owner)
        owner_prefix = f"{safe_owner}_" if safe_owner else ""
        csv_path = os.path.join(OUTPUT_DIR, f"session_{owner_prefix}{session_id}.csv")
        json_path = os.path.join(OUTPUT_DIR, f"session_{owner_prefix}{session_id}.json")
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        csv_writer.writeheader()

    try:
        with FaceLandmarker.create_from_options(_create_landmarker_options()) as landmarker:
            if baseline is None:
                calibrator = BaselineCalibrator(num_frames=90)
                baseline = run_calibration(cap, landmarker, calibrator)

            if baseline is not None:
                mode = "CALIBRATED"
            else:
                mode = "ABSOLUTE"
                baseline = None
            print(f"Running in [{mode}] mode.\n")

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

                    landmarks_3d = np.array([
                        [lm.x * w, lm.y * h, lm.z * w]
                        for lm in face_lms
                    ])

                    for x, y, *_ in landmarks_3d.astype(int):
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                    raw_params = extract_all_parameters(landmarks_3d)
                    smoothed = smoother.update(raw_params)
                    emotion = classify_emotion(smoothed, baseline)

                    frame_count += 1
                    if frame_count % 30 == 0:
                        print(f"[{mode}] [{emotion:>10s}]  EAR={smoothed['ear_avg']:.3f}  "
                              f"MAR={smoothed['mar']:.3f}  Smile={smoothed['smile_coeff']:.4f}  "
                              f"MouthW={smoothed['mouth_width']:.3f}  BrowD={smoothed['brow_dist']:.4f}  "
                              f"Asym={smoothed['mouth_asymmetry']:.4f}  LipRaise={smoothed['upper_lip_raise']:.3f}")

                    record = {
                        "frame": frame_count,
                        "timestamp": timestamp_ms,
                        "emotion": emotion,
                    }
                    for k in SCALAR_KEYS:
                        record[f"{k}_raw"] = round(raw_params[k], 5)
                        record[f"{k}_smooth"] = round(smoothed[k], 5)

                    if save_outputs:
                        csv_writer.writerow(record)
                        json_record = dict(record)
                        json_record["landmarks_3d"] = landmarks_to_list(landmarks_3d)
                        json_records.append(json_record)

                    y_offset = 30
                    lines = [
                        f"Emotion: {emotion}  [{mode}]",
                        f"EAR: {smoothed['ear_avg']:.3f}",
                        f"MAR: {smoothed['mar']:.3f}",
                        f"Smile: {smoothed['smile_coeff']:.4f}",
                        f"Mouth W: {smoothed['mouth_width']:.3f}",
                        f"Brow D: {smoothed['brow_dist']:.4f}",
                        f"Asym: {smoothed['mouth_asymmetry']:.4f}",
                        f"LipRaise: {smoothed['upper_lip_raise']:.3f}",
                    ]
                    for i, line in enumerate(lines):
                        color = (0, 255, 255) if i == 0 else (255, 255, 255)
                        cv2.putText(frame, line, (10, y_offset + i * 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                cv2.imshow("Facial Emotion Recognition", frame)
                key = cv2.waitKey(10)
                if key & 0xFF == ord("q") or key == 27:
                    break
    finally:
        if csv_file is not None:
            csv_file.close()
        cap.release()
        cv2.destroyAllWindows()

    if save_outputs:
        calibration_enabled = baseline is not None
        json_output = {
            "session": session_id,
            "owner": session_owner,
            "calibration": {
                "enabled": calibration_enabled,
                "baseline": baseline,
            },
            "smoothing": {
                "enabled": True,
                "alpha": smoother.alpha,
            },
            "landmarks_schema": {
                "format": "[x_px, y_px, z_px]",
                "count": 478,
                "z_reference": "face_centre_depth_same_scale_as_x",
            },
            "frames": json_records,
        }
        with open(json_path, "w") as f:
            json.dump(json_output, f, indent=2)
        print(f"\nResults saved to:\n  CSV:  {csv_path}\n  JSON: {json_path}")
