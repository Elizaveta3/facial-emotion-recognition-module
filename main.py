import os
import time

import cv2
import mediapipe as mp
import numpy as np

from landmark_utils import extract_all_parameters
from emotion_classifier import classify_emotion

MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")


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

    frame_count = 0

    with FaceLandmarker.create_from_options(options) as landmarker:
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

                # Compute parameters and classify
                params = extract_all_parameters(landmarks)
                emotion = classify_emotion(params)

                # Debug: print raw values every 30 frames for threshold tuning
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"[{emotion:>10s}]  EAR={params['ear_avg']:.3f}  "
                          f"MAR={params['mar']:.3f}  Smile={params['smile_coeff']:.4f}  "
                          f"MouthW={params['mouth_width']:.3f}  BrowD={params['brow_dist']:.4f}")

                # Overlay metrics
                y_offset = 30
                lines = [
                    f"Emotion: {emotion}",
                    f"EAR: {params['ear_avg']:.3f}",
                    f"MAR: {params['mar']:.3f}",
                    f"Smile: {params['smile_coeff']:.4f}",
                    f"Mouth W: {params['mouth_width']:.3f}",
                    f"Brow D: {params['brow_dist']:.4f}",
                ]
                for i, line in enumerate(lines):
                    color = (0, 255, 255) if i == 0 else (255, 255, 255)
                    cv2.putText(frame, line, (10, y_offset + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Facial Emotion Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
