import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

_vision = mp.tasks.vision
_conn   = _vision.HandLandmarksConnections

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

WRIST = 0


def _ensure_model() -> str:
    if not os.path.exists(MODEL_PATH):
        print(f"[INFO] Downloading hand landmarker model → '{MODEL_PATH}' …")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Download complete.")
    return MODEL_PATH


class HandDetector:
    def __init__(self, max_hands: int = 8,
                 detection_confidence: float = 0.7,
                 tracking_confidence: float = 0.5):
        options = _vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=_ensure_model()),
            running_mode=_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = _vision.HandLandmarker.create_from_options(options)
        self._t0 = time.monotonic()

    def process_and_draw(self, frame_bgr: np.ndarray) -> list:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int((time.monotonic() - self._t0) * 1000)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return []

        hands = []
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            raw_label = handedness[0].category_name
            label = "Right" if raw_label == "Left" else "Left"
            hands.append((label, landmarks))
            self._draw_hand(frame_bgr, landmarks)

        # Sort left-to-right so bit order is visually consistent
        hands.sort(key=lambda h: h[1][WRIST].x)
        return hands

    def close(self) -> None:
        self._landmarker.close()

    def _draw_hand(self, frame_bgr: np.ndarray, landmarks) -> None:
        h, w = frame_bgr.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        for conn in _conn.HAND_CONNECTIONS:
            cv2.line(frame_bgr, pts[conn.start], pts[conn.end],
                     (80, 200, 80), 2, cv2.LINE_AA)

        TIPS = {4, 8, 12, 16, 20}
        for i, (x, y) in enumerate(pts):
            r = 6 if i in TIPS else 4
            cv2.circle(frame_bgr, (x, y), r, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame_bgr, (x, y), r, (0, 120, 255),   1,  cv2.LINE_AA)
            #cv2.putText(frame_bgr, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
