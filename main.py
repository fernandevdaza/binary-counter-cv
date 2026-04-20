"""
Binary Hand Counter — N hands, N×4 bits
========================================
Detects up to MAX_HANDS hands. Each hand contributes 4 bits.
Hands are sorted left-to-right on screen, so the bit order is
always visually consistent regardless of how many people hold up hands.

  1 hand  →  4 bits  (0–15)
  2 hands →  8 bits  (0–255)
  4 hands → 16 bits  (0–65 535)
  8 hands → 32 bits  (0–4 294 967 295)

Controls
--------
  Q / ESC  →  quit
  S        →  save current frame as PNG
"""

import sys
import cv2

from hand_detector import HandDetector
from finger_logic  import build_binary_string, binary_to_decimal, SmoothedBinary
from utils         import FPSCounter, draw_info_panel, open_camera

CAMERA_INDEX = 0
MAX_HANDS    = 8       # raise/lower to taste; MediaPipe handles up to ~8 reliably
SMOOTH_WINDOW = 3
SAVE_KEY  = ord("s")
QUIT_KEYS = {ord("q"), 27}


def main() -> None:
    try:
        cap = open_camera(CAMERA_INDEX)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    detector    = HandDetector(max_hands=MAX_HANDS)
    smoother    = SmoothedBinary(window=SMOOTH_WINDOW)
    fps_counter = FPSCounter(alpha=0.1)

    print(f"[INFO] Binary Hand Counter started (max {MAX_HANDS} hands). Q/ESC to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARNING] Failed to grab frame — retrying…")
            continue

        frame = cv2.flip(frame, 1)

        # ── Detection ─────────────────────────────────────────────────────────
        # hands = [(label, landmarks), ...] sorted left→right
        hands = detector.process_and_draw(frame)

        # ── Binary logic ──────────────────────────────────────────────────────
        raw_binary    = build_binary_string(hands)
        stable_binary = smoother.update(raw_binary)

        # ── HUD ───────────────────────────────────────────────────────────────
        fps = fps_counter.tick()
        draw_info_panel(frame, stable_binary, hands, fps)

        # ── Display ───────────────────────────────────────────────────────────
        cv2.imshow("Binary Hand Counter", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in QUIT_KEYS:
            break
        if key == SAVE_KEY:
            decimal_val = binary_to_decimal(stable_binary)
            filename = f"capture_{decimal_val}.png"
            cv2.imwrite(filename, frame)
            print(f"[INFO] Saved {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Bye!")


if __name__ == "__main__":
    main()
