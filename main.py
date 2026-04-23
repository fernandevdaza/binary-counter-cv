import sys
import cv2

from hand_detector import HandDetector
from finger_logic  import build_binary_string, binary_to_decimal, SmoothedBinary
from utils         import FPSCounter, draw_info_panel, open_camera

CAMERA_INDEX = 0
MAX_HANDS    = 8
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

    print(f"[INFO] Binary Hand Counter iniciado (max {MAX_HANDS} manos). Presionar Q/ESC para salir.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARNING] No se pudo leer el frame")
            continue

        frame = cv2.flip(frame, 1)

        hands = detector.process_and_draw(frame)

        raw_binary    = build_binary_string(hands)
        stable_binary = smoother.update(raw_binary)

        fps = fps_counter.tick()
        draw_info_panel(frame, stable_binary, hands, fps)

        cv2.imshow("Binary Hand Counter", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in QUIT_KEYS:
            break
        if key == SAVE_KEY:
            decimal_val = binary_to_decimal(stable_binary)
            filename = f"capture_{decimal_val}.png"
            cv2.imwrite(filename, frame)
            print(f"[INFO] Guardado {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Saliendo!")


if __name__ == "__main__":
    main()
