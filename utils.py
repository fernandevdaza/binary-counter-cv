import time
import cv2
import numpy as np

from finger_logic import LEFT_ORDER, RIGHT_ORDER

# ── Colours (BGR) ──────────────────────────────────────────────────────────────
WHITE  = (255, 255, 255)
GREEN  = (0,   220,  80)
RED    = (0,    60, 220)
YELLOW = (0,   220, 220)
CYAN   = (220, 220,   0)
DARK   = (30,   30,  30)

# Short labels shown in the finger-state row
_FINGER_SHORT = {"index": "idx", "middle": "mid", "ring": "rng", "pinky": "pnk"}


class FPSCounter:
    def __init__(self, alpha: float = 0.1):
        self._alpha = alpha
        self._fps: float = 0.0
        self._prev: float = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        instant = 1.0 / max(now - self._prev, 1e-6)
        self._fps = self._alpha * instant + (1 - self._alpha) * self._fps
        self._prev = now
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps


def draw_info_panel(
    frame: np.ndarray,
    stable_binary: str,
    hands: list,           # [(label, landmarks), ...]  sorted left→right
    fps: float,
) -> None:
    """
    Adaptive HUD:
      Row 0 – bits grouped by hand, colour-coded, with separators
      Row 1 – Decimal value
      Row 2 – Hex value  |  bit-width
      Row 3+ – one finger-state row per hand
    """
    from finger_logic import get_finger_states

    h, w = frame.shape[:2]
    n_hands = len(hands)

    # ── Panel height scales with number of hands ───────────────────────────────
    rows_needed = 3 + max(n_hands, 1)   # header rows + finger rows
    panel_h = 44 * rows_needed
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), DARK, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    decimal_val = int(stable_binary, 2) if stable_binary else 0
    n_bits      = len(stable_binary)

    # ── Row 0: bit groups ──────────────────────────────────────────────────────
    y0 = 32
    cv2.putText(frame, "Bits:", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.75, YELLOW, 2)

    # Scale font based on total bits so they always fit
    bit_font  = max(0.45, 1.1 - n_bits * 0.025)
    bit_step  = max(16, int(28 - n_bits * 0.4))
    x_cursor  = 100

    for g, (bit_group_start, bit_group_end) in enumerate(
        zip(range(0, n_bits, 4), range(4, n_bits + 1, 4))
    ):
        group = stable_binary[bit_group_start:bit_group_end]
        for bit in group:
            colour = GREEN if bit == "1" else RED
            cv2.putText(frame, bit, (x_cursor, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, bit_font, colour, 2)
            x_cursor += bit_step
        # separator between groups
        if bit_group_end < n_bits:
            cv2.putText(frame, "|", (x_cursor, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, bit_font, WHITE, 1)
            x_cursor += bit_step // 2

    # ── Row 1: Decimal ─────────────────────────────────────────────────────────
    y1 = y0 + 38
    cv2.putText(frame, f"Decimal: {decimal_val}", (10, y1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)

    # ── Row 2: Hex + bit width ─────────────────────────────────────────────────
    y2 = y1 + 36
    hex_digits = max(2, (n_bits + 3) // 4)
    cv2.putText(frame, f"Hex: 0x{decimal_val:0{hex_digits}X}    [{n_bits} bits / 2^{n_bits}={2**n_bits}]",
                (10, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, CYAN, 2)

    # ── Rows 3+: per-hand finger states ───────────────────────────────────────
    for i, (label, landmarks) in enumerate(hands):
        y = y2 + 36 + i * 36
        states = get_finger_states(landmarks)
        order  = LEFT_ORDER if label == "Left" else RIGHT_ORDER

        hand_tag = f"H{i+1}:{label[0]}"   # e.g. "H1:L", "H2:R"
        cv2.putText(frame, hand_tag, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)

        for j, fname in enumerate(order):
            colour = GREEN if states[fname] == 1 else RED
            cv2.putText(frame, _FINGER_SHORT[fname], (90 + j * 50, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

    # ── FPS (top-right corner) ────────────────────────────────────────────────
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

    # ── No-hands message ──────────────────────────────────────────────────────
    if n_hands == 0:
        cv2.putText(frame, "No hands detected", (10, y2 + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, RED, 2)


def open_camera(index: int = 0) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera (index={index}). "
            "Check that the device is connected and not in use."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap
