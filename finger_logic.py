from collections import deque

FINGERS = {
    "index":  (8,  6),
    "middle": (12, 10),
    "ring":   (16, 14),
    "pinky":  (20, 18),
}

LEFT_ORDER  = ["index", "middle", "ring",  "pinky"]
RIGHT_ORDER = ["pinky", "ring",   "middle", "index"]

NOISE_MARGIN = 0.02


def get_finger_states(landmarks) -> dict:
    if landmarks is None:
        return {name: 0 for name in LEFT_ORDER}

    states = {}
    for name in LEFT_ORDER:
        tip_idx, pip_idx = FINGERS[name]
        tip_y = landmarks[tip_idx].y
        pip_y = landmarks[pip_idx].y
        states[name] = 1 if tip_y < pip_y - NOISE_MARGIN else 0

    return states


def hand_bits(label: str, landmarks) -> list[int]:
    states = get_finger_states(landmarks)
    order = LEFT_ORDER if label == "Left" else RIGHT_ORDER
    return [states[f] for f in order]


def build_binary_string(hands: list) -> str:
    bits = []
    for label, landmarks in hands:
        bits.extend(hand_bits(label, landmarks))
    return "".join(map(str, bits))


def binary_to_decimal(binary_str: str) -> int:
    return int(binary_str, 2) if binary_str else 0


class SmoothedBinary:

    def __init__(self, window: int = 3):
        self._window = window
        self._history: deque = deque(maxlen=window)
        self._stable: str = ""

    def update(self, binary_str: str) -> str:
        if self._history and len(self._history[-1]) != len(binary_str):
            self._history.clear()
            self._stable = binary_str

        self._history.append(binary_str)
        if len(self._history) == self._window and len(set(self._history)) == 1:
            self._stable = binary_str
        return self._stable
