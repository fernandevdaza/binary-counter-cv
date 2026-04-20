from collections import deque

# Landmark indices for each finger (tip, pip)
FINGERS = {
    "index":  (8,  6),
    "middle": (12, 10),
    "ring":   (16, 14),
    "pinky":  (20, 18),
}

# Left hand bits go index→pinky, right hand bits go pinky→index
LEFT_ORDER  = ["index", "middle", "ring",  "pinky"]
RIGHT_ORDER = ["pinky", "ring",   "middle", "index"]

NOISE_MARGIN = 0.02


def get_finger_states(landmarks) -> dict:
    """
    Returns {finger_name: 0|1} for the four non-thumb fingers of one hand.
    Pass None to get all zeros (hand not present).
    """
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
    """
    Return the 4 bits for one hand as a list of ints.
    Left  → index, middle, ring,  pinky
    Right → pinky, ring,   middle, index
    """
    states = get_finger_states(landmarks)
    order = LEFT_ORDER if label == "Left" else RIGHT_ORDER
    return [states[f] for f in order]


def build_binary_string(hands: list) -> str:
    """
    Build an N*4 bit binary string from a list of (label, landmarks) tuples.
    Hands must already be sorted left-to-right (done by HandDetector).
    Returns '' if no hands detected.
    """
    bits = []
    for label, landmarks in hands:
        bits.extend(hand_bits(label, landmarks))
    return "".join(map(str, bits))


def binary_to_decimal(binary_str: str) -> int:
    return int(binary_str, 2) if binary_str else 0


class SmoothedBinary:
    """
    Only updates the displayed value when the same binary string (same length
    AND same value) has been seen for `window` consecutive frames.
    Resets automatically when the number of detected hands changes.
    """

    def __init__(self, window: int = 3):
        self._window = window
        self._history: deque = deque(maxlen=window)
        self._stable: str = ""

    def update(self, binary_str: str) -> str:
        # If bit-length changed, flush history so we don't mix lengths
        if self._history and len(self._history[-1]) != len(binary_str):
            self._history.clear()
            self._stable = binary_str   # show immediately to avoid blank frame

        self._history.append(binary_str)
        if len(self._history) == self._window and len(set(self._history)) == 1:
            self._stable = binary_str
        return self._stable
