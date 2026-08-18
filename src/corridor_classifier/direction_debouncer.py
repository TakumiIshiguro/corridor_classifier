from typing import Hashable, List, Optional, Sequence


class ConsecutiveConfirmDebouncer:
    """Holds a stable output and only accepts a new value once it has been
    seen for ``confirm_frames`` consecutive updates in a row.

    This is deliberately not a moving average or majority vote over a
    window: given the robot's travel speed, the correct output is expected
    to hold for several consecutive frames, so a single frame that
    disagrees with an otherwise stable stream is more likely to be noise
    than a genuine transition. Averaging/voting over a fixed window was
    measured to slightly *hurt* accuracy (see
    scripts/compare_temporal_smoothing.py); requiring sustained agreement
    before switching is a different mechanism aimed at output stability,
    not at improving per-frame accuracy.

    Each position in the sequence is debounced independently (e.g. one
    slot per front/left/right direction), so a change in one direction
    does not reset the confirmation count of the others.
    """

    def __init__(self, initial: Sequence[Hashable], confirm_frames: int = 2):
        if confirm_frames < 1:
            raise ValueError("confirm_frames must be at least 1")
        self.confirm_frames = int(confirm_frames)
        self._current: List[Hashable] = list(initial)
        self._candidate: List[Hashable] = list(self._current)
        self._candidate_count: List[int] = [0] * len(self._current)

    def update(self, values: Sequence[Hashable]) -> List[Hashable]:
        if len(values) != len(self._current):
            raise ValueError("values length must match the debouncer's slot count")
        for index, value in enumerate(values):
            if value == self._current[index]:
                self._candidate[index] = value
                self._candidate_count[index] = 0
                continue
            if value == self._candidate[index]:
                self._candidate_count[index] += 1
            else:
                self._candidate[index] = value
                self._candidate_count[index] = 1
            if self._candidate_count[index] >= self.confirm_frames:
                self._current[index] = value
                self._candidate_count[index] = 0
        return list(self._current)

    def reset(self, initial: Optional[Sequence[Hashable]] = None) -> None:
        if initial is not None:
            self._current = list(initial)
        self._candidate = list(self._current)
        self._candidate_count = [0] * len(self._current)
