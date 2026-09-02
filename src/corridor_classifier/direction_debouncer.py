from typing import Hashable, List, Optional, Sequence


class ConsecutiveConfirmDebouncer:
    """Holds a stable output tuple. A disagreeing raw tuple must be seen for
    at least ``min_confirm_frames`` consecutive updates before the published
    value switches to it.

    This exists to reject single-frame noise (most visibly right after a
    reset, e.g. just after a turn, when a single bad frame would otherwise
    immediately become the published value).

    The whole tuple is debounced atomically: every slot switches together
    based on the complete raw tuple, not slot by slot. Debouncing each slot
    independently was tried first, but let each slot switch at a different
    time, so the published tuple could pass through combinations the model
    never actually predicted at any single instant (e.g. flickering through
    several intersection class names in a row while the raw per-frame
    prediction never changed), which downstream consumers such as
    scenario_navigation could misread as several distinct real transitions
    happening at once.

    An earlier version also enforced a minimum hold period between switches
    (``confirm_frames``), with a bypass for values matching
    scenario_navigation's current target. It was removed: scenario_navigation
    is /passage_type's only consumer, it already needs only
    ``min_confirm_frames`` worth of evidence to advance (via the bypass), and
    a non-matching reading flickering during the hold caused no incorrect
    behavior there (compareScenarioAndPassageType simply never matched it).
    The hold's only real effect was to delay -- and, for genuinely short
    segments, sometimes fully suppress -- transitions unrelated to the
    current target, without changing scenario_navigation's behavior.
    """

    def __init__(
        self,
        initial: Sequence[Hashable],
        min_confirm_frames: int = 3,
    ):
        if min_confirm_frames < 1:
            raise ValueError("min_confirm_frames must be at least 1")
        self.min_confirm_frames = int(min_confirm_frames)
        self._current: List[Hashable] = list(initial)
        self._candidate: List[Hashable] = list(self._current)
        self._candidate_streak: int = 0

    def update(self, values: Sequence[Hashable]) -> List[Hashable]:
        values = list(values)
        if len(values) != len(self._current):
            raise ValueError("values length must match the debouncer's slot count")

        if values != self._current:
            if values == self._candidate:
                self._candidate_streak += 1
            else:
                self._candidate = values
                self._candidate_streak = 1
        else:
            self._candidate = list(values)
            self._candidate_streak = 0

        if values != self._current and self._candidate_streak >= self.min_confirm_frames:
            self._current = values
            self._candidate_streak = 0
        return list(self._current)

    def reset(self, initial: Optional[Sequence[Hashable]] = None) -> None:
        if initial is not None:
            self._current = list(initial)
        self._candidate = list(self._current)
        self._candidate_streak = 0
