from typing import Hashable, List, Optional, Sequence


class ConsecutiveConfirmDebouncer:
    """Holds a stable output tuple. A disagreeing raw tuple must be seen
    for at least ``min_confirm_frames`` consecutive updates before it can
    switch at all, and then it is held for at least ``confirm_frames``
    updates before accepting another switch.

    Earlier versions of this debouncer required either a long run of
    ``confirm_frames`` consecutive matching updates before switching, or
    switched on a single disagreeing frame with no evidence at all. Both
    turned out to be problems in practice: the long-confirmation version
    made every genuine transition lag by up to ``confirm_frames`` frames,
    while the zero-evidence version could lock onto a single noisy
    misclassification (most visibly right after a reset, e.g. just after a
    turn, when a single bad frame would otherwise immediately become the
    published value for the entire next hold period). ``min_confirm_frames``
    is deliberately much smaller than ``confirm_frames``: it exists to
    reject single-frame noise, not to make transitions wait for
    ``confirm_frames`` like the old design did.

    A raw value can also be marked high-priority via ``update(...,
    bypass_hold=True)`` (e.g. it matches what a downstream consumer such as
    scenario_navigation is currently waiting for). Once it has the same
    ``min_confirm_frames`` worth of evidence, it switches immediately
    instead of waiting out an unrelated, lower-priority switch's
    ``confirm_frames`` hold.

    The whole tuple is debounced atomically: every slot switches together
    based on the complete raw tuple, not slot by slot. Debouncing each slot
    independently was tried first, but let each slot switch at a different
    time, so the published tuple could pass through combinations the model
    never actually predicted at any single instant (e.g. flickering
    through several intersection class names in a row while the raw
    per-frame prediction never changed), which downstream consumers such
    as scenario_navigation could misread as several distinct real
    transitions happening at once.
    """

    def __init__(
        self,
        initial: Sequence[Hashable],
        confirm_frames: int = 2,
        min_confirm_frames: int = 3,
    ):
        if confirm_frames < 1:
            raise ValueError("confirm_frames must be at least 1")
        if min_confirm_frames < 1:
            raise ValueError("min_confirm_frames must be at least 1")
        self.confirm_frames = int(confirm_frames)
        self.min_confirm_frames = int(min_confirm_frames)
        self._current: List[Hashable] = list(initial)
        # Start already past the hold period, so the first *confirmed*
        # candidate after init/reset can switch as soon as it has
        # min_confirm_frames worth of evidence, without also waiting out
        # confirm_frames.
        self._frames_since_switch: int = self.confirm_frames
        self._candidate: List[Hashable] = list(self._current)
        self._candidate_streak: int = 0

    def update(
        self, values: Sequence[Hashable], bypass_hold: bool = False
    ) -> List[Hashable]:
        """``bypass_hold=True`` marks ``values`` as coming from a
        high-priority source. Once ``values`` has accumulated
        ``min_confirm_frames`` worth of evidence, it switches immediately
        even mid-hold, instead of also waiting out ``confirm_frames``.
        """
        values = list(values)
        if len(values) != len(self._current):
            raise ValueError("values length must match the debouncer's slot count")
        self._frames_since_switch += 1

        if values != self._current:
            if values == self._candidate:
                self._candidate_streak += 1
            else:
                self._candidate = values
                self._candidate_streak = 1
        else:
            self._candidate = list(values)
            self._candidate_streak = 0

        should_switch = (
            values != self._current
            and self._candidate_streak >= self.min_confirm_frames
            and (bypass_hold or self._frames_since_switch >= self.confirm_frames)
        )
        if should_switch:
            self._current = values
            self._frames_since_switch = 0
            self._candidate_streak = 0
        return list(self._current)

    def reset(self, initial: Optional[Sequence[Hashable]] = None) -> None:
        if initial is not None:
            self._current = list(initial)
        self._frames_since_switch = self.confirm_frames
        self._candidate = list(self._current)
        self._candidate_streak = 0
