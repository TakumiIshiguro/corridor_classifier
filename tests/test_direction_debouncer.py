import pytest

from corridor_classifier.direction_debouncer import ConsecutiveConfirmDebouncer


def test_does_not_switch_on_a_single_disagreeing_frame():
    debouncer = ConsecutiveConfirmDebouncer(
        initial=(True, False, False), min_confirm_frames=3
    )

    assert debouncer.update((False, True, False)) == [True, False, False]


def test_single_frame_right_after_reset_does_not_immediately_lock_in():
    # This is the scenario that motivated min_confirm_frames: right after a
    # reset (e.g. just finished a turn), a single noisy misclassification
    # used to become the published value for the entire next hold period.
    debouncer = ConsecutiveConfirmDebouncer(initial=(True,), min_confirm_frames=3)
    debouncer.reset()

    assert debouncer.update((False,)) == [True]  # 1 noisy frame: not enough evidence
    assert debouncer.update((True,)) == [True]  # noise clears; still at the old value


def test_switches_after_min_confirm_frames_consecutive_matches():
    debouncer = ConsecutiveConfirmDebouncer(initial=(True,), min_confirm_frames=3)

    assert debouncer.update((False,)) == [True]  # 1st matching frame
    assert debouncer.update((False,)) == [True]  # 2nd
    assert debouncer.update((False,)) == [False]  # 3rd: enough evidence


def test_switches_on_the_very_next_frame_once_confirmed():
    debouncer = ConsecutiveConfirmDebouncer(initial=(True,), min_confirm_frames=1)

    assert debouncer.update((False,)) == [False]
    assert debouncer.update((True,)) == [True]  # no hold period: switches back immediately


def test_a_broken_streak_must_restart_from_scratch():
    debouncer = ConsecutiveConfirmDebouncer(initial=(True,), min_confirm_frames=3)

    assert debouncer.update((False,)) == [True]  # streak=1
    assert debouncer.update((False,)) == [True]  # streak=2
    assert debouncer.update((True,)) == [True]  # agrees with current: streak resets
    assert debouncer.update((False,)) == [True]  # streak=1 again, not 3
    assert debouncer.update((False,)) == [True]  # streak=2
    assert debouncer.update((False,)) == [False]  # streak=3: switches


def test_reset_clears_the_candidate_streak():
    debouncer = ConsecutiveConfirmDebouncer(initial=(True,), min_confirm_frames=3)
    debouncer.update((False,))  # streak=1

    debouncer.reset()

    assert debouncer.update((False,)) == [True]  # streak restarts at 1, not 2
    assert debouncer.update((False,)) == [True]
    assert debouncer.update((False,)) == [False]


def test_reset_with_initial_changes_the_current_value():
    debouncer = ConsecutiveConfirmDebouncer(initial=(True,), min_confirm_frames=1)

    debouncer.reset((False,))

    assert debouncer.update((False,)) == [False]


def test_directions_switch_as_one_atomic_tuple():
    debouncer = ConsecutiveConfirmDebouncer(initial=(True, True), min_confirm_frames=1)

    # Both slots flip together once the full tuple has enough evidence.
    assert debouncer.update((False, False)) == [False, False]


def test_mismatched_length_raises():
    debouncer = ConsecutiveConfirmDebouncer(initial=(True, True))
    with pytest.raises(ValueError):
        debouncer.update((True,))


def test_rejects_nonpositive_min_confirm_frames():
    with pytest.raises(ValueError):
        ConsecutiveConfirmDebouncer(initial=(True,), min_confirm_frames=0)
