import pytest

from corridor_classifier.direction_debouncer import ConsecutiveConfirmDebouncer


def test_single_disagreeing_frame_is_ignored():
    debouncer = ConsecutiveConfirmDebouncer(initial=(True, False, False), confirm_frames=2)

    assert debouncer.update((False, False, False)) == [True, False, False]
    assert debouncer.update((True, False, False)) == [True, False, False]


def test_sustained_disagreement_switches_after_confirm_frames():
    debouncer = ConsecutiveConfirmDebouncer(initial=(True, False, False), confirm_frames=2)

    assert debouncer.update((False, False, False)) == [True, False, False]
    assert debouncer.update((False, False, False)) == [False, False, False]


def test_flip_flopping_candidate_never_confirms():
    debouncer = ConsecutiveConfirmDebouncer(initial=(True,), confirm_frames=2)

    assert debouncer.update((False,)) == [True]
    assert debouncer.update((True,)) == [True]
    assert debouncer.update((False,)) == [True]
    assert debouncer.update((True,)) == [True]


def test_directions_debounce_independently():
    debouncer = ConsecutiveConfirmDebouncer(initial=(True, True), confirm_frames=2)

    # front flips twice in a row (confirms); left flips once then reverts.
    assert debouncer.update((False, False)) == [True, True]
    assert debouncer.update((False, True)) == [False, True]


def test_confirm_frames_one_switches_immediately():
    debouncer = ConsecutiveConfirmDebouncer(initial=(False,), confirm_frames=1)

    assert debouncer.update((True,)) == [True]


def test_reset_restores_initial_state_and_clears_candidate():
    debouncer = ConsecutiveConfirmDebouncer(initial=(True,), confirm_frames=2)
    debouncer.update((False,))  # candidate started, not yet confirmed

    debouncer.reset((False,))

    assert debouncer.update((True,)) == [False]  # candidate count restarted
    assert debouncer.update((True,)) == [True]


def test_mismatched_length_raises():
    debouncer = ConsecutiveConfirmDebouncer(initial=(True, True), confirm_frames=2)
    with pytest.raises(ValueError):
        debouncer.update((True,))
