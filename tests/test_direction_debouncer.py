import pytest

from corridor_classifier.direction_debouncer import ConsecutiveConfirmDebouncer


def test_does_not_switch_on_a_single_disagreeing_frame():
    debouncer = ConsecutiveConfirmDebouncer(
        initial=(True, False, False), confirm_frames=1, min_confirm_frames=3
    )

    assert debouncer.update((False, True, False)) == [True, False, False]


def test_single_frame_right_after_reset_does_not_immediately_lock_in():
    # This is the scenario that motivated min_confirm_frames: right after a
    # reset (e.g. just finished a turn), a single noisy misclassification
    # used to become the published value for the entire next hold period.
    debouncer = ConsecutiveConfirmDebouncer(
        initial=(True,), confirm_frames=32, min_confirm_frames=3
    )
    debouncer.reset()

    assert debouncer.update((False,)) == [True]  # 1 noisy frame: not enough evidence
    assert debouncer.update((True,)) == [True]  # noise clears; still at the old value


def test_switches_after_min_confirm_frames_consecutive_matches():
    debouncer = ConsecutiveConfirmDebouncer(
        initial=(True,), confirm_frames=1, min_confirm_frames=3
    )

    assert debouncer.update((False,)) == [True]  # 1st matching frame
    assert debouncer.update((False,)) == [True]  # 2nd
    assert debouncer.update((False,)) == [False]  # 3rd: enough evidence


def test_holds_new_value_until_confirm_frames_have_elapsed():
    debouncer = ConsecutiveConfirmDebouncer(
        initial=(True,), confirm_frames=3, min_confirm_frames=1
    )

    assert debouncer.update((False,)) == [False]  # switch accepted (min_confirm_frames=1)
    assert debouncer.update((True,)) == [False]  # 1 frame since switch: held
    assert debouncer.update((True,)) == [False]  # 2 frames since switch: held
    assert debouncer.update((True,)) == [True]  # 3 frames since switch: allowed again


def test_agreeing_frames_during_hold_do_not_reset_the_hold_timer_early():
    debouncer = ConsecutiveConfirmDebouncer(
        initial=(True,), confirm_frames=3, min_confirm_frames=1
    )

    assert debouncer.update((False,)) == [False]
    assert debouncer.update((False,)) == [False]  # agrees with current; still counts down
    assert debouncer.update((True,)) == [False]  # 2 frames since switch: still held
    assert debouncer.update((True,)) == [True]  # 3 frames since switch: allowed


def test_confirm_frames_and_min_confirm_frames_of_one_switches_on_every_disagreeing_frame():
    debouncer = ConsecutiveConfirmDebouncer(
        initial=(False,), confirm_frames=1, min_confirm_frames=1
    )

    assert debouncer.update((True,)) == [True]
    assert debouncer.update((False,)) == [False]
    assert debouncer.update((True,)) == [True]


def test_reset_allows_an_immediate_switch_again():
    debouncer = ConsecutiveConfirmDebouncer(
        initial=(True,), confirm_frames=3, min_confirm_frames=1
    )
    debouncer.update((False,))  # switch accepted, hold timer starts

    debouncer.reset((True,))

    assert debouncer.update((False,)) == [False]  # not still held after reset


def test_directions_switch_as_one_atomic_tuple():
    debouncer = ConsecutiveConfirmDebouncer(
        initial=(True, True), confirm_frames=1, min_confirm_frames=1
    )

    # Both slots flip together once the full tuple has enough evidence.
    assert debouncer.update((False, False)) == [False, False]


def test_mismatched_length_raises():
    debouncer = ConsecutiveConfirmDebouncer(initial=(True, True), confirm_frames=2)
    with pytest.raises(ValueError):
        debouncer.update((True,))


def test_rejects_nonpositive_confirm_frames():
    with pytest.raises(ValueError):
        ConsecutiveConfirmDebouncer(initial=(True,), confirm_frames=0)


def test_rejects_nonpositive_min_confirm_frames():
    with pytest.raises(ValueError):
        ConsecutiveConfirmDebouncer(initial=(True,), min_confirm_frames=0)


def test_bypass_hold_switches_after_min_confirm_frames_even_mid_hold():
    debouncer = ConsecutiveConfirmDebouncer(
        initial=(True,), confirm_frames=10, min_confirm_frames=2
    )

    assert debouncer.update((False,)) == [True]  # 1st matching frame: not enough evidence yet
    assert debouncer.update((False,), bypass_hold=True) == [False]  # 2nd: bypassed


def test_single_noisy_bypass_frame_does_not_switch():
    # A lone frame that happens to match a high-priority target should not
    # alone be enough to switch (and, for a consumer like
    # scenario_navigation, should not alone be enough to advance a step).
    debouncer = ConsecutiveConfirmDebouncer(
        initial=(True,), confirm_frames=10, min_confirm_frames=3
    )

    assert debouncer.update((False,), bypass_hold=True) == [True]
    assert debouncer.update((True,)) == [True]  # noise clears; candidate streak resets
    assert debouncer.update((False,), bypass_hold=True) == [True]  # streak restarts at 1


def test_bypass_hold_resets_the_hold_timer_for_the_next_switch():
    debouncer = ConsecutiveConfirmDebouncer(
        initial=(True,), confirm_frames=3, min_confirm_frames=1
    )
    debouncer.update((False,))
    debouncer.update((True,), bypass_hold=True)  # bypass switch back to True

    # The bypassed switch still starts its own hold period.
    assert debouncer.update((False,)) == [True]
    assert debouncer.update((False,)) == [True]
    assert debouncer.update((False,)) == [False]


def test_bypass_hold_has_no_effect_when_already_at_that_value():
    debouncer = ConsecutiveConfirmDebouncer(initial=(True,), confirm_frames=3)

    assert debouncer.update((True,), bypass_hold=True) == [True]
