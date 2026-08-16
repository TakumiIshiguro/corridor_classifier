import math

import pytest

from corridor_classifier.turn_detection import (
    bridge_short_turning_gaps,
    TurningDetector,
    replace_post_turn_label_islands,
)


def test_turning_detector_finds_sustained_yaw_change():
    stamps = [index * 0.1 for index in range(41)]
    yaws = []
    for stamp in stamps:
        if stamp <= 1.0:
            yaws.append(0.0)
        elif stamp <= 2.0:
            yaws.append(0.3 * (stamp - 1.0))
        else:
            yaws.append(0.3)
    detector = TurningDetector(
        stamps,
        yaws,
        angular_speed_threshold_rad_s=0.12,
        window_seconds=1.0,
        minimum_duration_seconds=0.5,
        padding_seconds=0.0,
        maximum_pose_gap_seconds=0.2,
    )

    assert detector.is_turning(1.5)
    assert not detector.is_turning(0.2)
    assert not detector.is_turning(3.0)
    assert detector.angular_speed(1.5) == pytest.approx(0.3)


def test_turning_detector_handles_yaw_wraparound():
    stamps = [index * 0.1 for index in range(21)]
    unwrapped = [3.0 + 0.25 * stamp for stamp in stamps]
    wrapped = [math.atan2(math.sin(yaw), math.cos(yaw)) for yaw in unwrapped]
    detector = TurningDetector(
        stamps,
        wrapped,
        angular_speed_threshold_rad_s=0.12,
        window_seconds=0.5,
        minimum_duration_seconds=0.2,
        padding_seconds=0.0,
        maximum_pose_gap_seconds=0.2,
    )

    assert detector.is_turning(1.0)
    assert detector.angular_speed(1.0) == pytest.approx(0.25)


def test_short_post_turn_old_label_is_replaced_with_following_label():
    assert replace_post_turn_label_islands(
        stamps=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25],
        labels=[3, 8, 8, 3, 3, 0],
        turn_class_index=8,
        maximum_seconds=6.0,
    ) == [3, 8, 8, 0, 0, 0]


def test_long_post_turn_old_label_is_not_replaced():
    assert replace_post_turn_label_islands(
        stamps=[0.0, 1.0, 2.0, 9.0, 10.0],
        labels=[3, 8, 3, 3, 0],
        turn_class_index=8,
        maximum_seconds=6.0,
    ) == [3, 8, 3, 3, 0]


def test_short_straight_gap_between_turns_is_replaced_with_turning():
    assert bridge_short_turning_gaps(
        stamps=[0.0, 0.25, 0.5, 0.75, 1.0],
        labels=[8, 8, 0, 8, 8],
        turn_class_index=8,
        maximum_seconds=1.5,
    ) == [8, 8, 8, 8, 8]


def test_long_straight_gap_between_turns_is_preserved():
    assert bridge_short_turning_gaps(
        stamps=[0.0, 0.25, 1.0, 3.0, 3.25],
        labels=[8, 8, 0, 0, 8],
        turn_class_index=8,
        maximum_seconds=1.5,
    ) == [8, 8, 0, 0, 8]
