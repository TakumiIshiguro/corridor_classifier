import pytest

from corridor_classifier.messages import (
    make_direction_passage_message,
    make_passage_message,
)


CLASS_NAMES = (
    "straight_road",
    "dead_end",
    "corner_right",
    "corner_left",
    "cross_road",
    "3_way_right",
    "3_way_center",
    "3_way_left",
)


def test_passage_message_contains_name_and_one_hot_label():
    msg = make_passage_message(5, CLASS_NAMES)

    assert list(msg.cmd_dir) == [0, 0, 0]
    assert list(msg.intersection_label) == [0, 0, 0, 0, 0, 1, 0, 0]
    assert msg.intersection_name == "3_way_right"


def test_passage_message_rejects_invalid_index():
    with pytest.raises(ValueError, match="class_index"):
        make_passage_message(8, CLASS_NAMES)


def test_turning_message_uses_name_with_legacy_all_zero_one_hot():
    msg = make_passage_message(8, CLASS_NAMES + ("turning",))

    assert list(msg.intersection_label) == [0] * 8
    assert msg.intersection_name == "turning"


def test_direction_message_reconstructs_legacy_shape():
    msg = make_direction_passage_message(
        [1, 0, 1], False, CLASS_NAMES + ("turning",)
    )

    assert list(msg.intersection_label) == [0, 0, 0, 0, 0, 1, 0, 0]
    assert msg.intersection_name == "3_way_right"


def test_direction_message_prioritizes_turning():
    msg = make_direction_passage_message(
        [1, 1, 1], True, CLASS_NAMES + ("turning",)
    )

    assert list(msg.intersection_label) == [0] * 8
    assert msg.intersection_name == "turning"
