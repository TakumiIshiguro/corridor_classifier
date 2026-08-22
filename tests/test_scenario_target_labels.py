from corridor_classifier.scenario_target_labels import parse_target_labels


def test_parse_target_labels_splits_on_comma():
    assert parse_target_labels("3_way_center,corner_left") == {
        "3_way_center",
        "corner_left",
    }


def test_parse_target_labels_handles_single_label():
    assert parse_target_labels("straight_road") == {"straight_road"}


def test_parse_target_labels_handles_empty_string():
    assert parse_target_labels("") == set()


def test_parse_target_labels_ignores_empty_entries():
    assert parse_target_labels("corner_left,,corner_right") == {
        "corner_left",
        "corner_right",
    }
